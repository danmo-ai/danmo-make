"""Danmo Make MCP server — creative tools over the local REST API."""

from __future__ import annotations

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from backend.mcp.bridge import MakeAPIBridge, dumps
from backend.mcp.model_guide import (
    enrich_model_list,
    frames_from_duration_sec,
    normalize_list_action,
    summarize_model,
)
from backend.mcp.wait import wait_task as poll_wait_task

_bridge: MakeAPIBridge | None = None


def get_bridge() -> MakeAPIBridge:
    global _bridge
    if _bridge is None:
        _bridge = MakeAPIBridge()
    return _bridge


def set_bridge(bridge: MakeAPIBridge) -> None:
    global _bridge
    _bridge = bridge


def create_mcp(*, name: str = "danmo-make") -> FastMCP:
    """Build FastMCP with streamable-HTTP path ``/`` (mount parent at ``/mcp``)."""
    mcp = FastMCP(
        name,
        instructions=(
            "Danmo Make local generation API. Prefer wait=true on generate_* tools. "
            "Use asset ids (ast_*) from upload_asset / task results."
        ),
        streamable_http_path="/",
        stateless_http=True,
    )
    _register_tools(mcp)
    return mcp


def _register_tools(mcp: FastMCP) -> None:
    @mcp.tool(description="Probe MLX/CUDA runtime health.")
    async def health() -> str:
        return dumps(await get_bridge().get("/api/system/health"))

    @mcp.tool(
        description=(
            "List runnable models (not LoRA adapters) grouped image/video/audio, "
            "sorted commercial→newer→distilled→smaller quant ([0]=suggestion). "
            "Cards include actions: generate|edit|upscale|…. "
            "For text-to-image/video pass action=generate (aliases: create). "
            "For edits pass action=edit. Prefer installed=true."
        )
    )
    async def list_models(
        media: Optional[str] = None,
        action: Optional[str] = None,
        installed: Optional[bool] = None,
    ) -> str:
        api_action = normalize_list_action(action, media=media)
        params: dict[str, Any] = {}
        if media:
            params["media"] = media
        if api_action:
            params["action"] = api_action
        if installed is not None:
            params["installed"] = installed
        bridge = get_bridge()
        index = await bridge.get("/api/models", params=params or None)
        registry = await bridge.get("/api/registry")
        reg_models = registry.get("models") if isinstance(registry, dict) else {}
        if not isinstance(reg_models, dict):
            reg_models = {}
        return dumps(
            enrich_model_list(
                index if isinstance(index, dict) else {},
                reg_models,
                require_action=api_action,
            )
        )

    @mcp.tool(
        description=(
            "Model card: actions + type + defaults + parameters "
            "(size options; video duration_sec; audio duration; steps/guidance/fps). "
            "Call after list_models. Reject type=lora for generate_*."
        )
    )
    async def get_model(model_id: str) -> str:
        detail = await get_bridge().get(f"/api/models/{model_id}")
        if not isinstance(detail, dict):
            raise RuntimeError(f"get_model: unexpected response for {model_id!r}")
        cfg = detail.get("config") if isinstance(detail.get("config"), dict) else {}
        index_row = {
            "media": detail.get("media"),
            "family": detail.get("family"),
            "actions": detail.get("actions"),
            "type": cfg.get("type"),
            "installed": None,
        }
        return dumps(
            summarize_model(
                str(detail.get("id") or model_id),
                index_row=index_row,
                config=cfg,
                full=True,
            )
        )

    @mcp.tool(description="Upload a local file path → asset id (ast_*).")
    async def upload_asset(path: str) -> str:
        return dumps(await get_bridge().upload_file(path))

    @mcp.tool(description="Get asset metadata and local file path.")
    async def get_asset(asset_id: str) -> str:
        aid = asset_id.removeprefix("asset:").strip()
        return dumps(await get_bridge().get(f"/api/assets/{aid}"))

    @mcp.tool(description="Get task status/result by tsk_* id.")
    async def get_task(task_id: str) -> str:
        return dumps(await get_bridge().get(f"/api/tasks/{task_id}"))

    @mcp.tool(description="Poll task until completed/failed/cancelled or timeout.")
    async def wait_task(task_id: str, wait_timeout_seconds: float = 300.0) -> str:
        return dumps(
            await poll_wait_task(
                get_bridge(),
                task_id,
                timeout_seconds=wait_timeout_seconds,
            )
        )

    @mcp.tool(description="Cancel a queued or running task.")
    async def cancel_task(task_id: str) -> str:
        return dumps(await get_bridge().delete(f"/api/tasks/{task_id}"))

    @mcp.tool(description="Fetch task diagnostic bundle (failure/graph/trace).")
    async def diagnose_task(task_id: str) -> str:
        return dumps(await get_bridge().get(f"/api/tasks/{task_id}/diagnostic"))

    async def _submit_and_maybe_wait(
        path: str,
        body: dict[str, Any],
        *,
        wait: bool,
        wait_timeout_seconds: float,
    ) -> str:
        submitted = await get_bridge().post(path, json_body=body)
        if not wait:
            return dumps(submitted)
        task = (submitted or {}).get("task") if isinstance(submitted, dict) else None
        if not isinstance(task, dict) or not task.get("id"):
            raise RuntimeError(f"submit missing task.id: {submitted!r}")
        tid = str(task["id"])
        final = await poll_wait_task(
            get_bridge(),
            tid,
            timeout_seconds=wait_timeout_seconds,
        )
        return dumps({"submit": submitted, "result": final})

    async def _model_card(model_id: str) -> dict[str, Any]:
        detail = await get_bridge().get(f"/api/models/{model_id}")
        if not isinstance(detail, dict):
            raise RuntimeError(f"model not found: {model_id!r}")
        cfg = detail.get("config") if isinstance(detail.get("config"), dict) else {}
        return summarize_model(
            str(detail.get("id") or model_id),
            index_row={
                "media": detail.get("media"),
                "family": detail.get("family"),
                "actions": detail.get("actions"),
                "type": cfg.get("type"),
            },
            config=cfg,
            full=False,
        )

    async def _require_model_action(model_id: str, api_action: str) -> dict[str, Any]:
        card = await _model_card(model_id)
        mtype = str(card.get("type") or "")
        if mtype == "lora":
            raise RuntimeError(
                f"{model_id!r} is a LoRA adapter (type=lora), not a generation model. "
                f"list_models(action={api_action!r}) for base models; attach LoRA in the UI."
            )
        actions = card.get("actions") if isinstance(card.get("actions"), list) else []
        if api_action not in actions:
            raise RuntimeError(
                f"model {model_id!r} does not support {api_action!r} "
                f"(actions={actions or []}). "
                f"Use list_models(installed=true, action={api_action!r})."
            )
        return card

    async def _model_defaults(model_id: str) -> dict[str, Any]:
        card = await _model_card(model_id)
        defaults = card.get("defaults")
        return defaults if isinstance(defaults, dict) else {}

    async def _apply_generation_defaults(body: dict[str, Any], *, keys: tuple[str, ...]) -> None:
        """Fill missing generation fields from registry model defaults (fail loud if no model)."""
        mid = str(body.get("model") or "").strip()
        if not mid:
            raise RuntimeError("model id is required")
        need = [k for k in keys if body.get(k) in (None, "", [])]
        if not need:
            return
        defaults = await _model_defaults(mid)
        for k in need:
            if k in defaults and defaults[k] is not None:
                body[k] = defaults[k]
        # size still missing → explicit error (do not invent WxH)
        if "size" in keys and not body.get("size"):
            raise RuntimeError(
                f"model {mid!r} has no default size; call get_model and pass size=WIDTHxHEIGHT "
                f"from size_options / defaults.size"
            )

    @mcp.tool(
        description=(
            "Text-to-image. model from list_models(action=generate).image[0] "
            "(must have actions containing generate; not LoRA/edit-only). "
            "size from get_model. wait=true default."
        )
    )
    async def generate_image(
        model: str,
        prompt: str,
        negative_prompt: str = "",
        size: str = "",
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        n: int = 1,
        priority: str = "normal",
        wait: bool = True,
        wait_timeout_seconds: float = 300.0,
    ) -> str:
        await _require_model_action(model, "generate")
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "n": n,
            "priority": priority,
        }
        size_s = (size or "").strip()
        if size_s:
            body["size"] = size_s
        if steps is not None:
            body["steps"] = steps
        if guidance is not None:
            body["guidance"] = guidance
        if seed is not None:
            body["seed"] = seed
        await _apply_generation_defaults(body, keys=("size", "steps", "guidance"))
        return await _submit_and_maybe_wait(
            "/api/images/generations",
            body,
            wait=wait,
            wait_timeout_seconds=wait_timeout_seconds,
        )

    @mcp.tool(
        description=(
            "Edit image (rewrite/retouch/extend). "
            "model from list_models(action=edit); needs source_asset_id."
        )
    )
    async def edit_image(
        model: str,
        operation: str,
        source_asset_id: str,
        prompt: str,
        mask_asset_id: Optional[str] = None,
        reference_asset_ids: Optional[list[str]] = None,
        negative_prompt: str = "",
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        source_fidelity: float = 0.6,
        priority: str = "normal",
        wait: bool = True,
        wait_timeout_seconds: float = 300.0,
    ) -> str:
        await _require_model_action(model, "edit")
        body: dict[str, Any] = {
            "model": model,
            "operation": operation,
            "source_asset_id": source_asset_id.removeprefix("asset:"),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "source_fidelity": source_fidelity,
            "priority": priority,
            "reference_asset_ids": [
                a.removeprefix("asset:") for a in (reference_asset_ids or [])
            ],
        }
        if mask_asset_id:
            body["mask_asset_id"] = mask_asset_id.removeprefix("asset:")
        if steps is not None:
            body["steps"] = steps
        if guidance is not None:
            body["guidance"] = guidance
        if seed is not None:
            body["seed"] = seed
        return await _submit_and_maybe_wait(
            "/api/images/edits",
            body,
            wait=wait,
            wait_timeout_seconds=wait_timeout_seconds,
        )

    @mcp.tool(description="Upscale image from source_asset_id. model must support upscale.")
    async def upscale_image(
        model: str,
        source_asset_id: str,
        scale: int = 2,
        denoise: float = 0.0,
        priority: str = "normal",
        wait: bool = True,
        wait_timeout_seconds: float = 300.0,
    ) -> str:
        await _require_model_action(model, "upscale")
        body = {
            "model": model,
            "source_asset_id": source_asset_id.removeprefix("asset:"),
            "scale": scale,
            "denoise": denoise,
            "priority": priority,
        }
        return await _submit_and_maybe_wait(
            "/api/images/upscales",
            body,
            wait=wait,
            wait_timeout_seconds=wait_timeout_seconds,
        )

    @mcp.tool(
        description=(
            "Text/image-to-video. Prefer duration_sec from get_model "
            "(converted to num_frames). size/fps from get_model. wait default true."
        )
    )
    async def generate_video(
        model: str,
        prompt: str,
        negative_prompt: str = "",
        size: str = "",
        duration_sec: Optional[float] = None,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        reference_asset_ids: Optional[list[str]] = None,
        priority: str = "normal",
        wait: bool = True,
        wait_timeout_seconds: float = 900.0,
    ) -> str:
        await _require_model_action(model, "generate")
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "priority": priority,
            "reference_asset_ids": [
                a.removeprefix("asset:") for a in (reference_asset_ids or [])
            ],
        }
        size_s = (size or "").strip()
        if size_s:
            body["size"] = size_s
        if fps is not None:
            body["fps"] = fps
        if num_frames is not None:
            body["num_frames"] = num_frames
        if steps is not None:
            body["steps"] = steps
        if guidance is not None:
            body["guidance"] = guidance
        if seed is not None:
            body["seed"] = seed
        await _apply_generation_defaults(
            body, keys=("size", "steps", "guidance", "fps", "num_frames")
        )
        if duration_sec is not None and num_frames is None:
            detail = await get_bridge().get(f"/api/models/{model}")
            cfg = detail.get("config") if isinstance(detail, dict) else {}
            params = cfg.get("parameters") if isinstance(cfg, dict) else {}
            nf = params.get("num_frames") if isinstance(params, dict) else {}
            rate = int(body.get("fps") or 16)
            body["num_frames"] = frames_from_duration_sec(
                float(duration_sec),
                rate,
                min_frames=int(nf["min"]) if isinstance(nf, dict) and "min" in nf else None,
                max_frames=int(nf["max"]) if isinstance(nf, dict) and "max" in nf else None,
            )
        return await _submit_and_maybe_wait(
            "/api/videos/generations",
            body,
            wait=wait,
            wait_timeout_seconds=wait_timeout_seconds,
        )

    @mcp.tool(description="Video edit (animate); needs source_asset_id.")
    async def edit_video(
        model: str,
        source_asset_id: str,
        prompt: str,
        operation: str = "animate",
        negative_prompt: str = "",
        size: str = "832x480",
        num_frames: int = 81,
        fps: int = 16,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        priority: str = "normal",
        wait: bool = True,
        wait_timeout_seconds: float = 900.0,
    ) -> str:
        await _require_model_action(model, "edit")
        body: dict[str, Any] = {
            "model": model,
            "operation": operation,
            "source_asset_id": source_asset_id.removeprefix("asset:"),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "size": size,
            "num_frames": num_frames,
            "fps": fps,
            "priority": priority,
        }
        if steps is not None:
            body["steps"] = steps
        if seed is not None:
            body["seed"] = seed
        return await _submit_and_maybe_wait(
            "/api/videos/edits",
            body,
            wait=wait,
            wait_timeout_seconds=wait_timeout_seconds,
        )

    @mcp.tool(description="Upscale video from source_asset_id.")
    async def upscale_video(
        model: str,
        source_asset_id: str,
        scale: int = 2,
        denoise: float = 0.3,
        priority: str = "normal",
        wait: bool = True,
        wait_timeout_seconds: float = 900.0,
    ) -> str:
        body = {
            "model": model,
            "source_asset_id": source_asset_id.removeprefix("asset:"),
            "scale": scale,
            "denoise": denoise,
            "priority": priority,
        }
        return await _submit_and_maybe_wait(
            "/api/videos/upscales",
            body,
            wait=wait,
            wait_timeout_seconds=wait_timeout_seconds,
        )

    @mcp.tool(
        description=(
            "Generate audio/music. model=list_models.audio[0].id (or user-named); "
            "wait default true."
        )
    )
    async def generate_audio(
        model: str,
        prompt: str,
        duration: Optional[int] = None,
        instrumental: bool = False,
        lyrics: str = "",
        n: int = 2,
        seed: Optional[int] = None,
        priority: str = "normal",
        wait: bool = True,
        wait_timeout_seconds: float = 600.0,
    ) -> str:
        await _require_model_action(model, "create_music")
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "instrumental": instrumental,
            "lyrics": lyrics,
            "n": n,
            "priority": priority,
        }
        if duration is not None:
            body["duration"] = duration
        if seed is not None:
            body["seed"] = seed
        return await _submit_and_maybe_wait(
            "/api/audios/generations",
            body,
            wait=wait,
            wait_timeout_seconds=wait_timeout_seconds,
        )

    @mcp.tool(description="Audio edit (cover); needs source_asset_id.")
    async def edit_audio(
        model: str,
        source_asset_id: str,
        prompt: str = "",
        operation: str = "cover",
        source_fidelity: float = 1.0,
        n: int = 1,
        seed: Optional[int] = None,
        priority: str = "normal",
        wait: bool = True,
        wait_timeout_seconds: float = 600.0,
    ) -> str:
        await _require_model_action(model, "edit")
        body: dict[str, Any] = {
            "model": model,
            "operation": operation,
            "source_asset_id": source_asset_id.removeprefix("asset:"),
            "prompt": prompt,
            "source_fidelity": source_fidelity,
            "n": n,
            "priority": priority,
        }
        if seed is not None:
            body["seed"] = seed
        return await _submit_and_maybe_wait(
            "/api/audios/edits",
            body,
            wait=wait,
            wait_timeout_seconds=wait_timeout_seconds,
        )


# Module-level instance for stdio / mount helpers
mcp = create_mcp()
