"""Danmo Make MCP server — creative tools over the local REST API."""

from __future__ import annotations

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from backend.mcp.bridge import MakeAPIBridge, dumps
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

    @mcp.tool(description="List models; filter media/action/installed.")
    async def list_models(
        media: Optional[str] = None,
        action: Optional[str] = None,
        installed: Optional[bool] = None,
    ) -> str:
        params: dict[str, Any] = {}
        if media:
            params["media"] = media
        if action:
            params["action"] = action
        if installed is not None:
            params["installed"] = installed
        return dumps(await get_bridge().get("/api/models", params=params or None))

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

    @mcp.tool(description="Text-to-image; wait=true until done (default).")
    async def generate_image(
        model: str,
        prompt: str,
        negative_prompt: str = "",
        size: str = "1024x1024",
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        n: int = 1,
        priority: str = "normal",
        wait: bool = True,
        wait_timeout_seconds: float = 300.0,
    ) -> str:
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "size": size,
            "n": n,
            "priority": priority,
        }
        if steps is not None:
            body["steps"] = steps
        if guidance is not None:
            body["guidance"] = guidance
        if seed is not None:
            body["seed"] = seed
        return await _submit_and_maybe_wait(
            "/api/images/generations",
            body,
            wait=wait,
            wait_timeout_seconds=wait_timeout_seconds,
        )

    @mcp.tool(description="Edit image (rewrite/retouch/extend); needs source_asset_id.")
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

    @mcp.tool(description="Upscale image from source_asset_id.")
    async def upscale_image(
        model: str,
        source_asset_id: str,
        scale: int = 2,
        denoise: float = 0.0,
        priority: str = "normal",
        wait: bool = True,
        wait_timeout_seconds: float = 300.0,
    ) -> str:
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

    @mcp.tool(description="Text/image-to-video; wait default true (use longer timeout).")
    async def generate_video(
        model: str,
        prompt: str,
        negative_prompt: str = "",
        size: str = "832x480",
        num_frames: int = 81,
        fps: int = 16,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        reference_asset_ids: Optional[list[str]] = None,
        priority: str = "normal",
        wait: bool = True,
        wait_timeout_seconds: float = 900.0,
    ) -> str:
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "size": size,
            "num_frames": num_frames,
            "fps": fps,
            "priority": priority,
            "reference_asset_ids": [
                a.removeprefix("asset:") for a in (reference_asset_ids or [])
            ],
        }
        if steps is not None:
            body["steps"] = steps
        if guidance is not None:
            body["guidance"] = guidance
        if seed is not None:
            body["seed"] = seed
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

    @mcp.tool(description="Generate audio/music; wait default true.")
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
