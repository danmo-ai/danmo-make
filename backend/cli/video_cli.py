"""
视频 CLI 命令 — 调用 DanQingVideoEngine。

与 REST API 端点一一对应：
  danqing-video-generate → POST /api/videos/generations → IVideoEngine.generate()
  danqing-video-edit     → POST /api/videos/edits       → IVideoEngine.edit()
  danqing-video-upscale  → POST /api/videos/upscales    → IVideoEngine.upscale()
"""
from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

from backend.cli.base import build_exec_context, engine_session
from backend.core.contracts import VideoEditRequest, VideoGenerationRequest, VideoUpscaleRequest


def generate(
    model: str,
    prompt: str,
    *,
    negative_prompt: str = "",
    size: str = "1024x1024",
    num_frames: int | None = None,
    fps: int | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    shift: float | None = None,
    seed: int | None = None,
    teacache_mode: str | None = None,
    output: str = "",
    project_root: Path | None = None,
) -> str:
    """文生视频。对应 POST /api/videos/generations。"""
    with engine_session(project_root) as ctx:
        exec_ctx = build_exec_context(
            work_dir=ctx.path_resolver.get_outputs_dir() / "cli_tmp",
            asset_store=ctx.asset_store,
            on_progress=lambda ev: None,
            on_log=lambda ev: print(f"  [{ev.level}] {ev.message}"),
        )

        payload: dict = dict(
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            size=size,
            steps=steps,
            guidance=guidance,
            seed=seed,
        )
        if num_frames is not None:
            payload["num_frames"] = num_frames
        if fps is not None:
            payload["fps"] = fps
        if shift is not None:
            payload["shift"] = shift
        if teacache_mode is not None:
            payload["teacache_mode"] = teacache_mode
        request = VideoGenerationRequest(**payload)

        if not ctx.video_engine.supports(model, "generate"):
            raise RuntimeError(
                f"Model {model!r} does not support text-to-video (create); "
                "check config/models_registry.json actions."
            )

        t0 = time.time()
        result = asyncio.run(ctx.video_engine.generate(request, exec_ctx))
        elapsed = time.time() - t0

        if result.metadata.get("status") == "cancelled":
            raise RuntimeError("Generation cancelled")

        if output and result.output_paths:
            out = Path(output)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result.output_paths[0], out)
            print(f"[cli] DONE ({elapsed:.1f}s) -> {out}")
            return str(out)

        if result.output_paths:
            print(f"[cli] DONE ({elapsed:.1f}s) -> {result.output_paths[0]}")
            return result.output_paths[0]

        raise RuntimeError("No output generated")


def edit(
    model: str,
    source_asset_id: str,
    prompt: str,
    *,
    tail_asset_id: str | None = None,
    negative_prompt: str = "",
    steps: int | None = None,
    seed: int | None = None,
    teacache_mode: str | None = None,
    output: str = "",
    project_root: Path | None = None,
) -> str:
    """视频编辑。对应 POST /api/videos/edits。"""
    with engine_session(project_root) as ctx:
        exec_ctx = build_exec_context(
            work_dir=ctx.path_resolver.get_outputs_dir() / "cli_tmp",
            asset_store=ctx.asset_store,
            on_progress=lambda ev: None,
            on_log=lambda ev: print(f"  [{ev.level}] {ev.message}"),
        )

        request = VideoEditRequest(
            model=model,
            source_asset_id=source_asset_id,
            tail_asset_id=tail_asset_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            seed=seed,
            teacache_mode=teacache_mode,
        )

        if not ctx.video_engine.supports(model, "edit"):
            raise RuntimeError(
                f"Model {model!r} does not support video edit (animate); "
                "check config/models_registry.json actions."
            )

        t0 = time.time()
        result = asyncio.run(ctx.video_engine.edit(request, exec_ctx))
        elapsed = time.time() - t0

        if result.metadata.get("status") == "cancelled":
            raise RuntimeError("Edit cancelled")

        if output and result.output_paths:
            out = Path(output)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result.output_paths[0], out)
            print(f"[cli] DONE ({elapsed:.1f}s) -> {out}")
            return str(out)

        if result.output_paths:
            print(f"[cli] DONE ({elapsed:.1f}s) -> {result.output_paths[0]}")
            return result.output_paths[0]

        raise RuntimeError("No output generated")


def upscale(
    model: str,
    source_asset_id: str,
    *,
    scale: int = 2,
    denoise: float = 0.3,
    max_frames: int = 300,
    seed: int | None = None,
    output: str = "",
    project_root: Path | None = None,
) -> str:
    """视频修复 / 超分。对应 POST /api/videos/upscales。"""
    with engine_session(project_root) as ctx:
        exec_ctx = build_exec_context(
            work_dir=ctx.path_resolver.get_outputs_dir() / "cli_tmp",
            asset_store=ctx.asset_store,
            on_progress=lambda ev: None,
            on_log=lambda ev: print(f"  [{ev.level}] {ev.message}"),
        )

        if scale not in (2, 4):
            raise ValueError("scale must be 2 or 4")
        meta: dict = {}
        if seed is not None:
            meta["seed"] = int(seed)
        request = VideoUpscaleRequest(
            model=model,
            source_asset_id=source_asset_id,
            scale=scale,  # type: ignore[arg-type]
            denoise=float(denoise),
            max_frames=int(max_frames),
            metadata=meta,
        )

        if not ctx.video_engine.supports(model, "upscale"):
            raise RuntimeError(
                f"Model {model!r} does not support video upscale; "
                "check config/models_registry.json actions."
            )

        t0 = time.time()
        result = asyncio.run(ctx.video_engine.upscale(request, exec_ctx))
        elapsed = time.time() - t0

        if result.metadata.get("status") == "cancelled":
            raise RuntimeError("Video upscale cancelled")

        if output and result.output_paths:
            out = Path(output)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result.output_paths[0], out)
            print(f"[cli] DONE ({elapsed:.1f}s) -> {out}")
            return str(out)

        if result.output_paths:
            print(f"[cli] DONE ({elapsed:.1f}s) -> {result.output_paths[0]}")
            return result.output_paths[0]

        raise RuntimeError("No output generated")
