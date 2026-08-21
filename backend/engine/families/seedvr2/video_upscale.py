"""SeedVR2 spatiotemporal video restoration — registered video upscale runner."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

from backend.engine.cache import ModelCache
from backend.engine.common.ops.scale_factor import ScaleFactor
from backend.engine.contracts import local_bundle_root, require_entry_family
from backend.engine.families.seedvr2.stem_mlx import (
    load_seedvr2_upscale_pipeline,
    run_seedvr2_spatiotemporal_video,
)
from backend.engine.runtime._base import RuntimeContext
from backend.utils.video_sr_ffmpeg import (
    extract_png_frames,
    mux_png_sequence_to_mp4,
    probe_video_fps,
)

_VIDEO_SUFFIXES = frozenset({".mp4", ".webm", ".mov", ".mkv", ".m4v"})


def run_seedvr2_video_upscale(
    *,
    ctx: RuntimeContext,
    request: Any,
    ctx_exec: Any,
    entry: Any,
    version_key: str | None,
    model_key: str,
    asset_store: Any,
    model_registry: Any,
    model_cache: ModelCache | None,
    project_root: Path,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Decode source video → SeedVR2 3D VAE chunks → mux MP4 (audio preserved when present)."""
    from backend.engine.common.model.dit_stem import require_mlx_ctx

    require_mlx_ctx(ctx, feature="SeedVR2 video upscale")
    _ = model_registry

    require_entry_family(entry, model_id=model_key)
    bundle_root = local_bundle_root(project_root, entry, version_key)
    if bundle_root is None:
        raise RuntimeError(f"SeedVR2 video bundle not installed for {model_key!r}")

    src_id = request.source_asset_id
    if not src_id:
        raise RuntimeError("Video upscale requires source_asset_id")

    src_path = asset_store.get_file_path(src_id)
    if src_path is None or not src_path.is_file():
        raise RuntimeError(f"Source asset not found: {src_id!r}")

    if src_path.suffix.lower() not in _VIDEO_SUFFIXES:
        raise RuntimeError(
            f"SeedVR2 video upscale expects a video file ({', '.join(sorted(_VIDEO_SUFFIXES))}); "
            f"got {src_path.suffix!r} on {src_path.name!r}"
        )

    if ctx_exec.cancel_token.is_cancelled():
        return None

    max_frames = int(getattr(request, "max_frames", 300) or 300)
    scale = int(getattr(request, "scale", 2) or 2)
    if scale not in (2, 4):
        raise RuntimeError(f"SeedVR2 video upscale scale must be 2 or 4, got {scale!r}")
    softness = float(getattr(request, "denoise", 0.3) or 0.3)
    chunk_frames = max(4, int(getattr(request, "temporal_window", 5) or 5))

    meta = getattr(request, "metadata", None) or {}
    seed_raw = meta.get("seed")
    seed_base = int(seed_raw) if seed_raw is not None else random.randint(0, 2 ** 31 - 1)

    work = Path(ctx_exec.work_dir)
    frames_in = work / "frames_in"
    frames_out = work / "frames_out"
    frames_in.mkdir(parents=True, exist_ok=True)
    frames_out.mkdir(parents=True, exist_ok=True)

    if on_log:
        on_log("info", f"seedvr2_video_upscale extract max_frames={max_frames} src={src_path.name}")

    n_frames = extract_png_frames(video=src_path, out_dir=frames_in, max_frames=max_frames)
    if n_frames < 1:
        raise RuntimeError(f"No frames extracted from {src_path}")

    fps = probe_video_fps(src_path)

    sr_cache_key = f"upscale:video:{entry.id}:{version_key or 'default'}"
    from backend.engine.common.bundle.weights import parse_size_gb
    from backend.engine.contracts import resolve_version_block

    ver = resolve_version_block(entry, version_key) or {}
    raw = getattr(entry, "raw", {}) or {}
    size_str = str((ver or {}).get("size") or raw.get("size") or "14GB")

    pipeline = load_seedvr2_upscale_pipeline(
        bundle_path=bundle_root,
        model_key=model_key,
        model_cache=model_cache,
        cache_key=sr_cache_key,
        cache_size_gb=parse_size_gb(size_str),
        on_log=on_log,
    )

    def _progress(frac: float, done: int, total: int) -> None:
        if on_progress:
            on_progress(float(frac), int(done), int(total), None, "denoise")

    def _cancelled() -> bool:
        return ctx_exec.cancel_token.is_cancelled()

    run_seedvr2_spatiotemporal_video(
        pipeline=pipeline,
        frames_dir=frames_in,
        n_frames=n_frames,
        resolution=ScaleFactor.parse(f"{scale}x"),
        softness=softness,
        seed_base=seed_base,
        frames_out_dir=frames_out,
        png_pattern_name="up",
        chunk_frames=chunk_frames,
        on_log=on_log,
        on_progress=_progress,
        is_cancelled=_cancelled,
    )

    if ctx_exec.cancel_token.is_cancelled():
        return None

    out_path = work / f"{model_key}_sr_{src_id}.mp4"
    mux_png_sequence_to_mp4(
        png_pattern=frames_out / "up_%06d.png",
        source_audio_from=src_path,
        output_mp4=out_path,
        fps=fps,
        include_audio=True,
    )

    if on_log:
        on_log("info", f"SeedVR2 video upscale saved {out_path} frames={n_frames} fps={fps:.3f}")
    if on_progress:
        on_progress(1.0, n_frames, n_frames, None, "complete")

    return str(out_path), {
        "model": request.model,
        "source_asset_id": src_id,
        "mime_type": "video/mp4",
        "frame_count": n_frames,
        "fps": fps,
        "scale": scale,
        "denoise": softness,
        "chunk_frames": chunk_frames,
        "seed": seed_base,
    }
