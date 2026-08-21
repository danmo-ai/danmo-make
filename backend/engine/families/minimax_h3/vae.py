"""MiniMax-H3 video/audio VAE — public decode/mux API."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from backend.engine.common.model.dit_stem import require_mlx_ctx

from .vae_mlx import (
    AutoencoderKLMiniMaxH3AudioMLX,
    AutoencoderKLMiniMaxH3MLX,
    load_audio_vae,
    load_video_vae,
)

__all__ = [
    "AutoencoderKLMiniMaxH3MLX",
    "AutoencoderKLMiniMaxH3AudioMLX",
    "decode_video_latents_ncthw",
    "decode_audio_latents",
    "load_video_vae",
    "load_audio_vae",
    "mux_video_audio_mp4",
    "decode_minimax_h3_av_to_mp4",
]


def decode_video_latents_ncthw(
    ctx: Any,
    latents_bcthw: Any,
    bundle_root: Path,
    *,
    on_stage: Callable[[float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> Any:
    require_mlx_ctx(ctx, feature="MiniMax-H3 VAE decode")
    from .vae_mlx import decode_video_latents_ncthw as _decode

    return _decode(ctx, latents_bcthw, bundle_root, on_stage=on_stage, on_log=on_log)


def decode_audio_latents(ctx: Any, audio_latent: Any, bundle_root: Path) -> Any:
    require_mlx_ctx(ctx, feature="MiniMax-H3 audio VAE decode")
    from .vae_mlx import decode_audio_latents as _decode

    return _decode(ctx, audio_latent, bundle_root)


def mux_video_audio_mp4(
    ctx: Any,
    video_latent: Any,
    audio_latent: Any,
    output_path: str,
    bundle_root: Path,
    *,
    frame_rate: float = 24.0,
    on_log: Callable[[str], None] | None = None,
) -> str:
    require_mlx_ctx(ctx, feature="MiniMax-H3 mux")
    from .vae_mlx import mux_video_audio_mp4 as _mux

    return _mux(
        ctx,
        video_latent,
        audio_latent,
        output_path,
        bundle_root,
        frame_rate=frame_rate,
        on_log=on_log,
    )


def decode_minimax_h3_av_to_mp4(
    ctx: Any,
    video_latent: Any,
    audio_latent: Any,
    output_path: str,
    bundle_root: Path | None,
    *,
    frame_rate: float = 24.0,
    on_log: Callable[[str], None] | None = None,
) -> str:
    """Decode joint A/V latents and mux to mp4 (requires ffmpeg on PATH)."""
    if bundle_root is None:
        raise RuntimeError("MiniMax-H3 mux requires a local model bundle path.")
    return mux_video_audio_mp4(
        ctx,
        video_latent,
        audio_latent,
        output_path,
        bundle_root,
        frame_rate=frame_rate,
        on_log=on_log,
    )
