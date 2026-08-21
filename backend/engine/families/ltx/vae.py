"""LTX 2.3 video/audio codec — public decode/mux API."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from backend.engine.common.codecs.vae.video_frames import ncthw_pixels_to_pil_frames
from backend.engine.common.model.dit_stem import require_mlx_ctx


def decode_ltx_latents_to_pil_frames(
    ctx: Any,
    latents_bcthw: Any,
    bundle_root: Path | None,
    on_stage: Callable[[float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> list:
    """Decode ``[B,C,T,H,W]`` video latents to RGB ``PIL.Image`` frames."""
    if bundle_root is None:
        raise RuntimeError("LTX 2.3 VAE decode requires a local model bundle path.")

    require_mlx_ctx(ctx, feature="LTX 2.3 VAE decode")
    from .vae_mlx import decode_latents_ncthw

    pixels_ncthw = decode_latents_ncthw(
        ctx, latents_bcthw, bundle_root, on_stage=on_stage, on_log=on_log,
    )
    return ncthw_pixels_to_pil_frames(ctx, pixels_ncthw, model_label="LTX")


def decode_ltx23_av_to_mp4(
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
        raise RuntimeError("LTX 2.3 mux requires a local model bundle path.")
    require_mlx_ctx(ctx, feature="LTX 2.3 mux")
    from .vae_mlx import mux_video_audio_mp4

    return mux_video_audio_mp4(
        ctx,
        video_latent,
        audio_latent,
        output_path,
        bundle_root,
        frame_rate=frame_rate,
        on_log=on_log,
    )


def get_ltx23_video_decoder(ctx: Any, bundle_root: Path):
    """Return cached ``LTX23VideoDecoder`` (fail loud if bundle files missing)."""
    if bundle_root is None:
        raise RuntimeError("LTX 2.3 video decoder requires a local model bundle path.")
    require_mlx_ctx(ctx, feature="LTX 2.3 video decoder")
    from .vae_mlx import load_ltx23_video_decoder

    return load_ltx23_video_decoder(bundle_root, load_fn=getattr(ctx, "load_weights", None))


def get_ltx23_video_encoder(ctx: Any, bundle_root: Path):
    if bundle_root is None:
        raise RuntimeError("LTX 2.3 video encoder requires a local model bundle path.")
    require_mlx_ctx(ctx, feature="LTX 2.3 video encoder")
    from .vae_mlx import load_ltx23_video_encoder

    return load_ltx23_video_encoder(bundle_root, load_fn=getattr(ctx, "load_weights", None))


def get_ltx23_latent_upsampler(ctx: Any, bundle_root: Path, *, variant: str = "spatial_x2"):
    if bundle_root is None:
        raise RuntimeError("LTX 2.3 latent upsampler requires a local model bundle path.")
    require_mlx_ctx(ctx, feature="LTX 2.3 latent upsampler")
    from .vae_mlx import load_ltx23_latent_upsampler

    return load_ltx23_latent_upsampler(
        bundle_root, variant=variant, load_fn=getattr(ctx, "load_weights", None),
    )


def get_ltx23_audio_decoder(ctx: Any, bundle_root: Path):
    if bundle_root is None:
        raise RuntimeError("LTX 2.3 audio decoder requires a local model bundle path.")
    require_mlx_ctx(ctx, feature="LTX 2.3 audio decoder")
    from .vae_mlx import load_ltx23_audio_decoder

    return load_ltx23_audio_decoder(bundle_root, load_fn=getattr(ctx, "load_weights", None))
