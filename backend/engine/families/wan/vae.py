"""Wan 2.2 3D causal VAE — decode/encode entry points (``VideoPipeline`` contract)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from backend.engine.common.codecs.vae.video_frames import ncthw_pixels_to_pil_frames
from backend.engine.common.model.dit_stem import require_mlx_ctx


def decode_wan_latents_to_pil_frames(
    ctx: Any,
    latents_bcthw: Any,
    bundle_root: Path | None,
    on_stage: Callable[[float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    *,
    spatial_tiling: bool = False,
    spatial_scale: int = 16,
) -> list:
    """Decode ``[B,C,T,H,W]`` latents to RGB ``PIL.Image`` frame list."""
    if bundle_root is None:
        raise RuntimeError("Wan VAE decode requires a local model bundle path.")

    if on_log is not None:
        on_log(f"Wan VAE decode start (latent shape {tuple(getattr(latents_bcthw, 'shape', ()))})")

    require_mlx_ctx(ctx, feature="Wan VAE decode")

    from .vae_mlx import decode_wan_vae_latents

    pixels_ncthw = decode_wan_vae_latents(
        ctx,
        latents_bcthw,
        bundle_root,
        on_stage=on_stage,
        on_log=on_log,
        spatial_tiling=spatial_tiling,
        spatial_scale=spatial_scale,
    )
    frames = ncthw_pixels_to_pil_frames(ctx, pixels_ncthw, model_label="Wan")
    if on_log is not None and frames:
        w, h = frames[0].size
        on_log(f"Wan VAE decode done ({len(frames)} frames, {w}x{h})")
    return frames


def encode_wan_image_to_latent(
    ctx: Any,
    image_chw: Any,
    bundle_root: Path | None,
) -> Any:
    """Encode a single RGB frame ``[C,H,W]`` (float ``[-1,1]``) to latents for I2V conditioning."""
    if bundle_root is None:
        raise RuntimeError("Wan VAE encode requires a local model bundle path.")

    require_mlx_ctx(ctx, feature="Wan VAE encode")

    from .vae_mlx import encode_wan_vae_image

    latents = encode_wan_vae_image(ctx, image_chw, bundle_root)
    if getattr(ctx, "is_tensor", lambda _x: False)(latents):
        ctx.eval(latents)
    return latents
