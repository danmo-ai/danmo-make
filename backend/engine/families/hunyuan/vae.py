"""HunyuanVideo-1.5 3D causal VAE — decode/encode entry points (``VideoPipeline`` contract)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from backend.engine.common.codecs.vae.video_frames import ncthw_pixels_to_pil_frames
from backend.engine.common.model.dit_stem import require_mlx_ctx


def decode_hunyuan_latents_to_pil_frames(
    ctx: Any,
    latents_bcthw: Any,
    bundle_root: Path | None,
    on_stage: Callable[[float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    *,
    temporal_chunk_size: int = 0,
    spatial_tiling: bool = False,
) -> list:
    """Decode ``[B,C,T,H,W]`` latents to RGB ``PIL.Image`` frame list."""
    if bundle_root is None:
        raise RuntimeError("HunyuanVideo VAE decode requires a local model bundle path.")

    require_mlx_ctx(ctx, feature="HunyuanVideo VAE decode")
    from .vae_mlx import decode_latents_ncthw

    pixels_ncthw = decode_latents_ncthw(
        ctx, latents_bcthw, bundle_root,
        on_stage=on_stage, on_log=on_log,
        temporal_chunk_size=temporal_chunk_size,
        spatial_tiling=spatial_tiling,
    )
    return ncthw_pixels_to_pil_frames(ctx, pixels_ncthw, model_label="HunyuanVideo")


def encode_hunyuan_rgb_to_latents(
    ctx: Any,
    pixels_bcthw: Any,
    bundle_root: Path | None,
    on_log: Callable[[str], None] | None = None,
) -> Any:
    """Encode ``[B,C,T,H,W]`` RGB float video (``[-1,1]``) to latents for I2V conditioning."""
    if bundle_root is None:
        raise RuntimeError("HunyuanVideo VAE encode requires a local model bundle path.")

    require_mlx_ctx(ctx, feature="HunyuanVideo VAE encode")
    from .vae_mlx import encode_video_ncthw

    latents = encode_video_ncthw(ctx, pixels_bcthw, bundle_root, on_log=on_log)
    if getattr(ctx, "is_tensor", lambda _x: False)(latents):
        ctx.eval(latents)
    return latents
