"""Wan text encoder — public MLX entry."""
from __future__ import annotations

from backend.engine.common.model.dit_stem import require_mlx_ctx
from backend.engine.runtime._base import RuntimeContext

from .text_encoder_mlx import WanUMT5EncoderMLX, resolve_wan_umt5_pth


def WanUMT5Encoder(ctx: RuntimeContext, checkpoint_path: str, tokenizer_path: str, *, text_len: int = 512):
    """Return MLX UMT5 encoder; fail loud for non-mlx backends."""
    require_mlx_ctx(ctx, feature="Wan UMT5 encoder")
    return WanUMT5EncoderMLX(ctx, checkpoint_path, tokenizer_path, text_len=text_len)


__all__ = ["WanUMT5Encoder", "WanUMT5EncoderMLX", "resolve_wan_umt5_pth"]
