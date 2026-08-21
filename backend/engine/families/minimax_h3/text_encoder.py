"""MiniMax-H3 text encoder — public facade (MLX Qwen3-VL conditioner)."""
from __future__ import annotations

from typing import Any

from backend.engine.common.model.dit_stem import require_mlx_ctx
from .text_encoder_mlx import MiniMaxH3TextEncoderMLX

__all__ = ["MiniMaxH3TextEncoder", "MiniMaxH3TextEncoderMLX"]


def MiniMaxH3TextEncoder(ctx: Any, *args: Any, **kwargs: Any) -> MiniMaxH3TextEncoderMLX:
    """Construct the MLX Qwen3-VL text/vision encoder; fail loud otherwise."""
    require_mlx_ctx(ctx, feature="MiniMax-H3 text encoder")
    return MiniMaxH3TextEncoderMLX(ctx, *args, **kwargs)
