"""MiniMax-H3 text encoder — public facade (MLX Qwen3-VL conditioner)."""
from __future__ import annotations

from typing import Any

from .text_encoder_mlx import MiniMaxH3TextEncoderMLX

__all__ = ["MiniMaxH3TextEncoder", "MiniMaxH3TextEncoderMLX"]


def MiniMaxH3TextEncoder(ctx: Any, *args: Any, **kwargs: Any) -> MiniMaxH3TextEncoderMLX:
    """Construct the MLX Qwen3-VL text/vision encoder; CUDA is not implemented (fail loud)."""
    if getattr(ctx, "backend", None) == "cuda":
        raise RuntimeError(
            "MiniMax-H3 text encoder CUDA backend is not implemented; use MLX or fail loud."
        )
    if getattr(ctx, "backend", None) != "mlx":
        raise RuntimeError(
            f"MiniMax-H3 text encoder requires MLX (got {getattr(ctx, 'backend', None)!r})"
        )
    return MiniMaxH3TextEncoderMLX(ctx, *args, **kwargs)
