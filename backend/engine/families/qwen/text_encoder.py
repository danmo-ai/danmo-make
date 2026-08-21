"""Qwen-Image text encoder — public MLX entry."""
from __future__ import annotations

from typing import Any

from backend.engine.common.model.dit_stem import require_mlx_ctx
from backend.engine.families.qwen.text_encoder_mlx import QwenImageTextEncoder as _QwenImageTextEncoderMlx

__all__ = ["QwenImageTextEncoder"]


class QwenImageTextEncoder(_QwenImageTextEncoderMlx):
    """Registry entry — MLX-only."""

    def __new__(
        cls,
        ctx: Any,
        model_path: str | Any,
        tokenizer_path: str = "",
        **kw: Any,
    ):
        require_mlx_ctx(ctx, feature="Qwen-Image text encoder")
        return super().__new__(cls)
