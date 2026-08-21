"""Z-Image Text Encoder — MLX stack."""
from __future__ import annotations

from typing import Any

from backend.engine.common.model.dit_stem import require_mlx_ctx
from .text_encoder_mlx import ZImageTextEncoder as _ZImageTextEncoderMlx

__all__ = ["ZImageTextEncoder"]


class ZImageTextEncoder(_ZImageTextEncoderMlx):
    """Registry entry — MLX-only."""

    def __new__(
        cls,
        ctx: Any,
        model_path: str,
        max_seq_len: int = 512,
        tokenizer_path: str = "",
        **kw: Any,
    ):
        require_mlx_ctx(ctx, feature="Z-Image text encoder")
        return super().__new__(cls)
