"""CLIP text encoder — MLX-only public entry."""
from __future__ import annotations

from typing import Any

from backend.engine.common.codecs.text_encoders.clip_mlx import CLIPEncoderMlx
from backend.engine.common.model.dit_stem import require_mlx_ctx


class CLIPEncoder(CLIPEncoderMlx):
    """Public CLIP encoder — MLX forward only."""

    def __init__(self, ctx: Any, *args: Any, **kwargs: Any) -> None:
        require_mlx_ctx(ctx, feature="CLIPEncoder")
        super().__init__(ctx, *args, **kwargs)
