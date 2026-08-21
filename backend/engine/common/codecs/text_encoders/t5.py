"""T5-XXL text encoder — MLX-only public entry."""
from __future__ import annotations

from typing import Any

from backend.engine.common.codecs.text_encoders.t5_mlx import T5EncoderMlx
from backend.engine.common.model.dit_stem import require_mlx_ctx


class T5Encoder(T5EncoderMlx):
    """Public T5 encoder — MLX forward only."""

    def __init__(self, ctx: Any, *args: Any, **kwargs: Any) -> None:
        require_mlx_ctx(ctx, feature="T5Encoder")
        super().__init__(ctx, *args, **kwargs)
