"""ERNIE-Image Transformer — MLX dispatch entry."""
from __future__ import annotations

from typing import Any

from backend.engine.common.model.dit_stem import DelegatingDiTStem


class ErnieImageTransformer(DelegatingDiTStem):
    """ERNIE-Image single-stream DiT — MLX on Apple Silicon."""

    def __init__(self, config: Any, ctx: Any):
        from .transformer_mlx import ErnieImageDiTMLX as _MLX

        super().__init__(
            config,
            ctx,
            mlx_cls=_MLX,
        )
