"""FIBO Transformer — 对外入口（MLX dispatch）。"""
from __future__ import annotations

from typing import Any

from backend.engine.common.model.dit_stem import DelegatingDiTStem


class FIBOTransformer(DelegatingDiTStem):
    """FIBO DiT — MLX DiT via ``DelegatingDiTStem``."""

    def __init__(self, config: Any, ctx: Any):
        from .transformer_mlx import FIBODiTMLX as _MLX

        super().__init__(
            config,
            ctx,
            mlx_cls=_MLX,
        )
