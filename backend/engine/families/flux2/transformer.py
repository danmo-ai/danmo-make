"""Flux.2 Transformer — 对外入口（MLX dispatch）。"""
from __future__ import annotations

from typing import Any

from backend.engine.common.model.dit_stem import DelegatingDiTStem


class Flux2Transformer(DelegatingDiTStem):
    """Flux.2 DiT — MLX DiT via ``DelegatingDiTStem``."""

    def __init__(self, config: Any, ctx: Any):
        from .transformer_mlx import Flux2DiTMLX as _MLX

        super().__init__(
            config,
            ctx,
            mlx_cls=_MLX,
        )
