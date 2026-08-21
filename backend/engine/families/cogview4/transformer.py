"""CogView4 Transformer — public stem; MLX in ``transformer_mlx``."""
from __future__ import annotations

from typing import Any

from backend.engine.common.model.dit_stem import DelegatingDiTStem


class CogView4Transformer(DelegatingDiTStem):
    """CogView4-6B joint-attention DiT — MLX on Apple Silicon."""

    def __init__(self, config: Any, ctx: Any):
        from .transformer_mlx import CogView4DiTMLX as _MLX

        super().__init__(
            config,
            ctx,
            mlx_cls=_MLX,
        )
