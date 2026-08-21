"""Qwen-Image DiT — public MLX entry."""
from __future__ import annotations

from typing import Any

from backend.engine.common.model.dit_stem import DelegatingDiTStem


class QwenImageTransformer(DelegatingDiTStem):
    """Qwen-Image DiT — MLX implementation from ``RuntimeContext``."""

    def __init__(self, config: Any, ctx: Any):
        from .transformer_mlx import QwenImageDiTMLX as _MLX

        super().__init__(config, ctx, mlx_cls=_MLX)

    @property
    def dit(self):
        return getattr(self._inner, "dit", None)
