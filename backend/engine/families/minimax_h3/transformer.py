"""MiniMax-H3 Transformer stem — MLX-only family_generator (no standard DiT load)."""
from __future__ import annotations

from typing import Any

from backend.engine.common.model.base import TransformerBase


class MinimaxH3Transformer(TransformerBase):
    """Placeholder stem; MiniMax-H3 uses ``family_generator`` (Shape C)."""

    def __init__(self, config: Any, ctx: Any):
        self.config = config
        self.ctx = ctx

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "MiniMax-H3 does not use the standard video denoise loop; "
            "video_pipeline_shape=family_generator"
        )
