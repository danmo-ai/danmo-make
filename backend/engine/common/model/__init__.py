"""L3 model contracts — ``TransformerBase`` and MLX DiT stem."""

from backend.engine.common.model.base import TransformerBase
from backend.engine.common.model.dit_stem import (
    DelegatingDiTStem,
    dispatch_dit_implementation,
    require_mlx_ctx,
)

__all__ = [
    "TransformerBase",
    "DelegatingDiTStem",
    "dispatch_dit_implementation",
    "require_mlx_ctx",
]
