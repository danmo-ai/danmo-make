"""CogView4 GLM-4 text encoder facade — MLX-only."""
from __future__ import annotations

from typing import Any

from backend.engine.common.model.dit_stem import require_mlx_ctx
from backend.engine.families.cogview4.text_encoder_mlx import CogView4TextEncoder as _CogView4TextEncoderMlx

__all__ = ["CogView4TextEncoder"]


class CogView4TextEncoder(_CogView4TextEncoderMlx):
    """Registry entry — MLX encode only."""

    def __init__(self, ctx: Any, *args: Any, **kwargs: Any) -> None:
        require_mlx_ctx(ctx, feature="CogView4 text encoder")
        super().__init__(ctx, *args, **kwargs)
