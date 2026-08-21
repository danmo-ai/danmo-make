"""
DiffRhythm 2 MuQ-MuLan style encoder — MLX-only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.engine.common.model.dit_stem import require_mlx_ctx


class MuQStyleEncoder:
    """Text style encoder for DiffRhythm 2 (MLX)."""

    def __init__(self, ctx: Any, cache_dir: Path, mulan_repo_id: str):
        require_mlx_ctx(ctx, feature="MuQ style encoder")
        self._ctx = ctx
        from .mulan_mlx import MuQStyleEncoderMLX

        self._enc = MuQStyleEncoderMLX(cache_dir, mulan_repo_id, ctx)

    def load(self) -> None:
        self._enc.load()

    def encode_text(self, style_prompt: str, *, array_fn: Any) -> Any:
        return self._enc.encode_text(style_prompt, array_fn=array_fn)
