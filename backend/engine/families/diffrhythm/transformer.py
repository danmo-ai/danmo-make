"""
DiffRhythm 2 Transformer — registry entry / MLX DiT+CFM lives in ``transformer_mlx``.

Music generation uses ``DiffRhythm2CFMMLX`` via ``generation_mlx`` directly.
"""
from __future__ import annotations

from typing import Any

from backend.engine.common.model.base import TransformerBase
from backend.engine.common.model.dit_stem import require_mlx_ctx


class DiffRhythmTransformer(TransformerBase):
    """Registry placeholder — inference uses ``DiffRhythm2CFMMLX`` in ``generation_mlx``."""

    def __init__(self, ctx: Any, **config: Any):
        super().__init__()
        self._ctx = ctx
        require_mlx_ctx(ctx, feature="DiffRhythmTransformer")
        from .transformer_mlx import DiffRhythm2CFMMLX, DiffRhythm2DiTMLX

        dit = DiffRhythm2DiTMLX(**config)
        self._model = DiffRhythm2CFMMLX(dit)
        self._build_param_map()

    def forward(self, *args: Any, **kwargs: Any):
        raise RuntimeError("Use DiffRhythmMlxGenerator.generate_waveform for inference")

    def parameters(self):
        return self._model.parameters()

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Strip ``module.`` prefix from checkpoint keys."""
        remapped: dict[str, Any] = {}
        for key, tensor in weights.items():
            nk = key[len("module."):] if key.startswith("module.") else key
            remapped[nk] = tensor
        return remapped

    def _build_param_map(self):
        from backend.engine.families.diffrhythm.transformer_mlx import parameters_flat

        self._param_map = parameters_flat(self._model.transformer)
