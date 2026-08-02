"""Causal 3D conv streaming state — WF-VAE style chunk decode (SeedVR2 / video VAE)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CausalConv3dStreamState:
    """Temporal tail frames carried across chunked ``CausalConv3d`` calls."""

    tail_frames: Any | None = None
    kernel_temporal: int = 1

    def reset(self) -> None:
        self.tail_frames = None


@dataclass
class VAEStreamCacheSession:
    """Per-decode session for causal conv layers keyed by module id."""

    enabled: bool
    layers: dict[str, CausalConv3dStreamState] = field(default_factory=dict)

    @classmethod
    def from_plan(cls, *, enabled: bool) -> VAEStreamCacheSession:
        return cls(enabled=bool(enabled))

    def layer_state(self, layer_id: str, *, kernel_temporal: int = 1) -> CausalConv3dStreamState:
        st = self.layers.get(layer_id)
        if st is None:
            st = CausalConv3dStreamState(kernel_temporal=int(kernel_temporal))
            self.layers[layer_id] = st
        return st

    def reset(self) -> None:
        for st in self.layers.values():
            st.reset()

    def prepend_causal_tail(self, x: Any, layer_id: str, *, kernel_temporal: int) -> Any:
        """Prepend cached frames for causal temporal conv input ``[B,C,T,H,W]``."""
        if not self.enabled:
            return x
        st = self.layer_state(layer_id, kernel_temporal=kernel_temporal)
        if st.tail_frames is None or int(kernel_temporal) <= 1:
            return x
        import mlx.core as mx

        return mx.concatenate([st.tail_frames, x], axis=2)

    def update_causal_tail(self, x: Any, layer_id: str, *, kernel_temporal: int) -> None:
        if not self.enabled or int(kernel_temporal) <= 1:
            return
        keep = int(kernel_temporal) - 1
        if keep <= 0:
            return
        st = self.layer_state(layer_id, kernel_temporal=kernel_temporal)
        st.tail_frames = x[:, :, -keep:, :, :]
