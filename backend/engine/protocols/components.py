"""Component protocols — single ABC layer, family implementations in families/."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from backend.engine.protocols.bundle import MediaBundle, TensorRef


@dataclass(frozen=True)
class EncodeResult:
    embeddings: TensorRef
    pooled: TensorRef | None = None
    attention_mask: TensorRef | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Backbone(Protocol):
    """DiT / UNet backbone — load + forward; hooks live on the DiT model itself."""

    def forward(self, latents: TensorRef, t: TensorRef, **kwargs: Any) -> TensorRef: ...

    def load(self, bundle: MediaBundle, platform: Any) -> None: ...

    def after_load(self, bundle: MediaBundle) -> None: ...


@runtime_checkable
class VAE(Protocol):
    def encode(self, pixels: TensorRef) -> TensorRef: ...

    def decode(self, latents: TensorRef) -> TensorRef: ...

    def load(self, bundle: MediaBundle, platform: Any) -> None: ...


@runtime_checkable
class TextEncoder(Protocol):
    def encode(self, text: str, **kwargs: Any) -> EncodeResult: ...

    def load(self, bundle: MediaBundle, platform: Any) -> None: ...
