"""Family plugin aggregate — v3 family-facing surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from backend.engine.protocols.components import Backbone, TextEncoder, VAE

LatentLayout = Literal["nchw", "packed_seq", "qwen_grid", "video_5d", "audio_1d", "pixel_patch"]
CfgMode = Literal["fused", "batched", "dual", "none"]
ParadigmKind = Literal["diffusion", "flow_matching", "block_ar", "two_stage", "job"]


@dataclass(frozen=True)
class FamilySpec:
    """Catalog-facing family metadata (v3 registry ``families`` block).

    Denoise / CFG / scheduler runtime semantics live on ``ModelConfig`` +
    ``FamilyRuntimeContract`` — do not grow this dataclass into a third contract.
    """

    family_id: str
    media: Literal["image", "video", "audio"]
    paradigm: ParadigmKind = "diffusion"
    latent_layout: LatentLayout = "nchw"
    latent_channels: int = 16
    vae_scale: int = 8
    cfg_mode: CfgMode = "dual"
    supports_guidance: bool = True
    default_scheduler: str = "flow_match_euler"
    step_kwargs_profile: str | None = None
    hooks: frozenset[str] = frozenset()
    backends: tuple[str, ...] = ("mlx",)


@dataclass
class FamilyPlugin:
    """Backbone composition root — load gate + family_id/spec; not a fifth strategy surface.

    Conditioning and extra-cond belong on ``TransformerBase`` hooks or
    ``_transformer_registry`` factory rows (``_STRUCTURAL_GUIDE_*``,
    ``_IMAGE_EDIT_EXTRA_COND``, …). Production encode uses ``build_*_run_context``.
    """

    family_id: str
    spec: FamilySpec
    backbone: Backbone
    vae: VAE | None = None
    text_encoder: TextEncoder | None = None
    secondary_encoders: dict[str, TextEncoder] = field(default_factory=dict)

    def select_paradigm(self) -> ParadigmKind:
        return self.spec.paradigm
