"""Shared video FamilyPlugin backbone — loads DiT via ``load_video_transformer``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.engine.pipelines.video_model_load import (
    latent_frame_count_for_video,
    load_video_transformer,
    prepare_video_config,
    resolve_video_num_frames,
    uses_family_video_avatar,
    uses_family_video_generator,
)
from backend.engine.platform.session import PlatformSession
from backend.engine.protocols.bundle import MediaBundle
from backend.engine.protocols.plugin import FamilyPlugin, FamilySpec


def plugin_video_backbone_model_if_ready(
    plugin: FamilyPlugin | None,
    *,
    config: Any,
    num_frames: int,
) -> Any | None:
    """Return plugin-preloaded video DiT when frame count matches encode phase."""
    if plugin is None:
        return None
    backbone = plugin.backbone
    if getattr(backbone, "_skip_load", False):
        return None
    expected = latent_frame_count_for_video(config, num_frames)
    if getattr(backbone, "_latent_frames", None) != expected:
        return None
    return getattr(backbone, "_model", None)


class VideoPluginBackbone:
    """v3 video backbone holder (wan/hunyuan diffusion path; LTX generator skips load)."""

    def __init__(self, spec: FamilySpec) -> None:
        self.spec = spec
        self._model: Any | None = None
        self._registry_entry: Any | None = None
        self._project_root: Path | None = None
        self._model_cache: Any | None = None
        self._latent_frames: int | None = None
        self._skip_load = False

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError(
                f"VideoPluginBackbone({self.spec.family_id!r}): call load() before forward"
            )
        return self._model

    def bind_load_context(
        self,
        *,
        registry_entry: Any,
        project_root: Path,
        model_cache: Any | None = None,
        bundle_root: Path,
        request: Any,
    ) -> bool:
        """Bind load context; return ``False`` when backbone load should be skipped."""
        self._registry_entry = registry_entry
        self._project_root = project_root
        self._model_cache = model_cache
        config = prepare_video_config(
            registry_entry,
            self.spec.family_id,
            bundle_root,
            project_root=project_root,
        )
        if uses_family_video_generator(config) or uses_family_video_avatar(config):
            self._skip_load = True
            return False
        pixel_frames = resolve_video_num_frames(request, registry_entry)
        self._latent_frames = latent_frame_count_for_video(config, pixel_frames)
        return True

    def load(self, bundle: MediaBundle, platform: PlatformSession) -> None:
        if self._skip_load:
            return
        if (
            self._registry_entry is None
            or self._project_root is None
            or self._latent_frames is None
        ):
            raise RuntimeError(
                f"VideoPluginBackbone({self.spec.family_id!r}): bind_load_context() required"
            )
        config = prepare_video_config(
            self._registry_entry,
            self.spec.family_id,
            bundle.root,
            project_root=self._project_root,
        )
        self._model = load_video_transformer(
            ctx=platform.kernels,
            family=self.spec.family_id,
            config=config,
            entry=self._registry_entry,
            version_key=bundle.version_key,
            project_root=self._project_root,
            num_frames=self._latent_frames,
            model_cache=self._model_cache,
        )
        if self._model is None:
            raise RuntimeError(
                f"VideoPluginBackbone({self.spec.family_id!r}): transformer weights missing "
                f"under {bundle.root}"
            )

    def after_load(self, bundle: MediaBundle) -> None:
        if self._model is not None:
            self._model.after_load_weights(bundle_root=str(bundle.root))

    def forward(self, latents: Any, t: Any, **kwargs: Any) -> Any:
        return self.model(latents, t, **kwargs)
