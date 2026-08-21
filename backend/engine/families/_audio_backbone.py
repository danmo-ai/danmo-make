"""Shared audio FamilyPlugin backbone — loads generator via ``load_audio_generator``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.engine.pipelines.audio_model_load import load_audio_generator
from backend.engine.platform.session import PlatformSession
from backend.engine.protocols.bundle import MediaBundle
from backend.engine.protocols.plugin import FamilyPlugin, FamilySpec


def plugin_audio_generator_if_ready(plugin: FamilyPlugin | None) -> Any | None:
    if plugin is None:
        return None
    return getattr(plugin.backbone, "_generator", None)


class AudioPluginBackbone:
    """v3 audio generator holder."""

    def __init__(self, spec: FamilySpec) -> None:
        self.spec = spec
        self._generator: Any | None = None
        self._registry_entry: Any | None = None
        self._project_root: Path | None = None
        self._model_cache: Any | None = None

    @property
    def model(self) -> Any:
        if self._generator is None:
            raise RuntimeError(
                f"AudioPluginBackbone({self.spec.family_id!r}): call load() before forward"
            )
        return self._generator

    def bind_load_context(
        self,
        *,
        registry_entry: Any,
        project_root: Path,
        model_cache: Any | None = None,
        bundle_root: Path,
    ) -> bool:
        _ = bundle_root
        self._registry_entry = registry_entry
        self._project_root = project_root
        self._model_cache = model_cache
        return True

    def load(self, bundle: MediaBundle, platform: PlatformSession) -> None:
        if self._registry_entry is None or self._project_root is None:
            raise RuntimeError(
                f"AudioPluginBackbone({self.spec.family_id!r}): bind_load_context() required"
            )
        self._generator = load_audio_generator(
            ctx=platform.kernels,
            family=self.spec.family_id,
            bundle_root=bundle.root,
            entry=self._registry_entry,
            version_key=bundle.version_key,
            model_cache=self._model_cache,
        )

    def after_load(self, bundle: MediaBundle) -> None:
        return None

    def forward(self, latents: Any, t: Any, **kwargs: Any) -> Any:
        return self.model(latents, t, **kwargs)
