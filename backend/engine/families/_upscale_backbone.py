"""Shared upscale / job FamilyPlugin backbone — loads via ``load_upscale_pipeline``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from backend.engine.pipelines.upscale_model_load import load_upscale_pipeline
from backend.engine.platform.session import PlatformSession
from backend.engine.protocols.bundle import MediaBundle
from backend.engine.protocols.plugin import FamilyPlugin, FamilySpec


def plugin_upscale_pipeline_if_ready(plugin: FamilyPlugin | None) -> Any | None:
    if plugin is None:
        return None
    return getattr(plugin.backbone, "_pipeline", None)


class UpscalePluginBackbone:
    """v3 job-paradigm backbone holder (seedvr2 upscale pipeline)."""

    def __init__(self, spec: FamilySpec) -> None:
        self.spec = spec
        self._pipeline: Any | None = None
        self._registry_entry: Any | None = None
        self._model_cache: Any | None = None
        self._model_key: str | None = None

    def bind_load_context(
        self,
        *,
        registry_entry: Any,
        project_root: Path,
        model_cache: Any | None = None,
        bundle_root: Path,
        request: Any,
    ) -> bool:
        _ = project_root, bundle_root, request
        self._registry_entry = registry_entry
        self._model_cache = model_cache
        self._model_key = str(getattr(registry_entry, "id", "") or "")
        return True

    def load(self, bundle: MediaBundle, platform: PlatformSession) -> None:
        _ = platform
        if self._registry_entry is None or not self._model_key:
            raise RuntimeError(
                f"UpscalePluginBackbone({self.spec.family_id!r}): bind_load_context() required"
            )
        self._pipeline = load_upscale_pipeline(
            family=self.spec.family_id,
            bundle_path=bundle.root,
            model_key=self._model_key,
            entry=self._registry_entry,
            version_key=bundle.version_key,
            model_cache=self._model_cache,
        )
        if self._pipeline is None:
            raise RuntimeError(
                f"UpscalePluginBackbone({self.spec.family_id!r}): pipeline loader missing"
            )

    def after_load(self, bundle: MediaBundle) -> None:
        return None

    def forward(self, latents: Any, t: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"UpscalePluginBackbone({self.spec.family_id!r}): forward delegates to job runner"
        )
