"""Shared image FamilyPlugin backbone — loads DiT via ``load_image_transformer``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.engine.config.model_configs import (
    apply_image_bundle_config_merger,
    apply_image_registry_config_overrides,
    get_config_class,
)
from backend.engine.pipelines.image_model_load import load_image_transformer
from backend.engine.platform.session import PlatformSession
from backend.engine.protocols.bundle import MediaBundle
from backend.engine.protocols.plugin import FamilyPlugin, FamilySpec


def plugin_backbone_model_if_ready(plugin: FamilyPlugin | None, *, request: Any) -> Any | None:
    """Return plugin-preloaded DiT when safe to reuse (no per-request LoRA reload)."""
    if plugin is None:
        return None
    if getattr(request, "adapters", None):
        return None
    backbone = plugin.backbone
    if getattr(backbone, "_skip_load", False):
        return None
    model = getattr(backbone, "_model", None)
    if model is not None:
        return model
    getter = getattr(type(backbone), "model", None)
    if isinstance(getter, property):
        try:
            return backbone.model  # type: ignore[attr-defined]
        except RuntimeError:
            return None
    return None


class ImagePluginBackbone:
    """Image plugin backbone — loads DiT; infer routes through ``DiffusionParadigm``."""

    def __init__(self, spec: FamilySpec) -> None:
        self.spec = spec
        self._model: Any | None = None
        self._registry_entry: Any | None = None
        self._project_root: Path | None = None
        self._model_cache: Any | None = None
        self._skip_load = False

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError(
                f"ImagePluginBackbone({self.spec.family_id!r}): call load() before forward"
            )
        return self._model

    def bind_load_context(
        self,
        *,
        registry_entry: Any,
        project_root: Path,
        model_cache: Any | None = None,
        bundle_root: Path | None = None,
        request: Any = None,
    ) -> bool:
        _ = bundle_root, request
        self._registry_entry = registry_entry
        self._project_root = project_root
        self._model_cache = model_cache
        config = get_config_class(self.spec.family_id)()
        apply_image_registry_config_overrides(registry_entry, config)
        if (
            str(getattr(config, "image_pipeline_shape", "dit_standard") or "dit_standard")
            == "family_generator"
        ):
            self._skip_load = True
            return False
        return True

    def load(self, bundle: MediaBundle, platform: PlatformSession) -> None:
        if self._skip_load:
            return
        if self._registry_entry is None or self._project_root is None:
            raise RuntimeError(
                f"ImagePluginBackbone({self.spec.family_id!r}): bind_load_context() required"
            )
        family = self.spec.family_id
        config = get_config_class(family)()
        apply_image_registry_config_overrides(self._registry_entry, config)
        apply_image_bundle_config_merger(config, bundle.root)
        self._model = load_image_transformer(
            ctx=platform.kernels,
            family=family,
            config=config,
            entry=self._registry_entry,
            version_key=bundle.version_key,
            project_root=self._project_root,
            model_cache=self._model_cache,
        )
        if self._model is None:
            raise RuntimeError(
                f"ImagePluginBackbone({family!r}): transformer weights missing under {bundle.root}"
            )

    def after_load(self, bundle: MediaBundle) -> None:
        if self._model is not None:
            self._model.after_load_weights(bundle_root=str(bundle.root))

    def forward(self, latents: Any, t: Any, **kwargs: Any) -> Any:
        return self.model(latents, t, **kwargs)
