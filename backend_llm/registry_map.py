"""Resolve registry model ids to local paths for mlx_vlm.server lazy loading."""

from __future__ import annotations

from pathlib import Path

from shared.danqing_config.llm import (
    build_sidecar_registry_paths,
    coerce_assistant_model_id,
    require_multimodal_assistant_model,
)
from shared.danqing_config.registry import ModelRecord, load_registry
from shared.danqing_config.settings import AppSettingsSnapshot, load_app_settings
from shared.danqing_config.paths import models_registry_path, workspace_root


def load_registry_models() -> dict[str, ModelRecord]:
    path = models_registry_path()
    if not path.is_file():
        raise RuntimeError(f"models_registry.json not found at {path}")
    return load_registry(path)


def assistant_model_map(settings: AppSettingsSnapshot | None = None) -> dict[str, Path]:
    snap = settings or load_app_settings()
    models = load_registry_models()
    return build_sidecar_registry_paths(snap, models, workspace=workspace_root())


def default_assistant_id(settings: AppSettingsSnapshot | None = None) -> str:
    snap = settings or load_app_settings()
    return coerce_assistant_model_id(snap.default_model_llm, load_registry_models())


def default_llm_id(settings: AppSettingsSnapshot | None = None) -> str:
    return default_assistant_id(settings)


def default_vlm_id(settings: AppSettingsSnapshot | None = None) -> str:
    return default_assistant_id(settings)


def validate_assistant_model(settings: AppSettingsSnapshot | None = None) -> None:
    snap = settings or load_app_settings()
    models = load_registry_models()
    mid = coerce_assistant_model_id(snap.default_model_llm, models)
    require_multimodal_assistant_model(mid, models)
