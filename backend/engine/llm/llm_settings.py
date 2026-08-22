"""LLM assistant default model coercion aligned with models_registry."""

from __future__ import annotations

import logging

from backend.core.interfaces import AppSettings
from backend.core.model_registry import ModelEntry, ModelRegistry

logger = logging.getLogger(__name__)

DEFAULT_ASSISTANT_MODEL_ID = "qwen3-vl-4b-instruct"
DEFAULT_LLM_MODEL_ID = DEFAULT_ASSISTANT_MODEL_ID
DEFAULT_VLM_MODEL_ID = DEFAULT_ASSISTANT_MODEL_ID


def _is_multimodal_assistant_entry(entry: ModelEntry | None) -> bool:
    if entry is None or entry.media != "llm":
        return False
    return "describe" in entry.actions


def _pick_first_assistant(registry: ModelRegistry) -> str | None:
    for mid in sorted(registry.all()):
        if _is_multimodal_assistant_entry(registry.get(mid)):
            return mid
    return None


def _resolve_assistant_model_id(preferred: str, registry: ModelRegistry) -> str:
    """Resolve registry id; unknown ids fall back to the default multimodal model."""
    candidate = (preferred or "").strip().split(":", 1)[0]
    if candidate and registry.get(candidate) is not None:
        return candidate
    fallback = DEFAULT_ASSISTANT_MODEL_ID
    if _is_multimodal_assistant_entry(registry.get(fallback)):
        return fallback
    picked = _pick_first_assistant(registry)
    return picked or fallback


def assistant_model_not_multimodal_message(model_id: str) -> str:
    mid = (model_id or "").strip().split(":", 1)[0]
    return (
        f"Assistant model {mid!r} is not multimodal. "
        "The LLM sidecar loads one model at a time; choose a VLM with vision support "
        f"(default: {DEFAULT_ASSISTANT_MODEL_ID!r})."
    )


def require_multimodal_assistant_model(model_id: str, registry: ModelRegistry) -> None:
    mid = (model_id or "").strip().split(":", 1)[0]
    if not _is_multimodal_assistant_entry(registry.get(mid)):
        raise RuntimeError(assistant_model_not_multimodal_message(mid))


def normalize_app_llm_settings(settings: AppSettings, registry: ModelRegistry) -> bool:
    changed = False
    coerced = _resolve_assistant_model_id(settings.default_model_llm, registry)
    if settings.default_model_llm != coerced:
        if (settings.default_model_llm or "").strip():
            logger.warning(
                "default_model_llm %r not in registry; using %r",
                settings.default_model_llm,
                coerced,
            )
        settings.default_model_llm = coerced
        changed = True

    if settings.default_model_vlm != settings.default_model_llm:
        settings.default_model_vlm = settings.default_model_llm
        changed = True
    return changed


def resolve_assistant_model_id(settings: AppSettings, registry: ModelRegistry) -> str:
    return _resolve_assistant_model_id(settings.default_model_llm, registry)


def resolve_llm_model_id(settings: AppSettings, registry: ModelRegistry) -> str:
    return resolve_assistant_model_id(settings, registry)


def resolve_vlm_model_id(settings: AppSettings, registry: ModelRegistry) -> str:
    return resolve_assistant_model_id(settings, registry)
