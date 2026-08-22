"""Backward-compatible re-exports (implementation: service.py + llm_settings.py)."""

from backend.engine.llm.llm_settings import (  # noqa: F401
    DEFAULT_ASSISTANT_MODEL_ID,
    DEFAULT_LLM_MODEL_ID,
    DEFAULT_VLM_MODEL_ID,
    normalize_app_llm_settings,
    resolve_assistant_model_id,
    resolve_llm_model_id,
    resolve_vlm_model_id,
)
from backend.engine.llm.service import LLMService

__all__ = [
    "DEFAULT_ASSISTANT_MODEL_ID",
    "DEFAULT_LLM_MODEL_ID",
    "DEFAULT_VLM_MODEL_ID",
    "LLMService",
    "normalize_app_llm_settings",
    "resolve_assistant_model_id",
    "resolve_llm_model_id",
    "resolve_vlm_model_id",
]
