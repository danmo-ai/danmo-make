"""LLM service — local inference via mlx-lm (standalone, not through TaskScheduler)."""

from backend.engine.llm.service_mlx import LLMService

__all__ = ["LLMService"]
