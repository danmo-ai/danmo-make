"""Unit tests for LLM assistant model selection (single multimodal sidecar)."""

from __future__ import annotations

import unittest
from pathlib import Path

from backend.core.interfaces import AppSettings
from backend.core.model_registry import ModelRegistry
from backend.engine.llm.llm_settings import (
    assistant_model_not_multimodal_message,
    require_multimodal_assistant_model,
    resolve_assistant_model_id,
)
from backend.engine.llm.service_mlx import LLMService
from backend.utils.path_utils import PathResolver
from shared.danqing_config.llm import is_thinking_model

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "default_config" / "models_registry.json"


class LlmModelResolveUnit(unittest.TestCase):
    def test_qwen36_is_thinking_model(self) -> None:
        self.assertTrue(is_thinking_model("qwen3.6-27b"))

    def test_resolve_assistant_model_unknown_falls_back_to_default_vlm(self) -> None:
        reg = ModelRegistry.load(REGISTRY)
        settings = AppSettings(default_model_llm="missing-model-id")
        self.assertEqual(resolve_assistant_model_id(settings, reg), "qwen3-vl-4b-instruct")

    def test_require_multimodal_rejects_text_only_llm(self) -> None:
        reg = ModelRegistry.load(REGISTRY)
        with self.assertRaises(RuntimeError) as ctx:
            require_multimodal_assistant_model("qwen3.5-4b", reg)
        self.assertIn("not multimodal", str(ctx.exception).lower())
        self.assertIn("qwen3.5-4b", assistant_model_not_multimodal_message("qwen3.5-4b"))

    def test_require_multimodal_accepts_vlm(self) -> None:
        reg = ModelRegistry.load(REGISTRY)
        require_multimodal_assistant_model("qwen3-vl-4b-instruct", reg)

    def test_llm_memory_policy_defaults(self) -> None:
        reg = ModelRegistry.load(REGISTRY)
        svc = LLMService(
            reg,
            PathResolver(project_root=ROOT),
            default_model_id="qwen3-vl-4b-instruct",
        )
        self.assertFalse(svc._unload_each_request)
        self.assertGreaterEqual(svc._llm_cache_ttl_minutes, 1)

    def test_unload_each_request_skips_sidecar_unload(self) -> None:
        reg = ModelRegistry.load(REGISTRY)
        svc = LLMService(
            reg,
            PathResolver(project_root=ROOT),
            default_model_id="qwen3-vl-4b-instruct",
            unload_each_request=True,
        )
        svc.unload_text_model()


if __name__ == "__main__":
    unittest.main()
