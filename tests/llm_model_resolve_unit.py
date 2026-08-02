"""Unit tests for LLM model selection (chapter analyze + thinking detection)."""

from __future__ import annotations

import unittest
from pathlib import Path

from backend.core.contracts import LongVideoChapterAnalyzeRequest
from backend.core.model_registry import ModelRegistry
from backend.engine.llm.service_mlx import LLMService
from backend.utils.path_utils import PathResolver

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "default_config" / "models_registry.json"


class LlmModelResolveUnit(unittest.TestCase):
    def test_chapter_analyze_request_model_field(self) -> None:
        req = LongVideoChapterAnalyzeRequest(
            chapter_text="sample",
            model="qwen3.6-27b",
        )
        self.assertEqual(req.model, "qwen3.6-27b")

    def test_qwen36_is_thinking_model(self) -> None:
        self.assertTrue(LLMService._is_thinking_model("qwen3.6-27b"))

    def test_resolve_request_llm_model_empty_uses_default(self) -> None:
        reg = ModelRegistry.load(REGISTRY)
        svc = LLMService(
            reg,
            PathResolver(project_root=ROOT),
            default_model_id="qwen3.5-4b",
        )
        self.assertEqual(svc._resolve_request_llm_model(""), "qwen3.5-4b")
        self.assertEqual(svc._resolve_request_llm_model(None), "qwen3.5-4b")

    def test_resolve_request_llm_model_coerces_registry_id(self) -> None:
        reg = ModelRegistry.load(REGISTRY)
        svc = LLMService(
            reg,
            PathResolver(project_root=ROOT),
            default_model_id="qwen3.5-4b",
        )
        self.assertEqual(svc._resolve_request_llm_model("qwen3.6-27b"), "qwen3.6-27b")


if __name__ == "__main__":
    unittest.main()
