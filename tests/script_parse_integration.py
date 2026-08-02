"""E2E integration tests for script-parse pipeline (local LLM, no HTTP server)."""
from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from backend.core.contracts import ScriptParseDecomposeRequest, ScriptParseExpandRequest

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "chapter_scripts"

SAMPLE_CHAPTER_NARRATIVE = """
第一章 雨夜访客

林晓住在城郊一栋旧公寓的三楼。这天深夜，窗外下着连绵的秋雨，霓虹灯在积水的路面上拉出长长的倒影。

林晓坐在狭小的工作台前，面前摊着一封泛黄的信。信上没有署名，只写着一行字：「今晚子时，老仓库见。」她揉了揉眉心，决定穿上那件深蓝色的连帽外套出门。

老仓库位于废弃的港区，铁门锈迹斑斑。林晓推开吱呀作响的门，昏黄的应急灯忽明忽暗。角落里站着的老周——她的前导师——正用拐杖轻轻敲击地面。

「你终于来了。」老周沉声说，「他们要在黎明前运走那批证据。」

林晓握紧背包带，跟随老周走向仓库深处。铁皮货架之间，成箱的档案在冷白灯下泛着灰光。忽然，门外传来脚步声。两人迅速躲进阴影，林晓屏住呼吸，从货架缝隙看见两名黑衣护卫手持电筒扫过通道。

对峙持续至拂晓前的最暗时刻。林晓与老周交换眼神，她缓步走出，正面迎向领头的护卫，而老周则从侧翼掀开了隐藏在后墙的通风口——那里通向港区的旧栈道。

当第一缕晨光刺破海平面时，林晓已在栈道上奔跑，怀中紧紧抱着从箱中取出的硬盘。雨停了，远处城市的轮廓在晨曦中渐渐清晰。
""".strip()


def _resolve_parse_model(svc) -> str:
    for mid in ("qwen3.6-27b", "qwen3.5-4b"):
        try:
            path = svc._resolve_model_path(mid)
            if svc._llm_weights_ready(path):
                return mid
        except Exception:
            continue
    return svc._model_id


def _entity_report(decomposed, expanded) -> str:
    char_names = [c.name for c in decomposed.characters]
    scene_names = [s.name for s in decomposed.scenes]
    beat_locs = [b.get("location", "") for b in decomposed.script_artifact.get("beats", [])]
    lines = [
        f"model beats={len(decomposed.script_artifact.get('beats', []))} scenes={decomposed.scene_count}",
        f"characters: {char_names}",
        f"scenes: {scene_names}",
        f"beat locations: {beat_locs}",
        f"shots={len(expanded.shots)} llm_calls={expanded.llm_calls}",
        f"quality_issues={[i.code for i in expanded.quality_issues]}",
    ]
    return "\n".join(lines)


def _assert_rainy_night_entities(test: unittest.TestCase, decomposed, expanded) -> None:
    char_names = [c.name for c in decomposed.characters]
    scene_blob = " ".join(s.name for s in decomposed.scenes)
    test.assertTrue(any("林晓" in n for n in char_names), f"missing protagonist 林晓 in {char_names}")
    test.assertTrue(any("老周" in n for n in char_names), f"missing 老周 in {char_names}")
    for keyword in ("公寓", "仓库"):
        test.assertIn(keyword, scene_blob, f"scene names missing {keyword!r}: {scene_blob}")
    beat_count = len(decomposed.script_artifact.get("beats", []))
    test.assertGreaterEqual(beat_count, 4, "expected at least 4 beats for sample chapter")
    test.assertGreaterEqual(len(expanded.shots), beat_count, "expected at least one shot per beat")
    critical = [i for i in expanded.quality_issues if i.severity == "critical"]
    test.assertEqual(critical, [], _entity_report(decomposed, expanded))


def _run_expand(svc, decomposed, *, parse_model: str):
    return svc.script_parse_expand(
        ScriptParseExpandRequest(
            script_artifact=decomposed.script_artifact,
            locale="zh",
            target_duration_sec=60.0,
            segment_duration_sec=5.0,
            max_clip_sec=10.0,
            model=parse_model,
        ),
    )


def _assert_wukong_entities(test: unittest.TestCase, decomposed, expanded) -> None:
    char_names = [c.name for c in decomposed.characters]
    test.assertTrue(any("赵今麦" in n for n in char_names), f"missing 赵今麦 in {char_names}")
    test.assertTrue(any("孙悟空" in n for n in char_names), f"missing 孙悟空 in {char_names}")
    test.assertGreaterEqual(len(decomposed.script_artifact.get("beats", [])), 4)
    test.assertGreater(len(expanded.shots), 0, "expand must produce shots")
    critical = [i for i in expanded.quality_issues if i.severity == "critical"]
    test.assertEqual(critical, [], _entity_report(decomposed, expanded))


def _load_llm_service():
    from backend.core.interfaces import AppSettings
    from backend.core.model_registry import ModelRegistry
    from backend.engine.llm.service_mlx import (
        LLMService,
        normalize_app_llm_settings,
        resolve_llm_model_id,
        resolve_vlm_model_id,
    )
    from backend.utils.path_utils import PathResolver

    root = Path(__file__).resolve().parents[1]
    pr = PathResolver(root)
    mr = ModelRegistry.load(pr.get_models_registry_path())
    settings = AppSettings()
    normalize_app_llm_settings(settings, mr)
    svc = LLMService(
        mr,
        pr,
        default_model_id=resolve_llm_model_id(settings, mr),
        vision_model_id=resolve_vlm_model_id(settings, mr),
        llm_think_enabled=settings.default_model_llm_think,
    )
    return svc


class ScriptParseIntegrationTests(unittest.TestCase):
    def test_rejects_empty_script_without_llm(self) -> None:
        svc = _load_llm_service()
        with self.assertRaises(RuntimeError):
            svc.script_parse_decompose(
                ScriptParseDecomposeRequest(script_text="   ", locale="zh"),
            )

    def test_decompose_and_expand_e2e(self) -> None:
        svc = _load_llm_service()
        if not svc.is_available():
            self.skipTest("no local LLM installed")

        parse_model = _resolve_parse_model(svc)
        decomposed = svc.script_parse_decompose(
            ScriptParseDecomposeRequest(
                script_text=SAMPLE_CHAPTER_NARRATIVE,
                title="雨夜访客",
                locale="zh",
                model=parse_model,
            ),
        )
        self.assertGreaterEqual(len(decomposed.synopsis.strip()), 12)
        self.assertGreaterEqual(decomposed.scene_count, 2)
        self.assertTrue(decomposed.script_artifact.get("version") == "2.0")
        self.assertGreaterEqual(decomposed.llm_calls, 1)

        expanded = _run_expand(svc, decomposed, parse_model=parse_model)
        self.assertGreater(len(expanded.shots), 0)
        self.assertGreaterEqual(expanded.llm_calls, 2)
        for shot in expanded.shots:
            dur = float(shot.duration_sec or 0)
            self.assertGreaterEqual(dur, 2.0)
            self.assertLessEqual(dur, 10.0)
            self.assertNotEqual(shot.flf_mode, "first_last")
        _assert_rainy_night_entities(self, decomposed, expanded)
        print(f"\n[script_parse e2e ok model={parse_model}]\n{_entity_report(decomposed, expanded)}")

    def test_wukong_decompose_and_expand_e2e(self) -> None:
        """大战悟空 — partial visibility + 无面部 camera zones (regression for beat_plan/shot_spec)."""
        svc = _load_llm_service()
        if not svc.is_available():
            self.skipTest("no local LLM installed")

        script_path = FIXTURES / "wukong_battle.txt"
        script_text = script_path.read_text(encoding="utf-8").strip()
        parse_model = _resolve_parse_model(svc)
        decomposed = svc.script_parse_decompose(
            ScriptParseDecomposeRequest(
                script_text=script_text,
                title="大战悟空",
                locale="zh",
                model=parse_model,
            ),
        )
        expanded = _run_expand(svc, decomposed, parse_model=parse_model)
        self.assertGreater(len(expanded.shots), 0)
        _assert_wukong_entities(self, decomposed, expanded)
        print(f"\n[wukong e2e ok model={parse_model}]\n{_entity_report(decomposed, expanded)}")

    def test_decompose_route_e2e(self) -> None:
        svc = _load_llm_service()
        if not svc.is_available():
            self.skipTest("no local LLM installed")

        from backend.api.routes.script_parse import script_parse_decompose

        http_request = MagicMock()
        http_request.headers.get.return_value = "zh-CN"

        resp = asyncio.run(
            script_parse_decompose(
                ScriptParseDecomposeRequest(
                    script_text=SAMPLE_CHAPTER_NARRATIVE,
                    title="雨夜访客",
                    locale="zh",
                ),
                http_request,
                service=svc,
                activity_store=MagicMock(),
            ),
        )
        self.assertGreaterEqual(resp.scene_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
