"""Unit tests for input-driven parse quality checks."""
from __future__ import annotations

import unittest

from backend.long_video.parse_quality import (
    _motion_core,
    protagonist_names_from_anchor,
    validate_parse_quality,
)
from backend.long_video.shot_contract_validator import validate_shot_contracts


ANCHOR = (
    "【角色·林晓·默认】定位：lead | 外貌：年轻女性\n"
    "---\n"
    "【角色·老周·默认】定位：supporting | 外貌：老年男性"
)


class ParseQualityTests(unittest.TestCase):
    def test_protagonist_from_structured_anchor(self):
        names = protagonist_names_from_anchor(ANCHOR)
        self.assertEqual(names[0], "林晓")

    def test_opening_check_uses_structured_protagonist(self):
        shots = [
            {
                "duration_sec": 5.0,
                "characters_on_screen": ["林晓"],
                "first_frame_visibility": "invisible",
                "start_frame_mode": "keyframe",
                "start_visual_prompt": "林晓在门厅",
                "video_prompt": "林晓走向门口",
                "flf_mode": "none",
                "segment_group_id": "beat_0",
            }
        ]
        result = validate_shot_contracts(shots, character_anchor=ANCHOR)
        self.assertFalse(result.ok)
        self.assertTrue(any(i.code == "opening_no_protagonist" for i in result.issues))

    def test_motion_duplicate_in_group(self):
        shots = [
            {
                "segment_group_id": "beat_0",
                "segment_role": "pre_anchor",
                "duration_sec": 3.0,
                "video_prompt": "Alice walks slowly toward the door in cold light",
                "characters_on_screen": ["Alice"],
                "flf_mode": "none",
            },
            {
                "segment_group_id": "beat_0",
                "segment_role": "face_anchor",
                "duration_sec": 2.0,
                "video_prompt": "Alice walks slowly toward the door in cold light",
                "characters_on_screen": ["Alice"],
                "flf_mode": "none",
            },
        ]
        result = validate_parse_quality(
            shots,
            beat_sheet=["开场 | 中景 | 室内 | Alice walks toward the door"],
            character_anchor="【角色·Alice·默认】定位：lead | 外貌：年轻女性",
            character_dtos=[{"name": "Alice", "id": "c1", "looks": []}],
        )
        codes = {i.code for i in result.issues}
        self.assertIn("motion_duplicate_in_group", codes)

    def test_roster_unknown_on_screen(self):
        shots = [
            {
                "segment_group_id": "beat_0",
                "duration_sec": 4.0,
                "video_prompt": "Bob enters",
                "characters_on_screen": ["Bob"],
                "flf_mode": "none",
            }
        ]
        result = validate_parse_quality(
            shots,
            beat_sheet=["入场 | 中景 | 大厅 | Bob enters"],
            character_anchor="【角色·Alice·默认】定位：lead | 外貌：年轻女性",
            character_dtos=[{"name": "Alice", "id": "c1", "looks": []}],
        )
        self.assertTrue(any(i.code == "roster_shot_unknown_character" for i in result.issues))

    def test_beat_no_shots(self):
        result = validate_parse_quality(
            [{"segment_group_id": "beat_0", "duration_sec": 4.0, "video_prompt": "x", "flf_mode": "none"}],
            beat_sheet=["a | 中景 | loc | narrative one", "b | 中景 | loc | narrative two"],
        )
        self.assertTrue(any(i.code == "beat_no_shots" for i in result.issues))

    def test_motion_core_strips_style_suffix(self):
        style = "冷色调夜景，高对比度明暗分割，数字噪点纹理"
        raw = f"Hero walks toward door，{style}，持续 3.0 秒"
        core = _motion_core(raw, style_anchor=style)
        self.assertNotIn("冷色调", core)
        self.assertIn("hero walks toward door", core)

    def test_motion_duplicate_with_style_boilerplate_only(self):
        style = "冷色调夜景，高对比度明暗分割，数字噪点纹理"
        motion = f"Alice walks slowly toward the door，{style}，持续 3.0 秒"
        shots = [
            {
                "segment_group_id": "beat_0",
                "segment_role": "pre_anchor",
                "duration_sec": 3.0,
                "video_prompt": motion,
                "flf_mode": "none",
            },
            {
                "segment_group_id": "beat_0",
                "segment_role": "face_anchor",
                "duration_sec": 2.0,
                "video_prompt": motion.replace("3.0", "2.0"),
                "flf_mode": "none",
            },
        ]
        result = validate_parse_quality(
            shots,
            beat_sheet=["开场 | 中景 | 室内 | Alice walks toward the door"],
            style_anchor=style,
        )
        self.assertIn("motion_duplicate_in_group", {i.code for i in result.issues})

    def test_strip_style_from_motion_prompt(self):
        from backend.long_video.parse_quality import (
            strip_style_from_motion_prompt,
            validate_parse_quality,
        )

        style = "冷蓝色调与昏黄应急灯光对比，手持镜头营造不安感，胶片颗粒质感"
        raw = f"{style}；林晓坐在昏暗工作台前，揉揉眉心后推门走入雨幕"
        cleaned = strip_style_from_motion_prompt(raw, style_anchor=style)
        self.assertNotIn("胶片颗粒", cleaned)
        self.assertIn("林晓", cleaned)
        spam = f"林晓走向门口，{style}"
        shots = [
            {
                "segment_group_id": f"beat_{i // 3}",
                "duration_sec": 3.0,
                "video_prompt": strip_style_from_motion_prompt(spam, style_anchor=style),
                "flf_mode": "none",
            }
            for i in range(6)
        ]
        result = validate_parse_quality(
            shots,
            beat_sheet=["a | 中景 | loc | 林晓 walks", "b | 中景 | loc | 林晓 continues"],
            style_anchor=style,
        )
        codes = {i.code for i in result.issues}
        self.assertNotIn("style_phrase_spam", codes)

    def test_style_phrase_spam(self):
        style = "冷色调夜景，高对比度明暗分割"
        spam = f"Alice walks slowly，{style}，数字噪点纹理"
        shots = [
            {
                "segment_group_id": f"beat_{i // 3}",
                "duration_sec": 3.0,
                "video_prompt": spam,
                "flf_mode": "none",
            }
            for i in range(6)
        ]
        result = validate_parse_quality(
            shots,
            beat_sheet=["a | 中景 | loc | Alice walks", "b | 中景 | loc | Alice continues"],
            style_anchor=style,
        )
        codes = {i.code for i in result.issues}
        self.assertTrue(codes & {"style_phrase_spam", "prompt_fragment_spam"})

    def test_cast_look_missing(self):
        shots = [
            {
                "segment_group_id": "beat_0",
                "duration_sec": 4.0,
                "video_prompt": "Hero enters",
                "characters_on_screen": ["Hero"],
                "cast_looks": [],
                "flf_mode": "none",
            }
        ]
        result = validate_parse_quality(
            shots,
            beat_sheet=["入场 | 中景 | 大厅 | Hero enters"],
            character_dtos=[{"name": "Hero", "id": "c1", "looks": []}],
        )
        self.assertTrue(any(i.code == "cast_look_missing" for i in result.issues))

    def test_intra_beat_visibility_jump(self):
        shots = [
            {
                "segment_group_id": "beat_1",
                "segment_role": "pre_anchor",
                "duration_sec": 3.0,
                "first_frame_visibility": "silhouette",
                "end_visibility": "silhouette",
                "characters_on_screen": ["Hero"],
                "video_prompt": "Hero approaches",
                "start_frame_mode": "keyframe",
                "flf_mode": "none",
            },
            {
                "segment_group_id": "beat_1",
                "segment_role": "face_anchor",
                "duration_sec": 2.0,
                "first_frame_visibility": "full_face",
                "characters_on_screen": ["Hero"],
                "video_prompt": "Hero face close-up",
                "start_frame_mode": "keyframe",
                "flf_mode": "none",
            },
        ]
        result = validate_parse_quality(
            shots,
            beat_sheet=["对决 | 特写 | 山顶 | Hero faces the enemy"],
        )
        self.assertTrue(any(i.code == "intra_beat_visibility_jump" for i in result.issues))


    def test_visual_inline_look_tag_warning(self) -> None:
        shots = [
            {
                "visual_prompt": "赵今麦（白T恤）坐在床边",
                "video_prompt": "赵今麦刷手机",
                "duration_sec": 5.0,
                "segment_group_id": "beat_0",
            }
        ]
        result = validate_parse_quality(shots, beat_sheet=["卧室 | 中景 | 卧室 | 赵今麦刷手机"])
        self.assertTrue(any(i.code == "visual_inline_look_tag" for i in result.issues))

    def test_scene_prompt_inline_look_tag_warning(self) -> None:
        shots = [
            {
                "scene_prompt": "赵今麦（睡衣）刷手机",
                "video_prompt": "赵今麦刷手机",
                "duration_sec": 5.0,
                "segment_group_id": "beat_0",
            }
        ]
        result = validate_parse_quality(shots, beat_sheet=["卧室 | 中景 | 卧室 | 赵今麦刷手机"])
        self.assertTrue(any(i.code == "scene_inline_look_tag" for i in result.issues))

    def test_beat_narrative_coverage_uses_start_visual(self) -> None:
        shots = [
            {
                "segment_group_id": "beat_0",
                "duration_sec": 5.0,
                "scene_prompt": "城郊旧公寓，窗外雨夜霓虹倒影",
                "video_prompt": "雨水沿玻璃滑落，镜头缓慢前推",
                "start_visual_prompt": "林晓坐在工作台前低头读泛黄信件",
                "characters_on_screen": ["林晓"],
                "flf_mode": "none",
            },
        ]
        beat = "雨夜读信 | 远景 | 城郊旧公寓 | 窗外秋雨连绵，林晓坐在工作台研读匿名信"
        result = validate_parse_quality(shots, beat_sheet=[beat])
        codes = {i.code for i in result.issues}
        self.assertNotIn("beat_narrative_undercovered", codes)


if __name__ == "__main__":
    unittest.main()
