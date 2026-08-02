"""Unit tests for visibility-driven T2I prompt policy."""
from __future__ import annotations

import unittest

from backend.long_video.keyframe_prompt_policy import (
    cast_reference_scope,
    should_merge_scene_prompt_into_t2i,
    validate_visibility_role_contract,
)
from backend.engine.llm.script_parse.schemas import (
    BeatPlanLLMSchema,
    BeatPlanRowLLM,
    BeatPlanSegmentLLM,
    FiveAspectArtifact,
    ShotLanguageArtifact,
    ShotSpecArtifact,
    ShotSpecRowLLM,
)


class KeyframePromptPolicyTests(unittest.TestCase):
    def test_cast_scope_from_visibility(self) -> None:
        self.assertEqual(cast_reference_scope(visibility="partial", segment_role="keyframe"), "wardrobe")
        self.assertEqual(cast_reference_scope(visibility="full_face", segment_role="face_anchor"), "face")
        self.assertEqual(cast_reference_scope(visibility="full_face", segment_role="keyframe"), "face")
        self.assertEqual(
            cast_reference_scope(visibility="invisible", segment_role="establishing", is_intentional_empty=True),
            "none",
        )

    def test_structured_start_visual_blocks_scene_merge(self) -> None:
        self.assertFalse(should_merge_scene_prompt_into_t2i(start_visual="手指特写", scene_prompt="卧室环境"))

    def test_visibility_role_contract(self) -> None:
        issues = validate_visibility_role_contract(
            segment_role="keyframe",
            start_visibility="full_face",
            beat_index=0,
        )
        self.assertEqual(len(issues), 1)
        self.assertIn("face_anchor", issues[0])

    def test_shot_spec_rejects_non_anchor_full_face(self) -> None:
        with self.assertRaises(ValueError):
            ShotSpecArtifact(
                segment_index=1,
                beat_index=0,
                role="keyframe",
                five_aspect=FiveAspectArtifact(
                    subject="赵今麦手指与手机",
                    subject_motion="按下确认",
                    scene="卧室",
                    spatial_framing="极特写",
                    camera="static",
                ),
                shot_language=ShotLanguageArtifact(camera_movement="static"),
                video_prompt="手指按下",
                start_visual="赵今麦手指悬停在手机上方",
                characters_on_screen=["赵今麦"],
                start_visibility="full_face",
            )

    def test_sanitize_strips_anchor_visual_on_keyframe(self) -> None:
        from backend.long_video.keyframe_prompt_policy import sanitize_shot_spec_prompts

        sv, av = sanitize_shot_spec_prompts(
            role="keyframe",
            start_visual="赵今麦手指",
            anchor_visual="赵今麦面部",
        )
        self.assertEqual(sv, "赵今麦手指")
        self.assertEqual(av, "")

    def test_coalesce_face_anchor_from_start_visual(self) -> None:
        from backend.long_video.keyframe_prompt_policy import coalesce_face_anchor_visual

        sv, av = coalesce_face_anchor_visual(
            anchor_visual="赵今麦",
            start_visual="面部特写，眼神惊恐，红色通知文字清晰可见",
            five_aspect_subject="赵今麦",
            primary_name="赵今麦",
        )
        self.assertIn("面部特写", av)
        self.assertIn("眼神惊恐", av)

    def test_normalize_face_anchor_single_character(self) -> None:
        from backend.long_video.keyframe_prompt_policy import (
            normalize_face_anchor_characters_on_screen,
        )

        self.assertEqual(
            normalize_face_anchor_characters_on_screen("face_anchor", ["赵今麦", "判官"]),
            ["赵今麦"],
        )
        self.assertEqual(
            normalize_face_anchor_characters_on_screen("keyframe", ["赵今麦", "判官"]),
            ["赵今麦", "判官"],
        )

    def test_beat_plan_rejects_multi_face_anchor_characters(self) -> None:
        from backend.long_video.keyframe_prompt_policy import validate_beat_plan_row_contract

        class Seg:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        issues = validate_beat_plan_row_contract(
            beat_index=4,
            segments=[
                Seg(
                    role="face_anchor",
                    start_visibility="full_face",
                    characters_on_screen=["赵今麦", "判官"],
                    first_frame_requirement="",
                    spatial=None,
                ),
            ],
        )
        self.assertTrue(any("one character on screen" in i for i in issues))

    def test_shot_spec_rejects_anchor_visual_on_keyframe(self) -> None:
        with self.assertRaises(ValueError):
            ShotSpecArtifact(
                segment_index=1,
                beat_index=0,
                role="keyframe",
                five_aspect=FiveAspectArtifact(
                    subject="赵今麦",
                    subject_motion="n",
                    scene="卧室",
                    spatial_framing="特写",
                    camera="static",
                ),
                shot_language=ShotLanguageArtifact(camera_movement="static"),
                video_prompt="static",
                start_visual="赵今麦手指",
                anchor_visual="赵今麦面部",
                characters_on_screen=["赵今麦"],
                start_visibility="partial",
            )


class CameraZonePartialVisibilityTests(unittest.TestCase):
    def test_face_hidden_zones_do_not_conflict(self) -> None:
        from backend.long_video.keyframe_prompt_policy import (
            camera_zone_conflicts_with_partial_visibility,
        )

        cases = [
            "手机屏幕及手指，面部仅露出轮廓或不可见。",
            "分身群与赵今麦的肢体轮廓，面部不可见。",
            "肢体轮廓与碎片，面部完全不可见。",
            "手机屏幕与手指细节，无面部",
            "腿部动作与外套细节，无面部",
            "躯干与破碎特效，无面部",
            "手部、面部、破碎镜面",
        ]
        for area in cases:
            with self.subTest(area=area):
                self.assertFalse(camera_zone_conflicts_with_partial_visibility(area))

    def test_face_demand_zones_conflict(self) -> None:
        from backend.long_video.keyframe_prompt_policy import (
            camera_zone_conflicts_with_partial_visibility,
        )

        self.assertTrue(camera_zone_conflicts_with_partial_visibility("面部特写"))
        self.assertTrue(camera_zone_conflicts_with_partial_visibility("五官清晰可读"))


class BeatPlanVisibilityValidationTests(unittest.TestCase):
    def test_partial_with_face_zone_fails(self) -> None:
        from backend.engine.llm.script_parse.beat_plan import _validate_beat_plan
        from backend.engine.llm.script_parse.schemas import (
            CameraZoneSnippet,
            ScriptArtifact,
            ScriptBeatArtifact,
            ScriptCharacterArtifact,
            ScriptSceneArtifact,
            SpatialSnippet,
        )

        script = ScriptArtifact(
            title="t",
            synopsis="赵今麦在卧室收到挑战通知后踏上云雾山。",
            mood="m",
            style_anchor="",
            beats=[
                ScriptBeatArtifact(
                    index=0,
                    title="b0",
                    location="卧室",
                    narrative="n0",
                    enhancement_cues=["cue"],
                    suggested_shot_size="特写",
                    estimated_duration_sec=8.0,
                ),
                ScriptBeatArtifact(
                    index=1,
                    title="b1",
                    location="山径",
                    narrative="n1",
                    enhancement_cues=["cue"],
                    suggested_shot_size="远景",
                    estimated_duration_sec=8.0,
                ),
            ],
            characters=[
                ScriptCharacterArtifact(
                    name="赵今麦",
                    role="protagonist",
                    looks=[{"label": "常服", "body": "短发"}],
                ),
            ],
            scenes=[ScriptSceneArtifact(name="卧室", looks=[{"label": "夜", "body": "室内"}])],
        )
        payload = BeatPlanLLMSchema(
            beats=[
                BeatPlanRowLLM(
                    beat_index=0,
                    shot_intent="intent",
                    narrative_role="establish_context",
                    segments=[
                        BeatPlanSegmentLLM(
                            role="keyframe",
                            duration_sec=4.0,
                            characters_on_screen=["赵今麦"],
                            start_visibility="partial",
                            spatial=SpatialSnippet(
                                location="卧室",
                                camera_zones=[
                                    CameraZoneSnippet(id="CZ2", description="d", visible_area="面部特写"),
                                ],
                            ),
                        ),
                    ],
                ),
            ],
        )
        ok, msg = _validate_beat_plan(payload, script)
        self.assertFalse(ok)
        self.assertIn("partial start_visibility", msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
