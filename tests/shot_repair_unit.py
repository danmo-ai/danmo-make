"""Unit tests for rule-based shot contract repair."""
from __future__ import annotations

import unittest

from backend.long_video.shot_contract_validator import validate_shot_contracts
from backend.long_video.shot_repair import repair_shot_contracts


DOORBELL_SHOT_0 = {
    "duration_sec": 6.0,
    "first_frame_visibility": "silhouette",
    "characters_on_screen": ["林晓"],
    "start_frame_mode": "keyframe",
    "start_visual_prompt": "门厅暖光，林晓在玄关远处侧影，门铃按钮在前景",
    "video_prompt": "镜头缓慢推进，林晓走向门口",
    "flf_mode": "none",
    "chain_mode": "keyframe_only",
}


class ShotRepairTests(unittest.TestCase):
    def test_doorbell_opening_passes_validation(self):
        anchor = "- 林晓：年轻女性，深蓝色连帽外套"
        result = validate_shot_contracts([DOORBELL_SHOT_0], character_anchor=anchor)
        self.assertTrue(result.ok)

    def test_repair_visibility_jump(self):
        shots = [
            {
                "duration_sec": 2.0,
                "first_frame_visibility": "silhouette",
                "end_visibility": "silhouette",
                "characters_on_screen": ["赵今麦"],
                "start_frame_mode": "keyframe",
                "flf_mode": "none",
            },
            {
                "duration_sec": 3.0,
                "first_frame_visibility": "full_face",
                "end_visibility": "full_face",
                "characters_on_screen": ["赵今麦"],
                "start_frame_mode": "keyframe",
                "flf_mode": "none",
            },
        ]
        fixed = repair_shot_contracts(shots, character_anchor="- 赵今麦：主角")
        from backend.long_video.shot_contract_validator import validate_shot_contracts

        result = validate_shot_contracts(fixed, character_anchor="- 赵今麦：主角")
        self.assertTrue(result.ok)
        self.assertEqual(fixed[1]["first_frame_visibility"], "partial")

    def test_normalize_preserves_post_when_over_guide(self):
        from backend.long_video.shot_repair import normalize_subsegment_plans
        from backend.long_video.segment_plan_types import SubsegmentPlan

        subsegs = [
            SubsegmentPlan(
                role="pre_anchor",
                duration_sec=3.0,
                shot_size="远景",
                flf_mode="none",
                start_visibility="silhouette",
                end_visibility="partial",
                characters_on_screen=("赵今麦",),
            ),
            SubsegmentPlan(
                role="face_anchor",
                duration_sec=2.5,
                shot_size="特写",
                flf_mode="none",
                start_visibility="partial",
                end_visibility="full_face",
                characters_on_screen=("赵今麦",),
            ),
            SubsegmentPlan(
                role="post_anchor",
                duration_sec=3.0,
                shot_size="中景",
                flf_mode="none",
                start_visibility="full_face",
                end_visibility="partial",
                characters_on_screen=("赵今麦",),
            ),
        ]
        out = normalize_subsegment_plans(subsegs, max_clip_sec=10.0)
        roles = [s.role for s in out]
        self.assertIn("post_anchor", roles)
        self.assertAlmostEqual(sum(s.duration_sec for s in out), 8.5, places=1)

    def test_repair_fixes_invisible_opening(self):
        bad = dict(DOORBELL_SHOT_0)
        bad["first_frame_visibility"] = "invisible"
        bad["characters_on_screen"] = []
        fixed = repair_shot_contracts([bad], character_anchor="- 林晓：主角")
        result = validate_shot_contracts(fixed, character_anchor="- 林晓：主角")
        self.assertTrue(result.ok)
        self.assertEqual(fixed[0]["first_frame_visibility"], "silhouette")
        self.assertIn("林晓", fixed[0]["characters_on_screen"])


if __name__ == "__main__":
    unittest.main()
