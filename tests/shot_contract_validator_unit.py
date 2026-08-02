"""Unit tests for shot contract validator."""
from __future__ import annotations

import unittest

from backend.long_video.shot_contract_validator import (
    clamp_shot_durations,
    validate_shot_contracts,
)


class ShotContractValidatorTests(unittest.TestCase):
    def test_duration_cap(self):
        shots = [{"duration_sec": 15.0, "flf_mode": "none", "start_frame_mode": "keyframe"}]
        result = validate_shot_contracts(shots, max_clip_sec=10.0)
        self.assertFalse(result.ok)
        self.assertTrue(any(i.code == "duration_out_of_range" for i in result.issues))

    def test_clamp_durations(self):
        shots = [{"duration_sec": 15.0}]
        clamped = clamp_shot_durations(shots, max_clip_sec=10.0)
        self.assertEqual(clamped[0]["duration_sec"], 10.0)

    def test_opening_protagonist_silhouette(self):
        shots = [
            {
                "duration_sec": 5.0,
                "characters_on_screen": ["李明"],
                "first_frame_visibility": "invisible",
                "start_frame_mode": "keyframe",
                "start_visual_prompt": "李明推门",
                "video_prompt": "李明进入",
                "flf_mode": "none",
            }
        ]
        result = validate_shot_contracts(
            shots,
            character_anchor="- 李明（主角）",
            max_clip_sec=10.0,
        )
        self.assertFalse(result.ok)
        self.assertTrue(any(i.code == "opening_no_protagonist" for i in result.issues))

    def test_flf_deprecated(self):
        shots = [{"duration_sec": 5.0, "flf_mode": "first_last", "start_frame_mode": "keyframe"}]
        result = validate_shot_contracts(shots)
        self.assertFalse(result.ok)


if __name__ == "__main__":
    unittest.main()
