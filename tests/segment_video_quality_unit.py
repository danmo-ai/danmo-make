"""Unit tests for segment_video batch motion validation."""
from __future__ import annotations

import unittest

from backend.long_video.segment_video_quality import (
    validate_segment_video_batch_payload,
    validate_segment_video_prompts,
)


class _Seg:
    def __init__(self, index: int, gid: str, role: str) -> None:
        self.segment_index = index
        self.segment_group_id = gid
        self.segment_role = role


class _Row:
    def __init__(self, index: int, video_prompt: str) -> None:
        self.index = index
        self.video_prompt = video_prompt


class SegmentVideoQualityTests(unittest.TestCase):
    def test_batch_rejects_duplicate_group(self):
        segs = [
            _Seg(0, "beat_0", "pre_anchor"),
            _Seg(1, "beat_0", "face_anchor"),
        ]
        rows = [
            _Row(0, "Alice walks toward the door in cold blue light"),
            _Row(1, "Alice walks toward the door in cold blue light"),
        ]
        ok, msg = validate_segment_video_batch_payload(segs, rows)
        self.assertFalse(ok)
        self.assertIn("beat_0", msg)

    def test_batch_accepts_different_roles(self):
        segs = [
            _Seg(0, "beat_0", "pre_anchor"),
            _Seg(1, "beat_0", "face_anchor"),
        ]
        rows = [
            _Row(0, "Wide shot: Alice walks from the hallway toward the doorway, camera slowly dollies in"),
            _Row(1, "MCU: Alice holds still, subtle breath and eye movement, minimal camera push"),
        ]
        ok, msg = validate_segment_video_batch_payload(segs, rows)
        self.assertTrue(ok, msg)

    def test_full_map_duplicate(self):
        segs = [
            _Seg(0, "beat_0", "pre_anchor"),
            _Seg(1, "beat_0", "face_anchor"),
        ]
        video = {
            0: "Same motion text repeated for both clips in the beat",
            1: "Same motion text repeated for both clips in the beat",
        }
        result = validate_segment_video_prompts(segs, video)
        self.assertFalse(result.ok)
        self.assertTrue(any(i.code == "motion_duplicate_in_group" for i in result.issues))


if __name__ == "__main__":
    unittest.main()
