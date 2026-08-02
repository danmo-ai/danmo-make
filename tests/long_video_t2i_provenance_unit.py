"""Unit tests for long-video T2I provenance (activity shots_summary)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend.long_video.activity import LongVideoActivityRecorder
from backend.long_video.t2i_provenance import (
    build_shot_t2i_provenance_summary,
    build_shots_summary_with_provenance,
    is_close_up_shot_size,
    locations_similar,
)
from backend.persistence.long_video_activity_store import LongVideoActivityStore


class LongVideoT2iProvenanceUnit(unittest.TestCase):
    def test_locations_similar_bigram(self) -> None:
        self.assertTrue(locations_similar("云雾山山径", "云雾山山路"))

    def test_face_anchor_skips_narrative_merge(self) -> None:
        row = build_shot_t2i_provenance_summary(
            {
                "id": "sh_1",
                "order": 1,
                "segment_role": "face_anchor",
                "location": "办公室",
                "scene_prompt": "主角推门进入",
                "start_visual_prompt": "特写，人物侧脸",
                "shot_size": "特写",
            }
        )
        self.assertFalse(row["narrative_merged"])
        self.assertEqual(row["narrative_skip_reason"], "face_anchor")

    def test_close_up_skips_narrative_merge(self) -> None:
        self.assertTrue(is_close_up_shot_size("特写"))
        row = build_shot_t2i_provenance_summary(
            {
                "id": "sh_2",
                "order": 2,
                "segment_role": "action",
                "location": "街道",
                "scene_prompt": "雨夜行人匆匆",
                "visual_prompt": "近景，撑伞的路人",
                "shot_size": "近景",
            }
        )
        self.assertEqual(row["narrative_skip_reason"], "close_up")

    def test_narrative_merged_when_low_coverage(self) -> None:
        row = build_shot_t2i_provenance_summary(
            {
                "id": "sh_3",
                "order": 3,
                "segment_role": "establishing",
                "location": "古城",
                "scene_prompt": "晨雾中的城墙与飞鸟",
                "visual_prompt": "广角镜头，空镜",
                "start_visual_prompt": "广角镜头，空镜",
                "shot_size": "远景",
                "first_frame_visibility": "invisible",
                "is_establishing_empty": True,
            }
        )
        self.assertFalse(row["narrative_merged"])
        self.assertEqual(row["composed_scene_preview"], "广角镜头，空镜")

    def test_legacy_sparse_visual_merges_beat_scene(self) -> None:
        row = build_shot_t2i_provenance_summary(
            {
                "id": "sh_3b",
                "order": 3,
                "segment_role": "establishing",
                "location": "古城",
                "scene_prompt": "晨雾中的城墙与飞鸟",
                "visual_prompt": "",
                "start_visual_prompt": "",
                "shot_size": "远景",
                "first_frame_visibility": "invisible",
            }
        )
        self.assertTrue(row["narrative_merged"])
        self.assertEqual(row["location_merge"], "prepended")
        self.assertIn("古城", row["composed_scene_preview"])

    def test_ffr_not_merged_into_t2i(self) -> None:
        row = build_shot_t2i_provenance_summary(
            {
                "id": "sh_4",
                "order": 4,
                "segment_role": "action",
                "location": "",
                "scene_prompt": "",
                "visual_prompt": "人物站在窗前",
                "first_frame_requirement": "人物站在窗前；窗外有雨",
                "shot_size": "中景",
            }
        )
        self.assertFalse(row["first_frame_requirement_merged"])
        self.assertEqual(row["ffr_skip_reason"], "inspector_only")
        self.assertEqual(row["composed_scene_preview"], "人物站在窗前")

    def test_build_shots_summary_stats(self) -> None:
        shots = [
            {
                "id": "a",
                "order": 0,
                "segment_role": "face_anchor",
                "scene_prompt": "主角特写",
                "visual_prompt": "x",
            },
            {"id": "b", "order": 1, "segment_role": "action", "location": "L", "scene_prompt": "S", "visual_prompt": "y"},
        ]
        rows, stats = build_shots_summary_with_provenance(shots)
        self.assertEqual(len(rows), 2)
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["face_anchor_skip_count"], 1)

    def test_record_completed_includes_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LongVideoActivityStore(Path(tmp) / "studio.db")
            rec = LongVideoActivityRecorder(store, "lvp_prov")
            response = SimpleNamespace(
                chapter_title="Ch1",
                scene_count=1,
                shots=[
                    SimpleNamespace(
                        id="sh_t",
                        order=0,
                        segment_role="establishing",
                        location="山谷",
                        scene_prompt="云雾缭绕",
                        visual_prompt="航拍",
                        shot_size="远景",
                        first_frame_requirement="",
                    )
                ],
                characters=[],
                llm_calls=3,
                quality_warnings=[],
                quality_issues=[],
                parse_phases=[],
            )
            rec.record_started()
            rec.record_completed(response)
            events = store.list_events("lvp_prov", category="script_parse")
            completed = [e for e in events if e["event_type"] == "parse_completed"]
            self.assertEqual(len(completed), 1)
            detail = completed[0]["detail"]
            self.assertIn("t2i_provenance_stats", detail)
            summary = detail["shots_summary"][0]
            self.assertIn("narrative_merged", summary)
            self.assertIn("location_merge", summary)
            self.assertIn("composed_scene_preview", summary)


if __name__ == "__main__":
    unittest.main()
