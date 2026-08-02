"""Unit tests for long-video project activity store."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.long_video.activity import (
    category_for_context,
    extract_long_video_context,
    record_task_activity,
)
from backend.persistence.long_video_activity_store import LongVideoActivityStore
import backend.core.task_kinds as TK


class LongVideoActivityUnit(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db = Path(self._tmpdir.name) / "studio.db"
        self.store = LongVideoActivityStore(db)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_extract_long_video_context_from_metadata(self) -> None:
        ctx = extract_long_video_context(
            {
                "metadata": {
                    "long_video_project_id": "lvp_abc",
                    "long_video_phase": "keyframe",
                    "long_video_shot_id": "shot_1",
                }
            }
        )
        self.assertEqual(ctx["project_id"], "lvp_abc")
        self.assertEqual(ctx["phase"], "keyframe")
        self.assertEqual(ctx["shot_id"], "shot_1")

    def test_category_for_context(self) -> None:
        self.assertEqual(category_for_context(phase="cast_portrait", task_kind=TK.IMAGE_GENERATION), "image_generation")
        self.assertEqual(category_for_context(phase="segment", task_kind=TK.VIDEO_GENERATION), "video_generation")
        self.assertEqual(category_for_context(phase="", task_kind=TK.VIDEO_LONG_GENERATION), "video_generation")

    def test_append_and_list_parse_events(self) -> None:
        project_id = "lvp_test1"
        parse_run_id = "prun_test1"
        self.store.append_event(
            project_id=project_id,
            category="script_parse",
            event_type="parse_started",
            phase="script_parse",
            parse_run_id=parse_run_id,
            summary="started",
        )
        self.store.append_event(
            project_id=project_id,
            category="script_parse",
            event_type="parse_phase",
            phase="plan",
            parse_run_id=parse_run_id,
            summary="plan",
            detail={"message": "plan"},
        )
        items = self.store.list_events(project_id, category="script_parse")
        self.assertEqual(len(items), 2)
        run = self.store.get_parse_run(project_id, parse_run_id)
        assert run is not None
        self.assertEqual(run["status"], "running")
        self.assertEqual(len(run["phases"]), 1)

    def test_record_task_activity_helper(self) -> None:
        record_task_activity(
            self.store,
            event_type="task_submitted",
            task_id="tsk_abc",
            task_kind=TK.IMAGE_GENERATION,
            model_id="flux1-dev",
            params={
                "metadata": {
                    "long_video_project_id": "lvp_task",
                    "long_video_phase": "keyframe",
                    "long_video_shot_id": "sh_01",
                }
            },
            status="queued",
        )
        items = self.store.list_events("lvp_task", task_id="tsk_abc")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["category"], "image_generation")
        self.assertEqual(items[0]["event_type"], "task_submitted")
        self.assertEqual(items[0]["shot_id"], "sh_01")


if __name__ == "__main__":
    unittest.main()
