"""Unit tests for scene grounding bundle."""
from __future__ import annotations

import unittest

from backend.long_video.scene_grounding import (
    build_camera_candidates,
    build_grounding_bundle,
    keyframe_grounding_metadata,
    pick_camera_zone,
    select_camera_zone_with_vlm,
)


class SceneGroundingTests(unittest.TestCase):
    def test_build_bundle(self):
        bundle = build_grounding_bundle(
            "living_room",
            location="客厅·夜",
            spatial_layout={"camera_zones": [{"id": "CZ1", "description": "wide"}]},
            environment_text="warm lamp light",
        )
        self.assertEqual(bundle.scene_key, "living_room")
        self.assertIn("360", bundle.panorama_prompt.lower())
        self.assertEqual(bundle.selected_camera_zone_id, "CZ1")
        self.assertEqual(len(bundle.camera_candidates), 1)

    def test_pick_silhouette_prefers_wide(self):
        candidates = build_camera_candidates(
            {
                "camera_zones": [
                    {"id": "close", "description": "close-up face"},
                    {"id": "door_wide", "description": "wide shot from door entry"},
                ]
            }
        )
        picked = pick_camera_zone(candidates, visibility="silhouette")
        self.assertEqual(picked.zone_id, "door_wide")

    def test_vlm_select_fallback(self):
        candidates = build_camera_candidates({"camera_zones": [{"id": "A"}, {"id": "B"}]})
        picked = select_camera_zone_with_vlm(candidates, shot_description="doorbell entry")
        self.assertIn(picked.zone_id, {"A", "B"})

    def test_keyframe_metadata(self):
        shot = {
            "first_frame_strategy": "t2i_from_grounding",
            "camera_zone_id": "CZ1",
            "first_frame_visibility": "silhouette",
            "first_frame_requirement": "protagonist silhouette at door",
        }
        scene = {
            "grounding_panorama_asset_id": "ast_pano",
            "grounding_depth_asset_id": "ast_depth",
            "spatial_layout_json": {"camera_zones": [{"id": "CZ1"}]},
        }
        meta = keyframe_grounding_metadata(shot, scene)
        self.assertEqual(meta["long_video_scene_grounding_camera_zone_id"], "CZ1")
        self.assertEqual(meta["long_video_scene_grounding_panorama_asset_id"], "ast_pano")
        self.assertEqual(meta["long_video_scene_grounding_depth_asset_id"], "ast_depth")


if __name__ == "__main__":
    unittest.main()
