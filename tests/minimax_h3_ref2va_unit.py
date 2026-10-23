"""MiniMax-H3 Ref2VA packing and request wiring (no weights)."""
from __future__ import annotations

import unittest

import numpy as np

import backend.engine.inference  # noqa: F401

from backend.engine.families.minimax_h3.ref2va import (
    build_ref2va_packed_sequence,
    build_ref2va_presentation,
    ref2va_duration_ok,
    validate_reference_kinds,
)
from backend.engine.families.minimax_h3.veda import ref2va_layout_spec


class Ref2VAContractTest(unittest.TestCase):
    def test_rejects_audio_only(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_reference_kinds(["audio"])

    def test_caps_and_order(self) -> None:
        validate_reference_kinds(["image", "video", "audio"])
        with self.assertRaises(RuntimeError):
            validate_reference_kinds(["image"] * 10)  # MAX_IMAGES = 9

    def test_duration_window(self) -> None:
        ref2va_duration_ok(125)  # ~5.2s @ 24fps
        with self.assertRaises(RuntimeError):
            ref2va_duration_ok(24)  # 1s

    def test_presentation_labels_picture_and_video(self) -> None:
        def tokenize(text: str) -> list[int]:
            return [ord(c) % 97 + 1 for c in text[:8]] or [1]

        ids, tags = build_ref2va_presentation(
            tokenize,
            "a walking person",
            ["image", "video"],
            [False, True],
            [4],
            [6],
            [[0.0, 0.5]],
        )
        self.assertGreater(len(ids), 10)
        self.assertEqual(len(ids), len(tags))

    def test_packed_sequence_ends_with_target_video(self) -> None:
        tags = np.array([1, 1, 1], dtype=np.int64)
        pack = build_ref2va_packed_sequence(
            tags,
            ["image"],
            [False],
            [(1, 16, 16)],
            [],
            num_latent_frames=2,
            latent_height=16,
            latent_width=16,
            num_audio_latents=4,
            patch_size=(1, 2, 2),
        )
        self.assertEqual(pack.target_grid, (2, 8, 8))
        self.assertEqual(pack.layout.num_condition_video_rows, 1 * 8 * 8)
        self.assertEqual(
            int(pack.layout.video_indices[-1]) + 1,
            pack.layout.sequence_length,
        )
        spec = ref2va_layout_spec(
            pack.layout.sequence_length,
            [(span.start, span.grid) for span in pack.visual_spans],
            pack.target_start,
            pack.target_grid,
        )
        self.assertEqual(spec.target.start, pack.target_start)
        self.assertEqual(len(spec.references), 1)

    def test_media_reference_contract(self) -> None:
        from backend.core.contracts import VideoGenerationRequest, VideoMediaReference

        req = VideoGenerationRequest(
            model="minimax-h3-fl2va:mlx-q4",
            prompt="test",
            media_references=[
                VideoMediaReference(kind="image", asset_id="ast_a"),
                VideoMediaReference(kind="audio", asset_id="ast_b"),
            ],
        )
        self.assertEqual(req.media_references[0].kind, "image")
        self.assertEqual(len(req.media_references), 2)


class Ref2VAMutexTest(unittest.TestCase):
    def test_generate_rejects_refs_with_keyframes(self) -> None:
        from unittest.mock import patch

        from backend.engine.config.model_configs import MinimaxH3Config
        from backend.engine.families.minimax_h3.generation_mlx import MinimaxH3MlxGenerator

        gen = MinimaxH3MlxGenerator.__new__(MinimaxH3MlxGenerator)
        gen.config = MinimaxH3Config()
        gen.ctx = object()
        gen.bundle_root = __import__("pathlib").Path(".")
        gen._dit = object()
        gen._text_encoder = object()
        gen._video_vae = object()
        gen._audio_vae = object()
        gen._project_root = None
        gen._registry = None
        gen._adapters = None
        gen._validate_inference_plan = lambda *a, **k: None  # type: ignore[method-assign]
        gen._license_notice = lambda *a, **k: None  # type: ignore[method-assign]
        gen._ensure_loaded = lambda *a, **k: None  # type: ignore[method-assign]
        gen._apply_turbo_lora = lambda *a, **k: None  # type: ignore[method-assign]

        with patch(
            "backend.engine.families.minimax_h3.generation_mlx.require_mlx_version",
            lambda: None,
        ):
            with self.assertRaises(RuntimeError) as raised:
                MinimaxH3MlxGenerator.generate_and_save(
                    gen,
                    prompt="x",
                    output_path="/tmp/out.mp4",
                    width=768,
                    height=768,
                    num_frames=125,
                    fps=24.0,
                    seed=0,
                    steps=8,
                    guidance=1.0,
                    step_distill=False,
                    image_path="/tmp/first.png",
                    last_frame_path=None,
                    media_references=[("image", "/tmp/ref.png")],
                    negative_prompt="",
                    on_log=None,
                    on_progress=None,
                )
        self.assertIn("Ref2VA", str(raised.exception))
        self.assertIn("keyframes", str(raised.exception).lower())


if __name__ == "__main__":
    unittest.main()
