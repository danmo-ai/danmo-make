"""NumPy contract tests for LTX 2.5 MSR Ref2VA (no MLX)."""
from __future__ import annotations

import unittest

import numpy as np

from backend.engine.families.ltx25.ref2va import (
    aspect_family,
    audio_token_bounds,
    guide_time_midpoints,
    lora_module_and_role,
    max_audio_tokens,
    read_downscale_factor,
    reference_frame_offsets,
    require_prepend_layout,
    shift_audio_midpoints,
    slot_embedding,
    split_msr_lora_keys,
    split_subject_and_background,
    validate_reference_kinds,
)


class Ltx25Ref2vaContractTest(unittest.TestCase):
    def test_rejects_empty_video_and_audio_only(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_reference_kinds([])
        with self.assertRaises(RuntimeError):
            validate_reference_kinds(["video"])
        with self.assertRaises(RuntimeError):
            validate_reference_kinds(["audio"])
        with self.assertRaises(RuntimeError):
            validate_reference_kinds(["image"] * 6)
        with self.assertRaises(RuntimeError):
            validate_reference_kinds(["image", "audio", "audio", "audio"])

    def test_accepts_subjects_background_and_paired_audio(self) -> None:
        validate_reference_kinds(["image"] * 4 + ["audio", "audio"])
        validate_reference_kinds(["image"] * 5 + ["audio", "audio"])
        subjects, background = split_subject_and_background(5)
        self.assertEqual((subjects, background), (4, True))
        subjects, background = split_subject_and_background(2)
        self.assertEqual((subjects, background), (2, False))
        with self.assertRaises(RuntimeError):
            validate_reference_kinds(["image"] * 5 + ["audio"] * 5)

    def test_pic1_is_the_furthest_negative_offset(self) -> None:
        self.assertEqual(reference_frame_offsets(3), [-3, -2, -1])
        self.assertEqual(reference_frame_offsets(1), [-1])

    def test_guide_midpoint_uses_causal_frame_then_offset(self) -> None:
        mids = guide_time_midpoints(1, frame_offset=-1, fps=24.0)
        self.assertAlmostEqual(float(mids[0]), -0.5 / 24.0, places=6)

    def test_slot_embedding_shape(self) -> None:
        frequencies = np.asarray([1.0, 2.0], dtype=np.float32)
        weight0 = np.eye(5, dtype=np.float32)
        state = {
            "frequencies": frequencies,
            "net.0.weight": weight0,
            "net.0.bias": np.zeros((5,), dtype=np.float32),
            "net.2.weight": np.ones((8, 5), dtype=np.float32),
            "net.2.bias": np.zeros((8,), dtype=np.float32),
        }
        embedding = slot_embedding(1, state)
        self.assertEqual(embedding.shape, (8,))

    def test_lora_keys_split_and_remap(self) -> None:
        keys = [
            "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.lora_A.weight",
            "model.diffusion_model.transformer_blocks.0.ff.net.2.lora_B.weight",
            "model.diffusion_model.transformer_blocks.0.attn1.to_out.0.lora_A.weight",
            "diffusion_model.reference_slot_embedding.net.0.weight",
            "model.diffusion_model.reference_audio_slot_embedding.frequencies",
        ]
        lora_keys, image, audio = split_msr_lora_keys(keys)
        self.assertEqual(len(lora_keys), 3)
        self.assertEqual(image["net.0.weight"], keys[3])
        self.assertEqual(audio["frequencies"], keys[4])
        module, role = lora_module_and_role(keys[0])
        self.assertEqual((module, role), ("transformer_blocks.0.ff.proj_in", "A"))
        module, role = lora_module_and_role(keys[1])
        self.assertEqual((module, role), ("transformer_blocks.0.ff.proj_out", "B"))
        module, role = lora_module_and_role(keys[2])
        self.assertEqual((module, role), ("transformer_blocks.0.attn1.to_out", "A"))

    def test_audio_window_shifts_to_absolute_slot_end(self) -> None:
        _, ends, mids = audio_token_bounds(4)
        shifted = shift_audio_midpoints(
            mids, slot_id=1, image_slot_count=2, slot_duration=5.0, end_margin=0.04,
        )
        slot_end = -((2 - 1) * 5.0) - 0.04
        self.assertAlmostEqual(float(shifted[-1] - mids[-1]), slot_end - float(ends[-1]), places=5)
        self.assertGreaterEqual(max_audio_tokens(5.0), 1)
        with self.assertRaises(RuntimeError):
            shift_audio_midpoints(mids, slot_id=3, image_slot_count=2, slot_duration=5.0, end_margin=0.04)

    def test_aspect_and_downscale(self) -> None:
        self.assertEqual(aspect_family(1280, 704), "landscape")
        self.assertEqual(aspect_family(704, 1280), "portrait")
        self.assertEqual(aspect_family(768, 768), "square")
        self.assertEqual(read_downscale_factor({}), 1)
        with self.assertRaises(RuntimeError):
            read_downscale_factor({"reference_downscale_factor": 2})
        require_prepend_layout({})
        with self.assertRaises(RuntimeError):
            require_prepend_layout({"reference_token_order": "append"})


if __name__ == "__main__":
    unittest.main()
