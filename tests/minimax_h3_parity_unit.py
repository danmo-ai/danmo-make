"""MiniMax-H3 parity tests (PipeNetwork / diffusers alignment, no GPU)."""
from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np

from backend.engine.families.minimax_h3 import packing as P
from backend.engine.families.minimax_h3.scheduler_mlx import MiniMaxH3Scheduler, linspace_1_to_0


class MiniMaxH3PackingParityTests(unittest.TestCase):
    def test_canvas_and_frames(self) -> None:
        for aw, ah in [(16, 9), (9, 16), (1, 1), (4, 1)]:
            h, w = P.resolve_canvas_size(aw, ah)
            self.assertEqual(h % 32, 0)
            self.assertEqual(w % 32, 0)
            self.assertLessEqual(h * w, P.MINIMAX_H3_MAX_PIXELS + 32**2)

        self.assertEqual(P.align_num_frames(22), 22)
        self.assertEqual(P.align_num_frames(23), 39)
        self.assertEqual(P.video_latent_num_frames(22), 7)
        self.assertEqual(P.audio_latent_num_frames(125), 208)

    def test_spatial_grid_aspect_normalized(self) -> None:
        sqrt_area = float(np.sqrt(24 * 24))
        grid = P._spatial_position_grid(24, 2, sqrt_area)
        self.assertAlmostEqual(float(grid[0]), 0.0, places=5)
        self.assertTrue(float(grid[-1]) < 32.0)
        wide = P._spatial_position_grid(48, 2, float(np.sqrt(48 * 24)))
        self.assertLess(float(wide[0]), 0.0)
        self.assertGreater(float(wide[-1]), 0.0)

    def test_packed_sequence_shapes(self) -> None:
        tags = np.full(7, P.MINIMAX_H3_TEXT_TAG, dtype=np.int64)
        layout = P.build_packed_sequence(
            tags,
            num_latent_frames=12,
            latent_height=24,
            latent_width=42,
            num_audio_latents=20,
            patch_size=(1, 2, 2),
            keyframe_anchors=("first",),
        )
        self.assertEqual(layout.position_ids.shape[1], 3)
        self.assertEqual(len(layout.token_tags), layout.sequence_length)
        self.assertGreater(layout.num_condition_video_rows, 0)

    def test_patchify_roundtrip(self) -> None:
        patch = (1, 2, 2)
        latents = np.random.default_rng(0).standard_normal((1, 4, 6, 8, 10)).astype(np.float32)
        rows = P.patchify_video_latents(latents, patch)
        back = P.unpatchify_video_tokens(
            rows[0],
            num_latent_frames=6,
            latent_height=8,
            latent_width=10,
            latent_channels=4,
            patch_size=patch,
        )
        self.assertEqual(back.shape, latents.shape)


class MiniMaxH3SchedulerParityTests(unittest.TestCase):
    def test_linspace_matches_torch_style(self) -> None:
        out = linspace_1_to_0(16)
        self.assertEqual(out.shape, (16,))
        self.assertAlmostEqual(float(out[0]), 1.0, places=6)
        self.assertAlmostEqual(float(out[-1]), 0.0, places=6)

    def test_scheduler_sigmas_dedup(self) -> None:
        for shift in (12.0, 3.0):
            sched = MiniMaxH3Scheduler(shift=shift)
            sched.set_timesteps(50)
            sigmas = np.array(sched.sigmas)
            self.assertTrue(np.all(sigmas[:-1] > sigmas[1:]))
            self.assertEqual(float(sigmas[-1]), 0.0)
            timesteps = np.array(sched.timesteps)
            self.assertEqual(len(timesteps), len(sigmas) - 1)
            np.testing.assert_allclose(timesteps, 1.0 - sigmas[:-1], rtol=0, atol=1e-6)

    def test_scheduler_trajectory(self) -> None:
        sched = MiniMaxH3Scheduler(shift=12.0)
        sched.set_timesteps(16)
        rng = np.random.default_rng(0)
        sample = mx.array(rng.standard_normal((4, 8)).astype(np.float32))

        for t in sched.timesteps.tolist():
            v = mx.array(rng.standard_normal((4, 8)).astype(np.float32))
            sample = sched.step(v, float(t), sample)
        self.assertEqual(sample.shape, (4, 8))

    def test_scale_noise_keyframe(self) -> None:
        sched = MiniMaxH3Scheduler(shift=12.0)

        x0 = mx.array(np.ones((2, 3), dtype=np.float32))
        noise = mx.zeros((2, 3), dtype=mx.float32)
        out = sched.scale_noise(x0, P.MINIMAX_H3_KEYFRAME_NOISE_AUG, noise)
        np.testing.assert_allclose(
            np.array(out),
            P.MINIMAX_H3_KEYFRAME_NOISE_AUG * np.ones((2, 3), dtype=np.float32),
            rtol=1e-6,
        )


class MiniMaxH3DiTParityTests(unittest.TestCase):
    def test_dit_module_key_tree(self) -> None:
        from backend.engine.families.minimax_h3.transformer_mlx import MiniMaxH3DiTMLX, expected_dit_param_keys

        dit = MiniMaxH3DiTMLX.from_config({})
        keys = expected_dit_param_keys(dit)
        self.assertIn("video_patch_proj.weight", keys)
        self.assertIn("blocks.0.attn.qkv_proj.weight", keys)
        self.assertIn("blocks.0.mlp.fc1.weight", keys)
        self.assertIn("final_layer.video_out.weight", keys)
        self.assertNotIn("rope.inv_freq", keys)


class MiniMaxH3LoraParityTests(unittest.TestCase):
    def test_remap_lora_ab_pairs(self) -> None:
        from backend.engine.families.minimax_h3.lora_weights import remap_minimax_h3_lora_keys

        weights = {
            "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": mx.zeros((4, 8)),
            "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": mx.zeros((16, 4)),
        }
        groups = remap_minimax_h3_lora_keys(weights, default_alpha=128.0)
        self.assertIn("blocks.0.attn.qkv_proj", groups)
        down, up, alpha = groups["blocks.0.attn.qkv_proj"]
        self.assertEqual(tuple(down.shape), (4, 8))
        self.assertEqual(tuple(up.shape), (16, 4))
        self.assertAlmostEqual(alpha, 128.0)


import os

_H3_BUNDLE = os.environ.get("DANQING_H3_BUNDLE", "").strip()


@unittest.skipUnless(_H3_BUNDLE, "Set DANQING_H3_BUNDLE to run GPU/bundle parity tests")
class MiniMaxH3SlowParityTests(unittest.TestCase):
    """Optional slow parity (DiT / TE / VAE / E2E) when bundle path is set."""

    def test_bundle_scheduler_e2e_smoke(self) -> None:
        from pathlib import Path

        root = Path(_H3_BUNDLE)
        self.assertTrue(root.is_dir(), f"DANQING_H3_BUNDLE not a directory: {root}")
        # Import-only smoke: generator resolves plan and packing without running denoise.
        from backend.engine.config.model_configs import MinimaxH3Config
        from backend.engine.families.minimax_h3.generation_mlx import MinimaxH3MlxGenerator
        from backend.engine.runtime.mlx_runtime import MLXContext

        gen = MinimaxH3MlxGenerator(MLXContext(), root, config=MinimaxH3Config())
        gen._validate_inference_plan(8, None)


if __name__ == "__main__":
    unittest.main()
