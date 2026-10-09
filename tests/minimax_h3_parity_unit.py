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


class MiniMaxH3UpgradeTests(unittest.TestCase):
    def test_requirements_pin_mlx_0323(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        macos = (root / "requirements-macos.txt").read_text(encoding="utf-8")
        linux = (root / "requirements-linux.txt").read_text(encoding="utf-8")
        self.assertIn("mlx>=0.32.3", macos)
        self.assertIn("mlx[cuda]>=0.32.3", linux)
        self.assertNotIn("mlx-vlm>=0.7", macos)
        self.assertNotIn("mlx-vlm>=0.7", linux)

    def test_version_parse_orders_0323_above_021(self) -> None:
        from backend.engine.runtime.mlx_version import parse_mlx_version

        self.assertEqual(parse_mlx_version("0.32.3"), (0, 32, 3))
        self.assertEqual(parse_mlx_version("0.32.3.dev0"), (0, 32, 3))
        self.assertLess(parse_mlx_version("0.21.0"), (0, 32, 3))

    def test_quality_presets_keep_full_euler(self) -> None:
        # inference/__init__ imports video_two_stage, which imports video_run_common.
        # Loading the package first finishes that cycle; importing video_run_common
        # first leaves it partial and cannot see execute_family_video_avatar.
        import backend.engine.inference  # noqa: F401
        from backend.engine.pipelines.video_run_common import _apply_minimax_h3_quality_preset

        class _Cfg:
            h3_denoiser_reuse = 3
            h3_active_layers = 40
            h3_internal_canvas = "384"
            h3_turbo = True
            h3_low_memory = False
            teacache_mode = "on"

        for name in ("balanced", "quality", "oracle"):
            cfg = _Cfg()
            _apply_minimax_h3_quality_preset(cfg, name)
            self.assertEqual(cfg.h3_denoiser_reuse, 1)
            self.assertEqual(cfg.h3_active_layers, 50)
            self.assertEqual(cfg.h3_internal_canvas, "off")
            self.assertFalse(cfg.h3_turbo)
            self.assertEqual(cfg.teacache_mode, "none")
        draft = _Cfg()
        _apply_minimax_h3_quality_preset(draft, "draft")
        self.assertTrue(draft.h3_turbo)
        self.assertEqual(draft.h3_denoiser_reuse, 1)
        self.assertEqual(draft.h3_active_layers, 50)
        self.assertEqual(draft.h3_internal_canvas, "off")
        with self.assertRaises(RuntimeError):
            _apply_minimax_h3_quality_preset(_Cfg(), "fast")

    def test_reuse_one_covers_every_sigma(self) -> None:
        from backend.engine.families.minimax_h3.generation_mlx import denoiser_eval_indices

        self.assertEqual(denoiser_eval_indices(8, 1), set(range(8)))
        self.assertEqual(denoiser_eval_indices(2, 3), {0, 1})
        skipped = denoiser_eval_indices(8, 2)
        self.assertIn(0, skipped)
        self.assertIn(7, skipped)
        self.assertLess(len(skipped), 8)

    def test_attention_mlx_matches_auto_on_head_128(self) -> None:
        from backend.engine.common.ops.attention import scaled_dot_product_attention_bhsd_mx

        mx.random.seed(0)
        q = mx.random.normal((1, 2, 8, 128))
        k = mx.random.normal((1, 2, 8, 128))
        v = mx.random.normal((1, 2, 8, 128))
        scale = 128 ** -0.5
        eager = scaled_dot_product_attention_bhsd_mx(
            mx, q, k, v, scale=scale, attention_backend="mlx",
        )
        auto = scaled_dot_product_attention_bhsd_mx(
            mx, q, k, v, scale=scale, attention_backend="auto",
        )
        mx.eval(eager, auto)
        err = float(mx.max(mx.abs(eager - auto)).item())
        self.assertLess(err, 1e-4)

    def test_quantized_matmul_finite_past_32k_rows(self) -> None:
        from backend.engine.families.minimax_h3.dit_quant_mlx import (
            assert_quantized_matmul_finite_over_32k,
        )

        assert_quantized_matmul_finite_over_32k()

    def test_adaln_cache_matches_online_and_text_is_stable(self) -> None:
        from backend.engine.families.minimax_h3.transformer_mlx import MiniMaxH3DiTMLX

        dit = _tiny_h3_dit()
        seq = 6
        text_idx = mx.array([0, 1], dtype=mx.int32)
        video_idx = mx.array([2, 3], dtype=mx.int32)
        audio_idx = mx.array([4, 5], dtype=mx.int32)
        tags = mx.array(
            [P.MINIMAX_H3_TEXT_TAG, P.MINIMAX_H3_TEXT_TAG, P.MINIMAX_H3_VIDEO_TAG,
             P.MINIMAX_H3_VIDEO_TAG, P.MINIMAX_H3_AUDIO_TAG, P.MINIMAX_H3_AUDIO_TAG],
            dtype=mx.int32,
        )
        position_ids = mx.zeros((seq, 3))
        prompt = mx.random.normal((1, 2, 16))
        video = mx.random.normal((1, 2, 4))
        audio = mx.random.normal((1, 2, 4))
        unique = np.array([0.75, 0.25], dtype=np.float32)
        row_inv = np.array([0, 0, 1, 1, 0, 1], dtype=np.int32)
        cache_ts = np.unique(unique)
        row_cache = P.map_unique_timesteps_to_cache(cache_ts, unique, row_inv)
        kwargs = dict(
            hidden_states=video,
            audio_hidden_states=audio,
            encoder_hidden_states=prompt,
            timestep=mx.array(unique),
            timestep_indices=mx.array(row_inv),
            token_tags=tags,
            position_ids=position_ids,
            video_indices=video_idx,
            audio_indices=audio_idx,
            text_indices=text_idx,
            return_dict=False,
        )
        online_v, online_a = dit(**kwargs)
        clip = dit.prepare_clip(
            encoder_hidden_states=prompt,
            position_ids=position_ids,
            token_tags=tags,
            schedule_timesteps=mx.array(cache_ts),
        )
        again = dit.prepare_clip(
            encoder_hidden_states=prompt,
            position_ids=position_ids,
            token_tags=tags,
            schedule_timesteps=mx.array(cache_ts),
        )
        mx.eval(clip.text_embeds, again.text_embeds, clip.rotary_emb[0], again.rotary_emb[0])
        self.assertLess(
            float(mx.max(mx.abs(clip.text_embeds - again.text_embeds)).item()),
            1e-5,
        )
        self.assertLess(
            float(mx.max(mx.abs(clip.rotary_emb[0] - again.rotary_emb[0])).item()),
            1e-5,
        )
        cached_v, cached_a = dit(
            **kwargs,
            clip_cache=clip,
            cache_row_indices=mx.array(row_cache),
        )
        mx.eval(online_v, online_a, cached_v, cached_a)
        self.assertLess(float(mx.max(mx.abs(online_v - cached_v)).item()), 1e-4)
        self.assertLess(float(mx.max(mx.abs(online_a - cached_a)).item()), 1e-4)
        dit.drop_adaln_projections()
        with self.assertRaises(RuntimeError):
            dit.prepare_clip(
                encoder_hidden_states=prompt,
                position_ids=position_ids,
                token_tags=tags,
                schedule_timesteps=mx.array(cache_ts),
            )

    def test_compiled_block_matches_eager(self) -> None:
        from backend.engine.families.minimax_h3.transformer_mlx import assert_compiled_block_close

        dit = _tiny_h3_dit()
        block = dit.blocks[0]
        hidden = mx.random.normal((1, 4, 32))
        temb = dit._temb(mx.array(np.array([0.3, 0.7], dtype=np.float32)))
        modulation = block.adaln_proj(temb)
        indices = mx.array([0, 1, 3, 4], dtype=mx.int32)
        rotary = dit.rope(mx.zeros((4, 3)))
        assert_compiled_block_close(block, hidden, modulation, indices, rotary)

    def test_turbo_merge_restores_dense_weights(self) -> None:
        import tempfile
        from pathlib import Path

        from backend.engine.families.minimax_h3.lora_mlx import merge_minimax_h3_turbo_lora
        from backend.engine.runtime.mlx import MLXContext

        dit = _tiny_h3_dit()
        weight = dit.blocks[0].attn.out_proj.weight
        before = mx.array(weight)
        out_d, in_d = int(weight.shape[0]), int(weight.shape[1])
        down = mx.ones((1, in_d)) * 0.01
        up = mx.ones((out_d, 1)) * 0.05
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "turbo_lora.safetensors"
            mx.save_safetensors(
                str(path),
                {
                    "blocks.0.attn.out_proj.lora_A.weight": down,
                    "blocks.0.attn.out_proj.lora_B.weight": up,
                },
            )
            restore = merge_minimax_h3_turbo_lora(
                dit,
                weight_path=path,
                strength=1.0,
                ctx=MLXContext(),
            )
        mx.eval(weight)
        self.assertGreater(float(mx.max(mx.abs(weight - before)).item()), 1e-5)
        restore()
        mx.eval(weight)
        self.assertLess(float(mx.max(mx.abs(weight - before)).item()), 1e-5)

    def test_mlx_vlm_symbols_or_absent(self) -> None:
        from backend.engine.families.minimax_h3.text_encoder_mlx import (
            require_minimax_h3_mlx_vlm_api,
        )

        try:
            require_minimax_h3_mlx_vlm_api()
        except RuntimeError as exc:
            message = str(exc)
            self.assertTrue(
                "requires `mlx-vlm`" in message or "missing MiniMax-H3" in message,
                message,
            )


def _tiny_h3_dit():
    from backend.engine.families.minimax_h3.transformer_mlx import MiniMaxH3DiTMLX

    mx.random.seed(0)
    return MiniMaxH3DiTMLX(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=32,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=64,
        in_channels=4,
        audio_in_channels=4,
        patch_size=(1, 1, 1),
        text_dim=16,
        freq_dim=8,
        time_embed_hidden_dim=16,
        time_embed_dim=16,
        rope_freq_dim=2,
    )


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
        from backend.engine.runtime.mlx import MLXContext

        gen = MinimaxH3MlxGenerator(MLXContext(), root, config=MinimaxH3Config())
        gen._validate_inference_plan(8, None)


if __name__ == "__main__":
    unittest.main()
