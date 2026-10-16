"""Veda tile selection and MLX gather attention for MiniMax-H3."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import backend.engine.inference  # noqa: F401

from backend.engine.families.minimax_h3.veda import (
    TILE_SIZE,
    Budget,
    LayoutSpec,
    SpanSpec,
    TileShape,
    TiledSpan,
    TilePlan,
    bresenham_extra,
    build_tile_layout,
    column_blocks,
    fl2va_layout_spec,
    least_padding_shape,
    select_block,
    span_tiles,
    split_budget,
)
from backend.engine.pipelines.video_run_common import parse_h3_veda


class VedaSelectionTest(unittest.TestCase):
    def test_fp8_e4m3_values(self) -> None:
        from backend.engine.families.minimax_h3.veda_mlx import e4m3fn_to_fp32

        raw = np.array([0x00, 0x38, 0x40, 0x7F, 0xB8], dtype=np.uint8)
        got = e4m3fn_to_fp32(raw)
        self.assertEqual(float(got[0]), 0.0)
        self.assertAlmostEqual(float(got[1]), 1.0)
        self.assertAlmostEqual(float(got[2]), 2.0)
        self.assertTrue(np.isnan(got[3]))
        self.assertAlmostEqual(float(got[4]), -1.0)

    def test_span_compacts_real_rows_to_a_prefix(self) -> None:
        shape = TileShape.parse("8x4x4")
        span = TiledSpan(0, (8, 4, 6), shape)
        tiles = span_tiles(span)
        self.assertEqual(tiles.shape, (2, TILE_SIZE))
        counts = (tiles >= 0).sum(1)
        self.assertEqual(int(counts[0]), 128)
        self.assertEqual(int(counts[1]), 64)
        self.assertTrue(np.all(tiles[1, :64] >= 0))
        self.assertTrue(np.all(tiles[1, 64:] < 0))
        self.assertEqual(int((tiles >= 0).sum()), 8 * 4 * 6)

    def test_own_tile_consumes_a_one_tile_budget(self) -> None:
        shape = least_padding_shape((2, 8, 16))
        layout = build_tile_layout([TiledSpan(0, (2, 8, 16), shape)], seq_len=256)
        self.assertEqual(layout.n_video_tiles, 2)
        self.assertEqual(layout.n_global_tiles, 0)
        scores = np.zeros((1, 2, 2), dtype=np.float32)
        scores[0, :, 0] = 10.0
        index, keep = select_block(
            scores,
            layout,
            column_blocks(layout, Budget(tiles=1), Budget(tiles=1)),
        )
        self.assertEqual(int(keep[0, 0].sum()), 1)
        self.assertEqual(int(keep[0, 1].sum()), 1)
        self.assertEqual(int(index[0, 0][keep[0, 0]][0]), 0)
        self.assertEqual(int(index[0, 1][keep[0, 1]][0]), 1)

    def test_bresenham_mean_matches_the_budget(self) -> None:
        extra = bresenham_extra(10, 0.3)
        self.assertEqual(int(extra.sum()), 3)
        k_lo, k_hi, frac = split_budget(2.3, 8)
        self.assertEqual((k_lo, k_hi, frac), (2, 3, 0.3))

    def test_fl2va_spans_leave_text_and_audio_global(self) -> None:
        class _Packed:
            text_indices = np.arange(3)
            num_condition_video_rows = 4
            video_indices = np.array([3, 4, 5, 6, 12, 13, 14, 15])
            sequence_length = 16

        spec = fl2va_layout_spec(_Packed(), num_latent_frames=1, token_h=2, token_w=2)
        self.assertEqual(spec.references[0].grid, (1, 2, 2))
        self.assertEqual(spec.references[0].start, 3)
        self.assertEqual(spec.target, SpanSpec("target", 12, (1, 2, 2)))
        layout = build_tile_layout(
            [
                TiledSpan(spec.references[0].start, spec.references[0].grid, least_padding_shape(spec.references[0].grid)),
                TiledSpan(spec.target.start, spec.target.grid, least_padding_shape(spec.target.grid)),
            ],
            spec.seq_len,
        )
        global_perm = layout.perm[layout.n_video_tiles * TILE_SIZE :]
        global_rows = global_perm[global_perm >= 0]
        self.assertIn(0, global_rows)
        self.assertIn(7, global_rows)
        self.assertNotIn(12, global_rows)

    def test_parse_flag(self) -> None:
        self.assertFalse(parse_h3_veda("off"))
        self.assertFalse(parse_h3_veda("none"))
        self.assertTrue(parse_h3_veda("on"))
        self.assertTrue(parse_h3_veda(True))
        with self.assertRaises(RuntimeError):
            parse_h3_veda("fast")


class VedaAttentionTest(unittest.TestCase):
    def _runtime(self, *, tiles: float | None, ratio: float | None):
        import mlx.core as mx

        from backend.engine.families.minimax_h3.veda import PlanTable
        from backend.engine.families.minimax_h3.veda_mlx import PredictorBundle, VedaRuntime

        budget = Budget(tiles=tiles) if tiles is not None else Budget(ratio=ratio)
        plan = TilePlan(
            name="16x8_t2",
            grid=(2, 8, 16),
            shapes=[TileShape.parse("2x8x8")],
            head_shape=[[0, 0]],
        )
        spec = LayoutSpec(seq_len=260, target=SpanSpec("target", 4, (2, 8, 16)))
        proj = mx.zeros((2, 24, 8), dtype=mx.bfloat16)
        bundle = PredictorBundle(
            path=Path("synthetic.safetensors"),
            num_layers=1,
            num_heads=2,
            head_dim=8,
            generated=budget,
            reference=budget,
            plans=PlanTable([plan]),
            proj_q=[proj],
            proj_k=[proj],
        )
        return VedaRuntime(bundle, spec, plan, "synthetic", True)

    def _qkv(self):
        import mlx.core as mx

        mx.random.seed(0)
        q = mx.random.normal((1, 2, 260, 8))
        k = mx.random.normal((1, 2, 260, 8))
        v = mx.random.normal((1, 2, 260, 8))
        mx.eval(q, k, v)
        return q, k, v

    def test_full_keep_matches_dense_attention(self) -> None:
        import mlx.core as mx

        q, k, v = self._qkv()
        dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=8**-0.5)
        got = self._runtime(tiles=None, ratio=1.0).attend(q, k, v, 0)
        mx.eval(dense, got)
        err = float(mx.max(mx.abs(dense - got)).item())
        self.assertTrue(np.isfinite(err), err)
        self.assertLess(err, 2e-3)

    def test_one_tile_budget_is_not_dense(self) -> None:
        import mlx.core as mx

        q, k, v = self._qkv()
        dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=8**-0.5)
        runtime = self._runtime(tiles=1, ratio=None)
        got = runtime.attend(q, k, v, 0)
        mx.eval(dense, got)
        err = float(mx.max(mx.abs(dense - got)).item())
        self.assertGreater(err, 1e-2)
        self.assertTrue(np.isfinite(err))
        self.assertLess(runtime.kept / runtime.video_pairs, 0.6)

    def test_bundle_roundtrip(self) -> None:
        from safetensors.numpy import save_file

        from backend.engine.families.minimax_h3.veda_mlx import load_veda_bundle

        plans = {
            "16x8_t2": {
                "geometry": "16x8_t2",
                "grid": [2, 8, 16],
                "shapes": ["2x8x8"],
                "head_shape": [[0, 0]],
            }
        }
        metadata = {
            "format": "miowtion-veda-predictor-v1",
            "num_layers": "1",
            "num_heads": "2",
            "head_dim": "8",
            "keep_ratio": "0.1",
            "dtype": "float32",
            "plans": json.dumps(plans),
            "step": "0",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictor.safetensors"
            save_file(
                {
                    "layers.0.proj_q": np.zeros((2, 24, 8), dtype=np.float32),
                    "layers.0.proj_k": np.ones((2, 24, 8), dtype=np.float32),
                },
                path,
                metadata=metadata,
            )
            bundle = load_veda_bundle(path)
        self.assertEqual(bundle.num_layers, 1)
        self.assertAlmostEqual(float(bundle.generated.ratio), 0.1)
        self.assertTrue(bundle.plans.select((2, 8, 16))[1])


if __name__ == "__main__":
    unittest.main()
