"""MiniMax-H3 Veda sparse attention on MLX.

The predictor scores 128-token tiles (TripPool + a residual projection). Video
queries keep the top tiles; text, audio, and keyframe-global rows stay dense.
Attention itself is gather + ``mx.fast.scaled_dot_product_attention``.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from backend.engine.contracts.pipeline_registry import local_bundle_root
from backend.engine.families.minimax_h3.veda import (
    TILE_SIZE,
    Budget,
    LayoutSpec,
    PlanTable,
    TileLayout,
    TilePlan,
    TiledSpan,
    block_mask,
    build_tile_layout,
    column_blocks,
    least_padding_shape,
    select_block,
)

H3_VEDA_ID = "minimax-h3-veda"
H3_VEDA_VERSION = "fp8"
H3_VEDA_FILENAME = "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors"
FORMAT = "miowtion-veda-predictor-v1"
_CHUNK_BYTES = 256 * 2**20

_BUNDLE_CACHE: dict[str, PredictorBundle] = {}


def e4m3fn_to_fp32(raw: np.ndarray) -> np.ndarray:
    """OCP float8 e4m3fn (no infinities; 0x7F / 0xFF are NaN)."""
    u = np.asarray(raw, dtype=np.uint8)
    sign = (u >> 7).astype(np.int32)
    exp = ((u >> 3) & 0xF).astype(np.int32)
    mant = (u & 0x7).astype(np.int32)
    is_nan = (exp == 15) & (mant == 7)
    is_zero = (exp == 0) & (mant == 0)
    is_sub = (exp == 0) & (mant != 0)
    normal = np.ldexp(1.0 + mant / 8.0, exp - 7)
    sub = np.ldexp(mant / 8.0, 1 - 7)
    out = np.where(is_sub, sub, normal).astype(np.float32)
    out = np.where(is_zero, np.float32(0.0), out)
    out = np.where(is_nan, np.float32(np.nan), out)
    return np.where(sign == 1, -out, out).astype(np.float32)


def _decode_tensor(dtype: str, shape: tuple[int, ...], raw: bytes) -> np.ndarray:
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
        return (u16.astype(np.uint32) << 16).view(np.float32).copy()
    if dtype in ("F8_E4M3", "F8_E4M3FN"):
        return e4m3fn_to_fp32(np.frombuffer(raw, dtype=np.uint8).reshape(shape))
    raise RuntimeError(f"Veda predictor tensor dtype {dtype!r} is not supported")


def _read_safetensors(path: Path) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        blob = handle.read()
    metadata = {str(k): str(v) for k, v in (header.pop("__metadata__", {}) or {}).items()}
    tensors: dict[str, np.ndarray] = {}
    for key, info in header.items():
        start, end = info["data_offsets"]
        shape = tuple(int(v) for v in info["shape"])
        tensors[str(key)] = _decode_tensor(str(info["dtype"]), shape, blob[start:end])
    return metadata, tensors


def _budget_from_metadata(metadata: dict[str, str], prefix: str, fallback: Budget | None) -> Budget:
    kind = metadata.get(f"{prefix}_budget_kind")
    if kind is None:
        if fallback is None:
            raise RuntimeError(f"Veda bundle is missing {prefix}_budget_kind and keep_ratio")
        return fallback
    try:
        value = float(metadata[f"{prefix}_budget_value"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"Veda bundle {prefix}_budget_value is unreadable") from exc
    if kind == "tiles":
        return Budget(tiles=value)
    if kind == "ratio":
        return Budget(ratio=value)
    raise RuntimeError(f"Veda bundle {prefix}_budget_kind {kind!r} is unknown")


@dataclass
class PredictorBundle:
    path: Path
    num_layers: int
    num_heads: int
    head_dim: int
    generated: Budget
    reference: Budget
    plans: PlanTable
    proj_q: list[mx.array]
    proj_k: list[mx.array]

    def describe(self) -> str:
        trained = self.generated.describe()
        if self.reference != self.generated:
            trained += f" / reference {self.reference.describe()}"
        return (
            f"{self.path.name} · {self.num_layers} layers x {self.num_heads} heads · {trained}"
        )


def load_veda_bundle(path: Path) -> PredictorBundle:
    key = str(Path(path).resolve())
    cached = _BUNDLE_CACHE.get(key)
    if cached is not None:
        return cached
    metadata, tensors = _read_safetensors(Path(path))
    if metadata.get("format") != FORMAT:
        raise RuntimeError(
            f"{Path(path).name} is not a Veda predictor bundle "
            f"(format {metadata.get('format')!r}, expected {FORMAT!r})."
        )
    try:
        num_layers = int(metadata["num_layers"])
        num_heads = int(metadata["num_heads"])
        head_dim = int(metadata["head_dim"])
        keep = metadata.get("keep_ratio")
        default = Budget(ratio=float(keep)) if keep is not None else None
        generated = _budget_from_metadata(metadata, "target", default)
        reference = _budget_from_metadata(metadata, "ref", generated)
        plan_json = json.loads(metadata["plans"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{Path(path).name}: incomplete Veda metadata ({exc})") from exc
    stored = metadata.get("dtype", "float32")
    shape = (num_heads, 3 * head_dim, head_dim)
    proj_q: list[mx.array] = []
    proj_k: list[mx.array] = []
    for layer in range(num_layers):
        for name, out in (("proj_q", proj_q), ("proj_k", proj_k)):
            tensor_key = f"layers.{layer}.{name}"
            value = tensors.pop(tensor_key, None)
            if value is None or tuple(value.shape) != shape:
                raise RuntimeError(f"{Path(path).name}: {tensor_key} missing or not {list(shape)}")
            if stored == "float8_e4m3fn":
                scale = tensors.pop(tensor_key + ".__scale", None)
                if scale is None or tuple(scale.shape) != (num_heads,):
                    raise RuntimeError(f"{Path(path).name}: {tensor_key} has no per-head fp8 scale")
                value = value * scale.astype(np.float32).reshape(-1, 1, 1)
            out.append(mx.array(np.ascontiguousarray(value, dtype=np.float32)).astype(mx.bfloat16))
    leftover = [key for key in tensors if not key.endswith(".__scale")]
    if leftover:
        raise RuntimeError(f"{Path(path).name}: unexpected tensors {sorted(leftover)[:4]}")
    table = PlanTable([TilePlan.from_json(item) for item in plan_json.values()])
    for plan in table.plans.values():
        if plan.num_layers != num_layers or plan.num_heads != num_heads:
            raise RuntimeError(
                f"{Path(path).name}: plan {plan.name} is {plan.num_layers}x{plan.num_heads}, "
                f"predictor is {num_layers}x{num_heads}"
            )
    bundle = PredictorBundle(
        path=Path(path),
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        generated=generated,
        reference=reference,
        plans=table,
        proj_q=proj_q,
        proj_k=proj_k,
    )
    mx.eval(*proj_q, *proj_k)
    _BUNDLE_CACHE[key] = bundle
    return bundle


def _resolve_file(root: Path) -> Path:
    named = root / H3_VEDA_FILENAME
    if named.is_file():
        return named
    files = sorted(root.rglob("*.safetensors"))
    preferred = [path for path in files if "veda" in path.name.lower()]
    if len(preferred) == 1:
        return preferred[0]
    if len(files) == 1:
        return files[0]
    raise RuntimeError(f"No MiniMax-H3 Veda predictor .safetensors under {root}")


def resolve_h3_veda_path(
    *,
    bundle_root: Path,
    project_root: Path | None,
    registry: Any,
) -> Path:
    bundle_file = Path(bundle_root) / H3_VEDA_FILENAME
    if bundle_file.is_file():
        return bundle_file
    if project_root is None or registry is None:
        raise RuntimeError(
            "MiniMax-H3 Veda requires the predictor weights. Install "
            f"'{H3_VEDA_ID}:{H3_VEDA_VERSION}' from Models, or place "
            f"{H3_VEDA_FILENAME} in {bundle_root}."
        )
    try:
        entry = registry.require(H3_VEDA_ID)
    except KeyError as exc:
        raise RuntimeError(
            f"Registry has no {H3_VEDA_ID!r}. Sync models_registry.json and install the predictor."
        ) from exc
    root = local_bundle_root(project_root, entry, H3_VEDA_VERSION)
    if root is None or not Path(root).exists():
        raise RuntimeError(
            "MiniMax-H3 Veda predictor is not installed. "
            f"Install '{H3_VEDA_ID}:{H3_VEDA_VERSION}' from Models "
            "(models/Tools/minimax-h3-veda-fp8)."
        )
    return _resolve_file(Path(root))


def _pool_video_tiles(x: mx.array, layout: TileLayout) -> mx.array:
    n_video = layout.n_video_tiles
    heads, dim = int(x.shape[1]), int(x.shape[2])
    tiles = x[: n_video * TILE_SIZE].reshape(n_video, TILE_SIZE, heads, dim).astype(mx.float32)
    count = np.maximum(layout.valid_count[:n_video], 1).astype(np.float32)
    mean = mx.sum(tiles, axis=1) / mx.array(count).reshape(n_video, 1, 1)
    positions = mx.arange(TILE_SIZE)
    valid_count = mx.array(layout.valid_count[:n_video].astype(np.int32))
    valid = (positions.reshape(1, TILE_SIZE) < valid_count.reshape(n_video, 1)).reshape(n_video, TILE_SIZE, 1, 1)
    tmax = mx.max(mx.where(valid, tiles, mx.array(-math.inf, dtype=mx.float32)), axis=1)
    tmin = mx.min(mx.where(valid, tiles, mx.array(math.inf, dtype=mx.float32)), axis=1)
    feats = mx.concatenate([mean, tmax, tmin], axis=-1)
    ok = mx.array(layout.kv_ok[:n_video].astype(bool)).reshape(n_video, 1, 1)
    feats = mx.where(ok, feats, mx.zeros_like(feats))
    return mx.transpose(feats, (1, 0, 2))


def _tile_logits(feats_q: mx.array, feats_k: mx.array, proj_q: mx.array, proj_k: mx.array) -> mx.array:
    dim = int(proj_q.shape[-1])
    fq = feats_q.astype(mx.float32)
    fk = feats_k.astype(mx.float32)
    q_hat = mx.matmul(fq, proj_q.astype(mx.float32)) + fq[..., :dim]
    k_hat = mx.matmul(fk, proj_k.astype(mx.float32)) + fk[..., :dim]
    return mx.matmul(q_hat, mx.swapaxes(k_hat, -1, -2)) / math.sqrt(dim)


def _gather_tiles(x: mx.array, layout: TileLayout, head_idx: np.ndarray) -> mx.array:
    picked = mx.take(x, mx.array(head_idx.astype(np.int32)), axis=1)
    gathered = mx.take(picked, mx.array(layout.gather_index), axis=0)
    if layout.pad_slots.size:
        n, heads, dim = int(gathered.shape[0]), int(gathered.shape[1]), int(gathered.shape[2])
        del n
        zeros = mx.zeros((int(layout.pad_slots.size), heads, dim), dtype=gathered.dtype)
        idx = mx.broadcast_to(
            mx.array(layout.pad_slots.astype(np.int32)).reshape(-1, 1, 1),
            zeros.shape,
        )
        gathered = mx.put_along_axis(gathered, idx, zeros, axis=0)
    return gathered


def _scatter_tiles(buf: mx.array, tiled: mx.array, layout: TileLayout, head_idx: np.ndarray) -> mx.array:
    """Write tile-ordered rows back. Padding slots are dropped, not stored."""
    real = layout.perm >= 0
    if not np.any(real):
        return buf
    seq_len, heads, dim = int(buf.shape[0]) - 1, int(buf.shape[1]), int(buf.shape[2])
    tiled_np_idx = np.flatnonzero(real).astype(np.int32)
    rows = layout.perm[real].astype(np.int32)
    values = mx.take(tiled, mx.array(tiled_np_idx), axis=0)
    head_ids = mx.array(head_idx.astype(np.int32))
    flat_rows = mx.array(rows).reshape(-1, 1) * heads + head_ids.reshape(1, -1)
    flat = buf.reshape((seq_len + 1) * heads, dim)
    n_write, n_heads = int(values.shape[0]), int(values.shape[1])
    idx = mx.broadcast_to(flat_rows.reshape(n_write * n_heads, 1), (n_write * n_heads, dim))
    flat = mx.put_along_axis(flat, idx, values.reshape(n_write * n_heads, dim), axis=0)
    return flat.reshape(seq_len + 1, heads, dim)


def _attend_selected(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: np.ndarray,
    layout: TileLayout,
) -> mx.array:
    """``q/k/v`` are tile-ordered ``[N, heads, D]``. ``mask`` is ``[heads, n_tiles, n_tiles]``."""
    n, n_video = layout.n_tiles, layout.n_video_tiles
    heads, dim = int(q.shape[1]), int(q.shape[2])
    scale = dim**-0.5
    selected = mask[:, :n_video]
    counts = selected.sum(-1)
    kmax = int(counts.max()) if counts.size else 0
    if kmax < 1:
        raise RuntimeError("Veda selected no key tiles for a video query")
    order = np.argsort(np.where(selected, 0, 1), axis=-1, kind="stable")[..., :kmax]
    slot_ok = np.arange(kmax)[None, None, :] < counts[..., None]
    key_ok = layout.slot_valid.reshape(n, TILE_SIZE).astype(bool)
    allowed = key_ok[order] & slot_ok[..., None]

    qm = mx.transpose(q.reshape(n, TILE_SIZE, heads, dim), (2, 0, 1, 3))
    km = mx.transpose(k.reshape(n, TILE_SIZE, heads, dim), (2, 0, 1, 3))
    vm = mx.transpose(v.reshape(n, TILE_SIZE, heads, dim), (2, 0, 1, 3))
    k_flat = km.reshape(heads * n, TILE_SIZE, dim)
    v_flat = vm.reshape(heads * n, TILE_SIZE, dim)
    flat_idx = (order + (np.arange(heads)[:, None, None] * n)).astype(np.int32)

    per_row = max(1, heads * kmax * TILE_SIZE * (2 * dim * 2 + TILE_SIZE * 4))
    step = max(1, _CHUNK_BYTES // per_row)
    pieces: list[mx.array] = []
    for r0 in range(0, n_video, step):
        r1 = min(n_video, r0 + step)
        count = r1 - r0
        idx = mx.array(np.ascontiguousarray(flat_idx[:, r0:r1].reshape(-1)))
        keys = mx.take(k_flat, idx, axis=0).reshape(heads * count, 1, kmax * TILE_SIZE, dim)
        values = mx.take(v_flat, idx, axis=0).reshape(heads * count, 1, kmax * TILE_SIZE, dim)
        allow = mx.array(np.ascontiguousarray(allowed[:, r0:r1].reshape(heads * count, 1, 1, kmax * TILE_SIZE)))
        query = qm[:, r0:r1].reshape(heads * count, 1, TILE_SIZE, dim)
        out = mx.fast.scaled_dot_product_attention(query, keys, values, scale=scale, mask=allow)
        mx.eval(out)
        out = mx.transpose(out.reshape(heads, count, TILE_SIZE, dim), (1, 2, 0, 3))
        pieces.append(out.reshape(count * TILE_SIZE, heads, dim))
    video_out = mx.concatenate(pieces, axis=0)

    if n_video >= n:
        return video_out
    allow_all = mx.array(layout.slot_valid.astype(bool).reshape(1, 1, 1, -1))
    keys = km.reshape(heads, 1, n * TILE_SIZE, dim)
    values = vm.reshape(heads, 1, n * TILE_SIZE, dim)
    global_pieces: list[mx.array] = []
    for r0 in range(n_video, n):
        query = qm[:, r0 : r0 + 1].reshape(heads, 1, TILE_SIZE, dim)
        out = mx.fast.scaled_dot_product_attention(query, keys, values, scale=scale, mask=allow_all)
        mx.eval(out)
        global_pieces.append(mx.transpose(out.reshape(heads, TILE_SIZE, dim), (1, 0, 2)))
    return mx.concatenate([video_out, mx.concatenate(global_pieces, axis=0)], axis=0)


class VedaRuntime:
    """One loaded predictor and the tile plan for this clip."""

    def __init__(self, bundle: PredictorBundle, spec: LayoutSpec, plan: TilePlan, how: str, exact: bool):
        self.bundle = bundle
        self.spec = spec
        self.plan = plan
        self.how = how
        self.exact = exact
        self.kept = 0
        self.video_pairs = 0
        self.work = 0
        self.dense_work = 0.0
        self._layouts: dict[tuple[str, bool], TileLayout] = {}

    def _layout(self, shape: Any) -> TileLayout:
        tile_refs = not self.bundle.reference.keeps_all
        key = (str(shape), tile_refs)
        cached = self._layouts.get(key)
        if cached is not None:
            return cached
        spans: list[TiledSpan] = []
        if tile_refs:
            spans = [
                TiledSpan(ref.start, ref.grid, least_padding_shape(ref.grid))
                for ref in self.spec.references
            ]
        spans.append(TiledSpan(self.spec.target.start, self.spec.target.grid, shape))
        layout = build_tile_layout(spans, self.spec.seq_len)
        self._layouts[key] = layout
        return layout

    def summary_line(self) -> str:
        if not self.video_pairs:
            return "MiniMax-H3 Veda ran no sparse attention calls"
        kept = self.kept / self.video_pairs
        compute = min(1.0, self.work / self.dense_work) if self.dense_work else 1.0
        return (
            f"MiniMax-H3 Veda kept {kept:.1%} of video tiles "
            f"({compute:.1%} of full-attention blocks); {self.how}"
        )

    def attend(self, query: mx.array, key: mx.array, value: mx.array, layer: int) -> mx.array:
        """``[1, heads, seq, dim]`` post-RoPE Q/K/V -> same layout."""
        if query.ndim != 4 or int(query.shape[0]) != 1:
            raise RuntimeError(f"Veda attention expects batch 1, got {tuple(query.shape)}")
        if layer < 0 or layer >= self.bundle.num_layers:
            raise RuntimeError(
                f"Veda predictor has {self.bundle.num_layers} layers; DiT asked for layer {layer}"
            )
        heads = int(query.shape[1])
        if heads != self.bundle.num_heads or int(query.shape[-1]) != self.bundle.head_dim:
            raise RuntimeError(
                f"Veda predictor is {self.bundle.num_heads}x{self.bundle.head_dim}, "
                f"attention is {heads}x{int(query.shape[-1])}"
            )
        if int(query.shape[2]) != self.spec.seq_len:
            raise RuntimeError(
                f"Veda layout is {self.spec.seq_len} tokens, attention sequence is {int(query.shape[2])}"
            )
        q = mx.transpose(query[0], (1, 0, 2))
        k = mx.transpose(key[0], (1, 0, 2))
        v = mx.transpose(value[0], (1, 0, 2))
        seq_len, _, dim = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
        out = mx.zeros((seq_len + 1, heads, dim), dtype=query.dtype)
        proj_q = self.bundle.proj_q[layer]
        proj_k = self.bundle.proj_k[layer]
        for group in self.plan.head_groups(layer):
            layout = self._layout(group.shape)
            group_blocks = column_blocks(layout, self.bundle.generated, self.bundle.reference)
            per_head = max(1, layout.num_slots * dim * 2)
            step = max(1, min(int(group.heads.shape[0]), _CHUNK_BYTES // per_head))
            for start in range(0, int(group.heads.shape[0]), step):
                head_idx = group.heads[start : start + step]
                q_t = _gather_tiles(q, layout, head_idx)
                k_t = _gather_tiles(k, layout, head_idx)
                v_t = _gather_tiles(v, layout, head_idx)
                scores = _tile_logits(
                    _pool_video_tiles(q_t, layout),
                    _pool_video_tiles(k_t, layout),
                    mx.take(proj_q, mx.array(head_idx.astype(np.int32)), axis=0),
                    mx.take(proj_k, mx.array(head_idx.astype(np.int32)), axis=0),
                )
                mx.eval(scores)
                index, keep = select_block(np.array(scores), layout, group_blocks)
                self.kept += int(keep.sum())
                self.video_pairs += int(keep.shape[0]) * layout.n_video_tiles ** 2
                global_blocks = layout.n_global_tiles * (layout.n_tiles + layout.n_video_tiles)
                self.work += int(keep.sum()) + int(keep.shape[0]) * global_blocks
                self.dense_work += int(keep.shape[0]) * (layout.seq_len / TILE_SIZE) ** 2
                mask = block_mask(index, keep, layout)
                attended = _attend_selected(q_t, k_t, v_t, mask, layout)
                out = _scatter_tiles(out, attended, layout, head_idx)
                mx.eval(out)
        result = mx.transpose(out[:seq_len], (1, 0, 2))[None]
        return result.astype(query.dtype)


def bind_h3_veda(
    dit: Any,
    *,
    spec: LayoutSpec,
    path: Path,
    on_log: Callable[[str, str], None] | None = None,
) -> VedaRuntime:
    bundle = load_veda_bundle(path)
    active = int(getattr(dit, "_active_layers", bundle.num_layers))
    if bundle.num_layers < active:
        raise RuntimeError(
            f"Veda predictor has {bundle.num_layers} layers but the DiT runs {active}. "
            "Use a predictor trained for this MiniMax-H3 depth."
        )
    attn = dit.blocks[0].attn
    if bundle.num_heads != int(attn.heads) or bundle.head_dim != int(attn.head_dim):
        raise RuntimeError(
            f"Veda predictor is {bundle.num_heads}x{bundle.head_dim}, "
            f"DiT attention is {attn.heads}x{attn.head_dim}"
        )
    plan, exact, how = bundle.plans.select(spec.target.grid)
    runtime = VedaRuntime(bundle, spec, plan, how, exact)
    dit._veda = runtime
    if on_log is not None:
        exact_note = "exact" if exact else "nearest plan"
        on_log(
            "info",
            f"MiniMax-H3 Veda sparse attention on ({exact_note}: {how}; "
            f"video {bundle.generated.describe()}). Text and audio stay dense. "
            "The released predictor was trained with the 8-step Turbo LoRA.",
        )
    return runtime
