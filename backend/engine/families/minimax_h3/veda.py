"""Veda tile layout and block selection for MiniMax-H3.

The released predictor was trained against this contract (128-token 3D tiles,
TripPool is applied later on device):

* A tile shape ``(t, h, w)`` with ``t * h * w = 128`` cuts a span's token grid
  into boxes. Box order is (h-block, w-block, t-block); rows inside a box are
  t, h, w row-major, then real rows are stably compacted to a prefix.
* Text and audio stay global: they are chunked into 128-row tiles after every
  video span, and attention to or from them stays dense.
* Only the video-to-video quadrant is sparse. Reference spans and the target
  video each keep their own budget, so a pooled top-k cannot starve keyframes.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

TILE_SIZE = 128
_NEG_INF = -np.inf
_POS_INF = np.inf


@dataclass(frozen=True, order=True)
class TileShape:
    t: int
    h: int
    w: int

    def __post_init__(self) -> None:
        if self.t * self.h * self.w != TILE_SIZE:
            raise ValueError(f"tile {self} does not hold {TILE_SIZE} tokens")

    def __str__(self) -> str:
        return f"{self.t}x{self.h}x{self.w}"

    @classmethod
    def parse(cls, text: str) -> TileShape:
        t, h, w = (int(v) for v in text.split("x"))
        return cls(t, h, w)

    def transposed(self) -> TileShape:
        return TileShape(self.t, self.w, self.h)

    def padded_grid(self, grid: Sequence[int]) -> tuple[int, int, int]:
        return tuple(-(-g // s) * s for g, s in zip(grid, (self.t, self.h, self.w)))

    def num_tiles(self, grid: Sequence[int]) -> int:
        tp, hp, wp = self.padded_grid(grid)
        return tp * hp * wp // TILE_SIZE

    def aspect_spread(self) -> float:
        return max(self.t, self.h, self.w) / min(self.t, self.h, self.w)


def all_shapes() -> list[TileShape]:
    exponent = TILE_SIZE.bit_length() - 1
    shapes: list[TileShape] = []
    for i, j in itertools.product(range(exponent + 1), repeat=2):
        if i + j <= exponent:
            shapes.append(TileShape(2**i, 2**j, 2 ** (exponent - i - j)))
    return sorted(shapes)


def least_padding_shape(grid: Sequence[int]) -> TileShape:
    fitting = [
        shape
        for shape in all_shapes()
        if shape.t <= grid[0] and shape.h <= grid[1] and shape.w <= grid[2]
    ]
    return min(
        fitting or all_shapes(),
        key=lambda shape: (shape.num_tiles(grid), shape.aspect_spread(), (shape.t, shape.h, shape.w)),
    )


@dataclass(frozen=True)
class TiledSpan:
    start: int
    grid: tuple[int, int, int]
    shape: TileShape

    @property
    def num_rows(self) -> int:
        t, h, w = self.grid
        return t * h * w


def span_tiles(span: TiledSpan) -> np.ndarray:
    """``[n_tiles, 128]`` packed row ids, ``-1`` on padding."""
    t, h, w = span.grid
    shape = span.shape
    tp, hp, wp = shape.padded_grid(span.grid)
    grid = np.full((tp, hp, wp), -1, dtype=np.int64)
    grid[:t, :h, :w] = span.start + np.arange(t * h * w, dtype=np.int64).reshape(t, h, w)
    tiles = grid.reshape(tp // shape.t, shape.t, hp // shape.h, shape.h, wp // shape.w, shape.w)
    tiles = np.transpose(tiles, (2, 4, 0, 1, 3, 5)).reshape(-1, TILE_SIZE)
    order = np.argsort((tiles < 0).astype(np.int8), axis=1, kind="stable")
    return np.take_along_axis(tiles, order, axis=1)


@dataclass
class TileLayout:
    perm: np.ndarray
    valid_count: np.ndarray
    n_video_tiles: int
    n_ref_tiles: int
    n_global_tiles: int
    ref_tokens: int
    target_tokens: int
    seq_len: int
    gather_index: np.ndarray
    scatter_index: np.ndarray
    pad_slots: np.ndarray
    kv_ok: np.ndarray
    slot_valid: np.ndarray

    @property
    def n_tiles(self) -> int:
        return self.n_video_tiles + self.n_global_tiles

    @property
    def num_slots(self) -> int:
        return self.n_tiles * TILE_SIZE


def build_tile_layout(spans: Sequence[TiledSpan], seq_len: int) -> TileLayout:
    if not spans:
        raise ValueError("at least the target span must be tiled")
    covered = np.zeros(seq_len, dtype=bool)
    tiles: list[np.ndarray] = []
    prev_stop = 0
    for span in spans:
        stop = span.start + span.num_rows
        if span.start < prev_stop or stop > seq_len:
            raise ValueError(f"span {span} overlaps or exceeds {seq_len}")
        prev_stop = stop
        covered[span.start:stop] = True
        tiles.append(span_tiles(span))
    n_ref_tiles = sum(tile.shape[0] for tile in tiles[:-1])
    video = np.concatenate(tiles, axis=0)
    global_rows = np.flatnonzero(~covered)
    n_global = -(-global_rows.size // TILE_SIZE)
    global_tiles = np.full((n_global * TILE_SIZE,), -1, dtype=np.int64)
    global_tiles[: global_rows.size] = global_rows
    perm = np.concatenate([video.reshape(-1), global_tiles])
    valid_count = (perm.reshape(-1, TILE_SIZE) >= 0).sum(1).astype(np.int32)
    return TileLayout(
        perm=perm,
        valid_count=valid_count,
        n_video_tiles=int(video.shape[0]),
        n_ref_tiles=int(n_ref_tiles),
        n_global_tiles=int(n_global),
        ref_tokens=sum(span.num_rows for span in spans[:-1]),
        target_tokens=spans[-1].num_rows,
        seq_len=int(seq_len),
        gather_index=np.clip(perm, 0, None).astype(np.int32),
        scatter_index=np.where(perm < 0, seq_len, perm).astype(np.int32),
        pad_slots=np.flatnonzero(perm < 0).astype(np.int32),
        kv_ok=valid_count > 0,
        slot_valid=(perm >= 0).astype(np.int32),
    )


@dataclass(frozen=True)
class Budget:
    ratio: float | None = None
    tiles: float | None = None

    def __post_init__(self) -> None:
        if (self.ratio is None) == (self.tiles is None):
            raise ValueError("set exactly one of ratio / tiles")
        value = self.ratio if self.tiles is None else self.tiles
        if value is None or value <= 0:
            raise ValueError(f"budget must be positive: {self}")

    @property
    def keeps_all(self) -> bool:
        return self.ratio is not None and self.ratio >= 1.0

    def per_row(self, real_tokens: int, n_cols: int) -> float:
        if self.tiles is not None:
            return float(self.tiles)
        n_ideal = math.ceil(real_tokens / TILE_SIZE)
        return float(self.ratio) * n_ideal * n_ideal / n_cols

    def describe(self) -> str:
        if self.tiles is not None:
            return f"{self.tiles:g} tiles/query"
        keep = min(float(self.ratio), 1.0)
        return f"keep_ratio={keep:g} (target ~{100.0 * (1.0 - keep):g}% of video tile-pairs)"


def split_budget(budget: float, n_cols: int) -> tuple[int, int, float]:
    if budget >= n_cols:
        return n_cols, n_cols, 0.0
    k_lo = min(max(1, math.floor(budget)), n_cols)
    frac = round(min(max(budget - k_lo, 0.0), 1.0), 12)
    return k_lo, min(k_lo + 1, n_cols), frac


def bresenham_extra(n_rows: int, frac: float) -> np.ndarray:
    ramp = np.floor(np.arange(n_rows + 1, dtype=np.float64) * frac)
    return ramp[1:] > ramp[:-1]


@dataclass(frozen=True)
class ColumnBlock:
    start: int
    stop: int
    budget: Budget
    real_tokens: int


def column_blocks(layout: TileLayout, generated: Budget, reference: Budget) -> list[ColumnBlock]:
    target = ColumnBlock(layout.n_ref_tiles, layout.n_video_tiles, generated, layout.target_tokens)
    if layout.n_ref_tiles == 0:
        return [target]
    return [
        ColumnBlock(0, layout.n_ref_tiles, reference, layout.ref_tokens),
        target,
    ]


def select_block(
    scores: np.ndarray,
    layout: TileLayout,
    blocks: Sequence[ColumnBlock],
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k over the video quadrant.

    Returns ``(index, keep)`` with shapes ``[heads, n_video, K]``.
    """
    heads, n_video = int(scores.shape[0]), layout.n_video_tiles
    rows = np.arange(n_video)
    indices: list[np.ndarray] = []
    keeps: list[np.ndarray] = []
    for block in blocks:
        n_cols = block.stop - block.start
        if block.budget.keeps_all:
            idx = np.arange(block.start, block.stop)
            indices.append(np.broadcast_to(idx, (heads, n_video, n_cols)))
            keeps.append(np.broadcast_to(layout.kv_ok[block.start:block.stop], (heads, n_video, n_cols)))
            continue
        picked = np.array(scores[:, :, block.start:block.stop], dtype=np.float32, copy=True)
        picked[:, :, ~layout.kv_ok[block.start:block.stop]] = _NEG_INF
        own = (rows >= block.start) & (rows < block.stop)
        col = np.clip(rows - block.start, 0, n_cols - 1)
        col = np.broadcast_to(col.reshape(1, n_video, 1), (heads, n_video, 1))
        gathered = np.take_along_axis(picked, col, axis=-1)
        own_b = own.reshape(1, n_video, 1)
        np.put_along_axis(picked, col, np.where(own_b, _POS_INF, gathered), axis=-1)
        budget = block.budget.per_row(block.real_tokens, n_cols)
        k_lo, k_hi, frac = split_budget(budget, n_cols)
        part = np.argpartition(-picked, k_hi - 1, axis=-1)[..., :k_hi]
        vals = np.take_along_axis(picked, part, axis=-1)
        order = np.argsort(-vals, axis=-1, kind="stable")
        idx = np.take_along_axis(part, order, axis=-1)
        vals = np.take_along_axis(vals, order, axis=-1)
        extra = bresenham_extra(n_video, frac)
        allowed = k_lo + extra.astype(np.int64)
        keep = (np.arange(k_hi)[None, None, :] < allowed[None, :, None]) & (vals > _NEG_INF)
        indices.append(idx + block.start)
        keeps.append(keep)
    return np.concatenate(indices, axis=-1), np.concatenate(keeps, axis=-1)


def block_mask(index: np.ndarray, keep: np.ndarray, layout: TileLayout) -> np.ndarray:
    """``[heads, n_tiles, n_tiles]`` bool. Rows are query tiles."""
    heads = int(index.shape[0])
    n, n_video = layout.n_tiles, layout.n_video_tiles
    mask = np.zeros((heads, n, n), dtype=bool)
    h_idx = np.arange(heads)[:, None, None]
    r_idx = np.arange(n_video)[None, :, None]
    mask[h_idx, r_idx, index] = keep
    if n_video < n:
        mask[:, :, n_video:] = layout.kv_ok[n_video:]
        mask[:, n_video:, :] = layout.kv_ok
    return mask


@dataclass(frozen=True)
class SpanSpec:
    kind: str
    start: int
    grid: tuple[int, int, int]


@dataclass(frozen=True)
class LayoutSpec:
    seq_len: int
    target: SpanSpec
    references: tuple[SpanSpec, ...] = ()


def ref2va_layout_spec(
    seq_len: int,
    visual_spans: Sequence[tuple[int, tuple[int, int, int]]],
    target_start: int,
    target_grid: tuple[int, int, int],
) -> LayoutSpec:
    """One Veda span per reference video block, then the generated video."""
    references = tuple(
        SpanSpec("cond", int(start), (int(grid[0]), int(grid[1]), int(grid[2])))
        for start, grid in visual_spans
    )
    return LayoutSpec(
        seq_len=int(seq_len),
        target=SpanSpec("target", int(target_start), target_grid),
        references=references,
    )


def fl2va_layout_spec(
    packed: object,
    *,
    num_latent_frames: int,
    token_h: int,
    token_w: int,
) -> LayoutSpec:
    """Map an FL2VA packed sequence onto Veda spans.

    Row order is ``text | keyframes | audio | target video``. Keyframes are one
    ``(n, H, W)`` row-major span; text and audio stay global.
    """
    text_n = int(np.asarray(getattr(packed, "text_indices")).shape[0])
    cond = int(getattr(packed, "num_condition_video_rows"))
    frame_tokens = int(token_h) * int(token_w)
    if frame_tokens <= 0 or num_latent_frames < 1:
        raise RuntimeError(
            f"MiniMax-H3 Veda token grid is empty ({num_latent_frames}, {token_h}, {token_w})."
        )
    references: list[SpanSpec] = []
    if cond:
        if cond % frame_tokens != 0:
            raise RuntimeError(
                f"MiniMax-H3 keyframe rows {cond} are not a multiple of the "
                f"token grid {token_h}x{token_w}."
            )
        references.append(SpanSpec("cond", text_n, (cond // frame_tokens, int(token_h), int(token_w))))
    target_rows = int(num_latent_frames) * frame_tokens
    video_indices = np.asarray(getattr(packed, "video_indices"))
    if video_indices.size <= cond:
        raise RuntimeError("MiniMax-H3 packed sequence has no target video rows.")
    target_start = int(video_indices[cond])
    seq_len = int(getattr(packed, "sequence_length"))
    if target_start + target_rows != seq_len:
        raise RuntimeError(
            f"MiniMax-H3 target video span {target_start}:{target_start + target_rows} "
            f"does not end at sequence length {seq_len}."
        )
    if cond and text_n + cond > target_start:
        raise RuntimeError("MiniMax-H3 condition span overlaps the target video.")
    return LayoutSpec(
        seq_len=seq_len,
        target=SpanSpec("target", target_start, (int(num_latent_frames), int(token_h), int(token_w))),
        references=tuple(references),
    )


@dataclass
class HeadGroup:
    shape: TileShape
    heads: np.ndarray


@dataclass
class TilePlan:
    name: str
    grid: tuple[int, int, int]
    shapes: list[TileShape]
    head_shape: list[list[int]]

    @property
    def num_layers(self) -> int:
        return len(self.head_shape)

    @property
    def num_heads(self) -> int:
        return len(self.head_shape[0]) if self.head_shape else 0

    def head_groups(self, layer: int) -> list[HeadGroup]:
        row = np.asarray(self.head_shape[layer], dtype=np.int64)
        return [
            HeadGroup(self.shapes[index], np.flatnonzero(row == index).astype(np.int32))
            for index in sorted(set(int(v) for v in row))
        ]

    def transposed(self) -> TilePlan:
        t, h, w = self.grid
        return TilePlan(
            self.name + "_T",
            (t, w, h),
            [shape.transposed() for shape in self.shapes],
            [list(row) for row in self.head_shape],
        )

    @classmethod
    def from_json(cls, data: dict) -> TilePlan:
        return cls(
            name=str(data["geometry"]),
            grid=tuple(int(v) for v in data["grid"]),
            shapes=[TileShape.parse(text) for text in data["shapes"]],
            head_shape=[[int(v) for v in row] for row in data["head_shape"]],
        )


def _aspect(grid: Sequence[int]) -> float:
    return float(grid[2]) / float(grid[1])


class PlanTable:
    def __init__(self, plans: list[TilePlan]):
        if not plans:
            raise ValueError("a Veda predictor bundle needs at least one tile plan")
        self.plans = {plan.name: plan for plan in plans}
        self._choices: dict[tuple[int, int, int], tuple[TilePlan, bool, str]] = {}

    def select(self, grid: tuple[int, int, int]) -> tuple[TilePlan, bool, str]:
        """Nearest trained aspect, then nearest duration, then least padding."""
        grid = tuple(int(v) for v in grid)
        cached = self._choices.get(grid)
        if cached is not None:
            return cached
        candidates = list(self.plans.values())
        candidates += [plan.transposed() for plan in list(candidates)]
        target = math.log(_aspect(grid))

        def cost(plan: TilePlan) -> tuple[float, int, int]:
            padding = sum(shape.num_tiles(grid) for shape in plan.shapes)
            return (
                round(abs(math.log(_aspect(plan.grid)) - target), 3),
                abs(plan.grid[0] - grid[0]),
                padding,
            )

        plan = min(candidates, key=cost)
        exact = plan.grid == grid
        if exact:
            how = f"trained for token grid {grid[0]}x{grid[1]}x{grid[2]}"
        else:
            how = (
                f"nearest trained token grid {plan.grid[0]}x{plan.grid[1]}x{plan.grid[2]} "
                f"(this clip {grid[0]}x{grid[1]}x{grid[2]})"
            )
        choice = (plan, exact, how)
        self._choices[grid] = choice
        return choice
