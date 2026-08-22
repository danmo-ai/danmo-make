"""MiniMax-H3 FL2VA packed-sequence geometry (MLX / NumPy).

Port of Diffusers ``modular_pipelines.minimax_h3.packing`` constants and builders.
Row order: ``[ text | keyframe conditions | target audio | target video ]``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

MINIMAX_H3_VIDEO_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
MINIMAX_H3_AUDIO_TAG = 2

MINIMAX_H3_FPS = 24
MINIMAX_H3_SHORT_EDGE = 768
MINIMAX_H3_MAX_PIXELS = 768 * 1344
MINIMAX_H3_CANVAS_MULTIPLE = 32
MINIMAX_H3_MIN_ASPECT_RATIO = 1 / 4
MINIMAX_H3_MAX_ASPECT_RATIO = 4
MINIMAX_H3_MIN_DURATION = 5.0
MINIMAX_H3_MAX_DURATION = 15.0

MINIMAX_H3_FRAMES_PER_CHUNK = 17
MINIMAX_H3_LATENTS_PER_CHUNK = 5

MINIMAX_H3_PIXEL_MEAN = (0.485, 0.456, 0.406)
MINIMAX_H3_PIXEL_STD = (0.229, 0.224, 0.225)

MINIMAX_H3_TEXT_ENCODER_LAYER = 50
MINIMAX_H3_AUDIO_LATENTS_PER_SECOND = 40
MINIMAX_H3_AUDIO_CHANNELS = 2
MINIMAX_H3_KEYFRAME_NOISE_AUG = 0.999
MINIMAX_H3_KEYFRAME_ENCODE_SEED = 42

_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32

PATCH_SIZE = (1, 2, 2)
VAE_SPATIAL_SCALE = 16


@dataclass
class MiniMaxH3PackedSequence:
    sequence_length: int
    position_ids: np.ndarray  # (S, 3) float64
    token_tags: np.ndarray  # (S,) int64
    video_indices: np.ndarray
    audio_indices: np.ndarray
    text_indices: np.ndarray
    num_condition_video_rows: int
    num_condition_audio_rows: int


def resolve_canvas_size(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}.")
    ratio = aspect_width / aspect_height
    if not MINIMAX_H3_MIN_ASPECT_RATIO <= ratio <= MINIMAX_H3_MAX_ASPECT_RATIO:
        raise ValueError(
            f"MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got {aspect_width}:{aspect_height} ({ratio:g})."
        )
    if ratio >= 1.0:
        width, height = MINIMAX_H3_SHORT_EDGE * ratio, float(MINIMAX_H3_SHORT_EDGE)
    else:
        width, height = float(MINIMAX_H3_SHORT_EDGE), MINIMAX_H3_SHORT_EDGE / ratio
    area = width * height
    if area > MINIMAX_H3_MAX_PIXELS:
        scale = (MINIMAX_H3_MAX_PIXELS / area) ** 0.5
        width, height = width * scale, height * scale
    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    return (
        max(multiple, round(height / multiple) * multiple),
        max(multiple, round(width / multiple) * multiple),
    )


def align_num_frames(num_frames: int) -> int:
    if num_frames < 1:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    while num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def video_latent_num_frames(num_frames: int) -> int:
    if num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        raise ValueError(f"`num_frames` must be of the form 17 * n + 5, got {num_frames}.")
    return (
        num_frames - MINIMAX_H3_LATENTS_PER_CHUNK
    ) // MINIMAX_H3_FRAMES_PER_CHUNK * MINIMAX_H3_LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int) -> int:
    return int(round(num_frames / MINIMAX_H3_FPS * MINIMAX_H3_AUDIO_LATENTS_PER_SECOND))


def validate_duration(num_frames: int) -> None:
    duration = num_frames / MINIMAX_H3_FPS
    if duration < MINIMAX_H3_MIN_DURATION or duration > MINIMAX_H3_MAX_DURATION:
        raise RuntimeError(
            f"MiniMax-H3 FL2VA duration must be in "
            f"[{MINIMAX_H3_MIN_DURATION}, {MINIMAX_H3_MAX_DURATION}]s @ {MINIMAX_H3_FPS}fps; "
            f"got {num_frames} frames ({duration:.3f}s)."
        )


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> np.ndarray:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE


def _temporal_position_span(num_latent_frames: int) -> float:
    spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
    for i in range(len(_ROPE_FRAMES_PER_LATENT)):
        spans[i :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[i]
    return float(spans.sum())


def _temporal_position_grid(num_latent_frames: int, origin: float) -> np.ndarray:
    spans = np.array(
        [
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)]
            for i in range(num_latent_frames)
        ],
        dtype=np.float64,
    )
    return origin + np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(spans[:-1])])


def patchify_video_latents(latents: np.ndarray, patch_size: tuple[int, int, int] = PATCH_SIZE) -> np.ndarray:
    """Pack ``(B, C, T, H, W)`` latents into ``(B, T*h*w, C*pt*ph*pw)`` rows."""
    if latents.ndim != 5:
        raise ValueError(f"Expected (B,C,T,H,W) latents, got shape {latents.shape}")
    b, c, t, h, w = latents.shape
    pt, ph, pw = patch_size
    if t % pt or h % ph or w % pw:
        raise ValueError(
            f"Latent shape {(t, h, w)} not divisible by patch_size {patch_size}"
        )
    x = latents.reshape(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
    x = np.transpose(x, (0, 2, 4, 6, 1, 3, 5, 7))
    return x.reshape(b, (t // pt) * (h // ph) * (w // pw), c * pt * ph * pw)


def unpatchify_video_tokens(
    tokens: np.ndarray,
    *,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    latent_channels: int = 24,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
) -> np.ndarray:
    """Inverse of :func:`patchify_video_latents` for batch=1 target rows."""
    pt, ph, pw = patch_size
    rows_h = latent_height // ph
    rows_w = latent_width // pw
    expected = num_latent_frames * rows_h * rows_w
    if tokens.ndim == 3:
        tokens = tokens[0]
    if tokens.shape[0] != expected:
        raise ValueError(f"Expected {expected} video tokens, got {tokens.shape[0]}")
    x = tokens.reshape(num_latent_frames, rows_h, rows_w, latent_channels, pt, ph, pw)
    x = np.transpose(x, (3, 0, 4, 1, 5, 2, 6))
    return x.reshape(1, latent_channels, num_latent_frames * pt, latent_height, latent_width)


def build_packed_sequence(
    text_token_tags: np.ndarray,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
    keyframe_anchors: Sequence[str] = (),
) -> MiniMaxH3PackedSequence:
    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text_tokens = int(text_token_tags.shape[0])
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows

    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows

    position_ids = np.zeros((sequence_length, 3), dtype=np.float64)
    position_ids[:num_text_tokens, 0] = np.arange(num_text_tokens, dtype=np.float64)

    sqrt_area = float(np.sqrt(latent_height * latent_width))
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    hh, ww = np.meshgrid(height_grid, width_grid, indexing="ij")
    frame_grid = np.stack([hh.reshape(-1), ww.reshape(-1)], axis=-1)

    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            anchor_time = (
                float(num_text_tokens)
                + _temporal_position_span(num_latent_frames)
                - _ROPE_FRAME_RESCALE
            )
        else:
            raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")
        rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    audio_time = float(num_text_tokens) + np.arange(num_audio_latents, dtype=np.float64)
    position_ids[audio_start:video_start, 0] = np.tile(audio_time, MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[audio_start:video_start, 2] = np.concatenate(
        [
            np.full(num_audio_latents, float(width_grid[0]), dtype=np.float64),
            np.full(num_audio_rows - num_audio_latents, float(width_grid[-1]), dtype=np.float64),
        ]
    )

    video_position_ids = np.empty((num_latent_frames, rows_per_frame, 3), dtype=np.float64)
    video_position_ids[:, :, 0] = _temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
    video_position_ids[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_position_ids.reshape(-1, 3)

    video_indices = np.concatenate(
        [
            np.arange(condition_start, audio_start),
            np.arange(video_start, sequence_length),
        ]
    ).astype(np.int64)
    audio_indices = np.arange(audio_start, video_start, dtype=np.int64)
    text_indices = np.arange(num_text_tokens, dtype=np.int64)

    token_tags = np.empty(sequence_length, dtype=np.int64)
    token_tags[text_indices] = text_token_tags.astype(np.int64)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG

    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_condition_rows,
        num_condition_audio_rows=0,
    )


def build_row_timesteps(
    layout: MiniMaxH3PackedSequence,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float,
    condition_audio_timestep: float,
) -> tuple[np.ndarray, np.ndarray]:
    row_timesteps = np.full((layout.sequence_length,), video_timestep, dtype=np.float32)
    if layout.num_condition_video_rows:
        row_timesteps[layout.video_indices[: layout.num_condition_video_rows]] = condition_video_timestep
    gen_audio = layout.audio_indices[layout.num_condition_audio_rows :]
    row_timesteps[gen_audio] = audio_timestep
    if layout.num_condition_audio_rows:
        row_timesteps[layout.audio_indices[: layout.num_condition_audio_rows]] = condition_audio_timestep
    distinct, inverse = np.unique(row_timesteps, return_inverse=True)
    return distinct.astype(np.float32), inverse.astype(np.int32)


def flow_match_sigmas(num_inference_steps: int, *, shift: float) -> np.ndarray:
    """Deprecated: prefer ``MiniMaxH3Scheduler.set_timesteps``."""
    from backend.engine.families.minimax_h3.scheduler_mlx import MiniMaxH3Scheduler

    sched = MiniMaxH3Scheduler(shift=shift)
    sched.set_timesteps(num_inference_steps)
    assert sched.sigmas is not None
    return np.array(sched.sigmas, dtype=np.float32)
