"""MiniMax-H3 Ref2VA geometry (NumPy).

Contract follows Diffusers ``MiniMaxH3Ref2VA*`` blocks: an ordered mix of image,
video, and audio references packed ahead of the generated rows. Order labels the
prompt (``<Picture N>``, ``<Video N>``, ``<Audio N>``) and advances the shared
rotary clock. Audio cannot be the only reference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from backend.engine.families.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_MAX_ASPECT_RATIO,
    MINIMAX_H3_MAX_DURATION,
    MINIMAX_H3_MIN_ASPECT_RATIO,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    PATCH_SIZE,
    MiniMaxH3PackedSequence,
    _ROPE_FRAME_RESCALE,
    _ROPE_FRAMES_PER_LATENT,
    _spatial_position_grid,
    _temporal_position_grid,
)

MAX_IMAGES = 9
MAX_VIDEOS = 3
MAX_AUDIOS = 3
MAX_REFERENCES = 12
MIN_CLIP_SECONDS = 2.0
MAX_CLIP_SECONDS = 15.0
MAX_TOTAL_VIDEO_SECONDS = 15.0
MAX_TOTAL_AUDIO_SECONDS = 15.0
IMAGE_SHORT_EDGE = 2048
VIDEO_CONDITION_FPS = 2.0

_VISION_START = 151652
_VISION_END = 151653
_IMAGE_PAD = 151655
_VIDEO_PAD = 151656


@dataclass(frozen=True)
class VisualSpan:
    """One reference video block in the packed sequence (latent grid, token axes)."""

    start: int
    grid: tuple[int, int, int]


@dataclass(frozen=True)
class Ref2VAPack:
    layout: MiniMaxH3PackedSequence
    visual_spans: tuple[VisualSpan, ...]
    target_start: int
    target_grid: tuple[int, int, int]


def validate_reference_kinds(kinds: Sequence[str]) -> None:
    if not kinds:
        raise RuntimeError(
            "MiniMax-H3 Ref2VA needs at least one reference. Use text-to-video for a prompt-only request."
        )
    allowed = {"image", "video", "audio"}
    for index, kind in enumerate(kinds):
        if kind not in allowed:
            raise RuntimeError(
                f"MiniMax-H3 reference {index} must be image, video, or audio, got {kind!r}."
            )
    counts = {kind: kinds.count(kind) for kind in allowed}
    limits = {"image": MAX_IMAGES, "video": MAX_VIDEOS, "audio": MAX_AUDIOS}
    for kind, limit in limits.items():
        if counts[kind] > limit:
            raise RuntimeError(
                f"MiniMax-H3 accepts at most {limit} {kind} references, got {counts[kind]}."
            )
    if len(kinds) > MAX_REFERENCES:
        raise RuntimeError(
            f"MiniMax-H3 accepts at most {MAX_REFERENCES} references in total, got {len(kinds)}."
        )
    if set(kinds) == {"audio"}:
        raise RuntimeError(
            "An audio reference has to be paired with at least one image or video reference."
        )


def validate_clip_duration(kind: str, seconds: float, *, index: int) -> None:
    if seconds < MIN_CLIP_SECONDS or seconds > MAX_CLIP_SECONDS:
        raise RuntimeError(
            f"MiniMax-H3 {kind} reference {index} must be "
            f"{MIN_CLIP_SECONDS:g}–{MAX_CLIP_SECONDS:g}s, got {seconds:.3f}s."
        )


def validate_total_duration(kind: str, seconds: float) -> None:
    cap = MAX_TOTAL_VIDEO_SECONDS if kind == "video" else MAX_TOTAL_AUDIO_SECONDS
    if seconds > cap + 1e-3:
        raise RuntimeError(
            f"MiniMax-H3 {kind} references must total at most {cap:g}s, got {seconds:.3f}s."
        )


def image_reference_size(width: int, height: int) -> tuple[int, int]:
    """Short edge 2048, axes snapped to 32, no area cap. Returns ``(height, width)``."""
    if width <= 0 or height <= 0:
        raise RuntimeError(f"A reference image must have a positive size, got {width}x{height}.")
    ratio = width / height
    if not MINIMAX_H3_MIN_ASPECT_RATIO <= ratio <= MINIMAX_H3_MAX_ASPECT_RATIO:
        raise RuntimeError(
            f"A reference image must be within 1:4 and 4:1, got {width}x{height}."
        )
    scale = IMAGE_SHORT_EDGE / min(width, height)
    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    target_h = max(multiple, round(height * scale / multiple) * multiple)
    target_w = max(multiple, round(width * scale / multiple) * multiple)
    return int(target_h), int(target_w)


def resample_video_frames(frames: np.ndarray, fps: float, target_fps: float = MINIMAX_H3_FPS) -> np.ndarray:
    """Hold-and-repeat onto ``target_fps``, matching ffmpeg's fps filter order."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(
            f"A reference video must be (frames, height, width, 3) RGB, got {frames.shape}."
        )
    if fps <= 0:
        raise RuntimeError(f"A reference video must have a positive frame rate, got {fps}.")
    if frames.shape[0] < 1:
        raise RuntimeError("A reference video has no frames.")
    if abs(fps - target_fps) < 1e-6:
        return frames
    scale = target_fps / fps
    slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
    repeats = np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5))
    if np.any(repeats < 0):
        raise RuntimeError("MiniMax-H3 video resample produced a negative hold.")
    if int(repeats.sum()) < 1:
        raise RuntimeError("MiniMax-H3 video resample produced no frames.")
    return np.repeat(frames, repeats, axis=0)


def snap_reference_frames(num_frames: int) -> int:
    """Snap a reference clip down to ``17n+5`` so the video VAE encodes it."""
    frames_per_chunk = 17
    latents_per_chunk = 5
    chunks = max(1, (int(num_frames) - latents_per_chunk) // frames_per_chunk)
    return chunks * frames_per_chunk + latents_per_chunk


def sample_video_condition_frames(
    num_frames: int,
    *,
    fps: float = MINIMAX_H3_FPS,
    sample_fps: float = VIDEO_CONDITION_FPS,
    temporal_patch: int = 2,
) -> tuple[list[int], list[float]]:
    """Frame indices the conditioner reads, and one timestamp per merged vision block."""
    stride = fps / sample_fps
    indices: list[int] = []
    cursor = 0.0
    while round(cursor) < num_frames:
        picked = round(cursor)
        if not indices or picked > indices[-1]:
            indices.append(picked)
        cursor += stride
    if len(indices) < temporal_patch:
        minimum = round((temporal_patch - 1) * stride) + 1
        raise RuntimeError(
            f"A reference video is read at {sample_fps:g} fps and merged in groups of "
            f"{temporal_patch}, so it must run at least {minimum} frames at {fps:g} fps "
            f"({minimum / fps:.2g}s), got {num_frames}."
        )
    timestamps = [index / sample_fps for index in range(len(indices))]
    pad = (-len(timestamps)) % temporal_patch
    timestamps += [timestamps[-1]] * pad
    block_timestamps = [
        (timestamps[index] + timestamps[index + temporal_patch - 1]) / 2
        for index in range(0, len(timestamps), temporal_patch)
    ]
    return indices, block_timestamps


def _frame_position_grid_np(
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> tuple[np.ndarray, np.ndarray]:
    sqrt_area = float(np.sqrt(latent_height * latent_width))
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    hh, ww = np.meshgrid(height_grid, width_grid, indexing="ij")
    return np.stack([hh.reshape(-1), ww.reshape(-1)], axis=-1), width_grid


def _fill_audio_positions_np(
    position_ids: np.ndarray,
    start: int,
    num_rows: int,
    num_audio_latents: int,
    rotary_time: float,
    width_grid: np.ndarray,
) -> None:
    if num_rows <= 0:
        return
    time = rotary_time + np.arange(num_audio_latents, dtype=np.float64)
    position_ids[start : start + num_rows, 0] = np.tile(time, MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[start : start + num_rows, 2] = np.concatenate(
        [
            np.full(num_audio_latents, float(width_grid[0]), dtype=np.float64),
            np.full(num_audio_latents, float(width_grid[-1]), dtype=np.float64),
        ]
    )


def _sequential_video_span(num_latent_frames: int) -> float:
    total = 0.0
    for index in range(num_latent_frames):
        total += _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
    return total


def build_ref2va_packed_sequence(
    text_token_tags: np.ndarray,
    kinds: Sequence[str],
    has_audio: Sequence[bool],
    visual_shapes: Sequence[tuple[int, int, int]],
    audio_row_counts: Sequence[int],
    *,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
) -> Ref2VAPack:
    """``[text | reference blocks | target audio | target video]``."""
    if len(kinds) != len(has_audio):
        raise RuntimeError("Ref2VA kind and audio flags must be the same length.")
    _, patch_h, patch_w = patch_size
    tags = np.asarray(text_token_tags).reshape(-1)
    num_text = int(tags.shape[0])
    num_target_video_rows = num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
    num_target_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS

    visual = iter(tuple(int(x) for x in shape) for shape in visual_shapes)
    audio_rows_iter = iter(int(n) for n in audio_row_counts)
    num_reference_video_rows = 0
    num_reference_audio_rows = 0
    planned: list[tuple[str, bool, tuple[int, int, int] | None, int]] = []
    for kind, audio_flag in zip(kinds, has_audio):
        shape = None
        audio_rows = 0
        if kind in ("image", "video"):
            shape = next(visual)
            t, h, w = shape
            num_reference_video_rows += t * (h // patch_h) * (w // patch_w)
        if audio_flag:
            audio_rows = next(audio_rows_iter)
            num_reference_audio_rows += audio_rows
        planned.append((kind, bool(audio_flag), shape, audio_rows))
    try:
        next(visual)
        raise RuntimeError("Ref2VA visual latents do not match the image and video references.")
    except StopIteration:
        pass
    try:
        next(audio_rows_iter)
        raise RuntimeError("Ref2VA audio latents do not match the audio-bearing references.")
    except StopIteration:
        pass

    sequence_length = (
        num_text
        + num_reference_video_rows
        + num_reference_audio_rows
        + num_target_audio_rows
        + num_target_video_rows
    )
    position_ids = np.zeros((sequence_length, 3), dtype=np.float64)
    position_ids[:num_text, 0] = np.arange(num_text, dtype=np.float64)
    target_frame_grid, target_width_grid = _frame_position_grid_np(
        latent_height, latent_width, patch_h, patch_w
    )

    video_parts: list[np.ndarray] = []
    audio_parts: list[np.ndarray] = []
    visual_spans: list[VisualSpan] = []
    cursor = num_text
    rotary_time = float(num_text)
    for kind, audio_flag, shape, audio_rows in planned:
        if kind == "image":
            assert shape is not None
            t, h, w = shape
            num_video_rows = t * (h // patch_h) * (w // patch_w)
            frame_grid, _width = _frame_position_grid_np(h, w, patch_h, patch_w)
            position_ids[cursor : cursor + num_video_rows, 0] = rotary_time
            position_ids[cursor : cursor + num_video_rows, 1:] = frame_grid
            visual_spans.append(
                VisualSpan(cursor, (t, h // patch_h, w // patch_w))
            )
            video_parts.append(np.arange(cursor, cursor + num_video_rows))
            cursor += num_video_rows
            rotary_time += 1.0
        elif kind == "audio":
            reference_audio_latents = audio_rows // MINIMAX_H3_AUDIO_CHANNELS
            _fill_audio_positions_np(
                position_ids,
                cursor,
                audio_rows,
                reference_audio_latents,
                rotary_time,
                target_width_grid,
            )
            audio_parts.append(np.arange(cursor, cursor + audio_rows))
            cursor += audio_rows
            rotary_time += float(reference_audio_latents)
        elif kind == "video":
            assert shape is not None
            reference_audio_latents = audio_rows // MINIMAX_H3_AUDIO_CHANNELS if audio_flag else 0
            t, h, w = shape
            num_video_rows = t * (h // patch_h) * (w // patch_w)
            frame_grid, width_grid = _frame_position_grid_np(h, w, patch_h, patch_w)
            if audio_rows:
                _fill_audio_positions_np(
                    position_ids,
                    cursor,
                    audio_rows,
                    reference_audio_latents,
                    rotary_time,
                    width_grid,
                )
                audio_parts.append(np.arange(cursor, cursor + audio_rows))
            video_start_ref = cursor + audio_rows
            frame_time = _temporal_position_grid(t, rotary_time)
            position_ids[video_start_ref : video_start_ref + num_video_rows, 0] = np.repeat(
                frame_time, frame_grid.shape[0]
            )
            position_ids[video_start_ref : video_start_ref + num_video_rows, 1:] = np.tile(
                frame_grid, (t, 1)
            )
            visual_spans.append(
                VisualSpan(video_start_ref, (t, h // patch_h, w // patch_w))
            )
            video_parts.append(np.arange(video_start_ref, video_start_ref + num_video_rows))
            cursor = video_start_ref + num_video_rows
            rotary_time += max(float(reference_audio_latents), _sequential_video_span(t))
        else:
            raise RuntimeError(f"A reference must be image, video, or audio, got {kind!r}.")

    audio_start = cursor
    video_start = audio_start + num_target_audio_rows
    _fill_audio_positions_np(
        position_ids,
        audio_start,
        num_target_audio_rows,
        num_audio_latents,
        rotary_time,
        target_width_grid,
    )
    frame_time = _temporal_position_grid(num_latent_frames, rotary_time)
    position_ids[video_start:, 0] = np.repeat(frame_time, target_frame_grid.shape[0])
    position_ids[video_start:, 1:] = np.tile(target_frame_grid, (num_latent_frames, 1))

    video_parts.append(np.arange(video_start, sequence_length))
    audio_parts.append(np.arange(audio_start, video_start))
    video_indices = np.concatenate(video_parts).astype(np.int64)
    audio_indices = np.concatenate(audio_parts).astype(np.int64)
    text_indices = np.arange(num_text, dtype=np.int64)

    token_tags = np.empty(sequence_length, dtype=np.int64)
    token_tags[text_indices] = tags.astype(np.int64)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG
    if video_start + num_target_video_rows != sequence_length:
        raise RuntimeError("Ref2VA target video does not end the packed sequence.")
    target_grid = (
        int(num_latent_frames),
        int(latent_height // patch_h),
        int(latent_width // patch_w),
    )
    layout = MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_reference_video_rows,
        num_condition_audio_rows=num_reference_audio_rows,
    )
    return Ref2VAPack(
        layout=layout,
        visual_spans=tuple(visual_spans),
        target_start=video_start,
        target_grid=target_grid,
    )


def build_ref2va_presentation(
    tokenize,
    prompt: str,
    kinds: Sequence[str],
    has_audio: Sequence[bool],
    image_token_counts: Sequence[int],
    video_block_token_counts: Sequence[int],
    video_block_timestamps: Sequence[Sequence[float]],
) -> tuple[list[int], list[int]]:
    """Token ids and modality tags for the Ref2VA conditioner presentation."""

    def text(value: str) -> tuple[list[int], list[int]]:
        ids = list(tokenize(value))
        return ids, [MINIMAX_H3_TEXT_TAG] * len(ids)

    def vision(pad: int, num_tokens: int) -> tuple[list[int], list[int]]:
        ids = [_VISION_START] + [pad] * int(num_tokens) + [_VISION_END]
        return ids, [MINIMAX_H3_VIDEO_TAG] * len(ids)

    token_ids: list[int] = []
    token_tags: list[int] = []

    def emit(segment: tuple[list[int], list[int]]) -> None:
        token_ids.extend(segment[0])
        token_tags.extend(segment[1])

    counts = {"image": 0, "video": 0, "audio": 0}
    image_counts = list(image_token_counts)
    video_counts = list(video_block_token_counts)
    video_times = [list(row) for row in video_block_timestamps]
    for kind, audio_flag in zip(kinds, has_audio):
        if audio_flag:
            counts["audio"] += 1
            emit(text(f"<Audio {counts['audio']}>: "))
        if kind == "image":
            counts["image"] += 1
            emit(text(f"<Picture {counts['image']}>: "))
            emit(vision(_IMAGE_PAD, image_counts[counts["image"] - 1]))
        elif kind == "video":
            counts["video"] += 1
            emit(text(f"<Video {counts['video']}>: "))
            per_block = video_counts[counts["video"] - 1]
            for timestamp in video_times[counts["video"] - 1]:
                emit(text(f"<{timestamp:.1f} seconds>"))
                emit(vision(_VIDEO_PAD, per_block))
    emit(text(prompt))
    if len(token_ids) != len(token_tags) or not token_ids:
        raise RuntimeError("MiniMax-H3 Ref2VA presentation produced an empty token sequence.")
    return token_ids, token_tags


def ref2va_duration_ok(num_frames: int) -> None:
    duration = num_frames / MINIMAX_H3_FPS
    if duration < 5.0 or duration > MINIMAX_H3_MAX_DURATION:
        raise RuntimeError(
            f"MiniMax-H3 generates between 5 and {MINIMAX_H3_MAX_DURATION:g} seconds at "
            f"{MINIMAX_H3_FPS} fps, got {num_frames} frames ({duration:.3f}s)."
        )
