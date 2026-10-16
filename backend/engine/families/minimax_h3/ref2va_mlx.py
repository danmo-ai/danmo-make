"""Load and encode MiniMax-H3 Ref2VA references (MLX).

Images are resized to a 2048 short edge. Videos are resampled to 24 fps and placed
on the canvas of their own aspect ratio. Soundtracks are stereo at the audio VAE
rate and stay clean. Visual references are noised to t = 0.999.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from backend.engine.families.minimax_h3.packing import (
    MINIMAX_H3_FPS,
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    PATCH_SIZE,
    patchify_video_latents,
    resolve_canvas_size,
)
from backend.engine.families.minimax_h3.ref2va import (
    image_reference_size,
    resample_video_frames,
    snap_reference_frames,
    validate_clip_duration,
    validate_reference_kinds,
    validate_total_duration,
)
from backend.engine.runtime.mlx_runtime import run_eval
from backend.utils.video_sr_ffmpeg import probe_video_fps, require_ffmpeg, require_ffprobe


@dataclass
class PreparedMedia:
    kind: str
    has_audio: bool
    image: Image.Image | None = None
    frames: np.ndarray | None = None
    waveform: np.ndarray | None = None


def _ffprobe_values(path: Path, entries: str, *, select: str) -> list[str]:
    ffprobe = require_ffprobe()
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            select,
            "-show_entries",
            entries,
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {path.name}: {(proc.stderr or proc.stdout or '').strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _decode_video_rgb(path: Path) -> tuple[np.ndarray, float]:
    values = _ffprobe_values(path, "stream=width,height", select="v:0")
    if len(values) < 2:
        raise RuntimeError(f"ffprobe did not return a frame size for {path.name}.")
    width, height = int(values[0]), int(values[1])
    if width < 1 or height < 1:
        raise RuntimeError(f"Reference video {path.name} has no frame size.")
    fps = probe_video_fps(path)
    ffmpeg = require_ffmpeg()
    proc = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to decode {path.name}: {proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    frame_bytes = width * height * 3
    if frame_bytes < 1 or len(proc.stdout) < frame_bytes or len(proc.stdout) % frame_bytes:
        raise RuntimeError(
            f"Reference video {path.name} decoded to {len(proc.stdout)} bytes, "
            f"not a whole number of {width}x{height} frames."
        )
    count = len(proc.stdout) // frame_bytes
    frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(count, height, width, 3)
    return np.array(frames), fps


def _decode_audio(path: Path, sample_rate: int, max_duration: float) -> np.ndarray:
    ffmpeg = require_ffmpeg()
    proc = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-t",
            f"{max_duration:.6f}",
            "-ac",
            "2",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to decode audio {path.name}: "
            f"{proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    if len(proc.stdout) < 8 or len(proc.stdout) % 8:
        raise RuntimeError(f"Reference audio {path.name} decoded to no stereo samples.")
    return np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 2).T.copy()


def _probe_duration(path: Path) -> float:
    ffprobe = require_ffprobe()
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(
            f"ffprobe returned no duration for {path.name}: {(proc.stderr or '').strip()}"
        )
    try:
        seconds = float(proc.stdout.strip().splitlines()[0])
    except ValueError as exc:
        raise RuntimeError(f"ffprobe returned no duration for {path.name}.") from exc
    if seconds <= 0:
        raise RuntimeError(f"Reference {path.name} has no positive duration.")
    return seconds


def _has_audio_stream(path: Path) -> bool:
    ffprobe = require_ffprobe()
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _open_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    try:
        from PIL import ImageOps

        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
    return image


def _resize_video(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    if frames.shape[1:3] == (height, width):
        return frames
    resized = [
        np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS))
        for frame in frames
    ]
    return np.stack(resized)


def prepare_references(
    specs: list[tuple[str, str]],
    *,
    num_frames: int,
    audio_sample_rate: int,
) -> list[PreparedMedia]:
    """Normalize ``(kind, path)`` references onto MiniMax-H3 rates, in request order."""
    kinds = [kind for kind, _path in specs]
    validate_reference_kinds(kinds)
    duration = num_frames / MINIMAX_H3_FPS
    video_total = 0.0
    audio_total = 0.0
    prepared: list[PreparedMedia] = []
    for index, (kind, raw_path) in enumerate(specs):
        path = Path(raw_path)
        if not path.is_file():
            raise RuntimeError(f"MiniMax-H3 reference file not found: {path}")
        if kind == "image":
            image = _open_image(path)
            height, width = image_reference_size(*image.size)
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            prepared.append(PreparedMedia(kind="image", has_audio=False, image=image))
            continue
        seconds = _probe_duration(path)
        validate_clip_duration(kind, seconds, index=index)
        if kind == "video":
            video_total += seconds
            frames, fps = _decode_video_rgb(path)
            frames = resample_video_frames(frames, fps, MINIMAX_H3_FPS)
            frames = frames[:num_frames]
            snapped = snap_reference_frames(int(frames.shape[0]))
            if snapped > int(frames.shape[0]):
                raise RuntimeError(
                    f"Reference video {path.name} is too short to encode after the 24 fps snap "
                    f"({frames.shape[0]} frames, need {snapped})."
                )
            frames = frames[:snapped]
            canvas_h, canvas_w = resolve_canvas_size(float(frames.shape[2]), float(frames.shape[1]))
            frames = _resize_video(frames, canvas_h, canvas_w)
            has_audio = _has_audio_stream(path)
            waveform = _decode_audio(path, audio_sample_rate, duration) if has_audio else None
            if has_audio:
                audio_total += min(seconds, duration)
            prepared.append(
                PreparedMedia(kind="video", has_audio=has_audio, frames=frames, waveform=waveform)
            )
        else:
            audio_total += seconds
            prepared.append(
                PreparedMedia(
                    kind="audio",
                    has_audio=True,
                    waveform=_decode_audio(path, audio_sample_rate, duration),
                )
            )
    if video_total:
        validate_total_duration("video", video_total)
    if audio_total:
        validate_total_duration("audio", audio_total)
    return prepared


def encode_visual_latents(video_vae: Any, prepared: list[PreparedMedia], *, ctx: Any) -> list[Any]:
    import mlx.core as mx

    mean = np.array(MINIMAX_H3_PIXEL_MEAN, dtype=np.float32).reshape(1, 3, 1, 1, 1)
    std = np.array(MINIMAX_H3_PIXEL_STD, dtype=np.float32).reshape(1, 3, 1, 1, 1)
    latents: list[Any] = []
    for item in prepared:
        if item.kind == "image":
            assert item.image is not None
            pixels = np.asarray(item.image, dtype=np.float32) / 255.0
            chw = np.transpose(pixels, (2, 0, 1))[None, :, None]
        elif item.kind == "video":
            assert item.frames is not None
            pixels = item.frames.astype(np.float32) / 255.0
            chw = np.transpose(pixels, (3, 0, 1, 2))[None]
        else:
            continue
        mx.random.seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
        encoded = video_vae.encode_sample(mx.array((chw - mean) / std), normalize=True)
        run_eval(getattr(ctx, "eval", None), encoded)
        latents.append(encoded)
    return latents


def encode_audio_rows(audio_vae: Any, prepared: list[PreparedMedia], *, ctx: Any) -> list[np.ndarray]:
    import mlx.core as mx

    rows: list[np.ndarray] = []
    for item in prepared:
        if not item.has_audio or item.waveform is None:
            continue
        encoded = audio_vae.encode(mx.array(item.waveform[:, None, :]), normalize=True)
        run_eval(getattr(ctx, "eval", None), encoded)
        arr = np.array(encoded.astype(mx.float32))
        rows.append(np.transpose(arr, (0, 2, 1)).reshape(-1, int(arr.shape[1])))
    return rows


def noise_visual_rows(latents: list[Any], scheduler: Any, *, ctx: Any) -> np.ndarray:
    import mlx.core as mx

    packed: list[np.ndarray] = []
    for latent in latents:
        noise = mx.random.normal(latent.shape).astype(mx.float32)
        noised = scheduler.scale_noise(latent, MINIMAX_H3_KEYFRAME_NOISE_AUG, noise)
        run_eval(getattr(ctx, "eval", None), noised)
        packed.append(patchify_video_latents(np.array(noised.astype(mx.float32)), PATCH_SIZE)[0])
    if not packed:
        raise RuntimeError("MiniMax-H3 Ref2VA encoded no visual references.")
    return np.concatenate(packed, axis=0)
