"""FFmpeg/ffprobe helpers for SeedVR2 视频修复（抽帧 / 封装）。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _exe_stem(name: str) -> str:
    return f"{name}.exe" if sys.platform == "win32" else name


def _sibling_exe(other_resolved: str, want: str) -> str | None:
    """If ``ffmpeg`` is found at ``.../bin/ffmpeg``, try ``.../bin/ffprobe`` (Homebrew layout)."""
    p = Path(other_resolved)
    sib = p.with_name(_exe_stem(want))
    return str(sib) if sib.is_file() else None


def resolve_ffmpeg() -> str | None:
    """Return absolute path to ``ffmpeg``, or ``None``."""
    raw = os.environ.get("FFMPEG_PATH")
    if raw:
        p = Path(raw).expanduser()
        if p.is_file():
            return str(p)
    w = shutil.which("ffmpeg")
    if w:
        return w
    fp = shutil.which("ffprobe")
    if fp:
        sib = _sibling_exe(fp, "ffmpeg")
        if sib:
            return sib
    for root in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"):
        cand = Path(root) / _exe_stem("ffmpeg")
        if cand.is_file():
            return str(cand)
    return None


def resolve_ffprobe() -> str | None:
    """Return absolute path to ``ffprobe``, or ``None``."""
    raw = os.environ.get("FFPROBE_PATH")
    if raw:
        p = Path(raw).expanduser()
        if p.is_file():
            return str(p)
    w = shutil.which("ffprobe")
    if w:
        return w
    ff = resolve_ffmpeg()
    if ff:
        sib = _sibling_exe(ff, "ffprobe")
        if sib:
            return sib
    for root in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"):
        cand = Path(root) / _exe_stem("ffprobe")
        if cand.is_file():
            return str(cand)
    return None


def require_ffmpeg() -> str:
    exe = resolve_ffmpeg()
    if not exe:
        raise RuntimeError(
            "ffmpeg not found (checked PATH, FFMPEG_PATH, /opt/homebrew/bin, /usr/local/bin). "
            "Install ffmpeg (e.g. `brew install ffmpeg`) or set FFMPEG_PATH to the binary."
        )
    return exe


def require_ffprobe() -> str:
    exe = resolve_ffprobe()
    if not exe:
        raise RuntimeError(
            "ffprobe not found (checked PATH, FFPROBE_PATH, sibling of ffmpeg, /opt/homebrew/bin, /usr/local/bin). "
            "Install ffmpeg (ffprobe ships with it) or set FFPROBE_PATH."
        )
    return exe


def _parse_rate(expr: str) -> float:
    s = (expr or "").strip()
    if not s or s == "0/0":
        return 0.0
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / max(float(b), 1e-9)
    return float(s)


def probe_video_fps(video: Path) -> float:
    """Return a positive FPS from ffprobe.

    Raises ``RuntimeError`` if ffprobe is missing, fails, or cannot read a valid frame rate
    (no silent fallback to request defaults).
    """
    ffprobe = resolve_ffprobe()
    if not ffprobe:
        raise RuntimeError(
            "Cannot determine video frame rate: ffprobe not found. "
            "Set PATH or FFPROBE_PATH, or install ffmpeg (ffprobe ships with it). "
            "Request `fps` is not used as a fallback."
        )
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(video),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Cannot determine video frame rate: ffprobe failed (exit {proc.returncode}) for {video}: "
            f"{(proc.stderr or proc.stdout or '').strip()[:800]}"
        )
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(
            f"Cannot determine video frame rate: ffprobe found no video stream in {video}."
        )
    st0 = streams[0]
    for key in ("r_frame_rate", "avg_frame_rate"):
        fps = _parse_rate(str(st0.get(key) or ""))
        if fps > 1e-3:
            return fps
    raise RuntimeError(
        f"Cannot determine video frame rate: ffprobe returned no usable r_frame_rate/avg_frame_rate "
        f"for {video} (stream keys: {list(st0.keys())})."
    )


def video_has_audio_stream(video: Path) -> bool:
    ffprobe = resolve_ffprobe()
    if not ffprobe:
        return False
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(video),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return False
    return bool((proc.stdout or "").strip())


def extract_png_frames(
    *,
    video: Path,
    out_dir: Path,
    max_frames: int,
) -> int:
    """Decode up to ``max_frames`` video frames to ``out_dir/frame_%06d.png``. Returns frame count on disk."""
    ffmpeg = require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%06d.png")
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        # ffmpeg 8+ dropped -vsync; 0 was passthrough (keep source timestamps).
        "-fps_mode",
        "passthrough",
        "-vframes",
        str(int(max_frames)),
        pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg frame extract failed (exit {proc.returncode}): {err[:1200]}")
    return len(sorted(out_dir.glob("frame_*.png")))


def mux_png_sequence_to_mp4(
    *,
    png_pattern: Path,
    source_audio_from: Path,
    output_mp4: Path,
    fps: float,
    include_audio: bool,
) -> None:
    """Mux ``png_pattern`` (printf-style e.g. /tmp/up/up_%06d.png) to H.264 MP4; optionally copy AAC from source."""
    ffmpeg = require_ffmpeg()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    fps_s = f"{max(fps, 1e-3):.6f}".rstrip("0").rstrip(".")
    if include_audio and video_has_audio_stream(source_audio_from):
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            fps_s,
            "-i",
            str(png_pattern),
            "-i",
            str(source_audio_from),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_mp4),
        ]
    else:
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            fps_s,
            "-i",
            str(png_pattern),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(output_mp4),
        ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg mux failed (exit {proc.returncode}): {err[:1200]}")
