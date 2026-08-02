"""Video segment stitch + last-frame extract (ffmpeg, fail loud)."""
from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from backend.long_video.plan import num_frames_for_duration_sec
from backend.engine.families.ltx.pipeline_math_mlx import AUDIO_SAMPLE_RATE
from backend.utils.video_sr_ffmpeg import require_ffmpeg, require_ffprobe


def _extract_video_rgb(video: Path, *, fps: float) -> np.ndarray:
    ffmpeg = require_ffmpeg()
    probe = subprocess.run(
        [ffmpeg, "-i", str(video), "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    import re

    m = re.search(r",\s*(\d{2,5})x(\d{2,5})", probe.stderr or "")
    if not m:
        raise RuntimeError(f"video stitch: cannot probe size of {video}")
    w, h = int(m.group(1)), int(m.group(2))
    rate = max(1.0, float(fps))
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps={rate:.6f},format=rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"video stitch: rgb extract failed for {video}")
    fb = w * h * 3
    n = len(proc.stdout) // fb
    return np.frombuffer(proc.stdout[: n * fb], dtype=np.uint8).reshape(n, h, w, 3).astype(np.float32) / 127.5 - 1.0


def _read_wav_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        rate = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"video stitch: unsupported WAV sample width {sw}")
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    if rate != AUDIO_SAMPLE_RATE and data.size > 0:
        ffmpeg = require_ffmpeg()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            in_path = tmp_in.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            out_path = tmp_out.name
        try:
            with wave.open(in_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(rate)
                wf.writeframes((np.clip(data, -1, 1) * 32767).astype(np.int16).tobytes())
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                in_path,
                "-ac",
                "1",
                "-ar",
                str(AUDIO_SAMPLE_RATE),
                out_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return _read_wav_mono(Path(out_path))
        finally:
            Path(in_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)
    return data.astype(np.float32)


def append_video_with_crossfade(
    *,
    accumulator: Path,
    new_segment: Path,
    output: Path,
    overlap_frames: int,
    fps: float,
) -> None:
    """Append ``new_segment`` to ``accumulator`` with linear crossfade on overlap."""
    acc = Path(accumulator)
    seg = Path(new_segment)
    out = Path(output)
    if not acc.is_file():
        raise RuntimeError(f"video stitch: accumulator missing: {acc}")
    if not seg.is_file():
        raise RuntimeError(f"video stitch: new segment missing: {seg}")

    overlap = max(0, int(overlap_frames))
    rate = max(1.0, float(fps))
    acc_rgb = _extract_video_rgb(acc, fps=rate)
    seg_rgb = _extract_video_rgb(seg, fps=rate)

    ffmpeg = require_ffmpeg()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ta:
        acc_wav = ta.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ts:
        seg_wav = ts.name
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(acc),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(AUDIO_SAMPLE_RATE),
                acc_wav,
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(seg),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(AUDIO_SAMPLE_RATE),
                seg_wav,
            ],
            check=True,
            capture_output=True,
        )
        acc_a = _read_wav_mono(Path(acc_wav))
        seg_a = _read_wav_mono(Path(seg_wav))
    finally:
        Path(acc_wav).unlink(missing_ok=True)
        Path(seg_wav).unlink(missing_ok=True)

    if overlap <= 0 or acc_rgb.shape[0] == 0 or seg_rgb.shape[0] == 0:
        merged_rgb = np.concatenate([acc_rgb, seg_rgb], axis=0)
        merged_a = np.concatenate([acc_a, seg_a], axis=0)
    else:
        ov = min(overlap, acc_rgb.shape[0], seg_rgb.shape[0])
        weights = np.linspace(0.0, 1.0, ov, dtype=np.float32)[:, None, None, None]
        blend = acc_rgb[-ov:] * (1.0 - weights) + seg_rgb[:ov] * weights
        merged_rgb = np.concatenate([acc_rgb[:-ov], blend, seg_rgb[ov:]], axis=0)
        ov_a = min(int(ov * AUDIO_SAMPLE_RATE / rate), acc_a.shape[0], seg_a.shape[0])
        if ov_a > 0:
            aw = np.linspace(0.0, 1.0, ov_a, dtype=np.float32)
            ab = acc_a[-ov_a:] * (1.0 - aw) + seg_a[:ov_a] * aw
            merged_a = np.concatenate([acc_a[:-ov_a], ab, seg_a[ov_a:]], axis=0)
        else:
            merged_a = np.concatenate([acc_a, seg_a], axis=0)

    h, w = merged_rgb.shape[1], merged_rgb.shape[2]
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tw:
        wav_path = tw.name
    try:
        pcm = np.clip(merged_a, -1.0, 1.0)
        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(AUDIO_SAMPLE_RATE)
            wf.writeframes((pcm * 32767).astype(np.int16).tobytes())

        raw = np.clip((merged_rgb + 1.0) * 127.5, 0, 255).astype(np.uint8).tobytes()
        fps_s = f"{rate:.6f}".rstrip("0").rstrip(".")
        cmd_v = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            fps_s,
            "-i",
            "pipe:0",
            "-i",
            wav_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(out),
        ]
        proc = subprocess.run(cmd_v, input=raw, capture_output=True, check=False)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or b"").decode(errors="ignore")
            raise RuntimeError(f"video stitch: mux append failed: {err[:1200]}")
    finally:
        Path(wav_path).unlink(missing_ok=True)


def extract_first_frame_image(
    video_path: Path,
    *,
    output_path: Path,
) -> Path:
    """Extract the first frame of a video as PNG."""
    ffmpeg = require_ffmpeg()
    require_ffprobe()
    video_path = Path(video_path)
    output_path = Path(output_path)
    if not video_path.is_file():
        raise RuntimeError(f"video stitch: video not found: {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-update",
        "1",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not output_path.is_file():
        err = (proc.stderr or proc.stdout or b"").decode(errors="ignore")
        raise RuntimeError(f"video stitch: first frame extract failed: {err[:800]}")
    return output_path


def extract_last_frame_image(
    video_path: Path,
    *,
    output_path: Path,
) -> Path:
    """Extract the last frame of a video as PNG."""
    ffmpeg = require_ffmpeg()
    require_ffprobe()
    video_path = Path(video_path)
    output_path = Path(output_path)
    if not video_path.is_file():
        raise RuntimeError(f"video stitch: video not found: {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-sseof",
        "-0.04",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-update",
        "1",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not output_path.is_file():
        err = (proc.stderr or proc.stdout or b"").decode(errors="ignore")
        raise RuntimeError(f"video stitch: last frame extract failed: {err[:800]}")
    return output_path


def stitch_segment_paths(
    segment_paths: list[Path],
    *,
    output: Path,
    overlap_frames: int,
    fps: float,
) -> Path:
    """Stitch ordered segment videos into one file."""
    if not segment_paths:
        raise RuntimeError("video stitch: no segments to stitch")
    if len(segment_paths) == 1:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(segment_paths[0], out)
        return out

    work = Path(output).parent
    acc = work / "_stitch_acc.mp4"
    import shutil

    shutil.copy2(segment_paths[0], acc)
    for idx, seg in enumerate(segment_paths[1:], start=1):
        nxt = work / f"_stitch_tmp_{idx}.mp4"
        append_video_with_crossfade(
            accumulator=acc,
            new_segment=Path(seg),
            output=nxt,
            overlap_frames=overlap_frames,
            fps=fps,
        )
        acc.unlink(missing_ok=True)
        acc = nxt
    out = Path(output)
    shutil.move(str(acc), str(out))
    return out
