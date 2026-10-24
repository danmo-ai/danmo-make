"""Audio WAV write + asset persistence (``AudioSession`` phased helpers)."""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import soundfile as sf

from backend.core.contracts import (
    AudioEditRequest,
    AudioGenerationRequest,
    ExecutionContext,
    work_title_metadata,
)
from backend.engine._transformer_registry import audio_lyrics_metadata
from backend.engine.lineage import resolve_lineage

ACE_STEP_SAMPLE_RATE = 48_000


def quality_log_message(quality: Any) -> str | None:
    """Family-agnostic quality log message (reads grade/score/warnings attrs)."""
    if quality is None:
        return None
    grade = getattr(quality, "grade", "")
    warnings = getattr(quality, "warnings", []) or []
    score = getattr(quality, "score", 0)
    if grade == "good" and not warnings:
        return None
    warn = ", ".join(warnings) if warnings else "none"
    return f"生成质量评估: {score:.0f}/100 ({grade}); signals={warn}"


def raise_if_cancelled(exec_ctx: ExecutionContext) -> None:
    if exec_ctx.cancel_token.is_cancelled():
        raise asyncio.CancelledError()


DENOISE_FILTERS = {
    "mild": "afftdn=nr=6:nf=-45:tn=1",
    "strong": "afftdn=nr=10:nf=-42:tn=1",
}


def audio_denoise_filter(mode: str) -> str | None:
    if mode == "off":
        return None
    try:
        return DENOISE_FILTERS[mode]
    except KeyError as exc:
        raise RuntimeError(f"Unknown audio denoise mode {mode!r}; use off, mild, or strong") from exc


def apply_audio_denoise(path: Path, mode: str) -> Path:
    audio_filter = audio_denoise_filter(mode)
    if audio_filter is None:
        return path
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Audio denoise requires ffmpeg on PATH")
    tmp_path = path.with_name(f".{path.stem}.{mode}.tmp.wav")
    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(path), "-af", audio_filter, "-c:a", "pcm_s16le", str(tmp_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            detail = (proc.stderr or b"").decode(errors="ignore").strip()[:1200]
            raise RuntimeError(f"Audio denoise ({mode}) failed: {detail or 'ffmpeg error'}")
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return path


def save_audio_waveform(
    project_root: Path,
    waveform: Any,
    model_id: str,
    seed: int,
    *,
    family: str,
    sample_rate: int,
    denoise: str = "off",
) -> Path:
    """Write one waveform to ``outputs/audio/*.wav``."""
    out_dir = project_root / "outputs" / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = model_id.replace("/", "_")
    fname = f"{family}_{safe_id}_{ts}_{seed}.wav"
    out_path = out_dir / fname

    wf = np.array(waveform) if not isinstance(waveform, np.ndarray) else waveform
    if wf.ndim == 3:
        wf = wf[0]
    if wf.ndim == 1:
        wf = wf[:, None]

    sf.write(str(out_path), wf, sample_rate)
    return apply_audio_denoise(out_path, denoise)


def persist_audio_create_assets(
    asset_store: Any,
    paths: list[str],
    request: AudioGenerationRequest,
    model_id: str,
    elapsed: float,
    task_id: str,
    durations: list[float] | None,
    *,
    family: str,
    lyrics_captures: list[Any] | None = None,
) -> list[str]:
    parent_id, relation = resolve_lineage(request.metadata)
    ids: list[str] = []
    fmt = (request.audio_format or "wav").lower()
    mime = "audio/mpeg" if fmt == "mp3" else f"audio/{fmt}"
    for idx, p in enumerate(paths):
        dur = None
        if durations and idx < len(durations):
            dur = durations[idx]
        asset_meta: dict[str, Any] = {
            "model": model_id,
            "prompt": request.prompt,
            "duration_seconds": dur if dur is not None else request.duration,
            "format": fmt,
            "denoise": request.denoise,
            "elapsed_seconds": elapsed,
            "output_path": str(p),
        }
        asset_meta.update(work_title_metadata(request.title))
        lyrics_capture = (
            lyrics_captures[idx] if lyrics_captures and idx < len(lyrics_captures) else None
        )
        if lyrics_capture is not None:
            dur_meta = asset_meta.get("duration_seconds")
            asset_meta.update(
                audio_lyrics_metadata(
                    family,
                    lyrics_capture,
                    duration_sec=float(dur_meta) if dur_meta is not None else None,
                )
            )
            sidecar = Path(p).with_name(f"{Path(p).stem}_lyrics.txt")
            if sidecar.is_file():
                asset_meta["lyrics_sidecar"] = str(sidecar)
            lrc_sidecar = Path(p).with_name(f"{Path(p).stem}.lrc")
            if lrc_sidecar.is_file():
                asset_meta["lyrics_lrc_sidecar"] = str(lrc_sidecar)
        aid = asset_store.create_from_file(
            Path(p),
            kind="audio",
            mime_type=mime,
            source_task_id=task_id,
            metadata=asset_meta,
            source_action="create",
            parent_asset_id=parent_id,
            relation_type=relation,
        )
        ids.append(aid)
    return ids


def persist_audio_edit_assets(
    asset_store: Any,
    paths: list[str],
    request: AudioEditRequest,
    model_id: str,
    elapsed: float,
    task_id: str,
    durations: list[float] | None,
    *,
    quality: Any = None,
) -> list[str]:
    parent_id, relation = resolve_lineage(
        request.metadata,
        parent_asset_id=request.source_asset_id,
        relation_type="cover",
    )
    ids: list[str] = []
    fmt = (request.audio_format or "wav").lower()
    mime = "audio/mpeg" if fmt == "mp3" else f"audio/{fmt}"
    for idx, p in enumerate(paths):
        dur = None
        if durations and idx < len(durations):
            dur = durations[idx]
        asset_meta: dict[str, Any] = {
            "model": model_id,
            "operation": request.operation,
            "source_asset_id": request.source_asset_id,
            "source_fidelity": request.source_fidelity,
            "prompt": request.prompt,
            "duration_seconds": dur,
            "format": fmt,
            "elapsed_seconds": elapsed,
            "output_path": str(p),
        }
        if quality is not None:
            asset_meta.update(quality.as_metadata())
        aid = asset_store.create_from_file(
            Path(p),
            kind="audio",
            mime_type=mime,
            source_task_id=task_id,
            metadata=asset_meta,
            source_action="cover",
            parent_asset_id=parent_id,
            relation_type=relation,
        )
        ids.append(aid)
    return ids
