"""
DiffRhythm 2 generation — shared helpers + public Pipeline entry.

Pipeline and engine must import from this module only, not from ``generation_*`` internals.
Upstream reference: ``ASLP-lab/DiffRhythm2`` (block flow matching + BigVGAN).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Protocol, Tuple

import numpy as np

from backend.core.contracts import AudioGenerationRequest
from backend.engine.common.model.dit_stem import require_mlx_ctx
from backend.engine.config.model_configs import DiffRhythmConfig

SAMPLE_RATE = 48_000
LATENT_FRAME_RATE = 5.0  # Music VAE latent rate (Hz)
MAX_DURATION_SECONDS = 210

# Lyric structure tokens (ASLP-lab/DiffRhythm2 inference.py)
STRUCT_TOKEN_IDS = {
    "[start]": 500,
    "[end]": 501,
    "[intro]": 502,
    "[verse]": 503,
    "[chorus]": 504,
    "[outro]": 505,
    "[inst]": 506,
    "[solo]": 507,
    "[bridge]": 508,
    "[hook]": 509,
    "[break]": 510,
    "[stop]": 511,
    "[space]": 512,
}

DIFFRHYTHM2_BUNDLE_FILES = (
    "model.safetensors",
    "config.json",
    "decoder.bin",
    "decoder.json",
)


def duration_to_latent_frames(duration: float, *, block_size: int = 10) -> int:
    """Latent frame count at 5 Hz, aligned to ``block_size`` for CFM blocks."""
    raw = max(1, int(round(float(duration) * LATENT_FRAME_RATE)))
    if raw % block_size == 0:
        return raw
    return raw + (block_size - (raw % block_size))


def snap_latent_frames_for_inference(frames: int, *, block_size: int = 10) -> int:
    """Round up to a multiple of ``block_size``."""
    if frames % block_size == 0:
        return frames
    return frames + (block_size - (frames % block_size))


def estimate_hum_ratio(wf: np.ndarray) -> float:
    mono = wf.mean(axis=1) if wf.ndim == 2 else wf
    seg = mono[: min(len(mono), SAMPLE_RATE * 5)]
    if len(seg) < SAMPLE_RATE:
        return 0.0
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / SAMPLE_RATE)
    low = float(np.sum(spec[(freqs >= 50) & (freqs < 120)] ** 2))
    mid = float(np.sum(spec[(freqs >= 200) & (freqs < 4000)] ** 2))
    return low / (mid + 1e-12)


def estimate_mains_correlation(wf: np.ndarray, mains_hz: float = 50.0) -> float:
    mono = wf.mean(axis=1) if wf.ndim == 2 else wf
    n = min(len(mono), SAMPLE_RATE * 10)
    if n < SAMPLE_RATE:
        return 0.0
    seg = mono[:n]
    acf = np.correlate(seg, seg, mode="full")
    acf = acf[acf.size // 2 :]
    if acf[0] < 1e-12:
        return 0.0
    lag = int(round(SAMPLE_RATE / mains_hz))
    if lag >= len(acf):
        return 0.0
    return float(acf[lag] / acf[0])


def lyrics_looks_chinese(lyrics: str) -> bool:
    if not lyrics:
        return False
    cjk = len(re.findall(r"[\u4e00-\u9fff]", lyrics))
    latin = len(re.findall(r"[A-Za-z]", lyrics))
    return cjk >= 4 and cjk >= latin


def resolve_vocal_language(lyrics: str, vocal_language: str) -> str:
    explicit = (vocal_language or "").strip().lower()
    if explicit and explicit not in ("auto", "detect", "automatic"):
        return explicit
    if lyrics_looks_chinese(lyrics):
        return "zh"
    return "en"


def normalize_waveform(wf: np.ndarray) -> np.ndarray:
    """Light post-decode cleanup — upstream writes decoder output without loudness normalization."""
    if wf.ndim == 2:
        wf = wf - np.mean(wf, axis=0, keepdims=True)
    else:
        wf = wf - float(np.mean(wf))
    peak = float(np.abs(wf).max())
    rms = float(np.sqrt(np.mean(wf**2)))
    if peak < 0.02 or rms < 0.005:
        raise RuntimeError(
            f"DiffRhythm 2 decode produced near-silent audio (peak={peak:.4f}, rms={rms:.4f}). "
            "Check BigVGAN decoder weights."
        )
    if peak > 1.0:
        wf = wf / peak
    return wf.astype(np.float32)


def latent_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float32).reshape(1, -1)
    vb = np.asarray(b, dtype=np.float32).reshape(1, -1)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-12:
        return 1.0
    return float(np.sum(va * vb) / denom)


def latent_diff_mean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))


def latents_collapsed_to_silence(
    latents: Any,
    src_latents: Any,
    *,
    cos_threshold: float = 0.95,
    diff_threshold: float = 0.10,
) -> tuple[bool, float, float]:
    cos = latent_cosine_similarity(latents, src_latents)
    diff = latent_diff_mean(latents, src_latents)
    return cos >= cos_threshold and diff < diff_threshold, cos, diff


def diffusion_retry_seed(base_seed: int, attempt: int) -> int:
    if attempt <= 0:
        return int(base_seed) & 0x7FFFFFFF
    mixed = int(base_seed) ^ (0x9E3779B9 * (attempt + 1))
    mixed = (mixed * 1_103_515_245 + 12_345) & 0x7FFFFFFF
    return mixed or 1


def resolve_dit_bundle(bundle_root: Path) -> Path:
    """Resolve DiffRhythm 2 CFM checkpoint directory (``model.safetensors`` + ``config.json``)."""
    root = Path(bundle_root)
    candidates = [
        root,
        root / "diffrhythm-v2",
        root / "diffrhythm2",
        root / "ASLP-lab" / "DiffRhythm2",
    ]
    for candidate in candidates:
        if (candidate / "model.safetensors").is_file() and (candidate / "config.json").is_file():
            return candidate
    for ckpt in sorted(root.rglob("model.safetensors")):
        parent = ckpt.parent
        if (parent / "config.json").is_file():
            return parent
    raise RuntimeError(
        f"No DiffRhythm 2 checkpoint (model.safetensors + config.json) under {root}. "
        "Install ASLP-lab/DiffRhythm2 weights at bundle root or nested org/repo folder."
    )


def assert_diffrhythm2_bundle(bundle_root: Path) -> Path:
    """Fail loud when required DiffRhythm 2 bundle artifacts are missing."""
    root = resolve_dit_bundle(bundle_root)
    missing = [name for name in DIFFRHYTHM2_BUNDLE_FILES if not (root / name).is_file()]
    if missing:
        raise RuntimeError(
            f"DiffRhythm 2 bundle at {root} is incomplete; missing: {', '.join(missing)}. "
            "Download ASLP-lab/DiffRhythm2 (model.safetensors, config.json, decoder.bin, decoder.json)."
        )
    return root


# --- Public API ---


class _DiffRhythmGeneratorProto(Protocol):
    def load(self) -> None: ...
    def generate_waveform(self, **kwargs: Any) -> Any: ...
    @property
    def model_config(self) -> Any: ...


@dataclass
class DiffRhythmPreparedRequest:
    """Registry-driven inference knobs for one audio generation request."""

    lyrics: str
    vocal_language: str
    effective_prompt: str
    steps: int
    shift: float
    duration: float
    guidance: float
    log_events: List[Tuple[str, str]] = field(default_factory=list)


def create_diffrhythm_generator(ctx: Any, bundle_root: Path) -> _DiffRhythmGeneratorProto:
    require_mlx_ctx(ctx, feature="DiffRhythm 2")
    from backend.engine.families.diffrhythm.generation_mlx import DiffRhythmMlxGenerator

    return DiffRhythmMlxGenerator(ctx, bundle_root)


def prepare_music_request(
    request: AudioGenerationRequest,
    config: DiffRhythmConfig,
    bundle_root: Path,
    *,
    backend: str = "mlx",
) -> DiffRhythmPreparedRequest:
    """Resolve lyrics/style/steps for DiffRhythm 2 (MLX)."""
    _ = backend
    events: List[Tuple[str, str]] = []

    assert_diffrhythm2_bundle(bundle_root)

    registry_max = float(config.max_duration_seconds)
    raw_duration = float(request.duration if request.duration is not None else 30)
    duration = max(5.0, min(registry_max, raw_duration))
    if duration != raw_duration:
        events.append(
            (
                "warning",
                f"Duration clamped from {raw_duration}s to {duration}s "
                f"(DiffRhythm 2 max {int(registry_max)}s)",
            )
        )

    if request.instrumental:
        raise RuntimeError(
            "DiffRhythm 2 does not support instrumental-only generation yet; provide lyrics."
        )

    raw_lyrics = (request.lyrics or "").strip()
    if not raw_lyrics:
        raise RuntimeError(
            "DiffRhythm 2 requires lyrics (with optional structure tags like [verse], [chorus])."
        )
    lyrics = raw_lyrics

    vocal_lang = resolve_vocal_language(lyrics, request.vocal_language or "")
    preview = lyrics.replace("\n", " ")[:80]
    events.append(
        ("info", f"歌词已启用（约 {len(lyrics)} 字）: {preview!r}…"),
    )
    if not (request.vocal_language or "").strip():
        events.append(
            ("info", f"人声语言未指定，已根据歌词自动设为 {vocal_lang!r}"),
        )

    effective_prompt = (request.prompt or "").strip()
    if not effective_prompt:
        raise RuntimeError(
            "DiffRhythm 2 requires a style prompt (text description of genre/mood)."
        )

    steps = request.steps or config.default_infer_steps
    guidance = request.guidance if request.guidance is not None else config.default_guidance

    return DiffRhythmPreparedRequest(
        lyrics=lyrics,
        vocal_language=vocal_lang,
        effective_prompt=effective_prompt,
        steps=steps,
        shift=1.0,
        duration=duration,
        guidance=guidance,
        log_events=events,
    )
