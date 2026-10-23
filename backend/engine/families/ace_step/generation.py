"""
ACE-Step generation — shared helpers + public Pipeline entry.

Pipeline and engine must import from this module only, not from ``generation_*`` internals.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

from backend.core.contracts import AudioEditRequest, AudioGenerationRequest
from backend.engine.common.model.dit_stem import require_mlx_ctx
from backend.engine.config.model_configs import AceStepConfig
from backend.engine.families.ace_step.lm.lm_format import (
    is_instrumental_lyrics,
    normalize_lyrics_body,
    vocal_lyrics_required_error,
)
from backend.engine.families.ace_step.vocals.vocal_prompt import apply_vocal_type_to_prompt

SFT_GEN_PROMPT = """# Instruction
{instruction}

# Caption
{caption}

# Metas
{metadata}<|endoftext|>
"""

DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
COVER_DIT_INSTRUCTION = (
    "Generate audio semantic tokens based on the given conditions:"
)
DEFAULT_LM_SIMPLE_INSTRUCTION = (
    "Generate audio semantic tokens based on the given conditions:"
)

SAMPLE_RATE = 48_000
LATENT_HOP_SAMPLES = 1920
VAE_TILE_CHUNK_SIZE = 512
VAE_TILE_MIN_CHUNK_SIZE = 192
VAE_TILE_OVERLAP = 64


TIMBRE_REFERENCE_FRAMES = 750
TIMBRE_SEGMENT_FRAMES = 250


def sample_timbre_reference_latents(latents: np.ndarray, seed: int) -> np.ndarray:
    """30 s timbre reference from front/middle/back 10 s segments (upstream ``process_reference_audio``).

    Upstream samples in waveform space before VAE encode; this samples the already-encoded
    ``[1, T, C]`` latents at 25 Hz so the source audio is encoded once.
    """
    lat = np.asarray(latents, dtype=np.float32)
    if lat.ndim == 2:
        lat = lat[np.newaxis, ...]
    total = int(lat.shape[1])
    if total <= 0:
        raise RuntimeError("ACE-Step timbre reference latents are empty")
    if total < TIMBRE_REFERENCE_FRAMES:
        lat = np.tile(lat, (1, int(math.ceil(TIMBRE_REFERENCE_FRAMES / total)), 1))
        total = int(lat.shape[1])
    rng = np.random.default_rng(int(seed))
    seg = total // 3
    starts = (
        int(rng.integers(0, max(0, seg - TIMBRE_SEGMENT_FRAMES) + 1)),
        seg + int(rng.integers(0, max(0, seg - TIMBRE_SEGMENT_FRAMES) + 1)),
        2 * seg + int(rng.integers(0, max(0, (total - 2 * seg) - TIMBRE_SEGMENT_FRAMES) + 1)),
    )
    return np.concatenate(
        [lat[:, s : s + TIMBRE_SEGMENT_FRAMES, :] for s in starts], axis=1
    ).astype(np.float32, copy=False)


def vae_tile_windows(t_len: int, chunk_size: int = VAE_TILE_CHUNK_SIZE) -> list[tuple[int, int, int, int]]:
    """Overlap-discard VAE decode windows ``(win_start, win_end, core_start, core_end)``.

    Mirrors upstream MLX ``_mlx_decode_single``: chunk is clamped to ≥192 latent
    frames and each window carries 64 frames of context that is trimmed after decode.
    """
    chunk = max(VAE_TILE_MIN_CHUNK_SIZE, int(chunk_size))
    if t_len <= chunk:
        return [(0, t_len, 0, t_len)]
    stride = chunk - 2 * VAE_TILE_OVERLAP
    windows = []
    for idx in range((t_len + stride - 1) // stride):
        core_start = idx * stride
        core_end = min(core_start + stride, t_len)
        windows.append(
            (
                max(0, core_start - VAE_TILE_OVERLAP),
                min(t_len, core_end + VAE_TILE_OVERLAP),
                core_start,
                core_end,
            )
        )
    return windows


def duration_to_latent_frames(duration: float) -> int:
    return max(128, int(round(float(duration) * SAMPLE_RATE / LATENT_HOP_SAMPLES)))


def snap_latent_frames_for_inference(frames: int) -> int:
    """Prefer even latent length (odd turbo lengths can collapse to silence/hum)."""
    if frames % 2 == 1:
        return max(128, frames - 1)
    return frames


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
    """Pick vocal language for ``# Languages`` block; auto-detect when UI leaves it empty."""
    explicit = (vocal_language or "").strip().lower()
    if explicit and explicit not in ("auto", "detect", "automatic"):
        return explicit
    if lyrics_looks_chinese(lyrics):
        return "zh"
    return "en"


def vocal_language_mismatch_warning(lyrics: str, lang: str) -> Optional[str]:
    if is_instrumental_lyrics(lyrics):
        return None
    lang_l = (lang or "").strip().lower()
    if lyrics_looks_chinese(lyrics) and lang_l.startswith("en"):
        return (
            "歌词为中文但人声语言为 en；条件不一致，演唱可能极弱或像器乐。"
            "请将「人声语言」设为 zh（中文）。"
        )
    return None


def format_lyrics(lyrics: str, language: str) -> str:
    body = normalize_lyrics_body(lyrics)
    return f"# Languages\n{language}\n\n# Lyric\n{body}<|endoftext|>"


def finalize_lyrics_for_inference(
    lyrics: str,
    *,
    instrumental: bool,
    lm_expanded: bool,
) -> str:
    """Normalize lyrics and fail loud when a vocal track has no singable text."""
    if instrumental:
        return "[Instrumental]"
    body = normalize_lyrics_body(lyrics)
    if body and not is_instrumental_lyrics(body):
        return body
    if lm_expanded:
        raise RuntimeError(vocal_lyrics_required_error())
    return "[Instrumental]"


@dataclass
class AceStepLyricsCapture:
    """Lyrics/caption actually fed to DiT after optional 5Hz LM rewrite."""

    lyrics_input: str = ""
    lyrics_effective: str = ""
    caption_effective: str = ""
    lm_expanded: bool = False
    lyrics_changed: bool = False
    # ``LyricsAlignmentResult`` from DiT cross-attention; ``None`` → structure estimate.
    timestamps: Any = None


def _lyrics_alignment_for(
    cap: AceStepLyricsCapture,
    duration_sec: float,
) -> Any:
    from backend.engine.families.ace_step.vocals.lyrics_alignment import estimate_lyrics_alignment

    if cap.timestamps is not None and cap.timestamps.segments:
        return cap.timestamps
    return estimate_lyrics_alignment(
        cap.lyrics_effective,
        duration_sec=float(duration_sec),
        instrumental=False,
    )


def capture_inference_lyrics(
    *,
    lyrics_input: str,
    lyrics_effective: str,
    caption_effective: str,
    lm_expanded: bool,
) -> AceStepLyricsCapture:
    inp = (lyrics_input or "").strip()
    eff = (lyrics_effective or "").strip()
    return AceStepLyricsCapture(
        lyrics_input=inp,
        lyrics_effective=eff,
        caption_effective=(caption_effective or "").strip(),
        lm_expanded=bool(lm_expanded),
        lyrics_changed=bool(lm_expanded) and eff != inp,
    )


def lyrics_capture_log_message(
    cap: AceStepLyricsCapture,
    *,
    preview_chars: int = 500,
) -> Optional[str]:
    """User-facing task log for effective lyrics (not word-level timed)."""
    if is_instrumental_lyrics(cap.lyrics_effective):
        return None
    text = (cap.lyrics_effective or "").strip()
    if not text:
        return None
    preview = text if len(text) <= preview_chars else text[:preview_chars] + "…"
    header = (
        "5Hz LM 扩写后歌词（送入模型的条件文本）"
        if cap.lyrics_changed
        else "最终歌词（送入模型的条件文本）"
    )
    return (
        f"{header}；与音频非逐字时间轴对齐，仅作演唱条件。\n"
        f"{preview}"
    )


def lyrics_capture_metadata(
    cap: AceStepLyricsCapture,
    *,
    duration_sec: Optional[float] = None,
) -> dict[str, Any]:
    """Serializable fields for task result / asset metadata."""
    if is_instrumental_lyrics(cap.lyrics_effective):
        return {
            "lyrics_alignment": "instrumental",
            "lm_expanded": cap.lm_expanded,
        }
    meta = {
        "lyrics_input": cap.lyrics_input,
        "lyrics_effective": cap.lyrics_effective,
        "lyrics_changed_by_lm": cap.lyrics_changed,
        "caption_effective": cap.caption_effective,
        "lm_expanded": cap.lm_expanded,
    }
    if duration_sec is not None and duration_sec > 0:
        meta.update(_lyrics_alignment_for(cap, float(duration_sec)).as_metadata())
    else:
        meta["lyrics_alignment"] = "conditioning_only"
    return meta


def write_lyrics_sidecar(audio_path: Path, lyrics: str) -> Optional[Path]:
    """Write ``<stem>_lyrics.txt`` next to the audio file."""
    if is_instrumental_lyrics(lyrics):
        return None
    text = (lyrics or "").strip()
    if not text:
        return None
    sidecar = audio_path.with_name(f"{audio_path.stem}_lyrics.txt")
    sidecar.write_text(text + "\n", encoding="utf-8")
    return sidecar


def write_lrc_sidecar(
    audio_path: Path,
    lyrics: str,
    *,
    duration_sec: float,
    alignment: Any = None,
) -> Optional[Path]:
    """Write ``<stem>.lrc`` (DiT-aligned when ``alignment`` is given, else structure estimate)."""
    from backend.engine.families.ace_step.vocals.lyrics_alignment import (
        estimate_lyrics_alignment,
        format_lrc,
    )

    if is_instrumental_lyrics(lyrics) or duration_sec <= 0:
        return None
    if alignment is None or not alignment.segments:
        alignment = estimate_lyrics_alignment(lyrics, duration_sec=float(duration_sec))
    lrc_text = format_lrc(alignment)
    if not lrc_text:
        return None
    sidecar = audio_path.with_name(f"{audio_path.stem}.lrc")
    sidecar.write_text(lrc_text, encoding="utf-8")
    return sidecar


LYRIC_TOKEN_MAX_LENGTH = 2048


def ace_step_condition_side_config(raw: dict) -> dict:
    """Runtime config for condition encoder / tokenizer / detokenizer.

    XL DiT ``hidden_size`` (e.g. 2560) differs from ``encoder_hidden_size`` (2048).
    Upstream builds a deepcopy and overlays encoder_* onto the side modules.
    """
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError("ACE-Step config must be a non-empty dict")
    cfg = dict(raw)
    cfg["hidden_size"] = int(raw.get("encoder_hidden_size", raw["hidden_size"]))
    cfg["intermediate_size"] = int(
        raw.get("encoder_intermediate_size", raw["intermediate_size"])
    )
    cfg["num_attention_heads"] = int(
        raw.get("encoder_num_attention_heads", raw["num_attention_heads"])
    )
    cfg["num_key_value_heads"] = int(
        raw.get("encoder_num_key_value_heads", raw["num_key_value_heads"])
    )
    return cfg


def warn_lyrics_char_heuristic(lyrics: str) -> Optional[str]:
    """Rough pre-check before tokenization (Chinese-heavy text ~1–2 chars/token)."""
    if is_instrumental_lyrics(lyrics):
        return None
    n = len(lyrics)
    if n >= 3500:
        return (
            f"歌词约 {n} 字，很可能超过模型 {LYRIC_TOKEN_MAX_LENGTH} token 上限，"
            "后段可能被截断而唱不出来；建议缩短或分段生成。"
        )
    if n >= 2200:
        return (
            f"歌词约 {n} 字，接近 token 上限，后段对齐可能变弱；"
            "可精简段落或拆成多次生成。"
        )
    return None


def warn_lyrics_token_truncation(
    token_count: int,
    *,
    max_length: int = LYRIC_TOKEN_MAX_LENGTH,
) -> Optional[str]:
    """Warn when lyric conditioning will be truncated (later verses often not sung)."""
    if token_count >= max_length:
        return (
            f"歌词编码已达 {token_count} token（上限 {max_length}），尾部已被截断，"
            "后段歌词很可能唱不出来。请缩短歌词或分段多次生成。"
        )
    if token_count >= int(max_length * 0.85):
        return (
            f"歌词较长（约 {token_count} token，上限 {max_length}），"
            "后段对齐可能变弱，建议精简或只保留关键段落。"
        )
    return None


def warn_weak_vocal_lyrics(lyrics: str) -> Optional[str]:
    """Return a user-facing warning when lyrics look unlikely to produce audible vocals."""
    if is_instrumental_lyrics(lyrics):
        return None
    lower = lyrics.lower()
    score_tags = (
        "cinematic",
        "instrumental",
        "guzheng",
        "war drum",
        "orchestral",
        "soundtrack",
        "bgm",
        "配乐",
        "史诗",
        "管弦",
        "古筝",
        "战鼓",
    )
    vocal_tags = (
        "vocal",
        "sing",
        "singer",
        "voice",
        "人声",
        "演唱",
        "歌声",
        "合唱",
        "主唱",
    )
    if any(k in lower for k in score_tags) and not any(k in lower for k in vocal_tags):
        return (
            "歌词/段落标签偏电影配乐或器乐（如 cinematic、guzheng、war drums），"
            "未强调演唱；成片可能几乎听不到人声。建议在歌词中写清可唱词句，"
            "并在描述中加入「清晰人声/主唱演唱」。"
        )
    bracket_tags = re.findall(r"\[[^\]]+\]", lyrics)
    singable_lines = [
        ln.strip()
        for ln in lyrics.splitlines()
        if ln.strip() and not ln.strip().startswith("[")
    ]
    singable_chars = sum(len(ln) for ln in singable_lines)
    if len(bracket_tags) >= 4 and singable_chars < 80:
        return (
            "歌词以 [Intro]/[Verse] 等制作标签为主、可唱正文很少；"
            "请在各段下补充具体歌词句子，否则模型易生成纯配乐。"
        )
    return None


def ensure_vocal_caption_hint(caption: str, lyrics: str, language: str = "en") -> str:
    """Nudge DiT caption toward lead vocals when non-instrumental lyrics are present."""
    cap = (caption or "").strip()
    if is_instrumental_lyrics(lyrics):
        return cap
    lower = cap.lower()
    lang = (language or "en").lower()
    vocal_in_cap = any(
        k in lower
        for k in (
            "vocal", "singing", "singer", "voice", "人声", "演唱", "歌声", "主唱",
            "male", "female", "choir", "chorus", "duet", "男声", "女声", "合唱", "对唱",
        )
    )
    if vocal_in_cap:
        return cap
    if lang.startswith("zh"):
        suffix = " 清晰的中文主唱人声，按所给歌词演唱，人声位于前景。"
        default = "带清晰中文主唱人声演唱的歌曲，人声位于前景。"
    else:
        suffix = " Clear lead vocals singing the provided lyrics, vocals in the front."
        default = "Song with clear lead vocals singing the provided lyrics."
    return (cap + suffix) if cap else default


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


def resolve_dit_bundle(bundle_root: Path, *, dit_subdir: str | None = None) -> Path:
    root = Path(bundle_root)
    if dit_subdir:
        sub = root / dit_subdir
        if sub.is_dir() and (
            (sub / "model.safetensors").is_file()
            or (sub / "model.safetensors.index.json").is_file()
        ):
            return sub
        raise RuntimeError(
            f"ACE-Step DiT subdir {dit_subdir!r} not found under {root}. "
            "Install the matching registry model version "
            "(shared VAE/text encoder bundle + XL DiT checkpoint required)."
        )
    if (root / "model.safetensors").is_file() or (root / "model.safetensors.index.json").is_file():
        return root
    for sub_name in (
        "acestep-v15-xl-sft",
        "acestep-v15-xl-turbo",
        "acestep-v15-xl-base",
        "acestep-v15-sft",
        "acestep-v15-turbo",
        "acestep-v15-base",
    ):
        sub = root / sub_name
        if sub.is_dir() and (
            (sub / "model.safetensors").is_file()
            or (sub / "model.safetensors.index.json").is_file()
        ):
            return sub
    raise RuntimeError(
        f"No ACE-Step DiT checkpoint (model.safetensors) under {root}. "
        "Expected weights at bundle root or in acestep-v15-*/ subdirectory."
    )


def resolve_silence_latent_path(bundle_root: Path, dit_bundle: Path) -> Path:
    for candidate in (dit_bundle / "silence_latent.pt", bundle_root / "silence_latent.pt"):
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        f"silence_latent.pt not found under {bundle_root} or {dit_bundle}"
    )


def resolve_silence_latent_npy_path(pt_path: Path) -> Path:
    return pt_path.with_suffix(".npy")


def load_silence_latent_numpy(pt_path: Path) -> np.ndarray:
    """Load silence latent as ``[B, T, C]`` float32 (``.npy`` cache or torch-free ``.pt``)."""
    npy_path = resolve_silence_latent_npy_path(pt_path)
    if npy_path.is_file():
        arr = np.asarray(np.load(npy_path), dtype=np.float32)
    else:
        from backend.engine.common.bundle.pytorch_bin_numpy import load_pytorch_bin

        obj = load_pytorch_bin(pt_path)
        if not isinstance(obj, np.ndarray):
            raise RuntimeError(
                f"silence_latent checkpoint must be a tensor ndarray, got {type(obj)!r}: {pt_path}"
            )
        arr = np.asarray(obj, dtype=np.float32)
        # Upstream stores [B, C, T]; runtime slices expect [B, T, C].
        if arr.ndim == 3 and arr.shape[1] < arr.shape[2] and arr.shape[1] <= 256:
            arr = np.transpose(arr, (0, 2, 1))
        try:
            np.save(npy_path, arr)
        except OSError:
            pass
    if arr.ndim != 3:
        raise RuntimeError(
            f"silence_latent must be rank-3 [B,T,C], got shape={arr.shape} from {pt_path}"
        )
    return arr


def build_nonperiodic_src_latents(silence_latent: np.ndarray, length: int) -> np.ndarray:
    """Build text-to-music ``src`` latents without tiling the silence template.

    Uses the mean channel vector of ``silence_latent`` broadcast to ``length``
    frames so long clips do not receive a periodic silence texture as DiT context.
    """
    arr = np.asarray(silence_latent, dtype=np.float32)
    if arr.ndim != 3:
        raise RuntimeError(
            f"silence_latent must be rank-3 [B,T,C], got shape={arr.shape}"
        )
    length = int(length)
    if length < 1:
        raise RuntimeError(f"src latent length must be >= 1, got {length}")
    channels = int(arr.shape[-1])
    mean_vec = np.mean(arr[0], axis=0, keepdims=True).astype(np.float32)
    return np.broadcast_to(mean_vec, (length, channels)).copy()


def format_metadata(
    *,
    duration: float,
    bpm: Optional[int] = None,
    key_scale: str = "",
    time_signature: str = "",
) -> str:
    bpm_s = bpm if bpm is not None else "N/A"
    ts_s = time_signature if time_signature else "N/A"
    key_s = key_scale if key_scale else "N/A"
    dur_s = f"{int(round(duration))} seconds"
    return (
        f"- bpm: {bpm_s}\n"
        f"- timesignature: {ts_s}\n"
        f"- keyscale: {key_s}\n"
        f"- duration: {dur_s}\n"
    )


def load_reference_waveform(path: Path, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load mono/stereo WAV for cover/reference tasks → float32 [T, C]."""
    import soundfile as sf

    wf, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if sr != sample_rate:
        raise RuntimeError(
            f"Reference audio must be {sample_rate} Hz (got {sr} Hz): {path}"
        )
    if wf.shape[0] < sample_rate // 2:
        raise RuntimeError(f"Reference audio too short (<0.5s): {path}")
    return wf.astype(np.float32)


def normalize_waveform(wf: np.ndarray) -> np.ndarray:
    if wf.ndim == 2:
        wf = wf - np.mean(wf, axis=0, keepdims=True)
    else:
        wf = wf - float(np.mean(wf))
    peak = float(np.abs(wf).max())
    rms = float(np.sqrt(np.mean(wf**2)))
    if peak < 1e-5 or rms < 1e-6:
        raise RuntimeError(
            f"ACE-Step VAE decode produced near-silent audio (peak={peak:.2e}, rms={rms:.2e})"
        )
    if peak > 1.0:
        wf = wf / peak
    target_amp = 10.0 ** (-1.0 / 20.0)
    peak = float(np.abs(wf).max())
    if peak > 1e-8:
        wf = wf * (target_amp / peak)
    return wf

# --- Public API ---

class _AceStepGeneratorProto(Protocol):
    def load(self) -> None: ...
    def generate_waveform(self, **kwargs: Any) -> Any: ...
    def generate_cover_waveform(self, **kwargs: Any) -> Any: ...
    @property
    def model_config(self) -> Any: ...


@dataclass
class AceStepPreparedRequest:
    """Registry-driven inference knobs for one audio generation request."""

    lyrics: str
    vocal_language: str
    effective_prompt: str
    steps: int
    shift: float
    is_turbo: bool
    duration: float
    guidance: float = 1.0
    lm_enabled: bool = True
    resource_tier: str = ""
    lm_quantize_bits: Optional[int] = None
    log_events: List[Tuple[str, str]] = field(default_factory=list)
    sampler_kwargs: Dict[str, Any] = field(default_factory=dict)


DCW_MODES = ("low", "high", "double", "pix")


@dataclass(frozen=True)
class AceStepDcwSettings:
    """Differential Correction in Wavelet domain (upstream ``dcw_*`` sampler knobs)."""

    enabled: bool
    mode: str = "double"
    scaler: float = 0.05
    high_scaler: float = 0.02

    def sampler_kwargs(self) -> Dict[str, Any]:
        return {
            "dcw_enabled": self.enabled,
            "dcw_mode": self.mode,
            "dcw_scaler": self.scaler,
            "dcw_high_scaler": self.high_scaler,
        }

    def describe(self) -> str:
        if not self.enabled:
            return "DCW off"
        return (
            f"DCW {self.mode} (haar, scaler={self.scaler:g}, "
            f"high_scaler={self.high_scaler:g})"
        )


def _metadata_bool(raw: Any, key: str) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise RuntimeError(f"ACE-Step metadata {key}={raw!r} is not a boolean")


def _metadata_float(raw: Any, key: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"ACE-Step metadata {key}={raw!r} is not a number") from exc
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise RuntimeError(f"ACE-Step metadata {key}={value} must be within [0, 1]")
    return value


def resolve_dcw_settings(
    metadata: Optional[Dict[str, Any]],
    config: AceStepConfig,
    *,
    is_turbo: bool,
) -> AceStepDcwSettings:
    """``metadata.dcw_*`` overrides on top of ``AceStepConfig``; enabled defaults to turbo."""
    meta = metadata or {}
    enabled = (
        _metadata_bool(meta["dcw_enabled"], "dcw_enabled")
        if meta.get("dcw_enabled") is not None
        else bool(is_turbo)
    )
    mode = str(meta.get("dcw_mode") or config.dcw_mode).strip().lower()
    if mode not in DCW_MODES:
        raise RuntimeError(f"ACE-Step dcw_mode={mode!r}; expected one of {', '.join(DCW_MODES)}")
    wavelet = str(meta.get("dcw_wavelet") or "haar").strip().lower()
    if wavelet != "haar":
        raise RuntimeError(
            f"ACE-Step dcw_wavelet={wavelet!r} is not supported on MLX; only 'haar' is implemented"
        )
    scaler = (
        _metadata_float(meta["dcw_scaler"], "dcw_scaler")
        if meta.get("dcw_scaler") is not None
        else float(config.dcw_scaler)
    )
    high_scaler = (
        _metadata_float(meta["dcw_high_scaler"], "dcw_high_scaler")
        if meta.get("dcw_high_scaler") is not None
        else float(config.dcw_high_scaler)
    )
    return AceStepDcwSettings(enabled=enabled, mode=mode, scaler=scaler, high_scaler=high_scaler)


def create_ace_step_generator(
    ctx: Any,
    bundle_root: Path,
    *,
    entry: Any | None = None,
    version_key: str | None = None,
) -> _AceStepGeneratorProto:
    require_mlx_ctx(ctx, feature="ACE-Step audio")
    from backend.engine.families.ace_step.generation_mlx import AceStepMlxGenerator

    return AceStepMlxGenerator(ctx, bundle_root, entry=entry, version_key=version_key)


_LM_EXPANSION_REASON_MESSAGES: dict[str, str] = {
    "override_format": (
        "高级覆盖：强制 5Hz LM 格式化扩写（format_sample）。"
    ),
    "override_off": "高级覆盖：已关闭 5Hz LM 扩写。",
    "auto_format": (
        "5Hz LM 将格式化 caption/元数据并生成 audio codes（llm_dit，对齐官方 Custom 路径）。"
    ),
    "lm_disabled": "5Hz LM 已关闭（ACESTEP_USE_LM=0）。",
}


def _normalize_lm_expansion_mode(raw: Any) -> Optional[str]:
    """Return format | off, or None for auto."""
    if raw is None:
        return None
    mode = str(raw).strip().lower()
    if mode in ("", "auto"):
        return None
    if mode in ("inspiration", "create", "create_sample", "simple", "simple_mode"):
        raise RuntimeError(
            "ACE-Step inspiration mode (create_sample) was removed. "
            "Use AI assistant for lyrics, then generate with lm_expansion auto or format."
        )
    if mode in ("format", "format_sample", "understand"):
        return "format"
    if mode in ("off", "none", "skip"):
        return "off"
    raise RuntimeError(
        f"Unknown lm_expansion mode {raw!r}; use auto, format, or off"
    )


def resolve_lm_expansion_override(request: AudioGenerationRequest) -> Optional[str]:
    """Explicit LM path override, or None for auto."""
    meta = request.metadata or {}
    raw = meta.get("lm_expansion")
    if raw is None:
        raw = getattr(request, "lm_expansion", None)
    return _normalize_lm_expansion_mode(raw)


def resolve_lm_expansion_state(
    request: AudioGenerationRequest,
    *,
    lm_enabled: bool,
) -> tuple[bool, str]:
    """Return (run_5hz_lm, reason_key). LM always uses format_sample when enabled."""
    if not lm_enabled:
        return False, "lm_disabled"
    override = resolve_lm_expansion_override(request)
    if override == "off":
        return False, "override_off"
    if override == "format":
        return True, "override_format"
    return True, "auto_format"


def vocal_lyrics_required_message() -> str:
    return vocal_lyrics_required_error()


def resolve_bundle_is_turbo(bundle_root: Path, *, dit_subdir: str | None = None) -> bool:
    """Read ``is_turbo`` from installed DiT ``config.json`` (authoritative over static defaults)."""
    dit = resolve_dit_bundle(bundle_root, dit_subdir=dit_subdir)
    cfg_path = dit / "config.json"
    if not cfg_path.is_file():
        return False
    with open(cfg_path, encoding="utf-8") as f:
        raw = json.load(f)
    return bool(raw.get("is_turbo", False))


def prepare_music_request(
    request: AudioGenerationRequest,
    config: AceStepConfig,
    bundle_root: Path,
    *,
    backend: str = "mlx",
) -> AceStepPreparedRequest:
    """Resolve lyrics/language/steps/shift from contract + registry config (no family branches in Pipeline)."""
    from backend.engine.families.ace_step.quality.resource_policy import (
        clamp_duration,
        resolve_resource_policy,
    )

    _ = backend
    events: List[Tuple[str, str]] = []
    policy = resolve_resource_policy()
    lm_enabled = os.environ.get("ACESTEP_USE_LM", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if policy.tier in ("minimal", "low") and lm_enabled:
        events.append(
            (
                "info",
                f"内存档位 {policy.tier}（约 {policy.memory_gb:.0f}GB）："
                f"建议 LM ≤ {policy.available_lm_models[-1] if policy.available_lm_models else '无'}，"
                f"最长 {policy.max_duration_with_lm}s（含 LM）。",
            )
        )

    registry_max = 600
    raw_duration = float(request.duration if request.duration is not None else 30)
    duration, dur_warn = clamp_duration(
        raw_duration,
        lm_enabled=lm_enabled,
        policy=policy,
        registry_max=600,
    )
    if dur_warn:
        events.append(("warning", dur_warn))

    raw_lyrics = (request.lyrics or "").strip()
    lm_use, lm_reason = resolve_lm_expansion_state(
        request,
        lm_enabled=lm_enabled,
    )
    lm_reason_msg = _LM_EXPANSION_REASON_MESSAGES.get(lm_reason)
    if lm_reason_msg and lm_reason not in ("lm_disabled", "override_off"):
        events.append(("info", lm_reason_msg))
    elif lm_reason == "override_off":
        events.append(("info", lm_reason_msg or ""))

    if request.instrumental:
        lyrics = "[Instrumental]"
    elif raw_lyrics and not is_instrumental_lyrics(raw_lyrics):
        lyrics = raw_lyrics
    elif not raw_lyrics:
        raise RuntimeError(vocal_lyrics_required_message())
    else:
        lyrics = raw_lyrics

    vocal_lang = resolve_vocal_language(
        lyrics or raw_lyrics,
        request.vocal_language or "",
    )

    if raw_lyrics and not is_instrumental_lyrics(raw_lyrics) and not request.instrumental:
        preview = lyrics.replace("\n", " ")[:80]
        events.append(
            ("info", f"人声歌词已启用（约 {len(lyrics)} 字）: {preview!r}…"),
        )
        weak = warn_weak_vocal_lyrics(lyrics)
        if weak:
            events.append(("warning", weak))
        long_hint = warn_lyrics_char_heuristic(lyrics)
        if long_hint:
            events.append(("warning", long_hint))
        mismatch = vocal_language_mismatch_warning(lyrics, vocal_lang)
        if mismatch:
            events.append(("warning", mismatch))
        if not (request.vocal_language or "").strip():
            events.append(
                ("info", f"人声语言未指定，已根据歌词自动设为 {vocal_lang!r}"),
            )

    effective_prompt, vocal_tpl_log = apply_vocal_type_to_prompt(
        request.prompt or "",
        request.vocal_type or "",
        language=vocal_lang,
        instrumental=request.instrumental or is_instrumental_lyrics(lyrics),
    )
    if vocal_tpl_log:
        events.append(("info", vocal_tpl_log))

    from backend.engine.families.ace_step.weights import ace_step_dit_subdir_for_model

    dit_subdir = ace_step_dit_subdir_for_model(request.model)
    is_turbo = resolve_bundle_is_turbo(bundle_root, dit_subdir=dit_subdir)
    steps = request.steps or config.default_infer_steps
    shift = float(config.default_shift)
    guidance = float(
        request.guidance if request.guidance is not None else config.default_guidance
    )
    if is_turbo:
        steps = min(int(steps), int(config.turbo_infer_steps))
        shift = float(config.turbo_shift)
        guidance = 1.0
        events.append(
            (
                "info",
                f"ACE-Step turbo DiT: {steps} diffusion steps, shift={shift} "
                f"({getattr(bundle_root, 'name', bundle_root)})",
            )
        )
    dcw = resolve_dcw_settings(request.metadata, config, is_turbo=is_turbo)
    events.append(("info", f"ACE-Step sampler: {dcw.describe()}"))

    return AceStepPreparedRequest(
        lyrics=lyrics,
        vocal_language=vocal_lang,
        effective_prompt=effective_prompt,
        steps=steps,
        shift=shift,
        is_turbo=is_turbo,
        duration=duration,
        guidance=guidance,
        lm_enabled=lm_use,
        resource_tier=policy.tier,
        lm_quantize_bits=policy.lm_quantize_bits,
        log_events=events,
        sampler_kwargs=dcw.sampler_kwargs(),
    )


@dataclass
class CoverEditBatch:
    """ACE-Step cover edit outputs before asset persistence."""

    waveforms: list[np.ndarray]
    seed: int
    shift: float
    lyrics: str
    vocal_lang: str
    quality: Any | None = None


def run_cover_edit(
    generator: Any,
    request: AudioEditRequest,
    *,
    config: AceStepConfig,
    bundle_root: Path,
    source_path: Path,
    raise_if_cancelled: Callable[[], None],
    on_progress: Callable[[float, int, int, str], None] | None = None,
    on_log: Callable[[str, str], None] | None = None,
) -> CoverEditBatch:
    """ACE-Step cover: reference waveform + prompt/lyrics → new waveform(s)."""
    import random

    from backend.engine.families.ace_step.weights import ace_step_dit_subdir_for_model

    dit_subdir = ace_step_dit_subdir_for_model(request.model)
    is_turbo = resolve_bundle_is_turbo(bundle_root, dit_subdir=dit_subdir)
    shift = float(config.turbo_shift if is_turbo else config.default_shift)
    dcw = resolve_dcw_settings(request.metadata, config, is_turbo=is_turbo)
    raw_timbre = (request.metadata or {}).get("cover_timbre_from_source")
    timbre_from_source = (
        _metadata_bool(raw_timbre, "cover_timbre_from_source") if raw_timbre is not None else False
    )
    ref_wf = load_reference_waveform(source_path)
    seed = (
        request.seed
        if request.seed is not None and request.seed >= 0
        else random.randint(0, 2**31 - 1)
    )
    n = max(request.n, 1)

    raw_lyrics = (request.lyrics or "").strip()
    use_instrumental = bool(request.instrumental) or not raw_lyrics
    lyrics = (
        "[Instrumental]"
        if use_instrumental
        else finalize_lyrics_for_inference(raw_lyrics, instrumental=False, lm_expanded=False)
    )
    vocal_lang = resolve_vocal_language(
        lyrics or raw_lyrics,
        request.vocal_language or "",
    )

    effective_prompt, vocal_tpl_log = apply_vocal_type_to_prompt(
        request.prompt or "",
        request.vocal_type or "",
    )
    if vocal_tpl_log and on_log:
        on_log("info", vocal_tpl_log)

    cover_duration: Optional[float] = None
    if request.duration is not None:
        cover_duration = float(max(10, min(600, int(request.duration))))

    log_parts = [
        f"ACE-Step cover: source={source_path.name}",
        f"strength={request.source_fidelity}",
        f"seed={seed}",
        f"n={n}",
        dcw.describe(),
        "timbre=source" if timbre_from_source else "timbre=silence (upstream default)",
    ]
    if lyrics != "[Instrumental]":
        log_parts.append(f"lyrics={len(lyrics)}chars lang={vocal_lang}")
    if cover_duration is not None:
        log_parts.append(f"duration={cover_duration}s")
    if request.bpm is not None:
        log_parts.append(f"bpm={request.bpm}")
    if request.key_scale:
        log_parts.append(f"key={request.key_scale}")
    if on_log:
        on_log("info", ", ".join(log_parts))

    waveforms: list[np.ndarray] = []
    quality: Any | None = None
    for i in range(n):
        raise_if_cancelled()
        batch_seed = seed + i
        if on_progress:
            on_progress((i + 0.1) / n, i + 1, n, f"Cover generation {i + 1}/{n}")
        cover_steps = (
            int(config.turbo_infer_steps)
            if is_turbo
            else int(request.steps or config.default_infer_steps)
        )
        cover_guidance = (
            1.0
            if is_turbo
            else float(
                request.guidance if request.guidance is not None else config.default_guidance
            )
        )
        waveform = generator.generate_cover_waveform(
            reference_waveform=ref_wf,
            prompt=effective_prompt,
            seed=batch_seed,
            shift=shift,
            audio_cover_strength=float(request.source_fidelity),
            lyrics=lyrics,
            vocal_language=vocal_lang,
            bpm=request.bpm,
            key_scale=request.key_scale or "",
            time_signature=request.time_signature or "",
            duration=cover_duration,
            steps=cover_steps,
            guidance=cover_guidance,
            timbre_from_source=timbre_from_source,
            **dcw.sampler_kwargs(),
        )
        quality = getattr(generator, "last_quality", None)
        waveforms.append(waveform)

    return CoverEditBatch(
        waveforms=waveforms,
        seed=seed,
        shift=shift,
        lyrics=lyrics,
        vocal_lang=vocal_lang,
        quality=quality,
    )


def post_generation_artifacts(
    out_path: Path,
    generator: Any,
    *,
    duration_sec: float,
    on_log: Callable[[str, str], None] | None = None,
    log_lyrics_preview: bool = True,
) -> dict[str, Any]:
    """ACE-Step: lyrics sidecars + metadata after WAV write (MusicPipeline post-hook)."""
    cap = getattr(generator, "last_lyrics_capture", None)
    if cap is None:
        return {}
    if log_lyrics_preview and on_log:
        log_msg = lyrics_capture_log_message(cap)
        if log_msg:
            on_log("info", log_msg)
    sidecar = write_lyrics_sidecar(out_path, cap.lyrics_effective)
    if sidecar is not None and on_log:
        on_log("info", f"歌词已写入: {sidecar.name}")
    lrc_path = write_lrc_sidecar(
        out_path,
        cap.lyrics_effective,
        duration_sec=duration_sec,
        alignment=cap.timestamps,
    )
    if lrc_path is not None and on_log:
        aligned = cap.timestamps is not None and bool(cap.timestamps.segments)
        on_log(
            "info",
            f"歌词时间轴已写入: {lrc_path.name}"
            + ("（DiT 交叉注意力对齐）" if aligned else "（按段落估算）"),
        )
    return lyrics_capture_metadata(cap, duration_sec=duration_sec)