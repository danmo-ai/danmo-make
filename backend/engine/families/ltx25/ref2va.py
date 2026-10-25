"""LTX-2.5 MSR reference conditioning (NumPy).

Community contract (ComfyUI-LTX2.5-MSR / Licon MSR V2):

* Up to four subject stills (``pic1``..``pic4``) plus one optional background.
* Each still is repeated to 25 or 33 frames, VAE-encoded, and appended after the
  target latent. RoPE time is a negative pixel-frame offset (pic1 is the
  furthest before the clip).
* A learned Fourier-MLP slot embedding from the MSR LoRA is added to every
  reference latent channel.
* Optional audio references pair with pic1 and pic2. They are prepended to the
  audio token stream inside absolute negative time windows. Audio cannot be
  the only reference, and a video clip is not an MSR slot.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

MAX_SUBJECTS = 4
MAX_IMAGES = 5
MAX_AUDIOS = 2
REFERENCE_FRAMES = (25, 33)
DEFAULT_REFERENCE_FRAMES = 33
SLOT_EMBED_SCALE = 16.0
VIDEO_TEMPORAL_SCALE = 8
AUDIO_DOWNSAMPLE = 4
AUDIO_HOP = 160
AUDIO_SAMPLE_RATE = 16000
DEFAULT_AUDIO_SLOT_SECONDS = 5.0
DEFAULT_AUDIO_END_MARGIN = 0.04

_IMAGE_SLOT_PREFIXES = (
    "model.diffusion_model.reference_slot_embedding.",
    "diffusion_model.reference_slot_embedding.",
    "reference_slot_embedding.",
)
_AUDIO_SLOT_PREFIXES = (
    "model.diffusion_model.reference_audio_slot_embedding.",
    "diffusion_model.reference_audio_slot_embedding.",
    "reference_audio_slot_embedding.",
)
_LORA_SUFFIXES = (
    ".lora_A.weight",
    ".lora_B.weight",
    ".lora_down.weight",
    ".lora_up.weight",
    ".alpha",
)
_REQUIRED_SLOT_KEYS = (
    "frequencies",
    "net.0.weight",
    "net.0.bias",
    "net.2.weight",
    "net.2.bias",
)


def validate_reference_kinds(kinds: Sequence[str]) -> None:
    """Fail loud on an empty, video, or audio-only MSR reference list."""
    if not kinds:
        raise RuntimeError(
            "LTX 2.5 Ref2VA needs at least one reference image. "
            "Use text-to-video for a prompt-only request."
        )
    allowed = {"image", "audio"}
    for index, kind in enumerate(kinds):
        if kind == "video":
            raise RuntimeError(
                "LTX 2.5 Ref2VA does not take a video reference. "
                "Add still images (and optional audio paired with the first two images)."
            )
        if kind not in allowed:
            raise RuntimeError(
                f"LTX 2.5 reference {index} must be image or audio, got {kind!r}."
            )
    images = kinds.count("image")
    audios = kinds.count("audio")
    if images > MAX_IMAGES:
        raise RuntimeError(
            f"LTX 2.5 accepts at most {MAX_IMAGES} reference images "
            f"(4 subjects + 1 background), got {images}."
        )
    if images < 1:
        raise RuntimeError(
            "LTX 2.5 Ref2VA needs at least one reference image. "
            "Audio references have to be paired with pic1 / pic2."
        )
    if audios > MAX_AUDIOS:
        raise RuntimeError(f"LTX 2.5 accepts at most {MAX_AUDIOS} audio references, got {audios}.")
    subjects = images - 1 if images == MAX_IMAGES else images
    if audios > subjects:
        raise RuntimeError(
            f"Each audio reference needs a matching subject image (pic1, then pic2). "
            f"Got {audios} audio reference(s) and {subjects} subject image(s)."
        )


def reference_frame_offsets(num_slots: int) -> list[int]:
    """Negative pixel-frame offsets, pic1 furthest from the clip (MSR order)."""
    if not 1 <= num_slots <= MAX_IMAGES:
        raise RuntimeError(f"LTX 2.5 MSR needs 1–{MAX_IMAGES} image slots, got {num_slots}.")
    return [-(num_slots - index) for index in range(num_slots)]


def split_subject_and_background(image_count: int) -> tuple[int, bool]:
    """Return ``(subject_count, last_image_is_background)``."""
    if image_count == MAX_IMAGES:
        return MAX_SUBJECTS, True
    return image_count, False


def aspect_family(width: int, height: int) -> str:
    ratio = width / max(height, 1)
    if ratio >= 1.25:
        return "landscape"
    if ratio <= 0.8:
        return "portrait"
    return "square"


def guide_time_midpoints(
    num_latent_frames: int,
    *,
    frame_offset: int,
    fps: float,
    temporal_scale: int = VIDEO_TEMPORAL_SCALE,
) -> np.ndarray:
    """Causal latent-frame midpoints in seconds, shifted by a pixel-frame offset.

    Matches ComfyUI ``latent_to_pixel_coords(causal_fix=True)`` plus the MSR
    ``append_keyframe`` time shift, then the midpoint used by
    ``use_middle_indices_grid``.
    """
    if num_latent_frames < 1:
        raise RuntimeError("A reference guide needs at least one latent frame.")
    if fps <= 0:
        raise RuntimeError(f"fps must be positive, got {fps}.")
    idx = np.arange(num_latent_frames, dtype=np.float32)
    starts = np.maximum(idx * temporal_scale + 1 - temporal_scale, 0.0) + float(frame_offset)
    ends = np.maximum((idx + 1) * temporal_scale + 1 - temporal_scale, 0.0) + float(frame_offset)
    return ((starts + ends) / 2.0 / float(fps)).astype(np.float32)


def audio_token_bounds(num_tokens: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal audio token ``(start, end, midpoint)`` times in seconds."""
    if num_tokens < 1:
        raise RuntimeError("Audio reference token count must be positive.")
    idx = np.arange(num_tokens, dtype=np.float64)
    starts = np.maximum(idx * AUDIO_DOWNSAMPLE + 1 - AUDIO_DOWNSAMPLE, 0.0) * AUDIO_HOP / AUDIO_SAMPLE_RATE
    ends = (
        np.maximum((idx + 1) * AUDIO_DOWNSAMPLE + 1 - AUDIO_DOWNSAMPLE, 0.0)
        * AUDIO_HOP
        / AUDIO_SAMPLE_RATE
    )
    mids = (starts + ends) / 2.0
    return starts.astype(np.float32), ends.astype(np.float32), mids.astype(np.float32)


def max_audio_tokens(slot_duration: float) -> int:
    """How many causal audio tokens fit in one MSR slot window."""
    if slot_duration <= 0:
        raise RuntimeError(f"Audio slot duration must be positive, got {slot_duration}.")
    count = 0
    while count < 100_000:
        _, ends, _ = audio_token_bounds(count + 1)
        if float(ends[-1]) <= slot_duration + 1e-6:
            count += 1
            continue
        break
    return max(1, count)


def shift_audio_midpoints(
    mids: np.ndarray,
    *,
    slot_id: int,
    image_slot_count: int,
    slot_duration: float,
    end_margin: float,
) -> np.ndarray:
    """Place a reference clip so its last token ends on the MSR absolute window."""
    if mids.ndim != 1 or mids.shape[0] < 1:
        raise RuntimeError("Audio midpoint vector must be 1-D and non-empty.")
    if slot_id < 1 or slot_id > image_slot_count:
        raise RuntimeError(
            f"Audio slot {slot_id} has no matching image slot (image slots: {image_slot_count})."
        )
    _, ends, _ = audio_token_bounds(int(mids.shape[0]))
    local_end = float(ends[-1])
    slot_end = -((image_slot_count - slot_id) * float(slot_duration)) - float(end_margin)
    return (mids.astype(np.float32) + np.float32(slot_end - local_end)).astype(np.float32)


def slot_embedding(slot_id: int, state: Mapping[str, np.ndarray]) -> np.ndarray:
    """Fourier-MLP slot identity. ``slot_id`` is 1-based, matching the MSR loader."""
    _validate_slot_arrays(state, label="slot")
    frequencies = np.asarray(state["frequencies"], dtype=np.float32).reshape(-1)
    scaled = np.float32(float(slot_id) / SLOT_EMBED_SCALE)
    phases = scaled * frequencies
    features = np.concatenate(
        [np.asarray([scaled], dtype=np.float32), np.sin(phases), np.cos(phases)]
    )
    weight0 = np.asarray(state["net.0.weight"], dtype=np.float32)
    bias0 = np.asarray(state["net.0.bias"], dtype=np.float32)
    hidden = _silu(features @ weight0.T + bias0)
    weight2 = np.asarray(state["net.2.weight"], dtype=np.float32)
    bias2 = np.asarray(state["net.2.bias"], dtype=np.float32)
    return (hidden @ weight2.T + bias2).astype(np.float32)


def _silu(values: np.ndarray) -> np.ndarray:
    return values / (1.0 + np.exp(-values))


def _validate_slot_arrays(state: Mapping[str, np.ndarray], *, label: str) -> None:
    missing = [key for key in _REQUIRED_SLOT_KEYS if key not in state]
    if missing:
        raise RuntimeError(
            f"MSR {label} embedding is missing {', '.join(missing)}."
        )
    frequencies = np.asarray(state["frequencies"])
    weight0 = np.asarray(state["net.0.weight"])
    bias0 = np.asarray(state["net.0.bias"])
    weight2 = np.asarray(state["net.2.weight"])
    bias2 = np.asarray(state["net.2.bias"])
    expected = 1 + 2 * int(frequencies.size)
    if frequencies.ndim != 1 or weight0.ndim != 2 or weight0.shape[1] != expected:
        raise RuntimeError(f"Invalid MSR {label} Fourier-MLP input shape.")
    if bias0.shape != weight0.shape[:1] or weight2.ndim != 2 or weight2.shape[1] != weight0.shape[0]:
        raise RuntimeError(f"Invalid MSR {label} Fourier-MLP hidden shape.")
    if bias2.shape != weight2.shape[:1]:
        raise RuntimeError(f"Invalid MSR {label} Fourier-MLP output shape.")


def split_msr_lora_keys(
    keys: Sequence[str],
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Split checkpoint keys into LoRA pairs, image-slot keys, and audio-slot keys.

    Slot dict values are the suffix after the embedding prefix (``net.0.weight``).
    """
    image: dict[str, str] = {}
    audio: dict[str, str] = {}
    lora_keys: list[str] = []
    for key in keys:
        image_suffix = _slot_suffix(key, _IMAGE_SLOT_PREFIXES)
        if image_suffix is not None:
            image[image_suffix] = key
            continue
        audio_suffix = _slot_suffix(key, _AUDIO_SLOT_PREFIXES)
        if audio_suffix is not None:
            audio[audio_suffix] = key
            continue
        if any(key.endswith(suffix) for suffix in _LORA_SUFFIXES):
            lora_keys.append(key)
    return lora_keys, image, audio


def _slot_suffix(key: str, prefixes: Sequence[str]) -> str | None:
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix):]
    return None


def lora_module_and_role(raw_key: str) -> tuple[str, str] | None:
    """Map a Comfy/LTX LoRA key to ``(module, role)`` with ``role`` in A/B/alpha."""
    key = raw_key
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    role = ""
    module = ""
    for suffix, name in (
        (".lora_A.weight", "A"),
        (".lora_down.weight", "A"),
        (".lora_B.weight", "B"),
        (".lora_up.weight", "B"),
        (".alpha", "alpha"),
    ):
        if key.endswith(suffix):
            module = key[: -len(suffix)]
            role = name
            break
    if not module or not role:
        return None
    module = module.replace(".emb.timestep_embedder.", ".emb.")
    module = module.replace(".ff.net.0.proj", ".ff.proj_in")
    module = module.replace(".ff.net.2", ".ff.proj_out")
    module = module.replace(".audio_ff.net.0.proj", ".audio_ff.proj_in")
    module = module.replace(".audio_ff.net.2", ".audio_ff.proj_out")
    for attn in (
        "attn1",
        "attn2",
        "audio_attn1",
        "audio_attn2",
        "audio_to_video_attn",
        "video_to_audio_attn",
    ):
        module = module.replace(f".{attn}.to_out.0", f".{attn}.to_out")
    return module, role


def read_downscale_factor(metadata: Mapping[str, Any] | None) -> int:
    raw = (metadata or {}).get("reference_downscale_factor", 1)
    try:
        factor = max(1, int(round(float(raw))))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"MSR LoRA metadata reference_downscale_factor={raw!r} is not a number."
        ) from exc
    if factor != 1:
        raise RuntimeError(
            "LTX 2.5 Ref2VA currently supports MSR checkpoints with "
            f"reference_downscale_factor=1 (this LoRA declares {factor})."
        )
    return factor


def require_prepend_layout(metadata: Mapping[str, Any] | None) -> None:
    meta = metadata or {}
    order = str(meta.get("reference_token_order", "prepend"))
    offsets = str(meta.get("reference_slot_time_offsets", "pic1_based_negative_time"))
    if order != "prepend":
        raise RuntimeError(f"Unsupported MSR reference_token_order={order!r}; expected 'prepend'.")
    if offsets != "pic1_based_negative_time":
        raise RuntimeError(
            "Unsupported MSR reference_slot_time_offsets="
            f"{offsets!r}; expected 'pic1_based_negative_time'."
        )


def avref_timing(metadata: Mapping[str, Any] | None) -> tuple[float, float]:
    """Return ``(slot_duration_seconds, end_margin_seconds)`` or fail loud."""
    meta = dict(metadata or {})
    expected = {
        "reference_token_order": "prepend",
        "reference_slot_time_offsets": "pic1_based_negative_time",
        "reference_audio_conditioning": "id_lora_clean_negative_rope",
        "reference_audio_token_order": "pic1_to_picN_then_target",
        "reference_audio_rope_layout": "absolute_image_slot_windows",
        "reference_audio_overflow_mode": "truncate",
    }
    for key, expected_value in expected.items():
        actual = meta.get(key)
        if actual is None:
            raise RuntimeError(f"The MSR LoRA is missing AVref metadata {key}.")
        if str(actual).strip() != expected_value:
            raise RuntimeError(f"Unsupported {key}={actual!r}; expected {expected_value!r}.")
    if not _metadata_bool(meta, "reference_audio_sparse_slots", False):
        raise RuntimeError("The MSR LoRA does not declare sparse absolute audio slots.")
    if not _metadata_bool(meta, "reference_slot_embedding_enabled", False):
        raise RuntimeError("The MSR LoRA does not enable image reference slot embeddings.")
    if not _metadata_bool(meta, "reference_audio_slot_embedding_enabled", False):
        raise RuntimeError("The MSR LoRA does not enable audio reference slot embeddings.")
    duration = _metadata_float(meta, "reference_audio_slot_duration_seconds", DEFAULT_AUDIO_SLOT_SECONDS)
    margin = _metadata_float(meta, "reference_audio_end_margin_seconds", DEFAULT_AUDIO_END_MARGIN)
    if duration <= 0 or margin <= 0:
        raise RuntimeError("Audio slot duration and end margin must both be positive.")
    return duration, margin


def _metadata_bool(metadata: Mapping[str, Any], key: str, default: bool) -> bool:
    value = metadata.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _metadata_float(metadata: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(metadata.get(key, default))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid numeric LoRA metadata {key}={metadata.get(key)!r}.") from exc
