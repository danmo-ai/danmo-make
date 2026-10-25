"""LTX-2.5 MSR Ref2VA runtime: LoRA slot embeddings, guide latents, audio windows."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from backend.engine.families.ltx.pipeline_math_mlx import (
    VIDEO_SPATIAL_SCALE,
    LatentState,
    VideoLatentPatchifier,
)
from backend.engine.families.ltx25.ref2va import (
    DEFAULT_REFERENCE_FRAMES,
    avref_timing,
    guide_time_midpoints,
    lora_module_and_role,
    max_audio_tokens,
    read_downscale_factor,
    require_prepend_layout,
    shift_audio_midpoints,
    slot_embedding,
    split_msr_lora_keys,
    split_subject_and_background,
    validate_reference_kinds,
    audio_token_bounds,
)
from backend.engine.runtime.mlx_runtime import load_weights_dict, run_eval

MSR_LORA_ID = "ltx25-msr-lora"
MSR_LORA_VERSION = "v2"
MSR_LORA_FILENAME = "LTX-2.5-Licon-MSR-V2.safetensors"


@dataclass
class MsrConditioning:
    image_slots: dict[str, np.ndarray]
    audio_slots: dict[str, np.ndarray]
    slot_duration: float
    end_margin: float
    has_audio_slots: bool


@dataclass
class PreparedReferences:
    image_paths: list[str]
    audio_paths: list[str]
    background_index: int | None


def prepare_reference_list(media_references: Sequence[tuple[str, str]]) -> PreparedReferences:
    kinds = [str(kind).strip().lower() for kind, _ in media_references]
    validate_reference_kinds(kinds)
    images = [path for kind, path in zip(kinds, (item[1] for item in media_references)) if kind == "image"]
    audios = [path for kind, path in zip(kinds, (item[1] for item in media_references)) if kind == "audio"]
    subjects, has_background = split_subject_and_background(len(images))
    background_index = subjects if has_background else None
    return PreparedReferences(image_paths=images, audio_paths=audios, background_index=background_index)


def resolve_msr_lora_path(*, project_root: Path, registry: Any) -> Path:
    if registry is None or project_root is None:
        raise RuntimeError(
            "LTX 2.5 Ref2VA needs the model registry to locate the MSR LoRA. "
            f"Install '{MSR_LORA_ID}:{MSR_LORA_VERSION}' from Models."
        )
    try:
        entry = registry.require(MSR_LORA_ID)
    except KeyError as exc:
        raise RuntimeError(
            f"LTX 2.5 Ref2VA requires registry model {MSR_LORA_ID!r}. "
            "Sync the model registry and install the MSR LoRA."
        ) from exc
    from backend.engine.contracts.pipeline_registry import local_bundle_root

    bundle = local_bundle_root(project_root, entry, MSR_LORA_VERSION)
    if bundle is None or not Path(bundle).exists():
        raise RuntimeError(
            "LTX 2.5 MSR LoRA is not installed. Install "
            f"'{MSR_LORA_ID}:{MSR_LORA_VERSION}' from Models "
            "(models/Lora/LTX-2.5/ltx25-msr-v2)."
        )
    root = Path(bundle)
    named = root / MSR_LORA_FILENAME
    if named.is_file():
        return named
    if root.is_file() and root.suffix == ".safetensors":
        return root
    files = sorted(root.rglob("*.safetensors"))
    if len(files) == 1:
        return files[0]
    preferred = [path for path in files if "msr" in path.name.lower() and "v2" in path.name.lower()]
    if len(preferred) == 1:
        return preferred[0]
    if files:
        return files[0]
    raise RuntimeError(f"No MSR LoRA .safetensors under {root}")


def read_safetensors_metadata(path: Path) -> dict[str, str]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required to read MSR LoRA metadata.") from exc
    with safe_open(str(path), framework="numpy") as handle:
        meta = handle.metadata() or {}
    return {str(key): str(value) for key, value in meta.items()}


def merge_msr_lora(
    stem: Any,
    *,
    weight_path: Path,
    ctx: Any,
    bits: int | None,
    group_size: int,
    on_log: Callable[[str, str], None] | None = None,
) -> MsrConditioning:
    """Merge MSR LoRA deltas into the live DiT and return slot-embedding weights."""
    import mlx.core as mx

    from backend.engine.common.bundle.lora_mlx import orient_lora_pair
    from backend.engine.common.model.quantized_lora_mlx import apply_lora_delta_to_weight

    weights = load_weights_dict(getattr(ctx, "load_weights", None), str(weight_path))
    if not weights:
        raise RuntimeError(f"MSR LoRA is empty: {weight_path}")
    metadata = read_safetensors_metadata(weight_path)
    require_prepend_layout(metadata)
    read_downscale_factor(metadata)

    lora_keys, image_index, audio_index = split_msr_lora_keys(list(weights))
    if not image_index:
        raise RuntimeError(
            f"{weight_path.name} has no reference_slot_embedding tensors and is not an MSR LoRA."
        )
    image_slots = {suffix: np.array(weights[key].astype(mx.float32)) for suffix, key in image_index.items()}
    audio_slots = {suffix: np.array(weights[key].astype(mx.float32)) for suffix, key in audio_index.items()}
    slot_embedding(1, image_slots)
    has_audio = bool(audio_slots)
    duration, margin = (0.0, 0.0)
    if has_audio:
        slot_embedding(1, audio_slots)
        duration, margin = avref_timing(metadata)

    model = stem.model
    if not getattr(stem, "_param_map", None):
        stem._build_param_map()
    model._param_map = stem._param_map

    groups: dict[str, dict[str, Any]] = {}
    for key in lora_keys:
        parsed = lora_module_and_role(key)
        if parsed is None:
            continue
        module, role = parsed
        groups.setdefault(module, {})[role] = weights[key]

    def _rebuild() -> None:
        stem._build_param_map()
        model._param_map = stem._param_map

    class _MergeRoot:
        def __getattr__(self, name: str) -> Any:
            return getattr(model, name)

        def _build_param_map(self) -> None:
            _rebuild()
            self._param_map = model._param_map

    root = _MergeRoot()
    root._param_map = model._param_map  # type: ignore[attr-defined]

    applied = 0
    unmatched: list[str] = []
    for module, parts in groups.items():
        if "A" not in parts or "B" not in parts:
            unmatched.append(module)
            continue
        wkey = f"{module}.weight"
        if wkey not in model._param_map:
            unmatched.append(module)
            continue
        param = model._param_map[wkey]
        dtype_label = str(getattr(getattr(param, "dtype", None), "name", None) or getattr(param, "dtype", "")).lower()
        quantized = "uint32" in dtype_label
        if quantized:
            scales = model._param_map.get(f"{module}.scales")
            if scales is None:
                raise RuntimeError(f"MSR LoRA merge needs {module}.scales for the quantized layer.")
            out_d = int(scales.shape[0])
            in_d = int(scales.shape[1]) * int(group_size)
        else:
            out_d, in_d = int(param.shape[0]), int(param.shape[1])
        down, up, rank = orient_lora_pair(
            parts["A"],
            parts["B"],
            out_d=out_d,
            in_d=in_d,
            lora_id=MSR_LORA_ID,
            wkey=wkey,
        )
        alpha = parts.get("alpha")
        if alpha is None:
            alpha_value = float(rank)
        elif hasattr(alpha, "item"):
            alpha_value = float(alpha.item())
        else:
            alpha_value = float(np.array(alpha).reshape(-1)[0])
        delta = (alpha_value / float(rank)) * mx.matmul(up.astype(mx.float32), down.astype(mx.float32))
        if quantized:
            if bits not in (4, 8):
                raise RuntimeError(
                    f"MSR LoRA hit a quantized layer {wkey} but the bundle bits are {bits!r}."
                )
            root._param_map = model._param_map
            apply_lora_delta_to_weight(
                model=root,
                wkey=wkey,
                delta=delta,
                ctx=ctx,
                bits=int(bits),
                group_size=int(group_size),
            )
            _rebuild()
        else:
            param[:] = (param.astype(mx.float32) + delta).astype(param.dtype)
        applied += 1

    if applied == 0:
        preview = ", ".join(unmatched[:8])
        raise RuntimeError(
            "MSR LoRA did not match any LTX 2.5 transformer weights. "
            f"Unmatched modules start with: {preview}"
        )
    if unmatched:
        preview = ", ".join(unmatched[:8])
        raise RuntimeError(
            "MSR LoRA has modules that do not match the LTX 2.5 transformer: "
            f"{preview}. Refusing to run with a partial merge."
        )
    run_eval(getattr(ctx, "eval", None), stem.model.scale_shift_table)
    if on_log is not None:
        on_log(
            "info",
            f"Merged MSR LoRA into {applied} layers "
            f"(image slots={len(image_slots)}, audio slots={len(audio_slots)}).",
        )
    return MsrConditioning(
        image_slots=image_slots,
        audio_slots=audio_slots,
        slot_duration=duration,
        end_margin=margin,
        has_audio_slots=has_audio,
    )


def _resize_reference(image: np.ndarray, target_w: int, target_h: int, *, background: bool) -> np.ndarray:
    from PIL import Image

    from backend.engine.families.ltx25.ref2va import aspect_family

    src = Image.fromarray(image, mode="RGB")
    if background or (
        aspect_family(src.width, src.height) == aspect_family(target_w, target_h)
        and (src.width > target_w or src.height > target_h)
    ):
        scale = max(target_w / src.width, target_h / src.height)
        resized = src.resize((max(1, round(src.width * scale)), max(1, round(src.height * scale))), Image.BILINEAR)
        left = max(0, (resized.width - target_w) // 2)
        top = max(0, (resized.height - target_h) // 2)
        resized = resized.crop((left, top, left + target_w, top + target_h))
        return np.asarray(resized, dtype=np.uint8)

    scale = min(target_w / src.width, target_h / src.height)
    resized_w = max(1, min(target_w, round(src.width * scale)))
    resized_h = max(1, min(target_h, round(src.height * scale)))
    resized = src.resize((resized_w, resized_h), Image.BILINEAR)
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    left = (target_w - resized_w) // 2
    top = (target_h - resized_h) // 2
    canvas[top : top + resized_h, left : left + resized_w] = np.asarray(resized, dtype=np.uint8)
    return canvas


def encode_image_guides(
    ctx: Any,
    *,
    image_paths: Sequence[str],
    background_index: int | None,
    pixel_h: int,
    pixel_w: int,
    reference_frames: int,
    video_encoder: Any,
    image_slots: dict[str, np.ndarray],
    fps: float,
) -> tuple[Any, Any, int]:
    """Return ``(tokens, positions, token_count)`` appended after the target video."""
    import mlx.core as mx

    if reference_frames not in (25, 33):
        raise RuntimeError(f"reference_frames must be 25 or 33, got {reference_frames}.")
    patchifier = VideoLatentPatchifier()
    token_blocks = []
    position_blocks = []
    num_slots = len(image_paths)
    from backend.engine.families.ltx25.ref2va import reference_frame_offsets

    offsets = reference_frame_offsets(num_slots)
    for index, path in enumerate(image_paths):
        from PIL import Image

        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        canvas = _resize_reference(image, pixel_w, pixel_h, background=index == background_index)
        pixels = canvas.astype(np.float32) / 127.5 - 1.0
        clip = np.repeat(pixels[None, None], reference_frames, axis=1)  # (1, F, H, W, 3)
        clip = np.transpose(clip, (0, 4, 1, 2, 3))
        latent = video_encoder.encode(ctx.array(clip.astype(np.float32)))
        run_eval(getattr(ctx, "eval", None), latent)
        embedding = slot_embedding(index + 1, image_slots)
        if int(embedding.shape[0]) != int(latent.shape[1]):
            raise RuntimeError(
                f"MSR slot embedding dim {int(embedding.shape[0])} does not match "
                f"video latent channels {int(latent.shape[1])}."
            )
        latent = latent + ctx.array(embedding.astype(np.float32)).reshape(1, -1, 1, 1, 1)
        tokens, _ = patchifier.patchify(latent, ctx)
        frames = int(latent.shape[2])
        height = int(latent.shape[3])
        width = int(latent.shape[4])
        mids = guide_time_midpoints(frames, frame_offset=offsets[index], fps=fps)
        h_mids = (np.arange(height, dtype=np.float32) * VIDEO_SPATIAL_SCALE) + VIDEO_SPATIAL_SCALE / 2.0
        w_mids = (np.arange(width, dtype=np.float32) * VIDEO_SPATIAL_SCALE) + VIDEO_SPATIAL_SCALE / 2.0
        positions = np.stack(
            [
                np.repeat(np.repeat(mids[:, None, None], height, axis=1), width, axis=2),
                np.repeat(np.repeat(h_mids[None, :, None], frames, axis=0), width, axis=2),
                np.repeat(np.repeat(w_mids[None, None, :], frames, axis=0), height, axis=1),
            ],
            axis=-1,
        ).reshape(1, -1, 3)
        token_blocks.append(tokens)
        position_blocks.append(ctx.array(positions.astype(np.float32)))
    tokens_out = mx.concatenate(token_blocks, axis=1)
    positions_out = mx.concatenate(position_blocks, axis=1)
    return tokens_out, positions_out, int(tokens_out.shape[1])


def append_clean_tokens(
    ctx: Any,
    state: LatentState,
    tokens: Any,
    positions: Any,
    *,
    prepend: bool,
) -> LatentState:
    """Pin clean reference tokens (denoise mask 0) onto a noised state."""
    count = int(tokens.shape[1])
    mask = ctx.zeros((int(tokens.shape[0]), count, 1), dtype=state.denoise_mask.dtype)
    tokens = tokens.astype(state.latent.dtype)
    if prepend:
        latent = ctx.concat([tokens, state.latent], axis=1)
        clean = ctx.concat([tokens, state.clean_latent], axis=1)
        denoise = ctx.concat([mask, state.denoise_mask], axis=1)
        pos = ctx.concat([positions, state.positions], axis=1)
    else:
        latent = ctx.concat([state.latent, tokens], axis=1)
        clean = ctx.concat([state.clean_latent, tokens], axis=1)
        denoise = ctx.concat([state.denoise_mask, mask], axis=1)
        pos = ctx.concat([state.positions, positions], axis=1)
    return LatentState(
        latent=latent,
        clean_latent=clean,
        denoise_mask=denoise,
        positions=pos,
        attention_mask=state.attention_mask,
    )


def crop_appended(tokens: Any, count: int) -> Any:
    if count <= 0:
        return tokens
    return tokens[:, :-count]


def crop_prepended(tokens: Any, count: int) -> Any:
    if count <= 0:
        return tokens
    return tokens[:, count:]


def _decode_waveform(path: str, sample_rate: int, seconds: float) -> np.ndarray:
    from backend.engine.families.ltx25.vae_mlx import _find_ffmpeg

    ffmpeg = _find_ffmpeg()
    proc = subprocess.run(
        [
            ffmpeg, "-v", "error", "-i", path,
            "-t", f"{seconds:.6f}",
            "-ac", "2", "-ar", str(sample_rate),
            "-f", "f32le", "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode audio reference {path}: {detail}")
    pcm = np.frombuffer(proc.stdout, dtype=np.float32)
    if pcm.size < 2 or pcm.size % 2:
        raise RuntimeError(f"Audio reference {path} decoded to {pcm.size} samples.")
    return np.array(pcm.reshape(-1, 2).T, dtype=np.float32)


def _hz_to_mel_slaney(freq: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    mels = freq / f_sp
    log_region = freq >= min_log_hz
    mels = np.where(log_region, min_log_mel + np.log(freq / min_log_hz) / logstep, mels)
    return mels


def _mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    freq = mels * f_sp
    log_region = mels >= min_log_mel
    freq = np.where(log_region, min_log_hz * np.exp(logstep * (mels - min_log_mel)), freq)
    return freq


def _slaney_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    mel_min = _hz_to_mel_slaney(np.asarray([0.0]))[0]
    mel_max = _hz_to_mel_slaney(np.asarray([sample_rate / 2.0]))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz_slaney(mel_points)
    weights = np.zeros((n_mels, fft_freqs.shape[0]), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = hz_points[index : index + 3]
        rising = (fft_freqs - left) / max(center - left, 1e-6)
        falling = (right - fft_freqs) / max(right - center, 1e-6)
        weights[index] = np.maximum(0.0, np.minimum(rising, falling))
    enorm = (2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])).astype(np.float32)
    return weights * enorm[:, None]


def waveform_to_log_mel(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int,
    hop: int,
    n_mels: int,
) -> np.ndarray:
    """Stereo waveform ``(2, T)`` → log-mel ``(1, 2, frames, n_mels)``."""
    window = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)
    basis = _slaney_filterbank(sample_rate, n_fft, n_mels)
    channels = []
    for channel in waveform:
        padded = np.pad(channel.astype(np.float32), (n_fft // 2, n_fft // 2), mode="reflect")
        frames = 1 + max(0, (padded.shape[0] - n_fft) // hop)
        spec = np.empty((frames, n_fft // 2 + 1), dtype=np.float32)
        for index in range(frames):
            start = index * hop
            chunk = padded[start : start + n_fft]
            if chunk.shape[0] < n_fft:
                chunk = np.pad(chunk, (0, n_fft - chunk.shape[0]))
            spec[index] = np.abs(np.fft.rfft(chunk * window)).astype(np.float32)
        mel = spec @ basis.T
        channels.append(np.log(np.maximum(mel, 1e-5)))
    stacked = np.stack(channels, axis=0)
    return stacked[None].astype(np.float32)


def encode_audio_references(
    ctx: Any,
    *,
    audio_paths: Sequence[str],
    image_slot_count: int,
    audio_encoder: Any,
    audio_slots: dict[str, np.ndarray],
    slot_duration: float,
    end_margin: float,
) -> tuple[Any, Any, int]:
    """Return prepended ``(tokens, positions, token_count)`` for pic-paired audio."""
    import mlx.core as mx

    limit = max_audio_tokens(slot_duration)
    blocks = []
    position_blocks = []
    for index, path in enumerate(audio_paths):
        slot_id = index + 1
        waveform = _decode_waveform(path, int(audio_encoder.sample_rate), float(slot_duration))
        mel = waveform_to_log_mel(
            waveform,
            sample_rate=int(audio_encoder.sample_rate),
            n_fft=int(audio_encoder.n_fft),
            hop=int(audio_encoder.mel_hop_length),
            n_mels=int(audio_encoder.mel_bins),
        )
        latent = audio_encoder.encode(ctx.array(mel))
        run_eval(getattr(ctx, "eval", None), latent)
        latent_np = np.array(latent.astype(mx.float32))
        tokens = np.transpose(latent_np, (0, 2, 1, 3)).reshape(1, latent_np.shape[2], -1)
        if tokens.shape[1] > limit:
            tokens = tokens[:, :limit]
        embedding = slot_embedding(slot_id, audio_slots)
        if int(embedding.shape[0]) != int(tokens.shape[-1]):
            raise RuntimeError(
                f"Audio slot embedding dim {int(embedding.shape[0])} does not match "
                f"audio tokens {int(tokens.shape[-1])}."
            )
        tokens = tokens + embedding.reshape(1, 1, -1)
        _, _, mids = audio_token_bounds(int(tokens.shape[1]))
        shifted = shift_audio_midpoints(
            mids,
            slot_id=slot_id,
            image_slot_count=image_slot_count,
            slot_duration=slot_duration,
            end_margin=end_margin,
        )
        blocks.append(tokens.astype(np.float32))
        position_blocks.append(shifted.reshape(1, -1, 1).astype(np.float32))
    tokens_np = np.concatenate(blocks, axis=1)
    positions_np = np.concatenate(position_blocks, axis=1)
    return ctx.array(tokens_np), ctx.array(positions_np), int(tokens_np.shape[1])


def default_reference_frames() -> int:
    return DEFAULT_REFERENCE_FRAMES
