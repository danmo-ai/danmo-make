"""LTX 2.3 pipeline math — scheduling, patchify, latent state, conditioning.

Merged from legacy ``conditioning.py`` + ``ltx_scheduling.py`` plus LTX-2.3
orchestration helpers (token BLC layout, distilled sigma tables, positions).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np

# ---------------------------------------------------------------------------
# Distilled sigma schedules (Lightricks LTX-2 / 2.3)
# ---------------------------------------------------------------------------

DISTILLED_SIGMAS: list[float] = [
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
]

STAGE_2_SIGMAS: list[float] = [
    0.909375,
    0.725,
    0.421875,
    0.0,
]

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)

VIDEO_TEMPORAL_SCALE = 8
VIDEO_SPATIAL_SCALE = 32
AUDIO_DOWNSAMPLE_FACTOR = 4
AUDIO_HOP_LENGTH = 160
AUDIO_SAMPLE_RATE = 16000
AUDIO_LATENTS_PER_SECOND = AUDIO_SAMPLE_RATE / AUDIO_HOP_LENGTH / AUDIO_DOWNSAMPLE_FACTOR


# ---------------------------------------------------------------------------
# Scheduling (legacy ltx_scheduling + dynamic LTX-2 schedule)
# ---------------------------------------------------------------------------

def ltx_calculate_shift(
    video_seq_len: int,
    base_image_seq_len: int = 1024,
    max_image_seq_len: int = 4096,
    base_shift: float = 0.95,
    max_shift: float = 2.05,
) -> float:
    """Resolution-dependent μ for LTX FlowMatch schedulers."""
    m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    b = base_shift - m * base_image_seq_len
    return float(video_seq_len * m + b)


def ltx_scheduler_shift_kwargs(bundle_root: Path | None) -> dict[str, float | int]:
    """Read LTX FlowMatch shift params from ``scheduler/scheduler_config.json`` when present."""
    defaults: dict[str, float | int] = {
        "base_image_seq_len": 1024,
        "max_image_seq_len": 4096,
        "base_shift": 0.95,
        "max_shift": 2.05,
    }
    if bundle_root is None:
        return defaults
    cfg_path = bundle_root / "scheduler" / "scheduler_config.json"
    if not cfg_path.is_file():
        return defaults
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"LTX: cannot read scheduler config {cfg_path}: {e}") from e
    for key in defaults:
        if key in data:
            defaults[key] = data[key]
    if "shift_terminal" in data:
        defaults["shift_terminal"] = data["shift_terminal"]
    return defaults


def ltx_video_sequence_length(
    *,
    latent_frames: int,
    pixel_w: int,
    pixel_h: int,
    vae_scale: int,
) -> int:
    latent_h = pixel_h // vae_scale
    latent_w = pixel_w // vae_scale
    return int(latent_frames * latent_h * latent_w)


def ltx_stretch_shift_to_terminal(sigmas: np.ndarray, shift_terminal: float | None) -> np.ndarray:
    """Match diffusers ``stretch_shift_to_terminal`` when ``shift_terminal`` is set."""
    if shift_terminal is None:
        return sigmas
    one_minus = 1.0 - sigmas
    scale = float(one_minus[-1]) / (1.0 - float(shift_terminal))
    if scale == 0.0:
        return sigmas
    return 1.0 - (one_minus / scale)


def ltx_rope_interpolation_scale(
    *,
    temporal_vae_scale: int,
    vae_scale: int,
    fps: float,
) -> tuple[float, float, float]:
    """Micro-conditions for LTX RoPE (diffusers ``LTXPipeline``)."""
    fps_val = max(float(fps), 1.0)
    return (
        float(temporal_vae_scale) / fps_val,
        float(vae_scale),
        float(vae_scale),
    )


def ltx_denormalize_latents(
    ctx: Any,
    latents: Any,
    latents_mean: Any,
    latents_std: Any,
    scaling_factor: float = 1.0,
) -> Any:
    """Undo VAE latent normalization: ``z = z * std / scaling_factor + mean`` ([B,C,T,H,W])."""
    mean = latents_mean.reshape(1, -1, 1, 1, 1)
    std = latents_std.reshape(1, -1, 1, 1, 1)
    return latents * std / float(scaling_factor) + mean


def ltx2_dynamic_schedule(
    steps: int,
    num_tokens: int,
    *,
    base_shift: float = 0.95,
    max_shift: float = 2.05,
    base_tokens: int = 1024,
    max_tokens: int = 4096,
    stretch: bool = True,
    terminal: float = 0.1,
) -> list[float]:
    """Token-count-adaptive flow-matching sigma schedule (LTX-2 dev pipelines)."""
    sigmas = np.linspace(1.0, 0.0, steps + 1)
    slope = (max_shift - base_shift) / (max_tokens - base_tokens)
    intercept = base_shift - slope * base_tokens
    sigma_shift = num_tokens * slope + intercept

    nonzero = sigmas != 0
    shifted = np.empty_like(sigmas)
    shifted[~nonzero] = 0.0
    shifted[nonzero] = math.exp(sigma_shift) / (
        math.exp(sigma_shift) + (1.0 / sigmas[nonzero] - 1.0)
    )
    sigmas = shifted

    if stretch:
        nz = sigmas != 0
        nz_sigmas = sigmas[nz]
        if len(nz_sigmas) > 0:
            one_minus_z = 1.0 - nz_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            if scale_factor != 0:
                sigmas[nz] = 1.0 - (one_minus_z / scale_factor)
    return sigmas.tolist()


def ltx2_schedule(
    steps: int,
    num_tokens: int,
    *,
    base_shift: float = 0.95,
    max_shift: float = 2.05,
    base_tokens: int = 1024,
    max_tokens: int = 4096,
    stretch: bool = True,
    terminal: float = 0.1,
) -> list[float]:
    """LTX-2 dev stage-1 sigma schedule (dgrauet / mlx-forge reference)."""
    return ltx2_dynamic_schedule(
        steps,
        num_tokens,
        base_shift=base_shift,
        max_shift=max_shift,
        base_tokens=base_tokens,
        max_tokens=max_tokens,
        stretch=stretch,
        terminal=terminal,
    )


def compute_video_latent_shape(
    num_frames: int,
    height: int,
    width: int,
    temporal_compression: int = VIDEO_TEMPORAL_SCALE,
    spatial_compression: int = VIDEO_SPATIAL_SCALE,
) -> tuple[int, int, int]:
    """Return ``(F', H', W')`` latent dimensions after LTX video VAE encode."""
    f_lat = (num_frames + temporal_compression - 1) // temporal_compression
    h_lat = height // spatial_compression
    w_lat = width // spatial_compression
    return f_lat, h_lat, w_lat


def compute_audio_token_count(num_video_frames: int, frame_rate: float = 24.0) -> int:
    """Audio latent token count aligned to video duration."""
    duration = num_video_frames / frame_rate
    return round(duration * AUDIO_LATENTS_PER_SECOND)


def compute_video_positions(
    ctx: Any,
    num_frames: int,
    height: int,
    width: int,
    frame_rate: float = 24.0,
) -> Any:
    """3D video token positions ``(1, F*H*W, 3)`` in pixel-space seconds + spatial midpoints."""
    idx = np.arange(num_frames, dtype=np.float32)
    f_starts = np.maximum(idx * VIDEO_TEMPORAL_SCALE + 1 - VIDEO_TEMPORAL_SCALE, 0.0)
    f_ends = np.maximum((idx + 1) * VIDEO_TEMPORAL_SCALE + 1 - VIDEO_TEMPORAL_SCALE, 0.0)
    f_mids = (f_starts + f_ends) / 2.0 / frame_rate

    h_mids = np.arange(height, dtype=np.float32) * VIDEO_SPATIAL_SCALE + VIDEO_SPATIAL_SCALE / 2.0
    w_mids = np.arange(width, dtype=np.float32) * VIDEO_SPATIAL_SCALE + VIDEO_SPATIAL_SCALE / 2.0

    f_grid = np.repeat(np.repeat(f_mids[:, None, None], height, axis=1), width, axis=2)
    h_grid = np.repeat(np.repeat(h_mids[None, :, None], num_frames, axis=0), width, axis=2)
    w_grid = np.repeat(np.repeat(w_mids[None, None, :], num_frames, axis=0), height, axis=1)

    positions = np.stack([f_grid, h_grid, w_grid], axis=-1).reshape(-1, 3)
    return ctx.array(positions[None, :, :].astype(np.float32))


def compute_audio_positions(ctx: Any, num_tokens: int) -> Any:
    """1D audio token positions ``(1, T, 1)`` in seconds."""
    idx = np.arange(num_tokens, dtype=np.float32)
    starts = (
        np.maximum(idx * AUDIO_DOWNSAMPLE_FACTOR + 1 - AUDIO_DOWNSAMPLE_FACTOR, 0.0)
        * AUDIO_HOP_LENGTH
        / AUDIO_SAMPLE_RATE
    )
    ends = (
        np.maximum((idx + 1) * AUDIO_DOWNSAMPLE_FACTOR + 1 - AUDIO_DOWNSAMPLE_FACTOR, 0.0)
        * AUDIO_HOP_LENGTH
        / AUDIO_SAMPLE_RATE
    )
    mids = (starts + ends) / 2.0
    return ctx.array(mids[None, :, None].astype(np.float32))


# ---------------------------------------------------------------------------
# Video latent patchifier (BLC token layout)
# ---------------------------------------------------------------------------

class AudioPatchifier:
    """Flatten audio latents ``(B,8,T,16)`` ↔ ``(B,T,128)`` token layout."""

    def unpatchify(self, tokens: Any, _time_dim: int | None = None) -> Any:
        b, t, _c = tokens.shape
        return tokens.reshape(b, t, 8, 16).transpose(0, 2, 1, 3)


class VideoLatentPatchifier:
    """Patchify ``(B,C,F,H,W)`` video latents to ``(B,N,C)`` tokens."""

    def __init__(
        self,
        patch_size_t: int = 1,
        patch_size_h: int = 1,
        patch_size_w: int = 1,
    ):
        self.patch_size_t = patch_size_t
        self.patch_size_h = patch_size_h
        self.patch_size_w = patch_size_w

    def patchify(self, latent: Any, ctx: Any) -> tuple[Any, tuple[int, int, int]]:
        b, c, f, h, w = latent.shape
        tokens = ctx.reshape(ctx.permute(latent, (0, 2, 3, 4, 1)), (b, f * h * w, c))
        return tokens, (f, h, w)

    def unpatchify(self, tokens: Any, spatial_dims: tuple[int, int, int], ctx: Any) -> Any:
        f, h, w = spatial_dims
        b, _n, c = tokens.shape
        x = ctx.reshape(tokens, (b, f, h, w, c))
        return ctx.permute(x, (0, 4, 1, 2, 3))


# ---------------------------------------------------------------------------
# LatentState (token BLC) + conditioning
# ---------------------------------------------------------------------------

@dataclass
class LatentState:
    """Diffusion state in ``(B, N, C)`` token layout."""

    latent: Any
    clean_latent: Any
    denoise_mask: Any
    positions: Any | None = None
    attention_mask: Any | None = None


@dataclass
class VideoConditionByLatentIndex:
    """Inject encoded image latents at a frame index (I2V)."""

    latent: Any
    frame_idx: int = 0
    strength: float = 1.0


def apply_denoise_mask(ctx: Any, denoised: Any, clean: Any, denoise_mask: Any) -> Any:
    """Blend denoised output with clean reference: ``denoised * mask + clean * (1 - mask)``."""
    one = ctx.ones((1,), dtype=denoised.dtype)
    out = denoised * denoise_mask + clean.astype(ctx.float32()) * (one - denoise_mask)
    return out.astype(denoised.dtype)


def pin_latent_by_mask(ctx: Any, latent: Any, clean: Any, denoise_mask: Any) -> Any:
    """Re-lock preserved tokens after an Euler step (I2V frame-0 pin).

    ``apply_denoise_mask`` only blends the x0 prediction; without this pin the
    noised ``latent`` input to the next step can drift on mask=0 tokens and
    break temporal coherence for long I2V clips.
    """
    one = ctx.ones((1,), dtype=latent.dtype)
    out = latent * denoise_mask + clean.astype(ctx.float32()) * (one - denoise_mask)
    return out.astype(latent.dtype)


def apply_conditioning(
    ctx: Any,
    state: LatentState,
    conditionings: List[VideoConditionByLatentIndex],
    spatial_dims: tuple[int, int, int],
) -> LatentState:
    """Apply I2V conditioning items to a token ``LatentState``."""
    f, h, w = spatial_dims
    tokens_per_frame = h * w
    latent = state.latent
    clean = state.clean_latent
    mask = state.denoise_mask
    mask_value_dtype = mask.dtype

    for cond in conditionings:
        cond_latent = cond.latent
        frame_idx = cond.frame_idx % f if cond.frame_idx < 0 else cond.frame_idx
        if frame_idx >= f:
            raise ValueError(f"Frame index {frame_idx} out of bounds for {f} latent frames")

        if cond_latent.ndim == 5:
            _, _cond_c, cond_f, cond_h, cond_w = cond_latent.shape
            patchifier = VideoLatentPatchifier()
            cond_tokens, _ = patchifier.patchify(cond_latent, ctx)
            lh, lw = cond_h, cond_w
        elif cond_latent.ndim == 3:
            cond_tokens = cond_latent
            cond_f = 1
            lh, lw = h, w
        else:
            raise ValueError(f"LTX conditioning latent must be BLC or BCFHW, got ndim={cond.latent.ndim}")

        if lh != h or lw != w:
            raise ValueError(
                f"Conditioning latent spatial ({lh}, {lw}) does not match target ({h}, {w})"
            )

        num_cond_frames = cond_f
        end_idx = min(frame_idx + num_cond_frames, f)

        for i in range(frame_idx, end_idx):
            start = i * tokens_per_frame
            end = start + tokens_per_frame
            src_start = (i - frame_idx) * tokens_per_frame
            src_end = src_start + tokens_per_frame
            frame_tokens = cond_tokens[:, src_start:src_end, :]
            latent = ctx.concat([latent[:, :start, :], frame_tokens, latent[:, end:, :]], axis=1)
            clean = ctx.concat([clean[:, :start, :], frame_tokens, clean[:, end:, :]], axis=1)
            frame_mask = ctx.full((mask.shape[0], tokens_per_frame, 1), 1.0 - cond.strength, dtype=mask_value_dtype)
            mask = ctx.concat([mask[:, :start, :], frame_mask, mask[:, end:, :]], axis=1)

    return LatentState(
        latent=latent,
        clean_latent=clean,
        denoise_mask=mask,
        positions=state.positions,
        attention_mask=state.attention_mask,
    )


def create_noised_state(
    ctx: Any,
    base_shape: tuple[int, ...],
    *,
    conditionings: list | None = None,
    spatial_dims: tuple[int, int, int] | None = None,
    positions: Any | None = None,
    seed: int,
    sigma: float = 1.0,
    initial_latent: Any | None = None,
    dtype: Any | None = None,
    legacy_scalar_blend: bool = False,
) -> LatentState:
    """Build a noised ``LatentState``: init → conditionings → mask-aware noise."""
    if dtype is None:
        dtype = ctx.bfloat16()

    if initial_latent is None:
        latent = ctx.zeros(base_shape, dtype=dtype)
    else:
        latent = initial_latent

    state = LatentState(
        latent=latent,
        clean_latent=latent,
        denoise_mask=ctx.ones((base_shape[0], base_shape[1], 1), dtype=dtype),
        positions=positions,
    )

    if legacy_scalar_blend:
        from backend.engine.runtime.mlx_runtime import set_random_seed

        set_random_seed(getattr(ctx, "set_random_seed", None), seed)
        noise = ctx.randn(state.clean_latent.shape, dtype=dtype)
        blended = noise * sigma + state.clean_latent * (1.0 - sigma)
        state = LatentState(
            latent=blended,
            clean_latent=state.clean_latent,
            denoise_mask=state.denoise_mask,
            positions=state.positions,
            attention_mask=state.attention_mask,
        )
        if conditionings and spatial_dims is not None:
            state = apply_conditioning(ctx, state, conditionings, spatial_dims)
        return state

    if conditionings and spatial_dims is not None:
        state = apply_conditioning(ctx, state, conditionings, spatial_dims)

    from backend.engine.runtime.mlx_runtime import set_random_seed

    set_random_seed(getattr(ctx, "set_random_seed", None), seed)
    noise = ctx.randn(state.clean_latent.shape, dtype=dtype)
    scaled_mask = state.denoise_mask * sigma
    one = ctx.ones((1,), dtype=dtype)
    noised = noise * scaled_mask + state.clean_latent * (one - scaled_mask)
    return LatentState(
        latent=noised,
        clean_latent=state.clean_latent,
        denoise_mask=state.denoise_mask,
        positions=state.positions,
        attention_mask=state.attention_mask,
    )


def reference_latent_frame_count(reference_pixel_frames: int, temporal_compression: int = VIDEO_TEMPORAL_SCALE) -> int:
    """How many latent time slices are fully covered by reference pixel frames."""
    pf = max(1, int(reference_pixel_frames))
    return max(1, (pf + temporal_compression - 1) // temporal_compression)


def build_ltx_av_extend_masks(
    ctx: Any,
    *,
    num_video_latent_frames: int,
    tokens_per_frame: int,
    num_audio_tokens: int,
    reference_latent_frames: int,
    reference_audio_tokens: int,
    dtype: Any | None = None,
) -> tuple[Any, Any]:
    """Build per-token denoise masks for extend-at-end (0=freeze, 1=denoise)."""
    if dtype is None:
        dtype = ctx.bfloat16()
    f_lat = max(1, int(num_video_latent_frames))
    tpf = max(1, int(tokens_per_frame))
    ref_f = max(0, min(int(reference_latent_frames), f_lat))
    ref_v_tokens = ref_f * tpf
    v_total = f_lat * tpf
    video_mask = ctx.ones((1, v_total, 1), dtype=dtype)
    if ref_v_tokens > 0:
        video_mask = ctx.concat(
            [
                ctx.zeros((1, ref_v_tokens, 1), dtype=dtype),
                ctx.ones((1, v_total - ref_v_tokens, 1), dtype=dtype),
            ],
            axis=1,
        )
    a_total = max(1, int(num_audio_tokens))
    ref_a = max(0, min(int(reference_audio_tokens), a_total))
    audio_mask = ctx.ones((1, a_total, 1), dtype=dtype)
    if ref_a > 0:
        audio_mask = ctx.concat(
            [
                ctx.zeros((1, ref_a, 1), dtype=dtype),
                ctx.ones((1, a_total - ref_a, 1), dtype=dtype),
            ],
            axis=1,
        )
    return video_mask, audio_mask


def merge_extend_latent_states(
    ctx: Any,
    *,
    video_tokens: Any,
    audio_tokens: Any,
    video_clean: Any,
    audio_clean: Any,
    video_mask: Any,
    audio_mask: Any,
    video_positions: Any,
    audio_positions: Any,
    seed: int,
    sigma: float = 1.0,
    spatial_dims: tuple[int, int, int],
    legacy_scalar_blend: bool = True,
) -> tuple[LatentState, LatentState]:
    """Initialize extend pass states from encoded reference + noise in extend region."""
    v_state = LatentState(
        latent=video_tokens,
        clean_latent=video_clean,
        denoise_mask=video_mask,
        positions=video_positions,
    )
    a_state = LatentState(
        latent=audio_tokens,
        clean_latent=audio_clean,
        denoise_mask=audio_mask,
        positions=audio_positions,
    )
    from backend.engine.runtime.mlx_runtime import set_random_seed

    set_random_seed(getattr(ctx, "set_random_seed", None), seed)
    dtype = video_tokens.dtype
    v_noise = ctx.randn(v_state.clean_latent.shape, dtype=dtype)
    a_noise = ctx.randn(a_state.clean_latent.shape, dtype=dtype)
    one = ctx.ones((1,), dtype=dtype)
    v_noised = v_noise * (v_state.denoise_mask * sigma) + v_state.clean_latent * (one - v_state.denoise_mask * sigma)
    a_noised = a_noise * (a_state.denoise_mask * sigma) + a_state.clean_latent * (one - a_state.denoise_mask * sigma)
    return (
        LatentState(
            latent=v_noised,
            clean_latent=v_state.clean_latent,
            denoise_mask=v_state.denoise_mask,
            positions=v_state.positions,
        ),
        LatentState(
            latent=a_noised,
            clean_latent=a_state.clean_latent,
            denoise_mask=a_state.denoise_mask,
            positions=a_state.positions,
        ),
    )


# Legacy NCTHW helpers (28-layer diffusers transformer path)
@dataclass
class LTXLatentState:
    latent: Any
    clean_latent: Any
    denoise_mask: Any

    def clone(self, ctx: Any) -> "LTXLatentState":
        return LTXLatentState(self.latent, self.clean_latent, self.denoise_mask)


def create_initial_ltx_state(
    ctx: Any,
    shape: tuple,
    seed: int | None = None,
    noise_scale: float = 1.0,
) -> LTXLatentState:
    """Initial noisy state in ``(B,C,F,H,W)`` layout (legacy LTX 28L path)."""
    if seed is not None:
        noise = ctx.seeded_randn(shape, seed, dtype=ctx.float32())
    else:
        noise = ctx.randn(shape, dtype=ctx.float32())
    b, _c, f, _h, _w = shape
    return LTXLatentState(
        latent=noise * noise_scale,
        clean_latent=ctx.zeros(shape, dtype=ctx.float32()),
        denoise_mask=ctx.ones((b, 1, f, 1, 1), dtype=ctx.float32()),
    )


# ---------------------------------------------------------------------------
# Dev guidance (CFG / STG / modality — ported from LTX-2.3 reference)
# ---------------------------------------------------------------------------

DEFAULT_LTX_IMAGE_CRF = 33
_DEFAULT_STG_BLOCKS = (28,)


@dataclass(frozen=True)
class MultiModalGuiderParams:
    cfg_scale: float = 1.0
    stg_scale: float = 0.0
    stg_blocks: tuple[int, ...] = ()
    rescale_scale: float = 0.0
    modality_scale: float = 1.0
    skip_step: int = 0


@dataclass(frozen=True)
class MultiModalGuider:
    params: MultiModalGuiderParams
    negative_context: Any | None = None

    def calculate(
        self,
        cond: Any,
        uncond_text: Any | float,
        uncond_perturbed: Any | float,
        uncond_modality: Any | float,
    ) -> Any:
        pred = (
            cond
            + (self.params.cfg_scale - 1) * (cond - uncond_text)
            + self.params.stg_scale * (cond - uncond_perturbed)
            + (self.params.modality_scale - 1) * (cond - uncond_modality)
        )
        if self.params.rescale_scale != 0:
            import mlx.core as mx

            factor = mx.sqrt(mx.var(cond)) / (mx.sqrt(mx.var(pred)) + 1e-8)
            factor = self.params.rescale_scale * factor + (1 - self.params.rescale_scale)
            pred = pred * factor
        return pred

    def do_unconditional_generation(self) -> bool:
        return not math.isclose(self.params.cfg_scale, 1.0)

    def do_perturbed_generation(self) -> bool:
        return not math.isclose(self.params.stg_scale, 0.0)

    def do_isolated_modality_generation(self) -> bool:
        return not math.isclose(self.params.modality_scale, 1.0)

    def should_skip_step(self, step: int) -> bool:
        if self.params.skip_step == 0:
            return False
        return step % (self.params.skip_step + 1) != 0


@dataclass(frozen=True)
class MultiModalGuiderFactory:
    negative_context: Any | None = None
    _params: MultiModalGuiderParams = MultiModalGuiderParams()

    @classmethod
    def constant(
        cls,
        params: MultiModalGuiderParams,
        *,
        negative_context: Any | None = None,
    ) -> MultiModalGuiderFactory:
        return cls(negative_context=negative_context, _params=params)

    def build_from_sigma(self, sigma: float | Any) -> MultiModalGuider:
        del sigma
        return MultiModalGuider(params=self._params, negative_context=self.negative_context)


def ltx_dev_video_guider_params(cfg_scale: float, *, stg_scale: float = 1.0) -> MultiModalGuiderParams:
    return MultiModalGuiderParams(
        cfg_scale=float(cfg_scale),
        stg_scale=float(stg_scale),
        stg_blocks=_DEFAULT_STG_BLOCKS,
        rescale_scale=0.7,
        modality_scale=3.0,
    )


def ltx_dev_audio_guider_params(*, stg_scale: float = 1.0) -> MultiModalGuiderParams:
    return MultiModalGuiderParams(
        cfg_scale=7.0,
        stg_scale=float(stg_scale),
        stg_blocks=_DEFAULT_STG_BLOCKS,
        rescale_scale=0.7,
        modality_scale=3.0,
    )

