"""LTX 2.3 two-stage T2V/I2V generation — in-repo MLX orchestration (dgrauet reference)."""
from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np
from PIL import Image

from backend.engine.runtime.mlx_runtime import run_eval
from backend.engine.config.model_configs import LTXConfig
from backend.engine.families.ltx.pipeline_math_mlx import (
    DEFAULT_LTX_IMAGE_CRF,
    DISTILLED_SIGMAS,
    STAGE_2_SIGMAS,
    AudioPatchifier,
    LatentState,
    MultiModalGuiderFactory,
    VideoConditionByLatentIndex,
    VideoLatentPatchifier,
    apply_denoise_mask,
    compute_audio_positions,
    compute_audio_token_count,
    compute_video_latent_shape,
    compute_video_positions,
    create_noised_state,
    pin_latent_by_mask,
    ltx2_schedule,
    ltx_dev_audio_guider_params,
    ltx_dev_video_guider_params,
    VIDEO_SPATIAL_SCALE,
)
from backend.engine.families.ltx.text_encoder_mlx import LTX23GemmaEncoder
from backend.engine.families.ltx.transformer_mlx import LTX23X0Model, load_ltx23_x0_model
from backend.engine.families.ltx.vae import decode_ltx23_av_to_mp4
from backend.engine.families.ltx.vae_mlx import load_ltx23_latent_upsampler, load_ltx23_video_encoder
from backend.engine.pipelines.pipeline_progress import emit_denoise_progress, emit_post_progress
from backend.engine.runtime._base import RuntimeContext
from backend.utils.video_sr_ffmpeg import require_ffmpeg

_DEFAULT_CFG = 3.0
_DISTILLED_STAGE1_FULL = 8


def _resolve_distilled_stage1_steps(steps: int, *, on_log: Callable[[str, str], None] | None) -> int:
    requested = max(1, int(steps))
    if requested >= _DISTILLED_STAGE1_FULL:
        return requested
    if on_log:
        on_log(
            "warning",
            f"LTX distilled stage1_steps={requested} is below required {_DISTILLED_STAGE1_FULL}; "
            f"using full stage-1 schedule (stage-2 adds 3 refine steps)",
        )
    return _DISTILLED_STAGE1_FULL


@dataclass
class _DenoiseOutput:
    video_latent: mx.array
    audio_latent: mx.array


def _materialize(ctx: RuntimeContext, *arrays: mx.array) -> None:
    run_eval(getattr(ctx, "eval", None), *arrays)


def _clear_cache(ctx: RuntimeContext) -> None:
    if hasattr(ctx, "clear_cache"):
        ctx.clear_cache()


def _euler_step(x: mx.array, x0: mx.array, sigma: float, sigma_next: float) -> mx.array:
    if sigma == 0:
        return x0
    d = (x - x0) / sigma
    return x + (sigma_next - sigma) * d


def _is_uniform_mask(mask: mx.array) -> bool:
    return bool(mx.all(mask == 1.0).item())


def _per_token_timesteps(ctx: RuntimeContext, sigma: float, denoise_mask: mx.array) -> mx.array:
    return (denoise_mask * sigma).squeeze(-1)


def _is_inrepo_x0(model: Any) -> bool:
    return isinstance(model, LTX23X0Model)


def _parse_ffmpeg_size(stderr: str) -> tuple[int, int] | None:
    match = re.search(r",\s*(\d{2,5})x(\d{2,5})", stderr)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _encode_single_frame_h264(image_array: np.ndarray, crf: float) -> bytes:
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)
    height, width, _ = image_array.shape
    pad_w = width + (width & 1)
    pad_h = height + (height & 1)
    if (pad_w, pad_h) != (width, height):
        padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        padded[:height, :width, :] = image_array
        image_array = padded
    ffmpeg = require_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{pad_w}x{pad_h}",
        "-r",
        "1",
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(int(crf)),
        "-frames:v",
        "1",
        "-f",
        "mp4",
        "-movflags",
        "frag_keyframe+empty_moov",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, input=image_array.tobytes(), capture_output=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(
            f"LTX I2V H.264 preprocess failed: {proc.stderr.decode(errors='ignore')}"
        )
    return proc.stdout


def _decode_single_frame_h264(data: bytes) -> np.ndarray:
    ffmpeg = require_ffmpeg()
    probe = subprocess.run(
        [ffmpeg, "-i", "pipe:0", "-f", "null", "-"],
        input=data,
        capture_output=True,
        timeout=30,
    )
    size = _parse_ffmpeg_size(probe.stderr.decode(errors="ignore"))
    if size is None:
        raise RuntimeError("LTX I2V H.264 preprocess: could not probe frame size")
    width, height = size
    proc = subprocess.run(
        [
            ffmpeg,
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        input=data,
        capture_output=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"LTX I2V H.264 decode failed: {proc.stderr.decode(errors='ignore')}"
        )
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(height, width, 3).copy()


def _preprocess_i2v_image(image: np.ndarray, crf: float) -> np.ndarray:
    if crf <= 0:
        return image
    h, w, _ = image.shape
    encoded = _encode_single_frame_h264(image, crf)
    decoded = _decode_single_frame_h264(encoded)
    if decoded.shape[0] != h or decoded.shape[1] != w:
        decoded = decoded[:h, :w, :]
    return decoded


def _resize_and_center_crop(image: Image.Image, height: int, width: int) -> Image.Image:
    src_w, src_h = image.size
    scale = max(height / src_h, width / src_w)
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    crop_left = (new_w - width) // 2
    crop_top = (new_h - height) // 2
    return image.crop((crop_left, crop_top, crop_left + width, crop_top + height))


def _load_i2v_image_tensor(
    image_path: str,
    height: int,
    width: int,
    ctx: RuntimeContext,
    *,
    crf: int = DEFAULT_LTX_IMAGE_CRF,
) -> mx.array:
    """I2V image path aligned with dgrauet reference (H.264 CRF round-trip)."""
    arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    arr = _preprocess_i2v_image(arr, float(crf))
    image = _resize_and_center_crop(Image.fromarray(arr, mode="RGB"), height, width)
    f = np.asarray(image, dtype=np.float32) / 255.0
    f = f * 2.0 - 1.0
    tensor = ctx.array(f).transpose(2, 0, 1)[None, ...]
    return tensor.astype(ctx.bfloat16())


def _denoise_loop(
    ctx: RuntimeContext,
    model: LTX23X0Model,
    video_state: LatentState,
    audio_state: LatentState,
    video_text_embeds: mx.array,
    audio_text_embeds: mx.array,
    sigmas: list[float],
    *,
    on_progress: Callable[..., None] | None = None,
    progress_step_offset: int = 0,
    progress_total_steps: int = 1,
    on_log: Callable[[str, str], None] | None = None,
    progress_label: str = "denoise",
) -> _DenoiseOutput:
    video_x = video_state.latent
    audio_x = audio_state.latent
    video_uniform = _is_uniform_mask(video_state.denoise_mask)
    audio_uniform = _is_uniform_mask(audio_state.denoise_mask)
    n_steps = max(1, len(sigmas) - 1)

    for step_idx, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        sigma_arr = ctx.array([sigma], dtype=ctx.bfloat16())
        b = int(video_x.shape[0])
        call_kwargs: dict[str, Any] = dict(
            video_latent=video_x,
            audio_latent=audio_x,
            sigma=ctx.broadcast_to(sigma_arr, (b,)),
            video_text_embeds=video_text_embeds,
            audio_text_embeds=audio_text_embeds,
            video_positions=video_state.positions,
            audio_positions=audio_state.positions,
        )
        if not video_uniform:
            call_kwargs["video_timesteps"] = _per_token_timesteps(ctx, sigma, video_state.denoise_mask)
        if not audio_uniform:
            call_kwargs["audio_timesteps"] = _per_token_timesteps(ctx, sigma, audio_state.denoise_mask)

        video_x0, audio_x0 = model(**call_kwargs)
        video_x0 = apply_denoise_mask(ctx, video_x0, video_state.clean_latent, video_state.denoise_mask)
        audio_x0 = apply_denoise_mask(ctx, audio_x0, audio_state.clean_latent, audio_state.denoise_mask)
        video_x = _euler_step(video_x, video_x0, sigma, sigma_next)
        audio_x = _euler_step(audio_x, audio_x0, sigma, sigma_next)
        video_x = pin_latent_by_mask(ctx, video_x, video_state.clean_latent, video_state.denoise_mask)
        audio_x = pin_latent_by_mask(ctx, audio_x, audio_state.clean_latent, audio_state.denoise_mask)
        _materialize(ctx, video_x, audio_x)

        step_1based = progress_step_offset + step_idx + 1
        emit_denoise_progress(on_progress, step_1based, progress_total_steps)
        if on_log:
            on_log(
                "info",
                f"{progress_label} step {step_1based}/{progress_total_steps} "
                f"(stage σ {step_idx + 1}/{n_steps})",
            )

    return _DenoiseOutput(video_latent=video_x, audio_latent=audio_x)


def _multimodal_guided_denoise_loop(
    ctx: RuntimeContext,
    model: LTX23X0Model,
    video_state: LatentState,
    audio_state: LatentState,
    video_text_embeds: mx.array,
    audio_text_embeds: mx.array,
    video_guider_factory: MultiModalGuiderFactory,
    audio_guider_factory: MultiModalGuiderFactory,
    sigmas: list[float],
    *,
    on_progress: Callable[..., None] | None = None,
    progress_step_offset: int = 0,
    progress_total_steps: int = 1,
    on_log: Callable[[str, str], None] | None = None,
) -> _DenoiseOutput:
    """Dev stage-1 denoise with CFG + STG + modality guidance (dgrauet reference)."""
    if not _is_inrepo_x0(model):
        raise RuntimeError(
            "LTX 2.3 dev guidance requires the in-repo LTX23X0Model DiT wrapper."
        )

    from backend.engine.families.ltx.perturbations_mlx import (
        BatchedPerturbationConfig,
        Perturbation,
        PerturbationConfig,
        PerturbationType,
    )

    video_x = video_state.latent
    audio_x = audio_state.latent
    video_uniform = _is_uniform_mask(video_state.denoise_mask)
    audio_uniform = _is_uniform_mask(audio_state.denoise_mask)
    n_steps = max(1, len(sigmas) - 1)

    for step_idx, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        video_guider = video_guider_factory.build_from_sigma(sigma)
        audio_guider = audio_guider_factory.build_from_sigma(sigma)
        sigma_arr = ctx.array([sigma], dtype=ctx.bfloat16())
        b = int(video_x.shape[0])

        base_kwargs: dict[str, Any] = dict(
            video_latent=video_x,
            audio_latent=audio_x,
            sigma=ctx.broadcast_to(sigma_arr, (b,)),
            video_positions=video_state.positions,
            audio_positions=audio_state.positions,
        )
        if not video_uniform:
            base_kwargs["video_timesteps"] = _per_token_timesteps(ctx, sigma, video_state.denoise_mask)
        if not audio_uniform:
            base_kwargs["audio_timesteps"] = _per_token_timesteps(ctx, sigma, audio_state.denoise_mask)

        def _predict(v_embeds: mx.array, a_embeds: mx.array, perturbations=None) -> tuple[mx.array, mx.array]:
            kw = {
                **base_kwargs,
                "video_text_embeds": v_embeds,
                "audio_text_embeds": a_embeds,
            }
            if perturbations is not None:
                kw["perturbations"] = perturbations
            v_x0, a_x0 = model(**kw)
            v_x0 = apply_denoise_mask(ctx, v_x0, video_state.clean_latent, video_state.denoise_mask)
            a_x0 = apply_denoise_mask(ctx, a_x0, audio_state.clean_latent, audio_state.denoise_mask)
            return v_x0, a_x0

        cond_v, cond_a = _predict(video_text_embeds, audio_text_embeds)

        neg_v: mx.array | float = 0.0
        neg_a: mx.array | float = 0.0
        if video_guider.do_unconditional_generation() or audio_guider.do_unconditional_generation():
            neg_v_embeds = (
                video_guider.negative_context if video_guider.negative_context is not None else video_text_embeds
            )
            neg_a_embeds = (
                audio_guider.negative_context if audio_guider.negative_context is not None else audio_text_embeds
            )
            neg_v, neg_a = _predict(neg_v_embeds, neg_a_embeds)

        ptb_v: mx.array | float = 0.0
        ptb_a: mx.array | float = 0.0
        if video_guider.do_perturbed_generation() or audio_guider.do_perturbed_generation():
            perturbations: list[Perturbation] = []
            if video_guider.do_perturbed_generation():
                perturbations.append(
                    Perturbation(
                        type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                        blocks=list(video_guider.params.stg_blocks),
                    )
                )
            if audio_guider.do_perturbed_generation():
                perturbations.append(
                    Perturbation(
                        type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
                        blocks=list(audio_guider.params.stg_blocks),
                    )
                )
            batched = BatchedPerturbationConfig(
                perturbations=[PerturbationConfig(perturbations=perturbations)] * b
            )
            ptb_v, ptb_a = _predict(video_text_embeds, audio_text_embeds, perturbations=batched)

        mod_v: mx.array | float = 0.0
        mod_a: mx.array | float = 0.0
        if video_guider.do_isolated_modality_generation() or audio_guider.do_isolated_modality_generation():
            mod_list = [
                Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
            ]
            batched = BatchedPerturbationConfig(
                perturbations=[PerturbationConfig(perturbations=mod_list)] * b
            )
            mod_v, mod_a = _predict(video_text_embeds, audio_text_embeds, perturbations=batched)

        video_x0 = video_guider.calculate(cond_v, neg_v, ptb_v, mod_v)
        audio_x0 = audio_guider.calculate(cond_a, neg_a, ptb_a, mod_a)
        video_x0 = apply_denoise_mask(ctx, video_x0, video_state.clean_latent, video_state.denoise_mask)
        audio_x0 = apply_denoise_mask(ctx, audio_x0, audio_state.clean_latent, audio_state.denoise_mask)

        video_x = _euler_step(video_x, video_x0, sigma, sigma_next)
        audio_x = _euler_step(audio_x, audio_x0, sigma, sigma_next)
        video_x = pin_latent_by_mask(ctx, video_x, video_state.clean_latent, video_state.denoise_mask)
        audio_x = pin_latent_by_mask(ctx, audio_x, audio_state.clean_latent, audio_state.denoise_mask)
        _materialize(ctx, video_x, audio_x)

        step_1based = progress_step_offset + step_idx + 1
        emit_denoise_progress(on_progress, step_1based, progress_total_steps)
        if on_log:
            on_log(
                "info",
                f"denoise step {step_1based}/{progress_total_steps} "
                f"(dev stage1 σ {step_idx + 1}/{n_steps})",
            )

    return _DenoiseOutput(video_latent=video_x, audio_latent=audio_x)


def _i2v_conditionings(
    ctx: RuntimeContext,
    image_path: str,
    *,
    enc_h: int,
    enc_w: int,
    video_encoder: Any,
    anchor_strength: float = 1.0,
) -> list[VideoConditionByLatentIndex]:
    pixels = _load_i2v_image_tensor(image_path, enc_h, enc_w, ctx)
    pixels = ctx.expand_dims(pixels, axis=2)
    latent = video_encoder.encode(pixels)
    patchifier = VideoLatentPatchifier()
    tokens, _ = patchifier.patchify(latent, ctx)
    strength = float(max(0.0, min(1.0, anchor_strength)))
    return [VideoConditionByLatentIndex(latent=tokens, frame_idx=0, strength=strength)]


def _pre_denoise_materialize(ctx: RuntimeContext, video_state: LatentState, audio_state: LatentState) -> None:
    """Force-eval noised states before the denoise loop (Metal watchdog + stable I2V)."""
    _materialize(
        ctx,
        video_state.latent,
        video_state.clean_latent,
        video_state.denoise_mask,
        audio_state.latent,
        audio_state.clean_latent,
        audio_state.denoise_mask,
    )


class LTX23MlxGenerator:
    """In-repo two-stage LTX 2.3 T2V/I2V generator (MLX, dgrauet algorithm)."""

    def __init__(
        self,
        ctx: RuntimeContext,
        bundle_root: Path,
        config: LTXConfig | None = None,
        *,
        entry: Any | None = None,
        version_key: str | None = None,
    ):
        self.ctx = ctx
        self.bundle_root = Path(bundle_root)
        self.config = config or LTXConfig()
        self._registry_entry = entry
        self._version_key = version_key
        self._encoder: LTX23GemmaEncoder | None = None
        self._dit: LTX23X0Model | None = None
        self._video_encoder = None
        self._upsampler = None
        self._video_patchifier = VideoLatentPatchifier()
        self._audio_patchifier = AudioPatchifier()

    def load(self) -> None:
        """No-op — components load lazily in ``generate_and_save``."""

    def _log(self, on_log: Callable[[str, str], None] | None, level: str, message: str) -> None:
        if on_log:
            on_log(level, message)

    def _weight_stem(self, *, step_distill: bool, stage2_dev_refine: bool) -> str:
        if step_distill or stage2_dev_refine:
            return "transformer-distilled"
        return "transformer-dev"

    def _load_dit(
        self,
        *,
        step_distill: bool,
        stage2_dev_refine: bool = False,
        on_log: Callable[[str, str], None] | None = None,
    ) -> LTX23X0Model:
        stem = self._weight_stem(step_distill=step_distill, stage2_dev_refine=stage2_dev_refine)
        self._log(on_log, "info", f"Loading LTX 2.3 transformer: {stem}")
        self._dit = load_ltx23_x0_model(
            self.ctx,
            self.bundle_root,
            self.config,
            weight_stem=stem,
            entry=self._registry_entry,
            version_key=self._version_key,
        )
        inner = self._dit.model
        self._log(
            on_log,
            "info",
            f"LTX 2.3 DiT scales ts={inner._timestep_scale} av_ca={inner._av_ca_timestep_scale}",
        )
        return self._dit

    def _encode_prompts(
        self,
        prompt: str,
        *,
        step_distill: bool,
        on_log: Callable[[str, str], None] | None = None,
    ) -> tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array] | None]:
        if self._encoder is None:
            self._encoder = LTX23GemmaEncoder(self.ctx, self.bundle_root, self.config)
        if step_distill:
            pos = self._encoder.encode(prompt, on_log=on_log)
            return pos, None
        pos, neg = self._encoder.encode_with_negative(prompt, on_log=on_log)
        return pos, neg

    def _generate_two_stage(
        self,
        *,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        seed: int,
        stage1_steps: int,
        stage2_steps: int,
        guidance: float,
        step_distill: bool,
        image_path: str | None,
        on_log: Callable[[str, str], None] | None,
        on_progress: Callable[..., None] | None = None,
        stage1_states: tuple[LatentState, LatentState] | None = None,
    ) -> tuple[mx.array, mx.array]:
        ctx = self.ctx
        load_fn = getattr(ctx, "load_weights", None)
        low_memory = bool(getattr(self.config, "ltx_low_memory", True))

        (video_embeds, audio_embeds), neg = self._encode_prompts(
            prompt, step_distill=step_distill, on_log=on_log
        )
        _materialize(ctx, video_embeds, audio_embeds)
        neg_video = neg_audio = None
        if neg is not None:
            neg_video, neg_audio = neg
            _materialize(ctx, neg_video, neg_audio)

        if low_memory and self._encoder is not None:
            self._encoder.free()
            self._encoder = None
            _clear_cache(ctx)

        half_h, half_w = height // 2, width // 2
        f_lat, h_half, w_half = compute_video_latent_shape(num_frames, half_h, half_w)
        video_shape = (1, f_lat * h_half * w_half, 128)
        audio_t = compute_audio_token_count(num_frames, frame_rate=fps)
        audio_shape = (1, audio_t, 128)
        num_tokens = f_lat * h_half * w_half

        video_positions_1 = compute_video_positions(ctx, f_lat, h_half, w_half, frame_rate=fps)
        audio_positions = compute_audio_positions(ctx, audio_t)

        if image_path and self._video_encoder is None:
            self._video_encoder = load_ltx23_video_encoder(self.bundle_root, load_fn=load_fn)

        i2v_anchor_strength = float(getattr(self.config, "ltx_i2v_anchor_strength", 1.0) or 1.0)

        conditionings_1: list[VideoConditionByLatentIndex] = []
        if image_path:
            if self._video_encoder is None:
                raise RuntimeError("LTX I2V requires video encoder but load failed")
            conditionings_1 = _i2v_conditionings(
                ctx,
                image_path,
                enc_h=h_half * VIDEO_SPATIAL_SCALE,
                enc_w=w_half * VIDEO_SPATIAL_SCALE,
                video_encoder=self._video_encoder,
                anchor_strength=i2v_anchor_strength,
            )

        if stage1_states is not None:
            video_state, audio_state = stage1_states
            if image_path:
                self._log(
                    on_log,
                    "warning",
                    "LTX extend pass: ignoring image_path when stage1_states are prebuilt",
                )
        else:
            video_state = create_noised_state(
                ctx,
                video_shape,
                conditionings=conditionings_1,
                spatial_dims=(f_lat, h_half, w_half),
                positions=video_positions_1,
                seed=seed,
                sigma=1.0,
                legacy_scalar_blend=True,
            )
            audio_state = create_noised_state(
                ctx,
                audio_shape,
                conditionings=[],
                spatial_dims=(f_lat, h_half, w_half),
                positions=audio_positions,
                seed=seed + 1,
                sigma=1.0,
                legacy_scalar_blend=True,
            )

        if step_distill:
            sigmas_1 = DISTILLED_SIGMAS[: stage1_steps + 1]
        else:
            sigmas_1 = ltx2_schedule(stage1_steps, num_tokens)

        stage1_denoise_steps = max(1, len(sigmas_1) - 1)
        stage2_denoise_steps = max(1, stage2_steps)
        total_denoise_steps = stage1_denoise_steps + stage2_denoise_steps

        model = self._load_dit(step_distill=step_distill, on_log=on_log)
        self._log(on_log, "info", f"LTX 2.3 stage 1 denoise steps={stage1_denoise_steps} at {half_w}x{half_h}")
        _pre_denoise_materialize(ctx, video_state, audio_state)

        if step_distill:
            output_1 = _denoise_loop(
                ctx,
                model,
                video_state,
                audio_state,
                video_embeds,
                audio_embeds,
                sigmas_1,
                on_progress=on_progress,
                progress_step_offset=0,
                progress_total_steps=total_denoise_steps,
                on_log=on_log,
                progress_label="denoise stage1",
            )
        else:
            cfg_scale = float(guidance if guidance > 0 else _DEFAULT_CFG)
            video_factory = MultiModalGuiderFactory.constant(
                ltx_dev_video_guider_params(cfg_scale),
                negative_context=neg_video,
            )
            audio_factory = MultiModalGuiderFactory.constant(ltx_dev_audio_guider_params())
            output_1 = _multimodal_guided_denoise_loop(
                ctx,
                model,
                video_state,
                audio_state,
                video_embeds,
                audio_embeds,
                video_factory,
                audio_factory,
                sigmas_1,
                on_progress=on_progress,
                progress_step_offset=0,
                progress_total_steps=total_denoise_steps,
                on_log=on_log,
            )

        if low_memory:
            _clear_cache(ctx)

        if not step_distill:
            model = self._load_dit(step_distill=False, stage2_dev_refine=True, on_log=on_log)

        gen_tokens_1 = output_1.video_latent[:, :num_tokens, :]
        video_half = self._video_patchifier.unpatchify(gen_tokens_1, (f_lat, h_half, w_half), ctx)

        if self._video_encoder is None:
            self._video_encoder = load_ltx23_video_encoder(self.bundle_root, load_fn=load_fn)
        if self._upsampler is None:
            self._upsampler = load_ltx23_latent_upsampler(self.bundle_root, load_fn=load_fn)

        video_mlx = video_half.transpose(0, 2, 3, 4, 1)
        video_denorm = self._video_encoder.denormalize_latent(video_mlx)
        video_denorm = video_denorm.transpose(0, 4, 1, 2, 3)
        video_upscaled = self._upsampler(video_denorm)
        video_up_mlx = video_upscaled.transpose(0, 2, 3, 4, 1)
        video_upscaled = self._video_encoder.normalize_latent(video_up_mlx)
        video_upscaled = video_upscaled.transpose(0, 4, 1, 2, 3)
        _materialize(ctx, video_upscaled)

        h_full = h_half * 2
        w_full = w_half * 2

        conditionings_2: list[VideoConditionByLatentIndex] = []
        if image_path:
            if self._video_encoder is None:
                raise RuntimeError("LTX I2V requires video encoder but load failed")
            conditionings_2 = _i2v_conditionings(
                ctx,
                image_path,
                enc_h=h_full * VIDEO_SPATIAL_SCALE,
                enc_w=w_full * VIDEO_SPATIAL_SCALE,
                video_encoder=self._video_encoder,
                anchor_strength=i2v_anchor_strength,
            )

        if low_memory:
            self._video_encoder = None
            self._upsampler = None
            if step_distill:
                self._dit = None
            _clear_cache(ctx)

        if step_distill and self._dit is None:
            model = self._load_dit(step_distill=True, on_log=on_log)

        video_tokens, _ = self._video_patchifier.patchify(video_upscaled, ctx)
        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1]
        start_sigma = sigmas_2[0]

        video_positions_2 = compute_video_positions(ctx, f_lat, h_full, w_full, frame_rate=fps)
        video_state_2 = create_noised_state(
            ctx,
            video_tokens.shape,
            conditionings=conditionings_2,
            spatial_dims=(f_lat, h_full, w_full),
            positions=video_positions_2,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=video_tokens,
            legacy_scalar_blend=True,
        )
        audio_state_2 = create_noised_state(
            ctx,
            output_1.audio_latent.shape,
            conditionings=[],
            spatial_dims=(f_lat, h_full, w_full),
            positions=audio_positions,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=output_1.audio_latent,
            legacy_scalar_blend=False,
        )

        self._log(on_log, "info", f"LTX 2.3 stage 2 denoise steps={stage2_denoise_steps} at {w_full}x{h_full}")
        _pre_denoise_materialize(ctx, video_state_2, audio_state_2)
        output_2 = _denoise_loop(
            ctx,
            model,
            video_state_2,
            audio_state_2,
            video_embeds,
            audio_embeds,
            sigmas_2,
            on_progress=on_progress,
            progress_step_offset=stage1_denoise_steps,
            progress_total_steps=total_denoise_steps,
            on_log=on_log,
            progress_label="denoise stage2",
        )

        if low_memory:
            self._dit = None
            _clear_cache(ctx)

        gen_tokens_2 = output_2.video_latent[:, : f_lat * h_full * w_full, :]
        video_latent = self._video_patchifier.unpatchify(gen_tokens_2, (f_lat, h_full, w_full), ctx)
        audio_latent = self._audio_patchifier.unpatchify(output_2.audio_latent)
        return video_latent, audio_latent

    def generate_and_save(
        self,
        *,
        prompt: str,
        output_path: str,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        seed: int,
        steps: int,
        guidance: float,
        step_distill: bool,
        image_path: str | None,
        negative_prompt: str = "",
        on_log: Callable[[str, str], None] | None,
        on_progress: Callable[..., None] | None = None,
    ) -> str:
        if getattr(self.ctx, "backend", None) != "mlx":
            raise RuntimeError(
                f"LTX 2.3 requires MLX runtime (got {getattr(self.ctx, 'backend', None)!r})"
            )

        stage2_steps = int(getattr(self.config, "ltx_stage2_steps", 3) or 3)
        stage1_steps = (
            _resolve_distilled_stage1_steps(steps, on_log=on_log)
            if step_distill
            else max(1, int(steps))
        )
        mode = "distilled" if step_distill else "dev"
        i2v = "i2v" if image_path else "t2v"
        if image_path and int(num_frames) > 121:
            self._log(
                on_log,
                "warning",
                f"LTX I2V: {num_frames} frames (~{(num_frames - 1) / max(float(fps), 1):.1f}s) — "
                "long single-anchor clips tend to blur or warp on fast motion (distilled/q4). "
                "Try 4s (97 frames), soften the prompt, or use ltx-2.3-dev + negative prompt.",
            )
        log_parts = [
            f"LTX 2.3 MLX pipeline={mode}",
            f"mode={i2v}",
            f"size={width}x{height}",
            f"frames={num_frames}",
            f"fps={fps}",
            f"seed={seed}",
            f"stage1_steps={stage1_steps}",
            f"stage2_steps={stage2_steps}",
            f"bundle={self.bundle_root.name}",
        ]
        if image_path:
            anchor = float(getattr(self.config, "ltx_i2v_anchor_strength", 1.0) or 1.0)
            log_parts.append(f"i2v_anchor={anchor:.2f}")
        self._log(
            on_log,
            "info",
            " ".join(log_parts),
        )

        video_latent, audio_latent = self._generate_two_stage(
            prompt=prompt,
            width=int(width),
            height=int(height),
            num_frames=int(num_frames),
            fps=float(fps),
            seed=int(seed),
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            guidance=float(guidance),
            step_distill=bool(step_distill),
            image_path=image_path,
            on_log=on_log,
            on_progress=on_progress,
        )

        _materialize(self.ctx, video_latent, audio_latent)
        stage2_steps_int = int(getattr(self.config, "ltx_stage2_steps", 3) or 3)
        total_steps = max(1, stage1_steps + stage2_steps_int)
        emit_post_progress(on_progress, n_steps=total_steps, within_post=0.2)
        self._log(on_log, "info", f"LTX 2.3 decode+mux → {output_path}")

        def _decode_log(msg: str) -> None:
            self._log(on_log, "info", msg)

        return decode_ltx23_av_to_mp4(
            self.ctx,
            video_latent,
            audio_latent,
            output_path,
            self.bundle_root,
            frame_rate=float(fps),
            on_log=_decode_log,
        )
