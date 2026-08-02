from __future__ import annotations

"""SeedVR2 超分：单次运行时上下文 + 编排管线（Studio 唯一热路径）。

- **注册表**决定模型 id、动作等；**``ModelConfig``**（``weights_mlx``）描述 3B/7B bundle 元数据。
- **``SeedVR2UpscaleRuntime``** 只描述本次调用的尺寸、步数、guidance、调度器。
"""

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np
import PIL.Image

from backend.engine.common.bundle.bundle_weights import WeightApplier
from backend.engine.runtime.mlx_runtime import seeded_random_normal
from backend.engine.common.ops.scale_factor import ScaleFactor
from backend.engine.common.codecs.vae.mlx_tiling import TilingConfig, VAEUtil
from .dit_mlx import SeedVR2DiT
from .preprocess_mlx import SeedVR2LatentCreator, SeedVR2PositiveEmbeddings, SeedVR2Util
from .vae_mlx import SeedVR2VAE
from .weights_mlx import ModelConfig, load_flat_bundle, load_seedvr2_transformer_flat_checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job-local Euler scheduler (multi-step upscale; not interchangeable with common/schedulers)
# ---------------------------------------------------------------------------


class SeedVR2EulerScheduler:
    def __init__(self, config):
        self.config = config
        self.num_inference_steps = config.num_inference_steps
        self.num_train_timesteps = (
            config.num_train_steps if config.num_train_steps is not None else 1000
        )
        self.cfg_scale = config.guidance
        self.T = float(self.num_train_timesteps)
        self._timesteps, self._sigmas = self._compute_timesteps_and_sigmas()

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    def _compute_timesteps_and_sigmas(self) -> tuple[mx.array, mx.array]:
        timesteps_arr = mx.linspace(
            self.T, 0.0, self.num_inference_steps + 1, dtype=mx.float32
        )
        sigmas_arr = timesteps_arr / self.T
        return timesteps_arr, sigmas_arr

    def step(
        self,
        noise: mx.array,
        timestep: int,
        latents: mx.array,
        **kwargs,
    ) -> mx.array:
        model_output = noise
        sample = latents
        timestep_idx = timestep
        t = self._timesteps[timestep_idx]
        s = self._timesteps[timestep_idx + 1]
        t_norm = t / self.T
        s_norm = s / self.T
        pred_x_0 = sample - t_norm * model_output
        pred_noise = sample + (1 - t_norm) * model_output
        if s > 0:
            next_sample = (1 - s_norm) * pred_x_0 + s_norm * pred_noise
        else:
            next_sample = pred_x_0
        return next_sample


SCHEDULER_REGISTRY: dict[str, type] = {
    "seedvr2_euler": SeedVR2EulerScheduler,
    "SeedVR2EulerScheduler": SeedVR2EulerScheduler,
}


def try_import_external_scheduler(scheduler_object_path: str) -> None:
    raise RuntimeError(
        f"External scheduler {scheduler_object_path!r} is not supported for SeedVR2 in Danmo Make."
    )


@dataclass
class SeedVR2UpscaleRuntime:
    """对齐画幅后的单次推理上下文；调度器按 ``scheduler_key`` 惰性构造。"""

    model_config: ModelConfig
    height: int
    width: int
    num_inference_steps: int = 1
    guidance: float = 1.0
    image_path: Path | str | None = None
    scheduler_key: str = "seedvr2_euler"

    _scheduler: SeedVR2EulerScheduler | None = field(default=None, repr=False)
    _time_steps: range | None = field(default=None, repr=False)

    @classmethod
    def from_aligned_hw(
        cls,
        *,
        model_config: ModelConfig,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance: float,
        image_path: Path | str | None,
        scheduler_key: str = "seedvr2_euler",
    ) -> SeedVR2UpscaleRuntime:
        if width % 16 != 0 or height % 16 != 0:
            logger.warning("Width and height should be multiples of 16; rounding down.")
        h = 16 * (height // 16)
        w = 16 * (width // 16)
        return cls(
            model_config=model_config,
            height=h,
            width=w,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            image_path=image_path,
            scheduler_key=scheduler_key,
        )

    @property
    def precision(self) -> mx.Dtype:
        return ModelConfig.precision

    @property
    def num_train_steps(self) -> int | None:
        return self.model_config.num_train_steps

    @property
    def image_seq_len(self) -> int:
        return (self.height // 16) * (self.width // 16)

    @property
    def init_time_step(self) -> int:
        return 0

    @property
    def time_steps(self) -> range:
        if self._time_steps is None:
            self._time_steps = range(self.init_time_step, self.num_inference_steps)
        return self._time_steps

    @property
    def controlnet_strength(self) -> float | None:
        return None

    @property
    def scheduler(self) -> SeedVR2EulerScheduler:
        if self._scheduler is None:
            registered = SCHEDULER_REGISTRY.get(self.scheduler_key)
            if registered is None:
                raise NotImplementedError(
                    f"The scheduler {self.scheduler_key!r} is not available for SeedVR2. "
                    f"Supported: {sorted(SCHEDULER_REGISTRY.keys())}"
                )
            self._scheduler = registered(self)
            if hasattr(self._scheduler, "set_image_seq_len") and self.model_config.requires_sigma_shift:
                self._scheduler.set_image_seq_len(self.image_seq_len)
        return self._scheduler


class _UpscaleDenoiseCtx:
    __slots__ = ()

    def before_loop(self, latents, *, canny_image=None, depth_image=None) -> None:
        return None

    def in_loop(self, t: int, latents, time_steps=None) -> None:
        return None

    def after_loop(self, latents) -> None:
        return None


def _resolve_eval_fn(dit: Any) -> Callable[..., None]:
    fn = mx.eval
    dit_ctx = getattr(dit, "ctx", None)
    if dit_ctx is not None and hasattr(dit_ctx, "eval"):
        fn = dit_ctx.eval
    return fn


def _resolve_array_fn(dit: Any) -> Callable[..., Any]:
    fn = mx.array
    dit_ctx = getattr(dit, "ctx", None)
    if dit_ctx is not None and hasattr(dit_ctx, "array"):
        fn = dit_ctx.array
    return fn


def _resolve_seeded_randn_fn(dit: Any) -> Callable[..., Any] | None:
    dit_ctx = getattr(dit, "ctx", None)
    if dit_ctx is not None and hasattr(dit_ctx, "seeded_randn"):
        return dit_ctx.seeded_randn
    return None


class SeedVR2UpscalePipeline:
    """从 bundle 构建并执行 SeedVR2 超分。"""

    __slots__ = (
        "model_config",
        "vae",
        "dit",
        "tiling_config",
        "bits",
        "_bundle_path",
        "_vae_stream_session",
    )

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        vae: SeedVR2VAE,
        dit: SeedVR2DiT,
        tiling_config: TilingConfig,
        bits: int | None,
        bundle_path: str | Path | None = None,
    ) -> None:
        self.model_config = model_config
        self.vae = vae
        self.dit = dit
        self.tiling_config = tiling_config
        self.bits = bits
        self._bundle_path = Path(bundle_path) if bundle_path is not None else None
        self._vae_stream_session = None

    def configure_vae_runtime(
        self,
        *,
        stream_enabled: bool,
        conv3d_backend: str = "auto",
        on_log: Callable[[str, str], None] | None = None,
    ) -> None:
        from backend.engine.common.ops.mfa_seedvr2_mlx import (
            log_conv3d_backend,
            resolve_conv3d_backend,
        )
        from backend.engine.common.ops.vae_stream_cache_mlx import VAEStreamCacheSession

        backend = resolve_conv3d_backend(conv3d_backend)
        log_conv3d_backend(backend, on_log=on_log)
        session = VAEStreamCacheSession.from_plan(enabled=bool(stream_enabled))
        self._vae_stream_session = session
        self.vae.bind_vae_runtime(stream=session, conv3d_backend=backend)

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path,
        model_config: ModelConfig,
        *,
        quantize: int | None = None,
    ) -> SeedVR2UpscalePipeline:
        path = str(bundle_path)
        weights, weight_definition_cls = load_flat_bundle(path, model_config)
        vae = SeedVR2VAE()
        dit = SeedVR2DiT(**(model_config.transformer_overrides or {}))

        from backend.engine.common.bundle.bundle_weights.resolution import QuantizationResolution
        from backend.engine.common.bundle.quant_inference import WeightInferenceMode
        from backend.engine.runtime.mlx import MLXContext

        stored_q = weights.meta_data.quantization_level
        bits, warning = QuantizationResolution.resolve(stored=stored_q, requested=quantize)
        if warning:
            print(f"⚠️  {warning}")

        ctx = MLXContext()
        components = {c.name: c for c in weight_definition_cls.get_components()}

        if stored_q is not None:
            flat, file_q = load_seedvr2_transformer_flat_checkpoint(path, model_config)
            bundle_bits = stored_q if stored_q is not None else file_q
            inference_mode = WeightInferenceMode(kind="quantized", bits=int(bundle_bits))
            dit.load_weights(
                list(flat.items()),
                strict=False,
                ctx=ctx,
                bundle_affine_bits=bundle_bits,
                inference_mode=inference_mode,
                module_root=dit,
            )
            setattr(dit, "_dq_inference_mode", inference_mode)
            vae_weights = weights.components.get("vae")
            if vae_weights is not None:
                vae.update(vae_weights, strict=False)
        else:
            WeightApplier.set_weights(weights, {"transformer": dit, "vae": vae}, components)
            if bits is not None:
                dit.quantize_runtime(bits=bits, ctx=ctx)
                vae.quantize_runtime(bits=bits, ctx=ctx)

        dit.after_load_weights(bundle_root=path)
        return cls(
            model_config=model_config,
            vae=vae,
            dit=dit,
            tiling_config=TilingConfig(),
            bits=bits,
            bundle_path=path,
        )

    def generate_image(
        self,
        seed: int,
        image_path: str | Path,
        resolution: int | ScaleFactor,
        softness: float = 0.0,
    ) -> GeneratedImage:
        from backend.utils.image_metadata import MetadataReader

        processed_image, true_height, true_width = SeedVR2Util.preprocess_image(
            image_path=image_path,
            resolution=resolution,
            softness=softness,
        )

        runtime = SeedVR2UpscaleRuntime.from_aligned_hw(
            model_config=self.model_config,
            height=true_height,
            width=true_width,
            num_inference_steps=1,
            guidance=1.0,
            image_path=image_path,
            scheduler_key="seedvr2_euler",
        )

        initial_latent = VAEUtil.encode(
            vae=self.vae, image=processed_image, tiling_config=self.tiling_config
        )
        static_condition = SeedVR2LatentCreator.create_condition(encoded_latent=initial_latent)
        latents = SeedVR2LatentCreator.create_noise_latents(
            seed=seed,
            height=initial_latent.shape[-2],
            width=initial_latent.shape[-1],
            seeded_randn_fn=_resolve_seeded_randn_fn(self.dit),
        )

        txt_pos = SeedVR2PositiveEmbeddings.load(bundle_path=self._bundle_path)
        eval_fn = _resolve_eval_fn(self.dit)

        t0 = time.perf_counter()
        ctx = _UpscaleDenoiseCtx()
        ctx.before_loop(latents)

        for t in runtime.time_steps:
            model_input = mx.concatenate([latents, static_condition], axis=1)

            noise = self.dit(
                txt=txt_pos,
                vid=model_input,
                timestep=runtime.scheduler.timesteps[t],
            )

            latents = runtime.scheduler.step(noise=noise, timestep=t, latents=latents)

            ctx.in_loop(t, latents)

            eval_fn(latents)

        ctx.after_loop(latents)

        elapsed_s = time.perf_counter() - t0

        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)
        decoded = decoded[:, :, :true_height, :true_width]
        style = processed_image[:, :, :true_height, :true_width]
        decoded = SeedVR2Util.apply_color_correction(decoded, style)

        init_metadata = MetadataReader.read_all_metadata(image_path) if image_path else None

        return ImageUtil.to_image(
            seed=seed,
            prompt="",
            runtime=runtime,
            quantization=self.bits,
            decoded_latents=decoded,
            generation_time=elapsed_s,
            init_metadata=init_metadata,
        )


@dataclass
class GeneratedImage:
    image: PIL.Image.Image
    model_config: ModelConfig
    seed: int
    prompt: str
    steps: int
    guidance: float | None
    precision: mx.Dtype
    quantization: int
    generation_time: float
    lora_paths: list[str] | None = None
    lora_scales: list[float] | None = None
    height: int | None = None
    width: int | None = None
    controlnet_image_path: str | Path | None = None
    controlnet_strength: float | None = None
    image_path: str | Path | None = None
    image_paths: list[str] | list[Path] | None = None
    image_strength: float | None = None
    masked_image_path: str | Path | None = None
    depth_image_path: str | Path | None = None
    redux_image_paths: list[str] | list[Path] | None = None
    redux_image_strengths: list[float] | None = None
    concept_heatmap: Any = None
    negative_prompt: str | None = None
    init_metadata: dict | None = None


class ImageUtil:
    @staticmethod
    def to_image(
        *,
        seed: int,
        prompt: str,
        runtime: SeedVR2UpscaleRuntime,
        quantization: int,
        decoded_latents: mx.array,
        generation_time: float,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        controlnet_image_path: str | Path | None = None,
        image_path: str | Path | None = None,
        image_paths: list[str] | list[Path] | None = None,
        redux_image_paths: list[str] | list[Path] | None = None,
        redux_image_strengths: list[float] | None = None,
        image_strength: float | None = None,
        masked_image_path: str | Path | None = None,
        depth_image_path: str | Path | None = None,
        concept_heatmap: Any = None,
        negative_prompt: str | None = None,
        init_metadata: dict | None = None,
    ) -> GeneratedImage:
        normalized = ImageUtil._denormalize(decoded_latents)
        normalized_numpy = ImageUtil._to_numpy(normalized)
        image = ImageUtil._numpy_to_pil(normalized_numpy)
        return GeneratedImage(
            image=image,
            model_config=runtime.model_config,
            seed=seed,
            steps=runtime.num_inference_steps,
            prompt=prompt,
            guidance=runtime.guidance,
            precision=runtime.precision,
            quantization=quantization,
            generation_time=generation_time,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            height=runtime.height,
            width=runtime.width,
            image_path=image_path,
            image_paths=image_paths,
            image_strength=image_strength,
            controlnet_image_path=controlnet_image_path,
            controlnet_strength=runtime.controlnet_strength,
            masked_image_path=masked_image_path,
            depth_image_path=depth_image_path,
            redux_image_paths=redux_image_paths,
            redux_image_strengths=redux_image_strengths,
            concept_heatmap=concept_heatmap,
            negative_prompt=negative_prompt,
            init_metadata=init_metadata,
        )

    @staticmethod
    def _denormalize(images: mx.array) -> mx.array:
        return mx.clip((images / 2 + 0.5), 0, 1)

    @staticmethod
    def _to_numpy(images: mx.array) -> np.ndarray:
        if len(images.shape) == 5:
            images = mx.squeeze(images, axis=2)
        images = mx.transpose(images, (0, 2, 3, 1))
        images = mx.array.astype(images, mx.float32)
        return np.array(images)

    @staticmethod
    def _numpy_to_pil(images: np.ndarray) -> PIL.Image.Image:
        images = (images * 255).round().astype("uint8")
        pil_images = [PIL.Image.fromarray(image) for image in images]
        return pil_images[0]


# ---------------------------------------------------------------------------
# Spatiotemporal video restoration (3D VAE + MM-DiT; ffmpeg only decode/mux)
# ---------------------------------------------------------------------------


def _pad_stack_rgb_frames(
    frame_paths: list[Path],
    *,
    resolution: int | ScaleFactor,
    softness: float,
    array_fn: Callable[..., Any] = mx.array,
) -> tuple[mx.array, int, int, int, int]:
    """将若干帧对齐为同一空间尺寸后堆成 ``(1,3,T,H,W)``（[-1,1]）。

    返回 ``(volume, max_h, max_w, true_h0, true_w0)``：``true_*`` 为首帧内容尺寸，用于输出裁剪。
    """
    planes: list[mx.array] = []
    max_h = 0
    max_w = 0
    true_h0 = true_w0 = 0
    for i, p in enumerate(frame_paths):
        t4, th, tw = SeedVR2Util.preprocess_image(image_path=p, resolution=resolution, softness=softness)
        if i == 0:
            true_h0, true_w0 = int(th), int(tw)
        planes.append(t4[0])
        max_h = max(max_h, int(t4.shape[-2]))
        max_w = max(max_w, int(t4.shape[-1]))
    padded: list[mx.array] = []
    for x in planes:
        h, w = int(x.shape[-2]), int(x.shape[-1])
        canvas = mx.zeros((3, max_h, max_w), dtype=x.dtype)
        canvas = mx.slice_update(
            canvas,
            x,
            start_indices=array_fn([0, 0], dtype=mx.int32),
            axes=(1, 2),
        )
        padded.append(canvas)
    vol = mx.stack(padded, axis=0)
    vol = mx.transpose(vol, (1, 0, 2, 3))
    vol = vol[None, ...]
    return vol, max_h, max_w, true_h0, true_w0


def _decode_latents_to_frame_tensors(
    decoded: mx.array,
    *,
    true_h: int,
    true_w: int,
) -> list[mx.array]:
    """``decoded`` 为 ``(1,3,T,H,W)`` 或 ``(1,3,1,H,W)``，裁剪到内容尺寸后返回每帧 ``(3,h,w)``。"""
    if decoded.ndim == 4:
        decoded = decoded[:, :, None, :, :]
    _, _, t, _, _ = decoded.shape
    out: list[mx.array] = []
    for ti in range(int(t)):
        fr = decoded[:, :, ti : ti + 1, :, :]
        fr = fr[:, :, 0, :true_h, :true_w]
        out.append(fr[0])
    return out


def restore_video_chunk_spatiotemporal(
    *,
    pipeline: SeedVR2UpscalePipeline,
    frame_paths: list[Path],
    resolution: int | ScaleFactor,
    softness: float,
    seed: int,
    bundle_path: Path | None,
) -> list[mx.array]:
    """对一段连续帧做 SeedVR2 视频修复，返回每帧 RGB ``(3,h,w)`` 张量列表（[-1,1]）。"""
    if not frame_paths:
        return []
    eval_fn = _resolve_eval_fn(pipeline.dit)
    array_fn = _resolve_array_fn(pipeline.dit)
    seeded_randn_fn = _resolve_seeded_randn_fn(pipeline.dit)

    processed, max_h, max_w, true_h, true_w = _pad_stack_rgb_frames(
        frame_paths,
        resolution=resolution,
        softness=softness,
        array_fn=array_fn,
    )

    runtime = SeedVR2UpscaleRuntime.from_aligned_hw(
        model_config=pipeline.model_config,
        height=max_h,
        width=max_w,
        num_inference_steps=1,
        guidance=1.0,
        image_path=None,
        scheduler_key="seedvr2_euler",
    )

    initial_latent = VAEUtil.encode(vae=pipeline.vae, image=processed, tiling_config=pipeline.tiling_config)
    eval_fn(initial_latent)

    static_condition = SeedVR2LatentCreator.create_condition(encoded_latent=initial_latent)
    t_lat = int(initial_latent.shape[2])
    h_lat = int(initial_latent.shape[-2])
    w_lat = int(initial_latent.shape[-1])
    latents = seeded_random_normal(
        seeded_randn_fn,
        (1, 16, t_lat, h_lat, w_lat),
        int(seed) & 0x7FFFFFFF,
    )
    txt_pos = SeedVR2PositiveEmbeddings.load(bundle_path=bundle_path)

    ctx = _UpscaleDenoiseCtx()
    ctx.before_loop(latents)

    for t in runtime.time_steps:
        model_input = mx.concatenate([latents, static_condition], axis=1)
        noise = pipeline.dit(
            txt=txt_pos,
            vid=model_input,
            timestep=runtime.scheduler.timesteps[t],
        )
        latents = runtime.scheduler.step(noise=noise, timestep=t, latents=latents)
        ctx.in_loop(t, latents)
        eval_fn(latents)

    ctx.after_loop(latents)

    decoded = VAEUtil.decode(vae=pipeline.vae, latent=latents, tiling_config=pipeline.tiling_config)
    eval_fn(decoded)
    if decoded.ndim == 4:
        decoded = decoded[:, :, None, :, :]

    t_in = int(processed.shape[2])
    t_dec = int(decoded.shape[2])
    if t_dec < t_in:
        raise RuntimeError(
            f"SeedVR2 VAE decode returned {t_dec} temporal slice(s) for {t_in} input frame(s); "
            "cannot align restored chunk."
        )
    if t_dec > t_in:
        decoded = decoded[:, :, :t_in, :, :]

    decoded = decoded[:, :, :, :true_h, :true_w]
    style = processed[:, :, :t_in, :true_h, :true_w]
    corrected_slices: list[mx.array] = []
    for ti in range(t_in):
        d4 = decoded[:, :, ti, :, :]
        s4 = style[:, :, ti, :, :]
        corrected_slices.append(SeedVR2Util.apply_color_correction(d4, s4))
    decoded = mx.stack(corrected_slices, axis=2)

    return _decode_latents_to_frame_tensors(decoded, true_h=true_h, true_w=true_w)


def run_seedvr2_spatiotemporal_video(
    *,
    pipeline: SeedVR2UpscalePipeline,
    frames_dir: Path,
    n_frames: int,
    resolution: ScaleFactor,
    softness: float,
    seed_base: int,
    frames_out_dir: Path,
    png_pattern_name: str,
    chunk_frames: int,
    on_log: Callable[[str, str], None] | None,
    on_progress: Callable[[float, int, int], None] | None,
    is_cancelled: Callable[[], bool] | None,
) -> None:
    """读取 ``frames_dir/frame_%06d.png``，按 ``chunk_frames`` 做视频修复并写出 ``frames_out_dir/{pattern}%06d.png``。"""
    if chunk_frames < 4:
        raise RuntimeError("SeedVR2 video restoration requires chunk_frames >= 4")

    paths = [frames_dir / f"frame_{i:06d}.png" for i in range(1, n_frames + 1)]
    for p in paths:
        if not p.is_file():
            raise RuntimeError(f"Missing decoded frame: {p}")

    frames_out_dir.mkdir(parents=True, exist_ok=True)
    out_idx = 0
    bundle = pipeline._bundle_path

    from backend.engine.config.model_configs import SeedVR2Config

    sr_cfg = SeedVR2Config()
    pipeline.configure_vae_runtime(
        stream_enabled=bool(sr_cfg.vae_stream_cache),
        conv3d_backend=str(sr_cfg.conv3d_backend),
        on_log=on_log,
    )
    pipeline.vae.reset_vae_stream()

    for start in range(0, n_frames, chunk_frames):
        if is_cancelled and is_cancelled():
            return
        end = min(start + chunk_frames, n_frames)
        logical_paths = paths[start:end]
        infer_paths = list(logical_paths)
        while len(infer_paths) < 4:
            infer_paths.append(infer_paths[-1])
        sd = (int(seed_base) + start * 1009) & 0x7FFFFFFF
        t0 = time.perf_counter()
        frames_rgb = restore_video_chunk_spatiotemporal(
            pipeline=pipeline,
            frame_paths=[Path(x) for x in infer_paths],
            resolution=resolution,
            softness=softness,
            seed=sd,
            bundle_path=bundle,
        )
        frames_rgb = frames_rgb[: len(logical_paths)]
        if len(frames_rgb) != len(logical_paths):
            raise RuntimeError(
                f"SeedVR2 video restoration chunk produced {len(frames_rgb)} frame(s) for "
                f"{len(logical_paths)} input frame(s); VAE temporal length mismatch."
            )
        if on_log:
            on_log(
                "info",
                f"seedvr2_video_restoration chunk start={start} end={end} "
                f"frames={len(frames_rgb)} elapsed_s={time.perf_counter() - t0:.2f}",
            )
        for fr in frames_rgb:
            out_idx += 1
            arr = np.array(fr, dtype=np.float32)
            arr = (arr + 1.0) * 0.5
            arr = np.clip(arr, 0.0, 1.0)
            arr = np.transpose(arr, (1, 2, 0))
            rgb8 = (arr * 255.0).round().astype(np.uint8)
            im = PIL.Image.fromarray(rgb8, mode="RGB")
            im.save(str(frames_out_dir / f"{png_pattern_name}_{out_idx:06d}.png"))
        if on_progress:
            on_progress(end / max(n_frames, 1), end, n_frames)

    pipeline.vae.reset_vae_stream()


def expected_seedvr2_weight_files(model_key: str) -> tuple[str, ...]:
    if "7b" in model_key.lower():
        return ("seedvr2_ema_7b_fp16.safetensors", "ema_vae_fp16.safetensors")
    return ("seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors")


def validate_seedvr2_bundle(bundle_path: Path, model_key: str) -> None:
    missing = [n for n in expected_seedvr2_weight_files(model_key) if not (bundle_path / n).is_file()]
    if missing:
        raise RuntimeError(
            f"SeedVR2 bundle at {bundle_path} is missing weight file(s): {missing}. "
            "Expected flat directory with `ema_vae_fp16.safetensors` plus "
            "`seedvr2_ema_7b_fp16.safetensors` or `seedvr2_ema_3b_fp16.safetensors` "
            "(see registry `local_path`, e.g. models/Upscaler/seedvr2-7b-fp16)."
        )


def load_seedvr2_upscale_pipeline(
    *,
    bundle_path: Path,
    model_key: str,
    model_cache: Any | None = None,
    cache_key: str | None = None,
    cache_size_gb: float | None = None,
    on_log: Callable[[str, str], None] | None = None,
) -> SeedVR2UpscalePipeline:
    """Load or reuse cached SeedVR2 upscale pipeline (``ImageUpscalePipeline`` registry path)."""
    if model_cache is not None and cache_key:
        cached = model_cache.get(cache_key)
        if cached is not None:
            return cached

    if "7b" in model_key.lower():
        model_config = ModelConfig.seedvr2_7b()
    else:
        model_config = ModelConfig.seedvr2_3b()

    pipeline = SeedVR2UpscalePipeline.from_bundle(bundle_path, model_config)
    if model_cache is not None and cache_key and cache_size_gb is not None:
        model_cache.put(cache_key, pipeline, cache_size_gb)
    return pipeline


def run_seedvr2_upscale(
    *,
    bundle_path: Path,
    model_key: str,
    source_image: Path,
    scale: int,
    softness: float,
    seed: int | None,
    output_png: Path,
    on_log: Callable[[str, str], None] | None = None,
    pipeline: SeedVR2UpscalePipeline | None = None,
) -> dict[str, Any]:
    """执行 SeedVR2 超分并写出 PNG。由 ``ImageUpscalePipeline`` 在 MLX 路径下调用。"""
    validate_seedvr2_bundle(bundle_path, model_key)

    if scale not in (2, 4):
        raise RuntimeError(f"SeedVR2 upscale scale must be 2 or 4, got {scale!r}")
    if not source_image.is_file():
        raise RuntimeError(f"SeedVR2 upscale source image not found: {source_image}")

    if "7b" in model_key.lower():
        model_config = ModelConfig.seedvr2_7b()
    else:
        model_config = ModelConfig.seedvr2_3b()

    if pipeline is None:
        pipeline = SeedVR2UpscalePipeline.from_bundle(bundle_path, model_config)
    resolution = ScaleFactor.parse(f"{int(scale)}x")
    soft = max(0.0, min(1.0, float(softness)))
    sd = int(seed) if seed is not None else random.randint(0, 2 ** 31 - 1)

    if on_log:
        on_log(
            "info",
            " ".join(
                [
                    "seedvr2_upscale backend=backend.engine.families.seedvr2.stem_mlx",
                    f"bundle={bundle_path}",
                    f"model_key={model_key}",
                    f"resolution={resolution}",
                    f"softness={soft}",
                    f"seed={sd}",
                ]
            ),
        )

    generated = pipeline.generate_image(
        seed=sd,
        image_path=source_image,
        resolution=resolution,
        softness=soft,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    generated.image.save(str(output_png))

    return {
        "upscale_backend": "backend.engine.families.seedvr2.stem_mlx",
        "seed": sd,
        "softness": soft,
        "scale": int(scale),
        "reference_model_name": getattr(model_config, "model_name", ""),
    }
