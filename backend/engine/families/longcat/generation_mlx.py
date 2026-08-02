"""LongCat-Video MLX generation orchestration (T2V / I2V)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

from backend.engine.common.ops.schedulers import FlowMatchEulerScheduler
from backend.engine.config.model_configs import LongCatConfig
from backend.engine.families.longcat import bundle_load_mlx
from backend.engine.families.longcat.conditioning_mlx import (
    cfg_combine,
    cfg_split_outputs,
    flip_velocity_for_scheduler,
)
from backend.engine.pipelines.pipeline_progress import emit_denoise_progress, emit_post_progress
from backend.engine.runtime.mlx_runtime import run_eval


@dataclass
class _LongCatSamplerConfig:
    num_sampling_steps: int = 50
    num_train_timesteps: int = 1000
    scheduler_shift: float = 12.0
    text_guidance_scale: float = 5.0
    cfg_collapse: bool = False
    dit_in_channels: int = 16
    vae_scale_temporal: int = 4
    vae_scale_spatial: int = 8
    target_fps: int = 15


class LongCatMlxGenerator:
    def __init__(
        self,
        ctx: Any,
        bundle_root: Path,
        *,
        config: LongCatConfig | None = None,
        entry: Any | None = None,
        version_key: str | None = None,
    ) -> None:
        self.ctx = ctx
        self.bundle_root = Path(bundle_root)
        self.config = config or LongCatConfig()
        self.entry = entry
        self.version_key = version_key
        self._vae = None
        self._umt5 = None
        self._dit = None
        self._variant_dir: Path | None = None
        self._cfg_lora_merged = False

    @staticmethod
    def _log(on_log: Callable[[str, str], None] | None, level: str, msg: str) -> None:
        if on_log:
            on_log(level, msg)

    def load(self) -> None:
        if getattr(self.ctx, "backend", None) != "mlx":
            raise RuntimeError(
                f"LongCat-Video requires MLX runtime (got {getattr(self.ctx, 'backend', None)!r})"
            )
        self._vae, self._umt5, self._dit, self._variant_dir = bundle_load.load_longcat_components(
            self.bundle_root,
        )

    def _sampler_config(
        self,
        *,
        steps: int,
        guidance: float,
        cfg_step_lora: bool,
    ) -> _LongCatSamplerConfig:
        cfg = _LongCatSamplerConfig(
            num_sampling_steps=int(steps),
            scheduler_shift=float(getattr(self.config, "scheduler_shift", 12.0)),
            text_guidance_scale=float(guidance),
            target_fps=int(getattr(self.config, "default_fps", 15)),
        )
        if cfg_step_lora:
            cfg.cfg_collapse = True
            cfg.num_sampling_steps = max(1, int(getattr(self.config, "cfg_step_lora_steps", 8)))
            cfg.text_guidance_scale = 0.0
        return cfg

    def _make_scheduler(self, cfg: _LongCatSamplerConfig) -> FlowMatchEulerScheduler:
        return FlowMatchEulerScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            shift=cfg.scheduler_shift,
            ctx=self.ctx,
        )

    @staticmethod
    def _latent_shape(
        num_frames: int,
        height: int,
        width: int,
        *,
        cfg: _LongCatSamplerConfig,
    ) -> tuple[int, int, int]:
        v = cfg.vae_scale_temporal
        s = cfg.vae_scale_spatial
        t_lat = 1 + (num_frames - 1) // v
        h_lat = height // s
        w_lat = width // s
        return t_lat, h_lat, w_lat

    def _make_noise(
        self,
        num_frames: int,
        height: int,
        width: int,
        seed: int,
        *,
        cfg: _LongCatSamplerConfig,
    ) -> mx.array:
        t_lat, h_lat, w_lat = self._latent_shape(num_frames, height, width, cfg=cfg)
        mx.random.seed(int(seed))
        return mx.random.normal((1, cfg.dit_in_channels, t_lat, h_lat, w_lat))

    def _cfg_forward(
        self,
        latents: mx.array,
        timestep: mx.array,
        text_embeds_cat: mx.array,
        text_mask_cat: mx.array,
        *,
        cfg: _LongCatSamplerConfig,
        num_cond_latents: int = 0,
    ) -> mx.array:
        if cfg.cfg_collapse:
            if timestep.ndim == 0:
                timestep = timestep[None]
            pred = self._dit(
                latents,
                timestep,
                text_embeds_cat[1:2],
                encoder_attention_mask=text_mask_cat[1:2],
                num_cond_latents=num_cond_latents,
            )
            return flip_velocity_for_scheduler(pred)

        latents_2 = mx.concatenate([latents, latents], axis=0)
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep_2 = mx.repeat(timestep, 2, axis=0)
        pred_2 = self._dit(
            latents_2,
            timestep_2,
            text_embeds_cat,
            encoder_attention_mask=text_mask_cat,
            num_cond_latents=num_cond_latents,
        )
        noise_uncond, noise_cond = cfg_split_outputs(pred_2)
        combined = cfg_combine(
            noise_cond,
            noise_uncond,
            text_guidance_scale=cfg.text_guidance_scale,
        )
        return flip_velocity_for_scheduler(combined)

    def _denoise(
        self,
        latents: mx.array,
        text_embeds: mx.array,
        text_mask: mx.array,
        uncond_embeds: mx.array,
        uncond_mask: mx.array,
        *,
        cfg: _LongCatSamplerConfig,
        num_cond_latents: int = 0,
        on_progress: Callable[..., None] | None = None,
    ) -> mx.array:
        scheduler = self._make_scheduler(cfg)
        scheduler.set_timesteps(
            cfg.num_sampling_steps,
            use_empirical_mu=False,
            scheduler_shift=cfg.scheduler_shift,
        )
        timesteps = scheduler.timesteps
        n_steps = len(timesteps) if hasattr(timesteps, "__len__") else cfg.num_sampling_steps

        text_embeds_cat = mx.concatenate([uncond_embeds, text_embeds], axis=0)
        text_mask_cat = mx.concatenate([uncond_mask, text_mask], axis=0)

        for step_idx, t in enumerate(timesteps):
            t_scalar = float(t) if not hasattr(t, "item") else float(t.item())
            t_arr = mx.array([t_scalar], dtype=mx.float32)
            noise_pred = self._cfg_forward(
                latents,
                t_arr,
                text_embeds_cat,
                text_mask_cat,
                cfg=cfg,
                num_cond_latents=num_cond_latents,
            )
            latents = scheduler.step(noise_pred, step_idx, latents)
            emit_denoise_progress(on_progress, step_idx + 1, n_steps)

        return latents

    def _maybe_merge_cfg_step_lora(
        self,
        *,
        cfg_step_lora: bool,
        on_log: Callable[[str, str], None] | None,
    ) -> None:
        if not cfg_step_lora or self._cfg_lora_merged or self._dit is None or self._variant_dir is None:
            return
        bundle_load.merge_cfg_step_lora(self._dit, self._variant_dir, on_log=on_log)
        self._cfg_lora_merged = True

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
        on_log: Callable[[str, str], None] | None = None,
        on_progress: Callable[..., None] | None = None,
    ) -> str:
        if self._dit is None or self._vae is None or self._variant_dir is None:
            raise RuntimeError("LongCatMlxGenerator.load() must be called before generate_and_save")

        cfg_step_lora = bool(step_distill or getattr(self.config, "cfg_step_lora_default", False))
        mode = "i2v" if image_path else "t2v"
        eff_steps = int(steps)
        if cfg_step_lora and eff_steps >= 50:
            eff_steps = int(getattr(self.config, "cfg_step_lora_steps", 8))

        self._log(
            on_log,
            "info",
            " ".join(
                [
                    "LongCat-Video MLX",
                    f"mode={mode}",
                    f"size={width}x{height}",
                    f"frames={num_frames}",
                    f"fps={fps}",
                    f"seed={seed}",
                    f"steps={eff_steps}",
                    f"guidance={guidance}",
                    f"cfg_step_lora={cfg_step_lora}",
                    f"bundle={self.bundle_root.name}",
                ]
            ),
        )

        sampler_cfg = self._sampler_config(
            steps=eff_steps,
            guidance=guidance,
            cfg_step_lora=False,
        )
        if cfg_step_lora:
            self._maybe_merge_cfg_step_lora(cfg_step_lora=True, on_log=on_log)
            sampler_cfg = self._sampler_config(
                steps=eff_steps,
                guidance=guidance,
                cfg_step_lora=True,
            )

        text_embeds, text_mask, uncond_embeds, uncond_mask = bundle_load.encode_prompts(
            self._umt5,
            prompt,
            negative_prompt,
            self._variant_dir,
        )

        n_steps = max(1, sampler_cfg.num_sampling_steps)
        emit_denoise_progress(on_progress, 0, n_steps)

        if image_path:
            ref_image = bundle_load.load_reference_image(Path(image_path), height, width)
            ref_latent = self._vae.normalize_latents(self._vae.encode(ref_image))
            noise = self._make_noise(num_frames, height, width, seed, cfg=sampler_cfg)
            latents = mx.concatenate([ref_latent, noise], axis=2)
            latents = self._denoise(
                latents,
                text_embeds,
                text_mask,
                uncond_embeds,
                uncond_mask,
                cfg=sampler_cfg,
                num_cond_latents=1,
                on_progress=on_progress,
            )
            denoised = latents[:, :, 1:]
        else:
            latents = self._make_noise(num_frames, height, width, seed, cfg=sampler_cfg)
            denoised = self._denoise(
                latents,
                text_embeds,
                text_mask,
                uncond_embeds,
                uncond_mask,
                cfg=sampler_cfg,
                num_cond_latents=0,
                on_progress=on_progress,
            )

        z_denorm = self._vae.denormalize_latents(denoised)
        video = self._vae.decode(z_denorm)
        run_eval(getattr(self.ctx, "eval", None), video)
        emit_post_progress(on_progress, n_steps=n_steps, within_post=0.5)

        frames = bundle_load.video_tensor_to_uint8(video)
        out_fps = float(fps) if fps else float(sampler_cfg.target_fps)
        self._log(on_log, "info", f"LongCat-Video encode mp4 → {output_path}")
        return bundle_load.save_mp4(frames, output_path, fps=out_fps)
