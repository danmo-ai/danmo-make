"""Pipeline-level CFG combiner + DMD distilled sigma schedule (Avatar 1.5)."""

from __future__ import annotations

import mlx.core as mx


def disentangled_cfg_combine(
    noise_pred_cond: mx.array,
    noise_pred_uncond_text: mx.array,
    noise_pred_uncond: mx.array,
    text_guidance_scale: float = 4.0,
    audio_guidance_scale: float = 4.0,
) -> mx.array:
    return (
        noise_pred_uncond
        + text_guidance_scale * (noise_pred_cond - noise_pred_uncond_text)
        + audio_guidance_scale * (noise_pred_uncond_text - noise_pred_uncond)
    )


def flip_velocity_for_scheduler(noise_pred: mx.array) -> mx.array:
    return -noise_pred


def get_dmd_distilled_sigmas(
    sampling_steps: int = 8,
    num_train_timesteps: int = 1000,
    num_distill_sample_steps: int = 8,
    model_type: str = "avatar-v1.5",
) -> mx.array:
    if model_type != "avatar-v1.5":
        raise NotImplementedError(
            f"Only model_type='avatar-v1.5' is implemented; got {model_type!r}."
        )
    step_size = num_train_timesteps // num_distill_sample_steps
    distill_idx = [round((i + 1) * step_size) for i in range(num_distill_sample_steps)]
    distill_idx = [num_train_timesteps - i for i in distill_idx]
    full = [(num_train_timesteps - 1 - i) / (num_train_timesteps - 1) for i in range(num_train_timesteps)]
    sigmas = [full[i] for i in distill_idx]
    sigmas = list(reversed(sigmas))
    del sampling_steps
    return mx.array(sigmas, dtype=mx.float32)


def cfg_split_outputs(noise_pred_2batch: mx.array) -> tuple[mx.array, mx.array]:
    uncond_text, cond = mx.split(noise_pred_2batch, 2, axis=0)
    return uncond_text, cond
