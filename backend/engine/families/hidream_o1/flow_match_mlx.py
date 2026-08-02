"""MLX port of FlashFlowMatchEulerDiscreteScheduler from HiDream-O1.

Reference: HiDream-ai/HiDream-O1-Image @ models/flash_scheduler.py.
Trimmed to the path the Dev recipe actually uses:
  - num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False
  - timesteps overridden by DEFAULT_TIMESTEPS after construction
  - karras/exponential/beta sigmas not used
  - step() with s_churn/s_tmin/s_tmax stripped (always defaults)

The math is verbatim from upstream — only the framework swap (torch -> mlx).
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np


# Verbatim from HiDream-O1 models/pipeline.py
DEFAULT_TIMESTEPS = [
    999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764, 737,
    707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
]


class FlashFlowMatchScheduler:
    """Euler scheduler for flow matching, with optional noise injection."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 1.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

        sigmas = np.linspace(1.0, 1.0 / num_train_timesteps, num_train_timesteps, dtype=np.float32)
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
        self.sigmas_np = sigmas
        self.timesteps_np = sigmas * num_train_timesteps

        self.num_inference_steps: int | None = None
        self._step_index: int | None = None

    def set_timesteps(self, num_inference_steps: int, custom_timesteps: list[int] | None = None):
        if custom_timesteps is not None:
            timesteps = np.asarray(custom_timesteps, dtype=np.float32)
            sigmas = (timesteps / self.num_train_timesteps).astype(np.float32)
            sigmas = np.append(sigmas, 0.0).astype(np.float32)
        else:
            timesteps = np.linspace(self.num_train_timesteps, 1.0, num_inference_steps, dtype=np.float32)
            sigmas = (timesteps / self.num_train_timesteps).astype(np.float32)
            sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)
            sigmas = np.append(sigmas, 0.0).astype(np.float32)

        self.num_inference_steps = len(timesteps)
        self.timesteps_np = timesteps
        self.sigmas_np = sigmas
        self._step_index = None

    @property
    def timesteps(self) -> mx.array:
        return mx.array(self.timesteps_np)

    @property
    def sigmas(self) -> mx.array:
        return mx.array(self.sigmas_np)

    def _init_step_index(self, timestep_value: float):
        ts = self.timesteps_np
        matches = np.where(np.isclose(ts, timestep_value, atol=1e-3))[0]
        if len(matches) == 0:
            raise ValueError(f"timestep {timestep_value!r} not in scheduler.timesteps")
        self._step_index = int(matches[1] if len(matches) > 1 else matches[0])

    def step(self, model_output, timestep, sample,
             s_noise=1.0, noise_clip_std=0.0, seed=None):
        if self._step_index is None:
            self._init_step_index(float(timestep))
        idx = self._step_index

        sigma = float(self.sigmas_np[idx])
        sigma_next = float(self.sigmas_np[idx + 1])

        sample_f = sample.astype(mx.float32)
        model_output_f = model_output.astype(mx.float32)

        denoised = sample_f - model_output_f * sigma

        if idx < self.num_inference_steps:
            if seed is not None:
                key = mx.random.key(seed + idx)
                noise = mx.random.normal(model_output_f.shape, key=key)
            else:
                noise = mx.random.normal(model_output_f.shape)

            if noise_clip_std > 0:
                std = float(mx.std(noise))
                clip = noise_clip_std * std
                noise = mx.clip(noise, -clip, clip)

            new_sample = sigma_next * noise * s_noise + (1.0 - sigma_next) * denoised
        else:
            new_sample = denoised

        self._step_index += 1
        return new_sample.astype(sample.dtype)
