"""MiniMax-H3 rectified-flow Euler scheduler (PipeNetwork / diffusers parity)."""
from __future__ import annotations

import mlx.core as mx
import numpy as np


def linspace_1_to_0(n: int) -> np.ndarray:
    """``linspace(1, 0, n)`` in float32, bit-identical to ``torch.linspace``."""
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    start, end = 1.0, 0.0
    step = float(np.float32((end - start) / np.float32(n - 1)))
    half = n // 2
    i = np.arange(n, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    out[:half] = start + step * i[:half]
    out[half:] = end - step * (n - 1 - i[half:])
    return out.astype(np.float32)


class MiniMaxH3Scheduler:
    """Rectified-flow Euler (``eta = 0``) with exponential sigma shift."""

    order = 1

    def __init__(self, shift: float = 12.0) -> None:
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")
        self._shift = float(shift)
        self.sigmas: mx.array | None = None
        self.timesteps: mx.array | None = None
        self.num_inference_steps: int | None = None
        self._step_index: int | None = None

    @property
    def shift(self) -> float:
        return self._shift

    @property
    def step_index(self) -> int | None:
        return self._step_index

    def set_shift(self, shift: float) -> None:
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")
        self._shift = float(shift)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        sigmas: list[float] | mx.array | None = None,
    ) -> None:
        if sigmas is None:
            if num_inference_steps is None or num_inference_steps < 2:
                raise ValueError(
                    "`set_timesteps` requires either explicit `sigmas` or "
                    f"`num_inference_steps` >= 2, got {num_inference_steps}."
                )
            base = linspace_1_to_0(int(num_inference_steps))
            shift32 = np.float32(self._shift)
            shifted = (shift32 * base) / (np.float32(1.0) + np.float32(self._shift - 1.0) * base)
            values: list[float] = []
            for v in shifted.tolist():
                if not values or v != values[-1]:
                    values.append(v)
        else:
            values = [float(v) for v in (sigmas.tolist() if isinstance(sigmas, mx.array) else sigmas)]
            decreasing = all(b < a for a, b in zip(values, values[1:]))
            if len(values) < 2 or not decreasing or values[-1] != 0.0:
                raise ValueError(
                    "`sigmas` must hold at least two strictly decreasing values ending at 0.0."
                )

        self.sigmas = mx.array(values, dtype=mx.float32)
        self.timesteps = mx.array([1.0 - s for s in values[:-1]], dtype=mx.float32)
        self.num_inference_steps = len(values) - 1
        self._step_index = None

    def index_for_timestep(self, timestep: float) -> int:
        assert self.timesteps is not None
        target = float(timestep)
        for i, t in enumerate(self.timesteps.tolist()):
            if t == target:
                return i
        raise ValueError(
            "Passed `timestep` is not in `self.timesteps`. Use values from `scheduler.timesteps`."
        )

    def scale_noise(self, sample: mx.array, timestep: float, noise: mx.array) -> mx.array:
        """``x_t = t * x0 + (1 - t) * noise`` for conditioning anchors."""
        t32 = np.float32(timestep)
        return float(t32) * sample + float(np.float32(1.0) - t32) * noise

    def step(self, model_output: mx.array, timestep: float, sample: mx.array) -> mx.array:
        if isinstance(timestep, int):
            raise ValueError(
                "Passing integer indices as timesteps is not supported; pass one of the "
                "`scheduler.timesteps` values."
            )
        if self._step_index is None:
            self._step_index = self.index_for_timestep(timestep)

        assert self.sigmas is not None
        sigma_from_timestep = float(np.float32(1.0) - np.float32(timestep))
        denoised = sample + sigma_from_timestep * model_output

        sigma = np.float32(self.sigmas[self._step_index].item())
        sigma_next = np.float32(self.sigmas[self._step_index + 1].item())
        ratio = sigma_next / sigma
        one_minus_ratio = float(np.float32(1.0) - ratio)

        prev = float(ratio) * sample.astype(mx.float32) + one_minus_ratio * denoised.astype(mx.float32)
        self._step_index += 1
        return prev.astype(sample.dtype)
