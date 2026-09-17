"""Shared flow-matching training loss (min-SNR weighting, prior preservation)."""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx

CLASS_PRIOR_LATENT_COUNT = 16


def min_snr_weight(sigma: mx.array, gamma: float) -> mx.array:
    """Per-sample SNR weight for linear flow-matching with x_t = (1-σ)x0 + σε."""
    if gamma <= 0:
        return mx.ones_like(sigma)
    s = mx.reshape(sigma, (-1,) + (1,) * (sigma.ndim - 1 if sigma.ndim > 1 else 0))
    s = mx.clip(s, 1e-4, 1.0 - 1e-4)
    snr = mx.square((1.0 - s) / s)
    return mx.minimum(snr, gamma) / mx.maximum(snr, 1e-8)


def flow_match_mse(
    pred: mx.array,
    x0: mx.array,
    eps: mx.array,
    *,
    sigma: mx.array,
    min_snr_gamma: float = 0.0,
) -> mx.array:
    err = mx.square(pred + x0 - eps)
    if min_snr_gamma > 0:
        w = min_snr_weight(sigma, min_snr_gamma)
        while w.ndim < err.ndim:
            w = mx.expand_dims(w, axis=-1)
        return mx.mean(w * err)
    return mx.mean(err)


def sample_noisy_latent(x0: mx.array, ctx: Any) -> tuple[mx.array, mx.array, mx.array]:
    b = x0.shape[0]
    t = mx.random.uniform(shape=(b,), dtype=ctx.float32())
    eps = mx.random.normal(x0.shape, dtype=ctx.bfloat16())
    sigma = mx.reshape(t, (b,) + (1,) * (x0.ndim - 1)).astype(ctx.bfloat16())
    x_t = (1.0 - sigma) * x0 + sigma * eps
    x_t = mx.stop_gradient(x_t)
    return x_t, eps, t


def apply_static_sigma_shift(u: mx.array, shift: float) -> mx.array:
    """SD3 / Z-Image static sigma shift: ``shift*u / (1 + (shift-1)*u)``.

    Monotonic map on ``[0, 1]`` that pushes probability mass toward high σ. Mirrors the
    inference ``FlowMatchEulerScheduler`` static-``shift`` schedule so training samples the
    same σ distribution the model denoises at generation time.
    """
    s = float(shift)
    if s == 1.0:
        return u
    return s * u / (1.0 + (s - 1.0) * u)


def _apply_sigma_uniform_bias(u: mx.array, bias: str) -> mx.array:
    """Bias uniform draws before static shift (high → identity band, low → detail band)."""
    mode = (bias or "uniform").strip().lower()
    if mode == "low":
        return mx.square(u)
    if mode == "high":
        return mx.sqrt(u)
    return u


def sample_noisy_latent_shifted(
    x0: mx.array,
    ctx: Any,
    *,
    sigma_shift: float = 1.0,
    sigma_bias: str = "uniform",
) -> tuple[mx.array, mx.array, mx.array]:
    """Flow-match noising with a static sigma shift matching inference.

    Plain uniform σ sampling under-trains the high-σ (structure / identity) region that a
    shifted inference schedule spends most of its steps in, so trained LoRAs fail to bind
    identity (faces). Applying the same shift at training time concentrates supervision
    where inference actually denoises.
    """
    b = x0.shape[0]
    u = mx.random.uniform(shape=(b,), dtype=ctx.float32())
    u = _apply_sigma_uniform_bias(u, sigma_bias)
    t = apply_static_sigma_shift(u, sigma_shift) if float(sigma_shift) != 1.0 else u
    eps = mx.random.normal(x0.shape, dtype=ctx.bfloat16())
    sigma = mx.reshape(t, (b,) + (1,) * (x0.ndim - 1)).astype(ctx.bfloat16())
    x_t = (1.0 - sigma) * x0 + sigma * eps
    x_t = mx.stop_gradient(x_t)
    return x_t, eps, t


def turbo_training_sigmas(
    ctx: Any,
    *,
    infer_steps: int,
    width: int,
    height: int,
) -> mx.array:
    """Inference sigmas for Z-Image-Turbo (LinearScheduler + sigma shift)."""
    from backend.engine.common.ops.schedulers import LinearScheduler

    sched = LinearScheduler(num_train_timesteps=1000, ctx=ctx)
    sched.set_timesteps(
        int(infer_steps),
        image_width=int(width),
        image_height=int(height),
        requires_sigma_shift=True,
    )
    return sched._sigmas[:-1]


def turbo_sigma_shift_mu(width: int, height: int) -> float:
    """Resolution-dependent μ used by ``LinearScheduler(requires_sigma_shift=True)``."""
    sigma_max_shift, sigma_base_shift = 1.15, 0.5
    sigma_max_seq_len, sigma_base_seq_len = 4096, 256
    m = (sigma_max_shift - sigma_base_shift) / (sigma_max_seq_len - sigma_base_seq_len)
    b = sigma_base_shift - m * sigma_base_seq_len
    return float(m * int(width) * int(height) / 256 + b)


def turbo_band_unshifted_range(
    *,
    infer_steps: int,
    timestep_low: int,
    timestep_high: int,
) -> tuple[float, float]:
    """Continuous unshifted ``u`` range covered by inference steps ``[low, high]`` (1-indexed).

    ``LinearScheduler`` step ``i`` starts at ``s_i = 1 - (i-1)/N`` and denoises to ``s_{i+1}``,
    so the band spans ``[1 - high/N, 1 - (low-1)/N]``; ``[1, N]`` covers the whole ``(0, 1]``.
    """
    n = max(1, int(infer_steps))
    lo_step = max(1, min(int(timestep_low), n))
    hi_step = max(lo_step, min(int(timestep_high), n))
    u_lo = 1.0 - hi_step / n
    u_hi = 1.0 - (lo_step - 1) / n
    return float(u_lo), float(u_hi)


def sample_turbo_sigmas(
    ctx: Any,
    batch: int,
    *,
    infer_steps: int,
    timestep_low: int,
    timestep_high: int,
    width: int,
    height: int,
    timestep_bias: str = "uniform",
) -> mx.array:
    """Continuous σ over the Turbo inference band, shifted like the inference schedule.

    Training on the handful of discrete inference σ values (and only the low-σ half of them)
    leaves the high-σ steps that fix global layout / identity untrained, so faces are never
    memorized. Sampling the whole band continuously matches the ``weighted`` timestep
    distribution used by reference Turbo trainers and generalizes across resolutions.
    """
    u_lo, u_hi = turbo_band_unshifted_range(
        infer_steps=infer_steps, timestep_low=timestep_low, timestep_high=timestep_high
    )
    v = mx.random.uniform(shape=(int(batch),), dtype=ctx.float32())
    v = _apply_sigma_uniform_bias(v, timestep_bias)
    u = u_lo + v * (u_hi - u_lo)
    u = mx.clip(u, 1e-3, 1.0)
    mu = turbo_sigma_shift_mu(width, height)
    exp_mu = mx.exp(mx.array(mu, dtype=ctx.float32()))
    return exp_mu / (exp_mu + (1.0 / u - 1.0))


def sample_noisy_latent_turbo(
    x0: mx.array,
    ctx: Any,
    *,
    infer_steps: int,
    timestep_low: int,
    timestep_high: int,
    width: int,
    height: int,
    timestep_bias: str = "uniform",
) -> tuple[mx.array, mx.array, mx.array]:
    """Flow-match noising with σ drawn continuously from the Turbo inference band."""
    b = x0.shape[0]
    t = sample_turbo_sigmas(
        ctx,
        b,
        infer_steps=infer_steps,
        timestep_low=timestep_low,
        timestep_high=timestep_high,
        width=width,
        height=height,
        timestep_bias=timestep_bias,
    )
    eps = mx.random.normal(x0.shape, dtype=ctx.bfloat16())
    sigma = mx.reshape(t, (b,) + (1,) * (x0.ndim - 1)).astype(ctx.bfloat16())
    x_t = (1.0 - sigma) * x0 + sigma * eps
    x_t = mx.stop_gradient(x_t)
    return x_t, eps, t


def combine_instance_prior_loss(
    instance_loss: mx.array,
    prior_loss: mx.array | None,
    *,
    prior_loss_weight: float,
) -> mx.array:
    if prior_loss is None or prior_loss_weight <= 0:
        return instance_loss
    return instance_loss + float(prior_loss_weight) * prior_loss


def make_prior_latent(x0: mx.array, ctx: Any) -> mx.array:
    """Fallback prior x0: standard normal (used when class latents are unavailable)."""
    return mx.random.normal(x0.shape, dtype=ctx.bfloat16())


def sample_prior_latent(
    x0: mx.array,
    ctx: Any,
    *,
    prior_latents: mx.array | None = None,
) -> mx.array:
    """DreamBooth prior x0: sample from cached class latents when available."""
    if prior_latents is not None and int(prior_latents.shape[0]) > 0:
        import random

        n = int(prior_latents.shape[0])
        idx = random.randrange(n)
        latent = prior_latents[idx]
        if latent.ndim == 3:
            latent = latent[None]
        if latent.shape[0] != x0.shape[0]:
            latent = mx.broadcast_to(latent, x0.shape)
        return latent.astype(ctx.bfloat16())
    return make_prior_latent(x0, ctx)


def merge_prior_cache_tensors(
    cache: Any,
    tensors: dict[str, mx.array],
    *,
    name: str = "prior",
) -> None:
    try:
        existing = cache.load_prior(name=name)
    except RuntimeError:
        existing = {}
    merged = {**existing, **tensors}
    cache.write_prior(merged, name=name)


def wrap_loss_with_prior(
    instance_loss_fn: Callable[..., mx.array],
    *,
    prior_loss_fn: Callable[..., mx.array] | None,
    prior_loss_weight: float,
) -> Callable[..., mx.array]:
    if prior_loss_fn is None or prior_loss_weight <= 0:
        return instance_loss_fn

    def _loss(*args: Any, **kwargs: Any) -> mx.array:
        inst = instance_loss_fn(*args, **kwargs)
        prior = prior_loss_fn(*args, **kwargs)
        return combine_instance_prior_loss(inst, prior, prior_loss_weight=prior_loss_weight)

    return _loss
