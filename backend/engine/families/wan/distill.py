"""Wan 2.2 step-distill scheduler presets (LightX2V 4-step)."""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

# LightX2V ``wan_*_distill_4step_cfg.json`` — denoising_step_list (train-scale indices).
WAN_DISTILL_DEFAULT_TIMESTEPS: tuple[float, ...] = (1000.0, 750.0, 500.0, 250.0)
WAN_DISTILL_DEFAULT_SHIFT: float = 5.0
WAN_DISTILL_DEFAULT_BOUNDARY_STEP_INDEX: int = 2


def apply_adapter_prepare(
    config: Any,
    adapters: Sequence[Any],
    registry: Any,
    *,
    steps_default: int,
    on_log: Callable | None = None,
) -> tuple[bool, int]:
    """Mutate config when Lightning LoRA is selected. Returns (lightning_distill, steps_default)."""
    from backend.engine.families.wan.lora_mlx import adapters_include_wan_lightning

    if not adapters_include_wan_lightning(adapters, registry):
        return False, steps_default
    config.step_distill = True
    config.supports_guidance = False
    config.moe_boundary_step_index = WAN_DISTILL_DEFAULT_BOUNDARY_STEP_INDEX
    if getattr(config, "wan_distill_shift", None) is None:
        config.wan_distill_shift = WAN_DISTILL_DEFAULT_SHIFT
    if on_log is not None:
        on_log(
            "info",
            "Wan Lightning LoRA: 4-step distill schedule (no CFG, shift=5, MoE boundary=2)",
        )
    return True, steps_default


def _wan_distill_full_shifted_sigmas(num_train: int, shift: float) -> np.ndarray:
    """Full 1000-step flow sigmas with shift — matches LightX2V ``WanStepDistillScheduler``."""
    sigmas = np.linspace(1.0, 0.0, int(num_train) + 1, dtype=np.float64)[:-1]
    sh = float(shift)
    sigmas = sh * sigmas / (1.0 + (sh - 1.0) * sigmas)
    return sigmas.astype(np.float32)


def resolve_wan_distill_shift(config: Any) -> float:
    for key in ("wan_distill_shift", "shift"):
        val = getattr(config, key, None)
        if val is not None:
            return float(val)
    return WAN_DISTILL_DEFAULT_SHIFT


def resolve_wan_distill_timesteps(config: Any, steps: int) -> tuple[float, ...]:
    """Resolve denoising timesteps for Wan step-distill runs."""
    custom = getattr(config, "wan_distill_timesteps", None) or ()
    if custom:
        values = tuple(float(x) for x in custom)
    else:
        values = WAN_DISTILL_DEFAULT_TIMESTEPS
    if len(values) != int(steps):
        raise RuntimeError(
            f"Wan step-distill requires {len(values)} inference steps (got steps={steps}). "
            f"Use the registry default or set wan_distill_timesteps to match."
        )
    return values


def configure_wan_step_distill_timesteps(
    ctx: Any,
    scheduler: Any,
    steps: int,
    *,
    config: Any | None = None,
) -> Any:
    """Apply LightX2V Wan step-distill schedule (shift + index into full sigma table)."""
    cfg = config if config is not None else getattr(scheduler, "config", None)
    values = resolve_wan_distill_timesteps(cfg, steps)
    shift = resolve_wan_distill_shift(cfg) if cfg is not None else WAN_DISTILL_DEFAULT_SHIFT
    n_train = int(getattr(scheduler, "num_train_timesteps", 1000))
    full_sigmas = _wan_distill_full_shifted_sigmas(n_train, shift)
    indices = [int(n_train - float(t)) for t in values]
    picked = full_sigmas[indices]
    sigmas = np.append(picked, 0.0).astype(np.float32)
    timesteps = (picked * float(n_train)).astype(np.float32)

    scheduler._sigmas = sigmas
    scheduler._sigmas_float = sigmas.tolist()
    scheduler._timesteps = ctx.array(timesteps)
    scheduler._step_index = 0
    scheduler._num_steps = int(steps)
    scheduler._model_outputs = [None] * max(1, int(getattr(scheduler, "_solver_order", 1)))
    scheduler._lower_order_nums = 0
    scheduler._last_sample = None
    scheduler._this_order = 1
    # LightX2V ``WanStepDistillScheduler.step_post`` — simple flow Euler, not UniPC.
    scheduler._wan_distill_simple_step = True
    return scheduler._timesteps
