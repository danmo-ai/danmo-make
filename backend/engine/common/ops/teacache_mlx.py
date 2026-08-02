"""TeaCache — training-free denoise step skipping (MLX gate primitives).

Polynomial gate adapted from upstream TeaCache / mlx-teacache (Apache-2.0).
Coefficients for FLUX.1-dev and Z-Image base are vendored from mlx-teacache variants.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

GateKind = Literal["computed", "forced", "skipped", "numerical-miss"]

# Vendored from mlx-teacache variants/flux1_dev/config.py
FLUX1_TEACACHE_COEFFICIENTS: tuple[float, float, float, float, float] = (
    498.651651244,
    -283.781631,
    55.8554382,
    -3.82021401,
    0.264230861,
)
FLUX1_TEACACHE_DEFAULT_THRESH: float = 0.19  # calibrated flux1-dev 512² seed42 steps28 (~35% skip)

# Vendored from mlx-teacache variants/z_image_base/config.py
Z_IMAGE_TEACACHE_COEFFICIENTS: tuple[float, float, float, float, float] = (
    -898.9907628349583,
    367.7086118008557,
    -45.41511572598643,
    3.95114319842774,
    0.0,
)
Z_IMAGE_TEACACHE_DEFAULT_THRESH: float = 0.11  # calibrated z-image 512² seed42 steps28 (~35% skip)

# Vendored from ComfyUI-TeaCache SUPPORTED_MODELS_COEFFICIENTS["wan2.1_t2v_14B"]
WAN_TEACACHE_COEFFICIENTS: tuple[float, float, float, float, float] = (
    -5784.54975374,
    5449.50911966,
    -1811.16591783,
    256.27178429,
    -13.02252404,
)
WAN_TEACACHE_DEFAULT_THRESH: float = 0.20

# Vendored from ComfyUI-TeaCache SUPPORTED_MODELS_COEFFICIENTS["hunyuan_video"]
HUNYUAN_TEACACHE_COEFFICIENTS: tuple[float, float, float, float, float] = (
    733.226126,
    -401.131952,
    67.5869174,
    -3.149878,
    0.0961237896,
)
HUNYUAN_TEACACHE_DEFAULT_THRESH: float = 0.15

TEACACHE_FAMILY_DEFAULTS: dict[str, tuple[tuple[float, float, float, float, float], float]] = {
    "flux1": (FLUX1_TEACACHE_COEFFICIENTS, FLUX1_TEACACHE_DEFAULT_THRESH),
    "z_image": (Z_IMAGE_TEACACHE_COEFFICIENTS, Z_IMAGE_TEACACHE_DEFAULT_THRESH),
    "wan": (WAN_TEACACHE_COEFFICIENTS, WAN_TEACACHE_DEFAULT_THRESH),
    "hunyuan": (HUNYUAN_TEACACHE_COEFFICIENTS, HUNYUAN_TEACACHE_DEFAULT_THRESH),
}


@dataclass(frozen=True)
class TeaCacheGateDecision:
    kind: GateKind
    should_compute: bool
    should_update_cache: bool
    rel_l1: float | None
    predicted_distance: float | None
    accumulated_distance: float


@dataclass
class TeaCacheBranchLane:
    """Per-CFG-branch TeaCache lane (cond / uncond / default)."""

    previous_mod_input: mx.array | None = None
    cached_residual: mx.array | None = None
    accumulated_distance: float = 0.0


@dataclass
class TeaCacheState:
    step_counter: int = 0
    lanes: dict[str, TeaCacheBranchLane] | None = None
    num_steps: int | None = None
    computed_count: int = 0
    skipped_count: int = 0

    def lane(self, branch: str = "default") -> TeaCacheBranchLane:
        key = str(branch or "default")
        lanes = self.lanes
        if lanes is None:
            lanes = {}
            self.lanes = lanes
        if key not in lanes:
            lanes[key] = TeaCacheBranchLane()
        return lanes[key]

    def reset_for_new_generation(self, *, num_steps: int) -> None:
        self.step_counter = 0
        self.lanes = {}
        self.num_steps = num_steps
        self.computed_count = 0
        self.skipped_count = 0


def poly_eval(coeffs: tuple[float, float, float, float, float], x: float) -> float:
    c4, c3, c2, c1, c0 = coeffs
    return ((((c4 * x) + c3) * x + c2) * x + c1) * x + c0


def mean_abs_rel_l1(current: mx.array, previous: mx.array) -> float:
    num = float(mx.mean(mx.abs(current - previous)))
    denom = float(mx.mean(mx.abs(previous)))
    return num / max(denom, 1e-12)


def resolve_teacache_settings(
    family: str,
    mode: str | None,
    *,
    num_steps: int,
    user_thresh: float | None = None,
) -> tuple[bool, float, tuple[float, float, float, float, float], int, int]:
    """Return (enabled, thresh, coefficients, skip_first, skip_last)."""
    m = str(mode or "none").strip().lower()
    if m in ("", "none", "off", "false", "0"):
        return False, 0.0, FLUX1_TEACACHE_COEFFICIENTS, 1, 1
    if m == "auto":
        defaults = TEACACHE_FAMILY_DEFAULTS.get(family)
        if defaults is None:
            return False, 0.0, FLUX1_TEACACHE_COEFFICIENTS, 1, 1
        coeffs, thresh = defaults
        # Distilled short schedules rarely benefit; keep off unless explicit mode.
        if num_steps <= 8:
            return False, 0.0, coeffs, 1, 1
        return True, float(user_thresh if user_thresh is not None else thresh), coeffs, 1, 1
    if m in ("fast", "medium", "quality"):
        defaults = TEACACHE_FAMILY_DEFAULTS.get(family, (FLUX1_TEACACHE_COEFFICIENTS, FLUX1_TEACACHE_DEFAULT_THRESH))
        coeffs, base = defaults
        scale = {"quality": 0.85, "medium": 1.0, "fast": 1.25}[m]
        thresh = float(user_thresh if user_thresh is not None else base * scale)
        return True, thresh, coeffs, 1, 1
    if m == "on":
        defaults = TEACACHE_FAMILY_DEFAULTS.get(family, (FLUX1_TEACACHE_COEFFICIENTS, FLUX1_TEACACHE_DEFAULT_THRESH))
        coeffs, base = defaults
        return True, float(user_thresh if user_thresh is not None else base), coeffs, 1, 1
    raise RuntimeError(f"unknown teacache_mode={mode!r}")


def gate_step(
    state: TeaCacheState,
    *,
    rel_l1_thresh: float,
    coefficients: tuple[float, float, float, float, float],
    skip_first: int,
    skip_last: int,
    num_steps: int,
    step_idx: int,
    mod_in: mx.array,
    branch: str = "default",
) -> TeaCacheGateDecision:
    lane = state.lane(branch)
    if rel_l1_thresh <= 0.0:
        return TeaCacheGateDecision(
            kind="computed",
            should_compute=True,
            should_update_cache=False,
            rel_l1=None,
            predicted_distance=None,
            accumulated_distance=lane.accumulated_distance,
        )

    if step_idx < skip_first or step_idx >= num_steps - skip_last:
        return TeaCacheGateDecision(
            kind="forced",
            should_compute=True,
            should_update_cache=False,
            rel_l1=None,
            predicted_distance=None,
            accumulated_distance=lane.accumulated_distance,
        )

    if not bool(mx.all(mx.isfinite(mod_in))):
        return TeaCacheGateDecision(
            kind="numerical-miss",
            should_compute=True,
            should_update_cache=False,
            rel_l1=None,
            predicted_distance=None,
            accumulated_distance=lane.accumulated_distance,
        )

    if lane.previous_mod_input is None:
        return TeaCacheGateDecision(
            kind="computed",
            should_compute=True,
            should_update_cache=True,
            rel_l1=None,
            predicted_distance=None,
            accumulated_distance=lane.accumulated_distance,
        )

    rel_l1 = mean_abs_rel_l1(mod_in, lane.previous_mod_input)
    predicted = max(0.0, poly_eval(coefficients, rel_l1))
    new_acc = lane.accumulated_distance + predicted

    if new_acc < rel_l1_thresh:
        lane.accumulated_distance = new_acc
        return TeaCacheGateDecision(
            kind="skipped",
            should_compute=False,
            should_update_cache=False,
            rel_l1=rel_l1,
            predicted_distance=predicted,
            accumulated_distance=new_acc,
        )

    lane.accumulated_distance = 0.0
    return TeaCacheGateDecision(
        kind="computed",
        should_compute=True,
        should_update_cache=True,
        rel_l1=rel_l1,
        predicted_distance=predicted,
        accumulated_distance=0.0,
    )
