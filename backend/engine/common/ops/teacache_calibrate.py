"""TeaCache calibration helpers — replay rel_l1 traces, suggest thresholds, fit coeffs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backend.engine.common.ops.teacache_mlx import TEACACHE_FAMILY_DEFAULTS, poly_eval


@dataclass(frozen=True)
class TeaCacheCalibrationReport:
    family: str
    num_steps: int
    rel_l1_count: int
    rel_l1_min: float
    rel_l1_max: float
    rel_l1_mean: float
    rel_l1_p50: float
    rel_l1_p90: float
    coefficients: tuple[float, float, float, float, float]
    default_thresh: float
    suggested_thresh: float
    target_skip_rate: float
    simulated_skip_rate: float
    threshold_sweep: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "num_steps": self.num_steps,
            "rel_l1_count": self.rel_l1_count,
            "rel_l1_min": self.rel_l1_min,
            "rel_l1_max": self.rel_l1_max,
            "rel_l1_mean": self.rel_l1_mean,
            "rel_l1_p50": self.rel_l1_p50,
            "rel_l1_p90": self.rel_l1_p90,
            "coefficients": list(self.coefficients),
            "default_thresh": self.default_thresh,
            "suggested_thresh": self.suggested_thresh,
            "target_skip_rate": self.target_skip_rate,
            "simulated_skip_rate": self.simulated_skip_rate,
            "threshold_sweep": [{"thresh": t, "skip_rate": r} for t, r in self.threshold_sweep],
        }


def format_coefficients_python(
    coeffs: tuple[float, float, float, float, float],
    *,
    name: str = "TEACACHE_COEFFICIENTS",
) -> str:
    c4, c3, c2, c1, c0 = coeffs
    return (
        f"{name}: tuple[float, float, float, float, float] = (\n"
        f"    {c4},\n"
        f"    {c3},\n"
        f"    {c2},\n"
        f"    {c1},\n"
        f"    {c0},\n"
        f")"
    )


def fit_teacache_polynomial(
    rel_l1_values: list[float],
    *,
    degree: int = 4,
) -> tuple[float, float, float, float, float]:
    """Fit ``poly_eval(coeffs, rel_l1) ≈ rel_l1`` (proxy when output deltas are unavailable)."""
    if len(rel_l1_values) < degree + 1:
        raise RuntimeError(
            f"Need at least {degree + 1} rel_l1 samples for degree-{degree} fit, got {len(rel_l1_values)}"
        )
    xs = np.asarray(rel_l1_values, dtype=np.float64)
    ys = xs.copy()
    deg = int(degree)
    poly = np.polyfit(xs, ys, deg)
    # np.polyfit returns highest degree first: c4*x^4 + ... + c0
    if deg != 4:
        raise RuntimeError("Only degree=4 is supported for TeaCache export")
    return (float(poly[0]), float(poly[1]), float(poly[2]), float(poly[3]), float(poly[4]))


def simulate_skip_rate(
    rel_l1_values: list[float],
    *,
    coefficients: tuple[float, float, float, float, float],
    thresh: float,
    num_steps: int,
    skip_first: int = 1,
    skip_last: int = 1,
) -> float:
    """Replay TeaCache gate on recorded rel_l1 series (one value per eligible transition)."""
    if num_steps <= 0:
        return 0.0
    eligible = max(0, num_steps - skip_first - skip_last)
    if eligible <= 0 or not rel_l1_values:
        return 0.0

    acc = 0.0
    skipped = 0
    # rel_l1_values align with steps skip_first .. num_steps-skip_last-1
    for step_idx in range(skip_first, num_steps - skip_last):
        rel_i = step_idx - skip_first
        if rel_i >= len(rel_l1_values):
            break
        rel_l1 = float(rel_l1_values[rel_i])
        predicted = max(0.0, poly_eval(coefficients, rel_l1))
        acc += predicted
        if acc < float(thresh):
            skipped += 1
        else:
            acc = 0.0
    used = min(len(rel_l1_values), eligible)
    return float(skipped) / float(max(used, 1))


def suggest_threshold(
    rel_l1_values: list[float],
    *,
    coefficients: tuple[float, float, float, float, float],
    num_steps: int,
    target_skip_rate: float = 0.35,
    skip_first: int = 1,
    skip_last: int = 1,
    thresh_min: float = 0.02,
    thresh_max: float = 0.80,
    steps: int = 40,
) -> tuple[float, float]:
    """Binary search threshold for target skip rate. Returns (thresh, achieved_rate)."""
    target = float(max(0.0, min(1.0, target_skip_rate)))
    lo, hi = float(thresh_min), float(thresh_max)
    best_t, best_r = lo, simulate_skip_rate(
        rel_l1_values,
        coefficients=coefficients,
        thresh=lo,
        num_steps=num_steps,
        skip_first=skip_first,
        skip_last=skip_last,
    )
    for _ in range(max(4, int(steps))):
        mid = (lo + hi) * 0.5
        rate = simulate_skip_rate(
            rel_l1_values,
            coefficients=coefficients,
            thresh=mid,
            num_steps=num_steps,
            skip_first=skip_first,
            skip_last=skip_last,
        )
        if abs(rate - target) < abs(best_r - target):
            best_t, best_r = mid, rate
        if rate < target:
            lo = mid
        else:
            hi = mid
    return best_t, best_r


def threshold_sweep(
    rel_l1_values: list[float],
    *,
    coefficients: tuple[float, float, float, float, float],
    num_steps: int,
    skip_first: int = 1,
    skip_last: int = 1,
    thresh_values: tuple[float, ...] | None = None,
) -> tuple[tuple[float, float], ...]:
    values = thresh_values or (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
    out: list[tuple[float, float]] = []
    for t in values:
        out.append(
            (
                float(t),
                simulate_skip_rate(
                    rel_l1_values,
                    coefficients=coefficients,
                    thresh=float(t),
                    num_steps=num_steps,
                    skip_first=skip_first,
                    skip_last=skip_last,
                ),
            )
        )
    return tuple(out)


def build_calibration_report(
    rel_l1_values: list[float],
    *,
    family: str,
    num_steps: int,
    coefficients: tuple[float, float, float, float, float] | None = None,
    default_thresh: float | None = None,
    target_skip_rate: float = 0.35,
    skip_first: int = 1,
    skip_last: int = 1,
    fit_coefficients: bool = False,
) -> TeaCacheCalibrationReport:
    if not rel_l1_values:
        raise RuntimeError("rel_l1 trace is empty — run probe mode generation first")

    defaults = TEACACHE_FAMILY_DEFAULTS.get(family)
    if coefficients is None:
        if fit_coefficients:
            coefficients = fit_teacache_polynomial(rel_l1_values)
        elif defaults is not None:
            coefficients = defaults[0]
        else:
            raise RuntimeError(f"No TeaCache coefficients for family={family!r}; pass --fit-coefficients")
    if default_thresh is None:
        default_thresh = defaults[1] if defaults is not None else 0.20

    arr = np.asarray(rel_l1_values, dtype=np.float64)
    suggested, sim_rate = suggest_threshold(
        rel_l1_values,
        coefficients=coefficients,
        num_steps=num_steps,
        target_skip_rate=target_skip_rate,
        skip_first=skip_first,
        skip_last=skip_last,
    )
    sweep = threshold_sweep(
        rel_l1_values,
        coefficients=coefficients,
        num_steps=num_steps,
        skip_first=skip_first,
        skip_last=skip_last,
    )
    return TeaCacheCalibrationReport(
        family=str(family),
        num_steps=int(num_steps),
        rel_l1_count=len(rel_l1_values),
        rel_l1_min=float(arr.min()),
        rel_l1_max=float(arr.max()),
        rel_l1_mean=float(arr.mean()),
        rel_l1_p50=float(np.percentile(arr, 50)),
        rel_l1_p90=float(np.percentile(arr, 90)),
        coefficients=coefficients,
        default_thresh=float(default_thresh),
        suggested_thresh=float(suggested),
        target_skip_rate=float(target_skip_rate),
        simulated_skip_rate=float(sim_rate),
        threshold_sweep=sweep,
    )


def write_trace_json(
    path: Path,
    *,
    family: str,
    num_steps: int,
    rel_l1: list[float],
    model: str = "",
    prompt: str = "",
) -> None:
    payload = {
        "family": family,
        "num_steps": num_steps,
        "model": model,
        "prompt": prompt,
        "rel_l1": [float(x) for x in rel_l1],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_trace_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid trace JSON (expected object): {path}")
    rel = data.get("rel_l1")
    if not isinstance(rel, list) or not rel:
        raise RuntimeError(f"Trace JSON missing non-empty rel_l1 list: {path}")
    return data


def resolve_family_from_model(model_id: str, *, project_root: Path | None = None) -> str:
    from backend.core.model_registry import ModelRegistry
    from backend.utils.path_utils import PathResolver

    resolver = PathResolver(project_root)
    reg = ModelRegistry.load(resolver.get_models_registry_path())
    return str(reg.require(model_id).family)


def publish_teacache_probe_from_model(model: Any) -> None:
    """Publish rel_l1 trace from the active DiT ``_step_cache`` (probe mode only)."""
    from backend.engine.common.ops.step_cache import find_step_cache_session

    cache = find_step_cache_session(model)
    if cache is not None and getattr(cache, "calibration_probe", False):
        cache.publish_probe_trace()


def teacache_probe_enabled() -> bool:
    import os

    return os.environ.get("DANQING_TEACACHE_PROBE", "").strip().lower() in ("1", "true", "yes", "on")


def print_calibration_report(report: TeaCacheCalibrationReport, *, fit_coefficients: bool = False) -> None:
    print(f"family={report.family} num_steps={report.num_steps} rel_l1_samples={report.rel_l1_count}")
    print(
        f"rel_l1: min={report.rel_l1_min:.6f} p50={report.rel_l1_p50:.6f} "
        f"p90={report.rel_l1_p90:.6f} max={report.rel_l1_max:.6f} mean={report.rel_l1_mean:.6f}"
    )
    print(f"default_thresh={report.default_thresh:.4f}")
    print(
        f"suggested_thresh={report.suggested_thresh:.4f} "
        f"(target_skip={report.target_skip_rate:.0%} simulated={report.simulated_skip_rate:.1%})"
    )
    print("threshold_sweep:")
    for t, r in report.threshold_sweep:
        print(f"  thresh={t:.3f}  skip_rate={r:.1%}")
    if fit_coefficients:
        print("\nSuggested coefficients (paste into teacache.py after validation):")
        print(format_coefficients_python(report.coefficients))
