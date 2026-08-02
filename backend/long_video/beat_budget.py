"""Beat-level duration guide helpers (soft totals; per-clip caps are hard)."""
from __future__ import annotations

import math
from typing import Any

from backend.long_video.constants import MIN_CLIP_SEC


def split_duration_parts(total_sec: float, *, max_clip_sec: float) -> list[float]:
    """Split a beat/segment duration into parts that each respect the I2V clip cap."""
    total = max(0.5, float(total_sec))
    cap = max(2.0, float(max_clip_sec))
    if total <= cap:
        return [round(total, 1)]
    n_parts = max(1, int(math.ceil(total / cap)))
    base = round(total / n_parts, 1)
    parts = [base] * (n_parts - 1)
    parts.append(round(total - base * (n_parts - 1), 1))
    return [max(2.0, p) for p in parts]


def group_shots_by_beat(shots: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, shot in enumerate(shots):
        gid = str(shot.get("segment_group_id") or f"beat_{i}")
        groups.setdefault(gid, []).append(i)
    return groups


def beat_duration_budgets_from_shots(shots: list[dict[str, Any]]) -> dict[str, float]:
    """Infer per-beat duration guide as sum of current segment durations (baseline)."""
    out: dict[str, float] = {}
    for gid, indices in group_shots_by_beat(shots).items():
        out[gid] = sum(float(shots[i].get("duration_sec") or 0) for i in indices)
    return out


def validate_beat_duration_sums(
    shots: list[dict[str, Any]],
    beat_budgets: dict[int, float],
    *,
    tolerance: float = 0.25,
) -> list[str]:
    """Soft warnings when a beat group's duration sum drifts from the initial guide."""
    issues: list[str] = []
    by_gid: dict[str, list[dict[str, Any]]] = {}
    for shot in shots:
        gid = str(shot.get("segment_group_id") or "")
        by_gid.setdefault(gid, []).append(shot)
    for beat_i, guide in beat_budgets.items():
        gid = f"beat_{beat_i}"
        group = by_gid.get(gid, [])
        if not group:
            continue
        total = sum(float(s.get("duration_sec") or 0) for s in group)
        g = float(guide)
        if g <= 0:
            continue
        if abs(total - g) / g > tolerance:
            issues.append(
                f"beat {beat_i}: subsegment sum {total:.1f}s drifts from guide {g:.1f}s "
                f"(story/flow priority — informational only)"
            )
    return issues
