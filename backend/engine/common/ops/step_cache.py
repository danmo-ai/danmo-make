"""Step-level cache primitives (TeaCache gate) — optimization layer above DiT families."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.engine.common.ops.teacache_mlx import (
    TeaCacheGateDecision,
    TeaCacheState,
    gate_step,
    resolve_teacache_settings,
)


@dataclass
class StepCacheSession:
    """Per-generation TeaCache session — gate-only; family owns residual tensors."""

    family: str
    enabled: bool
    thresh: float
    coefficients: tuple[float, float, float, float, float]
    skip_first: int
    skip_last: int
    num_steps: int
    state: TeaCacheState | None = None
    calibration_probe: bool = False
    calibration_trace: list[float] | None = None
    _probe_published: bool = False
    _probe_recorded_step: int = -1

    _last_probe_trace: list[float] | None = None

    @classmethod
    def consume_probe_trace(cls) -> list[float]:
        trace = cls._last_probe_trace
        cls._last_probe_trace = None
        return list(trace or [])

    @classmethod
    def configure_probe(
        cls,
        *,
        family: str,
        num_steps: int,
    ) -> StepCacheSession:
        """Record rel_l1 each step without skipping — for ``calibrate_teacache.py run``."""
        defaults = resolve_teacache_settings(family, "on", num_steps=num_steps)
        _, _, coeffs, skip_first, skip_last = defaults
        state = TeaCacheState()
        state.reset_for_new_generation(num_steps=num_steps)
        return cls(
            family=family,
            enabled=True,
            thresh=1e9,
            coefficients=coeffs,
            skip_first=skip_first,
            skip_last=skip_last,
            num_steps=num_steps,
            state=state,
            calibration_probe=True,
            calibration_trace=[],
        )

    def publish_probe_trace(self) -> None:
        if not self.calibration_probe or self._probe_published:
            return
        StepCacheSession._last_probe_trace = list(self.calibration_trace or [])
        self._probe_published = True

    @classmethod
    def configure(
        cls,
        *,
        family: str,
        mode: str | None,
        num_steps: int,
        user_thresh: float | None = None,
    ) -> StepCacheSession:
        enabled, thresh, coeffs, skip_first, skip_last = resolve_teacache_settings(
            family,
            mode,
            num_steps=num_steps,
            user_thresh=user_thresh,
        )
        state = TeaCacheState() if enabled else None
        if state is not None:
            state.reset_for_new_generation(num_steps=num_steps)
        return cls(
            family=family,
            enabled=enabled,
            thresh=thresh,
            coefficients=coeffs,
            skip_first=skip_first,
            skip_last=skip_last,
            num_steps=num_steps,
            state=state,
        )

    def reset(self, *, num_steps: int | None = None) -> None:
        if num_steps is not None:
            self.num_steps = num_steps
        if self.state is None:
            return
        self.state.reset_for_new_generation(num_steps=self.num_steps)

    @property
    def step_counter(self) -> int:
        if self.state is None:
            return 0
        return int(self.state.step_counter)

    def set_step_counter(self, step_idx: int) -> None:
        if self.state is not None:
            self.state.step_counter = int(step_idx)

    def gate(
        self,
        mod_in: Any,
        *,
        step_idx: int | None = None,
        branch: str = "default",
    ) -> TeaCacheGateDecision | None:
        if not self.enabled or self.state is None or self.num_steps <= 0:
            return None
        lane_key = str(branch or "default")
        idx = int(self.state.step_counter if step_idx is None else step_idx)
        if self.calibration_probe:
            from backend.engine.common.ops.teacache_mlx import mean_abs_rel_l1

            lane = self.state.lane(lane_key)
            decision = TeaCacheGateDecision(
                kind="computed",
                should_compute=True,
                should_update_cache=True,
                rel_l1=None,
                predicted_distance=None,
                accumulated_distance=lane.accumulated_distance,
            )
            if idx >= self.skip_first and idx < self.num_steps - self.skip_last:
                if idx == self._probe_recorded_step:
                    return decision
                if lane.previous_mod_input is not None and self.calibration_trace is not None:
                    rel_l1 = mean_abs_rel_l1(mod_in, lane.previous_mod_input)
                    self.calibration_trace.append(float(rel_l1))
                    decision = TeaCacheGateDecision(
                        kind="computed",
                        should_compute=True,
                        should_update_cache=True,
                        rel_l1=rel_l1,
                        predicted_distance=None,
                        accumulated_distance=lane.accumulated_distance,
                    )
                self._probe_recorded_step = idx
            lane.previous_mod_input = mod_in
            self.state.computed_count += 1
            return decision
        decision = gate_step(
            self.state,
            rel_l1_thresh=self.thresh,
            coefficients=self.coefficients,
            skip_first=self.skip_first,
            skip_last=self.skip_last,
            num_steps=self.num_steps,
            step_idx=idx,
            mod_in=mod_in,
            branch=lane_key,
        )
        if decision.should_compute:
            self.state.computed_count += 1
        elif not decision.should_compute:
            self.state.skipped_count += 1
        return decision

    def store_branch_mod_input(self, branch: str, mod_in: Any) -> None:
        if self.state is None:
            return
        self.state.lane(branch).previous_mod_input = mod_in

    def store_branch_residual(self, branch: str, residual: Any) -> None:
        if self.state is None:
            return
        self.state.lane(branch).cached_residual = residual

    def branch_cached_residual(self, branch: str) -> Any:
        if self.state is None:
            return None
        return self.state.lane(branch).cached_residual

    def should_skip(
        self,
        decision: TeaCacheGateDecision | None,
        *,
        branch: str = "default",
        has_residual: bool | None = None,
    ) -> bool:
        if has_residual is not None:
            return bool(
                decision is not None
                and not decision.should_compute
                and has_residual
            )
        residual = self.branch_cached_residual(branch)
        return bool(
            decision is not None
            and not decision.should_compute
            and residual is not None
        )


def find_step_cache_session(model: Any) -> StepCacheSession | None:
    """Walk DiT stem / inner impl for the active ``StepCacheSession``."""
    seen: set[int] = set()
    stack: list[tuple[Any, int]] = [(model, 0)]
    max_depth = 8
    while stack:
        obj, depth = stack.pop()
        if obj is None or depth > max_depth:
            continue
        # unittest.mock synthesizes unbounded child graphs via getattr.
        if (type(obj).__module__ or "").startswith("unittest.mock"):
            continue
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        cache = getattr(obj, "_step_cache", None)
        if isinstance(cache, StepCacheSession):
            return cache
        for attr in ("_inner", "impl", "_impl", "model"):
            child = getattr(obj, attr, None)
            if child is None or isinstance(child, (str, int, float, bool, bytes)):
                continue
            if (type(child).__module__ or "").startswith("unittest.mock"):
                continue
            stack.append((child, depth + 1))
    return None


def format_step_cache_summary(session: StepCacheSession | None) -> str | None:
    if session is None or not session.enabled or session.calibration_probe:
        return None
    state = session.state
    if state is None:
        return None
    total = int(state.computed_count) + int(state.skipped_count)
    if total <= 0:
        return None
    skip_pct = 100.0 * float(state.skipped_count) / float(total)
    return (
        f"TeaCache skipped {state.skipped_count}/{total} forwards ({skip_pct:.0f}%), "
        f"thresh={session.thresh:.3f}"
    )


def log_step_cache_summary(model: Any, on_log: Any | None) -> None:
    if on_log is None:
        return
    msg = format_step_cache_summary(find_step_cache_session(model))
    if msg:
        on_log("info", msg)


def collect_step_cache_stats(model: Any) -> dict[str, Any]:
    session = find_step_cache_session(model)
    if session is None or not session.enabled or session.calibration_probe:
        return {}
    state = session.state
    if state is None:
        return {}
    total = int(state.computed_count) + int(state.skipped_count)
    if total <= 0:
        return {}
    skipped = int(state.skipped_count)
    return {
        "teacache_skipped": skipped,
        "teacache_computed": int(state.computed_count),
        "teacache_skip_rate": round(float(skipped) / float(total), 4),
    }
