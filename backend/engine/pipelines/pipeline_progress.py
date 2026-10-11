"""Shared progress + graph-step helpers for image/video pipelines."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

# Denoise uses most of the bar; VAE decode + save use the tail (queue ETA uses 1 - progress).
DENOISE_PROGRESS_SHARE = 0.88
POST_PROGRESS_SHARE = 0.12


def pipeline_graph_step(
    node_id: str,
    on_log: Callable[[str, str], None] | None,
    *,
    message: str = "start",
) -> None:
    if on_log is not None:
        on_log("info", f"[{node_id}] {message}")


def validate_bundle_graph_step(
    bundle_root: Path | None,
    *,
    family: str,
    model_id: str,
    on_log: Callable[[str, str], None] | None,
    registry_entry: Any | None = None,
    project_root: Path | None = None,
) -> None:
    from backend.engine.common.bundle.layout import assert_media_bundle_ready

    assert_media_bundle_ready(
        bundle_root,
        family=family,
        model_id=model_id,
        registry_entry=registry_entry,
        project_root=project_root,
    )
    pipeline_graph_step(
        "validate_bundle",
        on_log,
        message=f"ok family={family} root={bundle_root}",
    )


def timestep_embed_schedule_from_scheduler(scheduler: Any) -> list[float] | None:
    """Continuous noise-level values for time-MLP live on ``scheduler.timesteps``."""
    sched_ts = getattr(scheduler, "timesteps", None)
    if sched_ts is None:
        return None
    arr = np.asarray(sched_ts, dtype=np.float64).reshape(-1)
    return [float(x) for x in arr.tolist()]


def emit_denoise_progress(
    on_progress: Callable[..., None] | None,
    step_1based: int,
    n_steps: int,
) -> None:
    if on_progress is None:
        return
    n = max(1, int(n_steps))
    s = min(max(1, int(step_1based)), n)
    on_progress(DENOISE_PROGRESS_SHARE * (s / n), s, n, "denoise", "denoising")


def emit_post_progress(
    on_progress: Callable[..., None] | None,
    *,
    n_steps: int,
    within_post: float,
) -> None:
    if on_progress is None:
        return
    n = max(1, int(n_steps))
    w = min(1.0, max(0.0, float(within_post)))
    on_progress(DENOISE_PROGRESS_SHARE + POST_PROGRESS_SHARE * w, n, n, "post", "decoding")


def emit_complete(on_progress: Callable[..., None] | None, n_steps: int) -> None:
    if on_progress is None:
        return
    n = max(1, int(n_steps))
    on_progress(1.0, n, n, None, "saving")


def emit_phase(
    on_progress: Callable[..., None] | None,
    *,
    phase: str,
    progress: float,
    n_steps: int,
) -> None:
    if on_progress is None:
        return
    n = max(1, int(n_steps))
    on_progress(min(1.0, max(0.0, float(progress))), 0, n, phase, phase)


def scale_batch_progress(
    cb: Callable | None,
    batch_idx: int,
    total: int,
) -> Callable | None:
    """Map per-batch progress into overall ``[batch_idx/total, (batch_idx+1)/total)``."""
    if cb is None or total <= 1:
        return cb

    def wrapped(p, s, t, msg=None, phase=None):
        overall_p = (batch_idx + float(p)) / total
        prefix = f"[{batch_idx + 1}/{total}]"
        msg_out = f"{prefix} {msg}" if msg else prefix
        cb(overall_p, s, t, msg_out, phase)

    return wrapped
