"""Inference optimization plans — mlx-diffusion-kit style orchestration (image + video)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

from backend.engine.common.ops.compile_policy import resolve_use_mlx_compile
from backend.engine.common.ops.lemica import lemica_enabled, normalize_lemica_mode
from backend.engine.common.ops.mfa_bridge_mlx import resolve_attention_backend
from backend.engine.common.ops.teacache_mlx import resolve_teacache_settings
from backend.engine.contracts import registry_scalar_default as _registry_scalar_default_fn

INFERENCE_PLAN_KEY = "_inference_plan"

_DISTILLED_STEP_THRESHOLD = 8


@dataclass(frozen=True)
class StepCachePlanSlice:
    enabled: bool
    mode: str
    thresh: float
    coefficients: tuple[float, float, float, float, float]
    skip_first: int
    skip_last: int


def _resolve_step_cache(
    family: str,
    mode: str | None,
    *,
    num_steps: int,
    user_thresh: float | None,
    config: Any,
) -> StepCachePlanSlice:
    tc_mode = str(mode if mode is not None else getattr(config, "teacache_mode", "auto")).strip().lower()
    enabled, thresh, coeffs, skip_first, skip_last = resolve_teacache_settings(
        family,
        tc_mode,
        num_steps=num_steps,
        user_thresh=user_thresh,
    )
    return StepCachePlanSlice(
        enabled=enabled,
        mode=tc_mode,
        thresh=thresh,
        coefficients=coeffs,
        skip_first=skip_first,
        skip_last=skip_last,
    )


@dataclass(frozen=True)
class ImageInferencePlan:
    """Frozen optimization decisions for one image denoise run."""

    family: str
    backend: str
    num_steps: int
    is_distilled_schedule: bool
    teacache_mode: str
    step_cache_enabled: bool
    step_cache_thresh: float
    step_cache_coefficients: tuple[float, float, float, float, float]
    step_cache_skip_first: int
    step_cache_skip_last: int
    lemica_mode: str
    lemica_enabled: bool
    use_mlx_compile: bool
    preview_mode: str
    preview_decoder: str
    use_batched_cfg: bool
    use_mm_dit_text_cache: bool
    use_text_embedding_cache: bool = True

    def needs_precompute_cap(self) -> bool:
        """Families may precompute static text/geo caches when compile or step cache is on."""
        return self.use_mlx_compile or self.step_cache_enabled


@dataclass(frozen=True)
class VideoInferencePlan:
    """Frozen optimization decisions for one video denoise run."""

    family: str
    backend: str
    num_steps: int
    is_distilled_schedule: bool
    teacache_mode: str
    step_cache_enabled: bool
    step_cache_thresh: float
    step_cache_coefficients: tuple[float, float, float, float, float]
    step_cache_skip_first: int
    step_cache_skip_last: int
    use_mlx_compile: bool
    use_batched_cfg: bool
    attention_backend: str
    vae_stream_cache: bool
    conv3d_backend: str
    use_text_embedding_cache: bool = True


InferencePlan = Union[ImageInferencePlan, VideoInferencePlan]

INFERENCE_PLAN_SNAPSHOT_KEY = "_inference_plan_snapshot"

PLAN_EXTRA_COND_KEY = INFERENCE_PLAN_KEY


def resolve_image_inference_plan(
    *,
    family: str,
    config: Any,
    entry: Any,
    ctx: Any,
    num_steps: int,
    teacache_mode: str | None = None,
    lemica_mode: str | None = None,
    teacache_thresh: float | None = None,
) -> ImageInferencePlan:
    """Resolve mutually compatible inference optimizations for one run."""
    backend = str(getattr(ctx, "backend", "mlx") or "mlx")
    n_steps = max(0, int(num_steps))
    distilled = n_steps <= _DISTILLED_STEP_THRESHOLD

    tc = _resolve_step_cache(
        family,
        teacache_mode,
        num_steps=n_steps,
        user_thresh=teacache_thresh,
        config=config,
    )
    lm_mode = normalize_lemica_mode(
        lemica_mode if lemica_mode is not None else getattr(config, "lemica_mode", "none")
    )
    lm_on = lemica_enabled(lm_mode) and not distilled
    if tc.enabled and lm_on:
        lm_on = False
        lm_mode = "none"

    use_compile = resolve_use_mlx_compile(
        config,
        num_steps=n_steps,
        backend=backend,
        teacache_mode=tc.mode if tc.enabled else "none",
    )

    preview_mode, _, _, preview_decoder = _preview_settings(entry)

    return ImageInferencePlan(
        family=str(family),
        backend=backend,
        num_steps=n_steps,
        is_distilled_schedule=distilled,
        teacache_mode=tc.mode,
        step_cache_enabled=tc.enabled,
        step_cache_thresh=tc.thresh,
        step_cache_coefficients=tc.coefficients,
        step_cache_skip_first=tc.skip_first,
        step_cache_skip_last=tc.skip_last,
        lemica_mode=lm_mode,
        lemica_enabled=lm_on,
        use_mlx_compile=use_compile,
        preview_mode=preview_mode,
        preview_decoder=preview_decoder,
        use_batched_cfg=bool(getattr(config, "use_batched_cfg", True)),
        use_mm_dit_text_cache=bool(getattr(config, "use_mm_dit_text_cache", False)),
        use_text_embedding_cache=bool(getattr(config, "use_text_embedding_cache", True)),
    )


def resolve_video_inference_plan(
    *,
    family: str,
    config: Any,
    ctx: Any,
    num_steps: int,
    teacache_mode: str | None = None,
    teacache_thresh: float | None = None,
) -> VideoInferencePlan:
    backend = str(getattr(ctx, "backend", "mlx") or "mlx")
    n_steps = max(0, int(num_steps))
    distilled = n_steps <= _DISTILLED_STEP_THRESHOLD

    tc = _resolve_step_cache(
        family,
        teacache_mode,
        num_steps=n_steps,
        user_thresh=teacache_thresh,
        config=config,
    )
    use_compile = resolve_use_mlx_compile(
        config,
        num_steps=n_steps,
        backend=backend,
        teacache_mode=tc.mode if tc.enabled else "none",
    )
    head_dim = int(getattr(config, "dim_in", 0) or getattr(config, "dim", 128))
    attn_backend = resolve_attention_backend(
        requested=getattr(config, "attention_backend", "auto"),
        head_dim=head_dim if head_dim > 0 else 128,
    )
    from backend.engine.common.ops.mfa_seedvr2_mlx import resolve_conv3d_backend

    conv3d_backend = resolve_conv3d_backend(getattr(config, "conv3d_backend", "auto"))

    return VideoInferencePlan(
        family=str(family),
        backend=backend,
        num_steps=n_steps,
        is_distilled_schedule=distilled,
        teacache_mode=tc.mode,
        step_cache_enabled=tc.enabled,
        step_cache_thresh=tc.thresh,
        step_cache_coefficients=tc.coefficients,
        step_cache_skip_first=tc.skip_first,
        step_cache_skip_last=tc.skip_last,
        use_mlx_compile=use_compile,
        use_batched_cfg=bool(getattr(config, "use_batched_cfg", True)),
        attention_backend=attn_backend,
        vae_stream_cache=bool(getattr(config, "vae_stream_cache", False)),
        conv3d_backend=conv3d_backend,
        use_text_embedding_cache=bool(getattr(config, "use_text_embedding_cache", True)),
    )


def plan_from_extra_cond(
    extra_cond: dict[str, Any],
    *,
    family: str,
    config: Any,
    entry: Any,
    ctx: Any,
    num_steps: int,
) -> InferencePlan:
    existing = extra_cond.get(INFERENCE_PLAN_KEY)
    if isinstance(existing, (ImageInferencePlan, VideoInferencePlan)):
        return existing
    if str(family) in ("wan", "ltx", "hunyuan", "seedvr2"):
        return resolve_video_inference_plan(
            family=family,
            config=config,
            ctx=ctx,
            num_steps=num_steps,
            teacache_mode=extra_cond.get("teacache_mode"),
            teacache_thresh=extra_cond.get("teacache_thresh"),
        )
    return resolve_image_inference_plan(
        family=family,
        config=config,
        entry=entry,
        ctx=ctx,
        num_steps=num_steps,
        teacache_mode=extra_cond.get("teacache_mode"),
        lemica_mode=extra_cond.get("lemica_mode"),
        teacache_thresh=extra_cond.get("teacache_thresh"),
    )


def inference_plan_snapshot(plan: InferencePlan) -> dict[str, Any]:
    """JSON-safe metadata fields for asset/task records (no tensor/plan object)."""
    snap: dict[str, Any] = {"teacache_mode": plan.teacache_mode}
    if plan.step_cache_enabled:
        snap["teacache_thresh"] = float(plan.step_cache_thresh)
    if plan.use_mlx_compile:
        snap["mlx_compile"] = True
    if plan.use_batched_cfg:
        snap["batched_cfg"] = True
    if isinstance(plan, ImageInferencePlan):
        if plan.lemica_enabled:
            snap["lemica_mode"] = plan.lemica_mode
        snap["preview_mode"] = plan.preview_mode
        if plan.preview_decoder != "auto":
            snap["preview_decoder"] = plan.preview_decoder
    if isinstance(plan, VideoInferencePlan):
        snap["attention_backend"] = plan.attention_backend
        if plan.vae_stream_cache:
            snap["vae_stream_cache"] = True
        snap["conv3d_backend"] = plan.conv3d_backend
    return snap


def inference_plan_log_line(plan: InferencePlan) -> str:
    parts = [
        f"family={plan.family}",
        f"steps={plan.num_steps}",
        f"teacache={plan.teacache_mode}",
    ]
    if plan.use_mlx_compile:
        parts.append("mlx_compile=on")
    if plan.use_batched_cfg:
        parts.append("batched_cfg=on")
    if isinstance(plan, ImageInferencePlan):
        if plan.lemica_enabled:
            parts.append(f"lemica={plan.lemica_mode}")
        parts.append(f"preview={plan.preview_mode}/{plan.preview_decoder}")
    if isinstance(plan, VideoInferencePlan):
        parts.append(f"attn={plan.attention_backend}")
        if plan.vae_stream_cache:
            parts.append("vae_stream_cache=on")
    return "[inference] " + " ".join(parts)


def log_inference_plan_from_cond(
    extra_cond: dict[str, Any],
    on_log: Any | None,
) -> None:
    plan = extra_cond.get(INFERENCE_PLAN_KEY)
    if on_log is not None and isinstance(plan, (ImageInferencePlan, VideoInferencePlan)):
        on_log("info", inference_plan_log_line(plan))


def merge_inference_run_metadata(model: Any, extra_cond: dict[str, Any]) -> dict[str, Any]:
    from backend.engine.common.ops.step_cache import collect_step_cache_stats

    meta = dict(extra_cond.get(INFERENCE_PLAN_SNAPSHOT_KEY) or {})
    meta.update(collect_step_cache_stats(model))
    return meta


def stash_inference_run_metadata(model: Any, extra_cond: dict[str, Any]) -> None:
    meta = merge_inference_run_metadata(model, extra_cond)
    if meta:
        setattr(model, "_dq_inference_run_meta", meta)


def pop_inference_run_metadata(model: Any) -> dict[str, Any]:
    meta = getattr(model, "_dq_inference_run_meta", None)
    if isinstance(meta, dict):
        try:
            delattr(model, "_dq_inference_run_meta")
        except AttributeError:
            pass
        return dict(meta)
    return {}


_TASK_INFERENCE_METADATA_KEYS = (
    "teacache_mode",
    "teacache_thresh",
    "teacache_skipped",
    "teacache_computed",
    "teacache_skip_rate",
    "mlx_compile",
    "batched_cfg",
    "lemica_mode",
    "preview_mode",
    "preview_decoder",
    "attention_backend",
    "vae_stream_cache",
    "conv3d_backend",
)


def inference_metadata_for_task(asset_metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Subset of asset metadata surfaced on ``EngineResult.metadata`` / task result JSON."""
    if not asset_metadata:
        return {}
    return {
        key: asset_metadata[key]
        for key in _TASK_INFERENCE_METADATA_KEYS
        if key in asset_metadata
    }


def attach_inference_plan(
    extra_cond: dict[str, Any],
    plan: InferencePlan,
) -> dict[str, Any]:
    out = dict(extra_cond)
    out[INFERENCE_PLAN_KEY] = plan
    out[INFERENCE_PLAN_SNAPSHOT_KEY] = inference_plan_snapshot(plan)
    if plan.step_cache_enabled:
        out["teacache_mode"] = plan.teacache_mode
    if isinstance(plan, ImageInferencePlan) and plan.lemica_enabled:
        out["lemica_mode"] = plan.lemica_mode
    if isinstance(plan, VideoInferencePlan):
        out["attention_backend"] = plan.attention_backend
        if plan.vae_stream_cache:
            out["vae_stream_cache"] = True
    return out


def attach_resolved_video_inference_plan(
    extra_cond: dict[str, Any],
    *,
    family: str,
    config: Any,
    ctx: Any,
    num_steps: int,
) -> dict[str, Any]:
    plan = resolve_video_inference_plan(
        family=family,
        config=config,
        ctx=ctx,
        num_steps=num_steps,
        teacache_mode=extra_cond.get("teacache_mode"),
        teacache_thresh=extra_cond.get("teacache_thresh"),
    )
    return attach_inference_plan(extra_cond, plan)


def attach_resolved_inference_plan(
    extra_cond: dict[str, Any],
    *,
    family: str,
    config: Any,
    entry: Any,
    ctx: Any,
    num_steps: int,
) -> dict[str, Any]:
    plan = resolve_image_inference_plan(
        family=family,
        config=config,
        entry=entry,
        ctx=ctx,
        num_steps=num_steps,
        teacache_mode=extra_cond.get("teacache_mode"),
        lemica_mode=extra_cond.get("lemica_mode"),
        teacache_thresh=extra_cond.get("teacache_thresh"),
    )
    return attach_inference_plan(extra_cond, plan)


def pop_inference_plan(cond: dict[str, Any]) -> InferencePlan | None:
    plan = cond.pop(INFERENCE_PLAN_KEY, None)
    if isinstance(plan, (ImageInferencePlan, VideoInferencePlan)):
        return plan
    return None


def _preview_settings(entry: Any | None) -> tuple[str, int, int, str]:
    if entry is None:
        return "stream", 2, 512, "auto"
    mode = _registry_scalar_default_fn(entry, "preview_mode", None)
    if mode is None:
        raw = getattr(entry, "raw", None) or {}
        model_type = str(raw.get("type", "") if isinstance(raw, dict) else "")
        mode = "none" if model_type != "diffusion" else "stream"
    mode = str(mode).strip().lower()
    if mode not in ("stream", "none"):
        mode = "none"
    interval = int(_registry_scalar_default_fn(entry, "preview_interval_steps", 2) or 2)
    max_edge = int(_registry_scalar_default_fn(entry, "preview_max_edge", 512) or 512)
    decoder = str(_registry_scalar_default_fn(entry, "preview_decoder", "auto") or "auto").strip().lower()
    if decoder not in ("auto", "standard", "taesd"):
        decoder = "auto"
    return mode, max(1, interval), max(64, min(2048, max_edge)), decoder
