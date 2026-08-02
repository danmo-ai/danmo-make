"""Latent-space refinement before VAE decode (Z-Image hires)."""
from __future__ import annotations

from typing import Any, Callable


def apply_latent_refine_if_requested(
    pipeline: Any,
    latents: Any,
    *,
    request: Any,
    entry: Any,
    version_key: str | None,
    model: Any,
    timesteps: list[Any],
    sigmas: Any,
    txt_embeds: Any,
    neg_embeds: Any | None,
    guidance: float,
    extra_cond: dict[str, Any],
    on_log: Callable[..., None] | None = None,
    scheduler: Any = None,
    config: Any = None,
    runtime_contract: Any = None,
    semantics: Any = None,
    exec_ctx: Any = None,
    encoder_type: str = "",
    sched_ts: Any = None,
    timestep_embed_schedule: list[float] | None = None,
    pooled_embeds: Any = None,
    neg_pooled_embeds: Any = None,
    txt_attn_mask: Any = None,
    neg_attn_mask: Any = None,
) -> Any:
    spec = getattr(request, "latent_refine", None)
    if spec is None:
        return latents
    scale = float(getattr(spec, "scale", 1.0) or 1.0)
    if scale <= 1.0 + 1e-6:
        return latents

    from backend.engine.config.model_configs import get_config_class

    family = str(getattr(entry, "family", "") or "")
    family_config = get_config_class(family)()
    if not getattr(family_config, "supports_latent_refine", False):
        raise RuntimeError(f"latent_refine is not supported for family={family!r}")

    from backend.engine.runtime.mlx_guards import require_mlx_backend

    require_mlx_backend(pipeline.ctx, feature="latent_refine")
    if scheduler is None or config is None or runtime_contract is None or semantics is None:
        raise RuntimeError(
            "latent_refine requires scheduler, config, runtime_contract, and semantics from the pipeline run context"
        )
    if sigmas is None:
        raise RuntimeError("latent_refine requires scheduler sigmas for img2img-style noise injection")

    ctx = pipeline.ctx
    denoise_strength = float(getattr(spec, "denoise_strength", 0.35) or 0.35)
    hires_steps = int(getattr(spec, "hires_steps", 0) or 0)
    if hires_steps <= 0:
        hires_steps = max(4, int(round(denoise_strength * 12)))

    _c, f, h, w = latents.shape if latents.ndim == 4 else (
        latents.shape[1],
        latents.shape[2],
        latents.shape[3],
        latents.shape[4],
    )
    if latents.ndim == 5:
        latents = latents[0]
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    mode = str(getattr(spec, "interpolation", "linear") or "linear").lower()
    upscaled = _resize_latent_nchw(ctx, latents, new_h, new_w, mode=mode)
    if on_log:
        on_log(
            "info",
            f"latent_refine scale={scale:.2f} {h}x{w}->{new_h}x{new_w} "
            f"hires_steps={hires_steps} denoise_strength={denoise_strength:.2f}",
        )

    n_steps = len(timesteps)
    init_timestep = 0
    if denoise_strength > 0.0:
        init_timestep = max(1, int(n_steps * denoise_strength))
    init_timestep = min(init_timestep, max(0, n_steps - 1))
    hires_timesteps = timesteps[init_timestep : init_timestep + hires_steps]
    if not hires_timesteps:
        return upscaled

    from backend.engine.pipelines.image_run_common import prepare_edit_rewrite_latents

    refine_seed = int(getattr(request, "seed", None) or 0) + 991
    latents = prepare_edit_rewrite_latents(
        pipeline,
        model=model,
        config=config,
        runtime_contract=runtime_contract,
        encoded=upscaled,
        seed=refine_seed,
        init_timestep=init_timestep,
        sigmas=sigmas,
    )

    local_extra = dict(extra_cond)
    local_extra.pop("zimage_geo_cache", None)
    local_extra.pop("zimage_neg_geo_cache", None)
    local_extra["lemica_mode"] = "none"
    latents, local_extra = model.before_denoise(
        latents,
        timesteps,
        sigmas,
        txt_embeds=txt_embeds,
        neg_embeds=neg_embeds,
        **local_extra,
    )

    vae_scale = int(getattr(family_config, "vae_scale", 8) or 8)
    hires_w = new_w * vae_scale
    hires_h = new_h * vae_scale

    from backend.engine.inference.image_denoise import run_image_denoise

    latents = run_image_denoise(
        pipeline,
        model=model,
        scheduler=scheduler,
        timesteps=hires_timesteps,
        latents=latents,
        config=config,
        runtime_contract=runtime_contract,
        guidance=guidance,
        txt_embeds=txt_embeds,
        neg_embeds=neg_embeds,
        pooled_embeds=pooled_embeds,
        neg_pooled_embeds=neg_pooled_embeds,
        txt_attn_mask=txt_attn_mask,
        neg_attn_mask=neg_attn_mask,
        encoder_type=encoder_type,
        width=hires_w,
        height=hires_h,
        sched_ts=sched_ts,
        sigmas=sigmas,
        timestep_embed_schedule=timestep_embed_schedule,
        extra_cond=local_extra,
        semantics=semantics,
        ctx_exec=exec_ctx,
        on_progress=None,
        on_log=on_log,
        preview_mode="none",
        timestep_offset=init_timestep,
    )
    if latents is None:
        raise RuntimeError("latent_refine denoise was cancelled")
    ctx.eval(latents)
    return latents


def _resize_latent_nchw(ctx: Any, latents: Any, new_h: int, new_w: int, *, mode: str) -> Any:
    import mlx.core as mx
    import mlx.nn as nn

    if getattr(ctx, "backend", None) != "mlx":
        raise RuntimeError("latent_refine is MLX-only today")
    # [C,F,H,W] → upsample H,W via nn.Upsample (NHWC; mx.core has no image API)
    c, f, h, w = latents.shape
    flat = mx.reshape(latents, (c * f, h, w))
    flat = mx.expand_dims(flat, 0)
    interp = str(mode or "linear").lower()
    if interp == "nearest":
        up_mode = "nearest"
    elif interp == "cubic":
        up_mode = "cubic"
    else:
        up_mode = "linear"
    scale_h = new_h / h
    scale_w = new_w / w
    flat_hwc = mx.transpose(flat, (0, 2, 3, 1))
    out_hwc = nn.Upsample(scale_factor=(scale_h, scale_w), mode=up_mode)(flat_hwc)
    out = mx.transpose(out_hwc, (0, 3, 1, 2))
    out = mx.reshape(out[0], (c, f, new_h, new_w))
    return out
