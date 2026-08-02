"""Shared resolve / encode / schedule / VAE helpers for video create + edit."""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from backend.core.contracts import (
    ExecutionContext,
    VideoAvatarRequest,
    VideoEditRequest,
    VideoGenerationRequest,
    parse_model_version,
    parse_size,
    work_title_metadata,
)
from backend.engine._transformer_registry import (
    encode_video_hunyuan_dual_cfg_batch as _encode_video_hunyuan_dual_cfg_batch_fn,
    encode_video_prompt as _encode_video_prompt_fn,
    get_video_avatar_factory as _get_video_avatar_factory,
    get_video_generation_factory as _get_video_generation_factory,
    merge_video_lora_adapters as _merge_video_lora_adapters,
    validate_video_avatar_params,
    validate_video_generation_params,
)
from backend.engine.common.bundle.layout import t5_encoder_bundle_paths
from backend.engine.common.codecs.text_encoders import T5Encoder
from backend.engine.common.ops.schedulers import get_scheduler
from backend.engine.config.model_configs import get_config_class
from backend.engine.contracts import (
    create_video_t5_encoder,
    inject_hunyuan_text_encoder_paths,
    inject_ltx_text_encoder_paths,
    local_bundle_root as _local_bundle_root_fn,
    merge_video_bundle_config,
    registry_scalar_default as _registry_scalar_default_fn,
    require_entry_family,
    resolve_project_path as _resolve_project_path_fn,
    resolve_version_block as _resolve_version_block_fn,
    resolve_wan_shift_value,
    video_apply_i2v_conditioning,
    video_apply_ltx_distilled_scheduler_timesteps,
    video_apply_hunyuan_step_distill_scheduler_timesteps,
    video_apply_wan_step_distill_scheduler_timesteps,
    video_cfg_negative_prompt,
    video_encoder_type,
    video_i2v_encode_failure_message,
    video_infer_log_extras,
    video_prepare_i2v_source_image,
    video_resolve_shift_value,
    video_rotary_model_kwargs,
    video_scheduler_ctor_kwargs,
    video_snap_pixel_dims_if_needed,
    video_t5_max_seq_len,
    video_uses_ltx_distilled_timesteps,
    video_uses_hunyuan_step_distill_timesteps,
    video_uses_wan_step_distill_timesteps,
    video_validate_generate_geometry,
    wan_t5_bundle_paths,
)
from backend.engine.inference.video_denoise import run_video_denoise
from backend.engine.pipelines.pipeline_progress import (
    emit_complete,
    emit_phase,
    emit_post_progress,
    pipeline_graph_step,
    timestep_embed_schedule_from_scheduler,
    validate_bundle_graph_step,
)
from backend.engine.pipelines.video_bundle_layout import ltx_flat_vae_decoder_file
from backend.engine.pipelines.video_model_load import (
    load_video_transformer,
    video_model_cache_key,
)
from backend.engine.video_codec_registry import get_video_decode_handler, get_video_encode_handler


def _video_post_denoise_clear_cache(config: Any) -> bool:
    return bool(getattr(config, "post_denoise_clear_cache", False))

# --- Video pipeline ops (extracted from VideoPipeline) ---

def configure_hunyuan_text_encoder_paths(pipeline, entry, config) -> None:
    """Resolve registry-declared native TE roots (ModelScope Qwen + ByT5)."""
    inject_hunyuan_text_encoder_paths(entry, config, pipeline._project_root)


def configure_ltx_text_encoder_paths(pipeline, entry, config) -> None:
    """Resolve registry-declared Gemma 3 root under ``models/Text/…``."""
    inject_ltx_text_encoder_paths(entry, config, pipeline._project_root)

def resolved_fp16_video_bundle_root(pipeline, entry) -> Path | None:
    """Registry full-weight ``local_path`` for the same model (T5 fallback for MLX-only trees)."""
    from backend.core.version_keys import resolve_full_bundle_version_key

    raw = getattr(entry, "raw", {}) or {}
    versions = raw.get("versions") or {}
    base_key = resolve_full_bundle_version_key(versions)
    if not base_key:
        return None
    ob = versions.get(base_key)
    if not isinstance(ob, dict):
        return None
    lp = (ob.get("local_path") or "").strip()
    if not lp:
        return None
    p = _resolve_project_path_fn(pipeline._project_root, lp)
    return p if p.is_dir() else None


def effective_t5_bundle_root(pipeline, entry, bundle_root: Path | None, config: Any) -> Path | None:
    """Prefer current version bundle; if T5 dirs are missing (typical MLX-forge flat HF), use full-weight tier."""
    if bundle_root is None or not bundle_root.is_dir():
        return None
    try:
        if bool(getattr(config, "uses_wan_t5_bundle", False)):
            wan_t5_bundle_paths(bundle_root)
        else:
            t5_encoder_bundle_paths(bundle_root)
        return bundle_root
    except RuntimeError as err:
        alt = resolved_fp16_video_bundle_root(pipeline, entry)
        if alt is not None:
            if bool(getattr(config, "uses_wan_t5_bundle", False)):
                wan_t5_bundle_paths(alt)
            else:
                t5_encoder_bundle_paths(alt)
            return alt
        raise RuntimeError(
            f"T5 text encoder assets not found under {bundle_root}, "
            f"and no installed full-weight registry version for ``{entry.id}``."
            + (
                " Wan bundles require ``models_t5*.pth`` and ``google/umt5-xxl``."
                if bool(getattr(config, "uses_wan_t5_bundle", False))
                else " Install a full model bundle with ``text_encoder`` + ``tokenizer``."
            )
        ) from err

def resolve_guidance_default(pipeline, entry) -> float:
    g = _registry_scalar_default_fn(entry, "guidance", None)
    if g is not None:
        return float(g)
    gs = _registry_scalar_default_fn(entry, "guide_scale", None)
    if gs is not None:
        return float(gs)
    return 0.0

def resolve_num_frames(pipeline, request: VideoGenerationRequest | VideoEditRequest, entry) -> int:
    if request.num_frames is not None:
        return int(request.num_frames)
    reg = _registry_scalar_default_fn(entry, "num_frames", None)
    if reg is not None:
        return int(reg)
    return 81

def resolve_fps(pipeline, request: VideoGenerationRequest | VideoEditRequest, entry) -> int:
    if request.fps is not None:
        return int(request.fps)
    reg = _registry_scalar_default_fn(entry, "fps", None)
    if reg is not None:
        return int(reg)
    return 16

def validate_wan_umt5_embeddings(pipeline,
    config: Any,
    txt_embeds: Any | None,
    on_log: Callable[[str, str], None] | None,
) -> None:
    if not on_log or not bool(getattr(config, "validate_umt5_embeddings", False)) or txt_embeds is None:
        return
    pipeline.ctx.eval(txt_embeds)
    peak = float(pipeline.ctx.sqrt(pipeline.ctx.max(pipeline.ctx.square(txt_embeds))))
    if peak < 1e-3:
        raise RuntimeError(
            "Wan UMT5 embeddings are near zero; text encoder weights may not be loaded"
        )
    on_log("info", f"Wan UMT5 text embeddings ready (peak={peak:.3f})")

def validate_generate_geometry(pipeline, config: Any, w: int, h: int, num_frames: int,
) -> None:
    video_validate_generate_geometry(config, w, h, num_frames)

def snap_wan_pixel_dims_if_needed(pipeline,
    config: Any,
    w: int,
    h: int,
    *,
    on_log: Callable | None = None,
) -> tuple[int, int]:
    return video_snap_pixel_dims_if_needed(config, w, h, on_log=on_log)

def apply_video_registry_config_overrides(pipeline, entry: Any, config: Any) -> None:
    for param_key in (
        "vae_scale",
        "default_scheduler",
        "text_encoder_device",
        "vae_temporal_chunk_size",
        "gemma_model_id",
        "low_ram_streaming",
        "ltx_stage2_steps",
        "video_edit_source_mode",
        "video_pipeline_shape",
        "bernini_renderer",
        "use_src_id_rotary_emb",
    ):
        val = _registry_scalar_default_fn(entry, param_key, None)
        if val is not None:
            setattr(config, param_key, val)
    sg = _registry_scalar_default_fn(entry, "supports_guidance", None)
    if sg is not None:
        config.supports_guidance = bool(sg)
    sd = _registry_scalar_default_fn(entry, "step_distill", None)
    if sd is not None:
        config.step_distill = bool(sd)
    lvs = _registry_scalar_default_fn(entry, "long_video_support", None)
    if lvs is not None:
        config.supports_long_video = bool(lvs)
    moe_boundary = _registry_scalar_default_fn(entry, "moe_boundary_step_index", None)
    if moe_boundary is not None:
        config.moe_boundary_step_index = int(moe_boundary)
    distill_ts = _registry_scalar_default_fn(entry, "wan_distill_timesteps", None)
    if isinstance(distill_ts, (list, tuple)) and distill_ts:
        config.wan_distill_timesteps = tuple(float(x) for x in distill_ts)
    hy_distill_ts = _registry_scalar_default_fn(entry, "hunyuan_distill_timesteps", None)
    if isinstance(hy_distill_ts, (list, tuple)) and hy_distill_ts:
        config.hunyuan_distill_timesteps = tuple(float(x) for x in hy_distill_ts)
    hy_distill_shift = _registry_scalar_default_fn(entry, "hunyuan_distill_shift", None)
    if hy_distill_shift is not None:
        config.hunyuan_distill_shift = float(hy_distill_shift)
    vst = _registry_scalar_default_fn(entry, "vae_spatial_tiling", None)
    if vst is not None:
        config.vae_spatial_tiling = bool(vst)
    br = _registry_scalar_default_fn(entry, "bernini_renderer", None)
    if br is not None:
        config.bernini_renderer = bool(br)
    vps = _registry_scalar_default_fn(entry, "video_pipeline_shape", None)
    if vps is not None:
        config.video_pipeline_shape = str(vps)
    usre = _registry_scalar_default_fn(entry, "use_src_id_rotary_emb", None)
    if usre is not None:
        config.use_src_id_rotary_emb = bool(usre)
    if getattr(config, "inject_text_encoder_paths", False):
        configure_hunyuan_text_encoder_paths(pipeline, entry, config)
        configure_ltx_text_encoder_paths(pipeline, entry, config)

def prepare_video_bundle_and_schedule(pipeline,
    *,
    entry: Any,
    config: Any,
    family: str,
    model_key: str,
    version_key: str | None,
    w: int,
    h: int,
    num_frames: int,
    request: VideoGenerationRequest | VideoEditRequest,
    on_log: Callable | None,
) -> tuple[Path | None, int, int, int, float, bool, str]:
    bundle_root = _local_bundle_root_fn(pipeline._project_root, entry, version_key or None)
    validate_bundle_graph_step(
        bundle_root,
        family=family,
        model_id=model_key,
        on_log=on_log,
        registry_entry=entry,
        project_root=pipeline._project_root,
    )
    merge_video_bundle_config(config, bundle_root)
    w, h = snap_wan_pixel_dims_if_needed(pipeline, config, w, h, on_log=on_log)
    validate_generate_geometry(pipeline, config, w, h, num_frames)
    encoder_type = video_encoder_type(config)
    if encoder_type == "t5":
        pipeline._t5_bundle_root = effective_t5_bundle_root(pipeline, entry, bundle_root, config)
        prepare_t5_context(pipeline, config)
    else:
        pipeline._t5_bundle_root = bundle_root
        pipeline._video_config = config

    steps_default = _registry_scalar_default_fn(entry, "steps", 40)
    guidance_default = resolve_guidance_default(pipeline, entry)
    scheduler_registry = _registry_scalar_default_fn(entry, "scheduler", None)
    scheduler_default = scheduler_registry or getattr(config, "default_scheduler", "unipc")

    adapters = getattr(request, "adapters", None) or []
    lightning_distill = False

    def _apply_wan_lightning_distill() -> bool:
        from backend.engine.families.wan.lora_mlx import adapters_include_wan_lightning

        if not adapters_include_wan_lightning(adapters, pipeline._registry):
            return False
        from backend.engine.families.wan.distill import WAN_DISTILL_DEFAULT_BOUNDARY_STEP_INDEX

        config.step_distill = True
        config.supports_guidance = False
        config.moe_boundary_step_index = WAN_DISTILL_DEFAULT_BOUNDARY_STEP_INDEX
        if getattr(config, "wan_distill_shift", None) is None:
            config.wan_distill_shift = 5.0
        if on_log is not None:
            on_log(
                "info",
                "Wan Lightning LoRA: 4-step distill schedule (no CFG, shift=5, MoE boundary=2)",
            )
        return True

    _adapter_distill_hooks = {"wan": _apply_wan_lightning_distill}
    _distill_hook = _adapter_distill_hooks.get(family)
    if _distill_hook is not None and adapters:
        lightning_distill = _distill_hook()

    steps = int(request.steps) if request.steps is not None else int(steps_default)
    if lightning_distill:
        steps = 4
    steps = max(1, steps)
    step_distill = bool(
        lightning_distill
        or getattr(config, "step_distill", False)
        or _registry_scalar_default_fn(entry, "step_distill", False)
    )
    validate_video_generation_params(
        family,
        entry=entry,
        config=config,
        step_distill=step_distill,
    )
    guidance = float(request.guidance) if request.guidance is not None else float(guidance_default)
    if step_distill or not getattr(config, "supports_guidance", True):
        guidance = 0.0
    return bundle_root, w, h, steps, guidance, step_distill, scheduler_default

def video_encode_load_and_condition(pipeline,
    *,
    request: VideoGenerationRequest | VideoEditRequest,
    entry: Any,
    config: Any,
    family: str,
    bundle_root: Path | None,
    version_key: str | None,
    model_key: str,
    num_frames: int,
    guidance: float,
    ctx_exec: ExecutionContext,
    on_log: Callable | None,
    preloaded_model: Any | None = None,
) -> tuple[Any, dict[str, Any], Any, Any, int] | None:
    pipeline_graph_step("encode_prompt", on_log)
    txt_embeds = neg_embeds = None
    txt_mask = txt_mask_2 = neg_mask = neg_mask_2 = None
    neg_embeds_2 = txt_embeds_2 = None
    if request.prompt and config.text_dim > 0:
        (
            txt_embeds, txt_mask, txt_embeds_2, txt_mask_2,
            neg_embeds, neg_mask, neg_embeds_2, neg_mask_2,
        ) = encode_video_text_with_cfg(pipeline, 
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            config=config,
            family=family,
            bundle_root=bundle_root,
            guidance=guidance,
            version_key=version_key,
        )
        validate_wan_umt5_embeddings(pipeline, config, txt_embeds, on_log)

    if ctx_exec.cancel_token.is_cancelled():
        return None

    encoder_type = video_encoder_type(config)
    release_video_t5_after_encode(pipeline, config, encoder_type)

    latent_frames = latent_frame_count(pipeline, config, num_frames)
    if preloaded_model is not None:
        model = preloaded_model
    else:
        from backend.engine.common.bundle.quant_inference import assert_quantized_dit_lora_compatible

        assert_quantized_dit_lora_compatible(
            entry, version_key or None, getattr(request, "adapters", None)
        )
        pipeline_graph_step("load_transformer", on_log)
        allow_cache = not (getattr(request, "adapters", None) or [])
        model = load_model(
            pipeline,
            config,
            entry,
            version_key or None,
            latent_frames,
            allow_cache=allow_cache,
            on_log=on_log,
        )
        if model is None:
            raise RuntimeError(f"Failed to load model: {model_key}")
    apply_video_lora_adapters(pipeline, family, model, request, on_log)
    if preloaded_model is None:
        model.after_load_weights(bundle_root=str(bundle_root) if bundle_root else None)
    extra_cond = model.prepare_conditioning(
        request,
        bundle_root=str(bundle_root) if bundle_root else None,
    )
    extra_cond = apply_video_text_to_extra_cond(pipeline, 
        extra_cond,
        txt_embeds=txt_embeds,
        txt_mask=txt_mask,
        txt_embeds_2=txt_embeds_2,
        txt_mask_2=txt_mask_2,
        neg_embeds=neg_embeds,
        neg_mask=neg_mask,
        neg_embeds_2=neg_embeds_2,
        neg_mask_2=neg_mask_2,
    )
    return model, extra_cond, txt_embeds, neg_embeds, latent_frames

def create_timesteps_for_video(pipeline,
    *,
    entry: Any,
    config: Any,
    request: VideoGenerationRequest | VideoEditRequest,
    steps: int,
    step_distill: bool,
    scheduler_default: str,
    bundle_root: Path | None,
    w: int,
    h: int,
    num_frames: int,
    on_log: Callable | None,
) -> tuple[Any, Any, Any, list[float] | None, float | None]:
    scheduler = create_video_scheduler(pipeline, 
        config=config,
        scheduler_name=scheduler_default,
        bundle_root=bundle_root,
    )
    wan_shift: float | None = None
    if video_uses_ltx_distilled_timesteps(
        config, step_distill=step_distill, scheduler_default=scheduler_default,
    ):
        vae_scale = int(getattr(config, "vae_scale", 32))
        latent_frames = latent_frame_count(pipeline, config, num_frames)
        timesteps = video_apply_ltx_distilled_scheduler_timesteps(
            pipeline.ctx,
            scheduler,
            bundle_root=bundle_root,
            steps=steps,
            w=w,
            h=h,
            latent_frames=latent_frames,
            vae_scale=vae_scale,
            on_log=on_log,
        )
    elif video_uses_hunyuan_step_distill_timesteps(
        config, step_distill=step_distill, scheduler_default=scheduler_default,
    ):
        timesteps = video_apply_hunyuan_step_distill_scheduler_timesteps(
            pipeline.ctx,
            scheduler,
            steps=steps,
            config=config,
        )
    elif video_uses_wan_step_distill_timesteps(
        config, step_distill=step_distill, scheduler_default=scheduler_default,
    ):
        timesteps = video_apply_wan_step_distill_scheduler_timesteps(
            pipeline.ctx,
            scheduler,
            steps=steps,
            config=config,
        )
        if on_log is not None:
            on_log(
                "info",
                f"Wan step-distill schedule: steps={steps}, "
                f"boundary_step_index={getattr(config, 'moe_boundary_step_index', 2)}",
            )
    else:
        sched_kwargs: dict[str, Any] = {}
        shift_default = _registry_scalar_default_fn(entry, "shift", None)
        shift_val = video_resolve_shift_value(
            config,
            request_shift=request.shift,
            registry_shift=shift_default,
            scheduler_default_shift=getattr(scheduler, "_default_shift", None),
            on_log=on_log,
        )
        if shift_val is not None:
            sched_kwargs["shift"] = shift_val
        timesteps = scheduler.set_timesteps(steps, **sched_kwargs)
        if bool(getattr(config, "uses_wan_shift", False)):
            wan_shift = float(
                sched_kwargs.get("shift", getattr(scheduler, "_default_shift", 1.0))
            )
    sigmas = getattr(scheduler, "sigmas", None)
    timestep_embed_schedule = timestep_embed_schedule_from_scheduler(scheduler)
    return scheduler, timesteps, sigmas, timestep_embed_schedule, wan_shift

def finalize_video_from_latents(pipeline,
    *,
    latents: Any,
    timesteps: Any,
    entry: Any,
    version_key: str | None,
    config: Any,
    model_key: str,
    seed: int,
    request: VideoGenerationRequest | VideoEditRequest,
    ctx_exec: ExecutionContext,
    fps: int,
    num_frames: int,
    w: int,
    h: int,
    steps: int,
    guidance: float,
    on_progress: Callable | None,
    on_log: Callable | None,
    model: Any | None = None,
) -> tuple[str, dict[str, Any]] | None:
    if getattr(pipeline.ctx, "backend", None) == "mlx":
        if on_log:
            on_log("info", "Materializing denoised latents for VAE decode...")
        pipeline.ctx.eval(latents)
        if _video_post_denoise_clear_cache(config):
            pipeline.ctx.clear_cache()

    n_steps = len(timesteps)
    if on_log:
        on_log("info", "Decoding video latents (VAE)...")
    emit_post_progress(on_progress, n_steps=n_steps, within_post=0.1)

    def _vae_post_progress(frac: float) -> None:
        emit_post_progress(
            on_progress, n_steps=n_steps, within_post=0.1 + 0.75 * min(1.0, max(0.0, frac)),
        )

    def _vae_post_log(msg: str) -> None:
        if on_log:
            on_log("info", msg)

    pipeline_graph_step("decode_vae", on_log)
    frames = vae_decode_video(pipeline, 
        latents, entry, version_key or None, config,
        on_post_progress=_vae_post_progress,
        on_post_log=_vae_post_log,
    )

    if ctx_exec.cancel_token.is_cancelled():
        return None

    emit_post_progress(on_progress, n_steps=n_steps, within_post=0.85)
    if on_log:
        on_log("info", f"Saving video ({len(frames)} frames)...")

    pipeline_graph_step("save_asset", on_log)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = Path(ctx_exec.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_path = str(work / f"{model_key}_{seed}_{timestamp}.mp4")
    save_video(pipeline, frames, out_path, fps=fps)

    emit_complete(on_progress, n_steps)

    metadata = {
        "model": request.model, "seed": seed,
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt or "",
        "steps": steps,
        "guidance": guidance, "num_frames": num_frames,
        "fps": fps, "width": w, "height": h,
        "mime_type": "video/mp4",
    }
    if model is not None:
        from backend.engine.inference.optimization_plan import pop_inference_run_metadata

        metadata.update(pop_inference_run_metadata(model))
    metadata.update(work_title_metadata(request.title))
    return out_path, metadata

def uses_family_video_generator(config: Any) -> bool:
    return str(getattr(config, "video_pipeline_shape", "dit_standard") or "dit_standard") == "family_generator"


def uses_family_video_avatar(config: Any) -> bool:
    return str(getattr(config, "video_pipeline_shape", "dit_standard") or "dit_standard") == "family_avatar"

def execute_family_video_generator(pipeline,
    request: VideoGenerationRequest | VideoEditRequest,
    ctx_exec: ExecutionContext,
    *,
    is_edit: bool,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Shape C video generation — family-owned stack (T2V / I2V)."""
    model_key, version_key = parse_model_version(request.model)
    w, h = parse_size(request.size)
    entry = pipeline._registry.require(model_key)
    family = require_entry_family(entry, model_id=model_key)
    config_cls = get_config_class(family)
    config = config_cls()
    num_frames = resolve_num_frames(pipeline, request, entry)
    fps = resolve_fps(pipeline, request, entry)
    seed = request.seed if request.seed is not None else random.randint(0, 2 ** 32 - 1)

    apply_video_registry_config_overrides(pipeline, entry, config)
    if not uses_family_video_generator(config):
        raise RuntimeError(
            f"Internal error: _run_family_video_generator called for "
            f"video_pipeline_shape={getattr(config, 'video_pipeline_shape', '')!r} "
            f"(family={family!r})"
        )

    if ctx_exec.cancel_token.is_cancelled():
        return None

    bundle_root, w, h, steps, guidance, step_distill, _scheduler_default = (
        prepare_video_bundle_and_schedule(pipeline, 
            entry=entry,
            config=config,
            family=family,
            model_key=model_key,
            version_key=version_key,
            w=w,
            h=h,
            num_frames=num_frames,
            request=request,
            on_log=on_log,
        )
    )

    if bundle_root is None or not bundle_root.is_dir():
        raise RuntimeError(f"Video model bundle not found for {request.model!r}")

    source_video_path: str | None = None
    source_image_path: str | None = None
    image_path: str | None = None
    reference_image_paths: list[str] = []
    ref_ids = list(getattr(request, "reference_asset_ids", None) or [])
    meta_refs = getattr(request, "metadata", None) or {}
    if not ref_ids and isinstance(meta_refs.get("reference_asset_ids"), list):
        ref_ids = [str(x) for x in meta_refs["reference_asset_ids"]]

    if ref_ids:
        from backend.engine.contracts.video_runtime_contracts import resolve_video_reference_asset_paths

        reference_image_paths = resolve_video_reference_asset_paths(
            pipeline._asset_store, ref_ids,
        )

    if is_edit:
        if bool(getattr(config, "bernini_renderer", False)):
            from backend.engine.contracts.video_runtime_contracts import resolve_bernini_source_paths

            work = Path(ctx_exec.work_dir)
            source_video_path, source_image_path = resolve_bernini_source_paths(
                pipeline._asset_store,
                request,
                config,
                work_dir=work,
            )
            if on_log is not None:
                on_log(
                    "info",
                    "Bernini-R source resolved "
                    f"asset={getattr(request, 'source_asset_id', '')!r} "
                    f"video={source_video_path is not None} image={source_image_path is not None}",
                )
        else:
            if not request.source_asset_id:
                raise RuntimeError("Video edit requires source_asset_id")
            src = pipeline._asset_store.get_file_path(request.source_asset_id)
            if src is None or not src.exists():
                raise RuntimeError(f"Source image asset not found: {request.source_asset_id!r}")
            image_path = str(src)

    if on_log:
        on_log(
            "info",
            " ".join(
                [
                    f"infer model={model_key}",
                    f"family={family}",
                    f"version={version_key or 'default'}",
                    f"size={w}x{h}",
                    f"frames={num_frames}",
                    f"fps={fps}",
                    f"seed={seed}",
                    f"steps={steps}",
                    f"guidance={guidance}",
                    f"step_distill={step_distill}",
                    f"mode={'video_edit' if is_edit else 'video_generate'}",
                    "video_pipeline_shape=family_generator",
                ]
            ),
        )

    factory = _get_video_generation_factory(family)
    factory_kwargs: dict[str, Any] = {
        "config": config,
        "entry": entry,
        "version_key": version_key,
        "project_root": pipeline._project_root,
    }
    import inspect

    accepted = set(inspect.signature(factory).parameters)
    generator = factory(
        pipeline.ctx,
        bundle_root,
        **{k: v for k, v in factory_kwargs.items() if k in accepted},
    )
    generator.load()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = Path(ctx_exec.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_path = str(work / f"{model_key}_{seed}_{timestamp}.mp4")

    pipeline_graph_step("denoise", on_log)
    stage2_steps = int(getattr(config, "ltx_stage2_steps", 0) or 0)
    progress_total_steps = max(1, int(steps) + stage2_steps)
    emit_phase(on_progress, phase="generate", progress=0.05, n_steps=progress_total_steps)

    if ctx_exec.cancel_token.is_cancelled():
        return None

    prompt = (request.prompt or "").strip()
    if not prompt and not (
        isinstance(request, VideoGenerationRequest)
        and request.long_video is not None
        and (request.long_video.opening_prompt or "").strip()
    ):
        if stage2_steps > 0:
            raise RuntimeError("LTX 2.3 generation requires a non-empty prompt")
        if not bool(getattr(config, "bernini_renderer", False)):
            raise RuntimeError("Video generation requires a non-empty prompt")

    long_spec = getattr(request, "long_video", None) if not is_edit else None
    lv_fps = float(fps)
    if long_spec is not None:
        strategy = getattr(long_spec, "strategy", "latent_extend") or "latent_extend"
        if strategy == "segmented_i2v":
            raise RuntimeError(
                "long_video strategy segmented_i2v must use POST /api/videos/long-generations "
                "(task kind video.long_generation), not standard video create"
            )
        if not getattr(config, "supports_long_video", False):
            raise RuntimeError(
                f"Model {model_key!r} does not support long_video generation "
                "(registry long_video_support=false)"
            )
        from backend.engine.families.ltx.ltx_long_video import run_ltx_long_video

        lv_fps = float(request.fps) if getattr(request, "fps", None) else 24.0
        max_frames = int(getattr(config, "ltx_long_video_max_frames", 257) or 257)
        result_path = run_ltx_long_video(
            generator,
            request=request,
            spec=long_spec,
            output_path=out_path,
            width=w,
            height=h,
            fps=lv_fps,
            seed=seed,
            steps=steps,
            guidance=guidance,
            step_distill=step_distill,
            max_frames=max_frames,
            on_log=on_log,
            on_progress=on_progress,
        )
    else:
        gen_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "output_path": out_path,
            "width": w,
            "height": h,
            "num_frames": num_frames,
            "fps": float(fps),
            "seed": seed,
            "steps": steps,
            "guidance": guidance,
            "step_distill": step_distill,
            "negative_prompt": getattr(request, "negative_prompt", None) or "",
            "on_log": on_log,
            "on_progress": on_progress,
        }
        if bool(getattr(config, "bernini_renderer", False)):
            gen_kwargs.update(
                {
                    "source_video_path": source_video_path,
                    "source_image_path": source_image_path,
                    "reference_image_paths": reference_image_paths,
                    "is_edit": is_edit,
                }
            )
        else:
            gen_kwargs["image_path"] = image_path
        result_path = generator.generate_and_save(**gen_kwargs)

    if ctx_exec.cancel_token.is_cancelled():
        return None

    pipeline_graph_step("save_asset", on_log)
    emit_complete(on_progress, progress_total_steps)

    metadata = {
        "model": request.model,
        "seed": seed,
        "prompt": request.prompt,
        "negative_prompt": getattr(request, "negative_prompt", None) or "",
        "steps": steps,
        "guidance": guidance,
        "num_frames": num_frames,
        "fps": fps,
        "width": w,
        "height": h,
        "mime_type": "video/mp4",
        "video_pipeline_shape": "family_generator",
        "step_distill": step_distill,
    }
    if long_spec is not None:
        metadata["long_video"] = long_spec.model_dump()
        metadata["fps"] = int(lv_fps)
    metadata.update(work_title_metadata(getattr(request, "title", None)))
    return result_path, metadata


def execute_family_video_avatar(
    pipeline,
    request: VideoAvatarRequest,
    ctx_exec: ExecutionContext,
    *,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Shape C avatar generation — audio-driven digital human."""
    model_key, version_key = parse_model_version(request.model)
    w, h = parse_size(request.size)
    entry = pipeline._registry.require(model_key)
    family = require_entry_family(entry, model_id=model_key)
    config_cls = get_config_class(family)
    config = config_cls()
    num_frames = resolve_num_frames(pipeline, request, entry)
    fps = resolve_fps(pipeline, request, entry)
    seed = request.seed if request.seed is not None else random.randint(0, 2 ** 32 - 1)

    apply_video_registry_config_overrides(pipeline, entry, config)
    if not uses_family_video_avatar(config):
        raise RuntimeError(
            f"Internal error: execute_family_video_avatar called for "
            f"video_pipeline_shape={getattr(config, 'video_pipeline_shape', '')!r} "
            f"(family={family!r})"
        )

    if ctx_exec.cancel_token.is_cancelled():
        return None

    bundle_root = _local_bundle_root_fn(pipeline._project_root, entry, version_key or None)
    validate_bundle_graph_step(
        bundle_root,
        family=family,
        model_id=model_key,
        on_log=on_log,
        registry_entry=entry,
        project_root=pipeline._project_root,
    )
    merge_video_bundle_config(config, bundle_root)
    w, h = snap_wan_pixel_dims_if_needed(pipeline, config, w, h, on_log=on_log)
    validate_generate_geometry(pipeline, config, w, h, num_frames)

    steps_default = _registry_scalar_default_fn(entry, "steps", getattr(config, "default_infer_steps", 8))
    steps = int(request.steps) if request.steps is not None else int(steps_default)
    steps = max(1, steps)
    validate_video_avatar_params(family, entry=entry, config=config)

    if not request.reference_asset_id:
        raise RuntimeError("Video avatar requires reference_asset_id (portrait image)")
    if not request.audio_asset_id:
        raise RuntimeError("Video avatar requires audio_asset_id (voice track)")

    ref_path = pipeline._asset_store.get_file_path(request.reference_asset_id)
    if ref_path is None or not ref_path.exists():
        raise RuntimeError(f"Reference image asset not found: {request.reference_asset_id!r}")
    audio_path = pipeline._asset_store.get_file_path(request.audio_asset_id)
    if audio_path is None or not audio_path.exists():
        raise RuntimeError(f"Audio asset not found: {request.audio_asset_id!r}")

    if bundle_root is None or not bundle_root.is_dir():
        raise RuntimeError(f"Video avatar model bundle not found for {request.model!r}")

    if on_log:
        on_log(
            "info",
            " ".join(
                [
                    f"infer model={model_key}",
                    f"family={family}",
                    f"version={version_key or 'default'}",
                    f"size={w}x{h}",
                    f"frames={num_frames}",
                    f"fps={fps}",
                    f"seed={seed}",
                    f"steps={steps}",
                    "mode=video_avatar",
                    "video_pipeline_shape=family_avatar",
                ]
            ),
        )

    factory = _get_video_avatar_factory(family)
    generator = factory(
        pipeline.ctx,
        bundle_root,
        config=config,
        entry=entry,
        version_key=version_key,
    )
    generator.load()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = Path(ctx_exec.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_path = str(work / f"{model_key}_{seed}_{timestamp}.mp4")

    pipeline_graph_step("denoise", on_log)
    progress_total_steps = max(1, int(steps))
    emit_phase(on_progress, phase="generate", progress=0.05, n_steps=progress_total_steps)

    if ctx_exec.cancel_token.is_cancelled():
        return None

    result_path = generator.generate_and_save(
        prompt=(request.prompt or "").strip() or " ",
        output_path=out_path,
        reference_image_path=str(ref_path),
        audio_path=str(audio_path),
        width=w,
        height=h,
        num_frames=num_frames,
        fps=float(fps),
        seed=seed,
        steps=steps,
        negative_prompt=request.negative_prompt or "",
        on_log=on_log,
        on_progress=on_progress,
    )

    if ctx_exec.cancel_token.is_cancelled():
        return None

    pipeline_graph_step("save_asset", on_log)
    emit_complete(on_progress, progress_total_steps)

    metadata = {
        "model": request.model,
        "seed": seed,
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt or "",
        "steps": steps,
        "num_frames": num_frames,
        "fps": fps,
        "width": w,
        "height": h,
        "mime_type": "video/mp4",
        "video_pipeline_shape": "family_avatar",
        "reference_asset_id": request.reference_asset_id,
        "audio_asset_id": request.audio_asset_id,
    }
    metadata.update(work_title_metadata(getattr(request, "title", None)))
    return result_path, metadata


def prepare_t5_context(pipeline, config: Any) -> None:
    pipeline._video_config = config
    pipeline._t5_max_seq_len = video_t5_max_seq_len(config)
    pipeline._t5 = None

def initial_video_latents(pipeline, config: Any, latent_shape: tuple, seed: int | None) -> Any:
    """Sample initial video noise; Wan uses global seed for reproducible latents."""
    if seed is not None:
        if bool(getattr(config, "uses_wan_shift", False)):
            pipeline.ctx.seed_random(int(seed))
            return pipeline.ctx.randn(latent_shape, dtype=pipeline.ctx.float32())
        return pipeline.ctx.seeded_randn(latent_shape, seed, dtype=pipeline.ctx.float32())
    return pipeline.ctx.randn(latent_shape, dtype=pipeline.ctx.float32())

def denoise_video(pipeline,
    *,
    latents: Any,
    timesteps: Any,
    scheduler: Any,
    model: Any,
    txt_embeds: Any,
    neg_embeds: Any,
    guidance: float,
    config: Any,
    sigmas: Any,
    timestep_embed_schedule: list[float] | None,
    extra_cond: dict,
    rope_kw: dict,
    cfg_renorm: bool,
    cfg_renorm_min: float,
    ctx_exec: ExecutionContext,
    on_progress: Callable | None,
    on_log: Callable | None,
) -> Any | None:
    """Denoise loop — ``DiffusionParadigm`` + loaded backbone."""
    return run_video_denoise(
        self,
        model=model,
        scheduler=scheduler,
        timesteps=timesteps,
        latents=latents,
        config=config,
        guidance=guidance,
        txt_embeds=txt_embeds,
        neg_embeds=neg_embeds,
        sigmas=sigmas,
        timestep_embed_schedule=timestep_embed_schedule,
        extra_cond=extra_cond,
        rope_kw=rope_kw,
        cfg_renorm=cfg_renorm,
        cfg_renorm_min=cfg_renorm_min,
        ctx_exec=ctx_exec,
        on_progress=on_progress,
        on_log=on_log,
    )

def encode_video_text(pipeline,
    text: str,
    config: Any,
    family: str,
    bundle_root: Path | None,
    *,
    version_key: str | None = None,
    guidance: float = 1.0,
    negative_prompt: str | None = None,
) -> tuple[Any, Any | None, Any | None, Any | None]:
    """Encode one prompt. Returns ``(txt_embeds, txt_mask, txt_embeds_2, txt_mask_2)``."""
    use_cache = bool(getattr(config, "use_text_embedding_cache", True))
    encoder_id = f"{video_encoder_type(config)}:{family}:{version_key or 'default'}"
    if use_cache:
        from backend.engine.common.ops.text_embedding_cache import global_text_embedding_cache

        cached = global_text_embedding_cache().get(
            encoder_id=encoder_id,
            prompt=text,
            negative_prompt=negative_prompt,
            guidance=guidance,
        )
        if cached is not None:
            return cached  # type: ignore[return-value]

    encoder_type = video_encoder_type(config)
    if encoder_type == "t5":
        if bool(getattr(config, "t5_attention_mask", False)):
            embeds, mask = encode_t5_texts_with_mask(pipeline, [text])
            return embeds[0:1], mask[0:1], None, None
        return encode_t5(pipeline, text), None, None, None
    if bundle_root is None:
        raise RuntimeError(
            f"Video model family {family!r} with encoder_type={encoder_type!r} "
            "requires a local bundle with text encoder assets."
        )
    raw = _encode_video_prompt_fn(
        pipeline.ctx, text, encoder_type=encoder_type, bundle_root=bundle_root, config=config,
    )
    result = raw[0], raw[1], raw[2], raw[3]
    if use_cache:
        from backend.engine.common.ops.text_embedding_cache import global_text_embedding_cache

        global_text_embedding_cache().put(
            encoder_id=encoder_id,
            prompt=text,
            negative_prompt=negative_prompt,
            guidance=guidance,
            value=result,
        )
    return result

def encode_video_text_with_cfg(pipeline,
    *,
    prompt: str,
    negative_prompt: str | None,
    config: Any,
    family: str,
    bundle_root: Path | None,
    guidance: float,
    version_key: str | None = None,
) -> tuple[
    Any, Any | None, Any | None, Any | None,
    Any | None, Any | None, Any | None, Any | None,
]:
    """Encode prompt + optional CFG negative; Hunyuan batches both in one Qwen/ByT5 forward."""
    empty_neg = (None, None, None, None)
    if not prompt or config.text_dim <= 0:
        return None, None, None, None, *empty_neg

    use_cfg = bool(getattr(config, "supports_guidance", True) and guidance > 1.0)
    encoder_type = video_encoder_type(config)
    if use_cfg and bundle_root is not None:
        batch = _encode_video_hunyuan_dual_cfg_batch_fn(
            pipeline.ctx,
            prompt=prompt,
            negative_prompt=negative_prompt,
            bundle_root=bundle_root,
            config=config,
            guidance=guidance,
            encoder_type=encoder_type,
        )
        if batch is not None:
            return batch

    if use_cfg and encoder_type == "t5":
        neg_txt = video_cfg_negative_prompt(config, negative_prompt)
        if bool(getattr(config, "t5_attention_mask", False)):
            embeds, masks = encode_t5_texts_with_mask(pipeline, [prompt, neg_txt])
            return (
                embeds[0:1], masks[0:1], None, None,
                embeds[1:2], masks[1:2], None, None,
            )
        embeds = encode_t5_texts(pipeline, [prompt, neg_txt])
        return embeds[0:1], None, None, None, embeds[1:2], None, None, None

    pos = encode_video_text(
        pipeline, prompt, config, family, bundle_root,
        version_key=version_key, guidance=guidance, negative_prompt=negative_prompt,
    )
    if not use_cfg:
        return (*pos, *empty_neg)
    neg_txt = video_cfg_negative_prompt(config, negative_prompt)
    neg = encode_video_text(
        pipeline, neg_txt, config, family, bundle_root,
        version_key=version_key, guidance=guidance, negative_prompt=neg_txt,
    )
    return (*pos, *neg)

def apply_video_text_to_extra_cond(pipeline,
    extra_cond: dict[str, Any],
    *,
    txt_embeds: Any,
    txt_mask: Any | None,
    txt_embeds_2: Any | None,
    txt_mask_2: Any | None,
    neg_embeds: Any | None,
    neg_mask: Any | None,
    neg_embeds_2: Any | None,
    neg_mask_2: Any | None,
) -> dict[str, Any]:
    if txt_mask is not None:
        extra_cond["txt_attn_mask"] = txt_mask
    if txt_embeds_2 is not None:
        extra_cond["txt_embeds_2"] = txt_embeds_2
    if txt_mask_2 is not None:
        extra_cond["txt_attn_mask_2"] = txt_mask_2
    if neg_mask is not None:
        extra_cond["neg_txt_attn_mask"] = neg_mask
    if neg_embeds_2 is not None:
        extra_cond["neg_txt_embeds_2"] = neg_embeds_2
    if neg_mask_2 is not None:
        extra_cond["neg_txt_attn_mask_2"] = neg_mask_2
    return extra_cond

def create_video_scheduler(pipeline,
    *,
    config: Any,
    scheduler_name: str,
    bundle_root: Path | None,
) -> Any:
    """Instantiate the denoise scheduler; bundle may supply JSON defaults via config flags."""
    ctor_kwargs = video_scheduler_ctor_kwargs(config, scheduler_name, bundle_root)
    return get_scheduler(scheduler_name, ctx=pipeline.ctx, **ctor_kwargs)

def latent_frame_count(pipeline, config: Any, requested_pixel_frames: int) -> int:
    """Pixel timeline frames → latent timeline frames (temporal VAE compression).

    Models without ``temporal_vae_scale`` use the requested count as-is (Wan/LTX latents).
    """
    tvs = getattr(config, "temporal_vae_scale", None)
    if tvs is not None and int(tvs) > 0:
        rf = max(int(requested_pixel_frames), 1)
        return (rf - 1) // int(tvs) + 1
    return int(requested_pixel_frames)

def encode_t5_texts(pipeline, texts: list[str]) -> Any:
    """Batch T5 encode — one tokenizer + one forward for multiple prompts."""
    contract = getattr(self, "_video_config", None)
    t5_root = getattr(self, "_t5_bundle_root", None)
    if t5_root is None:
        raise RuntimeError("T5 encoding requires _t5_bundle_root")
    bundle_root = t5_root if isinstance(t5_root, Path) else Path(t5_root)
    max_seq_len = int(getattr(self, "_t5_max_seq_len", 512))
    if pipeline._t5 is None:
        pipeline._t5 = create_video_t5_encoder(
            pipeline.ctx, bundle_root, contract or object(), max_seq_len,
        )
    return pipeline._t5.encode(texts)

def release_video_t5_after_encode(pipeline,
    config: Any,
    encoder_type: str,
) -> None:
    """Drop text-encoder weights before loading DiT (T5 pipeline cache or Wan UMT5 MLX cache)."""
    if not bool(getattr(config, "release_t5_after_encode", False)):
        return
    if encoder_type == "t5" and pipeline._t5 is not None:
        pipeline._t5.release_weights()
        pipeline._t5 = None
    elif encoder_type == "wan_umt5" and hasattr(pipeline.ctx, "clear_cache"):
        pipeline.ctx.clear_cache()

def encode_t5(pipeline, text: str) -> Any:
    """T5 文本编码（视频模型目前统一使用 T5）；权重来自当前请求的 bundle，不走 Hub。"""
    return encode_t5_texts(pipeline, [text])

def encode_t5_texts_with_mask(pipeline, texts: list[str]) -> tuple[Any, Any]:
    """Batch T5 encode returning ``(hidden_states, attention_mask)`` per batch row."""
    contract = getattr(self, "_video_config", None)
    t5_root = getattr(self, "_t5_bundle_root", None)
    if t5_root is None:
        raise RuntimeError("T5 encoding requires _t5_bundle_root")
    if contract is not None and bool(getattr(contract, "uses_wan_t5_bundle", False)):
        raise RuntimeError(
            "Wan UMT5 bundle does not support encode_with_mask; disable t5_attention_mask."
        )
    bundle_root = t5_root if isinstance(t5_root, Path) else Path(t5_root)
    max_seq_len = int(getattr(self, "_t5_max_seq_len", 512))
    t5_dir, t5_tok_dir = t5_encoder_bundle_paths(bundle_root)
    if pipeline._t5 is None:
        pipeline._t5 = T5Encoder(
            pipeline.ctx, t5_dir, max_seq_len=max_seq_len, tokenizer_path=t5_tok_dir,
        )
    return pipeline._t5.encode_with_mask(texts)

def model_cache_key(pipeline, entry, version_key: str | None, num_frames: int) -> str:
    return video_model_cache_key(entry, version_key, num_frames)

def load_model(
    pipeline,
    config,
    entry,
    version_key: str | None,
    num_frames: int,
    *,
    allow_cache: bool = True,
    on_log: Callable | None = None,
) -> Any:
    """加载视频模型 — 注册表驱动，零 family 分支。"""
    family = getattr(entry, "family", "")
    return load_video_transformer(
        ctx=pipeline.ctx,
        family=family,
        config=config,
        entry=entry,
        version_key=version_key,
        project_root=pipeline._project_root,
        num_frames=num_frames,
        model_cache=pipeline._cache if allow_cache else None,
        on_log=on_log,
    )


def apply_video_lora_adapters(
    pipeline,
    family: str,
    model: Any,
    request: VideoGenerationRequest | VideoEditRequest,
    on_log: Callable[..., None] | None,
) -> None:
    adapters = getattr(request, "adapters", None) or []
    if not adapters:
        return
    base_model_id, _ = parse_model_version(request.model)
    entry = pipeline._registry.get(base_model_id)
    if entry is not None:
        lora_support = _registry_scalar_default_fn(entry, "lora_support", False)
        if not lora_support:
            raise RuntimeError(
                f"Model {base_model_id!r} does not declare LoRA support; "
                "remove adapters from the request or use a LoRA-capable base model."
            )
    from backend.engine.runtime.mlx import MLXContext

    if not isinstance(pipeline.ctx, MLXContext):
        raise RuntimeError(
            "LoRA merging for Wan video is only implemented on the MLX runtime; "
            f"current runtime is {type(pipeline.ctx).__name__}."
        )
    _merge_video_lora_adapters(
        family=family,
        model=model,
        adapters=list(adapters),
        base_model_id=base_model_id,
        project_root=pipeline._project_root,
        registry=pipeline._registry,
        ctx=pipeline.ctx,
        on_log=on_log,
    )

def vae_decode_video(pipeline,
    latents,
    entry,
    version_key,
    config: Any,
    on_post_progress: Callable[[float], None] | None = None,
    on_post_log: Callable[[str], None] | None = None,
) -> list:
    """逐帧 VAE 解码视频 latent → PIL Image 列表。"""
    backend = str(getattr(config, "video_vae_backend", "generic") or "generic")
    handler = get_video_decode_handler(backend)
    if handler is not None:
        return handler(
            ctx=pipeline.ctx,
            latents=latents,
            entry=entry,
            version_key=version_key,
            local_bundle_root=lambda entry, vk: _local_bundle_root_fn(
                pipeline._project_root, entry, vk
            ),
            registry_scalar_default=_registry_scalar_default_fn,
            on_post_progress=on_post_progress,
            on_post_log=on_post_log,
            pipeline_config=config,
        )

    from backend.engine.common.bundle.quant_inference import resolve_component_inference_weight_mode
    from backend.engine.common.bundle.safetensors_affine_quant import read_bundle_affine_bits_if_quantized
    from backend.engine.common.codecs.vae import (
        apply_flux2_latent_preprocess_if_enabled,
        create_loaded_vae_decoder,
        load_vae_weight_dict,
        read_vae_dir_config,
        release_vae_decoder_memory,
        vae_forward_to_pil,
    )

    ctx = pipeline.ctx
    bundle_root = _local_bundle_root_fn(pipeline._project_root, entry, version_key)
    vae_dir = (bundle_root / "vae") if bundle_root else None
    vae_cfg, scaling_factor, shift_factor = read_vae_dir_config(vae_dir)
    latent_cfg = int(vae_cfg.get("latent_channels", 16)) if vae_cfg else 16

    vae_weights = load_vae_weight_dict(ctx, vae_dir)
    vae_affine_root = vae_dir or Path(".")
    if not vae_weights and bool(getattr(config, "uses_ltx_flat_vae_decoder", False)) and bundle_root is not None:
        dec_path = ltx_flat_vae_decoder_file(bundle_root)
        if dec_path is not None:
            raw_dec = ctx.load_weights(str(dec_path))
            vae_weights = {f"decoder.{k}": v for k, v in raw_dec.items()}
            vae_affine_root = dec_path

    bundle_affine_bits = read_bundle_affine_bits_if_quantized(vae_weights, vae_affine_root)
    vae_inference_mode = resolve_component_inference_weight_mode(
        entry,
        version_key,
        ctx,
        component="vae",
        weight_keys=frozenset(vae_weights.keys()),
        bundle_affine_bits=bundle_affine_bits,
    )
    release_after = bool(_registry_scalar_default_fn(entry, "vae_release_after_decode", True))

    if latents.ndim == 5:
        B, C, T, H, W = latents.shape
    else:
        B, C, H, W = latents.shape
        T = 1

    sample_latent = latents[:, :, 0, :, :]
    sample_latent, vae_sf, vae_shf = apply_flux2_latent_preprocess_if_enabled(
        ctx, sample_latent, vae_cfg, vae_weights, scaling_factor, shift_factor
    )
    vae = None
    try:
        vae, _, _, _ = create_loaded_vae_decoder(
            ctx,
            sample_latent,
            vae_weights,
            vae_sf,
            vae_shf,
            default_channels=latent_cfg,
            require_conv_in=False,
            bundle_affine_bits=bundle_affine_bits,
            inference_mode=vae_inference_mode,
        )

        frames = []
        for t_idx in range(T):
            frame_latent = latents[:, :, t_idx, :, :]
            frame_latent, _, _ = apply_flux2_latent_preprocess_if_enabled(
                ctx, frame_latent, vae_cfg, vae_weights, scaling_factor, shift_factor
            )
            frames.append(vae_forward_to_pil(ctx, vae, frame_latent))

        if on_post_log:
            on_post_log(f"vae_decode {vae_inference_mode.log_label()}")
        return frames
    finally:
        if release_after:
            release_vae_decoder_memory(ctx, vae)
            if on_post_log:
                on_post_log("vae_released_after_decode=yes")

def vae_encode_frame(pipeline,
    image_tensor,
    entry,
    version_key,
    config: Any,
) -> Any:
    """VAE 编码单帧图像 → latent（用于 I2V 首帧条件）。"""
    backend = str(getattr(config, "video_vae_backend", "generic") or "generic")
    handler = get_video_encode_handler(backend)
    if handler is None:
        return None
    return handler(
        ctx=pipeline.ctx,
        image_tensor=image_tensor,
        entry=entry,
        version_key=version_key,
        local_bundle_root=lambda entry, vk: _local_bundle_root_fn(
            pipeline._project_root, entry, vk
        ),
        registry_scalar_default=_registry_scalar_default_fn,
    )

def save_video(pipeline, frames: list, output_path: str, fps: int = 16):
    """将 PIL Image 帧列表保存为 MP4 视频。"""
    from backend.engine.common.codecs.vae.video_io import save_pil_frames_to_mp4

    save_pil_frames_to_mp4(frames, output_path, fps=fps)
