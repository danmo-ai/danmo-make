"""Shared video DiT load path — VideoPipeline and v3 FamilyPlugin backbones."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.engine._transformer_registry import (
    get_video_transformer_class,
    prepare_video_transformer_weights,
)
from backend.engine.config.model_configs import get_config_class
from backend.engine.contracts import (
    inject_hunyuan_text_encoder_paths,
    inject_ltx_text_encoder_paths,
    local_bundle_root,
    merge_video_bundle_config,
    registry_scalar_default,
    resolve_version_block,
)
from backend.engine.cache import ModelCache
from backend.engine.common.bundle.quant_inference import (
    WeightInferenceMode,
    estimate_dit_cache_size_gb,
    resolve_inference_weight_mode,
)
from backend.engine.pipelines.video_bundle_layout import (
    resolve_video_transformer_weight_sources,
    wan_is_moe_bundle,
    wan_moe_expert_shards,
    wan_moe_expert_tensor_root,
)


def video_model_cache_key(
    entry: Any,
    version_key: str | None,
    num_frames: int,
    inference_mode: WeightInferenceMode | None = None,
) -> str:
    suffix = inference_mode.cache_suffix() if inference_mode is not None else ":dense"
    return f"video:{entry.id}:{version_key or 'default'}:{num_frames}{suffix}"


def resolve_video_num_frames(request: Any, entry: Any) -> int:
    if getattr(request, "num_frames", None) is not None:
        return int(request.num_frames)
    reg = registry_scalar_default(entry, "num_frames", None)
    if reg is not None:
        return int(reg)
    return 81


def uses_family_video_generator(config: Any) -> bool:
    return str(getattr(config, "video_pipeline_shape", "dit_standard") or "dit_standard") == "family_generator"


def uses_family_video_avatar(config: Any) -> bool:
    return str(getattr(config, "video_pipeline_shape", "dit_standard") or "dit_standard") == "family_avatar"


def latent_frame_count_for_video(config: Any, requested_pixel_frames: int) -> int:
    tvs = getattr(config, "temporal_vae_scale", None)
    if tvs is not None and int(tvs) > 0:
        rf = max(int(requested_pixel_frames), 1)
        return (rf - 1) // int(tvs) + 1
    return int(requested_pixel_frames)


def apply_video_registry_config_overrides(
    entry: Any,
    config: Any,
    *,
    project_root: Path,
) -> None:
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
        val = registry_scalar_default(entry, param_key, None)
        if val is not None:
            setattr(config, param_key, val)
    sg = registry_scalar_default(entry, "supports_guidance", None)
    if sg is not None:
        config.supports_guidance = bool(sg)
    sd = registry_scalar_default(entry, "step_distill", None)
    if sd is not None:
        config.step_distill = bool(sd)
    moe_boundary = registry_scalar_default(entry, "moe_boundary_step_index", None)
    if moe_boundary is not None:
        config.moe_boundary_step_index = int(moe_boundary)
    distill_ts = registry_scalar_default(entry, "wan_distill_timesteps", None)
    if isinstance(distill_ts, (list, tuple)) and distill_ts:
        config.wan_distill_timesteps = tuple(float(x) for x in distill_ts)
    hy_distill_ts = registry_scalar_default(entry, "hunyuan_distill_timesteps", None)
    if isinstance(hy_distill_ts, (list, tuple)) and hy_distill_ts:
        config.hunyuan_distill_timesteps = tuple(float(x) for x in hy_distill_ts)
    hy_distill_shift = registry_scalar_default(entry, "hunyuan_distill_shift", None)
    if hy_distill_shift is not None:
        config.hunyuan_distill_shift = float(hy_distill_shift)
    vst = registry_scalar_default(entry, "vae_spatial_tiling", None)
    if vst is not None:
        config.vae_spatial_tiling = bool(vst)
    if getattr(config, "inject_text_encoder_paths", False):
        inject_hunyuan_text_encoder_paths(entry, config, project_root)
        inject_ltx_text_encoder_paths(entry, config, project_root)


def prepare_video_config(entry: Any, family: str, bundle_root: Path, *, project_root: Path) -> Any:
    config = get_config_class(family)()
    apply_video_registry_config_overrides(entry, config, project_root=project_root)
    merge_video_bundle_config(config, bundle_root)
    return config


def _wan_moe_expert_disk_gb(entry: Any, version_key: str | None) -> float:
    """MoE bundles store two experts; budget one expert per cache slot."""
    from backend.engine.common.bundle.weights import parse_size_gb

    ver = resolve_version_block(entry, version_key)
    size_str = str((ver or {}).get("size") or getattr(entry, "raw", {}).get("size") or "10GB")
    disk_gb = parse_size_gb(size_str)
    return max(disk_gb / 2.0, 0.5)


def _load_wan_single_expert(
    *,
    ctx: Any,
    family: str,
    config: Any,
    entry: Any,
    version_key: str | None,
    num_frames: int,
    shard_paths: list[Path],
    tensor_root: Path,
    model_cache: ModelCache | None,
    cache_key_suffix: str = "",
    on_log: Any | None = None,
    disk_gb_override: float | None = None,
) -> Any:
    weights: dict[str, Any] = {}
    for shard in shard_paths:
        weights.update(ctx.load_weights(str(shard)))

    weights = prepare_video_transformer_weights(family, config, weights)

    from backend.engine.common.bundle.safetensors_affine_quant import read_bundle_affine_bits_if_quantized

    bundle_affine_bits = read_bundle_affine_bits_if_quantized(weights, tensor_root)
    inference_mode = resolve_inference_weight_mode(
        entry,
        version_key,
        ctx,
        weight_keys=frozenset(weights.keys()),
        bundle_affine_bits=bundle_affine_bits,
    )

    cache_key = video_model_cache_key(entry, version_key, num_frames, inference_mode)
    if cache_key_suffix:
        cache_key = f"{cache_key}{cache_key_suffix}"
    if model_cache is not None:
        cached = model_cache.get(cache_key)
        if cached is not None:
            return cached

    trans_cls = get_video_transformer_class(family)
    if family in ("wan", "ltx"):
        model = trans_cls(config, ctx, num_frames=num_frames)
    else:
        model = trans_cls(config, ctx)

    model.load_weights(
        list(weights.items()),
        strict=False,
        ctx=ctx,
        bundle_affine_bits=bundle_affine_bits,
        inference_mode=inference_mode,
    )
    setattr(model, "_dq_inference_mode", inference_mode)
    ctx.eval(*[p for _, p in model.parameters()])

    if on_log is not None:
        from backend.engine.common.bundle.weights import parse_size_gb

        if disk_gb_override is not None:
            disk_gb = float(disk_gb_override)
        else:
            ver = resolve_version_block(entry, version_key)
            size_str = str((ver or {}).get("size") or getattr(entry, "raw", {}).get("size") or "10GB")
            disk_gb = parse_size_gb(size_str)
        est_gb = estimate_dit_cache_size_gb(disk_gb, inference_mode)
        on_log(
            "info",
            f"[load_transformer] {inference_mode.log_label()} "
            f"disk_gb={disk_gb:.1f} cache_est_gb={est_gb:.1f}",
        )

    if model_cache is not None:
        if disk_gb_override is not None:
            disk_gb = float(disk_gb_override)
        else:
            from backend.engine.common.bundle.weights import parse_size_gb

            ver = resolve_version_block(entry, version_key)
            size_str = ""
            if ver:
                size_str = str(ver.get("size") or "")
            if not size_str:
                raw = getattr(entry, "raw", {}) or {}
                size_str = str(raw.get("size") or "10GB")
            disk_gb = parse_size_gb(size_str)
        model_cache.put(
            cache_key,
            model,
            estimate_dit_cache_size_gb(disk_gb, inference_mode),
        )
    setattr(model, "_dq_cache_key", cache_key)
    return model


def _wan_moe_expert_cache(parent_cache: ModelCache | None) -> ModelCache:
    """Dedicated two-slot cache so lazy MoE swap does not cold-reload experts each step."""
    get_limit = getattr(parent_cache, "get_memory_limit", None) if parent_cache else None
    ttl = int(getattr(parent_cache, "ttl_minutes", 30) or 30)
    reserve = float(getattr(parent_cache, "_reserve_gb", 20.0) or 20.0)
    from backend.engine.memory_policy import release_cached_model

    return ModelCache(
        get_memory_limit=get_limit or (lambda: 120.0),
        reserve_gb=reserve,
        ttl_minutes=max(1, ttl),
        max_entries=2,
        release_fn=release_cached_model,
    )


def load_wan_moe_video_transformer(
    *,
    ctx: Any,
    config: Any,
    entry: Any,
    version_key: str | None,
    project_root: Path,
    num_frames: int,
    bundle_root: Path,
    model_cache: ModelCache | None = None,
    on_log: Any | None = None,
) -> Any:
    """Load Wan 14B MoE (high/low noise experts) for step-distill or base MoE bundles."""
    from backend.engine.families.wan.moe import WanMoETransformer

    high_shards = wan_moe_expert_shards(bundle_root, "high")
    low_shards = wan_moe_expert_shards(bundle_root, "low")
    if not high_shards or not low_shards:
        raise RuntimeError(
            f"Wan MoE bundle at {bundle_root} requires safetensors or .pth under "
            "high_noise_model/ and low_noise_model/."
        )

    boundary = int(getattr(config, "moe_boundary_step_index", 2))
    lazy = bool(getattr(config, "wan_moe_lazy_experts", True))
    expert_disk_gb = _wan_moe_expert_disk_gb(entry, version_key)
    base_key_suffix = ":moe"
    held_cache_keys: dict[str, str | None] = {"high": None, "low": None}
    expert_cache = _wan_moe_expert_cache(model_cache) if lazy else model_cache

    def _load_high() -> Any:
        model = _load_wan_single_expert(
            ctx=ctx,
            family="wan",
            config=config,
            entry=entry,
            version_key=version_key,
            num_frames=num_frames,
            shard_paths=high_shards,
            tensor_root=wan_moe_expert_tensor_root(bundle_root, "high"),
            model_cache=expert_cache,
            cache_key_suffix=f"{base_key_suffix}-high",
            on_log=on_log,
            disk_gb_override=expert_disk_gb,
        )
        held_cache_keys["high"] = str(getattr(model, "_dq_cache_key", "") or "") or None
        return model

    def _load_low() -> Any:
        model = _load_wan_single_expert(
            ctx=ctx,
            family="wan",
            config=config,
            entry=entry,
            version_key=version_key,
            num_frames=num_frames,
            shard_paths=low_shards,
            tensor_root=wan_moe_expert_tensor_root(bundle_root, "low"),
            model_cache=expert_cache,
            cache_key_suffix=f"{base_key_suffix}-low",
            on_log=on_log,
            disk_gb_override=expert_disk_gb,
        )
        held_cache_keys["low"] = str(getattr(model, "_dq_cache_key", "") or "") or None
        return model

    def _release_side(side: str) -> None:
        # Drop resident reference only; keep expert_cache entries for fast re-swap.
        held_cache_keys[side] = None

    if lazy:
        if on_log is not None:
            on_log(
                "info",
                f"Wan MoE lazy swap enabled (high={len(high_shards)} shard(s), "
                f"low={len(low_shards)} shard(s), boundary_step_index={boundary}, "
                f"expert_est_gb={expert_disk_gb:.1f})",
            )
        return WanMoETransformer(
            None,
            None,
            boundary_step_index=boundary,
            config=config,
            lazy=True,
            ctx=ctx,
            load_high=_load_high,
            load_low=_load_low,
            release_high=lambda: _release_side("high"),
            release_low=lambda: _release_side("low"),
        )

    high = _load_high()
    low = _load_low()
    if on_log is not None:
        on_log(
            "info",
            f"Wan MoE loaded (high={len(high_shards)} shard(s), low={len(low_shards)} shard(s), "
            f"boundary_step_index={boundary})",
        )
    return WanMoETransformer(
        high,
        low,
        boundary_step_index=boundary,
        config=config,
        lazy=False,
        ctx=ctx,
    )


def load_video_transformer(
    *,
    ctx: Any,
    family: str,
    config: Any,
    entry: Any,
    version_key: str | None,
    project_root: Path,
    num_frames: int,
    model_cache: ModelCache | None = None,
    on_log: Any | None = None,
    bundle_root: Path | None = None,
) -> Any | None:
    """Load video transformer weights from bundle (registry-driven)."""
    if bundle_root is None:
        bundle_root = local_bundle_root(project_root, entry, version_key)
    elif not bundle_root.is_dir():
        return None
    if bundle_root is not None and wan_is_moe_bundle(bundle_root):
        return load_wan_moe_video_transformer(
            ctx=ctx,
            config=config,
            entry=entry,
            version_key=version_key,
            project_root=project_root,
            num_frames=num_frames,
            bundle_root=bundle_root,
            model_cache=model_cache,
            on_log=on_log,
        )

    tensor_root, shard_paths = resolve_video_transformer_weight_sources(
        bundle_root, family, entry.id
    )
    if tensor_root is None or not shard_paths:
        return None

    return _load_wan_single_expert(
        ctx=ctx,
        family=family,
        config=config,
        entry=entry,
        version_key=version_key,
        num_frames=num_frames,
        shard_paths=shard_paths,
        tensor_root=tensor_root,
        model_cache=model_cache,
        on_log=on_log,
    )
