"""Flux.1 DreamBooth LoRA training on MLX (DanQing bundle + engine integration)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image

from backend.core.contracts import ExecutionContext, LoraTrainingRequest
from backend.engine._transformer_registry import get_transformer_class
from backend.engine.common.codecs.vae import VAEEncoder, infer_latent_channels, prepare_vae_encoder_weight_items
from backend.engine.config.model_configs import get_config_class
from backend.engine.contracts import local_bundle_root
from backend.engine.families.flux1.flux1_dual_mlx import Flux1TextEncoder
from backend.engine.families.flux1.weights import remap_flux1_lora_keys
from backend.engine.pipelines.image_model_load import load_image_transformer
from backend.engine.training.crop import prepare_training_rgb_image, resolve_training_resolution
from backend.engine.training.dataset_store import _dataset_meta, load_training_pairs_unified
from backend.engine.training.lora_layers_mlx import (
    apply_lora_to_flux1_dit,
    list_flux1_lora_blocks,
    prepare_dit_for_lora_training,
)
from backend.engine.training.dit_training_loss_mlx import (
    CLASS_PRIOR_LATENT_COUNT,
    combine_instance_prior_loss,
    flow_match_mse,
    merge_prior_cache_tensors,
    sample_noisy_latent,
    sample_prior_latent,
)
from backend.engine.training.latent_cache_mlx import LatentCache
from backend.engine.training.lora_train_loop_mlx import run_dit_lora_train_loop
from backend.engine.training.lora_train_runtime_mlx import (
    assert_training_memory,
    parse_lora_train_runtime_config,
    save_training_checkpoint,
    split_train_val_indices,
)
from backend.engine.training.presets import merge_training_request_config, resolve_preset
from backend.engine.training.training_log import training_log, training_progress
from backend.engine.training.user_lora_registry import register_user_lora


def _log(ctx: ExecutionContext, level: str, message: str) -> None:
    training_log(ctx, level, message)


def _progress(
    ctx: ExecutionContext,
    *,
    step: int,
    total: int,
    message: str = "",
    loss: float | None = None,
    phase: str = "training",
    progress: float | None = None,
) -> None:
    training_progress(
        ctx,
        step=step,
        total=total,
        message=message,
        loss=loss,
        phase=phase,
        progress=progress,
    )


def _load_vae_encoder(ctx: Any, bundle_root: Path) -> VAEEncoder:
    from backend.engine.common.codecs.vae import load_vae_weight_dict, read_vae_dir_config

    vae_dir = bundle_root / "vae"
    vae_cfg, _, _ = read_vae_dir_config(vae_dir)
    vae_weights = load_vae_weight_dict(ctx, vae_dir)
    if not vae_weights:
        raise RuntimeError(f"VAE encode: no weights under {vae_dir}")
    scaling_factor = float(vae_cfg.get("scaling_factor", 1.0))
    shift_factor = float(vae_cfg.get("shift_factor", 0.0))
    latent_c = infer_latent_channels(vae_cfg, vae_weights)
    enc = VAEEncoder(
        latent_channels=latent_c,
        ctx=ctx,
        scaling_factor=scaling_factor,
        shift_factor=shift_factor,
    )
    enc.load_weights(prepare_vae_encoder_weight_items(vae_weights), strict=False)
    enc.cast_floating_params(ctx.bfloat16())
    return enc


def _encode_dataset_to_cache(
    *,
    cache: LatentCache,
    ctx: Any,
    pairs: list[tuple[Path, str]],
    vae: VAEEncoder,
    text_encoder: Flux1TextEncoder,
    base_model_id: str,
    train_cfg: dict[str, Any],
    preset: str | None,
    num_augmentations: int,
    dataset_id: str,
    resolution: tuple[int, int],
    exec_ctx: ExecutionContext,
    class_prompt: str | None,
    caption_mode: str = "",
) -> int:
    cache.begin(
        dataset_id=dataset_id,
        n_pairs=len(pairs),
        num_augmentations=num_augmentations,
        resolution=resolution,
        family="flux1",
        tensor_keys=["latent", "t5", "clip"],
        caption_mode=caption_mode,
    )
    _log(exec_ctx, "info", f"Encoding {len(pairs)} images × {num_augmentations} augmentations …")
    sample_idx = 0
    for img_path, prompt in pairs:
        t5, pooled = text_encoder.encode([prompt])
        mx.eval(t5, pooled)
        for aug_i in range(num_augmentations):
            arr, _ = prepare_training_rgb_image(
                img_path,
                base_model_id,
                train_cfg,
                preset=preset,
                augmentation_index=aug_i,
            )
            nchw = mx.array(arr.transpose(2, 0, 1)[None].astype("float32"))
            n11 = nchw * 2.0 - 1.0
            z = vae.encode(n11)
            if getattr(z, "ndim", 0) == 5:
                z = z[:, :, 0, :, :]
            z = z.astype(ctx.bfloat16())
            mx.eval(z)
            cache.write_sample(sample_idx, {"latent": z[0], "t5": t5, "clip": pooled})
            sample_idx += 1
    if class_prompt:
        t5p, pooledp = text_encoder.encode([class_prompt])
        mx.eval(t5p, pooledp)
        cache.write_prior({"t5": t5p, "clip": pooledp})
    return cache.finalize()


def _training_loss(
    model: Any,
    x0: mx.array,
    t5: mx.array,
    clip_pooled: mx.array,
    guidance: mx.array,
    ctx: Any,
    *,
    min_snr_gamma: float = 0.0,
    prior_t5: mx.array | None = None,
    prior_clip: mx.array | None = None,
    prior_latents: mx.array | None = None,
    prior_loss_weight: float = 0.0,
) -> mx.array:
    x_t, eps, t = sample_noisy_latent(x0, ctx)
    pred = model(
        x_t,
        timestep=0,
        txt_embeds=t5,
        pooled_embeds=clip_pooled,
        sigmas=t,
        guidance_scale=float(guidance[0]),
    )
    b = x0.shape[0]
    sigma = mx.reshape(t, (b,) + (1,) * (x0.ndim - 1)).astype(ctx.bfloat16())
    inst = flow_match_mse(pred, x0, eps, sigma=sigma, min_snr_gamma=min_snr_gamma)
    if prior_t5 is None or prior_clip is None or prior_loss_weight <= 0:
        return inst
    x0p = sample_prior_latent(x0, ctx, prior_latents=prior_latents)
    x_tp, epsp, tp = sample_noisy_latent(x0p, ctx)
    predp = model(
        x_tp,
        timestep=0,
        txt_embeds=prior_t5,
        pooled_embeds=prior_clip,
        sigmas=tp,
        guidance_scale=float(guidance[0]),
    )
    sigmap = mx.reshape(tp, (b,) + (1,) * (x0.ndim - 1)).astype(ctx.bfloat16())
    prior = flow_match_mse(predp, x0p, epsp, sigma=sigmap, min_snr_gamma=min_snr_gamma)
    return combine_instance_prior_loss(inst, prior, prior_loss_weight=prior_loss_weight)


def _save_adapter(path: Path, model: Any, rank: int, meta: dict[str, Any], optimizer: Any = None) -> None:
    if optimizer is not None:
        save_training_checkpoint(path, model, optimizer, rank=rank, meta=meta)
        return
    from backend.engine.training.lora_layers_mlx import collect_lora_safetensors

    weights = collect_lora_safetensors(model, rank=rank)
    weights.pop("lora_rank", None)
    mx.save_safetensors(str(path), weights)
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_saved_lora(path: Path, *, lora_blocks: int = -1) -> None:
    from backend.engine.common.bundle.weights import load_safetensors
    from backend.engine.config.model_configs import Flux1Config
    from backend.engine.families.flux1.transformer import Flux1Transformer
    from backend.engine.runtime.mlx import MLXContext
    from backend.engine.training.lora_layers_mlx import (
        enumerate_flux1_lora_module_paths,
        repair_indexed_lora_weights,
    )

    flat = load_safetensors(str(path))
    weights = dict(flat)
    if any(key.startswith("lora_") and ".lora_A." in key for key in weights):
        config_path = path.parent / "lora_config.json"
        blocks = lora_blocks
        if blocks == -1 and config_path.is_file():
            blocks = int(json.loads(config_path.read_text()).get("lora_blocks") or -1)
        probe = Flux1Transformer(Flux1Config(), MLXContext())
        paths = enumerate_flux1_lora_module_paths(probe, lora_blocks=blocks)
        weights = repair_indexed_lora_weights(weights, module_paths=paths)
    remapped = remap_flux1_lora_keys(weights)
    if not remapped:
        raise RuntimeError(
            f"Saved LoRA {path} has no remappable (lora_down, lora_up) pairs for Flux.1"
        )
    infer = Flux1Transformer(Flux1Config(), MLXContext())
    matched = sum(1 for tgt in remapped if f"{tgt}.weight" in infer._param_map)
    if matched == 0:
        raise RuntimeError(
            f"Saved LoRA {path}: remapped {len(remapped)} groups but none match Flux.1 DiT weights "
            "(re-export with a current Studio build or retrain)."
        )
    if matched < len(remapped):
        raise RuntimeError(
            f"Saved LoRA {path}: only {matched}/{len(remapped)} groups match Flux.1 DiT weights"
        )


def _denoise_latents_for_prompt(
    *,
    model: Any,
    text_encoder: Flux1TextEncoder,
    prompt: str,
    resolution: tuple[int, int],
    guidance: float,
    ctx: Any,
    steps: int = 20,
) -> mx.array:
    from backend.engine.common.ops.schedulers import FlowMatchEulerScheduler

    w, h = resolution
    lh, lw = h // 8, w // 8
    latents = mx.random.normal((1, 16, lh, lw), dtype=ctx.bfloat16())
    t5, pooled = text_encoder.encode([prompt])
    mx.eval(latents, t5, pooled)
    sched = FlowMatchEulerScheduler(num_train_timesteps=1000, shift=1.0, ctx=ctx)
    sched.set_timesteps(steps, mu=1.0)
    sigma_schedule = sched._sigmas
    for i, _t in enumerate(sched._timesteps):
        t_val = float(np.asarray(sched._timesteps[i]).reshape(-1)[0])
        pred = model(
            latents,
            timestep=i,
            txt_embeds=t5,
            pooled_embeds=pooled,
            sigmas=sigma_schedule,
            guidance_scale=guidance,
            timestep_embed_value=t_val,
        )
        latents = sched.step(pred, i, latents)
        mx.eval(latents)
    return latents


def _ensure_flux_class_prior_latents(
    *,
    latent_cache: LatentCache,
    model: Any,
    bundle_root: Path,
    class_prompt: str,
    resolution: tuple[int, int],
    guidance: float,
    ctx: Any,
    exec_ctx: ExecutionContext,
) -> mx.array | None:
    try:
        prior_data = latent_cache.load_prior()
        if "prior_latents" in prior_data:
            return prior_data["prior_latents"]
    except RuntimeError:
        pass

    _log(
        exec_ctx,
        "info",
        f"Generating {CLASS_PRIOR_LATENT_COUNT} class prior latents for {class_prompt!r} …",
    )
    text_encoder = Flux1TextEncoder(ctx, bundle_root)
    latents_list: list[mx.array] = []
    for seed in range(CLASS_PRIOR_LATENT_COUNT):
        mx.random.seed(seed + 17)
        z = _denoise_latents_for_prompt(
            model=model,
            text_encoder=text_encoder,
            prompt=class_prompt,
            resolution=resolution,
            guidance=guidance,
            ctx=ctx,
            steps=20,
        )
        latents_list.append(z[0])
    prior_latents = mx.stack(latents_list)
    mx.eval(prior_latents)
    merge_prior_cache_tensors(latent_cache, {"prior_latents": prior_latents})
    text_encoder.release_weights()
    ctx.clear_cache()
    return prior_latents


def _generate_progress_image(
    *,
    model: Any,
    vae_dec: Any,
    text_encoder: Flux1TextEncoder,
    prompt: str,
    resolution: tuple[int, int],
    guidance: float,
    ctx: Any,
    steps: int = 20,
) -> np.ndarray:
    """Quick sampling for training progress preview."""
    latents = _denoise_latents_for_prompt(
        model=model,
        text_encoder=text_encoder,
        prompt=prompt,
        resolution=resolution,
        guidance=guidance,
        ctx=ctx,
        steps=steps,
    )
    img = vae_dec.forward(latents)
    mx.eval(img)
    arr = np.asarray(img[0].transpose(1, 2, 0), dtype=np.float32)
    arr = np.clip((arr + 1.0) * 0.5, 0, 1)
    return (arr * 255).astype(np.uint8)


def run_flux_dreambooth_training(
    request: LoraTrainingRequest,
    exec_ctx: ExecutionContext,
    *,
    registry: Any,
    project_root: Path,
    runtime: Any,
    path_resolver: Any,
) -> dict[str, Any]:
    if getattr(runtime, "backend", None) != "mlx":
        raise RuntimeError("LoRA training requires MLX runtime (Apple Silicon)")

    from backend.utils.path_utils import get_memory_gb

    base_model_id, version_key = request.base_model.split(":", 1) if ":" in request.base_model else (request.base_model, "")
    entry = registry.require(base_model_id)
    if str(getattr(entry, "family", "")) != "flux1":
        raise RuntimeError(
            f"Training MVP supports flux1-dev only (model {base_model_id!r} is family={entry.family!r})"
        )

    preset = resolve_preset(request.preset, base_model=request.base_model)
    cfg = merge_training_request_config(request, preset)
    train_runtime = parse_lora_train_runtime_config(cfg, defaults=preset)
    mem_gb = get_memory_gb()
    assert_training_memory(base_model_id, mem_gb, qlora_bits=train_runtime.qlora_bits)

    resolution = resolve_training_resolution(base_model_id, cfg, preset=request.preset)
    progress_prompt = (request.progress_prompt or "").strip()
    if not progress_prompt:
        raise RuntimeError("progress_prompt is required for LoRA training")

    ctx = runtime
    bundle_root = local_bundle_root(project_root, entry, version_key or None)
    if bundle_root is None or not bundle_root.is_dir():
        raise RuntimeError(f"Base model {base_model_id!r} is not installed")

    work_dir = Path(exec_ctx.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = work_dir / "adapters"
    adapter_dir.mkdir(exist_ok=True)

    pairs, training_caption = load_training_pairs_unified(
        project_root,
        request.dataset_id,
        progress_prompt=progress_prompt,
        caption_mode=getattr(request, "caption_mode", None),
    )
    if len(pairs) < 3:
        raise RuntimeError("Dataset must contain at least 3 images")
    _log(exec_ctx, "info", f"DreamBooth caption: {training_caption!r}")
    dataset_meta = _dataset_meta(project_root, request.dataset_id)
    trigger_word = str(dataset_meta.get("trigger_word") or "").strip()
    resolved_caption_mode = "per_image" if len({str(p or "").strip() for _, p in pairs}) > 1 else "unified"
    _log(exec_ctx, "info", f"Caption mode: {resolved_caption_mode} ({len(pairs)} images)")

    config = get_config_class("flux1")()
    _log(exec_ctx, "info", "Loading VAE encoder and text encoders …")
    vae_enc = _load_vae_encoder(ctx, bundle_root)
    text_encoder = Flux1TextEncoder(ctx, bundle_root)

    _log(
        exec_ctx,
        "info",
        f"Training crop {resolution[0]}×{resolution[1]} (portrait-biased cover, Flux VAE grid ÷8) …",
    )
    latent_cache = LatentCache(work_dir)
    class_prompt = train_runtime.class_prompt
    if latent_cache.is_valid(
        dataset_id=request.dataset_id,
        n_pairs=len(pairs),
        num_augmentations=train_runtime.num_augmentations,
        resolution=resolution,
        family="flux1",
        n_samples=len(pairs) * train_runtime.num_augmentations,
        caption_mode=resolved_caption_mode,
    ):
        _log(exec_ctx, "info", "Reusing cached latents from work_dir/latent_cache …")
        n_samples = len(pairs) * train_runtime.num_augmentations
    else:
        n_samples = _encode_dataset_to_cache(
            cache=latent_cache,
            ctx=ctx,
            pairs=pairs,
            vae=vae_enc,
            text_encoder=text_encoder,
            base_model_id=base_model_id,
            train_cfg=cfg,
            preset=request.preset,
            num_augmentations=train_runtime.num_augmentations,
            dataset_id=request.dataset_id,
            resolution=resolution,
            exec_ctx=exec_ctx,
            class_prompt=class_prompt if train_runtime.prior_loss_weight > 0 else None,
            caption_mode=resolved_caption_mode,
        )
    del vae_enc
    text_encoder.release_weights()
    ctx.clear_cache()

    _log(exec_ctx, "info", "Loading Flux.1 DiT …")
    model = load_image_transformer(
        ctx=ctx,
        family="flux1",
        config=config,
        entry=entry,
        version_key=version_key or None,
        project_root=project_root,
        model_cache=None,
        allow_cache=False,
    )
    if model is None:
        raise RuntimeError("Failed to load Flux.1 transformer from bundle")

    guidance_val = train_runtime.guidance
    prior_latents: Any | None = None
    if train_runtime.prior_loss_weight > 0 and class_prompt:
        prior_latents = _ensure_flux_class_prior_latents(
            latent_cache=latent_cache,
            model=model,
            bundle_root=bundle_root,
            class_prompt=class_prompt,
            resolution=resolution,
            guidance=guidance_val,
            ctx=ctx,
            exec_ctx=exec_ctx,
        )

    model, train_module = prepare_dit_for_lora_training(
        model,
        apply_lora_to_flux1_dit,
        list_lora_blocks_fn=list_flux1_lora_blocks,
        rank=train_runtime.lora_rank,
        lora_blocks=train_runtime.lora_blocks,
        lora_scale=train_runtime.lora_scale,
        lora_dropout=train_runtime.lora_dropout,
        lora_module_keys=train_runtime.lora_module_keys,
        qlora_bits=train_runtime.qlora_bits,
        grad_checkpoint=train_runtime.grad_checkpoint,
        train_type=train_runtime.train_type,
    )

    guidance_val = train_runtime.guidance
    prior_t5: Any | None = None
    prior_clip: Any | None = None
    if train_runtime.prior_loss_weight > 0:
        try:
            prior_data = latent_cache.load_prior()
            prior_t5 = prior_data["t5"]
            prior_clip = prior_data["clip"]
        except RuntimeError:
            _log(exec_ctx, "warning", "Prior preservation requested but prior cache missing; disabled")

    train_pairs, val_pairs = split_train_val_indices(len(pairs), val_split=train_runtime.val_split)
    train_indices = [
        pi * train_runtime.num_augmentations + aug
        for pi in train_pairs
        for aug in range(train_runtime.num_augmentations)
    ]
    val_indices = [
        pi * train_runtime.num_augmentations + aug
        for pi in val_pairs
        for aug in range(train_runtime.num_augmentations)
    ]

    _log(exec_ctx, "info", f"Streaming {n_samples} cached latents from disk …")

    def sample_batch(indices: list[int]) -> tuple[Any, ...]:
        return latent_cache.sample_flux(indices[0])

    def loss_fn(x0: mx.array, t5: mx.array, clip_p: mx.array) -> mx.array:
        return _training_loss(
            train_module,
            x0,
            t5,
            clip_p,
            mx.full((x0.shape[0],), guidance_val, dtype=ctx.bfloat16()),
            ctx,
            min_snr_gamma=train_runtime.min_snr_gamma,
            prior_t5=prior_t5,
            prior_clip=prior_clip,
            prior_latents=prior_latents,
            prior_loss_weight=train_runtime.prior_loss_weight
            if prior_t5 is not None
            else 0.0,
        )

    def preview_at(step: int) -> None:
        _log(exec_ctx, "info", f"Generating progress preview at step {step} …")
        from backend.engine.common.codecs.vae import load_vae_weight_dict, read_vae_dir_config
        from backend.engine.common.codecs.vae.decoder import create_loaded_vae_decoder

        vae_dir = bundle_root / "vae"
        vae_cfg, _, _ = read_vae_dir_config(vae_dir)
        vae_weights = load_vae_weight_dict(ctx, vae_dir)
        preview_latent, _, _ = latent_cache.sample_flux(0)
        dec, _, _, _ = create_loaded_vae_decoder(
            ctx,
            preview_latent,
            vae_weights,
            float(vae_cfg.get("scaling_factor", 1.0)),
            float(vae_cfg.get("shift_factor", 0.0)),
            vae_cfg=vae_cfg,
        )
        te = Flux1TextEncoder(ctx, bundle_root)
        preview = _generate_progress_image(
            model=train_module,
            vae_dec=dec,
            text_encoder=te,
            prompt=progress_prompt,
            resolution=resolution,
            guidance=guidance_val,
            ctx=ctx,
            steps=train_runtime.progress_steps,
        )
        out_png = work_dir / f"{step:07d}_progress.png"
        Image.fromarray(preview).save(out_png)
        te.release_weights()
        ctx.clear_cache()

    loss_history, best_path = run_dit_lora_train_loop(
        exec_ctx=exec_ctx,
        model=model,
        train_module=train_module,
        runtime=train_runtime,
        work_dir=work_dir,
        adapter_dir=adapter_dir,
        base_model_id=base_model_id,
        n_samples=n_samples,
        sample_batch=sample_batch,
        train_indices=train_indices,
        val_indices=val_indices,
        loss_fn=loss_fn,
        on_progress_preview=preview_at,
        mlx_ctx=ctx,
    )

    final_path = adapter_dir / "final_adapters.safetensors"
    if best_path is not None and best_path.is_file():
        final_path.write_bytes(best_path.read_bytes())
        best_meta = adapter_dir / "best_adapters.json"
        if best_meta.is_file():
            final_path.with_suffix(".json").write_text(
                best_meta.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
    else:
        meta = {
            "iteration": train_runtime.iterations,
            "lora_rank": train_runtime.lora_rank,
            "base_model": base_model_id,
            "progress_prompt": progress_prompt,
            "qlora_bits": train_runtime.qlora_bits,
            "train_type": train_runtime.train_type,
        }
        _save_adapter(final_path, train_module, train_runtime.lora_rank, meta)
    if train_runtime.fuse_adapters:
        from backend.engine.training.lora_layers_mlx import collect_fused_adapter_deltas

        fused_path = adapter_dir / "fused_adapters.safetensors"
        mx.save_safetensors(str(fused_path), collect_fused_adapter_deltas(train_module))
        fused_path.with_suffix(".json").write_text(
            json.dumps({"format": "dense_delta", "base_model": base_model_id}, indent=2),
            encoding="utf-8",
        )
    _validate_saved_lora(final_path, lora_blocks=train_runtime.lora_blocks)

    output_name = (request.output_name or f"{base_model_id}-{request.dataset_id}").strip()
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in output_name)[:64]
    loras_dir = path_resolver.get_loras_dir()
    dest_dir = loras_dir / slug
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / "adapter.safetensors"
    import shutil

    shutil.copy2(final_path, dest_file)
    lora_config = {
        "lora_rank": train_runtime.lora_rank,
        "base_model": base_model_id,
        "alpha": train_runtime.lora_alpha,
        "trigger_word": trigger_word,
        "training_caption": training_caption,
    }
    (dest_dir / "lora_config.json").write_text(json.dumps(lora_config, indent=2), encoding="utf-8")

    user_lora_id = ""
    if request.auto_register:
        entry_row = register_user_lora(
            path_resolver.get_workspace_config_dir(),
            name=output_name,
            base_model=base_model_id,
            local_path=f"models/Lora/{slug}",
            trigger_word=trigger_word,
            lora_rank=train_runtime.lora_rank,
            task_id=exec_ctx.task_id,
        )
        user_lora_id = entry_row["id"]

    _log(exec_ctx, "success", f"Training complete → {dest_file}")
    return {
        "adapter_path": str(dest_file),
        "user_lora_id": user_lora_id,
        "output_name": slug,
        "loss_history": loss_history,
        "training_caption": training_caption,
    }
