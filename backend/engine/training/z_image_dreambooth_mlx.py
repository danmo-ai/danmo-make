"""Z-Image Base / Turbo DreamBooth LoRA training on MLX."""

from __future__ import annotations

import json
import random
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image

from backend.core.contracts import ExecutionContext, LoraTrainingRequest, LogEvent, ProgressEvent
from backend.engine._transformer_registry import _instantiate_image_text_encoder, get_text_encoder
from backend.engine.common.codecs.vae import VAEEncoder, infer_latent_channels, prepare_vae_encoder_weight_items
from backend.engine.config.model_configs import apply_image_registry_config_overrides, get_config_class
from backend.engine.contracts import local_bundle_root
from backend.engine.contracts.pipeline_registry import registry_scalar_default
from backend.engine.families.z_image.weights import remap_zimage_lora_keys
from backend.engine.pipelines.image_model_load import load_image_transformer
from backend.engine.training.crop import (
    prepare_training_rgb_image,
    resolve_training_resolution,
    training_allows_flip,
)
from backend.engine.training.dataset_store import (
    _dataset_meta,
    load_training_pairs_unified,
    skipped_training_log_line,
    training_dataset_fingerprint,
)
from backend.engine.training.face_crop import face_crop_cache_key, plan_training_face_crops
from backend.engine.training.flux_dreambooth_mlx import _load_vae_encoder, _log, _progress, _save_adapter
from backend.engine.training.lora_layers_mlx import (
    apply_lora_to_zimage_dit,
    assert_no_trainable_assistant,
    list_zimage_lora_blocks,
    prepare_dit_for_lora_training,
)
from backend.engine.training.dit_training_loss_mlx import (
    CLASS_PRIOR_LATENT_COUNT,
    combine_instance_prior_loss,
    flow_match_mse,
    merge_prior_cache_tensors,
    caption_dropout_applies,
    sample_noisy_latent_shifted,
    sample_shifted_sigmas,
    sample_noisy_latent_turbo,
    sample_prior_latent,
    sample_turbo_sigmas,
    turbo_training_sigmas,
)
from backend.engine.training.latent_cache_mlx import LatentCache
from backend.engine.training.lora_train_loop_mlx import preserve_training_random_streams, run_dit_lora_train_loop
from backend.engine.training.lora_train_runtime_mlx import (
    assert_training_memory,
    parse_lora_train_runtime_config,
    split_train_val_indices,
)
from backend.engine.training.presets import merge_training_request_config, resolve_preset
from backend.engine.training.user_lora_registry import register_user_lora

from backend.engine.training.z_image_turbo_adapter_mlx import (
    install_zimage_turbo_training_assistant,
    resolve_zimage_turbo_training_adapter_path,
    TurboTrainingAssistantHandle,
)

_Z_IMAGE_TRAINABLE_IDS = frozenset({"z-image", "z-image-turbo"})

# Fallback when neither preset nor request sets ``train_sigma_shift``: the Z-Image base
# inference shift (registry ``scheduler_shift``, default 6.0).
_Z_IMAGE_BASE_SIGMA_SHIFT = 6.0


def _model_id(base_model: str) -> str:
    return (base_model or "").split(":", 1)[0].strip()


def _is_zimage_turbo(base_model_id: str) -> bool:
    return _model_id(base_model_id) == "z-image-turbo"


def _load_zimage_text_encoder(ctx: Any, bundle_root: Path, config: Any) -> Any:
    enc_cls = get_text_encoder("z_image")
    return _instantiate_image_text_encoder(
        ctx,
        enc_cls,
        encoder_type="z_image",
        bundle_root=bundle_root,
        config=config,
        enc_kwargs={
            "max_seq_len": getattr(config, "max_seq_len", 512),
            "hidden_state_layers": getattr(config, "text_encoder_out_layers", None),
            "enable_thinking": getattr(config, "enable_thinking", True),
        },
    )


def _encode_dataset_to_cache(
    *,
    cache: LatentCache,
    ctx: Any,
    pairs: list[tuple[Path, str]],
    vae: VAEEncoder,
    text_encoder: Any,
    base_model_id: str,
    train_cfg: dict[str, Any],
    preset: str | None,
    num_augmentations: int,
    dataset_id: str,
    resolution: tuple[int, int],
    exec_ctx: ExecutionContext,
    class_prompt: str | None,
    caption_mode: str = "",
    allow_flip: bool = True,
    crop_windows: dict[Path, tuple[int, int, int, int] | None] | None = None,
    crop_policy: str = "",
    dataset_fingerprint: str = "",
) -> int:
    total_samples = len(pairs) * num_augmentations
    cache.begin(
        dataset_id=dataset_id,
        n_pairs=len(pairs),
        num_augmentations=num_augmentations,
        resolution=resolution,
        family="z_image",
        tensor_keys=["latent", "cap"],
        caption_mode=caption_mode,
        crop_policy=crop_policy,
        dataset_fingerprint=dataset_fingerprint,
    )
    _log(exec_ctx, "info", f"Encoding {len(pairs)} images × {num_augmentations} augmentations …")
    _progress(
        exec_ctx,
        step=0,
        total=1,
        message=f"Encoding 0/{total_samples} samples …",
        phase="encoding",
        progress=0.02,
    )
    sample_idx = 0
    for img_path, prompt in pairs:
        cap = text_encoder.encode([prompt])
        mx.eval(cap)
        for aug_i in range(num_augmentations):
            arr, _ = prepare_training_rgb_image(
                img_path,
                base_model_id,
                train_cfg,
                preset=preset,
                augmentation_index=aug_i,
                allow_flip=allow_flip,
                crop_window=(crop_windows or {}).get(img_path),
            )
            nchw = mx.array(arr.transpose(2, 0, 1)[None].astype("float32"))
            n11 = nchw * 2.0 - 1.0
            z = vae.encode(n11)
            if getattr(z, "ndim", 0) == 5:
                z = z[:, :, 0, :, :]
            z = z.astype(ctx.bfloat16())
            mx.eval(z)
            cache.write_sample(sample_idx, {"latent": z[0], "cap": cap})
            sample_idx += 1
            if sample_idx == total_samples or sample_idx % max(1, total_samples // 8) == 0:
                frac = 0.02 + 0.08 * (sample_idx / max(total_samples, 1))
                _progress(
                    exec_ctx,
                    step=0,
                    total=1,
                    message=f"Encoding {sample_idx}/{total_samples} samples …",
                    phase="encoding",
                    progress=frac,
                )
    if class_prompt:
        prior_cap = text_encoder.encode([class_prompt])
        mx.eval(prior_cap)
        cache.write_prior({"cap": prior_cap})
    return cache.finalize()


def _ensure_zimage_uncond_caption(cache: LatentCache, text_encoder: Any) -> Any:
    """Cache the empty-prompt embedding used by caption dropout."""
    try:
        return cache.load_prior(name="uncond")["cap"]
    except RuntimeError:
        cap = text_encoder.encode([""])
        mx.eval(cap)
        cache.write_prior({"cap": cap}, name="uncond")
        return cap


def _training_loss(
    model: Any,
    x0: mx.array,
    cap: mx.array,
    ctx: Any,
    *,
    min_snr_gamma: float = 0.0,
    prior_cap: mx.array | None = None,
    prior_latents: mx.array | None = None,
    prior_loss_weight: float = 0.0,
    turbo: bool = False,
    turbo_infer_steps: int = 9,
    timestep_low: int = 4,
    timestep_high: int = 9,
    timestep_bias: str = "uniform",
    resolution: tuple[int, int] = (512, 512),
    sigma_shift: float = 1.0,
    sigma_bias: str = "uniform",
) -> mx.array:
    if turbo:
        x_t, eps, t = sample_noisy_latent_turbo(
            x0,
            ctx,
            infer_steps=turbo_infer_steps,
            timestep_low=timestep_low,
            timestep_high=timestep_high,
            width=resolution[0],
            height=resolution[1],
            timestep_bias=timestep_bias,
        )
    else:
        x_t, eps, t = sample_noisy_latent_shifted(
            x0, ctx, sigma_shift=sigma_shift, sigma_bias=sigma_bias
        )
    pred = model(x_t, timestep=0, txt_embeds=cap, sigmas=t)
    b = x0.shape[0]
    sigma = mx.reshape(t, (b,) + (1,) * (x0.ndim - 1)).astype(ctx.bfloat16())
    inst = flow_match_mse(pred, x0, eps, sigma=sigma, min_snr_gamma=min_snr_gamma)
    if prior_cap is None or prior_loss_weight <= 0:
        return inst
    x0p = sample_prior_latent(x0, ctx, prior_latents=prior_latents)
    if turbo:
        x_tp, epsp, tp = sample_noisy_latent_turbo(
            x0p,
            ctx,
            infer_steps=turbo_infer_steps,
            timestep_low=timestep_low,
            timestep_high=timestep_high,
            width=resolution[0],
            height=resolution[1],
            timestep_bias=timestep_bias,
        )
    else:
        x_tp, epsp, tp = sample_noisy_latent_shifted(
            x0p, ctx, sigma_shift=sigma_shift, sigma_bias=sigma_bias
        )
    predp = model(x_tp, timestep=0, txt_embeds=prior_cap, sigmas=tp)
    sigmap = mx.reshape(tp, (b,) + (1,) * (x0.ndim - 1)).astype(ctx.bfloat16())
    prior = flow_match_mse(predp, x0p, epsp, sigma=sigmap, min_snr_gamma=min_snr_gamma)
    return combine_instance_prior_loss(inst, prior, prior_loss_weight=prior_loss_weight)


@contextmanager
def _temporary_inference_adapters(
    model: Any,
    *,
    project_root: Path,
    registry: Any,
    ctx: Any,
) -> Any:
    """Temporarily merge Turbo SFT's DistillPatch, never the trainable user LoRA."""
    from backend.catalog.lora_meta import Z_IMAGE_DISTILLPATCH_ADAPTER_ID
    from backend.engine.families.z_image.lora_mlx import merge_z_image_lora_adapters

    inner = getattr(model, "_inner", model)
    param_map = inner._param_map
    snapshot = {key: value for key, value in param_map.items() if isinstance(value, mx.array)}
    # Detached base tensors are safe to mutate while ``value_and_grad`` is off. Restore
    # the original objects afterward so the optimizer and trainable LoRA stay untouched.
    for key, value in snapshot.items():
        param_map[key] = mx.array(value)
    try:
        merge_z_image_lora_adapters(
            model,
            [{"id": Z_IMAGE_DISTILLPATCH_ADAPTER_ID, "weight": 1.0}],
            base_model_id="z-image-turbo",
            project_root=project_root,
            registry=registry,
            ctx=ctx,
        )
        yield
    finally:
        param_map.update(snapshot)


def _validate_saved_lora(path: Path) -> None:
    from backend.engine.common.bundle.weights import load_safetensors
    from backend.engine.config.model_configs import ZImageConfig
    from backend.engine.families.z_image.transformer import ZImageTransformer
    from backend.engine.runtime.mlx import MLXContext
    from backend.engine.training.lora_layers_mlx import (
        enumerate_zimage_lora_module_paths,
        repair_indexed_lora_weights,
    )

    flat = load_safetensors(str(path))
    weights = dict(flat)
    if any(key.endswith(".delta.weight") for key in weights):
        infer = ZImageTransformer(ZImageConfig(), MLXContext())
        dense_modules = [k[: -len(".delta.weight")] for k in weights if k.endswith(".delta.weight")]
        matched = sum(1 for m in dense_modules if f"{m}.weight" in infer._param_map)
        if matched == 0:
            raise RuntimeError(
                f"Saved LoRA {path}: dense deltas present but none match Z-Image DiT weights"
            )
        if matched < len(dense_modules):
            raise RuntimeError(
                f"Saved LoRA {path}: only {matched}/{len(dense_modules)} dense deltas match Z-Image DiT"
            )
        return
    if any(key.startswith("lora_") and ".lora_A." in key for key in weights):
        config_path = path.parent / "lora_config.json"
        lora_blocks = -1
        if config_path.is_file():
            lora_blocks = int(json.loads(config_path.read_text()).get("lora_blocks") or -1)
        probe = ZImageTransformer(ZImageConfig(), MLXContext())
        paths = enumerate_zimage_lora_module_paths(probe, lora_blocks=lora_blocks)
        weights = repair_indexed_lora_weights(weights, module_paths=paths)
    remapped = remap_zimage_lora_keys(weights)
    if not remapped:
        raise RuntimeError(
            f"Saved LoRA {path} has no remappable (lora_A, lora_B) pairs for Z-Image"
        )
    infer = ZImageTransformer(ZImageConfig(), MLXContext())
    matched = sum(1 for tgt in remapped if f"{tgt}.weight" in infer._param_map)
    if matched == 0:
        raise RuntimeError(
            f"Saved LoRA {path}: remapped {len(remapped)} groups but none match Z-Image DiT weights "
            "(re-export with a current Studio build or retrain)."
        )
    if matched < len(remapped):
        raise RuntimeError(
            f"Saved LoRA {path}: only {matched}/{len(remapped)} groups match Z-Image DiT weights"
        )


def _denoise_latents_for_prompt(
    *,
    model: Any,
    text_encoder: Any,
    prompt: str,
    resolution: tuple[int, int],
    ctx: Any,
    steps: int = 20,
    turbo: bool = False,
    guidance: float = 4.0,
    base_sigma_shift: float = 6.0,
) -> mx.array:
    from backend.engine.common.ops.schedulers import FlowMatchEulerScheduler, LinearScheduler

    w, h = resolution
    lh, lw = h // 8, w // 8
    latents = mx.random.normal((1, 16, lh, lw), dtype=ctx.bfloat16())
    cap = text_encoder.encode([prompt])
    mx.eval(latents, cap)
    if turbo:
        sched = LinearScheduler(num_train_timesteps=1000, ctx=ctx)
        sched.set_timesteps(
            steps,
            image_width=w,
            image_height=h,
            requires_sigma_shift=True,
        )
        sigma_schedule = sched._sigmas
    else:
        image_seq_len = lh * lw
        sched = FlowMatchEulerScheduler(
            num_train_timesteps=1000, shift=base_sigma_shift, ctx=ctx
        )
        sched.set_timesteps(
            steps,
            image_seq_len=image_seq_len,
            scheduler_shift=float(base_sigma_shift),
            use_empirical_mu=False,
        )
        sigma_schedule = sched._sigmas
    uncond = None if turbo or float(guidance) == 0.0 else text_encoder.encode([""])
    for i, _t in enumerate(sched._timesteps):
        pred = model(latents, timestep=i, txt_embeds=cap, sigmas=sigma_schedule)
        if uncond is not None:
            uncond_pred = model(latents, timestep=i, txt_embeds=uncond, sigmas=sigma_schedule)
            pred = model.combine_cfg_noise(pred, uncond_pred, float(guidance))
        latents = sched.step(pred, i, latents)
        mx.eval(latents)
    return latents


def _ensure_zimage_class_prior_latents(
    *,
    latent_cache: LatentCache,
    model: Any,
    bundle_root: Path,
    config: Any,
    class_prompt: str,
    resolution: tuple[int, int],
    ctx: Any,
    exec_ctx: ExecutionContext,
    turbo: bool = False,
    turbo_steps: int = 8,
) -> mx.array | None:
    # Optional prior generation is not a source of training randomness.
    with preserve_training_random_streams():
        return _ensure_zimage_class_prior_latents_isolated(
            latent_cache=latent_cache,
            model=model,
            bundle_root=bundle_root,
            config=config,
            class_prompt=class_prompt,
            resolution=resolution,
            ctx=ctx,
            exec_ctx=exec_ctx,
            turbo=turbo,
            turbo_steps=turbo_steps,
        )


def _ensure_zimage_class_prior_latents_isolated(
    *,
    latent_cache: LatentCache,
    model: Any,
    bundle_root: Path,
    config: Any,
    class_prompt: str,
    resolution: tuple[int, int],
    ctx: Any,
    exec_ctx: ExecutionContext,
    turbo: bool = False,
    turbo_steps: int = 8,
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
    text_encoder = _load_zimage_text_encoder(ctx, bundle_root, config)
    latents_list: list[mx.array] = []
    for seed in range(CLASS_PRIOR_LATENT_COUNT):
        mx.random.seed(seed + 17)
        z = _denoise_latents_for_prompt(
            model=model,
            text_encoder=text_encoder,
            prompt=class_prompt,
            resolution=resolution,
            ctx=ctx,
            steps=turbo_steps if turbo else 20,
            turbo=turbo,
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
    text_encoder: Any,
    prompt: str,
    resolution: tuple[int, int],
    ctx: Any,
    steps: int = 20,
    turbo: bool = False,
    guidance: float = 4.0,
    base_sigma_shift: float = 6.0,
) -> np.ndarray:
    latents = _denoise_latents_for_prompt(
        model=model,
        text_encoder=text_encoder,
        prompt=prompt,
        resolution=resolution,
        ctx=ctx,
        steps=steps,
        turbo=turbo,
        guidance=guidance,
        base_sigma_shift=base_sigma_shift,
    )
    img = vae_dec.forward(latents)
    mx.eval(img)
    arr = np.asarray(img[0].transpose(1, 2, 0), dtype=np.float32)
    arr = np.clip((arr + 1.0) * 0.5, 0, 1)
    return (arr * 255).astype(np.uint8)


def _fmt_floats(values: Any, fmt: str = "%.3f") -> str:
    import numpy as np

    return "[" + ", ".join(fmt % float(v) for v in np.asarray(values).reshape(-1)) + "]"


def _log_training_sigma_distribution(
    exec_ctx: ExecutionContext,
    ctx: Any,
    *,
    is_turbo: bool,
    sigma_shift: float,
    resolution: tuple[int, int],
    train_runtime: Any,
) -> None:
    """Log the σ distribution training actually samples (confirms base shift / turbo band)."""
    import numpy as np

    n = 4096
    try:
        if is_turbo:
            sigmas = turbo_training_sigmas(
                ctx,
                infer_steps=train_runtime.turbo_infer_steps,
                width=resolution[0],
                height=resolution[1],
            )
            s = sample_turbo_sigmas(
                ctx,
                n,
                infer_steps=train_runtime.turbo_infer_steps,
                timestep_low=train_runtime.timestep_low,
                timestep_high=train_runtime.timestep_high,
                width=resolution[0],
                height=resolution[1],
                timestep_bias=train_runtime.timestep_bias,
            )
            mx.eval(sigmas, s)
            header = (
                f"turbo continuous band[{train_runtime.timestep_low}-{train_runtime.timestep_high}] "
                f"of {train_runtime.turbo_infer_steps} steps bias={train_runtime.timestep_bias} "
                f"inference_σ={_fmt_floats(sigmas)}"
            )
        else:
            s = sample_shifted_sigmas(
                ctx,
                n,
                sigma_shift=sigma_shift,
                sigma_bias=train_runtime.sigma_bias,
            )
            mx.eval(s)
            header = (
                f"base static shift={sigma_shift:g} bias={train_runtime.sigma_bias}"
            )
    except Exception as e:  # noqa: BLE001 — diagnostics must never break training
        _log(exec_ctx, "warning", f"[diag] σ distribution probe skipped: {e}")
        return

    arr = np.asarray(s).reshape(-1).astype(np.float64)
    pct = np.percentile(arr, [5, 25, 50, 75, 95])
    frac_lo = float((arr < 0.3).mean())
    frac_mid = float(((arr >= 0.3) & (arr <= 0.7)).mean())
    frac_hi = float((arr > 0.7).mean())
    _log(
        exec_ctx,
        "info",
        f"[diag] train σ dist ({header}): "
        f"p5/25/50/75/95={_fmt_floats(pct)} "
        f"frac σ<0.3={frac_lo:.2f} 0.3-0.7={frac_mid:.2f} >0.7={frac_hi:.2f} "
        "(base identity lives in high σ; skin texture in low σ)",
    )


def _log_noise_level_loss(
    exec_ctx: ExecutionContext,
    model: Any,
    latent_cache: LatentCache,
    ctx: Any,
    *,
    label: str,
    sample_idx: int = 0,
) -> None:
    """Off-graph per-σ reconstruction loss probe (reveals which noise band is under-fit)."""
    try:
        x0, cap = latent_cache.sample_z_image(sample_idx)
    except Exception as e:  # noqa: BLE001
        # Diagnostics must never make an otherwise valid training run fail.
        _log(exec_ctx, "warning", f"[diag] per-σ loss probe skipped ({label}): {e}")
        return
    levels = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
    mx.random.seed(4242)
    eps = mx.random.normal(x0.shape, dtype=ctx.bfloat16())
    parts: list[str] = []
    try:
        for sv in levels:
            sig_col = mx.reshape(
                mx.full((x0.shape[0],), sv, dtype=ctx.float32()),
                (x0.shape[0],) + (1,) * (x0.ndim - 1),
            ).astype(ctx.bfloat16())
            x_t = mx.stop_gradient((1.0 - sig_col) * x0 + sig_col * eps)
            pred = model(
                x_t,
                timestep=0,
                txt_embeds=cap,
                sigmas=mx.full((x0.shape[0],), sv, dtype=ctx.float32()),
            )
            loss = mx.mean(mx.square(pred + x0 - eps))
            mx.eval(loss)
            parts.append(f"σ={sv:.1f}:{float(loss.item()):.3f}")
    except Exception as e:  # noqa: BLE001
        _log(exec_ctx, "warning", f"[diag] per-σ loss probe failed ({label}): {e}")
        return
    _log(exec_ctx, "info", f"[diag] per-σ recon loss [{label}] sample#{sample_idx}: " + " ".join(parts))


def _log_zimage_latent_caption_stats(
    exec_ctx: ExecutionContext,
    latent_cache: LatentCache,
    ctx: Any,
    *,
    sample_idx: int = 0,
) -> None:
    """Log cached latent + caption statistics (catches VAE scaling / caption issues)."""
    try:
        x0, cap = latent_cache.sample_z_image(sample_idx)
        mx.eval(x0, cap)
        lat_mean = float(mx.mean(x0.astype(ctx.float32())).item())
        lat_std = float(mx.mean(mx.square(x0.astype(ctx.float32()) - lat_mean)).item()) ** 0.5
        lat_min = float(mx.min(x0.astype(ctx.float32())).item())
        lat_max = float(mx.max(x0.astype(ctx.float32())).item())
        cap_norm = float(mx.mean(mx.abs(cap.astype(ctx.float32()))).item())
        _log(
            exec_ctx,
            "info",
            f"[diag] latent#{sample_idx} shape={tuple(x0.shape)} "
            f"mean={lat_mean:.3f} std={lat_std:.3f} min={lat_min:.3f} max={lat_max:.3f} | "
            f"cap shape={tuple(cap.shape)} mean|x|={cap_norm:.3f} "
            "(latent std≈1 expected; near-0 cap ⇒ empty caption/text-encoder issue)",
        )
    except Exception as e:  # noqa: BLE001
        _log(exec_ctx, "warning", f"[diag] latent/caption stats skipped: {e}")


def run_z_image_dreambooth_training(
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

    base_model_id, version_key = (
        request.base_model.split(":", 1) if ":" in request.base_model else (request.base_model, "")
    )
    mid = _model_id(base_model_id)
    if mid not in _Z_IMAGE_TRAINABLE_IDS:
        raise RuntimeError(
            f"LoRA training supports Z-Image Base/Turbo ({sorted(_Z_IMAGE_TRAINABLE_IDS)!r}) only "
            f"(got {base_model_id!r})"
        )
    is_turbo = _is_zimage_turbo(base_model_id)
    preset_name = str(request.preset or "standard").strip().lower()
    is_turbo_sft = is_turbo and preset_name == "sft"

    entry = registry.require(mid)
    if str(getattr(entry, "family", "")) != "z_image":
        raise RuntimeError(
            f"Z-Image training runner expects family z_image (model {base_model_id!r} is {entry.family!r})"
        )

    preset = resolve_preset(request.preset, base_model=request.base_model)
    cfg = merge_training_request_config(request, preset)
    train_runtime = parse_lora_train_runtime_config(cfg, defaults=preset)

    # Base σ sampling is independent from the Base inference scheduler shift.
    # The latter still controls Base progress previews; Turbo samples its own inference band.
    if is_turbo:
        base_sigma_shift = 1.0
    elif train_runtime.train_sigma_shift is not None:
        base_sigma_shift = float(train_runtime.train_sigma_shift)
    else:
        base_sigma_shift = float(
            registry_scalar_default(entry, "scheduler_shift", _Z_IMAGE_BASE_SIGMA_SHIFT)
        )
    base_preview_shift = float(
        registry_scalar_default(entry, "scheduler_shift", _Z_IMAGE_BASE_SIGMA_SHIFT)
    )
    if mid == "z-image" and (
        train_runtime.prior_loss_weight > 0 or train_runtime.class_prompt
    ):
        raise RuntimeError(
            "Z-Image LoRA training does not support prior preservation; "
            "omit class_prompt and keep prior_loss_weight at 0."
        )
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
    skipped_note = skipped_training_log_line(project_root, request.dataset_id)
    if skipped_note:
        _log(exec_ctx, "info", skipped_note)
    dataset_meta = _dataset_meta(project_root, request.dataset_id)
    resolved_caption_mode = "unified"
    sample_cap = str(pairs[0][1] or "").strip() if pairs else ""
    _log(
        exec_ctx,
        "info",
        f"Caption mode: unified ({len(pairs)} images); "
        f"sample_caption={sample_cap[:120]!r}{'…' if len(sample_cap) > 120 else ''}",
    )

    config = get_config_class("z_image")()
    apply_image_registry_config_overrides(entry, config)
    _log(exec_ctx, "info", "Loading VAE encoder and Z-Image text encoder …")
    _progress(
        exec_ctx,
        step=0,
        total=1,
        message="Loading VAE and text encoder …",
        phase="loading_model",
        progress=0.01,
    )
    vae_enc = _load_vae_encoder(ctx, bundle_root)
    text_encoder = _load_zimage_text_encoder(ctx, bundle_root, config)

    _log(
        exec_ctx,
        "info",
        f"Training crop {resolution[0]}×{resolution[1]} (portrait-biased cover, Z-Image VAE grid ÷8) …",
    )
    latent_cache = LatentCache(work_dir)
    class_prompt = train_runtime.class_prompt
    allow_flip = training_allows_flip(dataset_meta)
    if not allow_flip:
        _log(exec_ctx, "info", "Concept dataset: horizontal flip augmentation disabled (faces are asymmetric)")
    from backend.engine.training.training_preflight import plan_and_write_training_crops

    face_plans = plan_and_write_training_crops(
        pairs=pairs,
        project_root=project_root,
        resolution=resolution,
        mode=train_runtime.face_crop,
        dataset_meta=dataset_meta,
        work_dir=work_dir,
        on_log=lambda level, msg: _log(exec_ctx, level, msg),
    )
    crop_windows = {p: plan.window for p, plan in face_plans.items()}
    crop_policy = face_crop_cache_key(
        face_plans,
        mode=train_runtime.face_crop,
        kind=str(dataset_meta.get("kind") or "concept"),
        allow_flip=allow_flip,
    )
    from backend.engine.training.dataset_store import training_dataset_fingerprint

    dataset_fingerprint = training_dataset_fingerprint(pairs)

    def _run_encode() -> int:
        return _encode_dataset_to_cache(
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
            allow_flip=allow_flip,
            crop_windows=crop_windows,
            crop_policy=crop_policy,
            dataset_fingerprint=dataset_fingerprint,
        )

    if latent_cache.is_valid(
        dataset_id=request.dataset_id,
        n_pairs=len(pairs),
        num_augmentations=train_runtime.num_augmentations,
        resolution=resolution,
        family="z_image",
        n_samples=len(pairs) * train_runtime.num_augmentations,
        caption_mode=resolved_caption_mode,
        crop_policy=crop_policy,
        dataset_fingerprint=dataset_fingerprint,
    ):
        _log(exec_ctx, "info", "Reusing cached latents from work_dir/latent_cache …")
        n_samples = len(pairs) * train_runtime.num_augmentations
    else:
        n_samples = _run_encode()
    uncond_cap = None
    if train_runtime.caption_dropout > 0:
        uncond_cap = _ensure_zimage_uncond_caption(latent_cache, text_encoder)
        _log(
            exec_ctx,
            "info",
            f"Caption dropout {train_runtime.caption_dropout:.2f}: "
            "empty prompt on that fraction of steps so scene words cannot explain the face",
        )
    del vae_enc
    text_encoder.release_weights()
    ctx.clear_cache()

    _log(exec_ctx, "info", "Loading Z-Image DiT …")
    _progress(
        exec_ctx,
        step=0,
        total=1,
        message="Loading Z-Image DiT …",
        phase="loading_model",
        progress=0.10,
    )
    model = load_image_transformer(
        ctx=ctx,
        family="z_image",
        config=config,
        entry=entry,
        version_key=version_key or None,
        project_root=project_root,
        model_cache=None,
        allow_cache=False,
    )
    if model is None:
        raise RuntimeError("Failed to load Z-Image transformer from bundle")

    training_assistant: TurboTrainingAssistantHandle | None = None

    model, train_module = prepare_dit_for_lora_training(
        model,
        apply_lora_to_zimage_dit,
        list_lora_blocks_fn=list_zimage_lora_blocks,
        rank=train_runtime.lora_rank,
        lora_blocks=train_runtime.lora_blocks,
        lora_scale=train_runtime.lora_scale,
        lora_dropout=train_runtime.lora_dropout,
        lora_module_keys=train_runtime.lora_module_keys,
        qlora_bits=train_runtime.qlora_bits,
        grad_checkpoint=train_runtime.grad_checkpoint,
        train_type=train_runtime.train_type,
    )

    if is_turbo_sft:
        from backend.catalog.lora_meta import distillpatch_lora_installed

        if not distillpatch_lora_installed(project_root, registry):
            raise RuntimeError(
                "Turbo SFT requires the Z-Image Turbo DistillPatch LoRA for matching progress previews "
                "and inference. Install 'z-image-turbo-distillpatch-lora:bf16' from Models → LoRA."
            )

    if is_turbo and not is_turbo_sft:
        from backend.engine.families.z_image.lora_mlx import _repair_indexed_zimage_weights

        adapter_path = resolve_zimage_turbo_training_adapter_path(project_root)
        _log(exec_ctx, "info", f"Loading Z-Image-Turbo training assistant from {adapter_path} …")
        training_assistant = install_zimage_turbo_training_assistant(
            model,
            adapter_path,
            ctx,
            repair_indexed_weights=_repair_indexed_zimage_weights,
        )
        # The assistant must not receive gradients: otherwise AdamW writes the subject into the
        # (larger) assistant, which is switched off at inference and never exported.
        lora_param_count = assert_no_trainable_assistant(train_module)
        _log(
            exec_ctx,
            "info",
            f"Training assistant attached ({training_assistant.count} layers, frozen; "
            f"base DiT weights unchanged; trainable LoRA params={lora_param_count:,})",
        )
        if train_runtime.turbo_assistant_off_prob > 0:
            _log(
                exec_ctx,
                "warning",
                f"turbo_assistant_off_prob={train_runtime.turbo_assistant_off_prob:g}: training part-time "
                "on the raw distilled model spends LoRA capacity on re-distillation instead of the "
                "subject; the reference recipe keeps the assistant on for every step.",
            )
        _log(
            exec_ctx,
            "info",
            f"Turbo LoRA inference hint: linear scheduler, {train_runtime.turbo_infer_steps} steps, CFG=0, "
            f"LoRA weight 0.8-1.0, FP16 base; training σ sampled continuously over inference steps "
            f"[{train_runtime.timestep_low}-{train_runtime.timestep_high}] "
            f"bias={train_runtime.timestep_bias} "
            f"assistant_off_prob={train_runtime.turbo_assistant_off_prob:g} "
            f"modules={train_runtime.lora_module_keys or 'all'}",
        )
    elif is_turbo:
        assert_no_trainable_assistant(train_module)
        _log(
            exec_ctx,
            "info",
            "Standard SFT on Z-Image-Turbo: no Ostris training assistant. "
            "Progress previews and exported inference load Z-Image-Turbo-DistillPatch "
            "after this LoRA (8 steps, linear scheduler, CFG=0).",
        )
    else:
        assert_no_trainable_assistant(train_module)
        _log(
            exec_ctx,
            "info",
            f"Base Z-Image training: σ bias={train_runtime.sigma_bias}, "
            f"uniform-only train_sigma_shift={base_sigma_shift:g}, "
            f"sigma_bias={train_runtime.sigma_bias} (inference scheduler_shift="
            f"{float(registry_scalar_default(entry, 'scheduler_shift', _Z_IMAGE_BASE_SIGMA_SHIFT)):g})",
        )
        if train_runtime.sigma_bias == "high" or base_sigma_shift >= 5.0:
            _log(
                exec_ctx,
                "warning",
                "σ sampling is concentrated near pure noise (shift>=5 and/or sigma_bias=high): most "
                "steps land at σ>0.9 and the 0.3–0.7 band that fixes facial structure is starved; "
                "identity LoRAs converge poorly. Prefer sigma_bias=sigmoid with train_sigma_shift=1.",
            )
    lora_layer_count = len(getattr(train_module, "_lora_paths", []) or [])
    _log(
        exec_ctx,
        "info",
        "[diag] config: "
        f"model={base_model_id} turbo={is_turbo} res={resolution[0]}x{resolution[1]} "
        f"iters={train_runtime.iterations} lr={train_runtime.learning_rate:g} "
        f"grad_accum={train_runtime.grad_accumulate} "
        f"rank={train_runtime.lora_rank} alpha={train_runtime.lora_alpha} "
        f"scale={train_runtime.lora_scale:g} blocks={train_runtime.lora_blocks} "
        f"modules={train_runtime.lora_module_keys or 'all'} lora_layers={lora_layer_count} "
        f"train_type={train_runtime.train_type} qlora={train_runtime.qlora_bits} "
        f"min_snr_gamma={train_runtime.min_snr_gamma:g} "
        f"sigma_shift={base_sigma_shift:g} sigma_bias={train_runtime.sigma_bias} "
        f"turbo_band=[{train_runtime.timestep_low}-{train_runtime.timestep_high}]/"
        f"{train_runtime.turbo_infer_steps} timestep_bias={train_runtime.timestep_bias} "
        f"turbo_asst_off={train_runtime.turbo_assistant_off_prob:g} "
        f"num_aug={train_runtime.num_augmentations} n_samples={n_samples} "
        f"images={len(pairs)} val_split={train_runtime.val_split:g} "
        f"optimizer_updates={max(1, train_runtime.iterations // max(1, train_runtime.grad_accumulate))}",
    )
    _log_training_sigma_distribution(
        exec_ctx,
        ctx,
        is_turbo=is_turbo,
        sigma_shift=base_sigma_shift,
        resolution=resolution,
        train_runtime=train_runtime,
    )
    _log_zimage_latent_caption_stats(exec_ctx, latent_cache, ctx)
    # Baseline per-σ loss (fresh LoRA ≈ frozen base): reference to compare training progress against.
    if is_turbo and training_assistant is not None:
        _log_noise_level_loss(exec_ctx, train_module, latent_cache, ctx, label="turbo+assistant-init")
        training_assistant.set_enabled(False)
        _log_noise_level_loss(exec_ctx, train_module, latent_cache, ctx, label="turbo-inference-init")
        training_assistant.set_enabled(True)
    else:
        _log_noise_level_loss(exec_ctx, train_module, latent_cache, ctx, label="base-init")
    prior_latents: Any | None = None
    if train_runtime.prior_loss_weight > 0 and class_prompt:
        prior_latents = _ensure_zimage_class_prior_latents(
            latent_cache=latent_cache,
            model=model,
            bundle_root=bundle_root,
            config=config,
            class_prompt=class_prompt,
            resolution=resolution,
            ctx=ctx,
            exec_ctx=exec_ctx,
            turbo=is_turbo,
            turbo_steps=train_runtime.progress_steps,
        )

    prior_cap: Any | None = None
    if train_runtime.prior_loss_weight > 0:
        try:
            prior_cap = latent_cache.load_prior()["cap"]
        except RuntimeError as exc:
            raise RuntimeError(
                "Prior preservation is enabled but the class/prior latent cache is missing. "
                "Re-encode the dataset with a class_prompt, or set prior_loss_weight=0."
            ) from exc

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
        return latent_cache.sample_z_image(indices[0])

    def loss_fn(x0: mx.array, cap: mx.array) -> mx.array:
        if uncond_cap is not None and caption_dropout_applies(train_runtime.caption_dropout):
            cap = uncond_cap
        assistant_was_off = False
        if (
            is_turbo
            and training_assistant is not None
            and train_runtime.turbo_assistant_off_prob > 0.0
        ):
            if random.random() < float(train_runtime.turbo_assistant_off_prob):
                training_assistant.set_enabled(False)
                assistant_was_off = True
        use_turbo_band = is_turbo
        try:
            return _training_loss(
                train_module,
                x0,
                cap,
                ctx,
                min_snr_gamma=train_runtime.min_snr_gamma,
                prior_cap=prior_cap,
                prior_latents=prior_latents,
                prior_loss_weight=train_runtime.prior_loss_weight if prior_cap is not None else 0.0,
                turbo=use_turbo_band,
                turbo_infer_steps=train_runtime.turbo_infer_steps,
                timestep_low=train_runtime.timestep_low,
                timestep_high=train_runtime.timestep_high,
                timestep_bias=train_runtime.timestep_bias,
                resolution=resolution,
                sigma_shift=base_sigma_shift,
                sigma_bias=train_runtime.sigma_bias,
            )
        finally:
            if assistant_was_off and training_assistant is not None:
                training_assistant.set_enabled(True)

    def preview_at(step: int) -> None:
        from backend.engine.common.codecs.vae import load_vae_weight_dict, read_vae_dir_config
        from backend.engine.common.codecs.vae.decoder import create_loaded_vae_decoder

        # Per-σ reconstruction loss probe (off training graph). For turbo, capture both the
        # training-time behavior (assistant on) and the actual inference behavior (assistant off,
        # matching generation) so we can see whether the low-σ / skin band is being fit.
        if is_turbo and training_assistant is not None:
            _log_noise_level_loss(exec_ctx, train_module, latent_cache, ctx, label="turbo+assistant")
            training_assistant.set_enabled(False)
            _log_noise_level_loss(exec_ctx, train_module, latent_cache, ctx, label="turbo-inference")
            training_assistant.set_enabled(True)
        else:
            _log_noise_level_loss(exec_ctx, train_module, latent_cache, ctx, label="base")
        vae_dir = bundle_root / "vae"
        vae_cfg, _, _ = read_vae_dir_config(vae_dir)
        vae_weights = load_vae_weight_dict(ctx, vae_dir)
        preview_latent, _ = latent_cache.sample_z_image(0)
        dec, _, _, _ = create_loaded_vae_decoder(
            ctx,
            preview_latent,
            vae_weights,
            float(vae_cfg.get("scaling_factor", 1.0)),
            float(vae_cfg.get("shift_factor", 0.0)),
            vae_cfg=vae_cfg,
        )
        te = _load_zimage_text_encoder(ctx, bundle_root, config)
        if is_turbo and training_assistant is not None:
            training_assistant.set_enabled(False)
        try:
            preview_scope = (
                _temporary_inference_adapters(model, project_root=project_root, registry=registry, ctx=ctx)
                if is_turbo_sft
                else nullcontext()
            )
            with preview_scope:
                preview = _generate_progress_image(
                    model=train_module,
                    vae_dec=dec,
                    text_encoder=te,
                    prompt=progress_prompt,
                    resolution=resolution,
                    ctx=ctx,
                    steps=train_runtime.progress_steps,
                    turbo=is_turbo,
                    guidance=0.0 if is_turbo else float(train_runtime.guidance),
                    base_sigma_shift=base_preview_shift,
                )
            Image.fromarray(preview).save(work_dir / f"{step:07d}_progress.png")
        finally:
            if is_turbo and training_assistant is not None:
                training_assistant.set_enabled(True)
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

    # Keep final / best / periodic checkpoints distinct so the candidate matrix can
    # compare them. ``best_adapters`` is lowest val loss only — not best likeness.
    del best_path
    final_path = adapter_dir / "final_adapters.safetensors"
    if train_runtime.fuse_adapters:
        from backend.engine.training.lora_layers_mlx import write_fused_adapters_from_published

        write_fused_adapters_from_published(
            train_module,
            final_path=final_path,
            fused_path=adapter_dir / "fused_adapters.safetensors",
            rank=train_runtime.lora_rank,
            base_model_id=base_model_id,
        )
    _validate_saved_lora(final_path)

    from backend.engine.training.lora_train_config import finalize_training_adapters

    published = finalize_training_adapters(
        work_dir=work_dir,
        adapter_dir=adapter_dir,
        final_path=final_path,
        output_name=request.output_name or f"{base_model_id}-{request.dataset_id}",
        params=request.model_dump(),
        dataset_meta=dataset_meta,
        loss_history=loss_history,
        training_caption=training_caption,
        path_resolver=path_resolver,
        task_id=exec_ctx.task_id,
        auto_register=bool(request.auto_register),
    )
    _log(
        exec_ctx,
        "success",
        (
            f"Training complete → awaiting candidate selection ({len(published.get('checkpoints') or [])} checkpoints)"
            if published.get("needs_candidate_evaluation")
            else f"Training complete → {published['adapter_path']}"
        ),
    )
    return published
