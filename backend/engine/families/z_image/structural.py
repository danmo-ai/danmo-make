"""Z-Image structural guide (Fun ControlNet Union) — pipeline hooks."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

from PIL import Image

GuideType = Literal["canny", "depth", "pose", "hed", "mlsd", "scribble", "gray", "auto"]

Z_IMAGE_CONTROLNET_DECLARED_BACKENDS: tuple[str, ...] = ("mlx",)


def controlnet_runtime_available() -> bool:
    from backend.engine.families.flux1.structural import controlnet_host_backends_available

    detected = controlnet_host_backends_available()
    return any(b in detected for b in Z_IMAGE_CONTROLNET_DECLARED_BACKENDS)


def require_controlnet_runtime(ctx: Any, *, feature: str = "structural_guide") -> None:
    from backend.engine.families.flux1.structural import controlnet_host_backends_available

    host = controlnet_host_backends_available()
    if not any(b in host for b in Z_IMAGE_CONTROLNET_DECLARED_BACKENDS):
        raise RuntimeError(
            f"{feature} requires MLX runtime on Apple Silicon; "
            f"detected backends={host!r}, required one of {Z_IMAGE_CONTROLNET_DECLARED_BACKENDS!r}"
        )
    backend = getattr(ctx, "backend", "mlx")
    if backend not in Z_IMAGE_CONTROLNET_DECLARED_BACKENDS:
        raise RuntimeError(
            f"{feature} requires MLX runtime (got backend={backend!r}); "
            f"required one of {Z_IMAGE_CONTROLNET_DECLARED_BACKENDS!r}"
        )

def infer_guide_type(controlnet_id: str) -> GuideType:
    k = (controlnet_id or "").strip().lower()
    for token in ("depth", "pose", "hed", "mlsd", "scribble", "gray", "canny"):
        if token in k:
            return token  # type: ignore[return-value]
    return "auto"


def _preprocess_guide_rgb(
    pil: Image.Image,
    *,
    guide_type: GuideType,
    width: int,
    height: int,
    registry: Any,
    project_root: Path,
    on_log: Callable[..., None] | None,
) -> Any:
    if guide_type in ("auto", "pose", "hed", "mlsd", "scribble", "gray"):
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        if pil.size != (width, height):
            pil = pil.resize((width, height), Image.Resampling.LANCZOS)
        import numpy as np

        return np.asarray(pil, dtype=np.float32) / 255.0

    from backend.engine.families.flux1.structural import preprocess_structural_rgb

    gt = "canny" if guide_type == "auto" else guide_type
    if gt not in ("canny", "depth"):
        raise RuntimeError(f"z_image structural_guide preprocess does not support type={guide_type!r}")
    return preprocess_structural_rgb(
        pil,
        guide_type=gt,
        width=width,
        height=height,
        registry=registry,
        project_root=project_root,
        on_log=on_log,
    )


def build_z_image_control_context_nchw(
    pipeline: Any,
    control_rgb01,
    *,
    entry: Any,
    version_key: str | None,
    height: int,
    width: int,
    on_log: Callable[..., None] | None,
    inpaint_rgb01=None,
    inpaint_mask_hw=None,
) -> Any:
    """Build ``[1, 33, 1, H_lat, W_lat]`` control context (16 + 1 + 16 channels).

    When ``inpaint_rgb01`` and ``inpaint_mask_hw`` are provided, channel 17 is the
    inverted preserve mask and channels 18–33 are VAE latents of the masked source
    (diffusers ``ZImageControlNetInpaintPipeline`` layout).
    """
    ctx = pipeline.ctx
    import numpy as np

    arr = np.asarray(control_rgb01, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise RuntimeError(f"z_image control guide must be HxWx3 RGB, got shape={arr.shape}")
    image_nchw = ctx.array(arr[None, ...])
    image_nchw = ctx.permute(image_nchw, (0, 3, 1, 2))

    from backend.engine.pipelines.image_run_common import image_vae_encode_tensor

    control_latents = image_vae_encode_tensor(
        pipeline,
        image_nchw,
        entry,
        version_key,
        height_px=height,
        width_px=width,
        on_log=on_log,
    )
    if control_latents.ndim == 4:
        _, c, h, w = control_latents.shape
        control_latents = ctx.reshape(control_latents, (1, c, 1, h, w))
    elif control_latents.ndim == 5:
        pass
    else:
        raise RuntimeError(f"unexpected control_latents shape {tuple(control_latents.shape)}")

    _, _c, _f, lh, lw = control_latents.shape

    if inpaint_rgb01 is not None and inpaint_mask_hw is not None:
        from backend.engine.families.flux1.fill_edit import apply_inpaint_mask_rgb
        from PIL import Image

        mask_hw = np.asarray(inpaint_mask_hw, dtype=np.float32)
        if mask_hw.ndim != 2:
            raise RuntimeError(f"inpaint mask must be HxW, got {mask_hw.shape}")
        if mask_hw.shape != (height, width):
            raise RuntimeError(
                f"inpaint mask shape {mask_hw.shape} does not match generation size {height}x{width}"
            )
        src = np.asarray(inpaint_rgb01, dtype=np.float32)
        if src.shape[:2] != (height, width):
            raise RuntimeError(
                f"inpaint source shape {src.shape[:2]} does not match generation size {height}x{width}"
            )
        masked_rgb = apply_inpaint_mask_rgb(src, mask_hw)
        masked_nchw = ctx.array(masked_rgb[None, ...])
        masked_nchw = ctx.permute(masked_nchw, (0, 3, 1, 2))
        inpaint_latents = image_vae_encode_tensor(
            pipeline,
            masked_nchw,
            entry,
            version_key,
            height_px=height,
            width_px=width,
            on_log=on_log,
        )
        if inpaint_latents.ndim == 4:
            _, ic, ih, iw = inpaint_latents.shape
            inpaint_latents = ctx.reshape(inpaint_latents, (1, ic, 1, ih, iw))
        elif inpaint_latents.ndim != 5:
            raise RuntimeError(f"unexpected inpaint_latents shape {tuple(inpaint_latents.shape)}")

        mask_small = Image.fromarray((np.clip(mask_hw, 0.0, 1.0) * 255.0).astype(np.uint8))
        mask_small = mask_small.resize((lw, lh), Image.Resampling.NEAREST)
        mask_arr = np.asarray(mask_small, dtype=np.float32) / 255.0
        preserve = (1.0 - mask_arr)[None, None, None, ...]
        mask_1ch = ctx.array(preserve.astype(np.float32))
        combined = ctx.concat([control_latents, mask_1ch, inpaint_latents], axis=1)
    else:
        zeros16 = ctx.zeros((1, 16, 1, lh, lw), dtype=control_latents.dtype)
        zeros1 = ctx.zeros((1, 1, 1, lh, lw), dtype=control_latents.dtype)
        combined = ctx.concat([control_latents, zeros1, zeros16], axis=1)

    if getattr(ctx, "backend", None) == "mlx":
        ctx.eval(combined)
    return combined


def load_z_image_controlnet_weights(
    *,
    registry: Any,
    project_root: Path,
    controlnet_model_id: str,
    ctx: Any,
    on_log: Callable[..., None] | None,
) -> dict[str, Any]:
    from backend.engine.contracts.pipeline_registry import local_bundle_root as bundle_root_fn
    from backend.engine.contracts.pipeline_registry import resolve_version_block as version_block_fn
    from backend.engine.families.z_image.weights import remap_zimage_control_weights

    entry = registry.require(controlnet_model_id)
    version_key = version_block_fn(entry, None)
    bundle_root = bundle_root_fn(project_root, entry, version_key)
    if bundle_root is None or not bundle_root.exists():
        raise RuntimeError(
            f"structural_guide requires installed controlnet bundle {controlnet_model_id!r} "
            f"(missing under {bundle_root}); install from Models → ControlNet"
        )

    raw: dict[str, Any] = {}
    for sf in sorted(bundle_root.rglob("*.safetensors")):
        raw.update(ctx.load_weights(str(sf)))
    remapped = remap_zimage_control_weights(raw)
    if not remapped:
        raise RuntimeError(
            f"controlnet bundle {controlnet_model_id!r} has no control_* weights under {bundle_root}"
        )
    if on_log:
        on_log("info", f"structural_guide loaded z_image controlnet keys={len(remapped)} from {controlnet_model_id}")
    return remapped


def augment_request_for_structural_guide(request: Any, ctx: Any) -> Any:
    """Z-Image Union controlnet does not use companion structural LoRAs."""
    guide = getattr(request, "structural_guide", None)
    if guide is None:
        return request
    require_controlnet_runtime(ctx, feature="structural_guide")
    return request


def attach_structural_conditioning(
    pipeline: Any,
    *,
    request: Any,
    family: str,
    model: Any,
    entry: Any,
    version_key: str | None,
    extra_cond: dict[str, Any],
    width: int,
    height: int,
    ctx_exec: Any,
    on_log: Callable[..., None] | None,
) -> tuple[dict[str, Any], Callable[[], None] | None]:
    guide = getattr(request, "structural_guide", None)
    if guide is None:
        return extra_cond, None

    require_controlnet_runtime(pipeline.ctx, feature="structural_guide")
    from backend.engine.config.model_configs import get_config_class

    family_config = get_config_class(family)()
    if not getattr(family_config, "supports_structural_guide", False):
        raise RuntimeError(
            f"structural_guide is not supported for family={family!r} "
            f"(ModelConfig.supports_structural_guide=false)"
        )

    controlnet_id = (getattr(guide, "model_id", None) or "").strip()
    if not controlnet_id:
        raise RuntimeError("structural_guide.model_id is required")
    guide_type = getattr(guide, "type", None) or infer_guide_type(controlnet_id)
    if guide_type == "redux":
        raise RuntimeError("redux structural_guide is FLUX-only; use z_image union controlnet with a control image")

    src_path = ctx_exec.asset_store.get_file_path(guide.asset_id)
    pil = Image.open(str(src_path))

    inpaint_rgb01 = None
    inpaint_mask_hw = None
    inp_src_id = (getattr(guide, "inpaint_source_asset_id", None) or "").strip()
    inp_msk_id = (getattr(guide, "inpaint_mask_asset_id", None) or "").strip()
    if inp_src_id and inp_msk_id:
        from backend.engine.families.flux1.fill_edit import mask_pil_to_weight

        src_path_inp = ctx_exec.asset_store.get_file_path(inp_src_id)
        msk_path_inp = ctx_exec.asset_store.get_file_path(inp_msk_id)
        src_pil = Image.open(str(src_path_inp)).convert("RGB")
        if src_pil.size != (width, height):
            src_pil = src_pil.resize((width, height), Image.Resampling.LANCZOS)
        msk_pil = Image.open(str(msk_path_inp))
        inpaint_mask_hw = mask_pil_to_weight(msk_pil)
        if inpaint_mask_hw.shape != (height, width):
            msk_pil = msk_pil.resize((width, height), Image.Resampling.NEAREST)
            inpaint_mask_hw = mask_pil_to_weight(msk_pil)
        import numpy as np

        inpaint_rgb01 = np.asarray(src_pil, dtype=np.float32) / 255.0

    rgb = _preprocess_guide_rgb(
        pil,
        guide_type=guide_type,
        width=width,
        height=height,
        registry=pipeline._registry,
        project_root=pipeline._project_root,
        on_log=on_log,
    )
    control_context = build_z_image_control_context_nchw(
        pipeline,
        rgb,
        entry=entry,
        version_key=version_key,
        height=height,
        width=width,
        on_log=on_log,
        inpaint_rgb01=inpaint_rgb01,
        inpaint_mask_hw=inpaint_mask_hw,
    )

    weights = load_z_image_controlnet_weights(
        registry=pipeline._registry,
        project_root=pipeline._project_root,
        controlnet_model_id=controlnet_id,
        ctx=pipeline.ctx,
        on_log=on_log,
    )

    activate = getattr(model, "activate_z_image_control", None)
    deactivate = getattr(model, "deactivate_z_image_control", None)
    if not callable(activate) or not callable(deactivate):
        raise RuntimeError(
            f"structural_guide requires ZImageTransformer control hooks; model={type(model).__name__}"
        )

    scale = float(guide.weight)
    if scale <= 0.0:
        if on_log:
            on_log("info", "structural_guide z_image skipped (weight <= 0)")
        return dict(extra_cond), None

    activate(weights, context_scale=scale)

    out = dict(extra_cond)
    out["zimage_control_context"] = control_context
    out["zimage_control_context_scale"] = scale
    if on_log:
        mode = "inpaint+control" if inpaint_rgb01 is not None else "control"
        on_log(
            "info",
            f"structural_guide z_image type={guide_type} mode={mode} controlnet={controlnet_id} "
            f"scale={scale:.3f} asset={guide.asset_id}",
        )
    return out, deactivate
