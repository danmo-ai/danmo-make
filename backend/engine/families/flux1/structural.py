"""FLUX.1 ControlNet — runtime contract, depth/canny preprocess, structural conditioning."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from backend.engine.runtime._base import RuntimeContext

GuideType = Literal["canny", "depth", "redux"]

# Registry controlnet id → companion LoRA (BFL structural-conditioning LoRA path).
CONTROLNET_LORA_MAP: dict[str, str] = {
    "flux-canny-controlnet": "flux1-canny-dev-lora",
    "flux-depth-controlnet": "flux1-depth-dev-lora",
}

_GUIDE_TYPE_HINTS: tuple[tuple[str, GuideType], ...] = (
    ("depth", "depth"),
    ("redux", "redux"),
    ("canny", "canny"),
)

# Registry ``backends`` for controlnets category rows until CUDA batch is ready.
CONTROLNET_DECLARED_BACKENDS: tuple[str, ...] = ("mlx",)
CONTROLNET_CUDA_BATCH_PLANNED = True


# ---------------------------------------------------------------------------
# ControlNet runtime (MLX today; CUDA unified batch TBD)
# ---------------------------------------------------------------------------


def controlnet_host_backends_available() -> tuple[str, ...]:
    from backend.engine.platform import PlatformInfo

    detected = set(PlatformInfo.detect())
    return tuple(b for b in CONTROLNET_DECLARED_BACKENDS if b in detected)


def controlnet_runtime_available() -> bool:
    return len(controlnet_host_backends_available()) > 0


def require_controlnet_runtime(ctx: RuntimeContext, *, feature: str) -> None:
    """Fail loud when this host or RuntimeContext cannot run ControlNet paths today."""
    from backend.engine.platform import PlatformInfo
    from backend.engine.runtime.mlx import MLXContext

    detected = PlatformInfo.detect()
    if not controlnet_runtime_available():
        raise RuntimeError(
            f"FLUX ControlNet ({feature}) requires host backends {CONTROLNET_DECLARED_BACKENDS}; "
            f"detected={detected}."
        )
    if not isinstance(ctx, MLXContext):
        raise RuntimeError(
            f"FLUX ControlNet ({feature}) is MLX-only; "
            f"current runtime={type(ctx).__name__}."
        )


# ---------------------------------------------------------------------------
# Depth Pro (structural depth guide)
# ---------------------------------------------------------------------------


def estimate_depth_rgb01(
    pil: Image.Image,
    *,
    width: int,
    height: int,
    depth_bundle_root: Path,
    on_log: Any = None,
) -> np.ndarray:
    """Return float01 RGB depth visualization ``[H,W,3]`` (BFL DepthImageEncoder style)."""
    from backend.engine.families.flux1.depth_encode_mlx import estimate_depth_rgb01_mlx

    return estimate_depth_rgb01_mlx(
        pil,
        width=width,
        height=height,
        depth_bundle_root=depth_bundle_root,
        on_log=on_log,
    )


def resolve_depth_pro_bundle_root(registry: Any, project_root: Path) -> Path:
    from backend.engine.contracts.pipeline_registry import local_bundle_root as bundle_root_fn
    from backend.engine.contracts.pipeline_registry import resolve_version_block as version_block_fn

    entry = registry.require("depth-pro")
    version_key = version_block_fn(entry, None)
    root = bundle_root_fn(project_root, entry, version_key)
    if root is None:
        raise RuntimeError("depth-pro registry entry has no local bundle path")
    return root


# ---------------------------------------------------------------------------
# Structural guide helpers
# ---------------------------------------------------------------------------


def infer_guide_type(model_id: str) -> GuideType:
    key = (model_id or "").strip().lower()
    for hint, gtype in _GUIDE_TYPE_HINTS:
        if hint in key:
            return gtype
    return "canny"


def is_fill_controlnet(model_id: str) -> bool:
    return "fill" in (model_id or "").strip().lower()


def is_redux_controlnet(model_id: str) -> bool:
    return "redux" in (model_id or "").strip().lower()


def companion_lora_id(controlnet_model_id: str) -> str | None:
    if is_redux_controlnet(controlnet_model_id) or is_fill_controlnet(controlnet_model_id):
        return None
    return CONTROLNET_LORA_MAP.get((controlnet_model_id or "").strip())


def preprocess_structural_rgb(
    pil: Image.Image,
    *,
    guide_type: GuideType,
    width: int,
    height: int,
    registry: Any,
    project_root: Path,
    on_log: Any = None,
) -> np.ndarray:
    """Return float01 RGB ``[H,W,3]`` ready for VAE encode (linear 0..1)."""
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    if pil.size != (width, height):
        pil = pil.resize((width, height), Image.Resampling.LANCZOS)
    rgb = np.asarray(pil, dtype=np.float32) / 255.0

    if guide_type == "canny":
        return _canny_rgb(rgb)
    if guide_type == "depth":
        depth_root = resolve_depth_pro_bundle_root(registry, project_root)
        return estimate_depth_rgb01(
            pil,
            width=width,
            height=height,
            depth_bundle_root=depth_root,
            on_log=on_log,
        )
    raise RuntimeError(f"preprocess_structural_rgb does not handle guide_type={guide_type!r}")


def _canny_rgb(rgb: np.ndarray, low: int = 50, high: int = 200) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "structural guide (canny) requires opencv-python-headless; "
            "pip install opencv-python-headless"
        ) from exc
    gray = cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    e = edges.astype(np.float32) / 255.0
    return np.stack([e, e, e], axis=-1)


def load_flux1_structural_patch_embed(
    *,
    registry: Any,
    project_root: Path,
    controlnet_model_id: str,
    ctx: Any,
    on_log: Any = None,
) -> tuple[Any, Any]:
    """Load 128-dim packed ``x_embedder`` from a controlnet bundle (Flux Canny/Depth)."""
    from backend.engine.contracts.pipeline_registry import local_bundle_root as bundle_root_fn
    from backend.engine.contracts.pipeline_registry import resolve_version_block as version_block_fn
    from backend.engine.families.flux1.weights import remap_flux1_weights

    entry = registry.require(controlnet_model_id)
    version_key = version_block_fn(entry, None)
    bundle_root = bundle_root_fn(project_root, entry, version_key)
    tp = (bundle_root / "transformer") if bundle_root else None
    if tp is None or not tp.exists():
        raise RuntimeError(
            f"structural guide requires installed controlnet bundle {controlnet_model_id!r} "
            f"(missing transformer/ under {bundle_root}); install from Models → ControlNet"
        )

    raw: dict[str, Any] = {}
    for sf in sorted(tp.glob("*.safetensors")):
        raw.update(ctx.load_weights(str(sf)))
    remapped = remap_flux1_weights(raw)
    weight = remapped.get("patch_embed.proj.weight")
    bias = remapped.get("patch_embed.proj.bias")
    if weight is None or bias is None:
        raise RuntimeError(
            f"controlnet bundle {controlnet_model_id!r} missing x_embedder "
            f"(patch_embed.proj.weight/bias)"
        )
    if hasattr(weight, "shape"):
        sh = tuple(weight.shape)
        in_ch = int(sh[-1]) if len(sh) == 4 else int(sh[1]) if len(sh) == 2 else -1
    else:
        in_ch = -1
    if in_ch != 128:
        raise RuntimeError(
            f"controlnet bundle {controlnet_model_id!r} x_embedder input dim {in_ch} "
            f"(expected 128 for Flux structural concat); wrong bundle or corrupt weights"
        )
    if on_log:
        on_log("info", f"structural_guide loaded x_embedder from {controlnet_model_id} in_ch=128")
    return weight, bias


# ---------------------------------------------------------------------------
# Pipeline hooks (registry-driven)
# ---------------------------------------------------------------------------


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
    """Attach structural guide conditioning (Canny / Depth / Redux) to ``extra_cond``."""
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
    if is_fill_controlnet(controlnet_id):
        raise RuntimeError(
            "flux-fill is an inpainting model — use image retouch/extend with a mask, "
            "not text-to-image structural_guide"
        )
    guide_type = getattr(guide, "type", None) or infer_guide_type(controlnet_id)

    src_path = ctx_exec.asset_store.get_file_path(guide.asset_id)
    pil = Image.open(str(src_path))

    if guide_type == "redux" or is_redux_controlnet(controlnet_id):
        from backend.engine.families.flux1.redux_encode import (
            encode_redux_context_tokens,
            resolve_redux_bundle_root,
        )

        redux_root = resolve_redux_bundle_root(
            pipeline._registry, pipeline._project_root, controlnet_id
        )
        tokens = encode_redux_context_tokens(
            pil,
            redux_bundle_root=redux_root,
            on_log=on_log,
        )
        w = float(guide.weight)
        if w <= 0.0:
            if on_log:
                on_log("info", "structural_guide redux skipped (weight <= 0)")
            return dict(extra_cond), None
        if w != 1.0:
            tokens = tokens * np.float32(w)
        redux_embeds = pipeline.ctx.array(tokens.astype(np.float32))
        if getattr(pipeline.ctx, "backend", None) == "mlx":
            pipeline.ctx.eval(redux_embeds)
        out = dict(extra_cond)
        out["redux_txt_embeds"] = redux_embeds
        if on_log:
            on_log(
                "info",
                f"structural_guide type=redux controlnet={controlnet_id} "
                f"weight={float(guide.weight):.3f} asset={guide.asset_id}",
            )
        return out, None

    rgb = preprocess_structural_rgb(
        pil,
        guide_type=guide_type,
        width=width,
        height=height,
        registry=pipeline._registry,
        project_root=pipeline._project_root,
        on_log=on_log,
    )
    arr = rgb[None, ...]
    image_nchw = pipeline.ctx.array(arr)
    image_nchw = pipeline.ctx.permute(image_nchw, (0, 3, 1, 2))

    from backend.engine.pipelines.image_run_common import image_vae_encode_tensor

    structural_latents = image_vae_encode_tensor(
        pipeline,
        image_nchw,
        entry,
        version_key,
        height_px=height,
        width_px=width,
        on_log=on_log,
    )
    if getattr(pipeline.ctx, "backend", None) == "mlx":
        pipeline.ctx.eval(structural_latents)

    activate = getattr(model, "activate_structural_patch_embed", None)
    deactivate = getattr(model, "deactivate_structural_patch_embed", None)
    if not callable(activate) or not callable(deactivate):
        raise RuntimeError(
            f"structural_guide requires Flux1Transformer structural patch embed; "
            f"model={type(model).__name__}"
        )
    pw, pb = load_flux1_structural_patch_embed(
        registry=pipeline._registry,
        project_root=pipeline._project_root,
        controlnet_model_id=controlnet_id,
        ctx=pipeline.ctx,
        on_log=on_log,
    )
    activate(pw, pb)

    out = dict(extra_cond)
    out["structural_latents_nchw"] = structural_latents
    if on_log:
        on_log(
            "info",
            f"structural_guide type={guide_type} controlnet={controlnet_id} "
            f"weight={float(guide.weight):.3f} asset={guide.asset_id}",
        )
    return out, deactivate


def augment_request_for_structural_guide(request: Any, ctx: Any) -> Any:
    """Inject companion structural LoRA when ``structural_guide`` is set."""
    guide = getattr(request, "structural_guide", None)
    if guide is None:
        return request
    from backend.core.contracts import AdapterRef

    require_controlnet_runtime(ctx, feature="structural_guide")
    model_id = (getattr(guide, "model_id", None) or "").strip()
    if not model_id:
        raise RuntimeError(
            "structural_guide.model_id is required (registry controlnet id, e.g. flux-canny-controlnet)"
        )
    lora_id = companion_lora_id(model_id)
    if not lora_id:
        return request
    adapters = list(getattr(request, "adapters", None) or [])
    if any(a.id == lora_id or a.id.startswith(f"{lora_id}:") for a in adapters):
        return request
    adapters.append(AdapterRef(id=lora_id, weight=float(guide.weight)))
    return request.model_copy(update={"adapters": adapters})
