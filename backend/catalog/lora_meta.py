"""Registry-driven LoRA adapter metadata (UI picklist, compose overrides, engine flags)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from backend.core.contracts import parse_model_version

Z_IMAGE_DISTILLPATCH_LORA_ID = "z-image-turbo-distillpatch-lora"
Z_IMAGE_DISTILLPATCH_VERSION = "bf16"
Z_IMAGE_DISTILLPATCH_ADAPTER_ID = f"{Z_IMAGE_DISTILLPATCH_LORA_ID}:{Z_IMAGE_DISTILLPATCH_VERSION}"


def _entry_params(entry: Any) -> dict[str, Any]:
    params = getattr(entry, "parameters", None)
    if isinstance(params, dict):
        return params
    raw = getattr(entry, "raw", None) or {}
    legacy = raw.get("parameters")
    return legacy if isinstance(legacy, dict) else {}


def _catalog_lora_meta(entry: Any) -> dict[str, Any]:
    raw = getattr(entry, "raw", None) or {}
    metadata: dict[str, Any] = {}
    catalog = raw.get("catalog") if isinstance(raw.get("catalog"), dict) else {}
    if isinstance(catalog.get("metadata"), dict):
        metadata = catalog["metadata"]
    elif isinstance(raw.get("metadata"), dict):
        metadata = raw["metadata"]
    else:
        top = getattr(entry, "metadata", None)
        if isinstance(top, dict):
            metadata = top
    lora = metadata.get("lora")
    return lora if isinstance(lora, dict) else {}


def lora_video_step_distill(entry: Any) -> bool:
    """LoRA requests step-distill video schedule when merged (Lightning, H3 Turbo, etc.)."""
    params = _entry_params(entry)
    return bool(
        params.get("video_step_distill")
        or params.get("wan_lightning_distill")
        or params.get("h3_turbo_distill")
    )


def lora_z_image_distill_patch(entry: Any) -> bool:
    """LoRA restores Z-Image-Turbo acceleration after ordinary Turbo SFT."""
    params = _entry_params(entry)
    return bool(params.get("z_image_distill_patch"))


def lora_moe_shards(entry: Any) -> bool:
    """LoRA bundle uses separate high/low noise MoE safetensors."""
    params = _entry_params(entry)
    return bool(params.get("lora_moe_shards") or params.get("wan_lightning_distill"))


def lora_compose_overrides(entry: Any) -> dict[str, Any]:
    """Composer param overrides when this LoRA is selected (steps, guidance, shift, …)."""
    params = _entry_params(entry)
    raw = params.get("compose_overrides")
    if not isinstance(raw, dict) or not raw:
        return {}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if value is not None:
            out[str(key)] = value
    return out


def lora_adapter_tags(entry: Any) -> list[str]:
    meta = _catalog_lora_meta(entry)
    tags = meta.get("tags")
    if not isinstance(tags, list):
        return []
    return [str(t).strip() for t in tags if str(t).strip()]


def lora_adapter_hint_key(entry: Any) -> str | None:
    meta = _catalog_lora_meta(entry)
    hint = meta.get("hint_key")
    text = str(hint).strip() if hint is not None else ""
    return text or None


def lora_base_model_key(entry: Any) -> str:
    raw = getattr(entry, "raw", None) or {}
    declared = getattr(entry, "base_model", None) or raw.get("base_model") or ""
    return str(declared).split(":", 1)[0].strip()


def lora_base_compatible(model_base_key: str, lora_entry: Any) -> bool:
    """Default LoRA ↔ base matching: exact registry ``base_model`` id."""
    lora_key = lora_base_model_key(lora_entry)
    model_key = (model_base_key or "").split(":", 1)[0].strip()
    if not model_key or not lora_key:
        return False
    return lora_key == model_key


def lora_config_picklist_extras(bundle_dir: Path, for_model: str | None = None) -> dict[str, Any]:
    """Tags / compose_overrides / hint from ``lora_config.json``.

    Selecting a user LoRA sets its weight only. Steps, CFG, and scheduler stay on
    the current base model; the inference block is a verification recipe, not a
    composer overwrite.
    """
    cfg_path = bundle_dir / "lora_config.json"
    if not cfg_path.is_file():
        return {}
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    inference = data.get("inference")
    if not isinstance(inference, dict):
        return {}
    return _inference_block_picklist_extras(inference, for_model=for_model)


Z_IMAGE_TURBO_TRAINED_HINT_KEY = "studio.loraHint.zImageTurboTrained"
Z_IMAGE_SAMPLER_STAYS_HINT_KEY = "studio.loraHint.zImageSamplerStays"


def _model_stem(model_id: str | None) -> str:
    return (model_id or "").split(":", 1)[0].strip()


def _inference_sampling_applies(inference: dict[str, Any], for_model: str | None) -> bool:
    """True when steps/CFG/scheduler from the recipe belong on ``for_model``.

    ``for_model`` empty keeps the full block (callers that are not a composer).
    Turbo recipes must declare their target model explicitly.
    """
    if not for_model:
        return True
    target = _model_stem(str(inference.get("model") or ""))
    if not target:
        return True
    return _model_stem(for_model) == target


def _inference_block_picklist_extras(
    inference: dict[str, Any],
    *,
    for_model: str | None = None,
) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    recipe_on_this_model = _inference_sampling_applies(inference, for_model)
    overrides: dict[str, Any] = {}
    lora_weight = inference.get("lora_weight")
    if isinstance(lora_weight, (int, float)) and not isinstance(lora_weight, bool):
        overrides["lora_scale"] = float(lora_weight)
    if overrides:
        extras["compose_overrides"] = overrides
    rng = inference.get("lora_weight_range")
    if (
        isinstance(rng, (list, tuple))
        and len(rng) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in rng)
    ):
        lo, hi = float(rng[0]), float(rng[1])
        if 0 <= lo <= hi:
            extras["recommended_lora_scale"] = [lo, hi]
    scheme = str(inference.get("scheme") or "").strip().lower()
    if scheme == "sft":
        extras["tags"] = ["Turbo SFT"]
        extras["hint_key"] = (
            "studio.loraHint.zImageDistillPatch"
            if recipe_on_this_model
            else Z_IMAGE_SAMPLER_STAYS_HINT_KEY
        )
    elif scheme == "turbo":
        extras["tags"] = ["Turbo"]
        extras["hint_key"] = (
            Z_IMAGE_TURBO_TRAINED_HINT_KEY if recipe_on_this_model else Z_IMAGE_SAMPLER_STAYS_HINT_KEY
        )
    elif not recipe_on_this_model and any(
        key in inference and inference[key] is not None for key in ("steps", "guidance", "scheduler")
    ):
        extras["hint_key"] = Z_IMAGE_SAMPLER_STAYS_HINT_KEY
    return extras


def lora_adapter_picklist_row(
    entry: Any,
    *,
    lora_id: str,
    name: str,
    base_model: str,
    source: str = "registry",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "kind": "lora",
        "id": lora_id,
        "name": name,
        "base_model": base_model,
        "source": source,
        "tags": lora_adapter_tags(entry),
        "compose_overrides": lora_compose_overrides(entry),
    }
    hint = lora_adapter_hint_key(entry)
    if hint:
        row["hint_key"] = hint
    return row


def adapters_include_video_step_distill(adapters: Sequence[Any], registry: Any) -> bool:
    from backend.engine.common.bundle.lora_mlx import adapter_id_weight

    for item in adapters or ():
        lora_id, _ = adapter_id_weight(item)
        mid, _ = parse_model_version(lora_id)
        entry = registry.get(mid) if registry is not None else None
        if entry is not None and lora_video_step_distill(entry):
            return True
    return False


def adapters_include_z_image_distill_patch(adapters: Sequence[Any], registry: Any) -> bool:
    from backend.engine.common.bundle.lora_mlx import adapter_id_weight

    for item in adapters or ():
        lora_id, _ = adapter_id_weight(item)
        mid, _ = parse_model_version(lora_id)
        if mid == Z_IMAGE_DISTILLPATCH_LORA_ID:
            return True
        entry = registry.get(mid) if registry is not None else None
        if entry is not None and lora_z_image_distill_patch(entry):
            return True
    return False


def _adapter_uses_z_image_turbo_sft(lora_id: str, *, config_dir: Any = None) -> bool:
    """True only for an explicit, same-base Turbo SFT inference contract."""
    mid, _ = parse_model_version(lora_id)
    if mid == Z_IMAGE_DISTILLPATCH_LORA_ID or not mid.startswith("user-lora-") or config_dir is None:
        return False
    from backend.engine.training.user_lora_registry import get_user_lora

    ul = get_user_lora(config_dir, lora_id)
    if ul is None or str(ul.get("base_model") or "").split(":", 1)[0].strip() != "z-image-turbo":
        return False
    inference = ul.get("inference")
    if not isinstance(inference, dict) or str(inference.get("scheme") or "").lower() != "sft":
        return False
    model = str(inference.get("model") or "").split(":", 1)[0].strip()
    if model != "z-image-turbo":
        raise ValueError("Turbo SFT inference must use z-image-turbo")
    extra_adapters = inference.get("extra_adapters")
    if not isinstance(extra_adapters, list) or not any(
        parse_model_version(str(item).split("@", 1)[0])[0] == Z_IMAGE_DISTILLPATCH_LORA_ID
        for item in extra_adapters
    ):
        raise ValueError("Turbo SFT inference requires an explicit DistillPatch adapter")
    return True


def distillpatch_lora_installed(project_root: Any, registry: Any) -> bool:
    """True when DistillPatch safetensors exist at the registry ``local_path``."""
    from backend.engine.contracts.pipeline_registry import local_bundle_root

    entry = registry.get(Z_IMAGE_DISTILLPATCH_LORA_ID) if registry is not None else None
    if entry is None:
        return False
    root = Path(project_root) if project_root is not None else None
    if root is None:
        return False
    bundle = local_bundle_root(root, entry, Z_IMAGE_DISTILLPATCH_VERSION)
    return bundle is not None


def expand_z_image_turbo_sft_adapters(
    adapters: Sequence[Any],
    *,
    base_model_id: str,
    registry: Any,
    config_dir: Any = None,
    project_root: Any = None,
) -> list[Any]:
    """Append DistillPatch after an explicit Turbo SFT adapter on the same base."""
    model_key = (base_model_id or "").split(":", 1)[0].strip()
    if model_key != "z-image-turbo":
        return list(adapters or ())
    if adapters_include_z_image_distill_patch(adapters, registry):
        return list(adapters or ())
    needs_patch = False
    from backend.engine.common.bundle.lora_mlx import adapter_id_weight

    for item in adapters or ():
        lora_id, _ = adapter_id_weight(item)
        if _adapter_uses_z_image_turbo_sft(lora_id, config_dir=config_dir):
            needs_patch = True
            break
    if not needs_patch:
        return list(adapters or ())
    if registry is not None and registry.get(Z_IMAGE_DISTILLPATCH_LORA_ID) is None:
        raise RuntimeError(
            "Turbo SFT requires Z-Image Turbo DistillPatch LoRA in models_registry.json. "
            "Run `make sync-models-registry` (or restore registry defaults), then install "
            f"{Z_IMAGE_DISTILLPATCH_LORA_ID!r} from the Models page."
        )
    if project_root is not None and not distillpatch_lora_installed(project_root, registry):
        raise RuntimeError(
            "Turbo SFT: Z-Image Turbo DistillPatch LoRA is not installed. "
            f"Install {Z_IMAGE_DISTILLPATCH_LORA_ID!r} (BF16, ~225MB) from Models → LoRA, "
            "then retry Turbo inference."
        )
    out = list(adapters or ())
    out.append({"id": Z_IMAGE_DISTILLPATCH_ADAPTER_ID, "weight": 1.0})
    return out
