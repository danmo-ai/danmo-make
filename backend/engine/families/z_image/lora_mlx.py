"""Z-Image LoRA merge via shared MLX skeleton (with CUDA dispatch)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from backend.engine.common.bundle.lora_mlx import adapter_id_weight, merge_lora_adapters_common
from backend.engine.families.z_image.weights import remap_zimage_lora_keys, z_image_lora_scope_key
from backend.engine.runtime._base import RuntimeContext


def _base_model_scope_key(value: str) -> str:
    return z_image_lora_scope_key(value)


def _repair_indexed_zimage_weights(
    weights: dict[str, Any],
    model: Any,
    lora_config: dict[str, Any],
) -> dict[str, Any]:
    from backend.engine.training.lora_layers_mlx import (
        enumerate_zimage_lora_module_paths,
        repair_indexed_lora_weights,
    )

    raw_blocks = lora_config.get("lora_blocks")
    lora_blocks = int(raw_blocks) if raw_blocks is not None else -1
    paths = enumerate_zimage_lora_module_paths(model, lora_blocks=lora_blocks)
    return repair_indexed_lora_weights(weights, module_paths=paths)


def _resolve_lora_bundle(
    lora_id: str,
    *,
    project_root: Path,
    registry: Any,
) -> Path | None:
    from backend.engine.common.bundle.lora_mlx import adapter_id_weight
    from backend.core.contracts import parse_model_version
    from backend.engine.contracts.pipeline_registry import local_bundle_root

    mid, ver = parse_model_version(lora_id)
    try:
        entry = registry.require(mid)
    except KeyError:
        if str(lora_id).startswith("user-lora-"):
            from backend.engine.training.user_lora_registry import (
                get_user_lora,
                resolve_user_lora_bundle,
            )

            config_dir = project_root / "config"
            ul = get_user_lora(config_dir, lora_id)
            if ul is None:
                return None
            return resolve_user_lora_bundle(project_root, config_dir, lora_id)
        return None
    bundle = local_bundle_root(project_root, entry, ver or None)
    return bundle


def merge_z_image_lora_adapters(
    model: Any,
    adapters: Sequence[Any],
    *,
    base_model_id: str,
    project_root: Path,
    registry: Any,
    ctx: RuntimeContext,
    patch_size: int | None = None,
    on_log: Callable[[str, str], None] | None = None,
) -> None:
    ps = int(patch_size) if patch_size is not None else int(
        getattr(getattr(model, "config", None), "patch_size", 2) or 2
    )

    backend = getattr(ctx, "backend", "mlx")
    if backend == "cuda":
        import importlib
        lora_cuda = importlib.import_module("backend.engine.families.z_image.lora_cuda")
        lora_cuda.merge_z_image_lora_adapters_cuda(
            model=model,
            adapters=adapters,
            base_model_id=base_model_id,
            project_root=project_root,
            registry=registry,
            ctx=ctx,
            patch_size=ps,
            on_log=on_log,
        )
        return

    merge_lora_adapters_common(
        model=model,
        adapters=adapters,
        base_model_id=base_model_id,
        project_root=project_root,
        registry=registry,
        ctx=ctx,
        family_name="Z-Image",
        remap_groups=lambda weights: remap_zimage_lora_keys(
            weights, patch_size=max(1, ps)
        ),
        param_key_for_module=lambda module_name: f"{module_name}.weight",
        base_model_scope_key=_base_model_scope_key,
        repair_indexed_weights=_repair_indexed_zimage_weights,
        on_log=on_log,
    )
    refresh = getattr(model, "_refresh_compiled_forward", None)
    if callable(refresh):
        refresh()
