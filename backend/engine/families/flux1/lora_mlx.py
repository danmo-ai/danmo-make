"""Flux.1 LoRA merge via shared MLX skeleton."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from backend.engine.common.bundle.lora_mlx import merge_lora_adapters_common
from backend.engine.families.flux1.weights import remap_flux1_lora_keys, remap_flux1_lora_module_prefix
from backend.engine.runtime._base import RuntimeContext


def merge_flux1_lora_adapters(
    model: Any,
    adapters: Sequence[Any],
    *,
    base_model_id: str,
    project_root: Path,
    registry: Any,
    ctx: RuntimeContext,
    on_log: Callable[[str, str], None] | None = None,
) -> None:
    merge_lora_adapters_common(
        model=model,
        adapters=adapters,
        base_model_id=base_model_id,
        project_root=project_root,
        registry=registry,
        ctx=ctx,
        family_name="Flux1",
        remap_groups=remap_flux1_lora_keys,
        param_key_for_module=lambda module_name: f"{module_name}.weight",
        remap_module_prefix=remap_flux1_lora_module_prefix,
        on_log=on_log,
    )
