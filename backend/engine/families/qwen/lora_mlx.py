"""Qwen Image LoRA merge via shared MLX skeleton."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from backend.engine.common.bundle.lora_mlx import merge_lora_adapters_common
from backend.engine.families.qwen.weights_mlx import remap_qwen_lora_keys
from backend.engine.runtime._base import RuntimeContext


from backend.engine.families.qwen.weights_mlx import qwen_image_lora_scope_key


def _base_model_scope_key(value: str) -> str:
    return qwen_image_lora_scope_key(value)


def _repair_indexed_qwen_weights(
    weights: dict[str, Any],
    model: Any,
    lora_config: dict[str, Any],
) -> dict[str, Any]:
    from backend.engine.training.lora_layers_mlx import (
        enumerate_qwen_lora_module_paths,
        repair_indexed_lora_weights,
    )

    raw_blocks = lora_config.get("lora_blocks")
    lora_blocks = int(raw_blocks) if raw_blocks is not None else -1
    paths = enumerate_qwen_lora_module_paths(model, lora_blocks=lora_blocks)
    return repair_indexed_lora_weights(weights, module_paths=paths)


def merge_qwen_image_lora_adapters(
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
        family_name="Qwen Image",
        remap_groups=remap_qwen_lora_keys,
        param_key_for_module=lambda module_name: f"dit.{module_name}.weight",
        base_model_scope_key=_base_model_scope_key,
        repair_indexed_weights=_repair_indexed_qwen_weights,
        on_log=on_log,
    )
