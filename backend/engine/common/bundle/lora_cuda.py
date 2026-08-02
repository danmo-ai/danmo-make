"""Shared PyTorch LoRA merge skeleton for image transformer families (CUDA path)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from backend.core.contracts import parse_model_version
from backend.engine.contracts.pipeline_registry import local_bundle_root
from backend.engine.runtime._base import RuntimeContext


def adapter_id_weight(item: Any) -> tuple[str, float]:
    if isinstance(item, dict):
        return str(item["id"]), float(item.get("weight", 1.0))
    return str(item.id), float(item.weight)


def load_lora_flat_weights(bundle: Path, ctx: RuntimeContext) -> dict[str, Any]:
    if bundle.is_file():
        if bundle.suffix != ".safetensors":
            raise RuntimeError(
                f"LoRA path must be a .safetensors file or a directory containing weights: {bundle}"
            )
        return dict(ctx.load_weights(str(bundle)))
    if bundle.is_dir():
        files = sorted(bundle.rglob("*.safetensors"))
        if not files:
            raise RuntimeError(f"No .safetensors files under LoRA directory {bundle}")
        out: dict[str, Any] = {}
        for fp in files:
            out.update(ctx.load_weights(str(fp)))
        return out
    raise RuntimeError(f"LoRA path is not a file or directory: {bundle}")


def orient_lora_pair_torch(
    down: torch.Tensor,
    up: torch.Tensor,
    *,
    out_d: int,
    in_d: int,
    lora_id: str,
    wkey: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return (down, up) with shapes [rank, in_d] and [out_d, rank]."""
    d, u = down, up
    if d.shape[1] == in_d and u.shape[0] == out_d and u.shape[1] == d.shape[0]:
        return d, u, int(d.shape[0])
    if d.shape[0] == in_d and u.shape[0] == out_d and u.shape[1] == d.shape[1]:
        return d.t(), u, int(d.shape[1])
    if d.shape[1] == in_d and u.shape[1] == out_d and u.shape[0] == d.shape[0]:
        return d, u.t(), int(d.shape[0])
    if d.shape[0] == in_d and u.shape[1] == out_d and u.shape[0] == d.shape[1]:
        return d.t(), u.t(), int(d.shape[1])
    raise RuntimeError(
        f"LoRA {lora_id!r} tensor shape mismatch for {wkey}: model ({out_d}, {in_d}), "
        f"lora_down {tuple(d.shape)}, lora_up {tuple(u.shape)}."
    )


def merge_lora_adapters_common_cuda(
    *,
    model: Any,
    adapters: Sequence[Any],
    base_model_id: str,
    project_root: Path,
    registry: Any,
    ctx: RuntimeContext,
    family_name: str,
    remap_groups: Callable[[dict[str, Any]], dict[str, tuple[Any, Any, float]]],
    param_key_for_module: Callable[[str], str],
    base_model_scope_key: Callable[[str], str] | None = None,
    on_log: Callable[[str, str], None] | None = None,
) -> None:
    if not adapters:
        return
    if not hasattr(model, "_param_map"):
        raise RuntimeError(f"{family_name} LoRA merge requires TransformerBase with ``_param_map``.")

    request_scope = (
        base_model_scope_key(base_model_id) if base_model_scope_key is not None else base_model_id
    ).strip()
    for item in adapters:
        lora_id, strength = adapter_id_weight(item)
        mid, ver = parse_model_version(lora_id)
        entry = None
        bundle: Path | None = None
        declared_base = ""
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
                    raise RuntimeError(f"Unknown user LoRA adapter {lora_id!r}") from None
                declared_base = str(ul.get("base_model") or "").strip()
                bundle = resolve_user_lora_bundle(project_root, config_dir, lora_id)
                mid = lora_id
            else:
                raise RuntimeError(
                    f"Unknown LoRA adapter {lora_id!r}: add it to config/models_registry.json "
                    "(category=loras) and install weights to the declared local_path."
                ) from None
        if entry is not None:
            raw = entry.raw or {}
            if str(raw.get("category") or "") != "loras":
                raise RuntimeError(
                    f"Adapter id {lora_id!r} refers to registry model {mid!r}, which is not a LoRA "
                    f"(category={raw.get('category')!r})."
                )
            declared_base = str(raw.get("base_model") or "").strip()
            bundle = local_bundle_root(project_root, entry, ver or None)
        declared_scope = (
            base_model_scope_key(declared_base) if base_model_scope_key is not None else declared_base
        ).strip()
        if declared_scope and declared_scope != request_scope:
            raise RuntimeError(
                f"LoRA {mid!r} is scoped to base_model={declared_base!r}, "
                f"but the image request uses {base_model_id!r}."
            )
        if bundle is None:
            raise RuntimeError(
                f"LoRA {lora_id!r} is not installed on disk (missing registry versions.local_path "
                f"for version {ver or 'default'})."
            )
        weights = load_lora_flat_weights(bundle, ctx)
        groups = remap_groups(weights)
        if not groups:
            raise RuntimeError(
                f"LoRA {lora_id!r}: after key remap no (lora_down, lora_up) pairs were found."
            )
        applied = 0
        for module_name, (down, up, alpha) in groups.items():
            wkey = param_key_for_module(module_name)
            if wkey not in model._param_map:
                continue
            param = model._param_map[wkey]
            out_d, in_d = int(param.shape[0]), int(param.shape[1])

            # Convert to torch tensors if needed
            if hasattr(down, "numpy"):
                down = torch.as_tensor(down.numpy(), device=param.device)
            elif not isinstance(down, torch.Tensor):
                import numpy as np
                down = torch.as_tensor(np.asarray(down), device=param.device)
            if hasattr(up, "numpy"):
                up = torch.as_tensor(up.numpy(), device=param.device)
            elif not isinstance(up, torch.Tensor):
                import numpy as np
                up = torch.as_tensor(np.asarray(up), device=param.device)

            d_orient, u_orient, rank = orient_lora_pair_torch(
                down, up, out_d=out_d, in_d=in_d, lora_id=lora_id, wkey=wkey
            )
            if rank <= 0:
                raise RuntimeError(f"LoRA {lora_id!r}: invalid rank for {wkey}.")
            scale = (float(alpha) / float(rank)) * float(strength)
            delta = torch.matmul(u_orient.float(), d_orient.float())
            scaled_delta = scale * delta
            from backend.engine.common.model.quantized_lora_mlx import (
                apply_lora_delta_to_weight,
                inference_mode_from_model,
            )

            inference_mode = inference_mode_from_model(model)
            if (
                inference_mode is not None
                and getattr(inference_mode, "kind", "dense") == "quantized"
                and getattr(inference_mode, "bits", None) in (4, 8)
            ):
                apply_lora_delta_to_weight(
                    model=model,
                    wkey=wkey,
                    delta=scaled_delta,
                    ctx=ctx,
                    bits=int(inference_mode.bits),
                    group_size=int(getattr(inference_mode, "group_size", 64) or 64),
                )
            else:
                with torch.no_grad():
                    updated = param.float() + scaled_delta.to(param.device)
                    param.copy_(updated.to(param.dtype))
            applied += 1
        if applied == 0:
            raise RuntimeError(
                f"LoRA {lora_id!r}: remap produced {len(groups)} groups, but none matched this transformer."
            )
        if on_log:
            from backend.engine.common.model.quantized_lora_mlx import inference_mode_from_model

            mode = inference_mode_from_model(model)
            quant_note = ""
            if mode is not None and getattr(mode, "kind", "") == "quantized":
                quant_note = " requantized_layers=yes"
            on_log(
                "info",
                f"lora merged source={mid} strength={strength} tensors={applied}{quant_note}",
            )
