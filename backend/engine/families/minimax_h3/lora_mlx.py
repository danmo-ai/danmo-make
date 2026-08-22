"""MiniMax-H3 Turbo LoRA merge (registry adapter + optional bundle ``turbo_lora.safetensors``)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import mlx.core as mx

from backend.catalog.lora_meta import lora_compose_overrides
from backend.core.contracts import parse_model_version
from backend.engine.common.bundle.lora_mlx import (
    adapter_id_weight,
    load_lora_flat_weights,
    orient_lora_pair,
    read_lora_config,
)
from backend.engine.common.model.base import _collect_params
from backend.engine.common.model.quantized_lora_mlx import apply_lora_delta_to_weight
from backend.engine.contracts.pipeline_registry import local_bundle_root
from backend.engine.families.minimax_h3.lora_weights import (
    minimax_h3_lora_param_key,
    remap_minimax_h3_lora_keys,
)
from backend.engine.runtime.mlx_runtime import load_weights_dict, run_eval

H3_TURBO_LORA_ID = "minimax-h3-turbo-lora"
H3_TURBO_LORA_VERSION = "v4-ema"
H3_TURBO_BUNDLE_FILENAME = "turbo_lora.safetensors"
H3_TURBO_REGISTRY_ADAPTER = f"{H3_TURBO_LORA_ID}:{H3_TURBO_LORA_VERSION}"


def minimax_h3_video_lora_base_compatible(model_base_key: str, lora_base_key: str) -> bool:
    model_key = (model_base_key or "").split(":", 1)[0].strip()
    lora_key = (lora_base_key or "").split(":", 1)[0].strip()
    return bool(model_key and lora_key and lora_key == model_key)


def adapters_include_h3_turbo(adapters: Sequence[Any], registry: Any) -> bool:
    for item in adapters or ():
        lora_id, _ = adapter_id_weight(item)
        mid, _ = parse_model_version(lora_id)
        if mid == H3_TURBO_LORA_ID:
            return True
        try:
            entry = registry.require(mid)
        except KeyError:
            continue
        params = getattr(entry, "parameters", None) or {}
        if bool(params.get("h3_turbo") or params.get("h3_turbo_distill")):
            return True
    return False


def _dit_param_map(dit: Any) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    _collect_params(dit, "", out)
    return out


def _resolve_lora_file(bundle: Path) -> Path:
    root = Path(bundle)
    if root.is_file() and root.suffix == ".safetensors":
        return root
    if root.is_dir():
        named = root / H3_TURBO_BUNDLE_FILENAME
        if named.is_file():
            return named
        files = sorted(root.rglob("*.safetensors"))
        if len(files) == 1:
            return files[0]
        preferred = [f for f in files if "turbo" in f.name.lower()]
        if len(preferred) == 1:
            return preferred[0]
        if files:
            return files[0]
    raise RuntimeError(f"No MiniMax-H3 Turbo LoRA .safetensors under {root}")


def resolve_h3_turbo_lora_path(
    *,
    bundle_root: Path,
    project_root: Path,
    registry: Any,
    adapters: Sequence[Any] | None = None,
) -> Path:
    """Resolve turbo LoRA weights: explicit adapter → bundle file → default registry LoRA."""
    for item in adapters or ():
        lora_id, _ = adapter_id_weight(item)
        mid, ver = parse_model_version(lora_id)
        if mid != H3_TURBO_LORA_ID and not str(mid).startswith(H3_TURBO_LORA_ID):
            continue
        try:
            entry = registry.require(mid)
        except KeyError as exc:
            raise RuntimeError(f"Unknown MiniMax-H3 Turbo LoRA adapter {lora_id!r}") from exc
        bundle = local_bundle_root(project_root, entry, ver or None)
        if bundle is None or not Path(bundle).exists():
            block = (getattr(entry, "raw", None) or {}).get("distribution", {})
            versions = block.get("versions") if isinstance(block, dict) else {}
            ver_block = (versions or {}).get(ver or "") if isinstance(versions, dict) else {}
            lp = (ver_block or {}).get("local_path") if isinstance(ver_block, dict) else None
            raise RuntimeError(
                f"MiniMax-H3 Turbo LoRA {lora_id!r} is not installed. "
                f"Install it from Models (expected under {lp!r})."
            )
        return _resolve_lora_file(Path(bundle))

    bundle_file = Path(bundle_root) / H3_TURBO_BUNDLE_FILENAME
    if bundle_file.is_file():
        return bundle_file

    try:
        entry = registry.require(H3_TURBO_LORA_ID)
        bundle = local_bundle_root(project_root, entry, H3_TURBO_LORA_VERSION)
        if bundle is not None and Path(bundle).exists():
            return _resolve_lora_file(Path(bundle))
    except KeyError:
        pass

    raise RuntimeError(
        "MiniMax-H3 turbo mode requires Turbo LoRA weights. Install adapter "
        f"'{H3_TURBO_REGISTRY_ADAPTER}' from Models, or place "
        f"{H3_TURBO_BUNDLE_FILENAME} in {bundle_root}."
    )


def merge_minimax_h3_turbo_lora(
    dit: Any,
    *,
    weight_path: Path,
    strength: float,
    ctx: Any,
    on_log: Callable[[str, str], None] | None = None,
) -> int:
    """Merge turbo LoRA into ``MiniMaxH3DiTMLX`` (dense deltas or A/B pairs; quant-aware)."""
    load_fn = getattr(ctx, "load_weights", None)
    weights = load_weights_dict(load_fn, str(weight_path))
    if not weights:
        raise RuntimeError(f"MiniMax-H3 Turbo LoRA empty: {weight_path}")

    param_map = _dit_param_map(dit)
    lora_config = read_lora_config(Path(weight_path))
    config_alpha = lora_config.get("lora_alpha", lora_config.get("alpha"))
    default_alpha = float(config_alpha) if config_alpha is not None else 128.0
    groups = remap_minimax_h3_lora_keys(weights, default_alpha=default_alpha)
    dense: dict[str, mx.array] = {}
    for key, tensor in weights.items():
        if ".lora_" in key.lower():
            continue
        if key.endswith(".delta.weight"):
            dense[key[: -len(".delta.weight")]] = tensor
        elif key in param_map:
            module = key[: -len(".weight")] if key.endswith(".weight") else key
            dense[module] = tensor

    applied = 0
    bits = None
    group_size = 64
    quant_cfg = getattr(getattr(dit, "config", None), "quantization", None)
    if isinstance(quant_cfg, dict):
        bits = int(quant_cfg.get("bits", 0) or 0) or None
        group_size = int(quant_cfg.get("group_size", 64) or 64)

    class _MergeTarget:
        _param_map = param_map

    target = _MergeTarget()

    for module, delta in dense.items():
        wkey = minimax_h3_lora_param_key(module)
        if wkey not in param_map:
            continue
        scaled = float(strength) * delta.astype(mx.float32)
        if bits in (4, 8):
            apply_lora_delta_to_weight(
                model=target,
                wkey=wkey,
                delta=scaled,
                ctx=ctx,
                bits=bits,
                group_size=group_size,
            )
        else:
            param = param_map[wkey]
            param[:] = (param.astype(mx.float32) + scaled).astype(param.dtype)
        applied += 1

    for module, (down, up, alpha) in groups.items():
        wkey = minimax_h3_lora_param_key(module)
        if wkey not in param_map:
            continue
        param = param_map[wkey]
        out_d, in_d = int(param.shape[0]), int(param.shape[1])
        d_orient, u_orient, rank = orient_lora_pair(
            down,
            up,
            out_d=out_d,
            in_d=in_d,
            lora_id=H3_TURBO_LORA_ID,
            wkey=wkey,
        )
        scale = (float(alpha) / float(rank)) * float(strength)
        delta = mx.matmul(u_orient.astype(mx.float32), d_orient.astype(mx.float32))
        scaled_delta = scale * delta
        if bits in (4, 8):
            apply_lora_delta_to_weight(
                model=target,
                wkey=wkey,
                delta=scaled_delta,
                ctx=ctx,
                bits=bits,
                group_size=group_size,
            )
        else:
            param[:] = (param.astype(mx.float32) + scaled_delta).astype(param.dtype)
        applied += 1

    if applied == 0:
        raise RuntimeError(
            f"MiniMax-H3 Turbo LoRA {weight_path.name} matched 0 DiT parameters "
            f"({len(groups)} A/B groups, {len(dense)} dense keys). "
            "Check LoRA checkpoint layout vs ddalcu Diffusers DiT keys."
        )
    run_eval(getattr(ctx, "eval", None), dit.parameters())
    if on_log:
        on_log("info", f"MiniMax-H3 turbo LoRA merged ({applied} tensors) from {weight_path.name}")
    return applied


def apply_minimax_h3_turbo_lora(
    dit: Any,
    *,
    bundle_root: Path,
    config: Any,
    adapters: Sequence[Any] | None,
    project_root: Path,
    registry: Any,
    ctx: Any,
    on_log: Callable[[str, str], None] | None = None,
) -> None:
    """Apply turbo LoRA when ``h3_turbo`` or a compatible adapter is present."""
    use_turbo = bool(getattr(config, "h3_turbo", False)) or adapters_include_h3_turbo(
        adapters or (), registry
    )
    if not use_turbo:
        return

    strength = 1.0
    for item in adapters or ():
        lora_id, weight = adapter_id_weight(item)
        mid, _ = parse_model_version(lora_id)
        if mid == H3_TURBO_LORA_ID:
            strength = float(weight)
            break

    path = resolve_h3_turbo_lora_path(
        bundle_root=bundle_root,
        project_root=project_root,
        registry=registry,
        adapters=adapters,
    )
    merge_minimax_h3_turbo_lora(
        dit,
        weight_path=path,
        strength=strength,
        ctx=ctx,
        on_log=on_log,
    )


def merge_minimax_h3_lora_adapters(
    model: Any,
    adapters: Sequence[Any],
    *,
    base_model_id: str,
    project_root: Path,
    registry: Any,
    ctx: Any,
    on_log: Callable[[str, str], None] | None = None,
) -> None:
    """Merge registry LoRA adapters onto H3 DiT (turbo distill)."""
    _ = project_root
    if not adapters:
        return
    for item in adapters:
        lora_id, strength = adapter_id_weight(item)
        mid, ver = parse_model_version(lora_id)
        try:
            entry = registry.require(mid)
        except KeyError as exc:
            raise RuntimeError(f"Unknown LoRA adapter {lora_id!r}") from exc
        raw = getattr(entry, "raw", None) or {}
        if str(raw.get("category") or "") != "loras":
            raise RuntimeError(f"Adapter {lora_id!r} is not a LoRA (category={raw.get('category')!r}).")
        declared = str(raw.get("base_model") or "").strip()
        if declared and not minimax_h3_video_lora_base_compatible(base_model_id, declared):
            raise RuntimeError(
                f"LoRA {mid!r} is scoped to base_model={declared!r}, "
                f"but the request uses {base_model_id!r}."
            )
        bundle = local_bundle_root(project_root, entry, ver or None)
        if bundle is None or not Path(bundle).exists():
            raise RuntimeError(
                f"LoRA {lora_id!r} is not installed. Download it from the Models page."
            )
        path = _resolve_lora_file(Path(bundle))
        merge_minimax_h3_turbo_lora(
            model,
            weight_path=path,
            strength=float(strength),
            ctx=ctx,
            on_log=on_log,
        )
