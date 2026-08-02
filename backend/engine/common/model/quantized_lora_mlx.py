"""LoRA merge on quantized DiT: dense delta → re-quantize touched layers."""

from __future__ import annotations

from typing import Any

from backend.engine.common.model.base import _mlx_affine_infer_bits_and_group_size


def inference_mode_from_model(model: Any) -> Any | None:
    mode = getattr(model, "_dq_inference_mode", None)
    if mode is not None:
        return mode
    inner = getattr(model, "_inner", None)
    if inner is not None:
        return getattr(inner, "_dq_inference_mode", None)
    return None


def _is_quantized_affine_weight(weight: Any) -> bool:
    dtype = getattr(weight, "dtype", None)
    name = str(getattr(dtype, "name", dtype) or "")
    shape = getattr(weight, "shape", ())
    return name == "uint32" and len(shape) == 2


def _rebind_module_at_base(root: Any, base: str, module: Any) -> None:
    parts = base.split(".")
    parent: Any = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    setattr(parent, parts[-1], module)


def _merge_root_for_param_map(model: Any) -> Any:
    inner = getattr(model, "_inner", None)
    if inner is not None and hasattr(inner, "_param_map"):
        return inner
    return model


def apply_lora_delta_to_weight(
    *,
    model: Any,
    wkey: str,
    delta: Any,
    ctx: Any,
    bits: int,
    group_size: int = 64,
) -> None:
    """Apply LoRA low-rank delta to a ``.weight`` param (dense or affine-quantized)."""
    import mlx.core as mx
    from importlib import import_module

    nn = import_module("mlx.nn")

    root = _merge_root_for_param_map(model)
    if not hasattr(root, "_param_map") or wkey not in root._param_map:
        raise RuntimeError(f"LoRA merge: missing parameter {wkey!r} in model._param_map")

    param = root._param_map[wkey]
    base = wkey[:-7] if wkey.endswith(".weight") else wkey

    if not _is_quantized_affine_weight(param):
        updated = param.astype(mx.float32) + delta
        param[:] = updated.astype(param.dtype)
        return

    scales_key = f"{base}.scales"
    biases_key = f"{base}.biases"
    scales = root._param_map.get(scales_key)
    if scales is None:
        raise RuntimeError(
            f"LoRA merge on quantized layer {base!r} requires {scales_key!r} in _param_map"
        )
    biases = root._param_map.get(biases_key)
    dense_shape = (int(delta.shape[0]), int(delta.shape[1]))
    bits_eff, gs = _mlx_affine_infer_bits_and_group_size(
        param,
        scales,
        dense_weight_shape=dense_shape,
        weight_key=wkey,
        bundle_affine_bits=bits,
    )
    if bits_eff != bits:
        raise RuntimeError(
            f"LoRA re-quantize bits mismatch for {wkey}: layer={bits_eff}, model={bits}"
        )
    if gs != group_size:
        group_size = gs

    dense_w = ctx.dequantize(
        param,
        scales=scales,
        biases=biases,
        group_size=group_size,
        bits=bits_eff,
    )
    dense_updated = dense_w.astype(mx.float32) + delta

    bias_key = f"{base}.bias"
    bias_param = root._param_map.get(bias_key)
    has_bias = bias_param is not None

    in_features = int(dense_updated.shape[1])
    out_features = int(dense_updated.shape[0])
    if in_features % group_size != 0:
        raise RuntimeError(
            f"LoRA re-quantize failed for {base!r}: in_features={in_features} "
            f"not divisible by group_size={group_size}"
        )

    linear = nn.Linear(in_features, out_features, bias=has_bias)
    linear.weight = dense_updated.astype(dense_w.dtype)
    if has_bias:
        linear.bias = bias_param

    q_linear = linear.to_quantized(bits=bits, group_size=group_size)
    _rebind_module_at_base(root, base, q_linear)

    if hasattr(root, "_build_param_map"):
        root._build_param_map()
    if root is not model and hasattr(model, "_param_map"):
        model._param_map = getattr(root, "_param_map", model._param_map)
