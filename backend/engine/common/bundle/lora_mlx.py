"""Shared MLX LoRA merge skeleton for image transformer families."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Sequence

import mlx.core as mx

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


def read_lora_config(bundle: Path) -> dict[str, Any]:
    """Read optional ``lora_config.json`` beside a LoRA safetensors file or directory."""
    if bundle.is_file():
        config_path = bundle.parent / "lora_config.json"
    elif bundle.is_dir():
        config_path = bundle / "lora_config.json"
        if not config_path.is_file():
            candidates = sorted(bundle.glob("*.safetensors"))
            if len(candidates) == 1:
                config_path = candidates[0].parent / "lora_config.json"
    else:
        return {}
    if not config_path.is_file():
        return {}
    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise RuntimeError(f"LoRA config must be a JSON object: {config_path}")
        return data
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read LoRA config {config_path}: {exc}") from exc


def _weights_use_indexed_lora_keys(weights: dict[str, Any]) -> bool:
    return any(re.match(r"^lora_\d+\.lora_[AB]\.weight$", key) for key in weights)


def _apply_config_alpha(
    groups: dict[str, tuple[mx.array, mx.array, float]],
    *,
    config_alpha: float | None,
) -> dict[str, tuple[mx.array, mx.array, float]]:
    if config_alpha is None:
        return groups
    alpha = float(config_alpha)
    return {key: (down, up, alpha) for key, (down, up, _) in groups.items()}


def _merge_param_map(model: Any) -> dict[str, Any]:
    """Use the implementation's map when a delegating stem holds a stale snapshot."""
    inner = getattr(model, "_inner", None)
    if inner is not None and hasattr(inner, "_param_map"):
        return inner._param_map
    return model._param_map


def _apply_dense_delta(model: Any, *, wkey: str, delta: mx.array, ctx: RuntimeContext, bits: Any) -> None:
    """Apply a dense effective-weight delta, re-quantizing packed layers when needed."""
    param = _merge_param_map(model)[wkey]
    from backend.engine.common.model.quantized_lora_mlx import apply_lora_delta_to_weight

    if bits is not None and getattr(bits, "kind", "dense") == "quantized" and bits.bits in (4, 8):
        apply_lora_delta_to_weight(
            model=model,
            wkey=wkey,
            delta=delta,
            ctx=ctx,
            bits=int(bits.bits),
            group_size=int(getattr(bits, "group_size", 64) or 64),
        )
    else:
        param[:] = (param.astype(mx.float32) + delta).astype(param.dtype)


def _dora_effective_delta(
    *,
    param: mx.array,
    down: mx.array,
    up: mx.array,
    magnitude: mx.array,
    scale: float,
    strength: float,
    lora_id: str,
    wkey: str,
) -> mx.array:
    """DoRA: ``m * normalize(W + ΔW) - W``; never discard magnitude/normalization."""
    w = param.astype(mx.float32)
    v = w + float(scale) * mx.matmul(up.astype(mx.float32), down.astype(mx.float32))
    norms = mx.sqrt(mx.sum(v * v, axis=1, keepdims=True))
    if bool(mx.any(norms <= 1e-8).item()):
        raise RuntimeError(f"DoRA {lora_id!r}: zero direction norm for {wkey}.")
    m = magnitude.astype(mx.float32)
    if m.ndim == 2 and m.shape[1] == 1:
        m = m[:, 0]
    if m.ndim != 1 or m.shape[0] != w.shape[0]:
        raise RuntimeError(
            f"DoRA {lora_id!r}: magnitude shape {tuple(magnitude.shape)} "
            f"does not match {wkey} output dimension {w.shape[0]}."
        )
    return float(strength) * (m[:, None] * (v / mx.maximum(norms, 1e-8)) - w)


def orient_lora_pair(
    down: mx.array,
    up: mx.array,
    *,
    out_d: int,
    in_d: int,
    lora_id: str,
    wkey: str,
) -> tuple[mx.array, mx.array, int]:
    """Return (down, up) with shapes [rank, in_d] and [out_d, rank]."""
    d, u = down, up
    if d.shape[1] == in_d and u.shape[0] == out_d and u.shape[1] == d.shape[0]:
        return d, u, int(d.shape[0])
    if d.shape[0] == in_d and u.shape[0] == out_d and u.shape[1] == d.shape[1]:
        return mx.transpose(d, (1, 0)), u, int(d.shape[1])
    if d.shape[1] == in_d and u.shape[1] == out_d and u.shape[0] == d.shape[0]:
        return d, mx.transpose(u, (1, 0)), int(d.shape[0])
    if d.shape[0] == in_d and u.shape[1] == out_d and u.shape[0] == d.shape[1]:
        return mx.transpose(d, (1, 0)), mx.transpose(u, (1, 0)), int(d.shape[1])
    raise RuntimeError(
        f"LoRA {lora_id!r} tensor shape mismatch for {wkey}: model ({out_d}, {in_d}), "
        f"lora_down {tuple(d.shape)}, lora_up {tuple(u.shape)}."
    )


def merge_lora_adapters_common(
    *,
    model: Any,
    adapters: Sequence[Any],
    base_model_id: str,
    project_root: Path,
    registry: Any,
    ctx: RuntimeContext,
    family_name: str,
    remap_groups: Callable[[dict[str, Any]], dict[str, tuple[mx.array, mx.array, float]]],
    param_key_for_module: Callable[[str], str],
    remap_module_prefix: Callable[[str], str] | None = None,
    base_model_scope_key: Callable[[str], str] | None = None,
    repair_indexed_weights: Callable[[dict[str, Any], Any, dict[str, Any]], dict[str, Any]] | None = None,
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
            if entry is not None:
                from backend.engine.contracts.pipeline_registry import resolve_version_block

                block = resolve_version_block(entry, ver or None)
                lp = (block or {}).get("local_path") if block else None
                if block and lp:
                    raise RuntimeError(
                        f"LoRA {lora_id!r} is not installed. Download it from the Models page "
                        f"(expected under {lp!r}, version {ver or 'default'})."
                    )
            raise RuntimeError(
                f"LoRA {lora_id!r} is not installed on disk (missing registry versions.local_path "
                f"for version {ver or 'default'})."
            )
        weights = load_lora_flat_weights(bundle, ctx)
        lora_config = read_lora_config(bundle)
        if repair_indexed_weights is not None and _weights_use_indexed_lora_keys(weights):
            weights = repair_indexed_weights(weights, model, lora_config)
            if on_log:
                on_log("info", f"lora indexed key repair source={mid}")
        groups = remap_groups(weights)
        dense_deltas = {
            k[: -len(".delta.weight")]: v
            for k, v in weights.items()
            if k.endswith(".delta.weight")
        }
        magnitudes = {
            key[: -len(".dora_magnitude.weight")]: value
            for key, value in weights.items()
            if key.endswith(".dora_magnitude.weight")
        }
        if magnitudes and not groups:
            raise RuntimeError(f"DoRA {lora_id!r}: magnitude tensors have no LoRA factor pairs.")
        if magnitudes and dense_deltas:
            raise RuntimeError(f"Adapter {lora_id!r}: DoRA factors cannot be mixed with dense deltas.")
        if not groups and not dense_deltas:
            raise RuntimeError(
                f"LoRA {lora_id!r}: after key remap no (lora_down, lora_up) pairs were found."
            )
        config_alpha = lora_config.get("lora_alpha", lora_config.get("alpha"))
        if config_alpha is not None and not any(".alpha" in key.lower() for key in weights):
            groups = _apply_config_alpha(groups, config_alpha=float(config_alpha))
        param_map = _merge_param_map(model)
        unmatched = [
            param_key_for_module(name)
            for name in (*groups.keys(), *dense_deltas.keys())
            if param_key_for_module(name) not in param_map
        ]
        if unmatched:
            raise RuntimeError(
                f"LoRA {lora_id!r}: {len(unmatched)} weight groups do not match this transformer "
                f"(e.g. {unmatched[:4]}). Check base model, architecture and lora_blocks; "
                "no partial adapter was applied."
            )
        magnitude_by_wkey: dict[str, Any] = {}
        for module_name, magnitude in magnitudes.items():
            if remap_module_prefix is None:
                raise RuntimeError(f"DoRA {lora_id!r}: no module prefix remapper is configured.")
            wkey = param_key_for_module(remap_module_prefix(module_name))
            if wkey not in param_map or wkey not in {
                param_key_for_module(name) for name in groups
            }:
                raise RuntimeError(f"DoRA {lora_id!r}: unmatched magnitude tensor {module_name!r}.")
            magnitude_by_wkey[wkey] = magnitude
        if on_log:
            on_log(
                "info",
                f"lora scale source={mid} alpha={config_alpha if config_alpha is not None else 'tensor-or-rank'} "
                f"rank={lora_config.get('lora_rank', 'per-tensor')} train_type={lora_config.get('train_type', 'lora')}"
                + (" dora_effective_weight=yes" if magnitudes else ""),
            )
        applied = 0
        # [diag] merge telemetry for train/inference comparison
        _group_total = len(groups)
        _dense_total = len(dense_deltas)
        _delta_means: list[mx.array] = []
        _ranks: list[int] = []
        from backend.engine.common.model.quantized_lora_mlx import inference_mode_from_model

        inference_mode = inference_mode_from_model(model)
        for module_name, delta in dense_deltas.items():
            wkey = param_key_for_module(module_name)
            scaled = float(strength) * delta.astype(mx.float32)
            _delta_means.append(mx.mean(mx.abs(scaled)))
            _apply_dense_delta(model, wkey=wkey, delta=scaled, ctx=ctx, bits=inference_mode)
            applied += 1
        for module_name, (down, up, alpha) in groups.items():
            wkey = param_key_for_module(module_name)
            param = _merge_param_map(model)[wkey]
            magnitude = magnitude_by_wkey.get(wkey)
            # LoRA/DoRA factors are dense ``(out, in)``; packed uint32 layouts must be
            # dequantized before shape checks and DoRA direction normalization.
            dense_param = param
            if str(param.dtype) == "uint32":
                base = wkey[: -len(".weight")]
                scales = _merge_param_map(model)[f"{base}.scales"]
                biases = _merge_param_map(model)[f"{base}.biases"]
                from backend.engine.common.model.base import _mlx_affine_infer_bits_and_group_size

                qbits, qgroup = _mlx_affine_infer_bits_and_group_size(
                    param,
                    scales,
                    dense_weight_shape=None,
                    weight_key=wkey,
                    bundle_affine_bits=(
                        int(inference_mode.bits)
                        if inference_mode is not None
                        and getattr(inference_mode, "kind", "") == "quantized"
                        and getattr(inference_mode, "bits", None) in (4, 8)
                        else None
                    ),
                )
                dense_param = ctx.dequantize(
                    param, scales=scales, biases=biases, group_size=qgroup, bits=qbits
                )
            out_d, in_d = int(dense_param.shape[0]), int(dense_param.shape[1])
            d_orient, u_orient, rank = orient_lora_pair(
                down, up, out_d=out_d, in_d=in_d, lora_id=lora_id, wkey=wkey
            )
            if rank <= 0:
                raise RuntimeError(f"LoRA {lora_id!r}: invalid rank for {wkey}.")
            scale = float(alpha) / float(rank)
            if magnitude is not None:
                scaled_delta = _dora_effective_delta(
                    param=dense_param,
                    down=d_orient,
                    up=u_orient,
                    magnitude=magnitude,
                    scale=scale,
                    strength=float(strength),
                    lora_id=lora_id,
                    wkey=wkey,
                )
            else:
                scaled_delta = float(strength) * scale * mx.matmul(
                    u_orient.astype(mx.float32), d_orient.astype(mx.float32)
                )
            _delta_means.append(mx.mean(mx.abs(scaled_delta)))
            _ranks.append(int(rank))
            _apply_dense_delta(model, wkey=wkey, delta=scaled_delta, ctx=ctx, bits=inference_mode)
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
            total = _group_total + _dense_total
            skipped = total - applied
            rank_note = ""
            if _ranks:
                rmin, rmax = min(_ranks), max(_ranks)
                rank_note = f" rank={rmin}" if rmin == rmax else f" rank={rmin}-{rmax}"
            delta_note = ""
            if _delta_means:
                try:
                    mean_abs = mx.mean(mx.stack(_delta_means))
                    ctx.eval(mean_abs)
                    delta_note = f" mean|Δw|={float(mean_abs.item()):.3e}"
                except Exception:  # noqa: BLE001 — telemetry must not break merge
                    delta_note = ""
            on_log(
                "info",
                f"[diag] lora merged source={mid} strength={strength} "
                f"matched={applied}/{total} skipped={skipped}{rank_note}{delta_note}{quant_note} "
                "(mean|Δw|≈0 ⇒ LoRA barely affects output; skipped>0 ⇒ key/scope mismatch)",
            )
    ctx.eval(*[t for _, t in model.parameters()])
