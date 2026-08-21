"""Z-Image-Turbo DiT weight merge (weighted sum / add difference)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Literal

MergeMethod = Literal["weighted_sum", "add_difference"]


def _is_quantized_flat(weights: dict[str, Any]) -> bool:
    for key in weights:
        if key.endswith(".scales") or key.endswith(".biases") or "quant" in key.lower():
            return True
    return False


def load_z_image_dit_weights(bundle_root: Path, *, ctx: Any) -> dict[str, Any]:
    transformer_dir = bundle_root / "transformer"
    if not transformer_dir.is_dir():
        raise RuntimeError(f"Z-Image bundle missing transformer/ under {bundle_root}")
    flat: dict[str, Any] = {}
    files = sorted(transformer_dir.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"No transformer/*.safetensors under {bundle_root}")
    for sf in files:
        flat.update(ctx.load_weights(str(sf)))
    if not flat:
        raise RuntimeError(f"Empty transformer weights under {transformer_dir}")
    if _is_quantized_flat(flat):
        raise RuntimeError(
            "Merging quantized (int4/int8) Z-Image weights is not supported; use FP16/BF16 bundles"
        )
    return flat


def resolve_z_image_bundle_root(
    registry: Any,
    project_root: Path,
    model_id: str,
) -> Path:
    from backend.core.contracts import parse_model_version
    from backend.engine.contracts.pipeline_registry import local_bundle_root, resolve_version_block

    mid, ver = parse_model_version(model_id)
    entry = registry.require(mid)
    family = str(getattr(entry, "family", "") or "")
    if family != "z_image":
        raise RuntimeError(f"model {mid!r} is family={family!r}; only z_image merge is supported")
    block = resolve_version_block(entry, ver or None)
    root = local_bundle_root(project_root, entry, ver or None)
    if root is None or not root.is_dir():
        raise RuntimeError(f"Model bundle not installed for {model_id!r} (expected under models/Image/)")
    return root


def weighted_sum_merge(
    weights_a: dict[str, Any],
    weights_b: dict[str, Any],
    *,
    alpha: float,
    ctx: Any,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    alpha = float(max(0.0, min(1.0, alpha)))
    for key, a in weights_a.items():
        b = weights_b.get(key)
        if b is None:
            merged[key] = a
            continue
        if tuple(getattr(a, "shape", ())) != tuple(getattr(b, "shape", ())):
            raise RuntimeError(f"merge shape mismatch for {key!r}: {a.shape} vs {b.shape}")
        merged[key] = (1.0 - alpha) * a + alpha * b
    for key, b in weights_b.items():
        if key not in merged:
            merged[key] = b
    return merged


def add_difference_merge(
    weights_a: dict[str, Any],
    weights_b: dict[str, Any],
    weights_c: dict[str, Any],
    *,
    alpha: float,
    ctx: Any,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    alpha = float(alpha)
    for key, a in weights_a.items():
        b = weights_b.get(key)
        c = weights_c.get(key)
        if b is not None and c is not None:
            if tuple(a.shape) != tuple(b.shape) or tuple(a.shape) != tuple(c.shape):
                raise RuntimeError(f"merge shape mismatch for {key!r}")
            merged[key] = a + alpha * (b - c)
        elif b is not None and tuple(a.shape) == tuple(b.shape):
            merged[key] = (1.0 - alpha * 0.5) * a + (alpha * 0.5) * b
        else:
            merged[key] = a
    return merged


def save_merged_transformer_shard(
    out_transformer_dir: Path,
    merged: dict[str, Any],
    *,
    ctx: Any,
) -> Path:
    out_transformer_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_transformer_dir / "merged.safetensors"
    ctx.save_weights(merged, str(out_path))
    return out_path


def run_z_image_merge(
    *,
    registry: Any,
    project_root: Path,
    ctx: Any,
    method: MergeMethod,
    model_a: str,
    model_b: str,
    model_c: str | None,
    alpha: float,
    output_name: str,
    work_dir: Path,
    on_log: Callable[[str, str], None] | None = None,
    auto_register: bool = True,
    registry_path: Path | None = None,
    config_dir: Path | None = None,
    task_id: str = "",
) -> dict[str, Any]:
    name = (output_name or "").strip()
    if not name:
        raise RuntimeError("output_name is required")
    if not name.replace("-", "").replace("_", "").isalnum():
        raise RuntimeError(f"invalid output_name {name!r} (use letters, digits, hyphen, underscore)")

    root_a = resolve_z_image_bundle_root(registry, project_root, model_a)
    root_b = resolve_z_image_bundle_root(registry, project_root, model_b)
    if on_log:
        on_log("info", f"z_image merge load A={model_a} B={model_b} method={method} alpha={alpha:.3f}")

    wa = load_z_image_dit_weights(root_a, ctx=ctx)
    wb = load_z_image_dit_weights(root_b, ctx=ctx)

    if method == "weighted_sum":
        merged = weighted_sum_merge(wa, wb, alpha=alpha, ctx=ctx)
    elif method == "add_difference":
        if not model_c:
            raise RuntimeError("add_difference merge requires model_c (original fine-tune base)")
        root_c = resolve_z_image_bundle_root(registry, project_root, model_c)
        wc = load_z_image_dit_weights(root_c, ctx=ctx)
        merged = add_difference_merge(wa, wb, wc, alpha=alpha, ctx=ctx)
    else:
        raise RuntimeError(f"unknown merge method {method!r}")

    out_root = project_root / "models" / "Image" / f"{name}-fp16"
    out_transformer = out_root / "transformer"
    shard_path = save_merged_transformer_shard(out_transformer, merged, ctx=ctx)

    # Copy non-transformer components from model A so bundle is runnable with base VAE/TE.
    for component in ("vae", "text_encoder", "tokenizer", "scheduler"):
        src = root_a / component
        dst = out_root / component
        if src.is_dir() and not dst.exists():
            shutil.copytree(src, dst, symlinks=True)

    for meta_file in ("model_index.json", "config.json"):
        src = root_a / meta_file
        dst = out_root / meta_file
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)

    manifest = {
        "merge_method": method,
        "alpha": alpha,
        "model_a": model_a,
        "model_b": model_b,
        "model_c": model_c,
        "base_bundle": str(root_a),
        "transformer_shard": str(shard_path.relative_to(project_root)),
        "keys_merged": len(merged),
    }
    manifest_path = out_root / "merge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if on_log:
        on_log("info", f"z_image merge saved to {out_root} ({len(merged)} keys)")

    registered_model_id = ""
    if auto_register:
        if registry_path is None or config_dir is None:
            raise RuntimeError("auto_register requires registry_path and config_dir")
        from backend.engine.tools.user_merged_model_registry import register_merged_z_image_model

        rel_local = f"models/Image/{name}-fp16"
        row = register_merged_z_image_model(
            registry_path=registry_path,
            config_dir=config_dir,
            output_name=name,
            local_path=rel_local,
            template_model_id=model_a,
            merge_manifest=manifest,
            task_id=task_id,
        )
        registered_model_id = str(row.get("id") or "")
        if on_log:
            on_log("info", f"z_image merge registered model_id={registered_model_id}")

    return {
        "output_name": name,
        "output_bundle": str(out_root),
        "transformer_path": str(shard_path),
        "manifest": manifest,
        "registered_model_id": registered_model_id,
    }
