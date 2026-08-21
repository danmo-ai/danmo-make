"""Registry-driven MLX DiT weight inference mode (dense vs quantized)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

AuxComponent = Literal["text_encoder", "vae"]


@dataclass(frozen=True)
class WeightInferenceMode:
    kind: Literal["dense", "quantized"]
    bits: int | None = None
    group_size: int = 64

    def cache_suffix(self) -> str:
        if self.kind == "quantized" and self.bits in (4, 8):
            return f":q{self.bits}"
        return ":dense"

    def log_label(self) -> str:
        if self.kind == "quantized" and self.bits:
            return f"inference=quantized bits={self.bits}"
        return "inference=dense"


def _version_quantization_block(entry: Any, version_key: str | None) -> dict[str, Any]:
    from backend.engine.contracts import resolve_version_block

    ver = resolve_version_block(entry, version_key)
    if not ver:
        raw = getattr(entry, "raw", None) or {}
        versions = raw.get("versions") or {}
        ver = versions.get(version_key or "") or {}
    return dict(ver.get("quantization") or {})


def entry_version_declares_quantization(entry: Any, version_key: str | None) -> bool:
    quant = _version_quantization_block(entry, version_key)
    return quant.get("bits") in (4, 8)


def _require_mlx_quant_inference(ctx: Any, *, version: str) -> None:
    backend = str(getattr(ctx, "backend", "") or "")
    if backend != "mlx":
        raise RuntimeError(_i18n("error.quantized_inference_mlx_only", version=version))


def resolve_inference_weight_mode_from_bundle(
    ctx: Any,
    *,
    weight_keys: set[str] | frozenset[str] | None = None,
    bundle_affine_bits: int | None = None,
) -> WeightInferenceMode:
    """Infer mode from checkpoint tensors when registry entry is unavailable (e.g. LTX lazy load)."""
    keys = weight_keys or frozenset()
    if bundle_affine_bits not in (4, 8) or not any(k.endswith(".scales") for k in keys):
        return WeightInferenceMode(kind="dense")
    _require_mlx_quant_inference(ctx, version="bundle")
    return WeightInferenceMode(kind="quantized", bits=int(bundle_affine_bits))


def resolve_inference_weight_mode(
    entry: Any,
    version_key: str | None,
    ctx: Any,
    *,
    weight_keys: set[str] | frozenset[str] | None = None,
    bundle_affine_bits: int | None = None,
) -> WeightInferenceMode:
    """Resolve how DiT weights should be loaded for inference (registry + bundle)."""
    quant = _version_quantization_block(entry, version_key)
    bits = quant.get("bits")
    scheme = str(quant.get("scheme") or "").strip()
    inference = str(quant.get("inference") or "").strip().lower()

    if bits not in (4, 8):
        return WeightInferenceMode(kind="dense")

    if inference == "dense":
        return WeightInferenceMode(kind="dense")

    _require_mlx_quant_inference(ctx, version=version_key or "default")

    if scheme and scheme != "mlx_affine":
        raise RuntimeError(
            f"Unsupported quantization scheme {scheme!r} for version {version_key!r}; "
            "only mlx_affine supports quantized inference."
        )

    keys = weight_keys or frozenset()
    if keys and not any(k.endswith(".scales") for k in keys):
        raise RuntimeError(
            f"Registry declares {bits}-bit quantized inference for version {version_key!r}, "
            "but the bundle has no MLX affine tensors (*.scales). "
            "Reinstall or re-convert the quantized weight folder."
        )

    if bundle_affine_bits is not None and int(bundle_affine_bits) != int(bits):
        raise RuntimeError(
            f"Registry quantization.bits={bits} conflicts with bundle metadata "
            f"quantization_level={bundle_affine_bits}."
        )

    group_size = int(quant.get("group_size") or 64)
    return WeightInferenceMode(kind="quantized", bits=int(bits), group_size=group_size)


def resolve_component_inference_weight_mode(
    entry: Any,
    version_key: str | None,
    ctx: Any,
    *,
    component: AuxComponent,
    weight_keys: set[str] | frozenset[str] | None = None,
    bundle_affine_bits: int | None = None,
) -> WeightInferenceMode:
    """Resolve TE/VAE inference mode (default dense; quant only when affine tensors exist)."""
    keys = weight_keys or frozenset()
    has_affine = any(k.endswith(".scales") for k in keys)

    quant = _version_quantization_block(entry, version_key)
    comp = dict(quant.get(component) or {})

    bits = comp.get("bits")
    if bits is None and has_affine and quant.get("bits") in (4, 8):
        bits = quant.get("bits")

    if bits not in (4, 8):
        return WeightInferenceMode(kind="dense")

    inference = str(comp.get("inference") or quant.get("inference") or "").strip().lower()
    if inference == "dense":
        return WeightInferenceMode(kind="dense")

    if comp.get("bits") in (4, 8) and not has_affine:
        raise RuntimeError(
            f"Registry declares {component} {bits}-bit quantized inference for version "
            f"{version_key!r}, but weights have no MLX affine tensors (*.scales)."
        )

    if not has_affine:
        return WeightInferenceMode(kind="dense")

    _require_mlx_quant_inference(ctx, version=version_key or "default")

    scheme = str(comp.get("scheme") or quant.get("scheme") or "").strip()
    if scheme and scheme != "mlx_affine":
        raise RuntimeError(
            f"Unsupported quantization scheme {scheme!r} for {component} on version {version_key!r}; "
            "only mlx_affine supports quantized inference."
        )

    if bundle_affine_bits is not None and int(bundle_affine_bits) != int(bits):
        raise RuntimeError(
            f"Registry {component} quantization.bits={bits} conflicts with bundle metadata "
            f"quantization_level={bundle_affine_bits}."
        )

    return WeightInferenceMode(kind="quantized", bits=int(bits))


def resolve_dit_inference_weight_mode(
    ctx: Any,
    *,
    entry: Any | None = None,
    version_key: str | None = None,
    weight_keys: set[str] | frozenset[str] | None = None,
    bundle_affine_bits: int | None = None,
) -> WeightInferenceMode:
    """Registry-first DiT mode; fall back to bundle metadata when entry is omitted."""
    if entry is not None and entry_version_declares_quantization(entry, version_key):
        return resolve_inference_weight_mode(
            entry,
            version_key,
            ctx,
            weight_keys=weight_keys,
            bundle_affine_bits=bundle_affine_bits,
        )
    return resolve_inference_weight_mode_from_bundle(
        ctx,
        weight_keys=weight_keys,
        bundle_affine_bits=bundle_affine_bits,
    )


def estimate_dit_cache_size_gb(disk_size_gb: float, mode: WeightInferenceMode) -> float:
    """Approximate in-memory DiT footprint for ModelCache LRU budgeting."""
    if mode.kind == "quantized" and mode.bits in (4, 8):
        return max(disk_size_gb * (float(mode.bits) / 16.0), 0.5)
    return disk_size_gb


def entry_allows_quantized_lora(entry: Any, version_key: str | None) -> bool:
    """Whether LoRA merge on a quantized DiT version is allowed (default: true)."""
    if not entry_version_declares_quantization(entry, version_key):
        return True
    quant = _version_quantization_block(entry, version_key)
    if str(quant.get("inference") or "").strip().lower() == "dense":
        return True
    raw = quant.get("quantized_lora")
    if raw is None:
        return True
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def assert_quantized_dit_lora_compatible(
    entry: Any,
    version_key: str | None,
    adapters: Any,
) -> None:
    """Fail loud when LoRA + quantized DiT is explicitly disabled in registry."""
    if not adapters:
        return
    if not entry_version_declares_quantization(entry, version_key):
        return
    quant = _version_quantization_block(entry, version_key)
    if str(quant.get("inference") or "").strip().lower() == "dense":
        return
    if entry_allows_quantized_lora(entry, version_key):
        return
    raise RuntimeError(_i18n("error.quantized_dit_lora_unsupported"))


def _i18n(key: str, **params: Any) -> str:
    try:
        from backend.core.i18n import t

        return t(key, "en", **params)
    except Exception:
        return key.format(**params) if params else key
