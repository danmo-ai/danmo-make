"""
Model base class — common interface for all image/video Transformers.

TransformerBase provides the unified interface; family transformers inherit
and override ``forward``, ``load_weights``, hooks, etc.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.engine.common.bundle.quant_inference import WeightInferenceMode


def _mlx_affine_infer_bits_and_group_size(
    qw: Any,
    qs: Any,
    *,
    dense_weight_shape: tuple[int, int] | None,
    weight_key: str,
    bundle_affine_bits: int | None = None,
) -> tuple[int, int]:
    """Infer ``(bits, group_size)`` for MLX **affine** QuantizedLinear-style checkpoints.

    Packed ``weight`` is ``uint32`` along the **last** axis; each uint32 holds ``32 // bits``
    consecutive unsigned integer codes in ``[0, 2**bits - 1]`` (see MLX ``mx.quantize`` /
    ``nn.Linear.to_quantized``). ``scales`` / ``biases`` use one entry per input-axis group;
    the last tensor dimension is the group count along the input feature axis.

    Prefer the model's dense ``Linear.weight`` shape ``(out, in)`` when available so 4- vs
    8-bit packing is unambiguous.
    """
    qw_shape = tuple(int(x) for x in qw.shape)
    qs_shape = tuple(int(x) for x in qs.shape)
    if len(qw_shape) < 2:
        raise RuntimeError(
            f"{weight_key}: expected quantized weight with ndim >= 2, got shape {qw_shape}"
        )
    packed_in = qw_shape[-1]
    out_rows = qw_shape[0]
    num_groups = int(qs_shape[-1])
    if num_groups < 1:
        raise RuntimeError(f"{weight_key}: invalid scales shape {qs_shape} (empty groups)")

    if dense_weight_shape is not None:
        to, ti = int(dense_weight_shape[0]), int(dense_weight_shape[1])
        if to != out_rows:
            raise RuntimeError(
                f"{weight_key}: quantized rows {out_rows} != dense weight rows {to}"
            )
        if ti % packed_in != 0:
            raise RuntimeError(
                f"{weight_key}: dense input dim {ti} not divisible by packed width {packed_in}"
            )
        ratio = ti // packed_in
        if ratio not in (4, 8):
            raise RuntimeError(
                f"{weight_key}: unsupported MLX uint32 packing ratio {ratio} "
                f"(expected 4 for 8-bit or 8 for 4-bit codes per uint32)"
            )
        bits = 32 // ratio
        in_dim = ti
        if bundle_affine_bits is not None and bundle_affine_bits != bits:
            raise RuntimeError(
                f"{weight_key}: bundle metadata quantization_level={bundle_affine_bits} "
                f"conflicts with shape-inferred {bits}-bit MLX affine packing"
            )
    else:
        candidates: list[tuple[int, int, int]] = []
        for bits_try in (8, 4):
            vals_per_u32 = 32 // bits_try
            in_dim_try = packed_in * vals_per_u32
            if in_dim_try % num_groups != 0:
                continue
            gs = in_dim_try // num_groups
            if gs >= 1:
                candidates.append((bits_try, gs, in_dim_try))
        if not candidates:
            raise RuntimeError(
                f"{weight_key}: cannot infer MLX affine quantization from weight shape {qw_shape} "
                f"and scales shape {qs_shape} (no matching 4- or 8-bit group tiling); "
                f"ensure keys align with the model's Linear weights."
            )
        if len(candidates) > 1:
            if bundle_affine_bits in (4, 8):
                chosen = [c for c in candidates if c[0] == bundle_affine_bits]
                if len(chosen) != 1:
                    raise RuntimeError(
                        f"{weight_key}: safetensors quantization_level={bundle_affine_bits} "
                        f"does not match any tiling consistent with shapes weight={qw_shape}, "
                        f"scales={qs_shape}"
                    )
                bits, group_size, in_dim = chosen[0]
            else:
                b1, g1, i1 = candidates[0]
                b2, g2, i2 = candidates[1]
                raise RuntimeError(
                    f"{weight_key}: ambiguous MLX affine packing: both (bits={b1}, in_dim={i1}, "
                    f"group_size={g1}) and (bits={b2}, in_dim={i2}, group_size={g2}) fit. "
                    f"Write ``quantization_level`` (4 or 8) in shard metadata or align keys "
                    f"so dense Linear shapes are available."
                )
        else:
            bits, group_size, in_dim = candidates[0]
            if bundle_affine_bits is not None and bundle_affine_bits != bits:
                raise RuntimeError(
                    f"{weight_key}: safetensors quantization_level={bundle_affine_bits} "
                    f"conflicts with shape-inferred {bits}-bit packing "
                    f"(weight={qw_shape}, scales={qs_shape})"
                )

    if in_dim % num_groups != 0:
        raise RuntimeError(
            f"{weight_key}: in_dim {in_dim} not divisible by num_groups {num_groups} from scales"
        )
    group_size = in_dim // num_groups
    if group_size < 1:
        raise RuntimeError(f"{weight_key}: inferred non-positive group_size")
    return bits, group_size


class TransformerBase:
    """Simple base class — all current models inherit from this.

    Hook methods (default no-op, subclasses override selectively):
    - sanitize()             → checkpoint key remapping (inside model, cf. mlx-lm)
    - after_load_weights()  → LoRA weight merging
    - prepare_conditioning() → ControlNet / visual encoder / custom preprocessing
    - before_denoise()       → ControlNet conditioning injection, latent modification
    - step_callback()        → per-step logging / time-varying condition injection
    - pack_latents() / unpack_latents()                  → latent packing
    """

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    # ── Hook: checkpoint key sanitization ──────────────────────────
    def sanitize(self, weights: dict) -> dict:
        """Transform checkpoint keys to match this model's ``_param_map`` keys.

        Called automatically during ``load_weights()`` **after** dequantization
        but **before** applying tensors to parameters. Override in subclasses to
        implement family-specific key remapping (cf. ``mlx-lm`` ``Model.sanitize``).

        Default: identity (no remap).
        """
        return weights

    # ── Hook: after weight loading ──────────────────────────────────
    def after_load_weights(self, bundle_root=None):
        """Called after weights are loaded. LoRA / Adapter merging happens here."""
        pass

    # ── Hook: pre-denoise preparation ──────────────────────────────────
    def prepare_conditioning(self, request, bundle_root=None) -> dict:
        """Handle model-specific conditioning inputs (ControlNet image encoding, etc.).
        Returns an extra cond dict that will be passed to before_denoise and forward.
        """
        return {}

    # ── Hook: before denoising loop ──────────────────────────────────
    def before_denoise(self, latents, timesteps, sigmas, **cond):
        """Called before the denoising loop starts. Can modify latents or inject ControlNet signals."""
        return latents, cond

    # ── Hook: per-step callback ──────────────────────────────────
    def step_callback(self, step_idx: int, latents, noise_pred):
        """Called after each denoising step. Default no-op."""
        pass

    def refine_cfg_noise(self, noise_cond, noise_pred, *, cfg_renorm_min: float):
        """After standard CFG combine (``eps_u + g * (eps_c - eps_u)``).

        Default identity. Override when the reference pipeline applies a further
        normalization in packed-token or model-specific space.
        Called only when registry ``enable_cfg_renorm`` is true and CFG is active.
        """
        return noise_pred

    def combine_cfg_noise(self, noise_cond, noise_uncond, guidance: float):
        """Merge conditional and unconditional model outputs for classifier-free guidance.

        Default matches diffusers-style ``eps_u + guidance * (eps_c - eps_u)`` where
        ``noise_cond`` / ``noise_uncond`` are forward outputs with positive / negative text.

        Note: This is **not** the same as changing the scalar ``guidance`` in the request; it is the
        **affine combination** of the two branches. Z-Image reference uses
        ``eps_c + guidance * (eps_c - eps_u)`` instead (see upstream ``ZImage._predict``); bit-level
        parity with reference images requires matching that convention **and** identical
        per-branch tensors.
        """
        return noise_uncond + guidance * (noise_cond - noise_uncond)

    def pack_latents(self, ctx: Any, latents: Any) -> Any:
        """Pack latents into denoise-space layout. Default: identity."""
        return latents

    def unpack_latents(self, ctx: Any, latents: Any, h: int, w: int) -> Any:
        """Unpack latents from denoise-space back to VAE-space. Default: identity."""
        return latents

    def quantize_runtime(self, bits: int = 4, *, ctx: Any = None) -> None:
        """Convert `nn.Linear` modules to `QuantizedLinear` in-place (MLX runtime quantization).

        Must be called **after** all weight loading/merging (``load_weights`` +
        ``after_load_weights`` + LoRA/Adapter merging) and **before** the first forward pass.

        CUDA backend is a no-op (runtime quantization not yet implemented).

        Works with both `nn.Module`-based models (SeedVR2-style) and plain Python class
        models (z-image-style) via manual attribute tree traversal.
        """
        if ctx is not None and getattr(ctx, "backend", "") != "mlx":
            return

        from importlib import import_module

        nn = import_module("mlx.nn")

        def _default_predicate(_path: str, module) -> bool:
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                return False
            if not hasattr(module, "to_quantized"):
                return False
            if not hasattr(module, "weight"):
                return False
            w = module.weight
            return w.ndim >= 2 and w.shape[-1] % 64 == 0

        if isinstance(self, nn.Module):
            nn.quantize(self, class_predicate=_default_predicate, bits=bits)
        else:
            _quantize_linear_tree(self, bits)

        if hasattr(self, '_param_map'):
            self._param_map.clear()

    def parameters(self):
        """Default parameter list (via _param_map)."""
        if not hasattr(self, '_param_map'):
            self._build_param_map()
        return list(self._param_map.items())

    def _rebind_param_by_key(self, key: str, new_param: Any) -> None:
        """Rebind module attribute referenced by flat ``_param_map`` key."""
        parts = key.split(".")
        obj: Any = self
        for part in parts[:-1]:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        last = parts[-1]
        if hasattr(obj, last):
            setattr(obj, last, new_param)
        elif hasattr(obj, "_parameters") and last in obj._parameters:
            obj._parameters[last] = new_param

    def _cast_param_map_dtype(self, target_dtype: Any) -> int:
        """Cast all ``_param_map`` tensors to ``target_dtype`` and rebind updated params."""
        if not hasattr(self, "_param_map"):
            self._build_param_map()
        changed = 0
        for key, param in list(self._param_map.items()):
            if param.dtype == target_dtype:
                continue
            new_param = param.astype(target_dtype)
            self._param_map[key] = new_param
            self._rebind_param_by_key(key, new_param)
            changed += 1
        return changed

    def load_weights(
        self,
        weights: list[tuple[str, Any]],
        strict: bool = False,
        ctx: Any = None,
        *,
        bundle_affine_bits: int | None = None,
        inference_mode: WeightInferenceMode | None = None,
        module_root: Any | None = None,
        skeleton_key_prefix: str = "",
    ):
        """Default weight loading (via _param_map).

        Automatically handles MLX **affine** QuantizedLinear checkpoints (``uint32`` packed codes
        + ``scales`` + ``biases``), including **4-bit** and **8-bit**, using the target parameter
        shape to disambiguate packing when needed. Optional ``bundle_affine_bits`` (from shard
        metadata ``quantization_level``) must agree with shape inference and resolves ambiguity
        when the dense map alone cannot.

        When ``inference_mode.kind == "quantized"``, loads packed weights into
        ``QuantizedLinear`` without dequantizing (low VRAM). Otherwise affine checkpoints are
        dequantized to dense ``nn.Linear`` (legacy / debug ``inference: dense``).

        Quantized bundles must be **self-contained**; there is **no** alternate fp16 directory
        merge (avoids silent cross-precision parameter mixing).

        After load, **every** key in ``_param_map`` must have been assigned; otherwise
        :class:`RuntimeError` is raised (missing tensor or shape mismatch). If ``strict`` is
        ``True``, checkpoint keys that do not map to any parameter also raise.
        """
        if (
            inference_mode is not None
            and inference_mode.kind == "quantized"
            and inference_mode.bits in (4, 8)
        ):
            from backend.engine.common.model.quantized_load_mlx import load_weights_quantized_inference

            return load_weights_quantized_inference(
                self,
                weights,
                strict=strict,
                ctx=ctx,
                bundle_affine_bits=bundle_affine_bits,
                bits=int(inference_mode.bits),
                group_size=int(inference_mode.group_size),
                module_root=module_root,
                skeleton_key_prefix=skeleton_key_prefix,
            )

        if not hasattr(self, '_param_map'):
            self._build_param_map()

        # ── Preprocess: detect and dequantize mlx QuantizedLinear weights ──
        weight_dict = dict(weights)
        scales_map: dict[str, dict] = {}

        for key in list(weight_dict.keys()):
            if key.endswith(".scales"):
                base = key[:-7]
                weight_key = base + ".weight"
                if weight_key in weight_dict:
                    scales_map[base] = {
                        "scales": weight_dict[key],
                        "biases": weight_dict.get(base + ".biases"),
                        "weight": weight_dict[weight_key],
                    }

        dequantized: dict[str, Any] = {}
        for key, tensor in weight_dict.items():
            if key.endswith(".scales") or key.endswith(".biases"):
                continue

            base = key[:-7] if key.endswith(".weight") else None
            if base and base in scales_map:
                group = scales_map[base]
                qw = group["weight"]       # uint32
                qs = group["scales"]       # float/bfloat16
                qb = group.get("biases")   # float/bfloat16

                dense_shape: tuple[int, int] | None = None
                p = self._param_map.get(key)
                if p is not None and hasattr(p, "shape") and len(p.shape) == 2:
                    dense_shape = (int(p.shape[0]), int(p.shape[1]))
                bits, group_size = _mlx_affine_infer_bits_and_group_size(
                    qw,
                    qs,
                    dense_weight_shape=dense_shape,
                    weight_key=key,
                    bundle_affine_bits=bundle_affine_bits,
                )

                deq = ctx.dequantize(
                    qw,
                    scales=qs,
                    biases=qb,
                    group_size=group_size,
                    bits=bits,
                )
                dequantized[key] = deq
            else:
                dequantized[key] = tensor

        # Fail loud: affine-quant bundles need a standalone dense ``*.bias`` when the model has
        # ``nn.Linear(..., bias=True)``. ``*.biases`` from ``mx.quantize`` is not interchangeable.
        for param_key in self._param_map:
            if not param_key.endswith(".bias"):
                continue
            base = param_key[:-5]
            if base not in scales_map:
                continue
            if param_key in dequantized:
                continue
            if param_key in weight_dict:
                continue
            raise RuntimeError(
                f"Quantized checkpoint is missing dense bias tensor {param_key!r} (base {base!r} has "
                f"{{weight, scales, biases}} only). Bundles produced by older int4/int8 conversion "
                f"dropped Linear biases; remove this quantized folder and re-convert from the "
                f"fp16/bf16 source, or reinstall a bundle saved with a fixed converter."
            )

        # ── Load weights ──
        dequantized = self.sanitize(dequantized)

        loaded = []
        skipped = []
        for key, tensor in dequantized.items():
            if key in self._param_map:
                param = self._param_map[key]
                if param.shape == tensor.shape:
                    _assign_param_tensor(param, tensor)
                    loaded.append(key)
                else:
                    skipped.append(f"{key} shape_mismatch: {param.shape} vs {tensor.shape}")
            else:
                skipped.append(key)

        loaded_set = set(loaded)
        missing = [k for k in self._param_map if k not in loaded_set]
        if missing:
            preview = missing[:40]
            more = f" (+{len(missing) - 40} more)" if len(missing) > 40 else ""
            mismatches = [s for s in skipped if isinstance(s, str) and "shape_mismatch" in s]
            mm_note = (
                f" Shape mismatches (checkpoint vs model, first): {mismatches[:20]!r}"
                + (f" (+{len(mismatches) - 20} more)" if len(mismatches) > 20 else "")
                if mismatches
                else ""
            )
            raise RuntimeError(
                f"Weight load failed: {len(missing)} model parameter(s) missing or shape mismatch "
                f"(no tensor applied). First keys: {preview!r}{more}.{mm_note}"
            )

        if strict:
            extras = [s for s in skipped if isinstance(s, str) and "shape_mismatch" not in s]
            if extras:
                prev = extras[:40]
                more = f" (+{len(extras) - 40} more)" if len(extras) > 40 else ""
                raise RuntimeError(
                    f"Weight load strict mode: {len(extras)} checkpoint key(s) not used by the "
                    f"model. First keys: {prev!r}{more}"
                )

        return loaded, skipped

    def _build_param_map(self):
        """Build parameter map (default implementation, subclasses may override)."""
        if hasattr(self, '_param_map'):
            self._param_map.clear()
        else:
            self._param_map = {}
        _collect_params(self, "", self._param_map)


def _assign_param_tensor(param: Any, tensor: Any) -> None:
    """In-place assign checkpoint tensor into a model parameter (MLX array or torch Tensor)."""
    pmod = getattr(param.__class__, "__module__", "")
    if pmod.startswith("torch."):
        if not getattr(tensor.__class__, "__module__", "").startswith("torch."):
            tensor = param.new_tensor(tensor)
        if tensor.shape != param.shape:
            raise RuntimeError(f"shape mismatch: param {tuple(param.shape)} vs tensor {tuple(tensor.shape)}")
        if tensor.device != param.device or tensor.dtype != param.dtype:
            tensor = tensor.to(device=param.device, dtype=param.dtype)
        param.copy_(tensor)
        return

    param[:] = tensor


def collect_params_legacy_tree(obj: Any, prefix: str, result: dict) -> None:
    """Attribute-tree param walk for Flux1-style MLX models (reference weight parity)."""
    if hasattr(obj, "parameters") and callable(obj.parameters):
        try:
            for pname, ptensor in obj.parameters().items():
                if isinstance(ptensor, dict):
                    continue
                result[f"{prefix}.{pname}" if prefix else pname] = ptensor
            return
        except Exception:
            pass
    for attr_name in sorted(dir(obj)):
        if attr_name.startswith("_") or attr_name in ("ctx", "config", "freqs_cis", "_param_map"):
            continue
        try:
            attr = getattr(obj, attr_name)
        except Exception:
            continue
        if attr is None or isinstance(attr, (int, float, str, bool, type)):
            continue
        new_prefix = f"{prefix}.{attr_name}" if prefix else attr_name
        if hasattr(attr, "parameters") and callable(attr.parameters):
            collect_params_legacy_tree(attr, new_prefix, result)
        elif isinstance(attr, (list, tuple)):
            for i, item in enumerate(attr):
                collect_params_legacy_tree(item, f"{new_prefix}.{i}", result)
        elif hasattr(attr, "__dict__") and not isinstance(attr, type):
            collect_params_legacy_tree(attr, new_prefix, result)


def _is_param_leaf(obj: Any) -> bool:
    """True when ``obj`` is a concrete MLX array or torch parameter tensor."""
    if obj is None:
        return False
    cls = obj.__class__
    module_name = getattr(cls, "__module__", "")
    qual_name = getattr(cls, "__qualname__", "")
    full_type_name = f"{module_name}.{qual_name}".lower()
    if full_type_name.startswith("torch.") and "tensor" in full_type_name:
        return True
    if full_type_name.startswith("mlx.") and "array" in full_type_name:
        return True
    return False


def _collect_params(obj, prefix: str, result: dict):
    """Recursively collect MLX dict trees and torch/MLX module parameter leaves."""
    if obj is None or isinstance(obj, (int, float, str, bool, type)):
        return

    if _is_param_leaf(obj):
        if prefix:
            result[prefix] = obj
        return

    cls = obj.__class__
    module_name = getattr(cls, "__module__", "")
    qual_name = getattr(cls, "__qualname__", "")
    full_type_name = f"{module_name}.{qual_name}".lower()

    # Collect PyTorch nn.Module leaves without importing torch.
    if full_type_name.startswith("torch.nn.") and hasattr(obj, "named_parameters"):
        try:
            for name, param in obj.named_parameters():
                key = f"{prefix}.{name}" if prefix else name
                result[key] = param
            return
        except Exception:
            pass

    if hasattr(obj, "parameters") and callable(obj.parameters):
        try:
            params = obj.parameters()
            if isinstance(params, dict) and params:
                for pname, ptensor in params.items():
                    key = f"{prefix}.{pname}" if prefix else pname
                    if isinstance(ptensor, dict):
                        for subname, subtensor in ptensor.items():
                            subkey = f"{key}.{subname}"
                            if _is_param_leaf(subtensor):
                                result[subkey] = subtensor
                            else:
                                _collect_params(subtensor, subkey, result)
                    elif _is_param_leaf(ptensor):
                        result[key] = ptensor
                    else:
                        _collect_params(ptensor, key, result)
                return
        except Exception:
            pass

    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            item_prefix = f"{prefix}.{i}" if prefix else str(i)
            _collect_params(item, item_prefix, result)
        return

    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        for attr_name, attr in sorted(vars(obj).items()):
            if attr_name.startswith("_"):
                continue
            if attr_name in ("ctx", "config", "freqs_cis", "_param_map"):
                continue
            if attr is None or isinstance(attr, (int, float, str, bool, type)):
                continue
            new_prefix = f"{prefix}.{attr_name}" if prefix else attr_name
            _collect_params(attr, new_prefix, result)


def _quantize_linear_tree(root, bits: int) -> None:
    """Walk plain-Python object tree, find `nn.Linear` instances, call ``to_quantized``.

    Used by ``TransformerBase.quantize_runtime()`` for models that do not inherit
    ``nn.Module`` (e.g. z-image family).

    Skips ``nn.Conv2d`` / ``nn.Conv3d`` and ``nn.Linear`` layers where
    ``weight.shape[-1] % 64 != 0`` (MLX group_size=64 constraint).
    """
    from importlib import import_module

    nn = import_module("mlx.nn")

    seen: set[int] = set()

    def _walk(obj):
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(obj, nn.Linear):
            if (hasattr(obj, 'to_quantized') and hasattr(obj, 'weight')
                    and obj.weight.ndim >= 2 and obj.weight.shape[-1] % 64 == 0):
                obj.to_quantized(bits=bits, group_size=64)
            return
        if isinstance(obj, (nn.Conv2d, nn.Conv3d)):
            return
        if isinstance(obj, nn.Module):
            for _child in obj.children():
                _walk(_child)
            return
        if hasattr(obj, '__dict__') and not isinstance(obj, (type, int, float, str, bool)):
            for _k in sorted(vars(obj)):
                if _k.startswith('_'):
                    continue
                try:
                    _v = getattr(obj, _k)
                except Exception:
                    continue
                if _v is None or isinstance(_v, (int, float, str, bool, type)):
                    continue
                _walk(_v)
        elif isinstance(obj, (list, tuple)):
            for _item in obj:
                _walk(_item)

    _walk(root)
