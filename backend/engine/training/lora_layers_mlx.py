"""MLX LoRA linear layers for DreamBooth training."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn


def _frozen_assistant_lora_factors(
    down: mx.array,
    up: mx.array,
    *,
    out_d: int,
    in_d: int,
    alpha: float,
    strength: float = 1.0,
    lora_id: str = "zimage-turbo-training-adapter",
    wkey: str | None = None,
) -> tuple[mx.array, mx.array, float]:
    """Map ``orient_lora_pair`` layout to LoRALinear matmul factors ``(in, r)``, ``(r, out)``."""
    from backend.engine.common.bundle.lora_mlx import orient_lora_pair

    key = wkey or f"{lora_id}.weight"
    d_orient, u_orient, rank = orient_lora_pair(
        down,
        up,
        out_d=out_d,
        in_d=in_d,
        lora_id=lora_id,
        wkey=key,
    )
    if rank <= 0:
        raise RuntimeError(f"Frozen assistant LoRA: invalid rank for {key}.")
    # orient_lora_pair: down [rank, in_d], up [out_d, rank]
    lora_a = mx.transpose(d_orient, (1, 0))
    lora_b = mx.transpose(u_orient, (1, 0))
    scale = (float(alpha) / float(rank)) * float(strength)
    return lora_a, lora_b, scale


_ASSISTANT_PARAM_KEYS = ("assistant_lora_a", "assistant_lora_b", "assistant_dense")


def _freeze_assistant_params(module: nn.Module) -> None:
    """Keep the turbo training assistant out of ``trainable_parameters()``.

    The assistant tensors are plain ``mx.array`` attributes, so MLX registers them as
    parameters. Without this, AdamW trains the (much larger) assistant instead of the
    exported LoRA and the learned subject is discarded at inference (assistant off).
    """
    module.freeze(keys=list(_ASSISTANT_PARAM_KEYS), recurse=False, strict=False)


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen ``nn.Linear``."""

    @staticmethod
    def from_base(
        linear: nn.Linear,
        r: int = 8,
        scale: float = 1.0,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        output_dims, input_dims = linear.weight.shape
        lora_lin = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            scale=scale,
            dropout=dropout,
        )
        lora_lin.linear = linear
        linear.freeze()
        return lora_lin

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        scale: float = 1.0,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.scale = scale
        self.dropout = float(dropout)
        init_scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-init_scale,
            high=init_scale,
            shape=(input_dims, r),
        )
        self.lora_b = mx.zeros(shape=(r, output_dims))
        self.assistant_lora_a: mx.array | None = None
        self.assistant_lora_b: mx.array | None = None
        self.assistant_dense: mx.array | None = None
        self.assistant_scale = 1.0
        self.assistant_enabled = False

    def attach_frozen_assistant(
        self,
        down: mx.array,
        up: mx.array,
        alpha: float,
        *,
        strength: float = 1.0,
        lora_id: str = "zimage-turbo-training-adapter",
    ) -> None:
        """Attach a frozen low-rank assistant path without mutating ``linear.weight``."""
        out_d, in_d = int(self.linear.weight.shape[0]), int(self.linear.weight.shape[1])
        lora_a, lora_b, scale = _frozen_assistant_lora_factors(
            down,
            up,
            out_d=out_d,
            in_d=in_d,
            alpha=alpha,
            strength=strength,
            lora_id=lora_id,
        )
        self.assistant_lora_a = lora_a
        self.assistant_lora_b = lora_b
        self.assistant_dense = None
        self.assistant_scale = scale
        self.assistant_enabled = True
        _freeze_assistant_params(self)

    def attach_frozen_assistant_dense(self, delta: mx.array, *, strength: float = 1.0) -> None:
        """Attach a frozen full-rank delta (``delta @ x``) without mutating ``linear.weight``."""
        out_d, in_d = int(self.linear.weight.shape[0]), int(self.linear.weight.shape[1])
        delta_f = delta.astype(mx.float32)
        if tuple(delta_f.shape) != (out_d, in_d):
            raise RuntimeError(
                f"Frozen assistant dense delta shape {tuple(delta_f.shape)} "
                f"does not match linear weight {(out_d, in_d)}"
            )
        self.assistant_dense = float(strength) * delta_f
        self.assistant_lora_a = None
        self.assistant_lora_b = None
        self.assistant_scale = 1.0
        self.assistant_enabled = True
        _freeze_assistant_params(self)

    def _assistant_contribution(self, x: mx.array) -> mx.array:
        if self.assistant_dense is not None:
            dense = mx.stop_gradient(self.assistant_dense)
            return x @ dense.T.astype(x.dtype)
        if self.assistant_lora_a is None or self.assistant_lora_b is None:
            raise RuntimeError("Frozen assistant LoRA tensors are missing")
        a = mx.stop_gradient(self.assistant_lora_a)
        b = mx.stop_gradient(self.assistant_lora_b)
        z = (x @ a) @ b
        return (self.assistant_scale * z).astype(x.dtype)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.linear(x)
        if self.assistant_enabled:
            y = y + self._assistant_contribution(x)
        z = (x @ self.lora_a) @ self.lora_b
        if self.dropout > 0:
            z = mx.dropout(z, self.dropout)
        return y + (self.scale * z).astype(x.dtype)


class FrozenAssistantLinear(nn.Module):
    """Frozen ``nn.Linear`` with optional turbo training assistant (no trainable LoRA)."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear
        linear.freeze()
        self.assistant_lora_a: mx.array | None = None
        self.assistant_lora_b: mx.array | None = None
        self.assistant_dense: mx.array | None = None
        self.assistant_scale = 1.0
        self.assistant_enabled = False

    @staticmethod
    def wrap_linear(
        linear: nn.Linear,
        down: mx.array,
        up: mx.array,
        alpha: float,
        *,
        strength: float = 1.0,
        lora_id: str = "zimage-turbo-training-adapter",
    ) -> "FrozenAssistantLinear":
        wrapped = FrozenAssistantLinear(linear)
        out_d, in_d = int(linear.weight.shape[0]), int(linear.weight.shape[1])
        lora_a, lora_b, scale = _frozen_assistant_lora_factors(
            down,
            up,
            out_d=out_d,
            in_d=in_d,
            alpha=alpha,
            strength=strength,
            lora_id=lora_id,
        )
        wrapped.assistant_lora_a = lora_a
        wrapped.assistant_lora_b = lora_b
        wrapped.assistant_dense = None
        wrapped.assistant_scale = scale
        wrapped.assistant_enabled = True
        _freeze_assistant_params(wrapped)
        return wrapped

    @staticmethod
    def wrap_linear_dense(linear: nn.Linear, delta: mx.array, *, strength: float = 1.0) -> "FrozenAssistantLinear":
        wrapped = FrozenAssistantLinear(linear)
        out_d, in_d = int(linear.weight.shape[0]), int(linear.weight.shape[1])
        delta_f = delta.astype(mx.float32)
        if tuple(delta_f.shape) != (out_d, in_d):
            raise RuntimeError(
                f"Frozen assistant dense delta shape {tuple(delta_f.shape)} "
                f"does not match linear weight {(out_d, in_d)}"
            )
        wrapped.assistant_dense = float(strength) * delta_f
        wrapped.assistant_lora_a = None
        wrapped.assistant_lora_b = None
        wrapped.assistant_scale = 1.0
        wrapped.assistant_enabled = True
        _freeze_assistant_params(wrapped)
        return wrapped

    def _assistant_contribution(self, x: mx.array) -> mx.array:
        if self.assistant_dense is not None:
            dense = mx.stop_gradient(self.assistant_dense)
            return x @ dense.T.astype(x.dtype)
        if self.assistant_lora_a is None or self.assistant_lora_b is None:
            raise RuntimeError("Frozen assistant LoRA tensors are missing")
        a = mx.stop_gradient(self.assistant_lora_a)
        b = mx.stop_gradient(self.assistant_lora_b)
        z = (x @ a) @ b
        return (self.assistant_scale * z).astype(x.dtype)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.linear(x)
        if self.assistant_enabled:
            y = y + self._assistant_contribution(x)
        return y


class DoRALinear(nn.Module):
    """Weight-Decomposed LoRA (magnitude + low-rank direction)."""

    @staticmethod
    def from_base(
        linear: nn.Linear,
        r: int = 8,
        scale: float = 1.0,
        dropout: float = 0.0,
    ) -> "DoRALinear":
        output_dims, input_dims = linear.weight.shape
        dora_lin = DoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            scale=scale,
            dropout=dropout,
        )
        dora_lin.linear = linear
        linear.freeze()
        dora_lin.magnitude = mx.linalg.norm(linear.weight, axis=1)
        return dora_lin

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        scale: float = 1.0,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.scale = scale
        self.dropout = float(dropout)
        self.magnitude = mx.ones((output_dims,))
        init_scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-init_scale,
            high=init_scale,
            shape=(input_dims, r),
        )
        self.lora_b = mx.zeros(shape=(r, output_dims))
        self.assistant_lora_a: mx.array | None = None
        self.assistant_lora_b: mx.array | None = None
        self.assistant_dense: mx.array | None = None
        self.assistant_scale = 1.0
        self.assistant_enabled = False

    def attach_frozen_assistant(
        self,
        down: mx.array,
        up: mx.array,
        alpha: float,
        *,
        strength: float = 1.0,
        lora_id: str = "zimage-turbo-training-adapter",
    ) -> None:
        out_d, in_d = int(self.linear.weight.shape[0]), int(self.linear.weight.shape[1])
        self.assistant_lora_a, self.assistant_lora_b, self.assistant_scale = _frozen_assistant_lora_factors(
            down, up, out_d=out_d, in_d=in_d, alpha=alpha, strength=strength, lora_id=lora_id,
        )
        self.assistant_dense = None
        self.assistant_enabled = True
        _freeze_assistant_params(self)

    def attach_frozen_assistant_dense(self, delta: mx.array, *, strength: float = 1.0) -> None:
        out_d, in_d = int(self.linear.weight.shape[0]), int(self.linear.weight.shape[1])
        if tuple(delta.shape) != (out_d, in_d):
            raise RuntimeError(f"Assistant delta {tuple(delta.shape)} does not match weight {(out_d, in_d)}")
        self.assistant_dense = float(strength) * delta.astype(mx.float32)
        self.assistant_lora_a = None
        self.assistant_lora_b = None
        self.assistant_scale = 1.0
        self.assistant_enabled = True
        _freeze_assistant_params(self)

    def _assistant_delta(self) -> mx.array:
        if not self.assistant_enabled:
            return mx.zeros_like(self.linear.weight, dtype=mx.float32)
        if self.assistant_dense is not None:
            return mx.stop_gradient(self.assistant_dense)
        if self.assistant_lora_a is None or self.assistant_lora_b is None:
            raise RuntimeError("Frozen assistant LoRA tensors are missing")
        return mx.stop_gradient(
            (self.assistant_lora_a @ self.assistant_lora_b).T * self.assistant_scale
        )

    def effective_weight(self) -> mx.array:
        w = self.linear.weight.astype(mx.float32) + self._assistant_delta()
        delta = (self.lora_a @ self.lora_b).T * self.scale
        v = w + delta
        norms = mx.sqrt(mx.sum(v * v, axis=1, keepdims=True)) + 1e-8
        return self.magnitude[:, None] * (v / norms)

    def __call__(self, x: mx.array) -> mx.array:
        w_eff = self.effective_weight()
        y = x @ w_eff.T
        if self.linear.bias is not None:
            y = y + self.linear.bias
        return y


def _is_frozen_assistant_linear(module: Any) -> bool:
    return isinstance(module, FrozenAssistantLinear)


def _is_adapter_linear(module: Any) -> bool:
    return isinstance(module, (LoRALinear, DoRALinear))


def _is_leaf_linear(module: Any) -> bool:
    return _is_adapter_linear(module) or _is_frozen_assistant_linear(module) or isinstance(module, nn.Linear)


def _normalize_lora_module_key(key: str) -> str:
    """Accept diffusers-style ``to_out.0`` for MLX attribute ``to_out`` (sanitize drops the ``.0``)."""
    k = (key or "").strip().strip(".")
    if k.endswith(".0"):
        k = k[:-2]
    return k


def _match_lora_module_key(attr_name: str, module_keys: list[str] | None) -> bool:
    if not module_keys:
        return True
    for raw in module_keys:
        k = _normalize_lora_module_key(raw)
        if not k:
            continue
        if attr_name == k or attr_name.endswith(f".{k}"):
            return True
    return False


def assert_no_trainable_assistant(train_module: Any) -> int:
    """Fail loud if turbo assistant tensors leaked into the trainable set; return LoRA param count."""
    from mlx.utils import tree_flatten

    flat = dict(tree_flatten(train_module.trainable_parameters()))
    leaked = sorted(k for k in flat if any(k.endswith(s) for s in _ASSISTANT_PARAM_KEYS))
    if leaked:
        raise RuntimeError(
            "Turbo training assistant tensors are trainable "
            f"({len(leaked)} entries, e.g. {leaked[0]!r}); the assistant must stay frozen or the "
            "learned subject is written into it and discarded at inference."
        )
    return int(sum(int(v.size) for v in flat.values()))


def _is_trainable_base_linear(module: Any) -> bool:
    return (
        isinstance(module, nn.Linear)
        and not _is_adapter_linear(module)
        and not _is_frozen_assistant_linear(module)
    )


def _iter_named_children(module: Any):
    """Direct children, including Linear lists MLX did not register (Qwen ``attn_to_out``)."""
    if isinstance(module, nn.Module) and not _is_leaf_linear(module):
        children = dict(module.children())
        yield from children.items()
        seen = set(children)
        for name, val in vars(module).items():
            if name.startswith("_") or name in seen:
                continue
            if isinstance(val, (list, nn.Module)) or _is_trainable_base_linear(val):
                yield name, val
        return
    if not hasattr(module, "__dict__"):
        return
    for name, val in vars(module).items():
        if name.startswith("_"):
            continue
        yield name, val


def _out_proj_list_selected(parent_name: str | None, module_keys: list[str] | None) -> bool:
    """Whether anonymous linears in a list (Qwen image ``to_out``) should take LoRA."""
    if not module_keys:
        return True
    if not parent_name:
        return False
    if _match_lora_module_key(parent_name, module_keys):
        return True
    # QwenAttention stores the image out-projection as ``attn_to_out = [Linear]``.
    if parent_name == "attn_to_out" and _match_lora_module_key("to_out", module_keys):
        return True
    return False


def _apply_lora_recursive(
    module: Any,
    rank: int,
    *,
    scale: float = 1.0,
    dropout: float = 0.0,
    module_keys: list[str] | None = None,
    adapter_cls: type = LoRALinear,
    parent_name: str | None = None,
) -> list[Any] | None:
    # Adapters and raw linears are leaves. Descending into a LoRALinear would wrap its
    # inner projection a second time.
    if _is_leaf_linear(module):
        return
    if isinstance(module, list):
        wrap = _out_proj_list_selected(parent_name, module_keys)
        for i, item in enumerate(module):
            if _is_trainable_base_linear(item):
                if wrap:
                    module[i] = adapter_cls.from_base(
                        item,
                        r=rank,
                        scale=scale,
                        dropout=dropout,
                    )
            else:
                _apply_lora_recursive(
                    item,
                    rank,
                    scale=scale,
                    dropout=dropout,
                    module_keys=module_keys,
                    adapter_cls=adapter_cls,
                )
        return module
    for name, child in _iter_named_children(module):
        if _is_trainable_base_linear(child) and _match_lora_module_key(name, module_keys):
            setattr(
                module,
                name,
                adapter_cls.from_base(child, r=rank, scale=scale, dropout=dropout),
            )
        else:
            replacement = _apply_lora_recursive(
                child,
                rank,
                scale=scale,
                dropout=dropout,
                module_keys=module_keys,
                adapter_cls=adapter_cls,
                parent_name=name,
            )
            if isinstance(child, list) and replacement is not None:
                # MLX nn.Module.__getattr__ materializes a fresh list container on
                # every access. Mutating ``child[i]`` therefore updates a temporary
                # list and silently leaves Qwen's real attn_to_out[0] untrained.
                # Reassign the whole list so MLX registers the adapter in the DiT.
                setattr(module, name, replacement)


def _dit_inner(model: Any) -> Any:
    return getattr(model, "_inner", model)


def _walk_dit_nodes(root: Any) -> Any:
    if _is_leaf_linear(root):
        yield root
        return
    if isinstance(root, list):
        for item in root:
            yield from _walk_dit_nodes(item)
        return
    for _name, child in _iter_named_children(root):
        yield from _walk_dit_nodes(child)


def _resolve_parent_and_name(root: Any, path: str) -> tuple[Any, str]:
    parts = path.split(".")
    if not parts:
        raise RuntimeError("Empty module path")
    cur = root
    for part in parts[:-1]:
        if part.isdigit():
            cur = cur[int(part)]
        elif isinstance(cur, nn.Module):
            cur = cur.children()[part]
        else:
            cur = getattr(cur, part)
    return cur, parts[-1]


def _replace_module_at_path(root: Any, path: str, module: Any) -> None:
    parent, name = _resolve_parent_and_name(root, path)
    if name.isdigit():
        parent[int(name)] = module
    else:
        setattr(parent, name, module)


def _walk_all_linear_paths(root: Any, prefix: str = "") -> list[tuple[Any, str]]:
    """Collect ``(linear_layer, dit_param_prefix)`` for LoRA, assistant, and base Linear."""
    found: list[tuple[Any, str]] = []

    def visit(node: Any, path: str) -> None:
        if _is_leaf_linear(node):
            found.append((node, path.rstrip(".")))
            return
        if isinstance(node, list):
            for i, item in enumerate(node):
                child = f"{path}.{i}" if path else str(i)
                visit(item, child)
            return
        if isinstance(node, nn.Module) and not _is_leaf_linear(node):
            for name, child in _iter_named_children(node):
                child_path = f"{path}.{name}" if path else name
                visit(child, child_path)
            return
        if not hasattr(node, "__dict__"):
            return
        for name, val in vars(node).items():
            if name.startswith("_"):
                continue
            child_path = f"{path}.{name}" if path else name
            visit(val, child_path)

    visit(root, prefix)
    return found


def set_training_assistant_enabled(layers: list[Any], *, enabled: bool) -> None:
    """Toggle frozen turbo training assistants on attached layers."""
    for layer in layers:
        if isinstance(layer, (LoRALinear, DoRALinear, FrozenAssistantLinear)):
            layer.assistant_enabled = enabled


def _walk_lora_paths(root: Any, prefix: str = "") -> list[tuple[Any, str]]:
    """Collect ``(adapter_layer, dit_param_prefix)`` in the same order as ``iter_lora_linears``."""
    found: list[tuple[Any, str]] = []

    def visit(node: Any, path: str) -> None:
        if _is_adapter_linear(node):
            found.append((node, path.rstrip(".")))
            return
        if isinstance(node, nn.Linear) or _is_frozen_assistant_linear(node):
            return
        if isinstance(node, list):
            for i, item in enumerate(node):
                child = f"{path}.{i}" if path else str(i)
                visit(item, child)
            return
        if isinstance(node, nn.Module) and not _is_adapter_linear(node) and not _is_frozen_assistant_linear(node):
            for name, child in _iter_named_children(node):
                child_path = f"{path}.{name}" if path else name
                visit(child, child_path)
            return
        if not hasattr(node, "__dict__"):
            return
        for name, val in vars(node).items():
            if name.startswith("_"):
                continue
            child_path = f"{path}.{name}" if path else name
            visit(val, child_path)

    visit(root, prefix)
    return found


def iter_lora_linears(model: Any) -> list[Any]:
    layers = [node for node in _walk_dit_nodes(_dit_inner(model)) if _is_adapter_linear(node)]
    if not layers:
        raise RuntimeError("No LoRA layers found on DiT after apply_lora")
    return layers


def _canonical_lora_export_path(path: str) -> str:
    """Qwen stores to_out[0] as attn_to_out; use the diffusers path in saved LoRAs."""
    if path.endswith(".attn.attn_to_out.0"):
        # Diffusers and remap_qwen_lora_module_prefix both use ``to_out.0``.
        # Dropping the list index would make every output projection unmatchable.
        return f"{path[:-len('.attn.attn_to_out.0')]}.attn.to_out.0"
    return path


def iter_lora_linears_with_paths(model: Any) -> list[tuple[Any, str]]:
    entries = [
        (layer, _canonical_lora_export_path(path))
        for layer, path in _walk_lora_paths(_dit_inner(model))
    ]
    if not entries:
        raise RuntimeError("No LoRA layers found on DiT after apply_lora")
    return entries


def freeze_dit_base_weights(model: Any) -> None:
    """Freeze all base ``nn.Linear`` weights; leave LoRA adapter tensors trainable."""
    for node in _walk_dit_nodes(_dit_inner(model)):
        if _is_adapter_linear(node):
            continue
        if isinstance(node, nn.Module) and hasattr(node, "freeze"):
            node.freeze()


def _is_quantized_linear(module: Any) -> bool:
    cls_name = type(module).__name__
    return cls_name == "QuantizedLinear" or "Quantized" in cls_name


def quantize_frozen_dit_linears(model: Any, *, bits: int, group_size: int = 64) -> int:
    """QLoRA: quantize frozen base linears (including inside ``LoRALinear``)."""
    if bits not in (4, 8):
        raise RuntimeError(f"quantize_frozen_dit_linears bits must be 4 or 8 (got {bits})")
    replacements: list[tuple[Any, str, nn.Linear]] = []

    def visit(obj: Any, parent: Any | None = None, key: str | None = None) -> None:
        if _is_adapter_linear(obj):
            if isinstance(obj.linear, nn.Linear) and not _is_quantized_linear(obj.linear):
                replacements.append((obj, "linear", obj.linear))
            return
        if isinstance(obj, nn.Linear) and not _is_adapter_linear(obj):
            if parent is not None and key is not None and not _is_quantized_linear(obj):
                replacements.append((parent, key, obj))
            return
        if isinstance(obj, list):
            for item in obj:
                visit(item)
            return
        if isinstance(obj, nn.Module) and not _is_adapter_linear(obj):
            for name, child in _iter_named_children(obj):
                visit(child, obj, name)
            return
        if not hasattr(obj, "__dict__"):
            return
        for name, val in vars(obj).items():
            if name.startswith("_"):
                continue
            visit(val, obj, name)

    visit(_dit_inner(model))
    count = 0
    for parent, key, linear in replacements:
        q = linear.to_quantized(bits=bits, group_size=group_size)
        q.freeze()
        setattr(parent, key, q)
        count += 1
    return count


def _checkpoint_trainable_modules(obj: Any) -> list[nn.Module]:
    """Nested ``nn.Module``s whose parameters must be checkpoint inputs.

    ``mx.checkpoint`` only differentiates arrays passed as arguments. Z-Image blocks are
    plain objects, so LoRA weights read off ``self`` would otherwise get zero gradients
    and ``lora_B`` would stay at its zero init for the whole run.
    """
    modules: list[nn.Module] = []
    seen: set[int] = set()

    def walk(node: Any) -> None:
        if node is None or isinstance(node, (str, bytes, bytearray, int, float, bool, complex)):
            return
        if isinstance(node, mx.array):
            return
        marker = id(node)
        if marker in seen:
            return
        if isinstance(node, nn.Module):
            seen.add(marker)
            modules.append(node)
            return
        seen.add(marker)
        if isinstance(node, dict):
            children = list(node.values())
        elif isinstance(node, (list, tuple)):
            children = list(node)
        elif hasattr(node, "__dict__"):
            children = list(vars(node).values())
        else:
            return
        for child in children:
            walk(child)

    walk(obj)
    return modules


def grad_checkpoint(layer: Any) -> None:
    """Checkpoint a block's forward pass without changing its public API.

    Flux/Qwen blocks are ``nn.Module`` (``__call__``). Z-Image refiners/layers are
    plain Python classes that expose ``forward``. Both recompute activations, and both
    pass trainable parameters into the checkpoint so LoRA up-projections receive gradients.
    """
    layer_cls = type(layer)
    if getattr(layer_cls, "_dq_grad_checkpoint", False):
        return
    if isinstance(layer, nn.Module):
        _install_module_grad_checkpoint(layer_cls)
        return
    if callable(getattr(layer_cls, "forward", None)):
        _install_plain_grad_checkpoint(layer_cls)
        return
    raise RuntimeError(
        f"Gradient checkpointing requires nn.Module or a .forward method, "
        f"got {layer_cls.__name__}; refusing to claim memory savings for an inactive implementation."
    )


def _install_module_grad_checkpoint(layer_cls: type) -> None:
    from mlx.nn.utils import checkpoint as nn_checkpoint

    orig = layer_cls.__call__

    def checkpointed_fn(model: Any, *args: Any, **kwargs: Any) -> Any:
        cached = getattr(model, "_dq_ckpt_call", None)
        if cached is None:
            def call(*inner_args: Any, **inner_kwargs: Any) -> Any:
                return orig(model, *inner_args, **inner_kwargs)

            cached = nn_checkpoint(model, call)
            model._dq_ckpt_call = cached
        return cached(*args, **kwargs)

    checkpointed_fn._dq_grad_checkpoint = True  # type: ignore[attr-defined]
    layer_cls.__call__ = checkpointed_fn
    layer_cls._dq_grad_checkpoint = True


def _install_plain_grad_checkpoint(layer_cls: type) -> None:
    orig = layer_cls.forward

    def inner(param_trees: list[Any], model: Any, *args: Any, **kwargs: Any) -> Any:
        modules = _checkpoint_trainable_modules(model)
        if len(modules) != len(param_trees):
            raise RuntimeError(
                f"Gradient checkpoint parameter tree changed for {type(model).__name__}: "
                f"{len(modules)} modules vs {len(param_trees)} trees"
            )
        for module, params in zip(modules, param_trees):
            if params:
                module.update(params)
        return orig(model, *args, **kwargs)

    checkpointed = mx.checkpoint(inner)

    def checkpointed_fn(model: Any, *args: Any, **kwargs: Any) -> Any:
        modules = _checkpoint_trainable_modules(model)
        trees = [module.trainable_parameters() for module in modules]
        return checkpointed(trees, model, *args, **kwargs)

    checkpointed_fn._dq_grad_checkpoint = True  # type: ignore[attr-defined]
    layer_cls.forward = checkpointed_fn
    layer_cls._dq_grad_checkpoint = True


def assert_lora_adapter_received_gradients(train_module: Any, *, step: int) -> None:
    """Fail when ``lora_B`` is still the zero init, so a no-op adapter cannot train silently."""
    if step < 1:
        return
    from mlx.utils import tree_flatten

    flat = dict(tree_flatten(train_module.trainable_parameters()))
    b_keys = [key for key in flat if key == "lora_b" or key.endswith(".lora_b")]
    if not b_keys:
        return
    max_abs = mx.zeros((), dtype=mx.float32)
    for key in b_keys:
        max_abs = mx.maximum(max_abs, mx.max(mx.abs(flat[key])))
    mx.eval(max_abs)
    if float(max_abs.item()) == 0.0:
        raise RuntimeError(
            f"LoRA up-projection (lora_B) is still all zeros at step {step}. "
            "Gradients are not reaching the adapter, so progress previews stay on the base "
            "model and will not resemble the training subject."
        )


def enable_grad_checkpointing_on_blocks(blocks: list[Any]) -> int:
    """Enable real activation recomputation for each selected block type."""
    for block in blocks:
        grad_checkpoint(block)
    return len(blocks)


def list_flux1_lora_blocks(model: Any, *, lora_blocks: int) -> list[Any]:
    dit = getattr(model, "_inner", model)
    joint = list(dit.transformer_blocks)
    single = list(dit.single_transformer_blocks)
    all_blocks = list(reversed(joint + single))
    if lora_blocks == -1:
        return all_blocks
    if lora_blocks < 1 or lora_blocks > len(all_blocks):
        raise RuntimeError(f"lora_blocks must be -1 or 1..{len(all_blocks)} (got {lora_blocks})")
    return all_blocks[:lora_blocks]


def list_zimage_lora_blocks(model: Any, *, lora_blocks: int) -> list[Any]:
    """Blocks that receive LoRA, identity-first.

    ``context_refiner`` is where caption tokens are prepared before they meet image tokens,
    and ``noise_refiner`` sets layout. Taking the deepest main layers only (the old order)
    learned texture and left the trigger unbound, so faces never stuck. A short budget still
    includes both refiners, then shallow main layers.
    """
    dit = getattr(model, "_inner", model)
    all_blocks: list[Any] = [
        *list(dit.context_refiner),
        *list(dit.noise_refiner),
        *list(dit.layers),
    ]
    if lora_blocks == -1:
        return all_blocks
    if lora_blocks < 1 or lora_blocks > len(all_blocks):
        raise RuntimeError(f"lora_blocks must be -1 or 1..{len(all_blocks)} (got {lora_blocks})")
    return all_blocks[:lora_blocks]


def list_qwen_lora_blocks(model: Any, *, lora_blocks: int) -> list[Any]:
    """Shallow joint blocks first. Early blocks route the trigger onto the face; the tail refines texture."""
    dit = getattr(model, "_inner", model)
    inner = getattr(dit, "dit", dit)
    all_blocks = list(inner.transformer_blocks)
    if lora_blocks == -1:
        return all_blocks
    if lora_blocks < 1 or lora_blocks > len(all_blocks):
        raise RuntimeError(f"lora_blocks must be -1 or 1..{len(all_blocks)} (got {lora_blocks})")
    return all_blocks[:lora_blocks]


class DiTLoRATrainModule(nn.Module):
    """Thin ``nn.Module`` wrapper so ``mlx.nn.value_and_grad`` can optimize LoRA only."""

    def __init__(self, dit: Any, lora_entries: list[tuple[Any, str]]):
        super().__init__()
        self._dit = dit
        self._lora_paths: list[str] = []
        for idx, (layer, path) in enumerate(lora_entries):
            setattr(self, f"lora_{idx}", layer)
            self._lora_paths.append(path)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._dit(*args, **kwargs)


def build_lora_train_module(model: Any) -> DiTLoRATrainModule:
    return DiTLoRATrainModule(model, iter_lora_linears_with_paths(model))


def _assert_full_attention_coverage(
    model: Any,
    *,
    module_keys: list[str] | None,
    lora_blocks: int,
) -> None:
    """Verify real module-tree coverage for the shipped full-block face presets."""
    if lora_blocks != -1 or not module_keys:
        return
    keys = {_normalize_lora_module_key(key) for key in module_keys}
    expected: tuple[str, list[Any]] | None = None
    if "add_q_proj" in keys:
        blocks = list_qwen_lora_blocks(model, lora_blocks=-1)
        required = {"to_q", "to_k", "to_v", "to_out", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"}
        expected = (f"Qwen {len(blocks)} blocks × 8 attention projections", blocks)
    elif {"to_q", "to_k", "to_v", "to_out"}.issubset(keys):
        blocks = list_zimage_lora_blocks(model, lora_blocks=-1)
        if not blocks or type(blocks[0]).__module__.split(".")[-2] != "z_image":
            return
        required = {"to_q", "to_k", "to_v", "to_out"}
        expected = (f"Z-Image {len(blocks)} blocks (both refiners + main layers)", blocks)
    if expected is None:
        return
    label, blocks = expected
    missing: list[str] = []
    for index, block in enumerate(blocks):
        paths = {
            (path.rsplit(".", 1)[0] if path.rsplit(".", 1)[-1].isdigit() else path)
            .replace(".attn_to_out", ".to_out")
            for _, path in _walk_lora_paths(block)
        }
        absent = sorted(
            key for key in required
            if not any(path == key or path.endswith(f".{key}") for path in paths)
        )
        if absent:
            missing.append(f"block {index}: {','.join(absent)}")
    if missing:
        raise RuntimeError(
            f"LoRA full-block coverage failed for {label}: {missing[:6]}. "
            "Refusing to train an incomplete face adapter."
        )


def prepare_dit_for_lora_training(
    model: Any,
    apply_lora_fn: Any,
    *,
    list_lora_blocks_fn: Any | None = None,
    **lora_kw: Any,
) -> tuple[Any, DiTLoRATrainModule]:
    rank = int(lora_kw.get("rank", lora_kw.get("lora_rank", 8)))
    lora_blocks = int(lora_kw.get("lora_blocks", -1))
    lora_scale = float(lora_kw.get("scale", lora_kw.get("lora_scale", 1.0)))
    lora_dropout = float(lora_kw.get("dropout", lora_kw.get("lora_dropout", 0.0)))
    module_keys = lora_kw.get("module_keys", lora_kw.get("lora_module_keys"))
    qlora_bits = lora_kw.get("qlora_bits")
    grad_checkpoint = bool(lora_kw.get("grad_checkpoint") or False)
    train_type = str(lora_kw.get("train_type") or "lora").strip().lower()
    if train_type == "dora" and qlora_bits in (4, 8):
        raise RuntimeError("DoRA requires a dequantized base weight; use LoRA with qlora_bits.")

    if train_type == "dora" and lora_dropout > 0:
        import logging

        logging.getLogger("danqing.lora").warning(
            "DoRA ignores lora_dropout=%s; dropout applies only to LoRA adapters.",
            lora_dropout,
        )

    apply_lora_fn(
        model,
        rank=rank,
        lora_blocks=lora_blocks,
        scale=lora_scale,
        dropout=lora_dropout,
        module_keys=module_keys,
        train_type=train_type,
    )
    freeze_dit_base_weights(model)
    if qlora_bits in (4, 8):
        n_q = quantize_frozen_dit_linears(model, bits=int(qlora_bits))
        if n_q == 0:
            raise RuntimeError("QLoRA requested but no quantizable Linear layers were found on DiT")
    if grad_checkpoint:
        if list_lora_blocks_fn is None:
            raise RuntimeError("grad_checkpoint requires list_lora_blocks_fn for this DiT family")
        blocks = list_lora_blocks_fn(model, lora_blocks=lora_blocks)
        enable_grad_checkpointing_on_blocks(blocks)
    train_module = build_lora_train_module(model)
    _assert_full_attention_coverage(model, module_keys=module_keys, lora_blocks=lora_blocks)
    mx.eval(train_module.parameters())
    return model, train_module


def apply_lora_to_flux1_dit(
    model: Any,
    *,
    rank: int,
    lora_blocks: int,
    scale: float | None = None,
    dropout: float = 0.0,
    module_keys: list[str] | None = None,
    train_type: str = "lora",
) -> None:
    """Replace Linear layers in the last ``lora_blocks`` joint+single blocks with LoRA."""
    lora_scale = float(1.0 if scale is None else scale)
    adapter_cls = DoRALinear if train_type == "dora" else LoRALinear
    dit = getattr(model, "_inner", model)
    for block in list_flux1_lora_blocks(model, lora_blocks=lora_blocks):
        _apply_lora_recursive(
            block,
            rank,
            scale=lora_scale,
            dropout=dropout,
            module_keys=module_keys,
            adapter_cls=adapter_cls,
        )
    if hasattr(dit, "_build_param_map"):
        dit._build_param_map()
    if hasattr(model, "_inner") and hasattr(dit, "_param_map"):
        model._param_map = dit._param_map


def enumerate_flux1_lora_module_paths(model: Any, *, lora_blocks: int) -> list[str]:
    """DiT module paths in the same order as ``iter_lora_linears_with_paths`` (indexed export)."""
    from backend.engine.config.model_configs import Flux1Config
    from backend.engine.families.flux1.transformer import Flux1Transformer
    from backend.engine.runtime.mlx import MLXContext

    inner = getattr(model, "_inner", model)
    ctx = getattr(inner, "ctx", None)
    config = getattr(model, "config", None) or getattr(inner, "config", None)
    if ctx is None or config is None:
        ctx = MLXContext()
        config = Flux1Config()
        probe = Flux1Transformer(config, ctx)
    else:
        probe = Flux1Transformer(config, ctx)
    apply_lora_to_flux1_dit(probe, rank=8, lora_blocks=lora_blocks)
    return [path for _, path in iter_lora_linears_with_paths(probe)]


def enumerate_zimage_lora_module_paths(model: Any, *, lora_blocks: int) -> list[str]:
    """DiT module paths in the same order as ``iter_lora_linears_with_paths`` (indexed export)."""
    probe = model
    inner = getattr(probe, "_inner", probe)
    ctx = getattr(inner, "ctx", None)
    config = getattr(probe, "config", None) or getattr(inner, "config", None)
    if ctx is None or config is None:
        from backend.engine.config.model_configs import ZImageConfig
        from backend.engine.runtime.mlx import MLXContext

        ctx = MLXContext()
        config = ZImageConfig()
        from backend.engine.families.z_image.transformer import ZImageTransformer

        probe = ZImageTransformer(config, ctx)
    else:
        from backend.engine.families.z_image.transformer import ZImageTransformer

        probe = ZImageTransformer(config, ctx)
    apply_lora_to_zimage_dit(probe, rank=8, lora_blocks=lora_blocks)
    return [path for _, path in iter_lora_linears_with_paths(probe)]


def repair_indexed_lora_weights(
    flat: dict[str, Any],
    *,
    module_paths: list[str],
) -> dict[str, Any]:
    """Rewrite legacy ``lora_{i}.lora_*`` safetensors keys to DiT module paths."""
    repaired: dict[str, Any] = {}
    passthrough: dict[str, Any] = {}
    for key, tensor in flat.items():
        if key.startswith("lora_") and (".lora_A." in key or ".lora_B." in key):
            continue
        passthrough[key] = tensor
    for idx, path in enumerate(module_paths):
        if f"{path}.lora_A.weight" in flat or f"{path}.lora_B.weight" in flat:
            continue
        a_key = f"lora_{idx}.lora_A.weight"
        b_key = f"lora_{idx}.lora_B.weight"
        if a_key not in flat or b_key not in flat:
            raise RuntimeError(
                f"Indexed LoRA export missing pair at index {idx} "
                f"(expected {len(module_paths)} adapters, keys like {a_key!r})"
            )
        repaired[f"{path}.lora_A.weight"] = flat[a_key]
        repaired[f"{path}.lora_B.weight"] = flat[b_key]
    repaired.update(passthrough)
    return repaired


def enumerate_qwen_lora_module_paths(model: Any, *, lora_blocks: int) -> list[str]:
    """DiT module paths (no ``dit.`` prefix) in ``iter_lora_linears_with_paths`` order."""
    probe = model
    inner = getattr(probe, "_inner", probe)
    ctx = getattr(inner, "ctx", None)
    config = getattr(probe, "config", None) or getattr(inner, "config", None)
    if ctx is None or config is None:
        from backend.engine.config.model_configs import QwenImageConfig
        from backend.engine.runtime.mlx import MLXContext

        ctx = MLXContext()
        config = QwenImageConfig()
        from backend.engine.families.qwen.transformer import QwenImageTransformer

        probe = QwenImageTransformer(config, ctx)
    else:
        from backend.engine.families.qwen.transformer import QwenImageTransformer

        probe = QwenImageTransformer(config, ctx)
    apply_lora_to_qwen_dit(probe, rank=8, lora_blocks=lora_blocks)
    return [
        path[4:] if path.startswith("dit.") else path
        for _, path in iter_lora_linears_with_paths(probe)
    ]


def apply_lora_to_qwen_dit(
    model: Any,
    *,
    rank: int,
    lora_blocks: int,
    scale: float | None = None,
    dropout: float = 0.0,
    module_keys: list[str] | None = None,
    train_type: str = "lora",
) -> None:
    """Replace Linear layers in the last ``lora_blocks`` Qwen-Image DiT blocks with LoRA."""
    lora_scale = float(1.0 if scale is None else scale)
    adapter_cls = DoRALinear if train_type == "dora" else LoRALinear
    dit = getattr(model, "_inner", model)
    inner = getattr(dit, "dit", dit)
    for block in list_qwen_lora_blocks(model, lora_blocks=lora_blocks):
        _apply_lora_recursive(
            block,
            rank,
            scale=lora_scale,
            dropout=dropout,
            module_keys=module_keys,
            adapter_cls=adapter_cls,
        )
    if hasattr(dit, "_build_param_map"):
        dit._build_param_map()
    if hasattr(model, "_inner") and hasattr(dit, "_param_map"):
        model._param_map = dit._param_map


def apply_lora_to_zimage_dit(
    model: Any,
    *,
    rank: int,
    lora_blocks: int,
    scale: float | None = None,
    dropout: float = 0.0,
    module_keys: list[str] | None = None,
    train_type: str = "lora",
) -> None:
    """Attach LoRA to Z-Image's live plain-Python DiT block tree."""
    lora_scale = float(1.0 if scale is None else scale)
    adapter_cls = DoRALinear if train_type == "dora" else LoRALinear
    dit = getattr(model, "_inner", model)
    for block in list_zimage_lora_blocks(model, lora_blocks=lora_blocks):
        _apply_lora_recursive(
            block,
            rank,
            scale=lora_scale,
            dropout=dropout,
            module_keys=module_keys,
            adapter_cls=adapter_cls,
        )
    # ``TransformerBase``'s legacy map points at the replaced Linear objects;
    # update it through the adapter-aware collector without dropping frozen weights.
    from backend.engine.common.model.base import flatten_adapter_parameters

    dit._param_map = flatten_adapter_parameters(dit.parameters(), "")
    for name in ("x_pad_token", "cap_pad_token"):
        if hasattr(dit, name):
            dit._param_map[name] = getattr(dit, name)
    if hasattr(model, "_inner"):
        model._param_map = dit._param_map


def add_grad_trees(left: Any, right: Any) -> Any:
    """Element-wise add for nested MLX grad trees from ``value_and_grad``."""
    from mlx.utils import tree_map

    return tree_map(lambda a, b: a + b, left, right)


def scale_grad_tree(tree: Any, divisor: float) -> Any:
    from mlx.utils import tree_map

    return tree_map(lambda g: g / divisor, tree)


def load_lora_into_train_module(model: Any, adapter_path: Any, *, rank: int) -> None:
    """Load exported adapter weights into ``DiTLoRATrainModule`` trainable tensors."""
    from pathlib import Path

    from mlx.utils import tree_unflatten

    from backend.engine.common.bundle.weights import load_safetensors

    flat_file = load_safetensors(str(adapter_path))
    paths = getattr(model, "_lora_paths", None)
    if not paths:
        raise RuntimeError("load_lora_into_train_module requires DiTLoRATrainModule with _lora_paths")
    updates: dict[str, mx.array] = {}
    for idx, path in enumerate(paths):
        a_key = f"{path}.lora_A.weight"
        b_key = f"{path}.lora_B.weight"
        if a_key not in flat_file or b_key not in flat_file:
            raise RuntimeError(f"Resume adapter missing LoRA pair for {path!r}")
        updates[f"lora_{idx}.lora_a"] = flat_file[a_key].T
        updates[f"lora_{idx}.lora_b"] = flat_file[b_key].T
        mag_key = f"{path}.dora_magnitude.weight"
        if mag_key in flat_file:
            updates[f"lora_{idx}.magnitude"] = flat_file[mag_key]
    model.update(tree_unflatten(list(updates.items())))
    mx.eval(model.parameters())


def collect_lora_safetensors(model: Any, *, rank: int) -> dict[str, mx.array]:
    """Export trainable LoRA weights with diffusers-compatible key names."""
    from mlx.utils import tree_flatten

    flat = dict(tree_flatten(model.trainable_parameters()))
    paths = getattr(model, "_lora_paths", None)
    out: dict[str, mx.array] = {}
    if paths:
        for idx, path in enumerate(paths):
            for leaf, suffix in (("lora_a", "lora_A"), ("lora_b", "lora_B")):
                key = f"lora_{idx}.{leaf}"
                tensor = flat.get(key)
                if tensor is not None:
                    out[f"{path}.{suffix}.weight"] = tensor.T
            mag = flat.get(f"lora_{idx}.magnitude")
            if mag is not None:
                out[f"{path}.dora_magnitude.weight"] = mag
    else:
        for key, tensor in flat.items():
            if ".lora_a" in key:
                base = key.replace(".lora_a", "")
                out[f"{base}.lora_A.weight"] = tensor.T
            elif ".lora_b" in key:
                base = key.replace(".lora_b", "")
                out[f"{base}.lora_B.weight"] = tensor.T
    if not out:
        raise RuntimeError("No LoRA trainable parameters found to save")
    out["lora_rank"] = mx.array([float(rank)])
    return out


def collect_fused_adapter_deltas(model: Any) -> dict[str, mx.array]:
    """Export dense weight deltas (trainable LoRA + optional frozen turbo assistant)."""
    out: dict[str, mx.array] = {}
    for layer, path in iter_lora_linears_with_paths(model):
        base_w = layer.linear.weight
        if isinstance(layer, DoRALinear):
            eff = layer.effective_weight()
        elif isinstance(layer, LoRALinear):
            delta = (layer.lora_a @ layer.lora_b).T * layer.scale
            if layer.assistant_enabled:
                if layer.assistant_dense is not None:
                    delta = delta + layer.assistant_dense.astype(mx.float32)
                elif layer.assistant_lora_a is not None and layer.assistant_lora_b is not None:
                    delta = delta + (
                        (layer.assistant_lora_a @ layer.assistant_lora_b).T * layer.assistant_scale
                    ).astype(mx.float32)
            eff = base_w + delta.astype(base_w.dtype)
        else:
            eff = base_w + (layer.lora_a @ layer.lora_b).T * layer.scale
        mx.eval(eff, base_w)
        out[f"{path}.delta.weight"] = (eff - base_w).astype(base_w.dtype)
    if not out:
        raise RuntimeError("No adapter layers found for fused export")
    return out


def write_fused_adapters_from_published(
    train_module: Any,
    *,
    final_path: Path,
    fused_path: Path,
    rank: int,
    base_model_id: str,
) -> None:
    """Fuse the published (best/final) adapter, not the last in-memory step."""
    import json

    if not final_path.is_file():
        raise RuntimeError(f"Cannot fuse adapters; published weights missing: {final_path}")
    load_lora_into_train_module(train_module, final_path, rank=rank)
    mx.save_safetensors(str(fused_path), collect_fused_adapter_deltas(train_module))
    fused_path.with_suffix(".json").write_text(
        json.dumps({"format": "dense_delta", "base_model": base_model_id, "source": final_path.name}, indent=2),
        encoding="utf-8",
    )
