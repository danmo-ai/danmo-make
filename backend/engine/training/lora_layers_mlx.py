"""MLX LoRA linear layers for DreamBooth training."""

from __future__ import annotations

import math
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

    def _assistant_contribution(self, x: mx.array) -> mx.array:
        if self.assistant_dense is not None:
            return x @ self.assistant_dense.T.astype(x.dtype)
        if self.assistant_lora_a is None or self.assistant_lora_b is None:
            raise RuntimeError("Frozen assistant LoRA tensors are missing")
        z = (x @ self.assistant_lora_a) @ self.assistant_lora_b
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
        return wrapped

    def _assistant_contribution(self, x: mx.array) -> mx.array:
        if self.assistant_dense is not None:
            return x @ self.assistant_dense.T.astype(x.dtype)
        if self.assistant_lora_a is None or self.assistant_lora_b is None:
            raise RuntimeError("Frozen assistant LoRA tensors are missing")
        z = (x @ self.assistant_lora_a) @ self.assistant_lora_b
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

    def effective_weight(self) -> mx.array:
        w = self.linear.weight
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


def _match_lora_module_key(attr_name: str, module_keys: list[str] | None) -> bool:
    if not module_keys:
        return True
    return any(attr_name == k or attr_name.endswith(f".{k}") for k in module_keys)


def _is_trainable_base_linear(module: Any) -> bool:
    return (
        isinstance(module, nn.Linear)
        and not _is_adapter_linear(module)
        and not _is_frozen_assistant_linear(module)
    )


def _apply_lora_recursive(
    module: Any,
    rank: int,
    *,
    scale: float = 1.0,
    dropout: float = 0.0,
    module_keys: list[str] | None = None,
    adapter_cls: type = LoRALinear,
) -> None:
    if _is_trainable_base_linear(module):
        return
    if isinstance(module, list):
        for i, item in enumerate(module):
            if _is_trainable_base_linear(item):
                if not module_keys:
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
        return
    if isinstance(module, nn.Module) and not _is_leaf_linear(module):
        for name, child in module.children().items():
            if _is_trainable_base_linear(child) and _match_lora_module_key(name, module_keys):
                setattr(
                    module,
                    name,
                    adapter_cls.from_base(child, r=rank, scale=scale, dropout=dropout),
                )
            else:
                _apply_lora_recursive(
                    child,
                    rank,
                    scale=scale,
                    dropout=dropout,
                    module_keys=module_keys,
                    adapter_cls=adapter_cls,
                )
        return
    if not hasattr(module, "__dict__"):
        return
    for name, val in list(vars(module).items()):
        if name.startswith("_"):
            continue
        if _is_trainable_base_linear(val) and _match_lora_module_key(name, module_keys):
            setattr(
                module,
                name,
                adapter_cls.from_base(val, r=rank, scale=scale, dropout=dropout),
            )
        else:
            _apply_lora_recursive(
                val,
                rank,
                scale=scale,
                dropout=dropout,
                module_keys=module_keys,
                adapter_cls=adapter_cls,
            )


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
    if isinstance(root, nn.Module) and not _is_leaf_linear(root):
        for child in root.children().values():
            yield from _walk_dit_nodes(child)
        return
    if not hasattr(root, "__dict__"):
        return
    for name, val in vars(root).items():
        if name.startswith("_"):
            continue
        yield from _walk_dit_nodes(val)


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
            for name, child in node.children().items():
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
        if isinstance(layer, (LoRALinear, FrozenAssistantLinear)):
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
            for name, child in node.children().items():
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


def iter_lora_linears_with_paths(model: Any) -> list[tuple[Any, str]]:
    entries = _walk_lora_paths(_dit_inner(model))
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
            for name, child in obj.children().items():
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


def grad_checkpoint(layer: Any) -> None:
    """Enable gradient checkpointing for all instances of *layer*'s type."""
    layer_cls = type(layer)
    if getattr(layer_cls, "_dq_grad_checkpoint", False):
        return
    fn = layer_cls.__call__

    def checkpointed_fn(model: Any, *args: Any, **kwargs: Any) -> Any:
        def inner_fn(params: Any, *inner_args: Any, **inner_kwargs: Any) -> Any:
            model.update(params)
            return fn(model, *inner_args, **inner_kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    checkpointed_fn._dq_grad_checkpoint = True  # type: ignore[attr-defined]
    layer_cls.__call__ = checkpointed_fn


def enable_grad_checkpointing_on_blocks(blocks: list[Any]) -> None:
    seen: set[type] = set()
    for block in blocks:
        cls = type(block)
        if cls in seen:
            continue
        seen.add(cls)
        grad_checkpoint(block)


def list_flux1_lora_blocks(model: Any, *, lora_blocks: int) -> list[Any]:
    dit = getattr(model, "_inner", model)
    joint = list(dit.transformer_blocks)
    single = list(dit.single_transformer_blocks)
    all_blocks = list(reversed(joint + single))
    n = len(all_blocks) if lora_blocks < 0 else min(lora_blocks, len(all_blocks))
    return all_blocks[:n]


def list_zimage_lora_blocks(model: Any, *, lora_blocks: int) -> list[Any]:
    dit = getattr(model, "_inner", model)
    all_blocks: list[Any] = []
    for layer in reversed(list(dit.layers)):
        all_blocks.append(layer)
    for layer in reversed(list(dit.noise_refiner)):
        all_blocks.append(layer)
    for layer in reversed(list(dit.context_refiner)):
        all_blocks.append(layer)
    n = len(all_blocks) if lora_blocks < 0 else min(lora_blocks, len(all_blocks))
    return all_blocks[:n]


def list_qwen_lora_blocks(model: Any, *, lora_blocks: int) -> list[Any]:
    dit = getattr(model, "_inner", model)
    inner = getattr(dit, "dit", dit)
    all_blocks = list(reversed(list(inner.transformer_blocks)))
    n = len(all_blocks) if lora_blocks < 0 else min(lora_blocks, len(all_blocks))
    return all_blocks[:n]


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
        enable_grad_checkpointing_on_blocks(list_lora_blocks_fn(model, lora_blocks=lora_blocks))
    train_module = build_lora_train_module(model)
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
    if hasattr(inner, "_build_param_map"):
        inner._build_param_map()
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
    """Replace Linear layers in the last ``lora_blocks`` Z-Image DiT blocks with LoRA."""
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
    if hasattr(dit, "_build_param_map"):
        dit._build_param_map()
    if hasattr(model, "_inner") and hasattr(dit, "_param_map"):
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
