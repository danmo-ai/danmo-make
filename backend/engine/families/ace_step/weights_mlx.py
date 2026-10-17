"""ACE-Step safetensors → MLX weight loading (mlx imports allowed here only)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, List, Tuple

from backend.engine.runtime.mlx_runtime import (
    iter_safetensors_float32_numpy as _iter_safetensors_float32_numpy,
)
from backend.engine.families.ace_step.weights import _convert_decoder_tensor_for_mlx


def resolve_dit_checkpoint_path(dit_bundle: Path) -> Path:
    """Return single ``model.safetensors`` or the DiT dir when only a sharded index exists."""
    root = Path(dit_bundle)
    single = root / "model.safetensors"
    if single.is_file():
        return single
    index = root / "model.safetensors.index.json"
    if index.is_file():
        return root
    if root.is_file() and root.name.endswith(".safetensors"):
        return root
    raise RuntimeError(
        f"ACE-Step DiT checkpoint missing under {root} "
        "(expected model.safetensors or model.safetensors.index.json)."
    )


def iter_dit_checkpoint_tensors(checkpoint: str | Path) -> Iterator[Tuple[str, Any]]:
    """Yield ``(key, float32 ndarray)`` from a single file or HF-style sharded index."""
    path = Path(checkpoint)
    if path.is_file():
        yield from _iter_safetensors_float32_numpy(str(path))
        return
    if not path.is_dir():
        raise RuntimeError(f"ACE-Step checkpoint path not found: {path}")
    single = path / "model.safetensors"
    if single.is_file():
        yield from _iter_safetensors_float32_numpy(str(single))
        return
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(
            f"ACE-Step sharded index missing: {index_path} "
            "(and no model.safetensors present)."
        )
    weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map") or {}
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"ACE-Step weight_map empty or invalid in {index_path}")
    shard_names = sorted(set(str(v) for v in weight_map.values()))
    for name in shard_names:
        shard = path / name
        if not shard.is_file():
            raise RuntimeError(f"ACE-Step shard missing: {shard}")
        yield from _iter_safetensors_float32_numpy(str(shard))


@dataclass
class AceStepWeightGroups:
    decoder: List[Tuple[str, Any]] = field(default_factory=list)
    encoder: List[Tuple[str, Any]] = field(default_factory=list)
    tokenizer: List[Tuple[str, Any]] = field(default_factory=list)
    detokenizer: List[Tuple[str, Any]] = field(default_factory=list)
    special: dict[str, Any] = field(default_factory=dict)


_SPECIAL_KEYS = frozenset(
    {
        "tokenizer.attention_pooler.special_token",
        "detokenizer.special_tokens",
        "null_condition_emb",
    }
)


def load_ace_step_weight_groups(
    dit_bundle: Path, *, array_fn: Any | None = None
) -> AceStepWeightGroups:
    """Single-pass load of DiT / encoder / codec tensors from a mono or sharded checkpoint."""
    import mlx.core as mx

    if array_fn is None:
        array_fn = mx.array
    ckpt = resolve_dit_checkpoint_path(dit_bundle)
    groups = AceStepWeightGroups()
    for key, tensor in iter_dit_checkpoint_tensors(ckpt):
        if key in _SPECIAL_KEYS:
            groups.special[key] = array_fn(tensor)
            continue
        if key.startswith("decoder.") and "rotary_emb" not in key:
            mlx_key, np_val = _convert_decoder_tensor_for_mlx(key, tensor)
            groups.decoder.append((mlx_key, array_fn(np_val)))
        elif key.startswith("encoder."):
            groups.encoder.append((key[len("encoder.") :], array_fn(tensor)))
        elif key.startswith("tokenizer."):
            groups.tokenizer.append((key[len("tokenizer.") :], array_fn(tensor)))
        elif key.startswith("detokenizer."):
            groups.detokenizer.append((key[len("detokenizer.") :], array_fn(tensor)))
    if not groups.decoder:
        raise RuntimeError(f"No decoder.* weights found under {ckpt}")
    if not groups.encoder:
        raise RuntimeError(f"No encoder.* weights found under {ckpt}")
    if not groups.tokenizer:
        raise RuntimeError(f"No tokenizer.* weights found under {ckpt}")
    if not groups.detokenizer:
        raise RuntimeError(f"No detokenizer.* weights found under {ckpt}")
    return groups


def load_prefix_weights_for_mlx(
    safetensors_path: str,
    prefix: str,
    *,
    strip_prefix: bool = True,
    array_fn: Any | None = None,
) -> List[Tuple[str, Any]]:
    import mlx.core as mx

    if array_fn is None:
        array_fn = mx.array
    weights: List[Tuple[str, Any]] = []
    for key, arr in iter_dit_checkpoint_tensors(safetensors_path):
        if not key.startswith(prefix):
            continue
        mlx_key = key[len(prefix) :] if strip_prefix else key
        weights.append((mlx_key, array_fn(arr)))
    if not weights:
        raise RuntimeError(f"No weights with prefix {prefix!r} in {safetensors_path}")
    return weights


def load_decoder_safetensors_for_mlx(
    safetensors_path: str, *, array_fn: Any | None = None
) -> List[Tuple[str, Any]]:
    import mlx.core as mx

    if array_fn is None:
        array_fn = mx.array
    weights: List[Tuple[str, Any]] = []
    for key, tensor in iter_dit_checkpoint_tensors(safetensors_path):
        if not key.startswith("decoder.") or "rotary_emb" in key:
            continue
        mlx_key, np_val = _convert_decoder_tensor_for_mlx(key, tensor)
        weights.append((mlx_key, array_fn(np_val)))
    return weights
