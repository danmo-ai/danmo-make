"""MiniMax-H3 DiT quantization layout (PipeNetwork ``quant_config.json`` parity)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.nn as nn

CORE_LINEARS = (
    ".attn.qkv_proj",
    ".attn.out_proj",
    ".mlp.fc1",
    ".mlp.fc2",
)

NEVER_QUANTIZE = (
    "video_patch_proj",
    "audio_patch_proj",
    "condition_proj",
    "time_embedder",
    "final_layer.video_out",
    "final_layer.audio_out",
    "final_layer.adaln_proj",
)


@dataclass
class MiniMaxH3QuantConfig:
    bits: int = 4
    group_size: int = 64
    quantize_adaln: bool = False
    adaln_bits: int = 8
    overrides: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_recipe(cls, recipe: dict[str, Any]) -> "MiniMaxH3QuantConfig":
        return cls(
            bits=int(recipe["bits"]),
            group_size=int(recipe.get("group_size", 64)),
            quantize_adaln=bool(recipe.get("quantize_adaln", False)),
            adaln_bits=int(recipe.get("adaln_bits") or 8),
        )

    def bits_for(self, path: str) -> int | None:
        if path in self.overrides:
            return self.overrides[path]
        if any(path.endswith(suffix) or f"{suffix}." in path for suffix in NEVER_QUANTIZE):
            return None
        if any(name in path for name in NEVER_QUANTIZE):
            return None
        if ".adaln_proj." in path or path.endswith(".adaln_proj.linear"):
            return self.adaln_bits if self.quantize_adaln else None
        if any(path.endswith(suffix) for suffix in CORE_LINEARS):
            return self.bits
        return None


def _class_predicate(config: MiniMaxH3QuantConfig):
    def predicate(path: str, module: nn.Module) -> bool | dict:
        if not isinstance(module, nn.Linear):
            return False
        bits = config.bits_for(path)
        if bits is None:
            return False
        if module.weight.shape[-1] % config.group_size:
            return False
        return {"group_size": config.group_size, "bits": bits}

    return predicate


def apply_minimax_h3_quant_structure(model: nn.Module, config: MiniMaxH3QuantConfig) -> None:
    """Convert ``MiniMaxH3DiTMLX`` to quantized layers before loading PipeNetwork shards."""
    nn.quantize(
        model,
        group_size=config.group_size,
        bits=config.bits,
        class_predicate=_class_predicate(config),
    )
