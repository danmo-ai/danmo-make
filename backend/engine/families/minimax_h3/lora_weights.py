"""MiniMax-H3 Turbo LoRA key remap (Diffusers / Comfy → ``MiniMaxH3DiTMLX`` flat keys)."""
from __future__ import annotations

import re
from typing import Any


def _lora_key_to_module(key: str) -> str:
    for suffix in (
        ".lora_A.weight",
        ".lora_B.weight",
        ".lora_A",
        ".lora_B",
        ".lora_down.weight",
        ".lora_up.weight",
        ".lora_down",
        ".lora_up",
        "lora_A.weight",
        "lora_B.weight",
        "lora_down.weight",
        "lora_up.weight",
    ):
        if suffix in key:
            return key.replace(suffix, "").rstrip(".")
    return key


def _normalize_module(module: str) -> str:
    out = module.strip()
    for prefix in (
        "model.diffusion_model.",
        "diffusion_model.",
        "model.",
    ):
        if out.startswith(prefix):
            out = out[len(prefix) :]
    # Comfy / upstream occasional alias → Diffusers layout used by ``transformer_mlx``.
    out = re.sub(
        r"^blocks\.(\d+)\.",
        r"transformer_blocks.\1.",
        out,
    )
    return out


def minimax_h3_lora_param_key(module: str) -> str:
    mod = _normalize_module(module)
    if mod.endswith(".weight"):
        return mod
    return f"{mod}.weight"


def remap_minimax_h3_lora_keys(
    lora_weights: dict[str, Any],
    *,
    default_alpha: float = 1.0,
) -> dict[str, tuple[Any, Any, float]]:
    """Group LoRA A/B tensors by target DiT module (without ``.weight`` suffix)."""
    groups: dict[str, dict[str, Any]] = {}
    alpha_by_module: dict[str, float] = {}

    for key, tensor in lora_weights.items():
        kl = key.lower()
        if "alpha" in kl and "lora" in kl:
            mod = _normalize_module(_lora_key_to_module(key))
            try:
                alpha_by_module[mod] = float(tensor.item() if hasattr(tensor, "item") else tensor)
            except (TypeError, ValueError):
                pass
            continue
        module = _normalize_module(_lora_key_to_module(key))
        bucket = groups.setdefault(module, {})
        if "lora_down" in kl or "lora_a" in kl:
            bucket["down"] = tensor
        elif "lora_up" in kl or "lora_b" in kl:
            bucket["up"] = tensor

    remapped: dict[str, tuple[Any, Any, float]] = {}
    for module, parts in groups.items():
        if "up" not in parts or "down" not in parts:
            continue
        alpha = float(alpha_by_module.get(module, default_alpha))
        remapped[module] = (parts["down"], parts["up"], alpha)
    return remapped
