"""Z-Image weight remapping."""
from __future__ import annotations

from typing import Any

Z_IMAGE_LORA_MODEL_IDS = frozenset({"z-image", "z-image-turbo"})


def z_image_lora_scope_key(value: str) -> str:
    """Base and Turbo share one LoRA scope (same DiT ``_param_map``)."""
    mid = value.split(":", 1)[0].strip() if value else ""
    if mid in Z_IMAGE_LORA_MODEL_IDS:
        return "z_image"
    return mid


def z_image_lora_base_compatible(model_id: str, lora_base: str) -> bool:
    model_key = (model_id or "").split(":", 1)[0].strip()
    lora_key = (lora_base or "").split(":", 1)[0].strip()
    if not model_key or not lora_key:
        return False
    return lora_key == model_key


def remap_zimage_weights(weights: dict, patch_size: int = 2) -> dict:
    """Map diffusers-format Z-Image weight keys to DanQing engine keys.

    Source format (diffusers):
        all_x_embedder.2-1.weight
        all_final_layer.2-1.adaLN_modulation.1.bias
        t_embedder.mlp.0.weight
        layers.0.adaLN_modulation.0.weight
        layers.0.attention.to_out.0.weight
        cap_embedder.0.weight  (RMSNorm)
        cap_embedder.1.bias    (Linear)
        context_refiner.0.attention_norm1.weight

    Target format (DanQing):
        x_embedder.weight
        final_layer.adaLN_modulation.bias
        t_embedder.linear1.weight
        layers.0.adaLN_modulation.weight
        layers.0.attention.to_out.weight
        cap_norm.weight
        cap_embedder.bias
        context_refiner.0.attn_norm1.weight
    """
    prefix_key = f"{patch_size}-1"
    remapped = {}

    for key, tensor in weights.items():
        new_key = key

        # all_x_embedder.{patch}-1 → x_embedder
        new_key = new_key.replace(f"all_x_embedder.{prefix_key}", "x_embedder")
        # all_final_layer.{patch}-1 → final_layer
        new_key = new_key.replace(f"all_final_layer.{prefix_key}", "final_layer")

        # t_embedder.mlp.0 → t_embedder.linear1
        new_key = new_key.replace("t_embedder.mlp.0.", "t_embedder.linear1.")
        # t_embedder.mlp.2 → t_embedder.linear2
        new_key = new_key.replace("t_embedder.mlp.2.", "t_embedder.linear2.")

        # attention_norm1 → attn_norm1
        new_key = new_key.replace("attention_norm1.", "attn_norm1.")
        # attention_norm2 → attn_norm2
        new_key = new_key.replace("attention_norm2.", "attn_norm2.")

        # .to_out.0 → .to_out (DanQing uses direct Linear, not Sequential/list)
        new_key = new_key.replace(".to_out.0.", ".to_out.")

        # adaLN_modulation: reference uses list [nn.Linear(...)], DanQing also uses list now
        # Keep .0. index for transformer blocks and noise_refiner/context_refiner
        # For final_layer: diffusers uses .1., reference maps to .0., keep .0.
        new_key = new_key.replace(".adaLN_modulation.1.", ".adaLN_modulation.0.")

        # cap_embedder.0 → cap_norm (RMSNorm)
        new_key = new_key.replace("cap_embedder.0.", "cap_norm.")
        # cap_embedder.1 → cap_embedder (Linear)
        new_key = new_key.replace("cap_embedder.1.", "cap_embedder.")

        remapped[new_key] = tensor

    return remapped


def remap_zimage_control_weights(weights: dict, patch_size: int = 2) -> dict:
    """Map diffusers Z-Image Fun ControlNet keys → DanQing ``ZImageControlRuntime`` flat keys."""
    prefix_key = f"{patch_size}-1"
    remapped: dict[str, Any] = {}
    for key, tensor in weights.items():
        if not key.startswith("control_"):
            continue
        new_key = key
        new_key = new_key.replace(f"control_all_x_embedder.{prefix_key}", "control_x_embedder")
        new_key = new_key.replace("attention_norm1.", "attn_norm1.")
        new_key = new_key.replace("attention_norm2.", "attn_norm2.")
        new_key = new_key.replace(".to_out.0.", ".to_out.")
        new_key = new_key.replace(".adaLN_modulation.1.", ".adaLN_modulation.0.")
        remapped[new_key] = tensor
    return remapped


def remap_zimage_lora_module_prefix(module: str, patch_size: int = 2) -> str:
    """Map a LoRA module path (from checkpoint keys) → DanQing ``ZImageTransformer`` parameter prefix.

    Output is a prefix without ``.weight`` / ``.bias`` (must match ``_collect_nn_params`` paths, and
    ``lora_mlx`` appends ``.weight`` when merging).
    """
    m = (module or "").strip().strip(".")
    while ".." in m:
        m = m.replace("..", ".")
    for pref in (
        "transformer.",
        "z_image_transformer.",
        "diffusion_model.",
        "pipe.transformer.",
        "model.",
    ):
        if m.startswith(pref):
            m = m[len(pref) :]
    m = m.replace(".base_model.model.", ".")
    m = m.replace(".default.", ".")
    if m.endswith(".default"):
        m = m[: -len(".default")].rstrip(".")

    prefix_key = f"{patch_size}-1"
    m = m.replace(f"all_x_embedder.{prefix_key}", "x_embedder")
    m = m.replace(f"all_final_layer.{prefix_key}", "final_layer")
    m = m.replace("attention_norm1.", "attn_norm1.")
    m = m.replace("attention_norm2.", "attn_norm2.")
    m = m.replace(".to_out.0.", ".to_out.")
    m = m.replace("t_embedder.mlp.0.", "t_embedder.linear1.")
    m = m.replace("t_embedder.mlp.2.", "t_embedder.linear2.")
    m = m.replace("cap_embedder.0.", "cap_norm.")
    m = m.replace("cap_embedder.1.", "cap_embedder.")
    m = m.replace(".adaLN_modulation.1.", ".adaLN_modulation.0.")
    m = m.strip(".").strip()
    # Keys like ``...to_q.lora_A.weight`` → ``...to_q.weight`` after ``_lora_key_to_module``; strip the
    # trailing parameter leaf so ``lora_mlx`` can append ``.weight`` once (``...to_q.weight``).
    if m.endswith(".weight"):
        m = m[: -len(".weight")].rstrip(".")
    elif m.endswith(".bias"):
        m = m[: -len(".bias")].rstrip(".")
    return m.strip(".").strip()


def remap_zimage_lora_keys(lora_weights: dict, patch_size: int = 2) -> dict[str, tuple]:
    """Group LoRA tensors and map module names to DanQing Z-Image ``_param_map`` prefixes (no ``.weight``)."""
    from backend.engine.common.bundle.weights import _lora_key_to_module

    # Safetensors alpha is optional; Kohya/diffusers use alpha=rank when omitted.
    # Studio always writes its trained alpha explicitly; never assume rank=8.
    default_alpha: float | None = None
    alphas_by_tgt: dict[str, float] = {}
    for key, tensor in lora_weights.items():
        lk = key.lower()
        if "alpha" not in lk:
            continue
        if any(x in lk for x in ("lora_down", "lora_up", "lora_a", "lora_b")):
            continue
        if not lk.endswith(".alpha"):
            continue
        base = key[: -len(".alpha")] if key.lower().endswith(".alpha") else key
        tgt = remap_zimage_lora_module_prefix(base, patch_size)
        if not tgt:
            continue
        try:
            val = tensor.item() if hasattr(tensor, "item") else float(tensor)
            alphas_by_tgt[tgt] = float(val)
        except (TypeError, ValueError):
            pass

    groups: dict[str, dict[str, Any]] = {}
    for key, tensor in lora_weights.items():
        kl = key.lower()
        if "alpha" in kl or "dora_magnitude" in kl or key.endswith(".delta.weight"):
            continue
        module = _lora_key_to_module(key)
        if module not in groups:
            groups[module] = {}
        if "lora_down" in kl or "lora_a." in kl or "lora_a_weight" in kl:
            groups[module]["down"] = tensor
        elif "lora_up" in kl or "lora_b." in kl or "lora_b_weight" in kl:
            groups[module]["up"] = tensor

    out: dict[str, tuple[Any, Any, float]] = {}
    for module, parts in groups.items():
        if "down" not in parts or "up" not in parts:
            raise RuntimeError(
                f"Z-Image LoRA module {module!r} is missing a down/up factor; "
                "refusing to apply a partial adapter."
            )
        down, up = parts["down"], parts["up"]
        tgt = remap_zimage_lora_module_prefix(module, patch_size)
        if not tgt:
            raise RuntimeError(f"Z-Image LoRA module {module!r} has no valid target prefix.")
        fallback_alpha = default_alpha if default_alpha is not None else float(down.shape[0])
        alpha = float(alphas_by_tgt.get(tgt, fallback_alpha))
        out[tgt] = (down, up, alpha)
    return out

