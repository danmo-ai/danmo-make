"""Flux.1 权重键映射 — diffusers ``FluxTransformer2DModel`` → DanQing 模块命名。"""
from __future__ import annotations

import re
from typing import Any


def _reshape_patch_embed_weight(key: str, tensor: Any) -> Any:
    """diffusers ``x_embedder`` Linear (out, in) → ``patch_embed.proj`` Conv2d (out, 1, 1, in)."""
    if key != "patch_embed.proj.weight":
        return tensor
    if not hasattr(tensor, "shape"):
        return tensor
    sh = tuple(tensor.shape)
    if len(sh) == 2:
        out_ch, in_ch = sh
        return tensor.reshape(out_ch, 1, 1, in_ch)
    return tensor


def remap_flux1_clip_weights(weights: dict) -> dict:
    """diffusers CLIP keys already match ``CLIPEncoder`` module paths."""
    return dict(weights)


def nest_flux1_clip_weights(flat: dict[str, Any]) -> dict[str, Any]:
    """Flat ``text_model.*`` keys → nested dict for ``mlx.nn.Module.update`` (reference loader shape)."""

    def _insert(root: Any, parts: list[str], value: Any) -> None:
        cur = root
        for i, part in enumerate(parts[:-1]):
            nxt = parts[i + 1]
            nxt_is_idx = nxt.isdigit()
            if part.isdigit():
                idx = int(part)
                while len(cur) <= idx:
                    cur.append(None)
                if cur[idx] is None:
                    cur[idx] = [] if nxt_is_idx else {}
                cur = cur[idx]
            else:
                if part not in cur:
                    cur[part] = [] if nxt_is_idx else {}
                cur = cur[part]
        leaf = parts[-1]
        if isinstance(cur, list):
            raise RuntimeError(f"nest_flux1_clip_weights: invalid leaf index path: {parts!r}")
        cur[leaf] = value

    nested: dict[str, Any] = {}
    for key, tensor in flat.items():
        if not key.startswith("text_model."):
            raise RuntimeError(f"nest_flux1_clip_weights: unexpected key prefix: {key!r}")
        _insert(nested, key.split("."), tensor)
    return nested


def remap_flux1_t5_weights(weights: dict) -> dict:
    """diffusers T5 keys → ``T5Encoder`` module names."""
    remapped: dict[str, Any] = {}
    rel_bias_key = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    rel_bias = weights.get(rel_bias_key)

    for key, tensor in weights.items():
        if "relative_attention_bias" in key and key != rel_bias_key:
            continue
        new_key = key
        new_key = new_key.replace("encoder.final_layer_norm.", "final_layer_norm.")
        new_key = new_key.replace("encoder.block.", "t5_blocks.")
        new_key = new_key.replace(".layer.0.", ".attention.")
        new_key = new_key.replace(".layer.1.", ".ff.")
        remapped[new_key] = tensor

    if rel_bias is not None:
        for i in range(24):
            remapped[
                f"t5_blocks.{i}.attention.SelfAttention.relative_attention_bias.weight"
            ] = rel_bias
    return remapped


def remap_flux1_weights(weights: dict) -> dict:
    """Normalize checkpoint keys for ``Flux1Transformer``.

    Supports diffusers sharded bundles (``x_embedder``, ``context_embedder``, ``time_text_embed``)
    and legacy BFL/BFL-style ``double_blocks.*`` (handled via block index rewrites below).
    """
    remapped: dict[str, Any] = {}
    for key, tensor in weights.items():
        new_key = key
        new_key = new_key.replace(".to_out.0.", ".to_out.")
        new_key = new_key.replace(".ff.net.0.proj.", ".ff.net_0_proj.")
        new_key = new_key.replace(".ff.net.2.", ".ff.net_2.")
        new_key = new_key.replace(".ff_context.net.0.proj.", ".ff_context.net_0_proj.")
        new_key = new_key.replace(".ff_context.net.2.", ".ff_context.net_2.")
        # diffusers input / context / time+text embeds
        new_key = new_key.replace("x_embedder.", "patch_embed.proj.")
        new_key = new_key.replace("context_embedder.", "txt_in.")
        new_key = new_key.replace(
            "time_text_embed.timestep_embedder.linear_1", "time_in.mlp.layers.0"
        )
        new_key = new_key.replace(
            "time_text_embed.timestep_embedder.linear_2", "time_in.mlp.layers.2"
        )
        new_key = new_key.replace(
            "time_text_embed.text_embedder.linear_1", "vector_in.layers.0"
        )
        new_key = new_key.replace(
            "time_text_embed.text_embedder.linear_2", "vector_in.layers.2"
        )
        new_key = new_key.replace(
            "time_text_embed.guidance_embedder.linear_1", "guidance_in.mlp.layers.0"
        )
        new_key = new_key.replace(
            "time_text_embed.guidance_embedder.linear_2", "guidance_in.mlp.layers.2"
        )
        # legacy alias (older remap table)
        new_key = new_key.replace("time_text_embed.text_proj.", "vector_in.layers.0.")
        tensor = _reshape_patch_embed_weight(new_key, tensor)
        remapped[new_key] = tensor
    return remapped


def remap_flux1_lora_module_prefix(module: str) -> str:
    """Map LoRA module path → DanQing ``Flux1Transformer`` ``_param_map`` prefix (no ``.weight``)."""
    m = (module or "").strip().strip(".")
    while ".." in m:
        m = m.replace("..", ".")
    for pref in (
        "transformer.",
        "diffusion_model.",
        "pipe.transformer.",
        "model.",
        "lora_unet_",
    ):
        if m.startswith(pref):
            m = m[len(pref) :]
    m = m.replace(".base_model.model.", ".")
    m = m.replace(".default.", ".")
    if m.endswith(".default"):
        m = m[: -len(".default")].rstrip(".")
    m = m.replace(".to_out.0.", ".to_out.")
    if m.endswith(".to_out.0"):
        m = m[: -len(".to_out.0")]
    m = m.replace(".ff.net.0.proj.", ".ff.net_0_proj.")
    m = m.replace(".ff.net.2.", ".ff.net_2.")
    m = m.replace(".ff_context.net.0.proj.", ".ff_context.net_0_proj.")
    m = m.replace(".ff_context.net.2.", ".ff_context.net_2.")
    # BFL alias names → DanQing ``ff.net_*_proj``
    m = re.sub(r"\.ff\.linear1$", ".ff.net_0_proj", m)
    m = re.sub(r"\.ff\.linear2$", ".ff.net_2", m)
    m = re.sub(r"\.ff_context\.linear1$", ".ff_context.net_0_proj", m)
    m = re.sub(r"\.ff_context\.linear2$", ".ff_context.net_2", m)
    if m.endswith(".weight"):
        m = m[: -len(".weight")].rstrip(".")
    elif m.endswith(".bias"):
        m = m[: -len(".bias")].rstrip(".")
    return m.strip(".").strip()


def _flux1_lora_split_qkv(
    module: str,
    down: Any,
    up: Any,
    alpha: float,
    *,
    q_suffix: str,
    k_suffix: str,
    v_suffix: str,
) -> dict[str, tuple[Any, Any, float]]:
    """Split fused QKV LoRA into three DanQing attention linear targets."""
    out: dict[str, tuple[Any, Any, float]] = {}
    n = 3
    if int(up.shape[0]) % n != 0 or int(down.shape[0]) % n != 0:
        return out
    up_chunk = int(up.shape[0]) // n
    down_chunk = int(down.shape[0]) // n
    for idx, suffix in enumerate((q_suffix, k_suffix, v_suffix)):
        tgt = remap_flux1_lora_module_prefix(f"{module}.{suffix}")
        if not tgt:
            continue
        u = up[idx * up_chunk : (idx + 1) * up_chunk]
        d = down[idx * down_chunk : (idx + 1) * down_chunk]
        out[tgt] = (d, u, alpha)
    return out


def remap_flux1_lora_keys(lora_weights: dict) -> dict[str, tuple[Any, Any, float]]:
    """Group LoRA tensors and map module names to DanQing Flux.1 ``_param_map`` prefixes."""
    from backend.engine.common.bundle.weights import _lora_key_to_module

    # External LoRAs commonly omit alpha; Kohya/diffusers then mean alpha=rank.
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
        tgt = remap_flux1_lora_module_prefix(base)
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
        if "lora_down" in kl or ".lora_a." in kl or kl.endswith(".lora_a.weight"):
            groups[module]["down"] = tensor
        elif "lora_up" in kl or ".lora_b." in kl or kl.endswith(".lora_b.weight"):
            groups[module]["up"] = tensor

    remapped: dict[str, tuple[Any, Any, float]] = {}
    for module, parts in groups.items():
        if "down" not in parts or "up" not in parts:
            raise RuntimeError(
                f"Flux1 LoRA module {module!r} is missing a down/up factor; "
                "refusing to apply a partial adapter."
            )
        down, up = parts["down"], parts["up"]
        fallback_alpha = default_alpha if default_alpha is not None else float(down.shape[0])
        alpha = float(
            alphas_by_tgt.get(remap_flux1_lora_module_prefix(module), fallback_alpha)
        )

        mod = module.replace("diffusion_model.", "")
        m_img = re.match(r"double_blocks\.(\d+)\.img_attn\.qkv", mod)
        if m_img:
            remapped.update(
                _flux1_lora_split_qkv(
                    f"transformer_blocks.{m_img.group(1)}.attn",
                    down,
                    up,
                    alpha,
                    q_suffix="to_q",
                    k_suffix="to_k",
                    v_suffix="to_v",
                )
            )
            continue
        m_txt = re.match(r"double_blocks\.(\d+)\.txt_attn\.qkv", mod)
        if m_txt:
            remapped.update(
                _flux1_lora_split_qkv(
                    f"transformer_blocks.{m_txt.group(1)}.attn",
                    down,
                    up,
                    alpha,
                    q_suffix="add_q_proj",
                    k_suffix="add_k_proj",
                    v_suffix="add_v_proj",
                )
            )
            continue
        m_unet_img = re.match(r"double_blocks_(\d+)_img_attn_qkv", mod)
        if m_unet_img:
            remapped.update(
                _flux1_lora_split_qkv(
                    f"transformer_blocks.{m_unet_img.group(1)}.attn",
                    down,
                    up,
                    alpha,
                    q_suffix="to_q",
                    k_suffix="to_k",
                    v_suffix="to_v",
                )
            )
            continue
        m_unet_txt = re.match(r"double_blocks_(\d+)_txt_attn_qkv", mod)
        if m_unet_txt:
            remapped.update(
                _flux1_lora_split_qkv(
                    f"transformer_blocks.{m_unet_txt.group(1)}.attn",
                    down,
                    up,
                    alpha,
                    q_suffix="add_q_proj",
                    k_suffix="add_k_proj",
                    v_suffix="add_v_proj",
                )
            )
            continue

        m = re.match(r"double_blocks\.(\d+)\.img_attn\.proj", mod)
        if m:
            tgt = remap_flux1_lora_module_prefix(f"transformer_blocks.{m.group(1)}.attn.to_out")
            remapped[tgt] = (down, up, alpha)
            continue
        m = re.match(r"double_blocks\.(\d+)\.txt_attn\.proj", mod)
        if m:
            tgt = remap_flux1_lora_module_prefix(f"transformer_blocks.{m.group(1)}.attn.to_add_out")
            remapped[tgt] = (down, up, alpha)
            continue
        m = re.match(r"double_blocks\.(\d+)\.img_mlp\.0", mod)
        if m:
            tgt = remap_flux1_lora_module_prefix(f"transformer_blocks.{m.group(1)}.ff.net_0_proj")
            remapped[tgt] = (down, up, alpha)
            continue
        m = re.match(r"double_blocks\.(\d+)\.img_mlp\.2", mod)
        if m:
            tgt = remap_flux1_lora_module_prefix(f"transformer_blocks.{m.group(1)}.ff.net_2")
            remapped[tgt] = (down, up, alpha)
            continue
        m = re.match(r"double_blocks\.(\d+)\.txt_mlp\.0", mod)
        if m:
            tgt = remap_flux1_lora_module_prefix(f"transformer_blocks.{m.group(1)}.ff_context.net_0_proj")
            remapped[tgt] = (down, up, alpha)
            continue
        m = re.match(r"double_blocks\.(\d+)\.txt_mlp\.2", mod)
        if m:
            tgt = remap_flux1_lora_module_prefix(f"transformer_blocks.{m.group(1)}.ff_context.net_2")
            remapped[tgt] = (down, up, alpha)
            continue

        tgt = remap_flux1_lora_module_prefix(module)
        if not tgt:
            continue
        if "norm" in tgt and "modulation" not in tgt and "linear" not in tgt:
            continue
        remapped[tgt] = (down, up, alpha)

    return remapped
