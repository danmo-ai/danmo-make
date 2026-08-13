"""LTX-2.5 weight remapping — split 22B transformer checkpoint → MLX module names.

The official ``ltx-2.5-22b-{dev,distilled}-transformer-bf16.safetensors`` file
uses Comfy-style keys under ``model.diffusion_model.``. The upstream module
tree (AdaLayerNormSingle.emb.timestep_embedder.*, FeedForward.net.{0,2},
Attention.to_out.0) is normalized to the in-repo ``LTX25Model`` layout.
"""
from __future__ import annotations

import re
from typing import Any


def normalize_ltx25_keys(weights: dict[str, Any]) -> dict[str, Any]:
    """Strip ``model.diffusion_model.`` and normalize 2.5 key tails."""
    out: dict[str, Any] = {}
    prefix = "model.diffusion_model."
    for key, tensor in weights.items():
        nk = key[len(prefix):] if key.startswith(prefix) else key
        nk = nk.replace(".emb.timestep_embedder.", ".emb.")
        # FeedForward: ff.net.0.proj → ff.proj_in ; ff.net.2 → ff.proj_out
        nk = nk.replace(".ff.net.0.proj.", ".ff.proj_in.")
        nk = nk.replace(".ff.net.2.", ".ff.proj_out.")
        nk = nk.replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.")
        nk = nk.replace(".audio_ff.net.2.", ".audio_ff.proj_out.")
        # Attention: to_out.0 → to_out
        nk = re.sub(r"(\.(?:attn1|attn2|audio_attn1|audio_attn2|audio_to_video_attn|video_to_audio_attn)\.)to_out\.0\.(weight|bias)$", r"\1to_out.\2", nk)
        out[nk] = tensor
    return out


def remap_ltx25_weights(weights: dict[str, Any]) -> dict[str, Any]:
    """Map the 2.5 transformer checkpoint keys to ``LTX25Transformer._param_map`` names."""
    return normalize_ltx25_keys(weights)


def is_ltx25_dit_checkpoint(weights: dict[str, Any]) -> bool:
    """True when the weight dict looks like an LTX-2.5 transformer checkpoint."""
    for key in weights:
        if "patchify_proj" in key or "audio_patchify_proj" in key:
            return True
        if key.startswith("model.diffusion_model.transformer_blocks."):
            return True
        if ".audio_to_video_attn." in key:
            return True
    return False
