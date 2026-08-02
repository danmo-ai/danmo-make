"""Standalone MLX model wrapper for HiDream-O1-Image."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.fc1 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)

    @staticmethod
    def timestep_embedding(t: mx.array, dim: int, max_period: float = 10000.0) -> mx.array:
        half = dim // 2
        freqs = mx.exp(-math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half)
        args = t[:, None].astype(mx.float32) * freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
        return emb

    def __call__(self, t: mx.array) -> mx.array:
        t_freq = self.timestep_embedding(t * 1000.0, self.frequency_embedding_size)
        return self.fc2(nn.silu(self.fc1(t_freq.astype(self.fc1.weight.dtype))))


class BottleneckPatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 32, in_chans: int = 3,
                 pca_dim: int = 1024, embed_dim: int = 4096):
        super().__init__()
        self.proj1 = nn.Linear(patch_size * patch_size * in_chans, pca_dim, bias=False)
        self.proj2 = nn.Linear(pca_dim, embed_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj2(self.proj1(x))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int = 32, out_channels: int = 3):
        super().__init__()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


CUSTOM_HEAD_KEY_MAP = {
    "model.t_embedder1.mlp.0.weight":   "t_embedder1.fc1.weight",
    "model.t_embedder1.mlp.0.bias":     "t_embedder1.fc1.bias",
    "model.t_embedder1.mlp.2.weight":   "t_embedder1.fc2.weight",
    "model.t_embedder1.mlp.2.bias":     "t_embedder1.fc2.bias",
    "model.x_embedder.proj1.weight":    "x_embedder.proj1.weight",
    "model.x_embedder.proj2.weight":    "x_embedder.proj2.weight",
    "model.x_embedder.proj2.bias":      "x_embedder.proj2.bias",
    "model.final_layer2.linear.weight": "final_layer2.linear.weight",
    "model.final_layer2.linear.bias":   "final_layer2.linear.bias",
}


@dataclass
class HiDreamConfig:
    hidden_size: int = 4096
    patch_size: int = 32
    in_channels: int = 3
    bottleneck_dim: int = 1024
    tms_token_id: int = 151673
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652


def build_model(cfg: HiDreamConfig, mlx_vlm_qwen3_vl_model):
    class HiDream(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = mlx_vlm_qwen3_vl_model.vision_tower
            self.language_model = mlx_vlm_qwen3_vl_model.language_model
            self.t_embedder1 = TimestepEmbedder(cfg.hidden_size)
            self.x_embedder = BottleneckPatchEmbed(
                patch_size=cfg.patch_size, in_chans=cfg.in_channels,
                pca_dim=cfg.bottleneck_dim, embed_dim=cfg.hidden_size,
            )
            self.final_layer2 = FinalLayer(
                hidden_size=cfg.hidden_size,
                patch_size=cfg.patch_size,
                out_channels=cfg.in_channels,
            )

    return HiDream()


def precompute_text_embeds_with_vision(model, cfg, input_ids, pixel_values=None, image_grid_thw=None):
    """Compute text embeddings + (in edit mode) inject vision features at image_token
    positions. Returns embeds [B, S_text, hidden]. Call once before the denoising
    loop — output is constant across timesteps.
    """
    embed_tokens = model.language_model.model.embed_tokens
    inputs_embeds = embed_tokens(input_ids)

    if pixel_values is None or image_grid_thw is None:
        return inputs_embeds

    vt_out = model.visual(pixel_values, image_grid_thw)
    image_features = vt_out[0] if isinstance(vt_out, tuple) else vt_out
    if isinstance(image_features, (list, tuple)):
        image_features = mx.concatenate(image_features, axis=0)

    # Build a [B, S, H] tensor that has image_features at image_token positions
    # and inputs_embeds everywhere else, via mx.where on a broadcast mask.
    ids_np = np.asarray(input_ids)
    img_positions = np.where(ids_np[0] == cfg.image_token_id)[0]
    if img_positions.shape[0] != image_features.shape[0]:
        raise RuntimeError(
            f"image_features {image_features.shape[0]} != "
            f"image_token_id positions {img_positions.shape[0]} (input_ids was: {ids_np.shape})"
        )

    B, S, H = inputs_embeds.shape
    # Build aligned-to-S features: zero everywhere except at image positions.
    aligned = np.zeros((B, S, H), dtype=np.float32)
    aligned[0, img_positions] = np.asarray(image_features.astype(mx.float32))
    aligned_mx = mx.array(aligned).astype(inputs_embeds.dtype)

    # Mask: 1 at image positions, 0 elsewhere
    mask_2d = (ids_np == cfg.image_token_id).astype(np.bool_)
    mask_3d = np.broadcast_to(mask_2d[..., None], (B, S, H))
    mask_mx = mx.array(mask_3d.copy())

    return mx.where(mask_mx, aligned_mx, inputs_embeds)


def forward_generation(model, cfg, inputs_embeds_with_vision, position_ids, vinputs, timestep,
                       input_ids, token_types, attention_mask_4d):
    """Per-step forward. Takes the precomputed text+vision inputs_embeds, the
    fresh-noise vinputs, and the timestep. Returns x_pred [B, S_total, patch_dim].

    Signature change vs the T2I-only version: pixel_values/image_grid_thw moved
    out (call precompute_text_embeds_with_vision once before the loop). input_ids
    is still needed inside because we look up tms_token positions for t_emb scatter.
    """
    inputs_embeds = inputs_embeds_with_vision

    t_emb = model.t_embedder1(timestep)
    tms_mask = (input_ids == cfg.tms_token_id)
    tms_mask_3d = mx.broadcast_to(tms_mask[..., None], inputs_embeds.shape)
    t_emb_expanded = mx.broadcast_to(t_emb[:, None, :], inputs_embeds.shape)
    inputs_embeds = mx.where(tms_mask_3d, t_emb_expanded, inputs_embeds)

    vinputs_embedded = model.x_embedder(vinputs).astype(inputs_embeds.dtype)
    inputs_embeds = mx.concatenate([inputs_embeds, vinputs_embedded], axis=1)

    text_model = model.language_model.model
    # mlx-vlm Qwen3VLModel.__call__ accepts (inputs, inputs_embeds, mask, cache, position_ids, ...).
    # Pass our 4D additive mask directly; it bypasses the internal causal mask.
    # `inputs` is required positionally but ignored when inputs_embeds is set
    # in mlx-vlm's implementation — pass a placeholder of correct shape.
    placeholder = mx.zeros(inputs_embeds.shape[:2], dtype=mx.int32)
    h = text_model(
        placeholder,
        inputs_embeds=inputs_embeds,
        mask=attention_mask_4d,
        cache=None,
        position_ids=position_ids,
    )
    # Apply final norm. mlx-vlm's Qwen3VLModel applies it internally and returns hidden_states.
    x_pred = model.final_layer2(h)
    return x_pred
