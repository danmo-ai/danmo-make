"""
ACE-Step DiT decoder — pure MLX implementation for Apple Silicon.

Mirrors ``acestep/models/mlx/dit_model.py`` with adaptations to the DanQing
``TransformerBase`` contract.  All tensor operations stay in ``mlx.nn`` / ``mlx.core``
(importing ``mlx`` is allowed in ``*_mlx.py`` files per the dual-platform rules).

Architecture
------------
32 transformer layers, alternating sliding-window self-attention and full
self-attention, each followed by cross-attention to encoder hidden states and
a SwiGLU MLP.  Modulation is via AdaLN (scale_shift_table + timestep projection).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.model.base import TransformerBase
from backend.engine.common.ops.attention import (
    build_window_with_padding_bias,
    repeat_kv_heads_mx,
    scaled_dot_product_attention_bhsd_mx,
    rotate_half,
)
from backend.engine.common.ops.embeddings import sinusoidal_timestep_proj
from backend.engine.common.ops.norm import apply_scale_shift, unpack_modulation_6table
from backend.engine.runtime.mlx import MLXContext
from backend.engine.common.codecs.text_encoders.qwen3_mlx import MlxSwiGLUMLP, MlxTimestepEmbeddingMLP

logger = logging.getLogger(__name__)

_MLX_CTX = MLXContext()

# ---------------------------------------------------------------------------
# Rotary helpers
# ---------------------------------------------------------------------------

def _apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array,
) -> Tuple[mx.array, mx.array]:
    q_embed = (q * cos) + (rotate_half(_MLX_CTX, q) * sin)
    k_embed = (k * cos) + (rotate_half(_MLX_CTX, k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class _AceStepRotaryEmbedding(nn.Module):
    """Pre-computes cos/sin tables for RoPE."""

    def __init__(self, head_dim: int, max_len: int = 32768, base: float = 1_000_000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_len = max_len
        self.base = base

        inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
        positions = mx.arange(max_len).astype(mx.float32)
        freqs = positions[:, None] * inv_freq[None, :]
        freqs = mx.concatenate([freqs, freqs], axis=-1)
        self._cos = mx.cos(freqs)
        self._sin = mx.sin(freqs)

    def __call__(self, seq_len: int) -> Tuple[mx.array, mx.array]:
        cos = self._cos[:seq_len][None, None, :, :]
        sin = self._sin[:seq_len][None, None, :, :]
        return cos, sin


# ---------------------------------------------------------------------------
# Cross-attention KV cache
# ---------------------------------------------------------------------------

class _CrossAttentionCache:
    """KV cache for cross-attention layers.

    Cross-attention K/V are computed once on the first diffusion step and
    re-used for all subsequent steps.
    """

    def __init__(self):
        self._keys: Dict[int, mx.array] = {}
        self._values: Dict[int, mx.array] = {}
        self._updated: set = set()

    def update(self, key: mx.array, value: mx.array, layer_idx: int):
        self._keys[layer_idx] = key
        self._values[layer_idx] = value
        self._updated.add(layer_idx)

    def is_updated(self, layer_idx: int) -> bool:
        return layer_idx in self._updated

    def get(self, layer_idx: int) -> Tuple[mx.array, mx.array]:
        return self._keys[layer_idx], self._values[layer_idx]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class _Attention(nn.Module):
    """Multi-head attention with QK-RMSNorm.  Supports self-attention (with RoPE)
    and cross-attention (with optional KV caching).
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        attention_bias: bool,
        layer_idx: int,
        is_cross_attention: bool = False,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.n_rep = num_attention_heads // num_key_value_heads
        self.scale = head_dim ** -0.5
        self.layer_idx = layer_idx
        self.is_cross_attention = is_cross_attention
        self.sliding_window = sliding_window

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

        self.q_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        position_cos_sin: Optional[Tuple[mx.array, mx.array]] = None,
        attention_mask: Optional[mx.array] = None,
        encoder_hidden_states: Optional[mx.array] = None,
        cache: Optional[_CrossAttentionCache] = None,
        use_cache: bool = False,
    ) -> mx.array:
        B, L, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        q = self.q_norm(q.reshape(B, L, self.num_heads, self.head_dim))
        q = q.transpose(0, 2, 1, 3)

        if self.is_cross_attention and encoder_hidden_states is not None:
            if cache is not None and cache.is_updated(self.layer_idx):
                k, v = cache.get(self.layer_idx)
            else:
                enc_L = encoder_hidden_states.shape[1]
                k = self.k_proj(encoder_hidden_states)
                k = self.k_norm(k.reshape(B, enc_L, self.num_kv_heads, self.head_dim))
                k = k.transpose(0, 2, 1, 3)
                v = self.v_proj(encoder_hidden_states).reshape(
                    B, enc_L, self.num_kv_heads, self.head_dim,
                ).transpose(0, 2, 1, 3)
                if cache is not None and use_cache:
                    cache.update(k, v, self.layer_idx)
        else:
            k = self.k_proj(hidden_states)
            k = self.k_norm(k.reshape(B, L, self.num_kv_heads, self.head_dim))
            k = k.transpose(0, 2, 1, 3)
            v = self.v_proj(hidden_states).reshape(
                B, L, self.num_kv_heads, self.head_dim,
            ).transpose(0, 2, 1, 3)

            if position_cos_sin is not None:
                cos, sin = position_cos_sin
                q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv_heads_mx(mx, k, self.n_rep)
        v = repeat_kv_heads_mx(mx, v, self.n_rep)

        attn_out = scaled_dot_product_attention_bhsd_mx(
            mx,
            q,
            k,
            v,
            scale=self.scale,
            mask=attention_mask,
        )
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(attn_out)


# ---------------------------------------------------------------------------
# DiT layer
# ---------------------------------------------------------------------------

class _DiTLayer(nn.Module):
    """Single DiT layer: self-attn (AdaLN) → cross-attn → MLP (AdaLN)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        attention_bias: bool,
        layer_idx: int,
        layer_type: str,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        self.layer_type = layer_type
        sw = sliding_window if layer_type == "sliding_attention" else None

        self.self_attn_norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = _Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            layer_idx=layer_idx,
            is_cross_attention=False,
            sliding_window=sw,
        )

        self.cross_attn_norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.cross_attn = _Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            layer_idx=layer_idx,
            is_cross_attention=True,
        )

        self.mlp_norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = MlxSwiGLUMLP(hidden_size, intermediate_size)

        self.scale_shift_table = mx.zeros((1, 6, hidden_size))

    def __call__(
        self,
        hidden_states: mx.array,
        position_cos_sin: Tuple[mx.array, mx.array],
        temb: mx.array,
        self_attn_mask: Optional[mx.array],
        encoder_hidden_states: Optional[mx.array],
        encoder_attention_mask: Optional[mx.array],
        cache: Optional[_CrossAttentionCache] = None,
        use_cache: bool = False,
    ) -> mx.array:
        modulation = self.scale_shift_table + temb
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            unpack_modulation_6table(modulation)
        )
        # Unpack yields [B, D]; expand to [B, 1, D] so CFG (B>1) broadcasts with [B, T, D].
        # With B=1, [1, D] happened to broadcast against [1, T, D]; B=2 does not.
        shift_msa = mx.expand_dims(shift_msa, 1)
        scale_msa = mx.expand_dims(scale_msa, 1)
        gate_msa = mx.expand_dims(gate_msa, 1)
        c_shift_msa = mx.expand_dims(c_shift_msa, 1)
        c_scale_msa = mx.expand_dims(c_scale_msa, 1)
        c_gate_msa = mx.expand_dims(c_gate_msa, 1)

        # 1) Self-attention
        normed = self.self_attn_norm(hidden_states)
        normed = apply_scale_shift(normed, scale_msa, shift_msa, add_one=True)
        attn_out = self.self_attn(
            normed,
            position_cos_sin=position_cos_sin,
            attention_mask=self_attn_mask,
        )
        hidden_states = hidden_states + attn_out * gate_msa

        # 2) Cross-attention
        normed = self.cross_attn_norm(hidden_states)
        cross_out = self.cross_attn(
            normed,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            cache=cache,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + cross_out

        # 3) MLP
        normed = self.mlp_norm(hidden_states)
        normed = apply_scale_shift(normed, c_scale_msa, c_shift_msa, add_one=True)
        ff_out = self.mlp(normed)
        hidden_states = hidden_states + ff_out * c_gate_msa

        return hidden_states


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class _TimestepEmbedding(MlxTimestepEmbeddingMLP):
    """Sinusoidal timestep embedding → MLP → (temb, 6-way projection)."""

    def __init__(self, in_channels: int = 256, time_embed_dim: int = 2048, scale: float = 1000.0):
        super().__init__(in_channels, time_embed_dim)
        self.in_channels = in_channels
        self.scale = scale
        self.act2 = nn.SiLU()
        self.time_proj = nn.Linear(time_embed_dim, time_embed_dim * 6, bias=True)

    def __call__(self, t: mx.array) -> Tuple[mx.array, mx.array]:
        t_freq = sinusoidal_timestep_proj(
            _MLX_CTX, t, self.in_channels, sin_first=False, scale=self.scale
        ).astype(t.dtype)
        temb = super().__call__(t_freq.astype(t.dtype))
        proj = self.time_proj(self.act2(temb))
        timestep_proj = proj.reshape(proj.shape[0], 6, -1)
        return temb, timestep_proj


# ---------------------------------------------------------------------------
# Full DiT decoder
# ---------------------------------------------------------------------------

class AceStepDiTMLX(nn.Module):
    """Native MLX DiT decoder for ACE-Step audio generation.

    Mirrors the PyTorch ``AceStepDiTModel``:
        - Conv1d patch embedding of input
        - Dual timestep conditioning
        - N DiT layers (self-attn / cross-attn / MLP) with AdaLN
        - ConvTranspose1d output projection
        - Adaptive output layer norm

    *Not* a ``TransformerBase`` — the ``transformer.py`` wrapper adds the
    DanQing base-class interface.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        in_channels: int = 192,
        audio_acoustic_hidden_dim: int = 64,
        patch_size: int = 2,
        sliding_window: int = 128,
        layer_types: Optional[list] = None,
        rope_theta: float = 1_000_000.0,
        max_position_embeddings: int = 32768,
        encoder_hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        # XL: condition tokens are encoder_hidden_size (2048), DiT is hidden_size (2560).
        condition_dim = int(encoder_hidden_size) if encoder_hidden_size is not None else int(hidden_size)

        if layer_types is None:
            layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention"
                for i in range(num_hidden_layers)
            ]

        self.rotary_emb = _AceStepRotaryEmbedding(
            head_dim, max_len=max_position_embeddings, base=rope_theta,
        )

        self.proj_in = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        self.time_embed = _TimestepEmbedding(time_embed_dim=hidden_size)
        self.time_embed_r = _TimestepEmbedding(time_embed_dim=hidden_size)

        self.condition_embedder = nn.Linear(condition_dim, hidden_size, bias=True)

        self.layers = [
            _DiTLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                attention_bias=attention_bias,
                layer_idx=i,
                layer_type=layer_types[i],
                sliding_window=sliding_window,
            )
            for i in range(num_hidden_layers)
        ]

        self.norm_out = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.proj_out = nn.ConvTranspose1d(
            in_channels=hidden_size,
            out_channels=audio_acoustic_hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        self.scale_shift_table = mx.zeros((1, 2, hidden_size))

        self._sliding_masks: Dict[tuple[int, str], mx.array] = {}
        self._sliding_window = sliding_window
        self._layer_types = layer_types

    def _get_sliding_mask(self, seq_len: int, dtype: mx.Dtype) -> mx.array:
        key = (int(seq_len), str(dtype))
        if key not in self._sliding_masks:
            self._sliding_masks[key] = build_window_with_padding_bias(
                mx,
                seq_len,
                dtype,
                attention_mask=None,
                sliding_window=self._sliding_window,
                neg_value=-1e9,
            )
        return self._sliding_masks[key]

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        timestep_r: mx.array,
        encoder_hidden_states: mx.array,
        context_latents: mx.array,
        cache: Optional[_CrossAttentionCache] = None,
        use_cache: bool = True,
    ) -> Tuple[mx.array, Optional[_CrossAttentionCache]]:
        temb_t, proj_t = self.time_embed(timestep)
        temb_r, proj_r = self.time_embed_r(timestep - timestep_r)
        temb = temb_t + temb_r
        timestep_proj = proj_t + proj_r

        hidden_states = mx.concatenate([context_latents, hidden_states], axis=-1)

        original_seq_len = hidden_states.shape[1]
        pad_length = 0
        if hidden_states.shape[1] % self.patch_size != 0:
            pad_length = self.patch_size - (hidden_states.shape[1] % self.patch_size)
            padding = mx.zeros(
                (hidden_states.shape[0], pad_length, hidden_states.shape[2]),
                dtype=hidden_states.dtype,
            )
            hidden_states = mx.concatenate([hidden_states, padding], axis=1)

        hidden_states = self.proj_in(hidden_states)
        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)

        seq_len = hidden_states.shape[1]
        dtype = hidden_states.dtype

        cos, sin = self.rotary_emb(seq_len)

        has_sliding = any(lt == "sliding_attention" for lt in self._layer_types)
        sliding_mask = self._get_sliding_mask(seq_len, dtype) if has_sliding else None

        for layer in self.layers:
            self_attn_mask = sliding_mask if layer.layer_type == "sliding_attention" else None
            hidden_states = layer(
                hidden_states,
                position_cos_sin=(cos, sin),
                temb=timestep_proj,
                self_attn_mask=self_attn_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=None,
                cache=cache,
                use_cache=use_cache,
            )

        shift, scale = mx.split(
            self.scale_shift_table + mx.expand_dims(temb, axis=1), 2, axis=1,
        )
        hidden_states = apply_scale_shift(self.norm_out(hidden_states), scale, shift, add_one=True)

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states[:, :original_seq_len, :]

        return hidden_states, cache


def collect_ace_step_dit_param_map(module: nn.Module) -> dict[str, mx.array]:
    """Flatten ``AceStepDiTMLX`` leaves for ``TransformerBase.load_weights``."""
    flat: dict[str, mx.array] = {}
    _collect_ace_step_dit_params(module, "", flat)
    return flat


def _collect_ace_step_dit_params(obj: Any, prefix: str, result: dict[str, mx.array]) -> None:
    if isinstance(obj, mx.array):
        if prefix:
            result[prefix] = obj
        return

    if isinstance(obj, nn.Module):
        params = obj.parameters()
        if isinstance(params, dict) and params:
            for name, value in params.items():
                key = f"{prefix}.{name}" if prefix else str(name)
                _collect_ace_step_dit_params(value, key, result)
            return

    if isinstance(obj, dict):
        for name, value in obj.items():
            key = f"{prefix}.{name}" if prefix else str(name)
            _collect_ace_step_dit_params(value, key, result)
        return

    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            key = f"{prefix}.{idx}" if prefix else str(idx)
            _collect_ace_step_dit_params(value, key, result)
