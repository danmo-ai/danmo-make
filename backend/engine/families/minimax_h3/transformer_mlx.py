"""MLX port of Diffusers ``MiniMaxH3Transformer3DModel``.

Parameter names mirror the Diffusers checkpoint (``proj_in``, ``adaln_proj.linear``,
``ff.net.0.proj``, …) so safetensors load with only Conv/layout remaps at the caller.
"""
from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.ops.attention import scaled_dot_product_attention_bhsd_mx
from backend.engine.common.ops.embeddings import sinusoidal_timestep_proj
from backend.engine.runtime.mlx import MLXContext

_MLX_CTX = MLXContext()

# Per-row modality tags address AdaLN: 0=video, 1=text, 2=audio (padding uses -1).
MINIMAX_H3_MODALITY_NUM = 3


def _apply_rotary_emb(hidden_states: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate leading ``rotary_dim`` channels; pass the rest through.

    ``hidden_states``: ``(B, S, H, D)``; ``cos``/``sin``: ``(S, rotary_dim)``.
    """
    rotary_dim = int(cos.shape[-1])
    rotary = hidden_states[..., :rotary_dim]
    passthrough = hidden_states[..., rotary_dim:]
    cos_b = cos[None, :, None, :].astype(hidden_states.dtype)
    sin_b = sin[None, :, None, :].astype(hidden_states.dtype)
    x1, x2 = mx.split(rotary, 2, axis=-1)
    rotated = mx.concatenate((-x2, x1), axis=-1)
    rotary = rotary * cos_b + rotated * sin_b
    return mx.concatenate((rotary, passthrough), axis=-1)


def _index_select_axis1(x: mx.array, indices: mx.array) -> mx.array:
    return mx.take(x, indices.astype(mx.int32), axis=1)


def _index_copy_axis1(buf: mx.array, indices: mx.array, values: mx.array) -> mx.array:
    """Scatter ``values`` ``[B, N, C]`` into ``buf`` ``[B, S, C]`` at ``indices`` ``[N]``."""
    b, _s, c = buf.shape
    n = int(indices.shape[0])
    idx = mx.broadcast_to(indices.astype(mx.int32).reshape(1, n, 1), (b, n, c))
    return mx.put_along_axis(buf, idx, values, axis=1)


def _index_select_rows(table: mx.array, indices: mx.array) -> mx.array:
    """``table[indices]`` for ``table`` ``[R, C]`` and ``indices`` ``[S]`` → ``[S, C]``."""
    return mx.take(table, indices.astype(mx.int32), axis=0)


class MiniMaxH3Timesteps(nn.Module):
    """Diffusers ``Timesteps`` — sinusoidal projection of unscaled ``[0, 1]`` timesteps."""

    def __init__(self, num_channels: int = 256, flip_sin_to_cos: bool = True, downscale_freq_shift: float = 0.0):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def __call__(self, timesteps: mx.array) -> mx.array:
        return sinusoidal_timestep_proj(
            _MLX_CTX,
            timesteps,
            self.num_channels,
            sin_first=True,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )


class MiniMaxH3TimestepEmbedding(nn.Module):
    """Diffusers ``TimestepEmbedding`` with optional ``out_dim`` (keys ``linear_1`` / ``linear_2``)."""

    def __init__(self, in_channels: int, time_embed_dim: int, out_dim: int | None = None):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, out, bias=True)

    def __call__(self, sample: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(sample)))


class MiniMaxH3RotaryPosEmbed(nn.Module):
    """3-axis RoPE over packed ``(t, h, w)`` coordinates."""

    def __init__(self, rope_freq_dim: int = 16, rope_theta: float = 10000.0):
        super().__init__()
        self.rope_freq_dim = rope_freq_dim
        inv_freq = 1.0 / (
            rope_theta
            ** (mx.arange(0, 2 * rope_freq_dim, 2, dtype=mx.float32) / (2 * rope_freq_dim))
        )
        self.inv_freq = inv_freq

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        # position_ids: (S, 3) -> cos/sin: (S, 2 * 3 * rope_freq_dim)
        position_ids = position_ids.astype(mx.float32)
        freqs = position_ids[:, :, None] * self.inv_freq.reshape(1, 1, -1)
        freqs_t, freqs_h, freqs_w = freqs[:, 0, :], freqs[:, 1, :], freqs[:, 2, :]
        freqs = mx.concatenate((freqs_t, freqs_h, freqs_w), axis=-1)
        freqs = mx.concatenate((freqs, freqs), axis=-1)
        return mx.cos(freqs), mx.sin(freqs)


class MiniMaxH3AdaLayerNormModulation(nn.Module):
    """Block-level AdaLN projection (checkpoint key ``adaln_proj.linear``)."""

    def __init__(self, time_embed_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(time_embed_dim, 6 * hidden_size * MINIMAX_H3_MODALITY_NUM, bias=True)

    def __call__(self, temb: mx.array) -> tuple[mx.array, ...]:
        w_dtype = self.linear.weight.dtype
        temb = self.linear(nn.silu(temb).astype(w_dtype))
        temb = temb.reshape(-1, 6 * self.hidden_size)
        return tuple(mx.split(temb, 6, axis=-1))


class MiniMaxH3AdaLayerNormOut(nn.Module):
    """Final packed-sequence norm (``norm`` + ``linear`` → shift then scale)."""

    def __init__(self, hidden_size: int, time_embed_dim: int, eps: float):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(time_embed_dim, 2 * hidden_size, bias=True)

    def __call__(self, hidden_states: mx.array, temb: mx.array, timestep_indices: mx.array) -> mx.array:
        w_dtype = self.linear.weight.dtype
        shift, scale = mx.split(self.linear(nn.silu(temb).astype(w_dtype)), 2, axis=-1)
        hidden_states = self.norm(hidden_states)
        scale_rows = _index_select_rows(scale, timestep_indices)
        shift_rows = _index_select_rows(shift, timestep_indices)
        return hidden_states * (1.0 + scale_rows) + shift_rows


class _SwiGLU(nn.Module):
    """Diffusers ``SwiGLU`` — fused ``proj`` to ``2 * dim_out`` (value || gate)."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.proj(hidden_states)
        value, gate = mx.split(hidden_states, 2, axis=-1)
        return value * nn.silu(gate)


class MiniMaxH3FeedForward(nn.Module):
    """Diffusers ``FeedForward(..., activation_fn='swiglu')`` — keys ``net.0.proj`` / ``net.2``."""

    def __init__(self, dim: int, inner_dim: int, bias: bool = False):
        super().__init__()
        self.net = [
            _SwiGLU(dim, inner_dim, bias=bias),
            nn.Identity(),
            nn.Linear(inner_dim, dim, bias=bias),
        ]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class MiniMaxH3Attention(nn.Module):
    """Full self-attention with RMSNorm Q/K (no cross-attention in MiniMax-H3)."""

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        dim_head: int,
        qk_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head**-0.5
        self.to_q = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.to_k = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.to_v = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=qk_norm_eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=qk_norm_eps)
        self.to_out = [nn.Linear(self.inner_dim, hidden_size, bias=False), nn.Identity()]

    def __call__(
        self,
        hidden_states: mx.array,
        rotary_emb: tuple[mx.array, mx.array] | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        b, s, _ = hidden_states.shape
        query = self.to_q(hidden_states).reshape(b, s, self.heads, self.head_dim)
        key = self.to_k(hidden_states).reshape(b, s, self.heads, self.head_dim)
        value = self.to_v(hidden_states).reshape(b, s, self.heads, self.head_dim)
        query = self.norm_q(query)
        key = self.norm_k(key)
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)
        # SDPA wants [B, H, S, D]
        q = mx.transpose(query, (0, 2, 1, 3))
        k = mx.transpose(key, (0, 2, 1, 3))
        v = mx.transpose(value, (0, 2, 1, 3))
        out = scaled_dot_product_attention_bhsd_mx(
            mx, q, k, v, scale=self.scale, mask=attention_mask,
        )
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(b, s, self.inner_dim)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out


class MiniMaxH3TokenRefinerBlock(nn.Module):
    """Pre-norm transformer block for the text stream (no AdaLN / RoPE)."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size=hidden_size,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm_eps=qk_norm_eps,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = MiniMaxH3FeedForward(hidden_size, inner_dim=ffn_dim, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        num_layers: int,
        norm_eps: float,
        qk_norm_eps: float,
        final_norm_eps: float,
    ):
        super().__init__()
        self.refiner_blocks = [
            MiniMaxH3TokenRefinerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                ffn_dim=ffn_dim,
                norm_eps=norm_eps,
                qk_norm_eps=qk_norm_eps,
            )
            for _ in range(num_layers)
        ]
        self.final_norm = nn.RMSNorm(hidden_size, eps=final_norm_eps)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)


class MiniMaxH3TransformerBlock(nn.Module):
    """AdaLN-modulated pre-norm self-attention + SwiGLU FFN."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        time_embed_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size=hidden_size,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm_eps=qk_norm_eps,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = MiniMaxH3FeedForward(hidden_size, inner_dim=ffn_dim, bias=False)
        self.adaln_proj = MiniMaxH3AdaLayerNormModulation(
            time_embed_dim=time_embed_dim, hidden_size=hidden_size,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        temb: mx.array,
        adaln_indices: mx.array,
        rotary_emb: tuple[mx.array, mx.array],
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)

        residual = hidden_states
        norm_hs = self.norm1(hidden_states)
        norm_hs = norm_hs * (1.0 + _index_select_rows(scale_msa, adaln_indices)) + _index_select_rows(
            shift_msa, adaln_indices
        )
        attn_out = self.attn(norm_hs, rotary_emb, attention_mask)
        hidden_states = residual + _index_select_rows(gate_msa, adaln_indices) * attn_out

        residual = hidden_states
        norm_hs = self.norm2(hidden_states)
        norm_hs = norm_hs * (1.0 + _index_select_rows(scale_mlp, adaln_indices)) + _index_select_rows(
            shift_mlp, adaln_indices
        )
        ff_out = self.ff(norm_hs)
        hidden_states = residual + _index_select_rows(gate_mlp, adaln_indices) * ff_out
        return hidden_states


class MiniMaxH3DiTMLX(nn.Module):
    """MLX MiniMax-H3 joint video+audio DiT (Diffusers ``MiniMaxH3Transformer3DModel``)."""

    def __init__(
        self,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 2688,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
    ):
        super().__init__()
        patch = tuple(int(x) for x in patch_size)
        if len(patch) != 3:
            raise ValueError(f"`patch_size` must be (t, h, w), got {patch_size!r}")
        self.patch_size = patch
        self.in_channels = in_channels
        self.audio_in_channels = audio_in_channels
        self.hidden_size = hidden_size
        video_patch_dim = in_channels * patch[0] * patch[1] * patch[2]

        self.proj_in = nn.Linear(video_patch_dim, hidden_size, bias=True)
        self.audio_proj_in = nn.Linear(audio_in_channels, hidden_size, bias=True)
        self.context_embedder = nn.Linear(text_dim, hidden_size, bias=True)

        self.time_proj = MiniMaxH3Timesteps(
            num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0.0,
        )
        self.time_embedder = MiniMaxH3TimestepEmbedding(
            in_channels=freq_dim, time_embed_dim=time_embed_hidden_dim, out_dim=time_embed_dim,
        )
        self.rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=rope_freq_dim, rope_theta=rope_theta)
        self.token_refiner = MiniMaxH3TokenRefiner(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            ffn_dim=ffn_dim,
            num_layers=num_refiner_layers,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
        )
        self.transformer_blocks = [
            MiniMaxH3TransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                ffn_dim=ffn_dim,
                time_embed_dim=time_embed_dim,
                norm_eps=norm_eps,
                qk_norm_eps=qk_norm_eps,
            )
            for _ in range(num_layers)
        ]
        self.norm_out = MiniMaxH3AdaLayerNormOut(
            hidden_size=hidden_size, time_embed_dim=time_embed_dim, eps=final_norm_eps,
        )
        self.proj_out = nn.Linear(hidden_size, video_patch_dim, bias=True)
        self.audio_proj_out = nn.Linear(hidden_size, audio_in_channels, bias=True)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "MiniMaxH3DiTMLX":
        patch = cfg.get("patch_size", (1, 2, 2))
        if isinstance(patch, list):
            patch = tuple(patch)
        return cls(
            num_attention_heads=int(cfg.get("num_attention_heads", 56)),
            attention_head_dim=int(cfg.get("attention_head_dim", 128)),
            hidden_size=int(cfg.get("hidden_size", 5376)),
            num_layers=int(cfg.get("num_layers", 50)),
            num_refiner_layers=int(cfg.get("num_refiner_layers", 2)),
            ffn_dim=int(cfg.get("ffn_dim", 14336)),
            in_channels=int(cfg.get("in_channels", 24)),
            audio_in_channels=int(cfg.get("audio_in_channels", 32)),
            patch_size=tuple(int(x) for x in patch),
            text_dim=int(cfg.get("text_dim", 5120)),
            freq_dim=int(cfg.get("freq_dim", 256)),
            time_embed_hidden_dim=int(cfg.get("time_embed_hidden_dim", 5376)),
            time_embed_dim=int(cfg.get("time_embed_dim", 2688)),
            rope_freq_dim=int(cfg.get("rope_freq_dim", 16)),
            rope_theta=float(cfg.get("rope_theta", 10000.0)),
            norm_eps=float(cfg.get("norm_eps", 1e-5)),
            qk_norm_eps=float(cfg.get("qk_norm_eps", 1e-5)),
            final_norm_eps=float(cfg.get("final_norm_eps", 1e-5)),
        )

    def _padding_attention_mask(self, token_tags: mx.array, dtype: mx.Dtype) -> mx.array | None:
        is_pad = token_tags < 0
        if not bool(mx.any(is_pad).item()):
            return None
        # Live↔live and pad↔pad attend; cross groups are blocked (additive -inf).
        same = is_pad[None, :] == is_pad[:, None]  # [S, S]
        neg = mx.full(same.shape, -math.inf, dtype=dtype)
        mask = mx.where(same, mx.zeros(same.shape, dtype=dtype), neg)
        return mask[None, None, :, :]  # [1, 1, S, S]

    def __call__(
        self,
        hidden_states: mx.array,
        audio_hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        timestep_indices: mx.array,
        token_tags: mx.array,
        position_ids: mx.array,
        video_indices: mx.array,
        audio_indices: mx.array,
        text_indices: mx.array,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> dict[str, mx.array] | tuple[mx.array, mx.array]:
        del attention_kwargs  # LoRA scale hook reserved for callers
        if position_ids.ndim != 2 or int(position_ids.shape[-1]) != 3:
            raise ValueError(f"`position_ids` must be (seq_len, 3), got {tuple(position_ids.shape)}")
        sequence_length = int(position_ids.shape[0])
        if tuple(token_tags.shape) != (sequence_length,) or tuple(timestep_indices.shape) != (sequence_length,):
            raise ValueError(
                "`token_tags` and `timestep_indices` must be (seq_len,) matching `position_ids`, "
                f"got {tuple(token_tags.shape)} and {tuple(timestep_indices.shape)} for seq_len={sequence_length}."
            )

        rotary_emb = self.rope(position_ids)

        video_embeds = self.proj_in(hidden_states.astype(self.proj_in.weight.dtype))
        audio_embeds = self.audio_proj_in(audio_hidden_states.astype(self.audio_proj_in.weight.dtype))
        text_embeds = self.context_embedder(encoder_hidden_states.astype(self.context_embedder.weight.dtype))
        text_embeds = self.token_refiner(text_embeds)

        packed = mx.zeros(
            (int(text_embeds.shape[0]), sequence_length, int(text_embeds.shape[-1])),
            dtype=text_embeds.dtype,
        )
        packed = _index_copy_axis1(packed, text_indices, text_embeds)
        packed = _index_copy_axis1(packed, video_indices, video_embeds.astype(text_embeds.dtype))
        packed = _index_copy_axis1(packed, audio_indices, audio_embeds.astype(text_embeds.dtype))

        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.astype(self.time_embedder.linear_1.weight.dtype))

        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + mx.maximum(token_tags, mx.zeros_like(token_tags))
        attention_mask = self._padding_attention_mask(token_tags, packed.dtype)

        for block in self.transformer_blocks:
            packed = block(packed, temb, adaln_indices, rotary_emb, attention_mask)

        packed = self.norm_out(packed, temb, timestep_indices).astype(self.proj_out.weight.dtype)
        video_output = _index_select_axis1(self.proj_out(packed), video_indices)
        audio_output = _index_select_axis1(self.audio_proj_out(packed), audio_indices)

        if not return_dict:
            return (video_output, audio_output)
        return {"sample": video_output, "audio_sample": audio_output}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Diffusers-style alias for ``__call__``."""
        return self(*args, **kwargs)
