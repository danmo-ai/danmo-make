"""MLX MiniMax-H3 DiT — PipeNetwork / upstream checkpoint key tree (1:1 load).

Module names match released safetensors (``video_patch_proj``, ``blocks.*.attn.qkv_proj``,
``mlp.fc1`` fused SwiGLU, ``final_layer.video_out``, …). Layout quirks handled in forward:

* ``qkv_proj`` rows are per-head interleaved ``[h0: q,k,v][h1: q,k,v]…``
* ``mlp.fc1`` is fused ``[gate; value]`` SwiGLU (``fc2(silu(gate) * value)``)
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

MINIMAX_H3_MODALITY_NUM = 3

_FP32_KEY_PREFIXES = (
    "video_patch_proj.",
    "audio_patch_proj.",
    "time_embedder.",
    "final_layer.video_out.",
    "final_layer.audio_out.",
)


def _param_dtype(layer: nn.Module) -> mx.Dtype:
    scales = getattr(layer, "scales", None)
    return scales.dtype if scales is not None else layer.weight.dtype


def _apply_rotary_emb(hidden_states: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
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
    b, _s, c = buf.shape
    n = int(indices.shape[0])
    idx = mx.broadcast_to(indices.astype(mx.int32).reshape(1, n, 1), (b, n, c))
    return mx.put_along_axis(buf, idx, values, axis=1)


def _index_select_rows(table: mx.array, indices: mx.array) -> mx.array:
    return mx.take(table, indices.astype(mx.int32), axis=0)


class MiniMaxH3RotaryPosEmbed3D:
    """3-axis RoPE; ``inv_freq`` is recomputed (not loaded from checkpoint)."""

    def __init__(self, rope_freq_dim: int = 16, rope_theta: float = 10000.0):
        n = rope_freq_dim
        self.inv_freq = 1.0 / (
            rope_theta ** (mx.arange(0, 2 * n, 2, dtype=mx.float32) / (2 * n))
        )

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        pos = position_ids.astype(mx.float32)
        freqs = pos[..., None] * self.inv_freq.reshape(1, 1, -1)
        freqs = mx.concatenate([freqs[:, 0], freqs[:, 1], freqs[:, 2]], axis=-1)
        freqs = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(freqs), mx.sin(freqs)


class MiniMaxH3TimestepEmbedder(nn.Module):
    """Timestep MLP — checkpoint keys ``time_embedder.proj_in`` / ``proj_out``."""

    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.proj_in = nn.Linear(in_channels, hidden_dim, bias=True)
        self.proj_out = nn.Linear(hidden_dim, out_dim, bias=True)

    def __call__(self, sinusoid: mx.array) -> mx.array:
        return self.proj_out(nn.silu(self.proj_in(sinusoid)))


class MiniMaxH3Attention(nn.Module):
    """Fused QKV with per-head interleaved layout."""

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
        self.qkv_proj = nn.Linear(hidden_size, 3 * self.inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(dim_head, eps=qk_norm_eps)
        self.k_norm = nn.RMSNorm(dim_head, eps=qk_norm_eps)
        self.out_proj = nn.Linear(self.inner_dim, hidden_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        rotary_emb: tuple[mx.array, mx.array] | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        b, s, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).reshape(b, s, self.heads, 3, self.head_dim)
        query = qkv[:, :, :, 0]
        key = qkv[:, :, :, 1]
        value = qkv[:, :, :, 2]

        query = self.q_norm(query).transpose(0, 2, 1, 3)
        key = self.k_norm(key).transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        out = scaled_dot_product_attention_bhsd_mx(
            mx, query, key, value, scale=self.scale, mask=attention_mask,
        )
        out = out.transpose(0, 2, 1, 3).reshape(b, s, self.inner_dim)
        return self.out_proj(out.astype(hidden_states.dtype))


class MiniMaxH3FeedForward(nn.Module):
    """SwiGLU — ``fc1`` is fused ``[gate; value]`` (upstream checkpoint layout)."""

    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 2 * ffn_hidden_size, bias=False)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)
        self._ffn = ffn_hidden_size

    def __call__(self, hidden_states: mx.array) -> mx.array:
        fused = self.fc1(hidden_states)
        gate, value = fused[..., : self._ffn], fused[..., self._ffn :]
        return self.fc2(nn.silu(gate) * value)


class MiniMaxH3AdaLayerNormModulation(nn.Module):
    def __init__(self, time_embed_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(
            time_embed_dim, 6 * hidden_size * MINIMAX_H3_MODALITY_NUM, bias=True,
        )

    def __call__(self, temb: mx.array) -> tuple[mx.array, ...]:
        h = nn.silu(temb).astype(_param_dtype(self.linear))
        h = self.linear(h).reshape(-1, 6 * self.hidden_size)
        return tuple(mx.split(h, 6, axis=-1))


class MiniMaxH3TokenRefinerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_hidden_size: int,
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
        self.mlp = MiniMaxH3FeedForward(hidden_size, ffn_hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_hidden_size: int,
        num_layers: int,
        norm_eps: float,
        qk_norm_eps: float,
        final_norm_eps: float,
    ):
        super().__init__()
        self.blocks = [
            MiniMaxH3TokenRefinerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                ffn_hidden_size=ffn_hidden_size,
                norm_eps=norm_eps,
                qk_norm_eps=qk_norm_eps,
            )
            for _ in range(num_layers)
        ]
        self.final_norm = nn.RMSNorm(hidden_size, eps=final_norm_eps)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)


class MiniMaxH3TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_hidden_size: int,
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
        self.mlp = MiniMaxH3FeedForward(hidden_size, ffn_hidden_size)
        self.adaln_proj = MiniMaxH3AdaLayerNormModulation(
            time_embed_dim=time_embed_dim, hidden_size=hidden_size,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        modulation: tuple[mx.array, ...],
        adaln_indices: mx.array,
        rotary_emb: tuple[mx.array, mx.array],
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation

        h = self.norm1(hidden_states)
        h = h * (1.0 + _index_select_rows(scale_msa, adaln_indices)) + _index_select_rows(
            shift_msa, adaln_indices,
        )
        hidden_states = hidden_states + _index_select_rows(gate_msa, adaln_indices) * self.attn(
            h, rotary_emb, attention_mask,
        )

        h = self.norm2(hidden_states)
        h = h * (1.0 + _index_select_rows(scale_mlp, adaln_indices)) + _index_select_rows(
            shift_mlp, adaln_indices,
        )
        return hidden_states + _index_select_rows(gate_mlp, adaln_indices) * self.mlp(h)


class MiniMaxH3FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, time_embed_dim: int, video_patch_dim: int, audio_dim: int, eps: float):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.adaln_proj = nn.Linear(time_embed_dim, 2 * hidden_size, bias=True)
        self.video_out = nn.Linear(hidden_size, video_patch_dim, bias=True)
        self.audio_out = nn.Linear(hidden_size, audio_dim, bias=True)
        self.hidden_size = hidden_size

    def norm_out(self, hidden_states: mx.array, temb: mx.array, timestep_indices: mx.array) -> mx.array:
        h = self.adaln_proj(nn.silu(temb).astype(_param_dtype(self.adaln_proj)))
        shift, scale = mx.split(h, 2, axis=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + _index_select_rows(scale, timestep_indices)) + _index_select_rows(
            shift, timestep_indices,
        )


class MiniMaxH3DiTMLX(nn.Module):
    """MLX MiniMax-H3 joint video+audio DiT (PipeNetwork / upstream keys)."""

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

        self.video_patch_proj = nn.Linear(video_patch_dim, hidden_size, bias=True)
        self.audio_patch_proj = nn.Linear(audio_in_channels, hidden_size, bias=True)
        self.condition_proj = nn.Linear(text_dim, hidden_size, bias=True)

        self.time_embedder = MiniMaxH3TimestepEmbedder(
            in_channels=freq_dim,
            hidden_dim=time_embed_hidden_dim,
            out_dim=time_embed_dim,
        )
        self.rope = MiniMaxH3RotaryPosEmbed3D(rope_freq_dim=rope_freq_dim, rope_theta=rope_theta)
        self.token_refiner = MiniMaxH3TokenRefiner(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            ffn_hidden_size=ffn_dim,
            num_layers=num_refiner_layers,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
        )
        self.blocks = [
            MiniMaxH3TransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                ffn_hidden_size=ffn_dim,
                time_embed_dim=time_embed_dim,
                norm_eps=norm_eps,
                qk_norm_eps=qk_norm_eps,
            )
            for _ in range(num_layers)
        ]
        self._active_layers = num_layers
        self.final_layer = MiniMaxH3FinalLayer(
            hidden_size=hidden_size,
            time_embed_dim=time_embed_dim,
            video_patch_dim=video_patch_dim,
            audio_dim=audio_in_channels,
            eps=final_norm_eps,
        )

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
            num_refiner_layers=int(
                cfg.get("num_refiner_layers", cfg.get("token_refiner_num_layers", 2)),
            ),
            ffn_dim=int(cfg.get("ffn_dim", cfg.get("ffn_hidden_size", 14336))),
            in_channels=int(cfg.get("in_channels", cfg.get("latents_dim", 24))),
            audio_in_channels=int(
                cfg.get("audio_in_channels", cfg.get("audio_latents_dim", 32)),
            ),
            patch_size=tuple(int(x) for x in patch),
            text_dim=int(cfg.get("text_dim", 5120)),
            freq_dim=int(cfg.get("freq_dim", cfg.get("timestep_input_dim", 256))),
            time_embed_hidden_dim=int(
                cfg.get("time_embed_hidden_dim", cfg.get("time_embed_hidden_size", 5376)),
            ),
            time_embed_dim=int(cfg.get("time_embed_dim", 2688)),
            rope_freq_dim=int(cfg.get("rope_freq_dim", cfg.get("rope_inv_freq_len", 16))),
            rope_theta=float(cfg.get("rope_theta", 10000.0)),
            norm_eps=float(cfg.get("norm_eps", 1e-5)),
            qk_norm_eps=float(cfg.get("qk_norm_eps", 1e-5)),
            final_norm_eps=float(cfg.get("final_norm_eps", 1e-5)),
        )

    def set_active_layers(self, count: int) -> None:
        count = int(count)
        if count < 1 or count > len(self.blocks):
            raise ValueError(
                f"h3_active_layers must be in [1, {len(self.blocks)}], got {count}",
            )
        self._active_layers = count

    def _padding_attention_mask(self, token_tags: mx.array, dtype: mx.Dtype) -> mx.array | None:
        is_pad = token_tags < 0
        if not bool(mx.any(is_pad).item()):
            return None
        same = is_pad[None, :] == is_pad[:, None]
        neg = mx.full(same.shape, -math.inf, dtype=dtype)
        mask = mx.where(same, mx.zeros(same.shape, dtype=dtype), neg)
        return mask[None, None, :, :]

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
        del attention_kwargs
        if position_ids.ndim != 2 or int(position_ids.shape[-1]) != 3:
            raise ValueError(f"`position_ids` must be (seq_len, 3), got {tuple(position_ids.shape)}")
        sequence_length = int(position_ids.shape[0])
        if tuple(token_tags.shape) != (sequence_length,) or tuple(timestep_indices.shape) != (sequence_length,):
            raise ValueError(
                "`token_tags` and `timestep_indices` must be (seq_len,) matching `position_ids`, "
                f"got {tuple(token_tags.shape)} and {tuple(timestep_indices.shape)} for seq_len={sequence_length}."
            )

        rotary_emb = self.rope(position_ids)

        video_embeds = self.video_patch_proj(
            hidden_states.astype(_param_dtype(self.video_patch_proj)),
        )
        audio_embeds = self.audio_patch_proj(
            audio_hidden_states.astype(_param_dtype(self.audio_patch_proj)),
        )
        text_embeds = self.condition_proj(
            encoder_hidden_states.astype(_param_dtype(self.condition_proj)),
        )
        text_embeds = self.token_refiner(text_embeds)

        packed = mx.zeros(
            (int(text_embeds.shape[0]), sequence_length, int(text_embeds.shape[-1])),
            dtype=text_embeds.dtype,
        )
        packed = _index_copy_axis1(packed, text_indices, text_embeds)
        packed = _index_copy_axis1(packed, video_indices, video_embeds.astype(text_embeds.dtype))
        packed = _index_copy_axis1(packed, audio_indices, audio_embeds.astype(text_embeds.dtype))

        temb_input = sinusoidal_timestep_proj(
            _MLX_CTX,
            timestep,
            self.time_embedder.proj_in.weight.shape[1],
            sin_first=True,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        temb = self.time_embedder(temb_input.astype(_param_dtype(self.time_embedder.proj_in)))

        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + mx.maximum(
            token_tags, mx.zeros_like(token_tags),
        )
        attention_mask = self._padding_attention_mask(token_tags, packed.dtype)

        for block in self.blocks[: self._active_layers]:
            modulation = block.adaln_proj(temb)
            packed = block(packed, modulation, adaln_indices, rotary_emb, attention_mask)

        packed = self.final_layer.norm_out(packed, temb, timestep_indices)
        video_output = _index_select_axis1(
            self.final_layer.video_out(packed.astype(_param_dtype(self.final_layer.video_out))),
            video_indices,
        )
        audio_output = _index_select_axis1(
            self.final_layer.audio_out(packed.astype(_param_dtype(self.final_layer.audio_out))),
            audio_indices,
        )

        if not return_dict:
            return (video_output, audio_output)
        return {"sample": video_output, "audio_sample": audio_output}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self(*args, **kwargs)


def is_fp32_dit_key(key: str) -> bool:
    return key.startswith(_FP32_KEY_PREFIXES)


def expected_dit_param_keys(model: MiniMaxH3DiTMLX) -> set[str]:
    from mlx.utils import tree_flatten

    return {key for key, _ in tree_flatten(model.parameters())}
