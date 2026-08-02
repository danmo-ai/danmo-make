"""LTX 2.3 joint audio/video DiT — MLX implementation (48-layer A/V transformer).

Public stem: ``families.ltx.transformer.LTXTransformer`` → ``LTX23Transformer``.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.model.base import TransformerBase
from backend.engine.common.ops.attention import scaled_dot_product_attention_bhsd_mx
from backend.engine.common.ops.embeddings import sinusoidal_timestep_proj
from backend.engine.config.model_configs import LTXConfig
from backend.engine.families.ltx.perturbations_mlx import BatchedPerturbationConfig, PerturbationType
from backend.engine.runtime._base import RuntimeContext

# Watchdog guard: flush lazy graph every N blocks (Metal ~10 s deadline).
_DIT_EVAL_EVERY = int(os.environ.get("LTX2_DIT_EVAL_EVERY", "8"))
_mx_eval = getattr(mx, "eval")


# ---------------------------------------------------------------------------
# RoPE helpers (from ltx-core rope.py — inlined, no external import)
# ---------------------------------------------------------------------------

def _generate_freq_grid(theta: float, num_pos_dims: int, inner_dim: int) -> mx.array:
    n_elem = 2 * num_pos_dims
    num_freqs = inner_dim // n_elem
    indices = theta ** mx.linspace(
        math.log(1.0) / math.log(theta),
        math.log(theta) / math.log(theta),
        num_freqs,
    ).astype(mx.float32)
    return indices * (math.pi / 2.0)


def _compute_freqs(freq_indices: mx.array, positions: mx.array, max_pos: list[int]) -> mx.array:
    num_pos_dims = positions.shape[-1]
    frac_positions = mx.stack(
        [positions[:, :, i].astype(mx.float32) / max_pos[i] for i in range(num_pos_dims)],
        axis=-1,
    )
    scaled = freq_indices * (frac_positions[..., None] * 2.0 - 1.0)
    return scaled.transpose(0, 1, 3, 2).reshape(positions.shape[0], positions.shape[1], -1)


def _precompute_rope_freqs(
    positions: mx.array,
    inner_dim: int,
    num_heads: int,
    theta: float = 10000.0,
    max_pos: list[int] | None = None,
    rope_type: str = "split",
) -> tuple[mx.array, mx.array, str]:
    num_pos_dims = positions.shape[-1]
    if max_pos is None:
        max_pos = [20, 2048, 2048][:num_pos_dims]

    freq_indices = _generate_freq_grid(theta, num_pos_dims, inner_dim)
    freqs = _compute_freqs(freq_indices, positions, max_pos)
    b, n, num_freqs = freqs.shape

    if rope_type == "interleaved":
        cos_f = mx.cos(freqs)
        sin_f = mx.sin(freqs)
        cos_f = mx.repeat(cos_f, 2, axis=-1)
        sin_f = mx.repeat(sin_f, 2, axis=-1)
        pad_size = inner_dim - cos_f.shape[-1]
        if pad_size > 0:
            cos_f = mx.concatenate([mx.ones((*cos_f.shape[:-1], pad_size)), cos_f], axis=-1)
            sin_f = mx.concatenate([mx.zeros((*sin_f.shape[:-1], pad_size)), sin_f], axis=-1)
        head_dim = inner_dim // num_heads
        cos_f = cos_f.reshape(b, n, num_heads, head_dim).transpose(0, 2, 1, 3)
        sin_f = sin_f.reshape(b, n, num_heads, head_dim).transpose(0, 2, 1, 3)
        return cos_f, sin_f, rope_type

    expected = inner_dim // 2
    pad_size = expected - num_freqs
    if pad_size > 0:
        freqs = mx.concatenate([mx.zeros((*freqs.shape[:-1], pad_size)), freqs], axis=-1)
    cos_f = mx.cos(freqs)
    sin_f = mx.sin(freqs)
    head_dim_half = inner_dim // (2 * num_heads)
    cos_f = cos_f.reshape(b, n, num_heads, head_dim_half).transpose(0, 2, 1, 3)
    sin_f = sin_f.reshape(b, n, num_heads, head_dim_half).transpose(0, 2, 1, 3)
    return cos_f, sin_f, rope_type


@mx.compile
def _apply_rope_interleaved(x: mx.array, cos_freqs: mx.array, sin_freqs: mx.array) -> mx.array:
    cos_f = cos_freqs.astype(x.dtype)
    sin_f = sin_freqs.astype(x.dtype)
    x_pairs = x.reshape(*x.shape[:-1], -1, 2)
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]
    x_rot = mx.stack([-x2, x1], axis=-1).reshape(x.shape)
    return x * cos_f + x_rot * sin_f


@mx.compile
def _apply_rope_split(x: mx.array, cos_freqs: mx.array, sin_freqs: mx.array) -> mx.array:
    cos_f = cos_freqs.astype(x.dtype)
    sin_f = sin_freqs.astype(x.dtype)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * cos_f - x2 * sin_f, x1 * sin_f + x2 * cos_f], axis=-1)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class LTX23TimestepEmbedder(nn.Module):
    """Weight keys: ``emb.timestep_embedder.linear_{1,2}.*``."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.timestep_embedder = _LTX23TimestepMLP(in_channels, time_embed_dim)

    def __call__(self, sample: mx.array) -> mx.array:
        return self.timestep_embedder(sample)


class _LTX23TimestepMLP(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, sample: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(sample)))


class LTX23AdaLayerNormSingle(nn.Module):
    """Adaptive Layer Norm producing modulation params from timestep."""

    def __init__(self, dim: int, num_params: int = 6, timestep_dim: int | None = None):
        super().__init__()
        self.num_params = num_params
        t_dim = timestep_dim or dim
        self.emb = LTX23TimestepEmbedder(t_dim, dim)
        self.linear = nn.Linear(dim, num_params * dim)

    def __call__(self, timestep_emb: mx.array) -> tuple[mx.array, mx.array]:
        embedded = self.emb(timestep_emb)
        params = self.linear(nn.silu(embedded))
        return params, embedded


class LTX23Attention(nn.Module):
    """Multi-head attention with RoPE and per-head gating."""

    def __init__(
        self,
        query_dim: int,
        kv_dim: int | None = None,
        out_dim: int | None = None,
        num_heads: int = 32,
        head_dim: int = 128,
        qkv_bias: bool = True,
        use_rope: bool = True,
        norm_eps: float = 1e-6,
        apply_gated_attention: bool = True,
    ):
        super().__init__()
        if kv_dim is None:
            kv_dim = query_dim
        if out_dim is None:
            out_dim = query_dim

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_rope = use_rope
        self.scale = head_dim**-0.5

        inner_dim = num_heads * head_dim
        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(kv_dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(kv_dim, inner_dim, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, out_dim, bias=True)
        self.to_gate_logits = nn.Linear(query_dim, num_heads, bias=True) if apply_gated_attention else None
        self.q_norm = nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(inner_dim, eps=norm_eps)

    def __call__(
        self,
        x: mx.array,
        encoder_hidden_states: mx.array | None = None,
        rope_freqs: tuple[mx.array, mx.array, str] | None = None,
        rope_freqs_k: tuple[mx.array, mx.array, str] | None = None,
        attention_mask: mx.array | None = None,
        perturbation_mask: mx.array | None = None,
    ) -> mx.array:
        b, n, _ = x.shape
        kv_input = encoder_hidden_states if encoder_hidden_states is not None else x

        q = self.q_norm(self.to_q(x))
        k = self.k_norm(self.to_k(kv_input))
        v = self.to_v(kv_input)

        q = q.reshape(b, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.use_rope and rope_freqs is not None:
            cos_f, sin_f, rtype = rope_freqs
            apply_fn = _apply_rope_split if rtype == "split" else _apply_rope_interleaved
            q = apply_fn(q, cos_f, sin_f)
            if rope_freqs_k is not None:
                cos_fk, sin_fk, _ = rope_freqs_k
            else:
                cos_fk, sin_fk = cos_f, sin_f
            k = apply_fn(k, cos_fk, sin_fk)

        out = scaled_dot_product_attention_bhsd_mx(mx, q, k, v, scale=self.scale, mask=attention_mask)

        if perturbation_mask is not None:
            out = out * perturbation_mask + v * (1.0 - perturbation_mask)

        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            gate = 2.0 * mx.sigmoid(gate_logits)
            out = out * gate.transpose(0, 2, 1)[:, :, :, None]

        out = out.transpose(0, 2, 1, 3).reshape(b, -1, self.num_heads * self.head_dim)
        return self.to_out(out)


class LTX23FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int | None = None, mult: float = 4.0):
        super().__init__()
        dim_out = dim_out or dim
        inner_dim = int(dim * mult)
        self.proj_in = nn.Linear(dim, inner_dim)
        self.proj_out = nn.Linear(inner_dim, dim_out)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj_out(nn.gelu_approx(self.proj_in(x)))


class LTX23AVBlock(nn.Module):
    """Joint audio+video transformer block."""

    def __init__(
        self,
        video_dim: int = 4096,
        audio_dim: int = 2048,
        video_num_heads: int = 32,
        audio_num_heads: int = 32,
        video_head_dim: int = 128,
        audio_head_dim: int = 64,
        av_cross_num_heads: int = 32,
        av_cross_head_dim: int = 64,
        ff_mult: float = 4.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.attn1 = LTX23Attention(
            query_dim=video_dim, num_heads=video_num_heads, head_dim=video_head_dim,
            use_rope=True, norm_eps=norm_eps,
        )
        self.audio_attn1 = LTX23Attention(
            query_dim=audio_dim, num_heads=audio_num_heads, head_dim=audio_head_dim,
            use_rope=True, norm_eps=norm_eps,
        )
        self.attn2 = LTX23Attention(
            query_dim=video_dim, num_heads=video_num_heads, head_dim=video_head_dim,
            use_rope=False, norm_eps=norm_eps,
        )
        self.audio_attn2 = LTX23Attention(
            query_dim=audio_dim, num_heads=audio_num_heads, head_dim=audio_head_dim,
            use_rope=False, norm_eps=norm_eps,
        )
        self.audio_to_video_attn = LTX23Attention(
            query_dim=video_dim, kv_dim=audio_dim, out_dim=video_dim,
            num_heads=av_cross_num_heads, head_dim=av_cross_head_dim,
            use_rope=True, norm_eps=norm_eps,
        )
        self.video_to_audio_attn = LTX23Attention(
            query_dim=audio_dim, kv_dim=video_dim, out_dim=audio_dim,
            num_heads=av_cross_num_heads, head_dim=av_cross_head_dim,
            use_rope=True, norm_eps=norm_eps,
        )
        self.ff = LTX23FeedForward(video_dim, dim_out=video_dim, mult=ff_mult)
        self.audio_ff = LTX23FeedForward(audio_dim, dim_out=audio_dim, mult=ff_mult)

        self.scale_shift_table = mx.zeros((9, video_dim))
        self.audio_scale_shift_table = mx.zeros((9, audio_dim))
        self.prompt_scale_shift_table = mx.zeros((2, video_dim))
        self.audio_prompt_scale_shift_table = mx.zeros((2, audio_dim))
        self.scale_shift_table_a2v_ca_video = mx.zeros((5, video_dim))
        self.scale_shift_table_a2v_ca_audio = mx.zeros((5, audio_dim))
        self._norm_eps = norm_eps

    @staticmethod
    def _unpack_adaln(params: mx.array, table: mx.array, num_params: int, dim: int) -> list[mx.array]:
        if params.ndim == 2:
            p = params.reshape(-1, num_params, dim)
            p = p + table[None, :num_params, :]
            return [p[:, i, :][:, None, :] for i in range(num_params)]
        b, n, _ = params.shape
        p = params.reshape(b, n, num_params, dim)
        p = p + table[None, None, :num_params, :]
        return [p[:, :, i, :] for i in range(num_params)]

    def _rms_norm(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, weight=None, eps=self._norm_eps)

    def __call__(
        self,
        video_hidden: mx.array,
        audio_hidden: mx.array,
        video_adaln_params: mx.array,
        audio_adaln_params: mx.array,
        video_prompt_adaln_params: mx.array,
        audio_prompt_adaln_params: mx.array,
        av_ca_video_params: mx.array,
        av_ca_audio_params: mx.array,
        av_ca_a2v_gate_params: mx.array,
        av_ca_v2a_gate_params: mx.array,
        video_text_embeds: mx.array | None = None,
        audio_text_embeds: mx.array | None = None,
        video_rope_freqs: tuple[mx.array, mx.array, str] | None = None,
        audio_rope_freqs: tuple[mx.array, mx.array, str] | None = None,
        video_cross_rope_freqs: tuple[mx.array, mx.array, str] | None = None,
        audio_cross_rope_freqs: tuple[mx.array, mx.array, str] | None = None,
        video_attention_mask: mx.array | None = None,
        audio_attention_mask: mx.array | None = None,
        block_idx: int = 0,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[mx.array, mx.array]:
        vdim = video_hidden.shape[-1]
        adim = audio_hidden.shape[-1]

        (
            v_shift_sa, v_scale_sa, v_gate_sa,
            v_shift_ff, v_scale_ff, v_gate_ff,
            v_shift_ca, v_scale_ca, v_gate_ca,
        ) = self._unpack_adaln(video_adaln_params, self.scale_shift_table, 9, vdim)

        (
            a_shift_sa, a_scale_sa, a_gate_sa,
            a_shift_ff, a_scale_ff, a_gate_ff,
            a_shift_ca, a_scale_ca, a_gate_ca,
        ) = self._unpack_adaln(audio_adaln_params, self.audio_scale_shift_table, 9, adim)

        av_v_scale_a2v, av_v_shift_a2v, av_v_scale_v2a, av_v_shift_v2a = self._unpack_adaln(
            av_ca_video_params, self.scale_shift_table_a2v_ca_video, 4, vdim,
        )
        if av_ca_a2v_gate_params.ndim == 2:
            av_v_gate_a2v = (av_ca_a2v_gate_params + self.scale_shift_table_a2v_ca_video[4, :])[:, None, :]
        else:
            av_v_gate_a2v = av_ca_a2v_gate_params + self.scale_shift_table_a2v_ca_video[None, None, 4, :]

        av_a_scale_a2v, av_a_shift_a2v, av_a_scale_v2a, av_a_shift_v2a = self._unpack_adaln(
            av_ca_audio_params, self.scale_shift_table_a2v_ca_audio, 4, adim,
        )
        if av_ca_v2a_gate_params.ndim == 2:
            av_a_gate_v2a = (av_ca_v2a_gate_params + self.scale_shift_table_a2v_ca_audio[4, :])[:, None, :]
        else:
            av_a_gate_v2a = av_ca_v2a_gate_params + self.scale_shift_table_a2v_ca_audio[None, None, 4, :]

        video_normed = self._rms_norm(video_hidden) * (1.0 + v_scale_sa) + v_shift_sa
        v_ptb_mask = None
        if perturbations is not None and perturbations.any_in_batch(
            PerturbationType.SKIP_VIDEO_SELF_ATTN, block_idx
        ):
            v_ptb_mask = perturbations.mask_like(
                PerturbationType.SKIP_VIDEO_SELF_ATTN,
                block_idx,
                video_hidden[:, :1, :1, None],
            )
        video_sa_out = self.attn1(
            video_normed,
            rope_freqs=video_rope_freqs,
            attention_mask=video_attention_mask,
            perturbation_mask=v_ptb_mask,
        )
        video_hidden = video_hidden + video_sa_out * v_gate_sa

        audio_normed = self._rms_norm(audio_hidden) * (1.0 + a_scale_sa) + a_shift_sa
        a_ptb_mask = None
        if perturbations is not None and perturbations.any_in_batch(
            PerturbationType.SKIP_AUDIO_SELF_ATTN, block_idx
        ):
            a_ptb_mask = perturbations.mask_like(
                PerturbationType.SKIP_AUDIO_SELF_ATTN,
                block_idx,
                audio_hidden[:, :1, :1, None],
            )
        audio_sa_out = self.audio_attn1(
            audio_normed,
            rope_freqs=audio_rope_freqs,
            attention_mask=audio_attention_mask,
            perturbation_mask=a_ptb_mask,
        )
        audio_hidden = audio_hidden + audio_sa_out * a_gate_sa

        if video_text_embeds is not None:
            video_normed = self._rms_norm(video_hidden) * (1.0 + v_scale_ca) + v_shift_ca
            vp_shift, vp_scale = self._unpack_adaln(
                video_prompt_adaln_params, self.prompt_scale_shift_table, 2, vdim,
            )
            text_scaled = video_text_embeds * (1.0 + vp_scale) + vp_shift
            video_hidden = video_hidden + self.attn2(video_normed, encoder_hidden_states=text_scaled) * v_gate_ca

        if audio_text_embeds is not None:
            audio_normed = self._rms_norm(audio_hidden) * (1.0 + a_scale_ca) + a_shift_ca
            ap_shift, ap_scale = self._unpack_adaln(
                audio_prompt_adaln_params, self.audio_prompt_scale_shift_table, 2, adim,
            )
            text_scaled = audio_text_embeds * (1.0 + ap_scale) + ap_shift
            audio_hidden = audio_hidden + self.audio_attn2(audio_normed, encoder_hidden_states=text_scaled) * a_gate_ca

        video_norm3 = self._rms_norm(video_hidden)
        audio_norm3 = self._rms_norm(audio_hidden)

        video_q_a2v = video_norm3 * (1.0 + av_v_scale_a2v) + av_v_shift_a2v
        audio_kv_a2v = audio_norm3 * (1.0 + av_a_scale_a2v) + av_a_shift_a2v
        a2v_out = (
            self.audio_to_video_attn(
                video_q_a2v,
                encoder_hidden_states=audio_kv_a2v,
                rope_freqs=video_cross_rope_freqs,
                rope_freqs_k=audio_cross_rope_freqs,
            )
            * av_v_gate_a2v
        )
        if perturbations is not None and perturbations.any_in_batch(
            PerturbationType.SKIP_A2V_CROSS_ATTN, block_idx
        ):
            a2v_mask = perturbations.mask_like(
                PerturbationType.SKIP_A2V_CROSS_ATTN, block_idx, video_hidden
            )
            a2v_out = a2v_out * a2v_mask
        video_hidden = video_hidden + a2v_out

        audio_q_v2a = audio_norm3 * (1.0 + av_a_scale_v2a) + av_a_shift_v2a
        video_kv_v2a = video_norm3 * (1.0 + av_v_scale_v2a) + av_v_shift_v2a
        v2a_out = (
            self.video_to_audio_attn(
                audio_q_v2a,
                encoder_hidden_states=video_kv_v2a,
                rope_freqs=audio_cross_rope_freqs,
                rope_freqs_k=video_cross_rope_freqs,
            )
            * av_a_gate_v2a
        )
        if perturbations is not None and perturbations.any_in_batch(
            PerturbationType.SKIP_V2A_CROSS_ATTN, block_idx
        ):
            v2a_mask = perturbations.mask_like(
                PerturbationType.SKIP_V2A_CROSS_ATTN, block_idx, audio_hidden
            )
            v2a_out = v2a_out * v2a_mask
        audio_hidden = audio_hidden + v2a_out

        video_normed = self._rms_norm(video_hidden) * (1.0 + v_scale_ff) + v_shift_ff
        video_hidden = video_hidden + self.ff(video_normed) * v_gate_ff

        audio_normed = self._rms_norm(audio_hidden) * (1.0 + a_scale_ff) + a_shift_ff
        audio_hidden = audio_hidden + self.audio_ff(audio_normed) * a_gate_ff

        return video_hidden, audio_hidden


class LTX23Model(nn.Module):
    """48-block joint A/V DiT. Forward returns ``(video_velocity, audio_velocity)``."""

    VIDEO_DIM = 4096
    AUDIO_DIM = 2048
    NUM_LAYERS = 48

    def __init__(self, config: LTXConfig | None = None, ctx: RuntimeContext | None = None):
        super().__init__()
        cfg = config or LTXConfig()
        self.config = cfg
        if ctx is None:
            raise RuntimeError("LTX23Model requires RuntimeContext (ctx)")
        self.ctx = ctx
        vd = self.VIDEO_DIM
        ad = self.AUDIO_DIM
        t_dim = 256
        patch_ch = int(getattr(cfg, "dim_in", 128) or 128)
        num_layers = int(getattr(cfg, "depth", self.NUM_LAYERS) or self.NUM_LAYERS)

        self.patchify_proj = nn.Linear(patch_ch, vd)
        self.audio_patchify_proj = nn.Linear(patch_ch, ad)
        self.proj_out = nn.Linear(vd, patch_ch)
        self.audio_proj_out = nn.Linear(ad, patch_ch)
        self.scale_shift_table = mx.zeros((2, vd))
        self.audio_scale_shift_table = mx.zeros((2, ad))

        self.adaln_single = LTX23AdaLayerNormSingle(vd, num_params=9, timestep_dim=t_dim)
        self.audio_adaln_single = LTX23AdaLayerNormSingle(ad, num_params=9, timestep_dim=t_dim)
        self.prompt_adaln_single = LTX23AdaLayerNormSingle(vd, num_params=2, timestep_dim=t_dim)
        self.audio_prompt_adaln_single = LTX23AdaLayerNormSingle(ad, num_params=2, timestep_dim=t_dim)
        self.av_ca_video_scale_shift_adaln_single = LTX23AdaLayerNormSingle(vd, num_params=4, timestep_dim=t_dim)
        self.av_ca_audio_scale_shift_adaln_single = LTX23AdaLayerNormSingle(ad, num_params=4, timestep_dim=t_dim)
        self.av_ca_a2v_gate_adaln_single = LTX23AdaLayerNormSingle(vd, num_params=1, timestep_dim=t_dim)
        self.av_ca_v2a_gate_adaln_single = LTX23AdaLayerNormSingle(ad, num_params=1, timestep_dim=t_dim)

        self.transformer_blocks = [
            LTX23AVBlock(video_dim=vd, audio_dim=ad)
            for _ in range(num_layers)
        ]

        # LTX 2.3 checkpoints ship 1000.0 for both multipliers (dgrauet issue #37).
        self._timestep_scale = 1000.0
        self._av_ca_timestep_scale = 1000.0
        self._norm_eps = 1e-6
        self._rope_theta = 10000.0
        self._rope_type = "split"
        self._positional_max_pos = (20, 2048, 2048)
        self._audio_positional_max_pos = (20,)

    def _embed_timestep_scalar(self, timestep: mx.array) -> mx.array:
        return sinusoidal_timestep_proj(self.ctx, timestep * self._timestep_scale, 256, sin_first=True, flip_sin_to_cos=True)

    def _embed_timestep_per_token(self, per_token_timesteps: mx.array) -> mx.array:
        b, n = per_token_timesteps.shape
        flat = (per_token_timesteps * self._timestep_scale).reshape(-1)
        emb = sinusoidal_timestep_proj(self.ctx, flat, 256, sin_first=True, flip_sin_to_cos=True)
        return emb.reshape(b, n, -1)

    def _adaln_per_token(
        self,
        adaln_module: LTX23AdaLayerNormSingle,
        t_emb_per_token: mx.array,
    ) -> tuple[mx.array, mx.array]:
        b, n, d = t_emb_per_token.shape
        flat = t_emb_per_token.reshape(b * n, d)
        params, embedded = adaln_module(flat)
        return params.reshape(b, n, -1), embedded.reshape(b, n, -1)

    def _compute_rope_freqs(
        self,
        positions: mx.array,
        num_heads: int,
        head_dim: int,
        max_pos_override: list[int] | None = None,
    ) -> tuple[mx.array, mx.array, str]:
        inner_dim = num_heads * head_dim
        if max_pos_override is not None:
            max_pos = max_pos_override
        else:
            max_pos = list(self._positional_max_pos[: positions.shape[-1]])
        return _precompute_rope_freqs(
            positions,
            inner_dim=inner_dim,
            num_heads=num_heads,
            theta=self._rope_theta,
            max_pos=max_pos,
            rope_type=self._rope_type,
        )

    def _output_block(
        self,
        x: mx.array,
        embedded_timestep: mx.array,
        scale_shift_table: mx.array,
        proj: nn.Linear,
    ) -> mx.array:
        if embedded_timestep.ndim == 2:
            embedded_timestep = embedded_timestep[:, None, :]
        scale_shift_values = scale_shift_table[None, None, :, :] + embedded_timestep[:, :, None, :]
        shift = scale_shift_values[:, :, 0, :]
        scale = scale_shift_values[:, :, 1, :]
        x = mx.fast.layer_norm(x, weight=None, bias=None, eps=self._norm_eps)
        x = x * (1.0 + scale) + shift
        return proj(x)

    def __call__(
        self,
        video_latent: mx.array,
        audio_latent: mx.array,
        timestep: mx.array,
        video_text_embeds: mx.array | None = None,
        audio_text_embeds: mx.array | None = None,
        video_positions: mx.array | None = None,
        audio_positions: mx.array | None = None,
        video_attention_mask: mx.array | None = None,
        audio_attention_mask: mx.array | None = None,
        video_timesteps: mx.array | None = None,
        audio_timesteps: mx.array | None = None,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[mx.array, mx.array]:
        video_latent = video_latent.astype(mx.bfloat16)
        audio_latent = audio_latent.astype(mx.bfloat16)
        if video_text_embeds is not None:
            video_text_embeds = video_text_embeds.astype(mx.bfloat16)
        if audio_text_embeds is not None:
            audio_text_embeds = audio_text_embeds.astype(mx.bfloat16)

        video_hidden = self.patchify_proj(video_latent)
        audio_hidden = self.audio_patchify_proj(audio_latent)

        timestep = timestep.astype(mx.bfloat16)
        t_emb = self._embed_timestep_scalar(timestep)

        av_ca_factor = self._av_ca_timestep_scale / self._timestep_scale
        t_emb_av_gate = sinusoidal_timestep_proj(
            self.ctx,
            timestep * self._timestep_scale * av_ca_factor,
            256,
            sin_first=True,
            flip_sin_to_cos=True,
        )

        if video_timesteps is not None:
            vt_emb = self._embed_timestep_per_token(video_timesteps)
            video_adaln_emb, video_embedded_ts = self._adaln_per_token(self.adaln_single, vt_emb)
            av_ca_video_emb, _ = self._adaln_per_token(self.av_ca_video_scale_shift_adaln_single, vt_emb)
        else:
            video_adaln_emb, video_embedded_ts = self.adaln_single(t_emb)
            av_ca_video_emb, _ = self.av_ca_video_scale_shift_adaln_single(t_emb)
        av_ca_a2v_gate_emb, _ = self.av_ca_a2v_gate_adaln_single(t_emb_av_gate)
        video_prompt_emb, _ = self.prompt_adaln_single(t_emb)

        if audio_timesteps is not None:
            at_emb = self._embed_timestep_per_token(audio_timesteps)
            audio_adaln_emb, audio_embedded_ts = self._adaln_per_token(self.audio_adaln_single, at_emb)
            av_ca_audio_emb, _ = self._adaln_per_token(self.av_ca_audio_scale_shift_adaln_single, at_emb)
        else:
            audio_adaln_emb, audio_embedded_ts = self.audio_adaln_single(t_emb)
            av_ca_audio_emb, _ = self.av_ca_audio_scale_shift_adaln_single(t_emb)
        av_ca_v2a_gate_emb, _ = self.av_ca_v2a_gate_adaln_single(t_emb_av_gate)
        audio_prompt_emb, _ = self.audio_prompt_adaln_single(t_emb)

        video_rope_freqs = None
        audio_rope_freqs = None
        if video_positions is not None:
            video_rope_freqs = self._compute_rope_freqs(video_positions, 32, 128)
        if audio_positions is not None:
            audio_rope_freqs = self._compute_rope_freqs(
                audio_positions, 32, 64,
                max_pos_override=list(self._audio_positional_max_pos),
            )

        video_cross_rope_freqs = None
        audio_cross_rope_freqs = None
        cross_pe_max_pos = max(self._positional_max_pos[0], self._audio_positional_max_pos[0])
        if video_positions is not None:
            video_cross_rope_freqs = self._compute_rope_freqs(
                video_positions[:, :, 0:1], 32, 64, max_pos_override=[cross_pe_max_pos],
            )
        if audio_positions is not None:
            audio_cross_rope_freqs = self._compute_rope_freqs(
                audio_positions[:, :, 0:1], 32, 64, max_pos_override=[cross_pe_max_pos],
            )

        for block_idx, block in enumerate(self.transformer_blocks):
            video_hidden, audio_hidden = block(
                video_hidden=video_hidden,
                audio_hidden=audio_hidden,
                video_adaln_params=video_adaln_emb,
                audio_adaln_params=audio_adaln_emb,
                video_prompt_adaln_params=video_prompt_emb,
                audio_prompt_adaln_params=audio_prompt_emb,
                av_ca_video_params=av_ca_video_emb,
                av_ca_audio_params=av_ca_audio_emb,
                av_ca_a2v_gate_params=av_ca_a2v_gate_emb,
                av_ca_v2a_gate_params=av_ca_v2a_gate_emb,
                video_text_embeds=video_text_embeds,
                audio_text_embeds=audio_text_embeds,
                video_rope_freqs=video_rope_freqs,
                audio_rope_freqs=audio_rope_freqs,
                video_cross_rope_freqs=video_cross_rope_freqs,
                audio_cross_rope_freqs=audio_cross_rope_freqs,
                video_attention_mask=video_attention_mask,
                audio_attention_mask=audio_attention_mask,
                block_idx=block_idx,
                perturbations=perturbations,
            )
            if _DIT_EVAL_EVERY > 0 and (block_idx + 1) % _DIT_EVAL_EVERY == 0:
                _mx_eval(video_hidden, audio_hidden)

        video_out = self._output_block(
            video_hidden, video_embedded_ts, self.scale_shift_table, self.proj_out,
        )
        audio_out = self._output_block(
            audio_hidden, audio_embedded_ts, self.audio_scale_shift_table, self.audio_proj_out,
        )
        return video_out, audio_out


class LTX23X0Model(nn.Module):
    """Velocity → x0 wrapper: ``x0 = x_t - sigma * v``."""

    def __init__(self, model: LTX23Model):
        super().__init__()
        self.model = model

    def __call__(
        self,
        video_latent: mx.array,
        audio_latent: mx.array,
        sigma: mx.array,
        video_timesteps: mx.array | None = None,
        audio_timesteps: mx.array | None = None,
        **kwargs: Any,
    ) -> tuple[mx.array, mx.array]:
        video_v, audio_v = self.model(
            video_latent=video_latent,
            audio_latent=audio_latent,
            timestep=sigma,
            video_timesteps=video_timesteps,
            audio_timesteps=audio_timesteps,
            **kwargs,
        )

        if video_timesteps is not None:
            video_sigma = video_timesteps[:, :, None].astype(mx.float32)
        else:
            video_sigma = sigma[:, None, None].astype(mx.float32)

        if audio_timesteps is not None:
            audio_sigma = audio_timesteps[:, :, None].astype(mx.float32)
        else:
            audio_sigma = sigma[:, None, None].astype(mx.float32)

        video_x0 = (
            video_latent.astype(mx.float32) - video_sigma * video_v.astype(mx.float32)
        ).astype(video_latent.dtype)
        audio_x0 = (
            audio_latent.astype(mx.float32) - audio_sigma * audio_v.astype(mx.float32)
        ).astype(audio_latent.dtype)
        return video_x0, audio_x0


# ---------------------------------------------------------------------------
# TransformerBase wrapper (generation_mlx / VideoPipeline internal use)
# ---------------------------------------------------------------------------

def _register_linear(param_map: dict[str, Any], prefix: str, linear: nn.Linear) -> None:
    param_map[f"{prefix}.weight"] = linear.weight
    if linear.bias is not None:
        param_map[f"{prefix}.bias"] = linear.bias


def _register_attention(param_map: dict[str, Any], prefix: str, attn: LTX23Attention) -> None:
    for part in ("to_q", "to_k", "to_v", "to_out"):
        _register_linear(param_map, f"{prefix}.{part}", getattr(attn, part))
    if attn.to_gate_logits is not None:
        _register_linear(param_map, f"{prefix}.to_gate_logits", attn.to_gate_logits)
    param_map[f"{prefix}.q_norm.weight"] = attn.q_norm.weight
    param_map[f"{prefix}.k_norm.weight"] = attn.k_norm.weight


def _register_ff(param_map: dict[str, Any], prefix: str, ff: LTX23FeedForward) -> None:
    _register_linear(param_map, f"{prefix}.proj_in", ff.proj_in)
    _register_linear(param_map, f"{prefix}.proj_out", ff.proj_out)


def _register_adaln(param_map: dict[str, Any], prefix: str, adaln: LTX23AdaLayerNormSingle) -> None:
    te = adaln.emb.timestep_embedder
    _register_linear(param_map, f"{prefix}.emb.timestep_embedder.linear_1", te.linear_1)
    _register_linear(param_map, f"{prefix}.emb.timestep_embedder.linear_2", te.linear_2)
    _register_linear(param_map, f"{prefix}.linear", adaln.linear)


class LTX23Transformer(TransformerBase):
    """LTX 2.3 A/V DiT — ``forward`` accepts patchified ``[B, L, C]`` latents."""

    def __init__(self, config: LTXConfig, ctx: RuntimeContext, num_frames: int = 33):
        self.config = config
        self.ctx = ctx
        self._num_frames = num_frames
        self.model = LTX23Model(config, ctx=ctx)
        self.x0_model = LTX23X0Model(self.model)
        self._param_map: dict[str, Any] = {}
        self._build_param_map()

    def _build_param_map(self) -> None:
        m = self.model
        pm: dict[str, Any] = {}
        _register_linear(pm, "patchify_proj", m.patchify_proj)
        _register_linear(pm, "audio_patchify_proj", m.audio_patchify_proj)
        _register_linear(pm, "proj_out", m.proj_out)
        _register_linear(pm, "audio_proj_out", m.audio_proj_out)
        pm["scale_shift_table"] = m.scale_shift_table
        pm["audio_scale_shift_table"] = m.audio_scale_shift_table

        for name, adaln in (
            ("adaln_single", m.adaln_single),
            ("audio_adaln_single", m.audio_adaln_single),
            ("prompt_adaln_single", m.prompt_adaln_single),
            ("audio_prompt_adaln_single", m.audio_prompt_adaln_single),
            ("av_ca_video_scale_shift_adaln_single", m.av_ca_video_scale_shift_adaln_single),
            ("av_ca_audio_scale_shift_adaln_single", m.av_ca_audio_scale_shift_adaln_single),
            ("av_ca_a2v_gate_adaln_single", m.av_ca_a2v_gate_adaln_single),
            ("av_ca_v2a_gate_adaln_single", m.av_ca_v2a_gate_adaln_single),
        ):
            _register_adaln(pm, name, adaln)

        for i, block in enumerate(m.transformer_blocks):
            bp = f"transformer_blocks.{i}"
            _register_attention(pm, f"{bp}.attn1", block.attn1)
            _register_attention(pm, f"{bp}.audio_attn1", block.audio_attn1)
            _register_attention(pm, f"{bp}.attn2", block.attn2)
            _register_attention(pm, f"{bp}.audio_attn2", block.audio_attn2)
            _register_attention(pm, f"{bp}.audio_to_video_attn", block.audio_to_video_attn)
            _register_attention(pm, f"{bp}.video_to_audio_attn", block.video_to_audio_attn)
            _register_ff(pm, f"{bp}.ff", block.ff)
            _register_ff(pm, f"{bp}.audio_ff", block.audio_ff)
            pm[f"{bp}.scale_shift_table"] = block.scale_shift_table
            pm[f"{bp}.audio_scale_shift_table"] = block.audio_scale_shift_table
            pm[f"{bp}.prompt_scale_shift_table"] = block.prompt_scale_shift_table
            pm[f"{bp}.audio_prompt_scale_shift_table"] = block.audio_prompt_scale_shift_table
            pm[f"{bp}.scale_shift_table_a2v_ca_video"] = block.scale_shift_table_a2v_ca_video
            pm[f"{bp}.scale_shift_table_a2v_ca_audio"] = block.scale_shift_table_a2v_ca_audio

        self._param_map = pm

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Transform checkpoint keys to match ``LTX23Transformer._param_map``.

        Routes LTX 2.3 (48-layer A/V) and legacy 28-layer diffusers checkpoints
        through the appropriate key normalization.
        """
        from backend.engine.families.ltx.weights_mlx import remap_ltx_weights

        return remap_ltx_weights(weights)

    def load_weights(
        self,
        weights,
        strict: bool = False,
        ctx: Any = None,
        *,
        bundle_affine_bits: int | None = None,
        inference_mode=None,
    ):
        load_ctx = ctx if ctx is not None else self.ctx
        if (
            inference_mode is not None
            and getattr(inference_mode, "kind", "dense") == "quantized"
            and getattr(inference_mode, "bits", None) in (4, 8)
        ):
            from backend.engine.common.model.quantized_load_mlx import load_weights_quantized_inference

            return load_weights_quantized_inference(
                self,
                weights,
                strict=strict,
                ctx=load_ctx,
                bundle_affine_bits=bundle_affine_bits,
                bits=int(inference_mode.bits),
                group_size=int(getattr(inference_mode, "group_size", 64) or 64),
                module_root=self.model,
            )
        return super().load_weights(
            weights,
            strict=strict,
            ctx=load_ctx,
            bundle_affine_bits=bundle_affine_bits,
            inference_mode=inference_mode,
        )

    def parameters(self):
        if not self._param_map:
            self._build_param_map()
        return dict(self._param_map)

    def before_denoise(self, latents: Any, timesteps: Any, sigmas: Any, **cond: Any):
        """Inject LTX RoPE interpolation scale when ``_pipeline_fps`` is present."""
        fps = cond.pop("_pipeline_fps", None)
        if fps is not None:
            from backend.engine.families.ltx.pipeline_math_mlx import ltx_rope_interpolation_scale

            temporal = int(getattr(self.config, "temporal_vae_scale", 8) or 8)
            vae_sf = int(getattr(self.config, "vae_scale", 32) or 32)
            cond["ltx_rope_interpolation_scale"] = ltx_rope_interpolation_scale(
                temporal_vae_scale=temporal,
                vae_scale=vae_sf,
                fps=float(fps),
            )
        return latents, cond

    def forward(
        self,
        video_latent: mx.array,
        audio_latent: mx.array | None = None,
        timestep: mx.array | None = None,
        *,
        txt_embeds: mx.array | None = None,
        audio_txt_embeds: mx.array | None = None,
        video_positions: mx.array | None = None,
        audio_positions: mx.array | None = None,
        video_timesteps: mx.array | None = None,
        audio_timesteps: mx.array | None = None,
        predict_x0: bool = False,
        **conditioning: Any,
    ) -> tuple[mx.array, mx.array]:
        if audio_latent is None:
            raise RuntimeError("LTX23Transformer requires audio_latent [B, L, C] (joint A/V DiT).")
        if timestep is None:
            sigma = conditioning.get("sigmas")
            if sigma is None:
                raise RuntimeError("LTX23Transformer requires timestep or conditioning['sigmas'].")
            timestep = sigma

        video_text = conditioning.get("video_text_embeds", txt_embeds)
        audio_text = conditioning.get("audio_text_embeds", audio_txt_embeds)
        video_positions = conditioning.get("video_positions", video_positions)
        audio_positions = conditioning.get("audio_positions", audio_positions)
        video_timesteps = conditioning.get("video_timesteps", video_timesteps)
        audio_timesteps = conditioning.get("audio_timesteps", audio_timesteps)

        fwd_kwargs = dict(
            video_latent=video_latent,
            audio_latent=audio_latent,
            video_text_embeds=video_text,
            audio_text_embeds=audio_text,
            video_positions=video_positions,
            audio_positions=audio_positions,
            video_timesteps=video_timesteps,
            audio_timesteps=audio_timesteps,
            video_attention_mask=conditioning.get("video_attention_mask"),
            audio_attention_mask=conditioning.get("audio_attention_mask"),
        )
        if predict_x0:
            return self.x0_model(sigma=timestep, **fwd_kwargs)
        return self.model(timestep=timestep, **fwd_kwargs)


_LTX23_DEFAULT_TIMESTEP_SCALE = 1000.0
_LTX23_DEFAULT_AV_CA_TIMESTEP_SCALE = 1000.0


def _load_ltx_transformer_checkpoint_dict(bundle_root: Path) -> dict[str, Any]:
    """Read transformer hyperparams from ``embedded_config.json`` / ``config.json``."""
    root = Path(bundle_root)
    for name in ("embedded_config.json", "config.json"):
        path = root / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise RuntimeError(f"LTX 2.3 cannot read transformer config {path}: {e}") from e
        if isinstance(data, dict):
            nested = data.get("transformer", data)
            return nested if isinstance(nested, dict) else data
    raise RuntimeError(
        f"LTX 2.3 transformer config missing under {root} "
        "(expected embedded_config.json or config.json)"
    )


def apply_ltx_transformer_bundle_config(model: LTX23Model, bundle_root: Path) -> None:
    """Apply checkpoint RoPE / timestep scales to an in-memory ``LTX23Model``."""
    t = _load_ltx_transformer_checkpoint_dict(bundle_root)
    model._timestep_scale = float(t.get("timestep_scale_multiplier", _LTX23_DEFAULT_TIMESTEP_SCALE))
    model._av_ca_timestep_scale = float(
        t.get("av_ca_timestep_scale_multiplier", _LTX23_DEFAULT_AV_CA_TIMESTEP_SCALE)
    )
    max_pos = t.get("positional_embedding_max_pos")
    if max_pos is not None:
        model._positional_max_pos = tuple(int(x) for x in max_pos)
    audio_max = t.get("audio_positional_embedding_max_pos")
    if audio_max is not None:
        model._audio_positional_max_pos = tuple(int(x) for x in audio_max)
    if "positional_embedding_theta" in t:
        model._rope_theta = float(t["positional_embedding_theta"])
    if "rope_type" in t:
        model._rope_type = str(t["rope_type"])


def _resolve_bundle_safetensors(bundle_root: Path, stem: str) -> Path:
    from pathlib import Path as _Path

    root = _Path(bundle_root)
    exact = root / f"{stem}.safetensors"
    if exact.is_file():
        return exact
    matches = sorted(root.glob(f"{stem}*.safetensors"))
    if not matches:
        raise RuntimeError(f"LTX 2.3 bundle file missing: {root / (stem + '.safetensors')}")
    return matches[0]


def _read_ltx_quant_group_size(bundle_root: Path, *, default: int = 64) -> int:
    import json

    cfg_path = bundle_root / "quantize_config.json"
    if not cfg_path.is_file():
        return default
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    q = data.get("quantization") if isinstance(data, dict) else None
    if isinstance(q, dict) and q.get("group_size") is not None:
        return int(q["group_size"])
    return default


def load_ltx23_x0_model(
    ctx: RuntimeContext,
    bundle_root: Path,
    config: LTXConfig | None = None,
    *,
    weight_stem: str,
    entry: Any | None = None,
    version_key: str | None = None,
) -> LTX23X0Model:
    """Load LTX 2.3 DiT from bundle safetensors and return an in-repo X0 wrapper."""
    from pathlib import Path as _Path

    from backend.engine.common.bundle.quant_inference import resolve_dit_inference_weight_mode
    from backend.engine.common.bundle.safetensors_affine_quant import read_bundle_affine_bits_if_quantized
    from backend.engine.families.ltx.weights_mlx import load_split_safetensors, remap_ltx23_weights
    from backend.engine.runtime.mlx_runtime import load_weights_dict

    path = _resolve_bundle_safetensors(_Path(bundle_root), weight_stem)
    raw = load_weights_dict(getattr(ctx, "load_weights", None), str(path))
    if not raw:
        raw = load_split_safetensors(path)
    if not raw:
        raise RuntimeError(f"LTX 2.3 transformer weights empty: {path}")

    remapped = remap_ltx23_weights(raw)
    ltx = LTX23Transformer(config or LTXConfig(), ctx)
    apply_ltx_transformer_bundle_config(ltx.model, _Path(bundle_root))
    bundle_affine_bits = read_bundle_affine_bits_if_quantized(remapped, path)
    inference_mode = resolve_dit_inference_weight_mode(
        ctx,
        entry=entry,
        version_key=version_key,
        weight_keys=frozenset(remapped.keys()),
        bundle_affine_bits=bundle_affine_bits,
    )
    ltx.load_weights(
        list(remapped.items()),
        strict=False,
        ctx=ctx,
        bundle_affine_bits=bundle_affine_bits,
        inference_mode=inference_mode,
    )
    setattr(ltx, "_dq_inference_mode", inference_mode)

    _mx_eval(ltx.model.parameters())
    return ltx.x0_model
