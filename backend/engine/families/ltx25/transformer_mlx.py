"""LTX-2.5 22B joint audio/video DiT — MLX implementation.

A config-driven port of the upstream ``BasicAVTransformerBlock`` /
``LTXModel`` stack (Lightricks LTX-2, 22B "gemma4" generation). All dims and
flags come from the checkpoint's embedded config (materialized into
``bundle_config.json`` by ``ingest.py``) — never hardcoded.

Key 2.5 differences from the in-repo LTX 2.3 DiT:

* ``caption_proj_before_connector=true`` — no caption projection lives in the
  transformer; the Gemma-4 text encoder + connectors supply final context.
* ``ff_bias=false`` — feed-forward linears have no bias.
* ``cross_attention_adaln`` / ``use_prompt_adaln_single`` / ``apply_gated_attention``
  are config-driven (both the 6-param and 9-param block variants are supported).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.model.base import TransformerBase
from backend.engine.common.ops.attention import scaled_dot_product_attention_bhsd_mx
from backend.engine.common.ops.embeddings import sinusoidal_timestep_proj
from backend.engine.config.model_configs import LTX25Config
from backend.engine.families.ltx.perturbations_mlx import BatchedPerturbationConfig, PerturbationType
from backend.engine.families.ltx.transformer_mlx import (
    _apply_rope_split,
    _precompute_rope_freqs,
)
from backend.engine.families.ltx25.pipeline_math_mlx import get_transformer_config
from backend.engine.runtime._base import RuntimeContext
from backend.engine.runtime.mlx_runtime import load_weights_dict, run_eval

# Watchdog guard: flush lazy graph every N blocks (Metal ~10 s deadline).
_DIT_EVAL_EVERY = int(os.environ.get("LTX25_DIT_EVAL_EVERY", "4"))
_mx_eval = getattr(mx, "eval")

_TIMESTEP_DIM = 256
_DEFAULT_TIMESTEP_SCALE = 1000.0
_DEFAULT_AV_CA_TIMESTEP_SCALE = 1.0
_DEFAULT_ROPE_THETA = 10000.0


def _materialize(*arrays: mx.array) -> None:
    run_eval(None, *arrays)


class LTX25TimestepEmbedder(nn.Module):
    """Sinusoidal timestep projection (flip-sin-to-cos, 256 channels)."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, sample: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(sample)))


class LTX25AdaLayerNormSingle(nn.Module):
    """AdaLN-single: sinusoidal proj -> SiLU -> Linear(num_params * dim)."""

    def __init__(self, dim: int, num_params: int = 6):
        super().__init__()
        self.num_params = num_params
        self.emb = LTX25TimestepEmbedder(_TIMESTEP_DIM, dim)
        self.linear = nn.Linear(dim, num_params * dim)

    def __call__(self, timestep_emb: mx.array) -> tuple[mx.array, mx.array]:
        embedded = self.emb(timestep_emb)
        params = self.linear(nn.silu(embedded))
        return params, embedded


class LTX25Attention(nn.Module):
    """Multi-head attention with q/k RMSNorm, optional split-RoPE and per-head gating."""

    def __init__(
        self,
        query_dim: int,
        kv_dim: int | None = None,
        out_dim: int | None = None,
        num_heads: int = 32,
        head_dim: int = 128,
        qkv_bias: bool = True,
        out_bias: bool = True,
        use_rope: bool = True,
        norm_eps: float = 1e-6,
        apply_gated_attention: bool = False,
    ):
        super().__init__()
        kv_dim = kv_dim if kv_dim is not None else query_dim
        out_dim = out_dim if out_dim is not None else query_dim

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_rope = use_rope
        self.scale = head_dim**-0.5

        inner_dim = num_heads * head_dim
        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(kv_dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(kv_dim, inner_dim, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, out_dim, bias=out_bias)
        self.to_gate_logits = nn.Linear(query_dim, num_heads, bias=True) if apply_gated_attention else None
        self.q_norm = nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(inner_dim, eps=norm_eps)

    def __call__(
        self,
        x: mx.array,
        encoder_hidden_states: mx.array | None = None,
        rope_freqs: tuple[mx.array, mx.array] | None = None,
        rope_freqs_k: tuple[mx.array, mx.array] | None = None,
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
            cos_f, sin_f = rope_freqs
            q = _apply_rope_split(q, cos_f, sin_f)
            cos_fk, sin_fk = rope_freqs_k if rope_freqs_k is not None else (cos_f, sin_f)
            k = _apply_rope_split(k, cos_fk, sin_fk)

        out = scaled_dot_product_attention_bhsd_mx(mx, q, k, v, scale=self.scale, mask=attention_mask)

        if perturbation_mask is not None:
            out = out * perturbation_mask + v * (1.0 - perturbation_mask)

        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            gate = 2.0 * mx.sigmoid(gate_logits)
            out = out * gate.transpose(0, 2, 1)[:, :, :, None]

        out = out.transpose(0, 2, 1, 3).reshape(b, -1, self.num_heads * self.head_dim)
        return self.to_out(out)


class LTX25FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int | None = None, mult: float = 4.0, bias: bool = True):
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim
        inner_dim = int(dim * mult)
        self.proj_in = nn.Linear(dim, inner_dim, bias=bias)
        self.proj_out = nn.Linear(inner_dim, dim_out, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj_out(nn.gelu_approx(self.proj_in(x)))


class LTX25AVBlock(nn.Module):
    """Joint audio+video transformer block (config-driven adaln layout)."""

    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        video_num_heads: int,
        audio_num_heads: int,
        video_head_dim: int,
        audio_head_dim: int,
        av_cross_num_heads: int,
        av_cross_head_dim: int,
        ff_mult: float = 4.0,
        norm_eps: float = 1e-6,
        ff_bias: bool = True,
        audio_ff_bias: bool = True,
        apply_gated_attention: bool = False,
        cross_attention_adaln: bool = False,
    ):
        super().__init__()
        self.attn1 = LTX25Attention(
            query_dim=video_dim, num_heads=video_num_heads, head_dim=video_head_dim,
            use_rope=True, norm_eps=norm_eps, apply_gated_attention=apply_gated_attention,
        )
        self.audio_attn1 = LTX25Attention(
            query_dim=audio_dim, num_heads=audio_num_heads, head_dim=audio_head_dim,
            use_rope=True, norm_eps=norm_eps, apply_gated_attention=apply_gated_attention,
        )
        self.attn2 = LTX25Attention(
            query_dim=video_dim, num_heads=video_num_heads, head_dim=video_head_dim,
            use_rope=False, norm_eps=norm_eps, apply_gated_attention=apply_gated_attention,
        )
        self.audio_attn2 = LTX25Attention(
            query_dim=audio_dim, num_heads=audio_num_heads, head_dim=audio_head_dim,
            use_rope=False, norm_eps=norm_eps, apply_gated_attention=apply_gated_attention,
        )
        self.audio_to_video_attn = LTX25Attention(
            query_dim=video_dim, kv_dim=audio_dim, out_dim=video_dim,
            num_heads=av_cross_num_heads, head_dim=av_cross_head_dim,
            use_rope=True, norm_eps=norm_eps, apply_gated_attention=apply_gated_attention,
        )
        self.video_to_audio_attn = LTX25Attention(
            query_dim=audio_dim, kv_dim=video_dim, out_dim=audio_dim,
            num_heads=av_cross_num_heads, head_dim=av_cross_head_dim,
            use_rope=True, norm_eps=norm_eps, apply_gated_attention=apply_gated_attention,
        )
        self.ff = LTX25FeedForward(video_dim, dim_out=video_dim, mult=ff_mult, bias=ff_bias)
        self.audio_ff = LTX25FeedForward(audio_dim, dim_out=audio_dim, mult=ff_mult, bias=audio_ff_bias)

        self._num_adaln_params = 9 if cross_attention_adaln else 6
        self._cross_attention_adaln = cross_attention_adaln
        self.scale_shift_table = mx.zeros((self._num_adaln_params, video_dim))
        self.audio_scale_shift_table = mx.zeros((self._num_adaln_params, audio_dim))
        if cross_attention_adaln:
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

    def _text_cross_attention(
        self,
        x_normed: mx.array,
        context: mx.array,
        attn: LTX25Attention,
        ca_scale: mx.array,
        ca_shift: mx.array,
        ca_gate: mx.array,
        prompt_shift: mx.array | None,
        prompt_scale: mx.array | None,
    ) -> mx.array:
        """Cross-attention with AdaLN modulation (``cross_attention_adaln=true``).

        ``prompt_shift`` / ``prompt_scale`` of ``None`` mean the per-token prompt
        AdaLN MLP is disabled (``use_prompt_adaln_single=false``): only the static
        per-block table modulates K/V.
        """
        attn_input = x_normed * (1.0 + ca_scale) + ca_shift
        if prompt_shift is None:
            encoder_hidden_states = context
        else:
            encoder_hidden_states = context * (1.0 + prompt_scale) + prompt_shift
        return attn(attn_input, encoder_hidden_states=encoder_hidden_states) * ca_gate

    def __call__(
        self,
        video_hidden: mx.array,
        audio_hidden: mx.array,
        video_adaln_params: mx.array,
        audio_adaln_params: mx.array,
        video_prompt_adaln_params: mx.array | None,
        audio_prompt_adaln_params: mx.array | None,
        av_ca_video_params: mx.array,
        av_ca_audio_params: mx.array,
        av_ca_a2v_gate_params: mx.array,
        av_ca_v2a_gate_params: mx.array,
        video_text_embeds: mx.array | None = None,
        audio_text_embeds: mx.array | None = None,
        video_rope_freqs: tuple[mx.array, mx.array] | None = None,
        audio_rope_freqs: tuple[mx.array, mx.array] | None = None,
        video_cross_rope_freqs: tuple[mx.array, mx.array] | None = None,
        audio_cross_rope_freqs: tuple[mx.array, mx.array] | None = None,
        video_attention_mask: mx.array | None = None,
        audio_attention_mask: mx.array | None = None,
        block_idx: int = 0,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[mx.array, mx.array]:
        vdim = video_hidden.shape[-1]
        adim = audio_hidden.shape[-1]

        if self._cross_attention_adaln:
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
        else:
            (
                v_shift_sa, v_scale_sa, v_gate_sa,
                v_shift_ff, v_scale_ff, v_gate_ff,
            ) = self._unpack_adaln(video_adaln_params, self.scale_shift_table, 6, vdim)
            (
                a_shift_sa, a_scale_sa, a_gate_sa,
                a_shift_ff, a_scale_ff, a_gate_ff,
            ) = self._unpack_adaln(audio_adaln_params, self.audio_scale_shift_table, 6, adim)
            v_shift_ca = v_scale_ca = v_gate_ca = None
            a_shift_ca = a_scale_ca = a_gate_ca = None

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
        video_post_sa_normed = self._rms_norm(video_hidden)

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
        audio_post_sa_normed = self._rms_norm(audio_hidden)

        if video_text_embeds is not None:
            if self._cross_attention_adaln:
                vp_shift, vp_scale = self._unpack_adaln(
                    video_prompt_adaln_params, self.prompt_scale_shift_table, 2, vdim,
                )
                video_hidden = video_hidden + self._text_cross_attention(
                    video_post_sa_normed,
                    video_text_embeds,
                    self.attn2,
                    v_scale_ca, v_shift_ca, v_gate_ca,
                    vp_shift, vp_scale,
                )
            else:
                video_hidden = video_hidden + self.attn2(video_post_sa_normed, encoder_hidden_states=video_text_embeds)

        if audio_text_embeds is not None:
            if self._cross_attention_adaln:
                ap_shift, ap_scale = self._unpack_adaln(
                    audio_prompt_adaln_params, self.audio_prompt_scale_shift_table, 2, adim,
                )
                audio_hidden = audio_hidden + self._text_cross_attention(
                    audio_post_sa_normed,
                    audio_text_embeds,
                    self.audio_attn2,
                    a_scale_ca, a_shift_ca, a_gate_ca,
                    ap_shift, ap_scale,
                )
            else:
                audio_hidden = audio_hidden + self.audio_attn2(audio_post_sa_normed, encoder_hidden_states=audio_text_embeds)

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


def _config_int(cfg: dict[str, Any], key: str, default: int, *, required: bool = False) -> int:
    value = cfg.get(key, default)
    if value is None:
        if required:
            raise RuntimeError(f"LTX 2.5 transformer config missing required field {key!r}")
        return default
    return int(value)


def _config_flag(cfg: dict[str, Any], key: str, default: bool) -> bool:
    value = cfg.get(key, default)
    if value is None:
        return bool(default)
    return bool(value)


class LTX25Model(nn.Module):
    """LTX-2.5 22B joint A/V DiT — dims resolved from checkpoint config."""

    def __init__(self, cfg: dict[str, Any], ctx: RuntimeContext | None = None):
        super().__init__()
        if ctx is None:
            raise RuntimeError("LTX25Model requires RuntimeContext (ctx)")
        self.ctx = ctx

        num_heads = _config_int(cfg, "num_attention_heads", 32, required=True)
        head_dim = _config_int(cfg, "attention_head_dim", 128, required=True)
        audio_num_heads = _config_int(cfg, "audio_num_attention_heads", 32, required=True)
        audio_head_dim = _config_int(cfg, "audio_attention_head_dim", 64, required=True)
        num_layers = _config_int(cfg, "num_layers", 48, required=True)
        in_channels = _config_int(cfg, "in_channels", 128)
        out_channels = _config_int(cfg, "out_channels", 128)
        audio_in_channels = _config_int(cfg, "audio_in_channels", 128)
        audio_out_channels = _config_int(cfg, "audio_out_channels", 128)
        video_cross_attention_dim = _config_int(cfg, "cross_attention_dim", 4096)
        audio_cross_attention_dim = _config_int(cfg, "audio_cross_attention_dim", 2048)

        caption_proj_before_connector = _config_flag(cfg, "caption_proj_before_connector", False)
        if not caption_proj_before_connector:
            raise RuntimeError(
                "LTX-2.5 (22B) requires transformer config 'caption_proj_before_connector'=true — "
                "the text projection lives inside the Gemma-4 checkpoint. Refusing to build."
            )

        ff_bias = _config_flag(cfg, "ff_bias", False)
        audio_ff_bias = _config_flag(cfg, "audio_ff_bias", True)
        self.audio_ff_bias = audio_ff_bias
        apply_gated_attention = _config_flag(cfg, "apply_gated_attention", False)
        self.cross_attention_adaln = _config_flag(cfg, "cross_attention_adaln", False)
        self.use_prompt_adaln_single = _config_flag(cfg, "use_prompt_adaln_single", True)

        video_dim = num_heads * head_dim
        audio_dim = audio_num_heads * audio_head_dim
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.audio_num_heads = audio_num_heads
        self.audio_head_dim = audio_head_dim

        self._timestep_scale = float(cfg.get("timestep_scale_multiplier", _DEFAULT_TIMESTEP_SCALE))
        self._av_ca_timestep_scale = float(cfg.get("av_ca_timestep_scale_multiplier", _DEFAULT_AV_CA_TIMESTEP_SCALE))
        self._rope_theta = float(cfg.get("positional_embedding_theta", _DEFAULT_ROPE_THETA))
        self._rope_type = str(cfg.get("rope_type", "split"))
        self._norm_eps = 1e-6
        self._positional_max_pos = list(cfg.get("positional_embedding_max_pos", [20, 2048, 2048]))
        self._audio_positional_max_pos = list(cfg.get("audio_positional_embedding_max_pos", [20]))
        self._use_middle_indices_grid = _config_flag(cfg, "use_middle_indices_grid", True)

        num_adaln_params = 9 if self.cross_attention_adaln else 6
        self.patchify_proj = nn.Linear(in_channels, video_dim, bias=True)
        self.audio_patchify_proj = nn.Linear(audio_in_channels, audio_dim, bias=True)
        self.proj_out = nn.Linear(video_dim, out_channels, bias=True)
        self.audio_proj_out = nn.Linear(audio_dim, audio_out_channels, bias=True)
        self.scale_shift_table = mx.zeros((2, video_dim))
        self.audio_scale_shift_table = mx.zeros((2, audio_dim))

        self.adaln_single = LTX25AdaLayerNormSingle(video_dim, num_params=num_adaln_params)
        self.audio_adaln_single = LTX25AdaLayerNormSingle(audio_dim, num_params=num_adaln_params)
        self.prompt_adaln_single = (
            LTX25AdaLayerNormSingle(video_dim, num_params=2)
            if self.cross_attention_adaln and self.use_prompt_adaln_single
            else None
        )
        self.audio_prompt_adaln_single = (
            LTX25AdaLayerNormSingle(audio_dim, num_params=2)
            if self.cross_attention_adaln and self.use_prompt_adaln_single
            else None
        )
        self.av_ca_video_scale_shift_adaln_single = LTX25AdaLayerNormSingle(video_dim, num_params=4)
        self.av_ca_audio_scale_shift_adaln_single = LTX25AdaLayerNormSingle(audio_dim, num_params=4)
        self.av_ca_a2v_gate_adaln_single = LTX25AdaLayerNormSingle(video_dim, num_params=1)
        self.av_ca_v2a_gate_adaln_single = LTX25AdaLayerNormSingle(audio_dim, num_params=1)

        self.transformer_blocks = [
            LTX25AVBlock(
                video_dim=video_dim,
                audio_dim=audio_dim,
                video_num_heads=num_heads,
                audio_num_heads=audio_num_heads,
                video_head_dim=head_dim,
                audio_head_dim=audio_head_dim,
                av_cross_num_heads=audio_num_heads,
                av_cross_head_dim=audio_head_dim,
                norm_eps=self._norm_eps,
                ff_bias=ff_bias,
                audio_ff_bias=audio_ff_bias,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            for _ in range(num_layers)
        ]

        self.video_cross_attention_dim = video_cross_attention_dim
        self.audio_cross_attention_dim = audio_cross_attention_dim

    def _embed_timestep_scalar(self, timestep: mx.array) -> mx.array:
        return sinusoidal_timestep_proj(
            self.ctx, timestep * self._timestep_scale, _TIMESTEP_DIM, sin_first=True, flip_sin_to_cos=True,
        )

    def _embed_timestep_per_token(self, per_token_timesteps: mx.array) -> mx.array:
        b, n = per_token_timesteps.shape
        flat = (per_token_timesteps * self._timestep_scale).reshape(-1)
        emb = sinusoidal_timestep_proj(self.ctx, flat, _TIMESTEP_DIM, sin_first=True, flip_sin_to_cos=True)
        return emb.reshape(b, n, -1)

    def _adaln_per_token(
        self,
        adaln_module: LTX25AdaLayerNormSingle,
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
    ) -> tuple[mx.array, mx.array]:
        inner_dim = num_heads * head_dim
        max_pos = max_pos_override if max_pos_override is not None else list(
            self._positional_max_pos[: positions.shape[-1]]
        )
        cos_freqs, sin_freqs, _ = _precompute_rope_freqs(
            positions,
            inner_dim=inner_dim,
            num_heads=num_heads,
            theta=self._rope_theta,
            max_pos=max_pos,
            rope_type=self._rope_type,
        )
        return cos_freqs, sin_freqs

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
            _TIMESTEP_DIM,
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
        video_prompt_emb = (
            self.prompt_adaln_single(t_emb)[0]
            if self.prompt_adaln_single is not None
            else None
        )

        if audio_timesteps is not None:
            at_emb = self._embed_timestep_per_token(audio_timesteps)
            audio_adaln_emb, audio_embedded_ts = self._adaln_per_token(self.audio_adaln_single, at_emb)
            av_ca_audio_emb, _ = self._adaln_per_token(self.av_ca_audio_scale_shift_adaln_single, at_emb)
        else:
            audio_adaln_emb, audio_embedded_ts = self.audio_adaln_single(t_emb)
            av_ca_audio_emb, _ = self.av_ca_audio_scale_shift_adaln_single(t_emb)
        av_ca_v2a_gate_emb, _ = self.av_ca_v2a_gate_adaln_single(t_emb_av_gate)
        audio_prompt_emb = (
            self.audio_prompt_adaln_single(t_emb)[0]
            if self.audio_prompt_adaln_single is not None
            else None
        )

        video_rope_freqs = None
        audio_rope_freqs = None
        if video_positions is not None:
            video_rope_freqs = self._compute_rope_freqs(
                video_positions, self.num_heads, self.head_dim,
            )
        if audio_positions is not None:
            audio_rope_freqs = self._compute_rope_freqs(
                audio_positions, self.audio_num_heads, self.audio_head_dim,
                max_pos_override=list(self._audio_positional_max_pos),
            )

        video_cross_rope_freqs = None
        audio_cross_rope_freqs = None
        cross_pe_max_pos = max(self._positional_max_pos[0], self._audio_positional_max_pos[0])
        if video_positions is not None:
            video_cross_rope_freqs = self._compute_rope_freqs(
                video_positions[:, :, 0:1], self.audio_num_heads, self.audio_head_dim,
                max_pos_override=[cross_pe_max_pos],
            )
        if audio_positions is not None:
            audio_cross_rope_freqs = self._compute_rope_freqs(
                audio_positions[:, :, 0:1], self.audio_num_heads, self.audio_head_dim,
                max_pos_override=[cross_pe_max_pos],
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


class LTX25X0Model(nn.Module):
    """Velocity → x0 wrapper: ``x0 = x_t - sigma * v`` (per-token sigmas supported)."""

    def __init__(self, model: LTX25Model):
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

        video_sigma = (
            video_timesteps[:, :, None].astype(mx.float32)
            if video_timesteps is not None
            else sigma[:, None, None].astype(mx.float32)
        )
        audio_sigma = (
            audio_timesteps[:, :, None].astype(mx.float32)
            if audio_timesteps is not None
            else sigma[:, None, None].astype(mx.float32)
        )

        video_x0 = (
            video_latent.astype(mx.float32) - video_sigma * video_v.astype(mx.float32)
        ).astype(video_latent.dtype)
        audio_x0 = (
            audio_latent.astype(mx.float32) - audio_sigma * audio_v.astype(mx.float32)
        ).astype(audio_latent.dtype)
        return video_x0, audio_x0


# ---------------------------------------------------------------------------
# TransformerBase wrapper (family generator internal use)
# ---------------------------------------------------------------------------

def _register_linear(param_map: dict[str, Any], prefix: str, linear: nn.Linear) -> None:
    param_map[f"{prefix}.weight"] = linear.weight
    bias = getattr(linear, "bias", None)
    if bias is not None:
        param_map[f"{prefix}.bias"] = bias


def _register_attention(param_map: dict[str, Any], prefix: str, attn: LTX25Attention) -> None:
    for part in ("to_q", "to_k", "to_v", "to_out"):
        _register_linear(param_map, f"{prefix}.{part}", getattr(attn, part))
    if attn.to_gate_logits is not None:
        _register_linear(param_map, f"{prefix}.to_gate_logits", attn.to_gate_logits)
    param_map[f"{prefix}.q_norm.weight"] = attn.q_norm.weight
    param_map[f"{prefix}.k_norm.weight"] = attn.k_norm.weight


def _register_ff(param_map: dict[str, Any], prefix: str, ff: LTX25FeedForward) -> None:
    _register_linear(param_map, f"{prefix}.proj_in", ff.proj_in)
    _register_linear(param_map, f"{prefix}.proj_out", ff.proj_out)


def _register_adaln(param_map: dict[str, Any], prefix: str, adaln: LTX25AdaLayerNormSingle) -> None:
    te = adaln.emb
    _register_linear(param_map, f"{prefix}.emb.linear_1", te.linear_1)
    _register_linear(param_map, f"{prefix}.emb.linear_2", te.linear_2)
    _register_linear(param_map, f"{prefix}.linear", adaln.linear)


class LTX25Transformer(TransformerBase):
    """LTX-2.5 22B A/V DiT — ``forward`` accepts patchified ``[B, L, C]`` latents."""

    def __init__(self, config: LTX25Config, ctx: RuntimeContext, bundle_root: Path, num_frames: int = 33):
        self.config = config
        self.ctx = ctx
        self.bundle_root = Path(bundle_root)
        self._num_frames = num_frames
        self.model = LTX25Model(get_transformer_config(self.bundle_root), ctx=ctx)
        self.x0_model = LTX25X0Model(self.model)
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
            if adaln is not None:
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
            if block._cross_attention_adaln:
                pm[f"{bp}.prompt_scale_shift_table"] = block.prompt_scale_shift_table
                pm[f"{bp}.audio_prompt_scale_shift_table"] = block.audio_prompt_scale_shift_table
            pm[f"{bp}.scale_shift_table_a2v_ca_video"] = block.scale_shift_table_a2v_ca_video
            pm[f"{bp}.scale_shift_table_a2v_ca_audio"] = block.scale_shift_table_a2v_ca_audio

        self._param_map = pm

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        from backend.engine.families.ltx25.weights_mlx import remap_ltx25_weights

        return remap_ltx25_weights(weights)

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
            raise RuntimeError("LTX25Transformer requires audio_latent [B, L, C] (joint A/V DiT).")
        if timestep is None:
            sigma = conditioning.get("sigmas")
            if sigma is None:
                raise RuntimeError("LTX25Transformer requires timestep or conditioning['sigmas'].")
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


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

_DIT_WEIGHT_FILE = "transformer.safetensors"


def _load_transformer_checkpoint_dict(bundle_root: Path, load_fn: Any | None = None) -> dict[str, mx.array]:
    path = Path(bundle_root) / _DIT_WEIGHT_FILE
    if not path.is_file():
        raise RuntimeError(f"LTX 2.5 transformer weights missing: {path}")
    return load_weights_dict(load_fn, str(path))


def load_ltx25_x0_model(
    ctx: RuntimeContext,
    bundle_root: Path,
    config: LTX25Config,
    *,
    entry: Any | None = None,
    version_key: str | None = None,
    load_fn: Any | None = None,
    on_log: Any | None = None,
) -> LTX25X0Model:
    from backend.engine.common.bundle.quant_inference import resolve_dit_inference_weight_mode
    from backend.engine.common.bundle.safetensors_affine_quant import read_bundle_affine_bits_if_quantized

    transformer = LTX25Transformer(config, ctx, bundle_root)
    raw_weights = _load_transformer_checkpoint_dict(bundle_root, load_fn=load_fn)
    weights = transformer.sanitize(raw_weights)
    bundle_affine_bits = read_bundle_affine_bits_if_quantized(weights, Path(bundle_root))
    inference_mode = resolve_dit_inference_weight_mode(
        ctx,
        entry=entry,
        version_key=version_key,
        weight_keys=set(weights.keys()),
        bundle_affine_bits=bundle_affine_bits,
    )
    _ = on_log
    transformer.load_weights(
        list(weights.items()),
        strict=False,
        bundle_affine_bits=bundle_affine_bits,
        inference_mode=inference_mode,
    )
    _materialize(transformer.model.scale_shift_table)
    return transformer.x0_model
