"""
Z-Image Transformer — PyTorch / CUDA native implementation.

Mirrors ``transformer_mlx.py`` architecture identically, using ``torch.nn``.
All dense attention/MLP operations stay in NLC (batch, seq, channels) layout.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.engine.common.model.base import TransformerBase
from backend.engine.common.ops.attention import scaled_dot_product_attention_bhsd_torch
from backend.engine.config.model_configs import ZImageConfig
from backend.engine.runtime._base import RuntimeContext


# ---------------------------------------------------------------------------
# RMSNorm (compatible with torch < 2.4)
# ---------------------------------------------------------------------------

class _RMSNormCuda(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * norm).to(dtype)


# ---------------------------------------------------------------------------
# RoPE Embedder — complex frequency RoPE
# ---------------------------------------------------------------------------

class _RopeEmbedderCuda:
    """Precompute RoPE frequencies (complex form cos/sin stacked)."""

    def __init__(
        self,
        config_or_dims: ZImageConfig | None = None,
        ctx: RuntimeContext | None = None,
        theta: float = 256.0,
        axes_dims: list[int] | None = None,
        axes_lens: list[int] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        if isinstance(config_or_dims, ZImageConfig):
            self.theta = config_or_dims.rope_theta
            axes_dims = [config_or_dims.rope_dim, 48, 48]
            axes_lens = [1024, 512, 512]
        else:
            self.theta = theta
        axes_dims = axes_dims or [32, 48, 48]
        axes_lens = axes_lens or [1024, 512, 512]
        self.axes_dims = axes_dims
        self.device = device
        self.dtype = dtype
        self.freqs_cis = self._precompute(axes_dims, axes_lens, self.theta)

    def _precompute(self, axes_dims, axes_lens, theta):
        freqs_cis = []
        for d, e in zip(axes_dims, axes_lens):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=self.dtype, device=self.device) / d))
            ts = torch.arange(0, e, dtype=self.dtype, device=self.device)
            freqs = torch.einsum("i,j->ij", ts, freqs)
            cos_f = torch.cos(freqs)
            sin_f = torch.sin(freqs)
            freqs_cis_i = torch.stack([cos_f, sin_f], dim=-1)
            freqs_cis.append(freqs_cis_i)
        return freqs_cis

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [N, 3] int32 -> freqs_cis [N, total_dim] float32 (cos/sin stacked)."""
        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i].long()
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=1)


# ---------------------------------------------------------------------------
# Timestep Embedder — sinusoidal timestep embedding
# ---------------------------------------------------------------------------

class _TimestepEmbedderCuda(nn.Module):
    def __init__(
        self,
        out_size: int,
        mid_size: int = 1024,
        frequency_embedding_size: int = 256,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.linear1 = nn.Linear(frequency_embedding_size, mid_size, bias=True)
        self.linear2 = nn.Linear(mid_size, out_size, bias=True)
        self.device = device
        self.dtype = dtype
        self.to(device=device, dtype=dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Z-Image reference timestep embedding uses [cos, sin] concat order.
        emb = self._sinusoidal_timestep_proj(t, self.frequency_embedding_size, sin_first=False)
        x = self.linear1(emb)
        x = F.silu(x)
        x = self.linear2(x)
        return x

    @staticmethod
    def _sinusoidal_timestep_proj(
        timesteps: torch.Tensor,
        embedding_dim: int,
        sin_first: bool = True,
        max_period: float = 10000.0,
    ) -> torch.Tensor:
        timesteps = timesteps.reshape(-1).float()
        half_dim = embedding_dim // 2
        denom = max(half_dim, 1e-8)
        exp_arg = -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) / denom
        emb_freq = torch.exp(exp_arg)
        emb = timesteps[:, None] * emb_freq[None, :]
        if sin_first:
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        else:
            emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            z = torch.zeros((emb.shape[0], 1), dtype=emb.dtype, device=emb.device)
            emb = torch.cat([emb, z], dim=-1)
        return emb


# ---------------------------------------------------------------------------
# FeedForward — SiLU-gated MLP (SwiGLU variant)
# ---------------------------------------------------------------------------

class _FeedForwardCuda(nn.Module):
    """SwiGLU-style FFN: w2(silu(w1(x)) * w3(x))."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))
        proj = self.w3(x)
        return self.w2(gate * proj)


# ---------------------------------------------------------------------------
# Complex RoPE application
# ---------------------------------------------------------------------------

def _apply_complex_rope_from_cis_bshd(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply complex rotary embedding on [B, S, H, D] with [..., 2] cis table."""
    b, s, h, d = x.shape
    half = d // 2
    x_pairs = x.reshape(b, s, h, half, 2)
    freqs = freqs_cis.reshape(1, s, 1, half, 2)
    x_real = x_pairs[..., 0]
    x_imag = x_pairs[..., 1]
    c_real = freqs[..., 0]
    c_imag = freqs[..., 1]
    out_real = x_real * c_real - x_imag * c_imag
    out_imag = x_real * c_imag + x_imag * c_real
    out = torch.stack([out_real, out_imag], dim=-1)
    return out.reshape(b, s, h, d)


# ---------------------------------------------------------------------------
# Attention — QKV projection + QK Norm + complex RoPE + SDPA
# ---------------------------------------------------------------------------

class _ZImageAttentionCuda(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        qk_norm: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        if qk_norm:
            self.norm_q = _RMSNormCuda(self.head_dim, eps=eps)
            self.norm_k = _RMSNormCuda(self.head_dim, eps=eps)
        else:
            self.norm_q = None
            self.norm_k = None

        self.to(device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)
        q = q.reshape(B, S, self.n_heads, self.head_dim)
        k = k.reshape(B, S, self.n_heads, self.head_dim)
        v = v.reshape(B, S, self.n_heads, self.head_dim)

        if self.norm_q is not None:
            q = self.norm_q(q)
            k = self.norm_k(k)

        if freqs_cis is not None:
            q = _apply_complex_rope_from_cis_bshd(q, freqs_cis)
            k = _apply_complex_rope_from_cis_bshd(k, freqs_cis)

        # [B, S, H, D] -> [B, H, S, D] for SDPA
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if attention_mask is not None:
            # attention_mask: [B, S] bool/byte with 1=valid, 0=pad
            # Convert to additive mask for SDPA
            mask = attention_mask[:, None, None, :S].float()
            mask = mask.masked_fill(mask < 0.5, float("-inf"))
            mask = mask.masked_fill(mask >= 0.5, 0.0)
        else:
            mask = None

        out = scaled_dot_product_attention_bhsd_torch(q, k, v, mask=mask, scale=self.scale)
        out = out.permute(0, 2, 1, 3).reshape(B, S, self.dim)
        out = self.to_out(out)
        return out


# ---------------------------------------------------------------------------
# Context Block — no AdaLN, caption refinement only
# ---------------------------------------------------------------------------

class _ZImageContextBlockCuda(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.attention = _ZImageAttentionCuda(dim, n_heads, device=device, dtype=dtype, qk_norm=qk_norm, eps=norm_eps)
        self.feed_forward = _FeedForwardCuda(dim, int(dim / 3 * 8), device=device, dtype=dtype)
        self.attn_norm1 = _RMSNormCuda(dim, eps=norm_eps)
        self.attn_norm2 = _RMSNormCuda(dim, eps=norm_eps)
        self.ffn_norm1 = _RMSNormCuda(dim, eps=norm_eps)
        self.ffn_norm2 = _RMSNormCuda(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        normed = self.attn_norm1(x)
        attn_out = self.attention(normed, attention_mask=attn_mask, freqs_cis=freqs_cis)
        x = x + self.attn_norm2(attn_out)
        normed = self.ffn_norm1(x)
        ffn_out = self.feed_forward(normed)
        x = x + self.ffn_norm2(ffn_out)
        return x


# ---------------------------------------------------------------------------
# Transformer Block — AdaLN modulation + Attention + FFN
# ---------------------------------------------------------------------------

class _ZImageTransformerBlockCuda(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.attention = _ZImageAttentionCuda(dim, n_heads, device=device, dtype=dtype, qk_norm=qk_norm, eps=norm_eps)
        self.feed_forward = _FeedForwardCuda(dim, int(dim / 3 * 8), device=device, dtype=dtype)
        self.attn_norm1 = _RMSNormCuda(dim, eps=norm_eps)
        self.attn_norm2 = _RMSNormCuda(dim, eps=norm_eps)
        self.ffn_norm1 = _RMSNormCuda(dim, eps=norm_eps)
        self.ffn_norm2 = _RMSNormCuda(dim, eps=norm_eps)
        self.adaLN_modulation = nn.Linear(min(dim, 256), 4 * dim, bias=True)
        self.adaLN_modulation.to(device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        freqs_cis: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        # AdaLN modulation: [B, 4*dim] -> 4 params
        modulation = self.adaLN_modulation(t_emb).reshape(-1, 1, 4 * self.attention.dim)
        scale_msa, gate_msa, scale_mlp, gate_mlp = modulation.chunk(4, dim=-1)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp
        gate_msa = torch.tanh(gate_msa)
        gate_mlp = torch.tanh(gate_mlp)

        # Attention with modulation
        normed = self.attn_norm1(x)
        normed = normed * scale_msa
        attn_out = self.attention(normed, attention_mask=attn_mask, freqs_cis=freqs_cis)
        x = x + gate_msa * self.attn_norm2(attn_out)

        # FFN with modulation
        normed = self.ffn_norm1(x)
        normed = normed * scale_mlp
        ffn_out = self.feed_forward(normed)
        x = x + gate_mlp * self.ffn_norm2(ffn_out)
        return x


# ---------------------------------------------------------------------------
# Final Layer — AdaLN + Linear output
# ---------------------------------------------------------------------------

class _FinalLayerCuda(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Linear(min(hidden_size, 256), hidden_size, bias=True)
        self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.adaLN_modulation(F.silu(c))
        scale = scale.reshape(-1, 1, x.shape[-1])
        return self.linear(self.norm(x) * scale)


# ---------------------------------------------------------------------------
# Pad helpers
# ---------------------------------------------------------------------------

def _pad_len_to_multiple(length: int, multiple: int = 32) -> int:
    return (-int(length)) % int(multiple)


def _build_tail_pad_mask(valid_len: int, pad_len: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    valid = torch.zeros((int(valid_len),), dtype=dtype, device=device) > 0
    if int(pad_len) <= 0:
        return valid
    tail = torch.ones((int(pad_len),), dtype=dtype, device=device) > 0
    return torch.cat([valid, tail], dim=0)


def _pad_tail_with_last(values: torch.Tensor, pad_len: int) -> torch.Tensor:
    if int(pad_len) <= 0:
        return values
    last = values[-1:]
    pad = last.repeat(int(pad_len), *([1] * (values.ndim - 1)))
    return torch.cat([values, pad], dim=0)


def _apply_pad_token(embeds: torch.Tensor, pad_mask: torch.Tensor, pad_token: torch.Tensor) -> torch.Tensor:
    pad_mask = pad_mask.reshape(-1, 1)
    return torch.where(pad_mask, pad_token, embeds)


# ---------------------------------------------------------------------------
# Main DiT Model
# ---------------------------------------------------------------------------

class ZImageDiTCuda(TransformerBase, nn.Module):
    """Z-Image / Z-Image-Turbo Transformer — PyTorch CUDA implementation."""

    def __init__(self, config: ZImageConfig, ctx: RuntimeContext):
        TransformerBase.__init__(self)
        nn.Module.__init__(self)
        self.config = config
        self.ctx = ctx
        self._device = torch.device(getattr(ctx, "_device", "cuda"))
        self._dtype = torch.bfloat16

        dim = config.dim
        n_heads = config.num_heads
        norm_eps = config.norm_eps
        qk_norm = config.qk_norm

        self.in_channels = config.in_channels
        self.out_channels = config.in_channels
        self.patch_size = config.patch_size
        self.f_patch_size = 1
        self.dim = dim
        self.n_heads = n_heads
        self.t_scale = config.t_scale

        embed_dim = self.f_patch_size * self.patch_size * self.patch_size * self.in_channels
        self.x_embedder = nn.Linear(embed_dim, dim, bias=True)
        self.final_layer = _FinalLayerCuda(dim, embed_dim, device=self._device, dtype=self._dtype)

        self.t_embedder = _TimestepEmbedderCuda(
            out_size=min(dim, 256), mid_size=1024, device=self._device, dtype=self._dtype
        )
        self.cap_norm = _RMSNormCuda(config.cap_feat_dim, eps=norm_eps)
        self.cap_embedder = nn.Linear(config.cap_feat_dim, dim, bias=True)
        self.register_buffer("x_pad_token", torch.zeros((1, dim), dtype=self._dtype, device=self._device))
        self.register_buffer("cap_pad_token", torch.zeros((1, dim), dtype=self._dtype, device=self._device))

        self.noise_refiner = nn.ModuleList([
            _ZImageTransformerBlockCuda(dim, n_heads, device=self._device, dtype=self._dtype, norm_eps=norm_eps, qk_norm=qk_norm)
            for _ in range(config.num_refiner_layers)
        ])
        self.context_refiner = nn.ModuleList([
            _ZImageContextBlockCuda(dim, n_heads, device=self._device, dtype=self._dtype, norm_eps=norm_eps, qk_norm=qk_norm)
            for _ in range(config.num_refiner_layers)
        ])
        self.layers = nn.ModuleList([
            _ZImageTransformerBlockCuda(dim, n_heads, device=self._device, dtype=self._dtype, norm_eps=norm_eps, qk_norm=qk_norm)
            for _ in range(config.num_layers)
        ])
        self.rope = _RopeEmbedderCuda(
            config_or_dims=config, device=self._device, dtype=torch.float32
        )

        spec = getattr(config, "latent_noise_dtype", None)
        self._act_dtype = torch.bfloat16 if isinstance(spec, str) and spec.lower() in ("bfloat16", "bf16") else torch.float32

        self._param_map: dict[str, Any] = {}
        self._build_param_map()

        self.x_embedder.to(device=self._device, dtype=self._dtype)
        self.cap_embedder.to(device=self._device, dtype=self._dtype)

    def _build_param_map(self) -> None:
        self._param_map = {}
        for name, param in self.named_parameters():
            self._param_map[name] = param
        for name, buf in self.named_buffers():
            self._param_map[name] = buf

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        from backend.engine.families.z_image.weights import remap_zimage_weights

        patch_size = getattr(self.config, "patch_size", 2)
        return remap_zimage_weights(weights, patch_size=patch_size)

    def parameters(self):
        return list(self._param_map.items())

    def load_weights(
        self,
        weights: list[tuple[str, Any]],
        strict: bool = False,
        ctx: Any = None,
        *,
        bundle_affine_bits: int | None = None,
    ):
        loaded = 0
        skipped = []
        for key, tensor in weights:
            if key in self._param_map:
                param = self._param_map[key]
                if hasattr(tensor, "numpy"):
                    tensor = torch.as_tensor(tensor.numpy(), device=self._device)
                elif not isinstance(tensor, torch.Tensor):
                    import numpy as np
                    tensor = torch.as_tensor(np.asarray(tensor), device=self._device)
                if param.shape != tensor.shape:
                    skipped.append((key, f"shape mismatch: expected {tuple(param.shape)}, got {tuple(tensor.shape)}"))
                    continue
                param.data.copy_(tensor.to(param.dtype))
                loaded += 1
            else:
                skipped.append((key, "not in _param_map"))
        return loaded, skipped

    def combine_cfg_noise(self, noise_cond, noise_uncond, guidance: float):
        """Z-Image reference CFG convention: eps_c + g * (eps_c - eps_u)."""
        return noise_cond + guidance * (noise_cond - noise_uncond)

    def before_denoise(self, latents, timesteps, sigmas, **cond):
        from backend.engine.runtime.mlx_guards import require_mlx_if_option_active

        lemica = cond.get("lemica_mode")
        if lemica is None:
            lemica = getattr(self.config, "lemica_mode", None)
        require_mlx_if_option_active(self.ctx, feature="lemica_mode", option=lemica)
        return super().before_denoise(latents, timesteps, sigmas, **cond)

    def forward(
        self,
        latents,
        timestep,
        txt_embeds=None,
        sigmas=None,
        **conditioning,
    ):
        input_shape = latents.shape
        input_ndim = latents.ndim
        latents_n = self._normalize_latents(latents)
        t = self._resolve_timestep(timestep, sigmas)

        cap_feats = self._resolve_cap_feats(txt_embeds, conditioning)
        output = self._forward_compute(latents_n, t, cap_feats)

        return self._reshape_output(-output, input_shape, input_ndim)

    def _resolve_cap_feats(self, txt_embeds, conditioning):
        cap_feats = txt_embeds if txt_embeds is not None else conditioning.get("cap_feats")
        if cap_feats is None:
            raise ValueError("ZImageDiTCuda requires txt_embeds (Qwen3 cap_feats)")
        if cap_feats.ndim == 3 and cap_feats.shape[0] == 1:
            cap_feats = cap_feats[0]
        return cap_feats

    def _normalize_latents(self, latents):
        if latents.shape[0] == 1 and latents.ndim in (4, 5):
            latents = latents[0]
        if latents.ndim == 3:
            latents = latents.reshape(latents.shape[0], 1, latents.shape[1], latents.shape[2])
        return latents

    def _resolve_timestep(self, timestep, sigmas):
        t = timestep
        idx = None
        if not isinstance(t, bool):
            if isinstance(t, int):
                idx = t
            else:
                try:
                    import numpy as np
                    if isinstance(t, np.integer):
                        idx = int(t)
                except ImportError:
                    pass
                if idx is None and isinstance(t, torch.Tensor) and t.dtype in (torch.int32, torch.int64):
                    idx = int(t.item())
        if idx is not None:
            if sigmas is None:
                raise ValueError("ZImageDiTCuda requires sigmas when timestep is an integer index")
            sigma_t = sigmas[idx].reshape(1)
            t = torch.ones_like(sigma_t) + sigma_t * (-1.0)
        else:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, dtype=torch.float32, device=self._device)
            if t.ndim == 0:
                t = t.reshape(1)
        return t

    def _reshape_output(self, output, input_shape, input_ndim):
        if input_ndim == 4:
            output = output[:, 0, :, :]
            return output.reshape(1, output.shape[0], output.shape[1], output.shape[2])
        if input_ndim == 5:
            return output.reshape(input_shape)
        return output.reshape((1,) + output.shape)

    def _forward_compute(self, latents, t, cap_feats):
        t_emb = self.t_embedder(t * self.t_scale)

        x_emb, cap_emb, x_size, x_pos_ids, cap_pos_ids, x_pad_mask, cap_pad_mask = self._patchify(
            image=latents, cap_feats=cap_feats,
        )

        x_emb = self.x_embedder(x_emb)
        x_emb = _apply_pad_token(x_emb, x_pad_mask, self.x_pad_token)
        x_freqs_cis = self.rope.forward(x_pos_ids)
        x_emb = x_emb.reshape(1, x_emb.shape[0], x_emb.shape[1])

        for layer in self.noise_refiner:
            x_emb = layer(x_emb, None, x_freqs_cis, t_emb)

        cap_emb = self.cap_norm(cap_emb)
        cap_emb = self.cap_embedder(cap_emb)
        cap_emb = _apply_pad_token(cap_emb, cap_pad_mask, self.cap_pad_token)
        cap_freqs_cis = self.rope.forward(cap_pos_ids)
        cap_emb = cap_emb.reshape(1, cap_emb.shape[0], cap_emb.shape[1])

        for layer in self.context_refiner:
            cap_emb = layer(cap_emb, None, cap_freqs_cis)

        x_len = x_emb.shape[1]
        unified = torch.cat([x_emb, cap_emb], dim=1)
        unified_freqs = torch.cat([x_freqs_cis, cap_freqs_cis], dim=0)

        for layer in self.layers:
            unified = layer(unified, None, unified_freqs, t_emb)

        unified = self.final_layer(unified, t_emb)
        return self._unpatchify(unified[0, :x_len], x_size)

    def _patchify(self, image, cap_feats):
        pH = pW = self.patch_size
        pF = self.f_patch_size

        # Caption padding
        cap_ori_len = cap_feats.shape[0]
        cap_pad_len = _pad_len_to_multiple(cap_ori_len, 32)
        cap_pos_ids = self._coord_grid((cap_ori_len + cap_pad_len, 1, 1), (1, 0, 0))
        cap_pos_ids = cap_pos_ids.reshape(-1, 3)
        cap_pad_mask = _build_tail_pad_mask(cap_ori_len, cap_pad_len, self._device)
        cap_padded = _pad_tail_with_last(cap_feats, cap_pad_len)

        # Image patchification
        C, F, H, W = image.shape
        image_size = (F, H, W)
        F_tok, H_tok, W_tok = F // pF, H // pH, W // pW

        img = image.reshape(C, F_tok, pF, H_tok, pH, W_tok, pW)
        img = img.permute(1, 3, 5, 2, 4, 6, 0)
        img = img.reshape(F_tok * H_tok * W_tok, pF * pH * pW * C)

        # Image padding
        img_ori_len = img.shape[0]
        img_pad_len = _pad_len_to_multiple(img_ori_len, 32)
        img_pos_ids = self._coord_grid((F_tok, H_tok, W_tok), (cap_ori_len + cap_pad_len + 1, 0, 0))
        img_pos_ids = img_pos_ids.reshape(-1, 3)

        if img_pad_len > 0:
            img_pos_ids = torch.cat([img_pos_ids, torch.zeros((img_pad_len, 3), dtype=torch.int32, device=self._device)], dim=0)
        img = _pad_tail_with_last(img, img_pad_len)

        img_pad_mask = _build_tail_pad_mask(img_ori_len, img_pad_len, self._device)

        return img, cap_padded, image_size, img_pos_ids, cap_pos_ids, img_pad_mask, cap_pad_mask

    def _unpatchify(self, x, size):
        pH = pW = self.patch_size
        pF = self.f_patch_size
        F, H, W = size
        ori_len = (F // pF) * (H // pH) * (W // pW)
        x = x[:ori_len]
        x = x.reshape(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
        x = x.permute(6, 0, 3, 1, 4, 2, 5)
        return x.reshape(self.out_channels, F, H, W)

    def _coord_grid(self, size, start=None):
        start = start or tuple(0 for _ in size)
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=self._device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(*axes, indexing="ij")
        return torch.stack(grids, dim=-1)
