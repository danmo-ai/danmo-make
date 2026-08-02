"""HF ``GlmModel`` stack for CogView4 — native MLX (penultimate hidden state)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention as mlx_lm_sdpa


@dataclass
class GlmEncoderConfig:
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    attention_bias: bool
    partial_rotary_factor: float
    rope_theta: float


def load_glm_encoder_config(model_path: str | Path) -> GlmEncoderConfig:
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.is_file():
        raise RuntimeError(f"CogView4 GLM encoder: missing config.json under {model_path}")
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    head_dim = int(data.get("head_dim") or data["hidden_size"] // data["num_attention_heads"])
    rope = data.get("rope_parameters") or {}
    partial = data.get("partial_rotary_factor", rope.get("partial_rotary_factor", 1.0))
    return GlmEncoderConfig(
        hidden_size=int(data["hidden_size"]),
        num_hidden_layers=int(data["num_hidden_layers"]),
        intermediate_size=int(data["intermediate_size"]),
        num_attention_heads=int(data["num_attention_heads"]),
        num_key_value_heads=int(data.get("num_key_value_heads", data["num_attention_heads"])),
        head_dim=head_dim,
        rms_norm_eps=float(data["rms_norm_eps"]),
        vocab_size=int(data["vocab_size"]),
        attention_bias=bool(data.get("attention_bias", False)),
        partial_rotary_factor=float(partial),
        rope_theta=float(data.get("rope_theta", rope.get("rope_theta", 10000.0))),
    )


class _GlmAttention(nn.Module):
    def __init__(self, cfg: GlmEncoderConfig):
        super().__init__()
        self.head_dim = cfg.head_dim
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            cfg.hidden_size,
            cfg.num_attention_heads * self.head_dim,
            bias=cfg.attention_bias,
        )
        self.k_proj = nn.Linear(
            cfg.hidden_size,
            cfg.num_key_value_heads * self.head_dim,
            bias=cfg.attention_bias,
        )
        self.v_proj = nn.Linear(
            cfg.hidden_size,
            cfg.num_key_value_heads * self.head_dim,
            bias=cfg.attention_bias,
        )
        self.o_proj = nn.Linear(
            cfg.num_attention_heads * self.head_dim,
            cfg.hidden_size,
            bias=False,
        )
        rope_dims = int(self.head_dim * cfg.partial_rotary_factor)
        self.rope = nn.RoPE(
            dims=rope_dims,
            base=cfg.rope_theta,
            traditional=True,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        b, seq_len, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = queries.reshape(b, seq_len, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(b, seq_len, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(b, seq_len, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        output = mlx_lm_sdpa(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(b, seq_len, -1)
        return self.o_proj(output)


class _GlmMLP(nn.Module):
    def __init__(self, cfg: GlmEncoderConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate_up = self.gate_up_proj(x)
        gate, up = mx.split(gate_up, 2, axis=-1)
        return self.down_proj(swiglu(gate, up))


class _GlmBlock(nn.Module):
    def __init__(self, cfg: GlmEncoderConfig):
        super().__init__()
        self.self_attn = _GlmAttention(cfg)
        self.mlp = _GlmMLP(cfg)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        residual = x
        x = self.input_layernorm(x)
        x = residual + self.self_attn(x, mask, cache)
        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)
        return x


class GlmEncoderModel(nn.Module):
    """Matches HF ``GlmModel`` flat keys — returns penultimate layer output."""

    def __init__(self, cfg: GlmEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [_GlmBlock(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def encode_penultimate(self, input_ids: mx.array) -> mx.array:
        hidden = self.embed_tokens(input_ids)
        mask = create_attention_mask(hidden, None)
        for layer in self.layers[:-1]:
            hidden = layer(hidden, mask)
        return hidden.astype(mx.bfloat16)


def _load_glm_weight_dict(model_path: str | Path, load_fn: Any | None) -> dict[str, mx.array]:
    from backend.engine.runtime.mlx_runtime import load_weights_dict

    root = Path(model_path)
    weights: dict[str, mx.array] = {}
    for sf in sorted(root.glob("*.safetensors")):
        weights.update(load_weights_dict(load_fn, str(sf)))
    if not weights:
        raise RuntimeError(f"CogView4 GLM encoder: no safetensors under {root}")
    return {
        k: v
        for k, v in weights.items()
        if "self_attn.rotary_emb.inv_freq" not in k and not k.startswith("lm_head.")
    }


def build_glm_encoder_mlx(
    model_path: str,
    ctx: Any,
    *,
    load_fn: Any | None = None,
) -> GlmEncoderModel:
    cfg = load_glm_encoder_config(model_path)
    model = GlmEncoderModel(cfg)
    weights = _load_glm_weight_dict(model_path, load_fn)
    model.load_weights(list(weights.items()), strict=False)
    ctx.eval(model.parameters())
    return model
