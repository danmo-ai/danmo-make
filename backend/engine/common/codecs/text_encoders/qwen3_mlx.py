"""Qwen3-family MLX primitives + Z-Image / Flux2 shared text encoder stack."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.ops.attention import (
    build_causal_with_padding_bias,
    rotate_half,
    scaled_dot_product_attention_bhsd_mx,
)
from backend.engine.common.ops.embeddings import build_position_ids_2d
from backend.engine.runtime.mlx_runtime import load_weights_dict
from backend.engine.common.ops.norm import apply_rms_norm
from backend.engine.runtime.mlx import MLXContext

_MLX_CTX = MLXContext()


def llama_swi_glu_hidden_dim(dim: int, hidden_dim: int | None = None) -> int:
    """LLaMA-style SwiGLU hidden size rounded up to a multiple of 256."""
    hidden = hidden_dim if hidden_dim is not None else int(8 * dim / 3)
    return ((hidden + 255) // 256) * 256


class MlxRMSNorm(nn.Module):
    """MLX ``nn.Module`` RMSNorm — shared by Qwen3-family text encoders and Flux2."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return apply_rms_norm(hidden_states, self.weight, self.eps)


Float32RMSNorm = MlxRMSNorm


class MlxTimestepEmbeddingMLP(nn.Module):
    """diffusers ``TimestepEmbedding`` as ``mlx.nn.Module`` (linear → SiLU → linear)."""

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        *,
        linear_1_name: str = "linear_1",
        linear_2_name: str = "linear_2",
    ):
        super().__init__()
        self._linear_1_name = linear_1_name
        self._linear_2_name = linear_2_name
        setattr(self, linear_1_name, nn.Linear(in_channels, time_embed_dim, bias=True))
        setattr(self, linear_2_name, nn.Linear(time_embed_dim, time_embed_dim, bias=True))

    def __call__(self, x: mx.array) -> mx.array:
        linear_1 = getattr(self, self._linear_1_name)
        linear_2 = getattr(self, self._linear_2_name)
        return linear_2(nn.silu(linear_1(x)))


class MlxLTXTimestepEmbeddingMLP(MlxTimestepEmbeddingMLP):
    """LTX timestep MLP — checkpoint keys ``mlp_in`` / ``mlp_out``."""

    def __init__(self, frequency_embedding_size: int, dim: int):
        super().__init__(
            frequency_embedding_size,
            dim,
            linear_1_name="mlp_in",
            linear_2_name="mlp_out",
        )


class MlxTimestepEmbeddingMLPWide(nn.Module):
    """Timestep MLP with configurable expansion (dim → dim×expansion → dim)."""

    def __init__(self, in_channels: int, out_channels: int, *, expansion: int = 4):
        super().__init__()
        hidden = out_channels * expansion
        self.linear1 = nn.Linear(in_channels, hidden, bias=True)
        self.linear2 = nn.Linear(hidden, out_channels, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(nn.silu(self.linear1(x)))


class MlxSwiGLUMLP(nn.Module):
    """SwiGLU FFN; projection attribute names are configurable for checkpoint keys."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        bias: bool = False,
        gate_name: str = "gate_proj",
        up_name: str = "up_proj",
        down_name: str = "down_proj",
    ):
        super().__init__()
        self._gate_name = gate_name
        self._up_name = up_name
        self._down_name = down_name
        setattr(self, gate_name, nn.Linear(dim, hidden_dim, bias=bias))
        setattr(self, up_name, nn.Linear(dim, hidden_dim, bias=bias))
        setattr(self, down_name, nn.Linear(hidden_dim, dim, bias=bias))

    def __call__(self, x: mx.array) -> mx.array:
        gate = getattr(self, self._gate_name)
        up = getattr(self, self._up_name)
        down = getattr(self, self._down_name)
        return down(nn.silu(gate(x)) * up(x))


class LlamaMLP(MlxSwiGLUMLP):
    """SwiGLU MLP with LLaMA hidden-dim rounding (256-byte alignment)."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        bias: bool = False,
    ):
        super().__init__(dim, llama_swi_glu_hidden_dim(dim, hidden_dim), bias=bias)


def _default_rms_norm(dims: int, eps: float = 1e-6):
    return nn.RMSNorm(dims, eps=eps)


class Qwen3EncoderRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=1000000.0):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    def __call__(self, x, position_ids):
        seq_len = position_ids.shape[-1]
        freqs = mx.outer(mx.arange(seq_len, dtype=mx.float32), self.inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)[None, :, :]
        sin = mx.sin(emb)[None, :, :]
        return cos.astype(x.dtype), sin.astype(x.dtype)


_ZImageEncoderRotaryEmbedding = Qwen3EncoderRotaryEmbedding


class Qwen3EncoderAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        *,
        rms_norm_eps: float = 1e-6,
        rms_norm_factory=_default_rms_norm,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = rms_norm_factory(head_dim, rms_norm_eps)
        self.k_norm = rms_norm_factory(head_dim, rms_norm_eps)

    def __call__(self, hidden_states, attention_mask=None, position_embeddings=None):
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            cos = mx.expand_dims(cos, axis=2)
            sin = mx.expand_dims(sin, axis=2)
            q_embed = (q * cos) + (rotate_half(_MLX_CTX, q) * sin)
            k_embed = (k * cos) + (rotate_half(_MLX_CTX, k) * sin)
            q, k = q_embed, k_embed
        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=2)
            v = mx.repeat(v, self.num_kv_groups, axis=2)
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))
        attn_output = scaled_dot_product_attention_bhsd_mx(
            mx,
            q,
            k,
            v,
            scale=self.scale,
            mask=attention_mask,
            compute_dtype=mx.float32,
            out_dtype=q.dtype,
        )
        attn_output = mx.transpose(attn_output, axes=(0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


_ZImageEncoderAttention = Qwen3EncoderAttention
_ZImageEncoderMLP = MlxSwiGLUMLP


class Qwen3EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        intermediate_size,
        head_dim,
        rms_norm_eps=1e-6,
        *,
        rms_norm_factory=_default_rms_norm,
    ):
        super().__init__()
        self.input_layernorm = rms_norm_factory(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = rms_norm_factory(hidden_size, rms_norm_eps)
        self.self_attn = Qwen3EncoderAttention(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps=rms_norm_eps,
            rms_norm_factory=rms_norm_factory,
        )
        self.mlp = MlxSwiGLUMLP(hidden_size, intermediate_size)

    def __call__(self, hidden_states, attention_mask=None, position_embeddings=None):
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states), attention_mask, position_embeddings
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


_ZImageEncoderLayer = Qwen3EncoderLayer


class Qwen3EncoderModel(nn.Module):
    """Qwen3 text encoder MLX model (Z-Image / Flux2 / Qwen-Image shared stack)."""

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2560,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=9728,
        head_dim=128,
        max_position_embeddings=40960,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            Qwen3EncoderLayer(
                hidden_size,
                num_attention_heads,
                num_key_value_heads,
                intermediate_size,
                head_dim,
                rms_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = Qwen3EncoderRotaryEmbedding(dim=head_dim, base=rope_theta)

    def __call__(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids).astype(mx.float32)
        position_ids = Qwen3EncoderModel._build_position_ids(batch_size, seq_len)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal_mask = Qwen3EncoderModel._get_causal_mask(
            attention_mask, batch_size, hidden_states, seq_len
        )
        for layer in self.layers[:-1]:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
            )
        return hidden_states.astype(input_ids.dtype)

    def get_prompt_embeds(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        hidden_state_layers: tuple[int, ...] = (9, 18, 27),
    ) -> mx.array:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids).astype(mx.float32)
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
        position_ids = Qwen3EncoderModel._build_position_ids(batch_size, seq_len)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal_mask = Qwen3EncoderModel._get_causal_mask(
            attention_mask, batch_size, hidden_states, seq_len
        )
        hidden_states_list = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
            )
            hidden_states_list.append(hidden_states)
        _ = self.norm(hidden_states)
        stacked = mx.stack([hidden_states_list[i] for i in hidden_state_layers], axis=1)
        batch_size, num_layers, seq_len, hidden_dim = stacked.shape
        return mx.transpose(stacked, (0, 2, 1, 3)).reshape(
            batch_size, seq_len, num_layers * hidden_dim
        )

    @staticmethod
    def _get_causal_mask(attention_mask, batch_size, hidden_states, seq_len):
        return build_causal_with_padding_bias(
            mx,
            attention_mask,
            seq_len,
            hidden_states.dtype,
            valid_value=1,
            neg_value=float("-inf"),
            batch_size=batch_size,
        )

    @staticmethod
    def _build_position_ids(batch_size: int, seq_len: int):
        return build_position_ids_2d(mx, batch_size, seq_len, dtype=mx.int32)


_ZImageEncoderModel = Qwen3EncoderModel


def _infer_qwen3_encoder_config_from_weights(weights: dict[str, Any]) -> dict[str, Any]:
    """Infer Qwen3 text-encoder dims when ``config.json`` is missing (mlx-community bundles)."""
    config: dict[str, Any] = {}
    embed_scales = weights.get("embed_tokens.scales")
    if embed_scales is not None and hasattr(embed_scales, "shape") and len(embed_scales.shape) >= 1:
        num_groups = int(embed_scales.shape[-1])
        if num_groups > 0:
            config["hidden_size"] = num_groups * 64

    max_layer = -1
    for key in weights:
        if not key.startswith("layers."):
            continue
        part = key.split(".", 2)[1]
        if part.isdigit():
            max_layer = max(max_layer, int(part))
    if max_layer >= 0:
        config["num_hidden_layers"] = max_layer + 1

    gate = weights.get("layers.0.mlp.gate_proj.weight")
    if gate is not None and hasattr(gate, "shape") and len(gate.shape) >= 1:
        config["intermediate_size"] = int(gate.shape[0])

    return config


def build_zimage_mlx_encoder(
    model_path: str,
    ctx: Any,
    *,
    load_fn: Any | None = None,
    registry_entry: Any | None = None,
    registry_version_key: str | None = None,
) -> Qwen3EncoderModel:
    """Load Z-Image Qwen3 text encoder weights from ``model_path`` (safetensors + config.json)."""
    from backend.engine.common.bundle.quant_inference import (
        resolve_component_inference_weight_mode,
        resolve_inference_weight_mode_from_bundle,
    )
    from backend.engine.common.bundle.safetensors_affine_quant import read_bundle_affine_bits_if_quantized
    from backend.engine.common.model.quantized_load_mlx import load_weights_quantized_inference

    model_dir = Path(model_path)
    weights: dict = {}
    for sf in sorted(model_dir.glob("*.safetensors")):
        weights.update(load_weights_dict(load_fn, str(sf)))

    config_path = model_dir / "config.json"
    config: dict = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = _infer_qwen3_encoder_config_from_weights(weights)

    model = Qwen3EncoderModel(
        vocab_size=config.get("vocab_size", 151936),
        hidden_size=config.get("hidden_size", 2560),
        num_hidden_layers=config.get("num_hidden_layers", 36),
        num_attention_heads=config.get("num_attention_heads", 32),
        num_key_value_heads=config.get("num_key_value_heads", 8),
        intermediate_size=config.get("intermediate_size", 9728),
        head_dim=config.get("head_dim", 128),
        max_position_embeddings=config.get("max_position_embeddings", 40960),
        rope_theta=config.get("rope_theta", 1000000.0),
        rms_norm_eps=config.get("rms_norm_eps", 1e-6),
    )

    remapped: dict = {}
    for key, tensor in weights.items():
        new_key = key[6:] if key.startswith("model.") else key
        remapped[new_key] = tensor

    bundle_affine_bits = read_bundle_affine_bits_if_quantized(weights, model_dir)
    if registry_entry is not None:
        te_mode = resolve_component_inference_weight_mode(
            registry_entry,
            registry_version_key,
            ctx,
            component="text_encoder",
            weight_keys=frozenset(remapped.keys()),
            bundle_affine_bits=bundle_affine_bits,
        )
    else:
        te_mode = resolve_inference_weight_mode_from_bundle(
            ctx,
            weight_keys=frozenset(remapped.keys()),
            bundle_affine_bits=bundle_affine_bits,
        )

    if te_mode.kind == "quantized" and te_mode.bits in (4, 8):
        load_weights_quantized_inference(
            model,
            list(remapped.items()),
            strict=False,
            ctx=ctx,
            bundle_affine_bits=bundle_affine_bits,
            bits=int(te_mode.bits),
            group_size=int(te_mode.group_size),
            module_root=model,
        )
    else:
        model.load_weights(list(remapped.items()), strict=False)
    ctx.eval(model.parameters())
    return model


def build_qwen3_mlx_encoder(
    model_path: str,
    ctx: Any,
    *,
    load_fn: Any | None = None,
    registry_entry: Any | None = None,
    registry_version_key: str | None = None,
) -> Qwen3EncoderModel:
    """Load shared Qwen3 MLX encoder (Flux2/Z-Image/Qwen-Image)."""
    return build_zimage_mlx_encoder(
        model_path=model_path,
        ctx=ctx,
        load_fn=load_fn,
        registry_entry=registry_entry,
        registry_version_key=registry_version_key,
    )

