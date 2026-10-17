"""
ACE-Step condition encoder + Qwen3 embedding + ``prepare_condition`` (MLX only).

Cover mode uses ``audio_codec_mlx`` tokenize/detokenize when ``is_covers > 0``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.ops.attention import (
    build_causal_with_padding_bias,
    repeat_kv_heads_mx,
    scaled_dot_product_attention_bhsd_mx,
    build_window_with_padding_bias,
    rotate_half,
)
from backend.engine.common.ops.embeddings import build_position_ids_2d
from backend.engine.runtime.mlx import MLXContext
from backend.engine.runtime.mlx_runtime import load_weights_dict, run_eval
from backend.engine.families.ace_step.weights_mlx import load_prefix_weights_for_mlx
from backend.engine.common.codecs.text_encoders.qwen3_mlx import MlxSwiGLUMLP
from backend.engine.families.z_image.text_encoder_mlx import (
    _ZImageEncoderLayer,
    _ZImageEncoderRotaryEmbedding,
)

_MLX_CTX = MLXContext()


def _pack_sequences(
    hidden1: mx.array,
    hidden2: mx.array,
    mask1: mx.array,
    mask2: mx.array,
) -> Tuple[mx.array, mx.array]:
    hidden_cat = mx.concatenate([hidden1, hidden2], axis=1)
    mask_cat = mx.concatenate([mask1, mask2], axis=1)
    bsz, length, dim = hidden_cat.shape
    sort_idx = mx.argsort(-mask_cat, axis=1)
    idx_exp = mx.expand_dims(sort_idx, axis=-1)
    idx_exp = mx.broadcast_to(idx_exp, (bsz, length, dim))
    hidden_left = mx.take_along_axis(hidden_cat, idx_exp, axis=1)
    lengths = mask_cat.sum(axis=1).astype(mx.int32)
    positions = mx.arange(length, dtype=mx.int32)[None, :]
    new_mask = (positions < lengths[:, None]).astype(mx.float32)
    return hidden_left, new_mask


class _ConditionEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        head_dim: int,
        rms_norm_eps: float,
        attention_type: str,
        sliding_window: int,
    ):
        super().__init__()
        self.attention_type = attention_type
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = _ConditionSelfAttention(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
        )
        self.mlp = MlxSwiGLUMLP(hidden_size, intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        attn_mask_map: dict[str, Optional[mx.array]],
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings,
            attn_mask_map.get(self.attention_type),
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class _ConditionSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.n_rep = num_heads // num_kv_heads
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        attention_mask: Optional[mx.array],
    ) -> mx.array:
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_norm(self.q_proj(hidden_states).reshape(bsz, seq_len, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(hidden_states).reshape(bsz, seq_len, self.num_kv_heads, self.head_dim))
        v = self.v_proj(hidden_states).reshape(bsz, seq_len, self.num_kv_heads, self.head_dim)
        cos, sin = position_embeddings
        cos = mx.expand_dims(cos, axis=2)
        sin = mx.expand_dims(sin, axis=2)
        q = (q * cos) + (rotate_half(_MLX_CTX, q) * sin)
        k = (k * cos) + (rotate_half(_MLX_CTX, k) * sin)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = repeat_kv_heads_mx(mx, mx.transpose(k, (0, 2, 1, 3)), self.n_rep)
        v = repeat_kv_heads_mx(mx, mx.transpose(v, (0, 2, 1, 3)), self.n_rep)
        out = scaled_dot_product_attention_bhsd_mx(
            mx,
            q,
            k,
            v,
            scale=self.scale,
            mask=attention_mask,
            compute_dtype=mx.float32,
            out_dtype=hidden_states.dtype,
        )
        out = mx.transpose(out, (0, 2, 1, 3))
        return self.o_proj(out.reshape(bsz, seq_len, -1))


class _LyricEncoderMLX(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self._cfg = config
        hidden = config["hidden_size"]
        self.embed_tokens = nn.Linear(config["text_hidden_dim"], hidden, bias=True)
        self.norm = nn.RMSNorm(hidden, eps=config["rms_norm_eps"])
        self.rotary_emb = _ZImageEncoderRotaryEmbedding(
            dim=config["head_dim"], base=config["rope_theta"]
        )
        layer_types: Sequence[str] = config["layer_types"]
        n_layers = config["num_lyric_encoder_hidden_layers"]
        self.layers = [
            _ConditionEncoderLayer(
                hidden,
                config["num_attention_heads"],
                config["num_key_value_heads"],
                config["intermediate_size"],
                config["head_dim"],
                config["rms_norm_eps"],
                layer_types[i],
                config["sliding_window"],
            )
            for i in range(n_layers)
        ]

    def __call__(self, inputs_embeds: mx.array, attention_mask: mx.array) -> mx.array:
        bsz, seq_len, _ = inputs_embeds.shape
        hidden_states = self.embed_tokens(inputs_embeds)
        position_ids = build_position_ids_2d(mx, bsz, seq_len, dtype=mx.int32)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        full_mask = build_window_with_padding_bias(
            mx,
            seq_len,
            hidden_states.dtype,
            attention_mask=attention_mask,
            sliding_window=None,
            neg_value=-1e9,
            valid_value=1,
        )
        slide_mask = (
            build_window_with_padding_bias(
                mx,
                seq_len,
                hidden_states.dtype,
                attention_mask=attention_mask,
                sliding_window=self._cfg["sliding_window"],
                neg_value=-1e9,
                valid_value=1,
            )
            if self._cfg["use_sliding_window"]
            else None
        )
        attn_map = {"full_attention": full_mask, "sliding_attention": slide_mask}
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, attn_map)
        return self.norm(hidden_states)


class _TimbreEncoderMLX(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self._cfg = config
        hidden = config["hidden_size"]
        self.embed_tokens = nn.Linear(config["timbre_hidden_dim"], hidden, bias=True)
        self.norm = nn.RMSNorm(hidden, eps=config["rms_norm_eps"])
        self.rotary_emb = _ZImageEncoderRotaryEmbedding(
            dim=config["head_dim"], base=config["rope_theta"]
        )
        layer_types: Sequence[str] = config["layer_types"]
        n_layers = config["num_timbre_encoder_hidden_layers"]
        self.layers = [
            _ConditionEncoderLayer(
                hidden,
                config["num_attention_heads"],
                config["num_key_value_heads"],
                config["intermediate_size"],
                config["head_dim"],
                config["rms_norm_eps"],
                layer_types[i],
                config["sliding_window"],
            )
            for i in range(n_layers)
        ]

    def __call__(self, refer_packed: mx.array, refer_order: mx.array) -> Tuple[mx.array, mx.array]:
        del refer_order  # text2music uses one reference per batch row
        bsz, seq_len, _ = refer_packed.shape
        hidden_states = self.embed_tokens(refer_packed)
        position_ids = build_position_ids_2d(mx, bsz, seq_len, dtype=mx.int32)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        attn_mask = mx.ones((bsz, seq_len), dtype=mx.float32)
        full_mask = build_window_with_padding_bias(
            mx,
            seq_len,
            hidden_states.dtype,
            attention_mask=attn_mask,
            sliding_window=None,
            neg_value=-1e9,
            valid_value=1,
        )
        slide_mask = (
            build_window_with_padding_bias(
                mx,
                seq_len,
                hidden_states.dtype,
                attention_mask=attn_mask,
                sliding_window=self._cfg["sliding_window"],
                neg_value=-1e9,
                valid_value=1,
            )
            if self._cfg["use_sliding_window"]
            else None
        )
        attn_map = {"full_attention": full_mask, "sliding_attention": slide_mask}
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, attn_map)
        hidden_states = self.norm(hidden_states)[:, 0, :]
        timbre = mx.expand_dims(hidden_states, axis=1)
        mask = mx.ones((bsz, 1), dtype=mx.float32)
        return timbre, mask


class AceStepConditionEncoderMLX(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.text_projector = nn.Linear(config["text_hidden_dim"], config["hidden_size"], bias=False)
        self.lyric_encoder = _LyricEncoderMLX(config)
        self.timbre_encoder = _TimbreEncoderMLX(config)

    def __call__(
        self,
        text_hidden_states: mx.array,
        text_attention_mask: mx.array,
        lyric_hidden_states: mx.array,
        lyric_attention_mask: mx.array,
        refer_packed: mx.array,
        refer_order: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        text_hidden_states = self.text_projector(text_hidden_states)
        lyric_hidden_states = self.lyric_encoder(lyric_hidden_states, lyric_attention_mask)
        timbre_hidden, timbre_mask = self.timbre_encoder(refer_packed, refer_order)
        enc_hs, enc_mask = _pack_sequences(
            lyric_hidden_states, timbre_hidden, lyric_attention_mask, timbre_mask
        )
        return _pack_sequences(enc_hs, text_hidden_states, enc_mask, text_attention_mask)


@dataclass
class ConditionMlxConfig:
    hidden_size: int
    text_hidden_dim: int
    timbre_hidden_dim: int
    audio_acoustic_hidden_dim: int
    num_lyric_encoder_hidden_layers: int
    num_timbre_encoder_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int
    use_sliding_window: bool
    layer_types: Tuple[str, ...]

    @classmethod
    def from_json(cls, path: Path) -> "ConditionMlxConfig":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        from backend.engine.families.ace_step.generation import ace_step_condition_side_config

        side = ace_step_condition_side_config(raw)
        layer_types = tuple(side["layer_types"])
        return cls(
            hidden_size=side["hidden_size"],
            text_hidden_dim=side["text_hidden_dim"],
            timbre_hidden_dim=side["timbre_hidden_dim"],
            audio_acoustic_hidden_dim=side["audio_acoustic_hidden_dim"],
            num_lyric_encoder_hidden_layers=side["num_lyric_encoder_hidden_layers"],
            num_timbre_encoder_hidden_layers=side["num_timbre_encoder_hidden_layers"],
            num_attention_heads=side["num_attention_heads"],
            num_key_value_heads=side["num_key_value_heads"],
            intermediate_size=side["intermediate_size"],
            head_dim=side["head_dim"],
            rms_norm_eps=side["rms_norm_eps"],
            rope_theta=side["rope_theta"],
            sliding_window=side["sliding_window"],
            use_sliding_window=side.get("use_sliding_window", True),
            layer_types=layer_types,
        )

    def as_dict(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "text_hidden_dim": self.text_hidden_dim,
            "timbre_hidden_dim": self.timbre_hidden_dim,
            "audio_acoustic_hidden_dim": self.audio_acoustic_hidden_dim,
            "num_lyric_encoder_hidden_layers": self.num_lyric_encoder_hidden_layers,
            "num_timbre_encoder_hidden_layers": self.num_timbre_encoder_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "head_dim": self.head_dim,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "sliding_window": self.sliding_window,
            "use_sliding_window": self.use_sliding_window,
            "layer_types": self.layer_types,
        }


def load_condition_encoder_mlx(
    dit_bundle: Path,
    *,
    eval_fn: Any | None = None,
    array_fn: Any | None = None,
    encoder_weights: list[tuple[str, Any]] | None = None,
) -> AceStepConditionEncoderMLX:
    cfg = ConditionMlxConfig.from_json(dit_bundle / "config.json")
    model = AceStepConditionEncoderMLX(cfg.as_dict())
    if encoder_weights is None:
        from backend.engine.families.ace_step.weights_mlx import resolve_dit_checkpoint_path

        weights = load_prefix_weights_for_mlx(
            str(resolve_dit_checkpoint_path(dit_bundle)),
            "encoder.",
            strip_prefix=True,
            array_fn=array_fn,
        )
    else:
        weights = encoder_weights
    model.load_weights(weights, strict=False)
    run_eval(eval_fn, model.parameters())
    return model


def prepare_condition_mlx(
    encoder: AceStepConditionEncoderMLX,
    *,
    text_hidden_states: mx.array,
    text_attention_mask: mx.array,
    lyric_hidden_states: mx.array,
    lyric_attention_mask: mx.array,
    refer_packed: mx.array,
    refer_order: mx.array,
    src_latents: mx.array,
    chunk_masks: mx.array,
    is_covers: mx.array,
    silence_latent: mx.array | None = None,
    attention_mask: mx.array | None = None,
    audio_codec: Any | None = None,
    audio_code_indices: mx.array | None = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Prepare encoder + context latents; cover mode tokenizes ``src_latents`` via audio codec."""
    enc_hs, enc_mask = encoder(
        text_hidden_states,
        text_attention_mask,
        lyric_hidden_states,
        lyric_attention_mask,
        refer_packed,
        refer_order,
    )
    if audio_code_indices is not None:
        if audio_codec is None:
            raise RuntimeError(
                "ACE-Step MLX audio_code_indices require audio_codec to be loaded"
            )
        from backend.engine.families.ace_step.audio.audio_codec_mlx import apply_audio_code_src_latents

        src_latents = apply_audio_code_src_latents(
            audio_codec,
            src_latents=src_latents,
            audio_code_indices=audio_code_indices,
            is_covers=is_covers,
        )
    elif bool(mx.any(is_covers > 0).item()):
        if audio_codec is None or silence_latent is None or attention_mask is None:
            raise RuntimeError(
                "ACE-Step MLX cover mode requires audio_codec, silence_latent, and attention_mask"
            )
        from backend.engine.families.ace_step.audio.audio_codec_mlx import apply_cover_src_latents

        src_latents = apply_cover_src_latents(
            audio_codec,
            hidden_states=src_latents,
            src_latents=src_latents,
            silence_latent=silence_latent,
            attention_mask=attention_mask,
            is_covers=is_covers,
        )
    context_latents = mx.concatenate([src_latents, chunk_masks.astype(src_latents.dtype)], axis=-1)
    return enc_hs, enc_mask, context_latents


# --- Qwen3-Embedding-0.6B (merged from embedding_encoder_mlx) ---


class Qwen3EmbeddingMLX(nn.Module):
    """Qwen3 embedding model — full forward + ``embed_tokens`` for lyrics."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        head_dim: int,
        rope_theta: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            _ZImageEncoderLayer(
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
        self.rotary_emb = _ZImageEncoderRotaryEmbedding(dim=head_dim, base=rope_theta)

    def encode(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        input_ids = input_ids.astype(mx.int32)
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids).astype(mx.float32)
        position_ids = build_position_ids_2d(mx, batch_size, seq_len, dtype=mx.int32)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal_mask = build_causal_with_padding_bias(
            mx,
            attention_mask,
            seq_len,
            hidden_states.dtype,
            valid_value=1,
            neg_value=float("-inf"),
            batch_size=batch_size,
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
            )
        return self.norm(hidden_states).astype(mx.float32)

    def token_embed(self, input_ids: mx.array) -> mx.array:
        return self.embed_tokens(input_ids.astype(mx.int32)).astype(mx.float32)


def load_qwen3_embedding_mlx(
    model_dir: str | Path, *, eval_fn: Any | None = None, load_fn: Any | None = None
) -> Qwen3EmbeddingMLX:
    model_path = Path(model_dir)
    weights: dict[str, mx.array] = {}
    for sf in sorted(model_path.glob("*.safetensors")):
        weights.update(load_weights_dict(load_fn, str(sf)))

    config: dict = {}
    cfg_path = model_path / "config.json"
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            config = json.load(f)

    model = Qwen3EmbeddingMLX(
        vocab_size=config.get("vocab_size", 151669),
        hidden_size=config.get("hidden_size", 1024),
        num_hidden_layers=config.get("num_hidden_layers", 28),
        num_attention_heads=config.get("num_attention_heads", 16),
        num_key_value_heads=config.get("num_key_value_heads", 8),
        intermediate_size=config.get("intermediate_size", 3072),
        head_dim=config.get("head_dim", 128),
        rope_theta=config.get("rope_theta", 1_000_000.0),
        rms_norm_eps=config.get("rms_norm_eps", 1e-6),
    )
    remapped: list[tuple[str, mx.array]] = []
    for key, tensor in weights.items():
        new_key = key[6:] if key.startswith("model.") else key
        remapped.append((new_key, tensor))
    model.load_weights(remapped, strict=False)
    run_eval(eval_fn, model.parameters())
    return model

