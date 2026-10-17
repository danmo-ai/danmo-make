"""
ACE-Step audio codec (tokenizer + detokenizer) for MLX cover mode.

Ports upstream ``AceStepAudioTokenizer`` / ``AudioTokenDetokenizer`` +
``ResidualFSQ`` (inference-only) without ``vector_quantize_pytorch``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.ops.embeddings import build_position_ids_2d
from backend.engine.runtime.mlx_runtime import run_eval
from backend.engine.families.ace_step.audio.condition_mlx import _ConditionEncoderLayer
from backend.engine.families.ace_step.weights_mlx import load_prefix_weights_for_mlx
from backend.engine.families.z_image.text_encoder_mlx import _ZImageEncoderRotaryEmbedding

_BUFFER_SUFFIXES = (
    "_levels",
    "_basis",
    "scales",
    "soft_clamp_input_value",
    "implicit_codebook",
)


def _filter_learned_weights(
    weights: List[Tuple[str, mx.array]],
) -> List[Tuple[str, mx.array]]:
    out: List[Tuple[str, mx.array]] = []
    for key, tensor in weights:
        if any(part in key for part in _BUFFER_SUFFIXES):
            continue
        out.append((key, tensor))
    return out


def _round_ste(z: mx.array) -> mx.array:
    zhat = mx.round(z)
    return z + mx.stop_gradient(zhat - z)


def _floor_ste(z: mx.array) -> mx.array:
    zhat = mx.floor(z)
    return z + mx.stop_gradient(zhat - z)


class _MlxFSQ(nn.Module):
    """Finite scalar quantizer (symmetry-preserving, inference)."""

    def __init__(self, levels: Sequence[int]):
        super().__init__()
        self.levels = tuple(int(v) for v in levels)
        basis = [1]
        for level in self.levels[:-1]:
            basis.append(basis[-1] * level)
        self._levels = mx.array(self.levels, dtype=mx.int32)
        self._basis = mx.array(basis, dtype=mx.int32)
        self.codebook_dim = len(self.levels)
        codebook_size = 1
        for level in self.levels:
            codebook_size *= level
        indices = mx.arange(codebook_size, dtype=mx.int32)
        self.implicit_codebook = self._indices_to_codes(indices)

    def _scale_and_shift(self, zhat_normalized: mx.array) -> mx.array:
        levels_f = self._levels.astype(mx.float32)
        return (zhat_normalized + 1.0) / (2.0 / (levels_f - 1.0))

    def _scale_and_shift_inverse(self, zhat: mx.array) -> mx.array:
        levels_f = self._levels.astype(mx.float32)
        return zhat * (2.0 / (levels_f - 1.0)) - 1.0

    def _indices_to_codes(self, indices: mx.array) -> mx.array:
        idx = mx.expand_dims(indices, axis=-1)
        codes_non_centered = (idx // self._basis) % self._levels
        return self._scale_and_shift_inverse(codes_non_centered.astype(mx.float32))

    def _symmetry_preserving_bound(self, z: mx.array) -> mx.array:
        levels_f = self._levels.astype(mx.float32)
        scale = 2.0 / (levels_f - 1.0)
        bracket = (levels_f - 1.0) * (mx.tanh(z) + 1.0) / 2.0 + 0.5
        bracket = _floor_ste(bracket)
        return scale * bracket - 1.0

    def quantize(self, z: mx.array) -> mx.array:
        return self._symmetry_preserving_bound(z)

    def codes_to_indices(self, codes: mx.array) -> mx.array:
        scaled = self._scale_and_shift(codes)
        return mx.sum(scaled * self._basis.astype(mx.float32), axis=-1).astype(mx.int32)

    def __call__(self, z: mx.array) -> Tuple[mx.array, mx.array]:
        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)
        return codes, indices


class _MlxResidualFSQ(nn.Module):
    def __init__(self, *, dim: int, levels: Sequence[int], num_quantizers: int):
        super().__init__()
        codebook_dim = len(levels)
        self.num_quantizers = num_quantizers
        self.has_projections = codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, codebook_dim, bias=True)
            if self.has_projections
            else None
        )
        self.project_out = (
            nn.Linear(codebook_dim, dim, bias=True)
            if self.has_projections
            else None
        )
        self.layers = [_MlxFSQ(levels) for _ in range(num_quantizers)]
        levels_f = mx.array(list(levels), dtype=mx.float32)
        self.scales = [levels_f ** (-i) for i in range(num_quantizers)]

    def get_output_from_indices(self, indices: mx.array) -> mx.array:
        if indices.ndim == 2:
            indices = mx.expand_dims(indices, axis=-1)
        q = indices.shape[-1]
        codes_summed = mx.zeros((*indices.shape[:-1], self.layers[0].codebook_dim), dtype=mx.float32)
        for qi in range(min(q, self.num_quantizers)):
            layer = self.layers[qi]
            scale = self.scales[qi]
            idx_q = indices[..., qi].astype(mx.int32)
            flat = idx_q.reshape(-1)
            code_flat = layer.implicit_codebook[flat]
            code = code_flat.reshape(*idx_q.shape, layer.codebook_dim)
            codes_summed = codes_summed + code * scale
        if self.project_out is not None:
            return self.project_out(codes_summed)
        return codes_summed

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        if self.project_in is not None:
            x = self.project_in(x)
        quantized_out = mx.zeros_like(x)
        residual = x
        all_indices: list[mx.array] = []
        for qi, (layer, scale) in enumerate(zip(self.layers, self.scales)):
            q, idx = layer(residual / scale)
            q = q * scale
            residual = residual - q
            quantized_out = quantized_out + q
            all_indices.append(idx)
        if self.project_out is not None:
            quantized_out = self.project_out(quantized_out)
        indices = mx.stack(all_indices, axis=-1)
        return quantized_out, indices


class _AttentionPoolerMLX(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self._cfg = config
        hidden = config["hidden_size"]
        self.embed_tokens = nn.Linear(hidden, hidden, bias=True)
        self.norm = nn.RMSNorm(hidden, eps=config["rms_norm_eps"])
        self.rotary_emb = _ZImageEncoderRotaryEmbedding(
            dim=config["head_dim"], base=config["rope_theta"]
        )
        self.special_token = mx.zeros((1, 1, hidden))
        layer_types: Sequence[str] = config["layer_types"]
        n_layers = config["num_attention_pooler_hidden_layers"]
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

    def __call__(self, x: mx.array) -> mx.array:
        bsz, t_len, p_len, dim = x.shape
        x = self.embed_tokens(x)
        special = mx.broadcast_to(self.special_token, (bsz, t_len, 1, dim))
        x = mx.concatenate([special, x], axis=2)
        x = x.reshape(bsz * t_len, p_len + 1, dim)
        position_ids = build_position_ids_2d(mx, bsz * t_len, p_len + 1, dtype=mx.int32)
        position_embeddings = self.rotary_emb(x, position_ids)
        seq_len = p_len + 1
        attn_mask = mx.ones((bsz * t_len, seq_len), dtype=mx.float32)
        from backend.engine.common.ops.attention import build_window_with_padding_bias

        full_mask = build_window_with_padding_bias(
            mx,
            seq_len,
            x.dtype,
            attention_mask=attn_mask,
            sliding_window=None,
            neg_value=-1e9,
            valid_value=1,
        )
        slide_mask = (
            build_window_with_padding_bias(
                mx,
                seq_len,
                x.dtype,
                attention_mask=attn_mask,
                sliding_window=self._cfg["sliding_window"],
                neg_value=-1e9,
                valid_value=1,
            )
            if self._cfg["use_sliding_window"]
            else None
        )
        attn_map = {"full_attention": full_mask, "sliding_attention": slide_mask}
        hidden_states = x
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, attn_map)
        hidden_states = self.norm(hidden_states)
        cls_output = hidden_states[:, 0, :]
        return cls_output.reshape(bsz, t_len, dim)


class _AudioTokenDetokenizerMLX(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self._cfg = config
        hidden = config["hidden_size"]
        pool = config["pool_window_size"]
        self.embed_tokens = nn.Linear(hidden, hidden, bias=True)
        self.norm = nn.RMSNorm(hidden, eps=config["rms_norm_eps"])
        self.rotary_emb = _ZImageEncoderRotaryEmbedding(
            dim=config["head_dim"], base=config["rope_theta"]
        )
        self.special_tokens = mx.zeros((1, pool, hidden))
        layer_types: Sequence[str] = config["layer_types"]
        n_layers = config["num_attention_pooler_hidden_layers"]
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
        self.proj_out = nn.Linear(hidden, config["audio_acoustic_hidden_dim"], bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        bsz, t_len, dim = x.shape
        pool = self._cfg["pool_window_size"]
        x = self.embed_tokens(x)
        x = mx.expand_dims(x, axis=2)
        x = mx.broadcast_to(x, (bsz, t_len, pool, dim))
        special = mx.broadcast_to(self.special_tokens, (bsz, t_len, pool, dim))
        x = x + special
        x = x.reshape(bsz * t_len, pool, dim)
        position_ids = build_position_ids_2d(mx, bsz * t_len, pool, dtype=mx.int32)
        position_embeddings = self.rotary_emb(x, position_ids)
        attn_mask = mx.ones((bsz * t_len, pool), dtype=mx.float32)
        from backend.engine.common.ops.attention import build_window_with_padding_bias

        full_mask = build_window_with_padding_bias(
            mx,
            pool,
            x.dtype,
            attention_mask=attn_mask,
            sliding_window=None,
            neg_value=-1e9,
            valid_value=1,
        )
        slide_mask = (
            build_window_with_padding_bias(
                mx,
                pool,
                x.dtype,
                attention_mask=attn_mask,
                sliding_window=self._cfg["sliding_window"],
                neg_value=-1e9,
                valid_value=1,
            )
            if self._cfg["use_sliding_window"]
            else None
        )
        attn_map = {"full_attention": full_mask, "sliding_attention": slide_mask}
        hidden_states = x
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, attn_map)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states.reshape(bsz, t_len * pool, self._cfg["audio_acoustic_hidden_dim"])


class _AceStepAudioTokenizerMLX(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.pool_window_size = int(config["pool_window_size"])
        self.audio_acoustic_proj = nn.Linear(
            config["audio_acoustic_hidden_dim"], config["hidden_size"], bias=True
        )
        self.attention_pooler = _AttentionPoolerMLX(config)
        self.quantizer = _MlxResidualFSQ(
            dim=config["fsq_dim"],
            levels=config["fsq_input_levels"],
            num_quantizers=config["fsq_input_num_quantizers"],
        )

    def __call__(self, hidden_states: mx.array) -> Tuple[mx.array, mx.array]:
        hidden_states = self.audio_acoustic_proj(hidden_states)
        hidden_states = self.attention_pooler(hidden_states)
        return self.quantizer(hidden_states)


@dataclass
class AceStepAudioCodecMLX:
    config: dict
    tokenizer: _AceStepAudioTokenizerMLX
    detokenizer: _AudioTokenDetokenizerMLX
    pool_window_size: int

    def tokenize(
        self,
        hidden_states: mx.array,
        silence_latent: mx.array,
        attention_mask: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        pool = self.pool_window_size
        seq_len_total = int(hidden_states.shape[1])
        if seq_len_total % pool != 0:
            pad_len = pool - (seq_len_total % pool)
            pad = silence_latent[:1, :pad_len, :]
            pad = mx.broadcast_to(pad, (int(hidden_states.shape[0]), pad_len, int(hidden_states.shape[2])))
            hidden_states = mx.concatenate([hidden_states, pad], axis=1)
            attention_mask = mx.pad(attention_mask, [(0, 0), (0, pad_len)], constant_values=0.0)

        bsz, total_len, dim = hidden_states.shape
        t_patch = total_len // pool
        x = hidden_states.reshape(bsz, t_patch, pool, dim)
        chunk = int(math.ceil(int(attention_mask.shape[1]) / max(t_patch, 1)))
        llm_mask = _max_pool1d(attention_mask, kernel_size=chunk, stride=chunk)
        quantized, indices = self.tokenizer(x)
        return quantized, indices, llm_mask

    def detokenize(self, quantized: mx.array) -> mx.array:
        return self.detokenizer(quantized)


def _max_pool1d(mask_bl: mx.array, *, kernel_size: int, stride: int) -> mx.array:
    bsz, length = mask_bl.shape
    if length == 0:
        return mask_bl
    out_len = (length + stride - 1) // stride
    pad_len = out_len * stride - length
    if pad_len > 0:
        mask_bl = mx.pad(mask_bl, [(0, 0), (0, pad_len)], constant_values=0.0)
    windows = []
    for start in range(0, mask_bl.shape[1], stride):
        chunk = mask_bl[:, start : start + kernel_size]
        windows.append(mx.max(chunk, axis=1, keepdims=True))
    return mx.concatenate(windows, axis=1)


def _align_codec_latents_to_target(
    hints: mx.array,
    target_len: int,
    *,
    pad_from: mx.array,
) -> mx.array:
    """Match codec detokenizer length to DiT ``src_latents`` (pool × codes may be shorter)."""
    cur = int(hints.shape[1])
    if cur == target_len:
        return hints
    if cur > target_len:
        return hints[:, :target_len, :]
    pad = pad_from[:, cur:target_len, :]
    if int(pad.shape[1]) != target_len - cur:
        raise RuntimeError(
            f"ACE-Step audio codec latent align failed: hints={cur}, target={target_len}, "
            f"pad={int(pad.shape[1])}"
        )
    return mx.concatenate([hints, pad], axis=1)


def apply_audio_code_src_latents(
    codec: AceStepAudioCodecMLX,
    *,
    src_latents: mx.array,
    audio_code_indices: mx.array,
    is_covers: mx.array,
) -> mx.array:
    """Replace ``src_latents`` with LM audio-code hints when cover flag is set."""
    audio_code_indices = audio_code_indices.astype(mx.int32)
    if audio_code_indices.ndim == 2:
        audio_code_indices = mx.expand_dims(audio_code_indices, axis=-1)
    lm_hints_5hz = codec.tokenizer.quantizer.get_output_from_indices(audio_code_indices)
    lm_hints_25hz = codec.detokenize(lm_hints_5hz)
    target_len = int(src_latents.shape[1])
    lm_hints_25hz = _align_codec_latents_to_target(
        lm_hints_25hz,
        target_len,
        pad_from=src_latents,
    )
    cover_flag = is_covers.reshape(-1, 1, 1) > 0
    return mx.where(cover_flag, lm_hints_25hz, src_latents)


def apply_cover_src_latents(
    codec: AceStepAudioCodecMLX,
    *,
    hidden_states: mx.array,
    src_latents: mx.array,
    silence_latent: mx.array,
    attention_mask: mx.array,
    is_covers: mx.array,
) -> mx.array:
    lm_hints_5hz, _, _ = codec.tokenize(hidden_states, silence_latent, attention_mask)
    lm_hints_25hz = codec.detokenize(lm_hints_5hz)
    target_len = int(src_latents.shape[1])
    lm_hints_25hz = _align_codec_latents_to_target(
        lm_hints_25hz,
        target_len,
        pad_from=src_latents,
    )
    cover_flag = is_covers.reshape(-1, 1, 1) > 0
    return mx.where(cover_flag, lm_hints_25hz, src_latents)


def _load_special_params(
    weights_path: Path,
    *,
    array_fn: Any | None = None,
) -> dict[str, mx.array]:
    import mlx.core as mx_mod
    import numpy as np

    from backend.engine.families.ace_step.weights_mlx import iter_dit_checkpoint_tensors

    if array_fn is None:
        array_fn = mx_mod.array
    out: dict[str, mx.array] = {}
    wanted = {
        "tokenizer.attention_pooler.special_token",
        "detokenizer.special_tokens",
    }
    for key, arr in iter_dit_checkpoint_tensors(weights_path):
        if key in wanted:
            out[key] = array_fn(np.asarray(arr, dtype=np.float32))
    return out


def load_audio_codec_mlx(
    dit_bundle: Path,
    *,
    eval_fn: Any | None = None,
    array_fn: Any | None = None,
    tokenizer_weights: List[Tuple[str, Any]] | None = None,
    detokenizer_weights: List[Tuple[str, Any]] | None = None,
    special_params: dict[str, Any] | None = None,
) -> AceStepAudioCodecMLX:
    cfg_path = dit_bundle / "config.json"
    with open(cfg_path, encoding="utf-8") as f:
        raw_config = json.load(f)
    from backend.engine.families.ace_step.generation import ace_step_condition_side_config

    # Tokenizer/detokenizer share encoder_* dims (2048 on XL), not DiT hidden_size.
    config = ace_step_condition_side_config(raw_config)
    if tokenizer_weights is None or detokenizer_weights is None:
        from backend.engine.families.ace_step.weights_mlx import resolve_dit_checkpoint_path

        weights_path = resolve_dit_checkpoint_path(dit_bundle)
        if tokenizer_weights is None:
            tokenizer_weights = load_prefix_weights_for_mlx(
                str(weights_path), "tokenizer.", strip_prefix=True, array_fn=array_fn
            )
        if detokenizer_weights is None:
            detokenizer_weights = load_prefix_weights_for_mlx(
                str(weights_path), "detokenizer.", strip_prefix=True, array_fn=array_fn
            )
        if special_params is None:
            special_params = _load_special_params(weights_path, array_fn=array_fn)
    tok_weights = _filter_learned_weights(list(tokenizer_weights))
    det_weights = list(detokenizer_weights)
    tokenizer = _AceStepAudioTokenizerMLX(config)
    detokenizer = _AudioTokenDetokenizerMLX(config)
    tokenizer.load_weights(tok_weights, strict=False)
    detokenizer.load_weights(det_weights, strict=False)
    special = special_params or {}
    if "tokenizer.attention_pooler.special_token" in special:
        tokenizer.attention_pooler.special_token = special[
            "tokenizer.attention_pooler.special_token"
        ]
    if "detokenizer.special_tokens" in special:
        detokenizer.special_tokens = special["detokenizer.special_tokens"]
    run_eval(eval_fn, tokenizer.parameters())
    run_eval(eval_fn, detokenizer.parameters())
    return AceStepAudioCodecMLX(
        config=config,
        tokenizer=tokenizer,
        detokenizer=detokenizer,
        pool_window_size=int(config["pool_window_size"]),
    )
