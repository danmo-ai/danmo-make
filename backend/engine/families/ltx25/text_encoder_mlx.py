"""LTX-2.5 prompt encoder — Gemma 4 (mlx-lm) + V2 feature extractor + 1D connectors.

Pipeline (mirrors upstream ``PromptEncoder``):

1. Gemma 4 unified 12B (the LTX fine-tune ``gemma4-12b-with-proj-ltx-2.5``)
   runs a masked forward and returns all layer hidden states.
2. ``FeatureExtractorV2`` applies per-token RMS normalization over the layer
   stack, concatenates ``[B, T, D*L]``, rescales, then projects through the
   video / audio aggregate-embed linears (the "text embedding projection").
3. The video / audio ``Embeddings1DConnector`` (2-layer 1D transformer blocks
   with RoPE + learnable registers) produce the final cross-attention context.

All dims come from ``bundle_config.json`` (transformer + gemma sections).
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from backend.engine.config.model_configs import LTX25Config
from backend.engine.families.ltx25.pipeline_math_mlx import (
    get_gemma_config,
    get_transformer_config,
)
from backend.engine.runtime.mlx_runtime import load_weights_dict, run_eval

_DEFAULT_MAX_LENGTH = 1024


def _materialize(*arrays: mx.array) -> None:
    run_eval(None, *arrays)


def _rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x, weight=None, eps=eps)


def _apply_rope_split(x: mx.array, cos_f: mx.array, sin_f: mx.array) -> mx.array:
    """RoPE SPLIT type on ``(B, heads, N, head_dim)``."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rot1 = x1 * cos_f - x2 * sin_f
    rot2 = x1 * sin_f + x2 * cos_f
    return mx.concatenate([rot1, rot2], axis=-1)


def _precompute_connector_rope(
    seq_len: int,
    *,
    inner_dim: int,
    num_heads: int,
    max_pos: float,
) -> tuple[mx.array, mx.array]:
    """RoPE for 1D connectors (upstream ``precompute_freqs_cis``, 1 pos dim)."""
    positions = mx.arange(seq_len).astype(mx.float32)[None, :, None]
    num_pos_dims = 1
    n_elem = 2 * num_pos_dims
    num_freqs = inner_dim // n_elem
    theta = 10000.0
    freq_indices = theta ** mx.linspace(
        math.log(1.0) / math.log(theta),
        math.log(theta) / math.log(theta),
        num_freqs,
    ).astype(mx.float32)
    frac = positions.astype(mx.float32) / float(max_pos)
    scaled = freq_indices * (frac * 2.0 - 1.0)
    freqs = scaled.reshape(1, seq_len, -1)
    expected = inner_dim // 2
    pad_size = expected - freqs.shape[-1]
    if pad_size > 0:
        freqs = mx.concatenate([mx.zeros((1, seq_len, pad_size)), freqs], axis=-1)
    head_dim_half = inner_dim // (2 * num_heads)
    cos_f = mx.cos(freqs).reshape(1, seq_len, num_heads, head_dim_half).transpose(0, 2, 1, 3)
    sin_f = mx.sin(freqs).reshape(1, seq_len, num_heads, head_dim_half).transpose(0, 2, 1, 3)
    return cos_f, sin_f


# ---------------------------------------------------------------------------
# Gemma 4 language model (mlx-lm)
# ---------------------------------------------------------------------------


class _Gemma4LanguageModel:
    """Gemma 4 via mlx-lm — extracts all layer hidden states.

    Replicates ``Gemma4TextModel.__call__`` (shared-KV layer routing, per-layer
    masks) so hidden states are captured per layer while a padded causal mask is
    respected. Uses the mlx-lm ``create_causal_mask`` with left-padding support.
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def load(self, model_path: str) -> None:
        from mlx_lm import load as mlx_lm_load

        self._model, self._tokenizer = mlx_lm_load(model_path)

    def tokenize(self, text: str, max_length: int) -> tuple[mx.array, mx.array]:
        if self._tokenizer is None:
            raise RuntimeError("Gemma not loaded")
        tokens = self._tokenizer.encode(text.strip())
        # Gemma 4's tokenizer.json post-processor adds no BOS (upstream
        # ``LTXGemmaTokenizer`` prepends it manually); bos_token_id=2 in the
        # bundle text_config, but AutoTokenizer reports None without a
        # tokenizer_config.json, so fall back to the well-known id.
        bos_id = self._tokenizer.bos_token_id if self._tokenizer.bos_token_id is not None else 2
        if not tokens or tokens[0] != bos_id:
            tokens = [bos_id, *tokens]
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        pad_token = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else 0
        pad_length = max_length - len(tokens)
        padded = [pad_token] * pad_length + tokens
        mask = [0] * pad_length + [1] * len(tokens)
        return mx.array([padded]), mx.array([mask])

    def get_all_hidden_states(
        self,
        token_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> list[mx.array]:
        if self._model is None:
            raise RuntimeError("Gemma not loaded")

        outer = self._model
        lm = getattr(outer, "language_model", None) or outer
        text_model = getattr(lm, "model", lm)
        if not hasattr(text_model, "embed_tokens") or not hasattr(text_model, "layers"):
            raise RuntimeError("Cannot find embed_tokens/layers in Gemma 4 model hierarchy")

        from mlx_lm.models.base import create_causal_mask

        h = text_model.embed_tokens(token_ids)
        h = h * mx.array(text_model.embed_scale, dtype=h.dtype)
        all_states: list[mx.array] = [h]

        left_padding = None
        if attention_mask is not None:
            left_padding = (1 - attention_mask).cumsum(axis=1)

        masks: dict[str, mx.array] = {}
        t = token_ids.shape[1]
        for layer in text_model.layers:
            lt = str(layer.layer_type)
            if lt not in masks:
                window = text_model.window_size if lt == "sliding_attention" else None
                layer_mask = create_causal_mask(t, window_size=window, left_padding=left_padding)
                # Normalize to (B, 1, L, L) boolean layout for mlx SDPA
                # (no-pad → (L, L); padded → (B, 1, 1, L, L)).
                if layer_mask.ndim == 2:
                    layer_mask = layer_mask[None, None]
                elif layer_mask.ndim > 4:
                    layer_mask = layer_mask.reshape(layer_mask.shape[0], 1, t, t)
                elif layer_mask.ndim == 3:
                    layer_mask = layer_mask[:, None]
                masks[lt] = layer_mask

        cache = [None] * len(text_model.layers)
        intermediates = [(None, None)] * len(text_model.layers)
        eval_every = int(os.environ.get("LTX25_GEMMA_EVAL_EVERY", "2"))
        for idx, (layer, prev_idx) in enumerate(zip(text_model.layers, text_model.previous_kvs)):
            kvs, offset = intermediates[prev_idx]
            h, kvs, offset = layer(
                h,
                masks[str(layer.layer_type)],
                cache[idx],
                per_layer_input=None,
                shared_kv=kvs,
                offset=offset,
            )
            intermediates[idx] = (kvs, offset)
            all_states.append(h)
            if eval_every and (idx + 1) % eval_every == 0:
                _materialize(h)
        # HF ``output_hidden_states`` returns the final-layer output AFTER the
        # model's terminal RMSNorm as the last entry (upstream feature
        # extractor expects exactly num_hidden_layers + 1 states).
        if hasattr(text_model, "norm"):
            all_states[-1] = text_model.norm(all_states[-1])
        return all_states


# ---------------------------------------------------------------------------
# Feature extractor V2 (22B: per-token RMS + dual aggregate embeds)
# ---------------------------------------------------------------------------
#
# Built by :func:`_build_extractor` (sizes come from ``bundle_config.json``).



# ---------------------------------------------------------------------------
# 1D connectors (video / audio)
# ---------------------------------------------------------------------------


class _ConnectorAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, apply_gated_attention: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        inner = num_heads * head_dim
        self.to_q = nn.Linear(dim, inner, bias=True)
        self.to_k = nn.Linear(dim, inner, bias=True)
        self.to_v = nn.Linear(dim, inner, bias=True)
        self.to_out = nn.Linear(inner, dim, bias=True)
        self.to_gate_logits = nn.Linear(dim, num_heads, bias=True) if apply_gated_attention else None
        self.q_norm = nn.RMSNorm(inner)
        self.k_norm = nn.RMSNorm(inner)

    def __call__(
        self,
        x: mx.array,
        rope_cos: mx.array | None = None,
        rope_sin: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        b, n, _ = x.shape
        q = self.q_norm(self.to_q(x))
        k = self.k_norm(self.to_k(x))
        v = self.to_v(x)
        q = q.reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        if rope_cos is not None and rope_sin is not None:
            q = _apply_rope_split(q, rope_cos, rope_sin)
            k = _apply_rope_split(k, rope_cos, rope_sin)
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v
        if self.to_gate_logits is not None:
            gate = 2.0 * mx.sigmoid(self.to_gate_logits(x))
            out = out * gate.transpose(0, 2, 1)[:, :, :, None]
        out = out.transpose(0, 2, 1, 3).reshape(b, n, self.num_heads * self.head_dim)
        return self.to_out(out)


class _ConnectorFF(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, bias: bool = True):
        super().__init__()
        inner = int(dim * mult)
        self.proj_in = nn.Linear(dim, inner, bias=bias)
        self.proj_out = nn.Linear(inner, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj_out(nn.gelu_approx(self.proj_in(x)))


class _ConnectorBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        ff_mult: float = 4.0,
        ff_bias: bool = True,
        apply_gated_attention: bool = False,
    ):
        super().__init__()
        self.attn1 = _ConnectorAttention(dim, num_heads, head_dim, apply_gated_attention=apply_gated_attention)
        self.ff = _ConnectorFF(dim, mult=ff_mult, bias=ff_bias)

    def __call__(
        self,
        x: mx.array,
        rope_cos: mx.array | None,
        rope_sin: mx.array | None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        x = x + self.attn1(_rms_norm(x), rope_cos, rope_sin, attention_mask)
        x = x + self.ff(_rms_norm(x))
        return x


class _Embeddings1DConnector(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        num_layers: int = 2,
        num_registers: int = 128,
        ff_mult: float = 4.0,
        ff_bias: bool = True,
        apply_gated_attention: bool = False,
        max_pos: float = 4096.0,
        norm_output: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_registers = num_registers
        self.max_pos = max_pos
        self.norm_output = norm_output
        self.head_dim = head_dim
        self.learnable_registers = mx.zeros((num_registers, dim))
        self.transformer_1d_blocks = [
            _ConnectorBlock(
                dim, num_heads, head_dim, ff_mult=ff_mult, ff_bias=ff_bias,
                apply_gated_attention=apply_gated_attention,
            )
            for _ in range(num_layers)
        ]

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        seq_len = hidden_states.shape[1]
        if self.num_registers > 0 and attention_mask is not None:
            hidden_states = _replace_padding_with_registers(
                hidden_states, attention_mask, self.learnable_registers
            )
        rope_cos, rope_sin = _precompute_connector_rope(
            hidden_states.shape[1],
            inner_dim=self.dim,
            num_heads=self.dim // self.head_dim,
            max_pos=self.max_pos,
        )
        eval_every = int(os.environ.get("LTX25_CONNECTOR_EVAL_EVERY", "1"))
        for block in self.transformer_1d_blocks:
            hidden_states = block(hidden_states, rope_cos, rope_sin, attention_mask=None)
            if eval_every:
                _materialize(hidden_states)
        if self.norm_output:
            hidden_states = _rms_norm(hidden_states)
        return hidden_states, mx.zeros((1, seq_len, 1), dtype=hidden_states.dtype)


def _replace_padding_with_registers(
    hidden_states: mx.array,
    attention_mask: mx.array,
    registers: mx.array,
) -> mx.array:
    """Replace padded positions with tiled learnable registers (left-padded input).

    Inputs are already sorted valid-first (see ``_right_pad_sort``), so valid
    tokens occupy the tail of the sequence — no reordering inside the connector.
    """
    b, seq_len, dim = hidden_states.shape
    num_registers = registers.shape[0]
    tiled = mx.tile(registers[None, :, :], (1, seq_len // num_registers + 1, 1))[:, :seq_len, :]
    mask_1d = attention_mask.astype(mx.int32)
    num_valid = mx.sum(mask_1d, axis=1)
    results = []
    for bi in range(b):
        n_valid = int(num_valid[bi].item())
        flipped = mx.concatenate([
            mx.zeros((n_valid, 1), dtype=hidden_states.dtype),
            mx.ones((seq_len - n_valid, 1), dtype=hidden_states.dtype),
        ])
        results.append((1.0 - flipped) * hidden_states[bi] + flipped * tiled[bi])
    return mx.stack(results, axis=0)


def _right_pad_sort(features: mx.array, attention_mask: mx.array) -> tuple[mx.array, mx.array]:
    """Stable-sort valid tokens first (upstream ``_compute_right_pad_order``)."""
    binary = attention_mask.astype(mx.int32)  # (B, S)
    sort_idx = mx.argsort(-binary, axis=-1)  # stable: valid tokens keep relative order
    gathered = mx.take_along_axis(features, mx.broadcast_to(sort_idx[:, :, None], features.shape), axis=1)
    mask_sorted = mx.take_along_axis(binary, sort_idx, axis=1)
    return gathered, mask_sorted


# ---------------------------------------------------------------------------
# Weight loading for connectors + extractor
# ---------------------------------------------------------------------------

_CONNECTOR_FILE = "connector.safetensors"


def _load_connector_weights(bundle_root: Path, load_fn: Any | None) -> dict[str, mx.array]:
    path = Path(bundle_root) / _CONNECTOR_FILE
    if not path.is_file():
        raise RuntimeError(f"LTX 2.5 connector weights missing: {path}")
    return load_weights_dict(load_fn, str(path))


def _build_extractor(bundle_root: Path) -> _FeatureExtractorV2:
    """Build the V2 feature extractor with explicit layer-stack width."""
    tcfg = get_transformer_config(bundle_root)
    gcfg = get_gemma_config(bundle_root)
    embedding_dim = int(gcfg["hidden_size"])
    num_layers = int(gcfg["num_hidden_layers"]) + 1
    video_dim = int(tcfg["num_attention_heads"]) * int(tcfg["attention_head_dim"])
    audio_dim = int(tcfg["audio_num_attention_heads"]) * int(tcfg["audio_attention_head_dim"])

    class _SizedFeatureExtractorV2(nn.Module):
        def __init__(self, embedding_dim: int, num_layers: int, video_dim: int, audio_dim: int):
            super().__init__()
            self.embedding_dim = embedding_dim
            flat = embedding_dim * num_layers
            self.video_aggregate_embed = nn.Linear(flat, video_dim, bias=True)
            self.audio_aggregate_embed = nn.Linear(flat, audio_dim, bias=True)

        def __call__(
            self, hidden_states: list[mx.array], attention_mask: mx.array
        ) -> tuple[mx.array, mx.array]:
            encoded = mx.stack(hidden_states, axis=-1)  # (B, T, D, L)
            b, t, d, _ = encoded.shape
            mask_3d = attention_mask.astype(mx.bool_)[:, :, None]
            variance = mx.mean(encoded * encoded, axis=2, keepdims=True)
            normed = encoded * mx.rsqrt(variance + 1e-6)
            normed = normed.reshape(b, t, d * len(hidden_states))
            normed = mx.where(mask_3d, normed, mx.zeros_like(normed))
            v_dim = self.video_aggregate_embed.weight.shape[0]
            a_dim = self.audio_aggregate_embed.weight.shape[0]
            video = self.video_aggregate_embed(normed * math.sqrt(v_dim / self.embedding_dim))
            audio = self.audio_aggregate_embed(normed * math.sqrt(a_dim / self.embedding_dim))
            return video, audio

    return _SizedFeatureExtractorV2(embedding_dim, num_layers, video_dim, audio_dim)


def _build_connector(bundle_root: Path, *, audio: bool) -> _Embeddings1DConnector:
    tcfg = get_transformer_config(bundle_root)
    if audio:
        num_heads = int(tcfg.get("audio_connector_num_attention_heads", tcfg.get("connector_num_attention_heads", 30)))
        head_dim = int(tcfg.get("audio_connector_attention_head_dim", tcfg.get("connector_attention_head_dim", 128)))
        num_layers = int(tcfg.get("audio_connector_num_layers", tcfg.get("connector_num_layers", 2)))
    else:
        num_heads = int(tcfg.get("connector_num_attention_heads", 30))
        head_dim = int(tcfg.get("connector_attention_head_dim", 128))
        num_layers = int(tcfg.get("connector_num_layers", 2))
    ff_bias = tcfg.get("connector_ff_bias", True) is not False
    gated = tcfg.get("connector_apply_gated_attention", False) is True
    pe_max_pos = tcfg.get("connector_positional_embedding_max_pos", [1])
    max_pos = float(pe_max_pos[0]) if pe_max_pos else 1.0
    dim = num_heads * head_dim
    return _Embeddings1DConnector(
        dim=dim,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        num_registers=128,
        ff_bias=ff_bias,
        apply_gated_attention=gated,
        max_pos=max_pos,
        norm_output=True,
    )


class LTX25PromptEncoder:
    """Gemma 4 + feature extractor + connectors → (video_embeds, audio_embeds)."""

    def __init__(self, ctx: Any, bundle_root: Path, config: LTX25Config | None = None):
        self.ctx = ctx
        self.bundle_root = Path(bundle_root)
        self.config = config or LTX25Config()
        self._gemma: _Gemma4LanguageModel | None = None
        self._extractor: Any | None = None
        self._video_connector: _Embeddings1DConnector | None = None
        self._audio_connector: _Embeddings1DConnector | None = None
        self._loaded = False

    def _gemma_root(self) -> Path:
        root = self.bundle_root / "text_encoder"
        if not (root / "config.json").is_file():
            raise RuntimeError(
                f"LTX 2.5 Gemma 4 text encoder missing at {root}. "
                "Run the LTX-2.5 ingest/converter to produce the bundle."
            )
        return root

    def _max_length(self) -> int:
        gcfg = get_gemma_config(self.bundle_root)
        return int(gcfg.get("tokenizer_max_length", _DEFAULT_MAX_LENGTH))

    def load(self, on_log: Callable[[str, str], None] | None = None) -> None:
        if self._loaded:
            return
        _ = on_log

        self._gemma = _Gemma4LanguageModel()
        self._gemma.load(str(self._gemma_root()))

        weights = _load_connector_weights(self.bundle_root, getattr(self.ctx, "load_weights", None))
        prefix = "connector."
        cleaned = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in weights.items()}
        # Normalize checkpoint internals (ff.net.0.proj → proj_in, ff.net.2 → proj_out,
        # to_out.0 → to_out) — the official connector reuses the DiT FeedForward/
        # Attention key layout.
        from backend.engine.families.ltx25.weights_mlx import normalize_ltx25_keys

        cleaned = normalize_ltx25_keys(cleaned)

        extractor_weights = {
            k[len("feature_extractor."):]: v for k, v in cleaned.items() if k.startswith("feature_extractor.")
        }
        video_connector_weights = {
            k[len("video_connector."):]: v for k, v in cleaned.items() if k.startswith("video_connector.")
        }
        audio_connector_weights = {
            k[len("audio_connector."):]: v for k, v in cleaned.items() if k.startswith("audio_connector.")
        }

        extractor = _build_extractor(self.bundle_root)
        missing = [k for k in ("video_aggregate_embed.weight", "video_aggregate_embed.bias",
                               "audio_aggregate_embed.weight", "audio_aggregate_embed.bias")
                   if k not in extractor_weights]
        if missing:
            raise RuntimeError(f"LTX 2.5 connector weights missing feature_extractor keys: {missing[:8]}")
        extractor.load_weights(list(extractor_weights.items()), strict=False)

        video_connector = _build_connector(self.bundle_root, audio=False)
        video_connector.load_weights(list(video_connector_weights.items()), strict=False)

        audio_connector = _build_connector(self.bundle_root, audio=True)
        audio_connector.load_weights(list(audio_connector_weights.items()), strict=False)

        self._extractor = extractor
        self._video_connector = video_connector
        self._audio_connector = audio_connector
        self._loaded = True

    def encode(
        self,
        prompt: str,
        *,
        on_log: Callable[[str, str], None] | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Encode a prompt → ``(video_embeds, audio_embeds)``, both ``(B, S, D)``."""
        if not self._loaded:
            self.load(on_log=on_log)
        token_ids, attention_mask = self._gemma.tokenize(prompt, self._max_length())
        hidden_states = self._gemma.get_all_hidden_states(token_ids, attention_mask)
        video_feats, audio_feats = self._extractor(hidden_states, attention_mask)
        sorted_video, mask_video = _right_pad_sort(video_feats, attention_mask)
        sorted_audio, _ = _right_pad_sort(audio_feats, attention_mask)
        video_embeds, _ = self._video_connector(sorted_video, mask_video)
        audio_embeds, _ = self._audio_connector(sorted_audio, mask_video)
        _materialize(video_embeds, audio_embeds)
        return video_embeds, audio_embeds
