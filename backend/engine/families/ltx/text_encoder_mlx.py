"""LTX 2.3 Gemma 3 text encoder + connector (MLX).

Loads Gemma from registry ``text_encoder_gemma_local`` (``models/Text/gemma-3-12b-it-4bit``)
and the bundle ``connector.safetensors`` weights (in-repo connector + feature extractor).
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from backend.engine.runtime.mlx_runtime import load_weights_dict, run_eval
from backend.engine.config.model_configs import LTXConfig
from backend.engine.families.ltx.gemma_bundle import resolve_gemma_load_path

logger = logging.getLogger(__name__)

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
    max_pos: int = 4096,
) -> tuple[mx.array, mx.array]:
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


class _ConnectorAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        inner = num_heads * head_dim
        self.to_q = nn.Linear(dim, inner, bias=True)
        self.to_k = nn.Linear(dim, inner, bias=True)
        self.to_v = nn.Linear(dim, inner, bias=True)
        self.to_out = [nn.Linear(inner, dim, bias=True)]
        self.to_gate_logits = nn.Linear(dim, num_heads, bias=True)
        self.q_norm = nn.RMSNorm(inner)
        self.k_norm = nn.RMSNorm(inner)

    def __call__(
        self,
        x: mx.array,
        rope_cos: mx.array | None = None,
        rope_sin: mx.array | None = None,
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
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v
        gate = 2.0 * mx.sigmoid(self.to_gate_logits(x))
        out = out * gate.transpose(0, 2, 1)[:, :, :, None]
        out = out.transpose(0, 2, 1, 3).reshape(b, n, self.num_heads * self.head_dim)
        return self.to_out[0](out)


class _ConnectorFF(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        inner = int(dim * mult)
        self.net = [
            nn.Linear(dim, inner, bias=True),
            None,
            nn.Linear(inner, dim, bias=True),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu_approx(self.net[0](x))
        return self.net[2](x)


class _ConnectorBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, ff_mult: float = 4.0):
        super().__init__()
        self.attn1 = _ConnectorAttention(dim, num_heads, head_dim)
        self.ff = _ConnectorFF(dim, mult=ff_mult)

    def __call__(self, x: mx.array, rope_cos: mx.array | None, rope_sin: mx.array | None) -> mx.array:
        x = x + self.attn1(_rms_norm(x), rope_cos, rope_sin)
        x = x + self.ff(_rms_norm(x))
        return x


class _Embeddings1DConnector(nn.Module):
    def __init__(
        self,
        dim: int = 4096,
        num_heads: int = 32,
        head_dim: int = 128,
        num_layers: int = 8,
        num_registers: int = 128,
        ff_mult: float = 4.0,
        max_pos: int = 4096,
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
            _ConnectorBlock(dim, num_heads, head_dim, ff_mult=ff_mult) for _ in range(num_layers)
        ]

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        b, seq_len, dim = hidden_states.shape
        if self.num_registers > 0 and attention_mask is not None:
            hidden_states = _replace_padding_with_registers(hidden_states, attention_mask, self.learnable_registers)
        rope_cos, rope_sin = _precompute_connector_rope(
            hidden_states.shape[1],
            inner_dim=self.dim,
            num_heads=self.dim // self.head_dim,
            max_pos=self.max_pos,
        )
        eval_every = int(os.environ.get("LTX2_GEMMA_EVAL_EVERY", "1"))
        for block in self.transformer_1d_blocks:
            hidden_states = block(hidden_states, rope_cos, rope_sin)
            if eval_every:
                _materialize(hidden_states)
        if self.norm_output:
            hidden_states = _rms_norm(hidden_states)
        return hidden_states


def _replace_padding_with_registers(
    hidden_states: mx.array,
    attention_mask: mx.array,
    registers: mx.array,
) -> mx.array:
    b, seq_len, dim = hidden_states.shape
    num_registers = registers.shape[0]
    tiled = mx.tile(registers[None, :, :], (1, seq_len // num_registers + 1, 1))[:, :seq_len, :]
    results = []
    mask_1d = attention_mask.astype(mx.int32)
    num_valid = mx.sum(mask_1d, axis=1)
    for bi in range(b):
        n_valid = int(num_valid[bi].item())
        valid = hidden_states[bi, seq_len - n_valid :, :]
        if n_valid < seq_len:
            valid = mx.concatenate([valid, mx.zeros((seq_len - n_valid, dim), dtype=valid.dtype)], axis=0)
        flipped = mx.concatenate([
            mx.ones((n_valid, 1), dtype=valid.dtype),
            mx.zeros((seq_len - n_valid, 1), dtype=valid.dtype),
        ])
        results.append(flipped * valid + (1.0 - flipped) * tiled[bi])
    return mx.stack(results, axis=0)


class _TextEmbeddingProjection(nn.Module):
    def __init__(self, input_dim: int, video_dim: int, audio_dim: int, embedding_dim: int = 3840):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.video_aggregate_embed = nn.Linear(input_dim, video_dim, bias=True)
        self.audio_aggregate_embed = nn.Linear(input_dim, audio_dim, bias=True)

    def __call__(self, hidden_states: mx.array) -> tuple[mx.array, mx.array]:
        v_dim = self.video_aggregate_embed.weight.shape[0]
        v_scale = math.sqrt(v_dim / self.embedding_dim)
        video = self.video_aggregate_embed(hidden_states * v_scale)
        _materialize(video)
        a_dim = self.audio_aggregate_embed.weight.shape[0]
        a_scale = math.sqrt(a_dim / self.embedding_dim)
        audio = self.audio_aggregate_embed(hidden_states * a_scale)
        _materialize(audio)
        return video, audio


class _TextEncoderConnector(nn.Module):
    def __init__(
        self,
        caption_channels: int = 3840,
        num_gemma_layers: int = 49,
        video_dim: int = 4096,
        audio_dim: int = 2048,
    ):
        super().__init__()
        input_dim = num_gemma_layers * caption_channels
        self.text_embedding_projection = _TextEmbeddingProjection(input_dim, video_dim, audio_dim)
        self.video_embeddings_connector = _Embeddings1DConnector(dim=video_dim, head_dim=128)
        self.audio_embeddings_connector = _Embeddings1DConnector(dim=audio_dim, head_dim=64)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        video, audio = self.text_embedding_projection(hidden_states)
        video = self.video_embeddings_connector(video, attention_mask=attention_mask)
        audio = self.audio_embeddings_connector(audio, attention_mask=attention_mask)
        return video, audio


class _GemmaFeaturesExtractor(nn.Module):
    def __init__(
        self,
        caption_channels: int = 3840,
        num_gemma_layers: int = 49,
        video_dim: int = 4096,
        audio_dim: int = 2048,
    ):
        super().__init__()
        self.num_gemma_layers = num_gemma_layers
        self.caption_channels = caption_channels
        self.connector = _TextEncoderConnector(
            caption_channels=caption_channels,
            num_gemma_layers=num_gemma_layers,
            video_dim=video_dim,
            audio_dim=audio_dim,
        )

    def __call__(
        self,
        all_hidden_states: list[mx.array],
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        encoded = mx.stack(all_hidden_states, axis=-1)
        variance = mx.mean(encoded * encoded, axis=2, keepdims=True)
        normed = encoded * mx.rsqrt(variance + 1e-6)
        b, t, d, layers = normed.shape
        stacked = normed.reshape(b, t, d * layers)
        if attention_mask is not None:
            stacked = stacked * attention_mask[:, :, None].astype(stacked.dtype)
        _materialize(stacked)
        return self.connector(stacked, attention_mask=attention_mask)


class _GemmaLanguageModel:
    """Gemma 3 via mlx-lm — extracts all layer hidden states."""

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
        if len(tokens) > max_length:
            tokens = tokens[-max_length:]
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

        inner = self._model
        for attr in ("model", "language_model", "model"):
            if hasattr(inner, attr):
                inner = getattr(inner, attr)
            if hasattr(inner, "embed_tokens"):
                break
        if not hasattr(inner, "embed_tokens"):
            raise RuntimeError("Cannot find embed_tokens in Gemma model hierarchy")

        h = inner.embed_tokens(token_ids)
        hidden_size = h.shape[-1]
        h = h * mx.array(hidden_size**0.5, dtype=mx.bfloat16).astype(h.dtype)
        all_states: list[mx.array] = [h]

        t = token_ids.shape[1]
        causal = mx.triu(mx.full((t, t), -1e9, dtype=mx.bfloat16), k=1)
        if attention_mask is not None:
            pad = (1 - attention_mask[:, None, None, :].astype(mx.bfloat16)) * -1e9
            combined = causal[None, None, :, :] + pad
        else:
            combined = causal[None, None, :, :]

        eval_every = int(os.environ.get("LTX2_GEMMA_EVAL_EVERY", "1"))
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask=combined, cache=None)
            if isinstance(h, tuple):
                h = h[0]
            all_states.append(h)
            if eval_every and (i + 1) % eval_every == 0:
                _materialize(h)
        return all_states


def _remap_connector_bundle_keys(weights: dict[str, Any]) -> dict[str, Any]:
    """Map dgrauet / mlx-forge connector FF keys to ``_ConnectorFF`` Linear names."""
    out: dict[str, Any] = {}
    for key, tensor in weights.items():
        nk = key.replace(".ff.net.0.proj.", ".ff.net.0.")
        out[nk] = tensor
    return out


def _load_connector_weights(bundle_root: Path, load_fn: Any | None) -> dict[str, mx.array]:
    path = bundle_root / "connector.safetensors"
    if not path.is_file():
        raise RuntimeError(f"LTX 2.3 connector weights missing: {path}")
    raw = load_weights_dict(load_fn, str(path))
    out: dict[str, mx.array] = {}
    prefix = "connector."
    for k, v in raw.items():
        out[k[len(prefix):] if k.startswith(prefix) else k] = v
    return _remap_connector_bundle_keys(out)


class LTX23GemmaEncoder:
    """LTX 2.3 prompt encoder: Gemma 3 + connector → (video_embeds, audio_embeds)."""

    def __init__(
        self,
        ctx: Any,
        bundle_root: Path,
        config: LTXConfig | None = None,
    ):
        self.ctx = ctx
        self.bundle_root = Path(bundle_root)
        self.config = config or LTXConfig()
        self._gemma_root: Path | None = None
        self._gemma: Any | None = None
        self._extractor: Any | None = None

    def _resolve_gemma_root(self) -> Path:
        if self._gemma_root is not None:
            return self._gemma_root
        project_root = getattr(self.ctx, "project_root", None)
        root = resolve_gemma_load_path(
            self.config,
            project_root=Path(project_root) if project_root else None,
        )
        self._gemma_root = root
        return root

    def _max_length(self) -> int:
        return int(os.environ.get("LTX2_GEMMA_MAX_LENGTH", str(_DEFAULT_MAX_LENGTH)))

    def load(self, on_log: Callable[[str, str], None] | None = None) -> None:
        if getattr(self.ctx, "backend", None) != "mlx":
            raise RuntimeError(
                f"LTX 2.3 Gemma encoder requires MLX runtime (got {getattr(self.ctx, 'backend', None)!r})"
            )
        if not self.bundle_root.is_dir():
            raise RuntimeError(f"LTX 2.3 bundle directory not found: {self.bundle_root}")

        if self._gemma is None:
            self._gemma = _GemmaLanguageModel()
            gemma_root = self._resolve_gemma_root()
            msg = f"LTX 2.3 loading Gemma text encoder from {gemma_root}"
            logger.info(msg)
            if on_log:
                on_log("info", msg)
            self._gemma.load(str(gemma_root))
            if on_log:
                on_log("info", "LTX 2.3 Gemma text encoder ready")

        if self._extractor is None:
            if on_log:
                on_log("info", "LTX 2.3 loading connector weights")
            self._extractor = _GemmaFeaturesExtractor()
            connector_weights = _load_connector_weights(
                self.bundle_root,
                getattr(self.ctx, "load_weights", None),
            )
            self._extractor.connector.load_weights(list(connector_weights.items()))
            _materialize(
                self._extractor.connector.video_embeddings_connector.learnable_registers,
                self._extractor.connector.audio_embeddings_connector.learnable_registers,
            )
            if on_log:
                on_log("info", "LTX 2.3 connector ready")

    def free(self) -> None:
        self._gemma = None
        self._extractor = None
        if hasattr(self.ctx, "clear_cache"):
            self.ctx.clear_cache()

    def encode(
        self,
        prompt: str,
        on_log: Callable[[str, str], None] | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Encode prompt → ``(video_embeds, audio_embeds)`` each ``(1, seq, dim)``."""
        self.load(on_log=on_log)
        assert self._gemma is not None and self._extractor is not None
        if on_log:
            on_log("info", "LTX 2.3 encoding prompt with Gemma")
        max_len = self._max_length()
        token_ids, attention_mask = self._gemma.tokenize(prompt, max_len)
        hidden_states = self._gemma.get_all_hidden_states(token_ids, attention_mask=attention_mask)
        video, audio = self._extractor(hidden_states, attention_mask=attention_mask)
        _materialize(video, audio)
        if on_log:
            on_log("info", "LTX 2.3 prompt encoding done")
        return video, audio

    def encode_with_negative(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        on_log: Callable[[str, str], None] | None = None,
    ) -> tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]]:
        """Return ``((pos_video, pos_audio), (neg_video, neg_audio))`` for CFG."""
        from backend.engine.families.ltx.pipeline_math_mlx import DEFAULT_NEGATIVE_PROMPT

        neg = negative_prompt if negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT
        pos = self.encode(prompt, on_log=on_log)
        neg_emb = self.encode(neg, on_log=on_log)
        return pos, neg_emb
