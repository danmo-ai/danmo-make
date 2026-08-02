"""Wan UMT5-XXL 文本编码 — MLX 前向（``.pth`` 权重，视频族共享）。"""
from __future__ import annotations

import html
import logging
import math
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from backend.engine.common.ops.embeddings import pad_ragged_2d_sequences
from backend.engine.common.codecs.text_encoders.qwen3_mlx import MlxRMSNorm

logger = logging.getLogger(__name__)


def _basic_clean(text: str) -> str:
    try:
        import ftfy

        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = html.unescape(html.unescape(text))
    return text.strip()


def _whitespace_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class _T5Attention(nn.Module):
    def __init__(self, dim: int, dim_attn: int, num_heads: int):
        super().__init__()
        assert dim_attn % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)

    def __call__(
        self,
        x: mx.array,
        context: mx.array | None = None,
        mask: mx.array | None = None,
        pos_bias: mx.array | None = None,
    ) -> mx.array:
        """UMT5 attention — no ``1/sqrt(d)`` scaling (matches official Wan)."""
        context = x if context is None else context
        b, lq, lk = x.shape[0], x.shape[1], context.shape[1]
        n, c = self.num_heads, self.head_dim
        q = self.q(x).reshape(b, lq, n, c)
        k = self.k(context).reshape(b, lk, n, c)
        v = self.v(context).reshape(b, lk, n, c)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        # T5 uses unscaled QK^T; softmax in float32 (official Wan).
        attn = q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)
        if pos_bias is not None:
            attn = attn + pos_bias.astype(mx.float32)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            attn = attn + mx.where(mask == 0, -3.389e38, 0.0).astype(mx.float32)
        attn = mx.softmax(attn, axis=-1).astype(q.dtype)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(b, lq, n * c)
        return self.o(out)


class _T5FeedForward(nn.Module):
    def __init__(self, dim: int, dim_ffn: int):
        super().__init__()
        self.gate = [nn.Linear(dim, dim_ffn, bias=False)]
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.gate[0](x)) * self.fc1(x))


class _T5RelativeEmbedding(nn.Module):
    def __init__(self, num_buckets: int, num_heads: int, bidirectional: bool, max_dist: int = 128):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def __call__(self, lq: int, lk: int) -> mx.array:
        rel_pos = mx.arange(lk).reshape(1, -1) - mx.arange(lq).reshape(-1, 1)
        rel_pos = self._relative_position_bucket(rel_pos)
        values = self.embedding(rel_pos)
        return values.transpose(2, 0, 1).reshape(1, self.num_heads, lq, lk)

    def _relative_position_bucket(self, rel_pos: mx.array) -> mx.array:
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).astype(mx.int64) * num_buckets
            rel_pos = mx.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -mx.minimum(rel_pos, mx.zeros_like(rel_pos))
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (
            mx.log(rel_pos.astype(mx.float32) / max_exact)
            / math.log(self.max_dist / max_exact)
            * (num_buckets - max_exact)
        ).astype(mx.int64)
        rel_pos_large = mx.minimum(rel_pos_large, mx.ones_like(rel_pos_large) * (num_buckets - 1))
        rel_buckets += mx.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class _T5SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_buckets: int,
        shared_pos: bool,
    ):
        super().__init__()
        self.shared_pos = shared_pos
        self.norm1 = MlxRMSNorm(dim)
        self.attn = _T5Attention(dim, dim_attn, num_heads)
        self.norm2 = MlxRMSNorm(dim)
        self.ffn = _T5FeedForward(dim, dim_ffn)
        self.pos_embedding = None if shared_pos else _T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        pos_bias: mx.array | None = None,
    ) -> mx.array:
        e = pos_bias if self.shared_pos else self.pos_embedding(x.shape[1], x.shape[1])
        x = x + self.attn(self.norm1(x), mask=mask, pos_bias=e)
        return x + self.ffn(self.norm2(x))


class _UMT5Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 4096
        dim_attn = 4096
        dim_ffn = 10240
        num_heads = 64
        num_layers = 24
        num_buckets = 32
        self.token_embedding = nn.Embedding(256384, dim)
        self.blocks = [
            _T5SelfAttentionBlock(
                dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos=False,
            )
            for _ in range(num_layers)
        ]
        self.norm = MlxRMSNorm(dim)

    def __call__(self, ids: mx.array, mask: mx.array | None = None) -> mx.array:
        x = self.token_embedding(ids)
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.norm(x)


def _looks_like_tokenizer_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    marker_files = (
        "tokenizer_config.json",
        "tokenizer.json",
        "spiece.model",
    )
    return any((path / m).is_file() for m in marker_files)


def _resolve_umt5_tokenizer_dir(bundle_root: Path) -> Path | None:
    tok_candidates = [
        bundle_root / "google" / "umt5-xxl",
        bundle_root / "umt5-xxl",
        bundle_root / "tokenizer",
    ]
    tok_dir = next((p for p in tok_candidates if _looks_like_tokenizer_dir(p)), None)
    if tok_dir is not None:
        return tok_dir
    video_root = bundle_root.parent
    if video_root.is_dir():
        for sibling in sorted(video_root.iterdir()):
            if not sibling.is_dir() or sibling.resolve() == bundle_root.resolve():
                continue
            for rel in ("google/umt5-xxl", "umt5-xxl", "tokenizer"):
                candidate = sibling / rel
                if _looks_like_tokenizer_dir(candidate):
                    return candidate
    return None


def _remap_wan_umt5_mlx_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """mlx-community Bernini / Wan flat ports: ``ffn.gate_proj`` → ``ffn.gate.0``."""
    if not any(".ffn.gate_proj." in k for k in weights):
        return weights
    remapped: dict[str, mx.array] = {}
    for key, tensor in weights.items():
        remapped[key.replace(".ffn.gate_proj.", ".ffn.gate.0.")] = tensor
    return remapped


def _load_umt5_state_dict(
    checkpoint_path: Path, *, array_fn: Any | None = None
) -> dict[str, mx.array]:
    from backend.engine.runtime.mlx_runtime import load_weights_dict

    logger.info("Loading Wan UMT5 weights from %s", checkpoint_path)
    if array_fn is None:
        array_fn = mx.array
    if checkpoint_path.suffix.lower() == ".safetensors":
        raw = load_weights_dict(None, str(checkpoint_path))
        out: dict[str, mx.array] = {}
        for k, v in raw.items():
            if isinstance(v, mx.array):
                out[k] = v.astype(mx.float32)
            else:
                out[k] = array_fn(np.asarray(v, dtype=np.float32))
        return _remap_wan_umt5_mlx_weights(out)
    from backend.engine.common.bundle.pytorch_bin_numpy import state_dict_to_numpy

    sd = state_dict_to_numpy(checkpoint_path)
    out: dict[str, mx.array] = {}
    for k, v in sd.items():
        out[k] = array_fn(np.asarray(v, dtype=np.float32))
    return out


def _build_umt5_param_map(model: _UMT5Encoder) -> dict[str, mx.array]:
    """Flat checkpoint keys → MLX parameter tensors (``models_t5*.pth`` layout)."""
    pmap: dict[str, mx.array] = {}
    pmap["token_embedding.weight"] = model.token_embedding.weight
    for i, blk in enumerate(model.blocks):
        prefix = f"blocks.{i}"
        pmap[f"{prefix}.norm1.weight"] = blk.norm1.weight
        for w in ("q", "k", "v", "o"):
            lin = getattr(blk.attn, w)
            pmap[f"{prefix}.attn.{w}.weight"] = lin.weight
        pmap[f"{prefix}.norm2.weight"] = blk.norm2.weight
        pmap[f"{prefix}.ffn.gate.0.weight"] = blk.ffn.gate[0].weight
        pmap[f"{prefix}.ffn.fc1.weight"] = blk.ffn.fc1.weight
        pmap[f"{prefix}.ffn.fc2.weight"] = blk.ffn.fc2.weight
        if blk.pos_embedding is not None:
            pmap[f"{prefix}.pos_embedding.embedding.weight"] = blk.pos_embedding.embedding.weight
    pmap["norm.weight"] = model.norm.weight
    return pmap


def _apply_umt5_weights(model: _UMT5Encoder, weights: dict[str, mx.array]) -> None:
    """Assign UMT5 checkpoint tensors; fail loud on missing or shape mismatch."""
    pmap = _build_umt5_param_map(model)
    missing = [k for k in pmap if k not in weights]
    if missing:
        raise RuntimeError(
            f"Wan UMT5 checkpoint missing {len(missing)} parameter(s), e.g. {missing[:8]}"
        )
    extra = [k for k in weights if k not in pmap]
    if extra:
        raise RuntimeError(
            f"Wan UMT5 checkpoint has {len(extra)} unmapped key(s), e.g. {extra[:8]}"
        )
    for key, param in pmap.items():
        src = weights[key]
        if tuple(param.shape) != tuple(src.shape):
            raise RuntimeError(
                f"Wan UMT5 shape mismatch for {key}: param {param.shape} vs ckpt {src.shape}"
            )
        param[:] = src.astype(mx.float32)


class WanUMT5EncoderMLX:
    """Original Wan bundle UMT5-XXL encoder (``models_t5_umt5-xxl-enc-bf16.pth``)."""

    def __init__(
        self,
        ctx: Any,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        *,
        text_len: int = 512,
    ):
        self.ctx = ctx
        self.text_len = int(text_len)
        self._checkpoint_path = Path(checkpoint_path)
        self._tokenizer_path = Path(tokenizer_path)
        self._tokenizer = None
        self._model: _UMT5Encoder | None = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(self._tokenizer_path))
        return self._tokenizer

    def _ensure_model(self) -> _UMT5Encoder:
        if self._model is None:
            model = _UMT5Encoder()
            weights = _load_umt5_state_dict(
                self._checkpoint_path, array_fn=self.ctx.array
            )
            _apply_umt5_weights(model, weights)
            self.ctx.eval(model.parameters())
            logger.info(
                "Wan UMT5-XXL loaded (%d tensors from %s)",
                len(weights),
                self._checkpoint_path.name,
            )
            self._model = model
        return self._model

    @staticmethod
    def _clean_text(text: str) -> str:
        return _whitespace_clean(_basic_clean(text))

    def encode(self, texts: list[str]) -> mx.array:
        cleaned = [self._clean_text(t) for t in texts]
        tokens = self.tokenizer(
            cleaned,
            padding="max_length",
            max_length=self.text_len,
            truncation=True,
            return_tensors="np",
        )
        ids = self.ctx.array(tokens["input_ids"], dtype=mx.int32)
        mask = self.ctx.array(tokens["attention_mask"], dtype=mx.float32)
        hidden = self._ensure_model()(ids, mask=mask)
        self.ctx.eval(hidden)
        # Match official Wan: truncate to real token length, zero-pad tail for DiT.
        seq_lens = [int(v) for v in mask.sum(axis=1).tolist()]
        trimmed = [
            hidden[i, : min(seq_len, self.text_len)].astype(mx.float32)
            for i, seq_len in enumerate(seq_lens)
        ]
        return pad_ragged_2d_sequences(
            self.ctx,
            trimmed,
            target_len=self.text_len,
            dtype=mx.float32,
            pad_value=0.0,
        )

    def release_weights(self) -> None:
        self._model = None
        self.ctx.clear_cache()


def resolve_wan_umt5_pth(bundle_root: Path) -> tuple[Path, Path] | None:
    """Return ``(checkpoint.pth, tokenizer_dir)`` for original Wan bundles."""
    root = Path(bundle_root)
    if not root.is_absolute():
        from backend.utils.config_paths import resolve_default_config_root
        from backend.utils.user_home import ensure_control_plane
        from backend.utils.workspace import resolve_workspace_root

        repo_root = Path(__file__).resolve().parents[4]
        default_cfg = resolve_default_config_root(bootstrap_root=repo_root, bundle_root=None)
        control = ensure_control_plane()
        workspace_root = resolve_workspace_root(
            repo_root,
            control_plane=control,
            legacy_default_config=default_cfg,
        )
        root = (workspace_root / root).resolve()
    if not root.is_dir():
        return None

    tok_dir = _resolve_umt5_tokenizer_dir(root)
    if tok_dir is None:
        return None
    pth_candidates = sorted(root.glob("models_t5*.pth"))
    pth = next((p for p in pth_candidates if p.is_file()), None)
    if pth is not None:
        return pth, tok_dir
    te_dir = root / "text_encoder"
    if te_dir.is_dir():
        te_shards = sorted(te_dir.glob("*.safetensors"))
        if te_shards:
            return te_shards[0], tok_dir
    flat_t5 = root / "t5_encoder.safetensors"
    if flat_t5.is_file():
        return flat_t5, tok_dir
    return None
