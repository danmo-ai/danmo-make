"""Wan UMT5-XXL 文本编码 — PyTorch (CUDA) 前向（``.pth`` 权重，视频族共享）。"""
from __future__ import annotations

import html
import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.engine.runtime.cuda import CudaContext

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


def _pad_ragged_2d_sequences_torch(
    sequences: list[torch.Tensor],
    target_len: int | None = None,
    dtype: torch.dtype | None = None,
    pad_value: float = 0.0,
) -> torch.Tensor:
    if not sequences:
        raise RuntimeError("pad_ragged_2d_sequences requires non-empty sequences")
    max_len = max(int(s.shape[0]) for s in sequences)
    t = target_len if target_len is not None else max_len
    padded: list[torch.Tensor] = []
    for s in sequences:
        cur = int(s.shape[0])
        dim = int(s.shape[1])
        use = s[:t] if cur > t else s
        if cur < t:
            pad = torch.full((t - cur, dim), float(pad_value), dtype=s.dtype, device=s.device)
            use = torch.cat([use, pad], dim=0)
        padded.append(use)
    out = torch.stack(padded, dim=0)
    return out.to(dtype) if dtype is not None else out


# ---------------------------------------------------------------------------
# RMSNorm (torch)
# ---------------------------------------------------------------------------

class _RMSNormTorch(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * norm).to(dtype)


# ---------------------------------------------------------------------------
# UMT5 modules
# ---------------------------------------------------------------------------

class _T5AttentionTorch(nn.Module):
    def __init__(self, dim: int, dim_attn: int, num_heads: int):
        super().__init__()
        assert dim_attn % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pos_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """UMT5 attention — no ``1/sqrt(d)`` scaling (matches official Wan)."""
        context = x if context is None else context
        b, lq, lk = x.shape[0], x.shape[1], context.shape[1]
        n, c = self.num_heads, self.head_dim
        q = self.q(x).reshape(b, lq, n, c)
        k = self.k(context).reshape(b, lk, n, c)
        v = self.v(context).reshape(b, lk, n, c)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        # T5 uses unscaled QK^T; softmax in float32 (official Wan).
        attn = q.float() @ k.float().transpose(-2, -1)
        if pos_bias is not None:
            attn = attn + pos_bias.float()
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            attn = attn + torch.where(mask == 0, torch.tensor(-3.389e38, device=mask.device), 0.0).float()
        attn = torch.softmax(attn, dim=-1).to(q.dtype)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(b, lq, n * c)
        return self.o(out)


class _T5FeedForwardTorch(nn.Module):
    def __init__(self, dim: int, dim_ffn: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim_ffn, bias=False)
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.gate(x), approximate="tanh") * self.fc1(x))


class _T5RelativeEmbeddingTorch(nn.Module):
    def __init__(self, num_buckets: int, num_heads: int, bidirectional: bool, max_dist: int = 128):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq: int, lk: int) -> torch.Tensor:
        rel_pos = torch.arange(lk, dtype=torch.int64).view(1, -1) - torch.arange(lq, dtype=torch.int64).view(-1, 1)
        rel_pos = self._relative_position_bucket(rel_pos)
        values = self.embedding(rel_pos)
        return values.permute(2, 0, 1).view(1, self.num_heads, lq, lk)

    def _relative_position_bucket(self, rel_pos: torch.Tensor) -> torch.Tensor:
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = rel_pos.abs()
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.minimum(rel_pos, torch.zeros_like(rel_pos))
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (
            torch.log(rel_pos.float() / max_exact)
            / math.log(self.max_dist / max_exact)
            * (num_buckets - max_exact)
        ).long()
        rel_pos_large = torch.minimum(rel_pos_large, torch.ones_like(rel_pos_large) * (num_buckets - 1))
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class _T5SelfAttentionBlockTorch(nn.Module):
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
        self.norm1 = _RMSNormTorch(dim)
        self.attn = _T5AttentionTorch(dim, dim_attn, num_heads)
        self.norm2 = _RMSNormTorch(dim)
        self.ffn = _T5FeedForwardTorch(dim, dim_ffn)
        self.pos_embedding = None if shared_pos else _T5RelativeEmbeddingTorch(
            num_buckets, num_heads, bidirectional=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        pos_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        e = pos_bias if self.shared_pos else self.pos_embedding(x.shape[1], x.shape[1])
        x = x + self.attn(self.norm1(x), mask=mask, pos_bias=e)
        return x + self.ffn(self.norm2(x))


class _UMT5EncoderTorch(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        dim = 4096
        dim_attn = 4096
        dim_ffn = 10240
        num_heads = 64
        num_layers = 24
        num_buckets = 32
        self.token_embedding = nn.Embedding(256384, dim).to(device)
        self.blocks = nn.ModuleList([
            _T5SelfAttentionBlockTorch(
                dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos=False,
            ).to(device)
            for _ in range(num_layers)
        ])
        self.norm = _RMSNormTorch(dim)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.token_embedding(ids)
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _load_umt5_state_dict_torch(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    from backend.engine.common.bundle.pytorch_bin_numpy import state_dict_to_numpy

    logger.info("Loading Wan UMT5 weights from %s", checkpoint_path)
    sd = state_dict_to_numpy(checkpoint_path)
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        out[k] = torch.from_numpy(np.asarray(v, dtype=np.float32))
    return out


def _build_umt5_param_map_torch(model: _UMT5EncoderTorch) -> dict[str, torch.Tensor]:
    """Flat checkpoint keys → PyTorch parameter tensors (``models_t5*.pth`` layout)."""
    pmap: dict[str, torch.Tensor] = {}
    pmap["token_embedding.weight"] = model.token_embedding.weight
    for i, blk in enumerate(model.blocks):
        prefix = f"blocks.{i}"
        pmap[f"{prefix}.norm1.weight"] = blk.norm1.weight
        for w in ("q", "k", "v", "o"):
            lin = getattr(blk.attn, w)
            pmap[f"{prefix}.attn.{w}.weight"] = lin.weight
        pmap[f"{prefix}.norm2.weight"] = blk.norm2.weight
        pmap[f"{prefix}.ffn.gate.0.weight"] = blk.ffn.gate.weight
        pmap[f"{prefix}.ffn.fc1.weight"] = blk.ffn.fc1.weight
        pmap[f"{prefix}.ffn.fc2.weight"] = blk.ffn.fc2.weight
        if blk.pos_embedding is not None:
            pmap[f"{prefix}.pos_embedding.embedding.weight"] = blk.pos_embedding.embedding.weight
    pmap["norm.weight"] = model.norm.weight
    return pmap


def _apply_umt5_weights_torch(model: _UMT5EncoderTorch, weights: dict[str, torch.Tensor]) -> None:
    """Assign UMT5 checkpoint tensors; fail loud on missing or shape mismatch."""
    pmap = _build_umt5_param_map_torch(model)
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
        param.data.copy_(src.float())


# ---------------------------------------------------------------------------
# Public encoder class
# ---------------------------------------------------------------------------

class WanUMT5EncoderCUDA:
    """Original Wan bundle UMT5-XXL encoder (``models_t5_umt5-xxl-enc-bf16.pth``)."""

    def __init__(
        self,
        ctx: CudaContext,
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
        self._model: _UMT5EncoderTorch | None = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(str(self._tokenizer_path))
        return self._tokenizer

    def _ensure_model(self) -> _UMT5EncoderTorch:
        if self._model is None:
            model = _UMT5EncoderTorch(self.ctx.device)
            weights = _load_umt5_state_dict_torch(self._checkpoint_path)
            _apply_umt5_weights_torch(model, weights)
            model.eval()
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

    def encode(self, texts: list[str]) -> torch.Tensor:
        cleaned = [self._clean_text(t) for t in texts]
        tokens = self.tokenizer(
            cleaned,
            padding="max_length",
            max_length=self.text_len,
            truncation=True,
            return_tensors="np",
        )
        ids = torch.tensor(tokens["input_ids"], dtype=torch.int32, device=self.ctx.device)
        mask = torch.tensor(tokens["attention_mask"], dtype=torch.float32, device=self.ctx.device)
        hidden = self._ensure_model()(ids, mask=mask)
        # Match official Wan: truncate to real token length, zero-pad tail for DiT.
        seq_lens = [int(v) for v in mask.sum(dim=1).tolist()]
        trimmed = [
            hidden[i, : min(seq_len, self.text_len)].float()
            for i, seq_len in enumerate(seq_lens)
        ]
        return _pad_ragged_2d_sequences_torch(
            trimmed,
            target_len=self.text_len,
            dtype=torch.float32,
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

    def _looks_like_tokenizer_dir(path: Path) -> bool:
        if not path.is_dir():
            return False
        marker_files = (
            "tokenizer_config.json",
            "tokenizer.json",
            "spiece.model",
        )
        return any((path / m).is_file() for m in marker_files)

    tok_candidates = [
        root / "google" / "umt5-xxl",
        root / "umt5-xxl",
        root / "tokenizer",
    ]
    tok_dir = next((p for p in tok_candidates if _looks_like_tokenizer_dir(p)), None)
    if tok_dir is None:
        return None
    pth_candidates = sorted(root.glob("models_t5*.pth"))
    pth = next((p for p in pth_candidates if p.is_file()), None)
    if pth is None:
        return None
    return pth, tok_dir
