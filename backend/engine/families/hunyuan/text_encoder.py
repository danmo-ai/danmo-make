"""HunyuanVideo-1.5 dual text encoder — in-repo MLX (no transformers / tokenizers / torch)."""
from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any

import numpy as np

from backend.engine.common.bundle.hf_tokenizer_json import load_hf_tokenizer
from backend.engine.common.codecs.text_encoders import T5Encoder
from backend.engine.families.hunyuan.qwen_encoder_mlx import HunyuanQwen25VLEncoder

_HUNYUAN_SYSTEM_MESSAGE = (
    "You are a helpful assistant. Describe the video by detailing the following aspects: "
    "1. The main content and theme of the video. "
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
    "4. background environment, light, style and atmosphere. "
    "5. camera angles, movements, and transitions used in the video."
)

_NUM_HIDDEN_LAYERS_TO_SKIP = 2

_ENCODER_CACHE: dict[str, "HunyuanVideoTextEncoder"] = {}


def _resolve_qwen_dirs(qwen_root: Path) -> tuple[Path, Path]:
    """Native HF repo root or legacy diffusers ``text_encoder/`` + ``tokenizer/``."""
    root = Path(qwen_root)
    legacy_enc = root / "text_encoder"
    legacy_tok = root / "tokenizer"
    if legacy_enc.is_dir() and legacy_tok.is_dir():
        return legacy_enc, legacy_tok
    if (root / "config.json").is_file():
        return root, root
    raise RuntimeError(
        f"HunyuanVideo Qwen bundle invalid under {root}. "
        "Install models/Text/qwen2.5-vl-7b-instruct (ModelScope Qwen/Qwen2.5-VL-7B-Instruct)."
    )


def _resolve_byt5_dirs(byt5_root: Path) -> tuple[Path, Path]:
    """Native HF repo root or legacy diffusers ``text_encoder_2/`` + ``tokenizer_2/``."""
    root = Path(byt5_root)
    legacy_enc = root / "text_encoder_2"
    legacy_tok = root / "tokenizer_2"
    if legacy_enc.is_dir() and legacy_tok.is_dir():
        return legacy_enc, legacy_tok
    if (root / "config.json").is_file():
        return root, root
    raise RuntimeError(
        f"HunyuanVideo ByT5 bundle invalid under {root}. "
        "Install models/Text/byt5-small (ModelScope google/byt5-small)."
    )


def _format_text_input(prompt: str, system_message: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        },
    ]


def _byt5_tokenize_batch(
    texts: list[str],
    tokenizer_dir: Path,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    tok = load_hf_tokenizer(str(tokenizer_dir))
    return tok.encode_batch(texts, max_length=max_length, add_special_tokens=True)


def extract_glyph_texts(prompt: str) -> str | None:
    """Extract quoted / bracketed glyph text segments for ByT5 (diffusers-compatible)."""
    parts: list[str] = []
    for m in re.finditer(r'"([^"]+)"|"([^"]+)"|「([^」]+)」|《([^》]+)》', prompt):
        g = next(g for g in m.groups() if g is not None)
        if g.strip():
            parts.append(g.strip())
    if not parts:
        return None
    return " ".join(parts)


def get_hunyuan_text_encoder(ctx: Any, bundle_root: Path, config: Any) -> "HunyuanVideoTextEncoder":
    """Reuse one loaded encoder pair per bundle + TE paths (avoids reloading ~7B Qwen each request)."""
    from backend.engine.common.model.dit_stem import require_mlx_ctx

    require_mlx_ctx(ctx, feature="HunyuanVideo text encoding")
    qwen = str(getattr(config, "text_encoder_qwen_local", "") or "").strip()
    byt5 = str(getattr(config, "text_encoder_byt5_local", "") or "").strip()
    key = f"{Path(bundle_root).resolve()}|{qwen}|{byt5}"
    cached = _ENCODER_CACHE.get(key)
    if cached is not None:
        return cached
    enc = HunyuanVideoTextEncoder(ctx, bundle_root, config)
    _ENCODER_CACHE[key] = enc
    return enc


class HunyuanVideoTextEncoder:
    """Registry-driven dual encoder — MLX inference only."""

    def __init__(self, ctx: Any, bundle_root: Path, config: Any):
        self.ctx = ctx
        self.bundle_root = Path(bundle_root)
        self.mllm_max_length = int(getattr(config, "mllm_max_length", 1000))
        self.byt5_max_length = int(getattr(config, "byt5_max_length", 256))
        self.crop_start = int(getattr(config, "prompt_template_crop_start", 108))
        self.release_after_encode = bool(getattr(config, "text_encoder_release_after_encode", True))

        self._qwen: HunyuanQwen25VLEncoder | None = None
        self._byt5_mlx: T5Encoder | None = None

        qwen_root = str(getattr(config, "text_encoder_qwen_local", "") or "").strip()
        byt5_root = str(getattr(config, "text_encoder_byt5_local", "") or "").strip()
        if qwen_root and byt5_root:
            self._enc1_dir, self._tok1_dir = _resolve_qwen_dirs(Path(qwen_root))
            self._enc2_dir, self._tok2_dir = _resolve_byt5_dirs(Path(byt5_root))
        else:
            enc1_dir = self.bundle_root / "text_encoder"
            tok1_dir = self.bundle_root / "tokenizer"
            enc2_dir = self.bundle_root / "text_encoder_2"
            tok2_dir = self.bundle_root / "tokenizer_2"
            for d, name in (
                (enc1_dir, "text_encoder"),
                (tok1_dir, "tokenizer"),
                (enc2_dir, "text_encoder_2"),
                (tok2_dir, "tokenizer_2"),
            ):
                if not d.is_dir():
                    raise RuntimeError(
                        f"HunyuanVideo text encoders not configured. "
                        f"Set registry text_encoder_qwen_local / text_encoder_byt5_local, "
                        f"or install TE assets under {self.bundle_root} (missing {name}/)."
                    )
            self._enc1_dir = enc1_dir
            self._tok1_dir = tok1_dir
            self._enc2_dir = enc2_dir
            self._tok2_dir = tok2_dir

    def release_weights(self) -> None:
        """Drop MLX TE weights after encode so DiT/VAE can use unified memory."""
        if self._qwen is not None:
            self._qwen.release_weights()
            self._qwen = None
        if self._byt5_mlx is not None:
            self._byt5_mlx.release_weights()
            self._byt5_mlx = None
        ctx = getattr(self, "ctx", None)
        if ctx is not None and hasattr(ctx, "clear_cache"):
            ctx.clear_cache()
        else:
            importlib.import_module("mlx.core").clear_cache()

    def _ensure_qwen(self) -> None:
        if self._qwen is not None:
            return
        self._qwen = HunyuanQwen25VLEncoder(self._enc1_dir, self._tok1_dir, ctx=self.ctx)

    def _ensure_byt5(self) -> None:
        if self._byt5_mlx is not None:
            return
        self._byt5_mlx = T5Encoder(
            self.ctx,
            str(self._enc2_dir),
            max_seq_len=self.byt5_max_length,
            tokenizer_path=str(self._tok2_dir),
            weight_dtype=self.ctx.bfloat16(),
            native_mlx_only=True,
        )

    def _encode_qwen(self, prompts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if not prompts:
            raise RuntimeError("HunyuanVideo Qwen encode requires at least one prompt.")
        self._ensure_qwen()
        chats = [_format_text_input(p, _HUNYUAN_SYSTEM_MESSAGE) for p in prompts]
        layer_index = -(_NUM_HIDDEN_LAYERS_TO_SKIP + 1)
        return self._qwen.encode_batch(
            chats,
            max_length=self.mllm_max_length + self.crop_start,
            crop_start=self.crop_start,
            layer_index=layer_index,
        )

    def _byt5_d_model(self) -> int:
        if self._byt5_mlx is not None:
            return int(self._byt5_mlx.text_dim)
        return 1472

    def _encode_byt5(self, prompts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if not prompts:
            raise RuntimeError("HunyuanVideo ByT5 encode requires at least one prompt.")

        d_model = self._byt5_d_model()
        glyphs: list[str | None] = [extract_glyph_texts(p) for p in prompts]
        if all(g is None for g in glyphs):
            b = len(prompts)
            return (
                np.zeros((b, self.byt5_max_length, d_model), dtype=np.float32),
                np.zeros((b, self.byt5_max_length), dtype=bool),
            )

        self._ensure_byt5()
        glyph_texts = [g if g is not None else "" for g in glyphs]
        input_ids, attention_mask = _byt5_tokenize_batch(
            glyph_texts, self._tok2_dir, self.byt5_max_length,
        )
        hidden, mask = self._byt5_mlx.encode_tokenized_np(input_ids, attention_mask)
        hidden_np = np.asarray(hidden, dtype=np.float32)
        mask_np = np.asarray(mask, dtype=bool)

        for i, g in enumerate(glyphs):
            if g is None:
                hidden_np[i] = 0.0
                mask_np[i] = False
        return hidden_np, mask_np

    def encode(self, texts: list[str]) -> tuple[Any, Any, Any, Any]:
        if not texts:
            raise RuntimeError("HunyuanVideoTextEncoder.encode requires at least one prompt.")

        mllm_emb, mllm_mask = self._encode_qwen(texts)
        self.ctx.clear_cache()
        byt5_emb, byt5_mask = self._encode_byt5(texts)

        out = (
            self.ctx.array(mllm_emb),
            self.ctx.array(mllm_mask),
            self.ctx.array(byt5_emb),
            self.ctx.array(byt5_mask),
        )
        if self.release_after_encode:
            self.release_weights()
        else:
            self.ctx.clear_cache()
        return out


def encode_hunyuan_prompt_dual(
    bundle_root: Path,
    prompts: list[str],
    *,
    mllm_max_length: int = 1000,
    byt5_max_length: int = 256,
    crop_start: int = 108,
    device: str = "auto",
    ctx: Any | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Legacy functional API — prefer ``get_hunyuan_text_encoder`` + ``encode``."""
    from types import SimpleNamespace

    if ctx is None:
        from backend.engine.runtime.mlx import MLXContext
        ctx = MLXContext()
    cfg = SimpleNamespace(
        mllm_max_length=mllm_max_length,
        byt5_max_length=byt5_max_length,
        prompt_template_crop_start=crop_start,
        text_encoder_device=device,
    )
    enc = get_hunyuan_text_encoder(ctx, bundle_root, cfg)
    return enc.encode(prompts)
