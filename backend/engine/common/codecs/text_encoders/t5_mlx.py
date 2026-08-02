"""T5-XXL 文本编码器 — MLX 前向；PyTorch 见 ``t5_cuda``。"""
from __future__ import annotations
import importlib
from typing import Any


class T5EncoderMlx:
    """T5-XXL 文本编码器。

    用于: Flux1 / Flux2 / LTX / Wan 全系列。
    文本 → [B, seq_len, text_dim] 嵌入。
    """

    def __init__(
        self,
        ctx: Any,
        model_path: str,
        max_seq_len: int = 512,
        text_dim: int = 4096,
        *,
        tokenizer_path: str | None = None,
        weight_dtype: Any | None = None,
        native_mlx_only: bool = False,
    ):
        self.ctx = ctx
        self.model_path = model_path
        self._tokenizer_path = tokenizer_path or model_path
        self.max_seq_len = max_seq_len
        self.text_dim = text_dim
        self._weight_dtype = weight_dtype
        self._native_mlx_only = native_mlx_only
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_path, use_fast=False,
            )
        return self._tokenizer

    def encode(self, texts: list[str]) -> Any:
        tokenizer = self.tokenizer
        ctx = self.ctx
        tokens = tokenizer(
            texts, padding="max_length", max_length=self.max_seq_len,
            truncation=True, return_tensors="np",
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        import mlx.core as mx
        input_ids_mx = ctx.array(input_ids, dtype=mx.int32)
        attention_mask_mx = ctx.array(attention_mask, dtype=mx.float32)
        return self._forward_mlx(input_ids_mx, attention_mask_mx)

    def encode_with_mask(self, texts: list[str]) -> tuple[Any, Any]:
        """Encode texts → ``(hidden_states, attention_mask_bool)`` as MLX arrays."""
        tokenizer = self.tokenizer
        ctx = self.ctx
        tokens = tokenizer(
            texts, padding="max_length", max_length=self.max_seq_len,
            truncation=True, return_tensors="np",
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        import mlx.core as mx
        input_ids_mx = ctx.array(input_ids, dtype=mx.int32)
        attention_mask_mx = ctx.array(attention_mask, dtype=mx.float32)
        hidden = self._forward_mlx(input_ids_mx, attention_mask_mx)
        mask = ctx.array(attention_mask.astype(bool))
        return hidden, mask

    def encode_tokenized_np(
        self,
        input_ids: Any,
        attention_mask: Any,
    ) -> tuple[Any, Any]:
        """Encode pre-tokenized numpy batches → MLX ``(hidden, bool_mask)``."""
        import numpy as np
        import mlx.core as mx

        input_ids_mx = self.ctx.array(input_ids, dtype=mx.int32)
        attention_mask_mx = self.ctx.array(attention_mask, dtype=mx.float32)
        hidden = self._forward_mlx(input_ids_mx, attention_mask_mx)
        mask = self.ctx.array(np.asarray(attention_mask).astype(bool))
        return hidden, mask

    def _load_t5_weight_dict(self) -> dict[str, Any]:
        from pathlib import Path

        from backend.engine.runtime.mlx_runtime import load_weights_dict

        root = Path(self.model_path)
        weights: dict[str, Any] = {}
        load_fn = getattr(self.ctx, "load_weights", None)
        for sf in sorted(root.glob("*.safetensors")):
            if load_fn is not None:
                weights.update(load_fn(str(sf)))
            else:
                weights.update(load_weights_dict(None, str(sf)))
        return weights

    def _forward_mlx(self, input_ids, attention_mask):
        import mlx.core as mx
        try:
            from mlx_lm.models.t5 import T5Model, T5Config
        except ImportError as e:
            raise RuntimeError(
                "T5 MLX forward requires mlx_lm.models.t5; torch bridge is not allowed on the MLX hot path."
            ) from e
        if self._model is None:
            from pathlib import Path

            from backend.engine.common.bundle.quant_inference import (
                resolve_inference_weight_mode_from_bundle,
            )
            from backend.engine.common.bundle.safetensors_affine_quant import (
                read_bundle_affine_bits_if_quantized,
            )
            from backend.engine.common.model.quantized_load_mlx import load_weights_quantized_inference

            config = T5Config.from_pretrained(self.model_path)
            self._model = T5Model(config)
            weights = self._load_t5_weight_dict()
            bundle_affine_bits = read_bundle_affine_bits_if_quantized(weights, Path(self.model_path))
            te_mode = resolve_inference_weight_mode_from_bundle(
                self.ctx,
                weight_keys=frozenset(weights.keys()),
                bundle_affine_bits=bundle_affine_bits,
            )
            if te_mode.kind == "quantized" and te_mode.bits in (4, 8):
                load_weights_quantized_inference(
                    self._model,
                    list(weights.items()),
                    strict=False,
                    ctx=self.ctx,
                    bundle_affine_bits=bundle_affine_bits,
                    bits=int(te_mode.bits),
                    group_size=int(te_mode.group_size),
                    module_root=self._model,
                )
            else:
                self._model.load_weights(str(self.model_path))
                if self._weight_dtype is not None:
                    from backend.engine.runtime.mlx_dtype import cast_module_parameters

                    cast_module_parameters(
                        self._model, self._weight_dtype, eval_fn=self.ctx.eval
                    )
            self.ctx.eval(self._model.parameters())
        return self._model(input_ids, attention_mask)

    def release_weights(self) -> None:
        """Drop loaded MLX / bridge weights (tokenizers unchanged)."""
        self._model = None
        clear_cache_fn = getattr(self.ctx, "clear_cache", None)
        if clear_cache_fn is not None:
            clear_cache_fn()
        else:
            importlib.import_module("mlx.core").clear_cache()
