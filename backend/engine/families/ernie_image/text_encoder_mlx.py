"""ERNIE-Image Ministral-3 text encoder (MLX via mlx-lm)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from backend.engine.runtime.mlx_runtime import load_weights_dict


def _mistral3_text_config_dict(cfg: dict) -> dict:
    text_cfg = cfg.get("text_config") or cfg
    rope = text_cfg.get("rope_parameters") or text_cfg.get("rope_scaling") or {}
    layer_types = text_cfg.get("layer_types")
    out: dict[str, Any] = {
        "model_type": "ministral3",
        "hidden_size": text_cfg.get("hidden_size", 3072),
        "num_hidden_layers": text_cfg.get("num_hidden_layers", 26),
        "intermediate_size": text_cfg.get("intermediate_size", 9216),
        "num_attention_heads": text_cfg.get("num_attention_heads", 32),
        "num_key_value_heads": text_cfg.get("num_key_value_heads", 8),
        "head_dim": text_cfg.get("head_dim", 128),
        "rms_norm_eps": text_cfg.get("rms_norm_eps", 1e-5),
        "vocab_size": text_cfg.get("vocab_size", 131072),
        "max_position_embeddings": text_cfg.get("max_position_embeddings", 262144),
        "tie_word_embeddings": text_cfg.get("tie_word_embeddings", True),
        "rope_parameters": {
            "rope_type": rope.get("rope_type", "yarn"),
            "rope_theta": rope.get("rope_theta", text_cfg.get("rope_theta", 1_000_000.0)),
            "factor": rope.get("factor", 16.0),
            "beta_fast": rope.get("beta_fast", 32.0),
            "beta_slow": rope.get("beta_slow", 1.0),
            "llama_4_scaling_beta": rope.get("llama_4_scaling_beta", 0.1),
            "mscale": rope.get("mscale", 1.0),
            "mscale_all_dim": rope.get("mscale_all_dim", 1.0),
            "original_max_position_embeddings": rope.get(
                "original_max_position_embeddings", 16384
            ),
        },
    }
    if layer_types is not None:
        out["layer_types"] = list(layer_types)
    sliding_window = text_cfg.get("sliding_window")
    if sliding_window is not None:
        out["sliding_window"] = int(sliding_window)
    return out


class _Mistral3TextCore:
    """mlx-lm Mistral3 language stack — penultimate hidden state for DiT."""

    def __init__(self, text_cfg: dict):
        from mlx_lm.models import mistral3

        args = mistral3.ModelArgs(
            model_type="mistral3",
            text_config=_mistral3_text_config_dict(text_cfg),
        )
        self._model = mistral3.Model(args)

    def sanitize(self, weights: dict) -> dict:
        return self._model.sanitize(weights)

    def encode(self, input_ids: mx.array) -> mx.array:
        from mlx_lm.models.base import create_attention_mask
        from mlx_lm.models.ministral3 import _get_llama_4_attn_scale

        lm = self._model.language_model.model
        h = lm.embed_tokens(input_ids)
        cache = [None] * len(lm.layers)
        fa_mask = create_attention_mask(h, cache[lm.fa_idx]) if lm.fa_idx is not None else None
        swa_mask = None
        if lm.swa_idx is not None:
            swa_mask = create_attention_mask(h, cache[lm.swa_idx], window_size=lm.sliding_window)
        attn_scale = _get_llama_4_attn_scale(
            input_ids.shape[1],
            0,
            lm.args.rope_parameters["llama_4_scaling_beta"],
            lm.args.rope_parameters["original_max_position_embeddings"],
        ).astype(h.dtype)
        for layer, c in zip(lm.layers[:-1], cache[:-1], strict=True):
            mask = swa_mask if layer.use_sliding else fa_mask
            h = layer(h, attn_scale, mask, cache=c)
        return h


def build_ernie_mistral3_encoder(
    model_path: str,
    ctx: Any,
    *,
    load_fn: Any | None = None,
) -> _Mistral3TextCore:
    model_dir = Path(model_path)
    weights: dict = {}
    for sf in sorted(model_dir.glob("*.safetensors")):
        weights.update(load_weights_dict(load_fn, str(sf)))
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file() and not weights:
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        shard_names = sorted(set(index.get("weight_map", {}).values()))
        for shard in shard_names:
            shard_path = model_dir / shard
            if shard_path.is_file():
                weights.update(load_weights_dict(load_fn, str(shard_path)))

    config_path = model_dir / "config.json"
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            raw_cfg = json.load(f)
    elif not weights:
        raise RuntimeError(f"ERNIE text encoder config missing: {config_path}")
    else:
        # mlx-community bundles may omit config.json; ModelArgs defaults match ERNIE-Image Ministral-3.
        raw_cfg = {"text_config": {}}

    core = _Mistral3TextCore(raw_cfg)
    sanitized = core.sanitize(weights)

    from backend.engine.common.bundle.quant_inference import resolve_inference_weight_mode_from_bundle
    from backend.engine.common.bundle.safetensors_affine_quant import read_bundle_affine_bits_if_quantized
    from backend.engine.common.model.quantized_load_mlx import load_weights_quantized_inference

    bundle_affine_bits = read_bundle_affine_bits_if_quantized(weights, model_dir)
    te_mode = resolve_inference_weight_mode_from_bundle(
        ctx,
        weight_keys=frozenset(sanitized.keys()),
        bundle_affine_bits=bundle_affine_bits,
    )
    if te_mode.kind == "quantized" and te_mode.bits in (4, 8):
        load_weights_quantized_inference(
            core._model,
            list(sanitized.items()),
            strict=False,
            ctx=ctx,
            bundle_affine_bits=bundle_affine_bits,
            bits=int(te_mode.bits),
            group_size=int(te_mode.group_size),
            module_root=core._model,
        )
    else:
        core._model.load_weights(list(sanitized.items()), strict=False)
    ctx.eval(core._model.parameters())
    return core


class ErnieImageTextEncoder:
    """Ministral-3 encoder for ERNIE-Image — returns ``(embeddings, text_lens)``."""

    def __init__(
        self,
        ctx: Any,
        model_path: str,
        max_seq_len: int = 2048,
        tokenizer_path: str = "",
        **_kw: Any,
    ):
        self.ctx = ctx
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.max_seq_len = max_seq_len
        self._tokenizer = None
        self._core: _Mistral3TextCore | None = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True,
            )
        return self._tokenizer

    def _tokenize_batch(self, texts: list[str]) -> tuple[mx.array, mx.array]:
        """Match diffusers ``encode_prompt``: variable length, then pad to batch max."""
        tokenizer = self.tokenizer
        rows: list[list[int]] = []
        lengths: list[int] = []
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        for text in texts:
            ids = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_seq_len,
                padding=False,
            )["input_ids"]
            if not ids:
                bos = tokenizer.bos_token_id
                ids = [int(bos if bos is not None else 0)]
            rows.append(list(ids))
            lengths.append(len(ids))
        tmax = max(lengths)
        padded = [row + [pad_id] * (tmax - len(row)) for row in rows]
        import numpy as np

        input_ids = self.ctx.array(np.asarray(padded, dtype=np.int32), dtype=mx.int32)
        text_lens = mx.array(lengths, dtype=mx.int32)
        return input_ids, text_lens

    def encode(self, texts: list[str]) -> tuple[mx.array, mx.array]:
        input_ids, text_lens = self._tokenize_batch(texts)

        if self._core is None:
            self._core = build_ernie_mistral3_encoder(
                self.model_path,
                self.ctx,
                load_fn=getattr(self.ctx, "load_weights", None),
            )

        embeddings = self._core.encode(input_ids)
        return embeddings.astype(mx.bfloat16), text_lens

    def release_weights(self) -> None:
        """Drop Ministral-3 MLX weights after encode (tokenizer kept)."""
        self._core = None
        clear_cache_fn = getattr(self.ctx, "clear_cache", None)
        if clear_cache_fn is not None:
            clear_cache_fn()
        else:
            import importlib
            importlib.import_module("mlx.core").clear_cache()
