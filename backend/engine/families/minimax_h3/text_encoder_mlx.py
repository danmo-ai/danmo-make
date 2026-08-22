"""MiniMax-H3 Qwen3-VL conditioner — thin glue over mlx-vlm.

Neural net / vision merge / MRoPE / deepstack: **mlx-vlm** ``qwen3_vl.Model``.
Local code is only what mlx-vlm cannot do for this product path:

- load ddalcu flat affine ``text_encoder.safetensors`` (not an mlx-vlm repo layout)
- MiniMax FL2VA presentation (`": "` + vision tags; PipeNetwork / diffusers parity)
- read unnormalized ``hidden_states[50]`` (mlx-vlm forward always applies final RMSNorm)

ddalcu keeps the first 50 decoder layers and drops final norm / lm_head; after those
layers the activation is the HS[50] MiniMax-H3 expects.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from backend.engine.common.model.quantized_load_mlx import (
    apply_quantized_skeleton,
    collect_affine_quant_bases,
)
from backend.engine.runtime.mlx_runtime import load_weights_dict, run_eval

from .packing import (
    MINIMAX_H3_TEXT_ENCODER_LAYER,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
)

_DEFAULT_TEXT = {
    "vocab_size": 151936,
    "hidden_size": 5120,
    "num_hidden_layers": 50,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 25600,
    "rms_norm_eps": 1e-6,
    "rope_theta": 5000000.0,
    "max_position_embeddings": 262144,
}

_DEFAULT_VISION = {
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "out_hidden_size": 5120,
    "num_heads": 16,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "num_position_embeddings": 2304,
    "deepstack_visual_indexes": [8, 16, 24],
}

_IMAGE_TOKEN_ID = 151655
_VIDEO_TOKEN_ID = 151656
_VISION_START_TOKEN_ID = 151652
_VISION_END_TOKEN_ID = 151653


def _remap_te_weight_key(key: str) -> str | None:
    """Map ddalcu / HF TE keys onto mlx-vlm ``Model`` parameter paths."""
    if key.startswith("visual."):
        return "vision_tower." + key[len("visual.") :]
    if key.startswith("model.visual."):
        return "vision_tower." + key[len("model.visual.") :]
    if key.startswith("model.language_model."):
        return "language_model.model." + key[len("model.language_model.") :]
    if key.startswith("language_model."):
        return key if key.startswith("language_model.model.") else "language_model.model." + key[len("language_model.") :]
    if key.startswith("model."):
        # ddalcu flat pack: language tower is ``model.layers.*`` / ``model.embed_tokens.*``
        rest = key[len("model.") :]
        if rest.startswith("visual."):
            return "vision_tower." + rest[len("visual.") :]
        return "language_model.model." + rest
    if key.startswith("vision_tower.") or key.startswith("language_model."):
        return key
    if "lm_head" in key:
        return None
    return key


class MiniMaxH3TextEncoderMLX:
    """Encode MiniMax-H3 presentations with Qwen3-VL (LM + vision) via mlx-vlm."""

    def __init__(
        self,
        ctx: Any,
        model_path: str | Path,
        tokenizer_path: str | Path | None = None,
        *,
        config: dict[str, Any] | None = None,
        vision_config: dict[str, Any] | None = None,
        quant_cfg: dict[str, Any] | None = None,
    ):
        self.ctx = ctx
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else self.model_path
        self.quant_cfg = dict(quant_cfg) if quant_cfg else None

        text_cfg = dict(_DEFAULT_TEXT)
        vis_cfg = dict(_DEFAULT_VISION)
        cfg_path = self.model_path / "config.json"
        if cfg_path.is_file():
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            # Full Qwen3-VL config (official MiniMax subfolder) or ddalcu abbreviated.
            nested_text = raw.get("text_config") or raw.get("llm_config")
            if isinstance(nested_text, dict):
                text_cfg.update({k: nested_text[k] for k in _DEFAULT_TEXT if k in nested_text})
            te = raw.get("text_encoder")
            if isinstance(te, dict):
                text_cfg.update(_normalize_te_abbrev(te))
            nested_vis = raw.get("vision_config")
            if isinstance(nested_vis, dict):
                vis_cfg.update({k: nested_vis[k] for k in _DEFAULT_VISION if k in nested_vis})
                if "deepstack_visual_indexes" in nested_vis:
                    vis_cfg["deepstack_visual_indexes"] = list(nested_vis["deepstack_visual_indexes"])
        if config:
            text_cfg.update(config)
        if vision_config:
            vis_cfg.update(vision_config)
            if "deepstack_visual_indexes" in vision_config:
                vis_cfg["deepstack_visual_indexes"] = list(vision_config["deepstack_visual_indexes"])

        self.text_config = text_cfg
        self.vision_config = vis_cfg
        self.vlm = _build_vlm(text_cfg, vis_cfg)
        # Back-compat alias used by bundle_load mx.eval / quantize hooks.
        self.model = self.vlm
        self._tokenizer = None
        self._image_processor = None
        self._loaded = False
        self._has_vision_weights = False

    def _ensure_weights(self) -> None:
        if self._loaded:
            return
        weight_path = self._resolve_weight_file()
        load_fn = getattr(self.ctx, "load_weights", None)
        raw = load_weights_dict(load_fn, str(weight_path))
        weights: dict[str, mx.array] = {}
        vision_keys = 0
        for k, v in raw.items():
            mapped = _remap_te_weight_key(k)
            if mapped is None:
                continue
            arr = v if isinstance(v, mx.array) else mx.array(v)
            weights[mapped] = arr
            if mapped.startswith("vision_tower."):
                vision_keys += 1
        if not weights:
            raise RuntimeError(
                f"MiniMax-H3 text encoder found no usable keys in {weight_path}"
            )
        # HF Conv3d → MLX channel-last via mlx-vlm VisionModel.sanitize.
        weights = self.vlm.vision_tower.sanitize(weights)

        bits = None
        group_size = 64
        if self.quant_cfg is not None:
            bits = int(self.quant_cfg["bits"])
            group_size = int(self.quant_cfg.get("group_size", 64))
        bases = collect_affine_quant_bases(weights)
        if bases:
            if bits is None:
                raise RuntimeError(
                    f"MiniMax-H3 text encoder {weight_path.name} contains affine-quant "
                    "scales/biases but bundle config.json has no quantization{{bits,group_size}}."
                )
            apply_quantized_skeleton(self.vlm, bases, bits=bits, group_size=group_size)

        from mlx.utils import tree_flatten, tree_unflatten

        params = dict(tree_flatten(self.vlm.parameters()))
        assigned = 0
        vision_assigned = 0
        for k, v in weights.items():
            if k not in params:
                continue
            if tuple(v.shape) != tuple(params[k].shape):
                continue
            params[k] = v.astype(params[k].dtype) if hasattr(v, "astype") else v
            assigned += 1
            if k.startswith("vision_tower."):
                vision_assigned += 1
        if assigned == 0:
            raise RuntimeError(
                f"MiniMax-H3 text encoder assigned 0 weights from {weight_path} "
                f"(saw {len(weights)} remapped keys; model has {len(params)} params)."
            )
        self.vlm.update(tree_unflatten(list(params.items())))
        self._has_vision_weights = vision_assigned > 0
        if vision_keys > 0 and vision_assigned == 0:
            raise RuntimeError(
                f"MiniMax-H3 text encoder saw {vision_keys} vision keys in {weight_path.name} "
                "but assigned 0 into vision_tower — key remap/quantize skeleton mismatch."
            )
        self._loaded = True

    def _resolve_weight_file(self) -> Path:
        candidates = [
            self.model_path / "text_encoder.safetensors",
            self.model_path / "text_encoder" / "model.safetensors",
        ]
        for path in candidates:
            if path.is_file():
                return path
        # HF-style sharded text_encoder/
        te_dir = self.model_path / "text_encoder"
        if te_dir.is_dir():
            shards = sorted(te_dir.glob("*.safetensors"))
            if len(shards) == 1:
                return shards[0]
            if shards:
                raise RuntimeError(
                    f"MiniMax-H3 text encoder under {te_dir} is sharded ({len(shards)} files); "
                    "Phase1 expects a single text_encoder.safetensors (ddalcu layout)."
                )
        raise RuntimeError(
            f"MiniMax-H3 text encoder weights not found under {self.model_path} "
            "(expected text_encoder.safetensors)."
        )

    def _tokenizer_fast(self):
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import Qwen2TokenizerFast
        except ImportError as exc:
            raise RuntimeError(
                "MiniMax-H3 text encoding requires `transformers` (Qwen2TokenizerFast)."
            ) from exc
        self._tokenizer = Qwen2TokenizerFast.from_pretrained(
            str(self.tokenizer_path), local_files_only=True
        )
        return self._tokenizer

    def _ensure_image_processor(self):
        if self._image_processor is not None:
            return self._image_processor
        try:
            from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor
        except ImportError as exc:
            raise RuntimeError(
                "MiniMax-H3 keyframe vision requires mlx-vlm Qwen3VLImageProcessor."
            ) from exc
        patch = int(self.vision_config.get("patch_size", 16))
        merge = int(self.vision_config.get("spatial_merge_size", 2))
        temporal = int(self.vision_config.get("temporal_patch_size", 2))
        # MiniMax FL2VA processor allows very large canvases; keep headroom above 768p.
        self._image_processor = Qwen3VLImageProcessor(
            patch_size=patch,
            temporal_patch_size=temporal,
            merge_size=merge,
            min_pixels=patch * patch,
            max_pixels=16_777_216,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )
        return self._image_processor

    def conditioning_hidden_states(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        *,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        layer: int = MINIMAX_H3_TEXT_ENCODER_LAYER,
    ) -> mx.array:
        """Return unnormalized hidden states at conditioner layer ``layer``."""
        from mlx_vlm.models.base import create_attention_mask

        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)

        emb_feats = self.vlm.get_input_embeddings(
            input_ids,
            pixel_values,
            image_grid_thw=image_grid_thw,
            mask=attention_mask,
        )
        hidden = emb_feats.inputs_embeds
        visual_pos_masks = emb_feats.visual_pos_masks
        deepstack = emb_feats.deepstack_visual_embeds
        position_ids = getattr(self.vlm.language_model, "_position_ids", None)

        base = self.vlm.language_model.model
        n = int(base.num_hidden_layers)
        if n < layer:
            raise RuntimeError(
                f"MiniMax-H3 conditions on hidden_states[{layer}] but the loaded "
                f"Qwen3 tower has only {n} decoder layers."
            )
        attn_mask = create_attention_mask(hidden, None)
        # outs[0]=embed, outs[i]=after layer i  → outs[layer] == HF hidden_states[layer]
        outs = [hidden]
        for layer_idx, decoder in enumerate(base.layers):
            hidden = decoder(hidden, attn_mask, None, position_ids, None)
            if deepstack is not None and layer_idx in range(len(deepstack)):
                hidden = base._deepstack_process(
                    hidden, visual_pos_masks, deepstack[layer_idx]
                )
            outs.append(hidden)

        feats = outs[layer]
        # Final norm intentionally unused — matches Diffusers unnormalized HS[50].
        _ = base.norm
        if feats.ndim != 3 or int(feats.shape[-1]) != int(self.text_config["hidden_size"]):
            raise RuntimeError(
                f"MiniMax-H3 text conditioning shape mismatch: expected "
                f"[B, L, {self.text_config['hidden_size']}], got {tuple(feats.shape)}"
            )
        return feats

    def encode_prompt(
        self,
        prompt: str,
        images: list | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Return ``(prompt_embeds [1,L,H], text_token_tags [L])``.

        Presentation matches Diffusers ``MiniMaxH3TextEncoderStep``: verbatim prompt;
        each keyframe prepends ``": "`` + vision block. Vision-block rows are tagged as video.
        """
        if not isinstance(prompt, str):
            raise ValueError(
                f"MiniMax-H3 packs one request into one sequence, so `prompt` must be a string, got {type(prompt)}"
            )
        self._ensure_weights()
        tok = self._tokenizer_fast()
        token_ids: list[int] = []
        token_tags: list[int] = []

        pixel_values = None
        image_grid_thw = None
        if images:
            if not self._has_vision_weights:
                raise RuntimeError(
                    "MiniMax-H3 fl2va keyframe conditioning requires vision weights "
                    f"(visual.*) in {self._resolve_weight_file().name}, but none were loaded."
                )
            proc = self._ensure_image_processor()
            vision = proc(images=images, return_tensors="np")
            pixel_np = np.asarray(vision["pixel_values"])
            grid_np = np.asarray(vision["image_grid_thw"])
            merge_size = int(proc.merge_size) ** 2
            if grid_np.ndim != 2 or int(grid_np.shape[0]) != len(images):
                raise RuntimeError(
                    f"MiniMax-H3 image_grid_thw shape {grid_np.shape} does not match "
                    f"{len(images)} keyframe image(s)."
                )
            for index in range(len(images)):
                num_image_tokens = int(np.prod(grid_np[index])) // merge_size
                if num_image_tokens <= 0:
                    raise RuntimeError(
                        f"MiniMax-H3 keyframe {index} produced {num_image_tokens} vision tokens "
                        f"(grid={tuple(grid_np[index].tolist())}, merge={merge_size})."
                    )
                label_ids = tok(": ", add_special_tokens=False)["input_ids"]
                vision_ids = (
                    [_VISION_START_TOKEN_ID]
                    + [_IMAGE_TOKEN_ID] * num_image_tokens
                    + [_VISION_END_TOKEN_ID]
                )
                token_ids += list(label_ids) + vision_ids
                token_tags += [MINIMAX_H3_TEXT_TAG] * len(label_ids) + [
                    MINIMAX_H3_VIDEO_TAG
                ] * len(vision_ids)
            pixel_values = mx.array(pixel_np)
            image_grid_thw = mx.array(grid_np)

        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        token_ids += list(prompt_ids)
        token_tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)
        if not token_ids:
            raise RuntimeError("MiniMax-H3 text encoder produced an empty token sequence.")

        input_ids = mx.array([token_ids], dtype=mx.int32)
        attention_mask = mx.ones_like(input_ids)
        embeds = self.conditioning_hidden_states(
            input_ids,
            attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            layer=MINIMAX_H3_TEXT_ENCODER_LAYER,
        )
        run_eval(getattr(self.ctx, "eval", None), embeds)
        hidden = int(self.text_config["hidden_size"])
        if int(embeds.shape[-1]) != hidden:
            raise RuntimeError(
                f"MiniMax-H3 prompt embeds last dim must be hidden_size ({hidden}), "
                f"got {tuple(embeds.shape)}"
            )
        tags = mx.array(np.asarray(token_tags, dtype=np.int32))
        if int(tags.shape[0]) != int(embeds.shape[1]):
            raise RuntimeError(
                f"text_token_tags length {int(tags.shape[0])} != embed seq {int(embeds.shape[1])}"
            )
        return embeds, tags


def _normalize_te_abbrev(raw: dict[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    mapping = {
        "hidden": "hidden_size",
        "layers": "num_hidden_layers",
        "heads": "num_attention_heads",
        "kv_heads": "num_key_value_heads",
        "head_dim": "head_dim",
        "intermediate": "intermediate_size",
        "theta": "rope_theta",
        "vocab_size": "vocab_size",
        "rms_norm_eps": "rms_norm_eps",
        "max_position_embeddings": "max_position_embeddings",
        "hidden_size": "hidden_size",
        "num_hidden_layers": "num_hidden_layers",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_key_value_heads",
        "intermediate_size": "intermediate_size",
        "rope_theta": "rope_theta",
    }
    for src, dst in mapping.items():
        if src in raw:
            cfg[dst] = raw[src]
    return cfg


def _build_vlm(text_cfg: dict[str, Any], vis_cfg: dict[str, Any]):
    try:
        from mlx_vlm.models.qwen3_vl.config import ModelConfig, TextConfig, VisionConfig
        from mlx_vlm.models.qwen3_vl.qwen3_vl import Model
    except ImportError as exc:
        raise RuntimeError(
            "MiniMax-H3 FL2VA text/vision encoding requires `mlx-vlm` (Qwen3-VL)."
        ) from exc

    text = TextConfig(
        model_type="qwen3_vl_text",
        num_hidden_layers=int(text_cfg["num_hidden_layers"]),
        hidden_size=int(text_cfg["hidden_size"]),
        intermediate_size=int(text_cfg["intermediate_size"]),
        num_attention_heads=int(text_cfg["num_attention_heads"]),
        rms_norm_eps=float(text_cfg.get("rms_norm_eps", 1e-6)),
        vocab_size=int(text_cfg["vocab_size"]),
        num_key_value_heads=int(text_cfg["num_key_value_heads"]),
        head_dim=int(text_cfg["head_dim"]),
        rope_theta=float(text_cfg["rope_theta"]),
        max_position_embeddings=int(text_cfg.get("max_position_embeddings", 262144)),
        rope_scaling={
            "type": "default",
            "mrope_section": [24, 20, 20],
            "mrope_interleaved": True,
        },
    )
    vision = VisionConfig(
        model_type="qwen3_vl",
        depth=int(vis_cfg["depth"]),
        hidden_size=int(vis_cfg["hidden_size"]),
        intermediate_size=int(vis_cfg["intermediate_size"]),
        out_hidden_size=int(vis_cfg["out_hidden_size"]),
        num_heads=int(vis_cfg["num_heads"]),
        patch_size=int(vis_cfg["patch_size"]),
        spatial_merge_size=int(vis_cfg["spatial_merge_size"]),
        temporal_patch_size=int(vis_cfg["temporal_patch_size"]),
        num_position_embeddings=int(vis_cfg.get("num_position_embeddings", 2304)),
        deepstack_visual_indexes=list(vis_cfg.get("deepstack_visual_indexes") or []),
    )
    cfg = ModelConfig(
        text_config=text,
        vision_config=vision,
        model_type="qwen3_vl",
        image_token_id=_IMAGE_TOKEN_ID,
        video_token_id=_VIDEO_TOKEN_ID,
        vision_start_token_id=_VISION_START_TOKEN_ID,
        vision_end_token_id=_VISION_END_TOKEN_ID,
        vocab_size=int(text_cfg["vocab_size"]),
    )
    return Model(cfg)


__all__ = [
    "MiniMaxH3TextEncoderMLX",
    "MINIMAX_H3_TEXT_ENCODER_LAYER",
    "MINIMAX_H3_TEXT_TAG",
    "MINIMAX_H3_VIDEO_TAG",
]
