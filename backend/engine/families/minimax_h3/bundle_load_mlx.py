"""Load MiniMax-H3 FL2VA MLX bundles (flat ddalcu layout)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from backend.engine.families.minimax_h3.text_encoder_mlx import MiniMaxH3TextEncoderMLX
from backend.engine.families.minimax_h3.transformer_mlx import MiniMaxH3DiTMLX
from backend.engine.families.minimax_h3.vae_mlx import load_audio_vae, load_video_vae

_REQUIRED_FILES = (
    "config.json",
    "transformer.safetensors",
    "text_encoder.safetensors",
    "video_vae.safetensors",
    "audio_vae.safetensors",
    "tokenizer.json",
)


def _require_bundle(bundle_root: Path) -> Path:
    root = Path(bundle_root)
    if not root.is_dir():
        raise RuntimeError(f"MiniMax-H3 bundle directory not found: {root}")
    missing = [name for name in _REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise RuntimeError(
            f"MiniMax-H3 bundle incomplete under {root}: missing {missing}. "
            "Install mlx-q4 or mlx-q8 (flat safetensors + tokenizer + config.json)."
        )
    return root


def _apply_affine_quant(module: nn.Module, quant_cfg: dict[str, Any]) -> None:
    bits = int(quant_cfg["bits"])
    group_size = int(quant_cfg.get("group_size", 64))
    skip_patterns: list[str] = list(
        quant_cfg.get(
            "skip_patterns",
            [
                "proj_in",
                "audio_proj_in",
                "proj_out",
                "audio_proj_out",
                "time_embedder",
                "time_proj",
                "rope",
                "embed_tokens",
                "lm_head",
            ],
        )
    )

    def predicate(path: str, mod: nn.Module) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        return not any(pat in path for pat in skip_patterns)

    nn.quantize(module, group_size=group_size, bits=bits, class_predicate=predicate)


def _normalize_dit_cfg(raw: dict[str, Any]) -> dict[str, Any]:
    """Map ddalcu / Diffusers FL2VA key aliases onto ``MiniMaxH3DiTMLX.from_config``."""
    cfg = dict(raw)
    if "ffn_dim" not in cfg and "ffn_hidden_size" in cfg:
        cfg["ffn_dim"] = cfg["ffn_hidden_size"]
    if "in_channels" not in cfg and "latents_dim" in cfg:
        cfg["in_channels"] = cfg["latents_dim"]
    if "audio_in_channels" not in cfg and "audio_latents_dim" in cfg:
        cfg["audio_in_channels"] = cfg["audio_latents_dim"]
    if "rope_freq_dim" not in cfg and "rope_inv_freq_len" in cfg:
        cfg["rope_freq_dim"] = cfg["rope_inv_freq_len"]
    if "num_refiner_layers" not in cfg and "token_refiner_num_layers" in cfg:
        cfg["num_refiner_layers"] = cfg["token_refiner_num_layers"]
    if "time_embed_hidden_dim" not in cfg and "time_embed_hidden_size" in cfg:
        cfg["time_embed_hidden_dim"] = cfg["time_embed_hidden_size"]
    cfg.setdefault("patch_size", (1, 2, 2))
    cfg.setdefault("freq_dim", 256)
    cfg.setdefault("time_embed_hidden_dim", int(cfg.get("hidden_size", 5376)))
    return cfg


def _normalize_te_cfg(raw: dict[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    # ddalcu abbreviated keys
    if "hidden" in raw:
        cfg["hidden_size"] = int(raw["hidden"])
    if "layers" in raw:
        cfg["num_hidden_layers"] = int(raw["layers"])
    if "heads" in raw:
        cfg["num_attention_heads"] = int(raw["heads"])
    if "kv_heads" in raw:
        cfg["num_key_value_heads"] = int(raw["kv_heads"])
    if "head_dim" in raw:
        cfg["head_dim"] = int(raw["head_dim"])
    if "intermediate" in raw:
        cfg["intermediate_size"] = int(raw["intermediate"])
    if "theta" in raw:
        cfg["rope_theta"] = float(raw["theta"])
    for k in (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "rope_theta",
        "vocab_size",
        "rms_norm_eps",
        "max_position_embeddings",
    ):
        if k in raw:
            cfg[k] = raw[k]
    return cfg


def load_minimax_h3_components(
    bundle_root: Path,
    *,
    ctx: Any,
    on_log: Callable[[str, str], None] | None = None,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Load video VAE, audio VAE, text encoder, and DiT from a ddalcu-style bundle."""
    root = _require_bundle(bundle_root)
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    model_type = str(cfg.get("model_type", "")).replace("-", "_")
    if model_type not in ("minimax_h3",):
        raise RuntimeError(
            f"Unexpected MiniMax-H3 config model_type={cfg.get('model_type')!r} in {root / 'config.json'}"
        )
    if str(cfg.get("partition", "fl2va")).lower() != "fl2va":
        raise RuntimeError(
            f"Phase1 supports FL2VA only; bundle partition={cfg.get('partition')!r}"
        )

    quant_cfg = cfg.get("quantization")
    dit_cfg = _normalize_dit_cfg(dict(cfg.get("transformer") or {}))
    te_cfg = _normalize_te_cfg(dict(cfg.get("text_encoder") or {}))

    if on_log:
        on_log("info", f"MiniMax-H3 loading weights from {root.name}")

    load_fn = getattr(ctx, "load_weights", None)
    video_vae = load_video_vae(root, load_fn=load_fn)
    audio_vae = load_audio_vae(root, load_fn=load_fn)

    # Flat ddalcu pack: language + vision in text_encoder.safetensors (+ tokenizer at root).
    # Affine skeleton is applied inside the encoder from checkpoint *.scales (mixed dense/quant).
    text_encoder = MiniMaxH3TextEncoderMLX(
        ctx,
        model_path=root,
        tokenizer_path=root,
        config=te_cfg,
        quant_cfg=dict(quant_cfg) if quant_cfg is not None else None,
    )
    if quant_cfg is not None and on_log:
        on_log(
            "info",
            f"MiniMax-H3 text encoder {quant_cfg.get('bits')}-bit affine "
            f"(group_size={quant_cfg.get('group_size', 64)}; vision+LM from text_encoder.safetensors)",
        )
    text_encoder._ensure_weights()

    dit = MiniMaxH3DiTMLX.from_config(dit_cfg)
    if quant_cfg is not None:
        if on_log:
            on_log(
                "info",
                f"MiniMax-H3 DiT {quant_cfg.get('bits')}-bit affine "
                f"(group_size={quant_cfg.get('group_size', 64)})",
            )
        _apply_affine_quant(dit, quant_cfg)
    # Prefer ctx.load_weights if available (handles quantized tensors).
    if load_fn is not None:
        weights = load_fn(str(root / "transformer.safetensors"))
        dit.load_weights(list(weights.items()) if isinstance(weights, dict) else weights, strict=False)
    else:
        dit.load_weights(str(root / "transformer.safetensors"), strict=False)

    mx.eval(video_vae.parameters(), audio_vae.parameters(), text_encoder.model.parameters(), dit.parameters())
    return video_vae, audio_vae, text_encoder, dit, cfg
