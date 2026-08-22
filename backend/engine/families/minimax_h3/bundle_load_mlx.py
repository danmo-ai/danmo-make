"""Load MiniMax-H3 FL2VA MLX bundles (PipeNetwork DiT + MiniMaxAI FL2VA aux)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
from mlx.utils import tree_unflatten

from backend.engine.families.minimax_h3.bundle_paths_mlx import minimax_h3_aux_root, minimax_h3_tokenizer_root
from backend.engine.families.minimax_h3.dit_quant_mlx import (
    MiniMaxH3QuantConfig,
    apply_minimax_h3_quant_structure,
)
from backend.engine.families.minimax_h3.text_encoder_mlx import MiniMaxH3TextEncoderMLX
from backend.engine.families.minimax_h3.transformer_mlx import (
    MiniMaxH3DiTMLX,
    expected_dit_param_keys,
    is_fp32_dit_key,
)
from backend.engine.families.minimax_h3.vae_mlx import load_audio_vae, load_video_vae

_SKIP_DIT_KEYS = ("rope.inv_freq",)


def _require_bundle(bundle_root: Path) -> Path:
    root = Path(bundle_root)
    if not root.is_dir():
        raise RuntimeError(f"MiniMax-H3 bundle directory not found: {root}")
    if not (root / "quant_config.json").is_file():
        raise RuntimeError(
            f"MiniMax-H3 bundle missing quant_config.json under {root}. "
            "Install a PipeNetwork MLX quant build (pipenetwork/MiniMax-H3-MLX-*bit)."
        )
    if not _dit_weight_shards(root):
        raise RuntimeError(
            f"MiniMax-H3 DiT shards not found under {root}. "
            "Expected model-*-of-*.safetensors from PipeNetwork."
        )
    aux = minimax_h3_aux_root(root)
    missing: list[str] = []
    if not (aux / "video_vae").is_dir() and not (aux / "video_vae" / "source" / "model.safetensors").is_file():
        missing.append("FL2VA/video_vae/")
    if not (aux / "audio_vae").is_dir():
        missing.append("FL2VA/audio_vae/")
    if not (aux / "text_encoder").is_dir():
        missing.append("FL2VA/text_encoder/")
    if not (minimax_h3_tokenizer_root(root) / "tokenizer.json").is_file():
        missing.append("FL2VA/processor/tokenizer.json (or text_encoder/tokenizer.json)")
    if missing:
        raise RuntimeError(
            f"MiniMax-H3 bundle incomplete under {root}: missing {missing}. "
            "Install MiniMaxAI/MiniMax-H3 FL2VA aux via bundle_repos follow-up."
        )
    return root


def _dit_weight_shards(root: Path) -> list[Path]:
    """PipeNetwork DiT shards at bundle root or under ``transformer/``."""
    transformer_dir = root / "transformer"
    if transformer_dir.is_dir():
        index = transformer_dir / "model.safetensors.index.json"
        if index.is_file():
            return _shards_from_index(transformer_dir, index)
        shards = sorted(transformer_dir.glob("model-*-of-*.safetensors"))
        if shards:
            return shards

    index = root / "model.safetensors.index.json"
    if index.is_file():
        return _shards_from_index(root, index)

    shards = sorted(root.glob("model-*-of-*.safetensors"))
    return shards


def _shards_from_index(model_dir: Path, index_path: Path) -> list[Path]:
    weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map") or {}
    names = sorted(set(weight_map.values()))
    return [model_dir / name for name in names]


def _normalize_dit_cfg(raw: dict[str, Any]) -> dict[str, Any]:
    """Map PipeNetwork / upstream FL2VA DiT config onto ``MiniMaxH3DiTMLX.from_config``."""
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
    if "freq_dim" not in cfg and "timestep_input_dim" in cfg:
        cfg["freq_dim"] = cfg["timestep_input_dim"]
    cfg.setdefault("patch_size", (1, 2, 2))
    cfg.setdefault("freq_dim", 256)
    cfg.setdefault("time_embed_hidden_dim", int(cfg.get("hidden_size", 5376)))
    return cfg


def _load_bundle_cfg(root: Path) -> dict[str, Any]:
    aux = minimax_h3_aux_root(root)
    index_path = aux / "model_index.json"
    if index_path.is_file():
        cfg = json.loads(index_path.read_text(encoding="utf-8"))
        meta = cfg.get("_minimax_h3") if isinstance(cfg.get("_minimax_h3"), dict) else {}
        cfg.setdefault("model_type", "minimax_h3")
        cfg.setdefault("partition", meta.get("partition", "fl2va"))
        return cfg
    cfg_path = root / "config.json"
    if cfg_path.is_file():
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    raise RuntimeError(
        f"MiniMax-H3 bundle metadata not found under {root} "
        "(expected FL2VA/model_index.json or config.json)."
    )


def _resolve_dit_config(root: Path) -> dict[str, Any]:
    cfg_path = root / "config.json"
    if not cfg_path.is_file():
        raise RuntimeError(f"MiniMax-H3 PipeNetwork DiT config.json missing under {root}")
    dit_only = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(dit_only, dict) or dit_only.get("hidden_size") is None:
        raise RuntimeError(f"Invalid PipeNetwork DiT config.json under {root}")
    return _normalize_dit_cfg(dit_only)


def _normalize_te_cfg(raw: dict[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    nested = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else raw
    if not isinstance(nested, dict):
        nested = raw
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
        if k in nested:
            cfg[k] = nested[k]
    return cfg


def _load_dit_shards(
    dit: MiniMaxH3DiTMLX,
    shards: list[Path],
    *,
    load_fn: Callable[[str], dict[str, mx.array]] | None,
    strict: bool = True,
) -> None:
    expected = expected_dit_param_keys(dit)
    weights: dict[str, mx.array] = {}
    unexpected: list[str] = []

    for shard in shards:
        if load_fn is not None:
            loaded = load_fn(str(shard))
            if not isinstance(loaded, dict):
                loaded = dict(loaded)
        else:
            loaded = dict(mx.load(str(shard)))
        for key, tensor in loaded.items():
            if key in _SKIP_DIT_KEYS:
                continue
            if key not in expected:
                unexpected.append(key)
                continue
            if is_fp32_dit_key(key) and tensor.dtype != mx.float32:
                tensor = tensor.astype(mx.float32)
            weights[key] = tensor

    missing = sorted(expected - weights.keys())
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"MiniMax-H3 DiT checkpoint/module mismatch: {len(missing)} missing "
            f"(e.g. {missing[:4]}), {len(unexpected)} unexpected (e.g. {unexpected[:4]})."
        )
    dit.update(tree_unflatten(list(weights.items())))


def load_minimax_h3_components(
    bundle_root: Path,
    *,
    ctx: Any,
    on_log: Callable[[str, str], None] | None = None,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Load VAE, text encoder, and DiT from PipeNetwork DiT + MiniMaxAI FL2VA aux."""
    root = _require_bundle(bundle_root)
    cfg = _load_bundle_cfg(root)
    model_type = str(cfg.get("model_type", "minimax_h3")).replace("-", "_")
    if model_type not in ("minimax_h3",):
        raise RuntimeError(
            f"Unexpected MiniMax-H3 model_type={cfg.get('model_type')!r} in bundle metadata"
        )
    partition = str(cfg.get("partition", "fl2va")).lower()
    if partition not in ("fl2va", ""):
        raise RuntimeError(f"Phase1 supports FL2VA only; bundle partition={cfg.get('partition')!r}")

    dit_cfg = _resolve_dit_config(root)
    aux = minimax_h3_aux_root(root)
    te_cfg_path = aux / "text_encoder" / "config.json"
    te_cfg = _normalize_te_cfg(
        json.loads(te_cfg_path.read_text(encoding="utf-8")) if te_cfg_path.is_file() else {},
    )

    if on_log:
        on_log("info", f"MiniMax-H3 loading weights from {root.name}")

    load_fn = getattr(ctx, "load_weights", None)
    video_vae = load_video_vae(root, load_fn=load_fn)
    audio_vae = load_audio_vae(root, load_fn=load_fn)

    text_encoder = MiniMaxH3TextEncoderMLX(
        ctx,
        model_path=root,
        tokenizer_path=root,
        config=te_cfg,
        quant_cfg=None,
    )
    text_encoder._ensure_weights()

    dit = MiniMaxH3DiTMLX.from_config(dit_cfg)
    quant_path = root / "quant_config.json"
    recipe = json.loads(quant_path.read_text(encoding="utf-8"))
    pn_quant = MiniMaxH3QuantConfig.from_recipe(recipe)
    if on_log:
        on_log(
            "info",
            f"MiniMax-H3 DiT PipeNetwork {pn_quant.bits}-bit "
            f"(group_size={pn_quant.group_size}, quantize_adaln={pn_quant.quantize_adaln})",
        )
    apply_minimax_h3_quant_structure(dit, pn_quant)

    shards = _dit_weight_shards(root)
    _load_dit_shards(dit, shards, load_fn=load_fn, strict=True)

    mx.eval(video_vae.parameters(), audio_vae.parameters(), text_encoder.model.parameters(), dit.parameters())
    return video_vae, audio_vae, text_encoder, dit, cfg
