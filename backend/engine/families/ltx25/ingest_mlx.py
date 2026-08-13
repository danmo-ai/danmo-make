"""LTX-2.5 bundle ingest — convert official checkpoints to MLX.

Input: the official ``Lightricks/LTX-2.5`` repository layout (available on
ModelScope without auth — modelscope.cn/models/Lightricks/LTX-2.5 — or the
gated Hugging Face repo with a read token):

    diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors
    text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
    vae/ltx-2.5-video-vae-conv-bf16.safetensors
    vae/ltx-2.5-audio-vae-bf16.safetensors
    latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors

Output: an MLX bundle usable by ``backend.engine.families.ltx25``:

    transformer.safetensors       (DiT, optional 4/8-bit affine quantization)
    quantize_config.json          (when quantized)
    connector.safetensors         (video/audio 1D connectors + V2 feature extractor)
    video_vae.safetensors         (encoder + decoder + per-channel statistics)
    audio_vae.safetensors         (audio VAE decoder + vocoder/BWE + mel-stft bases)
    upsampler.safetensors         (spatial x2 latent upsampler)
    text_encoder/                 (mlx-lm Gemma 4 bundle)
    bundle_config.json            (all checkpoint configs, extracted from metadata)

Usage:

    python -m backend.engine.families.ltx25.ingest_mlx \
        --source ~/models/ltx-2.5 --target models/Video/ltx-2.5-distilled-mlx-q4 \
        --quantize 4
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

_BUNDLE_CONFIG_NAME = "bundle_config.json"

_TRANSFORMER_FILE = "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
_TEXT_ENCODER_FILE = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
# Conv-decoder VAE — the in-repo MLX implementation decodes with the conv
# decoder (the diffusion decoder needs neighborhood attention, not ported).
_VIDEO_VAE_FILE = "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
_AUDIO_VAE_FILE = "vae/ltx-2.5-audio-vae-bf16.safetensors"
_UPSAMPLER_FILE = "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

# Hugging Face config of the Gemma 4 fine-tune (upstream ``GEMMA_CONFIG_METADATA_KEY``).
_GEMMA_CONFIG_METADATA_KEY = "gemma_config"


def _metadata(path: Path) -> dict[str, Any]:
    import safetensors

    with safetensors.safe_open(str(path), framework="mlx") as handle:
        return dict(handle.metadata() or {})


def _load_weights(path: Path) -> dict[str, mx.array]:
    return dict(mx.load(str(path)))


def _save(path: Path, weights: dict[str, mx.array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), {k: v.astype(mx.bfloat16) for k, v in weights.items()})


def _meta_config(metadata: dict[str, Any]) -> dict[str, Any]:
    return dict(metadata.get("config", {}) or {})


def _extract_transformer_weights(weights: dict[str, mx.array]) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
    """Split DiT weights from connector/feature-extractor weights."""
    from backend.engine.families.ltx25.weights_mlx import normalize_ltx25_keys

    dit: dict[str, mx.array] = {}
    connector: dict[str, mx.array] = {}
    for key, tensor in weights.items():
        if key.startswith("model.diffusion_model.video_embeddings_connector."):
            connector["video_connector." + key[len("model.diffusion_model.video_embeddings_connector."):]] = tensor
        elif key.startswith("model.diffusion_model.audio_embeddings_connector."):
            connector["audio_connector." + key[len("model.diffusion_model.audio_embeddings_connector."):]] = tensor
        elif key.startswith("model.diffusion_model."):
            dit[key] = tensor
        else:
            raise RuntimeError(f"Unexpected transformer key: {key!r}")
    return normalize_ltx25_keys(dit), connector


def _extract_feature_extractor_weights(te_weights: dict[str, mx.array]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    for key, tensor in te_weights.items():
        if key.startswith("text_embedding_projection."):
            out["feature_extractor." + key[len("text_embedding_projection."):]] = tensor
    if not out:
        raise RuntimeError(
            "No text_embedding_projection.* keys in the LTX 2.5 text encoder checkpoint — "
            "expected the 'with-proj' file (caption_proj_before_connector layout)."
        )
    return out


def _quantize_weights(weights: dict[str, mx.array], *, bits: int, group_size: int) -> dict[str, mx.array]:
    return dict(mx.quantize(weights, group_size=group_size, bits=bits))


def _build_bundle_config(
    transformer_meta: dict[str, Any],
    video_vae_meta: dict[str, Any],
    audio_vae_meta: dict[str, Any],
    upsampler_meta: dict[str, Any],
    gemma_hf_config: dict[str, Any],
) -> dict[str, Any]:
    tcfg = _meta_config(transformer_meta)
    transformer_cfg = dict(tcfg.get("transformer", {}) or {})
    for key in ("caption_proj_before_connector", "gemma_source_checkpoint", "model_version"):
        if key in tcfg:
            transformer_cfg[key] = tcfg[key]

    vae_cfg = _meta_config(video_vae_meta).get("vae", {}) or {}
    audio_cfg = _meta_config(audio_vae_meta).get("audio_vae", {}) or {}
    vocoder_cfg = _meta_config(audio_vae_meta).get("vocoder", {}) or {}
    upsampler_cfg = _meta_config(upsampler_meta)

    if not transformer_cfg.get("num_layers"):
        raise RuntimeError("Transformer config missing 'num_layers' in checkpoint metadata.")
    if not vae_cfg.get("decoder_blocks"):
        raise RuntimeError("Video VAE config missing 'decoder_blocks' in checkpoint metadata.")

    # Audio VAE latent geometry (decoder ddconfig + preprocessing mel bins).
    ddconfig = audio_cfg.get("model", {}).get("params", {}).get("ddconfig", {}) or {}
    mel_cfg = audio_cfg.get("preprocessing", {}).get("mel", {}) or {}
    variables = audio_cfg.get("variables", {}) or {}
    mel_bins = ddconfig.get("mel_bins") or mel_cfg.get("n_mel_channels") or variables.get("mel_bins")
    audio_cfg = dict(audio_cfg)
    audio_cfg["ddconfig"] = ddconfig
    audio_cfg["mel_bins"] = mel_bins

    text_config = dict(gemma_hf_config.get("text_config", {}) or {})
    gemma_cfg: dict[str, Any] = {
        "hidden_size": int(text_config.get("hidden_size", 0)),
        "num_hidden_layers": int(text_config.get("num_hidden_layers", 0)),
        "tokenizer_max_length": 1024,
    }
    if not gemma_cfg["hidden_size"] or not gemma_cfg["num_hidden_layers"]:
        raise RuntimeError("Gemma text_config missing hidden_size / num_hidden_layers.")

    return {
        "transformer": transformer_cfg,
        "vae": vae_cfg,
        "audio_vae": audio_cfg,
        "vocoder": vocoder_cfg,
        "upsampler": upsampler_cfg,
        "gemma": gemma_cfg,
    }


def _resolve_gemma_hf_config(source_te_path: Path) -> dict[str, Any]:
    """Locate the Gemma 4 HF config for the LTX text-encoder checkpoint.

    Priority (upstream ``GemmaAssets.load``):
    1. safetensors metadata ``gemma_config`` (JSON-encoded HF config)
    2. nested ``config.text_encoder_config`` metadata block
    3. ``config.json`` beside the checkpoint
    """
    import safetensors

    with safetensors.safe_open(str(source_te_path), framework="mlx") as handle:
        metadata = dict(handle.metadata() or {})
    raw_config = metadata.get(_GEMMA_CONFIG_METADATA_KEY)
    if raw_config:
        return json.loads(raw_config)
    config_meta = metadata.get("config")
    nested = config_meta.get("text_encoder_config") if isinstance(config_meta, dict) else None
    if isinstance(nested, dict) and nested:
        return dict(nested)
    config_json = source_te_path.with_name("config.json")
    if config_json.is_file():
        return json.loads(config_json.read_text(encoding="utf-8"))
    raise RuntimeError("Cannot locate the Gemma 4 HF config for the LTX text encoder checkpoint.")


def _convert_text_encoder(source_te_path: Path, target_dir: Path) -> None:
    """Convert the HF gemma4-with-proj checkpoint into an mlx-lm bundle."""
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        from mlx_lm import convert as mlx_lm_convert
    except ImportError as exc:
        raise RuntimeError("mlx_lm is required to convert the LTX 2.5 Gemma 4 text encoder.") from exc

    hf_config = _resolve_gemma_hf_config(source_te_path)

    with tempfile.TemporaryDirectory(prefix="ltx25-te-") as tmp:
        tmp_path = Path(tmp)
        shutil.copy2(source_te_path, tmp_path / "model.safetensors")
        rewritten = dict(hf_config)
        rewritten["model_type"] = "gemma4"
        (tmp_path / "config.json").write_text(json.dumps(rewritten, indent=2), encoding="utf-8")
        mlx_lm_convert(
            hf_path=str(tmp_path),
            mlx_path=str(target_dir),
            quantize=False,
        )
    if not (target_dir / "config.json").is_file() or not (target_dir / "model.safetensors").is_file():
        raise RuntimeError("mlx_lm.convert did not produce the expected Gemma 4 bundle layout.")


def ingest_ltx25_bundle(
    *,
    source: Path,
    target: Path,
    quantize: int | None = None,
    group_size: int = 64,
    on_log: Any | None = None,
) -> Path:
    def _log(message: str) -> None:
        if on_log:
            on_log("info", message)

    source = Path(source)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)

    transformer_path = source / _TRANSFORMER_FILE
    te_path = source / _TEXT_ENCODER_FILE
    video_vae_path = source / _VIDEO_VAE_FILE
    audio_vae_path = source / _AUDIO_VAE_FILE
    upsampler_path = source / _UPSAMPLER_FILE
    for path in (transformer_path, te_path, video_vae_path, audio_vae_path, upsampler_path):
        if not path.is_file():
            raise RuntimeError(
                f"LTX 2.5 source file missing: {path}\n"
                "Download the Lightricks/LTX-2.5 repository first — ModelScope "
                "(modelscope.cn/models/Lightricks/LTX-2.5, no auth) or the gated "
                "Hugging Face repo with a read token."
            )

    _log("Reading checkpoint metadata")
    transformer_meta = _metadata(transformer_path)
    video_vae_meta = _metadata(video_vae_path)
    audio_vae_meta = _metadata(audio_vae_path)
    upsampler_meta = _metadata(upsampler_path)
    gemma_hf_config = _resolve_gemma_hf_config(te_path)

    _log("Splitting transformer + connector weights")
    transformer_weights = _load_weights(transformer_path)
    dit_weights, connector_weights = _extract_transformer_weights(transformer_weights)

    _log("Extracting feature-extractor weights from the text encoder checkpoint")
    te_weights = _load_weights(te_path)
    feature_weights = _extract_feature_extractor_weights(te_weights)
    _save(target / "connector.safetensors", {**connector_weights, **feature_weights})

    _log("Converting text encoder (Gemma 4, mlx-lm)")
    _convert_text_encoder(te_path, target / "text_encoder")

    _log("Writing video VAE")
    video_vae_weights = {k: v for k, v in _load_weights(video_vae_path).items()}
    _save(target / "video_vae.safetensors", video_vae_weights)

    _log("Writing audio VAE + vocoder")
    audio_vae_weights = {k: v for k, v in _load_weights(audio_vae_path).items()}
    _save(target / "audio_vae.safetensors", audio_vae_weights)

    _log("Writing spatial upsampler")
    _save(target / "upsampler.safetensors", _load_weights(upsampler_path))

    _log("Writing DiT weights")
    if quantize in (4, 8):
        dit_weights = _quantize_weights(dit_weights, bits=int(quantize), group_size=group_size)
        (target / "quantize_config.json").write_text(
            json.dumps({"quantization": {"bits": int(quantize), "group_size": int(group_size)}}, indent=2),
            encoding="utf-8",
        )
    _save(target / "transformer.safetensors", dit_weights)

    _log("Writing bundle config")
    bundle_config = _build_bundle_config(
        transformer_meta, video_vae_meta, audio_vae_meta, upsampler_meta, gemma_hf_config,
    )
    (target / _BUNDLE_CONFIG_NAME).write_text(json.dumps(bundle_config, indent=2), encoding="utf-8")

    _log(f"LTX 2.5 bundle ready at {target}")
    return target


def run_ltx25_ingest_hook(
    *,
    bundle_root: Path,
    model_name: str,
    version_key: str | None,
    hook_spec: dict[str, Any],
) -> None:
    """Install-hook entry: converts the downloaded HF files into the MLX bundle."""
    _ = model_name
    source = str(hook_spec.get("source") or ".")
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = Path(bundle_root) / source
    quantize = hook_spec.get("quantize")
    group_size = int(hook_spec.get("group_size") or 64)
    ingest_ltx25_bundle(
        source=source_path,
        target=Path(bundle_root),
        quantize=int(quantize) if quantize in (4, 8) else None,
        group_size=group_size,
        on_log=print,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert official LTX-2.5 checkpoints to an MLX bundle.")
    parser.add_argument("--source", required=True, help="Directory holding the downloaded Lightricks/LTX-2.5 files")
    parser.add_argument("--target", required=True, help="Output bundle directory")
    parser.add_argument("--quantize", type=int, choices=(4, 8), default=None, help="Quantize the DiT (4/8-bit)")
    parser.add_argument("--group-size", type=int, default=64, help="Quantization group size")
    args = parser.parse_args(argv)
    ingest_ltx25_bundle(
        source=Path(args.source),
        target=Path(args.target),
        quantize=args.quantize,
        group_size=args.group_size,
        on_log=lambda level, message: print(f"[{level}] {message}"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
