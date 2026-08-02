"""Load LongCat-Video-Avatar 1.5 MLX bundles."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from backend.engine.families.longcat.text_encoder_mlx import UMT5EncoderModel
from backend.engine.families.longcat.vae_mlx import AutoencoderKLWan
from backend.engine.families.longcat_avatar.lora_mlx import merge_lora_into_model
from backend.engine.families.longcat_avatar.transformer_mlx import LongCatVideoAvatarTransformer3DModel
from backend.engine.families.longcat_avatar.whisper_mlx import WhisperEncoder


def _apply_quantization_for_load(dit: nn.Module, quant_cfg: dict[str, Any]) -> None:
    skip_patterns: list[str] = quant_cfg.get(
        "skip_patterns",
        [
            "final_layer.linear",
            "t_embedder.",
            "y_embedder.",
            "adaLN_modulation.",
            "audio_adaLN_modulation.",
        ],
    )

    def predicate(path: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        return not any(pat in path for pat in skip_patterns)

    nn.quantize(
        dit,
        group_size=int(quant_cfg.get("group_size", 64)),
        bits=int(quant_cfg["bits"]),
        class_predicate=predicate,
    )


def load_longcat_avatar_components(
    bundle_root: Path,
    *,
    on_log: Callable[[str, str], None] | None = None,
) -> tuple[Any, Any, Any, Any, Path]:
    """Load VAE, umT5, Whisper, Avatar DiT from an installed avatar bundle root."""
    variant_dir = Path(bundle_root)
    if not variant_dir.is_dir():
        raise RuntimeError(f"LongCat-Avatar bundle directory not found: {variant_dir}")

    for label, path in (
        ("dit", variant_dir / "dit"),
        ("vae", variant_dir / "vae"),
        ("text_encoder", variant_dir / "text_encoder"),
        ("audio_encoder", variant_dir / "audio_encoder"),
    ):
        if not path.is_dir():
            raise RuntimeError(
                f"LongCat-Avatar bundle missing {label}/ under {variant_dir}. "
                "Install mlx-community LongCat-Video-Avatar-1.5-* variant."
            )

    if on_log:
        on_log("info", f"LongCat-Avatar loading weights from {variant_dir.name}")

    vae_cfg = json.loads((variant_dir / "vae" / "config.json").read_text(encoding="utf-8"))
    vae = AutoencoderKLWan.from_config(vae_cfg)
    vae.load_weights(str(variant_dir / "vae" / "diffusion_pytorch_model.safetensors"), strict=False)

    umt5_cfg = json.loads((variant_dir / "text_encoder" / "config.json").read_text(encoding="utf-8"))
    umt5 = UMT5EncoderModel.from_config(umt5_cfg)
    umt5_idx = json.loads((variant_dir / "text_encoder" / "model.safetensors.index.json").read_text(encoding="utf-8"))
    for shard_name in sorted(set(umt5_idx["weight_map"].values())):
        umt5.load_weights(str(variant_dir / "text_encoder" / shard_name), strict=False)

    whisper_cfg = json.loads((variant_dir / "audio_encoder" / "config.json").read_text(encoding="utf-8"))
    whisper = WhisperEncoder.from_config(whisper_cfg)
    whisper.load_weights(str(variant_dir / "audio_encoder" / "model.safetensors"), strict=False)

    dit_cfg = json.loads((variant_dir / "dit" / "config.json").read_text(encoding="utf-8"))
    quant_cfg = dit_cfg.get("quantization")
    dit = LongCatVideoAvatarTransformer3DModel.from_config(dit_cfg)
    if quant_cfg is not None:
        if on_log:
            on_log(
                "info",
                f"LongCat-Avatar DiT {quant_cfg.get('bits')}-bit quant (group_size="
                f"{quant_cfg.get('group_size', 64)})",
            )
        _apply_quantization_for_load(dit, quant_cfg)
    dit_idx = json.loads(
        (variant_dir / "dit" / "diffusion_pytorch_model.safetensors.index.json").read_text(encoding="utf-8")
    )
    for shard_name in sorted(set(dit_idx["weight_map"].values())):
        dit.load_weights(str(variant_dir / "dit" / shard_name), strict=False)

    lora_path = variant_dir / "lora" / "dmd_lora.safetensors"
    if lora_path.is_file():
        from safetensors import safe_open

        state_dict: dict[str, mx.array] = {}
        with safe_open(str(lora_path), framework="numpy") as f:
            for key in f.keys():
                state_dict[key] = mx.array(f.get_tensor(key))
        result = merge_lora_into_model(dit, state_dict, multiplier=1.0)
        if len(result.get("applied") or []) == 0:
            raise RuntimeError("LongCat-Avatar dmd_lora merge applied zero modules")
        if on_log:
            on_log(
                "info",
                f"LongCat-Avatar dmd_lora merged ({len(result['applied'])} modules)",
            )

    mx.eval(vae.parameters(), umt5.parameters(), whisper.parameters(), dit.parameters())
    return vae, umt5, whisper, dit, variant_dir


def encode_prompts(
    text_encoder: Any,
    prompt: str,
    negative_prompt: str,
    variant_dir: Path,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    from transformers import T5TokenizerFast

    tok = T5TokenizerFast.from_pretrained(str(variant_dir / "tokenizer"))

    def _encode_one(text: str) -> tuple[mx.array, mx.array]:
        enc = tok(text, return_tensors="np", padding="max_length", max_length=512, truncation=True)
        ids = mx.array(enc.input_ids)
        mask = mx.array(enc.attention_mask)
        hidden = text_encoder(ids, mask=mask)
        embeds = hidden[:, None, :, :]
        attn = mask[:, None, None, :]
        return embeds, attn

    text_embeds, text_mask = _encode_one(prompt)
    if negative_prompt.strip():
        uncond_embeds, uncond_mask = _encode_one(negative_prompt)
    else:
        empty_ids = mx.zeros((1, 512), dtype=mx.int32)
        empty_mask = mx.zeros((1, 512), dtype=mx.int32)
        uncond_hidden = text_encoder(empty_ids, mask=empty_mask)
        uncond_embeds = uncond_hidden[:, None, :, :]
        uncond_mask = empty_mask[:, None, None, :]

    return text_embeds, text_mask, uncond_embeds, uncond_mask


def load_reference_image(path: Path, height: int, width: int) -> mx.array:
    import numpy as np
    from PIL import Image

    img = Image.open(str(path)).convert("RGB")
    src_w, src_h = img.size
    scale = max(width / src_w, height / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    img = img.crop((left, top, left + width, top + height))
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    arr = arr.transpose(2, 0, 1)[None, :, None, :, :]
    return mx.array(arr)


def video_tensor_to_uint8(video: mx.array) -> Any:
    import numpy as np

    return (
        np.asarray(video).transpose(0, 2, 3, 4, 1)[0] * 127.5 + 127.5
    ).clip(0, 255).astype(np.uint8)


def save_mp4(frames: Any, output_path: str, *, fps: float) -> str:
    import imageio

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return output_path
