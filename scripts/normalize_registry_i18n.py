#!/usr/bin/env python3
"""Normalize models_registry.json user-facing i18n labels.

- versions.*.name → bilingual {zh, en}; quant tier only (no ModelScope / HF in label)
- name / description → capability text only; download source is shown via badges
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "default_config" / "models_registry.json"

sys.path.insert(0, str(ROOT))

from backend.core.version_labels import expected_version_display_name  # noqa: E402

CJK_RE = re.compile(r"[\u4e00-\u9fff]")

DESCRIPTION_OVERRIDES: dict[str, dict[str, str]] = {
    "flux1-krea-dev": {
        "zh": "FLUX.1 Krea 合作版，增强真实感，12B 参数。",
        "en": "FLUX.1 Krea collaboration variant with enhanced realism; 12B parameters.",
    },
    "flux2-klein-4b": {
        "zh": "FLUX.2 Klein 4B 蒸馏版，4 步快速出图，质量优秀。",
        "en": "FLUX.2 Klein 4B distilled variant; fast 4-step generation with strong quality.",
    },
    "flux2-klein-base-4b": {
        "zh": "FLUX.2 Klein 4B 基础版，50 步高质量，支持训练。",
        "en": "FLUX.2 Klein 4B base model; high quality at 50 steps and trainable.",
    },
    "flux2-klein-9b": {
        "zh": "FLUX.2 Klein 9B 蒸馏版，8 步真实感出图。",
        "en": "FLUX.2 Klein 9B distilled; realistic 8-step generation.",
    },
    "flux2-klein-base-9b": {
        "zh": "FLUX.2 Klein 9B 基础版，50 步高质量。",
        "en": "FLUX.2 Klein 9B base model; high quality at 50 steps.",
    },
    "flux1-dev": {
        "zh": "FLUX.1 开发版，28 步高质量出图，12B 参数。",
        "en": "FLUX.1 dev variant; high quality at 28 steps, 12B parameters.",
    },
    "fibo-lite": {
        "zh": "Bria FIBO-Lite 轻量版，8B 参数，支持 JSON 结构化提示。",
        "en": "Bria FIBO-Lite lightweight 8B model with JSON-structured prompts.",
    },
    "fibo-edit": {
        "zh": "Bria FIBO-Edit 图像编辑模型，支持自然语言图像编辑。",
        "en": "Bria FIBO-Edit image editing with natural-language instructions.",
    },
    "fibo-edit-rmbg": {
        "zh": "Bria FIBO-Edit-RMBG 背景移除模型，输出透明 PNG。",
        "en": "Bria FIBO-Edit-RMBG background removal; outputs transparent PNG.",
    },
    "seedvr2-3b": {
        "zh": "SeedVR2 3B 超分辨率模型，用于图像放大与增强。",
        "en": "SeedVR2 3B upscaler for image enlargement and enhancement.",
    },
    "seedvr2-7b": {
        "zh": "SeedVR2 7B 超分辨率模型，更高质量的放大效果。",
        "en": "SeedVR2 7B upscaler for higher-quality enlargement.",
    },
    "depth-pro": {
        "zh": "Apple Depth Pro 深度估计模型，生成高质量深度图。",
        "en": "Apple Depth Pro depth estimation for high-quality depth maps.",
    },
    "flux-canny-controlnet": {
        "zh": "FLUX.1 Canny 边缘控制，让生成遵循边缘轮廓。",
        "en": "FLUX.1 Canny edge control for generation guided by edge maps.",
    },
    "flux-depth-controlnet": {
        "zh": "FLUX.1 Depth 深度控制，让生成遵循空间深度关系。",
        "en": "FLUX.1 Depth control for generation guided by depth structure.",
    },
    "flux-fill-controlnet": {
        "zh": "FLUX.1 Fill 局部重绘与图像修复/外扩。",
        "en": "FLUX.1 Fill for inpainting, retouch, and outpainting.",
    },
    "flux-redux": {
        "zh": "FLUX.1 Redux 图像变体与风格迁移，混合参考图与文本提示。",
        "en": "FLUX.1 Redux for image variation and style transfer from reference + prompt.",
    },
    "flux1-kontext": {
        "zh": "FLUX.1 Kontext 上下文编辑，支持多模态条件生成与编辑。",
        "en": "FLUX.1 Kontext for multimodal conditioned generation and editing.",
    },
    "qwen-image-edit": {
        "zh": "通义千问 Qwen Image Edit 2511 指令编辑。人物与多人一致性更好，最多三张参考图。",
        "en": "Qwen Image Edit 2511 instruction edit. Stronger character and multi-person consistency, with up to three reference images.",
    },
    "firered-image-edit-1.1": {
        "zh": "FireRed Image Edit 1.1 指令编辑（仅图生图编辑，不支持文生图），强身份一致性，支持多参考图。",
        "en": "FireRed Image Edit 1.1 — instruction-based image edit only (no text-to-image); strong identity consistency and multi-reference input.",
    },
    "ace-step-xl-sft": {
        "zh": "ACE-Step-1.5 XL SFT，4B DiT 解码器，高质量音乐生成。",
        "en": "ACE-Step-1.5 XL SFT; 4B DiT decoder for high-quality music generation.",
    },
    "realism-lora": {
        "zh": "适用于 FLUX.1-dev；真实感增强 LoRA，提升人物细节。",
        "en": "For FLUX.1-dev; realism enhancement LoRA for finer portrait detail.",
    },
    "awportraitcn-hf": {
        "zh": "面向中国人像审美训练的 FLUX.1-dev LoRA；棚拍、时尚与室内外写真，肤质细腻自然。建议权重 0.9–1.0。",
        "en": "FLUX.1-dev portrait LoRA tuned for Chinese aesthetics; studio, fashion, and outdoor portraits. Weight 0.9–1.0 recommended.",
    },
    "awportraitcn2-hf": {
        "zh": "AWPortrait CN 升级版；更广的东方美学与全年龄段面谱，场景与民族服饰响应更好。适用于 FLUX.1-dev，建议权重 0.9–1.0。",
        "en": "Upgraded AWPortrait CN with broader Eastern aesthetics and age coverage; better scenes and ethnic costumes. For FLUX.1-dev; weight 0.9–1.0.",
    },
    "awportrait-z": {
        "zh": "基于 Z-Image 的东方人像美颜 LoRA；降噪、改善光比与面孔多样性。适用于 Z-Image-Turbo（与 Z-Image Base 适配器互通）；建议权重 0.8。",
        "en": "Z-Image portrait-beauty LoRA: reduced grain, improved lighting, and face diversity. For Z-Image-Turbo (compatible with Z-Image Base adapters). Weight 0.8 recommended.",
    },
    "awportrait-fl-hf": {
        "zh": "时尚摄影向 FLUX.1-dev LoRA；构图与肤质更细腻写实，适合人像与服装大片。建议权重 0.9。",
        "en": "Fashion-photography FLUX.1-dev LoRA with finer composition and realistic skin. Weight 0.9 recommended.",
    },
    "flux1-dev-lora-add-details-hf": {
        "zh": "FLUX.1-dev 真实质感增强 LoRA；自然肤质与细节，无固定触发词，建议权重 1.0（可为负向微调）。",
        "en": "FLUX.1-dev realism/detail LoRA for natural skin; no trigger words. Weight 1.0 recommended (can go negative).",
    },
    "flux-asian-portrait-hf": {
        "zh": "FLUX.1-dev 亚洲女性写实肖像 LoRA；触发词 1girl，建议权重 0.6–1.0。",
        "en": "FLUX.1-dev realistic Asian female portrait LoRA. Trigger: 1girl. Weight 0.6–1.0.",
    },
    "flux2-klein-asian-real-hf": {
        "zh": "FLUX.2 Klein 9B 亚洲青年写实人像 LoRA；动漫/插画转真人或文生图均可。触发词如 realistic style of a young Asian girl，建议权重 0.6–1.0。",
        "en": "FLUX.2 Klein 9B Asian photoreal portrait LoRA for T2I or anime-to-real. Trigger e.g. realistic style of a young Asian girl. Weight 0.6–1.0.",
    },
    "starface-z-image-turbo-ms": {
        "zh": "StarFace 中国高颜值气质人像 LoRA，绑定 Z-Image-Turbo 快速出图；建议权重 0.6–0.8。",
        "en": "StarFace Chinese portrait LoRA for Z-Image-Turbo fast generation. Weight 0.6–0.8 recommended.",
    },
    "flux1-dev-lora-antiblur": {
        "zh": "LiblibAI 去糊/清晰增强 LoRA，适用于 FLUX.1-dev。",
        "en": "LiblibAI anti-blur/sharpness LoRA for FLUX.1-dev.",
    },
    "flux1-dev-lora-ghibli": {
        "zh": "InstantX 吉卜力动画风格 LoRA，适用于 FLUX.1-dev。",
        "en": "InstantX Ghibli-style LoRA for FLUX.1-dev.",
    },
    "flux1-dev-lora-makoto-shinkai": {
        "zh": "InstantX 新海诚画风 LoRA，适用于 FLUX.1-dev。",
        "en": "InstantX Makoto Shinkai–style LoRA for FLUX.1-dev.",
    },
    "flux1-dev-lora-canny": {
        "zh": "官方 Canny 控制 LoRA，需配合边缘/Canny 条件流程。",
        "en": "Official Canny control LoRA; use with edge/Canny conditioning.",
    },
    "flux1-canny-dev-lora": {
        "zh": "官方 Canny 控制 LoRA，需配合边缘/Canny 条件流程。",
        "en": "Official Canny control LoRA; use with edge/Canny conditioning.",
    },
    "flux1-dev-lora-depth": {
        "zh": "官方 Depth 控制 LoRA，需配合深度图条件。",
        "en": "Official Depth control LoRA; requires depth conditioning.",
    },
    "flux2-klein-ac-style-lora": {
        "zh": "Klein AC 风格 LoRA；默认绑定 9B，若作者标明仅 4B 请改 base 或勿混用。",
        "en": "AC-style LoRA for FLUX.2 Klein; defaults to 9B—verify upstream if 4B-only.",
    },
    "flux2-klein-4b-spritesheet-lora": {
        "zh": "Klein 4B 精灵表/序列帧向 LoRA。",
        "en": "Spritesheet-oriented LoRA for Klein 4B.",
    },
    "flux2-klein-4b-zoom-lora": {
        "zh": "Klein 4B 变焦效果向 LoRA。",
        "en": "Zoom-effect LoRA for Klein 4B.",
    },
    "flux2-klein-4b-outpaint-lora": {
        "zh": "Klein 4B 画面外扩向 LoRA。",
        "en": "Outpainting LoRA for Klein 4B.",
    },
    "flux2-klein-4b-object-remove-lora": {
        "zh": "Klein 4B 物体移除向 LoRA。",
        "en": "Object-removal LoRA for Klein 4B.",
    },
    "starface-z-image-ms": {
        "zh": "明星脸/中国高颜值气质人像风格 LoRA，基于通义 Z-Image 训练。",
        "en": "Celebrity-style Chinese portrait LoRA trained on Tongyi Z-Image.",
    },
    "starface-qwen-image-2512-ms": {
        "zh": "明星脸/中国高颜值气质人像风格 LoRA，基于通义 Qwen Image 训练；引擎内合并仅 MLX。",
        "en": "Celebrity-style Chinese portrait LoRA for Qwen Image; in-engine merge is MLX-only.",
    },
    "qwen-image-2512-lightning-lora": {
        "zh": "Qwen Image 2512 LightX2V 4 步蒸馏 LoRA（V1.0 BF16）。需配合 qwen-image FP16 完整版；建议 4 步、CFG 1.0。",
        "en": "Qwen Image 2512 LightX2V 4-step distill LoRA (V1.0 BF16). Requires qwen-image FP16 full weights; use 4 steps and CFG 1.0.",
    },
    "qwen-image-edit-2511-lightning-lora": {
        "zh": "Qwen Image Edit 2511 LightX2V 蒸馏 LoRA（4 步或 8 步 BF16）。需配合 Qwen-Image-Edit-2511 FP16；建议 CFG 1.0。",
        "en": "Qwen Image Edit 2511 LightX2V distill LoRA (4-step or 8-step BF16). Requires Qwen-Image-Edit-2511 FP16; use CFG 1.0.",
    },
    "bernini-r-14b": {
        "zh": "ByteDance Bernini-R 14B：文生视频、参考生视频、视频编辑与参考+视频编辑（SA-3D RoPE）；各版本均为完整自包含 bundle（DiT + UMT5 + VAE）。",
        "en": "ByteDance Bernini-R 14B: T2V, R2V, V2V, RV2V (SA-3D RoPE); every tier is a self-contained bundle (DiT + UMT5 + VAE).",
    },
    "bernini-r-1.3b": {
        "zh": "ByteDance Bernini-R 1.3B：文生视频、参考生视频、视频编辑（SA-3D RoPE）；各版本均为完整自包含 bundle（DiT + UMT5 + VAE）。",
        "en": "ByteDance Bernini-R 1.3B: T2V, R2V, V2V (SA-3D RoPE); every tier is a self-contained bundle (DiT + UMT5 + VAE).",
    },
    "hunyuan-video-1.5-t2v-distill": {
        "zh": "HunyuanVideo 1.5 文生视频 LightX2V 4 步真蒸馏（单文件 DiT，480p）；完整自包含 bundle（DiT + VAE + 文本编码器）。",
        "en": "HunyuanVideo 1.5 T2V LightX2V 4-step distill (480p); self-contained bundle (DiT + VAE + text encoders).",
    },
    "hunyuan-video-1.5-1080p-sr": {
        "zh": "HunyuanVideo 1.5 超分修复，1080p 超分变体。",
        "en": "HunyuanVideo 1.5 super-resolution; 1080p SR variant.",
    },
    "ltx-2.3-distilled": {
        "zh": "Lightricks LTX-2.3 蒸馏版，含音视频 DiT，4 步快速生成。",
        "en": "Lightricks LTX-2.3 distilled A/V DiT; fast 4-step generation.",
    },
    "ltx-2.3-dev": {
        "zh": "Lightricks LTX-2.3 开发版，含音视频 DiT，支持 CFG，质量更高。",
        "en": "Lightricks LTX-2.3 dev A/V DiT; supports CFG for higher quality.",
    },
}

NAME_OVERRIDES: dict[str, dict[str, str]] = {
    "awportraitcn-hf": {
        "zh": "AWPortrait CN",
        "en": "AWPortrait CN",
    },
    "awportraitcn2-hf": {
        "zh": "AWPortrait CN2",
        "en": "AWPortrait CN2",
    },
    "awportrait-z": {
        "zh": "AWPortrait Z",
        "en": "AWPortrait Z",
    },
    "awportrait-fl-hf": {
        "zh": "AWPortrait FL",
        "en": "AWPortrait FL",
    },
    "flux1-dev-lora-add-details-hf": {
        "zh": "FLUX.1-dev Add Details",
        "en": "FLUX.1-dev Add Details",
    },
    "flux-asian-portrait-hf": {
        "zh": "Asian Portrait (FLUX.1)",
        "en": "Asian Portrait (FLUX.1)",
    },
    "flux2-klein-asian-real-hf": {
        "zh": "Klein 9B Asian Real",
        "en": "Klein 9B Asian Real",
    },
    "starface-z-image-turbo-ms": {
        "zh": "StarFace (Z-Image Turbo)",
        "en": "StarFace (Z-Image Turbo)",
    },
    "starface-z-image-ms": {
        "zh": "StarFace (Z-Image)",
        "en": "StarFace (Z-Image)",
    },
    "starface-qwen-image-2512-ms": {
        "zh": "StarFace (Qwen Image)",
        "en": "StarFace (Qwen Image)",
    },
    "qwen-image-2512-lightning-lora": {
        "zh": "Qwen Image 2512 Lightning",
        "en": "Qwen Image 2512 Lightning",
    },
    "qwen-image-edit-2511-lightning-lora": {
        "zh": "Qwen Image Edit 2511 Lightning",
        "en": "Qwen Image Edit 2511 Lightning",
    },
}


def _catalog_entry(model_entry: dict[str, Any]) -> dict[str, Any]:
    catalog = model_entry.get("catalog")
    return catalog if isinstance(catalog, dict) else model_entry


def _distribution_entry(model_entry: dict[str, Any]) -> dict[str, Any]:
    dist = model_entry.get("distribution")
    return dist if isinstance(dist, dict) else model_entry


def _skip_version_label(model_entry: dict[str, Any], version_key: str) -> bool:
    catalog = _catalog_entry(model_entry)
    catalog_type = str(catalog.get("type") or "").lower()
    catalog_category = str(catalog.get("category") or "").lower()
    if catalog_type in ("lora", "controlnet") or catalog_category in ("loras", "controlnets"):
        return True
    vk = str(version_key or "").lower()
    return bool(re.match(r"^v[\w-]+$", vk) or re.search(r"\d+steps$", vk))


def resolve_version_name(
    model_entry: dict[str, Any],
    version_key: str,
    version_entry: dict[str, Any],
) -> dict[str, str]:
    if _skip_version_label(model_entry, version_key):
        old_name = version_entry.get("name")
        if isinstance(old_name, dict):
            cleaned_zh = clean_user_text(str(old_name.get("zh") or ""), lang="zh")
            cleaned_en = clean_user_text(str(old_name.get("en") or ""), lang="en")
            if cleaned_zh or cleaned_en:
                return {
                    "zh": cleaned_zh or cleaned_en or version_key,
                    "en": cleaned_en or cleaned_zh or version_key,
                }
        old_plain = str(old_name or version_key)
        cleaned = clean_user_text(old_plain, lang="zh")
        return {"zh": cleaned or version_key, "en": cleaned or version_key}

    label = expected_version_display_name(version_key, version_entry)
    if label is not None:
        return label

    old_name = version_entry.get("name")
    if isinstance(old_name, dict):
        old_plain = str(old_name.get("zh") or old_name.get("en") or "")
    else:
        old_plain = str(old_name or "")
    cleaned = clean_user_text(old_plain, lang="zh")
    if cleaned:
        return {
            "zh": cleaned,
            "en": clean_user_text(
                str(old_name.get("en") if isinstance(old_name, dict) else old_plain),
                lang="en",
            )
            or cleaned,
        }
    return {"zh": version_key, "en": version_key}


def clean_user_text(text: str, *, lang: str) -> str:
    text = str(text or "").strip()
    if not text:
        return text

    text = re.sub(
        r"[（(]\s*(?:魔搭|ModelScope|Hugging Face|HF)\s*(?:[）)]|$)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"[（(][^）)]*(?:魔搭|ModelScope|Hugging Face|\bHF\b)[^）)]*[）)]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?:魔搭|ModelScope|Hugging Face|\bHF\b)\s+[A-Za-z0-9._-]+/[A-Za-z0-9._/-]+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?:与魔搭|与 ModelScope)\s+[^；。,.]+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[，,]\s*[；;]", "；", text)
    text = re.sub(r"^[；;]\s*", "", text)
    text = re.sub(r"[；;]{2,}", "；", text)
    text = re.sub(r"[，,]{2,}", "，", text)
    text = text.strip(" ；;，,.")

    if lang == "en":
        text = re.sub(r"\bModelScope\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bHugging Face\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip(" ,.;")

    return text.strip()


def _ensure_sentence(text: str, *, lang: str) -> str:
    text = text.strip()
    if not text:
        return text
    if lang == "zh":
        if text[-1] in "。；":
            return text
        return f"{text}。"
    if text[-1] in ".;":
        return text
    return f"{text}."


def normalize_name(model_id: str, model_entry: dict[str, Any]) -> dict[str, str]:
    if model_id in NAME_OVERRIDES:
        return dict(NAME_OVERRIDES[model_id])

    name = model_entry.get("name") or {}
    if not isinstance(name, dict):
        raw = str(name or model_id)
        return {"zh": raw, "en": raw}

    zh = clean_user_text(str(name.get("zh") or model_id), lang="zh")
    en = clean_user_text(str(name.get("en") or zh), lang="en")
    return {"zh": zh or model_id, "en": en or zh or model_id}


def normalize_description(model_id: str, model_entry: dict[str, Any]) -> dict[str, str]:
    if model_id in DESCRIPTION_OVERRIDES:
        item = DESCRIPTION_OVERRIDES[model_id]
        return {
            "zh": _ensure_sentence(item["zh"], lang="zh"),
            "en": _ensure_sentence(item["en"], lang="en"),
        }

    desc = model_entry.get("description") or {}
    if not isinstance(desc, dict):
        raw = clean_user_text(str(desc or ""), lang="zh")
        return {"zh": _ensure_sentence(raw, lang="zh"), "en": _ensure_sentence(raw, lang="en")}

    zh = clean_user_text(str(desc.get("zh") or ""), lang="zh")
    en = clean_user_text(str(desc.get("en") or ""), lang="en")
    if CJK_RE.search(en):
        en = clean_user_text(str(desc.get("en") or ""), lang="en")
        if CJK_RE.search(en):
            en = zh
    if not en and zh:
        en = zh
    if not zh and en:
        zh = en

    return {
        "zh": _ensure_sentence(zh, lang="zh") if zh else "",
        "en": _ensure_sentence(en, lang="en") if en else "",
    }


def normalize_registry(data: dict[str, Any]) -> tuple[dict[str, Any], int]:
    out = json.loads(json.dumps(data))
    changes = 0
    models = out.get("models") or {}
    for model_id, model_entry in models.items():
        if not isinstance(model_entry, dict):
            continue

        catalog = _catalog_entry(model_entry)
        dist = _distribution_entry(model_entry)

        new_name = normalize_name(model_id, catalog)
        if catalog.get("name") != new_name:
            catalog["name"] = new_name
            changes += 1

        new_desc = normalize_description(model_id, catalog)
        if catalog.get("description") != new_desc:
            catalog["description"] = new_desc
            changes += 1

        versions = dist.get("versions") or {}
        if not isinstance(versions, dict):
            continue
        for version_key, version_entry in versions.items():
            if not isinstance(version_entry, dict):
                continue
            new_vname = resolve_version_name(model_entry, version_key, version_entry)
            if version_entry.get("name") != new_vname:
                version_entry["name"] = new_vname
                changes += 1

    return out, changes


def main() -> int:
    path = REGISTRY_PATH
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    normalized, changes = normalize_registry(data)
    path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Normalized {path} ({changes} field updates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
