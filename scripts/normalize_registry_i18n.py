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
    'qwen3-4b-thinking-2507': {
        "zh": '本地小助手，会先思考再回答，适合短提示扩写和简单推理',
        "en": 'Small local assistant that thinks before answering, for short prompt expansion and light reasoning',
    },
    'qwen3.5-4b': {
        "zh": '本地小助手，直接回答，不做延长思考，比思考版更快',
        "en": 'Small local assistant that answers directly, without extended thinking, and faster than the thinking build',
    },
    'qwen3.6-27b': {
        "zh": '本地大助手，适合长剧本和结构化拆解，比 4B 更稳也更慢',
        "en": 'Large local assistant for long scripts and structured breakdowns, steadier and slower than the 4B models',
    },
    'qwen3-vl-4b-instruct': {
        "zh": '看图助手，根据画面写描述。文本助手不能看图',
        "en": 'Vision assistant that writes a description from a picture. Text assistants cannot see images',
    },
    'flux1-schnell': {
        "zh": '4 步快速文生图。参考图按强度重绘整张图，质量低于 dev',
        "en": 'Fast text-to-image in 4 steps. A reference redraws the whole frame by strength, at lower quality than dev',
    },
    'flux1-dev': {
        "zh": '28 步高质量文生图，也是 LoRA 训练底模。参考图按强度重绘整张图',
        "en": 'High-quality text-to-image at 28 steps, and the base for LoRA training. A reference redraws the whole frame by strength',
    },
    'flux1-krea-dev': {
        "zh": '与 dev 同档，更偏照片真实感。参考图按强度重绘整张图',
        "en": 'Same tier as dev, biased toward photographic realism. A reference redraws the whole frame by strength',
    },
    'flux2-klein-4b': {
        "zh": '小模型 4 步快速文生图，速度优先。参考图按强度重绘整张图',
        "en": 'Small text-to-image model in 4 steps, speed first. A reference redraws the whole frame by strength',
    },
    'flux2-klein-9b': {
        "zh": '比 4B 更大，8 步，真实感更好。参考图按强度重绘整张图',
        "en": 'Larger than 4B, 8 steps, stronger realism. A reference redraws the whole frame by strength',
    },
    'flux2-klein-base-4b': {
        "zh": '4B 未蒸馏底模，约 50 步，可训练。参考图按强度重绘整张图',
        "en": 'Undistilled 4B base, about 50 steps, trainable. A reference redraws the whole frame by strength',
    },
    'flux2-klein-base-9b': {
        "zh": '9B 未蒸馏底模，约 50 步，质量高于 4B 基础版。参考图按强度重绘整张图',
        "en": 'Undistilled 9B base, about 50 steps, higher quality than the 4B base. A reference redraws the whole frame by strength',
    },
    'z-image': {
        "zh": '文生图，支持 CFG 和负向提示。参考图按强度重绘整张图。人像 LoRA 在这里训练，再到 Turbo 上加速',
        "en": 'Text-to-image with CFG and a negative prompt. A reference redraws the whole frame by strength. Train portrait LoRAs here, then run them on Turbo',
    },
    'z-image-turbo': {
        "zh": '默认约 9 步快速文生图。参考图按强度重绘整张图。挂上 Base 训练的 LoRA 时会自动补蒸馏补丁并切到 8 步',
        "en": 'Fast text-to-image, about 9 steps by default. A reference redraws the whole frame by strength. A LoRA trained on Base auto-loads the distill patch and switches to 8 steps',
    },
    'fibo': {
        "zh": '用 JSON 结构化提示做文生图。参考图按强度重绘整张图。比 Lite 更完整',
        "en": 'Text-to-image from JSON-structured prompts. A reference redraws the whole frame by strength. Fuller than Lite',
    },
    'fibo-lite': {
        "zh": '更轻的 FIBO，同样用 JSON 结构化提示。参考图按强度重绘整张图',
        "en": 'Lighter FIBO with the same JSON-structured prompts. A reference redraws the whole frame by strength',
    },
    'fibo-edit': {
        "zh": '仅指令改图。单张参考图拼进画面条件，不靠去噪强度混合',
        "en": 'Instruction edit only. One reference image is concatenated into the condition, not blended by denoise strength',
    },
    'fibo-edit-rmbg': {
        "zh": '与 FIBO-Edit 相同的指令改图，并去掉背景，输出透明图',
        "en": 'Same instruction edit as FIBO-Edit, then removes the background for a transparent image',
    },
    'qwen-image': {
        "zh": '文生图，人像和自然纹理较强。参考图按强度重绘整张图，不单独锁脸',
        "en": 'Text-to-image with stronger portraits and textures. A reference redraws the whole frame by strength and does not lock a face on its own',
    },
    'qwen-image-edit': {
        "zh": '仅指令改图，最多 3 张参考图，人物和多人更稳',
        "en": 'Instruction edit only, up to 3 reference images, with stronger character and multi-person consistency',
    },
    'firered-image-edit-1.1': {
        "zh": '仅指令改图，最多 3 张参考图，身份一致性强于 Qwen Image Edit 2511',
        "en": 'Instruction edit only, up to 3 reference images, with stronger identity consistency than Qwen Image Edit 2511',
    },
    'hidream-o1-image-dev': {
        "zh": '28 步蒸馏。文生图；BF16 可指令改图，最多 3 张参考图。8-bit 只能文生图',
        "en": '28-step distill. Text-to-image; BF16 can instruction-edit with up to 3 reference images. The 8-bit build is text-to-image only',
    },
    'hidream-o1-image-full': {
        "zh": '50 步完整版，细节高于 Dev。文生图，并可指令改图，最多 3 张参考图',
        "en": '50-step full model, more detail than Dev. Text-to-image, and instruction edit with up to 3 reference images',
    },
    'step1x-edit': {
        "zh": '文生图，以及单张参考图的指令改图',
        "en": 'Text-to-image, plus instruction edit from one reference image',
    },
    'boogu-image-turbo': {
        "zh": '仅文生图，8 步快速写实，中英文文字较好。量化版的人像虚化背景可能出现细网格',
        "en": 'Text-to-image only, 8-step fast photorealism with stronger bilingual text. Quantized builds may show a fine grid in portrait bokeh',
    },
    'boogu-image-edit': {
        "zh": '仅指令改图，单张参考图，适合摄影调整和改字',
        "en": 'Instruction edit only, one reference image, for photo adjustments and text changes',
    },
    'ernie-image-turbo': {
        "zh": '仅文生图，8 步，复杂指令和画面文字较好',
        "en": 'Text-to-image only, 8 steps, stronger at complex instructions and rendered text',
    },
    'cogview4-6b': {
        "zh": '中文提示和画面文字较好。文生图，参考图按强度重绘整张图',
        "en": 'Stronger Chinese prompts and rendered text. Text-to-image; a reference redraws the whole frame by strength',
    },
    'seedvr2-3b': {
        "zh": '图像放大，3B，比 7B 更快更轻，细节也少一些',
        "en": 'Image upscale, 3B, faster and lighter than 7B, with less detail',
    },
    'seedvr2-7b': {
        "zh": '图像放大，7B，细节高于 3B，也慢于 Real-ESRGAN',
        "en": 'Image upscale, 7B, more detail than 3B, and slower than Real-ESRGAN',
    },
    'real-esrgan-x4plus': {
        "zh": '4 倍像素放大，很快、很省显存，细节低于 SeedVR2',
        "en": '4x pixel upscale. Fast and light on memory, with less detail than SeedVR2',
    },
    'seedvr2-video-7b': {
        "zh": '视频放大与修复，7B，质量高于 3B',
        "en": 'Video upscale and restoration, 7B, higher quality than 3B',
    },
    'seedvr2-video-3b': {
        "zh": '视频放大与修复，3B，比 7B 更轻',
        "en": 'Video upscale and restoration, 3B, lighter than 7B',
    },
    'depth-pro': {
        "zh": '从照片估计深度图，供深度控制使用',
        "en": 'Estimates a depth map from a photo for depth control',
    },
    'flux-canny-controlnet': {
        "zh": '用边缘图约束 FLUX.1 的轮廓',
        "en": 'Guides FLUX.1 contours with an edge map',
    },
    'flux-depth-controlnet': {
        "zh": '用深度图约束 FLUX.1 的空间关系',
        "en": 'Guides FLUX.1 spatial layout with a depth map',
    },
    'flux-fill-controlnet': {
        "zh": 'FLUX.1 局部重绘和外扩',
        "en": 'FLUX.1 inpainting and outpainting',
    },
    'flux-redux': {
        "zh": '用参考图给 FLUX.1 做变体和风格迁移，不锁脸',
        "en": 'FLUX.1 variations and style transfer from a reference, without face lock',
    },
    'z-image-turbo-fun-controlnet-union': {
        "zh": '一张控制图提供边缘、深度、姿态等结构，需先做好预处理图',
        "en": 'One control image supplies edges, depth, pose, and similar structure. Provide a preprocessed map',
    },
    'flux1-kontext': {
        "zh": 'FLUX.1 上下文版。参考图按强度重绘整张图',
        "en": 'FLUX.1 context variant. A reference image redraws the whole frame by strength',
    },
    'flux1-dev-lora-antiblur': {
        "zh": '挂在 FLUX.1 dev。减轻模糊、提高清晰度',
        "en": 'FLUX.1 dev LoRA that reduces blur and raises sharpness',
    },
    'flux1-dev-lora-ghibli': {
        "zh": '挂在 FLUX.1 dev。吉卜力式动画画风',
        "en": 'FLUX.1 dev LoRA for a Ghibli-like animated look',
    },
    'flux1-dev-lora-makoto-shinkai': {
        "zh": '挂在 FLUX.1 dev。新海诚式光影和天空',
        "en": 'FLUX.1 dev LoRA for a Makoto Shinkai look, with light and skies',
    },
    'flux1-canny-dev-lora': {
        "zh": '配合 Canny 边缘控制的官方 LoRA，单独挂上不会产生边缘约束',
        "en": 'Official LoRA paired with Canny edge control. On its own it does not apply edge guidance',
    },
    'flux1-depth-dev-lora': {
        "zh": '配合深度图控制的官方 LoRA',
        "en": 'Official LoRA paired with depth-map control',
    },
    'flux-super-realism-lora': {
        "zh": '挂在 FLUX.1 dev。加强写实人像和皮肤质感',
        "en": 'FLUX.1 dev LoRA that pushes photoreal portraits and skin texture',
    },
    'flux1-schnell-pvc-style-lora': {
        "zh": '挂在 FLUX.1 schnell。PVC 手办质感',
        "en": 'PVC figure look for FLUX.1 schnell',
    },
    'flux2-klein-ac-style-lora': {
        "zh": '挂在 FLUX.2 Klein 9B。AC 插画风格。权重若只适用于 4B，不要挂到 9B',
        "en": 'AC illustration style for FLUX.2 Klein 9B. If the weights are 4B-only, do not load them on 9B',
    },
    'flux2-klein-4b-spritesheet-lora': {
        "zh": '挂在 Klein 4B。生成精灵表和序列帧',
        "en": 'Spritesheets and animation frames for Klein 4B',
    },
    'flux2-klein-4b-zoom-lora': {
        "zh": '挂在 Klein 4B。变焦镜头效果',
        "en": 'Zoom-lens effect for Klein 4B',
    },
    'flux2-klein-4b-outpaint-lora': {
        "zh": '挂在 Klein 4B。把画面向外扩展',
        "en": 'Extends the frame outward for Klein 4B',
    },
    'flux2-klein-4b-object-remove-lora': {
        "zh": '挂在 Klein 4B。去掉画面中的物体',
        "en": 'Removes objects from the frame for Klein 4B',
    },
    'wan2.2-t2v-lightning-lora': {
        "zh": '挂在 Wan 2.2 文生视频 14B。固定 4 步，不用 CFG。含高噪和低噪两份，不能拿去挂图生视频',
        "en": '4-step distill for Wan 2.2 text-to-video 14B, no CFG. High-noise and low-noise pair, not for image-to-video',
    },
    'wan2.2-i2v-lightning-lora': {
        "zh": '挂在 Wan 2.2 图生视频 14B。固定 4 步，不用 CFG。与文生视频那份不能互换',
        "en": '4-step distill for Wan 2.2 image-to-video 14B, no CFG. Not interchangeable with the text-to-video LoRA',
    },
    'starface-z-image-ms': {
        "zh": '挂在 Z-Image。中国高颜值气质人像',
        "en": 'Chinese portrait look for Z-Image',
    },
    'z-image-turbo-distillpatch-lora': {
        "zh": '挂在 Z-Image Turbo。Base 上训练的 LoRA 在 Turbo 推理时自动加载它，恢复 8 步加速',
        "en": 'Auto-loaded on Z-Image Turbo when a LoRA was trained on Base, restoring 8-step acceleration',
    },
    'starface-qwen-image-2512-ms': {
        "zh": '挂在 Qwen Image。中国高颜值气质人像',
        "en": 'Chinese portrait look for Qwen Image',
    },
    'qwen-image-2512-lightning-lora': {
        "zh": '挂在 Qwen Image 完整精度底模。把采样压到 4 步，CFG 设为 1',
        "en": '4-step distill for the full-precision Qwen Image base. Set CFG to 1',
    },
    'qwen-image-edit-2511-lightning-lora': {
        "zh": '挂在 Qwen Image Edit 2511 完整精度底模。4 步或 8 步，CFG 设为 1',
        "en": '4-step or 8-step distill for full-precision Qwen Image Edit 2511. Set CFG to 1',
    },
    'ltx-2.3-distilled': {
        "zh": '文生视频和图生视频，自带声音，4 步。质量低于开发版',
        "en": 'Text-to-video and image-to-video with audio, in 4 steps. Lower quality than the dev build',
    },
    'ltx-2.3-dev': {
        "zh": '同一代音视频模型，支持 CFG，步数更多、质量更高',
        "en": 'Same-generation audio-video model with CFG, more steps, and higher quality',
    },
    'ltx-2.5-distilled': {
        "zh": '更大的音视频模型，两阶段采样，可到 4K。比 2.3 更重',
        "en": 'Larger audio-video model, two-stage sampling, up to 4K. Heavier than 2.3',
    },
    'hunyuan-video-1.5-t2v-distill': {
        "zh": '仅文生视频，4 步，480p',
        "en": 'Text-to-video only, 4 steps, 480p',
    },
    'hunyuan-video-1.5-1080p-sr': {
        "zh": '把已有视频放大到 1080p',
        "en": 'Upscales an existing video to 1080p',
    },
    'hunyuan-video-1.5-i2v-720p': {
        "zh": '图生视频，720p，完整步数',
        "en": 'Image-to-video at 720p with the full step count',
    },
    'hunyuan-video-1.5-i2v-720p-distill': {
        "zh": '图生视频 720p 的少步版，CFG 为 1，比完整版更快',
        "en": 'Faster 720p image-to-video with CFG at 1',
    },
    'wan-2.2-t2v-14b': {
        "zh": '文生视频 14B，质量优先、步数多。要 4 步请挂 Lightning LoRA',
        "en": '14B text-to-video, quality first and many steps. Add the Lightning LoRA for 4 steps',
    },
    'wan-2.2-i2v-14b': {
        "zh": '图生视频 14B 完整版。与文生视频权重不能互换',
        "en": 'Full 14B image-to-video. Not interchangeable with the text-to-video weights',
    },
    'wan-2.2-i2v-14b-distill': {
        "zh": '图生视频 14B 的 4 步版，720p，不用 CFG',
        "en": '4-step 720p image-to-video 14B, no CFG',
    },
    'bernini-r-14b': {
        "zh": '文生视频、参考图生视频、视频编辑，以及参考图加视频一起编辑。比 1.3B 更重、更稳',
        "en": 'Text-to-video, reference-to-video, video edit, and reference plus video together. Heavier and steadier than 1.3B',
    },
    'bernini-r-1.3b': {
        "zh": '文生视频、参考图生视频和视频编辑。比 14B 更轻，没有参考图加视频',
        "en": 'Text-to-video, reference-to-video, and video edit. Lighter than 14B, without reference plus video',
    },
    'longcat-video': {
        "zh": '文生视频和图生视频，默认 8 步快速路径',
        "en": 'Text-to-video and image-to-video, default 8-step fast path',
    },
    'minimax-h3-turbo-lora': {
        "zh": '挂在 MiniMax H3。把联合音视频压到 4 到 8 步，建议 6 到 8 步',
        "en": 'Distill LoRA for MiniMax H3. Joint audio-video in 4 to 8 steps; 6 to 8 is the useful range',
    },
    'minimax-h3-veda': {
        "zh": '挂在 MiniMax H3。视频注意力只算一小部分，文本和音频仍完整。可与 Turbo LoRA 叠加，用于 8 步预览',
        "en": 'Sparse attention for MiniMax H3: video blocks compute only a fraction, while text and audio stay dense. Stacks with the Turbo LoRA for an 8-step preview',
    },
    'minimax-h3-fl2va': {
        "zh": '文生或首尾帧生成 768p 音视频，24 帧，约 4 到 15 秒。无负向提示',
        "en": 'Text or first-and-last-frame audio-video at 768p, 24 fps, about 4 to 15 seconds. No negative prompt',
    },
    'longcat-avatar-1.5': {
        "zh": '数字人。用人像和配音驱动口型，默认 8 步。与 LongCat 文生视频不是同一条任务',
        "en": 'Digital human. A portrait plus speech drives the lips, 8 steps by default. Separate from LongCat text-to-video',
    },
    'wan-2.2-ti2v-5b': {
        "zh": '文本或图片引导的 5B 视频，默认 480p，比 14B 轻',
        "en": '5B video from text or an image, 480p by default, lighter than 14B',
    },
    'ace-step-xl-sft': {
        "zh": '歌曲生成和翻唱，质量优先，步数更多',
        "en": 'Song generation and cover, quality first, more steps',
    },
    'ace-step-xl-turbo': {
        "zh": '同一代的 8 步快速版，质量低于 SFT。同样支持翻唱',
        "en": 'Same generation, 8-step fast path, lower quality than SFT. Cover is included',
    },
    'ace-step-lora-super-eurobeats': {
        "zh": '挂在 ACE-Step。Super Eurobeat 节奏。提示里写上 Eurobeats',
        "en": 'Super Eurobeat rhythm for ACE-Step. Put Eurobeats in the prompt',
    },
    'ace-step-lora-accompaniment-ms': {
        "zh": '挂在 ACE-Step SFT。加强伴奏层次',
        "en": 'Richer accompaniment for ACE-Step SFT',
    },
    'ace-step-lora-chinese-new-year': {
        "zh": '挂在 ACE-Step。春节和喜庆节日气氛',
        "en": 'Chinese New Year and festive mood for ACE-Step',
    },
    'diffrhythm-v2': {
        "zh": '只生成歌曲，最长约 210 秒，48 kHz。没有翻唱',
        "en": 'Song generation only, up to about 210 seconds at 48 kHz. No cover',
    },
    'realism-lora': {
        "zh": '挂在 FLUX.1 dev。真实感人像细节。比 Add Details 更偏整体真实感',
        "en": 'Photoreal portrait detail for FLUX.1 dev. Broader realism than Add Details',
    },
    'awportraitcn-hf': {
        "zh": '挂在 FLUX.1 dev。中国人像审美，棚拍和室内外写真。建议权重 0.9 到 1.0',
        "en": 'Chinese portrait taste for FLUX.1 dev: studio and indoor-outdoor photos. Weight 0.9 to 1.0',
    },
    'awportraitcn2-hf': {
        "zh": 'AWPortrait CN 的升级，年龄段和民族服饰更广。同样挂在 FLUX.1 dev，建议权重 0.9 到 1.0',
        "en": 'Newer than AWPortrait CN, with wider ages and ethnic costume. Same FLUX.1 dev base, weight 0.9 to 1.0',
    },
    'awportrait-z': {
        "zh": '挂在 Z-Image Turbo。东方人像，颗粒更少、光比更稳。建议权重 0.8',
        "en": 'Eastern portraits for Z-Image Turbo: less grain and steadier lighting. Weight 0.8',
    },
    'awportrait-fl-hf': {
        "zh": '挂在 FLUX.1 dev。时尚大片的构图和肤质。建议权重 0.9',
        "en": 'Fashion-editorial composition and skin for FLUX.1 dev. Weight 0.9',
    },
    'flux1-dev-lora-add-details-hf': {
        "zh": '挂在 FLUX.1 dev。自然肤质和细节，无固定触发词。建议权重 1.0，也可以用负权重减弱',
        "en": 'Natural skin and detail for FLUX.1 dev, no trigger word. Weight 1.0, or a negative weight to soften it',
    },
    'flux-asian-portrait-hf': {
        "zh": '挂在 FLUX.1 dev。亚洲女性写实肖像。触发词 1girl。建议权重 0.6 到 1.0',
        "en": 'Realistic Asian female portraits for FLUX.1 dev. Trigger word 1girl. Weight 0.6 to 1.0',
    },
    'flux2-klein-asian-real-hf': {
        "zh": '挂在 FLUX.2 Klein 9B。亚洲青年写实人像，插画转真人或直接文生图。建议权重 0.6 到 1.0',
        "en": 'Photoreal young Asian portraits for FLUX.2 Klein 9B, from an illustration or from text. Weight 0.6 to 1.0',
    },
    'starface-z-image-turbo-ms': {
        "zh": '挂在 Z-Image Turbo。中国高颜值气质人像，走快速出图。建议权重 0.6 到 0.8',
        "en": 'Chinese portrait look for fast Z-Image Turbo. Weight 0.6 to 0.8',
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
