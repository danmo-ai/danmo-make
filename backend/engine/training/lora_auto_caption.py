"""VLM auto-caption for LoRA datasets (person name + scene, or style description)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import ChatMessage
from backend.engine.llm.chat_invoke import build_text_messages
from backend.engine.llm.message_content import extract_vision_instruction
from backend.engine.llm.prompts.system import (
    CONCEPT_LORA_CAPTION_RETRY_SYSTEM,
    CONCEPT_LORA_CAPTION_SYSTEM,
    STYLE_LORA_CAPTION_RETRY_SYSTEM,
    STYLE_LORA_CAPTION_SYSTEM,
)

_LEGACY_TEMPLATE_MARKERS = (
    "a photo of",
    "photo of",
    "sks",
    "dreambooth",
    "portrait of",
)

_GARBAGE_SCENE_RE = re.compile(r"^[\s!！?？.。,，、…\-_=~#@*]+$")
_PUNCT_RUN_RE = re.compile(r"([!！?？.。,，、…])\1{5,}")
_BANNED_BEAUTY_TEXTURE_TERMS = (
    "光滑",
    "磨皮",
    "无瑕",
    "零瑕疵",
    "细腻皮肤",
    "smooth skin",
    "flawless skin",
    "poreless",
    "airbrushed",
    "beauty retouch",
    "over-retouched",
    "plastic skin",
    "waxy skin",
)

VisionAnalyzeFn = Callable[[Path, list[ChatMessage]], str]
VisionBatchAnalyzeFn = Callable[..., list[str]]


def _count_meaningful_chars(text: str) -> int:
    return sum(1 for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def is_usable_scene_caption(text: str, *, min_meaningful: int = 2) -> bool:
    scene = (text or "").strip()
    if not scene:
        return False
    if _GARBAGE_SCENE_RE.match(scene):
        return False
    if _PUNCT_RUN_RE.search(scene):
        return False
    meaningful = _count_meaningful_chars(scene)
    if meaningful < min_meaningful:
        return False
    compact = scene.replace(" ", "")
    if len(compact) >= 8 and meaningful / max(len(compact), 1) < 0.15:
        return False
    return True


def build_concept_caption_messages(subject_name: str) -> list[ChatMessage]:
    subject = (subject_name or "").strip()
    parts = ["## Task", "Caption the attached photo."]
    if subject:
        parts.extend(
            [
                "",
                "## Training trigger word",
                f"{subject} (do NOT include in output)",
            ]
        )
        if any("\u4e00" <= ch <= "\u9fff" for ch in subject):
            parts.extend(["", "## Output language", "Use Chinese phrases in the caption body."])
    else:
        parts.extend(
            [
                "",
                "## Note",
                "No trigger word configured. Do not identify or name the person.",
            ]
        )
    return build_text_messages(system=CONCEPT_LORA_CAPTION_SYSTEM, user="\n".join(parts))


def build_concept_caption_retry_messages() -> list[ChatMessage]:
    return build_text_messages(
        system=CONCEPT_LORA_CAPTION_RETRY_SYSTEM,
        user="## Task\nCaption the attached photo.",
    )


def build_style_caption_messages() -> list[ChatMessage]:
    return build_text_messages(
        system=STYLE_LORA_CAPTION_SYSTEM,
        user="## Task\nCaption the attached image.",
    )


def build_style_caption_retry_messages() -> list[ChatMessage]:
    return build_text_messages(
        system=STYLE_LORA_CAPTION_RETRY_SYSTEM,
        user="## Task\nCaption the attached image.",
    )


def build_concept_auto_caption_instruction(subject_name: str) -> str:
    """Legacy VLM flat prompt — prefer ``build_concept_caption_messages``."""
    return extract_vision_instruction(build_concept_caption_messages(subject_name))


def build_style_auto_caption_instruction() -> str:
    return extract_vision_instruction(build_style_caption_messages())


def build_style_auto_caption_retry_instruction() -> str:
    return extract_vision_instruction(build_style_caption_retry_messages())


def build_concept_auto_caption_retry_instruction() -> str:
    return extract_vision_instruction(build_concept_caption_retry_messages())


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _looks_like_legacy_template(text: str) -> bool:
    low = (text or "").strip().lower()
    return any(marker in low for marker in _LEGACY_TEMPLATE_MARKERS)


def resolve_lora_subject_name(meta: dict[str, Any]) -> str:
    """Explicit trigger used for concept LoRA captions; dataset names are not identities."""
    trigger = str(meta.get("trigger_word") or "").strip()
    return trigger if trigger and not _looks_like_legacy_template(trigger) else ""


def clean_scene_caption(raw: str, *, subject_name: str = "") -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if ":" in text.splitlines()[0]:
        first = text.splitlines()[0]
        key, val = first.split(":", 1)
        if key.strip().upper() in {"DESCRIPTION", "SCENE", "CAPTION", "OUTPUT", "STYLE"}:
            text = val.strip()
    text = text.strip().strip('"\'')
    text = re.sub(
        r"^(the scene (is|shows)|description:|the image (shows|depicts|displays)|this (photo|image) (shows|depicts))\s*",
        "",
        text,
        flags=re.I,
    ).strip()
    text = re.sub(
        r"^(图片(展示了|显示了|描绘了)|照片(展示了|显示了|描绘了)|画面(展示了|显示了|中有|中有))\s*",
        "",
        text,
    ).strip()
    text = re.sub(r"^(描述|场景|画面描述|风格描述)\s*[：:]\s*", "", text).strip()

    subject = (subject_name or "").strip()
    if subject:
        for prefix in (f"{subject}，", f"{subject},", f"{subject}、", f"{subject}:"):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                break
        text = re.sub(rf"^{re.escape(subject)}[\s\-—:：]+", "", text).strip()
        if text == subject:
            text = ""
    return text.strip()


def _drop_banned_beauty_texture_phrases(text: str) -> str:
    phrases = [p.strip() for p in re.split(r"[，,、;；]+", text or "") if p.strip()]
    if not phrases:
        return text
    kept = [
        p
        for p in phrases
        if not any(term in p.lower() for term in _BANNED_BEAUTY_TEXTURE_TERMS)
    ]
    sep = "，" if _has_cjk(text) else ", "
    return sep.join(kept)


def normalize_scene_caption(raw: str, *, subject_name: str = "") -> str:
    """Clean VLM output and reject punctuation-only / garbage captions."""
    text = clean_scene_caption(raw, subject_name=subject_name)
    if not text:
        return ""
    text = re.sub(r"^[\s!！?？.。,，、…\-_=~#@*]+", "", text)
    text = re.sub(r"[\s!！?？.。,，、…\-_=~#@*]+$", "", text)
    text = re.sub(r"([!！?？.。,，、])\1{2,}", r"\1", text)
    text = text.strip()
    if (subject_name or "").strip():
        text = _drop_banned_beauty_texture_phrases(text).strip()
        if not text:
            return ""
    if not is_usable_scene_caption(text):
        return ""
    if len(text) > 200:
        for sep in ("，", ","):
            if sep in text[:200]:
                text = text[:200].rsplit(sep, 1)[0].strip()
                break
        else:
            text = text[:200].strip()
    return text


def compose_person_caption(subject_name: str, scene: str) -> str:
    subject = (subject_name or "").strip()
    scene = normalize_scene_caption(scene, subject_name=subject)
    if not subject:
        return scene
    if not scene:
        return subject
    sep = "，" if _has_cjk(subject) else ", "
    return f"{subject}{sep}{scene}"


def _default_batch_analyze(
    image_paths: list[Path],
    messages: list[ChatMessage],
    model_dir: Path,
    *,
    max_tokens: int = 128,
    temperature: float = 0.2,
) -> list[str]:
    from backend.engine.llm.vision_mlx import analyze_image_files_batch_messages

    return analyze_image_files_batch_messages(
        image_paths,
        model_dir,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def caption_dataset_image(
    path: Path,
    model_dir: Path,
    *,
    audit_kind: str = "concept",
    subject_name: str = "",
    analyze_fn: VisionAnalyzeFn | None = None,
) -> str:
    if analyze_fn is not None:

        def _analyze(image_path: Path, messages: list[ChatMessage], *, temperature: float = 0.2) -> str:
            del temperature
            return analyze_fn(image_path, messages)

        kind = (audit_kind or "concept").strip().lower()
        if kind == "style":
            raw = _analyze(path, build_style_caption_messages())
            scene = normalize_scene_caption(raw)
            if not scene:
                raw_retry = _analyze(path, build_style_caption_retry_messages())
                scene = normalize_scene_caption(raw_retry)
            return scene or "style reference"

        subject = (subject_name or "").strip()
        raw = _analyze(path, build_concept_caption_messages(subject))
        scene = normalize_scene_caption(raw, subject_name=subject)
        if not scene:
            raw_retry = _analyze(path, build_concept_caption_retry_messages())
            scene = normalize_scene_caption(raw_retry, subject_name=subject)
        return compose_person_caption(subject, scene)

    caps = caption_dataset_images_batch(
        [path],
        model_dir,
        audit_kind=audit_kind,
        subject_name=subject_name,
    )
    if not caps:
        raise RuntimeError(f"VLM auto-caption returned no caption for {path.name}")
    cap = str(caps[0] or "").strip()
    if not cap:
        raise RuntimeError(f"VLM auto-caption returned empty caption for {path.name}")
    return cap


def caption_dataset_images_batch(
    paths: list[Path],
    model_dir: Path,
    *,
    audit_kind: str = "concept",
    subject_name: str = "",
    batch_analyze_fn: VisionBatchAnalyzeFn | None = None,
) -> list[str]:
    """Caption many images with one VLM load per pass (primary + optional retry pass)."""
    if not paths:
        return []

    def analyze_batch(
        image_paths: list[Path],
        messages: list[ChatMessage],
        *,
        max_tokens: int = 128,
        temperature: float = 0.2,
    ) -> list[str]:
        if batch_analyze_fn is not None:
            return batch_analyze_fn(
                image_paths,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return _default_batch_analyze(
            image_paths,
            messages,
            model_dir,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    kind = (audit_kind or "concept").strip().lower()
    if kind == "style":
        raw_list = analyze_batch(paths, build_style_caption_messages())
        style_captions: list[str | None] = [None] * len(paths)
        retry_paths_style: list[Path] = []
        retry_indices_style: list[int] = []
        for idx, raw in enumerate(raw_list):
            scene = normalize_scene_caption(raw)
            if scene:
                style_captions[idx] = scene
            else:
                retry_paths_style.append(paths[idx])
                retry_indices_style.append(idx)
        if retry_paths_style:
            raw_retry = analyze_batch(
                retry_paths_style,
                build_style_caption_retry_messages(),
                max_tokens=128,
                temperature=0.1,
            )
            for idx, raw in zip(retry_indices_style, raw_retry, strict=False):
                style_captions[idx] = normalize_scene_caption(raw) or "style reference"
        return [cap if cap is not None else "style reference" for cap in style_captions]

    subject = (subject_name or "").strip()
    raw_list = analyze_batch(paths, build_concept_caption_messages(subject), max_tokens=128, temperature=0.2)

    captions: list[str | None] = [None] * len(paths)
    retry_paths: list[Path] = []
    retry_indices: list[int] = []

    for idx, raw in enumerate(raw_list):
        scene = normalize_scene_caption(raw, subject_name=subject)
        if scene:
            captions[idx] = compose_person_caption(subject, scene)
        else:
            retry_paths.append(paths[idx])
            retry_indices.append(idx)

    if retry_paths:
        raw_retry = analyze_batch(
            retry_paths,
            build_concept_caption_retry_messages(),
            max_tokens=128,
            temperature=0.1,
        )
        for idx, raw in zip(retry_indices, raw_retry, strict=False):
            captions[idx] = compose_person_caption(subject, raw)

    missing = [
        paths[i].name
        for i, cap in enumerate(captions)
        if cap is None or not str(cap).strip()
    ]
    if missing:
        preview = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise RuntimeError(
            f"VLM auto-caption failed for {len(missing)}/{len(paths)} image(s): {preview}{more}. "
            "Retry auto-caption or edit captions manually."
        )
    return [str(cap).strip() for cap in captions]
