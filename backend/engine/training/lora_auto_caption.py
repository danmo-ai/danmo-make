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

# Identity attributes the trigger word must learn; a phrase mentioning any of these is dropped
# from concept captions so the caption cannot "explain away" the face.
_IDENTITY_TERMS_EN = (
    "eye",
    "eyes",
    "eyebrow",
    "eyebrows",
    "nose",
    "lip",
    "lips",
    "mouth",
    "chin",
    "jaw",
    "jawline",
    "cheek",
    "cheeks",
    "cheekbones",
    "teeth",
    "face shape",
    "oval face",
    "round face",
    "square face",
    "heart-shaped face",
    "young",
    "youthful",
    "teen",
    "teenage",
    "middle-aged",
    "elderly",
    "old man",
    "old woman",
    "years old",
    "in her 20s",
    "in his 20s",
    "in her 30s",
    "in his 30s",
    "asian",
    "east asian",
    "chinese",
    "japanese",
    "korean",
    "caucasian",
    "white woman",
    "white man",
    "black woman",
    "black man",
    "african",
    "european",
    "latina",
    "latino",
    "hispanic",
    "ethnicity",
    "skin tone",
    "fair skin",
    "pale skin",
    "light skin",
    "dark skin",
    "tan skin",
    "tanned skin",
    "olive skin",
    "slender",
    "petite",
    "curvy",
    "chubby",
    "overweight",
    "muscular",
    "black hair",
    "dark hair",
    "brown hair",
    "blonde hair",
    "blond hair",
    "red hair",
    "grey hair",
    "gray hair",
    "white hair",
    "long hair",
    "short hair",
    "medium-length hair",
    "shoulder-length hair",
    "freckles",
    "mole",
    "moles",
    "acne",
    "wrinkles",
    "double eyelid",
    "monolid",
)
_IDENTITY_TERMS_CJK = (
    "眼睛",
    "双眼",
    "眼神",
    "眉毛",
    "鼻子",
    "鼻梁",
    "嘴唇",
    "嘴巴",
    "下巴",
    "下颌",
    "脸型",
    "脸颊",
    "颧骨",
    "牙齿",
    "五官",
    "年轻",
    "年龄",
    "中年",
    "老年",
    "少女",
    "少年",
    "岁",
    "亚洲",
    "东亚",
    "中国人",
    "日本人",
    "韩国人",
    "欧美",
    "白人",
    "黑人",
    "肤色",
    "白皮肤",
    "皮肤白",
    "小麦色",
    "黝黑",
    "身材",
    "苗条",
    "纤细",
    "丰满",
    "微胖",
    "高挑",
    "黑发",
    "黑色头发",
    "棕发",
    "棕色头发",
    "金发",
    "红发",
    "白发",
    "长发",
    "短发",
    "中长发",
    "齐肩",
    "雀斑",
    "痣",
    "痘",
    "皱纹",
    "双眼皮",
    "单眼皮",
)
_IDENTITY_TERMS_EN_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in _IDENTITY_TERMS_EN) + r")\b",
    flags=re.IGNORECASE,
)

# Concept captions must stay short so the trigger dominates the Qwen3 embedding.
MAX_CONCEPT_CAPTION_PHRASES = 8
MAX_CONCEPT_CAPTION_CHARS = 160

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


def _split_caption_phrases(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"[，,、;；。]+", text or "") if p.strip()]


def _join_caption_phrases(phrases: list[str], like: str) -> str:
    sep = "，" if _has_cjk(like) else ", "
    return sep.join(phrases)


def _mentions_identity(phrase: str) -> bool:
    low = phrase.lower()
    if _IDENTITY_TERMS_EN_RE.search(low):
        return True
    return any(term in phrase for term in _IDENTITY_TERMS_CJK)


def drop_identity_phrases(text: str) -> str:
    """Remove phrases describing the person's identity (face, age, ethnicity, hair colour, build).

    Concept LoRA captions should only carry what varies between photos; identity phrases teach the
    text encoder to explain the face away instead of binding it to the trigger word.
    """
    phrases = _split_caption_phrases(text)
    if not phrases:
        return text
    kept = [p for p in phrases if not _mentions_identity(p)]
    return _join_caption_phrases(kept, text)


def cap_caption_phrases(text: str, *, max_phrases: int = MAX_CONCEPT_CAPTION_PHRASES, max_chars: int = MAX_CONCEPT_CAPTION_CHARS) -> str:
    phrases = _split_caption_phrases(text)
    if not phrases:
        return text
    kept: list[str] = []
    total = 0
    for phrase in phrases[:max_phrases]:
        extra = len(phrase) + (2 if kept else 0)
        if kept and total + extra > max_chars:
            break
        kept.append(phrase)
        total += extra
    return _join_caption_phrases(kept, text)


def is_trigger_anchored_short_caption(caption: str, trigger: str) -> bool:
    """True when ``caption`` starts with (or contains) ``trigger`` and stays within the concept budget."""
    text = (caption or "").strip()
    trig = (trigger or "").strip()
    if not text or not trig or trig.lower() not in text.lower():
        return False
    body = re.sub(re.escape(trig), "", text, count=1, flags=re.IGNORECASE)
    phrases = _split_caption_phrases(body)
    return len(phrases) <= MAX_CONCEPT_CAPTION_PHRASES and len(body) <= MAX_CONCEPT_CAPTION_CHARS + 8


def normalize_scene_caption(raw: str, *, subject_name: str = "", concept: bool | None = None) -> str:
    """Clean VLM output and reject punctuation-only / garbage captions.

    ``concept`` (defaults to ``bool(subject_name)``) additionally strips identity / beauty-texture
    phrases and caps the phrase count so the trigger word dominates the embedding.
    """
    text = clean_scene_caption(raw, subject_name=subject_name)
    if not text:
        return ""
    text = re.sub(r"^[\s!！?？.。,，、…\-_=~#@*]+", "", text)
    text = re.sub(r"[\s!！?？.。,，、…\-_=~#@*]+$", "", text)
    text = re.sub(r"([!！?？.。,，、])\1{2,}", r"\1", text)
    text = text.strip()
    is_concept = bool((subject_name or "").strip()) if concept is None else bool(concept)
    if is_concept:
        text = _drop_banned_beauty_texture_phrases(text).strip()
        text = drop_identity_phrases(text).strip()
        if not text:
            return ""
    if not is_usable_scene_caption(text):
        return ""
    if is_concept:
        text = cap_caption_phrases(text)
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
    scene = normalize_scene_caption(scene, subject_name=subject, concept=True)
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
    from backend.persistence.stores import JsonConfigStore
    from backend.utils.path_utils import PathResolver
    from backend.engine.llm.vlm_http import analyze_image_files_batch_messages

    root = Path(__file__).resolve().parents[3]
    settings = JsonConfigStore(PathResolver(root)).load()
    from backend.engine.llm.llm_settings import DEFAULT_VLM_MODEL_ID

    return analyze_image_files_batch_messages(
        image_paths,
        model_dir,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        settings=settings,
        registry_model_id=DEFAULT_VLM_MODEL_ID,
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
        scene = normalize_scene_caption(raw, subject_name=subject, concept=True)
        if not scene:
            raw_retry = _analyze(path, build_concept_caption_retry_messages())
            scene = normalize_scene_caption(raw_retry, subject_name=subject, concept=True)
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
        scene = normalize_scene_caption(raw, subject_name=subject, concept=True)
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
