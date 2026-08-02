"""Character roster + multi-look cast for long-video keyframe prompts."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from backend.engine.llm.storyboard import (
    KEYFRAME_REF_DIVIDER,
    _LABELED_BLOCK_ZH,
    _STYLE_BLOCK_EN,
    _STYLE_LABELS,
    _parse_legacy_character_clause,
    _split_anchor_raw_blocks,
    extract_keyframe_shot_scene,
    normalize_storyboard_locale,
    prompt_locale,
)

_LOOK_TRIPLE_ZH = re.compile(r"^【角色·([^·】]+)·([^】]+)】\s*(.+)", re.S)
_LOOK_TRIPLE_EN = re.compile(r"^\[Character:\s*([^|]+)\|\s*([^\]]+)\]\s*(.+)", re.I | re.S)
_LOOK_TAG_ZH = re.compile(r"([^（(]+)[（(]([^）)]+)[）)]")
_LOOK_TAG_EN = re.compile(r"([A-Za-z\u4e00-\u9fff][^\s,]+)\s*\(([^)]+)\)")
_NAME_LOOK_TAG_RE = re.compile(r"([\u4e00-\u9fffA-Za-z·]{2,24})[（(]([^）)]+)[）)]")
_INVALID_CHARACTER_NAME_MARKERS = (
    "收到",
    "遭遇",
    "穿过",
    "攀登",
    "坠入",
    "之后",
    "然后",
    "短信",
    "挑战",
    "云端",
    "山顶",
)


def _is_valid_character_name(name: str) -> bool:
    n = (name or "").strip()
    if not n or len(n) > 10:
        return False
    return not any(marker in n for marker in _INVALID_CHARACTER_NAME_MARKERS)


@dataclass
class CharacterLook:
    id: str
    label: str
    body: str
    role: str = ""


@dataclass
class StoryboardCharacter:
    id: str
    name: str
    looks: list[CharacterLook] = field(default_factory=list)
    default_look_id: str = ""


@dataclass
class ShotCastLook:
    character_id: str
    look_id: str


def _stable_id(prefix: str, key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _default_look_label(locale: str) -> str:
    return "默认" if normalize_storyboard_locale(locale) == "zh" else "default"


_INVALID_LOOK_LABELS = frozenset(
    {
        "",
        "无标签",
        "（无标签）",
        "(无标签)",
        "无",
        "未命名",
        "（未命名）",
        "(未命名)",
        "untagged",
        "untitled",
        "default",
        "none",
        "n/a",
        "na",
        "-",
        "—",
    }
)


def is_placeholder_look_label(label: str) -> bool:
    raw = (label or "").strip()
    if not raw:
        return True
    if raw in _INVALID_LOOK_LABELS:
        return True
    lowered = raw.lower()
    if lowered in {x.lower() for x in _INVALID_LOOK_LABELS if x}:
        return True
    if lowered.startswith("<") and lowered.endswith(">"):
        return True
    return "无标签" in raw or "untagged" in lowered


def infer_look_labels_for_character(name: str, beat_sheet: list[str] | None) -> list[str]:
    if not name or not beat_sheet:
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for beat in beat_sheet:
        for m in _LOOK_TAG_ZH.finditer(beat or ""):
            if m.group(1).strip() != name:
                continue
            lbl = m.group(2).strip()
            if lbl and lbl not in seen and not is_placeholder_look_label(lbl):
                seen.add(lbl)
                labels.append(lbl)
    return labels


def _wardrobe_slug(wardrobe: str, *, max_len: int = 8) -> str:
    w = re.sub(r"\s+", "", (wardrobe or "").strip())
    if not w:
        return ""
    return w[:max_len]


def normalize_look_label(
    label: str,
    *,
    locale: str,
    name: str = "",
    wardrobe: str = "",
    beat_sheet: list[str] | None = None,
    look_index: int = 0,
) -> str:
    raw = (label or "").strip()
    if not is_placeholder_look_label(raw):
        return raw
    from_beats = infer_look_labels_for_character(name, beat_sheet)
    if from_beats:
        idx = min(max(look_index, 0), len(from_beats) - 1)
        return from_beats[idx]
    slug = _wardrobe_slug(wardrobe)
    if slug:
        return slug
    return _default_look_label(locale)


def strip_name_look_tags(text: str) -> str:
    """Remove Name（…） / Name(...) inline tags; keep bare character names."""
    raw = (text or "").strip()
    if not raw:
        return raw
    return _NAME_LOOK_TAG_RE.sub(r"\1", raw)


def find_name_look_tags(text: str) -> list[tuple[str, str]]:
    return [
        (m.group(1).strip(), m.group(2).strip())
        for m in _NAME_LOOK_TAG_RE.finditer(text or "")
        if m.group(1).strip()
    ]


def parse_character_roster(
    character_anchor: str,
    *,
    locale: str | None = None,
) -> tuple[list[StoryboardCharacter], str]:
    """Parse [Anchor]/[Cast] text into characters (multi-look) + shared style line."""
    anchor = (character_anchor or "").strip()
    if not anchor:
        return [], ""
    loc = normalize_storyboard_locale(locale) if locale else (
        "zh" if prompt_locale(anchor) == "zh" else "en"
    )
    default_label = _default_look_label(loc)
    style = ""
    by_name: dict[str, StoryboardCharacter] = {}

    for raw in _split_anchor_raw_blocks(anchor):
        block = raw.strip()
        if not block:
            continue
        m = _LOOK_TRIPLE_ZH.match(block)
        if m:
            name, look_label, body = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            _add_look(by_name, name, look_label, body, default_label)
            continue
        m = _LOOK_TRIPLE_EN.match(block)
        if m:
            name, look_label, body = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            _add_look(by_name, name, look_label, body, default_label)
            continue
        m = _LABELED_BLOCK_ZH.match(block)
        if m:
            label, body = m.group(1).strip(), m.group(2).strip()
            if label.startswith("角色·"):
                rest = label[3:]
                if "·" in rest:
                    name, look_label = rest.split("·", 1)
                    _add_look(by_name, name.strip(), look_label.strip(), body, default_label)
                else:
                    _add_look(by_name, rest.strip(), default_label, body, default_label)
                continue
            if label.lower() in _STYLE_LABELS or label in ("画风", "风格"):
                style = body
                continue
        m = _STYLE_BLOCK_EN.match(block)
        if m:
            style = m.group(2).strip()
            continue
        if not style and _looks_like_style_only_block(block):
            style = block
            continue
        legacy = _parse_legacy_character_clause(block)
        if legacy:
            name, body = legacy
            _add_look(by_name, name, default_label, body, default_label)
            continue

    return list(by_name.values()), style


def _looks_like_style_only_block(block: str) -> bool:
    s = (block or "").strip()
    if not s or len(s) > 120:
        return False
    if s.startswith("【") or s.startswith("["):
        return False
    return any(k in s for k in ("胶片", "色调", "镜头", "film", "palette", "35mm"))


def _add_look(
    by_name: dict[str, StoryboardCharacter],
    name: str,
    look_label: str,
    body: str,
    default_label: str,
) -> None:
    if not name or not body or not _is_valid_character_name(name):
        return
    char_id = _stable_id("char", name)
    if name not in by_name:
        by_name[name] = StoryboardCharacter(id=char_id, name=name, looks=[], default_look_id="")
    ch = by_name[name]
    label = look_label.strip()
    if is_placeholder_look_label(label):
        label = default_label
    look_id = _stable_id("look", f"{name}|{label}")
    if any(lk.id == look_id for lk in ch.looks):
        return
    ch.looks.append(CharacterLook(id=look_id, label=label, body=body.strip()))
    if not ch.default_look_id:
        ch.default_look_id = look_id


def format_character_roster(
    characters: list[StoryboardCharacter],
    style_anchor: str,
    *,
    locale: str | None = None,
) -> str:
    """Serialize roster to --- separated Anchor/Cast blocks."""
    loc = normalize_storyboard_locale(locale) if locale else "zh"
    lines: list[str] = []
    for ch in characters:
        for lk in ch.looks:
            if loc == "zh":
                lines.append(f"【角色·{ch.name}·{lk.label}】{lk.body}")
            else:
                lines.append(f"[Character: {ch.name} | {lk.label}] {lk.body}")
    if style_anchor.strip():
        lines.append(f"【画风】{style_anchor.strip()}" if loc == "zh" else f"[Style] {style_anchor.strip()}")
    return f"\n{KEYFRAME_REF_DIVIDER}\n".join(lines)


def roster_to_dtos(characters: list[StoryboardCharacter]) -> list[dict]:
    out: list[dict] = []
    for ch in characters:
        out.append(
            {
                "id": ch.id,
                "name": ch.name,
                "default_look_id": ch.default_look_id,
                "looks": [
                    {
                        "id": lk.id,
                        "label": lk.label,
                        "body": lk.body,
                        **({"role": lk.role} if lk.role else {}),
                    }
                    for lk in ch.looks
                ],
            }
        )
    return out


def dtos_to_roster(items: list[dict]) -> list[StoryboardCharacter]:
    chars: list[StoryboardCharacter] = []
    for row in items or []:
        looks = [
            CharacterLook(
                id=str(lk.get("id", "")),
                label=str(lk.get("label", "")),
                body=str(lk.get("body", "")),
                role=str(lk.get("role") or ""),
            )
            for lk in (row.get("looks") or [])
            if lk.get("body")
        ]
        if not looks:
            continue
        chars.append(
            StoryboardCharacter(
                id=str(row.get("id", "")),
                name=str(row.get("name", "")),
                looks=looks,
                default_look_id=str(row.get("default_look_id") or looks[0].id),
            )
        )
    return chars


def characters_on_screen(scene: str, characters: list[StoryboardCharacter]) -> list[StoryboardCharacter]:
    text = (scene or "").strip()
    if not text:
        return []
    return [ch for ch in characters if ch.name and ch.name in text]


def _match_look_by_hint(ch: StoryboardCharacter, hint: str) -> CharacterLook | None:
    hint = (hint or "").strip()
    if not hint:
        return None
    for lk in ch.looks:
        if hint == lk.label or hint in lk.label or lk.label in hint:
            return lk
    for lk in ch.looks:
        if hint in lk.body:
            return lk
    return None


def infer_look_label_from_text(text: str, character_name: str) -> str | None:
    raw = text or ""
    for m in _LOOK_TAG_ZH.finditer(raw):
        if m.group(1).strip() == character_name:
            return m.group(2).strip()
    for m in _LOOK_TAG_EN.finditer(raw):
        if m.group(1).strip() == character_name:
            return m.group(2).strip()
    return None


_LEAD_ROLE_VALUES = frozenset({"lead", "protagonist", "主角"})
_ROLE_IN_BODY_RE = re.compile(r"(?:定位|Role)\s*[：:]\s*(\S+)", re.I)


def _normalize_role_token(value: str) -> str:
    return (value or "").strip().lower()


def is_lead_character_role(role: str, *, body: str = "") -> bool:
    token = _normalize_role_token(role)
    if token in _LEAD_ROLE_VALUES:
        return True
    m = _ROLE_IN_BODY_RE.search(body or "")
    if m and _normalize_role_token(m.group(1)) in _LEAD_ROLE_VALUES:
        return True
    return False


def _label_scene_hint_score(label: str, hints: list[str]) -> float:
    """Score look label against scene entity names / location hints (substring, language-agnostic)."""
    label = (label or "").strip()
    if not label or not hints:
        return 0.0
    best = 0.0
    for hint in hints:
        h = (hint or "").strip()
        if not h:
            continue
        if label in h or h in label:
            return 1.0
        for n in range(min(len(label), 8), 1, -1):
            for i in range(0, len(label) - n + 1):
                frag = label[i : i + n]
                if len(frag) >= 2 and frag in h:
                    best = max(best, n / max(len(label), 1))
                    break
    return best


def infer_look_id_for_context(
    character: StoryboardCharacter,
    context: str,
    *,
    scene_hints: list[str] | None = None,
) -> str | None:
    """Pick a look when narrative/location/scene tokens overlap look labels (names-only; no inline tags)."""
    if len(character.looks) <= 1:
        return None
    from backend.long_video.prompt_overlap import prompt_token_coverage

    hint_blob = " ".join(h.strip() for h in (scene_hints or []) if h and h.strip())
    ctxt = "\n".join(p for p in ((context or "").strip(), hint_blob) if p).strip()
    if not ctxt:
        return None
    best_id: str | None = None
    best_score = 0.0
    for lk in character.looks:
        label = (lk.label or "").strip()
        if not label or is_placeholder_look_label(label):
            continue
        score = 1.0 if label in ctxt else prompt_token_coverage(ctxt, label)
        if hint_blob and label in hint_blob:
            score = max(score, 0.9)
        score = max(score, _label_scene_hint_score(label, scene_hints or []))
        body_snip = (lk.body or "")[:80]
        if body_snip:
            score = max(score, prompt_token_coverage(ctxt, body_snip) * 0.85)
        if score > best_score:
            best_score = score
            best_id = lk.id
    if best_score >= 0.34:
        return best_id
    return None


def infer_shot_cast_looks(
    *,
    scene: str,
    beat: str,
    characters: list[StoryboardCharacter],
    prev: list[ShotCastLook] | None = None,
    on_screen_names: list[str] | None = None,
    scene_hints: list[str] | None = None,
) -> list[ShotCastLook]:
    """Bind on-screen characters to looks: prev shot, context label overlap, else default."""
    prev_map = {c.character_id: c.look_id for c in (prev or [])}
    context = "\n".join(p for p in (scene or "", beat or "") if p).strip()
    cast: list[ShotCastLook] = []
    if on_screen_names:
        name_set = {n.strip() for n in on_screen_names if n.strip()}
        screen_chars = [ch for ch in characters if ch.name in name_set]
    else:
        screen_chars = characters_on_screen(context, characters)
    for ch in screen_chars:
        look_id = prev_map.get(ch.id)
        if not look_id:
            look_id = infer_look_id_for_context(ch, context, scene_hints=scene_hints)
        if not look_id:
            look_id = ch.default_look_id
        if not look_id and ch.looks:
            look_id = ch.looks[0].id
        if look_id:
            cast.append(ShotCastLook(character_id=ch.id, look_id=look_id))
    return cast


def _extra_look_label(name: str, *, locale: str) -> str:
    loc = normalize_storyboard_locale(locale)
    raw = (name or "").strip()
    if loc == "zh":
        return raw if 2 <= len(raw) <= 8 else "出镜"
    return raw if 2 <= len(raw) <= 16 else "on_screen"


def _extra_look_body(*, locale: str) -> str:
    loc = normalize_storyboard_locale(locale)
    if loc == "zh":
        return "定位：extra | 外貌：按镜头画面呈现"
    return "Role: extra | Appearance: as shown in frame"


def supplement_roster_from_shots(
    character_dtos: list[dict],
    shots: list[dict],
    *,
    locale: str = "zh",
) -> list[dict]:
    """Add minimal roster rows for on-screen names missing from the cast roster (generic extras)."""
    from backend.engine.llm.storyboard import normalize_storyboard_locale as _norm_loc

    loc = _norm_loc(locale)
    roster = dtos_to_roster(character_dtos)
    known = {ch.name for ch in roster if ch.name}
    on_screen: set[str] = set()
    for shot in shots or []:
        for name in shot.get("characters_on_screen") or []:
            n = str(name).strip()
            if n and _is_valid_character_name(n):
                on_screen.add(n)
    if not on_screen:
        return list(character_dtos)

    out = [dict(row) for row in character_dtos]
    for name in sorted(on_screen):
        if name in known:
            continue
        label = _extra_look_label(name, locale=loc)
        look_id = _stable_id("look", f"{name}|{label}")
        char_id = _stable_id("char", name)
        out.append(
            {
                "id": char_id,
                "name": name,
                "default_look_id": look_id,
                "looks": [
                    {
                        "id": look_id,
                        "label": label,
                        "body": _extra_look_body(locale=loc),
                        "role": "extra",
                    }
                ],
            }
        )
        known.add(name)
    return out


def cast_looks_to_dtos(cast: list[ShotCastLook]) -> list[dict]:
    return [{"character_id": c.character_id, "look_id": c.look_id} for c in cast]


def ensure_cast_covers_on_screen(
    cast: list[ShotCastLook],
    *,
    on_screen_names: list[str],
    characters: list[StoryboardCharacter],
) -> list[ShotCastLook]:
    """Every characters_on_screen name gets a cast row (default look fallback)."""
    if not on_screen_names:
        return list(cast)
    by_name = {ch.name: ch for ch in characters}
    present = {c.character_id for c in cast}
    out = list(cast)
    for name in on_screen_names:
        n = (name or "").strip()
        if not n:
            continue
        ch = by_name.get(n)
        if not ch or ch.id in present:
            continue
        look_id = ch.default_look_id or (ch.looks[0].id if ch.looks else "")
        if look_id:
            out.append(ShotCastLook(character_id=ch.id, look_id=look_id))
            present.add(ch.id)
    return out


def dtos_to_cast_looks(items: list[dict]) -> list[ShotCastLook]:
    out: list[ShotCastLook] = []
    for row in items or []:
        cid = str(row.get("character_id", "")).strip()
        lid = str(row.get("look_id", "")).strip()
        if cid and lid:
            out.append(ShotCastLook(character_id=cid, look_id=lid))
    return out


def format_cast_reference_blocks(
    characters: list[StoryboardCharacter],
    cast: list[ShotCastLook],
    style_anchor: str,
    *,
    locale: str | None = None,
) -> str:
    loc = normalize_storyboard_locale(locale) if locale else "zh"
    cast_map = {c.character_id: c.look_id for c in cast}
    lines: list[str] = []
    for ch in characters:
        look_id = cast_map.get(ch.id)
        if cast and ch.id not in cast_map:
            continue
        lk = next((l for l in ch.looks if l.id == look_id), None) if look_id else None
        if not lk and ch.looks:
            lk = next((l for l in ch.looks if l.id == ch.default_look_id), ch.looks[0])
        if not lk:
            continue
        if loc == "zh":
            lines.append(f"【角色·{ch.name}·{lk.label}】{lk.body}")
        else:
            lines.append(f"[Character: {ch.name} | {lk.label}] {lk.body}")
    if style_anchor.strip():
        lines.append(f"【画风】{style_anchor.strip()}" if loc == "zh" else f"[Style] {style_anchor.strip()}")
    return f"\n{KEYFRAME_REF_DIVIDER}\n".join(lines)


def compose_keyframe_with_cast(
    scene: str,
    *,
    characters: list[StoryboardCharacter],
    cast: list[ShotCastLook],
    style_anchor: str = "",
    locale: str | None = None,
    character_anchor: str = "",
) -> str:
    """Build T2I prompt: scene first; cast/style reference appended after ---."""
    scene = (scene or "").strip()
    if not scene and not characters and not character_anchor:
        return ""
    loc = normalize_storyboard_locale(locale) if locale else (
        "zh" if prompt_locale(scene or character_anchor) == "zh" else "en"
    )
    if not characters and character_anchor.strip():
        from backend.engine.llm.storyboard import compose_keyframe_visual_prompt

        return compose_keyframe_visual_prompt(scene or character_anchor, character_anchor, locale=loc)
    screen_chars = characters_on_screen(scene, characters) if scene else list(characters)
    on_screen = {ch.id for ch in screen_chars}
    cast_filtered = [c for c in cast if c.character_id in on_screen] if on_screen else cast
    if not cast_filtered and screen_chars:
        cast_filtered = infer_shot_cast_looks(scene=scene, beat=scene, characters=characters)
    ref_chars = screen_chars if screen_chars else characters
    ref = format_cast_reference_blocks(ref_chars, cast_filtered, style_anchor, locale=loc)
    if not scene:
        return ref
    if not ref:
        return scene
    from backend.engine.llm.storyboard import join_keyframe_prompt

    return join_keyframe_prompt(scene, ref, locale=loc)


def shot_cast_from_visual(
    visual: str,
    characters: list[StoryboardCharacter],
) -> list[ShotCastLook]:
    """Best-effort parse look labels from an already-composed visual prompt."""
    scene = extract_keyframe_shot_scene(visual)
    return infer_shot_cast_looks(scene=scene, beat=visual, characters=characters)
