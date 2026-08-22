"""models_registry.json 解析辅助（供 ModelRegistry / Settings）。

各模型条目可含 ``commercial_use_allowed``：``true`` / ``false`` / ``null``。
``null`` 表示未在注册表断言，须自行核对上游许可；非法律意见。
``recommended: true`` 仅用于 ``commercial_use_allowed: true`` 的开源可商用模型。
``commercial_use`` 等字段由注册表维护者按 HF/魔搭公开许可信息手工填写。
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Tuple


def parse_i18n_string_pair(
    name_val: Any,
    desc_val: Any,
    key: str,
) -> Tuple[str, str, str, str]:
    """v2 注册表：name/description 为 {zh,en}；否则仅作占位字符串。返回 (name_zh, name_en, desc_zh, desc_en)。"""
    if isinstance(name_val, dict) and "zh" in name_val:
        nz = str(name_val.get("zh") or key)
        ne = str(name_val.get("en") or nz)
    else:
        nz = str(name_val or key)
        ne = nz
    if isinstance(desc_val, dict) and "zh" in desc_val:
        dz = str(desc_val.get("zh") or "")
        de = str(desc_val.get("en") or dz)
    else:
        dz = str(desc_val or "")
        de = dz
    return nz, ne, dz, de


def resolve_registry_label(val: Any, fallback: str = "", *, locale: str = "zh") -> str:
    """Pick zh/en string from registry name/description/version label."""
    if val is None:
        return fallback
    if isinstance(val, dict):
        loc = (locale or "zh").lower()
        if loc.startswith("en"):
            return str(val.get("en") or val.get("zh") or fallback)
        return str(val.get("zh") or val.get("en") or fallback)
    text = str(val).strip()
    return text if text else fallback


def registry_declares_action(actions: Any, action: str) -> bool:
    """True when models_registry ``actions`` includes *action* (registry verb, not API alias)."""
    return isinstance(actions, dict) and actions.get(action) is not None


def api_action_frozenset(actions: Any, *, media: str) -> FrozenSet[str]:
    """调度器 / IImageEngine.supports 使用的 API 级动作名。"""
    if not isinstance(actions, dict):
        return frozenset()
    if media == "video":
        s: set[str] = set()
        if actions.get("create") is not None:
            s.add("generate")
        if actions.get("animate") is not None:
            s.add("edit")
        if actions.get("upscale") is not None:
            s.add("upscale")
        if actions.get("avatar") is not None:
            s.add("avatar")
        if actions.get("avatar_script") is not None:
            s.add("avatar_script")
        return frozenset(s)
    if media == "audio":
        s: set[str] = set()
        if actions.get("create") is not None:
            s.add("create_music")
        if any(actions.get(k) is not None for k in ("cover", "repaint")):
            s.add("edit")
        return frozenset(s)
    if media == "llm":
        s: set[str] = set()
        if actions.get("chat") is not None:
            s.add("chat")
        if actions.get("enhance") is not None:
            s.add("enhance")
        if actions.get("describe") is not None:
            s.add("describe")
        return frozenset(s)
    s = set()
    if actions.get("create") is not None:
        s.add("generate")
    if any(actions.get(k) is not None for k in ("rewrite", "retouch", "extend")):
        s.add("edit")
    if actions.get("upscale") is not None:
        s.add("upscale")
    return frozenset(s)


def media_from_record(raw: Dict[str, Any]) -> str:
    m = raw.get("media")
    if m in ("image", "video", "audio", "llm"):
        return str(m)
    cat = str(raw.get("category") or "")
    return "video" if cat == "video_models" else "image"


def typed_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """为缺少 type 的 parameter 条目补上简单 schema（幂等）。"""
    out: Dict[str, Any] = {}
    for key, spec in parameters.items():
        if not isinstance(spec, dict):
            out[key] = spec
            continue
        if "type" in spec:
            out[key] = dict(spec)
            continue
        if isinstance(spec.get("default"), bool) or key.endswith("_support"):
            out[key] = {**spec, "type": "bool"}
            continue
        if "options" in spec and isinstance(spec["options"], list):
            out[key] = {**spec, "type": "enum"}
            continue
        if "min" in spec and "max" in spec:
            d = spec.get("default")
            t = "int" if isinstance(d, int) and not isinstance(d, bool) else "float"
            out[key] = {**spec, "type": t}
            continue
        out[key] = {**spec, "type": "object"}
    return out
