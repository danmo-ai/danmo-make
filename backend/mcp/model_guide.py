"""Compact model cards for MCP agents — only generation-essential fields."""

from __future__ import annotations

import re
from typing import Any

_MEDIA_BUCKETS = ("image", "video", "audio")

_DISTILL_TOKENS = ("distill", "turbo", "schnell", "lightning", "dmd")

# Registry verbs (create/rewrite/…) → API verbs used by /api/models?action= and engines.
_ACTION_ALIASES = {
    "create": "generate",
    "generate": "generate",
    "rewrite": "edit",
    "retouch": "edit",
    "extend": "edit",
    "animate": "edit",
    "edit": "edit",
    "cover": "edit",
    "repaint": "edit",
    "upscale": "upscale",
    "create_music": "create_music",
    "avatar": "avatar",
    "avatar_script": "avatar_script",
    "chat": "chat",
    "enhance": "enhance",
}


def normalize_list_action(action: str | None, *, media: str | None = None) -> str | None:
    """Map registry/skill verbs to /api/models action filter (API surface)."""
    if action is None:
        return None
    raw = str(action).strip().lower()
    if not raw:
        return None
    if raw in ("create", "generate"):
        if (media or "").strip().lower() == "audio":
            return "create_music"
        return "generate"
    return _ACTION_ALIASES.get(raw, raw)


def _model_type(cfg: dict[str, Any], row: dict[str, Any]) -> str:
    for src in (cfg, row):
        t = src.get("type")
        if isinstance(t, str) and t.strip():
            return t.strip().lower()
    cat = cfg.get("category") or row.get("category")
    if cat == "loras":
        return "lora"
    return ""


def _api_actions(cfg: dict[str, Any], row: dict[str, Any]) -> list[str]:
    """API-level actions (generate/edit/upscale/…), sorted."""
    raw = row.get("actions")
    if isinstance(raw, list):
        return sorted({str(a) for a in raw if a})
    if isinstance(raw, (set, frozenset)):
        return sorted({str(a) for a in raw if a})
    acts = cfg.get("actions")
    if isinstance(acts, dict):
        media = str(row.get("media") or cfg.get("media") or "image")
        from backend.core.registry_format import api_action_frozenset

        return sorted(api_action_frozenset(acts, media=media))
    return []


def _param_default(params: dict[str, Any], key: str) -> Any:
    spec = params.get(key)
    if isinstance(spec, dict) and "default" in spec:
        return spec.get("default")
    return None


def _slim_range(spec: dict[str, Any]) -> dict[str, Any]:
    """Keep type/default/min/max/step/options only (no UI labels/notes)."""
    out: dict[str, Any] = {}
    for k in ("type", "default", "min", "max", "step", "options"):
        if k in spec:
            out[k] = spec[k]
    return out


def _bilingual_name(raw: dict[str, Any]) -> dict[str, str] | str:
    name = raw.get("name")
    if isinstance(name, dict):
        out = {k: str(v) for k, v in name.items() if v}
        return out or str(raw.get("id") or "")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return ""


def _size_spec(params: dict[str, Any]) -> dict[str, Any] | None:
    """Necessary size field: default + options (WIDTHxHEIGHT)."""
    presets = params.get("resolution_presets")
    default_size = None
    options: list[Any] | None = None
    if isinstance(presets, dict):
        default_size = presets.get("default")
        opts = presets.get("options")
        if isinstance(opts, list) and opts:
            options = opts[:32]
    if not default_size:
        w = _param_default(params, "width")
        h = _param_default(params, "height")
        if w is not None and h is not None:
            default_size = f"{w}x{h}"
    if default_size is None and not options:
        return None
    out: dict[str, Any] = {"type": "enum"}
    if default_size is not None:
        out["default"] = default_size
    if options:
        out["options"] = options
    return out


def _fps_value(params: dict[str, Any]) -> int:
    fps = _param_default(params, "fps")
    try:
        return max(1, int(fps))
    except (TypeError, ValueError):
        return 16


def duration_sec_from_frames(num_frames: int, fps: int) -> float:
    rate = max(1, int(fps))
    nf = max(1, int(num_frames))
    return round(max(0.0, (nf - 1) / rate), 1)


def frames_from_duration_sec(
    duration_sec: float,
    fps: int,
    *,
    min_frames: int | None = None,
    max_frames: int | None = None,
) -> int:
    """UI formula: num_frames ≈ duration_sec * fps + 1."""
    rate = max(1, int(fps))
    frames = int(round(max(0.0, float(duration_sec)) * rate + 1))
    if min_frames is not None:
        frames = max(int(min_frames), frames)
    if max_frames is not None:
        frames = min(int(max_frames), frames)
    return max(1, frames)


def _duration_spec(params: dict[str, Any]) -> dict[str, Any] | None:
    """Video duration in seconds (derived from num_frames + fps), for ask_user."""
    nf = params.get("num_frames")
    if not isinstance(nf, dict):
        return None
    fps = _fps_value(params)
    default_frames = nf.get("default")
    min_f = nf.get("min")
    max_f = nf.get("max")
    default_sec: float | None = None
    if isinstance(default_frames, (int, float)) and default_frames > 0:
        default_sec = duration_sec_from_frames(int(default_frames), fps)
    candidates = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20)
    options: list[float] = []
    for sec in candidates:
        frames = frames_from_duration_sec(sec, fps)
        if isinstance(min_f, (int, float)) and frames < int(min_f):
            continue
        if isinstance(max_f, (int, float)) and frames > int(max_f):
            continue
        options.append(float(sec))
    if default_sec is not None and default_sec not in options:
        options.append(default_sec)
        options.sort()
    if default_sec is None and not options:
        return None
    out: dict[str, Any] = {"type": "number", "unit": "seconds"}
    if default_sec is not None:
        out["default"] = default_sec
    if options:
        out["options"] = options
    if isinstance(min_f, (int, float)):
        out["min"] = duration_sec_from_frames(int(min_f), fps)
    if isinstance(max_f, (int, float)):
        out["max"] = duration_sec_from_frames(int(max_f), fps)
    return out


def _version_bits(version_key: str, version_cfg: dict[str, Any]) -> int:
    """Lower bits = smaller quant (preferred). Full precision → 16."""
    q = version_cfg.get("quantization")
    if isinstance(q, dict) and isinstance(q.get("bits"), (int, float)):
        return int(q["bits"])
    key = version_key.lower()
    for token, bits in (
        ("q4", 4),
        ("int4", 4),
        ("q8", 8),
        ("int8", 8),
        ("fp8", 8),
        ("bf16", 16),
        ("fp16", 16),
    ):
        if token in key:
            return bits
    return 16


def _best_quant_bits(versions: dict[str, Any]) -> int:
    bits: list[int] = []
    for vk, vv in versions.items():
        if isinstance(vv, dict):
            bits.append(_version_bits(str(vk), vv))
    return min(bits) if bits else 16


def _preferred_version_key(versions: dict[str, Any]) -> str:
    """Smallest quant first; tie-break default flag then key name."""
    ranked: list[tuple[int, int, str]] = []
    for vk, vv in versions.items():
        if not isinstance(vv, dict):
            continue
        ranked.append(
            (
                _version_bits(str(vk), vv),
                0 if vv.get("default") else 1,
                str(vk),
            )
        )
    if not ranked:
        return ""
    ranked.sort()
    return ranked[0][2]


def _is_distill(model_id: str, cfg: dict[str, Any]) -> bool:
    if cfg.get("distilled_from"):
        return True
    idl = model_id.lower()
    return any(t in idl for t in _DISTILL_TOKENS)


def _recency_score(model_id: str, cfg: dict[str, Any]) -> int:
    """Higher = newer. Superseded models (have successor) rank lowest."""
    if cfg.get("successor"):
        return 0
    stamps = [int(x) for x in re.findall(r"(?<![0-9])(\d{4})(?![0-9])", model_id)]
    dotted = re.search(r"(?<![0-9])(\d+)\.(\d+)", model_id)
    score = 1
    if stamps:
        score = max(score, max(stamps))
    if dotted:
        score = max(score, int(dotted.group(1)) * 100 + int(dotted.group(2)))
    return score


def _priority_tuple(model_id: str, cfg: dict[str, Any], row: dict[str, Any]) -> tuple:
    """商用优先 → 新版本优先 → 蒸馏优先 → 量化越小越优先."""
    commercial = cfg.get("commercial_use_allowed")
    if commercial is None:
        commercial = row.get("commercial_use_allowed")
    versions = cfg.get("versions") if isinstance(cfg.get("versions"), dict) else {}
    return (
        0 if commercial is True else 1,
        0 if not cfg.get("successor") else 1,
        -_recency_score(model_id, cfg),
        0 if _is_distill(model_id, cfg) else 1,
        _best_quant_bits(versions) if isinstance(versions, dict) else 16,
        model_id,
    )


def summarize_model(
    model_id: str,
    *,
    index_row: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    full: bool = False,
) -> dict[str, Any]:
    """Agent-facing card: list = id/name/media/defaults; full = +parameters ranges."""
    row = index_row or {}
    cfg = config or {}
    params = cfg.get("parameters") if isinstance(cfg.get("parameters"), dict) else {}
    if not isinstance(params, dict):
        params = {}
    versions = cfg.get("versions") if isinstance(cfg.get("versions"), dict) else {}

    media = row.get("media") or cfg.get("media")
    size = _size_spec(params)

    defaults: dict[str, Any] = {}
    if size and "default" in size:
        defaults["size"] = size["default"]
    for key in ("steps", "guidance", "num_frames", "fps", "duration"):
        val = _param_default(params, key)
        if val is not None:
            defaults[key] = val
    duration = _duration_spec(params) if media == "video" else None
    if media == "video" and duration and "default" in duration:
        defaults["duration_sec"] = duration["default"]

    actions = _api_actions(cfg, row)
    mtype = _model_type(cfg, row)
    card: dict[str, Any] = {
        "id": model_id,
        "name": _bilingual_name(cfg) or model_id,
        "media": media,
        "type": mtype or None,
        "actions": actions,
        "installed": row.get("installed"),
        "defaults": defaults,
    }
    if card["type"] is None:
        card.pop("type", None)
    pref_ver = _preferred_version_key(versions) if isinstance(versions, dict) else ""
    if pref_ver:
        card["preferred_version"] = pref_ver

    if full:
        parameters: dict[str, Any] = {}
        if size:
            parameters["size"] = size
        for key in ("steps", "guidance", "num_frames", "fps", "duration"):
            spec = params.get(key)
            if isinstance(spec, dict):
                slim = _slim_range(spec)
                if slim:
                    parameters[key] = slim
        if media == "image":
            for k in ("num_frames", "fps", "duration", "duration_sec"):
                parameters.pop(k, None)
                defaults.pop(k, None)
        elif media == "video":
            # Prefer ask_user duration_sec; keep fps range; hide raw num_frames from agent surface.
            parameters.pop("duration", None)
            parameters.pop("num_frames", None)
            defaults.pop("duration", None)
            defaults.pop("num_frames", None)
            if duration:
                parameters["duration_sec"] = duration
        elif media == "audio":
            for k in ("size", "num_frames", "fps", "duration_sec"):
                parameters.pop(k, None)
                defaults.pop(k, None)
        card["parameters"] = parameters
        card["defaults"] = defaults
        commercial = cfg.get("commercial_use_allowed")
        if commercial is None:
            commercial = row.get("commercial_use_allowed")
        card["commercial_use_allowed"] = commercial is True

    return card


def enrich_model_list(
    index: dict[str, Any],
    registry_models: dict[str, Any],
    *,
    defaults_by_media: dict[str, str] | None = None,
    require_action: str | None = None,
) -> dict[str, Any]:
    """Group by media; sort 商用 → 新版本 → 蒸馏 → 小量化 (first = preferred).

    Excludes LoRA/adapter rows (type=lora) — they are not generation models.
    Cards always include API ``actions`` (generate/edit/upscale/…).
    """
    del defaults_by_media
    empty = {m: [] for m in _MEDIA_BUCKETS}
    models = index.get("models") if isinstance(index, dict) else None
    if not isinstance(models, dict):
        return {
            **empty,
            "hint": "No models matched filters. Prefer list_models(installed=true, action=generate).",
        }

    want = normalize_list_action(require_action)
    buckets: dict[str, list[tuple[tuple, dict[str, Any]]]] = {m: [] for m in _MEDIA_BUCKETS}
    for mid, row in models.items():
        cfg = registry_models.get(mid) if isinstance(registry_models, dict) else None
        if not isinstance(cfg, dict):
            cfg = {}
        row_d = row if isinstance(row, dict) else {}
        if "commercial_use_allowed" not in cfg:
            cfg = {**cfg, "commercial_use_allowed": row_d.get("commercial_use_allowed")}
        if _model_type(cfg, row_d) == "lora":
            continue
        card = summarize_model(mid, index_row=row_d, config=cfg, full=False)
        actions = card.get("actions") if isinstance(card.get("actions"), list) else []
        if not actions:
            # Empty actions = adapter / non-runnable catalog row.
            continue
        if want and want not in actions:
            continue
        media = str(card.get("media") or "")
        if media not in buckets:
            continue
        buckets[media].append((_priority_tuple(mid, cfg, row_d), card))

    out_buckets: dict[str, list[dict[str, Any]]] = {}
    for media, items in buckets.items():
        items.sort(key=lambda it: it[0])
        out_buckets[media] = [card for _, card in items]

    return {
        "image": out_buckets["image"],
        "video": out_buckets["video"],
        "audio": out_buckets["audio"],
        "hint": (
            "Grouped by media; sorted commercial → newer → distilled → smaller quant. "
            "Each card has actions (API: generate/edit/upscale/…). "
            "For text-to-image/video use list_models(action=generate); "
            "for edits use action=edit. LoRA adapters are omitted. "
            "Suggest [0], ask_user, then get_model for size/duration options."
        ),
    }
