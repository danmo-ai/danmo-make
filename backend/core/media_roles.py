"""Submit-time media roles for image and video requests.

One ordered ``media[]`` list. The role set selects the task. Models declare
which roles each registry action accepts via ``actions.<action>.media_roles``.
When that key is absent, defaults follow the action:

- image ``rewrite`` → ``reference_image``
- image ``retouch`` → ``mask``
- image ``extend`` → ``extend`` (direction spec, not an asset)
- image / video ``create`` → no attached roles (text-only)
- video ``animate`` → ``first_frame``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

KEYFRAME_ROLES = frozenset({"first_frame", "last_frame"})
REFERENCE_ROLES = frozenset({"reference_image", "reference_video", "reference_audio"})

_IMAGE_ACTION_DEFAULTS: dict[str, frozenset[str]] = {
    "create": frozenset(),
    "rewrite": frozenset({"reference_image"}),
    "retouch": frozenset({"mask"}),
    "extend": frozenset({"extend"}),
}
_VIDEO_ACTION_DEFAULTS: dict[str, frozenset[str]] = {
    "create": frozenset(),
    "animate": frozenset({"first_frame"}),
}

_KIND_TO_ROLE = {
    "image": "reference_image",
    "video": "reference_video",
    "audio": "reference_audio",
}


@dataclass(frozen=True)
class MediaRoleRejection:
    code: Literal["unsupported", "invalid"]
    message: str


def _actions(entry: Any) -> dict[str, Any]:
    raw = getattr(entry, "raw", None) or {}
    acts = raw.get("actions") if isinstance(raw, dict) else None
    return acts if isinstance(acts, dict) else {}


def _parameters(entry: Any) -> dict[str, Any]:
    params = getattr(entry, "parameters", None)
    return params if isinstance(params, dict) else {}


def action_media_roles(entry: Any, action: str, *, media: str) -> frozenset[str]:
    """Roles allowed while serving registry ``action``. Explicit list replaces the default."""
    block = _actions(entry).get(action)
    if isinstance(block, dict) and "media_roles" in block:
        listed = block.get("media_roles") or []
        if not isinstance(listed, list):
            return frozenset()
        return frozenset(str(item) for item in listed if item)
    defaults = _VIDEO_ACTION_DEFAULTS if media == "video" else _IMAGE_ACTION_DEFAULTS
    return defaults.get(action, frozenset())


def model_media_roles(entry: Any, *, media: str) -> list[str]:
    """Union of roles the model declares, for API/MCP cards."""
    acts = _actions(entry)
    found: set[str] = set()
    for name in acts:
        if acts.get(name) is None:
            continue
        found |= action_media_roles(entry, str(name), media=media)
    return sorted(found)


def _asset_id(value: Any) -> str:
    return str(value or "").strip().removeprefix("asset:")


def _items(req: Any) -> list[Any]:
    return list(getattr(req, "media", None) or [])


def _role_of(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("role") or "").strip()
    return str(getattr(item, "role", "") or "").strip()


def _id_of(item: Any) -> str:
    raw = item.get("asset_id") if isinstance(item, dict) else getattr(item, "asset_id", None)
    return _asset_id(raw)


def _assign(current: str, new: str, field: str) -> str:
    if not new:
        return current
    if current and current != new:
        raise ValueError(f"{field} {current!r} conflicts with media asset {new!r}")
    return new


def _reference(kind: str, asset_id: str) -> Any:
    from backend.core.contracts import VideoMediaReference

    return VideoMediaReference(kind=kind, asset_id=asset_id)  # type: ignore[arg-type]


def project_image_edit(req: Any) -> None:
    """Copy ``media`` roles onto source / mask / reference fields when those are empty."""
    refs = list(getattr(req, "reference_asset_ids", None) or [])
    source = _asset_id(getattr(req, "source_asset_id", ""))
    mask = _asset_id(getattr(req, "mask_asset_id", "") or "")
    for item in _items(req):
        role = _role_of(item)
        aid = _id_of(item)
        if not aid:
            raise ValueError(f"media role {role or '?'} requires asset_id")
        if role == "mask":
            mask = _assign(mask, aid, "mask_asset_id")
        elif role == "reference_image":
            if not source:
                source = aid
            elif aid != source and aid not in refs:
                refs.append(aid)
        elif role:
            raise ValueError(f"image edit does not accept media role {role!r}")
    req.source_asset_id = source
    req.mask_asset_id = mask or None
    req.reference_asset_ids = refs


def project_video_edit(req: Any) -> None:
    source = _asset_id(getattr(req, "source_asset_id", ""))
    tail = _asset_id(getattr(req, "tail_asset_id", "") or "")
    refs = list(getattr(req, "media_references", None) or [])
    stills = [_asset_id(x) for x in (getattr(req, "reference_asset_ids", None) or []) if _asset_id(x)]
    for item in _items(req):
        role = _role_of(item)
        aid = _id_of(item)
        if not aid:
            raise ValueError(f"media role {role or '?'} requires asset_id")
        if role in ("first_frame", "source_video"):
            source = _assign(source, aid, "source_asset_id")
        elif role == "last_frame":
            tail = _assign(tail, aid, "tail_asset_id")
        elif role in _KIND_TO_ROLE.values():
            kind = next(k for k, mapped in _KIND_TO_ROLE.items() if mapped == role)
            if not _has_ref(refs, kind, aid):
                refs.append(_reference(kind, aid))
            if role == "reference_image" and aid not in stills:
                stills.append(aid)
        elif role:
            raise ValueError(f"video edit does not accept media role {role!r}")
    req.source_asset_id = source
    req.tail_asset_id = tail or None
    req.media_references = refs
    req.reference_asset_ids = stills


def project_video_generate(req: Any) -> None:
    refs = list(getattr(req, "media_references", None) or [])
    stills = [_asset_id(x) for x in (getattr(req, "reference_asset_ids", None) or []) if _asset_id(x)]
    for item in _items(req):
        role = _role_of(item)
        aid = _id_of(item)
        if role in KEYFRAME_ROLES or role == "source_video":
            continue
        if not role:
            continue
        if not aid:
            raise ValueError(f"media role {role} requires asset_id")
        if role not in _KIND_TO_ROLE.values():
            raise ValueError(f"video generation does not accept media role {role!r}")
        kind = next(k for k, mapped in _KIND_TO_ROLE.items() if mapped == role)
        if not _has_ref(refs, kind, aid):
            refs.append(_reference(kind, aid))
        if role == "reference_image" and aid not in stills:
            stills.append(aid)
    req.media_references = refs
    req.reference_asset_ids = stills


def _has_ref(refs: list[Any], kind: str, asset_id: str) -> bool:
    for ref in refs:
        ref_kind = str(getattr(ref, "kind", "") or "").strip().lower()
        if ref_kind == kind and _asset_id(getattr(ref, "asset_id", "")) == asset_id:
            return True
    return False


def _explicit_roles(req: Any) -> set[str]:
    return {role for item in _items(req) if (role := _role_of(item))}


def collect_image_roles(req: Any, *, endpoint: str) -> set[str]:
    roles = set(_explicit_roles(req))
    if endpoint == "generate":
        return roles
    operation = str(getattr(req, "operation", "") or "")
    if operation == "rewrite":
        if _asset_id(getattr(req, "source_asset_id", "")) or list(getattr(req, "reference_asset_ids", None) or []):
            roles.add("reference_image")
    elif list(getattr(req, "reference_asset_ids", None) or []):
        roles.add("reference_image")
    if _asset_id(getattr(req, "mask_asset_id", "") or ""):
        roles.add("mask")
    if getattr(req, "extend", None) is not None:
        roles.add("extend")
    return roles


def collect_video_roles(req: Any, entry: Any, *, endpoint: str) -> set[str]:
    roles = set(_explicit_roles(req))
    if endpoint == "edit":
        if "first_frame" not in roles and "source_video" not in roles and _asset_id(getattr(req, "source_asset_id", "")):
            mode = str(_parameters(entry).get("video_edit_source_mode") or "image_only")
            roles.add("source_video" if mode == "source_video" else "first_frame")
        if "last_frame" not in roles and _asset_id(getattr(req, "tail_asset_id", "") or ""):
            roles.add("last_frame")
    for aid in getattr(req, "reference_asset_ids", None) or []:
        if _asset_id(aid):
            roles.add("reference_image")
    for ref in getattr(req, "media_references", None) or []:
        kind = str(getattr(ref, "kind", "") or "").strip().lower()
        mapped = _KIND_TO_ROLE.get(kind)
        if mapped and _asset_id(getattr(ref, "asset_id", "")):
            roles.add(mapped)
    return roles


def _unsupported(model_id: str, what: str) -> MediaRoleRejection:
    return MediaRoleRejection(
        "unsupported",
        f"model {model_id!r} does not support {what}",
    )


def _invalid(message: str) -> MediaRoleRejection:
    return MediaRoleRejection("invalid", message)


def reject_image_roles(entry: Any, req: Any, *, endpoint: str) -> MediaRoleRejection | None:
    model_id = str(getattr(req, "model", "") or getattr(entry, "id", "")).split(":", 1)[0]
    roles = collect_image_roles(req, endpoint=endpoint)
    if endpoint == "generate":
        allowed = action_media_roles(entry, "create", media="image")
        extra = roles - allowed
        if extra:
            return _unsupported(model_id, f"media role {sorted(extra)[0]}")
        return None
    operation = str(getattr(req, "operation", "") or "")
    if operation not in ("rewrite", "retouch", "extend"):
        return _invalid(f"image edit operation {operation!r} is not rewrite, retouch, or extend")
    if not _asset_id(getattr(req, "source_asset_id", "")):
        return _invalid("image edit requires a source image")
    if _actions(entry).get(operation) is None:
        return _unsupported(model_id, operation)
    if operation == "retouch" and "mask" not in roles:
        return _invalid("retouch requires a mask asset")
    if operation == "extend" and "extend" not in roles:
        return _invalid("extend requires extend.directions and extend.pixels")
    if operation == "rewrite" and roles & {"mask", "extend"}:
        return _invalid("rewrite does not accept a mask or extend spec")
    if operation == "retouch" and "extend" in roles:
        return _invalid("retouch does not accept an extend spec")
    if operation == "extend" and "mask" in roles:
        return _invalid("extend does not accept a mask")
    allowed = action_media_roles(entry, operation, media="image")
    extra = roles - allowed
    if extra:
        return _unsupported(model_id, f"media role {sorted(extra)[0]}")
    return None


def reject_video_roles(entry: Any, req: Any, *, endpoint: str) -> MediaRoleRejection | None:
    model_id = str(getattr(req, "model", "") or getattr(entry, "id", "")).split(":", 1)[0]
    roles = collect_video_roles(req, entry, endpoint=endpoint)
    if "last_frame" in roles and "first_frame" not in roles:
        return _invalid("last_frame requires first_frame")
    keyframes = roles & KEYFRAME_ROLES
    references = roles & REFERENCE_ROLES
    if keyframes and references:
        return _invalid("first/last frames cannot be combined with reference media")
    if keyframes and "source_video" in roles:
        return _invalid("first/last frames cannot be combined with source_video")
    if endpoint == "edit" and _actions(entry).get("animate") is None:
        return _unsupported(model_id, "animate")
    if endpoint == "edit" and not _asset_id(getattr(req, "source_asset_id", "")):
        return _invalid("video edit requires a first_frame or source_video asset")
    action = "create" if endpoint == "generate" else "animate"
    allowed = action_media_roles(entry, action, media="video")
    extra = roles - allowed
    if extra:
        return _unsupported(model_id, f"media role {sorted(extra)[0]}")
    return None


def reject_request_media(entry: Any, req: Any, *, media: str, api_action: str) -> MediaRoleRejection | None:
    if media == "image" and api_action in ("generate", "edit"):
        return reject_image_roles(entry, req, endpoint=api_action)
    if media == "video" and api_action in ("generate", "edit"):
        return reject_video_roles(entry, req, endpoint=api_action)
    return None
