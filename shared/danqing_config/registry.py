"""Load models_registry.json and expose flat model records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, FrozenSet

from backend.catalog.loader import expand_catalog_document
from backend.core.registry_format import api_action_frozenset, media_from_record


@dataclass(frozen=True)
class ModelRecord:
    id: str
    raw: dict[str, Any]
    media: str
    actions: FrozenSet[str]


def load_registry(registry_path: Path) -> dict[str, ModelRecord]:
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    expanded = expand_catalog_document(data)
    raw_models = expanded.get("models") or {}
    built: dict[str, ModelRecord] = {}
    for mid, raw in raw_models.items():
        if not isinstance(raw, dict):
            continue
        media = media_from_record(raw)
        acts_block = raw.get("actions") if isinstance(raw.get("actions"), dict) else {}
        actions = api_action_frozenset(acts_block, media=media)
        built[mid] = ModelRecord(id=mid, raw=raw, media=media, actions=actions)
    return built


def default_version_local_path(entry: ModelRecord) -> str:
    versions = entry.raw.get("versions") or {}
    default_ver = next(
        (v for v in versions.values() if isinstance(v, dict) and v.get("default")),
        next((v for v in versions.values() if isinstance(v, dict)), None),
    )
    if default_ver is None:
        raise RuntimeError(f"No versions defined for model {entry.id!r} in registry")
    local_path = default_ver.get("local_path")
    if not local_path:
        raise RuntimeError(f"No local_path for default version of {entry.id!r}")
    return str(local_path)
