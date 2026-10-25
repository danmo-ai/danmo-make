"""User-trained LoRA registry (workspace ``config/user_loras.json``)."""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def user_loras_path(config_dir: Path) -> Path:
    return config_dir / "user_loras.json"


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"items": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"items": []}
    items = data.get("items")
    if not isinstance(items, list):
        data["items"] = []
    return data


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def list_user_loras(config_dir: Path) -> list[dict[str, Any]]:
    return list(_load(user_loras_path(config_dir)).get("items") or [])


def register_user_lora(
    config_dir: Path,
    *,
    name: str,
    base_model: str,
    local_path: str,
    trigger_word: str = "",
    lora_rank: int = 8,
    nsfw: bool = False,
    task_id: str = "",
    source: str = "user_trained",
    lora_alpha: int | None = None,
    inference: dict[str, Any] | None = None,
    repo_id: str = "",
    remote_hub_source: str = "",
    family: str = "",
    lora_module_keys: list[str] | None = None,
    train_type: str = "lora",
) -> dict[str, Any]:
    path = user_loras_path(config_dir)
    data = _load(path)
    items: list[dict[str, Any]] = list(data.get("items") or [])
    lora_id = "user-lora-" + secrets.token_hex(6)
    entry = {
        "id": lora_id,
        "name": name.strip() or lora_id,
        "base_model": base_model,
        "local_path": local_path,
        "trigger_word": trigger_word,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha if lora_alpha is not None else lora_rank,
        "inference": inference or {},
        "family": family,
        "train_type": train_type,
        "lora_module_keys": list(lora_module_keys or []),
        "nsfw": nsfw,
        "source": source,
        "task_id": task_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if repo_id:
        entry["repo_id"] = repo_id
    if remote_hub_source:
        entry["remote_hub_source"] = remote_hub_source
    items.insert(0, entry)
    data["items"] = items
    _save(path, data)
    return entry


def get_user_lora(config_dir: Path, lora_id: str) -> dict[str, Any] | None:
    for item in list_user_loras(config_dir):
        if item.get("id") == lora_id:
            return item
    return None


def delete_user_lora(config_dir: Path, lora_id: str, *, remove_files: bool = False, project_root: Path | None = None) -> bool:
    path = user_loras_path(config_dir)
    data = _load(path)
    items: list[dict[str, Any]] = list(data.get("items") or [])
    kept: list[dict[str, Any]] = []
    removed: dict[str, Any] | None = None
    for item in items:
        if item.get("id") == lora_id:
            removed = item
        else:
            kept.append(item)
    if removed is None:
        return False
    data["items"] = kept
    _save(path, data)
    if remove_files and project_root is not None:
        local_path = str(removed.get("local_path") or "").strip()
        if local_path:
            target = project_root / local_path
            if target.is_dir():
                import shutil

                shutil.rmtree(target, ignore_errors=True)
            elif target.is_file():
                target.unlink(missing_ok=True)
    return True


def resolve_user_lora_bundle(project_root: Path, config_dir: Path, lora_id: str) -> Path | None:
    """Return directory or safetensors path for a user-trained LoRA."""
    item = get_user_lora(config_dir, lora_id)
    if item is None:
        return None
    local_path = str(item.get("local_path") or "").strip()
    if not local_path:
        return None
    if local_path.startswith("models/"):
        bundle = project_root / local_path
    else:
        bundle = project_root / local_path
    if bundle.is_dir():
        adapter = bundle / "adapter.safetensors"
        if adapter.is_file():
            return adapter
        return bundle
    return bundle if bundle.is_file() else None
