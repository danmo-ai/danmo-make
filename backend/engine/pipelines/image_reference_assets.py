"""Edit reference order: the source image is picture 1, then ``reference_asset_ids``."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def family_reference_cap(config: Any) -> int:
    """How many reference images a family generator accepts, including the source."""
    for key in ("hidream_edit_max_refs", "edit_max_reference_images"):
        if not hasattr(config, key):
            continue
        raw = getattr(config, key)
        try:
            count = int(raw)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count
    return 1


def ordered_edit_asset_ids(
    source_asset_id: str | None,
    reference_asset_ids: list[str] | None,
) -> list[str]:
    """Picture 1 is the source. Later ids are extras, in order, with blanks and duplicates dropped."""
    source = str(source_asset_id or "").strip()
    if not source:
        raise RuntimeError("Image edit requires source_asset_id")
    ids = [source]
    seen = {source}
    for asset_id in reference_asset_ids or []:
        value = str(asset_id or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ids.append(value)
    return ids


def assert_reference_count(count: int, max_refs: int) -> None:
    cap = max(1, int(max_refs))
    if count > cap:
        raise RuntimeError(
            f"Edit accepts at most {cap} reference images; received {count}"
        )


def require_asset_file(asset_store: Any, asset_id: str) -> Path:
    path = asset_store.get_file_path(asset_id)
    if path is None:
        raise RuntimeError(f"Reference image asset not found: {asset_id!r}")
    file_path = Path(path)
    if not file_path.exists():
        raise RuntimeError(f"Reference image asset not found: {asset_id!r}")
    return file_path


def assemble_edit_reference_paths(
    asset_store: Any,
    source_asset_id: str | None,
    reference_asset_ids: list[str] | None,
    *,
    max_refs: int,
) -> list[str]:
    """Resolve picture 1..N. Too many ids, or a missing file, raises ``RuntimeError``."""
    ids = ordered_edit_asset_ids(source_asset_id, reference_asset_ids)
    assert_reference_count(len(ids), max_refs)
    return [str(require_asset_file(asset_store, asset_id)) for asset_id in ids]
