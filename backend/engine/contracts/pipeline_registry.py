"""Shared registry / path helpers for ImagePipeline and VideoPipeline.

Keeps bundle resolution and parameter defaults identical across media pipelines.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.core.bundle_repos import version_primary_local_path
from backend.core.version_keys import resolve_registry_version_key


def resolve_project_path(project_root: Path, local_path: str) -> Path:
    p = Path(local_path)
    if p.is_absolute():
        return p
    return (project_root / local_path).resolve()


def require_entry_family(entry: Any, *, model_id: str = "") -> str:
    """Return registry ``family`` or fail loud (no silent defaults like ``flux1`` / ``ltx``)."""
    family = getattr(entry, "family", None)
    if family is None or not str(family).strip():
        label = model_id or getattr(entry, "id", None) or "?"
        raise RuntimeError(
            f"Registry model {label!r} missing required 'family' field."
        )
    return str(family)


def registry_scalar_default(entry: Any, key: str, fallback: Any) -> Any:
    spec = (entry.parameters or {}).get(key)
    if spec is None:
        return fallback
    if isinstance(spec, dict):
        return spec.get("default", fallback)
    return spec  # direct value (list / int / str etc.), not wrapped in dict


def resolve_version_block(entry: Any, version_key: str | None) -> dict | None:
    raw = getattr(entry, "raw", {}) or {}
    versions = raw.get("versions") or {}
    resolved = resolve_registry_version_key(versions, version_key)
    if resolved and resolved in versions and isinstance(versions[resolved], dict):
        return versions[resolved]
    for vinfo in versions.values():
        if isinstance(vinfo, dict) and vinfo.get("default"):
            return vinfo
    return None


def _bundle_path_for_block(project_root: Path, block: dict | None) -> Path | None:
    if not block:
        return None
    lp = (block.get("local_path") or "").strip()
    if not lp:
        try:
            lp = version_primary_local_path(block)
        except ValueError:
            return None
    path = resolve_project_path(project_root, lp)
    return path if path.exists() else None


def _installed_version_rank(version_key: str, block: dict) -> tuple[int, int, str]:
    """Smaller quant first, then the registry default, then key name."""
    quant = block.get("quantization")
    bits = quant.get("bits") if isinstance(quant, dict) else None
    if not isinstance(bits, int):
        key = version_key.lower()
        if "q4" in key or "int4" in key:
            bits = 4
        elif "q8" in key or "int8" in key or "fp8" in key:
            bits = 8
        else:
            bits = 16
    return (bits, 0 if block.get("default") else 1, version_key)


def local_bundle_root(project_root: Path, entry: Any, version_key: str | None) -> Path | None:
    """Directory for the requested version, or the registry default when version is omitted.

    Does not substitute a different installed quant. Callers that accept a bare
    model id and must load whichever bundle is on disk use ``resolve_bundle_version``.
    """
    return _bundle_path_for_block(project_root, resolve_version_block(entry, version_key))


def resolve_bundle_version(
    project_root: Path,
    entry: Any,
    version_key: str | None,
) -> tuple[str | None, Path | None]:
    """Resolve the on-disk bundle for a model id.

    An explicit ``version_key`` stays on that version. A bare model id uses the
    registry default when that directory exists, otherwise the installed version
    (so ``ltx-2.5-distilled`` loads ``mlx-q8`` when only the 8-bit bundle is present).
    """
    raw = getattr(entry, "raw", {}) or {}
    versions = raw.get("versions") or {}
    explicit = bool(str(version_key or "").strip())
    resolved_key = resolve_registry_version_key(versions, version_key) if versions else version_key
    key_out = str(resolved_key) if resolved_key else None
    path = local_bundle_root(project_root, entry, version_key)
    if path is not None or explicit:
        return key_out, path

    installed: list[tuple[tuple[int, int, str], str, Path]] = []
    for vk, vinfo in versions.items():
        if not isinstance(vinfo, dict):
            continue
        found = _bundle_path_for_block(project_root, vinfo)
        if found is None:
            continue
        installed.append((_installed_version_rank(str(vk), vinfo), str(vk), found))
    if not installed:
        return key_out, None
    installed.sort(key=lambda item: item[0])
    return installed[0][1], installed[0][2]
