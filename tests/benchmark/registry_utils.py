"""Registry + workspace helpers for image eval benchmark."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from backend.core.version_keys import is_quantized_registry_version

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_REGISTRY_PATH = _REPO_ROOT / "default_config" / "models_registry.json"
_CONTROL_POINTER = Path.home() / ".danmo-make" / "workspace.pointer.json"
_LEGACY_POINTER = _REPO_ROOT / "default_config" / "workspace.pointer.json"

EDIT_ACTIONS = frozenset({"rewrite", "retouch", "extend"})

_MIN_SAFETENSORS_BYTES = 1024

_EVAL_FULL_VERSION_ORDER: tuple[str, ...] = ("fp16", "bf16", "fp8", "mlx-bf16")
_EVAL_QUANT_VERSION_ORDER: tuple[str, ...] = ("mlx-q4", "int4", "mlx-q8", "int8")


def repo_root() -> Path:
    return _REPO_ROOT


def resolve_benchmark_data_root() -> Path:
    """Control-plane / legacy pointer → workspace root, else repo root."""
    for pointer in (_CONTROL_POINTER, _LEGACY_POINTER):
        if not pointer.is_file():
            continue
        try:
            data = json.loads(pointer.read_text(encoding="utf-8"))
            custom = str(data.get("custom_workspace_dir") or "").strip()
            if custom:
                root = Path(custom).expanduser().resolve()
                if root.is_dir():
                    return root
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return _REPO_ROOT


def load_registry() -> dict[str, Any]:
    from backend.catalog.loader import expand_catalog_document, load_catalog_json

    return expand_catalog_document(load_catalog_json(_REGISTRY_PATH))


def _profile_engine(reg: dict[str, Any], profile_id: str) -> str:
    profiles = reg.get("profiles") or reg.get("ui_profiles") or {}
    profile = profiles.get(profile_id) or {}
    return str(profile.get("engine") or "").strip()


def _model_engine(reg: dict[str, Any], spec: dict[str, Any]) -> str:
    explicit = str(spec.get("engine") or "").strip()
    if explicit:
        return explicit
    profile_id = str(spec.get("profile") or "").strip()
    if profile_id:
        return _profile_engine(reg, profile_id)
    return ""


def param_default(spec: dict[str, Any], name: str, fallback: float | int | str) -> float | int | str:
    params = spec.get("parameters") or {}
    node = params.get(name) or {}
    if isinstance(node, dict) and "default" in node:
        return node["default"]
    return fallback


def _version_bundle_dir(
    vinfo: dict[str, Any],
    *,
    data_root: Path | None = None,
) -> Path | None:
    rel = str(vinfo.get("local_path") or "").strip()
    if not rel:
        return None
    root = data_root or resolve_benchmark_data_root()
    path = (root / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
    return path if path.is_dir() else None


def _version_sort_key(version_key: str, *, quant_pool: bool) -> tuple[int, str]:
    order = _EVAL_QUANT_VERSION_ORDER if quant_pool else _EVAL_FULL_VERSION_ORDER
    try:
        return (order.index(version_key), version_key)
    except ValueError:
        return (len(order), version_key)


def _installed_version_rows(
    model_id: str,
    spec: dict[str, Any],
    *,
    data_root: Path | None = None,
) -> list[tuple[str, dict[str, Any], Path]]:
    versions = spec.get("versions") or {}
    if not isinstance(versions, dict):
        return []
    root = data_root or resolve_benchmark_data_root()
    rows: list[tuple[str, dict[str, Any], Path]] = []
    for vk, vinfo in versions.items():
        if not isinstance(vinfo, dict):
            continue
        path = _version_bundle_dir(vinfo, data_root=root)
        if path is None:
            continue
        incomplete = _incomplete_download_reason(path)
        if incomplete:
            continue
        rows.append((str(vk), vinfo, path))
    return rows


def resolve_eval_version_key(
    model_id: str,
    *,
    reg: dict[str, Any] | None = None,
    data_root: Path | None = None,
) -> str | None:
    """Pick eval version: prefer installed full weights, else installed quantized."""
    reg = reg or load_registry()
    spec = (reg.get("models") or {}).get(model_id)
    if not isinstance(spec, dict):
        return None
    installed = _installed_version_rows(model_id, spec, data_root=data_root)
    if not installed:
        return None

    full_rows = [
        row for row in installed if not is_quantized_registry_version(row[0], row[1])
    ]
    quant_rows = [
        row for row in installed if is_quantized_registry_version(row[0], row[1])
    ]
    pool = full_rows if full_rows else quant_rows
    quant_pool = not full_rows
    best = min(pool, key=lambda row: _version_sort_key(row[0], quant_pool=quant_pool))
    return best[0]


def resolve_eval_bundle_dir(
    model_id: str,
    *,
    reg: dict[str, Any] | None = None,
    data_root: Path | None = None,
) -> Path | None:
    """On-disk bundle for eval: full when installed, quantized when that is all that exists."""
    reg = reg or load_registry()
    spec = (reg.get("models") or {}).get(model_id)
    if not isinstance(spec, dict):
        return None
    version_key = resolve_eval_version_key(model_id, reg=reg, data_root=data_root)
    if not version_key:
        return None
    vinfo = (spec.get("versions") or {}).get(version_key)
    if not isinstance(vinfo, dict):
        return None
    return _version_bundle_dir(vinfo, data_root=data_root)


def resolve_benchmark_model_bundle(model_id: str) -> Path | None:
    """Installed bundle path for engine_unit / integration tests (same rule as eval)."""
    return resolve_eval_bundle_dir(model_id)


def resolve_default_bundle_dir(model_id: str, *, reg: dict[str, Any] | None = None) -> Path | None:
    return resolve_eval_bundle_dir(model_id, reg=reg)


def _incomplete_download_reason(bundle_root: Path) -> str | None:
    for path in bundle_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.endswith(".incomplete"):
            return f"incomplete download ({path.name})"
        if path.suffix == ".safetensors" and path.stat().st_size < _MIN_SAFETENSORS_BYTES:
            return f"empty weight file ({path.name})"
    return None


def _bundle_manifest_module():
    import importlib.util
    import sys

    name = "dq_bundle_manifest"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached

    path = _REPO_ROOT / "backend" / "core" / "bundle_manifest.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load bundle manifest module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def bundle_ready(model_id: str, *, reg: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Return (ready, skip_reason). Uses FamilyBundleContract when family is known."""
    path = resolve_eval_bundle_dir(model_id, reg=reg)
    if path is None or not path.is_dir():
        return False, "missing default bundle"

    incomplete = _incomplete_download_reason(path)
    if incomplete:
        return False, incomplete

    reg = reg or load_registry()
    spec = (reg.get("models") or {}).get(model_id)
    if not isinstance(spec, dict):
        return False, "unknown model"
    family = str(spec.get("family") or "").strip()
    if not family:
        return True, ""

    try:
        manifest = _bundle_manifest_module()
        manifest.assert_bundle_ready_for_family(path, family=family, model_id=model_id)
    except RuntimeError as exc:
        return False, str(exc)
    return True, ""


def bundle_installed(model_id: str, *, reg: dict[str, Any] | None = None) -> bool:
    ready, _ = bundle_ready(model_id, reg=reg)
    return ready


def iter_image_eval_models(*, reg: dict[str, Any] | None = None) -> Iterator[tuple[str, dict[str, Any]]]:
    reg = reg or load_registry()
    models = reg.get("models") or {}
    for model_id, spec in sorted(models.items()):
        if not isinstance(spec, dict):
            continue
        if spec.get("media") != "image":
            continue
        actions = spec.get("actions") or {}
        if not isinstance(actions, dict) or not actions:
            continue
        engine = _model_engine(reg, spec)
        if engine and engine != "danqing-image":
            continue
        yield model_id, spec
