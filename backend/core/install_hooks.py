"""Registry-driven post-download install hooks (version block in models_registry.json)."""
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# hook ``type`` -> ``module.path:function_name``
_HOOK_RUNNERS: dict[str, str] = {
    "ltx25_ingest": "backend.engine.families.ltx25.ingest_mlx:run_ltx25_ingest_hook",
}


def install_hooks_from_version(ver_config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return normalized hook entries from a registry version dict."""
    if not ver_config:
        return []
    raw = ver_config.get("install_hooks")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [{"type": raw.strip()}]
    if not isinstance(raw, list):
        raise ValueError(
            "install_hooks must be a list of hook objects or type strings "
            f"(got {type(raw).__name__})"
        )
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            t = item.strip()
            if t:
                out.append({"type": t})
            continue
        if isinstance(item, dict):
            t = str(item.get("type") or "").strip()
            if not t:
                raise ValueError("install_hooks entry requires non-empty 'type'")
            entry = dict(item)
            entry["type"] = t
            out.append(entry)
            continue
        raise ValueError(
            f"install_hooks entries must be strings or objects (got {type(item).__name__})"
        )
    return out


def _resolve_runner(hook_type: str) -> Callable[..., None]:
    target = _HOOK_RUNNERS.get(hook_type)
    if not target:
        known = ", ".join(sorted(_HOOK_RUNNERS))
        raise RuntimeError(
            f"Unknown install_hooks type {hook_type!r} (known: {known})"
        )
    mod_name, func_name = target.rsplit(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name, None)
    if fn is None or not callable(fn):
        raise RuntimeError(f"Install hook runner not found: {target}")
    return fn


def run_install_hooks(
    *,
    model_name: str,
    version_key: str | None,
    ver_config: dict[str, Any] | None,
    bundle_root: Path,
) -> None:
    """Run all hooks declared on the version block (fail loud on error)."""
    hooks = install_hooks_from_version(ver_config)
    if not hooks:
        return
    root = Path(bundle_root)
    if not root.is_dir():
        raise RuntimeError(
            f"install_hooks: bundle root not found for {model_name!r}: {root}"
        )
    for spec in hooks:
        hook_type = str(spec["type"])
        logger.info(
            "Running install hook %s for %s:%s at %s",
            hook_type,
            model_name,
            version_key or "default",
            root,
        )
        runner = _resolve_runner(hook_type)
        runner(
            bundle_root=root,
            model_name=model_name,
            version_key=version_key,
            hook_spec=spec,
        )
