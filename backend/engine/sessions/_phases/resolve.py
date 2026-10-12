"""Phase: resolve model entry, platform, bundle paths."""

from __future__ import annotations

from typing import Any

from backend.core.contracts import parse_model_version
from backend.engine.contracts import require_entry_family, resolve_bundle_version
from backend.engine.platform.session import platform_from_runtime
from backend.engine.sessions._context import ResolvedRun


def resolve_phase(
    session: Any,
    request: Any,
    exec_ctx: Any,
    *,
    runtime_ctx: Any,
) -> ResolvedRun:
    """Registry + platform resolution — no family string branches."""
    model_key, version_key = parse_model_version(request.model)
    entry = session._registry.require(model_key)
    family_id = require_entry_family(entry, model_id=model_key)
    selected_version, bundle_root = resolve_bundle_version(
        session._project_root, entry, version_key or None
    )
    platform = platform_from_runtime(runtime_ctx)
    return ResolvedRun(
        model_id=model_key,
        version_key=selected_version,
        family_id=family_id,
        platform=platform,
        plugin=None,
        bundle_root=bundle_root,
        registry_entry=entry,
        exec_ctx=exec_ctx,
        request=request,
    )


def load_plugin_phase(
    resolved: ResolvedRun,
    *,
    project_root: Any | None = None,
    model_cache: Any | None = None,
) -> ResolvedRun:
    """Load ``FamilyPlugin`` via family registry (Phase 1+)."""
    from backend.engine.protocols.bundle import MediaBundle
    from backend.engine.registry.family_registry import build_family_plugin

    if resolved.bundle_root is None:
        raise RuntimeError(f"bundle root missing for model {resolved.model_id!r}")
    resolved.plugin = build_family_plugin(
        resolved.family_id,
        resolved.platform,
        model_id=resolved.model_id,
        bundle_root=resolved.bundle_root,
        version_key=resolved.version_key,
    )
    bundle = MediaBundle(
        family_id=resolved.family_id,
        model_id=resolved.model_id,
        root=resolved.bundle_root,
        version_key=resolved.version_key,
    )
    backbone = resolved.plugin.backbone
    should_load = _bind_plugin_backbone(
        backbone,
        registry_entry=resolved.registry_entry,
        project_root=project_root,
        model_cache=model_cache,
        bundle_root=resolved.bundle_root,
        request=resolved.request,
    )

    if should_load:
        trace = getattr(resolved.exec_ctx, "trace", None)
        if trace is not None:
            with trace.span_ctx(
                "load_backbone",
                kind="phase",
                family_id=resolved.family_id,
                model_id=resolved.model_id,
            ):
                backbone.load(bundle, resolved.platform)
                backbone.after_load(bundle)
        else:
            backbone.load(bundle, resolved.platform)
            backbone.after_load(bundle)
    return resolved


def _bind_plugin_backbone(
    backbone: Any,
    *,
    registry_entry: Any,
    project_root: Any | None,
    model_cache: Any | None,
    bundle_root: Any,
    request: Any,
) -> bool:
    bind_ctx = getattr(backbone, "bind_load_context", None)
    if not callable(bind_ctx):
        return True
    if project_root is None:
        raise RuntimeError(
            f"load_plugin_phase: project_root required for backbone {registry_entry.family!r}"
        )
    import inspect

    sig = inspect.signature(bind_ctx)
    kwargs = {
        "registry_entry": registry_entry,
        "project_root": project_root,
        "model_cache": model_cache,
        "bundle_root": bundle_root,
        "request": request,
    }
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    result = bind_ctx(**filtered)
    return result is not False
