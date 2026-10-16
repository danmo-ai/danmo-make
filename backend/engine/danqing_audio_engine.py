"""DanQing 音频引擎 — 音乐生成管线接入，支持 ACE-Step DiT + VAE。"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List

from backend.core.contracts import (
    AudioEditRequest,
    AudioGenerationRequest,
    EngineResult,
    ExecutionContext,
    LogEvent,
    parse_model_version,
)
from backend.core.interfaces import IPathResolver
from backend.core.media_interfaces import IAudioEngine
from backend.core.model_registry import ModelRegistry
from backend.core.version_keys import resolve_registry_version_key


def _version_config(entry_raw: dict[str, Any], version_key: str) -> dict[str, Any]:
    versions = entry_raw.get("versions")
    if not isinstance(versions, dict) or not versions:
        return {}
    resolved = resolve_registry_version_key(versions, version_key or None)
    if resolved and resolved in versions:
        cfg = versions[resolved]
        return cfg if isinstance(cfg, dict) else {}
    for _k, cfg in versions.items():
        if isinstance(cfg, dict) and cfg.get("default"):
            return cfg
    first = next(iter(versions.values()))
    return first if isinstance(first, dict) else {}


def _resolve_models_subpath(path_resolver: IPathResolver, local_path: str) -> Path:
    if local_path.startswith("models/"):
        rel = local_path[len("models/"):]
        return path_resolver.get_models_dir() / rel
    return Path(local_path).expanduser()


def _dir_ready(p: Path) -> bool:
    return p.is_dir() and any(p.iterdir())


class DanQingAudioEngine(IAudioEngine):
    media_type = "audio"
    engine_id = "danqing-audio"

    def __init__(
        self,
        path_resolver: IPathResolver,
        registry: ModelRegistry,
        runtimes: dict[str, Any],
        model_cache: Any = None,
    ):
        self._paths = path_resolver
        self._registry = registry
        self._runtimes = runtimes
        self._cache = model_cache

    def is_available(self) -> bool:
        return True

    def is_model_ready(self, model_name: str, version: str = "") -> bool:
        m, v = parse_model_version(model_name) if ":" in model_name else (model_name, version)
        entry = self._registry.get(m)
        if not entry or entry.media != "audio":
            return False
        raw = entry.raw
        if isinstance(raw, dict) and raw.get("stub_no_download"):
            return True
        ver_cfg = _version_config(raw, v)
        from backend.core.bundle_repos import bundle_local_paths, version_primary_local_path

        try:
            lp = version_primary_local_path(ver_cfg)
        except ValueError:
            return False
        bundle_path = _resolve_models_subpath(self._paths, str(lp))
        if not _dir_ready(bundle_path):
            return False
        for sub_lp in bundle_local_paths(ver_cfg):
            if sub_lp == lp:
                continue
            sub_path = _resolve_models_subpath(self._paths, sub_lp)
            if not _dir_ready(sub_path):
                return False
        return True

    def get_supported_models(self) -> List[str]:
        return [mid for mid, e in self._registry.all().items() if e.media == "audio"]

    def supports(self, model_id: str, action: str) -> bool:
        mid, _ = parse_model_version(model_id)
        e = self._registry.get(mid)
        if not e or e.media != "audio":
            return False
        return action in e.actions

    def _resolve_runtime(self, entry: Any) -> Any:
        """Select the appropriate RuntimeContext for this model (fail loud if none match)."""
        import os

        forced = (os.environ.get("DANQING_FORCE_AUDIO_BACKEND") or "").strip().lower()
        if forced:
            rt = self._runtimes.get(forced)
            if rt is None:
                raise RuntimeError(
                    f"DANQING_FORCE_AUDIO_BACKEND={forced!r} but runtime not available "
                    f"(active: {list(self._runtimes.keys())!r})"
                )
            return rt
        backends = getattr(entry, "backends", None) or entry.raw.get("backends", ["mlx"])
        for b in backends:
            rt = self._runtimes.get(b)
            if rt is not None:
                return rt
        raise RuntimeError(
            f"No runtime available for model backends {list(backends)!r}; "
            f"active runtimes: {list(self._runtimes.keys())!r}"
        )

    async def generate(
        self, request: AudioGenerationRequest, ctx: ExecutionContext
    ) -> EngineResult:
        mid, _ver = parse_model_version(request.model)
        if not self.supports(mid, "create_music"):
            raise RuntimeError(
                f"Model {mid!r} does not support text-to-music (create); "
                "see config/models_registry.json actions."
            )
        entry = self._registry.get(mid)
        if entry is None:
            raise RuntimeError(f"Model {request.model!r} not found in registry")

        is_stub = entry.raw.get("stub_no_download", False)
        if is_stub:
            ctx.on_log(LogEvent(level="error", message="Audio stub model has no backend"))
            raise RuntimeError(
                "Audio generation is not implemented for the stub model. "
                "Use a real audio model (e.g. ace-step-xl-sft)."
            )

        ctx.on_log(LogEvent(level="info", message=f"Initializing audio pipeline for {request.model}"))

        runtime = self._resolve_runtime(entry)
        project_root = (
            self._paths.get_project_root() if hasattr(self._paths, "get_project_root") else None
        )

        from backend.engine.sessions.engine_dispatch import dispatch_audio_create

        return await asyncio.to_thread(
            dispatch_audio_create,
            runtime=runtime,
            registry=self._registry,
            asset_store=ctx.asset_store,
            model_cache=self._cache,
            project_root=project_root,
            request=request,
            exec_ctx=ctx,
        )

    async def edit(self, request: AudioEditRequest, ctx: ExecutionContext) -> EngineResult:
        # Registry cover/repaint map to API action "edit" (see api_action_frozenset).
        if not self.supports(request.model, "edit"):
            raise RuntimeError(
                f"Model {request.model!r} does not declare audio edit "
                f"(registry cover/repaint); see config/models_registry.json."
            )
        entry = self._registry.get(parse_model_version(request.model)[0])
        if entry is None:
            raise RuntimeError(f"Model {request.model!r} not found in registry")

        ctx.on_log(
            LogEvent(
                level="info",
                message=f"Initializing audio edit pipeline ({request.operation}) for {request.model}",
            )
        )
        runtime = self._resolve_runtime(entry)
        project_root = (
            self._paths.get_project_root() if hasattr(self._paths, "get_project_root") else None
        )

        from backend.engine.sessions.engine_dispatch import dispatch_audio_edit

        return await asyncio.to_thread(
            dispatch_audio_edit,
            runtime=runtime,
            registry=self._registry,
            asset_store=ctx.asset_store,
            model_cache=self._cache,
            project_root=project_root,
            request=request,
            exec_ctx=ctx,
        )

    async def cancel(self, task_id: str) -> bool:
        return False
