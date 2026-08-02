"""
DanQingVideoEngine — IVideoEngine 实现。

MLX 操作在 TaskScheduler 线程同步执行（全局串行队列保证互斥）。
与 DanQingImageEngine 保持一致的架构模式。
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar, List, Any

from backend.core.contracts import (
    EngineResult, ExecutionContext,
    VideoAvatarRequest, VideoAvatarScriptRequest, VideoEditRequest,
    VideoGenerationRequest, VideoLongGenerationRequest, VideoUpscaleRequest, parse_model_version,
)
from backend.core.media_interfaces import IVideoEngine
from backend.core.interfaces import IPathResolver
from backend.core.i18n import t
from backend.engine.group_utils import resolve_asset_group_id
from .cache import ModelCache
from .lineage import resolve_lineage, video_edit_relation_type
from .progress_bridge import make_pipeline_progress_callback
from backend.long_video.validate import (
    LongVideoValidationError,
    validate_long_video_request,
)
from .sessions.engine_dispatch import (
    dispatch_long_video,
    dispatch_video_avatar,
    dispatch_video_create,
    dispatch_video_edit,
    dispatch_video_upscale,
)
from backend.engine.inference.optimization_plan import inference_metadata_for_task


class DanQingVideoEngine(IVideoEngine):
    media_type: ClassVar[str] = "video"
    engine_id: ClassVar[str] = "danqing-video"

    def __init__(
        self,
        path_resolver: IPathResolver,
        registry: Any,
        runtimes: dict[str, RuntimeContext],
        model_cache: ModelCache | None = None,
    ):
        self._paths = path_resolver
        self._registry = registry
        self._runtimes = runtimes
        self._cache = model_cache

    def is_available(self) -> bool:
        return len(self._runtimes) > 0

    def is_model_ready(self, model_name: str, version: str = "") -> bool:
        m, v = parse_model_version(model_name) if ":" in model_name else (model_name, version)
        try:
            entry = self._registry.require(m)
            backends = entry.backends
            for b in backends:
                if b in self._runtimes:
                    return True
        except Exception:
            pass
        return False

    def get_supported_models(self) -> List[str]:
        return [mid for mid, entry in self._registry.entries.items() if entry.media == "video"]

    def supports(self, model_id: str, action: str) -> bool:
        e = self._registry.get(model_id)
        if not e or e.media != "video":
            return False
        return action in e.actions

    def _resolve_runtime(self, model_id: str) -> RuntimeContext:
        entry = self._registry.require(model_id)
        backends = entry.backends
        for b in backends:
            if b in self._runtimes:
                return self._runtimes[b]
        raise RuntimeError(f"No available backend for model {model_id} (backends: {backends})")

    def _dispatch_kwargs(self, runtime: RuntimeContext, ctx: ExecutionContext) -> dict[str, Any]:
        return {
            "runtime": runtime,
            "registry": self._registry,
            "asset_store": ctx.asset_store,
            "model_cache": self._cache,
            "project_root": self._paths.get_project_root(),
        }

    async def generate(self, request: VideoGenerationRequest,
                       ctx: ExecutionContext) -> EngineResult:
        import asyncio
        if not self.supports(request.model, "generate"):
            mid = request.model.split(":", 1)[0]
            raise RuntimeError(
                f"Model {mid!r} does not support text-to-video (create) for this engine; "
                "see config/models_registry.json actions."
            )
        runtime = self._resolve_runtime(request.model)
        on_progress = make_pipeline_progress_callback(ctx)

        result = await asyncio.to_thread(
            dispatch_video_create,
            **self._dispatch_kwargs(runtime, ctx),
            request=request,
            exec_ctx=ctx,
            on_progress=on_progress,
            on_log=ctx.on_log,
        )
        if result is None:
            return EngineResult(primary_asset_id="", metadata={"status": "cancelled"})

        output_path, metadata = result
        parent_id, relation = resolve_lineage(request.metadata)
        group_id = resolve_asset_group_id(request.metadata, ctx.asset_store)
        aid = ctx.asset_store.create_from_file(
            Path(output_path), kind="video", mime_type="video/mp4",
            source_task_id=ctx.task_id, metadata=metadata, source_action="create",
            parent_asset_id=parent_id, relation_type=relation,
            group_id=group_id,
        )
        return EngineResult(
            primary_asset_id=aid,
            asset_ids=[aid],
            output_paths=[output_path],
            metadata=inference_metadata_for_task(metadata),
        )

    async def generate_long(
        self,
        request: VideoLongGenerationRequest,
        ctx: ExecutionContext,
        *,
        image_engine: Any,
    ) -> EngineResult:
        import asyncio

        spec = request.long_video
        phase = (request.metadata or {}).get("long_video_phase") or ""
        try:
            validate_long_video_request(
                request, video_engine=self, image_engine=image_engine
            )
        except LongVideoValidationError as exc:
            raise RuntimeError(exc.message) from exc
        video_runtime = self._resolve_runtime(request.model)
        if spec.strategy == "segmented_i2v" and phase != "assemble_only":
            kf_model = (spec.keyframe_model or "").strip()
            image_runtime = image_engine._resolve_runtime(kf_model)
        else:
            image_runtime = video_runtime

        on_progress = make_pipeline_progress_callback(ctx)
        result = await asyncio.to_thread(
            dispatch_long_video,
            image_runtime=image_runtime,
            video_runtime=video_runtime,
            registry=self._registry,
            asset_store=ctx.asset_store,
            model_cache=self._cache,
            project_root=self._paths.get_project_root(),
            request=request,
            exec_ctx=ctx,
            on_progress=on_progress,
            on_log=ctx.on_log,
        )
        if result is None:
            return EngineResult(primary_asset_id="", metadata={"status": "cancelled"})

        output_path, metadata = result
        parent_id, relation = resolve_lineage(request.metadata)
        group_id = resolve_asset_group_id(request.metadata, ctx.asset_store)
        aid = ctx.asset_store.create_from_file(
            Path(output_path), kind="video", mime_type="video/mp4",
            source_task_id=ctx.task_id, metadata=metadata, source_action="long_video",
            parent_asset_id=parent_id, relation_type=relation,
            group_id=group_id,
        )
        return EngineResult(
            primary_asset_id=aid,
            asset_ids=[aid],
            output_paths=[output_path],
            metadata=inference_metadata_for_task(metadata),
        )

    async def edit(self, request: VideoEditRequest, ctx: ExecutionContext) -> EngineResult:
        import asyncio
        if not self.supports(request.model, "edit"):
            mid = request.model.split(":", 1)[0]
            raise RuntimeError(
                f"Model {mid!r} does not support video edit (animate) for this engine; "
                "see config/models_registry.json actions."
            )
        runtime = self._resolve_runtime(request.model)
        on_progress = make_pipeline_progress_callback(ctx)

        result = await asyncio.to_thread(
            dispatch_video_edit,
            **self._dispatch_kwargs(runtime, ctx),
            request=request,
            exec_ctx=ctx,
            on_progress=on_progress,
            on_log=ctx.on_log,
        )
        if result is None:
            return EngineResult(primary_asset_id="", metadata={"status": "cancelled"})

        output_path, metadata = result
        parent_id, relation = resolve_lineage(
            request.metadata,
            parent_asset_id=request.source_asset_id,
            relation_type=video_edit_relation_type(request.operation),
        )
        group_id = resolve_asset_group_id(request.metadata, ctx.asset_store)
        aid = ctx.asset_store.create_from_file(
            Path(output_path), kind="video", mime_type="video/mp4",
            source_task_id=ctx.task_id, metadata=metadata, source_action="animate",
            parent_asset_id=parent_id, relation_type=relation,
            group_id=group_id,
        )
        return EngineResult(
            primary_asset_id=aid,
            asset_ids=[aid],
            output_paths=[output_path],
            metadata=inference_metadata_for_task(metadata),
        )

    async def avatar(self, request: VideoAvatarRequest, ctx: ExecutionContext) -> EngineResult:
        import asyncio
        if not self.supports(request.model, "avatar"):
            mid = request.model.split(":", 1)[0]
            raise RuntimeError(
                f"Model {mid!r} does not support video avatar for this engine; "
                "see config/models_registry.json actions."
            )
        runtime = self._resolve_runtime(request.model)
        on_progress = make_pipeline_progress_callback(ctx)

        result = await asyncio.to_thread(
            dispatch_video_avatar,
            **self._dispatch_kwargs(runtime, ctx),
            request=request,
            exec_ctx=ctx,
            on_progress=on_progress,
            on_log=ctx.on_log,
        )
        if result is None:
            return EngineResult(primary_asset_id="", metadata={"status": "cancelled"})

        output_path, metadata = result
        parent_id, relation = resolve_lineage(
            request.metadata,
            parent_asset_id=request.reference_asset_id,
            relation_type="avatar",
        )
        group_id = resolve_asset_group_id(request.metadata, ctx.asset_store)
        aid = ctx.asset_store.create_from_file(
            Path(output_path), kind="video", mime_type="video/mp4",
            source_task_id=ctx.task_id, metadata=metadata, source_action="avatar",
            parent_asset_id=parent_id, relation_type=relation,
            group_id=group_id,
        )
        return EngineResult(primary_asset_id=aid, asset_ids=[aid], output_paths=[output_path])

    async def avatar_script(
        self, request: VideoAvatarScriptRequest, ctx: ExecutionContext
    ) -> EngineResult:
        """Phase 1: fail-loud placeholder for script-to-avatar workflow.

        In phase 2 this will synthesize audio via a TTS family and then delegate
        to :meth:`avatar` with the generated audio asset.
        """
        if not self.supports(request.model, "avatar_script"):
            mid = request.model.split(":", 1)[0]
            raise RuntimeError(
                f"Model {mid!r} does not support video avatar script for this engine; "
                "see config/models_registry.json actions."
            )
        raise RuntimeError(
            t("avatar.ttsBackendNotImplemented", ctx.locale)
        )

    async def upscale(self, request: VideoUpscaleRequest, ctx: ExecutionContext) -> EngineResult:
        if not self.supports(request.model, "upscale"):
            mid = request.model.split(":", 1)[0]
            raise RuntimeError(
                f"Model {mid!r} does not support video upscale for this engine; "
                "see config/models_registry.json actions."
            )
        runtime = self._resolve_runtime(request.model)
        import asyncio

        on_progress = make_pipeline_progress_callback(ctx)

        result = await asyncio.to_thread(
            dispatch_video_upscale,
            **self._dispatch_kwargs(runtime, ctx),
            request=request,
            exec_ctx=ctx,
            on_progress=on_progress,
            on_log=ctx.on_log,
        )
        if result is None:
            return EngineResult(primary_asset_id="", metadata={"status": "cancelled"})

        output_path, metadata = result
        parent_id, relation = resolve_lineage(
            request.metadata,
            parent_asset_id=request.source_asset_id,
            relation_type="upscale",
        )
        group_id = resolve_asset_group_id(request.metadata, ctx.asset_store)
        aid = ctx.asset_store.create_from_file(
            Path(output_path), kind="video", mime_type="video/mp4",
            source_task_id=ctx.task_id, metadata=metadata, source_action="upscale",
            parent_asset_id=parent_id, relation_type=relation,
            group_id=group_id,
        )
        return EngineResult(primary_asset_id=aid, asset_ids=[aid], output_paths=[output_path])

    async def cancel(self, task_id: str) -> bool:
        return True
