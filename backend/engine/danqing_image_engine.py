"""
DanQingImageEngine — IImageEngine 实现。

MLX 操作在 TaskScheduler 线程同步执行（全局串行队列保证互斥）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, List

from backend.core.contracts import (
    EngineResult, ExecutionContext, ImageEditRequest,
    ImageGenerationRequest, ImageUpscaleRequest, parse_model_version,
)
from backend.core.media_interfaces import IImageEngine
from backend.core.interfaces import IPathResolver
from backend.engine.group_utils import resolve_asset_group_id
from .cache import ModelCache
from .lineage import image_edit_relation_type, resolve_lineage
from .progress_bridge import make_pipeline_progress_callback
from .runtime._base import RuntimeContext
from .sessions.engine_dispatch import (
    dispatch_image_create,
    dispatch_image_edit,
    dispatch_image_upscale,
)
from backend.engine.inference.optimization_plan import inference_metadata_for_task


class DanQingImageEngine(IImageEngine):
    media_type: ClassVar[str] = "image"
    engine_id: ClassVar[str] = "danqing-image"

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
        return [mid for mid, entry in self._registry.entries.items() if entry.media == "image"]

    def supports(self, model_id: str, action: str) -> bool:
        e = self._registry.get(model_id)
        if not e or e.media != "image":
            return False
        return action in e.actions

    def _reject_media(self, request: Any, endpoint: str) -> None:
        from backend.core.media_roles import reject_image_roles

        entry = self._registry.get(request.model)
        if entry is None:
            return
        rejected = reject_image_roles(entry, request, endpoint=endpoint)
        if rejected is not None:
            raise RuntimeError(rejected.message)

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

    async def generate(self, request: ImageGenerationRequest,
                       ctx: ExecutionContext) -> EngineResult:
        import asyncio
        if not self.supports(request.model, "generate"):
            mid = request.model.split(":", 1)[0]
            raise RuntimeError(
                f"Model {mid!r} does not support text-to-image (create) for this engine; "
                "see config/models_registry.json actions."
            )
        self._reject_media(request, "generate")
        runtime = self._resolve_runtime(request.model)

        on_progress = make_pipeline_progress_callback(ctx)

        result = await asyncio.to_thread(
            dispatch_image_create,
            **self._dispatch_kwargs(runtime, ctx),
            request=request,
            exec_ctx=ctx,
            on_progress=on_progress,
            on_log=ctx.on_log,
        )
        if result is None:
            return EngineResult(primary_asset_id="", metadata={"status": "cancelled"})

        parent_id, relation = resolve_lineage(request.metadata)
        group_id = resolve_asset_group_id(request.metadata, ctx.asset_store)
        if isinstance(result, list):
            asset_ids: list[str] = []
            output_paths: list[str] = []
            for output_path, metadata in result:
                aid = ctx.asset_store.create_from_file(
                    Path(output_path), kind="image", mime_type="image/png",
                    source_task_id=ctx.task_id, metadata=metadata, source_action="create",
                    parent_asset_id=parent_id, relation_type=relation,
                    group_id=group_id,
                )
                asset_ids.append(aid)
                output_paths.append(output_path)
            return EngineResult(
                primary_asset_id=asset_ids[0] if asset_ids else "",
                asset_ids=asset_ids,
                output_paths=output_paths,
                metadata=inference_metadata_for_task(result[0][1] if result else None),
            )

        output_path, metadata = result
        aid = ctx.asset_store.create_from_file(
            Path(output_path), kind="image", mime_type="image/png",
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

    async def edit(self, request: ImageEditRequest, ctx: ExecutionContext) -> EngineResult:
        import asyncio
        if not self.supports(request.model, "edit"):
            mid = request.model.split(":", 1)[0]
            raise RuntimeError(
                f"Model {mid!r} does not support image edit (rewrite/retouch/extend) for this engine; "
                "see config/models_registry.json actions."
            )
        self._reject_media(request, "edit")
        runtime = self._resolve_runtime(request.model)
        on_progress = make_pipeline_progress_callback(ctx)

        result = await asyncio.to_thread(
            dispatch_image_edit,
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
            relation_type=image_edit_relation_type(
                request.operation, rewrite_mode=request.rewrite_mode or ""
            ),
        )
        group_id = resolve_asset_group_id(request.metadata, ctx.asset_store)
        aid = ctx.asset_store.create_from_file(
            Path(output_path), kind="image", mime_type="image/png",
            source_task_id=ctx.task_id, metadata=metadata,
            source_action=request.operation,
            parent_asset_id=parent_id, relation_type=relation,
            group_id=group_id,
        )
        return EngineResult(
            primary_asset_id=aid,
            asset_ids=[aid],
            output_paths=[output_path],
            metadata=inference_metadata_for_task(metadata),
        )

    async def upscale(self, request: ImageUpscaleRequest, ctx: ExecutionContext) -> EngineResult:
        if not self.supports(request.model, "upscale"):
            mid = request.model.split(":", 1)[0]
            raise RuntimeError(
                f"Model {mid!r} does not support upscale for this engine; "
                "see config/models_registry.json actions."
            )
        runtime = self._resolve_runtime(request.model)
        import asyncio

        on_progress = make_pipeline_progress_callback(ctx)

        result = await asyncio.to_thread(
            dispatch_image_upscale,
            **self._dispatch_kwargs(runtime, ctx),
            request=request,
            exec_ctx=ctx,
            on_progress=on_progress,
            on_log=ctx.on_log,
        )
        if result is None:
            return EngineResult(primary_asset_id="", metadata={"status": "cancelled"})

        output_path, metadata = result
        group_id = resolve_asset_group_id(request.metadata, ctx.asset_store)
        aid = ctx.asset_store.create_from_file(
            Path(output_path), kind="image", mime_type="image/png",
            source_task_id=ctx.task_id, metadata=metadata, source_action="upscale",
            parent_asset_id=request.source_asset_id, relation_type="upscale",
            group_id=group_id,
        )
        return EngineResult(primary_asset_id=aid, asset_ids=[aid], output_paths=[output_path])

    async def cancel(self, task_id: str) -> bool:
        return True
