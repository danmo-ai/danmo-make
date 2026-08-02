"""
Danmo Make — FastAPI 主入口 (v4 引擎)
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    exe_dir = Path(sys.executable).parent.resolve()
    if sys.platform == "darwin" and exe_dir.name == "MacOS" and (exe_dir.parent / "Resources").exists():
        project_root = exe_dir.parent / "Resources"
    else:
        project_root = exe_dir
else:
    project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.mcp.bridge import MakeAPIBridge
from backend.mcp.base_url import resolve_api_base_url
from backend.mcp.server import create_mcp, set_bridge

_mcp = create_mcp()
_mcp_http_app = _mcp.streamable_http_app()

from backend.core.container import register_services, get_container
from backend.core.i18n import set_locale, _load_translations
from backend.utils.path_utils import PathResolver
from backend.persistence.stores import JsonConfigStore, JsonPresetStore
from backend.persistence.asset_store import SQLiteAssetStore
from backend.persistence.canvas_session_store import CanvasSessionStore
from backend.persistence.long_video_project_store import LongVideoProjectStore
from backend.persistence.long_video_activity_store import LongVideoActivityStore
from backend.persistence.v3_task_store import V3TaskStore
from backend.core.model_registry import ModelRegistry
from backend.engine.engine_registry import EngineRegistry
from backend.engine.platform import PlatformInfo
from backend.engine.danqing_audio_engine import DanQingAudioEngine
from backend.engine.danqing_image_engine import DanQingImageEngine
from backend.engine.danqing_video_engine import DanQingVideoEngine
from backend.engine.memory_policy import (
    apply_memory_settings,
    build_gpu_runtimes,
    build_shared_model_cache,
)
from backend.services.services import SettingsService
from backend.services.download_service import DownloadService
from backend.scheduler.task_scheduler import TaskScheduler

from backend.engine.llm import LLMService
from backend.engine.llm.service_mlx import normalize_app_llm_settings, resolve_llm_model_id, resolve_vlm_model_id

from backend.api.routes import (
    adapters, assets, audios, download, gallery, images, loras,
    models, presets, queue, registry, settings, system, tasks, tools, videos,
)


_logger = __import__("logging").getLogger(__name__)


def _resolve_frontend_static_dir(project_root: Path) -> Path | None:
    """Vite build at ``out/frontend/dist``; PyInstaller bundles under ``_MEIPASS/frontend/dist``."""
    candidates: list[Path] = []
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        bundle = Path(sys._MEIPASS)
        candidates.append(bundle / "frontend" / "dist")
        candidates.append(bundle / "frontend")
    repo_root = Path(__file__).resolve().parents[1]
    candidates.append(repo_root / "out" / "frontend" / "dist")
    candidates.append(project_root / "frontend" / "dist")
    candidates.append(project_root / "frontend")
    for path in candidates:
        if path.is_dir() and any(path.iterdir()):
            return path
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    set_bridge(MakeAPIBridge(base_url=resolve_api_base_url()))
    async with _mcp.session_manager.run():
        c = get_container()
        sched = c.try_resolve(TaskScheduler)
        if sched:
            await sched.start()

        cache = c.try_resolve_named("shared_model_cache")
        if cache:
            import asyncio
            cache.start_cleanup(asyncio.get_event_loop())

        yield

        if cache:
            cache.stop_cleanup()
        if sched:
            await sched.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Danmo Make API",
        description="Danmo Make — MLX/CUDA 双后端图像/视频生成引擎",
        version="4.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5800", "http://127.0.0.1:5800", "http://localhost:7800", "http://127.0.0.1:7800", "*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Remote (non-loopback) /api and /mcp require separate API keys; loopback skips.
    from backend.api.access_auth import AccessAuthMiddleware

    app.add_middleware(AccessAuthMiddleware)

    @app.middleware("http")
    async def no_cache_frontend(request, call_next):
        import re
        response = await call_next(request)
        if re.search(r"\.(js|css|html?)$", request.url.path) or request.url.path == "/":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    _setup_dependencies()
    _load_locale()

    app.include_router(registry.router)
    app.include_router(images.router)
    app.include_router(videos.router)
    app.include_router(assets.router)
    app.include_router(tasks.router)
    app.include_router(queue.router)
    app.include_router(audios.router)
    app.include_router(models.router)
    app.include_router(presets.router)
    app.include_router(adapters.router)
    app.include_router(loras.router)
    app.include_router(tools.router)
    app.include_router(system.router)
    app.include_router(gallery.router)
    app.include_router(download.router)
    app.include_router(settings.router)
    import backend.api.routes.setup as setup_routes
    app.include_router(setup_routes.router)

    # LLM service (standalone, not through TaskScheduler)
    import backend.api.routes.llm as llm_routes
    import backend.api.routes.script_parse as script_parse_routes
    import backend.api.routes.canvas as canvas_routes
    import backend.api.routes.long_video_projects as long_video_routes
    app.include_router(llm_routes.router)
    app.include_router(script_parse_routes.router)
    app.include_router(canvas_routes.router)
    app.include_router(long_video_routes.router)

    # MCP streamable-http at /mcp (before StaticFiles catch-all).
    # Starlette Mount: POST /mcp (no trailing slash) → 405; POST /mcp/ works.
    # Normalize so clients that TrimRight("/", url) still connect.
    @app.middleware("http")
    async def mcp_trailing_slash(request, call_next):
        if request.scope.get("path") == "/mcp":
            request.scope["path"] = "/mcp/"
        return await call_next(request)

    app.mount("/mcp", _mcp_http_app)

    frontend_dir = _resolve_frontend_static_dir(project_root)
    if frontend_dir is not None:
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


def _setup_dependencies():
    path_resolver = PathResolver(project_root)
    config_store = JsonConfigStore(path_resolver)
    preset_store = JsonPresetStore(path_resolver)
    registry_json = path_resolver.get_models_registry_path()
    model_registry = ModelRegistry.load(registry_json)

    app_settings = config_store.load()
    if normalize_app_llm_settings(app_settings, model_registry):
        config_store.save(app_settings)
    shared_cache = build_shared_model_cache(config_store.load)
    platforms = PlatformInfo.detect()
    _logger.info(f"Detected GPU backends: {platforms}")
    runtimes = build_gpu_runtimes(app_settings)
    if not runtimes:
        raise RuntimeError("No GPU backend available (need MLX on Apple Silicon or CUDA on NVIDIA)")
    apply_memory_settings(app_settings, runtimes, shared_cache)

    # v4 Danmo Make engines
    from backend.engine.registry import bootstrap_family_plugins

    bootstrap_family_plugins()

    danqing_image = DanQingImageEngine(
        path_resolver, model_registry, runtimes, model_cache=shared_cache,
    )
    danqing_video = DanQingVideoEngine(
        path_resolver, model_registry, runtimes, model_cache=shared_cache,
    )
    from backend.engine.danqing_lora_train_engine import DanQingLoraTrainEngine
    from backend.engine.danqing_tools_engine import DanQingToolsEngine

    danqing_audio = DanQingAudioEngine(
        path_resolver, model_registry, runtimes, model_cache=shared_cache,
    )
    danqing_lora_train = DanQingLoraTrainEngine(
        path_resolver, model_registry, runtimes,
    )
    danqing_tools = DanQingToolsEngine(
        path_resolver, model_registry, runtimes,
    )

    # LLM service (standalone, reuses registry for model path resolution)
    llm_service = LLMService(
        model_registry=model_registry,
        path_resolver=path_resolver,
        default_model_id=resolve_llm_model_id(app_settings, model_registry),
        vision_model_id=resolve_vlm_model_id(app_settings, model_registry),
        llm_think_enabled=app_settings.default_model_llm_think,
    )
    container = get_container()
    container.register_instance(LLMService, llm_service)

    engine_registry = EngineRegistry(model_registry)
    engine_registry.register(danqing_image)
    engine_registry.register(danqing_video)
    engine_registry.register(danqing_audio)
    engine_registry.register_lora_train(danqing_lora_train)
    engine_registry.register_tools(danqing_tools)
    _logger.info("Danmo Make engines registered: image=%s video=%s audio=%s",
                 danqing_image.is_available(), danqing_video.is_available(), danqing_audio.is_available())

    # 持久化层
    v3_db = path_resolver.get_project_root() / "db" / "studio.db"
    v3_tasks = V3TaskStore(v3_db)
    asset_root = path_resolver.get_project_root() / "outputs" / "assets"
    asset_store = SQLiteAssetStore(v3_db, asset_root)
    canvas_session_store = CanvasSessionStore(v3_db)
    long_video_project_store = LongVideoProjectStore(v3_db)
    long_video_activity_store = LongVideoActivityStore(v3_db)
    container.register_instance(CanvasSessionStore, canvas_session_store)
    container.register_instance(LongVideoProjectStore, long_video_project_store)
    container.register_instance(LongVideoActivityStore, long_video_activity_store)

    scheduler = TaskScheduler(
        path_resolver=path_resolver,
        task_store=v3_tasks,
        asset_store=asset_store,
        engine_registry=engine_registry,
        config_store=config_store,
        activity_store=long_video_activity_store,
    )

    download_service = DownloadService(path_resolver, config_store)
    settings_service = SettingsService(
        config_store, path_resolver, danqing_image, danqing_video
    )

    register_services(
        path_resolver=path_resolver,
        config_store=config_store,
        preset_store=preset_store,
        model_registry=model_registry,
        engine_registry=engine_registry,
        image_media_engine=danqing_image,
        download_service=download_service,
        settings_service=settings_service,
        video_media_engine=danqing_video,
        task_scheduler=scheduler,
        asset_store_v3=asset_store,
        shared_model_cache=shared_cache,
        gpu_runtimes=runtimes,
    )


def _load_locale():
    try:
        _load_translations()
        config_store = JsonConfigStore(PathResolver(project_root))
        settings = config_store.load()
        if hasattr(settings, "language") and settings.language:
            set_locale(settings.language)
    except Exception:
        pass


app = create_app()

if __name__ == "__main__":
    import os
    import uvicorn

    host = os.environ.get("DANQING_HTTP_HOST", "0.0.0.0")
    port = int(os.environ.get("DANQING_HTTP_PORT", "7800"))
    uvicorn.run(app, host=host, port=port)
