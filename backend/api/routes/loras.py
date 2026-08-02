"""LoRA training + dataset API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.api.deps import get_asset_store, get_engine_registry, get_llm_service, get_task_scheduler
from backend.core import task_kinds as TK
from backend.core.container import get_container
from backend.core.contracts import (
    DatasetAutoCaptionRequest,
    DatasetCreateRequest,
    DatasetCaptionUpdate,
    DatasetHealthVlmRequest,
    DatasetImportAssetsRequest,
    LoraRegisterRequest,
    LoraTrainingRequest,
)
from backend.core.interfaces import IPathResolver, ISettingsService
from backend.engine.engine_registry import EngineRegistry
from backend.engine.training import dataset_store
from backend.engine.training.crop import presets_with_training_resolution, training_crop_policy
from backend.engine.training.lora_train_runtime_mlx import (
    resume_checkpoint_incompatibility,
    train_min_memory_gb,
)
from backend.engine.training.presets import (
    FLUX1_TRAIN_MIN_MEMORY_GB,
    PRESETS,
    QWEN_IMAGE_PRESETS,
    TRAINABLE_BASE_MODELS,
    Z_IMAGE_PRESETS,
    Z_IMAGE_TURBO_PRESETS,
)
from backend.engine.training.user_lora_registry import delete_user_lora, list_user_loras, register_user_lora
from backend.persistence.asset_store import SQLiteAssetStore
from backend.scheduler.task_scheduler import TaskScheduler
from backend.utils.path_utils import get_memory_gb

router = APIRouter(prefix="/api/loras", tags=["loras"])


def _paths() -> IPathResolver:
    return get_container().resolve(IPathResolver)


@router.get("/datasets")
def list_datasets():
    root = _paths().get_project_root()
    return {"items": dataset_store.list_datasets(root)}


@router.post("/datasets", status_code=201)
def create_dataset(body: DatasetCreateRequest):
    root = _paths().get_project_root()
    return dataset_store.create_dataset(
        root,
        name=body.name,
        kind=body.kind,
        trigger_word=body.trigger_word,
        default_prompt=body.default_prompt,
        nsfw=body.nsfw,
    )


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    root = _paths().get_project_root()
    try:
        return dataset_store.get_dataset(root, dataset_id)
    except FileNotFoundError as e:
        raise HTTPException(404, detail={"code": "not_found", "message": str(e)}) from e


@router.get("/datasets/{dataset_id}/health")
def dataset_health(dataset_id: str):
    root = _paths().get_project_root()
    try:
        from backend.engine.training.lora_quality import analyze_dataset_health

        report = analyze_dataset_health(root, dataset_id)
        report["vision_available"] = get_llm_service().is_vision_available()
        return report
    except FileNotFoundError as e:
        raise HTTPException(404, detail={"code": "not_found", "message": str(e)}) from e


@router.post("/datasets/{dataset_id}/health/vlm")
async def dataset_health_vlm(dataset_id: str, body: DatasetHealthVlmRequest | None = None):
    service = get_llm_service()
    if not service.is_available():
        raise HTTPException(
            503,
            detail={"code": "llm_unavailable", "message": "LLM model not installed. Install via Models page."},
        )
    if not service.is_vision_available():
        raise HTTPException(
            503,
            detail={"code": "vision_unavailable", "message": "Vision model not available for VLM audit."},
        )

    root = _paths().get_project_root()
    max_samples = body.max_samples if body else 0
    audit_kind_override = body.audit_kind if body else None
    try:
        from backend.core.container import get_container
        from backend.engine.llm.vlm_subprocess import run_vlm_audit_subprocess
        from backend.engine.memory_policy import prepare_host_for_vlm_audit
        from backend.engine.training import dataset_store
        from backend.engine.training.lora_quality import analyze_dataset_health
        from backend.engine.training.lora_quality_vlm import (
            build_dataset_audit_instruction,
            collect_dataset_image_paths,
            compile_portrait_dataset_audit,
            merge_vlm_hints,
            resolve_audit_paths,
            resolve_dataset_audit_kind,
        )

        base = analyze_dataset_health(root, dataset_id)
        audit_kind = resolve_dataset_audit_kind(root, dataset_id, audit_kind_override)
        paths = collect_dataset_image_paths(root, dataset_id)
        if not paths:
            raise HTTPException(400, detail={"code": "invalid", "message": "No readable images to audit"})
        audit_paths, truncated = resolve_audit_paths(paths, max_samples=max_samples)
        dataset_root = dataset_store.datasets_root(root) / dataset_id
        file_keys = [
            str(p.relative_to(dataset_root)) if p.is_relative_to(dataset_root) else p.name
            for p in audit_paths
        ]
        all_file_keys = [
            str(p.relative_to(dataset_root)) if p.is_relative_to(dataset_root) else p.name
            for p in paths
        ]
        model_dir = service._resolve_vision_model_path()
        mlx_runtime = None
        try:
            runtimes = get_container().try_resolve_named("gpu_runtimes") or {}
            if isinstance(runtimes, dict):
                mlx_runtime = runtimes.get("mlx")
        except Exception:
            mlx_runtime = None
        worker_mem = prepare_host_for_vlm_audit(mlx_runtime=mlx_runtime)
        texts = await run_vlm_audit_subprocess(
            image_paths=audit_paths,
            model_dir=model_dir,
            instruction=build_dataset_audit_instruction(audit_kind),
            worker_memory_gb=worker_mem,
        )
        vlm = compile_portrait_dataset_audit(
            paths,
            audit_paths,
            texts,
            all_file_keys=all_file_keys,
            vlm_file_keys=file_keys,
            audit_kind=audit_kind,
            truncated=truncated,
            total_images=len(paths),
        )
        merged = merge_vlm_hints(base, vlm)
        merged["vision_available"] = True
        merged["vlm_audited"] = True
        merged["audit_kind"] = audit_kind
        return merged
    except FileNotFoundError as e:
        raise HTTPException(404, detail={"code": "not_found", "message": str(e)}) from e
    except RuntimeError as e:
        raise HTTPException(503, detail={"code": "vlm_audit_failed", "message": str(e)}) from e


@router.patch("/datasets/{dataset_id}")
def patch_dataset(dataset_id: str, body: DatasetCreateRequest):
    root = _paths().get_project_root()
    try:
        return dataset_store.update_dataset_meta(
            root,
            dataset_id,
            body.model_dump(),
        )
    except FileNotFoundError as e:
        raise HTTPException(404, detail={"code": "not_found", "message": str(e)}) from e


@router.post("/datasets/{dataset_id}/images", status_code=201)
async def upload_dataset_images(
    dataset_id: str,
    files: list[UploadFile] = File(...),
    default_prompt: Optional[str] = Form(None),
):
    root = _paths().get_project_root()
    payload: list[tuple[str, bytes]] = []
    for f in files:
        payload.append((f.filename or "image.png", await f.read()))
    try:
        return dataset_store.add_dataset_images(
            root, dataset_id, payload, default_prompt=default_prompt
        )
    except FileNotFoundError as e:
        raise HTTPException(404, detail={"code": "not_found", "message": str(e)}) from e
    except ValueError as e:
        raise HTTPException(400, detail={"code": "invalid", "message": str(e)}) from e


@router.patch("/datasets/{dataset_id}/captions")
def patch_captions(dataset_id: str, body: DatasetCaptionUpdate):
    root = _paths().get_project_root()
    try:
        return dataset_store.update_dataset_captions(root, dataset_id, body.captions)
    except FileNotFoundError as e:
        raise HTTPException(404, detail={"code": "not_found", "message": str(e)}) from e


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str):
    root = _paths().get_project_root()
    try:
        dataset_store.delete_dataset(root, dataset_id)
    except FileNotFoundError as e:
        raise HTTPException(404, detail={"code": "not_found", "message": str(e)}) from e
    except ValueError as e:
        raise HTTPException(400, detail={"code": "invalid", "message": str(e)}) from e


@router.delete("/datasets/{dataset_id}/images/{file_path:path}")
def delete_dataset_image(dataset_id: str, file_path: str):
    root = _paths().get_project_root()
    try:
        return dataset_store.remove_dataset_image(root, dataset_id, file_path)
    except FileNotFoundError as e:
        raise HTTPException(404, detail={"code": "not_found", "message": str(e)}) from e


@router.post("/datasets/{dataset_id}/import-assets", status_code=201)
def import_dataset_assets(dataset_id: str, body: DatasetImportAssetsRequest, store: SQLiteAssetStore = Depends(get_asset_store)):
    root = _paths().get_project_root()

    def _resolve(aid: str) -> Path:
        return store.get_file_path(aid)

    try:
        return dataset_store.import_dataset_from_assets(
            root,
            dataset_id,
            body.asset_ids,
            resolve_asset_path=_resolve,
            default_prompt=body.default_prompt,
            asset_captions=body.captions,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, detail={"code": "not_found", "message": str(e)}) from e
    except ValueError as e:
        raise HTTPException(400, detail={"code": "invalid", "message": str(e)}) from e


@router.post("/datasets/{dataset_id}/auto-caption")
async def auto_caption_dataset(
    dataset_id: str,
    body: DatasetAutoCaptionRequest,
):
    """Generate captions for dataset images using the vision LLM (isolated subprocess)."""
    service = get_llm_service()
    if not service.is_available():
        raise HTTPException(
            503,
            detail={"code": "llm_unavailable", "message": "LLM model not installed. Install via Models page."},
        )
    if not service.is_vision_available():
        raise HTTPException(
            503,
            detail={"code": "vision_unavailable", "message": "Vision model not available for auto-caption."},
        )
    root = _paths().get_project_root()
    try:
        ds = dataset_store.get_dataset(root, dataset_id)
    except FileNotFoundError as e:
        raise HTTPException(404, detail={"code": "not_found", "message": str(e)}) from e
    targets = [img for img in ds.get("images") or [] if img.get("exists")]
    if body.files:
        allow = set(body.files)
        targets = [img for img in targets if img.get("file") in allow]
    if not targets:
        raise HTTPException(400, detail={"code": "invalid", "message": "No images to caption"})
    audit_kind = str(ds.get("kind") or "concept").strip().lower()
    if audit_kind not in ("concept", "style"):
        audit_kind = "concept"
    from backend.core.container import get_container
    from backend.engine.llm.vlm_subprocess import run_lora_caption_subprocess
    from backend.engine.memory_policy import prepare_host_for_vlm_audit
    from backend.engine.training.lora_auto_caption import resolve_lora_subject_name

    meta = {
        "name": ds.get("name") or "",
        "kind": audit_kind,
        "trigger_word": ds.get("trigger_word") or "",
        "default_prompt": ds.get("default_prompt") or "",
    }
    subject_name = resolve_lora_subject_name(meta) if audit_kind == "concept" else ""
    image_paths: list[Path] = []
    file_keys: list[str] = []
    for img in targets:
        file_rel = str(img.get("file") or "")
        img_path = root / "datasets" / dataset_id / file_rel
        if not img_path.is_file():
            continue
        image_paths.append(img_path)
        file_keys.append(file_rel)
    if not image_paths:
        raise HTTPException(400, detail={"code": "invalid", "message": "No readable images to caption"})

    model_dir = service._resolve_vision_model_path()
    mlx_runtime = None
    try:
        runtimes = get_container().try_resolve_named("gpu_runtimes") or {}
        if isinstance(runtimes, dict):
            mlx_runtime = runtimes.get("mlx")
    except Exception:
        mlx_runtime = None
    worker_mem = prepare_host_for_vlm_audit(mlx_runtime=mlx_runtime)
    try:
        prompts = await run_lora_caption_subprocess(
            image_paths=image_paths,
            model_dir=model_dir,
            audit_kind=audit_kind,
            subject_name=subject_name,
            worker_memory_gb=worker_mem,
        )
    except RuntimeError as e:
        raise HTTPException(503, detail={"code": "caption_failed", "message": str(e)}) from e

    captions = [{"file": file_key, "prompt": prompt.strip()} for file_key, prompt in zip(file_keys, prompts, strict=False)]
    if not captions:
        raise HTTPException(400, detail={"code": "invalid", "message": "Caption generation produced no results"})
    result = dataset_store.update_dataset_captions(root, dataset_id, captions)

    return result


@router.post("/datasets/import-dog6", status_code=201)
def import_dog6():
    paths = _paths()
    root = paths.get_project_root()
    bundled = dataset_store.bundled_dog6_example_dir(paths.get_default_config_root())
    try:
        return dataset_store.import_dog6_example(root, bundled_root=bundled)
    except Exception as e:
        raise HTTPException(500, detail={"code": "import_failed", "message": str(e)}) from e


@router.get("/trainable-models")
def trainable_models():
    service = get_container().resolve(ISettingsService)
    registry = service.get_model_registry()
    detailed = service.get_models_detailed_status()
    items: list[dict[str, Any]] = []
    for mid in sorted(TRAINABLE_BASE_MODELS):
        cfg = registry.get(mid)
        status = detailed.get(mid, {})
        crop = training_crop_policy(mid)
        items.append(
            {
                "id": mid,
                "name": (cfg.name if cfg else {}) or {},
                "ready": bool(status.get("ready")),
                "trainable": mid in TRAINABLE_BASE_MODELS,
                "phase": 1 if mid in TRAINABLE_BASE_MODELS else 2,
                "training_crop": {
                    "vae_scale": int(crop["vae_scale"]),
                    "auto_crop": True,
                },
            }
        )
    return {
        "items": items,
        "presets": presets_with_training_resolution("flux1-dev", PRESETS),
        "presets_by_model": {
            "flux1-dev": presets_with_training_resolution("flux1-dev", PRESETS),
            "z-image": presets_with_training_resolution("z-image", Z_IMAGE_PRESETS),
            "z-image-turbo": presets_with_training_resolution("z-image-turbo", Z_IMAGE_TURBO_PRESETS),
            "qwen-image": presets_with_training_resolution("qwen-image", QWEN_IMAGE_PRESETS),
        },
    }


def _raise_resume_incompatible(message: str) -> None:
    raise HTTPException(
        400,
        detail={"code": "resume_incompatible", "message": message},
    )


def _validate_resume_checkpoint(
    body: LoraTrainingRequest,
    sched: TaskScheduler,
    adapter_path: Path,
    *,
    task_id: str | None,
) -> None:
    source_params: dict[str, Any] | None = None
    if task_id:
        row = sched.get_task(task_id)
        if row:
            raw = row.get("params")
            if isinstance(raw, dict):
                source_params = raw
    reason = resume_checkpoint_incompatibility(
        base_model=body.base_model,
        adapter_path=adapter_path,
        source_task_params=source_params,
    )
    if reason:
        _raise_resume_incompatible(reason)


def _resolve_resume_adapter(body: LoraTrainingRequest, sched: TaskScheduler) -> LoraTrainingRequest:
    if body.resume_from:
        path = Path(body.resume_from)
        if not path.is_file():
            raise HTTPException(
                404,
                detail={
                    "code": "resume_not_found",
                    "message": f"Resume adapter not found: {body.resume_from}",
                },
            )
        _validate_resume_checkpoint(body, sched, path, task_id=(body.resume_task_id or "").strip() or None)
        return body
    task_id = (body.resume_task_id or "").strip()
    checkpoint = (body.resume_checkpoint or "").strip()
    if not task_id and not checkpoint:
        return body
    if not task_id or not checkpoint:
        raise HTTPException(
            400,
            detail={
                "code": "invalid_resume",
                "message": "resume_task_id and resume_checkpoint must both be set",
            },
        )
    if ".." in checkpoint or "/" in checkpoint or "\\" in checkpoint:
        raise HTTPException(
            400,
            detail={"code": "invalid_resume", "message": "resume_checkpoint must be a filename only"},
        )
    work = sched.task_work_dir(task_id)
    path = work / "adapters" / checkpoint
    if not path.is_file():
        raise HTTPException(
            404,
            detail={
                "code": "resume_not_found",
                "message": f"Checkpoint not found: {checkpoint} for task {task_id}",
            },
        )
    _validate_resume_checkpoint(body, sched, path, task_id=task_id)
    return body.model_copy(update={"resume_from": str(path)})


@router.get("/training/requirements")
def training_requirements(base_model: str = "flux1-dev", qlora_bits: Optional[int] = None):
    if qlora_bits is not None and int(qlora_bits) not in (4, 8):
        raise HTTPException(
            400,
            detail={"code": "invalid_qlora_bits", "message": "qlora_bits must be 4 or 8"},
        )
    mem = get_memory_gb()
    qbits = int(qlora_bits) if qlora_bits in (4, 8) else None
    min_gb = train_min_memory_gb(base_model, qlora_bits=qbits)
    return {
        "min_memory_gb": min_gb,
        "detected_memory_gb": mem,
        "mlx_required": True,
        "qlora_bits": qbits,
        "can_submit": mem <= 0 or mem >= min_gb - 2,
    }


@router.post("/trainings", status_code=202)
async def submit_training(
    body: LoraTrainingRequest,
    sched: TaskScheduler = Depends(get_task_scheduler),
    engines: EngineRegistry = Depends(get_engine_registry),
):
    body = _resolve_resume_adapter(body, sched)
    mem = get_memory_gb()
    min_gb = train_min_memory_gb(body.base_model, qlora_bits=body.qlora_bits)
    if mem > 0 and mem < min_gb - 2:
        raise HTTPException(
            409,
            detail={
                "code": "insufficient_memory",
                "message": f"LoRA training for {body.base_model.split(':', 1)[0]!r} "
                f"requires ~{min_gb:.0f}GB unified memory",
            },
        )
    train_eng = engines.get_lora_train()
    if not train_eng.supports_base_model(body.base_model):
        raise HTTPException(
            409,
            detail={"code": "unsupported", "message": f"base model {body.base_model!r} not trainable"},
        )
    root = _paths().get_project_root()
    try:
        dataset_store.validate_dataset_for_training(root, body.dataset_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, detail={"code": "invalid_dataset", "message": str(e)}) from e

    r = await sched.submit(
        kind=TK.LORA_TRAINING,
        model_id=body.base_model.split(":", 1)[0],
        params=body.model_dump(),
        priority=body.priority,
    )
    return {"task": r}


@router.get("/datasets/{dataset_id}/file/{file_path:path}")
def dataset_image_file(dataset_id: str, file_path: str):
    from fastapi.responses import FileResponse

    root = _paths().get_project_root()
    path = root / "datasets" / dataset_id / file_path
    if not path.is_file():
        raise HTTPException(404, detail={"code": "not_found", "message": "image not found"})
    return FileResponse(path)


@router.get("/trainings/{task_id}/artifacts/file/{filename}")
def training_artifact_file(task_id: str, filename: str, sched: TaskScheduler = Depends(get_task_scheduler)):
    from fastapi.responses import FileResponse

    work = sched.task_work_dir(task_id)
    candidates = [
        work / filename,
        work / "adapters" / filename,
    ]
    for path in candidates:
        if path.is_file() and path.suffix.lower() in {".png", ".safetensors", ".json"}:
            return FileResponse(path)
    raise HTTPException(404, detail={"code": "not_found", "message": "artifact not found"})


@router.get("/trainings/{task_id}/artifacts")
def training_artifacts(task_id: str, sched: TaskScheduler = Depends(get_task_scheduler)):
    work = sched.task_work_dir(task_id)
    if not work.is_dir():
        raise HTTPException(404, detail={"code": "not_found", "message": "task work dir not found"})
    progress_images = sorted(work.glob("*_progress.png"))
    checkpoints = sorted((work / "adapters").glob("*_adapters.safetensors")) if (work / "adapters").is_dir() else []
    loss_path = work / "loss_history.json"
    loss_history = []
    if loss_path.is_file():
        import json

        loss_history = json.loads(loss_path.read_text(encoding="utf-8"))

    quality = None
    vision_available = get_llm_service().is_vision_available()
    if loss_history:
        from backend.engine.training.lora_quality import analyze_dataset_health, analyze_training_quality

        task_row = sched.get_task(task_id) or {}
        params = task_row.get("params") or {}
        dataset_id = str(params.get("dataset_id") or "").strip()
        dataset_health = None
        if dataset_id:
            root = _paths().get_project_root()
            try:
                dataset_health = analyze_dataset_health(root, dataset_id)
            except FileNotFoundError:
                dataset_health = None
        quality = analyze_training_quality(loss_history, dataset_health=dataset_health)
        if quality is not None:
            quality["vision_available"] = vision_available

    return {
        "progress_images": [p.name for p in progress_images],
        "checkpoints": [p.name for p in checkpoints],
        "loss_history": loss_history,
        "quality": quality,
        "vision_available": vision_available,
    }


@router.post("/trainings/{task_id}/quality/vlm")
async def training_quality_vlm(task_id: str, sched: TaskScheduler = Depends(get_task_scheduler)):
    import json

    service = get_llm_service()
    if not service.is_available():
        raise HTTPException(
            503,
            detail={"code": "llm_unavailable", "message": "LLM model not installed. Install via Models page."},
        )
    if not service.is_vision_available():
        raise HTTPException(
            503,
            detail={"code": "vision_unavailable", "message": "Vision model not available for VLM audit."},
        )

    work = sched.task_work_dir(task_id)
    if not work.is_dir():
        raise HTTPException(404, detail={"code": "not_found", "message": "task work dir not found"})

    progress_paths = sorted(work.glob("*_progress.png"))
    if not progress_paths:
        raise HTTPException(
            400,
            detail={"code": "invalid", "message": "No progress preview images to audit"},
        )

    loss_path = work / "loss_history.json"
    loss_history: list[dict[str, Any]] = []
    if loss_path.is_file():
        loss_history = json.loads(loss_path.read_text(encoding="utf-8"))

    from backend.engine.training.lora_quality import analyze_dataset_health, analyze_training_quality
    from backend.engine.training.lora_quality_vlm import (
        build_progress_audit_instruction,
        compile_progress_vlm_report,
        merge_vlm_hints,
        pick_progress_preview_paths,
        resolve_dataset_audit_kind,
    )
    from backend.core.container import get_container
    from backend.engine.llm.vlm_subprocess import run_vlm_audit_subprocess
    from backend.engine.memory_policy import prepare_host_for_vlm_audit

    task_row = sched.get_task(task_id) or {}
    params = task_row.get("params") or {}
    progress_prompt = str(params.get("progress_prompt") or "")
    dataset_id = str(params.get("dataset_id") or "").strip()
    dataset_health = None
    if dataset_id:
        root = _paths().get_project_root()
        try:
            dataset_health = analyze_dataset_health(root, dataset_id)
        except FileNotFoundError:
            dataset_health = None

    base_quality = analyze_training_quality(loss_history, dataset_health=dataset_health) if loss_history else {
        "level": "fair",
        "score": 50,
        "metrics": {"steps_logged": 0},
        "hints": [],
        "dataset_health": dataset_health,
    }

    audit_kind = "concept"
    if dataset_id:
        root = _paths().get_project_root()
        audit_kind = resolve_dataset_audit_kind(root, dataset_id)

    try:
        sample_paths = pick_progress_preview_paths(progress_paths)
        instruction = build_progress_audit_instruction(progress_prompt, audit_kind=audit_kind)
        model_dir = service._resolve_vision_model_path()
        mlx_runtime = None
        try:
            runtimes = get_container().try_resolve_named("gpu_runtimes") or {}
            if isinstance(runtimes, dict):
                mlx_runtime = runtimes.get("mlx")
        except Exception:
            mlx_runtime = None
        worker_mem = prepare_host_for_vlm_audit(mlx_runtime=mlx_runtime)
        texts = await run_vlm_audit_subprocess(
            image_paths=sample_paths,
            model_dir=model_dir,
            instruction=instruction,
            worker_memory_gb=worker_mem,
        )
        vlm = compile_progress_vlm_report(sample_paths, texts, audit_kind=audit_kind)
        merged = merge_vlm_hints(base_quality, vlm)
        merged["vision_available"] = True
        merged["vlm_audited"] = True
        merged["audit_kind"] = audit_kind
        return merged
    except RuntimeError as e:
        raise HTTPException(503, detail={"code": "vlm_audit_failed", "message": str(e)}) from e


@router.get("/user-adapters")
def list_user_adapters():
    paths = _paths()
    items = list_user_loras(paths.get_workspace_config_dir())
    items = [
        ul for ul in items
        if str(ul.get("source") or "user_trained") == "user_trained"
    ]
    out: list[dict[str, Any]] = []
    for ul in items:
        local_path = str(ul.get("local_path") or "")
        resolved = paths.resolve_registry_local_path(local_path) if local_path else None
        out.append(
            {
                **ul,
                "installed": bool(resolved and (resolved.is_file() or resolved.is_dir())),
            }
        )
    return {"items": out}


@router.get("/downloaded-adapters")
def list_downloaded_adapters():
    """LoRAs fetched via model-library remote search (``source=remote_download``)."""
    paths = _paths()
    items = list_user_loras(paths.get_workspace_config_dir())
    items = [
        ul for ul in items
        if str(ul.get("source") or "") == "remote_download"
    ]
    out: list[dict[str, Any]] = []
    for ul in items:
        local_path = str(ul.get("local_path") or "")
        resolved = paths.resolve_registry_local_path(local_path) if local_path else None
        out.append(
            {
                **ul,
                "installed": bool(resolved and (resolved.is_file() or resolved.is_dir())),
            }
        )
    return {"items": out}


@router.delete("/user-adapters/{lora_id}")
def remove_user_adapter(lora_id: str, remove_files: bool = False):
    paths = _paths()
    ok = delete_user_lora(
        paths.get_workspace_config_dir(),
        lora_id,
        remove_files=remove_files,
        project_root=paths.get_project_root(),
    )
    if not ok:
        raise HTTPException(404, detail={"code": "not_found", "message": f"user LoRA {lora_id!r} not found"})
    return {"ok": True, "id": lora_id}


@router.post("/trainings/{task_id}/register")
def register_training_checkpoint(
    task_id: str,
    body: LoraRegisterRequest,
    sched: TaskScheduler = Depends(get_task_scheduler),
):
    work = sched.task_work_dir(task_id)
    ckpt = work / "adapters" / body.checkpoint
    if not ckpt.is_file():
        raise HTTPException(404, detail={"code": "not_found", "message": f"checkpoint {body.checkpoint!r} not found"})
    task_row = sched.get_task(task_id) or {}
    params = task_row.get("params") or {}
    base_model = str(params.get("base_model") or "flux1-dev").split(":", 1)[0]
    dataset_id = str(params.get("dataset_id") or "").strip()
    trigger_word = ""
    paths = _paths()
    if dataset_id:
        try:
            trigger_word = str(dataset_store._dataset_meta(paths.get_project_root(), dataset_id).get("trigger_word") or "").strip()
        except Exception:
            trigger_word = ""
    slug = body.name.strip() or f"trained-{task_id[-8:]}"
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in slug)[:64]
    dest_dir = paths.get_loras_dir() / slug
    dest_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    dest_file = dest_dir / "adapter.safetensors"
    shutil.copy2(ckpt, dest_file)
    (dest_dir / "lora_config.json").write_text(
        json.dumps(
            {
                "base_model": base_model,
                "trigger_word": trigger_word,
                "task_id": task_id,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    entry = register_user_lora(
        paths.get_workspace_config_dir(),
        name=body.name or slug,
        base_model=base_model,
        local_path=f"models/Lora/{slug}",
        trigger_word=trigger_word,
        task_id=task_id,
    )
    return {"user_lora": entry, "adapter_path": str(dest_file)}
