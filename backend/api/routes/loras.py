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
    DatasetCreateRequest,
    DatasetHealthVlmRequest,
    DatasetSkippedImagesRequest,
    DatasetImportAssetsRequest,
    LoraContinueRequest,
    LoraEvaluateEndpointRequest,
    LoraEvaluationRequest,
    LoraRegisterRequest,
    LoraSelectCheckpointRequest,
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


def _face_audit_resolution(width: int, height: int) -> tuple[int, int]:
    from backend.engine.training.crop import align_training_edge

    if not 256 <= width <= 2048 or not 256 <= height <= 2048:
        raise ValueError("Face-audit resolution must be between 256 and 2048 pixels")
    return (
        align_training_edge(width, vae_scale=16),
        align_training_edge(height, vae_scale=16),
    )


@router.get("/datasets/{dataset_id}/health")
def dataset_health(dataset_id: str, width: int = 512, height: int = 512):
    root = _paths().get_project_root()
    try:
        from backend.engine.training.lora_quality import analyze_dataset_health_shared

        resolution = _face_audit_resolution(width, height)
        report = analyze_dataset_health_shared(root, dataset_id, training_resolution=resolution)
        if report.get("faces") is not None:
            report["faces"]["resolution"] = list(resolution)
        report["vision_available"] = get_llm_service().is_vision_available()
        return report
    except FileNotFoundError as e:
        raise HTTPException(404, detail={"code": "not_found", "message": str(e)}) from e
    except ValueError as e:
        raise HTTPException(400, detail={"code": "invalid_face_audit_resolution", "message": str(e)}) from e
    except TimeoutError as e:
        raise HTTPException(503, detail={"code": "face_preflight_busy", "message": str(e)}) from e


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

        resolution = _face_audit_resolution(
            body.width if body else 512,
            body.height if body else 512,
        )
        base = analyze_dataset_health(root, dataset_id, training_resolution=resolution)
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
            instruction=build_dataset_audit_instruction(audit_kind, resolution=resolution),
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
            project_root=root,
            training_resolution=resolution,
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


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str):
    root = _paths().get_project_root()
    try:
        dataset_store.delete_dataset(root, dataset_id)
    except FileNotFoundError as e:
        raise HTTPException(404, detail={"code": "not_found", "message": str(e)}) from e
    except ValueError as e:
        raise HTTPException(400, detail={"code": "invalid", "message": str(e)}) from e


@router.put("/datasets/{dataset_id}/skipped-images")
def put_skipped_images(dataset_id: str, body: DatasetSkippedImagesRequest):
    root = _paths().get_project_root()
    try:
        return dataset_store.set_skipped_images(root, dataset_id, body.files)
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
        )
    except FileNotFoundError as e:
        raise HTTPException(404, detail={"code": "not_found", "message": str(e)}) from e
    except ValueError as e:
        raise HTTPException(400, detail={"code": "invalid", "message": str(e)}) from e


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
    from backend.engine.training.lora_train_config import resolve_training_config

    source_params: dict[str, Any] | None = None
    if task_id:
        row = sched.get_task(task_id)
        if row:
            raw = row.get("params")
            if isinstance(raw, dict):
                source_params = raw
        else:
            _raise_resume_incompatible(f"Resume source task not found: {task_id}")
    reason = resume_checkpoint_incompatibility(
        base_model=body.base_model,
        adapter_path=adapter_path,
        source_task_params=source_params,
        expected_config=resolve_training_config(body.model_dump()),
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
    # Studio always keeps checkpoints unregistered until the user picks one from the matrix.
    body.auto_register = False
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
        from backend.engine.training.lora_train_config import training_inference_config
        from backend.engine.training.training_preflight import validate_training_request

        validate_training_request(root, body, sched=sched)
        body.metadata = {
            **body.metadata,
            "inference": training_inference_config(body.model_dump()),
        }
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, detail={"code": "invalid_dataset", "message": str(e)}) from e

    r = await sched.submit(
        kind=TK.LORA_TRAINING,
        model_id=body.base_model.split(":", 1)[0],
        params=body.model_dump(),
        priority=body.priority,
    )
    return {"task": r}


@router.post("/trainings/{task_id}/continue", status_code=202)
async def continue_training(
    task_id: str,
    body: LoraContinueRequest | None = None,
    sched: TaskScheduler = Depends(get_task_scheduler),
):
    """Requeue a cancelled/failed LoRA training task in place (same work_dir)."""
    req = body or LoraContinueRequest()
    try:
        task = await sched.continue_lora_training(
            task_id,
            checkpoint=(req.checkpoint or None),
            priority=req.priority,
        )
    except RuntimeError as exc:
        msg = str(exc)
        code = "not_resumable"
        status = 409
        low = msg.lower()
        if "not found" in low:
            code = "not_found"
            status = 404
        elif "filename only" in low or "missing optimizer" in low or "no resumable" in low:
            code = "no_checkpoint"
            status = 400
        elif "cancelled/failed" in low or "in flight" in low:
            code = "not_resumable"
            status = 409
        raise HTTPException(status, detail={"code": code, "message": msg}) from exc
    return {"task": task}


@router.get("/datasets/{dataset_id}/file/{file_path:path}")
def dataset_image_file(dataset_id: str, file_path: str):
    from fastapi.responses import FileResponse

    root = _paths().get_project_root()
    path = root / "datasets" / dataset_id / file_path
    if not path.is_file():
        raise HTTPException(404, detail={"code": "not_found", "message": "image not found"})
    return FileResponse(path)


@router.get("/trainings/{task_id}/artifacts/file/{filename:path}")
def training_artifact_file(task_id: str, filename: str, sched: TaskScheduler = Depends(get_task_scheduler)):
    from fastapi.responses import FileResponse

    work = sched.task_work_dir(task_id).resolve()
    # Allow nested candidate images under candidates/, but never path escape.
    rel = str(filename or "").lstrip("/")
    if not rel or ".." in Path(rel).parts:
        raise HTTPException(400, detail={"code": "invalid", "message": "invalid artifact path"})
    candidates = [
        (work / rel).resolve(),
        (work / "adapters" / Path(rel).name).resolve(),
        (work / "candidates" / Path(rel).name).resolve(),
    ]
    for path in candidates:
        if work not in path.parents and path != work:
            continue
        if path.is_file() and path.suffix.lower() in {".png", ".safetensors", ".json"}:
            return FileResponse(path)
    raise HTTPException(404, detail={"code": "not_found", "message": "artifact not found"})


@router.get("/trainings/{task_id}/artifacts")
def training_artifacts(task_id: str, sched: TaskScheduler = Depends(get_task_scheduler)):
    work = sched.task_work_dir(task_id)
    if not work.is_dir():
        raise HTTPException(404, detail={"code": "not_found", "message": "task work dir not found"})
    progress_images = sorted(work.glob("*_progress*.png"))
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
                stored_resolution = (params.get("metadata") or {}).get("training_resolution")
                if isinstance(stored_resolution, (list, tuple)) and len(stored_resolution) == 2:
                    audit_resolution = _face_audit_resolution(
                        int(stored_resolution[0]), int(stored_resolution[1])
                    )
                else:
                    from backend.engine.training.crop import resolve_training_resolution
                    from backend.engine.training.lora_train_config import resolve_training_config

                    cfg = resolve_training_config(params)
                    audit_resolution = resolve_training_resolution(
                        str(params.get("base_model") or ""), cfg, preset=params.get("preset")
                    )
                dataset_health = analyze_dataset_health(
                    root, dataset_id, training_resolution=audit_resolution
                )
            except FileNotFoundError:
                dataset_health = None
        quality = analyze_training_quality(loss_history, dataset_health=dataset_health)
        if quality is not None:
            quality["vision_available"] = vision_available

    task_row = sched.get_task(task_id) or {}
    params = task_row.get("params") or {}
    dataset_id = str(params.get("dataset_id") or "").strip()
    from backend.engine.training.lora_checkpoint_evaluation import (
        list_candidate_checkpoints,
        load_candidate_evaluation,
    )
    from backend.engine.training.lora_train_config import adapter_export_config, training_inference_config

    dataset_meta = dataset_store._dataset_meta(_paths().get_project_root(), dataset_id) if dataset_id else {}
    crop_plan_path = work / "crop_plan.json"
    crop_plan = json.loads(crop_plan_path.read_text(encoding="utf-8")) if crop_plan_path.is_file() else None
    candidate_eval = load_candidate_evaluation(work)
    adapter_dir = work / "adapters"
    candidate_checkpoints = list_candidate_checkpoints(adapter_dir) if adapter_dir.is_dir() else []
    return {
        "progress_images": [p.name for p in progress_images],
        "checkpoints": [p.name for p in checkpoints],
        "candidate_checkpoints": candidate_checkpoints,
        "candidate_evaluation": candidate_eval,
        "loss_history": loss_history,
        "quality": quality,
        "vision_available": vision_available,
        "inference": training_inference_config(params),
        "training_contract": adapter_export_config(params=params, dataset_meta=dataset_meta),
        "crop_plan": crop_plan,
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
            stored_resolution = (params.get("metadata") or {}).get("training_resolution")
            if isinstance(stored_resolution, (list, tuple)) and len(stored_resolution) == 2:
                audit_resolution = _face_audit_resolution(
                    int(stored_resolution[0]), int(stored_resolution[1])
                )
            else:
                from backend.engine.training.crop import resolve_training_resolution
                from backend.engine.training.lora_train_config import resolve_training_config

                cfg = resolve_training_config(params)
                audit_resolution = resolve_training_resolution(
                    str(params.get("base_model") or ""), cfg, preset=params.get("preset")
                )
            dataset_health = analyze_dataset_health(
                root, dataset_id, training_resolution=audit_resolution
            )
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


@router.delete("/trainings/{task_id}")
async def delete_training_result(task_id: str, sched: TaskScheduler = Depends(get_task_scheduler)):
    """Delete a terminal training task, its exported LoRAs, and its work directory."""
    import shutil

    from backend.engine.training.lora_checkpoint_evaluation import load_candidate_evaluation

    task = sched.get_task(task_id)
    if task is None:
        raise HTTPException(404, detail={"code": "not_found", "message": "training task not found"})
    if not TK.is_lora_training_kind(str(task.get("kind") or "")):
        raise HTTPException(400, detail={"code": "invalid", "message": "task is not a LoRA training task"})
    if str(task.get("status") or "") not in {"completed", "failed", "cancelled"}:
        raise HTTPException(409, detail={"code": "active", "message": "stop the training task before deleting its results"})

    paths = _paths()
    root = paths.get_project_root().resolve()
    config_dir = paths.get_workspace_config_dir()
    work = (paths.get_outputs_dir() / "work" / task_id).resolve()
    work_root = (paths.get_outputs_dir() / "work").resolve()
    if work.parent != work_root:
        raise HTTPException(400, detail={"code": "invalid", "message": "invalid training task id"})

    candidate = load_candidate_evaluation(work) if work.is_dir() else None
    eval_task_id = str((candidate or {}).get("evaluation_task_id") or "").strip()
    removed_eval_tasks: list[str] = []
    if eval_task_id:
        eval_row = sched.get_task(eval_task_id)
        if eval_row and str(eval_row.get("status") or "") in {"queued", "running"}:
            raise HTTPException(
                409,
                detail={
                    "code": "evaluation_active",
                    "message": "stop or wait for checkpoint evaluation before deleting this training result",
                },
            )
        if eval_row and str(eval_row.get("status") or "") in {"completed", "failed", "cancelled"}:
            eval_work = (paths.get_outputs_dir() / "work" / eval_task_id).resolve()
            if eval_work.parent == work_root and eval_work.is_dir():
                shutil.rmtree(eval_work)
            if not sched.delete_task(eval_task_id):
                raise RuntimeError(f"failed to delete evaluation task {eval_task_id}")
            removed_eval_tasks.append(eval_task_id)

    lora_root = paths.get_loras_dir().resolve()
    if lora_root != root / "models" / "Lora":
        raise HTTPException(409, detail={"code": "invalid_path", "message": "LoRA export root is not inside the workspace"})
    registered = [
        item for item in list_user_loras(config_dir)
        if str(item.get("task_id") or "") == task_id and str(item.get("source") or "user_trained") == "user_trained"
    ]
    paths_to_remove: list[tuple[str, Path]] = []
    removed_loras: list[str] = []
    retained_loras: list[str] = []
    other_training_tasks = [
        row for row in sched.list_tasks(limit=10000, kind=TK.LORA_TRAINING)
        if str(row.get("id")) != task_id
    ]
    for item in registered:
        lora_id = str(item.get("id") or "")
        if any(
            str((other.get("result") or {}).get("metadata", {}).get("user_lora_id") or "") == lora_id
            for other in other_training_tasks
        ):
            retained_loras.append(lora_id)
            continue
        local_path = str(item.get("local_path") or "").strip()
        if local_path:
            target = (root / local_path).resolve()
            if target != lora_root and lora_root not in target.parents:
                raise HTTPException(409, detail={"code": "invalid_path", "message": f"unsafe LoRA export path: {local_path}"})
            paths_to_remove.append((lora_id, target))
        removed_loras.append(lora_id)
    paths_to_remove.append(("work", work))
    for _, target in paths_to_remove:
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    for lora_id in removed_loras:
        if not delete_user_lora(config_dir, lora_id, project_root=root):
            raise RuntimeError(f"failed to remove registered LoRA {lora_id}")
    if not sched.delete_task(task_id):
        raise RuntimeError(f"failed to delete training task {task_id}")
    return {
        "ok": True,
        "task_id": task_id,
        "removed_loras": removed_loras,
        "retained_loras": retained_loras,
        "removed_evaluation_tasks": removed_eval_tasks,
        "removed_paths": [str(target.relative_to(root)) for _, target in paths_to_remove if not target.exists()],
    }


@router.post("/trainings/{task_id}/evaluate", status_code=202)
async def submit_checkpoint_evaluation(
    task_id: str,
    body: LoraEvaluateEndpointRequest | None = None,
    sched: TaskScheduler = Depends(get_task_scheduler),
):
    """Queue (or force re-queue) the same-condition candidate matrix for a finished training run."""
    from backend.engine.training.lora_checkpoint_evaluation import (
        load_candidate_evaluation,
        mark_evaluation_queued,
    )

    task = sched.get_task(task_id)
    if task is None:
        raise HTTPException(404, detail={"code": "not_found", "message": "training task not found"})
    if not TK.is_lora_training_kind(str(task.get("kind") or "")):
        raise HTTPException(400, detail={"code": "invalid", "message": "task is not a LoRA training task"})
    if str(task.get("status") or "") != "completed":
        raise HTTPException(409, detail={"code": "not_ready", "message": "training must complete before evaluation"})
    work = sched.task_work_dir(task_id)
    existing = load_candidate_evaluation(work)
    req = body or LoraEvaluateEndpointRequest()
    force = bool(req.force)
    priority = req.priority
    eval_id = str((existing or {}).get("evaluation_task_id") or "").strip()
    if eval_id:
        eval_row = sched.get_task(eval_id)
        if eval_row and str(eval_row.get("status") or "") in {"queued", "running"} and not force:
            return {"task": sched.public_task_view(eval_row), "reused": True}
    params = task.get("params") or {}
    model_id = str(params.get("base_model") or task.get("model_id") or "").split(":", 1)[0] or "lora-eval"
    submitted = await sched.submit(
        kind=TK.LORA_EVALUATION,
        model_id=model_id,
        params={
            "training_task_id": task_id,
            "priority": priority,
            "force": force,
        },
        priority=priority,
    )
    mark_evaluation_queued(work, evaluation_task_id=str(submitted["id"]))
    return {"task": submitted, "reused": False}


@router.post("/trainings/{task_id}/register")
def register_training_checkpoint(
    task_id: str,
    body: LoraSelectCheckpointRequest | LoraRegisterRequest,
    sched: TaskScheduler = Depends(get_task_scheduler),
):
    work = sched.task_work_dir(task_id)
    checkpoint = str(getattr(body, "checkpoint", "") or "").strip()
    if not checkpoint:
        raise HTTPException(400, detail={"code": "invalid", "message": "checkpoint is required"})
    ckpt = work / "adapters" / checkpoint
    if not ckpt.is_file():
        raise HTTPException(404, detail={"code": "not_found", "message": f"checkpoint {checkpoint!r} not found"})
    task_row = sched.get_task(task_id) or {}
    if str(task_row.get("status") or "") != "completed":
        raise HTTPException(409, detail={"code": "not_ready", "message": "training must complete before registration"})
    params = task_row.get("params") or {}
    dataset_id = str(params.get("dataset_id") or "").strip()
    paths = _paths()
    from backend.engine.training.lora_checkpoint_evaluation import (
        load_candidate_evaluation,
        mark_selection,
    )
    from backend.engine.training.lora_train_config import publish_trained_adapter

    # Studio path: require a completed candidate matrix. Legacy auto-registered runs may
    # lack the manifest; only allow those when a user LoRA already exists for the task.
    candidate = load_candidate_evaluation(work)
    if candidate is None:
        raise HTTPException(
            409,
            detail={
                "code": "evaluation_required",
                "message": "Run checkpoint candidate evaluation and pick a checkpoint before registering",
            },
        )
    if str(candidate.get("status") or "") == "registered" and candidate.get("user_lora_id"):
        entry = next(
            (
                item
                for item in list_user_loras(paths.get_workspace_config_dir())
                if item["id"] == candidate["user_lora_id"]
            ),
            None,
        )
        if entry is not None and str(candidate.get("selected_checkpoint") or "") == checkpoint:
            return {"user_lora": entry, "adapter_path": entry.get("local_path"), "already_registered": True}
    if str(candidate.get("status") or "") != "ready":
        raise HTTPException(
            409,
            detail={
                "code": "evaluation_not_ready",
                "message": f"Candidate evaluation status is {candidate.get('status')!r}; wait or retry evaluation",
            },
        )

    try:
        checkpoint_meta = json.loads(ckpt.with_suffix(".json").read_text(encoding="utf-8")) if ckpt.with_suffix(".json").is_file() else {}
        name = str(getattr(body, "name", "") or "").strip() or str(candidate.get("output_name") or f"trained-{task_id[-8:]}")
        published = publish_trained_adapter(
            final_path=ckpt,
            output_name=name,
            dest_dir=paths.get_loras_dir(),
            config_dir=paths.get_workspace_config_dir(),
            params=params,
            checkpoint_meta=checkpoint_meta,
            dataset_meta=dataset_store._dataset_meta(paths.get_project_root(), dataset_id) if dataset_id else {},
            auto_register=True,
            task_id=task_id,
        )
        mark_selection(work, checkpoint=checkpoint, user_lora_id=published["user_lora_id"])
        entry = next(item for item in list_user_loras(paths.get_workspace_config_dir()) if item["id"] == published["user_lora_id"])
        return {"user_lora": entry, "adapter_path": published["adapter_path"]}
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as e:
        raise HTTPException(409, detail={"code": "adapter_export_failed", "message": str(e)}) from e
