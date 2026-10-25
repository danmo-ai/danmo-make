"""Synchronous face-LoRA preflight shared by REST and CLI submission."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.engine.training import dataset_store
from backend.engine.training.face_crop import opencv_face_detector_available


def resolve_request_caption_mode(root: Path, request: Any) -> str:
    pairs = dataset_store.load_training_pairs(root, request.dataset_id)
    meta = dataset_store._dataset_meta(root, request.dataset_id)
    requested = str(getattr(request, "caption_mode", None) or "auto").lower()
    if requested in {"unified", "per_image"}:
        return requested
    return dataset_store.detect_caption_mode(pairs, dataset_meta=meta)


def _assert_resume_same_identity(
    *,
    root: Path,
    request: Any,
    pairs: list[tuple[Path, str]],
    caption: str,
    old_params: dict[str, Any],
    new_cfg: dict[str, Any],
) -> None:
    """Fail loud when a resume request changes the identity binding or recipe."""
    from backend.engine.training.lora_train_config import resolve_training_config

    old_dataset = str(old_params.get("dataset_id") or "")
    if old_dataset and old_dataset != request.dataset_id:
        raise ValueError("Resume requires the same dataset; create a fresh run when changing identity data.")
    if str(old_params.get("preset") or "") != str(request.preset or ""):
        raise ValueError("Resume requires the same preset; select the source run's preset.")
    old_cfg = resolve_training_config(old_params)
    for key in ("lora_rank", "lora_alpha", "lora_blocks", "lora_module_keys", "train_type"):
        if old_cfg.get(key) != new_cfg.get(key):
            raise ValueError(f"Resume checkpoint is incompatible: {key} changed from the source run.")
    old_pairs = dataset_store.load_training_pairs(root, request.dataset_id)
    old_cap = str(old_params.get("progress_prompt") or "")
    old_mode = str(old_params.get("caption_mode") or "auto")
    _, old_caption = dataset_store.load_training_pairs_unified(
        root, request.dataset_id, progress_prompt=old_cap, caption_mode=old_mode
    )
    if old_caption != caption or old_params.get("progress_prompt") != request.progress_prompt:
        raise ValueError("Resume requires the original trigger/training caption, not a changed identity binding.")
    source_fp = str((old_params.get("metadata") or {}).get("dataset_fingerprint") or "").strip()
    current_fp = dataset_store.training_dataset_fingerprint(pairs)
    if not source_fp:
        raise ValueError("Source task has no immutable dataset fingerprint; start a fresh training task.")
    if source_fp != current_fp:
        raise ValueError("Dataset bytes/captions changed since the source run; start a fresh task.")
    if len(old_pairs) != len(pairs):
        raise ValueError("The dataset changed since the source run; start a fresh training task.")


def validate_training_request(root: Path, request: Any, *, sched: Any | None = None) -> None:
    """Reject an unusable identity run before a long GPU task is queued.

    Style datasets remain supported; concept datasets require the detector because
    silently substituting a cover crop changes the face learning objective.
    """
    from backend.engine.training.crop import resolve_training_resolution
    from backend.engine.training.lora_train_config import resolve_training_config
    from backend.engine.training.lora_train_runtime_mlx import resume_checkpoint_incompatibility

    meta = dataset_store._dataset_meta(root, request.dataset_id)
    cfg = resolve_training_config(request.model_dump())
    resolution = resolve_training_resolution(request.base_model, cfg, preset=request.preset)
    pairs, caption = dataset_store.load_training_pairs_unified(
        root,
        request.dataset_id,
        progress_prompt=request.progress_prompt,
        caption_mode=str(getattr(request, "caption_mode", None) or "auto"),
    )
    kind = str(meta.get("kind") or "concept")
    trigger = str(meta.get("trigger_word") or "").strip()
    if kind == "concept":
        if not trigger:
            raise ValueError("Face LoRA requires a unique trigger word on the dataset.")
        if trigger not in caption:
            raise ValueError(
                f"Training/verification caption must contain the exact trigger word {trigger!r}."
            )
        if str(getattr(request, "face_crop", None) or "auto") == "off":
            raise ValueError("Face LoRA requires face_crop=auto or on; off is only for style datasets.")
        available, reason = opencv_face_detector_available()
        if not available:
            raise ValueError(
                f"Face LoRA requires YuNet/OpenCV face detection; {reason}. "
                "Install the face preprocessing dependency before submitting."
            )
        from backend.engine.training.face_crop import audit_dataset_faces

        report = audit_dataset_faces(
            [path for path, _ in pairs],
            project_root=root,
            resolution=resolution,
        )
        if not report.get("available"):
            raise ValueError(f"Face detection is required for identity training: {report.get('reason')}")
        counts = report.get("counts") or {}
        errors = []
        if counts.get("none"):
            errors.append(f"{counts['none']} image(s) have no detectable face")
        if counts.get("multi"):
            errors.append(f"{counts['multi']} image(s) contain competing faces")
        if counts.get("tiny"):
            errors.append(f"{counts['tiny']} image(s) have faces too small for the selected resolution")
        if counts.get("error"):
            errors.append(f"{counts['error']} image(s) failed face detection")
        if errors:
            raise ValueError("Identity dataset preflight failed: " + "; ".join(errors))

    # Persist the actual conditioning fingerprint in the task row. A resumed run
    # must compare against this immutable snapshot, not the mutable dataset library.
    request.metadata = {
        **(getattr(request, "metadata", None) or {}),
        "dataset_fingerprint": dataset_store.training_dataset_fingerprint(pairs),
        "resolved_caption_mode": resolve_request_caption_mode(root, request),
        "training_resolution": list(resolution),
    }

    new_cfg = resolve_training_config(request.model_dump())
    old_params: dict[str, Any] | None = None
    resume_task_id = str(getattr(request, "resume_task_id", None) or "").strip()
    resume_checkpoint = str(getattr(request, "resume_checkpoint", None) or "").strip()
    resume_from = str(getattr(request, "resume_from", None) or "").strip()
    if resume_checkpoint and not resume_task_id and not resume_from:
        raise ValueError("resume_checkpoint requires resume_task_id.")
    if resume_task_id:
        if sched is None:
            raise ValueError("resume_task_id requires the task scheduler so identity continuity can be verified.")
        source = sched.get_task(resume_task_id)
        if source is None:
            raise ValueError(f"Resume source task not found: {resume_task_id}")
        old_params = source.get("params") or {}
        if not isinstance(old_params, dict):
            raise ValueError(f"Resume source task {resume_task_id} has no usable params.")
        # Canonical UI resume and direct resume_from+task_id must share the same gate.
        _assert_resume_same_identity(
            root=root,
            request=request,
            pairs=pairs,
            caption=caption,
            old_params=old_params,
            new_cfg=new_cfg,
        )

    if resume_from:
        reason = resume_checkpoint_incompatibility(
            base_model=request.base_model,
            adapter_path=Path(resume_from),
            source_task_params=old_params,
            expected_config=new_cfg,
        )
        if reason:
            raise ValueError(reason)


def plan_and_write_training_crops(
    *,
    pairs: list[tuple[Path, str]],
    project_root: Path,
    resolution: tuple[int, int],
    mode: str,
    dataset_meta: dict[str, Any],
    work_dir: Path,
    on_log: Any,
) -> dict[Path, Any]:
    """Apply exactly the YuNet policy shown by preflight and persist its evidence."""
    import json

    from backend.engine.training.face_crop import plan_training_face_crops

    plans = plan_training_face_crops(
        pairs,
        project_root=project_root,
        resolution=resolution,
        mode=mode,
        dataset_meta=dataset_meta,
        on_log=on_log,
    )
    report = {
        "mode": mode,
        "resolution": list(resolution),
        "images": [
            {
                "file": path.name,
                "window": list(plan.window) if plan.window else None,
                "reason": plan.reason,
                "face": {
                    "x": plan.face.x, "y": plan.face.y,
                    "w": plan.face.w, "h": plan.face.h,
                    "score": plan.face.score,
                } if plan.face else None,
            }
            for path, plan in sorted(plans.items(), key=lambda row: str(row[0]))
        ],
    }
    (work_dir / "crop_plan.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return plans
