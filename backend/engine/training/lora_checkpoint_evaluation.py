"""Same-condition LoRA checkpoint candidate evaluation (no auto-scoring).

All checkpoints for one training run share a fixed prompt × seed matrix and the
training run's portable inference recipe. The user picks one checkpoint visually;
``best_adapters`` only means lowest validation loss, never best likeness.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from backend.core.contracts import (
    AdapterRef,
    ExecutionContext,
    ImageGenerationRequest,
    LoraEvaluationRequest,
)

CANDIDATE_EVAL_FILENAME = "candidate_evaluation.json"
CANDIDATE_DIRNAME = "candidates"
CHECKPOINT_ADAPTER_PREFIX = "training-checkpoint:"
EVAL_SEEDS: tuple[int, ...] = (42, 3407)

_CONCEPT_PROMPT_TEMPLATES: tuple[tuple[str, str], ...] = (
    (
        "closeup",
        "a clear close-up portrait photo of {identity}, facing the camera, "
        "sharp eyes, natural skin texture, soft studio light",
    ),
    (
        "three_quarter",
        "a three-quarter view portrait of {identity}, looking slightly aside, "
        "natural daylight, detailed face",
    ),
    (
        "outdoor_half",
        "an outdoor half-body photo of {identity}, soft daylight, "
        "shallow depth of field, candid expression",
    ),
    (
        "warm_light",
        "a portrait of {identity} under warm golden-hour light, "
        "gentle smile, high detail face",
    ),
)

_STYLE_PROMPT_TEMPLATES: tuple[tuple[str, str], ...] = (
    (
        "subject_a",
        "a detailed illustration in the style of {identity}, "
        "clean composition, rich texture",
    ),
    (
        "subject_b",
        "a landscape scene in the style of {identity}, "
        "cohesive palette, clear artistic look",
    ),
    (
        "subject_c",
        "a character portrait in the style of {identity}, "
        "consistent rendering and brushwork",
    ),
    (
        "subject_d",
        "an everyday object still life in the style of {identity}, "
        "matching color and line quality",
    ),
)

_SAFE_CHECKPOINT_RE = re.compile(r"^[A-Za-z0-9._-]+\.safetensors$")
_STEP_FROM_NAME_RE = re.compile(r"^(\d+)_adapters\.safetensors$")


def training_checkpoint_adapter_id(training_task_id: str, checkpoint: str) -> str:
    return f"{CHECKPOINT_ADAPTER_PREFIX}{training_task_id}:{checkpoint}"


def parse_training_checkpoint_adapter_id(adapter_id: str) -> tuple[str, str] | None:
    raw = str(adapter_id or "").strip()
    if not raw.startswith(CHECKPOINT_ADAPTER_PREFIX):
        return None
    rest = raw[len(CHECKPOINT_ADAPTER_PREFIX) :]
    task_id, sep, checkpoint = rest.partition(":")
    if not sep or not task_id.strip() or not checkpoint.strip():
        return None
    return task_id.strip(), checkpoint.strip()


def resolve_training_checkpoint_path(
    project_root: Path,
    *,
    training_task_id: str,
    checkpoint: str,
) -> Path:
    """Resolve a work-dir adapter with a hard path boundary (no arbitrary paths)."""
    task_id = str(training_task_id or "").strip()
    filename = str(checkpoint or "").strip()
    if not task_id or "/" in task_id or "\\" in task_id or ".." in task_id:
        raise RuntimeError(f"Invalid training task id for checkpoint load: {training_task_id!r}")
    if not _SAFE_CHECKPOINT_RE.match(filename):
        raise RuntimeError(f"Invalid checkpoint filename: {checkpoint!r}")
    work_root = (project_root / "outputs" / "work").resolve()
    adapters = (work_root / task_id / "adapters").resolve()
    if adapters.parent.parent != work_root:
        raise RuntimeError(f"Checkpoint path escaped work root: {adapters}")
    path = (adapters / filename).resolve()
    if path.parent != adapters:
        raise RuntimeError(f"Checkpoint path escaped adapters dir: {path}")
    if not path.is_file():
        raise RuntimeError(f"Training checkpoint not found: {filename}")
    return path


def checkpoint_step_label(filename: str) -> str:
    name = Path(filename).name
    if name == "best_adapters.safetensors":
        return "best (val loss)"
    if name == "final_adapters.safetensors":
        return "final"
    m = _STEP_FROM_NAME_RE.match(name)
    if m:
        return f"step {int(m.group(1))}"
    return name


def list_candidate_checkpoints(adapter_dir: Path) -> list[dict[str, Any]]:
    """Deduped candidate list: periodic steps + best + final, ordered by step."""
    if not adapter_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(path: Path, *, kind: str, step: int | None) -> None:
        if not path.is_file():
            return
        name = path.name
        if name in seen:
            return
        seen.add(name)
        meta: dict[str, Any] = {}
        meta_path = path.with_suffix(".json")
        if meta_path.is_file():
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    meta = payload
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                meta = {}
        iter_step = step
        if iter_step is None and meta.get("iteration") is not None:
            try:
                iter_step = int(meta["iteration"])
            except (TypeError, ValueError):
                iter_step = None
        rows.append(
            {
                "checkpoint": name,
                "kind": kind,
                "step": iter_step,
                "label": checkpoint_step_label(name),
                "val_loss": meta.get("val_loss"),
                "path": str(path),
            }
        )

    for path in sorted(adapter_dir.glob("*_adapters.safetensors")):
        name = path.name
        if name in {"best_adapters.safetensors", "final_adapters.safetensors", "fused_adapters.safetensors"}:
            continue
        m = _STEP_FROM_NAME_RE.match(name)
        step = int(m.group(1)) if m else None
        _add(path, kind="periodic", step=step)

    _add(adapter_dir / "best_adapters.safetensors", kind="best", step=None)
    _add(adapter_dir / "final_adapters.safetensors", kind="final", step=None)

    def _sort_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
        step = row.get("step")
        kind = str(row.get("kind") or "")
        # Within the same step: periodic → best → final.
        kind_order = {"periodic": 0, "best": 1, "final": 2}.get(kind, 3)
        if isinstance(step, int):
            return (0, step, kind_order, str(row.get("checkpoint") or ""))
        return (1, kind_order, 0, str(row.get("checkpoint") or ""))

    return sorted(rows, key=_sort_key)


def build_eval_prompts(
    *,
    dataset_kind: str,
    trigger_word: str,
    training_caption: str = "",
) -> list[dict[str, str]]:
    kind = str(dataset_kind or "concept").strip().lower()
    identity = (trigger_word or "").strip() or (training_caption or "").strip() or "the subject"
    templates = _STYLE_PROMPT_TEMPLATES if kind == "style" else _CONCEPT_PROMPT_TEMPLATES
    out: list[dict[str, str]] = []
    for prompt_id, template in templates:
        out.append(
            {
                "id": prompt_id,
                "prompt": template.format(identity=identity),
            }
        )
    return out


def candidate_eval_path(work_dir: Path) -> Path:
    return Path(work_dir) / CANDIDATE_EVAL_FILENAME


def load_candidate_evaluation(work_dir: Path) -> dict[str, Any] | None:
    path = candidate_eval_path(work_dir)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid candidate evaluation payload: {path}")
    return data


def save_candidate_evaluation(work_dir: Path, payload: dict[str, Any]) -> Path:
    path = candidate_eval_path(work_dir)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_adapters_lora_config(adapter_dir: Path, config: dict[str, Any]) -> Path:
    """Shared load contract for every unregistered checkpoint in ``adapter_dir``."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    path = adapter_dir / "lora_config.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def finish_training_without_register(
    *,
    work_dir: Path,
    adapter_dir: Path,
    final_path: Path,
    output_name: str,
    params: dict[str, Any],
    dataset_meta: dict[str, Any],
    loss_history: list[dict[str, Any]],
    training_caption: str,
    task_id: str,
) -> dict[str, Any]:
    """Persist export contract beside adapters; leave registration to the user."""
    from backend.engine.training.lora_train_config import adapter_export_config

    if not final_path.is_file():
        raise RuntimeError(f"Final adapters missing after training: {final_path}")
    config = adapter_export_config(
        params=params,
        checkpoint_meta=json.loads(final_path.with_suffix(".json").read_text(encoding="utf-8"))
        if final_path.with_suffix(".json").is_file()
        else {},
        dataset_meta=dataset_meta,
    )
    config["task_id"] = task_id
    write_adapters_lora_config(adapter_dir, config)
    checkpoints = list_candidate_checkpoints(adapter_dir)
    if not checkpoints:
        raise RuntimeError("Training finished but no candidate checkpoints were saved")
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in (output_name or "").strip())[:64]
    slug = slug.strip("-") or f"face-lora-{task_id[-8:]}"
    pending = {
        "status": "pending_evaluation",
        "training_task_id": task_id,
        "output_name": slug,
        "checkpoints": checkpoints,
        "inference": config.get("inference"),
        "training_contract": {
            key: config.get(key)
            for key in (
                "base_model",
                "family",
                "lora_rank",
                "lora_alpha",
                "lora_blocks",
                "train_type",
                "trigger_word",
                "training_caption",
            )
        },
        "selected_checkpoint": None,
        "user_lora_id": "",
        "evaluation_task_id": "",
    }
    save_candidate_evaluation(work_dir, pending)
    return {
        "adapter_path": str(final_path),
        "user_lora_id": "",
        "output_name": slug,
        "loss_history": loss_history,
        "training_caption": training_caption,
        "needs_candidate_evaluation": True,
        "checkpoints": [row["checkpoint"] for row in checkpoints],
    }


def mark_evaluation_queued(work_dir: Path, *, evaluation_task_id: str) -> dict[str, Any]:
    payload = load_candidate_evaluation(work_dir) or {}
    payload["status"] = "evaluating"
    payload["evaluation_task_id"] = evaluation_task_id
    payload["error"] = ""
    save_candidate_evaluation(work_dir, payload)
    return payload


def mark_selection(
    work_dir: Path,
    *,
    checkpoint: str,
    user_lora_id: str,
) -> dict[str, Any]:
    payload = load_candidate_evaluation(work_dir)
    if payload is None:
        raise RuntimeError("Candidate evaluation manifest is missing; run evaluation first")
    if str(payload.get("status") or "") != "ready":
        raise RuntimeError(
            f"Candidate evaluation is not ready for registration (status={payload.get('status')!r})"
        )
    names = {str(row.get("checkpoint") or "") for row in (payload.get("checkpoints") or [])}
    if checkpoint not in names:
        raise RuntimeError(f"Checkpoint {checkpoint!r} is not part of this training run's candidates")
    cells = payload.get("cells") or []
    if not cells:
        raise RuntimeError("Candidate grid is empty; refuse to register without evaluated images")
    missing = [
        cell
        for cell in cells
        if str(cell.get("checkpoint") or "") == checkpoint and str(cell.get("status") or "") != "ok"
    ]
    if missing:
        raise RuntimeError(
            f"Checkpoint {checkpoint!r} has incomplete candidate images; re-run evaluation before registering"
        )
    payload["status"] = "registered"
    payload["selected_checkpoint"] = checkpoint
    payload["user_lora_id"] = user_lora_id
    save_candidate_evaluation(work_dir, payload)
    return payload


def _eval_size(inference: dict[str, Any] | None, params: dict[str, Any]) -> str:
    resolution = params.get("resolution")
    if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
        return f"{int(resolution[0])}x{int(resolution[1])}"
    return "1024x1024"


def _build_generation_request(
    *,
    prompt: str,
    seed: int,
    inference: dict[str, Any],
    params: dict[str, Any],
    training_task_id: str,
    checkpoint: str,
) -> ImageGenerationRequest:
    model = str(inference.get("model") or params.get("base_model") or "").split(":", 1)[0]
    if not model:
        raise RuntimeError("Candidate evaluation requires an inference model")
    weight = float(inference.get("lora_weight") or 1.0)
    adapters = [
        AdapterRef(id=training_checkpoint_adapter_id(training_task_id, checkpoint), weight=weight)
    ]
    for extra in inference.get("extra_adapters") or []:
        adapters.append(AdapterRef(id=str(extra), weight=1.0))
    scheduler = inference.get("scheduler")
    return ImageGenerationRequest(
        model=model,
        prompt=prompt,
        size=_eval_size(inference, params),
        n=1,
        steps=int(inference.get("steps") or 30),
        guidance=float(inference.get("guidance") or 0.0),
        seed=int(seed),
        scheduler=str(scheduler) if scheduler else None,
        adapters=adapters,
        metadata={
            "lora_candidate_eval": True,
            "training_task_id": training_task_id,
            "checkpoint": checkpoint,
        },
    )


def run_lora_checkpoint_evaluation(
    request: LoraEvaluationRequest,
    exec_ctx: ExecutionContext,
    *,
    registry: Any,
    project_root: Path,
    runtime: Any,
    path_resolver: Any,
    model_cache: Any | None = None,
) -> dict[str, Any]:
    """Generate the full prompt×seed×checkpoint matrix into the training work dir."""
    from backend.engine.sessions.engine_dispatch import dispatch_image_create
    from backend.engine.training import dataset_store
    from backend.engine.training.lora_train_config import adapter_export_config, training_inference_config
    from backend.engine.memory_policy import unload_model_cache_if_present

    training_task_id = str(request.training_task_id or "").strip()
    if not training_task_id:
        raise RuntimeError("training_task_id is required for LoRA checkpoint evaluation")
    train_work = path_resolver.get_outputs_dir() / "work" / training_task_id
    if not train_work.is_dir():
        raise RuntimeError(f"Training work dir not found: {training_task_id}")
    adapter_dir = train_work / "adapters"
    existing = load_candidate_evaluation(train_work)
    if existing and str(existing.get("status") or "") == "ready" and not request.force:
        return existing

    # Training params live on the training task row; the evaluation worker passes them via
    # a sidecar written when the follow-up was queued, or we read from candidate payload.
    params_path = train_work / "lora_worker_job.json"
    if not params_path.is_file():
        raise RuntimeError(f"Training job snapshot missing for {training_task_id}")
    job = json.loads(params_path.read_text(encoding="utf-8"))
    params = dict(job.get("request") or {})
    if not params:
        raise RuntimeError(f"Training request snapshot empty for {training_task_id}")

    dataset_id = str(params.get("dataset_id") or "").strip()
    dataset_meta = dataset_store._dataset_meta(project_root, dataset_id) if dataset_id else {}
    inference = training_inference_config(params)
    if inference is None:
        raise RuntimeError("No portable inference recipe for this training run; refuse evaluation")
    export_cfg = adapter_export_config(params=params, dataset_meta=dataset_meta)
    write_adapters_lora_config(adapter_dir, {**export_cfg, "task_id": training_task_id})

    checkpoints = list_candidate_checkpoints(adapter_dir)
    if not checkpoints:
        raise RuntimeError("No candidate checkpoints available for evaluation")
    prompts = build_eval_prompts(
        dataset_kind=str(dataset_meta.get("kind") or "concept"),
        trigger_word=str(dataset_meta.get("trigger_word") or ""),
        training_caption=str(params.get("progress_prompt") or ""),
    )
    out_dir = train_work / CANDIDATE_DIRNAME
    if request.force and out_dir.is_dir():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells: list[dict[str, Any]] = []
    total = max(1, len(checkpoints) * len(prompts) * len(EVAL_SEEDS))
    done = 0
    on_log = getattr(exec_ctx, "on_log", None)

    def _log(level: str, message: str) -> None:
        if on_log is None:
            return
        from backend.core.contracts import LogEvent

        on_log(LogEvent(level=level, message=message))

    _log(
        "info",
        f"Evaluating {len(checkpoints)} checkpoints × {len(prompts)} prompts × "
        f"{len(EVAL_SEEDS)} seeds (same inference recipe)",
    )

    for ck in checkpoints:
        ck_name = str(ck["checkpoint"])
        # Fresh weights per checkpoint — clear DiT cache so adapters cannot leak.
        unload_model_cache_if_present()
        if runtime is not None and hasattr(runtime, "clear_cache"):
            runtime.clear_cache()
        for prompt_row in prompts:
            for seed in EVAL_SEEDS:
                exec_ctx.cancel_token.raise_if_cancelled()
                rel_name = f"{Path(ck_name).stem}__{prompt_row['id']}__s{seed}.png"
                out_path = out_dir / rel_name
                cell = {
                    "checkpoint": ck_name,
                    "checkpoint_label": ck.get("label"),
                    "step": ck.get("step"),
                    "prompt_id": prompt_row["id"],
                    "prompt": prompt_row["prompt"],
                    "seed": seed,
                    "filename": rel_name,
                    "status": "pending",
                }
                try:
                    gen_req = _build_generation_request(
                        prompt=prompt_row["prompt"],
                        seed=seed,
                        inference=inference,
                        params=params,
                        training_task_id=training_task_id,
                        checkpoint=ck_name,
                    )
                    # Isolate outputs under the evaluation task work dir, then copy into
                    # the training candidate folder so the training artifacts API can serve them.
                    result = dispatch_image_create(
                        runtime=runtime,
                        registry=registry,
                        asset_store=exec_ctx.asset_store,
                        model_cache=model_cache,
                        project_root=project_root,
                        request=gen_req,
                        exec_ctx=exec_ctx,
                        on_progress=None,
                        on_log=lambda level, msg: _log(str(level), str(msg)),
                    )
                    if result is None:
                        raise RuntimeError("Candidate generation cancelled")
                    if isinstance(result, list):
                        src_path = Path(result[0][0])
                    else:
                        src_path = Path(result[0])
                    if not src_path.is_file():
                        raise RuntimeError(f"Candidate image missing after generation: {src_path}")
                    shutil.copy2(src_path, out_path)
                    cell["status"] = "ok"
                except Exception as exc:
                    cell["status"] = "failed"
                    cell["error"] = str(exc)
                    cells.append(cell)
                    failed_payload = {
                        "status": "failed",
                        "training_task_id": training_task_id,
                        "evaluation_task_id": exec_ctx.task_id,
                        "output_name": (existing or {}).get("output_name") or training_task_id,
                        "checkpoints": checkpoints,
                        "prompts": prompts,
                        "seeds": list(EVAL_SEEDS),
                        "inference": inference,
                        "cells": cells,
                        "selected_checkpoint": None,
                        "user_lora_id": "",
                        "error": str(exc),
                    }
                    if existing and existing.get("output_name"):
                        failed_payload["output_name"] = existing["output_name"]
                    save_candidate_evaluation(train_work, failed_payload)
                    _log(
                        "error",
                        f"Candidate failed checkpoint={ck_name} prompt={prompt_row['id']} seed={seed}: {exc}",
                    )
                    raise
                finally:
                    done += 1
                    if exec_ctx.on_progress is not None:
                        from backend.core.contracts import ProgressEvent

                        exec_ctx.on_progress(
                            ProgressEvent(
                                progress=done / total,
                                step=done,
                                total=total,
                                message=f"{ck_name} / {prompt_row['id']} / seed {seed}",
                                phase="candidate_eval",
                            )
                        )
                cells.append(cell)
        unload_model_cache_if_present()
        if runtime is not None and hasattr(runtime, "clear_cache"):
            runtime.clear_cache()

    payload = {
        "status": "ready",
        "training_task_id": training_task_id,
        "evaluation_task_id": exec_ctx.task_id,
        "output_name": (existing or {}).get("output_name") or training_task_id,
        "checkpoints": checkpoints,
        "prompts": prompts,
        "seeds": list(EVAL_SEEDS),
        "inference": inference,
        "training_contract": {
            key: export_cfg.get(key)
            for key in (
                "base_model",
                "family",
                "lora_rank",
                "lora_alpha",
                "lora_blocks",
                "train_type",
                "trigger_word",
                "training_caption",
            )
        },
        "cells": cells,
        "selected_checkpoint": None,
        "user_lora_id": "",
        "error": "",
    }
    # Preserve slug chosen at training finish when present.
    if existing and existing.get("output_name"):
        payload["output_name"] = existing["output_name"]
    save_candidate_evaluation(train_work, payload)
    _log("success", f"Candidate evaluation ready ({len(cells)} images)")
    return payload
