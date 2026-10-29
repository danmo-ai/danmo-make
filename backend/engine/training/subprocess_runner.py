"""Spawn LoRA training in a child process so MLX does not block the API server."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from backend.core.contracts import ExecutionContext, LoraTrainingRequest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def collect_resume_lineage_work_dirs(resume_from: Path) -> list[Path]:
    """Return resume work dirs oldest→newest ending at ``resume_from``'s training work dir."""
    adapter = Path(resume_from)
    if not adapter.is_file():
        raise RuntimeError(f"Resume checkpoint is missing: {adapter}")
    current = adapter.parent.parent
    seen: set[Path] = set()
    newest_first: list[Path] = []
    while current.is_dir():
        resolved = current.resolve()
        if resolved in seen:
            break
        seen.add(resolved)
        newest_first.append(current)
        prov_path = current / "resume_source.json"
        if not prov_path.is_file():
            break
        try:
            payload = json.loads(prov_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            break
        if not isinstance(payload, dict):
            break
        nxt = Path(str(payload.get("source_work_dir") or "")).expanduser()
        if not nxt.is_dir():
            break
        current = nxt
    return list(reversed(newest_first))


def _copy_adapter_bundle(src_weights: Path, dest_dir: Path, *, overwrite: bool) -> None:
    """Copy one checkpoint weight file and its json/optimizer/rng sidecars when present."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_weights = dest_dir / src_weights.name
    if not dest_weights.exists() or overwrite:
        if src_weights.resolve() != dest_weights.resolve():
            shutil.copy2(src_weights, dest_weights)
    stem = (
        src_weights.name[: -len(".safetensors")]
        if src_weights.name.endswith(".safetensors")
        else src_weights.stem
    )
    sidecar_paths = [
        src_weights.with_suffix(".json"),
        src_weights.parent / f"{stem}_optimizer.safetensors",
        src_weights.parent / f"{stem}_rng.safetensors",
    ]
    for src_side in sidecar_paths:
        if not src_side.is_file():
            continue
        dest_side = dest_dir / src_side.name
        if dest_side.exists() and not overwrite:
            continue
        if src_side.resolve() != dest_side.resolve():
            shutil.copy2(src_side, dest_side)


def copy_resume_checkpoint_lineage(*, resume_from: Path, dest_work: Path) -> dict[str, Any]:
    """Bring prior periodic/best checkpoints into the new resume work dir for the candidate matrix.

    Only overwrites a destination file when copying the exact ``resume_from`` bundle (needed so
    optimizer/RNG match the resume point). Older lineage files fill gaps without clobbering
    checkpoints the new run has already written.
    """
    resume_adapter = Path(resume_from)
    dest = Path(dest_work)
    dest.mkdir(parents=True, exist_ok=True)
    lineage = collect_resume_lineage_work_dirs(resume_adapter)
    if not lineage:
        raise RuntimeError(f"Resume checkpoint work dir missing for {resume_adapter}")

    history_src = lineage[-1] / "loss_history.json"
    if history_src.is_file():
        shutil.copy2(history_src, dest / "loss_history.json")

    dest_adapters = dest / "adapters"
    dest_adapters.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    resume_name = resume_adapter.name
    for work in lineage:
        src_adapters = work / "adapters"
        if not src_adapters.is_dir():
            continue
        for src_weights in sorted(src_adapters.glob("*_adapters.safetensors")):
            # Skip fused exports; candidates are periodic/best/final only.
            if src_weights.name == "fused_adapters.safetensors":
                continue
            overwrite = src_weights.name == resume_name and work.resolve() == lineage[-1].resolve()
            before = (dest_adapters / src_weights.name).is_file()
            _copy_adapter_bundle(src_weights, dest_adapters, overwrite=overwrite)
            if (dest_adapters / src_weights.name).is_file() and (overwrite or not before):
                copied.append(src_weights.name)

    provenance = {
        "source_checkpoint": str(resume_adapter),
        "source_work_dir": str(lineage[-1]),
        "lineage_work_dirs": [str(p) for p in lineage],
        "copied_checkpoints": sorted(set(copied)),
    }
    meta_path = resume_adapter.with_suffix(".json")
    if meta_path.is_file():
        try:
            provenance["source_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    (dest / "resume_source.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return provenance


def _worker_exit_error(*, returncode: int | None, stderr: bytes | None, stdout: bytes | None) -> RuntimeError:
    tail = (stderr or b"").decode("utf-8", errors="replace")[-2000:]
    out = (stdout or b"").decode("utf-8", errors="replace")[-500:]
    code = returncode if returncode is not None else "?"
    if returncode == -9:
        return RuntimeError(
            "LoRA training worker was killed (SIGKILL, exit code -9). "
            "This usually means unified memory pressure: the API process and trainer "
            "competed for MLX Metal RAM, or macOS terminated the child. "
            "Retry with preset quick, lower lora_rank/lora_blocks, close other GPU apps, "
            "or reduce mlx_memory_limit in Settings. "
            f"stderr={tail!r} stdout={out!r}"
        )
    if returncode == -6 or "Insufficient Memory" in tail or "OutOfMemory" in tail:
        return RuntimeError(
            "LoRA training ran out of unified GPU memory (Metal OOM). "
            "Enable QLoRA 4-bit and gradient checkpointing in training settings, "
            "use preset quick, lower resolution/lora_blocks/lora_rank, close other GPU apps, "
            "or reduce mlx_memory_limit in Settings. "
            f"stderr={tail!r} stdout={out!r}"
        )
    return RuntimeError(
        f"LoRA worker exited without result (code={code}). stderr={tail!r} stdout={out!r}"
    )


async def run_lora_training_subprocess(
    *,
    runner_family: str,
    request: LoraTrainingRequest,
    exec_ctx: ExecutionContext,
    bootstrap_root: Path,
    worker_memory_gb: int | None = None,
) -> dict[str, Any]:
    work_dir = Path(exec_ctx.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if request.resume_from:
        resume_path = Path(request.resume_from).resolve()
        work_resolved = work_dir.resolve()
        try:
            same_work = resume_path.is_relative_to(work_resolved)
        except AttributeError:
            same_work = str(resume_path).startswith(str(work_resolved) + "/")
        if same_work:
            # In-place continue: adapters / latent_cache already live in this work_dir.
            provenance = {
                "source_checkpoint": str(resume_path),
                "source_work_dir": str(work_resolved),
                "inplace": True,
            }
            (work_dir / "resume_source.json").write_text(
                json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            copy_resume_checkpoint_lineage(
                resume_from=Path(request.resume_from),
                dest_work=work_dir,
            )
    job_path = work_dir / "lora_worker_job.json"
    result_path = work_dir / "lora_worker_result.json"
    if result_path.is_file():
        result_path.unlink()

    job = {
        "task_id": exec_ctx.task_id,
        "work_dir": str(work_dir),
        "bootstrap_root": str(bootstrap_root),
        "family": runner_family,
        "request": request.model_dump(),
        "result_path": str(result_path),
    }
    job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")

    repo = _repo_root()
    env = dict(__import__("os").environ)
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo) if not py_path else f"{repo}{Path.pathsep}{py_path}"
    if worker_memory_gb is not None:
        env["DANQING_MLX_MEMORY_LIMIT_GB"] = str(int(worker_memory_gb))

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "backend.engine.training.lora_worker",
        str(job_path),
        cwd=str(repo),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if not result_path.is_file():
        raise _worker_exit_error(
            returncode=proc.returncode,
            stderr=stderr,
            stdout=stdout,
        )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("cancelled"):
        raise asyncio.CancelledError()
    if not payload.get("ok"):
        err = str(payload.get("error") or "LoRA worker failed")
        tb = payload.get("traceback")
        if tb:
            err = f"{err}\n{tb}"
        raise RuntimeError(err)
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("LoRA worker returned invalid result payload")
    return result
