"""Spawn LoRA training in a child process so MLX does not block the API server."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from backend.core.contracts import ExecutionContext, LoraTrainingRequest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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
    env["PYTHONPATH"] = str(repo) if not py_path else f"{repo}{os.pathsep}{py_path}"
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
