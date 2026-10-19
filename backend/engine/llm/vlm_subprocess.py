"""Spawn VLM audits in a child process so MLX-VLM crashes do not take down the API server."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _worker_exit_error(*, returncode: int | None, stderr: bytes | None, stdout: bytes | None) -> RuntimeError:
    tail = (stderr or b"").decode("utf-8", errors="replace")[-2000:]
    out = (stdout or b"").decode("utf-8", errors="replace")[-500:]
    code = returncode if returncode is not None else "?"
    if returncode == -9:
        return RuntimeError(
            "VLM audit worker was killed (SIGKILL). Unified memory may be full — "
            "close other GPU apps, lower mlx_memory_limit, or retry with fewer sample images. "
            f"stderr={tail!r} stdout={out!r}"
        )
    if returncode == -11 or returncode == -6 or returncode == -10:
        return RuntimeError(
            "VLM worker crashed (native MLX/VLM fault). "
            "Retry after closing other MLX workloads or restart the backend. "
            f"stderr={tail!r} stdout={out!r}"
        )
    return RuntimeError(
        f"VLM audit worker failed (exit code {code}). stderr={tail!r} stdout={out!r}"
    )


async def run_vlm_audit_subprocess(
    *,
    image_paths: list[Path],
    model_dir: Path,
    instruction: str,
    max_tokens: int = 200,
    temperature: float = 0.2,
    worker_memory_gb: int | None = None,
) -> list[str]:
    if not image_paths:
        return []

    payload = await _run_vlm_worker_job(
        {
            "mode": "audit",
            "image_paths": [str(p) for p in image_paths],
            "model_dir": str(model_dir),
            "instruction": instruction,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        worker_memory_gb=worker_memory_gb,
    )
    texts = payload.get("texts")
    if not isinstance(texts, list):
        raise RuntimeError("VLM audit worker returned invalid payload")
    return [str(t) for t in texts]


_LORA_CAPTION_CHUNK_SIZE = 24


async def _run_vlm_worker_job(job: dict[str, Any], *, worker_memory_gb: int | None = None) -> dict[str, Any]:
    work_dir = Path(tempfile.gettempdir()) / "dq_vlm_audit" / uuid.uuid4().hex
    work_dir.mkdir(parents=True, exist_ok=True)
    result_path = work_dir / "result.json"
    job = dict(job)
    job["result_path"] = str(result_path)
    job_path = work_dir / "job.json"
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
        "backend.engine.llm.vlm_audit_worker",
        str(job_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=str(repo),
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        detail = ""
        if result_path.is_file():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                detail = str(payload.get("error") or "")
            except Exception:
                detail = ""
        err = _worker_exit_error(returncode=proc.returncode, stderr=stderr, stdout=stdout)
        if detail:
            raise RuntimeError(f"{err} ({detail})") from None
        raise err

    if not result_path.is_file():
        raise RuntimeError("VLM worker finished without result.json")

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload


async def run_lora_caption_subprocess(
    *,
    image_paths: list[Path],
    model_dir: Path,
    audit_kind: str = "concept",
    subject_name: str = "",
    worker_memory_gb: int | None = None,
    chunk_size: int = _LORA_CAPTION_CHUNK_SIZE,
) -> list[str]:
    """Run LoRA auto-caption in an isolated child process (chunked batch loads)."""
    if not image_paths:
        return []

    size = max(1, int(chunk_size))
    captions: list[str] = []
    for start in range(0, len(image_paths), size):
        chunk = image_paths[start : start + size]
        payload = await _run_vlm_worker_job(
            {
                "mode": "lora_caption",
                "image_paths": [str(p) for p in chunk],
                "model_dir": str(model_dir),
                "audit_kind": audit_kind,
                "subject_name": subject_name,
            },
            worker_memory_gb=worker_memory_gb,
        )
        chunk_caps = payload.get("captions")
        if not isinstance(chunk_caps, list) or len(chunk_caps) != len(chunk):
            raise RuntimeError("VLM caption worker returned invalid payload")
        captions.extend(str(c) for c in chunk_caps)
    return captions

