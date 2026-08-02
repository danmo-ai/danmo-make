"""Isolated subprocess worker for VLM batch audits (keeps API process off the Metal crash path)."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    mode = str(job.get("mode") or "audit").strip().lower()
    image_paths = [Path(p) for p in job.get("image_paths") or []]
    if not image_paths:
        raise RuntimeError("VLM worker job has no image_paths")
    model_dir = Path(str(job.get("model_dir") or ""))
    if not model_dir.is_dir():
        raise RuntimeError(f"VLM model_dir not found: {model_dir}")

    if mode == "lora_caption":
        from backend.engine.training.lora_auto_caption import caption_dataset_images_batch

        captions = caption_dataset_images_batch(
            image_paths,
            model_dir,
            audit_kind=str(job.get("audit_kind") or "concept"),
            subject_name=str(job.get("subject_name") or ""),
        )
        return {"captions": captions}

    from backend.engine.llm.vision_mlx import analyze_image_files_batch

    instruction = str(job.get("instruction") or "").strip()
    if not instruction:
        raise RuntimeError("VLM audit job missing instruction")
    max_tokens = int(job.get("max_tokens") or 200)
    temperature = float(job.get("temperature") or 0.2)
    texts = analyze_image_files_batch(
        image_paths,
        model_dir,
        instruction=instruction,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {"texts": texts}


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m backend.engine.llm.vlm_audit_worker <job.json>", file=sys.stderr)
        return 2
    job_path = Path(args[0])
    job = json.loads(job_path.read_text(encoding="utf-8"))
    result_path = Path(str(job.get("result_path") or ""))
    try:
        result = run_job(job)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return 0
    except Exception as exc:
        payload = {"error": str(exc), "traceback": traceback.format_exc()[-4000:]}
        try:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
