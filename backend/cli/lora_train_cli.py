"""LoRA training CLI — mirrors POST /api/loras/trainings."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from backend.cli.base import build_exec_context, engine_session
from backend.core.contracts import LoraTrainingRequest


def train(
    *,
    base_model: str,
    dataset_id: str,
    progress_prompt: str,
    preset: str = "standard",
    output_name: str = "",
    project_root: Path | None = None,
    **overrides,
) -> dict:
    with engine_session(project_root) as ctx:
        from backend.engine.danqing_lora_train_engine import DanQingLoraTrainEngine

        train_engine = DanQingLoraTrainEngine(
            ctx.path_resolver,
            ctx.model_registry,
            ctx.runtimes,
        )
        # CLI batch jobs register the final checkpoint unless the caller opts out.
        if "auto_register" not in overrides:
            overrides = {**overrides, "auto_register": True}
        req = LoraTrainingRequest(
            base_model=base_model,
            dataset_id=dataset_id,
            progress_prompt=progress_prompt,
            preset=preset,  # type: ignore[arg-type]
            output_name=output_name,
            **{k: v for k, v in overrides.items() if v is not None},
        )
        work = ctx.path_resolver.get_outputs_dir() / "cli_train" / f"run_{int(time.time())}"
        work.mkdir(parents=True, exist_ok=True)
        exec_ctx = build_exec_context(
            work_dir=work,
            asset_store=ctx.asset_store,
            on_progress=lambda ev: print(
                f"  [{ev.step}/{ev.total}] {ev.progress * 100:.1f}% {ev.message or ''}"
            ),
            on_log=lambda ev: print(f"  [{ev.level}] {ev.message}"),
        )

        async def _run():
            return await train_engine.train(req, exec_ctx)

        result = asyncio.run(_run())
        print(f"Training done: {result.metadata}")
        return dict(result.metadata or {})
