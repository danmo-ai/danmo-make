"""
音频 CLI 命令 — 调用 IAudioEngine。

与 REST API 端点一一对应：
  danqing-audio-generate → POST /api/audios/generations → IAudioEngine.generate()
  danqing-audio-edit     → POST /api/audios/edits       → IAudioEngine.edit()
"""
from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

from backend.cli.base import build_exec_context, engine_session
from backend.core.contracts import AudioEditRequest, AudioGenerationRequest


def generate(
    model: str,
    prompt: str,
    *,
    negative_prompt: str = "",
    duration: int | None = None,
    instrumental: bool = False,
    lyrics: str = "",
    vocal_language: str = "",
    vocal_type: str = "",
    bpm: int | None = None,
    key_scale: str = "",
    time_signature: str = "",
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    n: int = 2,
    audio_format: str = "mp3",
    output: str = "",
    project_root: Path | None = None,
) -> str:
    """文生音乐。对应 POST /api/audios/generations。"""
    with engine_session(project_root) as ctx:
        exec_ctx = build_exec_context(
            work_dir=ctx.path_resolver.get_outputs_dir() / "cli_tmp",
            asset_store=ctx.asset_store,
            on_progress=lambda ev: None,
            on_log=lambda ev: print(f"  [{ev.level}] {ev.message}"),
        )

        request = AudioGenerationRequest(
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            duration=duration,
            instrumental=instrumental,
            lyrics=lyrics,
            vocal_language=vocal_language,
            vocal_type=vocal_type,
            bpm=bpm,
            key_scale=key_scale,
            time_signature=time_signature,
            steps=steps,
            guidance=guidance,
            seed=seed,
            n=n,
            audio_format=audio_format,
        )

        if not ctx.audio_engine.supports(model, "create_music"):
            raise RuntimeError(
                f"Model {model!r} does not support text-to-music (create); "
                "check config/models_registry.json actions."
            )

        t0 = time.time()
        result = asyncio.run(ctx.audio_engine.generate(request, exec_ctx))
        elapsed = time.time() - t0

        if result.metadata.get("status") == "cancelled":
            raise RuntimeError("Generation cancelled")

        if output and result.output_paths:
            out = Path(output)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result.output_paths[0], out)
            print(f"[cli] DONE ({elapsed:.1f}s) -> {out}")
            return str(out)

        if result.output_paths:
            print(f"[cli] DONE ({elapsed:.1f}s) -> {result.output_paths[0]}")
            return result.output_paths[0]

        raise RuntimeError("No output generated")


def edit(
    model: str,
    operation: str,
    *,
    source_asset_id: str = "",
    source_audio: str = "",
    prompt: str = "",
    source_fidelity: float = 1.0,
    duration: int | None = None,
    instrumental: bool = False,
    lyrics: str = "",
    vocal_language: str = "",
    vocal_type: str = "",
    bpm: int | None = None,
    key_scale: str = "",
    time_signature: str = "",
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    n: int = 1,
    audio_format: str = "mp3",
    output: str = "",
    project_root: Path | None = None,
) -> str:
    """音频编辑（cover）。对应 POST /api/audios/edits。

    ``source_asset_id`` 或 ``source_audio``（本地 wav/mp3 路径）二选一。
    """
    with engine_session(project_root) as ctx:
        exec_ctx = build_exec_context(
            work_dir=ctx.path_resolver.get_outputs_dir() / "cli_tmp",
            asset_store=ctx.asset_store,
            on_progress=lambda ev: None,
            on_log=lambda ev: print(f"  [{ev.level}] {ev.message}"),
        )

        if source_audio and not source_asset_id:
            src = Path(source_audio)
            if not src.is_file():
                raise FileNotFoundError(f"Source audio not found: {src}")
            mime = "audio/wav" if src.suffix.lower() == ".wav" else "audio/mpeg"
            source_asset_id = ctx.asset_store.create_from_file(
                src,
                kind="audio",
                mime_type=mime,
                source_task_id="",
            )
            print(f"[cli] uploaded {source_audio} → asset {source_asset_id}")

        if not source_asset_id:
            raise ValueError("source_asset_id or source_audio is required")

        request = AudioEditRequest(
            model=model,
            operation=operation,  # type: ignore[arg-type]
            source_asset_id=source_asset_id,
            prompt=prompt,
            source_fidelity=source_fidelity,
            duration=duration,
            instrumental=instrumental,
            lyrics=lyrics,
            vocal_language=vocal_language,
            vocal_type=vocal_type,
            bpm=bpm,
            key_scale=key_scale,
            time_signature=time_signature,
            steps=steps,
            guidance=guidance,
            seed=seed,
            n=n,
            audio_format=audio_format,
        )

        # API action is "edit"; registry verbs are cover/repaint.
        if not ctx.audio_engine.supports(model, "edit"):
            raise RuntimeError(
                f"Model {model!r} does not support audio edit "
                f"(operation={operation!r}); check config/models_registry.json actions."
            )

        t0 = time.time()
        result = asyncio.run(ctx.audio_engine.edit(request, exec_ctx))
        elapsed = time.time() - t0

        if result.metadata.get("status") == "cancelled":
            raise RuntimeError("Edit cancelled")

        if output and result.output_paths:
            out = Path(output)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result.output_paths[0], out)
            print(f"[cli] DONE ({elapsed:.1f}s) -> {out}")
            return str(out)

        if result.output_paths:
            print(f"[cli] DONE ({elapsed:.1f}s) -> {result.output_paths[0]}")
            return result.output_paths[0]

        raise RuntimeError("No output generated")
