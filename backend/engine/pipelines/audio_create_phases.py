"""Audio create phased helpers (``AudioSession``)."""

from __future__ import annotations

import logging
import random
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List

from backend.core.contracts import (
    AudioGenerationRequest,
    EngineResult,
    ExecutionContext,
    LogEvent,
    ProgressEvent,
)
from backend.engine._transformer_registry import (
    audio_lyrics_metadata,
    get_audio_post_generation,
    get_audio_prepare_request,
)
from backend.engine.config.model_configs import get_config_class
from backend.engine.families._audio_backbone import plugin_audio_generator_if_ready
from backend.engine.inference.audio_waveform import run_audio_waveform
from backend.engine.pipelines.audio_run_common import load_audio_generator_for_request
from backend.engine.pipelines.audio_persist import (
    ACE_STEP_SAMPLE_RATE,
    persist_audio_create_assets,
    quality_log_message,
    raise_if_cancelled,
    save_audio_waveform,
)
from backend.engine.protocols.plugin import FamilyPlugin, ParadigmKind
from backend.engine.sessions._context import MediaRunContext, ResolvedRun, require_resolved_bundle

logger = logging.getLogger(__name__)

PhaseCmFactory = Callable[[str], AbstractContextManager[Any]]


@dataclass
class AudioCreateRunContext(MediaRunContext):
    """State for one audio create run (prepare → infer batch → persist)."""

    pipeline: Any
    request: AudioGenerationRequest
    exec_ctx: ExecutionContext
    model_id: str
    version_key: str | None
    entry: Any
    bundle_root: Path
    config: Any
    family: str
    prepared: Any
    generator: Any
    paradigm: ParadigmKind
    steps: int
    lyrics: Any
    vocal_lang: Any
    shift: Any
    duration: float
    guidance: float
    lm_enabled: bool
    lm_quantize_bits: Any
    sample_rate: int
    seed: int
    n: int
    t0: float

    def session_infer(self, **_ignored: Any) -> tuple[list[str], list[float], Any, Any]:
        return execute_audio_infer_batch(self)


def build_audio_create_run_context(
    pipeline: Any,
    request: AudioGenerationRequest,
    exec_ctx: ExecutionContext,
    *,
    resolved: ResolvedRun,
    t0: float,
    phase_cm: PhaseCmFactory | None = None,
    plugin: FamilyPlugin | None = None,
    paradigm: ParadigmKind = "flow_matching",
) -> AudioCreateRunContext:
    phase_cm = phase_cm or (lambda _name: nullcontext())
    bundle_root = require_resolved_bundle(resolved)
    entry = resolved.registry_entry
    model_id = resolved.model_id
    version_key = resolved.version_key
    family = resolved.family_id
    config = get_config_class(family)()

    with phase_cm("prepare"):
        prepared = get_audio_prepare_request(family)(
            request,
            config,
            bundle_root,
            backend=getattr(pipeline.ctx, "backend", "mlx"),
        )
        for level, message in getattr(prepared, "log_events", []):
            exec_ctx.on_log(LogEvent(level=level, message=message))

        exec_ctx.on_log(
            LogEvent(
                level="info",
                message=f"Loading {family} model: {model_id} from {bundle_root}",
            )
        )
        generator = plugin_audio_generator_if_ready(plugin)
        if generator is None:

            def _on_log(level: str, message: str) -> None:
                exec_ctx.on_log(LogEvent(level=level, message=message))

            generator = load_audio_generator_for_request(
                pipeline,
                family=family,
                bundle_root=bundle_root,
                entry=entry,
                version_key=version_key,
                request=request,
                on_log=_on_log,
            )

    steps = prepared.steps
    lyrics = prepared.lyrics
    vocal_lang = prepared.vocal_language
    shift = prepared.shift
    duration = float(prepared.duration)
    guidance = getattr(prepared, "guidance", None)
    if guidance is None:
        guidance = request.guidance if request.guidance is not None else 3.0
    lm_enabled = getattr(prepared, "lm_enabled", False)
    lm_quantize_bits = getattr(prepared, "lm_quantize_bits", None)
    sample_rate = getattr(config, "sample_rate", ACE_STEP_SAMPLE_RATE)

    seed = (
        request.seed
        if request.seed is not None and request.seed >= 0
        else random.randint(0, 2**31 - 1)
    )
    n = max(request.n, 1)

    exec_ctx.on_log(
        LogEvent(
            level="info",
            message=(
                f"Generating {n} audio(s): duration={duration}s, steps={steps}, "
                f"guidance={guidance}, seed={seed}, "
                f"instrumental={request.instrumental}, vocal_language={vocal_lang}"
            ),
        )
    )
    if request.negative_prompt:
        exec_ctx.on_log(
            LogEvent(
                level="warning",
                message=f"negative_prompt is not used by {family} (ignored)",
            )
        )

    return AudioCreateRunContext(
        pipeline=pipeline,
        request=request,
        exec_ctx=exec_ctx,
        model_id=model_id,
        version_key=version_key,
        entry=entry,
        bundle_root=bundle_root,
        config=config,
        family=family,
        prepared=prepared,
        generator=generator,
        paradigm=paradigm,
        steps=steps,
        lyrics=lyrics,
        vocal_lang=vocal_lang,
        shift=shift,
        duration=duration,
        guidance=float(guidance),
        lm_enabled=lm_enabled,
        lm_quantize_bits=lm_quantize_bits,
        sample_rate=sample_rate,
        seed=seed,
        n=n,
        t0=t0,
    )


def execute_audio_infer_batch(
    ctx: AudioCreateRunContext,
) -> tuple[list[str], list[float], Any, Any]:
    """Run batched waveform generation; returns paths, durations, per-sample lyrics captures, generator."""
    pipeline = ctx.pipeline
    exec_ctx = ctx.exec_ctx
    family = ctx.family
    generator = ctx.generator
    post_gen = get_audio_post_generation(family)

    output_paths: List[str] = []
    output_durations: List[float] = []
    lyrics_captures: List[Any] = []

    for i in range(ctx.n):
        raise_if_cancelled(exec_ctx)
        batch_seed = ctx.seed + i
        exec_ctx.on_progress(
            ProgressEvent(
                progress=(i + 0.1) / ctx.n,
                step=i + 1,
                total=ctx.n,
                message=f"Generating audio {i + 1}/{ctx.n}",
            )
        )
        exec_ctx.on_log(
            LogEvent(
                level="info",
                message=f"Running {family} inference ({i + 1}/{ctx.n}, seed={batch_seed})...",
            )
        )
        t_gen = time.monotonic()
        try:
            waveform = run_audio_waveform(
                ctx, batch_seed=batch_seed, batch_idx=i
            )
        except Exception as exc:
            logger.exception(
                "%s generation failed (item %d/%d, seed=%s)",
                family,
                i + 1,
                ctx.n,
                batch_seed,
            )
            exec_ctx.on_log(
                LogEvent(
                    level="error",
                    message=f"{family} generation failed ({i + 1}/{ctx.n}): {exc}",
                )
            )
            raise
        raise_if_cancelled(exec_ctx)

        gen_s = time.monotonic() - t_gen
        latent_frames = getattr(generator, "last_latent_frames", 0)
        hum_ratio = getattr(generator, "last_hum_ratio", 0.0)
        mains_acf = getattr(generator, "last_mains_acf", 0.0)
        decode_mode = getattr(generator, "last_decode_mode", "")
        latent_cos = getattr(generator, "last_latent_cos", 0.0)
        latent_diff = getattr(generator, "last_latent_diff_mean", 0.0)
        lm_expanded = getattr(generator, "last_lm_expanded", False)
        quality = getattr(generator, "last_quality", None)
        q_msg = quality_log_message(quality)
        if q_msg:
            exec_ctx.on_log(LogEvent(level="info", message=q_msg))

        lyrics_captures.append(getattr(generator, "last_lyrics_capture", None))

        exec_ctx.on_log(
            LogEvent(
                level="info",
                message=(
                    f"{family} inference done ({i + 1}/{ctx.n}): {gen_s:.1f}s, "
                    f"latent_frames={latent_frames}, decode={decode_mode}, "
                    f"lm_expanded={lm_expanded}, latent_cos={latent_cos:.4f}, "
                    f"latent_diff={latent_diff:.4f}, mains_acf={mains_acf:.3f}"
                ),
            )
        )
        if mains_acf > 0.4 or hum_ratio > 0.25:
            exec_ctx.on_log(
                LogEvent(
                    level="warning",
                    message=(
                        f"Audio {i + 1}/{ctx.n} may contain mains hum "
                        f"(mains_acf={mains_acf:.3f}); try another seed or a descriptive prompt"
                    ),
                )
            )

        exec_ctx.on_progress(
            ProgressEvent(
                progress=(i + 0.9) / ctx.n,
                step=i + 1,
                total=ctx.n,
                message=f"Saving audio {i + 1}/{ctx.n}",
            )
        )

        out_path = save_audio_waveform(
            pipeline._project_root,
            waveform,
            ctx.model_id,
            batch_seed,
            family=family,
            sample_rate=ctx.sample_rate,
        )
        n_samples = int(waveform.shape[0]) if hasattr(waveform, "shape") else 0
        dur_written = n_samples / ctx.sample_rate if n_samples else 0.0

        if post_gen is not None:
            def _post_log(level: str, message: str) -> None:
                exec_ctx.on_log(LogEvent(level=level, message=message))

            post_gen(
                out_path,
                generator,
                duration_sec=dur_written,
                on_log=_post_log,
                log_lyrics_preview=(i == 0),
            )

        exec_ctx.on_log(
            LogEvent(
                level="success",
                message=(
                    f"Saved audio {i + 1}/{ctx.n}: {out_path.name} "
                    f"({dur_written:.1f}s, {out_path.stat().st_size // 1024}KB)"
                ),
            )
        )
        output_paths.append(str(out_path))
        output_durations.append(dur_written)

    return output_paths, output_durations, lyrics_captures, generator


def persist_audio_create(
    ctx: AudioCreateRunContext,
    output_paths: list[str],
    output_durations: list[float],
    lyrics_captures: list[Any],
    generator: Any,
) -> EngineResult:
    exec_ctx = ctx.exec_ctx
    pipeline = ctx.pipeline
    family = ctx.family

    raise_if_cancelled(exec_ctx)
    exec_ctx.on_progress(
        ProgressEvent(progress=1.0, step=ctx.n, total=ctx.n, message="Complete")
    )
    exec_ctx.on_log(
        LogEvent(
            level="success",
            message=(
                f"{family} complete: {len(output_paths)} file(s) "
                f"in {time.monotonic() - ctx.t0:.1f}s"
            ),
        )
    )
    elapsed = time.monotonic() - ctx.t0

    asset_ids = persist_audio_create_assets(
        pipeline._asset_store,
        output_paths,
        ctx.request,
        ctx.model_id,
        elapsed,
        exec_ctx.task_id,
        output_durations,
        family=family,
        lyrics_captures=lyrics_captures,
    )

    result_meta: dict[str, Any] = {
        "model": ctx.model_id,
        "seed": ctx.seed,
        "steps": ctx.steps,
        "guidance": ctx.guidance,
        "duration_seconds": ctx.duration,
    }
    lyrics_capture = next((c for c in reversed(lyrics_captures) if c is not None), None)
    if lyrics_capture is not None:
        result_meta.update(
            audio_lyrics_metadata(family, lyrics_capture, duration_sec=ctx.duration)
        )
    quality = getattr(generator, "last_quality", None)
    if quality is not None:
        result_meta.update(quality.as_metadata())

    return EngineResult(
        primary_asset_id=asset_ids[0] if asset_ids else "",
        asset_ids=asset_ids,
        output_paths=output_paths,
        metadata=result_meta,
    )

