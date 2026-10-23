"""Audio waveform generation — flow-matching / block-AR."""

from __future__ import annotations

from typing import Any

from backend.engine.inference._runtime import inference_span
from backend.engine.protocols.plugin import ParadigmKind


def _audio_span_name(paradigm: ParadigmKind) -> str:
    if paradigm == "flow_matching":
        return "flow_matching_paradigm"
    if paradigm == "block_ar":
        return "block_ar_paradigm"
    return "audio_generate_paradigm"


def run_audio_waveform(ctx: Any, *, batch_seed: int, batch_idx: int) -> Any:
    with inference_span(ctx.exec_ctx, _audio_span_name(ctx.paradigm)):
        return ctx.generator.generate_waveform(
            prompt=ctx.prepared.effective_prompt or ctx.request.prompt or "",
            lyrics=ctx.lyrics,
            vocal_language=ctx.vocal_lang,
            duration=ctx.duration,
            steps=ctx.steps,
            guidance=ctx.guidance,
            seed=batch_seed,
            bpm=ctx.request.bpm,
            key_scale=ctx.request.key_scale or "",
            time_signature=ctx.request.time_signature or "",
            shift=ctx.shift,
            instrumental=bool(ctx.request.instrumental),
            lm_enabled=ctx.lm_enabled,
            lm_quantize_bits=ctx.lm_quantize_bits,
            **getattr(ctx.prepared, "sampler_kwargs", {}),
        )
