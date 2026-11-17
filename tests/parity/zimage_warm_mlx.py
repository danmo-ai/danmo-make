"""Warm-run main Z-Image baseline for the P0 S4 performance gate.

Runs one fixed real-weight vertical slice several times in one process and
records per-pass timings with the weights resident (the production steady
state). Pass 0 is cold (weight load + first kernels); passes 1+ are the warm
baseline the Rust engine must stay within 105% of (plan §8.2).

Paths validated separately (plan S2): default eager, `--compile` (main's
`use_mlx_compile=True` through `before_denoise`), and `--teacache` (the
registry default `teacache_mode="auto"` optimization). `--native-dtype`
restores every checkpoint tensor's stored dtype after load, matching the
Rust production `WeightDtype::Native` policy (§1.5.1).

The denoise timing boundary matches the parity fixtures: it starts before
`before_denoise` (cap/geo precompute when the optimization plan enables it)
and ends after the final latent is evaluated; per-step evals mirror main's
production step loop (velocity + latent, plus the residual eval the TeaCache
path performs inside `_forward_cached_compute`).
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--teacache", action="store_true")
    parser.add_argument("--teacache-thresh", type=float)
    parser.add_argument("--native-dtype", action="store_true")
    args = parser.parse_args()
    if args.compile and args.teacache:
        raise SystemExit("--compile and --teacache are separate validation paths")
    if args.width % 16 or args.height % 16:
        raise SystemExit("width/height must be multiples of 16 (VAE 8 x patch 2)")
    sys.path.insert(0, args.main_root)
    width, height, steps = args.width, args.height, args.steps
    latent_h, latent_w = height // 8, width // 8

    import mlx.core as mx
    from backend.engine.common.codecs.text_encoders.qwen3_mlx import build_zimage_mlx_encoder
    from backend.engine.common.codecs.vae.decoder import (
        build_standard_vae_preview_session,
        vae_output_to_uint8_hwc,
    )
    from backend.engine.common.model.base import TransformerBase
    from backend.engine.common.ops.schedulers import LinearScheduler
    from backend.engine.config.model_configs import ZImageConfig
    from backend.engine.families.z_image.transformer_mlx import ZImageDiTMLX
    from backend.engine.runtime.mlx import MLXContext

    if args.native_dtype:
        mx.reset_peak_memory()
    bundle = Path(args.bundle)
    ids = [151644, 872, 198, 64, 2518, 23739, 151645, 198, 151644, 77091, 198]
    ctx = MLXContext()

    load_started = time.monotonic()
    encoder = build_zimage_mlx_encoder(str(bundle / "text_encoder"), ctx)
    config = ZImageConfig()
    if args.compile:
        config.use_mlx_compile = True
        config.teacache_mode = "none"
    if args.teacache:
        config.use_mlx_compile = False
        config.teacache_mode = "auto"
    model = ZImageDiTMLX(config, ctx)
    weights = {}
    for shard in sorted((bundle / "transformer").glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    model.load_weights(list(weights.items()), strict=True)
    if args.native_dtype:
        native_weights = model.sanitize(weights)
        for key, param in list(model._param_map.items()):
            native = native_weights[key]
            if param.dtype == native.dtype:
                continue
            model._param_map[key] = native
            model._rebind_param_by_key(key, native)
    del weights
    mx.clear_cache()
    session = build_standard_vae_preview_session(ctx, bundle / "vae")
    if session is None:
        raise RuntimeError("VAE preview session did not load")
    if args.native_dtype:
        # Restore each VAE parameter to its published checkpoint dtype
        # (all BF16 for this VAE), never a blanket downcast.
        vae = session["vae"]
        from backend.engine.common.codecs.vae.weight_remap import remap_vae_weights

        vae_weights = remap_vae_weights(session["vae_weights"])
        for key, param in list(vae._param_map.items()):
            native = vae_weights.get(key)
            if native is None:
                raise RuntimeError(f"VAE checkpoint has no tensor for {key}")
            if param.dtype == native.dtype:
                continue
            vae._param_map[key] = native
            TransformerBase._rebind_param_by_key(vae, key, native)
        del vae_weights
        mx.clear_cache()
    load_seconds = time.monotonic() - load_started

    latent_values = np.array(
        [((index % 31) * 0.02 - 0.3) for index in range(16 * latent_h * latent_w)],
        dtype=np.float32,
    )
    passes = []
    denoised_reference = None
    pixels_reference = None
    cache_reference = None
    caption_reference = None
    output_reference = None
    velocities_reference = None
    step_latents_reference = None
    sigmas_reference = None
    for pass_index in range(args.warmup + args.passes):
        started = time.monotonic()
        input_ids = mx.array([ids], dtype=mx.int32)
        attention_mask = mx.ones((1, len(ids)), dtype=mx.int32)
        hidden = encoder.embed_tokens(input_ids).astype(mx.bfloat16)
        causal_mask = encoder._get_causal_mask(attention_mask, 1, hidden, len(ids))
        position_ids = encoder._build_position_ids(1, len(ids))
        positions = encoder.rotary_emb(hidden, position_ids)
        for layer in encoder.layers[:-1]:
            hidden = layer(
                hidden_states=hidden,
                attention_mask=causal_mask,
                position_embeddings=positions,
            )
        caption = hidden[:, : len(ids), :].astype(mx.bfloat16)
        mx.eval(caption)
        encode_seconds = time.monotonic() - started
        caption_values = np.asarray(caption.astype(mx.float32), dtype=np.float32).reshape(-1)
        if caption_reference is None:
            caption_reference = caption_values
        elif not np.array_equal(caption_values, caption_reference):
            raise RuntimeError("warm passes produced different captions")

        dit_started = time.monotonic()
        latents = mx.array(latent_values.reshape(1, 16, latent_h, latent_w), dtype=mx.float32).astype(
            mx.bfloat16
        )
        scheduler = LinearScheduler(ctx=ctx)
        timesteps = scheduler.set_timesteps(
            steps,
            image_width=width,
            image_height=height,
            requires_sigma_shift=True,
        )
        conditioning = {"txt_embeds": caption}
        if args.teacache_thresh is not None:
            conditioning["teacache_thresh"] = args.teacache_thresh
        if args.compile or args.teacache:
            latents, conditioning = model.before_denoise(
                latents, timesteps, scheduler.sigmas, **conditioning
            )
        if args.compile and (not model._use_mlx_compile_run or model._compiled_forward is None):
            raise RuntimeError("main compiled forward was not enabled")
        cache_decisions = []
        if args.teacache:
            if model._step_cache is None or not model._step_cache.enabled:
                raise RuntimeError("main TeaCache was not enabled")
            if model._use_mlx_compile_run:
                raise RuntimeError("TeaCache must disable compiled forward")
            if args.teacache_thresh is not None:
                model._step_cache.thresh = float(args.teacache_thresh)
            original_gate = model._step_cache.gate

            def recording_gate(mod_in, **kwargs):
                decision = original_gate(mod_in, **kwargs)
                cache_decisions.append(
                    {
                        "kind": decision.kind,
                        "should_compute": decision.should_compute,
                        "should_update_cache": decision.should_update_cache,
                        "rel_l1": decision.rel_l1,
                        "predicted_distance": decision.predicted_distance,
                        "accumulated_distance": decision.accumulated_distance,
                    }
                )
                return decision

            model._step_cache.gate = recording_gate
        step_seconds = []
        step_velocities = []
        step_latents = []
        for timestep in timesteps:
            step_started = time.monotonic()
            velocity = model.forward(
                latents, timestep, sigmas=scheduler.sigmas, **conditioning
            )
            latents = scheduler.step(velocity, timestep, latents)
            if args.teacache:
                model.step_callback(timestep, latents, velocity)
            mx.eval(velocity, latents)
            step_seconds.append(time.monotonic() - step_started)
            step_velocities.append(
                np.asarray(velocity.astype(mx.float32), dtype=np.float32).reshape(-1)
            )
            step_latents.append(
                np.asarray(latents.astype(mx.float32), dtype=np.float32).reshape(-1)
            )
        mx.eval(latents)
        if velocities_reference is None:
            velocities_reference = step_velocities
            step_latents_reference = step_latents
            sigmas_reference = np.asarray(scheduler.sigmas, dtype=np.float32)
        else:
            for actual, expected in zip(step_velocities, velocities_reference):
                if not np.array_equal(actual, expected):
                    raise RuntimeError("warm passes produced different velocities")
            for actual, expected in zip(step_latents, step_latents_reference):
                if not np.array_equal(actual, expected):
                    raise RuntimeError("warm passes produced different latents")
        denoise_seconds = time.monotonic() - dit_started
        denoised_values = np.asarray(latents.astype(mx.float32), dtype=np.float32).reshape(-1)
        cache_counts = None
        if args.teacache:
            cache_counts = {
                "computed": model._step_cache.state.computed_count,
                "skipped": model._step_cache.state.skipped_count,
            }
            if cache_reference is None:
                cache_reference = cache_counts
            elif cache_counts != cache_reference:
                raise RuntimeError("warm passes changed TeaCache decisions")
        if denoised_reference is None:
            denoised_reference = denoised_values
        elif not np.array_equal(denoised_values, denoised_reference):
            raise RuntimeError("warm passes diverged from the first pass")

        vae_started = time.monotonic()
        output = session["vae"].forward(latents)
        mx.eval(output)
        pixels = vae_output_to_uint8_hwc(
            output.astype(mx.float32) if args.native_dtype else output, ctx
        )
        vae_seconds = time.monotonic() - vae_started
        output_values = np.asarray(output.astype(mx.float32), dtype=np.float32).reshape(-1)
        if output_reference is None:
            output_reference = output_values
        elif not np.array_equal(output_values, output_reference):
            raise RuntimeError("warm passes produced different VAE outputs")
        if pixels_reference is None:
            pixels_reference = pixels
        elif not np.array_equal(pixels, pixels_reference):
            raise RuntimeError("warm passes produced different pixels")
        passes.append(
            {
                "encode": encode_seconds,
                "denoise": denoise_seconds,
                "steps": step_seconds,
                "vae": vae_seconds,
                "total": time.monotonic() - started,
            }
        )
        print(
            f"pass {pass_index}: encode={encode_seconds:.3f}s "
            f"denoise={denoise_seconds:.3f}s vae={vae_seconds:.3f}s"
            + (f" cache={cache_counts}" if cache_counts else "")
        )

    warm = passes[args.warmup:] if len(passes) > args.warmup else passes
    payload = {
        "ids": ids,
        "width": width,
        "height": height,
        "steps": steps,
        "compile": args.compile,
        "teacache": args.teacache,
        "teacache_thresh": args.teacache_thresh,
        "native_dtype": "checkpoint" if args.native_dtype else None,
        "load_seconds": load_seconds,
        "passes": passes,
        "warm_p50": {
            key: statistics.median(entry[key] for entry in warm)
            for key in ("encode", "denoise", "vae", "total")
        },
        "cache_counts": cache_reference,
        "cache_decisions": cache_decisions,
        "caption_shape": [len(ids), 2560],
        "caption": caption_reference.tolist(),
        "latent": latent_values.tolist(),
        "sigmas": sigmas_reference.tolist(),
        "velocities": [values.tolist() for values in velocities_reference],
        "step_latents": [values.tolist() for values in step_latents_reference],
        "denoised": denoised_reference.tolist(),
        "output": output_reference.tolist(),
        "pixels": pixels_reference.reshape(-1).tolist(),
        "memory": {
            "peak": int(mx.get_peak_memory()),
            "active": int(mx.get_active_memory()),
            "cache": int(mx.get_cache_memory()),
        },
    }
    # Steady-state resident footprint with the reusable buffer pool emptied,
    # comparable across sides regardless of pass-count pool accumulation.
    mx.clear_cache()
    payload["memory"]["steady_active"] = int(mx.get_active_memory())
    payload["memory"]["steady_cache"] = int(mx.get_cache_memory())
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
