import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--teacache", action="store_true")
    parser.add_argument("--teacache-thresh", type=float)
    parser.add_argument("--native-dtype", action="store_true")
    parser.add_argument("--quantization-bits", type=int, choices=(4, 8))
    args = parser.parse_args()
    if args.compile and args.teacache:
        raise SystemExit("--compile and --teacache are separate validation paths")
    sys.path.insert(0, args.main_root)
    width, height, steps = args.width, args.height, args.steps
    if width % 16 or height % 16:
        raise SystemExit("width/height must be multiples of 16 (VAE 8 x patch 2)")
    latent_h, latent_w = height // 8, width // 8

    import mlx.core as mx
    from backend.engine.common.codecs.text_encoders.qwen3_mlx import build_zimage_mlx_encoder
    from backend.engine.common.model.base import TransformerBase
    from backend.engine.common.bundle.quant_inference import WeightInferenceMode
    from backend.engine.common.codecs.vae.decoder import (
        build_standard_vae_preview_session,
        vae_output_to_uint8_hwc,
    )
    from backend.engine.common.ops.schedulers import LinearScheduler
    from backend.engine.config.model_configs import ZImageConfig
    from backend.engine.families.z_image.transformer_mlx import ZImageDiTMLX
    from backend.engine.runtime.mlx import MLXContext

    if args.native_dtype:
        mx.reset_peak_memory()
    bundle = Path(args.bundle)
    ids = [151644, 872, 198, 64, 2518, 23739, 151645, 198, 151644, 77091, 198]
    started = time.monotonic()
    ctx = MLXContext()
    encoder = build_zimage_mlx_encoder(str(bundle / "text_encoder"), ctx)
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
    caption_values = np.asarray(caption.astype(mx.float32), dtype=np.float32).reshape(-1)
    encode_seconds = time.monotonic() - started
    del encoder, hidden, caption
    mx.clear_cache()

    dit_started = time.monotonic()
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
    inference_mode = (
        WeightInferenceMode(kind="quantized", bits=args.quantization_bits)
        if args.quantization_bits
        else None
    )
    model.load_weights(
        list(weights.items()),
        strict=True,
        bundle_affine_bits=args.quantization_bits,
        inference_mode=inference_mode,
    )
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
    if args.seed is None:
        latent_values = np.array(
            [((index % 31) * 0.02 - 0.3) for index in range(16 * latent_h * latent_w)],
            dtype=np.float32,
        )
    else:
        latent_values = np.asarray(
            mx.random.normal((1, 16, latent_h, latent_w), key=mx.random.key(args.seed)),
            dtype=np.float32,
        ).reshape(-1)
    latents = mx.array(latent_values.reshape(1, 16, latent_h, latent_w), dtype=mx.float32).astype(mx.bfloat16)
    caption = mx.array(caption_values.reshape(len(ids), 2560), dtype=mx.bfloat16)
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
    velocities = []
    step_latents = []
    for timestep in timesteps:
        velocity = model.forward(latents, timestep, sigmas=scheduler.sigmas, **conditioning)
        latents = scheduler.step(velocity, timestep, latents)
        if args.teacache:
            model.step_callback(timestep, latents, velocity)
        mx.eval(velocity, latents)
        velocities.append(np.asarray(velocity.astype(mx.float32), dtype=np.float32).reshape(-1).tolist())
        step_latents.append(np.asarray(latents.astype(mx.float32), dtype=np.float32).reshape(-1).tolist())
    mx.eval(latents)
    denoised_values = np.asarray(latents.astype(mx.float32), dtype=np.float32).reshape(-1)
    denoise_seconds = time.monotonic() - dit_started
    cache_counts = {
        "computed": model._step_cache.state.computed_count if args.teacache else 0,
        "skipped": model._step_cache.state.skipped_count if args.teacache else 0,
    }
    del model
    mx.clear_cache()

    vae_started = time.monotonic()
    session = build_standard_vae_preview_session(ctx, bundle / "vae")
    if session is None:
        raise RuntimeError("VAE preview session did not load")
    if args.native_dtype:
        vae = session["vae"]
        for key, param in list(vae._param_map.items()):
            if param.dtype == mx.bfloat16:
                continue
            native = param.astype(mx.bfloat16)
            vae._param_map[key] = native
            TransformerBase._rebind_param_by_key(vae, key, native)
    output = session["vae"].forward(latents)
    mx.eval(output)
    output_values = np.asarray(output.astype(mx.float32), dtype=np.float32).reshape(-1)
    pixels = vae_output_to_uint8_hwc(
        output.astype(mx.float32) if args.native_dtype else output, ctx
    )
    vae_seconds = time.monotonic() - vae_started
    payload = {
        "ids": ids,
        "width": width,
        "height": height,
        "steps": steps,
        "seed": args.seed,
        "compile": args.compile,
        "teacache": args.teacache,
        "teacache_thresh": args.teacache_thresh,
        "cache_decisions": cache_decisions,
        "cache_counts": cache_counts,
        "native_dtype": "checkpoint" if args.native_dtype else None,
        "quantization_bits": args.quantization_bits,
        "caption_shape": [len(ids), 2560],
        "caption": caption_values.tolist(),
        "latent_shape": [1, 16, latent_h, latent_w],
        "latent": latent_values.tolist(),
        "sigmas": np.asarray(scheduler.sigmas, dtype=np.float32).tolist(),
        "denoised": denoised_values.tolist(),
        "velocities": velocities,
        "step_latents": step_latents,
        "output_shape": list(output.shape),
        "output": output_values.tolist(),
        "pixels": pixels.reshape(-1).tolist(),
        "timings": {
            "encode": encode_seconds,
            "dit_load_and_denoise": denoise_seconds,
            "vae": vae_seconds,
            "total": time.monotonic() - started,
        },
        "memory": {
            "peak": int(mx.get_peak_memory()),
            "active": int(mx.get_active_memory()),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    print(
        f"main real Z-Image: encode={encode_seconds:.3f}s "
        f"dit={denoise_seconds:.3f}s vae={vae_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
