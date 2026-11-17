"""Warm-run main Z-Image baseline for the P0 S4 performance gate.

Runs the same fixed 16x16 / 8-step real-weight vertical slice as
`zimage_real_mlx.py` three times in one process and records per-pass timings.
Pass 0 is cold (weight load + first kernels); passes 1+ are the warm baseline
the Rust engine must reach ~95% of. `--compile` additionally enables main's
`ctx.compile` DiT path (`use_mlx_compile=True`); the default eager mode matches
the production policy for 8-step turbo schedules (compile auto-off at <=8
steps without step distill).
"""

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
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, args.main_root)

    import mlx.core as mx
    from backend.engine.common.codecs.text_encoders.qwen3_mlx import build_zimage_mlx_encoder
    from backend.engine.common.codecs.vae.decoder import (
        build_standard_vae_preview_session,
        vae_output_to_uint8_hwc,
    )
    from backend.engine.common.ops.schedulers import LinearScheduler
    from backend.engine.config.model_configs import ZImageConfig
    from backend.engine.families.z_image.transformer_mlx import ZImageDiTMLX
    from backend.engine.runtime.mlx import MLXContext

    bundle = Path(args.bundle)
    ids = [151644, 872, 198, 64, 2518, 23739, 151645, 198, 151644, 77091, 198]
    ctx = MLXContext()

    encoder = build_zimage_mlx_encoder(str(bundle / "text_encoder"), ctx)
    config = ZImageConfig()
    if args.compile:
        config.use_mlx_compile = True
    model = ZImageDiTMLX(config, ctx)
    weights = {}
    for shard in sorted((bundle / "transformer").glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    model.load_weights(list(weights.items()), strict=True)
    del weights
    mx.clear_cache()
    session = build_standard_vae_preview_session(ctx, bundle / "vae")
    if session is None:
        raise RuntimeError("VAE preview session did not load")

    latent_values = np.array(
        [((index % 31) * 0.02 - 0.3) for index in range(16 * 2 * 2)],
        dtype=np.float32,
    )
    passes = []
    denoised_reference = None
    for pass_index in range(args.passes):
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

        dit_started = time.monotonic()
        latents = mx.array(latent_values.reshape(1, 16, 2, 2), dtype=mx.float32).astype(
            mx.bfloat16
        )
        scheduler = LinearScheduler(ctx=ctx)
        timesteps = scheduler.set_timesteps(
            8,
            image_width=16,
            image_height=16,
            requires_sigma_shift=True,
        )
        step_seconds = []
        for timestep in timesteps:
            step_started = time.monotonic()
            velocity = model.forward(
                latents, timestep, txt_embeds=caption, sigmas=scheduler.sigmas
            )
            latents = scheduler.step(velocity, timestep, latents)
            mx.eval(velocity, latents)
            step_seconds.append(time.monotonic() - step_started)
        mx.eval(latents)
        denoise_seconds = time.monotonic() - dit_started
        denoised_values = np.asarray(latents.astype(mx.float32), dtype=np.float32).reshape(-1)
        if denoised_reference is None:
            denoised_reference = denoised_values
        elif not np.array_equal(denoised_values, denoised_reference):
            raise RuntimeError("warm passes diverged from the first pass")

        vae_started = time.monotonic()
        output = session["vae"].forward(latents)
        mx.eval(output)
        pixels = vae_output_to_uint8_hwc(output, ctx)
        vae_seconds = time.monotonic() - vae_started
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
        )

    payload = {
        "ids": ids,
        "width": 16,
        "height": 16,
        "steps": 8,
        "compile": args.compile,
        "passes": passes,
        "denoised": denoised_reference.tolist(),
        "pixels": pixels.reshape(-1).tolist(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
