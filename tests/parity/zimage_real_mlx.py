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
    args = parser.parse_args()
    sys.path.insert(0, args.main_root)
    width, height, steps = args.width, args.height, args.steps
    if width % 16 or height % 16:
        raise SystemExit("width/height must be multiples of 16 (VAE 8 x patch 2)")
    latent_h, latent_w = height // 8, width // 8

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
    model = ZImageDiTMLX(ZImageConfig(), ctx)
    weights = {}
    for shard in sorted((bundle / "transformer").glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    model.load_weights(list(weights.items()), strict=True)
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
    velocities = []
    step_latents = []
    for timestep in timesteps:
        velocity = model.forward(latents, timestep, txt_embeds=caption, sigmas=scheduler.sigmas)
        latents = scheduler.step(velocity, timestep, latents)
        mx.eval(velocity, latents)
        velocities.append(np.asarray(velocity.astype(mx.float32), dtype=np.float32).reshape(-1).tolist())
        step_latents.append(np.asarray(latents.astype(mx.float32), dtype=np.float32).reshape(-1).tolist())
    mx.eval(latents)
    denoised_values = np.asarray(latents.astype(mx.float32), dtype=np.float32).reshape(-1)
    denoise_seconds = time.monotonic() - dit_started
    del model
    mx.clear_cache()

    vae_started = time.monotonic()
    session = build_standard_vae_preview_session(ctx, bundle / "vae")
    if session is None:
        raise RuntimeError("VAE preview session did not load")
    output = session["vae"].forward(latents)
    mx.eval(output)
    output_values = np.asarray(output.astype(mx.float32), dtype=np.float32).reshape(-1)
    pixels = vae_output_to_uint8_hwc(output, ctx)
    vae_seconds = time.monotonic() - vae_started
    payload = {
        "ids": ids,
        "width": width,
        "height": height,
        "steps": steps,
        "seed": args.seed,
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
