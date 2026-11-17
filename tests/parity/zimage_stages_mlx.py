"""Stage-by-stage main Z-Image DiT outputs for divergence localization.

Runs the same fixed slice as `zimage_real_mlx.py` but hooks every block of a
selected denoise step and dumps intermediate activations (F32) to safetensors
so the Rust forward can be compared stage by stage at production resolution.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--teacache", action="store_true")
    parser.add_argument("--native-dtype", action="store_true")
    parser.add_argument("--injection-fixture", help="warm fixture whose latent/caption/sigmas to replay")
    args = parser.parse_args()
    sys.path.insert(0, args.main_root)
    width, height, steps = args.width, args.height, args.steps
    latent_h, latent_w = height // 8, width // 8

    import mlx.core as mx
    from backend.engine.common.codecs.text_encoders.qwen3_mlx import build_zimage_mlx_encoder
    from backend.engine.common.ops.schedulers import LinearScheduler
    from backend.engine.config.model_configs import ZImageConfig
    from backend.engine.families.z_image.transformer_mlx import ZImageDiTMLX
    from backend.engine.runtime.mlx import MLXContext

    bundle = Path(args.bundle)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = [151644, 872, 198, 64, 2518, 23739, 151645, 198, 151644, 77091, 198]
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
    del encoder, hidden
    mx.clear_cache()

    config = ZImageConfig()
    if args.teacache:
        config.use_mlx_compile = False
        config.teacache_mode = "auto"
    model = ZImageDiTMLX(config, ctx)
    weights = {}
    for shard in sorted((bundle / "transformer").glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    model.load_weights(list(weights.items()), strict=True)
    if args.native_dtype:
        from backend.engine.common.model.base import TransformerBase  # noqa: F401
        native_weights = model.sanitize(weights)
        for key, param in list(model._param_map.items()):
            native = native_weights[key]
            if param.dtype == native.dtype:
                continue
            model._param_map[key] = native
            model._rebind_param_by_key(key, native)
        del native_weights
    del weights
    mx.clear_cache()

    stages = {}

    def record(name, value):
        stages[name] = value.astype(mx.float32)

    def hook(module, name):
        if hasattr(module, "forward"):
            original = module.forward

            def wrapped(*a, **kw):
                result = original(*a, **kw)
                record(name, result)
                return result

            module.forward = wrapped
        else:
            # mlx.nn.Module declares __call__ as an instance attribute.
            original = module.__call__

            def wrapped(*a, **kw):
                result = original(*a, **kw)
                record(name, result)
                return result

            module.__call__ = wrapped

    hook(model.t_embedder, "t_embedder")
    for i, layer in enumerate(model.noise_refiner):
        hook(layer, f"noise_refiner.{i}")
    for i, layer in enumerate(model.context_refiner):
        hook(layer, f"context_refiner.{i}")
    for i, layer in enumerate(model.layers):
        hook(layer, f"layers.{i}")
    hook(model.final_layer, "final_layer")

    injection = None
    if args.injection_fixture:
        injection = json.loads(Path(args.injection_fixture).read_text())
        if (injection["width"], injection["height"], injection["steps"]) != (width, height, steps):
            raise SystemExit("injection fixture geometry/steps mismatch")
        if injection["ids"] != ids:
            raise SystemExit("injection fixture token ids mismatch")
        caption_values = np.asarray(injection["caption"], dtype=np.float32)
        latent_values = np.asarray(injection["latent"], dtype=np.float32)
    else:
        latent_values = np.asarray(
            mx.random.normal((1, 16, latent_h, latent_w), key=mx.random.key(args.seed)),
            dtype=np.float32,
        ).reshape(-1)
    initial_latent_values = latent_values.copy()
    if injection is not None and args.step > 0:
        # Start directly from the recorded step's INPUT latent.
        latent_values = np.asarray(injection["step_latents"][args.step - 1], dtype=np.float32)
    latents = mx.array(latent_values.reshape(1, 16, latent_h, latent_w), dtype=mx.float32).astype(
        mx.bfloat16
    )
    caption_in = mx.array(caption_values.reshape(len(ids), 2560), dtype=mx.bfloat16)
    scheduler = LinearScheduler(ctx=ctx)
    timesteps = scheduler.set_timesteps(
        steps,
        image_width=width,
        image_height=height,
        requires_sigma_shift=True,
    )
    if not 0 <= args.step < steps:
        raise SystemExit(f"--step must be in [0, {steps})")
    conditioning = {"txt_embeds": caption_in}
    if args.teacache:
        latents, conditioning = model.before_denoise(
            latents, timesteps, scheduler.sigmas, **conditioning
        )
        if model._step_cache is None or not model._step_cache.enabled:
            raise RuntimeError("main TeaCache was not enabled")
        original_gate = model._step_cache.gate

        def traced_gate(mod_in, **kwargs):
            decision = original_gate(mod_in, **kwargs)
            print(
                f"main gate: kind={decision.kind} rel_l1={decision.rel_l1} "
                f"pred={decision.predicted_distance} acc={decision.accumulated_distance}"
            )
            return decision

        model._step_cache.gate = traced_gate
    step_velocities = []
    if injection is not None and args.step > 0:
        step_velocities = injection["velocities"][: args.step]
        if args.teacache:
            # Replay the earlier steps so the cache state (previous mod
            # input, residual, accumulated distance, step counter) matches
            # the production loop at the recorded step.
            replay = mx.array(
                initial_latent_values.reshape(1, 16, latent_h, latent_w), dtype=mx.float32
            ).astype(mx.bfloat16)
            for index in range(args.step):
                velocity = model.forward(
                    replay, timesteps[index], sigmas=scheduler.sigmas, **conditioning
                )
                replay = scheduler.step(velocity, timesteps[index], replay)
                model.step_callback(timesteps[index], replay, velocity)
                mx.eval(velocity, replay)
        start_step = args.step
    else:
        start_step = 0
    for step in range(start_step, args.step + 1):
        stages.clear()
        velocity = model.forward(
            latents, timesteps[step], sigmas=scheduler.sigmas, **conditioning
        )
        latents = scheduler.step(velocity, timesteps[step], latents)
        if args.teacache:
            model.step_callback(timesteps[step], latents, velocity)
        mx.eval(velocity, latents)
        step_velocities.append(
            np.asarray(velocity.astype(mx.float32), dtype=np.float32).reshape(-1).tolist()
        )
    record("velocity", velocity)
    mx.eval(*stages.values())
    mx.save_safetensors(str(out / "stages.safetensors"), stages)
    payload = {
        "ids": ids,
        "width": width,
        "height": height,
        "steps": steps,
        "seed": args.seed,
        "step": args.step,
        "caption_shape": [len(ids), 2560],
        "caption": caption_values.tolist(),
        "latent_shape": [1, 16, latent_h, latent_w],
        "latent": latent_values.tolist(),
        "initial_latent": initial_latent_values.tolist(),
        "sigmas": np.asarray(scheduler.sigmas, dtype=np.float32).tolist(),
        "velocities": step_velocities,
        "velocity_shape": list(velocity.shape),
        "stages": {name: list(value.shape) for name, value in stages.items()},
    }
    (out / "expected.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"main stages: {len(stages)} tensors, velocity shape {velocity.shape}")


if __name__ == "__main__":
    main()
