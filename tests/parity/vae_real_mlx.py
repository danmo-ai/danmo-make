import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latent-json")
    args = parser.parse_args()
    sys.path.insert(0, args.main_root)

    import mlx.core as mx
    from backend.engine.common.codecs.vae.decoder import build_standard_vae_preview_session
    from backend.engine.runtime.mlx import MLXContext

    started = time.monotonic()
    ctx = MLXContext()
    session = build_standard_vae_preview_session(ctx, Path(args.vae))
    if session is None:
        raise RuntimeError("VAE preview session did not load")
    if args.latent_json:
        latent_values = np.asarray(
            json.loads(Path(args.latent_json).read_text(encoding="utf-8"))["denoised"],
            dtype=np.float32,
        )
    else:
        latent_values = np.array(
            [((index % 31) * 0.02 - 0.3) for index in range(16 * 2 * 2)],
            dtype=np.float32,
        )
    latents = mx.array(latent_values.reshape(1, 16, 2, 2), dtype=mx.bfloat16)
    vae = session["vae"]
    if vae.scaling_factor != 1.0 or vae.shift_factor != 0.0:
        latents = (latents / vae.scaling_factor) + vae.shift_factor
    stages = {}

    def record(name, value):
        stages[name] = np.asarray(value.astype(mx.float32), dtype=np.float32).reshape(-1).tolist()

    x = mx.transpose(latents, (0, 2, 3, 1))
    x = vae.conv_in(x)
    x = mx.transpose(x, (0, 3, 1, 2))
    record("conv_in", x)
    x = vae.mid_resnet1.forward(x)
    record("mid_resnet1", x)
    if vae.mid_attn is not None:
        x = vae.mid_attn.forward(x)
        record("mid_attention", x)
    x = vae.mid_resnet2.forward(x)
    record("mid_resnet2", x)
    for index, (resnets, upsample) in enumerate(vae._up_stages):
        for resnet in resnets:
            x = resnet.forward(x)
        if upsample is not None:
            x = upsample.forward(x)
        record(f"up_block_{index}", x)
    x = mx.transpose(x, (0, 2, 3, 1))
    x = vae.norm_out(x)
    record("conv_norm_out", mx.transpose(x, (0, 3, 1, 2)))
    x = mx.sigmoid(x) * x
    output = vae.conv_out(x)
    output = mx.transpose(output, (0, 3, 1, 2))
    mx.eval(output)
    values = np.asarray(output.astype(mx.float32), dtype=np.float32).reshape(-1)
    cfg = session["vae_cfg"]
    payload = {
        "latent_shape": [1, 16, 2, 2],
        "latent": latent_values.tolist(),
        "latent_channels": 16,
        "block_out": cfg.get("block_out_channels", [128, 256, 512, 512]),
        "resnets": cfg.get("layers_per_block", 3),
        "groups": cfg.get("norm_num_groups", 32),
        "scaling": vae.scaling_factor,
        "shift": vae.shift_factor,
        "attention": cfg.get("mid_block_add_attention", True),
        "output_shape": list(output.shape),
        "output": values.tolist(),
        "stages": stages,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"main real VAE: shape={output.shape}, seconds={time.monotonic() - started:.3f}")


if __name__ == "__main__":
    main()
