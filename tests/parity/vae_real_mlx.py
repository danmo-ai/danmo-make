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
    parser.add_argument("--native-dtype", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, args.main_root)

    import mlx.core as mx
    from backend.engine.common.codecs.vae.decoder import build_standard_vae_preview_session
    from backend.engine.common.model.base import TransformerBase
    from backend.engine.runtime.mlx import MLXContext

    started = time.monotonic()
    ctx = MLXContext()
    session = build_standard_vae_preview_session(ctx, Path(args.vae))
    if session is None:
        raise RuntimeError("VAE preview session did not load")
    if args.latent_json:
        source = json.loads(Path(args.latent_json).read_text(encoding="utf-8"))
        latent_shape = source["latent_shape"]
        latent_values = np.asarray(source["denoised"], dtype=np.float32)
    else:
        latent_shape = [1, 16, 2, 2]
        latent_values = np.array(
            [((index % 31) * 0.02 - 0.3) for index in range(16 * 2 * 2)],
            dtype=np.float32,
        )
    latents = mx.array(latent_values.reshape(latent_shape), dtype=mx.bfloat16)
    vae = session["vae"]
    if args.native_dtype:
        for key, param in list(vae._param_map.items()):
            if param.dtype == mx.bfloat16:
                continue
            native = param.astype(mx.bfloat16)
            vae._param_map[key] = native
            TransformerBase._rebind_param_by_key(vae, key, native)
    if vae.scaling_factor != 1.0 or vae.shift_factor != 0.0:
        latents = (latents / vae.scaling_factor) + vae.shift_factor
    stages = {}

    def record(name, value):
        stages[name] = np.asarray(value.astype(mx.float32), dtype=np.float32).reshape(-1).tolist()

    def run_resnet(resnet, prefix, value):
        h = mx.transpose(value, (0, 2, 3, 1))
        h = resnet.norm1(h)
        record(f"{prefix}.norm1", mx.transpose(h, (0, 3, 1, 2)))
        h = mx.sigmoid(h) * h
        h = resnet.conv1(h)
        record(f"{prefix}.conv1", mx.transpose(h, (0, 3, 1, 2)))
        h = resnet.norm2(h)
        record(f"{prefix}.norm2", mx.transpose(h, (0, 3, 1, 2)))
        h = mx.sigmoid(h) * h
        h = resnet.conv2(h)
        record(f"{prefix}.conv2", mx.transpose(h, (0, 3, 1, 2)))
        h = mx.transpose(h, (0, 3, 1, 2))
        if resnet.conv_shortcut is not None:
            shortcut = resnet.conv_shortcut(mx.transpose(value, (0, 2, 3, 1)))
            shortcut = mx.transpose(shortcut, (0, 3, 1, 2))
            record(f"{prefix}.shortcut", shortcut)
            value = shortcut
        return value + h

    x = mx.transpose(latents, (0, 2, 3, 1))
    x = vae.conv_in(x)
    x = mx.transpose(x, (0, 3, 1, 2))
    record("conv_in", x)
    x = run_resnet(vae.mid_resnet1, "decoder.mid_block.resnets.0", x)
    record("mid_resnet1", x)
    if vae.mid_attn is not None:
        x = vae.mid_attn.forward(x)
        record("mid_attention", x)
    x = run_resnet(vae.mid_resnet2, "decoder.mid_block.resnets.1", x)
    record("mid_resnet2", x)
    for index, (resnets, upsample) in enumerate(vae._up_stages):
        for resnet_index, resnet in enumerate(resnets):
            x = run_resnet(resnet, f"decoder.up_blocks.{index}.resnets.{resnet_index}", x)
            record(f"up_{index}_resnet_{resnet_index}", x)
        if upsample is not None:
            x = upsample.forward(x)
            record(f"up_{index}_upsample", x)
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
        "native_dtype": "checkpoint" if args.native_dtype else None,
        "latent_shape": latent_shape,
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
