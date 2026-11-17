import argparse
import json
import sys
from pathlib import Path

import numpy as np


def add_tensor(tensors, name, shape):
    count = int(np.prod(shape))
    values = np.array(
        [((index + len(name)) % 13) * 0.015 - 0.07 for index in range(count)],
        dtype=np.float32,
    ).reshape(shape)
    tensors[name] = values


def to_diffusers(name):
    if name.startswith("mid_resnet1."):
        return name.replace("mid_resnet1.", "decoder.mid_block.resnets.0.", 1)
    if name.startswith("mid_resnet2."):
        return name.replace("mid_resnet2.", "decoder.mid_block.resnets.1.", 1)
    if name.startswith("mid_attn."):
        return name.replace("mid_attn.", "decoder.mid_block.attentions.0.", 1)
    for prefix in ("up1_", "up2_", "up3_", "up4_"):
        if name.startswith(prefix):
            stage = int(prefix[2]) - 1
            suffix = name[len(prefix):]
            if suffix.startswith("resnets."):
                return f"decoder.up_blocks.{stage}.{suffix}"
            if suffix.startswith("up."):
                return f"decoder.up_blocks.{stage}.upsamplers.0.{suffix.removeprefix('up.')}"
    if name.startswith("conv_in."):
        return f"decoder.{name}"
    if name.startswith("conv_out."):
        return f"decoder.{name}"
    if name.startswith("norm_out."):
        return name.replace("norm_out.", "decoder.conv_norm_out.", 1)
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.main_root)

    import mlx.core as mx
    from backend.engine.common.codecs.vae.decoder import VAEDecoder
    from backend.engine.runtime.mlx import MLXContext

    ctx = MLXContext()
    model = VAEDecoder(
        latent_channels=2,
        ctx=ctx,
        scaling_factor=1.0,
        shift_factor=0.0,
        block_out_channels=(32, 32),
        layers_per_block=1,
        norm_num_groups=32,
        mid_block_add_attention=True,
    )
    model._build_param_map()
    tensors = {}
    for name, parameter in model._param_map.items():
        add_tensor(tensors, name, parameter.shape)
    weights = {name: mx.array(value, dtype=mx.bfloat16) for name, value in tensors.items()}
    model.load_weights(list(weights.items()), strict=True, ctx=ctx)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    published_weights = {}
    for name, value in weights.items():
        if value.ndim == 4 and name.endswith("weight"):
            value = mx.transpose(value, (0, 3, 1, 2))
        published_weights[to_diffusers(name)] = value
    mx.save_safetensors(str(output_dir / "fixture.safetensors"), published_weights)
    latent_values = np.array([index * 0.1 - 0.2 for index in range(8)], dtype=np.float32)
    latents = mx.array(latent_values.reshape(1, 2, 2, 2), dtype=mx.bfloat16)
    output = model.forward(latents)
    mx.eval(output)
    values = np.asarray(output.astype(mx.float32), dtype=np.float32).reshape(-1)
    payload = {
        "latent_shape": [1, 2, 2, 2],
        "latent": latent_values.tolist(),
        "latent_channels": 2,
        "block_out": [32, 32],
        "resnets": 1,
        "groups": 32,
        "scaling": 1.0,
        "shift": 0.0,
        "attention": True,
        "output_shape": list(output.shape),
        "output": values.tolist(),
    }
    (output_dir / "expected.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"main tiny VAE output: shape={output.shape}, max_abs={np.max(np.abs(values))}")


if __name__ == "__main__":
    main()
