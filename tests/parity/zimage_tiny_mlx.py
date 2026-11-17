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


def add_block(tensors, prefix, config, modulated):
    dim = config.dim
    ffn = int(dim / 3 * 8)
    for name in ("to_q", "to_k", "to_v", "to_out"):
        add_tensor(tensors, f"{prefix}.attention.{name}.weight", (dim, dim))
    add_tensor(tensors, f"{prefix}.attention.norm_q.weight", (dim // config.num_heads,))
    add_tensor(tensors, f"{prefix}.attention.norm_k.weight", (dim // config.num_heads,))
    for name in ("attn_norm1", "attn_norm2", "ffn_norm1", "ffn_norm2"):
        add_tensor(tensors, f"{prefix}.{name}.weight", (dim,))
    add_tensor(tensors, f"{prefix}.feed_forward.w1.weight", (ffn, dim))
    add_tensor(tensors, f"{prefix}.feed_forward.w2.weight", (dim, ffn))
    add_tensor(tensors, f"{prefix}.feed_forward.w3.weight", (ffn, dim))
    if modulated:
        add_tensor(tensors, f"{prefix}.adaLN_modulation.0.weight", (4 * dim, config.adaln_dim))
        add_tensor(tensors, f"{prefix}.adaLN_modulation.0.bias", (4 * dim,))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.main_root)

    import mlx.core as mx
    from backend.engine.config.model_configs import ZImageConfig
    from backend.engine.families.z_image.transformer_mlx import ZImageDiTMLX
    from backend.engine.runtime.mlx import MLXContext

    config = ZImageConfig()
    config.in_channels = 2
    config.dim = 6
    config.hidden_dim = 6
    config.num_heads = 1
    config.num_layers = 1
    config.num_refiner_layers = 1
    config.cap_feat_dim = 4
    config.text_dim = 4
    config.frequency_embedding_size = 256
    config.timestep_mid = 1024
    config.adaln_dim = 6
    config.rope_axes_dim = (2, 2, 2)
    config.use_mlx_compile = False
    config.use_mlx_compile_auto = False
    config.teacache_mode = "none"
    config.lemica_mode = "none"

    tensors = {}
    token = config.patch_size * config.patch_size * config.in_channels
    add_tensor(tensors, "x_embedder.weight", (config.dim, token))
    add_tensor(tensors, "x_embedder.bias", (config.dim,))
    add_tensor(tensors, "t_embedder.linear1.weight", (config.timestep_mid, 256))
    add_tensor(tensors, "t_embedder.linear1.bias", (config.timestep_mid,))
    add_tensor(tensors, "t_embedder.linear2.weight", (config.adaln_dim, config.timestep_mid))
    add_tensor(tensors, "t_embedder.linear2.bias", (config.adaln_dim,))
    add_tensor(tensors, "cap_norm.weight", (config.cap_feat_dim,))
    add_tensor(tensors, "cap_embedder.weight", (config.dim, config.cap_feat_dim))
    add_tensor(tensors, "cap_embedder.bias", (config.dim,))
    add_block(tensors, "noise_refiner.0", config, True)
    add_block(tensors, "context_refiner.0", config, False)
    add_block(tensors, "layers.0", config, True)
    add_tensor(tensors, "final_layer.adaLN_modulation.0.weight", (config.dim, config.adaln_dim))
    add_tensor(tensors, "final_layer.adaLN_modulation.0.bias", (config.dim,))
    add_tensor(tensors, "final_layer.linear.weight", (token, config.dim))
    add_tensor(tensors, "final_layer.linear.bias", (token,))
    tensors["x_pad_token"] = np.zeros((1, config.dim), dtype=np.float32)
    tensors["cap_pad_token"] = np.zeros((1, config.dim), dtype=np.float32)
    weights = {name: mx.array(value, dtype=mx.bfloat16) for name, value in tensors.items()}

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output_dir / "fixture.safetensors"), weights)

    ctx = MLXContext()
    model = ZImageDiTMLX(config, ctx)
    model.rope.axes_dims = [2, 2, 2]
    model.rope.freqs_cis = model.rope._precompute([2, 2, 2], [1024, 512, 512], 256.0)
    model.load_weights(list(weights.items()), strict=True, ctx=ctx)
    latent_values = np.array([index * 0.1 - 0.2 for index in range(8)], dtype=np.float32)
    caption_values = np.array([index * 0.07 - 0.15 for index in range(8)], dtype=np.float32)
    latents = mx.array(latent_values.reshape(2, 1, 2, 2), dtype=mx.bfloat16)
    caption = mx.array(caption_values.reshape(2, 4), dtype=mx.bfloat16)
    sigma = 0.25
    output = model.forward(latents, 1.0 - sigma, txt_embeds=caption)
    mx.eval(output)
    expected = np.asarray(output.astype(mx.float32), dtype=np.float32).reshape(-1)
    payload = {
        "sigma": sigma,
        "latent_shape": [2, 2, 2],
        "caption_shape": [2, 4],
        "latent": latent_values.tolist(),
        "caption": caption_values.tolist(),
        "output_shape": list(output.shape),
        "output": expected.tolist(),
    }
    (output_dir / "expected.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"main tiny DiT output: shape={output.shape}, max_abs={np.max(np.abs(expected))}")


if __name__ == "__main__":
    main()
