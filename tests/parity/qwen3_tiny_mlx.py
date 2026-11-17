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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.main_root)

    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from backend.engine.common.codecs.text_encoders.qwen3_mlx import Qwen3EncoderModel

    vocab = 8
    hidden = 4
    layers = 2
    heads = 2
    kv_heads = 1
    head_dim = 4
    intermediate = 8
    prefix = "model."
    tensors = {}
    add_tensor(tensors, f"{prefix}embed_tokens.weight", (vocab, hidden))
    for layer in range(layers):
        path = f"{prefix}layers.{layer}"
        add_tensor(tensors, f"{path}.input_layernorm.weight", (hidden,))
        add_tensor(tensors, f"{path}.self_attn.q_proj.weight", (heads * head_dim, hidden))
        add_tensor(tensors, f"{path}.self_attn.k_proj.weight", (kv_heads * head_dim, hidden))
        add_tensor(tensors, f"{path}.self_attn.v_proj.weight", (kv_heads * head_dim, hidden))
        add_tensor(tensors, f"{path}.self_attn.o_proj.weight", (hidden, heads * head_dim))
        add_tensor(tensors, f"{path}.self_attn.q_norm.weight", (head_dim,))
        add_tensor(tensors, f"{path}.self_attn.k_norm.weight", (head_dim,))
        add_tensor(tensors, f"{path}.post_attention_layernorm.weight", (hidden,))
        add_tensor(tensors, f"{path}.mlp.gate_proj.weight", (intermediate, hidden))
        add_tensor(tensors, f"{path}.mlp.up_proj.weight", (intermediate, hidden))
        add_tensor(tensors, f"{path}.mlp.down_proj.weight", (hidden, intermediate))
    add_tensor(tensors, f"{prefix}norm.weight", (hidden,))
    weights = {name: mx.array(value, dtype=mx.bfloat16) for name, value in tensors.items()}

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output_dir / "fixture.safetensors"), weights)

    model = Qwen3EncoderModel(
        vocab_size=vocab,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        intermediate_size=intermediate,
        head_dim=head_dim,
    )
    model.update(tree_unflatten([(name.removeprefix(prefix), value) for name, value in weights.items()]))
    ids = [1, 2, 3]
    input_ids = mx.array([ids], dtype=mx.int32)
    attention_mask = mx.ones((1, len(ids)), dtype=mx.int32)
    hidden_states = model.embed_tokens(input_ids).astype(mx.bfloat16)
    causal_mask = model._get_causal_mask(attention_mask, 1, hidden_states, len(ids))
    position_ids = model._build_position_ids(1, len(ids))
    position_embeddings = model.rotary_emb(hidden_states, position_ids)
    for layer in model.layers[:-1]:
        hidden_states = layer(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
        )
    result = hidden_states[:, :len(ids), :].astype(mx.bfloat16)
    mx.eval(result)
    output = np.asarray(result.astype(mx.float32), dtype=np.float32).reshape(-1)
    payload = {
        "ids": ids,
        "valid": [True] * len(ids),
        "selected": [1],
        "output_shape": list(result.shape),
        "output": output.tolist(),
    }
    (output_dir / "expected.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"main Qwen3 output: shape={result.shape}, max_abs={np.max(np.abs(output))}")


if __name__ == "__main__":
    main()
