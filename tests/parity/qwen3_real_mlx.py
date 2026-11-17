import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.main_root)

    import mlx.core as mx
    from backend.engine.common.codecs.text_encoders.qwen3_mlx import build_zimage_mlx_encoder
    from backend.engine.runtime.mlx import MLXContext

    ids = [151644, 872, 198, 64, 2518, 23739, 151645, 198, 151644, 77091, 198]
    started = time.monotonic()
    ctx = MLXContext()
    model = build_zimage_mlx_encoder(args.encoder, ctx)
    input_ids = mx.array([ids], dtype=mx.int32)
    attention_mask = mx.ones((1, len(ids)), dtype=mx.int32)
    hidden_states = model.embed_tokens(input_ids).astype(mx.bfloat16)
    causal_mask = model._get_causal_mask(attention_mask, 1, hidden_states, len(ids))
    position_ids = model._build_position_ids(1, len(ids))
    position_embeddings = model.rotary_emb(hidden_states, position_ids)
    checkpoint_outputs = {}
    checkpoints = {1, 2, 10, 20, 30, 35}
    for index, layer in enumerate(model.layers[:-1], 1):
        hidden_states = layer(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
        )
        if index in checkpoints:
            checkpoint = hidden_states[:, :len(ids), :].astype(mx.bfloat16)
            mx.eval(checkpoint)
            checkpoint_outputs[str(index)] = np.asarray(
                checkpoint.astype(mx.float32), dtype=np.float32
            ).reshape(-1).tolist()
    result = hidden_states[:, :len(ids), :].astype(mx.bfloat16)
    mx.eval(result)
    output = np.asarray(result.astype(mx.float32), dtype=np.float32).reshape(-1)
    payload = {
        "ids": ids,
        "valid": [True] * len(ids),
        "selected": [35],
        "output_shape": list(result.shape),
        "output": output.tolist(),
        "layer_outputs": checkpoint_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"main real Qwen3: shape={result.shape}, seconds={time.monotonic() - started:.3f}")


if __name__ == "__main__":
    main()
