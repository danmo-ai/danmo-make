use danmo_mlx::{Array, Error, LinearWeight, Result, Stream};

use super::Qwen3Config;
use crate::weights::{MlxWeights, WeightDtype};

pub fn load_directory(directory: &std::path::Path, dtype: WeightDtype) -> Result<MlxWeights> {
    crate::weights::load_mlx_directory(directory, dtype, None)
}

pub fn encode_ids(
    weights: &MlxWeights,
    config: Qwen3Config,
    prefix: &str,
    ids: &[i32],
    valid: &[bool],
    selected: &[i64],
    stream: &Stream,
) -> Result<Array> {
    if ids.is_empty() || ids.len() != valid.len() || selected.is_empty() {
        return Err(Error::new(
            "Qwen3 MLX encode",
            "empty or mismatched token inputs",
        ));
    }
    let last = selected.iter().max().copied().unwrap_or(0);
    if last < 1 || last > config.layers {
        return Err(Error::new(
            "Qwen3 MLX encode",
            "selected layer is outside the tower",
        ));
    }
    let sequence = ids.len() as i32;
    let indices = Array::from_i32_slice(ids, &[sequence])?;
    let embedding = weights.embedding(&format!("{prefix}embed_tokens.weight"), stream)?;
    let mut hidden = embedding
        .take_axis(&indices, 0, stream)?
        .reshape(&[1, sequence, config.hidden as i32], stream)?
        .astype(embedding.dtype(), stream)?;
    let mask = causal_padding_mask(valid, stream)?;
    let (cos, sin) = position_embeddings(config, sequence, hidden.dtype(), stream)?;
    let mut captured = Vec::new();
    for layer in 0..last {
        hidden = layer_forward(
            weights, config, prefix, layer, &hidden, &mask, &cos, &sin, stream,
        )?;
        if selected.contains(&(layer + 1)) {
            captured.push(hidden.clone());
        }
    }
    let mut output = Array::concatenate_axis(&captured, 2, stream)?;
    let valid_count = valid.iter().filter(|value| **value).count() as i32;
    output = output.slice(
        &[0, 0, 0],
        &[1, valid_count, output.shape()[2]],
        &[1, 1, 1],
        stream,
    )?;
    Ok(output)
}

fn layer_forward(
    weights: &MlxWeights,
    config: Qwen3Config,
    prefix: &str,
    layer: i64,
    hidden: &Array,
    mask: &Array,
    cos: &Array,
    sin: &Array,
    stream: &Stream,
) -> Result<Array> {
    let path = format!("{prefix}layers.{layer}");
    let normed = hidden.rms_norm(
        Some(weight(weights, &path, "input_layernorm.weight")?),
        1e-6,
        stream,
    )?;
    let attended = attention(weights, config, &path, &normed, mask, cos, sin, stream)?;
    let hidden = Array::add(hidden, &attended, stream)?;
    let normed = hidden.rms_norm(
        Some(weight(weights, &path, "post_attention_layernorm.weight")?),
        1e-6,
        stream,
    )?;
    let gate = linear(weights, &path, "mlp.gate_proj.weight")?
        .apply(&normed, None, stream)?
        .silu(stream)?;
    let up = linear(weights, &path, "mlp.up_proj.weight")?.apply(&normed, None, stream)?;
    let mid = Array::multiply(&gate, &up, stream)?;
    let mlp = linear(weights, &path, "mlp.down_proj.weight")?.apply(&mid, None, stream)?;
    Array::add(&hidden, &mlp, stream)
}

fn attention(
    weights: &MlxWeights,
    config: Qwen3Config,
    prefix: &str,
    hidden: &Array,
    mask: &Array,
    cos: &Array,
    sin: &Array,
    stream: &Stream,
) -> Result<Array> {
    let shape = hidden.shape();
    let (batch, sequence) = (shape[0], shape[1]);
    let head = config.head_dim as i32;
    let heads = config.heads as i32;
    let kv_heads = config.kv_heads as i32;
    let mut q = linear(weights, prefix, "self_attn.q_proj.weight")?
        .apply(hidden, None, stream)?
        .reshape(&[batch, sequence, heads, head], stream)?
        .rms_norm(
            Some(weight(weights, prefix, "self_attn.q_norm.weight")?),
            1e-6,
            stream,
        )?;
    let mut k = linear(weights, prefix, "self_attn.k_proj.weight")?
        .apply(hidden, None, stream)?
        .reshape(&[batch, sequence, kv_heads, head], stream)?
        .rms_norm(
            Some(weight(weights, prefix, "self_attn.k_norm.weight")?),
            1e-6,
            stream,
        )?;
    let mut v = linear(weights, prefix, "self_attn.v_proj.weight")?
        .apply(hidden, None, stream)?
        .reshape(&[batch, sequence, kv_heads, head], stream)?;
    q = apply_rope(&q, cos, sin, stream)?;
    k = apply_rope(&k, cos, sin, stream)?;
    let groups = heads / kv_heads;
    if groups > 1 {
        k = k.repeat_axis(groups, 2, stream)?;
        v = v.repeat_axis(groups, 2, stream)?;
    }
    q = q.transpose_axes(&[0, 2, 1, 3], stream)?;
    k = k.transpose_axes(&[0, 2, 1, 3], stream)?;
    v = v.transpose_axes(&[0, 2, 1, 3], stream)?;
    let compute = q.astype(danmo_mlx::Dtype::Float32, stream)?;
    let keys = k.astype(danmo_mlx::Dtype::Float32, stream)?;
    let values = v.astype(danmo_mlx::Dtype::Float32, stream)?;
    let output = Array::scaled_dot_product_attention(
        &compute,
        &keys,
        &values,
        1.0 / (config.head_dim as f32).sqrt(),
        "",
        Some(mask),
        stream,
    )?
    .astype(q.dtype(), stream)?
    .transpose_axes(&[0, 2, 1, 3], stream)?
    .reshape(&[batch, sequence, heads * head], stream)?;
    linear(weights, prefix, "self_attn.o_proj.weight")?.apply(&output, None, stream)
}

fn apply_rope(input: &Array, cos: &Array, sin: &Array, stream: &Stream) -> Result<Array> {
    let dim = input.shape()[3];
    let parts = input.split_sections(&[dim / 2], -1, stream)?;
    let rotated =
        Array::concatenate_axis(&[parts[1].negative(stream)?, parts[0].clone()], -1, stream)?;
    Array::add(
        &Array::multiply(input, cos, stream)?,
        &Array::multiply(&rotated, sin, stream)?,
        stream,
    )
}

fn position_embeddings(
    config: Qwen3Config,
    sequence: i32,
    dtype: danmo_mlx::Dtype,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let mut cos = Vec::with_capacity((sequence * config.head_dim as i32) as usize);
    let mut sin = cos.clone();
    for position in 0..sequence {
        for pair in 0..config.head_dim / 2 {
            let angle = position as f32
                / config
                    .rope_theta
                    .powf(pair as f32 * 2.0 / config.head_dim as f32);
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
        for pair in 0..config.head_dim / 2 {
            let angle = position as f32
                / config
                    .rope_theta
                    .powf(pair as f32 * 2.0 / config.head_dim as f32);
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    let shape = &[1, sequence, config.head_dim as i32];
    Ok((
        Array::from_f32_slice(&cos, shape)?
            .expand_dims(2, stream)?
            .astype(dtype, stream)?,
        Array::from_f32_slice(&sin, shape)?
            .expand_dims(2, stream)?
            .astype(dtype, stream)?,
    ))
}

fn causal_padding_mask(valid: &[bool], _stream: &Stream) -> Result<Array> {
    let sequence = valid.len();
    let mut mask = vec![f32::NEG_INFINITY; sequence * sequence];
    for query in 0..sequence {
        for key in 0..sequence {
            if key <= query && valid[key] {
                mask[query * sequence + key] = 0.0;
            }
        }
    }
    Array::from_f32_slice(&mask, &[1, 1, sequence as i32, sequence as i32])
}

fn qualified(prefix: &str, name: &str) -> String {
    if prefix.ends_with('.') {
        format!("{prefix}{name}")
    } else {
        format!("{prefix}.{name}")
    }
}

fn weight<'a>(weights: &'a MlxWeights, prefix: &str, name: &str) -> Result<&'a Array> {
    weights.tensor(&qualified(prefix, name))
}

fn linear<'a>(weights: &'a MlxWeights, prefix: &str, name: &str) -> Result<&'a LinearWeight> {
    weights.linear(&qualified(prefix, name))
}

#[cfg(test)]
mod tests {
    use super::Qwen3Config;
    use super::{encode_ids, load_directory};
    use crate::weights::{MlxWeights, WeightDtype};
    use danmo_mlx::{SafeTensors, Stream};
    use std::collections::BTreeMap;

    #[test]
    #[ignore]
    fn python_main_qwen3_fixture_matches_rust() {
        let root =
            std::path::PathBuf::from(std::env::var("DANQING_TEST_QWEN3_PARITY_DIR").unwrap());
        let payload: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("expected.json")).unwrap())
                .unwrap();
        let ids: Vec<i32> = payload["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap() as i32)
            .collect();
        let valid: Vec<bool> = payload["valid"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_bool().unwrap())
            .collect();
        let selected: Vec<i64> = payload["selected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap())
            .collect();
        #[cfg(feature = "mlx-metal")]
        let stream = Stream::gpu().unwrap();
        #[cfg(not(feature = "mlx-metal"))]
        let stream = Stream::cpu().unwrap();
        let io = Stream::cpu().unwrap();
        let tensors = SafeTensors::load(&root.join("fixture.safetensors"), &io).unwrap();
        let mut weights = BTreeMap::new();
        for name in tensors.names().unwrap() {
            weights.insert(name.clone(), tensors.get(&name).unwrap().unwrap());
        }
        let weights = MlxWeights::from_dense(weights);
        let config = Qwen3Config {
            vocab: 8,
            hidden: 4,
            layers: 2,
            heads: 2,
            kv_heads: 1,
            head_dim: 4,
            intermediate: 8,
            rope_theta: 1_000_000.0,
        };
        let actual = encode_ids(&weights, config, "model.", &ids, &valid, &selected, &stream)
            .unwrap()
            .to_vec_f32(&stream)
            .unwrap();
        let expected: Vec<f32> = payload["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(actual.len(), expected.len());
        let mut max_error = 0.0f32;
        for (actual, expected) in actual.iter().zip(expected) {
            let error = (actual - expected).abs();
            max_error = max_error.max(error);
            let exponent = expected.abs().max(f32::MIN_POSITIVE).log2().floor();
            let allowed = 4.0 * 2f32.powf(exponent - 7.0) + 1e-4;
            assert!(
                error <= allowed,
                "error {error} exceeds four BF16 ULP ({allowed})"
            );
        }
        println!("main/Rust Qwen3 max abs error: {max_error}");
    }

    #[test]
    #[ignore]
    fn real_qwen3_encoder_matches_main_embedding() {
        let encoder =
            std::path::PathBuf::from(std::env::var("DANQING_TEST_QWEN3_ENCODER_DIR").unwrap());
        let payload: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(std::env::var("DANQING_TEST_QWEN3_REAL_JSON").unwrap())
                .unwrap(),
        )
        .unwrap();
        #[cfg(feature = "mlx-metal")]
        let stream = Stream::gpu().unwrap();
        #[cfg(not(feature = "mlx-metal"))]
        let stream = Stream::cpu().unwrap();
        let started = std::time::Instant::now();
        let weights = load_directory(&encoder, WeightDtype::Native).unwrap();
        let config = Qwen3Config {
            vocab: 151_936,
            hidden: 2560,
            layers: 36,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            intermediate: 9728,
            rope_theta: 1_000_000.0,
        };
        let ids: Vec<i32> = payload["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap() as i32)
            .collect();
        let valid = vec![true; ids.len()];
        for layer in [1_i64, 2, 10, 20, 30, 35] {
            let actual = encode_ids(&weights, config, "model.", &ids, &valid, &[layer], &stream)
                .unwrap()
                .to_vec_f32(&stream)
                .unwrap();
            let expected: Vec<f32> = payload["layer_outputs"][layer.to_string()]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect();
            assert_eq!(actual.len(), expected.len());
            let mut max_error = 0.0f32;
            for (actual, expected) in actual.iter().zip(expected) {
                let error = (actual - expected).abs();
                max_error = max_error.max(error);
                let exponent = expected.abs().max(f32::MIN_POSITIVE).log2().floor();
                let allowed = 4.0 * 2f32.powf(exponent - 7.0) + 1e-4;
                assert!(
                    error <= allowed,
                    "layer {layer}: error {error} exceeds four BF16 ULP ({allowed})"
                );
            }
            println!("layer {layer}: max abs error {max_error}");
        }
        println!("real Qwen3 encode: {:.3}s", started.elapsed().as_secs_f64());
    }
}
