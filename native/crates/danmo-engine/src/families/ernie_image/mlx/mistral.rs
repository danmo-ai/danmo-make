use std::collections::BTreeMap;
use std::path::Path;

use danmo_mlx::{Array, Dtype, Error, Result, Stream};

use super::config::MistralConfig;
use crate::weights::{MlxWeights, WeightDtype};

pub fn load_text_encoder(directory: &Path, stream: &Stream) -> Result<MlxWeights> {
    let weights = crate::weights::load_mlx_directory(directory, WeightDtype::Native, None)?
        .retain(|name| name.starts_with("language_model.model."));
    let names: BTreeMap<String, String> = weights
        .names()
        .map(|name| {
            let target = name
                .strip_prefix("language_model.")
                .unwrap_or(name)
                .to_string();
            (name.to_string(), target)
        })
        .collect();
    if names.is_empty() {
        return Err(Error::new(
            "ERNIE text encoder",
            "language_model tensors are missing",
        ));
    }
    weights
        .rename(&names)?
        .materialize_embedding("model.embed_tokens.weight", stream)
}

pub fn load_pe(directory: &Path, stream: &Stream) -> Result<MlxWeights> {
    crate::weights::load_mlx_directory(directory, WeightDtype::Native, None)?
        .materialize_embedding("model.embed_tokens.weight", stream)
}

pub fn encode(
    weights: &MlxWeights,
    config: MistralConfig,
    ids: &[i32],
    stream: &Stream,
) -> Result<Array> {
    if ids.is_empty() || config.layers < 2 {
        return Err(Error::new(
            "ERNIE text encoder",
            "empty tokens or insufficient hidden layers",
        ));
    }
    let sequence = ids.len() as i32;
    let indices = Array::from_i32_slice(ids, &[sequence])?;
    let embedding = weights.embedding("model.embed_tokens.weight", stream)?;
    let mut hidden = embedding
        .take_axis(&indices, 0, stream)?
        .reshape(&[1, sequence, config.hidden], stream)?;
    let mask = causal_mask(sequence, sequence, stream)?;
    let (cos, sin) = position_embeddings(config, 0, sequence, hidden.dtype(), stream)?;
    for layer in 0..config.layers - 1 {
        hidden = layer_forward(weights, config, layer, &hidden, &mask, &cos, &sin, stream)?;
    }
    Ok(hidden)
}

fn layer_forward(
    weights: &MlxWeights,
    config: MistralConfig,
    layer: i32,
    hidden: &Array,
    mask: &Array,
    cos: &Array,
    sin: &Array,
    stream: &Stream,
) -> Result<Array> {
    let prefix = format!("model.layers.{layer}");
    let input_norm = rms_norm(
        hidden,
        weights.tensor(&format!("{prefix}.input_layernorm.weight"))?,
        config.rms_eps,
        stream,
    )?;
    let attended = attention(
        weights,
        config,
        &prefix,
        &input_norm,
        None,
        None,
        mask,
        cos,
        sin,
        0,
        stream,
    )?
    .0;
    let hidden = Array::add(hidden, &attended, stream)?;
    let post_norm = rms_norm(
        &hidden,
        weights.tensor(&format!("{prefix}.post_attention_layernorm.weight"))?,
        config.rms_eps,
        stream,
    )?;
    let mlp = mlp(weights, &prefix, &post_norm, stream)?;
    Array::add(&hidden, &mlp, stream)
}

fn mlp(weights: &MlxWeights, prefix: &str, hidden: &Array, stream: &Stream) -> Result<Array> {
    let gate = weights
        .linear(&format!("{prefix}.mlp.gate_proj.weight"))?
        .apply(hidden, None, stream)?
        .silu(stream)?;
    let up = weights
        .linear(&format!("{prefix}.mlp.up_proj.weight"))?
        .apply(hidden, None, stream)?;
    let mid = Array::multiply(&gate, &up, stream)?;
    weights
        .linear(&format!("{prefix}.mlp.down_proj.weight"))?
        .apply(&mid, None, stream)
}

fn attention(
    weights: &MlxWeights,
    config: MistralConfig,
    prefix: &str,
    hidden: &Array,
    past_key: Option<&Array>,
    past_value: Option<&Array>,
    mask: &Array,
    cos: &Array,
    sin: &Array,
    start: i32,
    stream: &Stream,
) -> Result<(Array, Array, Array)> {
    let shape = hidden.shape();
    let (batch, sequence) = (shape[0], shape[1]);
    let q = weights
        .linear(&format!("{prefix}.self_attn.q_proj.weight"))?
        .apply(hidden, None, stream)?
        .reshape(&[batch, sequence, config.heads, config.head_dim], stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
    let k = weights
        .linear(&format!("{prefix}.self_attn.k_proj.weight"))?
        .apply(hidden, None, stream)?
        .reshape(&[batch, sequence, config.kv_heads, config.head_dim], stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
    let v = weights
        .linear(&format!("{prefix}.self_attn.v_proj.weight"))?
        .apply(hidden, None, stream)?
        .reshape(&[batch, sequence, config.kv_heads, config.head_dim], stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
    let mut q = apply_rope(&q, cos, sin, stream)?;
    let k = apply_rope(&k, cos, sin, stream)?;
    let q_scale = llama_4_attention_scale(config, start, sequence, q.dtype(), stream)?;
    q = Array::multiply(&q, &q_scale, stream)?;
    let mut key = k;
    let mut value = v;
    if let (Some(past_key), Some(past_value)) = (past_key, past_value) {
        key = Array::concatenate_axis(&[past_key.clone(), key], 2, stream)?;
        value = Array::concatenate_axis(&[past_value.clone(), value], 2, stream)?;
    }
    let groups = config.heads / config.kv_heads;
    let mut key_attn = key.clone();
    let mut value_attn = value.clone();
    if groups > 1 {
        key_attn = key_attn.repeat_axis(groups, 1, stream)?;
        value_attn = value_attn.repeat_axis(groups, 1, stream)?;
    }
    let mask = mask.astype(q.dtype(), stream)?;
    let output = Array::scaled_dot_product_attention(
        &q,
        &key_attn,
        &value_attn,
        1.0 / (config.head_dim as f32).sqrt(),
        "",
        Some(&mask),
        stream,
    )?
    .transpose_axes(&[0, 2, 1, 3], stream)?
    .reshape(&[batch, sequence, config.heads * config.head_dim], stream)?;
    let output = weights
        .linear(&format!("{prefix}.self_attn.o_proj.weight"))?
        .apply(&output, None, stream)?;
    Ok((output, key, value))
}

pub struct LmLayerCache {
    key: Array,
    value: Array,
}

pub struct LmCache {
    layers: Vec<LmLayerCache>,
    length: i32,
}

pub fn step_logits(
    weights: &MlxWeights,
    config: MistralConfig,
    ids: &[i32],
    cache: &mut Option<LmCache>,
    stream: &Stream,
) -> Result<Vec<f32>> {
    if ids.is_empty() {
        return Err(Error::new("ERNIE PE", "token step is empty"));
    }
    let start = cache.as_ref().map(|value| value.length).unwrap_or(0);
    let sequence = ids.len() as i32;
    let indices = Array::from_i32_slice(ids, &[sequence])?;
    let embedding = weights.embedding("model.embed_tokens.weight", stream)?;
    let mut hidden = embedding
        .take_axis(&indices, 0, stream)?
        .reshape(&[1, sequence, config.hidden], stream)?;
    let total = start + sequence;
    let mask = if start == 0 {
        causal_mask(sequence, total, stream)?
    } else {
        Array::from_f32_slice(
            &vec![0.0; (sequence * total) as usize],
            &[1, 1, sequence, total],
        )?
    };
    let (cos, sin) = position_embeddings(config, start, sequence, hidden.dtype(), stream)?;
    let mut next = Vec::with_capacity(config.layers as usize);
    for layer in 0..config.layers {
        let prefix = format!("model.layers.{layer}");
        let input_norm = rms_norm(
            &hidden,
            weights.tensor(&format!("{prefix}.input_layernorm.weight"))?,
            config.rms_eps,
            stream,
        )?;
        let past = cache
            .as_ref()
            .and_then(|value| value.layers.get(layer as usize));
        let (attended, key, value) = attention(
            weights,
            config,
            &prefix,
            &input_norm,
            past.map(|value| &value.key),
            past.map(|value| &value.value),
            &mask,
            &cos,
            &sin,
            start,
            stream,
        )?;
        hidden = Array::add(&hidden, &attended, stream)?;
        let post_norm = rms_norm(
            &hidden,
            weights.tensor(&format!("{prefix}.post_attention_layernorm.weight"))?,
            config.rms_eps,
            stream,
        )?;
        hidden = Array::add(&hidden, &mlp(weights, &prefix, &post_norm, stream)?, stream)?;
        next.push(LmLayerCache { key, value });
    }
    let norm = weights.tensor("model.norm.weight")?;
    let last = hidden.slice(
        &[0, sequence - 1, 0],
        &[1, sequence, hidden.shape()[2]],
        &[1, 1, 1],
        stream,
    )?;
    let last = rms_norm(&last, norm, config.rms_eps, stream)?;
    let logits = if weights.has("lm_head.weight") {
        weights
            .linear("lm_head.weight")?
            .apply(&last, None, stream)?
    } else {
        let table = embedding.transpose_axes(&[1, 0], stream)?;
        Array::matmul(&last, &table, stream)?
    }
    .astype(Dtype::Float32, stream)?
    .reshape(&[config.vocab], stream)?;
    let mut held = vec![&logits];
    for layer in &next {
        held.push(&layer.key);
        held.push(&layer.value);
    }
    Array::eval_all(&held)?;
    *cache = Some(LmCache {
        layers: next,
        length: total,
    });
    logits.to_vec_f32(stream)
}

fn rms_norm(input: &Array, weight: &Array, eps: f32, stream: &Stream) -> Result<Array> {
    let dtype = input.dtype();
    let input = input.astype(Dtype::Float32, stream)?;
    let weight = weight.astype(Dtype::Float32, stream)?;
    input
        .rms_norm(Some(&weight), eps, stream)?
        .astype(dtype, stream)
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
    config: MistralConfig,
    start: i32,
    length: i32,
    dtype: Dtype,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let inv_freq = yarn_inv_freq(config, stream)?;
    let positions = Array::arange(
        start as f64,
        (start + length) as f64,
        1.0,
        Dtype::Float32,
        stream,
    )?
    .expand_dims(1, stream)?;
    let angles = Array::multiply(&positions, &inv_freq.expand_dims(0, stream)?, stream)?;
    let embedding = Array::concatenate_axis(&[angles.clone(), angles], -1, stream)?;
    let scale = Array::from_f32_slice(&[config.attention_scale], &[])?;
    let cos = Array::multiply(&embedding.cos(stream)?, &scale, stream)?
        .expand_dims(0, stream)?
        .expand_dims(1, stream)?
        .astype(dtype, stream)?;
    let sin = Array::multiply(&embedding.sin(stream)?, &scale, stream)?
        .expand_dims(0, stream)?
        .expand_dims(1, stream)?
        .astype(dtype, stream)?;
    Ok((cos, sin))
}

fn yarn_inv_freq(config: MistralConfig, stream: &Stream) -> Result<Array> {
    let dim = config.head_dim;
    let half = dim / 2;
    let arange = Array::arange(0.0, dim as f64, 2.0, Dtype::Float32, stream)?;
    let exponent = Array::divide(&arange, &Array::from_f32_slice(&[dim as f32], &[])?, stream)?;
    let pos_freq = Array::power(
        &Array::from_f32_slice(&[config.rope_theta], &[])?,
        &exponent,
        stream,
    )?;
    let one = Array::from_f32_slice(&[1.0], &[])?;
    let extrapolation = Array::divide(&one, &pos_freq, stream)?;
    let interpolation = Array::divide(
        &extrapolation,
        &Array::from_f32_slice(&[config.rope_factor], &[])?,
        stream,
    )?;
    let correction = |rotations: f32| {
        (dim as f32
            * (config.original_positions as f32 / (rotations * 2.0 * std::f32::consts::PI)).ln())
            / (2.0 * config.rope_theta.ln())
    };
    let low = correction(config.beta_fast).floor().max(0.0);
    let high = correction(config.beta_slow).ceil().min((dim - 1) as f32);
    let ramp = Array::arange(0.0, half as f64, 1.0, Dtype::Float32, stream)?;
    let ramp = Array::subtract(&ramp, &Array::from_f32_slice(&[low], &[])?, stream)?;
    let ramp = Array::divide(
        &ramp,
        &Array::from_f32_slice(&[(high - low).max(0.001)], &[])?,
        stream,
    )?
    .clip(0.0, 1.0, stream)?;
    let extrapolation_factor = Array::subtract(&one, &ramp, stream)?;
    Array::add(
        &Array::multiply(
            &interpolation,
            &Array::subtract(&one, &extrapolation_factor, stream)?,
            stream,
        )?,
        &Array::multiply(&extrapolation, &extrapolation_factor, stream)?,
        stream,
    )
}

fn llama_4_attention_scale(
    config: MistralConfig,
    start: i32,
    length: i32,
    dtype: Dtype,
    stream: &Stream,
) -> Result<Array> {
    let values: Vec<f32> = (start..start + length)
        .map(|position| {
            1.0 + config.llama_4_scaling_beta
                * (1.0 + (position as f32 / config.original_positions as f32).floor()).ln()
        })
        .collect();
    Array::from_f32_slice(&values, &[1, 1, length, 1])?.astype(dtype, stream)
}

fn causal_mask(query: i32, total: i32, stream: &Stream) -> Result<Array> {
    let _ = stream;
    let mut values = vec![f32::NEG_INFINITY; (query * total) as usize];
    for row in 0..query {
        for col in 0..=row {
            values[(row * total + col) as usize] = 0.0;
        }
    }
    Array::from_f32_slice(&values, &[1, 1, query, total])
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use danmo_mlx::{Array, Stream};

    use super::{
        causal_mask, encode, layer_forward, position_embeddings, step_logits, MistralConfig,
    };
    use crate::weights::MlxWeights;

    fn insert(
        tensors: &mut BTreeMap<String, Array>,
        name: String,
        rows: i32,
        cols: i32,
        scale: f32,
    ) {
        let values: Vec<f32> = (0..rows * cols)
            .map(|index| ((index % 17) as f32 - 8.0) * scale)
            .collect();
        tensors.insert(name, Array::from_f32_slice(&values, &[rows, cols]).unwrap());
    }

    fn weights(config: MistralConfig) -> MlxWeights {
        let mut tensors = BTreeMap::new();
        insert(
            &mut tensors,
            "model.embed_tokens.weight".into(),
            config.vocab,
            config.hidden,
            0.02,
        );
        for layer in 0..config.layers {
            let prefix = format!("model.layers.{layer}");
            for norm in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                tensors.insert(
                    format!("{prefix}.{norm}"),
                    Array::from_f32_slice(&vec![1.0; config.hidden as usize], &[config.hidden])
                        .unwrap(),
                );
            }
            for (name, rows, cols) in [
                (
                    "self_attn.q_proj.weight",
                    config.heads * config.head_dim,
                    config.hidden,
                ),
                (
                    "self_attn.k_proj.weight",
                    config.kv_heads * config.head_dim,
                    config.hidden,
                ),
                (
                    "self_attn.v_proj.weight",
                    config.kv_heads * config.head_dim,
                    config.hidden,
                ),
                (
                    "self_attn.o_proj.weight",
                    config.hidden,
                    config.heads * config.head_dim,
                ),
                ("mlp.gate_proj.weight", config.intermediate, config.hidden),
                ("mlp.up_proj.weight", config.intermediate, config.hidden),
                ("mlp.down_proj.weight", config.hidden, config.intermediate),
            ] {
                insert(&mut tensors, format!("{prefix}.{name}"), rows, cols, 0.01);
            }
        }
        tensors.insert(
            "model.norm.weight".into(),
            Array::from_f32_slice(&vec![1.0; config.hidden as usize], &[config.hidden]).unwrap(),
        );
        insert(
            &mut tensors,
            "lm_head.weight".into(),
            config.vocab,
            config.hidden,
            0.01,
        );
        MlxWeights::from_dense(tensors)
    }

    #[test]
    fn text_encoder_returns_the_penultimate_layer() {
        let config = MistralConfig::smoke();
        let weights = weights(config);
        let stream = Stream::cpu().unwrap();
        let ids = [1, 3, 4];
        let actual = encode(&weights, config, &ids, &stream).unwrap();
        let embedding = weights
            .embedding("model.embed_tokens.weight", &stream)
            .unwrap();
        let indices = Array::from_i32_slice(&ids, &[3]).unwrap();
        let hidden = embedding
            .take_axis(&indices, 0, &stream)
            .unwrap()
            .reshape(&[1, 3, config.hidden], &stream)
            .unwrap();
        let mask = causal_mask(3, 3, &stream).unwrap();
        let (cos, sin) = position_embeddings(config, 0, 3, hidden.dtype(), &stream).unwrap();
        let expected =
            layer_forward(&weights, config, 0, &hidden, &mask, &cos, &sin, &stream).unwrap();
        assert_eq!(actual.shape(), vec![1, 3, config.hidden]);
        let actual = actual.to_vec_f32(&stream).unwrap();
        let expected = expected.to_vec_f32(&stream).unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn cached_decode_matches_full_prefill() {
        let config = MistralConfig::smoke();
        let weights = weights(config);
        let stream = Stream::cpu().unwrap();
        let mut full_cache = None;
        let expected = step_logits(&weights, config, &[1, 3, 4], &mut full_cache, &stream).unwrap();
        let mut cached = None;
        step_logits(&weights, config, &[1, 3], &mut cached, &stream).unwrap();
        let actual = step_logits(&weights, config, &[4], &mut cached, &stream).unwrap();
        assert_eq!(actual.len(), config.vocab as usize);
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-4, "{actual} != {expected}");
        }
    }
}

#[cfg(test)]
mod parity_tests {
    use std::collections::BTreeMap;

    use danmo_mlx::{SafeTensors, Stream};

    use super::{encode, step_logits};
    use crate::families::ernie_image::mlx::config::MistralConfig;
    use crate::weights::MlxWeights;

    #[test]
    #[ignore = "requires the official Transformers miniature fixture"]
    fn official_transformers_mistral_fixture_matches() {
        let root =
            std::path::PathBuf::from(std::env::var("DANQING_TEST_ERNIE_MISTRAL_FIXTURE").unwrap());
        let expected: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("expected.json")).unwrap())
                .unwrap();
        let stream = Stream::cpu().unwrap();
        let tensors = SafeTensors::load(&root.join("fixture.safetensors"), &stream).unwrap();
        let mut dense = BTreeMap::new();
        for name in tensors.names().unwrap() {
            dense.insert(name.clone(), tensors.get(&name).unwrap().unwrap());
        }
        let weights = MlxWeights::from_dense(dense);
        let ids = [1, 3, 4];
        let config = MistralConfig::smoke();
        let actual = encode(&weights, config, &ids, &stream)
            .unwrap()
            .to_vec_f32(&stream)
            .unwrap();
        let values = |key: &str| {
            expected[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        compare("text", &actual, &values("output"));
        let mut cache = None;
        let logits = step_logits(&weights, config, &ids, &mut cache, &stream).unwrap();
        compare("logits", &logits, &values("logits"));
    }

    fn compare(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        let mut max_abs = 0.0f32;
        let mut error = 0.0f32;
        let mut norm = 0.0f32;
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            max_abs = max_abs.max((actual - expected).abs());
            error += (actual - expected).powi(2);
            norm += expected.powi(2);
        }
        let relative = (error / norm.max(1e-20)).sqrt();
        println!("ERNIE Mistral {label} parity max_abs={max_abs} relative_l2={relative}");
        assert!(
            relative < 3e-4,
            "{label} relative L2 {relative}, max abs {max_abs}"
        );
    }
}

#[cfg(test)]
mod real_parity_tests {
    use super::{encode, load_text_encoder};
    use crate::families::ernie_image::mlx::config::MistralConfig;
    use danmo_mlx::Stream;

    #[test]
    #[ignore = "loads the official 7.7 GB text encoder"]
    fn installed_text_encoder_matches_reference_on_device() {
        let directory =
            std::path::PathBuf::from(std::env::var("DANQING_TEST_ERNIE_TEXT_ENCODER_DIR").unwrap());
        let reference: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(std::env::var("DANQING_TEST_ERNIE_TEXT_REFERENCE").unwrap())
                .unwrap(),
        )
        .unwrap();
        let ids: Vec<i32> = reference["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap() as i32)
            .collect();
        #[cfg(feature = "mlx-metal")]
        let stream = Stream::gpu().unwrap();
        #[cfg(not(feature = "mlx-metal"))]
        let stream = Stream::cpu().unwrap();
        let config = MistralConfig::load_text_encoder(&directory).unwrap();
        let weights = load_text_encoder(&directory, &stream).unwrap();
        let actual = encode(&weights, config, &ids, &stream)
            .unwrap()
            .to_vec_f32(&stream)
            .unwrap();
        let expected: Vec<f32> = reference["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(actual.len(), expected.len());
        let mut error = 0.0;
        let mut norm = 0.0;
        let mut max_abs = 0.0f32;
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            error += (actual - expected).powi(2);
            norm += expected.powi(2);
            max_abs = max_abs.max((actual - expected).abs());
        }
        let relative = (error / norm.max(1e-20)).sqrt();
        println!("ERNIE real text parity max_abs={max_abs} relative_l2={relative}");
        assert!(relative < 0.03, "relative L2 {relative}, max abs {max_abs}");
    }
}

#[cfg(test)]
mod real_pe_parity_tests {
    use super::{load_pe, step_logits};
    use crate::families::ernie_image::mlx::config::MistralConfig;
    use danmo_mlx::Stream;

    #[test]
    #[ignore = "loads the official 7.7 GB prompt enhancer"]
    fn installed_prompt_enhancer_logits_match_reference_on_device() {
        let directory = std::path::PathBuf::from(std::env::var("DANQING_TEST_ERNIE_PE_DIR").unwrap());
        let reference: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(std::env::var("DANQING_TEST_ERNIE_PE_REFERENCE").unwrap()).unwrap()).unwrap();
        let ids: Vec<i32> = reference["ids"].as_array().unwrap().iter().map(|value| value.as_i64().unwrap() as i32).collect();
        #[cfg(feature = "mlx-metal")]
        let stream = Stream::gpu().unwrap();
        #[cfg(not(feature = "mlx-metal"))]
        let stream = Stream::cpu().unwrap();
        let config = MistralConfig::load_pe(&directory).unwrap();
        let weights = load_pe(&directory, &stream).unwrap();
        let mut cache = None;
        let actual = step_logits(&weights, config, &ids, &mut cache, &stream).unwrap();
        let expected: Vec<f32> = reference["logits"].as_array().unwrap().iter().map(|value| value.as_f64().unwrap() as f32).collect();
        let mut error = 0.0;
        let mut norm = 0.0;
        let mut max_abs = 0.0f32;
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            error += (actual - expected).powi(2);
            norm += expected.powi(2);
            max_abs = max_abs.max((actual - expected).abs());
        }
        let relative = (error / norm.max(1e-20)).sqrt();
        println!("ERNIE real PE parity max_abs={max_abs} relative_l2={relative}");
        assert!(relative < 0.03, "relative L2 {relative}, max abs {max_abs}");
    }
}
