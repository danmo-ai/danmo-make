use danmo_mlx::{Array, Error, LinearWeight, Result, Stream};

use super::Qwen3Config;
use crate::weights::{MlxWeights, WeightDtype};

pub fn load_directory(directory: &std::path::Path, dtype: WeightDtype) -> Result<MlxWeights> {
    let weights = crate::weights::load_mlx_directory(directory, dtype, None)?;
    // mflux-serialized text encoders omit the `model.` prefix that the
    // official Qwen3 checkpoint carries. Normalize to the prefixed layout so
    // callers keep a single `model.` lookup contract. Detection is by key
    // shape, never by family id.
    let has_prefix = weights.names().any(|name| name.starts_with("model."));
    if has_prefix {
        return Ok(weights);
    }
    let bare = weights
        .names()
        .any(|name| name.starts_with("embed_tokens."));
    if !bare {
        return Ok(weights);
    }
    let names: std::collections::BTreeMap<String, String> = weights
        .names()
        .map(|name| (name.to_string(), format!("model.{name}")))
        .collect();
    weights.rename(&names)
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
    let sequence = ids.len() as i32;
    let indices = Array::from_i32_slice(ids, &[sequence])?;
    encode_token_array(
        weights,
        config,
        prefix,
        &indices,
        valid,
        selected,
        true,
        // Z-Image: cast embed to BF16, build RoPE from that cast, then the
        // quantized tower may promote activations to F32.
        RopeSource::Embedding,
        stream,
    )
}

/// Same tower as [`encode_ids`], but padding positions stay in the sequence.
///
/// Flux2 Klein concatenates layers 9/18/27 over the padded chat length. The
/// causal mask still blocks pad keys. Dropping those positions would move
/// every later RoPE id.
pub fn encode_ids_full(
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
    let sequence = ids.len() as i32;
    let indices = Array::from_i32_slice(ids, &[sequence])?;
    encode_token_array(
        weights,
        config,
        prefix,
        &indices,
        valid,
        selected,
        false,
        // Flux2 `Qwen3EncoderModel.get_prompt_embeds`: cast embed to F32 first,
        // then `rotary_emb(hidden)` so cos/sin follow the activation dtype.
        RopeSource::Activation,
        stream,
    )
}

#[derive(Clone, Copy)]
enum RopeSource {
    Embedding,
    Activation,
}

fn encode_token_array(
    weights: &MlxWeights,
    config: Qwen3Config,
    prefix: &str,
    indices: &danmo_mlx::Array,
    valid: &[bool],
    selected: &[i64],
    trim_padding: bool,
    rope: RopeSource,
    stream: &Stream,
) -> Result<danmo_mlx::Array> {
    let last = selected.iter().max().copied().unwrap_or(0);
    if last < 1 || last > config.layers {
        return Err(Error::new(
            "Qwen3 MLX encode",
            "selected layer is outside the tower",
        ));
    }
    let sequence = indices.shape()[0];
    let embedding = weights.embedding(&format!("{prefix}embed_tokens.weight"), stream)?;
    // main's quantized encoder keeps dense norms at the F32 module-init dtype
    // (slice assignment preserves it), so the whole tower promotes to F32;
    // the dense fp16 encoder replaces params with BF16 and stays BF16.
    let quantized_tower = weights.quantization().is_some();
    let tower_dtype = if quantized_tower {
        danmo_mlx::Dtype::Float32
    } else {
        embedding.dtype()
    };
    let mut hidden = embedding
        .take_axis(&indices, 0, stream)?
        .reshape(&[1, sequence, config.hidden as i32], stream)?
        .astype(tower_dtype, stream)?;
    let mask = causal_padding_mask(valid, stream)?;
    let rope_dtype = match rope {
        RopeSource::Embedding => embedding.dtype(),
        RopeSource::Activation => tower_dtype,
    };
    let (cos, sin) = position_embeddings(config, sequence, rope_dtype, stream)?;
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
    if trim_padding {
        let valid_count = valid.iter().filter(|value| **value).count() as i32;
        output = output.slice(
            &[0, 0, 0],
            &[1, valid_count, output.shape()[2]],
            &[1, 1, 1],
            stream,
        )?;
    }
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
    let input_norm = norm_weight(
        weights,
        &path,
        "input_layernorm.weight",
        hidden.dtype(),
        stream,
    )?;
    let normed = hidden.rms_norm(Some(&input_norm), 1e-6, stream)?;
    let attended = attention(weights, config, &path, &normed, mask, cos, sin, stream)?;
    let hidden = Array::add(hidden, &attended, stream)?;
    let post_norm = norm_weight(
        weights,
        &path,
        "post_attention_layernorm.weight",
        hidden.dtype(),
        stream,
    )?;
    let normed = hidden.rms_norm(Some(&post_norm), 1e-6, stream)?;
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
    let q_proj = linear(weights, prefix, "self_attn.q_proj.weight")?
        .apply(hidden, None, stream)?
        .reshape(&[batch, sequence, heads, head], stream)?;
    let q_norm = norm_weight(
        weights,
        prefix,
        "self_attn.q_norm.weight",
        q_proj.dtype(),
        stream,
    )?;
    let mut q = q_proj.rms_norm(Some(&q_norm), 1e-6, stream)?;
    let k_proj = linear(weights, prefix, "self_attn.k_proj.weight")?
        .apply(hidden, None, stream)?
        .reshape(&[batch, sequence, kv_heads, head], stream)?;
    let k_norm = norm_weight(
        weights,
        prefix,
        "self_attn.k_norm.weight",
        k_proj.dtype(),
        stream,
    )?;
    let mut k = k_proj.rms_norm(Some(&k_norm), 1e-6, stream)?;
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
    position_embeddings_from(config, 0, sequence, dtype, stream)
}

fn position_embeddings_from(
    config: Qwen3Config,
    start: i32,
    length: i32,
    dtype: danmo_mlx::Dtype,
    stream: &Stream,
) -> Result<(Array, Array)> {
    // main computes inv_freq / outer / cos / sin with MLX operators in F32,
    // then casts to the activation dtype; host-side powf/cosf drifts by ~2e-3
    // and amplifies through the tower. Reproduce the exact operator sequence.
    let dim = config.head_dim as i32;
    let arange = Array::arange(0.0, dim as f64, 2.0, danmo_mlx::Dtype::Float32, stream)?;
    let dim_scalar = Array::from_f32_slice(&[dim as f32], &[])?;
    let exponent = Array::divide(&arange, &dim_scalar, stream)?;
    let base = Array::from_f32_slice(&[config.rope_theta], &[])?;
    let power = Array::power(&base, &exponent, stream)?;
    let one = Array::from_f32_slice(&[1.0], &[])?;
    let inv_freq = Array::divide(&one, &power, stream)?;
    let end = i32::saturating_add(start, length);
    let positions = Array::arange(
        start as f64,
        end as f64,
        1.0,
        danmo_mlx::Dtype::Float32,
        stream,
    )?;
    // outer(positions, inv_freq) == positions[:, None] * inv_freq[None, :]
    let positions_col = positions.expand_dims(1, stream)?;
    let inv_freq_row = inv_freq.expand_dims(0, stream)?;
    let freqs = Array::multiply(&positions_col, &inv_freq_row, stream)?;
    let emb = Array::concatenate_axis(&[freqs.clone(), freqs], -1, stream)?;
    let cos = emb
        .cos(stream)?
        .expand_dims(0, stream)?
        .expand_dims(2, stream)?;
    let sin = emb
        .sin(stream)?
        .expand_dims(0, stream)?
        .expand_dims(2, stream)?;
    Ok((cos.astype(dtype, stream)?, sin.astype(dtype, stream)?))
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

/// RMSNorm weight promoted to the activation dtype. main's quantized encoder
/// keeps dense norms at F32 (module-init dtype preserved by slice assignment),
/// which promotes the tower to F32; the dense fp16 path stays BF16. Matching
/// the activation dtype reproduces both without a family branch.
fn norm_weight(
    weights: &MlxWeights,
    prefix: &str,
    name: &str,
    dtype: danmo_mlx::Dtype,
    stream: &Stream,
) -> Result<Array> {
    let tensor = weight(weights, prefix, name)?;
    if tensor.dtype() == dtype {
        Ok(tensor.clone())
    } else {
        tensor.astype(dtype, stream)
    }
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

/// RoPE'd K/V for one Qwen3 layer, sequence axis already transposed to
/// `[batch, kv_heads, time, head]`.
pub(crate) struct LmLayerCache {
    key: Array,
    value: Array,
}

pub(crate) struct LmCache {
    layers: Vec<LmLayerCache>,
    length: i32,
}

/// One LM step. `cache` empty means `ids` is the prefill. A populated cache
/// appends `ids` (normally one token) and attends over the stored keys.
pub(crate) fn step_logits(
    weights: &MlxWeights,
    config: Qwen3Config,
    prefix: &str,
    ids: &[i32],
    cache: &mut Option<LmCache>,
    stream: &Stream,
) -> Result<Vec<f32>> {
    if ids.is_empty() {
        return Err(Error::new("Qwen3 LM", "token step is empty"));
    }
    let start = cache.as_ref().map(|item| item.length).unwrap_or(0);
    let sequence = ids.len() as i32;
    let indices = Array::from_i32_slice(ids, &[sequence])?;
    let embedding = weights.embedding(&format!("{prefix}embed_tokens.weight"), stream)?;
    let quantized_tower = weights.quantization().is_some();
    let tower_dtype = if quantized_tower {
        danmo_mlx::Dtype::Float32
    } else {
        embedding.dtype()
    };
    let mut hidden = embedding
        .take_axis(&indices, 0, stream)?
        .reshape(&[1, sequence, config.hidden as i32], stream)?
        .astype(tower_dtype, stream)?;
    let (cos, sin) = position_embeddings_from(config, start, sequence, embedding.dtype(), stream)?;
    let mut next = Vec::with_capacity(config.layers as usize);
    for layer in 0..config.layers {
        let past = cache
            .as_ref()
            .and_then(|item| item.layers.get(layer as usize));
        let (updated, key, value) = layer_step(
            weights,
            config,
            prefix,
            layer,
            &hidden,
            past.map(|item| &item.key),
            past.map(|item| &item.value),
            &cos,
            &sin,
            stream,
        )?;
        hidden = updated;
        next.push(LmLayerCache { key, value });
    }
    let norm = norm_weight(weights, prefix, "norm.weight", hidden.dtype(), stream)?;
    let last = hidden.slice(
        &[0, sequence - 1, 0],
        &[1, sequence, hidden.shape()[2]],
        &[1, 1, 1],
        stream,
    )?;
    let normed = last
        .rms_norm(Some(&norm), 1e-6, stream)?
        .reshape(&[1, config.hidden as i32], stream)?;
    let table = embedding.transpose_axes(&[1, 0], stream)?;
    let logits =
        Array::matmul(&normed, &table, stream)?.astype(danmo_mlx::Dtype::Float32, stream)?;
    let mut held = Vec::with_capacity(next.len() * 2 + 1);
    held.push(&logits);
    for layer in &next {
        held.push(&layer.key);
        held.push(&layer.value);
    }
    Array::eval_all(&held)?;
    *cache = Some(LmCache {
        layers: next,
        length: start + sequence,
    });
    logits.to_vec_f32(stream)
}

fn layer_step(
    weights: &MlxWeights,
    config: Qwen3Config,
    prefix: &str,
    layer: i64,
    hidden: &Array,
    past_key: Option<&Array>,
    past_value: Option<&Array>,
    cos: &Array,
    sin: &Array,
    stream: &Stream,
) -> Result<(Array, Array, Array)> {
    let path = format!("{prefix}layers.{layer}");
    let input_norm = norm_weight(
        weights,
        &path,
        "input_layernorm.weight",
        hidden.dtype(),
        stream,
    )?;
    let normed = hidden.rms_norm(Some(&input_norm), 1e-6, stream)?;
    let (attended, key, value) = attention_step(
        weights, config, &path, &normed, past_key, past_value, cos, sin, stream,
    )?;
    let hidden = Array::add(hidden, &attended, stream)?;
    let post_norm = norm_weight(
        weights,
        &path,
        "post_attention_layernorm.weight",
        hidden.dtype(),
        stream,
    )?;
    let normed = hidden.rms_norm(Some(&post_norm), 1e-6, stream)?;
    let gate = linear(weights, &path, "mlp.gate_proj.weight")?
        .apply(&normed, None, stream)?
        .silu(stream)?;
    let up = linear(weights, &path, "mlp.up_proj.weight")?.apply(&normed, None, stream)?;
    let mid = Array::multiply(&gate, &up, stream)?;
    let mlp = linear(weights, &path, "mlp.down_proj.weight")?.apply(&mid, None, stream)?;
    Ok((Array::add(&hidden, &mlp, stream)?, key, value))
}

fn attention_step(
    weights: &MlxWeights,
    config: Qwen3Config,
    prefix: &str,
    hidden: &Array,
    past_key: Option<&Array>,
    past_value: Option<&Array>,
    cos: &Array,
    sin: &Array,
    stream: &Stream,
) -> Result<(Array, Array, Array)> {
    let shape = hidden.shape();
    let (batch, sequence) = (shape[0], shape[1]);
    let head = config.head_dim as i32;
    let heads = config.heads as i32;
    let kv_heads = config.kv_heads as i32;
    let q_proj = linear(weights, prefix, "self_attn.q_proj.weight")?
        .apply(hidden, None, stream)?
        .reshape(&[batch, sequence, heads, head], stream)?;
    let q_norm = norm_weight(
        weights,
        prefix,
        "self_attn.q_norm.weight",
        q_proj.dtype(),
        stream,
    )?;
    let mut q = q_proj.rms_norm(Some(&q_norm), 1e-6, stream)?;
    let k_proj = linear(weights, prefix, "self_attn.k_proj.weight")?
        .apply(hidden, None, stream)?
        .reshape(&[batch, sequence, kv_heads, head], stream)?;
    let k_norm = norm_weight(
        weights,
        prefix,
        "self_attn.k_norm.weight",
        k_proj.dtype(),
        stream,
    )?;
    let mut k = k_proj.rms_norm(Some(&k_norm), 1e-6, stream)?;
    let v = linear(weights, prefix, "self_attn.v_proj.weight")?
        .apply(hidden, None, stream)?
        .reshape(&[batch, sequence, kv_heads, head], stream)?;
    q = apply_rope(&q, cos, sin, stream)?;
    k = apply_rope(&k, cos, sin, stream)?;
    q = q.transpose_axes(&[0, 2, 1, 3], stream)?;
    let mut k = k.transpose_axes(&[0, 2, 1, 3], stream)?;
    let mut v = v.transpose_axes(&[0, 2, 1, 3], stream)?;
    if let (Some(past_key), Some(past_value)) = (past_key, past_value) {
        k = Array::concatenate_axis(&[past_key.clone(), k], 2, stream)?;
        v = Array::concatenate_axis(&[past_value.clone(), v], 2, stream)?;
    }
    let groups = heads / kv_heads;
    let mut key_attn = k.clone();
    let mut value_attn = v.clone();
    if groups > 1 {
        key_attn = key_attn.repeat_axis(groups, 1, stream)?;
        value_attn = value_attn.repeat_axis(groups, 1, stream)?;
    }
    let total = k.shape()[2];
    let mask = if past_key.is_none() {
        let valid = vec![true; sequence as usize];
        causal_padding_mask(&valid, stream)?
    } else {
        Array::from_f32_slice(
            &vec![0.0; (sequence * total) as usize],
            &[1, 1, sequence, total],
        )?
    };
    let compute = q.astype(danmo_mlx::Dtype::Float32, stream)?;
    let keys = key_attn.astype(danmo_mlx::Dtype::Float32, stream)?;
    let values = value_attn.astype(danmo_mlx::Dtype::Float32, stream)?;
    let output = Array::scaled_dot_product_attention(
        &compute,
        &keys,
        &values,
        1.0 / (config.head_dim as f32).sqrt(),
        "",
        Some(&mask),
        stream,
    )?
    .astype(q.dtype(), stream)?
    .transpose_axes(&[0, 2, 1, 3], stream)?
    .reshape(&[batch, sequence, heads * head], stream)?;
    let projected =
        linear(weights, prefix, "self_attn.o_proj.weight")?.apply(&output, None, stream)?;
    Ok((projected, k, v))
}

#[cfg(test)]
mod tests {
    use super::Qwen3Config;
    use super::{
        apply_rope, causal_padding_mask, encode_ids, load_directory, norm_weight,
        position_embeddings,
    };
    use crate::weights::{MlxWeights, WeightDtype};
    use danmo_mlx::{Array, SafeTensors, Stream};
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

    /// Sub-stage layer-0 localization against a main quantized-encoder dump.
    #[test]
    #[ignore]
    fn real_quantized_qwen3_layer0_substages() {
        let encoder =
            std::path::PathBuf::from(std::env::var("DANQING_TEST_QWEN3_ENCODER_DIR").unwrap());
        let stages: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(std::env::var("DANQING_TEST_QWEN3_STAGES_JSON").unwrap())
                .unwrap(),
        )
        .unwrap();
        let stream = Stream::gpu().unwrap();
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
        let ids: Vec<i32> = vec![
            151644, 872, 198, 64, 2518, 23739, 151645, 198, 151644, 77091, 198,
        ];
        let sequence = ids.len() as i32;
        let values = |name: &str| -> Vec<f32> {
            stages[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        };
        let compare = |name: &str, actual: &Array| {
            let actual = actual.to_vec_f32(&stream).unwrap();
            let expected = values(name);
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("{name}: dtype {:?} max abs error {error}", actual.len());
        };
        let indices = danmo_mlx::Array::from_i32_slice(&ids, &[sequence]).unwrap();
        let embedding = weights
            .embedding("model.embed_tokens.weight", &stream)
            .unwrap();
        let hidden = embedding
            .take_axis(&indices, 0, &stream)
            .unwrap()
            .reshape(&[1, sequence, config.hidden as i32], &stream)
            .unwrap()
            .astype(danmo_mlx::Dtype::Float32, &stream)
            .unwrap();
        let path = "model.layers.0";
        let input_norm = norm_weight(
            &weights,
            path,
            "input_layernorm.weight",
            hidden.dtype(),
            &stream,
        )
        .unwrap();
        let normed = hidden.rms_norm(Some(&input_norm), 1e-6, &stream).unwrap();
        compare("input_norm", &normed);
        let heads = config.heads as i32;
        let kv_heads = config.kv_heads as i32;
        let head = config.head_dim as i32;
        let q_proj = weights
            .linear(&format!("{path}.self_attn.q_proj.weight"))
            .unwrap()
            .apply(&normed, None, &stream)
            .unwrap()
            .reshape(&[1, sequence, heads, head], &stream)
            .unwrap();
        compare("q_proj", &q_proj);
        let v_proj = weights
            .linear(&format!("{path}.self_attn.v_proj.weight"))
            .unwrap()
            .apply(&normed, None, &stream)
            .unwrap()
            .reshape(&[1, sequence, kv_heads, head], &stream)
            .unwrap();
        compare("v_proj", &v_proj);
        let q_norm = norm_weight(
            &weights,
            path,
            "self_attn.q_norm.weight",
            q_proj.dtype(),
            &stream,
        )
        .unwrap();
        let qn = q_proj.rms_norm(Some(&q_norm), 1e-6, &stream).unwrap();
        compare("q_norm", &qn);
        let (cos, sin) = position_embeddings(config, sequence, embedding.dtype(), &stream).unwrap();
        compare("cos", &cos);
        compare("sin", &sin);
        let qe = apply_rope(&qn, &cos, &sin, &stream).unwrap();
        compare("q_rope", &qe);
        let k_proj = weights
            .linear(&format!("{path}.self_attn.k_proj.weight"))
            .unwrap()
            .apply(&normed, None, &stream)
            .unwrap()
            .reshape(&[1, sequence, kv_heads, head], &stream)
            .unwrap();
        let k_norm = norm_weight(
            &weights,
            path,
            "self_attn.k_norm.weight",
            k_proj.dtype(),
            &stream,
        )
        .unwrap();
        let kn = k_proj.rms_norm(Some(&k_norm), 1e-6, &stream).unwrap();
        let ke = apply_rope(&kn, &cos, &sin, &stream).unwrap();
        compare("k_rope", &ke);
        let groups = heads / kv_heads;
        let kg = ke.repeat_axis(groups, 2, &stream).unwrap();
        let vg = v_proj.repeat_axis(groups, 2, &stream).unwrap();
        let qt = qe.transpose_axes(&[0, 2, 1, 3], &stream).unwrap();
        let kt = kg.transpose_axes(&[0, 2, 1, 3], &stream).unwrap();
        let vt = vg.transpose_axes(&[0, 2, 1, 3], &stream).unwrap();
        let mask = causal_padding_mask(&vec![true; ids.len()], &stream).unwrap();
        let compute = qt.astype(danmo_mlx::Dtype::Float32, &stream).unwrap();
        let keys = kt.astype(danmo_mlx::Dtype::Float32, &stream).unwrap();
        let vals = vt.astype(danmo_mlx::Dtype::Float32, &stream).unwrap();
        let sdpa = Array::scaled_dot_product_attention(
            &compute,
            &keys,
            &vals,
            1.0 / (config.head_dim as f32).sqrt(),
            "",
            Some(&mask),
            &stream,
        )
        .unwrap()
        .astype(qe.dtype(), &stream)
        .unwrap();
        compare("sdpa", &sdpa);
        let sdpa_t = sdpa
            .transpose_axes(&[0, 2, 1, 3], &stream)
            .unwrap()
            .reshape(&[1, sequence, heads * head], &stream)
            .unwrap();
        let o = weights
            .linear(&format!("{path}.self_attn.o_proj.weight"))
            .unwrap()
            .apply(&sdpa_t, None, &stream)
            .unwrap();
        compare("o_proj", &o);
    }

    /// Stage-by-stage localization against a main quantized-encoder dump.
    #[test]
    #[ignore]
    fn real_quantized_qwen3_stages_localize_divergence() {
        let encoder =
            std::path::PathBuf::from(std::env::var("DANQING_TEST_QWEN3_ENCODER_DIR").unwrap());
        let stages: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(std::env::var("DANQING_TEST_QWEN3_STAGES_JSON").unwrap())
                .unwrap(),
        )
        .unwrap();
        let stream = Stream::gpu().unwrap();
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
        let ids: Vec<i32> = vec![
            151644, 872, 198, 64, 2518, 23739, 151645, 198, 151644, 77091, 198,
        ];
        let valid = vec![true; ids.len()];
        let values = |name: &str| -> Vec<f32> {
            stages[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        };
        // Embedding probe: replicate encode_ids' first step.
        let indices = danmo_mlx::Array::from_i32_slice(&ids, &[ids.len() as i32]).unwrap();
        let embedding = weights
            .embedding("model.embed_tokens.weight", &stream)
            .unwrap();
        let hidden = embedding
            .take_axis(&indices, 0, &stream)
            .unwrap()
            .reshape(&[1, ids.len() as i32, config.hidden as i32], &stream)
            .unwrap()
            .astype(embedding.dtype(), &stream)
            .unwrap();
        let embed_actual = hidden.to_vec_f32(&stream).unwrap();
        let embed_expected = values("embed");
        let embed_error = embed_actual
            .iter()
            .zip(&embed_expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "embed: dtype {:?}, max abs error {embed_error}",
            embedding.dtype()
        );
        for (stage, layer) in [
            ("layer0", 1_i64),
            ("layer1", 2),
            ("layer5", 6),
            ("layer17", 18),
            ("layer34", 35),
        ] {
            let actual = encode_ids(&weights, config, "model.", &ids, &valid, &[layer], &stream)
                .unwrap()
                .to_vec_f32(&stream)
                .unwrap();
            let expected = values(stage);
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("{stage}: max abs error {error}");
        }
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
