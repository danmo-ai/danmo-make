//! LTX 2.5 Gemma 4 prompt encoder, feature extractor, and A/V connectors.
//!
//! The family owns the published Gemma 4 configuration and connector policy.
//! Generic tensor operations live in `common::mlx_ops`; packed affine matmuls
//! remain inside `LinearWeight` and are never expanded for DiT/LLM linears.

use std::path::Path;

use danmo_mlx::{AffineQuantization, Array, Dtype, Error, LinearWeight, Result, Stream};
use serde_json::Value;

use super::tokenizer::GemmaTokenizer;
use crate::common::mlx_ops::{apply_split_rope, gelu_tanh, rope_table_1d, scalar};
use crate::weights::{load_mlx_file, MlxWeights, WeightDtype, WeightError};

const PREFIX: &str = "language_model.model";

#[derive(Debug, Clone)]
struct GemmaConfig {
    hidden: usize,
    layers: usize,
    intermediate: usize,
    heads: usize,
    kv_heads: usize,
    global_kv_heads: usize,
    head_dim: usize,
    global_head_dim: usize,
    sliding_window: usize,
    layer_types: Vec<String>,
    attention_k_eq_v: bool,
    full_theta: f32,
    full_rotary_factor: f32,
    sliding_theta: f32,
    max_length: usize,
}

impl GemmaConfig {
    fn load(root: &Path, bundle_gemma: &Value) -> Result<Self> {
        let path = root.join("text_encoder/config.json");
        let text = std::fs::read_to_string(&path).map_err(|error| {
            Error::new(
                "LTX 2.5 Gemma config",
                format!("{}: {error}", path.display()),
            )
        })?;
        let value: Value = serde_json::from_str(&text)
            .map_err(|error| Error::new("LTX 2.5 Gemma config", error.to_string()))?;
        let cfg = value.get("text_config").unwrap_or(&value);
        for flag in ["enable_moe_block", "use_double_wide_mlp"] {
            if cfg.get(flag).and_then(Value::as_bool).unwrap_or(false) {
                return Err(Error::new(
                    "LTX 2.5 Gemma config",
                    format!("{flag}=true is not the published 12B text encoder"),
                ));
            }
        }
        for field in ["num_kv_shared_layers", "hidden_size_per_layer_input"] {
            if cfg.get(field).and_then(Value::as_u64).unwrap_or(0) != 0 {
                return Err(Error::new(
                    "LTX 2.5 Gemma config",
                    format!("{field} must be zero for the published LTX 12B encoder"),
                ));
            }
        }
        let layer_types = strings(cfg, "layer_types")?;
        let layers = integer(cfg, "num_hidden_layers")?;
        if layer_types.len() != layers {
            return Err(Error::new(
                "LTX 2.5 Gemma config",
                "layer_types does not cover num_hidden_layers",
            ));
        }
        let rope = cfg
            .get("rope_parameters")
            .and_then(Value::as_object)
            .ok_or_else(|| Error::new("LTX 2.5 Gemma config", "missing rope_parameters"))?;
        let full = rope
            .get("full_attention")
            .ok_or_else(|| Error::new("LTX 2.5 Gemma config", "missing full RoPE"))?;
        let sliding = rope
            .get("sliding_attention")
            .ok_or_else(|| Error::new("LTX 2.5 Gemma config", "missing sliding RoPE"))?;
        if full.get("rope_type").and_then(Value::as_str) != Some("proportional") {
            return Err(Error::new(
                "LTX 2.5 Gemma config",
                "full attention must use proportional RoPE",
            ));
        }
        let max_length = bundle_gemma
            .get("tokenizer_max_length")
            .and_then(Value::as_u64)
            .unwrap_or(1024) as usize;
        Ok(Self {
            hidden: integer(cfg, "hidden_size")?,
            layers,
            intermediate: integer(cfg, "intermediate_size")?,
            heads: integer(cfg, "num_attention_heads")?,
            kv_heads: integer(cfg, "num_key_value_heads")?,
            global_kv_heads: integer(cfg, "num_global_key_value_heads")?,
            head_dim: integer(cfg, "head_dim")?,
            global_head_dim: integer(cfg, "global_head_dim")?,
            sliding_window: integer(cfg, "sliding_window")?,
            layer_types,
            attention_k_eq_v: cfg
                .get("attention_k_eq_v")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            full_theta: float(full, "rope_theta")?,
            full_rotary_factor: float(full, "partial_rotary_factor")?,
            sliding_theta: float(sliding, "rope_theta")?,
            max_length,
        })
    }

    fn is_sliding(&self, layer: usize) -> bool {
        self.layer_types[layer] == "sliding_attention"
    }

    fn head_dim(&self, layer: usize) -> usize {
        if self.is_sliding(layer) {
            self.head_dim
        } else {
            self.global_head_dim
        }
    }
}

pub struct PromptEmbeddings {
    pub video: Array,
    pub audio: Array,
}

pub fn encode_prompt(
    bundle: &Path,
    bundle_config: &super::config::BundleConfig,
    prompt: &str,
    stream: &Stream,
) -> std::result::Result<PromptEmbeddings, WeightError> {
    encode_prompt_inner(bundle, bundle_config, prompt, stream)
        .map_err(|error| WeightError::NotBuilt(error.to_string()))
}

fn encode_prompt_inner(
    bundle: &Path,
    bundle_config: &super::config::BundleConfig,
    prompt: &str,
    stream: &Stream,
) -> Result<PromptEmbeddings> {
    let config = GemmaConfig::load(bundle, &bundle_config.gemma)?;
    let tokenizer = GemmaTokenizer::load(&bundle.join("text_encoder"))
        .map_err(|error| Error::new("LTX 2.5 Gemma tokenizer", error.to_string()))?;
    let tokens = tokenizer
        .encode(prompt, config.max_length)
        .map_err(|error| Error::new("LTX 2.5 Gemma tokenizer", error.to_string()))?;
    let quantization = gemma_quantization(bundle)?;
    let weights = load_mlx_file(
        &bundle.join("text_encoder/model.safetensors"),
        WeightDtype::Native,
        quantization,
    )?
    .materialize_embedding(&format!("{PREFIX}.embed_tokens.weight"), stream)?;
    let states = gemma_hidden_states(
        &weights,
        &config,
        &tokens.ids,
        &tokens.attention_mask,
        stream,
    )?;
    Array::eval_all(&states.iter().collect::<Vec<_>>())?;
    drop(weights);
    let connector = load_mlx_file(
        &bundle.join("connector.safetensors"),
        WeightDtype::Native,
        None,
    )?;
    let (video, audio) =
        feature_extractor(&connector, &config, &states, &tokens.attention_mask, stream)?;
    let sorted = right_sort(&video, &tokens.attention_mask, stream)?;
    let sorted_audio = right_sort(&audio, &tokens.attention_mask, stream)?;
    let video = connector_forward(
        &connector,
        &bundle_config.transformer,
        &sorted,
        &tokens.attention_mask,
        false,
        stream,
    )?;
    let audio = connector_forward(
        &connector,
        &bundle_config.transformer,
        &sorted_audio,
        &tokens.attention_mask,
        true,
        stream,
    )?;
    Array::eval_all(&[&video, &audio])?;
    Ok(PromptEmbeddings { video, audio })
}

fn gemma_quantization(bundle: &Path) -> Result<Option<AffineQuantization>> {
    let path = bundle.join("text_encoder/config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| Error::new("LTX 2.5 Gemma quantization", error.to_string()))?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|error| Error::new("LTX 2.5 Gemma quantization", error.to_string()))?;
    let Some(quant) = value.get("quantization") else {
        return Ok(None);
    };
    Ok(Some(AffineQuantization::new(
        integer(quant, "bits")? as i32,
        integer(quant, "group_size")? as i32,
    )?))
}

fn gemma_hidden_states(
    weights: &MlxWeights,
    config: &GemmaConfig,
    ids: &[i32],
    attention_mask: &[i32],
    stream: &Stream,
) -> Result<Vec<Array>> {
    let length = ids.len();
    let ids_array = Array::from_i32_slice(ids, &[1, length as i32])?;
    let embedding = weights.tensor(&format!("{PREFIX}.embed_tokens.weight"))?;
    let mut hidden = embedding.take_axis(&ids_array, 0, stream)?;
    hidden = Array::multiply(&hidden, &scalar((config.hidden as f32).sqrt())?, stream)?;
    let mut states = vec![hidden.clone()];
    let full_mask = causal_mask(attention_mask, None, stream)?;
    let sliding_mask = causal_mask(attention_mask, Some(config.sliding_window), stream)?;
    let full_rope = rope_table_1d(
        length,
        config.global_head_dim,
        (config.global_head_dim as f32 * config.full_rotary_factor) as usize,
        config.full_theta,
        stream,
    )?;
    let sliding_rope = rope_table_1d(
        length,
        config.head_dim,
        config.head_dim,
        config.sliding_theta,
        stream,
    )?;
    for layer in 0..config.layers {
        let sliding = config.is_sliding(layer);
        hidden = gemma_layer(
            weights,
            config,
            layer,
            &hidden,
            if sliding { &sliding_mask } else { &full_mask },
            if sliding { &sliding_rope } else { &full_rope },
            stream,
        )?;
        if (layer + 1) % 2 == 0 {
            hidden.eval()?;
        }
        states.push(hidden.clone());
    }
    let last = states
        .last_mut()
        .expect("one state per layer plus embedding");
    *last = last.rms_norm(
        Some(weights.tensor(&format!("{PREFIX}.norm.weight"))?),
        1e-6,
        stream,
    )?;
    Ok(states)
}

fn gemma_layer(
    weights: &MlxWeights,
    config: &GemmaConfig,
    layer: usize,
    input: &Array,
    mask: &Array,
    rope: &(Array, Array),
    stream: &Stream,
) -> Result<Array> {
    let prefix = format!("{PREFIX}.layers.{layer}");
    let normed = input.rms_norm(
        Some(weights.tensor(&format!("{prefix}.input_layernorm.weight"))?),
        1e-6,
        stream,
    )?;
    let attention = gemma_attention(weights, config, layer, &normed, mask, rope, stream)?;
    let attention = attention.rms_norm(
        Some(weights.tensor(&format!("{prefix}.post_attention_layernorm.weight"))?),
        1e-6,
        stream,
    )?;
    let hidden = Array::add(input, &attention, stream)?;
    let normed = hidden.rms_norm(
        Some(weights.tensor(&format!("{prefix}.pre_feedforward_layernorm.weight"))?),
        1e-6,
        stream,
    )?;
    let gate = linear(weights, &format!("{prefix}.mlp.gate_proj"))?.apply(&normed, None, stream)?;
    let up = linear(weights, &format!("{prefix}.mlp.up_proj"))?.apply(&normed, None, stream)?;
    let activated = Array::multiply(&gelu_tanh(&gate, stream)?, &up, stream)?;
    let mlp =
        linear(weights, &format!("{prefix}.mlp.down_proj"))?.apply(&activated, None, stream)?;
    let mlp = mlp.rms_norm(
        Some(weights.tensor(&format!("{prefix}.post_feedforward_layernorm.weight"))?),
        1e-6,
        stream,
    )?;
    let hidden = Array::add(&hidden, &mlp, stream)?;
    Array::multiply(
        &hidden,
        weights.tensor(&format!("{prefix}.layer_scalar"))?,
        stream,
    )
}

fn gemma_attention(
    weights: &MlxWeights,
    config: &GemmaConfig,
    layer: usize,
    input: &Array,
    mask: &Array,
    rope: &(Array, Array),
    stream: &Stream,
) -> Result<Array> {
    let prefix = format!("{PREFIX}.layers.{layer}.self_attn");
    let shape = input.shape();
    let (batch, sequence) = (shape[0], shape[1]);
    let sliding = config.is_sliding(layer);
    let head_dim = config.head_dim(layer) as i32;
    let kv_heads = if sliding && config.attention_k_eq_v {
        config.kv_heads
    } else if config.attention_k_eq_v {
        config.global_kv_heads
    } else {
        config.kv_heads
    } as i32;
    let heads = config.heads as i32;
    let q = linear(weights, &format!("{prefix}.q_proj"))?
        .apply(input, None, stream)?
        .reshape(&[batch, sequence, heads, head_dim], stream)?
        .rms_norm(
            Some(weights.tensor(&format!("{prefix}.q_norm.weight"))?),
            1e-6,
            stream,
        )?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
    let k = linear(weights, &format!("{prefix}.k_proj"))?
        .apply(input, None, stream)?
        .reshape(&[batch, sequence, kv_heads, head_dim], stream)?;
    let values = if config.attention_k_eq_v && !sliding {
        k.clone()
    } else {
        linear(weights, &format!("{prefix}.v_proj"))?
            .apply(input, None, stream)?
            .reshape(&[batch, sequence, kv_heads, head_dim], stream)?
    };
    let k = k
        .rms_norm(
            Some(weights.tensor(&format!("{prefix}.k_norm.weight"))?),
            1e-6,
            stream,
        )?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
    let v = values
        .rms_norm(None, 1e-6, stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
    let q = apply_split_rope(&q, &rope.0, &rope.1, stream)?;
    let k = apply_split_rope(&k, &rope.0, &rope.1, stream)?;
    let attended = Array::scaled_dot_product_attention(&q, &k, &v, 1.0, "", Some(mask), stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, sequence, heads * head_dim], stream)?;
    linear(weights, &format!("{prefix}.o_proj"))?.apply(&attended, None, stream)
}

fn causal_mask(attention: &[i32], window: Option<usize>, stream: &Stream) -> Result<Array> {
    let length = attention.len();
    let mut left_padding = 0usize;
    let mut values = Vec::with_capacity(length * length);
    for query in 0..length {
        if attention[query] == 0 {
            left_padding += 1;
        }
        for key in 0..length {
            let causal = key <= query && window.map_or(true, |size| query < key + size);
            let valid_key = key >= left_padding;
            values.push(i32::from(causal && valid_key));
        }
    }
    Array::from_i32_slice(&values, &[1, 1, length as i32, length as i32])?
        .astype(Dtype::Bool, stream)
}

fn feature_extractor(
    weights: &MlxWeights,
    config: &GemmaConfig,
    states: &[Array],
    attention_mask: &[i32],
    stream: &Stream,
) -> Result<(Array, Array)> {
    let encoded = Array::stack(states, -1, stream)?;
    let variance = encoded.square(stream)?.mean_axes(&[2], true, stream)?;
    let normed = Array::divide(
        &encoded,
        &Array::add(&variance, &scalar(1e-6)?, stream)?.sqrt(stream)?,
        stream,
    )?
    .reshape(
        &[
            1,
            attention_mask.len() as i32,
            (config.hidden * states.len()) as i32,
        ],
        stream,
    )?;
    let mask = Array::from_i32_slice(attention_mask, &[1, attention_mask.len() as i32, 1])?
        .astype(Dtype::Float32, stream)?;
    let normed = Array::multiply(&normed, &mask, stream)?;
    let video_dim = weights
        .tensor("feature_extractor.video_aggregate_embed.bias")?
        .shape()[0];
    let audio_dim = weights
        .tensor("feature_extractor.audio_aggregate_embed.bias")?
        .shape()[0];
    let video = linear(weights, "feature_extractor.video_aggregate_embed")?.apply(
        &Array::multiply(
            &normed,
            &scalar((video_dim as f32 / config.hidden as f32).sqrt())?,
            stream,
        )?,
        Some(weights.tensor("feature_extractor.video_aggregate_embed.bias")?),
        stream,
    )?;
    let audio = linear(weights, "feature_extractor.audio_aggregate_embed")?.apply(
        &Array::multiply(
            &normed,
            &scalar((audio_dim as f32 / config.hidden as f32).sqrt())?,
            stream,
        )?,
        Some(weights.tensor("feature_extractor.audio_aggregate_embed.bias")?),
        stream,
    )?;
    Ok((video, audio))
}

fn right_sort(input: &Array, attention_mask: &[i32], stream: &Stream) -> Result<Array> {
    let valid = attention_mask.iter().filter(|value| **value != 0).count();
    let length = attention_mask.len();
    let shape = input.shape();
    let valid_part = input.slice(
        &[0, (length - valid) as i32, 0],
        &[shape[0], length as i32, shape[2]],
        &[1, 1, 1],
        stream,
    )?;
    if valid == length {
        return Ok(valid_part);
    }
    let padding = input.slice(
        &[0, 0, 0],
        &[shape[0], (length - valid) as i32, shape[2]],
        &[1, 1, 1],
        stream,
    )?;
    Array::concatenate_axis(&[valid_part, padding], 1, stream)
}

fn connector_forward(
    weights: &MlxWeights,
    transformer: &Value,
    input: &Array,
    original_mask: &[i32],
    audio: bool,
    stream: &Stream,
) -> Result<Array> {
    let prefix = if audio {
        "audio_connector"
    } else {
        "video_connector"
    };
    let valid = original_mask.iter().filter(|value| **value != 0).count();
    let length = original_mask.len();
    let shape = input.shape();
    let dim = shape[2] as usize;
    let head_dim = if audio { 64 } else { 128 };
    let heads = if audio {
        transformer
            .get("audio_connector_num_attention_heads")
            .or_else(|| transformer.get("connector_num_attention_heads"))
    } else {
        transformer.get("connector_num_attention_heads")
    }
    .and_then(Value::as_u64)
    .ok_or_else(|| {
        Error::new(
            "LTX 2.5 connector config",
            "missing connector_num_attention_heads",
        )
    })? as usize;
    if heads * head_dim != dim {
        return Err(Error::new(
            "LTX 2.5 connector config",
            format!("{heads} heads x {head_dim} does not match connector width {dim}"),
        ));
    }
    let layers = if audio {
        transformer
            .get("audio_connector_num_layers")
            .or_else(|| transformer.get("connector_num_layers"))
    } else {
        transformer.get("connector_num_layers")
    }
    .and_then(Value::as_u64)
    .ok_or_else(|| Error::new("LTX 2.5 connector config", "missing connector_num_layers"))?
        as usize;
    let max_pos = transformer
        .get("positional_embedding_max_pos")
        .and_then(Value::as_f64)
        .unwrap_or(4096.0) as f32;
    let registers = weights.tensor(&format!("{prefix}.learnable_registers"))?;
    let tiled = tile_registers(registers, length, stream)?;
    let mut mask_values = vec![0.0f32; length];
    mask_values[..valid].fill(1.0);
    let mask = Array::from_f32_slice(&mask_values, &[1, length as i32, 1])?;
    let inverse = Array::subtract(&scalar(1.0)?, &mask, stream)?;
    let mut hidden = Array::add(
        &Array::multiply(input, &mask, stream)?,
        &Array::multiply(&tiled, &inverse, stream)?,
        stream,
    )?;
    let (cos, sin) = connector_rope(length, dim, head_dim, max_pos, stream)?;
    for layer in 0..layers {
        let block = format!("{prefix}.transformer_1d_blocks.{layer}");
        let normed = hidden.rms_norm(None, 1e-6, stream)?;
        let attention = connector_attention(
            weights, &block, &normed, heads, head_dim, &cos, &sin, stream,
        )?;
        hidden = Array::add(&hidden, &attention, stream)?;
        let normed = hidden.rms_norm(None, 1e-6, stream)?;
        let ff = connector_linear(
            weights,
            &format!("{block}.ff.proj_in"),
            &format!("{block}.ff.net.0.proj"),
        )?
        .apply(
            &normed,
            Some(weights.tensor(&format!("{block}.ff.net.0.proj.bias"))?),
            stream,
        )?;
        let ff = gelu_tanh(&ff, stream)?;
        let ff = connector_linear(
            weights,
            &format!("{block}.ff.proj_out"),
            &format!("{block}.ff.net.2"),
        )?
        .apply(
            &ff,
            Some(weights.tensor(&format!("{block}.ff.net.2.bias"))?),
            stream,
        )?;
        hidden = Array::add(&hidden, &ff, stream)?;
        hidden.eval()?;
    }
    hidden.rms_norm(None, 1e-6, stream)
}

#[allow(clippy::too_many_arguments)]
fn connector_attention(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    heads: usize,
    head_dim: usize,
    cos: &Array,
    sin: &Array,
    stream: &Stream,
) -> Result<Array> {
    let attn = format!("{prefix}.attn1");
    let shape = input.shape();
    let (batch, sequence) = (shape[0], shape[1]);
    let dim = heads * head_dim;
    let project = |name: &str| -> Result<Array> {
        linear(weights, &format!("{attn}.{name}"))?.apply(
            input,
            Some(weights.tensor(&format!("{attn}.{name}.bias"))?),
            stream,
        )
    };
    let q = project("to_q")?
        .rms_norm(
            Some(weights.tensor(&format!("{attn}.q_norm.weight"))?),
            1e-6,
            stream,
        )?
        .reshape(&[batch, sequence, heads as i32, head_dim as i32], stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
    let k = project("to_k")?
        .rms_norm(
            Some(weights.tensor(&format!("{attn}.k_norm.weight"))?),
            1e-6,
            stream,
        )?
        .reshape(&[batch, sequence, heads as i32, head_dim as i32], stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
    let v = project("to_v")?
        .reshape(&[batch, sequence, heads as i32, head_dim as i32], stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
    let q = apply_split_rope(&q, cos, sin, stream)?;
    let k = apply_split_rope(&k, cos, sin, stream)?;
    let mut output = Array::scaled_dot_product_attention(
        &q,
        &k,
        &v,
        1.0 / (head_dim as f32).sqrt(),
        "",
        None,
        stream,
    )?;
    let gate = linear(weights, &format!("{attn}.to_gate_logits"))?.apply(
        input,
        Some(weights.tensor(&format!("{attn}.to_gate_logits.bias"))?),
        stream,
    )?;
    let gate = Array::multiply(&gate.sigmoid(stream)?, &scalar(2.0)?, stream)?
        .transpose_axes(&[0, 2, 1], stream)?
        .expand_dims(-1, stream)?;
    output = Array::multiply(&output, &gate, stream)?;
    let output = output
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, sequence, dim as i32], stream)?;
    connector_linear(
        weights,
        &format!("{attn}.to_out"),
        &format!("{attn}.to_out.0"),
    )?
    .apply(
        &output,
        Some(weights.tensor(&format!("{attn}.to_out.0.bias"))?),
        stream,
    )
}

fn connector_rope(
    sequence: usize,
    dim: usize,
    head_dim: usize,
    max_pos: f32,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let heads = dim / head_dim;
    let half = head_dim / 2;
    let frequencies = dim / 2;
    let mut cos = Vec::with_capacity(heads * sequence * half);
    let mut sin = Vec::with_capacity(cos.capacity());
    for head in 0..heads {
        for position in 0..sequence {
            for index in 0..half {
                let frequency_index = head * half + index;
                let exponent = frequency_index as f32 / (frequencies - 1) as f32;
                let angle = 10000.0f32.powf(exponent) * (2.0 * position as f32 / max_pos - 1.0);
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
    }
    let shape = [1, heads as i32, sequence as i32, half as i32];
    Ok((
        Array::from_f32_slice(&cos, &shape)?,
        Array::from_f32_slice(&sin, &shape)?,
    ))
}

fn tile_registers(registers: &Array, sequence: usize, stream: &Stream) -> Result<Array> {
    let count = registers.shape()[0];
    let mut tiles = Vec::new();
    let mut covered = 0;
    while covered < sequence {
        tiles.push(registers.clone());
        covered += count as usize;
    }
    let tiled = Array::concatenate_axis(&tiles, 0, stream)?.expand_dims(0, stream)?;
    tiled.slice(
        &[0, 0, 0],
        &[1, sequence as i32, registers.shape()[1]],
        &[1, 1, 1],
        stream,
    )
}

fn linear<'a>(weights: &'a MlxWeights, name: &str) -> Result<&'a LinearWeight> {
    weights.linear(&format!("{name}.weight"))
}

fn connector_linear<'a>(
    weights: &'a MlxWeights,
    normalized: &str,
    original: &str,
) -> Result<&'a LinearWeight> {
    let normalized = format!("{normalized}.weight");
    let original = format!("{original}.weight");
    if weights.has(&normalized) {
        weights.linear(&normalized)
    } else {
        weights.linear(&original)
    }
}

fn integer(value: &Value, key: &str) -> Result<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .ok_or_else(|| Error::new("LTX 2.5 config", format!("missing integer {key}")))
}

fn float(value: &Value, key: &str) -> Result<f32> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .ok_or_else(|| Error::new("LTX 2.5 config", format!("missing number {key}")))
}

fn strings(value: &Value, key: &str) -> Result<Vec<String>> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new("LTX 2.5 config", format!("missing string list {key}")))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| Error::new("LTX 2.5 config", format!("{key} contains a non-string")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore]
    fn real_bundle_encodes_prompt() {
        let root = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_LTX25_BUNDLE").expect("DANQING_TEST_LTX25_BUNDLE"),
        );
        let config = crate::families::ltx25::config::BundleConfig::load(&root).unwrap();
        let stream = danmo_mlx::Stream::gpu().unwrap();
        let output = super::encode_prompt(
            &root,
            &config,
            "A cinematic shot of a red fox walking through a forest.",
            &stream,
        )
        .unwrap();
        assert_eq!(output.video.shape(), &[1, 1024, 4096]);
        assert_eq!(output.audio.shape(), &[1, 1024, 2048]);
        let probe = output
            .video
            .slice(&[0, 0, 0], &[1, 8, 16], &[1, 1, 1], &stream)
            .unwrap()
            .to_vec_f32(&stream)
            .unwrap();
        assert!(probe.iter().all(|value| value.is_finite()));
        println!("LTX25 prompt probe: {:?}", &probe[..8]);
    }
}
