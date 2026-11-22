//! ACE-Step XL decoder and condition encoder on MLX.
//!
//! The decoder follows `acestep/models/mlx/dit_model.py`. The condition
//! encoder follows `AceStepConditionEncoder` in
//! `modeling_acestep_v15_xl_turbo.py`: lyric transformer, timbre CLS,
//! then text. Conv kernels stay in the checkpoint layout until the call,
//! where PyTorch `[out, in, K]` becomes MLX `[out, K, in]`.

use danmo_mlx::{Array, Dtype, Result, Stream};

use crate::weights::MlxWeights;

#[derive(Clone, Debug)]
pub struct AceConfig {
    pub hidden: i32,
    pub intermediate: i32,
    pub layers: i32,
    pub heads: i32,
    pub kv_heads: i32,
    pub head_dim: i32,
    pub in_channels: i32,
    pub out_channels: i32,
    pub patch: i32,
    pub sliding_window: i32,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub encoder_hidden: i32,
    pub encoder_intermediate: i32,
    pub encoder_heads: i32,
    pub encoder_kv_heads: i32,
    pub text_hidden: i32,
    pub lyric_layers: i32,
    pub timbre_layers: i32,
    pub is_turbo: bool,
    /// FSQ detokenizer repeats each 5 Hz code across this many 25 Hz frames.
    pub pool_window: i32,
    /// Detokenizer transformer depth (`num_attention_pooler_hidden_layers`).
    pub pool_layers: i32,
    /// ResidualFSQ depth. Cover encode supports one quantizer.
    pub fsq_quantizers: i32,
    /// Per-dimension FSQ levels. Must match `fsq_quant::FSQ_LEVELS`.
    pub fsq_levels: Vec<i32>,
    /// `true` when `layer_types[i]` is sliding attention.
    pub sliding: Vec<bool>,
    /// Sorted `(layer, heads)` from `lyric_alignment_layers_config`. Empty when the key is absent.
    pub align_layers: Vec<(i32, Vec<i32>)>,
}

impl AceConfig {
    pub fn from_json(text: &str) -> std::result::Result<Self, String> {
        let value: serde_json::Value = serde_json::from_str(text)
            .map_err(|error| format!("ACE-Step config.json is not json: {error}"))?;
        let hidden = json_i32(&value, "hidden_size")?;
        let heads = json_i32(&value, "num_attention_heads")?;
        let head_dim = value
            .get("head_dim")
            .and_then(|item| item.as_i64())
            .map(|item| item as i32)
            .unwrap_or(hidden / heads);
        let layers = json_i32(&value, "num_hidden_layers")?;
        let sliding = value
            .get("layer_types")
            .and_then(|item| item.as_array())
            .map(|items| {
                items
                    .iter()
                    .map(|item| item.as_str() == Some("sliding_attention"))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| (0..layers).map(|index| index % 2 == 0).collect());
        if sliding.len() < layers as usize {
            return Err(format!(
                "ACE-Step layer_types has {} entries for {layers} layers",
                sliding.len()
            ));
        }
        let is_turbo = value
            .get("is_turbo")
            .and_then(|item| item.as_bool())
            .unwrap_or(false);
        Ok(Self {
            hidden,
            intermediate: json_i32(&value, "intermediate_size")?,
            layers,
            heads,
            kv_heads: json_i32(&value, "num_key_value_heads")?,
            head_dim,
            in_channels: json_i32(&value, "in_channels")?,
            out_channels: json_i32(&value, "audio_acoustic_hidden_dim")?,
            patch: json_i32(&value, "patch_size")?,
            sliding_window: value
                .get("sliding_window")
                .and_then(|item| item.as_i64())
                .map(|item| item as i32)
                .filter(|item| *item > 0)
                .unwrap_or(128),
            rope_theta: value
                .get("rope_theta")
                .and_then(|item| item.as_f64())
                .unwrap_or(1_000_000.0) as f32,
            rms_eps: value
                .get("rms_norm_eps")
                .and_then(|item| item.as_f64())
                .unwrap_or(1e-6) as f32,
            encoder_hidden: json_i32(&value, "encoder_hidden_size").unwrap_or(hidden),
            encoder_intermediate: json_i32(&value, "encoder_intermediate_size")
                .unwrap_or_else(|_| json_i32(&value, "intermediate_size").unwrap_or(hidden * 4)),
            encoder_heads: json_i32(&value, "encoder_num_attention_heads").unwrap_or(heads),
            encoder_kv_heads: json_i32(&value, "encoder_num_key_value_heads")
                .unwrap_or_else(|_| json_i32(&value, "num_key_value_heads").unwrap_or(heads)),
            text_hidden: json_i32(&value, "text_hidden_dim")?,
            lyric_layers: json_i32(&value, "num_lyric_encoder_hidden_layers")?,
            timbre_layers: json_i32(&value, "num_timbre_encoder_hidden_layers")?,
            pool_window: json_i32(&value, "pool_window_size")?,
            pool_layers: json_i32(&value, "num_attention_pooler_hidden_layers")?,
            fsq_quantizers: json_i32(&value, "fsq_input_num_quantizers")?,
            fsq_levels: json_i32_array(&value, "fsq_input_levels")?,
            is_turbo,
            sliding,
            align_layers: alignment_layers(&value)?,
        })
    }
}

fn alignment_layers(
    value: &serde_json::Value,
) -> std::result::Result<Vec<(i32, Vec<i32>)>, String> {
    let Some(raw) = value.get("lyric_alignment_layers_config") else {
        return Ok(Vec::new());
    };
    let object = raw
        .as_object()
        .ok_or("ACE-Step lyric_alignment_layers_config must be an object")?;
    let mut layers = Vec::with_capacity(object.len());
    for (key, heads) in object {
        let layer = key
            .parse::<i32>()
            .map_err(|_| format!("ACE-Step lyric alignment layer '{key}' is not an integer"))?;
        let heads = heads.as_array().ok_or_else(|| {
            format!("ACE-Step lyric alignment layer {layer} heads must be an array")
        })?;
        if heads.is_empty() {
            return Err(format!(
                "ACE-Step lyric alignment layer {layer} selects no heads"
            ));
        }
        let mut selected = Vec::with_capacity(heads.len());
        for head in heads {
            let head = head.as_i64().ok_or_else(|| {
                format!("ACE-Step lyric alignment layer {layer} has a non-integer head")
            })?;
            selected.push(head as i32);
        }
        layers.push((layer, selected));
    }
    layers.sort_by_key(|(layer, _)| *layer);
    Ok(layers)
}

fn json_i32(value: &serde_json::Value, key: &str) -> std::result::Result<i32, String> {
    value
        .get(key)
        .and_then(|item| item.as_i64())
        .map(|item| item as i32)
        .ok_or_else(|| format!("ACE-Step config.json is missing {key}"))
}

fn json_i32_array(value: &serde_json::Value, key: &str) -> std::result::Result<Vec<i32>, String> {
    let items = value
        .get(key)
        .and_then(|item| item.as_array())
        .ok_or_else(|| format!("ACE-Step config.json is missing {key}"))?;
    items
        .iter()
        .map(|item| {
            item.as_i64()
                .map(|item| item as i32)
                .ok_or_else(|| format!("ACE-Step {key} contains a non-integer"))
        })
        .collect()
}

/// Condition encoder: lyric ‖ timbre CLS ‖ projected text. Every token is valid.
pub fn encode_condition(
    weights: &MlxWeights,
    config: &AceConfig,
    text: &Array,
    lyric: &Array,
    timbre: &Array,
    stream: &Stream,
) -> Result<Array> {
    let text = text.astype(Dtype::Float32, stream)?;
    let lyric = lyric.astype(Dtype::Float32, stream)?;
    let timbre = timbre.astype(Dtype::Float32, stream)?;
    let text = apply_linear(weights, "encoder.text_projector", &text, stream)?;
    let lyric = encode_stack(
        weights,
        config,
        "encoder.lyric_encoder",
        config.lyric_layers,
        config.encoder_hidden,
        config.encoder_intermediate,
        config.encoder_heads,
        config.encoder_kv_heads,
        &apply_linear(
            weights,
            "encoder.lyric_encoder.embed_tokens",
            &lyric,
            stream,
        )?,
        stream,
    )?;
    let timbre = encode_timbre(weights, config, &timbre, stream)?;
    Array::concatenate_axis(&[lyric, timbre, text], 1, stream)
}

/// Quantize 25 Hz acoustic latents and expand them back to 25 Hz.
///
/// `latents` is `[1, frames, 64]` and `frames` is a positive multiple of
/// `pool_window`. Each window is projected, pooled by the CLS token, and
/// mapped through one ResidualFSQ. The existing detokenizer expands the
/// code to `pool_window` frames.
pub fn retokenize_cover_latents(
    weights: &MlxWeights,
    config: &AceConfig,
    latents: &Array,
    stream: &Stream,
) -> Result<Array> {
    let shape = latents.shape();
    let pool = config.pool_window;
    if shape.len() != 3 || shape[0] != 1 || shape[2] != config.out_channels {
        return Err(danmo_mlx::Error::new(
            "ACE-Step FSQ cover",
            format!(
                "acoustic latents must be [1, frames, {}], got {shape:?}",
                config.out_channels
            ),
        ));
    }
    let frames = shape[1];
    if pool < 1 || config.pool_layers < 1 || frames < 1 || frames % pool != 0 {
        return Err(danmo_mlx::Error::new(
            "ACE-Step FSQ cover",
            format!("frames {frames} must be a positive multiple of pool window {pool}"),
        ));
    }
    if config.fsq_quantizers != 1
        || config.fsq_levels.as_slice() != super::super::fsq_quant::FSQ_LEVELS
    {
        return Err(danmo_mlx::Error::new(
            "ACE-Step FSQ cover",
            format!(
                "tokenizer expects 1 quantizer and levels {:?}, got {} and {:?}",
                super::super::fsq_quant::FSQ_LEVELS,
                config.fsq_quantizers,
                config.fsq_levels
            ),
        ));
    }
    let patches = frames / pool;
    let width = config.encoder_hidden;
    let projected = apply_linear(weights, "tokenizer.audio_acoustic_proj", latents, stream)?
        .reshape(&[1, patches, pool, width], stream)?;
    let embedded = apply_linear(
        weights,
        "tokenizer.attention_pooler.embed_tokens",
        &projected,
        stream,
    )?;
    let special = weights
        .tensor("tokenizer.attention_pooler.special_token")?
        .reshape(&[1, 1, 1, width], stream)?
        .broadcast_to(&[1, patches, 1, width], stream)?;
    let pooled = Array::concatenate_axis(&[special, embedded], 2, stream)?
        .reshape(&[patches, pool + 1, width], stream)?;
    let pooled = encode_stack(
        weights,
        config,
        "tokenizer.attention_pooler",
        config.pool_layers,
        width,
        config.encoder_intermediate,
        config.encoder_heads,
        config.encoder_kv_heads,
        &pooled,
        stream,
    )?;
    let cls = pooled
        .slice(&[0, 0, 0], &[patches, 1, width], &[1, 1, 1], stream)?
        .reshape(&[1, patches, width], stream)?;
    let raw = apply_linear(weights, "tokenizer.quantizer.project_in", &cls, stream)?
        .astype(Dtype::Float32, stream)?;
    raw.eval()?;
    let values = raw.to_vec_f32(stream)?;
    let dims = super::super::fsq_quant::FSQ_NDIMS;
    if values.len() != patches as usize * dims {
        return Err(danmo_mlx::Error::new(
            "ACE-Step FSQ cover",
            format!(
                "project_in wrote {} values for {patches} patches of {dims}",
                values.len()
            ),
        ));
    }
    let mut indices = Vec::with_capacity(patches as usize);
    for chunk in values.chunks(dims) {
        let mut row = [0.0f32; super::super::fsq_quant::FSQ_NDIMS];
        row.copy_from_slice(chunk);
        indices.push(super::super::fsq_quant::encode_index(&row));
    }
    detokenize_audio_codes(weights, config, &indices, stream)
}

/// Turn 5 Hz FSQ indices into 25 Hz acoustic context `[1, codes * pool, 64]`.
///
/// One quantizer: the index is the implicit 6-d codebook row, `project_out`
/// lifts it to the encoder width, and the detokenizer repeats it across the
/// pool window. Indices outside the 64000-entry book fail.
pub fn detokenize_audio_codes(
    weights: &MlxWeights,
    config: &AceConfig,
    indices: &[i32],
    stream: &Stream,
) -> Result<Array> {
    if indices.is_empty() {
        return Err(danmo_mlx::Error::new(
            "ACE-Step audio codes",
            "audio code list is empty",
        ));
    }
    if config.pool_window < 1 || config.pool_layers < 1 {
        return Err(danmo_mlx::Error::new(
            "ACE-Step audio codes",
            format!(
                "pool window {} and pool layers {} must be positive",
                config.pool_window, config.pool_layers
            ),
        ));
    }
    let book = super::super::fsq_quant::FSQ_CODEBOOK_SIZE;
    let mut raw = Vec::with_capacity(indices.len() * super::super::fsq_quant::FSQ_NDIMS);
    for &index in indices {
        if index < 0 || index >= book {
            return Err(danmo_mlx::Error::new(
                "ACE-Step audio codes",
                format!("code {index} is outside [0, {book})"),
            ));
        }
        raw.extend_from_slice(&super::super::fsq_quant::decode_index(index));
    }
    let time = indices.len() as i32;
    let codes = Array::from_f32_slice(&raw, &[1, time, super::super::fsq_quant::FSQ_NDIMS as i32])?;
    let hidden = apply_linear(weights, "tokenizer.quantizer.project_out", &codes, stream)?;
    let embedded = apply_linear(weights, "detokenizer.embed_tokens", &hidden, stream)?;
    let width = config.encoder_hidden;
    let pool = config.pool_window;
    let expanded = embedded
        .expand_dims(2, stream)?
        .broadcast_to(&[1, time, pool, width], stream)?;
    let special = weights
        .tensor("detokenizer.special_tokens")?
        .reshape(&[1, 1, pool, width], stream)?
        .broadcast_to(&[1, time, pool, width], stream)?;
    let tokens = Array::add(&expanded, &special, stream)?.reshape(&[time, pool, width], stream)?;
    let decoded = encode_stack(
        weights,
        config,
        "detokenizer",
        config.pool_layers,
        config.encoder_hidden,
        config.encoder_intermediate,
        config.encoder_heads,
        config.encoder_kv_heads,
        &tokens,
        stream,
    )?;
    apply_linear(weights, "detokenizer.proj_out", &decoded, stream)?
        .reshape(&[1, time * pool, config.out_channels], stream)
}

/// One denoising velocity. `latents` and `context` are `[1, T, C]`.
pub fn velocity(
    weights: &MlxWeights,
    config: &AceConfig,
    latents: &Array,
    context: &Array,
    encoder: &Array,
    timestep: f32,
    stream: &Stream,
) -> Result<Array> {
    let mut state = prepare_decoder(weights, config, latents, context, encoder, timestep, stream)?;
    let sliding = state.sliding.clone();
    for layer in 0..config.layers {
        let mask = if config.sliding[layer as usize] {
            Some(&sliding)
        } else {
            None
        };
        let (hidden, _) = dit_layer(
            weights,
            config,
            layer,
            &state.hidden,
            &state.modulation,
            &state.cos,
            &state.sin,
            &state.encoder,
            mask,
            &[],
            stream,
        )?;
        state.hidden = hidden;
    }
    let mut hidden = state.hidden;
    let temb = state.temb;
    let original = state.original;
    let patch = config.patch;
    let temb = temb.expand_dims(1, stream)?;
    let table = weights.tensor("decoder.scale_shift_table")?;
    let shift_scale = Array::add(table, &temb, stream)?;
    let parts = shift_scale.split_sections(&[1], 1, stream)?;
    let norm = rms(
        weights,
        "decoder.norm_out.weight",
        &hidden,
        config.rms_eps,
        stream,
    )?;
    let one = Array::from_f32_slice(&[1.0], &[])?;
    let scaled = Array::multiply(&norm, &Array::add(&one, &parts[1], stream)?, stream)?;
    hidden = Array::add(&scaled, &parts[0], stream)?;
    hidden = conv_transpose1d_pt(weights, "decoder.proj_out.1", &hidden, patch, 0, stream)?;
    hidden.slice(
        &[0, 0, 0],
        &[1, original, config.out_channels],
        &[1, 1, 1],
        stream,
    )
}

fn encode_timbre(
    weights: &MlxWeights,
    config: &AceConfig,
    timbre: &Array,
    stream: &Stream,
) -> Result<Array> {
    let mut hidden = apply_linear(
        weights,
        "encoder.timbre_encoder.embed_tokens",
        timbre,
        stream,
    )?;
    let special = weights.tensor("encoder.timbre_encoder.special_token")?;
    let batch = hidden.shape()[0];
    let special = special.broadcast_to(&[batch, 1, config.encoder_hidden], stream)?;
    hidden = Array::concatenate_axis(&[special, hidden], 1, stream)?;
    hidden = encode_stack(
        weights,
        config,
        "encoder.timbre_encoder",
        config.timbre_layers,
        config.encoder_hidden,
        config.encoder_intermediate,
        config.encoder_heads,
        config.encoder_kv_heads,
        &hidden,
        stream,
    )?;
    hidden.slice(
        &[0, 0, 0],
        &[batch, 1, config.encoder_hidden],
        &[1, 1, 1],
        stream,
    )
}

fn encode_stack(
    weights: &MlxWeights,
    config: &AceConfig,
    prefix: &str,
    layers: i32,
    hidden_size: i32,
    intermediate: i32,
    heads: i32,
    kv_heads: i32,
    input: &Array,
    stream: &Stream,
) -> Result<Array> {
    let mut hidden = input.clone();
    let sequence = hidden.shape()[1];
    let (cos, sin) = rope(
        config.head_dim,
        config.rope_theta,
        sequence,
        Dtype::Float32,
        stream,
    )?;
    let sliding = sliding_mask(sequence, config.sliding_window, stream)?;
    for layer in 0..layers {
        let mask = if config.sliding[layer as usize] {
            Some(&sliding)
        } else {
            None
        };
        hidden = encoder_layer(
            weights,
            &format!("{prefix}.layers.{layer}"),
            hidden_size,
            intermediate,
            heads,
            kv_heads,
            config.head_dim,
            config.rms_eps,
            &hidden,
            &cos,
            &sin,
            mask,
            stream,
        )?;
    }
    rms(
        weights,
        &format!("{prefix}.norm.weight"),
        &hidden,
        config.rms_eps,
        stream,
    )
}

fn encoder_layer(
    weights: &MlxWeights,
    prefix: &str,
    _hidden_size: i32,
    intermediate: i32,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    eps: f32,
    hidden: &Array,
    cos: &Array,
    sin: &Array,
    mask: Option<&Array>,
    stream: &Stream,
) -> Result<Array> {
    let normed = rms(
        weights,
        &format!("{prefix}.input_layernorm.weight"),
        hidden,
        eps,
        stream,
    )?;
    let attended = attention(
        weights,
        &format!("{prefix}.self_attn"),
        &normed,
        None,
        heads,
        kv_heads,
        head_dim,
        eps,
        Some(cos),
        Some(sin),
        mask,
        stream,
    )?;
    let hidden = Array::add(hidden, &attended, stream)?;
    let normed = rms(
        weights,
        &format!("{prefix}.post_attention_layernorm.weight"),
        &hidden,
        eps,
        stream,
    )?;
    let mlp = swiglu(
        weights,
        &format!("{prefix}.mlp"),
        &normed,
        intermediate,
        stream,
    )?;
    Array::add(&hidden, &mlp, stream)
}

struct DecoderState {
    hidden: Array,
    encoder: Array,
    modulation: Array,
    temb: Array,
    cos: Array,
    sin: Array,
    sliding: Array,
    original: i32,
}

fn prepare_decoder(
    weights: &MlxWeights,
    config: &AceConfig,
    latents: &Array,
    context: &Array,
    encoder: &Array,
    timestep: f32,
    stream: &Stream,
) -> Result<DecoderState> {
    let hidden = Array::concatenate_axis(&[context.clone(), latents.clone()], -1, stream)?;
    let original = hidden.shape()[1];
    let patch = config.patch;
    let mut hidden = hidden;
    if original % patch != 0 {
        let pad = patch - original % patch;
        let channels = hidden.shape()[2];
        let zeros = Array::from_f32_slice(
            &vec![0.0; pad as usize * channels as usize],
            &[1, pad, channels],
        )?;
        hidden = Array::concatenate_axis(&[hidden, zeros], 1, stream)?;
    }
    let hidden = conv1d_pt(weights, "decoder.proj_in.1", &hidden, patch, 0, stream)?;
    let encoder = apply_linear(weights, "decoder.condition_embedder", encoder, stream)?;
    let steps = scalar(timestep, stream)?;
    let (temb_t, proj_t) =
        timestep_embed(weights, "decoder.time_embed", &steps, config.hidden, stream)?;
    let zero = scalar(0.0, stream)?;
    let (temb_r, proj_r) = timestep_embed(
        weights,
        "decoder.time_embed_r",
        &zero,
        config.hidden,
        stream,
    )?;
    let sequence = hidden.shape()[1];
    let (cos, sin) = rope(
        config.head_dim,
        config.rope_theta,
        sequence,
        Dtype::Float32,
        stream,
    )?;
    Ok(DecoderState {
        hidden,
        encoder,
        modulation: Array::add(&proj_t, &proj_r, stream)?,
        temb: Array::add(&temb_t, &temb_r, stream)?,
        cos,
        sin,
        sliding: sliding_mask(sequence, config.sliding_window, stream)?,
        original,
    })
}

/// Cross-attention of the configured lyric heads, softmaxed over the encoder.
/// Layout is `[heads, frames, encoder]`.
pub fn capture_lyric_attention(
    weights: &MlxWeights,
    config: &AceConfig,
    latents: &Array,
    context: &Array,
    encoder: &Array,
    timestep: f32,
    stream: &Stream,
) -> Result<(usize, usize, usize, Vec<f32>)> {
    if config.align_layers.is_empty() {
        return Err(danmo_mlx::Error::new(
            "ACE-Step lyric alignment",
            "config.json has no lyric_alignment_layers_config",
        ));
    }
    let last = config
        .align_layers
        .iter()
        .map(|(layer, _)| *layer)
        .max()
        .unwrap();
    if last < 0 || last >= config.layers {
        return Err(danmo_mlx::Error::new(
            "ACE-Step lyric alignment",
            format!("layer {last} is outside {} decoder layers", config.layers),
        ));
    }
    let mut state = prepare_decoder(weights, config, latents, context, encoder, timestep, stream)?;
    let sliding = state.sliding.clone();
    let mut weights_out = Vec::new();
    let mut heads = 0usize;
    let mut frames = 0usize;
    let mut encoder_len = 0usize;
    for layer in 0..=last {
        let selected = config
            .align_layers
            .iter()
            .find(|(id, _)| *id == layer)
            .map(|(_, heads)| heads.as_slice())
            .unwrap_or(&[]);
        let mask = if config.sliding[layer as usize] {
            Some(&sliding)
        } else {
            None
        };
        let (hidden, probs) = dit_layer(
            weights,
            config,
            layer,
            &state.hidden,
            &state.modulation,
            &state.cos,
            &state.sin,
            &state.encoder,
            mask,
            selected,
            stream,
        )?;
        state.hidden = hidden;
        if let Some((layer_frames, layer_encoder, values)) = probs {
            if frames == 0 {
                frames = layer_frames;
                encoder_len = layer_encoder;
            } else if frames != layer_frames || encoder_len != layer_encoder {
                return Err(danmo_mlx::Error::new(
                    "ACE-Step lyric alignment",
                    "cross-attention shapes differ across layers",
                ));
            }
            heads += selected.len();
            weights_out.extend(values);
        }
    }
    if heads == 0 || frames == 0 || encoder_len == 0 {
        return Err(danmo_mlx::Error::new(
            "ACE-Step lyric alignment",
            "no cross-attention heads were captured",
        ));
    }
    Ok((heads, frames, encoder_len, weights_out))
}

fn dit_layer(
    weights: &MlxWeights,
    config: &AceConfig,
    layer: i32,
    hidden: &Array,
    modulation: &Array,
    cos: &Array,
    sin: &Array,
    encoder: &Array,
    mask: Option<&Array>,
    capture: &[i32],
    stream: &Stream,
) -> Result<(Array, Option<(usize, usize, Vec<f32>)>)> {
    let prefix = format!("decoder.layers.{layer}");
    let table = weights.tensor(&format!("{prefix}.scale_shift_table"))?;
    let mods =
        Array::add(table, modulation, stream)?.split_sections(&[1, 2, 3, 4, 5], 1, stream)?;
    let normed = rms(
        weights,
        &format!("{prefix}.self_attn_norm.weight"),
        hidden,
        config.rms_eps,
        stream,
    )?;
    let one = Array::from_f32_slice(&[1.0], &[])?;
    let normed = Array::add(
        &Array::multiply(&normed, &Array::add(&one, &mods[1], stream)?, stream)?,
        &mods[0],
        stream,
    )?;
    let attended = attention(
        weights,
        &format!("{prefix}.self_attn"),
        &normed,
        None,
        config.heads,
        config.kv_heads,
        config.head_dim,
        config.rms_eps,
        Some(cos),
        Some(sin),
        mask,
        stream,
    )?;
    let hidden = Array::add(
        hidden,
        &Array::multiply(&attended, &mods[2], stream)?,
        stream,
    )?;
    let normed = rms(
        weights,
        &format!("{prefix}.cross_attn_norm.weight"),
        &hidden,
        config.rms_eps,
        stream,
    )?;
    let captured = if capture.is_empty() {
        None
    } else {
        Some(attention_probs(
            weights,
            &format!("{prefix}.cross_attn"),
            &normed,
            encoder,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.rms_eps,
            capture,
            stream,
        )?)
    };
    let cross = attention(
        weights,
        &format!("{prefix}.cross_attn"),
        &normed,
        Some(encoder),
        config.heads,
        config.kv_heads,
        config.head_dim,
        config.rms_eps,
        None,
        None,
        None,
        stream,
    )?;
    let hidden = Array::add(&hidden, &cross, stream)?;
    let normed = rms(
        weights,
        &format!("{prefix}.mlp_norm.weight"),
        &hidden,
        config.rms_eps,
        stream,
    )?;
    let normed = Array::add(
        &Array::multiply(&normed, &Array::add(&one, &mods[4], stream)?, stream)?,
        &mods[3],
        stream,
    )?;
    let mlp = swiglu(
        weights,
        &format!("{prefix}.mlp"),
        &normed,
        config.intermediate,
        stream,
    )?;
    Ok((
        Array::add(&hidden, &Array::multiply(&mlp, &mods[5], stream)?, stream)?,
        captured,
    ))
}

fn attention_probs(
    weights: &MlxWeights,
    prefix: &str,
    hidden: &Array,
    encoder: &Array,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    eps: f32,
    selected: &[i32],
    stream: &Stream,
) -> Result<(usize, usize, Vec<f32>)> {
    let sequence = hidden.shape()[1];
    let encoder_len = encoder.shape()[1];
    let mut q = apply_linear(weights, &format!("{prefix}.q_proj"), hidden, stream)?
        .reshape(&[1, sequence, heads, head_dim], stream)?;
    q = rms_last(weights, &format!("{prefix}.q_norm.weight"), &q, eps, stream)?;
    let mut k = apply_linear(weights, &format!("{prefix}.k_proj"), encoder, stream)?
        .reshape(&[1, encoder_len, kv_heads, head_dim], stream)?;
    k = rms_last(weights, &format!("{prefix}.k_norm.weight"), &k, eps, stream)?;
    q = q.transpose_axes(&[0, 2, 1, 3], stream)?;
    k = k.transpose_axes(&[0, 2, 1, 3], stream)?;
    let groups = heads / kv_heads.max(1);
    let scale = Array::from_f32_slice(&[1.0 / (head_dim as f32).sqrt()], &[])?;
    let mut out = Vec::with_capacity(selected.len() * sequence as usize * encoder_len as usize);
    for &head in selected {
        if head < 0 || head >= heads {
            return Err(danmo_mlx::Error::new(
                "ACE-Step lyric alignment",
                format!("head {head} is outside {heads} attention heads"),
            ));
        }
        let kv = head / groups;
        let qh = q
            .slice(
                &[0, head, 0, 0],
                &[1, head + 1, sequence, head_dim],
                &[1, 1, 1, 1],
                stream,
            )?
            .reshape(&[sequence, head_dim], stream)?;
        let kh = k
            .slice(
                &[0, kv, 0, 0],
                &[1, kv + 1, encoder_len, head_dim],
                &[1, 1, 1, 1],
                stream,
            )?
            .reshape(&[encoder_len, head_dim], stream)?
            .transpose_axes(&[1, 0], stream)?;
        let mut scores = Array::multiply(&Array::matmul(&qh, &kh, stream)?, &scale, stream)?;
        scores.eval()?;
        let mut row = scores.to_vec_f32(stream)?;
        let width = encoder_len as usize;
        for frame in 0..sequence as usize {
            softmax_inplace(&mut row[frame * width..(frame + 1) * width]);
        }
        out.extend(row);
    }
    Ok((sequence as usize, encoder_len as usize, out))
}

fn softmax_inplace(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for value in row.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    let inv = 1.0 / (sum + 1e-9);
    for value in row.iter_mut() {
        *value *= inv;
    }
}

fn attention(
    weights: &MlxWeights,
    prefix: &str,
    hidden: &Array,
    encoder: Option<&Array>,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    eps: f32,
    cos: Option<&Array>,
    sin: Option<&Array>,
    mask: Option<&Array>,
    stream: &Stream,
) -> Result<Array> {
    let shape = hidden.shape();
    let (batch, sequence) = (shape[0], shape[1]);
    let source = encoder.unwrap_or(hidden);
    let source_len = source.shape()[1];
    let mut q = apply_linear(weights, &format!("{prefix}.q_proj"), hidden, stream)?
        .reshape(&[batch, sequence, heads, head_dim], stream)?;
    q = rms_last(weights, &format!("{prefix}.q_norm.weight"), &q, eps, stream)?;
    let mut k = apply_linear(weights, &format!("{prefix}.k_proj"), source, stream)?
        .reshape(&[batch, source_len, kv_heads, head_dim], stream)?;
    k = rms_last(weights, &format!("{prefix}.k_norm.weight"), &k, eps, stream)?;
    let mut v = apply_linear(weights, &format!("{prefix}.v_proj"), source, stream)?
        .reshape(&[batch, source_len, kv_heads, head_dim], stream)?;
    if let (Some(cos), Some(sin)) = (cos, sin) {
        q = apply_rope(&q, cos, sin, stream)?;
        k = apply_rope(&k, cos, sin, stream)?;
    }
    let groups = heads / kv_heads;
    if groups > 1 {
        k = k.repeat_axis(groups, 2, stream)?;
        v = v.repeat_axis(groups, 2, stream)?;
    }
    q = q.transpose_axes(&[0, 2, 1, 3], stream)?;
    k = k.transpose_axes(&[0, 2, 1, 3], stream)?;
    v = v.transpose_axes(&[0, 2, 1, 3], stream)?;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let output = Array::scaled_dot_product_attention(&q, &k, &v, scale, "", mask, stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, sequence, heads * head_dim], stream)?;
    apply_linear(weights, &format!("{prefix}.o_proj"), &output, stream)
}

fn swiglu(
    weights: &MlxWeights,
    prefix: &str,
    hidden: &Array,
    _intermediate: i32,
    stream: &Stream,
) -> Result<Array> {
    let gate =
        apply_linear(weights, &format!("{prefix}.gate_proj"), hidden, stream)?.silu(stream)?;
    let up = apply_linear(weights, &format!("{prefix}.up_proj"), hidden, stream)?;
    let product = Array::multiply(&gate, &up, stream)?;
    apply_linear(weights, &format!("{prefix}.down_proj"), &product, stream)
}

fn timestep_embed(
    weights: &MlxWeights,
    prefix: &str,
    timestep: &Array,
    hidden: i32,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let freq = sinusoidal(timestep, 256, stream)?;
    let mut temb =
        apply_linear(weights, &format!("{prefix}.linear_1"), &freq, stream)?.silu(stream)?;
    temb = apply_linear(weights, &format!("{prefix}.linear_2"), &temb, stream)?;
    let proj = apply_linear(
        weights,
        &format!("{prefix}.time_proj"),
        &temb.silu(stream)?,
        stream,
    )?;
    let batch = proj.shape()[0];
    let proj = proj.reshape(&[batch, 6, hidden], stream)?;
    Ok((temb, proj))
}

fn sinusoidal(timestep: &Array, dim: i32, stream: &Stream) -> Result<Array> {
    let scaled = Array::multiply(timestep, &Array::from_f32_slice(&[1000.0], &[])?, stream)?;
    let half = dim / 2;
    let arange = Array::arange(0.0, half as f64, 1.0, Dtype::Float32, stream)?;
    let half_scalar = Array::from_f32_slice(&[half as f32], &[])?;
    let exponent = Array::divide(&arange, &half_scalar, stream)?;
    let period = Array::from_f32_slice(&[10_000.0], &[])?;
    let freqs = Array::power(&period, &exponent.negative(stream)?, stream)?;
    let args = Array::multiply(
        &scaled.expand_dims(1, stream)?,
        &freqs.expand_dims(0, stream)?,
        stream,
    )?;
    Array::concatenate_axis(&[args.cos(stream)?, args.sin(stream)?], -1, stream)
}

fn rope(
    head_dim: i32,
    theta: f32,
    sequence: i32,
    dtype: Dtype,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let arange = Array::arange(0.0, head_dim as f64, 2.0, Dtype::Float32, stream)?;
    let dim_scalar = Array::from_f32_slice(&[head_dim as f32], &[])?;
    let exponent = Array::divide(&arange, &dim_scalar, stream)?;
    let base = Array::from_f32_slice(&[theta], &[])?;
    let inv_freq = Array::divide(
        &Array::from_f32_slice(&[1.0], &[])?,
        &Array::power(&base, &exponent, stream)?,
        stream,
    )?;
    let positions = Array::arange(0.0, sequence as f64, 1.0, Dtype::Float32, stream)?;
    let freqs = Array::multiply(
        &positions.expand_dims(1, stream)?,
        &inv_freq.expand_dims(0, stream)?,
        stream,
    )?;
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

fn sliding_mask(sequence: i32, window: i32, stream: &Stream) -> Result<Array> {
    let n = sequence as usize;
    let mut mask = vec![0.0f32; n * n];
    for row in 0..n {
        for col in 0..n {
            let distance = (row as i32 - col as i32).unsigned_abs();
            if distance > window as u32 {
                mask[row * n + col] = -1.0e9;
            }
        }
    }
    let mask = Array::from_f32_slice(&mask, &[1, 1, sequence, sequence])?;
    let _ = stream;
    Ok(mask)
}

fn conv1d_pt(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    stride: i32,
    padding: i32,
    stream: &Stream,
) -> Result<Array> {
    let kernel = weights
        .tensor(&format!("{prefix}.weight"))?
        .transpose_axes(&[0, 2, 1], stream)?;
    let output = Array::conv1d(input, &kernel, stride, padding, 1, 1, stream)?;
    add_bias(weights, prefix, output, stream)
}

fn conv_transpose1d_pt(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    stride: i32,
    padding: i32,
    stream: &Stream,
) -> Result<Array> {
    let kernel = weights
        .tensor(&format!("{prefix}.weight"))?
        .transpose_axes(&[1, 2, 0], stream)?;
    let output = Array::conv_transpose1d(input, &kernel, stride, padding, 1, 0, 1, stream)?;
    add_bias(weights, prefix, output, stream)
}

fn add_bias(weights: &MlxWeights, prefix: &str, output: Array, stream: &Stream) -> Result<Array> {
    let name = format!("{prefix}.bias");
    if weights.has(&name) {
        Array::add(&output, weights.tensor(&name)?, stream)
    } else {
        Ok(output)
    }
}

fn apply_linear(weights: &MlxWeights, name: &str, input: &Array, stream: &Stream) -> Result<Array> {
    let bias_name = format!("{name}.bias");
    let bias = if weights.has(&bias_name) {
        Some(weights.tensor(&bias_name)?.clone())
    } else {
        None
    };
    weights
        .linear(&format!("{name}.weight"))?
        .apply(input, bias.as_ref(), stream)
}

fn rms(
    weights: &MlxWeights,
    name: &str,
    input: &Array,
    eps: f32,
    stream: &Stream,
) -> Result<Array> {
    input.rms_norm(Some(weights.tensor(name)?), eps, stream)
}

fn rms_last(
    weights: &MlxWeights,
    name: &str,
    input: &Array,
    eps: f32,
    stream: &Stream,
) -> Result<Array> {
    input.rms_norm(Some(weights.tensor(name)?), eps, stream)
}

fn scalar(value: f32, _stream: &Stream) -> Result<Array> {
    Array::from_f32_slice(&[value], &[1])
}

#[cfg(test)]
mod tests {
    use super::alignment_layers;

    #[test]
    fn alignment_layers_sort_and_keep_head_order() {
        let value = serde_json::json!({
            "lyric_alignment_layers_config": {
                "7": [20, 21],
                "3": [18, 27]
            }
        });
        let layers = alignment_layers(&value).unwrap();
        assert_eq!(layers, vec![(3, vec![18, 27]), (7, vec![20, 21])]);
    }
}
