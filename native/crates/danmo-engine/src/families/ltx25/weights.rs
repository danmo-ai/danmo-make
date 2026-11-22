//! LTX 2.5 checkpoint-specific key and tensor layout conversion.
//!
//! These mappings are not a shared weight loader. `danmo_mlx` owns tensor I/O,
//! transpose, dtype conversion, and affine quantization; this module only
//! describes the official LTX 2.5 checkpoint schema.

use danmo_mlx::{Array, Stream};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

pub type Weights = BTreeMap<String, Array>;

const DIFFUSION_PREFIX: &str = "model.diffusion_model.";
const ATTENTION_KINDS: &[&str] = &[
    "attn1",
    "attn2",
    "audio_attn1",
    "audio_attn2",
    "audio_to_video_attn",
    "video_to_audio_attn",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentLayout {
    VideoVae,
    AudioVae,
    Upsampler,
}

pub fn normalize_key(key: &str) -> String {
    let mut normalized = key
        .strip_prefix(DIFFUSION_PREFIX)
        .unwrap_or(key)
        .to_string();
    normalized = normalized.replace(".emb.timestep_embedder.", ".emb.");
    normalized = normalized.replace(".ff.net.0.proj.", ".ff.proj_in.");
    normalized = normalized.replace(".ff.net.2.", ".ff.proj_out.");
    normalized = normalized.replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.");
    normalized = normalized.replace(".audio_ff.net.2.", ".audio_ff.proj_out.");
    for attention in ATTENTION_KINDS {
        normalized = normalized.replace(
            &format!(".{attention}.to_out.0."),
            &format!(".{attention}.to_out."),
        );
    }
    normalized
}

pub fn normalize_weights(weights: Weights) -> Weights {
    weights
        .into_iter()
        .map(|(key, tensor)| (normalize_key(&key), tensor))
        .collect()
}

pub fn is_dit_checkpoint(weights: &Weights) -> bool {
    weights.keys().any(|key| {
        key.contains("patchify_proj")
            || key.starts_with("model.diffusion_model.transformer_blocks.")
            || key.contains(".audio_to_video_attn.")
    })
}

pub fn split_transformer(weights: Weights) -> Result<(Weights, Weights), String> {
    let mut transformer = BTreeMap::new();
    let mut connector = BTreeMap::new();
    const VIDEO_CONNECTOR: &str = "model.diffusion_model.video_embeddings_connector.";
    const AUDIO_CONNECTOR: &str = "model.diffusion_model.audio_embeddings_connector.";
    for (key, tensor) in weights {
        if let Some(rest) = key.strip_prefix(VIDEO_CONNECTOR) {
            connector.insert(format!("video_connector.{rest}"), tensor);
        } else if let Some(rest) = key.strip_prefix(AUDIO_CONNECTOR) {
            connector.insert(format!("audio_connector.{rest}"), tensor);
        } else if key.starts_with(DIFFUSION_PREFIX) {
            transformer.insert(normalize_key(&key), tensor);
        } else if key.starts_with("connectors.") {
            // Legacy unit-test / older checkpoint layout.
            connector.insert(key, tensor);
        } else {
            return Err(format!("unexpected LTX 2.5 transformer key: {key}"));
        }
    }
    Ok((transformer, connector))
}

pub fn split_audio_vae(weights: Weights) -> (Weights, Weights) {
    let mut audio_vae = BTreeMap::new();
    let mut vocoder = BTreeMap::new();
    for (key, tensor) in weights {
        if key.starts_with("vocoder.") {
            vocoder.insert(key, tensor);
        } else {
            audio_vae.insert(key, tensor);
        }
    }
    (audio_vae, vocoder)
}

/// Pull caption-projection weights out of the with-proj Gemma checkpoint.
///
/// Published keys use `text_embedding_projection.*`; they are stored under
/// `feature_extractor.*` in `connector.safetensors` (Python ingest parity).
pub fn extract_feature_extractor(weights: &mut Weights) -> Result<Weights, String> {
    let mut output = BTreeMap::new();
    let projection_keys: Vec<String> = weights
        .keys()
        .filter(|key| key.starts_with("text_embedding_projection."))
        .cloned()
        .collect();
    for key in projection_keys {
        let tensor = weights
            .remove(&key)
            .expect("key listed from weights must exist");
        let rest = key
            .strip_prefix("text_embedding_projection.")
            .expect("prefix checked above");
        output.insert(format!("feature_extractor.{rest}"), tensor);
    }
    if !output.is_empty() {
        return Ok(output);
    }
    // Legacy fixture layout used by older unit tests.
    for (source, destination) in [
        ("feature_extractor.projection.weight", "proj_in.weight"),
        ("feature_extractor.projection.bias", "proj_in.bias"),
        (
            "feature_extractor.audio_projection.weight",
            "audio_proj_in.weight",
        ),
        (
            "feature_extractor.audio_projection.bias",
            "audio_proj_in.bias",
        ),
        (
            "feature_extractor.shared_audio_projection.weight",
            "shared_audio_proj.weight",
        ),
        (
            "feature_extractor.shared_audio_projection.bias",
            "shared_audio_proj.bias",
        ),
    ] {
        if let Some(tensor) = weights.remove(source) {
            output.insert(destination.to_string(), tensor);
        }
    }
    for source in ["weight", "bias"] {
        if let Some(tensor) = weights.remove(&format!("feature_extractor.normalization.{source}")) {
            output.insert(format!("video_feature_norm.{source}"), tensor);
        }
    }
    for source in ["weight", "bias"] {
        if let Some(tensor) = weights.remove(&format!("feature_extractor.project_mlp.0.{source}")) {
            output.insert(format!("per_stack.projection.0.{source}"), tensor);
        }
        if let Some(tensor) = weights.remove(&format!("feature_extractor.project_mlp.2.{source}")) {
            output.insert(format!("per_stack.projection.2.{source}"), tensor);
        }
    }
    if output.is_empty() {
        return Err(
            "No text_embedding_projection.* keys in the LTX 2.5 text encoder checkpoint — expected the with-proj file"
                .into(),
        );
    }
    Ok(output)
}

/// MLX channel-last keeps kernel axes right after `O`. Published OIK/OIHW/OIDHW
/// puts `I` there instead. Used by ingest transpose (idempotent) and validate.
///
/// Vocoder `*.ups.*` are ConvTranspose1d: published `(in,out,k)` → MLX `(out,k,in)`.
pub fn conv_weight_is_channel_last(key: &str, shape: &[i32]) -> bool {
    let shape: Vec<usize> = shape.iter().map(|&dim| dim as usize).collect();
    match shape.len() {
        5 => shape[1] == shape[2] && shape[2] == shape[3],
        4 => shape[1] == shape[2],
        3 if key.contains(".ups.") => shape[0] <= shape[2],
        3 => shape[1] <= shape[2],
        _ => true,
    }
}

fn conv_transpose_axes(key: &str, shape: &[i32]) -> Result<Option<&'static [i32]>, String> {
    if !key.ends_with(".weight") {
        return Ok(None);
    }
    match shape.len() {
        3 if key.contains(".ups.") => {
            // Kernel is the smallest axis for BigVGAN ups; its position tells
            // published (in,out,k) vs MLX (out,k,in) vs a mistaken (in,k,out).
            let kernel_axis = (0..3).min_by_key(|&axis| shape[axis]).expect("shape len 3");
            match kernel_axis {
                1 if shape[0] <= shape[2] => Ok(None), // MLX (out, k, in)
                1 => Ok(Some(&[2, 1, 0])),             // mistaken (in, k, out)
                2 => Ok(Some(&[1, 2, 0])),             // published (in, out, k)
                _ => Err(format!(
                    "cannot infer ConvTranspose1d layout for {key} {shape:?}"
                )),
            }
        }
        3 if !conv_weight_is_channel_last(key, shape) => Ok(Some(&[0, 2, 1])),
        4 if !conv_weight_is_channel_last(key, shape) => Ok(Some(&[0, 2, 3, 1])),
        5 if !conv_weight_is_channel_last(key, shape) => Ok(Some(&[0, 2, 3, 4, 1])),
        _ => Ok(None),
    }
}

pub fn transpose_component_weights(
    component: ComponentLayout,
    weights: Weights,
    stream: &Stream,
) -> Result<Weights, String> {
    weights
        .into_iter()
        .map(|(key, tensor)| {
            // Published LTX convs → MLX channel-last. Same family for VAE /
            // vocoder / upsampler; `.ups.` uses ConvTranspose1d axes.
            // Skip tensors that are already channel-last (idempotent repair).
            let axes = match component {
                ComponentLayout::VideoVae
                | ComponentLayout::AudioVae
                | ComponentLayout::Upsampler => conv_transpose_axes(&key, &tensor.shape())?,
            };
            let tensor = match axes {
                Some(axes) => tensor
                    .transpose_axes(axes, stream)
                    .and_then(|value| value.contiguous(stream))
                    .map_err(|error| format!("{component:?} tensor {key}: {error}"))?,
                None => tensor,
            };
            Ok((key, tensor))
        })
        .collect()
}

fn flatten_vocoder_config(audio_vae: &Value, audio: &Map<String, Value>) -> Map<String, Value> {
    if let Some(package) = audio_vae.get("vocoder") {
        if package.get("vocoder").is_some() || package.get("bwe").is_some() {
            let mut out = Map::new();
            if let Some(inner) = package.get("vocoder").cloned() {
                out.insert("vocoder".into(), inner);
            }
            if let Some(bwe) = package
                .get("bwe")
                .cloned()
                .or_else(|| audio_vae.get("bwe").cloned())
            {
                out.insert("bwe".into(), bwe);
            }
            return out;
        }
        let mut out = Map::new();
        out.insert("vocoder".into(), package.clone());
        if let Some(bwe) = audio_vae.get("bwe").cloned() {
            out.insert("bwe".into(), bwe);
        }
        return out;
    }
    if audio_vae.get("bwe").is_some() {
        let mut out = Map::new();
        if let Some(bwe) = audio_vae.get("bwe").cloned() {
            out.insert("bwe".into(), bwe);
        }
        return out;
    }
    // Legacy flat audio-vae config used by unit tests.
    audio
        .iter()
        .filter(|(key, _)| {
            matches!(
                key.as_str(),
                "upsampling_factors"
                    | "sampling_rate"
                    | "resblock_kernel_sizes"
                    | "resblock_dilations"
                    | "upsample_kernel_sizes"
            )
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

pub fn build_bundle_config(
    transformer: &Value,
    video_vae: &Value,
    audio_vae: &Value,
    upsampler: &Value,
    gemma: &Value,
) -> Result<Value, String> {
    // Published checkpoints nest sections under config.transformer / config.vae.
    let mut transformer = nested_or_flat(transformer, "transformer")?;
    if !transformer.contains_key("num_layers") {
        return Err("Transformer config missing 'num_layers' in checkpoint metadata".into());
    }
    transformer
        .entry("in_dims".to_string())
        .or_insert(Value::from(128));
    transformer
        .entry("audio_in_dims".to_string())
        .or_insert(Value::from(128));

    let vae = nested_or_flat(video_vae, "vae")?;
    validate_decoder_blocks(
        vae.get("decoder_blocks")
            .unwrap_or(&Value::Array(Vec::new())),
    )?;

    let mut audio = nested_or_flat(audio_vae, "audio_vae")?;
    if let Some(model) = audio_vae.get("model").cloned() {
        audio.insert("model".into(), model);
    }
    if let Some(preprocessing) = audio_vae.get("preprocessing").cloned() {
        audio.insert("preprocessing".into(), preprocessing);
    }
    let ddconfig = audio
        .get("model")
        .and_then(|value| value.get("params"))
        .and_then(|value| value.get("ddconfig"))
        .cloned()
        .or_else(|| audio.get("ddconfig").cloned())
        .unwrap_or(Value::Object(Map::new()));
    let mel_bins = ddconfig
        .get("mel_bins")
        .cloned()
        .or_else(|| {
            audio
                .get("preprocessing")
                .and_then(|value| value.get("mel"))
                .and_then(|value| value.get("n_mel_channels"))
                .cloned()
        })
        .or_else(|| {
            audio
                .get("variables")
                .and_then(|value| value.get("mel_bins"))
                .cloned()
        })
        .or_else(|| audio.get("mel_bins").cloned());
    audio.insert("ddconfig".into(), ddconfig);
    if let Some(mel_bins) = mel_bins {
        audio.insert("mel_bins".into(), mel_bins);
    }

    // Desired product shape: { "vocoder": {...params}, "bwe": {...params} }.
    // Published audio-vae metadata may already nest that package under
    // "vocoder" — unwrap once instead of wrapping again.
    let vocoder = flatten_vocoder_config(audio_vae, &audio);

    let upsampler = nested_or_flat(upsampler, "upsampler")?;

    let text_config = gemma
        .get("text_config")
        .or_else(|| gemma.get("text_encoder_config"))
        .unwrap_or(gemma);
    let hidden_size = text_config
        .get("hidden_size")
        .and_then(Value::as_u64)
        .ok_or_else(|| "Gemma text_config missing hidden_size".to_string())?;
    let num_hidden_layers = text_config
        .get("num_hidden_layers")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mut gemma_out = Map::from_iter([("hidden_size".into(), Value::from(hidden_size))]);
    if num_hidden_layers > 0 {
        gemma_out.insert("num_hidden_layers".into(), Value::from(num_hidden_layers));
        gemma_out.insert("tokenizer_max_length".into(), Value::from(1024));
    }

    Ok(Value::Object(Map::from_iter([
        ("transformer".into(), Value::Object(transformer)),
        ("vae".into(), Value::Object(vae)),
        ("audio_vae".into(), Value::Object(audio)),
        ("vocoder".into(), Value::Object(vocoder)),
        ("upsampler".into(), Value::Object(upsampler)),
        ("gemma".into(), Value::Object(gemma_out)),
    ])))
}

fn nested_or_flat(value: &Value, section: &str) -> Result<Map<String, Value>, String> {
    if let Some(nested) = value.get(section) {
        return object(nested, section);
    }
    object(value, section)
}

fn object(value: &Value, name: &str) -> Result<Map<String, Value>, String> {
    value
        .as_object()
        .cloned()
        .ok_or_else(|| format!("{name} checkpoint config must be a JSON object"))
}

fn validate_decoder_blocks(blocks: &Value) -> Result<(), String> {
    let blocks = blocks
        .as_array()
        .ok_or_else(|| "video VAE decoder_blocks must be an array".to_string())?;
    for block in blocks {
        if let Some(rows) = block.as_array() {
            // Published CausalVideoAutoencoder: ["res_x", {"num_layers": N}], ...
            if rows.len() == 2 && rows[0].is_string() && rows[1].is_object() {
                continue;
            }
            // Legacy compact triple used by unit fixtures.
            if rows.len() == 3 && rows.iter().all(Value::is_u64) {
                continue;
            }
        }
        if let Some(rows) = block.as_object() {
            for value in rows.values() {
                let parts = value
                    .as_array()
                    .ok_or_else(|| "VAE residual block entry must be an array".to_string())?;
                if parts.len() != 3
                    || !parts[0].is_array()
                    || !parts[1].is_array()
                    || !parts[2].is_u64()
                {
                    return Err(
                        "VAE residual block entry must be (channels, res_blocks, strided)".into(),
                    );
                }
            }
            continue;
        }
        return Err(format!("unsupported VAE decoder_blocks entry: {block}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        build_bundle_config, extract_feature_extractor, is_dit_checkpoint, normalize_key,
        normalize_weights, split_audio_vae, split_transformer, transpose_component_weights,
        ComponentLayout, Weights,
    };
    use danmo_mlx::{Array, Stream};
    use serde_json::json;

    fn weights(keys: &[&str]) -> Weights {
        keys.iter()
            .map(|key| {
                (
                    (*key).to_string(),
                    Array::from_f32_slice(&[1.0], &[1]).unwrap(),
                )
            })
            .collect()
    }

    #[test]
    fn official_transformer_keys_map_to_the_model_layout() {
        let key = "model.diffusion_model.transformer_blocks.0.attn1.to_out.0.weight";
        assert_eq!(
            normalize_key(key),
            "transformer_blocks.0.attn1.to_out.weight"
        );
        assert_eq!(
            normalize_key(
                "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight"
            ),
            "adaln_single.emb.linear_1.weight"
        );
        assert_eq!(
            normalize_key("model.diffusion_model.transformer_blocks.0.audio_ff.net.0.proj.weight"),
            "transformer_blocks.0.audio_ff.proj_in.weight"
        );
        let official = weights(&[key, "connectors.proj_in.weight"]);
        assert!(is_dit_checkpoint(&official));
        let normalized = normalize_weights(official);
        assert!(normalized.contains_key("transformer_blocks.0.attn1.to_out.weight"));
    }

    #[test]
    fn component_and_feature_extractor_splits_keep_namespaces_separate() {
        let (transformer, connector) = split_transformer(weights(&[
            "model.diffusion_model.patchify_proj.weight",
            "model.diffusion_model.video_embeddings_connector.learnable_registers",
            "model.diffusion_model.audio_embeddings_connector.learnable_registers",
        ]))
        .unwrap();
        assert!(transformer.contains_key("patchify_proj.weight"));
        assert!(connector.contains_key("video_connector.learnable_registers"));
        assert!(connector.contains_key("audio_connector.learnable_registers"));
        let (audio, vocoder) = split_audio_vae(weights(&["decoder.weight", "vocoder.res.weight"]));
        assert!(audio.contains_key("decoder.weight"));
        assert!(vocoder.contains_key("vocoder.res.weight"));

        let mut gemma = weights(&[
            "text_embedding_projection.video_aggregate_embed.weight",
            "text_embedding_projection.audio_aggregate_embed.bias",
            "model.layers.0.weight",
        ]);
        let feature = extract_feature_extractor(&mut gemma).unwrap();
        assert!(feature.contains_key("feature_extractor.video_aggregate_embed.weight"));
        assert!(feature.contains_key("feature_extractor.audio_aggregate_embed.bias"));
        assert_eq!(
            gemma.keys().collect::<Vec<_>>(),
            vec!["model.layers.0.weight"]
        );
    }

    #[test]
    fn video_convolutions_move_channels_to_the_last_axis() {
        let tensor = Array::from_f32_slice(
            &(0..16).map(|value| value as f32).collect::<Vec<_>>(),
            &[1, 2, 2, 2, 2],
        )
        .unwrap();
        let converted = transpose_component_weights(
            ComponentLayout::VideoVae,
            Weights::from([("decoder.conv.weight".into(), tensor)]),
            &Stream::cpu().unwrap(),
        )
        .unwrap();
        assert_eq!(
            converted["decoder.conv.weight"].shape(),
            vec![1, 2, 2, 2, 2]
        );
        // Published OIK [O,I,K]=[2,4,3] → OKI [O,K,I]=[2,3,4]
        let matrix = Array::from_f32_slice(
            &(0..2 * 4 * 3).map(|value| value as f32).collect::<Vec<_>>(),
            &[2, 4, 3],
        )
        .unwrap();
        let converted = transpose_component_weights(
            ComponentLayout::Upsampler,
            Weights::from([("conv.weight".into(), matrix)]),
            &Stream::cpu().unwrap(),
        )
        .unwrap();
        assert_eq!(converted["conv.weight"].shape(), vec![2, 3, 4]);
        // Idempotent on already channel-last OKI.
        let again = transpose_component_weights(
            ComponentLayout::Upsampler,
            Weights::from([("conv.weight".into(), converted["conv.weight"].clone())]),
            &Stream::cpu().unwrap(),
        )
        .unwrap();
        assert_eq!(again["conv.weight"].shape(), vec![2, 3, 4]);
        // Published OIDHW → MLX ODHWI (same rule as VAE).
        let conv3d = Array::from_f32_slice(
            &(0..2 * 4 * 3 * 3 * 3)
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
            &[2, 4, 3, 3, 3],
        )
        .unwrap();
        let converted = transpose_component_weights(
            ComponentLayout::Upsampler,
            Weights::from([("initial_conv.weight".into(), conv3d)]),
            &Stream::cpu().unwrap(),
        )
        .unwrap();
        assert_eq!(
            converted["initial_conv.weight"].shape(),
            vec![2, 3, 3, 3, 4]
        );
        let conv2d = Array::from_f32_slice(
            &(0..2 * 4 * 3 * 3)
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
            &[2, 4, 3, 3],
        )
        .unwrap();
        let converted = transpose_component_weights(
            ComponentLayout::Upsampler,
            Weights::from([("upsampler.0.weight".into(), conv2d)]),
            &Stream::cpu().unwrap(),
        )
        .unwrap();
        assert_eq!(converted["upsampler.0.weight"].shape(), vec![2, 3, 3, 4]);

        // ConvTranspose1d ups: published (in,out,k) → MLX (out,k,in).
        let ups = Array::from_f32_slice(
            &(0..8 * 4 * 3).map(|value| value as f32).collect::<Vec<_>>(),
            &[8, 4, 3],
        )
        .unwrap();
        let converted = transpose_component_weights(
            ComponentLayout::AudioVae,
            Weights::from([("vocoder.vocoder.ups.0.weight".into(), ups)]),
            &Stream::cpu().unwrap(),
        )
        .unwrap();
        assert_eq!(
            converted["vocoder.vocoder.ups.0.weight"].shape(),
            vec![4, 3, 8]
        );
        // Mistaken regular-conv pass left (in,k,out) — still recoverable.
        let mistaken = Array::from_f32_slice(
            &(0..8 * 3 * 4).map(|value| value as f32).collect::<Vec<_>>(),
            &[8, 3, 4],
        )
        .unwrap();
        let converted = transpose_component_weights(
            ComponentLayout::AudioVae,
            Weights::from([("vocoder.vocoder.ups.0.weight".into(), mistaken)]),
            &Stream::cpu().unwrap(),
        )
        .unwrap();
        assert_eq!(
            converted["vocoder.vocoder.ups.0.weight"].shape(),
            vec![4, 3, 8]
        );
    }

    #[test]
    fn bundle_config_keeps_checkpoint_shapes_and_splits_vocoder() {
        let config = build_bundle_config(
            &json!({"num_layers": 48}),
            &json!({"decoder_blocks": [[2, 2, 2]], "patch_size": 4, "latent_log_var": "one"}),
            &json!({"sampling_rate": 48000, "mel_bins": 16, "upsampling_ratios": [2, 2]}),
            &json!({"in_channels": 128}),
            &json!({"text_encoder_config": {"hidden_size": 3840}, "architectures": ["Gemma4"]}),
        )
        .unwrap();
        assert_eq!(config["transformer"]["in_dims"], 128);
        assert_eq!(config["vae"]["patch_size"], 4);
        assert_eq!(config["audio_vae"]["mel_bins"], 16);
        assert_eq!(config["vocoder"]["sampling_rate"], 48000);
        assert!(config["vocoder"].get("mel_bins").is_none());
        assert_eq!(config["gemma"]["hidden_size"], 3840);

        // Published audio-vae metadata nests {vocoder,bwe} under "vocoder".
        let nested = build_bundle_config(
            &json!({"num_layers": 48}),
            &json!({"decoder_blocks": [[2, 2, 2]], "patch_size": 4, "latent_log_var": "one"}),
            &json!({
                "audio_vae": {"mel_bins": 16},
                "vocoder": {
                    "vocoder": {"upsample_rates": [5, 2]},
                    "bwe": {"input_sampling_rate": 16000, "output_sampling_rate": 48000}
                }
            }),
            &json!({"in_channels": 128}),
            &json!({"text_config": {"hidden_size": 3840}}),
        )
        .unwrap();
        assert_eq!(
            nested["vocoder"]["vocoder"]["upsample_rates"],
            json!([5, 2])
        );
        assert_eq!(nested["vocoder"]["bwe"]["input_sampling_rate"], 16000);
    }
}
