//! Native MLX conversion and integrity checks for an official LTX 2.5 bundle.
//!
//! The registry install hook calls this family module after all published
//! source files have landed. No Python converter or second runtime is used.

use super::weights::{
    build_bundle_config, extract_feature_extractor, split_transformer, transpose_component_weights,
    ComponentLayout, Weights,
};
use danmo_mlx::{save_safetensors, Array, Dtype, Stream};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const TRANSFORMER_SOURCE: &str =
    "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors";
pub const TEXT_ENCODER_SOURCE: &str = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors";
pub const VIDEO_VAE_SOURCE: &str = "vae/ltx-2.5-video-vae-conv-bf16.safetensors";
pub const AUDIO_VAE_SOURCE: &str = "vae/ltx-2.5-audio-vae-bf16.safetensors";
pub const UPSAMPLER_SOURCE: &str =
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors";

const REQUIRED_BUNDLE_FILES: &[&str] = &[
    "bundle_config.json",
    "transformer.safetensors",
    "connector.safetensors",
    "video_vae.safetensors",
    "audio_vae.safetensors",
    "upsampler.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder/tokenizer.json",
];
const SOURCE_FILES: &[&str] = &[
    TRANSFORMER_SOURCE,
    TEXT_ENCODER_SOURCE,
    VIDEO_VAE_SOURCE,
    AUDIO_VAE_SOURCE,
    UPSAMPLER_SOURCE,
];
const SOURCE_DIRS: &[&str] = &[
    "diffusion_models",
    "text_encoders",
    "vae",
    "latent_upscale_models",
    "loras",
    "model_patches",
];

pub use super::config::Quantization;

fn quantization_config(quant: Quantization) -> Value {
    json!({
        "group_size": quant.group_size,
        "bits": quant.bits,
        "mode": "affine",
        "class": "QuantizedLinear",
    })
}

pub fn ingest_from_hook(
    bundle_root: &Path,
    hook_spec: &Value,
    cancel: Box<dyn Fn() -> bool + Send>,
    log: Box<dyn Fn(&str) + Send>,
) -> Result<(), String> {
    let source = hook_spec
        .get("source")
        .and_then(Value::as_str)
        .unwrap_or(".");
    let source = PathBuf::from(source);
    let source = if source.is_absolute() {
        source
    } else {
        bundle_root.join(source)
    };
    let quantize = match hook_spec.get("quantize") {
        None | Some(Value::Null) => None,
        Some(value) => {
            let bits = value
                .as_i64()
                .ok_or_else(|| "LTX 2.5 install quantize must be 4 or 8".to_string())?;
            if bits != 4 && bits != 8 {
                return Err(format!(
                    "LTX 2.5 install quantize must be 4 or 8, got {bits}"
                ));
            }
            Some(bits as i32)
        }
    };
    let group_size = hook_spec
        .get("group_size")
        .and_then(Value::as_i64)
        .unwrap_or(32);
    if group_size <= 0 || group_size > i32::MAX as i64 {
        return Err(format!(
            "LTX 2.5 quantization group_size {group_size} is invalid"
        ));
    }
    let cleanup_source = hook_spec
        .get("cleanup_source")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    // Conversion runs on the process MLX executor thread; no private stream.
    let target = bundle_root.to_path_buf();
    let quantization = quantize.map(|bits| Quantization {
        bits,
        group_size: group_size as i32,
    });
    crate::runtime::mlx_executor()
        .map_err(|error| format!("LTX 2.5 MLX install executor: {error}"))?
        .execute("ltx25 ingest", move |_, stream, _cache, _events| {
            Ok((|| {
                // Layout repair does not need published sources (post-cleanup).
                ensure_installed_conv_layouts(&target, stream, &log)?;
                ingest(&source, &target, quantization, stream, &cancel, &log)?;
                if cleanup_source {
                    cleanup_published_sources(&target, &log)?;
                }
                Ok(())
            })())
        })
        .map_err(|error| format!("LTX 2.5 MLX install executor: {error}"))?
}

pub fn bundle_is_ready(bundle_root: &Path, stream: &Stream) -> bool {
    validate_bundle(bundle_root, stream).is_ok()
}

pub fn validate_bundle(bundle_root: &Path, _stream: &Stream) -> Result<(), String> {
    let missing: Vec<&str> = REQUIRED_BUNDLE_FILES
        .iter()
        .copied()
        .filter(|relative| !bundle_root.join(relative).is_file())
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "LTX 2.5 MLX bundle is incomplete at {}: missing {}",
            bundle_root.display(),
            missing.join(", ")
        ));
    }
    // Fail loud if conv shards were saved in published OIK/OIHW/OIDHW layout.
    require_conv_weights_channel_last(&bundle_root.join("upsampler.safetensors"))?;
    require_conv_weights_channel_last(&bundle_root.join("audio_vae.safetensors"))?;
    require_conv_weights_channel_last(&bundle_root.join("video_vae.safetensors"))?;
    let quantize_path = bundle_root.join("quantize_config.json");
    if !quantize_path.is_file() {
        return Ok(());
    }
    let text = std::fs::read_to_string(&quantize_path)
        .map_err(|error| format!("{}: {error}", quantize_path.display()))?;
    let config: Value = serde_json::from_str(&text)
        .map_err(|error| format!("{} is not valid JSON: {error}", quantize_path.display()))?;
    let Some(bits) = config.pointer("/quantization/bits").and_then(Value::as_i64) else {
        return Ok(());
    };
    if bits != 4 && bits != 8 {
        return Ok(());
    }
    quantized_intact(&bundle_root.join("transformer.safetensors"), bits as i32)?;
    let text_config_path = bundle_root.join("text_encoder/config.json");
    let text_config = std::fs::read_to_string(&text_config_path)
        .map_err(|error| format!("{}: {error}", text_config_path.display()))?;
    let text_config: Value = serde_json::from_str(&text_config)
        .map_err(|error| format!("{text_config_path:?} is not valid JSON: {error}"))?;
    if matches!(
        text_config
            .pointer("/quantization/bits")
            .and_then(Value::as_i64),
        Some(4 | 8)
    ) {
        quantized_intact(
            &bundle_root.join("text_encoder/model.safetensors"),
            bits as i32,
        )?;
    }
    Ok(())
}

fn quantized_intact(path: &Path, bits: i32) -> Result<(), String> {
    let quantization = crate::weights::probe_affine_quantization(path)
        .map_err(|error| format!("quantized bundle check {}: {error}", path.display()))?
        .ok_or_else(|| format!("{} has no packed scales tensors", path.display()))?;
    if quantization.bits != bits {
        return Err(format!(
            "{} packs {} bits, quantize_config.json declares {bits}",
            path.display(),
            quantization.bits
        ));
    }
    Ok(())
}

fn conv_weight_is_channel_last(key: &str, shape: &[usize]) -> bool {
    let dims: Vec<i32> = shape.iter().map(|&dim| dim as i32).collect();
    super::weights::conv_weight_is_channel_last(key, &dims)
}

fn read_safetensors_shapes(path: &Path) -> Result<BTreeMap<String, Vec<usize>>, String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .map_err(|error| format!("read header length {}: {error}", path.display()))?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    if header_len == 0 || header_len > 64 * 1024 * 1024 {
        return Err(format!(
            "{} safetensors header length {header_len} is invalid",
            path.display()
        ));
    }
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)
        .map_err(|error| format!("read header {}: {error}", path.display()))?;
    let header: Value = serde_json::from_slice(&header)
        .map_err(|error| format!("{} header is not JSON: {error}", path.display()))?;
    let mut shapes = BTreeMap::new();
    for (key, value) in header.as_object().into_iter().flatten() {
        if key == "__metadata__" {
            continue;
        }
        let Some(shape) = value.get("shape").and_then(Value::as_array) else {
            continue;
        };
        let dims: Result<Vec<usize>, String> = shape
            .iter()
            .map(|dim| {
                dim.as_u64().map(|value| value as usize).ok_or_else(|| {
                    format!("{}:{key} shape entry is not an integer", path.display())
                })
            })
            .collect();
        shapes.insert(key.clone(), dims?);
    }
    Ok(shapes)
}

fn require_conv_weights_channel_last(path: &Path) -> Result<(), String> {
    let shapes = read_safetensors_shapes(path)?;
    let mut bad = Vec::new();
    for (name, shape) in &shapes {
        if !name.ends_with(".weight") {
            continue;
        }
        if !conv_weight_is_channel_last(name, shape) {
            bad.push(format!("{name} {shape:?}"));
        }
    }
    if bad.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{} still uses published OIK/OIHW/OIDHW conv layout (expected channel-last): {}",
        path.display(),
        bad.join("; ")
    ))
}

fn ensure_component_channel_last(
    path: &Path,
    component: ComponentLayout,
    stream: &Stream,
    log: &dyn Fn(&str),
) -> Result<(), String> {
    if !path.is_file() || require_conv_weights_channel_last(path).is_ok() {
        return Ok(());
    }
    // At most two passes: a prior whole-file re-layout may have left 4D/5D
    // weights one step past channel-last (back at published OIHW); the second
    // pass brings them home. Per-tensor transpose is idempotent on OK tensors.
    for pass in 1..=2 {
        if require_conv_weights_channel_last(path).is_ok() {
            return Ok(());
        }
        log(&format!(
            "Re-layout LTX 2.5 {component:?} to channel-last (pass {pass}): {}",
            path.display()
        ));
        let io = io_stream()?;
        let (weights, _) = crate::weights::read_component(path, &io)
            .map_err(|error| format!("load {}: {error}", path.display()))?;
        let weights = transpose_component_weights(component, weights, stream)?;
        save_weights(path, weights, None, stream)?;
    }
    require_conv_weights_channel_last(path)
}

fn ensure_installed_conv_layouts(
    target: &Path,
    stream: &Stream,
    log: &dyn Fn(&str),
) -> Result<(), String> {
    ensure_component_channel_last(
        &target.join("upsampler.safetensors"),
        ComponentLayout::Upsampler,
        stream,
        log,
    )?;
    ensure_component_channel_last(
        &target.join("audio_vae.safetensors"),
        ComponentLayout::AudioVae,
        stream,
        log,
    )?;
    ensure_component_channel_last(
        &target.join("video_vae.safetensors"),
        ComponentLayout::VideoVae,
        stream,
        log,
    )?;
    Ok(())
}

pub fn ingest(
    source: &Path,
    target: &Path,
    quantization: Option<Quantization>,
    stream: &Stream,
    cancel: &dyn Fn() -> bool,
    log: &dyn Fn(&str),
) -> Result<(), String> {
    ensure_installed_conv_layouts(target, stream, log)?;
    if bundle_is_ready(target, stream) {
        log(&format!(
            "LTX 2.5 bundle already converted at {}; skipping ingest",
            target.display()
        ));
        return Ok(());
    }
    for relative in SOURCE_FILES {
        let path = source.join(relative);
        if !path.is_file() {
            return Err(format!(
                "LTX 2.5 source file missing: {}\nDownload Lightricks/LTX-2.5 first (ModelScope mirror or gated Hugging Face).",
                path.display()
            ));
        }
    }
    std::fs::create_dir_all(target.join("text_encoder"))
        .map_err(|error| format!("create {}: {error}", target.display()))?;
    check_cancel(cancel)?;

    let transformer_ready = target.join("transformer.safetensors").is_file()
        && target.join("connector.safetensors").is_file();
    let text_ready = target.join("text_encoder/model.safetensors").is_file()
        && target.join("text_encoder/tokenizer.json").is_file()
        && target.join("text_encoder/config.json").is_file();
    let video_path = target.join("video_vae.safetensors");
    let audio_path = target.join("audio_vae.safetensors");
    let upsampler_path = target.join("upsampler.safetensors");
    let video_ready =
        video_path.is_file() && require_conv_weights_channel_last(&video_path).is_ok();
    let audio_ready =
        audio_path.is_file() && require_conv_weights_channel_last(&audio_path).is_ok();
    let upsampler_ready =
        upsampler_path.is_file() && require_conv_weights_channel_last(&upsampler_path).is_ok();

    // Configs come from safetensors headers (no tensor load). Weight shards skip
    // when a previous ingest attempt already wrote them.
    let transformer_config = read_checkpoint_config(&source.join(TRANSFORMER_SOURCE))?;
    let gemma_hf_config = read_gemma_hf_config(&source.join(TEXT_ENCODER_SOURCE))?;
    let video_vae_config = read_checkpoint_config(&source.join(VIDEO_VAE_SOURCE))?;
    let audio_vae_config = read_checkpoint_config(&source.join(AUDIO_VAE_SOURCE))?;
    let upsampler_config = read_checkpoint_config(&source.join(UPSAMPLER_SOURCE))?;

    if !transformer_ready || !text_ready {
        log("Converting LTX 2.5 transformer / connector / Gemma text encoder");
        let (transformer, _) = load_checkpoint(&source.join(TRANSFORMER_SOURCE), stream)?;
        let (dit, mut connector) = split_transformer(transformer)?;
        let (mut gemma, _) = load_gemma_checkpoint(&source.join(TEXT_ENCODER_SOURCE), stream)?;
        connector.extend(extract_feature_extractor(&mut gemma)?);
        if !transformer_ready {
            save_weights(
                &target.join("transformer.safetensors"),
                dit,
                quantization,
                stream,
            )?;
            save_weights(
                &target.join("connector.safetensors"),
                connector,
                None,
                stream,
            )?;
        }
        if !text_ready {
            convert_text_encoder(&mut gemma, &gemma_hf_config, quantization, target, stream)?;
        }
        check_cancel(cancel)?;
    } else {
        log("Reusing existing transformer/connector/text_encoder shards");
    }

    if !video_ready {
        log("Converting LTX 2.5 video VAE");
        let (video_vae, _) = load_checkpoint(&source.join(VIDEO_VAE_SOURCE), stream)?;
        let video_vae = transpose_component_weights(ComponentLayout::VideoVae, video_vae, stream)?;
        save_weights(
            &target.join("video_vae.safetensors"),
            video_vae,
            None,
            stream,
        )?;
        check_cancel(cancel)?;
    }

    if !audio_ready {
        log("Converting LTX 2.5 audio VAE (includes vocoder.* keys)");
        let (audio_vae, _) = load_checkpoint(&source.join(AUDIO_VAE_SOURCE), stream)?;
        let audio_vae = transpose_component_weights(ComponentLayout::AudioVae, audio_vae, stream)?;
        save_weights(
            &target.join("audio_vae.safetensors"),
            audio_vae,
            None,
            stream,
        )?;
        check_cancel(cancel)?;
    }

    if !upsampler_ready {
        log("Converting LTX 2.5 spatial upsampler");
        let (upsampler, _) = load_checkpoint(&source.join(UPSAMPLER_SOURCE), stream)?;
        let upsampler = transpose_component_weights(ComponentLayout::Upsampler, upsampler, stream)?;
        save_weights(
            &target.join("upsampler.safetensors"),
            upsampler,
            None,
            stream,
        )?;
        check_cancel(cancel)?;
    }

    let bundle_config = build_bundle_config(
        &transformer_config,
        &video_vae_config,
        &audio_vae_config,
        &upsampler_config,
        &gemma_hf_config,
    )?;
    write_json(&target.join("bundle_config.json"), &bundle_config)?;
    if let Some(quantization) = quantization {
        write_json(
            &target.join("quantize_config.json"),
            &json!({"quantization": quantization_config(quantization)}),
        )?;
    }
    validate_bundle(target, stream)?;
    log(&format!("LTX 2.5 bundle ready at {}", target.display()));
    Ok(())
}

fn cleanup_published_sources(bundle_root: &Path, log: &dyn Fn(&str)) -> Result<(), String> {
    for relative in SOURCE_FILES {
        let path = bundle_root.join(relative);
        if path.is_file() {
            std::fs::remove_file(&path)
                .map_err(|error| format!("cleanup {}: {error}", path.display()))?;
        }
    }
    for dir in SOURCE_DIRS {
        let path = bundle_root.join(dir);
        if path.is_dir() {
            let _ = std::fs::remove_dir(&path);
        }
    }
    log(&format!(
        "LTX 2.5 source files cleaned from {}",
        bundle_root.display()
    ));
    Ok(())
}

fn io_stream() -> Result<Stream, String> {
    // Safetensors Load/Save must stay on CPU — GPU eval of Load is unimplemented.
    Stream::cpu().map_err(|error| format!("LTX 2.5 CPU IO stream: {error}"))
}

fn read_safetensors_metadata(path: &Path) -> Result<BTreeMap<String, String>, String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .map_err(|error| format!("read header length {}: {error}", path.display()))?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    if header_len == 0 || header_len > 64 * 1024 * 1024 {
        return Err(format!(
            "{} safetensors header length {header_len} is invalid",
            path.display()
        ));
    }
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)
        .map_err(|error| format!("read header {}: {error}", path.display()))?;
    let header: Value = serde_json::from_slice(&header)
        .map_err(|error| format!("{} header is not JSON: {error}", path.display()))?;
    let Some(meta) = header.get("__metadata__").and_then(Value::as_object) else {
        return Ok(BTreeMap::new());
    };
    Ok(meta
        .iter()
        .filter_map(|(key, value)| value.as_str().map(|text| (key.clone(), text.to_string())))
        .collect())
}

fn read_checkpoint_config(path: &Path) -> Result<Value, String> {
    let meta = read_safetensors_metadata(path)?;
    let config = meta
        .get("config")
        .ok_or_else(|| format!("{} lacks checkpoint config metadata", path.display()))?;
    serde_json::from_str(config)
        .map_err(|error| format!("{} config metadata is not JSON: {error}", path.display()))
}

fn read_gemma_hf_config(path: &Path) -> Result<Value, String> {
    let meta = read_safetensors_metadata(path)?;
    if let Some(raw) = meta.get("gemma_config") {
        return serde_json::from_str(raw)
            .map_err(|error| format!("{} gemma_config is not JSON: {error}", path.display()));
    }
    let config = read_checkpoint_config(path)?;
    if let Some(nested) = config.get("text_encoder_config").cloned() {
        return Ok(nested);
    }
    Ok(config)
}

fn load_checkpoint(path: &Path, _stream: &Stream) -> Result<(Weights, Value), String> {
    let io = io_stream()?;
    let (weights, config) = crate::weights::read_component(path, &io)
        .map_err(|error| format!("load {}: {error}", path.display()))?;
    let config =
        config.ok_or_else(|| format!("{} lacks checkpoint config metadata", path.display()))?;
    let config = serde_json::from_str(&config)
        .map_err(|error| format!("{} config metadata is not JSON: {error}", path.display()))?;
    Ok((weights, config))
}

fn load_gemma_checkpoint(path: &Path, _stream: &Stream) -> Result<(Weights, Value), String> {
    let io = io_stream()?;
    let (weights, meta) =
        crate::weights::read_component_meta(path, &io, &["gemma_config", "config"])
            .map_err(|error| format!("load {}: {error}", path.display()))?;
    if let Some(raw) = meta.get("gemma_config") {
        let config = serde_json::from_str(raw)
            .map_err(|error| format!("{} gemma_config is not JSON: {error}", path.display()))?;
        return Ok((weights, config));
    }
    let config = meta
        .get("config")
        .ok_or_else(|| format!("{} lacks gemma_config / config metadata", path.display()))?;
    let parsed: Value = serde_json::from_str(config)
        .map_err(|error| format!("{} config metadata is not JSON: {error}", path.display()))?;
    if let Some(nested) = parsed.get("text_encoder_config").cloned() {
        return Ok((weights, nested));
    }
    Ok((weights, parsed))
}

fn convert_text_encoder(
    gemma: &mut Weights,
    gemma_hf_config: &Value,
    quantization: Option<Quantization>,
    target: &Path,
    stream: &Stream,
) -> Result<(), String> {
    let text_config = gemma_hf_config
        .get("text_config")
        .or_else(|| gemma_hf_config.get("text_encoder_config"))
        .cloned()
        .ok_or_else(|| "Gemma HF config lacks text_config".to_string())?;
    let vocab_size = text_config
        .get("vocab_size")
        .and_then(Value::as_u64)
        .unwrap_or(262_144);

    let tokenizer_json = gemma
        .remove("tokenizer_json")
        .ok_or_else(|| "LTX text encoder checkpoint has no packed tokenizer_json".to_string())?;
    let tokenizer_bytes = tokenizer_json
        .to_raw_bytes(stream)
        .map_err(|error| format!("read tokenizer_json: {error}"))?;
    std::fs::write(target.join("text_encoder/tokenizer.json"), tokenizer_bytes)
        .map_err(|error| format!("write text_encoder/tokenizer.json: {error}"))?;

    // Optional sidecars when present.
    for (key, filename) in [
        ("hf_asset__tokenizer_config.json", "tokenizer_config.json"),
        ("hf_asset__generation_config.json", "generation_config.json"),
        ("hf_asset__tokenizer_config", "tokenizer_config.json"),
        ("hf_asset__generation_config", "generation_config.json"),
    ] {
        if let Some(tensor) = gemma.remove(key) {
            let bytes = tensor
                .to_raw_bytes(stream)
                .map_err(|error| format!("read {key}: {error}"))?;
            let _ = std::fs::write(target.join("text_encoder").join(filename), bytes);
        }
    }

    // Drop multimodal / projector leftovers that do not belong in the LM shard.
    let drop_prefixes = [
        "hf_asset__",
        "text_embedding_projection.",
        "vision_model.",
        "multi_modal_projector.",
        "audio_projector.",
        "tokenizer_json",
    ];
    let discard: Vec<String> = gemma
        .keys()
        .filter(|key| drop_prefixes.iter().any(|prefix| key.starts_with(prefix)))
        .cloned()
        .collect();
    for key in discard {
        gemma.remove(&key);
    }

    let mut lm = BTreeMap::new();
    for (key, tensor) in std::mem::take(gemma) {
        if let Some(rest) = key.strip_prefix("model.") {
            lm.insert(format!("language_model.model.{rest}"), tensor);
        } else if key.starts_with("language_model.") {
            lm.insert(key, tensor);
        }
        // Unknown top-level keys are ignored (packed sidecars already handled).
    }
    if lm.is_empty() {
        return Err(
            "No Gemma language-model weights found in the LTX text encoder checkpoint".into(),
        );
    }

    let mut bundle_config = json!({
        "model_type": "gemma4",
        "vocab_size": vocab_size,
        "text_config": text_config,
    });
    if let Some(quantization) = quantization {
        bundle_config["quantization"] = quantization_config(quantization);
    }
    save_weights(
        &target.join("text_encoder/model.safetensors"),
        lm,
        quantization,
        stream,
    )?;
    write_json(&target.join("text_encoder/config.json"), &bundle_config)?;
    Ok(())
}

fn save_weights(
    path: &Path,
    weights: Weights,
    quantization: Option<Quantization>,
    stream: &Stream,
) -> Result<(), String> {
    if weights.is_empty() {
        return Err(format!(
            "refusing to write empty weight file {}",
            path.display()
        ));
    }
    let io = io_stream()?;
    // Quantize/cast may run on the executor GPU stream; flatten onto CPU before
    // mlx_save_safetensors (GPU Load/eval is unimplemented in mlx-c).
    let mut converted = BTreeMap::new();
    for (name, tensor) in weights {
        let skip_norm = name.ends_with(".q_norm.weight") || name.ends_with(".k_norm.weight");
        let quantize = quantization.is_some_and(|config| {
            !skip_norm
                && name.ends_with(".weight")
                && tensor.shape().len() == 2
                && tensor.shape()[1] > 0
                && tensor.shape()[1] % config.group_size == 0
        });
        if let Some(config) = quantization.filter(|_| quantize) {
            let packed =
                crate::weights::pack_affine(&tensor, config.group_size, config.bits, stream)
                    .map_err(|error| format!("quantize {}:{name}: {error}", path.display()))?;
            converted.insert(name.clone(), materialize_on_cpu(&packed.weight, &io)?);
            converted.insert(
                format!("{}.scales", name.trim_end_matches(".weight")),
                materialize_on_cpu(&packed.scales, &io)?,
            );
            converted.insert(
                format!("{}.biases", name.trim_end_matches(".weight")),
                materialize_on_cpu(&packed.biases, &io)?,
            );
        } else if tensor.dtype().is_floating() && tensor.dtype() != Dtype::BFloat16 {
            let cast = tensor
                .astype(Dtype::BFloat16, stream)
                .map_err(|error| format!("cast {} to BF16: {error}", path.display()))?;
            converted.insert(name, materialize_on_cpu(&cast, &io)?);
        } else {
            converted.insert(name, materialize_on_cpu(&tensor, &io)?);
        }
    }
    let held: Vec<&Array> = converted.values().collect();
    Array::eval_all(&held).map_err(|error| format!("eval {}: {error}", path.display()))?;
    let tensors: Vec<(&str, &Array)> = converted
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    let quantization_level = quantization.map(|config| config.bits.to_string());
    let metadata: Vec<(&str, &str)> = if let Some(level) = quantization_level.as_deref() {
        vec![("quantization_level", level)]
    } else {
        vec![("format", "pt")]
    };
    save_safetensors(path, &tensors, &metadata)
        .map_err(|error| format!("write {}: {error}", path.display()))
}

fn materialize_on_cpu(tensor: &Array, cpu: &Stream) -> Result<Array, String> {
    tensor
        .contiguous(cpu)
        .map_err(|error| format!("CPU materialize for safetensors save: {error}"))
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("encode {}: {error}", path.display()))?;
    std::fs::write(path, bytes).map_err(|error| format!("write {}: {error}", path.display()))
}

fn check_cancel(cancel: &dyn Fn() -> bool) -> Result<(), String> {
    if cancel() {
        Err("LTX 2.5 ingest cancelled".into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        bundle_is_ready, ingest, validate_bundle, Quantization, AUDIO_VAE_SOURCE,
        TEXT_ENCODER_SOURCE, TRANSFORMER_SOURCE, UPSAMPLER_SOURCE, VIDEO_VAE_SOURCE,
    };
    use danmo_mlx::{save_safetensors, Array, Dtype, SafeTensors, Stream};
    use serde_json::Value;
    use std::path::{Path, PathBuf};

    fn scratch() -> PathBuf {
        std::env::temp_dir().join(format!(
            "danmo-ltx25-ingest-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn checkpoint(path: &Path, config: Value, tensors: &[(&str, &Array)]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let config = config.to_string();
        save_safetensors(path, tensors, &[("config", &config)]).unwrap();
    }

    fn write_source(root: &Path) {
        let matrix = Array::from_f32_slice(
            &(0..128)
                .map(|index| index as f32 / 128.0)
                .collect::<Vec<_>>(),
            &[2, 64],
        )
        .unwrap();
        let registers = Array::from_f32_slice(&[0.1, 0.2], &[2]).unwrap();
        checkpoint(
            &root.join(TRANSFORMER_SOURCE),
            serde_json::json!({"transformer": {"num_layers": 1}}),
            &[
                ("model.diffusion_model.patchify_proj.weight", &matrix),
                (
                    "model.diffusion_model.video_embeddings_connector.learnable_registers",
                    &registers,
                ),
                (
                    "model.diffusion_model.audio_embeddings_connector.learnable_registers",
                    &registers,
                ),
            ],
        );
        let tokens = Array::from_u8_slice(br#"{"model":{}}"#, &[12]).unwrap();
        let gemma_config = serde_json::json!({
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "text_config": {"hidden_size": 8, "num_hidden_layers": 1, "vocab_size": 32}
        })
        .to_string();
        std::fs::create_dir_all(root.join(TEXT_ENCODER_SOURCE).parent().unwrap()).unwrap();
        save_safetensors(
            &root.join(TEXT_ENCODER_SOURCE),
            &[
                (
                    "text_embedding_projection.video_aggregate_embed.weight",
                    &matrix,
                ),
                ("model.layers.0.mlp.down_proj.weight", &matrix),
                ("tokenizer_json", &tokens),
            ],
            &[("gemma_config", &gemma_config)],
        )
        .unwrap();
        let video = Array::from_f32_slice(&[0.5; 16], &[1, 2, 2, 2, 2]).unwrap();
        checkpoint(
            &root.join(VIDEO_VAE_SOURCE),
            serde_json::json!({"vae": {"decoder_blocks": [[2, 2, 2]], "patch_size": 4}}),
            &[("decoder.conv.weight", &video)],
        );
        let audio = Array::from_f32_slice(&[0.25; 16], &[1, 2, 2, 2, 2]).unwrap();
        let vocoder = Array::from_f32_slice(&[0.5; 6], &[1, 2, 3]).unwrap();
        checkpoint(
            &root.join(AUDIO_VAE_SOURCE),
            serde_json::json!({
                "audio_vae": {"mel_bins": 16},
                "vocoder": {"sampling_rate": 48000},
                "bwe": {"sampling_rate": 48000}
            }),
            &[("decoder.weight", &audio), ("vocoder.res.weight", &vocoder)],
        );
        let upsampler = Array::from_f32_slice(&[0.5; 6], &[1, 2, 3]).unwrap();
        checkpoint(
            &root.join(UPSAMPLER_SOURCE),
            serde_json::json!({"in_channels": 128}),
            &[("conv.weight", &upsampler)],
        );
    }

    #[test]
    fn native_ingest_writes_a_complete_quantized_bundle() {
        let root = scratch();
        let source = root.join("source");
        let target = root.join("target");
        write_source(&source);
        let stream = Stream::cpu().unwrap();
        ingest(
            &source,
            &target,
            Some(Quantization {
                bits: 4,
                group_size: 32,
            }),
            &stream,
            &|| false,
            &|_| {},
        )
        .unwrap();
        assert!(bundle_is_ready(&target, &stream));
        assert_eq!(
            std::fs::read(target.join("text_encoder/tokenizer.json")).unwrap(),
            br#"{"model":{}}"#
        );
        let config: Value = serde_json::from_slice(
            &std::fs::read(target.join("text_encoder/config.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(config["quantization"]["bits"], 4);
        let transformer =
            SafeTensors::load(&target.join("transformer.safetensors"), &stream).unwrap();
        let packed = transformer.get("patchify_proj.weight").unwrap().unwrap();
        assert_eq!(packed.dtype(), Dtype::UInt32);
        assert_eq!(packed.shape(), vec![2, 8]);
        assert_eq!(
            transformer
                .metadata("quantization_level")
                .unwrap()
                .as_deref(),
            Some("4")
        );
        let connector = SafeTensors::load(&target.join("connector.safetensors"), &stream).unwrap();
        assert!(connector
            .get("feature_extractor.video_aggregate_embed.weight")
            .unwrap()
            .is_some());
        assert!(connector
            .get("video_connector.learnable_registers")
            .unwrap()
            .is_some());

        let dense = Array::from_f32_slice(&[0.0; 128], &[2, 64]).unwrap();
        save_safetensors(
            &target.join("transformer.safetensors"),
            &[("patchify_proj.weight", &dense)],
            &[("quantization_level", "4")],
        )
        .unwrap();
        let error = validate_bundle(&target, &stream).err().unwrap();
        assert!(error.contains("scales"), "{error}");
        drop(transformer);
        drop(connector);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn missing_bundle_files_fail_with_their_names() {
        let root = scratch();
        let error = validate_bundle(&root, &Stream::cpu().unwrap())
            .err()
            .unwrap();
        assert!(error.contains("bundle_config.json"), "{error}");
        assert!(error.contains("transformer.safetensors"), "{error}");
    }

    #[test]
    fn upsampler_oidhw_is_relayouted_to_channel_last() {
        let root = scratch();
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("upsampler.safetensors");
        // Published OIDHW: [O, I, kD, kH, kW]
        let weight = Array::from_f32_slice(
            &(0..2 * 4 * 3 * 3 * 3).map(|v| v as f32).collect::<Vec<_>>(),
            &[2, 4, 3, 3, 3],
        )
        .unwrap();
        save_safetensors(&path, &[("initial_conv.weight", &weight)], &[]).unwrap();
        let err = super::require_conv_weights_channel_last(&path)
            .err()
            .unwrap();
        assert!(
            err.contains("OIDHW") || err.contains("OIHW") || err.contains("OIK"),
            "{err}"
        );
        let stream = Stream::cpu().unwrap();
        super::ensure_component_channel_last(
            &path,
            super::ComponentLayout::Upsampler,
            &stream,
            &|_| {},
        )
        .unwrap();
        super::require_conv_weights_channel_last(&path).unwrap();
        let shapes = super::read_safetensors_shapes(&path).unwrap();
        assert_eq!(shapes["initial_conv.weight"], vec![2, 3, 3, 3, 4]);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    #[ignore = "workspace repair: set DANQING_TEST_LTX25_BUNDLE to a converted bundle root"]
    fn repair_installed_conv_layouts() {
        let root = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_LTX25_BUNDLE").expect("DANQING_TEST_LTX25_BUNDLE"),
        );
        let stream = Stream::cpu().unwrap();
        super::ensure_installed_conv_layouts(&root, &stream, &|message| eprintln!("{message}"))
            .unwrap();
        super::require_conv_weights_channel_last(&root.join("upsampler.safetensors")).unwrap();
        super::require_conv_weights_channel_last(&root.join("audio_vae.safetensors")).unwrap();
        super::require_conv_weights_channel_last(&root.join("video_vae.safetensors")).unwrap();
    }
}
