mod graph;
mod png;
mod text;
mod transformer;
mod weights;

pub use png::encode_rgb;
pub use text::{write_chain_bundle, write_qwen3_bundle};
pub use transformer::Flux2Denoise;
pub use weights::{weight_layout, Flux2Config};

use crate::common::ops::normal_latent;
use crate::runtime::{preview_conv, GgmlContext, TYPE_F32, TYPE_Q8_0};
use crate::weights::{
    assert_gguf_bundle, matmul_type, quantize_q8_0, type_block, write_f32_tensors,
    write_typed_tensors, F32Tensor, GgufWeights, TypedTensor, WeightError,
};
use std::collections::HashMap;
use std::path::Path;

pub fn execute(
    ctx: &GgmlContext,
    bundle: &Path,
    prompt: &str,
    steps: u32,
    width: u32,
    height: u32,
    guidance: f32,
    seed: Option<i64>,
    scheduler: Option<&str>,
    on_progress: impl FnMut(u32, u32),
) -> Result<crate::families::DitImageRun, WeightError> {
    match scheduler.unwrap_or("flow_match_euler_discrete") {
        "flow_match_euler" | "flow_match_euler_discrete" => {}
        other => {
            return Err(WeightError::NotBuilt(format!(
                "scheduler {other} is not built"
            )))
        }
    }
    let files = assert_gguf_bundle(bundle)?;
    let loaded = GgufWeights::open_parts(&files.transformer)?;
    let config = config_from_weights(&loaded)?;
    let scale = config.vae_scale.max(1);
    if width % scale != 0 || height % scale != 0 {
        return Err(WeightError::NotBuilt(format!(
            "size {width}x{height} is not divisible by vae scale {scale}"
        )));
    }
    if let Some(max_edge) = manifest_max_edge(bundle)? {
        if width > max_edge || height > max_edge {
            return Err(WeightError::NotBuilt(format!(
                "this bundle accepts edges up to {max_edge}, got {width}x{height}"
            )));
        }
    }
    let grid_w = (width / scale) as i64;
    let grid_h = (height / scale) as i64;
    let seq_img = grid_w * grid_h;
    let tokenizer = std::fs::read_to_string(bundle.join("tokenizer.json")).map_err(|err| {
        WeightError::Open {
            path: bundle.join("tokenizer.json").display().to_string(),
            detail: err.to_string(),
        }
    })?;
    let (txt, seq_txt) = text_tokens(
        ctx,
        &files.text_encoder,
        &tokenizer,
        prompt,
        config.joint_attention_dim,
    )?;
    let mut types = HashMap::new();
    let mut stored = Vec::new();
    for (name, ne0, ne1) in weight_layout(config) {
        let ggml_type = loaded.ggml_type_of(&name)?;
        check_stored_type(&name, ggml_type, ne0)?;
        types.insert(name.clone(), ggml_type);
        stored.push((name.clone(), loaded.raw_exact(&name, &[ne0, ne1])?));
    }
    let mut denoise =
        transformer::Flux2Denoise::build_typed(&ctx, config, grid_w, grid_h, seq_txt, &types)?;
    for (name, raw) in &stored {
        denoise.load_bytes(name, raw)?;
    }
    let latent = match seed {
        Some(seed) => normal_latent(seed, denoise.latent_len()),
        None => vec![0.1f32; denoise.latent_len()],
    };
    let latents = denoise.denoise(
        latent,
        &txt,
        steps,
        width,
        height,
        config.requires_sigma_shift,
        guidance,
        on_progress,
    )?;
    let png = decode_preview(
        ctx,
        &files.vae,
        &latents,
        config.in_channels as usize,
        seq_img as usize,
        grid_w as u32,
        grid_h as u32,
        width,
        height,
    )?;
    Ok(crate::families::DitImageRun {
        latents,
        graph_computes: denoise.graph_computes(),
        preview_png: png,
    })
}

fn config_from_weights(weights: &GgufWeights) -> Result<Flux2Config, WeightError> {
    let x = weights.shape("x_embedder.weight")?;
    let context = weights.shape("context_embedder.weight")?;
    let freq = weights.shape("time_guidance_embed.linear_1.weight")?;
    let in_channels = x[0];
    let dim = x[1];
    if context[1] != dim || freq[1] != dim {
        return Err(WeightError::Shape {
            name: "x_embedder.weight".into(),
            expected: vec![in_channels, dim],
            got: vec![context[0], context[1], freq[0], freq[1]],
        });
    }
    let norm = weights.shape("transformer_blocks.0.attn.norm_q.weight")?;
    let head = norm[0];
    if head < 1 || norm[1] != 1 || dim % head != 0 {
        return Err(WeightError::Shape {
            name: "transformer_blocks.0.attn.norm_q.weight".into(),
            expected: vec![head.max(1), 1],
            got: vec![norm[0], norm[1], dim],
        });
    }
    let heads = dim / head;
    let layers = count_blocks(weights, "transformer_blocks", "attn.to_q.weight");
    let singles = count_blocks(
        weights,
        "single_transformer_blocks",
        "attn.to_qkv_mlp_proj.weight",
    );
    if layers < 1 || singles < 1 {
        return Err(WeightError::MissingTensor {
            path: "transformer".into(),
            name: "transformer_blocks.0.attn.to_q.weight".into(),
        });
    }
    Ok(Flux2Config {
        in_channels,
        inner_dim: dim,
        num_heads: heads,
        attn_head_dim: head,
        num_layers: layers,
        num_single_layers: singles,
        joint_attention_dim: context[0],
        freq_dim: freq[0],
        mlp_ratio: 3,
        requires_sigma_shift: true,
        vae_scale: 16,
        rope_theta: 2000.0,
    })
}

fn check_stored_type(name: &str, ggml_type: u32, ne0: i64) -> Result<(), WeightError> {
    if name.contains(".norm_") {
        if ggml_type != TYPE_F32 {
            return Err(WeightError::NotBuilt(format!(
                "{name} is ggml type {ggml_type}. Norm weights stay F32 inside a quantized GGUF"
            )));
        }
        return Ok(());
    }
    if !matmul_type(ggml_type) {
        return Err(WeightError::Type {
            name: name.to_string(),
            ggml_type,
        });
    }
    let block = type_block(ggml_type);
    if block > 1 && (ne0 % block != 0) {
        return Err(WeightError::NotBuilt(format!(
            "{name} length {ne0} is not divisible by ggml block {block}"
        )));
    }
    Ok(())
}

fn count_blocks(weights: &GgufWeights, prefix: &str, suffix: &str) -> i64 {
    let mut n = 0i64;
    loop {
        let name = format!("{prefix}.{n}.{suffix}");
        if !weights.contains(&name) {
            break;
        }
        n += 1;
    }
    n
}

fn text_tokens(
    ctx: &GgmlContext,
    paths: &[std::path::PathBuf],
    tokenizer_json: &str,
    prompt: &str,
    joint: i64,
) -> Result<(Vec<f32>, i64), WeightError> {
    let text = GgufWeights::open_parts(paths)?;
    if text.names().any(|name| name.starts_with("blk.")) {
        return Err(WeightError::NotBuilt(
            "text encoder tensor names are blk.*; this graph reads layers.*".into(),
        ));
    }
    if text.contains("layers.0.self_attn.q_proj.weight") {
        return text::encode_prompt(ctx, paths, tokenizer_json, prompt, joint);
    }
    let shape = text.shape("token_embed.weight")?;
    if shape[0] != joint || shape[1] != 1 {
        return Err(WeightError::NotBuilt(format!(
            "text encoder graph is not built (token_embed.weight is {:?} not [{joint}, 1])",
            shape
        )));
    }
    Ok((text.f32_exact("token_embed.weight", &[joint, 1])?, 1))
}

fn manifest_max_edge(bundle: &Path) -> Result<Option<u32>, WeightError> {
    let path = bundle.join("bundle.manifest.json");
    let text = std::fs::read_to_string(&path).map_err(|err| WeightError::Open {
        path: path.display().to_string(),
        detail: err.to_string(),
    })?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|err| WeightError::NotBuilt(format!("bundle.manifest.json is not json: {err}")))?;
    Ok(value
        .get("max_edge")
        .and_then(|item| item.as_u64())
        .map(|edge| edge as u32))
}

fn decode_preview(
    ctx: &GgmlContext,
    paths: &[std::path::PathBuf],
    latent: &[f32],
    in_ch: usize,
    seq: usize,
    grid_w: u32,
    grid_h: u32,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, WeightError> {
    let vae = GgufWeights::open_parts(paths)?;
    let names: Vec<String> = vae.names().map(|name| name.to_string()).collect();
    if names.len() != 1 || names.first().map(String::as_str) != Some("decoder.conv.weight") {
        return Err(WeightError::NotBuilt("vae graph is not built".into()));
    }
    let weight = vae.f32_exact("decoder.conv.weight", &[in_ch as i64, 3])?;
    if latent.len() != in_ch * seq {
        return Err(WeightError::Shape {
            name: "latent".into(),
            expected: vec![in_ch as i64, seq as i64],
            got: vec![latent.len() as i64],
        });
    }
    let planar = preview_conv(
        ctx,
        latent,
        &weight,
        in_ch as i64,
        grid_w as i64,
        grid_h as i64,
    )?;
    let rgb = planar_rgb(&planar, grid_w, grid_h);
    let rgb = nearest_expand(&rgb, grid_w, grid_h, width, height);
    Ok(png::encode_rgb(width, height, &rgb))
}

fn planar_rgb(planar: &[f32], grid_w: u32, grid_h: u32) -> Vec<u8> {
    let width = grid_w as usize;
    let height = grid_h as usize;
    let plane = width * height;
    let mut rgb = vec![0u8; plane * 3];
    for y in 0..height {
        for x in 0..width {
            for channel in 0..3 {
                let v = planar
                    .get(x + y * width + channel * plane)
                    .copied()
                    .unwrap_or(0.0);
                rgb[(y * width + x) * 3 + channel] = (v.clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
    }
    rgb
}

fn nearest_expand(src: &[u8], grid_w: u32, grid_h: u32, width: u32, height: u32) -> Vec<u8> {
    if grid_w == width && grid_h == height {
        return src.to_vec();
    }
    let mut out = vec![0u8; (width as usize) * (height as usize) * 3];
    for y in 0..height {
        let sy = ((y as u64) * (grid_h as u64) / (height as u64)) as usize;
        for x in 0..width {
            let sx = ((x as u64) * (grid_w as u64) / (width as u64)) as usize;
            let from = (sy * grid_w as usize + sx) * 3;
            let to = (y as usize * width as usize + x as usize) * 3;
            out[to..to + 3].copy_from_slice(&src[from..from + 3]);
        }
    }
    out
}

fn write_transformer(
    dir: &Path,
    config: Flux2Config,
    values: fn(&str, i64, i64) -> Vec<f32>,
) -> Result<(), WeightError> {
    let layout = weight_layout(config);
    let owned: Vec<(String, i64, i64, Vec<f32>)> = layout
        .iter()
        .map(|(name, ne0, ne1)| {
            let data = values(name, *ne0, *ne1);
            (name.clone(), *ne0, *ne1, data)
        })
        .collect();
    let tensors: Vec<F32Tensor<'_>> = owned
        .iter()
        .map(|(name, ne0, ne1, data)| F32Tensor {
            name,
            ne0: *ne0,
            ne1: *ne1,
            data,
        })
        .collect();
    write_f32_tensors(&dir.join("transformer.gguf"), &tensors)
}

pub(super) fn rewrite_chain_transformer(
    dir: &Path,
    config: Flux2Config,
) -> Result<(), WeightError> {
    write_transformer(dir, config, weights::chain_dit_values)
}

/// Tiny bf16-tier bundle whose DiT file uses Flux2 parameter names.
pub fn write_smoke_bundle(dir: &Path, dim: i64) -> Result<(), WeightError> {
    write_config_bundle(dir, Flux2Config::from_dim(dim))
}

/// Same files as [`write_smoke_bundle`], with an explicit head split.
pub fn write_config_bundle(dir: &Path, config: Flux2Config) -> Result<(), WeightError> {
    std::fs::create_dir_all(dir).map_err(|err| WeightError::Open {
        path: dir.display().to_string(),
        detail: err.to_string(),
    })?;
    write_transformer(dir, config, weights::smoke_values)?;
    let joint = config.joint_attention_dim;
    let embed = vec![0.1f32; joint as usize];
    write_f32_tensors(
        &dir.join("text_encoder.gguf"),
        &[F32Tensor {
            name: "token_embed.weight",
            ne0: joint,
            ne1: 1,
            data: &embed,
        }],
    )?;
    let in_ch = config.in_channels as usize;
    let mut conv = vec![0f32; in_ch * 3];
    for i in 0..in_ch.min(3) {
        conv[i + i * in_ch] = 1.0;
    }
    write_f32_tensors(
        &dir.join("vae.gguf"),
        &[F32Tensor {
            name: "decoder.conv.weight",
            ne0: config.in_channels,
            ne1: 3,
            data: &conv,
        }],
    )?;
    std::fs::write(dir.join("tokenizer.json"), "{}\n").map_err(|err| WeightError::Open {
        path: dir.display().to_string(),
        detail: err.to_string(),
    })?;
    let manifest = r#"{
  "format": "gguf",
  "quant": "bf16",
  "files": {
    "transformer": "transformer.gguf",
    "text_encoder": "text_encoder.gguf",
    "vae": "vae.gguf"
  }
}
"#;
    std::fs::write(dir.join("bundle.manifest.json"), manifest).map_err(|err| {
        WeightError::Open {
            path: dir.display().to_string(),
            detail: err.to_string(),
        }
    })?;
    Ok(())
}

/// Matmul weights are Q8_0. Norm weights stay F32. `ne0` of every matmul row must be a multiple of 32.
pub fn write_q8_bundle(dir: &Path, config: Flux2Config) -> Result<(), WeightError> {
    std::fs::create_dir_all(dir).map_err(|err| WeightError::Open {
        path: dir.display().to_string(),
        detail: err.to_string(),
    })?;
    let layout = weight_layout(config);
    let mut owned = Vec::new();
    for (name, ne0, ne1) in &layout {
        let values = weights::smoke_values(name, *ne0, *ne1);
        if name.contains(".norm_") {
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            owned.push((name.clone(), *ne0, *ne1, TYPE_F32, bytes));
        } else {
            let bytes = quantize_q8_0(*ne0, *ne1, &values)?;
            owned.push((name.clone(), *ne0, *ne1, TYPE_Q8_0, bytes));
        }
    }
    let tensors: Vec<TypedTensor<'_>> = owned
        .iter()
        .map(|(name, ne0, ne1, ggml_type, data)| TypedTensor {
            name,
            ne0: *ne0,
            ne1: *ne1,
            ggml_type: *ggml_type,
            data,
        })
        .collect();
    write_typed_tensors(&dir.join("transformer.gguf"), &tensors)?;
    let joint = config.joint_attention_dim;
    let embed = vec![0.1f32; joint as usize];
    write_f32_tensors(
        &dir.join("text_encoder.gguf"),
        &[F32Tensor {
            name: "token_embed.weight",
            ne0: joint,
            ne1: 1,
            data: &embed,
        }],
    )?;
    let in_ch = config.in_channels as usize;
    let mut conv = vec![0f32; in_ch * 3];
    for i in 0..in_ch.min(3) {
        conv[i + i * in_ch] = 1.0;
    }
    write_f32_tensors(
        &dir.join("vae.gguf"),
        &[F32Tensor {
            name: "decoder.conv.weight",
            ne0: config.in_channels,
            ne1: 3,
            data: &conv,
        }],
    )?;
    std::fs::write(dir.join("tokenizer.json"), "{}\n").map_err(|err| WeightError::Open {
        path: dir.display().to_string(),
        detail: err.to_string(),
    })?;
    let manifest = r#"{
  "format": "gguf",
  "quant": "q8_0",
  "files": {
    "transformer": "transformer.gguf",
    "text_encoder": "text_encoder.gguf",
    "vae": "vae.gguf"
  }
}
"#;
    std::fs::write(dir.join("bundle.manifest.json"), manifest).map_err(|err| {
        WeightError::Open {
            path: dir.display().to_string(),
            detail: err.to_string(),
        }
    })?;
    Ok(())
}
