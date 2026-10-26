mod png;
mod transformer;
mod weights;

pub use transformer::Flux2Denoise;
pub use weights::{weight_layout, Flux2Config};

use crate::weights::{assert_gguf_bundle, write_f32_tensors, F32Tensor, GgufWeights, WeightError};
use std::path::Path;

pub fn run(
    bundle: &Path,
    steps: u32,
    width: u32,
    height: u32,
    guidance: f32,
    backend: crate::runtime::BackendKind,
    on_progress: impl FnMut(u32, u32),
) -> Result<crate::families::DitImageRun, WeightError> {
    let files = assert_gguf_bundle(bundle)?;
    let loaded = GgufWeights::open_parts(&files.transformer)?;
    let config = config_from_weights(&loaded)?;
    if config.num_heads != 1 {
        return Err(WeightError::NotBuilt(format!(
            "flux2 attention for {} heads is not built",
            config.num_heads
        )));
    }
    let scale = config.vae_scale.max(1);
    if width % scale != 0 || height % scale != 0 {
        return Err(WeightError::NotBuilt(format!(
            "size {width}x{height} is not divisible by vae scale {scale}"
        )));
    }
    let grid_w = (width / scale) as i64;
    let grid_h = (height / scale) as i64;
    let seq_img = grid_w * grid_h;
    let ctx = crate::runtime::GgmlContext::new(backend)?;
    let mut denoise = transformer::Flux2Denoise::build(&ctx, config, seq_img)?;
    for (name, ne0, ne1) in weight_layout(config) {
        let data = loaded.f32_exact(&name, &[ne0, ne1])?;
        denoise.load_f32(&name, &data)?;
    }
    let txt = text_tokens(&files.text_encoder, config.joint_attention_dim)?;
    let latent = vec![0.1f32; denoise.latent_len()];
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
        &files.vae,
        &latents,
        config.in_channels as usize,
        seq_img as usize,
        grid_w as u32,
        grid_h as u32,
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
    let (heads, head) = if dim >= 128 && dim % 128 == 0 {
        (dim / 128, 128)
    } else {
        (1, dim)
    };
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

fn text_tokens(paths: &[std::path::PathBuf], joint: i64) -> Result<Vec<f32>, WeightError> {
    let text = GgufWeights::open_parts(paths)?;
    for name in text.names() {
        if name.contains("layers.") || name.starts_with("blk.") {
            return Err(WeightError::NotBuilt(
                "text encoder graph is not built".into(),
            ));
        }
    }
    let shape = text.shape("token_embed.weight")?;
    if shape[0] != joint || shape[1] != 1 {
        return Err(WeightError::NotBuilt(format!(
            "text encoder graph is not built (token_embed.weight is {:?} not [{joint}, 1])",
            shape
        )));
    }
    text.f32_exact("token_embed.weight", &[joint, 1])
}

fn decode_preview(
    paths: &[std::path::PathBuf],
    latent: &[f32],
    in_ch: usize,
    seq: usize,
    grid_w: u32,
    grid_h: u32,
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
    let mut rgb = vec![0u8; seq * 3];
    for token in 0..seq {
        for channel in 0..3 {
            let mut acc = 0f32;
            for i in 0..in_ch {
                acc += latent[i + token * in_ch] * weight[i + channel * in_ch];
            }
            rgb[token * 3 + channel] = (acc.clamp(0.0, 1.0) * 255.0) as u8;
        }
    }
    Ok(png::encode_rgb(grid_w, grid_h, &rgb))
}

/// Tiny bf16-tier bundle whose DiT file uses Flux2 parameter names.
pub fn write_smoke_bundle(dir: &Path, dim: i64) -> Result<(), WeightError> {
    std::fs::create_dir_all(dir).map_err(|err| WeightError::Open {
        path: dir.display().to_string(),
        detail: err.to_string(),
    })?;
    let config = Flux2Config::from_dim(dim);
    let layout = weight_layout(config);
    let owned: Vec<(String, i64, i64, Vec<f32>)> = layout
        .iter()
        .map(|(name, ne0, ne1)| {
            let data = weights::smoke_values(name, *ne0, *ne1);
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
    write_f32_tensors(&dir.join("transformer.gguf"), &tensors)?;
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
