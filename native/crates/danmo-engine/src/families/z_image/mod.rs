//! Z-Image create path. Tensor names are the unsloth / SD.cpp GGUF names.
//!
//! The block math follows Tongyi and `main`'s `transformer_mlx.py`. SD.cpp agrees
//! on those ops. Two SD.cpp details are exact-arithmetic identities and are not
//! applied: the flash-attention `1/128` rescale, and the w2 `1/128` rescale.
//! Sequence order is image then caption. SD.cpp concatenates caption then image;
//! full attention makes the image tokens match.

mod geom;
mod graph;
#[cfg(test)]
mod reference;
mod transformer;
mod weights;

use crate::common::ops::{flow_match_shift_sigmas, linear_sigmas, normal_latent};
use crate::runtime::{preview_conv, GgmlContext, TYPE_F32};
use crate::weights::{assert_gguf_bundle, matmul_type, type_block, GgufWeights, WeightError};
#[cfg(test)]
use crate::weights::{write_f32_tensors, F32Tensor};
use std::collections::HashMap;
use std::path::Path;

use transformer::ZImageDenoise;
#[cfg(test)]
use weights::ZImageConfig;
use weights::{config_from_weights, vector_param, weight_layout};

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
    scheduler_shift: Option<f32>,
    empirical_mu: bool,
    on_progress: impl FnMut(u32, u32),
) -> Result<crate::families::DitImageRun, WeightError> {
    let _ = prompt;
    let files = assert_gguf_bundle(bundle)?;
    let loaded = GgufWeights::open_parts(&files.transformer)?;
    let config = config_from_weights(&loaded)?;
    let txt = text_tokens(&files.text_encoder, config.cap_dim)?;
    let cap_real = 1;
    let grid = geom::token_grid(width, height, cap_real, config.vae_scale)?;
    let mut types = HashMap::new();
    let mut stored = Vec::new();
    for (name, ne0, ne1) in weight_layout(config) {
        let ggml_type = loaded.ggml_type_of(&name)?;
        check_stored_type(&name, ggml_type, ne0)?;
        types.insert(name.clone(), ggml_type);
        stored.push((name.clone(), loaded.raw_exact(&name, &[ne0, ne1])?));
    }
    let mut denoise = ZImageDenoise::build(ctx, config, width, height, cap_real, &types)?;
    for (name, raw) in &stored {
        denoise.load_bytes(name, raw)?;
    }
    let latent = match seed {
        Some(seed) => normal_latent(seed, denoise.latent_len()),
        None => vec![0.1f32; denoise.latent_len()],
    };
    let sigmas = schedule(
        config,
        steps,
        width,
        height,
        scheduler,
        scheduler_shift,
        empirical_mu,
    )?;
    let latents = denoise.denoise(latent, &txt, &sigmas, guidance, on_progress)?;
    let png = decode_preview(
        ctx,
        &files.vae,
        &latents,
        config.in_channels,
        grid.latent_w,
        grid.latent_h,
        width,
        height,
    )?;
    Ok(crate::families::DitImageRun {
        latents,
        graph_computes: denoise.graph_computes(),
        preview_png: png,
    })
}

fn schedule(
    config: weights::ZImageConfig,
    steps: u32,
    width: u32,
    height: u32,
    scheduler: Option<&str>,
    scheduler_shift: Option<f32>,
    empirical_mu: bool,
) -> Result<Vec<f32>, WeightError> {
    match scheduler.unwrap_or("linear") {
        "linear" => {
            if empirical_mu {
                return Err(WeightError::NotBuilt(
                    "empirical mu schedule is not built".into(),
                ));
            }
            Ok(linear_sigmas(
                steps,
                width,
                height,
                config.requires_sigma_shift,
            ))
        }
        "flow_match_euler" => {
            if empirical_mu {
                return Err(WeightError::NotBuilt(
                    "empirical mu schedule is not built".into(),
                ));
            }
            let shift = scheduler_shift.unwrap_or(6.0);
            if !shift.is_finite() || shift <= 0.0 {
                return Err(WeightError::NotBuilt(format!(
                    "scheduler shift {shift} is not a positive finite value"
                )));
            }
            Ok(flow_match_shift_sigmas(steps, shift))
        }
        "flow_match_euler_discrete" => Err(WeightError::NotBuilt(
            "empirical mu schedule is not built".into(),
        )),
        other => Err(WeightError::NotBuilt(format!(
            "scheduler {other} is not built"
        ))),
    }
}

fn text_tokens(paths: &[std::path::PathBuf], cap_dim: i64) -> Result<Vec<f32>, WeightError> {
    let text = GgufWeights::open_parts(paths)?;
    if text.names().any(|name| name.starts_with("blk."))
        || text.contains("layers.0.self_attn.q_proj.weight")
    {
        return Err(WeightError::NotBuilt(
            "text encoder graph is not built".into(),
        ));
    }
    let shape = text.shape("token_embed.weight")?;
    if shape[0] != cap_dim || shape[1] != 1 {
        return Err(WeightError::NotBuilt(format!(
            "text encoder graph is not built (token_embed.weight is {:?} not [{cap_dim}, 1])",
            shape
        )));
    }
    text.f32_exact("token_embed.weight", &[cap_dim, 1])
}

fn check_stored_type(name: &str, ggml_type: u32, ne0: i64) -> Result<(), WeightError> {
    if vector_param(name) {
        if ggml_type != TYPE_F32 {
            return Err(WeightError::NotBuilt(format!(
                "{name} is ggml type {ggml_type}. Norm, bias, and pad tokens stay F32"
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
    if block > 1 && ne0 % block != 0 {
        return Err(WeightError::NotBuilt(format!(
            "{name} length {ne0} is not divisible by ggml block {block}"
        )));
    }
    Ok(())
}

fn decode_preview(
    ctx: &GgmlContext,
    paths: &[std::path::PathBuf],
    latent: &[f32],
    in_ch: i64,
    grid_w: i64,
    grid_h: i64,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, WeightError> {
    let vae = GgufWeights::open_parts(paths)?;
    let names: Vec<String> = vae.names().map(|name| name.to_string()).collect();
    if names.len() != 1 || names.first().map(String::as_str) != Some("decoder.conv.weight") {
        return Err(WeightError::NotBuilt("vae graph is not built".into()));
    }
    let weight = vae.f32_exact("decoder.conv.weight", &[in_ch, 3])?;
    let seq = grid_w * grid_h;
    if latent.len() as i64 != in_ch * seq {
        return Err(WeightError::Shape {
            name: "latent".into(),
            expected: vec![in_ch, seq],
            got: vec![latent.len() as i64],
        });
    }
    let planar = repack_channel_fast(latent, grid_w, grid_h, in_ch);
    let decoded = preview_conv(ctx, &planar, &weight, in_ch, grid_w, grid_h)?;
    let rgb = planar_rgb(&decoded, grid_w as u32, grid_h as u32);
    let rgb = nearest_expand(&rgb, grid_w as u32, grid_h as u32, width, height);
    Ok(super::flux2::encode_rgb(width, height, &rgb))
}

/// `[W, H, C]` (x fastest) to `[C, W*H]` (channel fastest) for `preview_conv`.
fn repack_channel_fast(src: &[f32], width: i64, height: i64, channels: i64) -> Vec<f32> {
    let width = width as usize;
    let height = height as usize;
    let channels = channels as usize;
    let mut out = vec![0f32; src.len()];
    for channel in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let from = x + y * width + channel * width * height;
                let to = channel + (x + y * width) * channels;
                out[to] = src[from];
            }
        }
    }
    out
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

#[cfg(test)]
pub fn write_smoke_bundle(dir: &Path) -> Result<(), WeightError> {
    let config = ZImageConfig::smoke();
    std::fs::create_dir_all(dir).map_err(|err| WeightError::Open {
        path: dir.display().to_string(),
        detail: err.to_string(),
    })?;
    let layout = weight_layout(config);
    let owned: Vec<(String, i64, i64, Vec<f32>)> = layout
        .iter()
        .map(|(name, ne0, ne1)| {
            (
                name.clone(),
                *ne0,
                *ne1,
                weights::smoke_values(name, *ne0, *ne1),
            )
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
    let cap = config.cap_dim as usize;
    let embed: Vec<f32> = (0..cap).map(|i| ((i % 5) as f32 - 2.0) * 0.05).collect();
    write_f32_tensors(
        &dir.join("text_encoder.gguf"),
        &[F32Tensor {
            name: "token_embed.weight",
            ne0: config.cap_dim,
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

#[cfg(test)]
mod tests {
    use super::write_smoke_bundle;
    use crate::image_create;
    use crate::runtime::BackendKind;
    use crate::weights::{write_f32_tensors, F32Tensor};
    use danmo_contracts::ImageGenerationRequest;

    #[test]
    fn same_seed_repeats_and_flow_match_runs() {
        let dir = scratch("z-image-seed");
        write_smoke_bundle(&dir).unwrap();
        let mut request = request();
        request.guidance = Some(0.0);
        request.seed = Some(7);
        request.scheduler = Some("flow_match_euler".into());
        request.scheduler_shift = Some(6.0);
        let first = image_create(&request, "z_image", &dir, BackendKind::Cpu, |_, _| ()).unwrap();
        let second = image_create(&request, "z_image", &dir, BackendKind::Cpu, |_, _| ()).unwrap();
        assert_eq!(first.latents, second.latents);
        request.seed = Some(8);
        let other = image_create(&request, "z_image", &dir, BackendKind::Cpu, |_, _| ()).unwrap();
        assert_ne!(first.latents, other.latents);
        request.scheduler = Some("unipc".into());
        let err = match image_create(&request, "z_image", &dir, BackendKind::Cpu, |_, _| ()) {
            Err(err) => err,
            Ok(_) => panic!("unknown scheduler was accepted"),
        };
        assert!(err.to_string().contains("not built"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn smoke_create_is_one_forward_and_a_png() {
        let dir = scratch("z-image-create");
        write_smoke_bundle(&dir).unwrap();
        let mut request = request();
        request.guidance = Some(0.0);
        let out = image_create(&request, "z_image", &dir, BackendKind::Cpu, |_, _| ()).unwrap();
        assert_eq!(out.graph_computes, 1);
        assert!(
            out.latents.iter().all(|v| v.is_finite()),
            "{:?}",
            &out.latents[..8]
        );
        assert_eq!(&out.preview_png[..8], b"\x89PNG\r\n\x1a\n");
        let width = u32::from_be_bytes(out.preview_png[16..20].try_into().unwrap());
        let height = u32::from_be_bytes(out.preview_png[20..24].try_into().unwrap());
        assert_eq!((width, height), (64, 64));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn qwen_text_tower_fails_loud() {
        let dir = scratch("z-image-qwen");
        write_smoke_bundle(&dir).unwrap();
        let row = vec![0.1f32; 8];
        write_f32_tensors(
            &dir.join("text_encoder.gguf"),
            &[F32Tensor {
                name: "layers.0.self_attn.q_proj.weight",
                ne0: 8,
                ne1: 1,
                data: &row,
            }],
        )
        .unwrap();
        let mut request = request();
        request.guidance = Some(0.0);
        let err = match image_create(&request, "z_image", &dir, BackendKind::Cpu, |_, _| ()) {
            Err(err) => err,
            Ok(_) => panic!("qwen text tower was accepted"),
        };
        assert!(
            err.to_string().contains("text encoder graph is not built"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn request() -> ImageGenerationRequest {
        ImageGenerationRequest {
            model: "z-image".into(),
            title: String::new(),
            prompt: "a cat".into(),
            negative_prompt: String::new(),
            size: "64x64".into(),
            n: 1,
            steps: Some(1),
            guidance: Some(0.0),
            seed: Some(1),
            scheduler: None,
            scheduler_shift: None,
            use_empirical_mu: false,
            priority: "normal".into(),
        }
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "danmo-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }
}
