//! Z-Image create path. Tensor names are the unsloth / SD.cpp GGUF names.
//!
//! The block math follows Tongyi and `main`'s `transformer_mlx.py`. SD.cpp agrees
//! on those ops. Two SD.cpp details are exact-arithmetic identities and are not
//! applied: the flash-attention `1/128` rescale, and the w2 `1/128` rescale.
//! Sequence order is image then caption. SD.cpp concatenates caption then image;
//! full attention makes the image tokens match.
//!
//! The caption is the shared Qwen3 tower's penultimate hidden state. Thinking
//! stays on. A one-vector `token_embed` fixture does not read the prompt.
//! The pixel decoder is the shared AutoencoderKL graph with this family's
//! spec. A one-tensor `decoder.conv.weight` file stays the 1×1 preview.

mod geom;
mod graph;
#[cfg(test)]
mod reference;
mod transformer;
mod vae;
mod weights;

use crate::common::ops::{
    empirical_mu_sigmas, flow_match_shift_sigmas, linear_sigmas, normal_latent,
};
use crate::runtime::{GgmlContext, TYPE_F32};
use crate::weights::{
    assert_gguf_bundle, load_lora_factors, matmul_type, type_block, GgufWeights, LoraFile,
    WeightError,
};
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
    negative: &str,
    loras: &[LoraFile],
    on_progress: impl FnMut(u32, u32),
) -> Result<crate::families::DitImageRun, WeightError> {
    let files = assert_gguf_bundle(bundle)?;
    let loaded = GgufWeights::open_parts(&files.transformer)?;
    let config = config_from_weights(&loaded)?;
    let (txt, cap_real) = encode_caption(ctx, bundle, &files.text_encoder, prompt, config.cap_dim)?;
    let uncond = encode_negative(
        ctx,
        bundle,
        &files.text_encoder,
        negative,
        config.cap_dim,
        cap_real,
        guidance,
    )?;
    let grid = geom::token_grid(width, height, cap_real, config.vae_scale)?;
    let layout = weight_layout(config);
    let factors = load_lora_factors(loras, &layout)?;
    let mut types = HashMap::new();
    for (name, ne0, _) in &layout {
        let ggml_type = loaded.ggml_type_of(name)?;
        check_stored_type(name, ggml_type, *ne0)?;
        types.insert(name.clone(), ggml_type);
    }
    let mut denoise =
        ZImageDenoise::build_with_loras(ctx, config, width, height, cap_real, &types, &factors)?;
    for (name, ne0, ne1) in &layout {
        let raw = loaded.raw_exact(name, &[*ne0, *ne1])?;
        denoise.load_bytes(name, &raw)?;
    }
    for (index, factor) in factors.iter().enumerate() {
        denoise.load_f32(&format!("lora.{index}.a"), &factor.a)?;
        denoise.load_f32(&format!("lora.{index}.b"), &factor.b)?;
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
    let latents = denoise.denoise(
        latent,
        &txt,
        uncond.as_deref(),
        &sigmas,
        guidance,
        on_progress,
    )?;
    if let Some(detail) = crate::common::ops::non_finite("z-image latent", &latents) {
        return Err(WeightError::NotBuilt(detail));
    }
    let png = vae::decode(
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
        // LinearScheduler keeps the resolution shift. Empirical mu belongs to FlowMatchEuler.
        "linear" => Ok(linear_sigmas(
            steps,
            width,
            height,
            config.requires_sigma_shift,
        )),
        "flow_match_euler" | "flow_match_euler_discrete" => {
            if empirical_mu {
                let scale = u32::try_from(config.vae_scale).unwrap_or(0);
                if scale == 0 || width % scale != 0 || height % scale != 0 {
                    return Err(WeightError::NotBuilt(format!(
                        "size {width}x{height} is not divisible by vae scale {}",
                        config.vae_scale
                    )));
                }
                let seq = (width / scale) * (height / scale);
                return Ok(empirical_mu_sigmas(steps, seq));
            }
            let shift = scheduler_shift.unwrap_or(6.0);
            if !shift.is_finite() || shift <= 0.0 {
                return Err(WeightError::NotBuilt(format!(
                    "scheduler shift {shift} is not a positive finite value"
                )));
            }
            Ok(flow_match_shift_sigmas(steps, shift))
        }
        other => Err(WeightError::NotBuilt(format!(
            "scheduler {other} is not built"
        ))),
    }
}

fn encode_caption(
    ctx: &GgmlContext,
    bundle: &Path,
    paths: &[std::path::PathBuf],
    prompt: &str,
    cap_dim: i64,
) -> Result<(Vec<f32>, i64), WeightError> {
    let text = GgufWeights::open_parts(paths)?;
    if text.names().any(|name| name.starts_with("blk.")) {
        return Err(WeightError::NotBuilt(
            "text encoder tensor names are blk.*; this graph reads layers.*".into(),
        ));
    }
    if crate::common::qwen3::tower_present(&text) {
        let tokenizer = crate::common::qwen3::load_tokenizer(bundle)?;
        let loaded = crate::common::qwen3::Loaded::open(paths)?;
        let config = loaded.config();
        if config.layers < 2 {
            return Err(WeightError::NotBuilt(
                "penultimate text encoder needs at least 2 layers".into(),
            ));
        }
        if config.hidden != cap_dim {
            return Err(WeightError::NotBuilt(format!(
                "text encoder hidden {} is not the caption width {cap_dim}",
                config.hidden
            )));
        }
        let layer = config.layers - 1;
        return loaded.encode(ctx, &tokenizer, prompt, &[layer], true, cap_dim);
    }
    let shape = text.shape("token_embed.weight")?;
    if shape[0] != cap_dim || shape[1] != 1 {
        return Err(WeightError::NotBuilt(format!(
            "text encoder graph is not built (token_embed.weight is {:?} not [{cap_dim}, 1])",
            shape
        )));
    }
    let _ = prompt;
    Ok((text.f32_exact("token_embed.weight", &[cap_dim, 1])?, 1))
}

fn encode_negative(
    ctx: &GgmlContext,
    bundle: &Path,
    paths: &[std::path::PathBuf],
    negative: &str,
    cap_dim: i64,
    cap_real: i64,
    guidance: f32,
) -> Result<Option<Vec<f32>>, WeightError> {
    if negative.trim().is_empty() {
        return Ok(None);
    }
    if guidance.abs() < f32::EPSILON {
        return Err(WeightError::NotBuilt(
            "negative prompt needs a guidance other than 0".into(),
        ));
    }
    let (encoded, seq) = encode_caption(ctx, bundle, paths, negative, cap_dim)?;
    if seq != cap_real {
        return Err(WeightError::NotBuilt(format!(
            "negative prompt tokenized to {seq} tokens, prompt has {cap_real}"
        )));
    }
    Ok(Some(encoded))
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

#[cfg(test)]
fn write_smoke_text(dir: &Path, cap_dim: i64) -> Result<(), WeightError> {
    let enc = crate::common::qwen3::Qwen3Config::smoke();
    if enc.hidden != cap_dim {
        return Err(WeightError::NotBuilt(format!(
            "smoke text hidden {} is not caption {cap_dim}",
            enc.hidden
        )));
    }
    let owned: Vec<(String, i64, i64, Vec<f32>)> = crate::common::qwen3::encoder_layout(enc)
        .into_iter()
        .map(|(name, ne0, ne1)| {
            let n = (ne0 as usize) * (ne1 as usize);
            let data = if name.ends_with("norm.weight") || name.ends_with("layernorm.weight") {
                vec![1.0; n]
            } else {
                (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.02).collect()
            };
            (name, ne0, ne1, data)
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
    write_f32_tensors(&dir.join("text_encoder.gguf"), &tensors)
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
    write_smoke_text(dir, config.cap_dim)?;
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
    std::fs::write(
        dir.join("tokenizer.json"),
        "{\"vocab\":{\"a\":0,\"cat\":1,\"dog\":2}}\n",
    )
    .map_err(|err| WeightError::Open {
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
    use super::weights::ZImageConfig;
    use super::write_smoke_bundle;
    use super::{weight_layout, ZImageDenoise};
    use crate::image_create;
    use crate::runtime::{BackendKind, GgmlContext, TYPE_F32};
    use crate::weights::{write_f32_tensors, F32Tensor};
    use danmo_contracts::ImageGenerationRequest;
    use std::collections::HashMap;

    /// 30 layers and 2 refiner pairs. Same node count as the published DiT.
    #[test]
    fn published_depth_reserves() {
        let config = ZImageConfig {
            dim: 8,
            heads: 1,
            head_dim: 8,
            axes: [2, 2, 4],
            layers: 30,
            refiners: 2,
            in_channels: 4,
            cap_dim: 8,
            ffn: 16,
            mid: 32,
            freq: 8,
            adaln: 8,
            rope_theta: 256.0,
            t_scale: 1000.0,
            vae_scale: 8,
            patch: 2,
            requires_sigma_shift: true,
        };
        let mut types = HashMap::new();
        for (name, _, _) in weight_layout(config) {
            types.insert(name, TYPE_F32);
        }
        let ctx = GgmlContext::new(BackendKind::Cpu).expect("cpu");
        let graph = ZImageDenoise::build(&ctx, config, 64, 64, 4, &types)
            .expect("published-depth z-image graph reserves");
        drop(graph);
    }

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
    fn discrete_matches_static_shift_and_empirical_mu_decodes() {
        let dir = scratch("z-image-schedule");
        write_smoke_bundle(&dir).unwrap();
        let mut euler = request();
        euler.guidance = Some(0.0);
        euler.steps = Some(4);
        euler.scheduler = Some("flow_match_euler".into());
        euler.scheduler_shift = Some(6.0);
        let base = image_create(&euler, "z_image", &dir, BackendKind::Cpu, |_, _| ()).unwrap();
        let mut discrete = euler.clone();
        discrete.scheduler = Some("flow_match_euler_discrete".into());
        let alias = image_create(&discrete, "z_image", &dir, BackendKind::Cpu, |_, _| ()).unwrap();
        assert_eq!(base.latents, alias.latents);
        discrete.use_empirical_mu = true;
        let shifted =
            image_create(&discrete, "z_image", &dir, BackendKind::Cpu, |_, _| ()).unwrap();
        assert_ne!(base.latents, shifted.latents);
        assert!(shifted.latents.iter().all(|v| v.is_finite()));
        assert_eq!(&shifted.preview_png[..8], b"\x89PNG\r\n\x1a\n");
        let mut linear = request();
        linear.guidance = Some(0.0);
        linear.steps = Some(4);
        linear.scheduler = Some("linear".into());
        let plain = image_create(&linear, "z_image", &dir, BackendKind::Cpu, |_, _| ()).unwrap();
        linear.use_empirical_mu = true;
        let flagged = image_create(&linear, "z_image", &dir, BackendKind::Cpu, |_, _| ()).unwrap();
        assert_eq!(plain.latents, flagged.latents);
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
    fn prompt_changes_the_latent_and_a_same_length_negative_is_uncond() {
        let dir = scratch("z-image-prompt");
        write_smoke_bundle(&dir).unwrap();
        let mut cat = request();
        cat.guidance = Some(0.0);
        cat.prompt = "a cat".into();
        let cat_out = image_create(&cat, "z_image", &dir, BackendKind::Cpu, |_, _| ()).unwrap();
        assert_eq!(cat_out.graph_computes, 1);
        let mut dog = cat.clone();
        dog.prompt = "a dog".into();
        let dog_out = image_create(&dog, "z_image", &dir, BackendKind::Cpu, |_, _| ()).unwrap();
        assert_ne!(cat_out.latents, dog_out.latents);
        assert!(cat_out.latents.iter().all(|v| v.is_finite()));
        assert!(dog_out.latents.iter().all(|v| v.is_finite()));

        let mut guided = cat.clone();
        guided.guidance = Some(2.0);
        let zeros = image_create(&guided, "z_image", &dir, BackendKind::Cpu, |_, _| ()).unwrap();
        assert_eq!(zeros.graph_computes, 2);
        guided.negative_prompt = "a dog".into();
        let negated = image_create(&guided, "z_image", &dir, BackendKind::Cpu, |_, _| ()).unwrap();
        assert_eq!(negated.graph_computes, 2);
        assert_ne!(zeros.latents, negated.latents);
        assert!(negated.latents.iter().all(|v| v.is_finite()));

        guided.negative_prompt = "cat".into();
        let mismatch = match image_create(&guided, "z_image", &dir, BackendKind::Cpu, |_, _| ()) {
            Err(err) => err,
            Ok(_) => panic!("a shorter negative prompt was accepted"),
        };
        assert!(
            mismatch.to_string().contains("tokenized to 1"),
            "{mismatch}"
        );
        guided.guidance = Some(0.0);
        guided.negative_prompt = "a dog".into();
        let unguided = match image_create(&guided, "z_image", &dir, BackendKind::Cpu, |_, _| ()) {
            Err(err) => err,
            Ok(_) => panic!("guidance 0 accepted a negative prompt"),
        };
        assert!(
            unguided
                .to_string()
                .contains("needs a guidance other than 0"),
            "{unguided}"
        );
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
        let message = err.to_string();
        assert!(
            message.contains("embed_tokens") || message.contains("not built"),
            "{message}"
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
            adapters: Vec::new(),
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
