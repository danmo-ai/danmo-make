//! Z-Image. Image create and rewrite edit run on the MLX session. This module
//! still owns the sigma schedule and the GGUF smoke bundles that prove a
//! removed ggml graph fails loud. Tensor names in those bundles are the
//! unsloth / SD.cpp GGUF names.
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

mod edit_session;
mod geom;
#[cfg(feature = "mlx")]
mod mlx;
#[cfg(test)]
mod reference;
mod session;
#[cfg(feature = "mlx")]
pub mod train;
mod weights;

pub(crate) use edit_session::ZImageEditFamily;
pub(crate) use session::ZImageFamily;

use crate::common::ops::{empirical_mu_sigmas, flow_match_shift_sigmas, linear_sigmas};
use crate::weights::{write_f32_tensors, F32Tensor, WeightError};
use std::path::Path;

use weights::{weight_layout, ZImageConfig};

pub(crate) fn schedule(
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

/// Edit smoke bundle: the smoke DiT and token-embed fixture plus a VAE file
/// with full encoder and decoder towers (latent 4, blocks 32/32/64/64, 32
/// groups) so the rewrite img2img path can encode and decode.
pub fn write_edit_smoke_bundle(dir: &Path) -> Result<(), WeightError> {
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
    let embed = vec![0.1f32; config.cap_dim as usize];
    write_f32_tensors(
        &dir.join("text_encoder.gguf"),
        &[F32Tensor {
            name: "token_embed.weight",
            ne0: config.cap_dim,
            ne1: 1,
            data: &embed,
        }],
    )?;
    let spec = crate::common::vae::VaeSpec {
        latent_channels: config.in_channels,
        block_out: vec![32, 32, 64, 64],
        resnets: 3,
        groups: 32,
        scaling: 0.3611,
        shift: 0.1159,
        attention: true,
    };
    let mut params = crate::common::vae::parameter_list(&spec)?;
    params.extend(crate::common::vae::encoder_parameter_list(
        &spec,
        "encoder.conv_norm_out",
    )?);
    let owned: Vec<(String, [i64; 4], Vec<f32>)> = params
        .iter()
        .map(|param| {
            let mut ne = [1i64; 4];
            for (index, dim) in param.shape.iter().enumerate() {
                ne[index] = *dim;
            }
            let n = param.shape.iter().map(|dim| *dim as usize).product();
            let data = if param.name.ends_with(".bias") {
                vec![0.0; n]
            } else if param.name.contains("norm") {
                vec![1.0; n]
            } else {
                (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.02).collect()
            };
            (param.name.clone(), ne, data)
        })
        .collect();
    let tensors: Vec<crate::weights::F32Nd<'_>> = owned
        .iter()
        .map(|(name, ne, data)| crate::weights::F32Nd {
            name,
            ne: *ne,
            data,
        })
        .collect();
    crate::weights::write_f32_nd(&dir.join("vae.gguf"), &tensors)?;
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

/// Control smoke pair: the edit smoke bundle (full VAE encoder) plus a
/// controlnet adapter bundle with the smoke control branch weights.
pub fn write_control_smoke_bundle(base: &Path, controlnet: &Path) -> Result<(), WeightError> {
    write_edit_smoke_bundle(base)?;
    let config = ZImageConfig::smoke();
    let control = weights::ControlConfig {
        refiner_blocks: config.refiners,
        layer_blocks: 1,
        in_channels: config.in_channels * 2 + 1,
    };
    std::fs::create_dir_all(controlnet).map_err(|err| WeightError::Open {
        path: controlnet.display().to_string(),
        detail: err.to_string(),
    })?;
    let owned: Vec<(String, i64, i64, Vec<f32>)> = weights::control_weight_layout(config, control)
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
    write_f32_tensors(&controlnet.join("controlnet.gguf"), &tensors)?;
    let manifest = r#"{
  "format": "gguf",
  "quant": "bf16",
  "files": {
    "transformer": "controlnet.gguf"
  }
}
"#;
    std::fs::write(controlnet.join("bundle.manifest.json"), manifest).map_err(|err| {
        WeightError::Open {
            path: controlnet.display().to_string(),
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
    use danmo_contracts::ImageGenerationRequest;

    #[test]
    fn gguf_create_fails_loud_without_the_ggml_graph() {
        let dir = scratch("z-image-gguf-retired");
        write_smoke_bundle(&dir).unwrap();
        let err = match image_create(&request(), "z_image", &dir, BackendKind::Cpu, |_, _| ()) {
            Err(err) => err,
            Ok(_) => panic!("gguf create still ran the removed ggml graph"),
        };
        assert!(
            err.to_string().contains("no longer has a ggml graph"),
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
            use_pe: None,
            adapters: Vec::new(),
            structural_guide: None,
            metadata: serde_json::json!({}),
            preview: true,
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
