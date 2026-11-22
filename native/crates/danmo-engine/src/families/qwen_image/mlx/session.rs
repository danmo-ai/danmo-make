//! Safetensors Qwen-Image create on the process MLX executor.
//!
//! Text is the Qwen2.5-VL trunk with the image describe template (34-token
//! prefix drop). The schedule is the mflux Qwen shift on the packed token
//! grid (max 0.9 at 8192, terminal sigma 0.02). Guidance > 1 runs positive and
//! negative branches batched on axis 0. Each branch keeps its own text mask,
//! and the combined velocity is rescaled to the conditional token L2
//! (mflux `QwenImage.compute_guided_noise`). The negative prompt defaults to
//! a single space. Rewrite goes through the
//! separate edit facade, not this create session.

use std::path::{Path, PathBuf};

use danmo_contracts::ImageGenerationRequest;
use danmo_mlx::{Array, Dtype, Error, Stream};

use super::forward;
use super::resident::{self, cache_key};
use super::text;
use crate::common::ops::non_finite;
use crate::contract::ExecutionContext;
use crate::error::EngineError;
use crate::families::{DitDecode, DitImageSession};
use crate::runtime::GgmlContext;
use crate::weights::{LoraFile, MlxWeights, WeightError};

const VAE_SCALE: u32 = 16;

pub(crate) struct MlxSession {
    bundle: PathBuf,
    request: ImageGenerationRequest,
    loras: Vec<LoraFile>,
    width: u32,
    height: u32,
    grid_w: i32,
    grid_h: i32,
    ids: Vec<i32>,
    neg_ids: Option<Vec<i32>>,
    sigmas: Option<Vec<f32>>,
}

impl MlxSession {
    pub(crate) fn prepare(
        bundle: &Path,
        request: &ImageGenerationRequest,
        loras: Vec<LoraFile>,
    ) -> Result<Self, EngineError> {
        if !crate::weights::is_safetensors_image_bundle(bundle) {
            return Err(EngineError::bad_request(format!(
                "{} is not an MLX Qwen-Image bundle",
                bundle.display()
            )));
        }
        let guidance = request.guidance.unwrap_or(1.0);
        if !guidance.is_finite() || guidance < 1.0 {
            return Err(EngineError::bad_request(format!(
                "Qwen-Image guidance {guidance} must be finite and >= 1"
            )));
        }
        if guidance <= 1.0 && !request.negative_prompt.trim().is_empty() {
            return Err(EngineError::bad_request(
                "Qwen-Image negative prompt requires guidance > 1",
            ));
        }
        let (width, height) =
            danmo_contracts::parse_size(&request.size).map_err(EngineError::InvalidRequest)?;
        if width % VAE_SCALE != 0 || height % VAE_SCALE != 0 {
            return Err(WeightError::NotBuilt(format!(
                "size {width}x{height} is not divisible by vae scale {VAE_SCALE}"
            ))
            .into());
        }
        Ok(Self {
            bundle: bundle.to_path_buf(),
            request: request.clone(),
            loras,
            width,
            height,
            grid_w: (width / VAE_SCALE) as i32,
            grid_h: (height / VAE_SCALE) as i32,
            ids: Vec::new(),
            neg_ids: None,
            sigmas: None,
        })
    }
}

impl DitImageSession for MlxSession {
    fn encode(&mut self, _ctx: &GgmlContext, _exec: &ExecutionContext) -> Result<(), EngineError> {
        resident::load_transformer(&self.bundle.join("transformer")).map_err(from_mlx)?;
        resident::load_text_encoder(&self.bundle.join("text_encoder")).map_err(from_mlx)?;
        resident::load_vae(&self.bundle.join("vae")).map_err(from_mlx)?;
        let vocab = text_vocab(&self.bundle)?;
        self.ids = text::prompt_ids(&self.bundle, &self.request.prompt, vocab)?;
        let guidance = self.request.guidance.unwrap_or(1.0);
        self.neg_ids = if guidance > 1.0 {
            let negative = if self.request.negative_prompt.trim().is_empty() {
                " "
            } else {
                self.request.negative_prompt.trim()
            };
            Some(text::prompt_ids(&self.bundle, negative, vocab)?)
        } else {
            None
        };
        Ok(())
    }

    fn schedule(&mut self) -> Result<(), EngineError> {
        if self.ids.is_empty() {
            return Err(EngineError::bad_request("encode must run before schedule"));
        }
        self.sigmas = Some(super::super::schedule(
            self.request.steps.unwrap_or(20),
            self.width,
            self.height,
            VAE_SCALE,
            self.request.scheduler.as_deref(),
            self.request.scheduler_shift,
            self.request.use_empirical_mu,
        )?);
        Ok(())
    }

    fn infer(&mut self, exec: &ExecutionContext) -> Result<Vec<f32>, EngineError> {
        let sigmas = self
            .sigmas
            .clone()
            .ok_or_else(|| EngineError::bad_request("schedule must run before infer"))?;
        let seed = self
            .request
            .seed
            .ok_or_else(|| EngineError::bad_request("Qwen-Image create requires a seed"))?;
        if seed < 0 {
            return Err(EngineError::bad_request(
                "Qwen-Image seed must be non-negative",
            ));
        }
        let text_key = cache_key(&self.bundle.join("text_encoder")).map_err(from_mlx)?;
        let dit_key = cache_key(&self.bundle.join("transformer")).map_err(from_mlx)?;
        let ids = self.ids.clone();
        let neg_ids = self.neg_ids.clone();
        let grid_w = self.grid_w;
        let grid_h = self.grid_h;
        let guidance = self.request.guidance.unwrap_or(1.0);
        let loras = self.loras.clone();
        let exec = exec.clone();
        crate::runtime::mlx_executor()
            .map_err(from_mlx)?
            .execute("qwen_image mlx infer", move |_, stream, cache, _| {
                denoise(
                    cache,
                    &text_key,
                    &dit_key,
                    &ids,
                    neg_ids.as_deref(),
                    grid_h,
                    grid_w,
                    seed as u64,
                    &sigmas,
                    guidance,
                    &loras,
                    &exec,
                    stream,
                )
            })
            .map_err(from_mlx)
    }

    fn decode(
        self: Box<Self>,
        _ctx: &GgmlContext,
        latents: &[f32],
    ) -> Result<DitDecode, EngineError> {
        let vae_key = cache_key(&self.bundle.join("vae")).map_err(from_mlx)?;
        let width = self.width;
        let height = self.height;
        let grid_w = self.grid_w;
        let grid_h = self.grid_h;
        let latents = latents.to_vec();
        let steps = self.request.steps.unwrap_or(20);
        let pixels = crate::runtime::mlx_executor()
            .map_err(from_mlx)?
            .execute("qwen_image mlx decode", move |_, stream, cache, _| {
                decode_pixels(
                    cache, &vae_key, &latents, grid_w, grid_h, width, height, stream,
                )
            })
            .map_err(from_mlx)?;
        Ok(DitDecode {
            preview_png: crate::common::png::encode_rgb(width, height, &pixels),
            graph_computes: steps,
            metadata: serde_json::json!({}),
        })
    }
}

fn text_vocab(bundle: &Path) -> Result<i64, EngineError> {
    let key = cache_key(&bundle.join("text_encoder")).map_err(from_mlx)?;
    crate::runtime::mlx_executor()
        .map_err(from_mlx)?
        .execute("qwen_image text spec", move |_, _, cache, _| {
            let weights = resident_weights(cache, &key, "text encoder")?;
            Ok(text::text_spec(weights)?.vocab)
        })
        .map_err(from_mlx)
}

fn denoise(
    cache: &mut danmo_mlx::ModelCache,
    text_key: &danmo_mlx::CacheKey,
    dit_key: &danmo_mlx::CacheKey,
    ids: &[i32],
    neg_ids: Option<&[i32]>,
    grid_h: i32,
    grid_w: i32,
    seed: u64,
    sigmas: &[f32],
    guidance: f32,
    loras: &[LoraFile],
    exec: &ExecutionContext,
    stream: &Stream,
) -> danmo_mlx::Result<Vec<f32>> {
    if sigmas.len() < 2 || sigmas.iter().any(|sigma| !sigma.is_finite()) {
        return Err(Error::new(
            "Qwen-Image denoise",
            "sigma schedule is empty or non-finite",
        ));
    }
    let spec = {
        let weights = resident_weights(cache, dit_key, "transformer")?;
        forward::assert_contract(weights)?
    };
    let seq_img = grid_h * grid_w;
    let cfg = neg_ids.is_some();

    // Text: encode each branch, right-pad to the joint length with zeros.
    // The negative prompt is shorter than the ChatML template, so its padding
    // must not share the positive mask.
    let (text, txt_mask) = {
        let weights = resident_weights(cache, text_key, "text encoder")?;
        let text_spec = text::text_spec(weights)?;
        if i64::from(spec.text_dim) != i64::from(text_spec.hidden) {
            return Err(Error::new(
                "Qwen-Image denoise",
                format!(
                    "text width {} does not match DiT txt_in {}",
                    text_spec.hidden, spec.text_dim
                ),
            ));
        }
        let pos = text::encode_prompt(weights, text_spec, ids, stream)?;
        let pos_len = pos.shape()[1];
        match neg_ids {
            None => (pos, None),
            Some(neg) => {
                let neg = text::encode_prompt(weights, text_spec, neg, stream)?;
                let neg_len = neg.shape()[1];
                let joint = pos_len.max(neg_len);
                let pos = pad_sequence(&pos, joint, stream)?;
                let neg = pad_sequence(&neg, joint, stream)?;
                let text = Array::concatenate_axis(&[pos, neg], 0, stream)?;
                let mask = forward::joint_text_bias(
                    &[valid_prefix(joint, pos_len), valid_prefix(joint, neg_len)],
                    seq_img,
                    stream,
                )?;
                (text, Some(mask))
            }
        }
    };
    text.eval()?;

    let weights = resident_weights(cache, dit_key, "transformer")?.clone();
    let weights = attach_dit_lora(weights, loras, stream)?;
    let seq_txt = text.shape()[1] as usize;
    let rope = forward::rope_tables(i64::from(grid_h), i64::from(grid_w), seq_txt, stream)?;

    // main create noise: seeded_randn((1, seq, 64), fp32) → bf16, then the
    // NCHW round-trip is an identity on the packed sequence.
    let noise = danmo_mlx::RandomKey::new(seed)?
        .normal(&[1, seq_img, spec.in_channels], Dtype::Float32, stream)?
        .astype(Dtype::BFloat16, stream)?;
    let mut latent = if cfg {
        Array::concatenate_axis(&[noise.clone(), noise], 0, stream)?
    } else {
        noise
    };
    let steps = sigmas.len() - 1;
    for (index, pair) in sigmas.windows(2).enumerate() {
        exec.check_cancelled()
            .map_err(|error| Error::new("Qwen-Image denoise", error.to_string()))?;
        let velocity = forward::forward(
            &weights,
            spec,
            &latent,
            &text,
            pair[0],
            &rope,
            txt_mask.as_ref(),
            stream,
        )?;
        let velocity = if cfg {
            let cond = velocity.slice(
                &[0, 0, 0],
                &[1, seq_img, spec.in_channels],
                &[1, 1, 1],
                stream,
            )?;
            let uncond = velocity.slice(
                &[1, 0, 0],
                &[2, seq_img, spec.in_channels],
                &[1, 1, 1],
                stream,
            )?;
            guided_velocity(&cond, &uncond, guidance, spec.in_channels, stream)?
        } else {
            velocity
        };
        velocity.eval()?;
        let finite = velocity
            .isfinite_all(stream)?
            .astype(Dtype::Float32, stream)?
            .to_vec_f32(stream)?;
        if finite.first().copied() != Some(1.0) {
            let values = velocity.to_vec_f32(stream)?;
            let detail = non_finite(&format!("dit velocity at step {}", index + 1), &values)
                .unwrap_or_else(|| {
                    format!("dit velocity at step {} failed the finite check", index + 1)
                });
            return Err(Error::new("Qwen-Image denoise", detail));
        }
        let dt = pair[1] - pair[0];
        let step = Array::from_f32_slice(&[dt], &[])?.astype(latent.dtype(), stream)?;
        let velocity = velocity.astype(latent.dtype(), stream)?;
        let current = if cfg {
            latent.slice(
                &[0, 0, 0],
                &[1, seq_img, spec.in_channels],
                &[1, 1, 1],
                stream,
            )?
        } else {
            latent
        };
        let updated = Array::add(
            &current,
            &Array::multiply(&velocity, &step, stream)?,
            stream,
        )?;
        latent = if cfg {
            Array::concatenate_axis(&[updated.clone(), updated], 0, stream)?
        } else {
            updated
        };
        latent.eval()?;
        exec.progress("denoise", index as u32 + 1, steps as u32)
            .map_err(|error| Error::new("Qwen-Image denoise", error.to_string()))?;
    }
    let final_latent = if cfg {
        latent.slice(
            &[0, 0, 0],
            &[1, seq_img, spec.in_channels],
            &[1, 1, 1],
            stream,
        )?
    } else {
        latent
    };
    let values = final_latent.to_vec_f32(stream)?;
    if let Some(detail) = non_finite("latent", &values) {
        return Err(Error::new("Qwen-Image denoise", detail));
    }
    if values.len() != (seq_img * spec.in_channels) as usize {
        return Err(Error::new(
            "Qwen-Image denoise",
            format!(
                "latent has {} values, expected {}",
                values.len(),
                seq_img * spec.in_channels
            ),
        ));
    }
    Ok(values)
}

pub(super) fn valid_prefix(joint: i32, kept: i32) -> Vec<bool> {
    (0..joint).map(|index| index < kept).collect()
}

/// mflux `QwenImage.compute_guided_noise`: `u + g*(c-u)`, then each token is
/// scaled so its L2 matches the conditional prediction.
pub(super) fn guided_velocity(
    cond: &Array,
    uncond: &Array,
    guidance: f32,
    channels: i32,
    stream: &Stream,
) -> danmo_mlx::Result<Array> {
    let scale = Array::from_f32_slice(&[guidance], &[])?.astype(cond.dtype(), stream)?;
    let combined = Array::add(
        uncond,
        &Array::multiply(&scale, &Array::subtract(cond, uncond, stream)?, stream)?,
        stream,
    )?;
    let combined = combined.astype(Dtype::Float32, stream)?;
    let ratio = Array::divide(
        &token_l2(cond, channels, stream)?,
        &token_l2(&combined, channels, stream)?,
        stream,
    )?;
    Array::multiply(&combined, &ratio, stream)
}

fn token_l2(input: &Array, channels: i32, stream: &Stream) -> danmo_mlx::Result<Array> {
    let input = input.astype(Dtype::Float32, stream)?;
    let width = Array::from_f32_slice(&[channels as f32], &[])?;
    let eps = Array::from_f32_slice(&[1e-12], &[])?;
    let sum = Array::multiply(
        &input.square(stream)?.mean_axes(&[-1], true, stream)?,
        &width,
        stream,
    )?;
    Array::sqrt(&Array::add(&sum, &eps, stream)?, stream)
}

pub(super) fn pad_sequence(text: &Array, joint: i32, stream: &Stream) -> danmo_mlx::Result<Array> {
    let shape = text.shape();
    let missing = joint - shape[1];
    if missing < 0 {
        return Err(Error::new(
            "Qwen-Image denoise",
            format!("text seq {} exceeds joint length {joint}", shape[1]),
        ));
    }
    if missing == 0 {
        return Ok(text.clone());
    }
    text.pad(&[1], &[0], &[missing], 0.0, "constant", stream)
}

pub(crate) fn decode_pixels(
    cache: &mut danmo_mlx::ModelCache,
    vae_key: &danmo_mlx::CacheKey,
    latents: &[f32],
    grid_w: i32,
    grid_h: i32,
    width: u32,
    height: u32,
    stream: &Stream,
) -> danmo_mlx::Result<Vec<u8>> {
    let weights = resident_weights(cache, vae_key, "vae")?;
    let latent = super::vae::unpack_latent(latents, grid_h, grid_w, stream)?
        .astype(Dtype::BFloat16, stream)?;
    let decoded = super::vae::decode_frame(weights, &latent, stream)?;
    decoded.eval()?;
    let shape = decoded.shape();
    if shape != vec![1, 3, height as i32, width as i32] {
        return Err(Error::new(
            "Qwen-Image decode",
            format!("decoded shape {shape:?} is not [1, 3, {height}, {width}]"),
        ));
    }
    let values = decoded.to_vec_f32(stream)?;
    let plane = (width as usize) * (height as usize);
    let mut rgb = vec![0u8; plane * 3];
    for y in 0..height as usize {
        for x in 0..width as usize {
            for channel in 0..3 {
                let sample = values[channel * plane + y * width as usize + x];
                let byte = ((sample + 1.0) * 0.5 * 255.0).round().clamp(0.0, 255.0);
                rgb[(y * width as usize + x) * 3 + channel] = byte as u8;
            }
        }
    }
    Ok(rgb)
}

pub(super) fn attach_dit_lora(
    weights: MlxWeights,
    loras: &[LoraFile],
    stream: &Stream,
) -> danmo_mlx::Result<MlxWeights> {
    weights
        .attach_lora_routed(loras, stream, &super::lora::route_qwen_image)
        .map_err(|error| Error::new("Qwen-Image LoRA", error.to_string()))
}

pub(super) fn resident_weights<'a>(
    cache: &'a mut danmo_mlx::ModelCache,
    key: &danmo_mlx::CacheKey,
    role: &str,
) -> danmo_mlx::Result<&'a MlxWeights> {
    cache.get::<MlxWeights>(key)?.ok_or_else(|| {
        Error::new(
            "Qwen-Image resident",
            format!("{role} is not in the executor cache"),
        )
    })
}

fn from_mlx(error: Error) -> EngineError {
    let message = error.to_string();
    if message.contains("cancelled") {
        EngineError::Cancelled
    } else {
        WeightError::NotBuilt(message).into()
    }
}

#[cfg(test)]
mod tests {
    use super::{guided_velocity, MlxSession};
    use danmo_contracts::ImageGenerationRequest;

    fn request() -> ImageGenerationRequest {
        serde_json::from_value(serde_json::json!({
            "model": "qwen-image",
            "prompt": "a red apple",
            "size": "512x512",
            "steps": 4,
            "guidance": 1,
            "seed": 1
        }))
        .unwrap()
    }

    #[test]
    fn prepare_rejects_bad_geometry_and_guidance() {
        let dir = std::env::temp_dir().join(format!("qwen-image-mlx-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for role in ["transformer", "text_encoder", "vae"] {
            let directory = dir.join(role);
            std::fs::create_dir_all(&directory).unwrap();
            let array = danmo_mlx::Array::from_f32_slice(&[0.5], &[1]).unwrap();
            danmo_mlx::save_safetensors(
                &directory.join("model.safetensors"),
                &[("probe.weight", &array)],
                &[],
            )
            .unwrap();
        }
        assert!(MlxSession::prepare(&dir, &request(), Vec::new()).is_ok());
        let mut bad = request();
        bad.size = "500x512".into();
        let err = match MlxSession::prepare(&dir, &bad, Vec::new()) {
            Ok(_) => panic!("500x512 must fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("divisible"), "{err}");
        let mut bad = request();
        bad.guidance = Some(0.5);
        let err = match MlxSession::prepare(&dir, &bad, Vec::new()) {
            Ok(_) => panic!("guidance 0.5 must fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("guidance"), "{err}");
        let mut bad = request();
        bad.negative_prompt = "blurry".into();
        let err = match MlxSession::prepare(&dir, &bad, Vec::new()) {
            Ok(_) => panic!("negative prompt at guidance 1 must fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("guidance > 1"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn guided_velocity_matches_mflux_token_rescale() {
        let stream = danmo_mlx::Stream::cpu().unwrap();
        let cond = danmo_mlx::Array::from_f32_slice(&[3.0, 4.0], &[1, 1, 2]).unwrap();
        let uncond = danmo_mlx::Array::from_f32_slice(&[0.0, 0.0], &[1, 1, 2]).unwrap();
        let out = guided_velocity(&cond, &uncond, 4.0, 2, &stream).unwrap();
        out.eval().unwrap();
        let values = out.to_vec_f32(&stream).unwrap();
        assert!((values[0] - 3.0).abs() < 1e-4, "{values:?}");
        assert!((values[1] - 4.0).abs() < 1e-4, "{values:?}");
    }
}
