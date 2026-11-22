//! Safetensors Flux2 Klein create and rewrite on the process MLX executor.
//!
//! Text matches danmo-make `text_encoder_mlx.py`: chat template, thinking off,
//! pad to 512, concatenate Qwen3 layers 9/18/27. The tower is Qwen3-4B or
//! Qwen3-8B, read from the checkpoint. The schedule is empirical-mu flow
//! matching. Rewrite follows BFL / diffusers Klein edit: clean source tokens
//! are concatenated as reference context (`T = 10, 20, …`), the target starts
//! from noise, and only target tokens are stepped. LoRA uses the Flux2 key
//! remap. Guidance is the raw scale into `guidance_linear_*` when those
//! weights exist.

use std::path::{Path, PathBuf};

use danmo_contracts::ImageGenerationRequest;
use danmo_mlx::{Array, Dtype, Error, Stream};

use super::forward::{self, KleinSpec};
use super::resident::{self, cache_key};
use super::text::{self, TEXT_LAYERS, TEXT_VOCAB};
use crate::common::ops::{non_finite, scaled_sinusoidal_timestep, sinusoidal_timestep};
use crate::common::vae::VaeSpec;
use crate::contract::ExecutionContext;
use crate::error::EngineError;
use crate::families::{DitDecode, DitImageSession};
use crate::runtime::GgmlContext;
use crate::weights::{LoraFile, WeightError};

const TEXT_LEN: usize = 512;
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
    valid: Vec<bool>,
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
                "{} is not an MLX Flux2 bundle",
                bundle.display()
            )));
        }
        if let Some(value) = request.guidance {
            if !value.is_finite() || value <= 0.0 {
                return Err(EngineError::bad_request(format!(
                    "Flux2 guidance {value} is not a positive finite number"
                )));
            }
        }
        let (width, height) =
            danmo_contracts::parse_size(&request.size).map_err(EngineError::InvalidRequest)?;
        if width % VAE_SCALE != 0 || height % VAE_SCALE != 0 {
            return Err(WeightError::NotBuilt(format!(
                "size {width}x{height} is not divisible by vae scale {VAE_SCALE}"
            ))
            .into());
        }
        if let Some(max_edge) = super::super::manifest_max_edge(bundle)? {
            if width > max_edge || height > max_edge {
                return Err(WeightError::NotBuilt(format!(
                    "this bundle accepts edges up to {max_edge}, got {width}x{height}"
                ))
                .into());
            }
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
            valid: Vec::new(),
            sigmas: None,
        })
    }
}

impl DitImageSession for MlxSession {
    fn encode(&mut self, _ctx: &GgmlContext, _exec: &ExecutionContext) -> Result<(), EngineError> {
        resident::load_transformer(&self.bundle.join("transformer")).map_err(from_mlx)?;
        resident::load_text_encoder(&self.bundle.join("text_encoder")).map_err(from_mlx)?;
        resident::load_vae(&self.bundle.join("vae")).map_err(from_mlx)?;
        let (ids, valid) = prompt_batch(&self.bundle, &self.request.prompt)?;
        self.ids = ids;
        self.valid = valid;
        Ok(())
    }

    fn schedule(&mut self) -> Result<(), EngineError> {
        if self.ids.len() != TEXT_LEN {
            return Err(EngineError::bad_request("encode must run before schedule"));
        }
        let steps = self.request.steps.unwrap_or(4);
        self.sigmas = Some(super::super::schedule(
            steps,
            i64::from(self.grid_w),
            i64::from(self.grid_h),
        ));
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
            .ok_or_else(|| EngineError::bad_request("Flux2 Klein create requires a seed"))?;
        let text_key = cache_key(&self.bundle.join("text_encoder")).map_err(from_mlx)?;
        let dit_key = cache_key(&self.bundle.join("transformer")).map_err(from_mlx)?;
        let ids = self.ids.clone();
        let valid = self.valid.clone();
        let grid_w = self.grid_w;
        let grid_h = self.grid_h;
        let guidance = self.request.guidance;
        let loras = self.loras.clone();
        let exec = exec.clone();
        crate::runtime::mlx_executor()
            .map_err(from_mlx)?
            .execute("flux2 klein infer", move |_, stream, cache, _| {
                denoise(
                    cache, &text_key, &dit_key, &ids, &valid, grid_h, grid_w, seed, &sigmas,
                    guidance, &loras, &exec, stream,
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
        let steps = self.request.steps.unwrap_or(4);
        let pixels = crate::runtime::mlx_executor()
            .map_err(from_mlx)?
            .execute("flux2 klein decode", move |_, stream, cache, _| {
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

fn denoise(
    cache: &mut danmo_mlx::ModelCache,
    text_key: &danmo_mlx::CacheKey,
    dit_key: &danmo_mlx::CacheKey,
    ids: &[i32],
    valid: &[bool],
    grid_h: i32,
    grid_w: i32,
    seed: i64,
    sigmas: &[f32],
    guidance: Option<f32>,
    loras: &[LoraFile],
    exec: &ExecutionContext,
    stream: &Stream,
) -> danmo_mlx::Result<Vec<f32>> {
    if sigmas.len() < 2 {
        return Err(Error::new("Flux2 Klein", "denoise schedule is empty"));
    }
    let spec = {
        let weights = resident_weights(cache, dit_key, "transformer")?;
        forward::assert_contract(weights)?
    };
    let noise = packed_noise(seed, spec.in_channels, grid_h, grid_w, stream)?;
    step_latent(
        cache, text_key, dit_key, ids, valid, grid_h, grid_w, &noise, None, sigmas, guidance,
        loras, exec, stream,
    )
}

fn step_latent(
    cache: &mut danmo_mlx::ModelCache,
    text_key: &danmo_mlx::CacheKey,
    dit_key: &danmo_mlx::CacheKey,
    ids: &[i32],
    valid: &[bool],
    grid_h: i32,
    grid_w: i32,
    initial: &[f32],
    references: Option<&[f32]>,
    sigmas: &[f32],
    guidance: Option<f32>,
    loras: &[LoraFile],
    exec: &ExecutionContext,
    stream: &Stream,
) -> danmo_mlx::Result<Vec<f32>> {
    if sigmas.len() < 2 {
        return Err(Error::new("Flux2 Klein", "denoise schedule is empty"));
    }
    let spec = {
        let weights = resident_weights(cache, dit_key, "transformer")?;
        forward::assert_contract(weights)?
    };
    let seq = (grid_h * grid_w) as usize;
    let channels = spec.in_channels as usize;
    if initial.len() != seq * channels {
        return Err(Error::new(
            "Flux2 Klein",
            format!(
                "latent has {} values, expected {}",
                initial.len(),
                seq * channels
            ),
        ));
    }
    let mut img_ids = forward::image_token_ids(0, grid_h, grid_w)?;
    let ref_seq = match references {
        Some(values) => {
            if values.len() != seq * channels {
                return Err(Error::new(
                    "Flux2 Klein",
                    format!(
                        "reference latent has {} values, expected {}",
                        values.len(),
                        seq * channels
                    ),
                ));
            }
            img_ids.extend(forward::image_token_ids(
                forward::REF_T_SCALE,
                grid_h,
                grid_w,
            )?);
            seq
        }
        None => 0,
    };
    let guidance_freq = guidance_embedding(spec, guidance)?;
    let text = {
        let weights = resident_weights(cache, text_key, "text encoder")?;
        let config = text::text_config_from_weights(weights)?;
        let joint = config.hidden * TEXT_LAYERS.len() as i64;
        if joint != i64::from(spec.joint) {
            return Err(Error::new(
                "Flux2 Klein",
                format!(
                    "text width {joint} does not match context_embedder {}",
                    spec.joint
                ),
            ));
        }
        let encoded = crate::common::qwen3::mlx::encode_ids_full(
            weights,
            config,
            "model.",
            ids,
            valid,
            &TEXT_LAYERS,
            stream,
        )?;
        let shape = encoded.shape();
        if shape != vec![1, TEXT_LEN as i32, spec.joint] {
            return Err(Error::new(
                "Flux2 Klein",
                format!(
                    "text embedding {shape:?} is not [1, {TEXT_LEN}, {}]",
                    spec.joint
                ),
            ));
        }
        // Match danmo-make Flux2TextEncoder.encode → astype(bfloat16).
        encoded.astype(Dtype::BFloat16, stream)?
    };
    text.eval()?;
    let weights = resident_weights(cache, dit_key, "transformer")?.clone();
    let weights = attach_dit_lora(weights, loras, stream)?;
    // Match main: sample fp32 noise, then cast to latent_noise_dtype (bf16)
    // before the Euler loop. Reference tokens stay clean and are rebuilt each
    // step (diffusers Flux2KleinPipeline).
    let mut latent = Array::from_f32_slice(initial, &[1, seq as i32, spec.in_channels])?
        .astype(Dtype::BFloat16, stream)?;
    let reference = match references {
        Some(values) => Some(
            Array::from_f32_slice(values, &[1, ref_seq as i32, spec.in_channels])?
                .astype(Dtype::BFloat16, stream)?,
        ),
        None => None,
    };
    let steps = sigmas.len() - 1;
    exec.log("info", &format!("flux2 denoise starting ({steps} steps)"))
        .map_err(|error| Error::new("Flux2 Klein", error.to_string()))?;
    for (index, pair) in sigmas.windows(2).enumerate() {
        exec.check_cancelled()
            .map_err(|error| Error::new("Flux2 Klein", error.to_string()))?;
        let temb = scaled_sinusoidal_timestep(pair[0], spec.freq as usize);
        let model_input = match &reference {
            Some(refs) => Array::concatenate_axis(&[latent.clone(), refs.clone()], 1, stream)?,
            None => latent.clone(),
        };
        let velocity = forward::forward(
            &weights,
            spec,
            &model_input,
            &text,
            &temb,
            guidance_freq.as_deref(),
            &img_ids,
            seq as i32,
            stream,
        )?;
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
            return Err(Error::new("Flux2 Klein", detail));
        }
        let dt = pair[1] - pair[0];
        let step = Array::from_f32_slice(&[dt], &[])?.astype(velocity.dtype(), stream)?;
        latent = latent.astype(velocity.dtype(), stream)?;
        latent = Array::add(&latent, &Array::multiply(&velocity, &step, stream)?, stream)?;
        latent.eval()?;
        exec.progress("denoise", index as u32 + 1, steps as u32)
            .map_err(|error| Error::new("Flux2 Klein", error.to_string()))?;
    }
    let values = latent.to_vec_f32(stream)?;
    if let Some(detail) = non_finite("latent", &values) {
        return Err(Error::new("Flux2 Klein", detail));
    }
    if values.len() != seq * channels {
        return Err(Error::new(
            "Flux2 Klein",
            format!(
                "latent has {} values, expected {}",
                values.len(),
                seq * channels
            ),
        ));
    }
    Ok(values)
}

fn decode_pixels(
    cache: &mut danmo_mlx::ModelCache,
    vae_key: &danmo_mlx::CacheKey,
    latents: &[f32],
    grid_w: i32,
    grid_h: i32,
    width: u32,
    height: u32,
    stream: &Stream,
) -> danmo_mlx::Result<Vec<u8>> {
    let vae = resident_weights(cache, vae_key, "vae")?;
    let mean = vae.tensor("bn.running_mean")?.to_vec_f32(stream)?;
    let var = vae.tensor("bn.running_var")?.to_vec_f32(stream)?;
    let weight = vae.tensor("post_quant_conv.weight")?.to_vec_f32(stream)?;
    let bias = vae.tensor("post_quant_conv.bias")?.to_vec_f32(stream)?;
    let unpacked = super::super::vae::kl_latent_from_packed(
        latents,
        &mean,
        &var,
        &weight,
        &bias,
        i64::from(grid_w),
        i64::from(grid_h),
    )
    .map_err(|error| Error::new("Flux2 Klein", error.to_string()))?;
    let latent_h = grid_h * 2;
    let latent_w = grid_w * 2;
    let expected = (32 * latent_h * latent_w) as usize;
    if unpacked.len() != expected {
        return Err(Error::new(
            "Flux2 Klein",
            format!(
                "unpacked latent has {} values, expected {expected}",
                unpacked.len()
            ),
        ));
    }
    let latent = Array::from_f32_slice(&unpacked, &[1, 32, latent_h, latent_w])?
        .astype(Dtype::BFloat16, stream)?;
    let decoded = crate::common::vae::mlx::decode(vae, &vae_spec(), &latent, stream)?;
    decoded.eval()?;
    let values = decoded.to_vec_f32(stream)?;
    let shape = decoded.shape();
    if shape != vec![1, 3, height as i32, width as i32] {
        return Err(Error::new(
            "Flux2 Klein",
            format!("decoded shape {shape:?} is not [1, 3, {height}, {width}]"),
        ));
    }
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

/// Decoder layout. Each up block has `layers_per_block + 1` resnets, so Klein
/// uses 3 here. The encoder uses `layers_per_block` and is counted separately.
fn vae_spec() -> VaeSpec {
    VaeSpec {
        latent_channels: 32,
        block_out: vec![128, 256, 512, 512],
        resnets: 3,
        groups: 32,
        scaling: 1.0,
        shift: 0.0,
        attention: true,
    }
}

fn encoder_spec(weights: &crate::weights::MlxWeights) -> danmo_mlx::Result<VaeSpec> {
    let mut spec = vae_spec();
    spec.resnets = shared_encoder_resnets(weights.names(), spec.block_out.len())?;
    Ok(spec)
}

fn shared_encoder_resnets<'a>(
    names: impl IntoIterator<Item = &'a str>,
    blocks: usize,
) -> danmo_mlx::Result<usize> {
    let names: Vec<&str> = names.into_iter().collect();
    let counts: Vec<usize> = (0..blocks)
        .map(|index| resnet_count(&names, &format!("encoder.down_blocks.{index}.resnets.")))
        .collect();
    let count = counts.first().copied().unwrap_or(0);
    if count < 1 || counts.iter().any(|other| *other != count) {
        return Err(Error::new(
            "Flux2 Klein",
            format!("encoder down blocks have resnet counts {counts:?}"),
        ));
    }
    Ok(count)
}

fn resnet_count(names: &[&str], prefix: &str) -> usize {
    let mut seen = Vec::new();
    for name in names {
        let Some(rest) = name.strip_prefix(prefix) else {
            continue;
        };
        if !rest.ends_with(".norm1.weight") {
            continue;
        }
        let Some(index) = rest.split('.').next() else {
            continue;
        };
        if let Ok(index) = index.parse::<usize>() {
            seen.push(index);
        }
    }
    seen.sort_unstable();
    seen.dedup();
    if seen.iter().copied().eq(0..seen.len()) {
        seen.len()
    } else {
        0
    }
}

fn guidance_embedding(
    spec: KleinSpec,
    guidance: Option<f32>,
) -> danmo_mlx::Result<Option<Vec<f32>>> {
    match (spec.guidance, guidance) {
        (false, None) => Ok(None),
        (false, Some(value)) if (value - 1.0).abs() < 1e-6 => Ok(None),
        (false, Some(value)) => Err(Error::new(
            "Flux2 Klein",
            format!("this bundle has no guidance embedder, got guidance {value}"),
        )),
        (true, None) => Err(Error::new(
            "Flux2 Klein",
            "this bundle has a guidance embedder and guidance was not set",
        )),
        (true, Some(value)) if !value.is_finite() || value <= 0.0 => Err(Error::new(
            "Flux2 Klein",
            format!("guidance {value} is not a positive finite number"),
        )),
        (true, Some(value)) => Ok(Some(sinusoidal_timestep(value, spec.freq as usize))),
    }
}

fn attach_dit_lora(
    weights: crate::weights::MlxWeights,
    loras: &[LoraFile],
    stream: &Stream,
) -> danmo_mlx::Result<crate::weights::MlxWeights> {
    weights
        .attach_lora_routed(loras, stream, &super::lora::route_flux2)
        .map_err(|error| Error::new("Flux2 LoRA", error.to_string()))
}

pub(crate) fn prompt_batch(
    bundle: &Path,
    prompt: &str,
) -> Result<(Vec<i32>, Vec<bool>), EngineError> {
    let tokenizer = crate::common::qwen3::load_tokenizer(bundle)?;
    let tokens = crate::common::qwen3::tokenize(&tokenizer, prompt, TEXT_VOCAB, false)?;
    if tokens.len() > TEXT_LEN {
        return Err(WeightError::NotBuilt(format!(
            "prompt tokenized to {} tokens, Flux2 Klein accepts {TEXT_LEN}",
            tokens.len()
        ))
        .into());
    }
    let pad = pad_token(&tokenizer, TEXT_VOCAB)?;
    let mut ids = tokens.clone();
    ids.resize(TEXT_LEN, pad);
    let mut valid = vec![false; TEXT_LEN];
    for slot in valid.iter_mut().take(tokens.len()) {
        *slot = true;
    }
    Ok((ids, valid))
}

/// Match danmo-make `seeded_randn([1, C, H, W], seed, fp32)` then the DiT pack
/// `NCHW → NHWC → [1, H*W, C]`.
pub(crate) fn packed_noise(
    seed: i64,
    channels: i32,
    grid_h: i32,
    grid_w: i32,
    stream: &Stream,
) -> danmo_mlx::Result<Vec<f32>> {
    danmo_mlx::RandomKey::new(seed as u64)?
        .normal(&[1, channels, grid_h, grid_w], Dtype::Float32, stream)?
        .transpose_axes(&[0, 2, 3, 1], stream)?
        .reshape(&[1, grid_h * grid_w, channels], stream)?
        .to_vec_f32(stream)
}

pub(crate) fn encode_packed_image(
    bundle: &Path,
    image: &[f32],
    width: u32,
    height: u32,
) -> Result<Vec<f32>, EngineError> {
    resident::load_vae(&bundle.join("vae")).map_err(from_mlx)?;
    let vae_key = cache_key(&bundle.join("vae")).map_err(from_mlx)?;
    let image = image.to_vec();
    crate::runtime::mlx_executor()
        .map_err(from_mlx)?
        .execute("flux2 klein vae encode", move |_, stream, cache, _| {
            let weights = resident_weights(cache, &vae_key, "vae")?.clone();
            let pixels = Array::from_f32_slice(&image, &[1, 3, height as i32, width as i32])?;
            let spec = encoder_spec(&weights)?;
            let conv = crate::common::vae::mlx::conv_out(&weights, &spec, &pixels, stream)?;
            conv.eval()?;
            let shape = conv.shape();
            if shape.len() != 4 || shape[0] != 1 || shape[1] != 64 {
                return Err(Error::new(
                    "Flux2 Klein",
                    format!("encoder conv_out {shape:?} is not [1, 64, H, W]"),
                ));
            }
            let values = conv.to_vec_f32(stream)?;
            let quant_weight = weights.tensor("quant_conv.weight")?.to_vec_f32(stream)?;
            let quant_bias = weights.tensor("quant_conv.bias")?.to_vec_f32(stream)?;
            let mean = weights.tensor("bn.running_mean")?.to_vec_f32(stream)?;
            let var = weights.tensor("bn.running_var")?.to_vec_f32(stream)?;
            super::super::vae::pack_from_conv_out(
                &values,
                i64::from(shape[2]),
                i64::from(shape[3]),
                &quant_weight,
                &quant_bias,
                &mean,
                &var,
            )
            .map_err(|error| Error::new("Flux2 Klein", error.to_string()))
        })
        .map_err(from_mlx)
}

pub(crate) fn denoise_latents(
    bundle: &Path,
    ids: &[i32],
    valid: &[bool],
    grid_h: i32,
    grid_w: i32,
    latent: &[f32],
    reference: Option<&[f32]>,
    sigmas: &[f32],
    guidance: Option<f32>,
    loras: &[LoraFile],
    exec: &ExecutionContext,
) -> Result<Vec<f32>, EngineError> {
    resident::load_transformer(&bundle.join("transformer")).map_err(from_mlx)?;
    resident::load_text_encoder(&bundle.join("text_encoder")).map_err(from_mlx)?;
    let text_key = cache_key(&bundle.join("text_encoder")).map_err(from_mlx)?;
    let dit_key = cache_key(&bundle.join("transformer")).map_err(from_mlx)?;
    let ids = ids.to_vec();
    let valid = valid.to_vec();
    let latent = latent.to_vec();
    let reference = reference.map(|values| values.to_vec());
    let sigmas = sigmas.to_vec();
    let loras = loras.to_vec();
    let exec = exec.clone();
    crate::runtime::mlx_executor()
        .map_err(from_mlx)?
        .execute("flux2 klein edit denoise", move |_, stream, cache, _| {
            step_latent(
                cache,
                &text_key,
                &dit_key,
                &ids,
                &valid,
                grid_h,
                grid_w,
                &latent,
                reference.as_deref(),
                &sigmas,
                guidance,
                &loras,
                &exec,
                stream,
            )
        })
        .map_err(from_mlx)
}

pub(crate) fn decode_png(
    bundle: &Path,
    latents: &[f32],
    grid_w: i32,
    grid_h: i32,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, EngineError> {
    resident::load_vae(&bundle.join("vae")).map_err(from_mlx)?;
    let vae_key = cache_key(&bundle.join("vae")).map_err(from_mlx)?;
    let latents = latents.to_vec();
    let pixels = crate::runtime::mlx_executor()
        .map_err(from_mlx)?
        .execute("flux2 klein edit decode", move |_, stream, cache, _| {
            decode_pixels(
                cache, &vae_key, &latents, grid_w, grid_h, width, height, stream,
            )
        })
        .map_err(from_mlx)?;
    Ok(crate::common::png::encode_rgb(width, height, &pixels))
}

fn pad_token(tokenizer: &str, vocab: i64) -> Result<i32, WeightError> {
    let ids = crate::common::qwen3::tokenize_plain(tokenizer, "<|endoftext|>", vocab)?;
    let [id] = ids.as_slice() else {
        return Err(WeightError::NotBuilt(format!(
            "<|endoftext|> tokenized to {} ids, Flux2 padding needs one id",
            ids.len()
        )));
    };
    Ok(*id)
}

fn resident_weights<'a>(
    cache: &'a mut danmo_mlx::ModelCache,
    key: &danmo_mlx::CacheKey,
    role: &str,
) -> danmo_mlx::Result<&'a crate::weights::MlxWeights> {
    cache
        .get::<crate::weights::MlxWeights>(key)?
        .ok_or_else(|| {
            Error::new(
                "Flux2 resident",
                format!("{role} is not in the executor cache"),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::shared_encoder_resnets;

    #[test]
    fn klein_encoder_is_two_resnets_and_a_gap_fails() {
        let names = [
            "encoder.down_blocks.0.resnets.0.norm1.weight",
            "encoder.down_blocks.0.resnets.1.norm1.weight",
            "encoder.down_blocks.1.resnets.0.norm1.weight",
            "encoder.down_blocks.1.resnets.1.norm1.weight",
            "decoder.up_blocks.0.resnets.2.norm1.weight",
        ];
        assert_eq!(shared_encoder_resnets(names, 2).unwrap(), 2);
        let gapped = ["encoder.down_blocks.0.resnets.0.norm1.weight"];
        assert!(shared_encoder_resnets(gapped, 2).is_err());
    }
}

fn from_mlx(error: Error) -> EngineError {
    let message = error.to_string();
    if message.contains("cancelled") {
        EngineError::Cancelled
    } else {
        WeightError::NotBuilt(message).into()
    }
}
