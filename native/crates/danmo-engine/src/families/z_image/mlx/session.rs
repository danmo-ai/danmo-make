//! Safetensors Z-Image create on the process MLX executor.
//!
//! The three component directories are admitted into `ModelCache`, then
//! encode, denoise, and decode run on that thread. Guidance 0 and 1 are the
//! single-forward path. Any other guidance or a negative prompt fails instead
//! of being dropped. A structural guide with weight > 0 runs the eager DiT
//! path with ControlNet hint injection (compiled DiT has no control hooks).
//! LoRA is the shared residual on a clone of the DiT; the compiled graph for
//! that clone is keyed by the adapter files.

use std::path::{Path, PathBuf};

use danmo_contracts::ImageGenerationRequest;
use danmo_mlx::{Array, Dtype, Error, Stream};

use super::control::ControlConditioning;
use super::model::{
    forward_cached_with_control, forward_compiled_with_step_cache, CaptionCache, CompiledCompute,
};
use super::resident::{self, cache_key};
use super::step_cache::StepCache;
use super::MlxZImageConfig;
use crate::common::qwen3::Qwen3Config;
use crate::contract::ExecutionContext;
use crate::error::EngineError;
use crate::families::{DitDecode, DitImageSession};
use crate::runtime::GgmlContext;
use crate::weights::{LoraFile, MlxWeights, WeightError};

pub(crate) struct ControlInpaint {
    /// Inpaint/source RGB resized to the create size, NCHW n11 planar.
    pub source_n11: Vec<f32>,
    /// Pixel-space preserve mask `[H*W]`, 1 = keep source (UI white = repaint inverted).
    pub preserve_hw: Vec<f32>,
}

pub(crate) struct ControlGuide {
    pub controlnet: PathBuf,
    /// Preprocessed structural map (canny/depth/pose/…) as NCHW n11 planar.
    /// `None` = DiffSynth inpaint-only: pack control latents as all `-1`.
    pub image_n11: Option<Vec<f32>>,
    pub scale: f32,
    pub inpaint: Option<ControlInpaint>,
}

pub(crate) struct MlxSession {
    bundle: PathBuf,
    request: ImageGenerationRequest,
    loras: Vec<LoraFile>,
    ids: Vec<i32>,
    sigmas: Option<Vec<f32>>,
    width: u32,
    height: u32,
    control: Option<ControlGuide>,
}

pub(crate) fn is_mlx_bundle(bundle: &Path) -> bool {
    crate::weights::is_safetensors_image_bundle(bundle)
}

impl MlxSession {
    pub(crate) fn prepare(
        bundle: &Path,
        request: &ImageGenerationRequest,
        loras: Vec<LoraFile>,
        control: Option<ControlGuide>,
    ) -> Result<Self, EngineError> {
        if !is_mlx_bundle(bundle) {
            return Err(EngineError::bad_request(format!(
                "{} is not an MLX Z-Image bundle",
                bundle.display()
            )));
        }
        if !guidance_is_single_pass(request.guidance) {
            return Err(EngineError::bad_request(
                "MLX Z-Image create does not apply guidance other than 0 or 1",
            ));
        }
        if !request.negative_prompt.trim().is_empty() {
            return Err(EngineError::bad_request(
                "MLX Z-Image create does not apply a negative prompt",
            ));
        }
        if let Some(guide) = &request.structural_guide {
            if guide.weight > 0.0 && control.is_none() {
                return Err(EngineError::bad_request(
                    "MLX Z-Image structural guide requires a resolved controlnet bundle",
                ));
            }
        }
        if let Some(ctrl) = &control {
            if !ctrl.scale.is_finite() || ctrl.scale <= 0.0 {
                return Err(EngineError::bad_request(format!(
                    "structural_guide.weight {} must be finite and > 0",
                    ctrl.scale
                )));
            }
            if ctrl.image_n11.is_none() && ctrl.inpaint.is_none() {
                return Err(EngineError::bad_request(
                    "structural guide requires a control image or an inpaint source/mask pair",
                ));
            }
        }
        let (width, height) =
            danmo_contracts::parse_size(&request.size).map_err(EngineError::InvalidRequest)?;
        if let Some(ctrl) = &control {
            let expected_rgb = 3 * width as usize * height as usize;
            if let Some(pixels) = &ctrl.image_n11 {
                if pixels.len() != expected_rgb {
                    return Err(EngineError::bad_request(format!(
                        "control image has {} values, expected {expected_rgb} for {width}x{height}",
                        pixels.len()
                    )));
                }
            }
            if let Some(inpaint) = &ctrl.inpaint {
                if inpaint.source_n11.len() != expected_rgb {
                    return Err(EngineError::bad_request(format!(
                        "inpaint source has {} values, expected {expected_rgb} for {width}x{height}",
                        inpaint.source_n11.len()
                    )));
                }
                let expected_mask = width as usize * height as usize;
                if inpaint.preserve_hw.len() != expected_mask {
                    return Err(EngineError::bad_request(format!(
                        "inpaint preserve mask has {} values, expected {expected_mask} for {width}x{height}",
                        inpaint.preserve_hw.len()
                    )));
                }
            }
        }
        Ok(Self {
            bundle: bundle.to_path_buf(),
            request: request.clone(),
            loras,
            ids: Vec::new(),
            sigmas: None,
            width,
            height,
            control,
        })
    }
}

impl DitImageSession for MlxSession {
    fn encode(&mut self, _ctx: &GgmlContext) -> Result<(), EngineError> {
        let config = MlxZImageConfig::published();
        resident::load_transformer(
            &self.bundle.join("transformer"),
            config,
            crate::weights::WeightDtype::Native,
            None,
        )
        .map_err(from_mlx)?;
        resident::load_text_encoder(&self.bundle.join("text_encoder")).map_err(from_mlx)?;
        resident::load_vae(&self.bundle.join("vae")).map_err(from_mlx)?;
        if let Some(control) = &self.control {
            resident::load_controlnet(&control.controlnet, config.patch).map_err(from_mlx)?;
        }
        self.ids = config.token_ids(&self.bundle, &self.request.prompt)?;
        Ok(())
    }

    fn schedule(&mut self) -> Result<(), EngineError> {
        if self.ids.is_empty() {
            return Err(EngineError::bad_request("encode must run before schedule"));
        }
        self.sigmas = Some(super::super::schedule(
            schedule_config(),
            self.request.steps.unwrap_or(4),
            self.width,
            self.height,
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
            .ok_or_else(|| EngineError::bad_request("MLX Z-Image create requires a seed"))?;
        if seed < 0 {
            return Err(EngineError::bad_request(
                "MLX Z-Image seed must be non-negative",
            ));
        }
        let text_key = cache_key(&self.bundle.join("text_encoder")).map_err(from_mlx)?;
        let dit_key = cache_key(&self.bundle.join("transformer")).map_err(from_mlx)?;
        let control_key = self
            .control
            .as_ref()
            .map(|guide| cache_key(&guide.controlnet))
            .transpose()
            .map_err(from_mlx)?;
        let control_image = self
            .control
            .as_ref()
            .and_then(|guide| guide.image_n11.clone());
        let control_scale = self.control.as_ref().map(|guide| guide.scale);
        let control_inpaint_source = self
            .control
            .as_ref()
            .and_then(|guide| guide.inpaint.as_ref())
            .map(|inpaint| inpaint.source_n11.clone());
        let control_preserve = self
            .control
            .as_ref()
            .and_then(|guide| guide.inpaint.as_ref())
            .map(|inpaint| inpaint.preserve_hw.clone());
        let vae_key = cache_key(&self.bundle.join("vae")).map_err(from_mlx)?;
        let ids = self.ids.clone();
        let width = self.width;
        let height = self.height;
        let exec = exec.clone();
        let loras = self.loras.clone();
        crate::runtime::mlx_executor()
            .map_err(from_mlx)?
            .execute("z_image mlx infer", move |_, stream, cache, _| {
                denoise(
                    cache,
                    &text_key,
                    &dit_key,
                    &vae_key,
                    control_key.as_ref(),
                    control_image.as_deref(),
                    control_scale,
                    control_inpaint_source.as_deref(),
                    control_preserve.as_deref(),
                    &ids,
                    width,
                    height,
                    seed as u64,
                    None,
                    &sigmas,
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
        let latents = latents.to_vec();
        // Diffusers / VideoX ControlNet inpaint: no post-decode hard composite —
        // preserve is carried by the 33-ch mask + inpaint latents alone.
        let pixels = crate::runtime::mlx_executor()
            .map_err(from_mlx)?
            .execute("z_image mlx decode", move |_, stream, cache, _| {
                decode_pixels(cache, &vae_key, &latents, width, height, stream)
            })
            .map_err(from_mlx)?;
        Ok(DitDecode {
            preview_png: crate::common::png::encode_rgb(width, height, &pixels),
            graph_computes: 1,
        })
    }
}

fn lora_cache_tag(loras: &[LoraFile]) -> String {
    if loras.is_empty() {
        return String::new();
    }
    let mut tag = String::from("#lora");
    for file in loras {
        tag.push('#');
        tag.push_str(&file.id);
        tag.push('@');
        tag.push_str(&file.path.display().to_string());
        tag.push(':');
        tag.push_str(&format!("{}", file.weight));
    }
    tag
}

fn attach_dit_lora(
    weights: MlxWeights,
    loras: &[LoraFile],
    patch: i64,
    stream: &Stream,
) -> danmo_mlx::Result<MlxWeights> {
    weights
        .attach_lora_mapped(loras, stream, &|module| {
            super::weights::remap_lora_module_prefix(module, patch)
        })
        .map_err(|error| Error::new("Z-Image LoRA", error.to_string()))
}

fn encode_control_rgb(
    pixels: &[f32],
    width: u32,
    height: u32,
    vae: &MlxWeights,
    stream: &Stream,
) -> danmo_mlx::Result<Array> {
    let image = Array::from_f32_slice(pixels, &[1, 3, height as i32, width as i32])?;
    let latent = crate::common::vae::mlx::encode(vae, &vae_spec(), &image, stream)?;
    if latent.shape().len() == 4 {
        Ok(latent)
    } else if latent.shape().len() == 3 {
        latent.reshape(
            &[1, latent.shape()[0], latent.shape()[1], latent.shape()[2]],
            stream,
        )
    } else {
        Err(Error::new(
            "Z-Image control",
            format!("vae encode returned unexpected shape {:?}", latent.shape()),
        ))
    }
}

/// DiffSynth `ZImageUnit_PAIControlNet`: when `controlnet_input.image` is
/// absent, `control_latents = ones([1,16,H,W]) * -1` (not mid-gray RGB, not
/// the photo VAE encode).
fn control_latents_absent(
    latent_h: i32,
    latent_w: i32,
    dtype: Dtype,
    stream: &Stream,
) -> danmo_mlx::Result<Array> {
    Array::from_f32_slice(&[-1.0], &[])?
        .astype(dtype, stream)?
        .broadcast_to(&[1, 16, latent_h, latent_w], stream)
}

/// Zero repaint pixels in NCHW n11 RGB (preserve=1 keep, 0 → mid-gray 0.0).
fn mask_inpaint_source_n11(
    source_n11: &[f32],
    preserve_hw: &[f32],
    width: usize,
    height: usize,
) -> danmo_mlx::Result<Vec<f32>> {
    let plane = width * height;
    if source_n11.len() != plane * 3 {
        return Err(Error::new(
            "Z-Image control",
            format!(
                "inpaint source has {} values, expected {} for {width}x{height}",
                source_n11.len(),
                plane * 3
            ),
        ));
    }
    if preserve_hw.len() != plane {
        return Err(Error::new(
            "Z-Image control",
            format!(
                "preserve mask has {} values, expected {plane} for {width}x{height}",
                preserve_hw.len()
            ),
        ));
    }
    let mut out = source_n11.to_vec();
    for channel in 0..3 {
        let base = channel * plane;
        for i in 0..plane {
            out[base + i] *= preserve_hw[i].clamp(0.0, 1.0);
        }
    }
    Ok(out)
}

/// Nearest-neighbor pixel preserve mask down to latent HxW, matching
/// diffusers `F.interpolate(mask, size=...)` (default nearest). Averaging
/// would bleed repaint/keep across mask edges; the checkpoint was trained on
/// hard 0/1 latent masks.
fn downsample_preserve_mask(
    preserve_hw: &[f32],
    width: usize,
    height: usize,
    latent_w: usize,
    latent_h: usize,
    _stream: &Stream,
) -> danmo_mlx::Result<Array> {
    if preserve_hw.len() != width * height {
        return Err(Error::new(
            "Z-Image control",
            format!(
                "preserve mask has {} values, expected {}",
                preserve_hw.len(),
                width * height
            ),
        ));
    }
    if latent_w == 0 || latent_h == 0 || width % latent_w != 0 || height % latent_h != 0 {
        return Err(Error::new(
            "Z-Image control",
            format!(
                "cannot downsample preserve mask {width}x{height} to latent {latent_w}x{latent_h}"
            ),
        ));
    }
    let scale_x = width / latent_w;
    let scale_y = height / latent_h;
    let mut out = vec![0f32; latent_w * latent_h];
    for ly in 0..latent_h {
        for lx in 0..latent_w {
            let y = ly * scale_y;
            let x = lx * scale_x;
            out[ly * latent_w + lx] = preserve_hw[y * width + x];
        }
    }
    Array::from_f32_slice(&out, &[1, 1, latent_h as i32, latent_w as i32])
}

fn denoise(
    cache: &mut danmo_mlx::ModelCache,
    text_key: &danmo_mlx::CacheKey,
    dit_key: &danmo_mlx::CacheKey,
    vae_key: &danmo_mlx::CacheKey,
    control_key: Option<&danmo_mlx::CacheKey>,
    control_image: Option<&[f32]>,
    control_scale: Option<f32>,
    control_inpaint_source: Option<&[f32]>,
    control_preserve: Option<&[f32]>,
    ids: &[i32],
    width: u32,
    height: u32,
    seed: u64,
    initial: Option<&[f32]>,
    sigmas: &[f32],
    loras: &[LoraFile],
    exec: &ExecutionContext,
    stream: &Stream,
) -> danmo_mlx::Result<Vec<f32>> {
    let config = MlxZImageConfig::published();
    let grid = super::super::geom::token_grid(width, height, ids.len() as i64, config.vae_scale)
        .map_err(|error| Error::new("Z-Image denoise", error.to_string()))?;
    let caption = {
        let text = resident_weights(cache, text_key, "text encoder")?;
        let caption = crate::common::qwen3::mlx::encode_ids(
            text,
            text_config(),
            "model.",
            ids,
            &vec![true; ids.len()],
            &[35],
            stream,
        )?;
        caption
            .squeeze_axis(0, stream)?
            .astype(Dtype::BFloat16, stream)?
    };
    caption.eval()?;
    let weights = resident_weights(cache, dit_key, "transformer")?.clone();
    let weights = attach_dit_lora(weights, loras, config.patch, stream)?;
    let caption_cache = CaptionCache::new(&weights, config, grid, &caption, stream)?;
    let control_active = control_key.is_some();
    let control_weights = match control_key {
        Some(key) => Some(resident_weights(cache, key, "controlnet")?.clone()),
        None => None,
    };
    // DiffSynth / Diffusers Fun ControlNet Union: pack
    // [control(16), preserve_mask(1), holed_inpaint(16)]. Denoise from pure
    // noise — no per-step latent blend and no post-decode hard composite.
    // DiffSynth inpaint-only (no pose/canny): control latents = ones * -1.
    let control_context = if let (Some(_), Some(_)) = (control_key, control_scale) {
        let expected = 3 * width as usize * height as usize;
        if let Some(pixels) = control_image {
            if pixels.len() != expected {
                return Err(Error::new(
                    "Z-Image control",
                    format!(
                        "control image has {} values, expected {expected} for {width}x{height}",
                        pixels.len()
                    ),
                ));
            }
        }
        let vae = resident_weights(cache, vae_key, "vae")?.clone();
        let context = match (control_inpaint_source, control_preserve) {
            (Some(source_pixels), Some(preserve_hw)) => {
                if source_pixels.len() != expected {
                    return Err(Error::new(
                        "Z-Image control",
                        format!(
                            "inpaint source has {} values, expected {expected} for {width}x{height}",
                            source_pixels.len()
                        ),
                    ));
                }
                let masked_source = mask_inpaint_source_n11(
                    source_pixels,
                    preserve_hw,
                    width as usize,
                    height as usize,
                )?;
                let inpaint_latent =
                    encode_control_rgb(&masked_source, width, height, &vae, stream)?;
                let latent_h = inpaint_latent.shape()[2];
                let latent_w = inpaint_latent.shape()[3];
                let control_latent = match control_image {
                    Some(pixels) => encode_control_rgb(pixels, width, height, &vae, stream)?,
                    None => {
                        control_latents_absent(latent_h, latent_w, inpaint_latent.dtype(), stream)?
                    }
                };
                let mask = downsample_preserve_mask(
                    preserve_hw,
                    width as usize,
                    height as usize,
                    latent_w as usize,
                    latent_h as usize,
                    stream,
                )?;
                super::control::build_control_context_inpaint_nchw(
                    &control_latent,
                    &mask,
                    &inpaint_latent,
                    stream,
                )?
            }
            (None, None) => {
                let pixels = control_image.ok_or_else(|| {
                    Error::new(
                        "Z-Image control",
                        "structural control image required when inpaint is not set",
                    )
                })?;
                let control_latent = encode_control_rgb(pixels, width, height, &vae, stream)?;
                super::control::build_control_context_nchw(&control_latent, stream)?
            }
            _ => {
                return Err(Error::new(
                    "Z-Image control",
                    "inpaint source and preserve mask must both be set",
                ));
            }
        };
        context.eval()?;
        Some(context)
    } else {
        None
    };
    let lora_key = lora_cache_tag(loras);
    let compiled_key = danmo_mlx::CacheKey {
        id: format!(
            "{}#compiled-dit#{}x{}#cap{}{lora_key}",
            dit_key.id, grid.tokens_w, grid.tokens_h, grid.cap_len
        ),
        revision: dit_key.revision,
    };
    if !control_active && cache.get::<CompiledCompute>(&compiled_key)?.is_none() {
        let compiled = CompiledCompute::new(&weights, config, grid, stream)?;
        let bytes = weights.nbytes();
        cache.insert(compiled_key.clone(), compiled, bytes)?;
    }
    let latent_h = (height as i64 / config.vae_scale) as i32;
    let latent_w = (width as i64 / config.vae_scale) as i32;
    let noise = danmo_mlx::RandomKey::new(seed)?.normal(
        &[1, 16, latent_h, latent_w],
        Dtype::Float32,
        stream,
    )?;
    let latent = match initial {
        None => noise
            .squeeze_axis(0, stream)?
            .astype(Dtype::BFloat16, stream)?,
        Some(reference) => {
            let expected = (16 * latent_h * latent_w) as usize;
            if reference.len() != expected {
                return Err(Error::new(
                    "Z-Image denoise",
                    format!(
                        "reference latent has {} values, expected {expected}",
                        reference.len()
                    ),
                ));
            }
            if sigmas.is_empty() {
                return Err(Error::new("Z-Image denoise", "sigma schedule is empty"));
            }
            let reference = Array::from_f32_slice(reference, &[1, 16, latent_h, latent_w])?
                .astype(Dtype::BFloat16, stream)?;
            let noise = noise.astype(Dtype::BFloat16, stream)?;
            let sigma = sigmas[0];
            let keep =
                Array::from_f32_slice(&[1.0 - sigma], &[])?.astype(Dtype::BFloat16, stream)?;
            let mix = Array::from_f32_slice(&[sigma], &[])?.astype(Dtype::BFloat16, stream)?;
            Array::add(
                &Array::multiply(&reference, &keep, stream)?,
                &Array::multiply(&noise, &mix, stream)?,
                stream,
            )?
            .squeeze_axis(0, stream)?
        }
    };
    let mut step_cache = StepCache::z_image(sigmas.len().saturating_sub(1));
    let total = (sigmas.len().saturating_sub(1)) as u32;
    let interval = exec.observer.preview_interval_steps();
    let max_edge = exec.observer.preview_max_edge();
    if sigmas.len() < 2 || sigmas.iter().any(|sigma| !sigma.is_finite()) {
        return Err(Error::new(
            "Z-Image denoise",
            "scheduler needs finite sigma endpoints",
        ));
    }
    let mut current = latent;
    for step_index in 0..sigmas.len() - 1 {
        if exec.check_cancelled().is_err() {
            return Err(Error::new(
                "Z-Image denoise",
                format!("cancelled at step {step_index}"),
            ));
        }
        exec.progress("denoise", step_index as u32, total)
            .map_err(|error| Error::new("Z-Image denoise", error.to_string()))?;
        let velocity = if let (Some(control_weights), Some(context), Some(scale)) = (
            control_weights.as_ref(),
            control_context.as_ref(),
            control_scale,
        ) {
            forward_cached_with_control(
                &weights,
                config,
                grid,
                &current,
                &caption_cache,
                sigmas[step_index],
                &mut step_cache,
                step_index,
                ControlConditioning {
                    weights: control_weights,
                    context,
                    scale,
                },
                stream,
            )?
        } else {
            let compiled = cache
                .get::<CompiledCompute>(&compiled_key)?
                .ok_or_else(|| {
                    Error::new(
                        "Z-Image denoise",
                        "compiled DiT is not in the executor cache",
                    )
                })?;
            let (velocity, _) = forward_compiled_with_step_cache(
                &weights,
                config,
                compiled,
                &current,
                &caption_cache,
                sigmas[step_index],
                &mut step_cache,
                step_index,
                stream,
            )?;
            velocity
        };
        let velocity = if velocity.dtype() == current.dtype() {
            velocity
        } else {
            velocity.astype(current.dtype(), stream)?
        };
        let delta = sigmas[step_index + 1] - sigmas[step_index];
        let delta = Array::from_f32_slice(&[delta], &[])?.astype(current.dtype(), stream)?;
        current = Array::add(
            &current,
            &Array::multiply(&velocity, &delta, stream)?,
            stream,
        )?;
        if let Some(interval) = interval {
            let step_1 = (step_index + 1) as u32;
            if should_emit_preview(step_1, total, interval) {
                let png = step_preview_png(
                    cache, &current, width, height, latent_h, latent_w, max_edge, stream,
                )?;
                exec.observer
                    .write_step_preview(&png)
                    .map_err(|error| Error::new("Z-Image denoise", error.to_string()))?;
            }
        }
    }
    current.eval()?;
    current.to_vec_f32(stream)
}

fn cached_decoder<'a>(
    cache: &'a mut danmo_mlx::ModelCache,
    vae_key: &danmo_mlx::CacheKey,
    latent_h: i32,
    latent_w: i32,
    stream: &Stream,
) -> danmo_mlx::Result<&'a crate::common::vae::mlx::CompiledDecoder> {
    let weights = resident_weights(cache, vae_key, "vae")?.clone();
    let decoder_key = danmo_mlx::CacheKey {
        id: format!("{}#compiled-vae#{latent_h}x{latent_w}", vae_key.id),
        revision: vae_key.revision,
    };
    if cache
        .get::<crate::common::vae::mlx::CompiledDecoder>(&decoder_key)?
        .is_none()
    {
        let decoder = crate::common::vae::mlx::CompiledDecoder::new(
            weights.clone(),
            vae_spec(),
            stream.clone(),
        )?;
        let bytes = weights.nbytes();
        cache.insert(decoder_key.clone(), decoder, bytes)?;
    }
    cache
        .get::<crate::common::vae::mlx::CompiledDecoder>(&decoder_key)?
        .ok_or_else(|| {
            Error::new(
                "Z-Image decode",
                "compiled VAE is not in the executor cache",
            )
        })
}

fn step_preview_png(
    cache: &mut danmo_mlx::ModelCache,
    latent: &Array,
    width: u32,
    height: u32,
    latent_h: i32,
    latent_w: i32,
    max_edge: u32,
    stream: &Stream,
) -> danmo_mlx::Result<Vec<u8>> {
    let spec = vae_spec();
    let image = {
        let decoder = crate::common::vae::taesd::cached(
            cache,
            latent_h,
            latent_w,
            crate::common::vae::taesd::PreviewScale {
                scaling: spec.scaling,
                shift: spec.shift,
            },
            stream,
        )?;
        decoder.apply(latent)?
    };
    image.eval()?;
    let shape = image.shape();
    if shape.len() != 4 || shape[3] != 3 {
        return Err(Error::new(
            "Z-Image preview",
            format!("TAEF1 output shape {shape:?} is not [1, H, W, 3]"),
        ));
    }
    let (out_h, out_w) = (shape[1] as u32, shape[2] as u32);
    if out_h != height || out_w != width {
        return Err(Error::new(
            "Z-Image preview",
            format!("TAEF1 decoded {out_w}x{out_h}, expected {width}x{height}"),
        ));
    }
    let values = image.to_vec_f32(stream)?;
    let plane = (width as usize).saturating_mul(height as usize);
    if values.len() != plane * 3 {
        return Err(Error::new(
            "Z-Image preview",
            format!(
                "TAEF1 returned {} values, expected {}",
                values.len(),
                plane * 3
            ),
        ));
    }
    // NHWC, then the same channel reversal main applies before saving the PNG.
    let rgb: Vec<u8> = (0..plane)
        .flat_map(|pixel| {
            let base = pixel * 3;
            [2, 1, 0].map(|channel| {
                (values[base + channel] * 255.0)
                    .clamp(0.0, 255.0)
                    .round_ties_even() as u8
            })
        })
        .collect();
    let (dw, dh) = preview_edge(width, height, max_edge);
    let rgb = if dw == width && dh == height {
        rgb
    } else {
        crate::common::image::resize_bicubic(&rgb, width, height, dw, dh)
            .map_err(|error| Error::new("Z-Image preview", error.to_string()))?
    };
    Ok(crate::common::png::encode_rgb(dw, dh, &rgb))
}

fn decode_pixels(
    cache: &mut danmo_mlx::ModelCache,
    vae_key: &danmo_mlx::CacheKey,
    latents: &[f32],
    width: u32,
    height: u32,
    stream: &Stream,
) -> danmo_mlx::Result<Vec<u8>> {
    let config = MlxZImageConfig::published();
    let latent_h = (height as i64 / config.vae_scale) as i32;
    let latent_w = (width as i64 / config.vae_scale) as i32;
    let expected = 16 * latent_h * latent_w;
    if latents.len() != expected as usize {
        return Err(Error::new(
            "Z-Image decode",
            format!("latent has {} values, expected {expected}", latents.len()),
        ));
    }
    let latent = Array::from_f32_slice(latents, &[1, 16, latent_h, latent_w])?
        .astype(Dtype::BFloat16, stream)?;
    let decoder = cached_decoder(cache, vae_key, latent_h, latent_w, stream)?;
    let decoded = decoder.apply(&latent)?;
    decoded.eval()?;
    let values = decoded.to_vec_f32(stream)?;
    let plane = (width as usize).saturating_mul(height as usize);
    if values.len() != plane * 3 {
        return Err(Error::new(
            "Z-Image decode",
            format!(
                "decoded {} values, expected {} for {width}x{height}",
                values.len(),
                plane * 3
            ),
        ));
    }
    Ok((0..plane * 3)
        .map(|index| {
            let channel = index % 3;
            let pixel = index / 3;
            let value = values[channel * plane + pixel];
            ((value + 1.0) * 127.5).clamp(0.0, 255.0).round_ties_even() as u8
        })
        .collect())
}

/// Step 1, every `interval` steps, and the last step. Matches main
/// `maybe_emit_image_step_preview`.
pub(crate) fn should_emit_preview(step_1: u32, total: u32, interval: u32) -> bool {
    let interval = interval.max(1);
    let is_last = step_1 >= total && total > 0;
    !(step_1 > 1 && step_1 % interval != 0 && !is_last)
}

fn preview_edge(width: u32, height: u32, max_edge: u32) -> (u32, u32) {
    let long = width.max(height);
    if max_edge == 0 || long <= max_edge {
        return (width.max(1), height.max(1));
    }
    let scale = max_edge as f32 / long as f32;
    (
        ((width as f32) * scale).round().max(1.0) as u32,
        ((height as f32) * scale).round().max(1.0) as u32,
    )
}

pub(crate) fn encode_image(
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
        .execute("z_image mlx vae encode", move |_, stream, cache, _| {
            let weights = resident_weights(cache, &vae_key, "vae")?.clone();
            let pixels = Array::from_f32_slice(&image, &[1, 3, height as i32, width as i32])?;
            let latent = crate::common::vae::mlx::encode(&weights, &vae_spec(), &pixels, stream)?;
            latent.eval()?;
            latent.to_vec_f32(stream)
        })
        .map_err(from_mlx)
}

pub(crate) fn denoise_reference(
    bundle: &Path,
    ids: &[i32],
    width: u32,
    height: u32,
    seed: u64,
    reference: &[f32],
    sigmas: &[f32],
    loras: &[LoraFile],
    exec: &ExecutionContext,
) -> Result<Vec<f32>, EngineError> {
    resident::load_transformer(
        &bundle.join("transformer"),
        MlxZImageConfig::published(),
        crate::weights::WeightDtype::Native,
        None,
    )
    .map_err(from_mlx)?;
    resident::load_text_encoder(&bundle.join("text_encoder")).map_err(from_mlx)?;
    let text_key = cache_key(&bundle.join("text_encoder")).map_err(from_mlx)?;
    let dit_key = cache_key(&bundle.join("transformer")).map_err(from_mlx)?;
    let vae_key = cache_key(&bundle.join("vae")).map_err(from_mlx)?;
    let ids = ids.to_vec();
    let reference = reference.to_vec();
    let sigmas = sigmas.to_vec();
    let exec = exec.clone();
    let loras = loras.to_vec();
    crate::runtime::mlx_executor()
        .map_err(from_mlx)?
        .execute("z_image mlx edit denoise", move |_, stream, cache, _| {
            denoise(
                cache,
                &text_key,
                &dit_key,
                &vae_key,
                None,
                None,
                None,
                None,
                None,
                &ids,
                width,
                height,
                seed,
                Some(&reference),
                &sigmas,
                &loras,
                &exec,
                stream,
            )
        })
        .map_err(from_mlx)
}

pub(crate) fn decode_latent(
    bundle: &Path,
    latents: &[f32],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, EngineError> {
    resident::load_vae(&bundle.join("vae")).map_err(from_mlx)?;
    let vae_key = cache_key(&bundle.join("vae")).map_err(from_mlx)?;
    let latents = latents.to_vec();
    let pixels = crate::runtime::mlx_executor()
        .map_err(from_mlx)?
        .execute("z_image mlx edit decode", move |_, stream, cache, _| {
            decode_pixels(cache, &vae_key, &latents, width, height, stream)
        })
        .map_err(from_mlx)?;
    Ok(crate::common::png::encode_rgb(width, height, &pixels))
}

pub(crate) fn vae_spec() -> crate::common::vae::VaeSpec {
    crate::common::vae::VaeSpec {
        latent_channels: 16,
        block_out: vec![128, 256, 512, 512],
        resnets: 2,
        groups: 32,
        scaling: 0.3611,
        shift: 0.1159,
        attention: true,
    }
}

fn resident_weights<'a>(
    cache: &'a mut danmo_mlx::ModelCache,
    key: &danmo_mlx::CacheKey,
    role: &str,
) -> danmo_mlx::Result<&'a MlxWeights> {
    cache.get::<MlxWeights>(key)?.ok_or_else(|| {
        Error::new(
            "Z-Image resident",
            format!("{role} is not in the executor cache"),
        )
    })
}

pub(crate) fn text_config() -> Qwen3Config {
    Qwen3Config {
        vocab: 151_936,
        hidden: 2560,
        layers: 36,
        heads: 32,
        kv_heads: 8,
        head_dim: 128,
        intermediate: 9728,
        rope_theta: 1_000_000.0,
    }
}

fn schedule_config() -> super::super::weights::ZImageConfig {
    let config = MlxZImageConfig::published();
    super::super::weights::ZImageConfig {
        dim: config.dim,
        heads: config.heads,
        head_dim: config.head_dim(),
        axes: config.axes,
        layers: config.layers,
        refiners: config.refiners,
        in_channels: config.in_channels,
        cap_dim: config.cap_dim,
        ffn: config.ffn_hidden(),
        mid: config.timestep_mid,
        freq: config.frequency_dim,
        adaln: config.adaln_dim,
        rope_theta: config.rope_theta,
        t_scale: config.t_scale,
        vae_scale: config.vae_scale,
        patch: config.patch,
        requires_sigma_shift: true,
    }
}

fn from_mlx(error: Error) -> EngineError {
    let message = error.to_string();
    if message.contains("cancelled at step") {
        EngineError::Cancelled
    } else {
        WeightError::NotBuilt(message).into()
    }
}

fn guidance_is_single_pass(guidance: Option<f32>) -> bool {
    match guidance {
        None => true,
        Some(value) => value.abs() < 1e-6 || (value - 1.0).abs() < 1e-6,
    }
}

#[cfg(test)]
mod tests {
    use super::{is_mlx_bundle, should_emit_preview, MlxSession};
    use crate::contract::{ExecutionContext, NoopObserver};
    use crate::families::DitImageSession;
    use crate::runtime::{BackendKind, GgmlContext};
    use danmo_contracts::ImageGenerationRequest;
    use danmo_mlx::{Array, Stream};
    use std::path::Path;
    use std::sync::Arc;

    #[test]
    fn preview_follows_the_main_interval() {
        let emit: Vec<u32> = (1..=9)
            .filter(|step| should_emit_preview(*step, 9, 2))
            .collect();
        assert_eq!(emit, vec![1, 2, 4, 6, 8, 9]);
        assert!(should_emit_preview(2, 2, 2));
        assert!(!should_emit_preview(3, 9, 2));
    }

    #[test]
    fn absent_control_latents_are_neg_one() {
        let stream = Stream::cpu().unwrap();
        let latents =
            super::control_latents_absent(2, 3, danmo_mlx::Dtype::Float32, &stream).unwrap();
        latents.eval().unwrap();
        assert_eq!(latents.shape(), vec![1, 16, 2, 3]);
        let values = latents.to_vec_f32(&stream).unwrap();
        assert!(values.iter().all(|v| (*v + 1.0).abs() < 1e-6));
    }

    #[test]
    fn inpaint_source_zeros_repaint_pixels() {
        // NCHW 2x2 RGB: channel 0 all 1.0, others 0.5; preserve keeps left column.
        let source = vec![
            1.0, 1.0, 1.0, 1.0, // R
            0.5, 0.5, 0.5, 0.5, // G
            0.5, 0.5, 0.5, 0.5, // B
        ];
        let preserve = vec![1.0, 0.0, 1.0, 0.0];
        let out = super::mask_inpaint_source_n11(&source, &preserve, 2, 2).unwrap();
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 1.0);
        assert_eq!(out[3], 0.0);
        assert_eq!(out[4], 0.5);
        assert_eq!(out[5], 0.0);
        assert!(super::mask_inpaint_source_n11(&source, &[1.0], 2, 2).is_err());
    }

    fn write_shard(path: &Path) {
        let values = [0.5f32; 4];
        let array = Array::from_f32_slice(&values, &[1, 4]).unwrap();
        danmo_mlx::save_safetensors(path, &[("layers.0.attention_norm1.weight", &array)], &[])
            .unwrap();
        let _ = Stream::cpu();
    }

    fn request() -> ImageGenerationRequest {
        serde_json::from_value(serde_json::json!({
            "model": "z-image-turbo",
            "prompt": "red",
            "size": "16x16",
            "steps": 2,
            "guidance": 0,
            "seed": 1
        }))
        .unwrap()
    }

    #[test]
    fn safetensors_bundle_reaches_the_mlx_text_encoder_and_keeps_weights_resident() {
        let bundle =
            std::env::temp_dir().join(format!("zimage-mlx-session-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&bundle);
        for role in ["transformer", "text_encoder", "vae"] {
            let directory = bundle.join(role);
            std::fs::create_dir_all(&directory).unwrap();
            write_shard(&directory.join("model.safetensors"));
        }
        std::fs::write(bundle.join("tokenizer.json"), r#"{"vocab":{"red":1}}"#).unwrap();
        assert!(is_mlx_bundle(&bundle));

        let mut session = MlxSession::prepare(&bundle, &request(), Vec::new(), None).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        session.encode(&ctx).unwrap();
        let again = super::resident::load_transformer(
            &bundle.join("transformer"),
            super::MlxZImageConfig::published(),
            crate::weights::WeightDtype::Native,
            None,
        )
        .unwrap();
        assert!(again.hit);
        session.schedule().unwrap();
        let exec = ExecutionContext::new("mlx-session", Arc::new(NoopObserver));
        let error = session.infer(&exec).unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("embed_tokens") || message.contains("missing tensor"),
            "{message}"
        );
        assert!(!message.contains("old MLX"));
        let _ = std::fs::remove_dir_all(&bundle);
    }

    /// Real safetensors bundle through the create session. Writes a PNG.
    #[test]
    #[ignore]
    fn real_session_writes_a_png() {
        let bundle = std::path::PathBuf::from(std::env::var("DANQING_TEST_ZIMAGE_BUNDLE").unwrap());
        let output = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_ZIMAGE_SESSION_PNG")
                .unwrap_or_else(|_| "out/zimage-mlx-session.png".into()),
        );
        let width = std::env::var("DANQING_TEST_ZIMAGE_WIDTH").unwrap_or_else(|_| "768".into());
        let height = std::env::var("DANQING_TEST_ZIMAGE_HEIGHT").unwrap_or_else(|_| "1280".into());
        let steps: u32 = std::env::var("DANQING_TEST_ZIMAGE_STEPS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(9);
        let prompt = std::env::var("DANQING_TEST_ZIMAGE_PROMPT")
            .unwrap_or_else(|_| "a red cube on a white table".into());
        let request = serde_json::from_value(serde_json::json!({
            "model": "z-image-turbo",
            "prompt": prompt,
            "size": format!("{width}x{height}"),
            "steps": steps,
            "guidance": 0,
            "seed": 1,
            "scheduler": "linear"
        }))
        .unwrap();
        let mut session = MlxSession::prepare(&bundle, &request, Vec::new(), None).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        session.encode(&ctx).unwrap();
        session.schedule().unwrap();
        let exec = ExecutionContext::new("mlx-session-real", Arc::new(NoopObserver));
        let latents = session.infer(&exec).unwrap();
        assert!(
            latents.iter().any(|value| value.abs() > 1e-4),
            "latent is empty"
        );
        let decoded = Box::new(session).decode(&ctx, &latents).unwrap();
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&output, &decoded.preview_png).unwrap();
        assert_eq!(&decoded.preview_png[..8], b"\x89PNG\r\n\x1a\n");
        let distinct = decoded
            .preview_png
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(
            distinct > 8,
            "png has only {distinct} distinct bytes: {}",
            output.display()
        );
        eprintln!("wrote {}", output.display());
    }
}
