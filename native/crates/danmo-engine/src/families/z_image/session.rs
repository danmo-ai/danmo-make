use super::geom;
use super::graph::ControlBuild;
use super::transformer::{ControlInput, ZImageDenoise};
use super::weights::{
    config_from_weights, control_config_from_weights, control_weight_layout, vector_param,
    weight_layout, ControlConfig, ZImageConfig,
};
use crate::contract::{Action, ExecutionContext, ResolvedRun};
use crate::error::EngineError;
use crate::families::{DitDecode, DitImageFamily, DitImageSession};
use crate::ports::ResolvePort;
use crate::runtime::GgmlContext;
use crate::weights::{
    assert_gguf_bundle, load_lora_factors, matmul_type, type_block, GgufWeights, LoraFile,
    WeightError,
};
use danmo_contracts::ImageGenerationRequest;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct ZImageFamily;

impl DitImageFamily for ZImageFamily {
    fn load(
        &self,
        run: &ResolvedRun,
        request: &ImageGenerationRequest,
        loras: Vec<LoraFile>,
        resolver: &dyn ResolvePort,
    ) -> Result<Box<dyn DitImageSession>, EngineError> {
        let files = assert_gguf_bundle(&run.bundle)?;
        let loaded = GgufWeights::open_parts(&files.transformer)?;
        let config = config_from_weights(&loaded)?;
        let control = match &request.structural_guide {
            Some(guide) if guide.weight > 0.0 => {
                Some(prepare_control(run, config, guide, resolver)?)
            }
            _ => None,
        };
        Ok(Box::new(ZImageSession {
            bundle: run.bundle.clone(),
            request: request.clone(),
            loras,
            config,
            grid: None,
            denoise: None,
            txt: None,
            uncond: None,
            sigmas: None,
            control,
            control_context: None,
        }))
    }
}

struct PreparedControl {
    files: Vec<PathBuf>,
    config: ControlConfig,
    rgb: Vec<u8>,
    width: u32,
    height: u32,
    scale: f32,
    inpaint: Option<PreparedInpaint>,
}

struct PreparedInpaint {
    rgb: Vec<u8>,
    width: u32,
    height: u32,
    mask: Vec<u8>,
    mask_width: u32,
    mask_height: u32,
}

fn prepare_control(
    run: &ResolvedRun,
    config: ZImageConfig,
    guide: &danmo_contracts::StructuralGuide,
    resolver: &dyn ResolvePort,
) -> Result<PreparedControl, EngineError> {
    if guide.model_id.trim().is_empty() {
        return Err(EngineError::bad_request(
            "structural_guide.model_id is required",
        ));
    }
    let control_run = resolver.resolve(&guide.model_id, Action::ImageCreate)?;
    let files = controlnet_files(&control_run.bundle)?;
    let weights = GgufWeights::open_parts(&files)?;
    let control = control_config_from_weights(config, &weights)?;
    let (rgb, width, height) = resolver.read_source_image(run, &guide.asset_id)?;
    let inpaint = match (
        guide.inpaint_source_asset_id.as_deref(),
        guide.inpaint_mask_asset_id.as_deref(),
    ) {
        (Some(source), Some(mask)) if !source.trim().is_empty() && !mask.trim().is_empty() => {
            let (rgb, width, height) = resolver.read_source_image(run, source)?;
            let (mask, mask_width, mask_height) = resolver.read_source_image(run, mask)?;
            Some(PreparedInpaint {
                rgb,
                width,
                height,
                mask,
                mask_width,
                mask_height,
            })
        }
        (None, None) => None,
        (Some(source), None) if source.trim().is_empty() => None,
        (None, Some(mask)) if mask.trim().is_empty() => None,
        _ => {
            return Err(EngineError::bad_request(
                "structural_guide inpaint source and mask must both be set",
            ));
        }
    };
    Ok(PreparedControl {
        files,
        config: control,
        rgb,
        width,
        height,
        scale: guide.weight,
        inpaint,
    })
}

/// A controlnet bundle is a single-adapter GGUF directory (registry category
/// `controlnets`), not a full family stack.
fn controlnet_files(bundle: &Path) -> Result<Vec<PathBuf>, EngineError> {
    let mut files: Vec<PathBuf> = Vec::new();
    let entries = std::fs::read_dir(bundle).map_err(|err| {
        EngineError::Resolve(format!("controlnet bundle {}: {err}", bundle.display()))
    })?;
    for entry in entries {
        let path = entry
            .map_err(|err| EngineError::Resolve(err.to_string()))?
            .path();
        if path.extension().is_some_and(|ext| ext == "gguf") {
            files.push(path);
        }
    }
    files.sort();
    if files.is_empty() {
        return Err(EngineError::Resolve(format!(
            "controlnet bundle {} has no GGUF file; install the controlnet first",
            bundle.display()
        )));
    }
    Ok(files)
}

struct ZImageSession {
    bundle: PathBuf,
    request: ImageGenerationRequest,
    loras: Vec<LoraFile>,
    config: ZImageConfig,
    grid: Option<geom::TokenGrid>,
    denoise: Option<ZImageDenoise>,
    txt: Option<Vec<f32>>,
    uncond: Option<Vec<f32>>,
    sigmas: Option<Vec<f32>>,
    control: Option<PreparedControl>,
    control_context: Option<Vec<f32>>,
}

impl DitImageSession for ZImageSession {
    fn encode(&mut self, ctx: &GgmlContext) -> Result<(), EngineError> {
        let files = assert_gguf_bundle(&self.bundle)?;
        let loaded = GgufWeights::open_parts(&files.transformer)?;
        let (txt, cap_real) = super::encode_caption(
            ctx,
            &self.bundle,
            &files.text_encoder,
            &self.request.prompt,
            self.config.cap_dim,
        )?;
        self.uncond = super::encode_negative(
            ctx,
            &self.bundle,
            &files.text_encoder,
            &self.request.negative_prompt,
            self.config.cap_dim,
            cap_real,
            self.request.guidance.unwrap_or(1.0),
        )?;
        let (width, height) =
            danmo_contracts::parse_size(&self.request.size).map_err(EngineError::InvalidRequest)?;
        let grid = geom::token_grid(width, height, cap_real, self.config.vae_scale)?;
        let layout = weight_layout(self.config);
        let factors = load_lora_factors(&self.loras, &layout)?;
        let mut types = HashMap::new();
        for (name, ne0, _) in &layout {
            let ggml_type = loaded.ggml_type_of(name)?;
            check_stored_type(name, ggml_type, *ne0)?;
            types.insert(name.clone(), ggml_type);
        }
        let control_context = match &self.control {
            Some(control) => Some(build_control_context(
                ctx,
                &files.vae,
                control,
                self.config,
                grid,
                width,
                height,
            )?),
            None => None,
        };
        let control_build = match &self.control {
            Some(control) => {
                let controlnet = GgufWeights::open_parts(&control.files)?;
                let control_layout = control_weight_layout(self.config, control.config);
                let mut control_types = HashMap::new();
                for (name, ne0, _) in &control_layout {
                    let ggml_type = controlnet.ggml_type_of(name)?;
                    check_stored_type(name, ggml_type, *ne0)?;
                    control_types.insert(name.clone(), ggml_type);
                }
                Some((controlnet, control_layout, control_types))
            }
            None => None,
        };
        let plan = control_build
            .as_ref()
            .map(|(_, _, control_types)| ControlBuild {
                config: self.control.as_ref().expect("control").config,
                types: control_types.clone(),
            });
        let mut denoise = ZImageDenoise::build(
            ctx,
            self.config,
            width,
            height,
            cap_real,
            &types,
            &factors,
            plan,
        )?;
        for (name, ne0, ne1) in &layout {
            let raw = loaded.raw_exact(name, &[*ne0, *ne1])?;
            denoise.load_bytes(name, &raw)?;
        }
        if let Some((controlnet, control_layout, _)) = &control_build {
            for (name, ne0, ne1) in control_layout {
                let raw = controlnet.raw_exact(name, &[*ne0, *ne1])?;
                denoise.load_bytes(name, &raw)?;
            }
        }
        for (index, factor) in factors.iter().enumerate() {
            denoise.load_f32(&format!("lora.{index}.a"), &factor.a)?;
            denoise.load_f32(&format!("lora.{index}.b"), &factor.b)?;
        }
        self.grid = Some(grid);
        self.txt = Some(txt);
        self.denoise = Some(denoise);
        self.control_context = control_context;
        Ok(())
    }

    fn schedule(&mut self) -> Result<(), EngineError> {
        let (width, height) =
            danmo_contracts::parse_size(&self.request.size).map_err(EngineError::InvalidRequest)?;
        self.sigmas = Some(super::schedule(
            self.config,
            self.request.steps.unwrap_or(4),
            width,
            height,
            self.request.scheduler.as_deref(),
            self.request.scheduler_shift,
            self.request.use_empirical_mu,
        )?);
        Ok(())
    }

    fn infer(&mut self, exec: &ExecutionContext) -> Result<Vec<f32>, EngineError> {
        let denoise = self
            .denoise
            .as_mut()
            .ok_or_else(|| EngineError::bad_request("encode must run before infer"))?;
        let txt = self
            .txt
            .as_ref()
            .ok_or_else(|| EngineError::bad_request("text encoding is missing"))?;
        let sigmas = self
            .sigmas
            .as_ref()
            .ok_or_else(|| EngineError::bad_request("schedule must run before infer"))?;
        let latent = match self.request.seed {
            Some(seed) => crate::common::ops::normal_latent(seed, denoise.latent_len()),
            None => vec![0.1f32; denoise.latent_len()],
        };
        let control = match (&self.control, &self.control_context) {
            (Some(control), Some(context)) => Some(ControlInput {
                context,
                scale: control.scale,
            }),
            _ => None,
        };
        let mut callback_error = None;
        let result = denoise.denoise(
            latent,
            txt,
            self.uncond.as_deref(),
            sigmas,
            self.request.guidance.unwrap_or(1.0),
            control,
            |step, total| {
                if callback_error.is_none() {
                    if let Err(error) = exec.progress("denoise", step, total) {
                        callback_error = Some(error);
                    }
                }
            },
        );
        if let Some(error) = callback_error {
            return Err(error);
        }
        let latents = result?;
        if let Some(detail) = crate::common::ops::non_finite("z-image latent", &latents) {
            return Err(WeightError::NotBuilt(detail).into());
        }
        Ok(latents)
    }

    fn decode(
        self: Box<Self>,
        ctx: &GgmlContext,
        latents: &[f32],
    ) -> Result<DitDecode, EngineError> {
        let files = assert_gguf_bundle(&self.bundle)?;
        let grid = self
            .grid
            .ok_or_else(|| EngineError::bad_request("encode must run before decode"))?;
        let denoise = self
            .denoise
            .as_ref()
            .ok_or_else(|| EngineError::bad_request("encode must run before decode"))?;
        let (width, height) =
            danmo_contracts::parse_size(&self.request.size).map_err(EngineError::InvalidRequest)?;
        let png = super::vae::decode(
            ctx,
            &files.vae,
            latents,
            self.config.in_channels,
            grid.latent_w,
            grid.latent_h,
            width,
            height,
        )?;
        Ok(DitDecode {
            preview_png: png,
            graph_computes: denoise.graph_computes(),
        })
    }
}

/// Diffusers `ZImageControlNetInpaintPipeline` layout: `[control latents;
/// preserve mask; inpaint latents]` as planar channels. Without the inpaint
/// pair the mask and inpaint planes stay zero.
fn build_control_context(
    ctx: &GgmlContext,
    vae_paths: &[PathBuf],
    control: &PreparedControl,
    config: ZImageConfig,
    grid: geom::TokenGrid,
    width: u32,
    height: u32,
) -> Result<Vec<f32>, EngineError> {
    let rgb = crate::common::image::resize_lanczos(
        &control.rgb,
        control.width,
        control.height,
        width,
        height,
    )?;
    let n11 = crate::common::image::rgb_to_n11_planar(&rgb, width, height)?;
    let control_latents = super::vae::encode(ctx, vae_paths, &n11, width, height)?;
    let in_ch = config.in_channels as usize;
    let plane = (grid.latent_w * grid.latent_h) as usize;
    let expect = plane * in_ch;
    if control_latents.len() != expect {
        return Err(WeightError::NotBuilt(format!(
            "control vae encode produced {} values but the latent plane is {expect}",
            control_latents.len()
        ))
        .into());
    }
    let total = plane * control.config.in_channels as usize;
    let mut context = vec![0f32; total];
    context[..expect].copy_from_slice(&control_latents);
    if let Some(inpaint) = &control.inpaint {
        let source = crate::common::image::resize_lanczos(
            &inpaint.rgb,
            inpaint.width,
            inpaint.height,
            width,
            height,
        )?;
        let mask = crate::common::image::resize_lanczos(
            &inpaint.mask,
            inpaint.mask_width,
            inpaint.mask_height,
            width,
            height,
        )?;
        let pixels = (width as usize) * (height as usize);
        let mut masked = vec![0u8; source.len()];
        for px in 0..pixels {
            let repaint = mask[px * 3] > 127;
            for c in 0..3 {
                masked[px * 3 + c] = if repaint { 0 } else { source[px * 3 + c] };
            }
        }
        let n11 = crate::common::image::rgb_to_n11_planar(&masked, width, height)?;
        let masked_latents = super::vae::encode(ctx, vae_paths, &n11, width, height)?;
        if masked_latents.len() != expect {
            return Err(WeightError::NotBuilt(format!(
                "inpaint vae encode produced {} values but the latent plane is {expect}",
                masked_latents.len()
            ))
            .into());
        }
        context[plane * (in_ch + 1)..plane * (2 * in_ch + 1)].copy_from_slice(&masked_latents);
        let lw = grid.latent_w as usize;
        let lh = grid.latent_h as usize;
        for y in 0..lh {
            for x in 0..lw {
                let sx =
                    (((x as f32 + 0.5) * width as f32 / lw as f32).floor() as u32).min(width - 1);
                let sy =
                    (((y as f32 + 0.5) * height as f32 / lh as f32).floor() as u32).min(height - 1);
                let repaint = mask[(sy as usize * width as usize + sx as usize) * 3] > 127;
                context[x + y * lw + plane * in_ch] = if repaint { 0.0 } else { 1.0 };
            }
        }
    }
    Ok(context)
}

fn check_stored_type(name: &str, ggml_type: u32, ne0: i64) -> Result<(), WeightError> {
    if vector_param(name) {
        if ggml_type != crate::runtime::TYPE_F32 {
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
