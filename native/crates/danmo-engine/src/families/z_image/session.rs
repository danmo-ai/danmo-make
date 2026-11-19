//! Z-Image image create. The ggml graph for this action is gone.
//!
//! Safetensors bundles run on the MLX session when the engine is built with
//! `mlx`. A GGUF bundle fails here instead of falling back to ggml. LoRA is
//! attached by the shared weight layer. Rewrite edit uses the same MLX session.
//! Structural guide (Fun ControlNet Union) resolves the controlnet bundle and
//! control image here, then runs eager DiT with hint injection.

use crate::contract::ResolvedRun;
use crate::error::EngineError;
use crate::families::{DitImageFamily, DitImageSession};
use crate::ports::ResolvePort;
use crate::weights::{LoraFile, WeightError};
use danmo_contracts::ImageGenerationRequest;

pub struct ZImageFamily;

impl DitImageFamily for ZImageFamily {
    fn load(
        &self,
        run: &ResolvedRun,
        request: &ImageGenerationRequest,
        loras: Vec<LoraFile>,
        resolver: &dyn ResolvePort,
    ) -> Result<Box<dyn DitImageSession>, EngineError> {
        let _ = &run.bundle;
        #[cfg(feature = "mlx")]
        if super::mlx::session::is_mlx_bundle(&run.bundle) {
            let control = resolve_control_guide(run, request, resolver)?;
            return Ok(Box::new(super::mlx::session::MlxSession::prepare(
                &run.bundle,
                request,
                loras,
                control,
            )?));
        }
        let _ = (loras, request, resolver);
        Err(WeightError::NotBuilt(
            "z_image image create no longer has a ggml graph; install the safetensors bundle and build with the mlx feature"
                .into(),
        )
        .into())
    }
}

#[cfg(feature = "mlx")]
fn resolve_control_guide(
    run: &ResolvedRun,
    request: &ImageGenerationRequest,
    resolver: &dyn ResolvePort,
) -> Result<Option<super::mlx::session::ControlGuide>, EngineError> {
    let Some(guide) = request.structural_guide.as_ref() else {
        return Ok(None);
    };
    if !(guide.weight > 0.0) {
        return Ok(None);
    }
    if guide.model_id.trim().is_empty() {
        return Err(EngineError::bad_request(
            "structural_guide.model_id is required",
        ));
    }
    let inpaint_source_id = guide
        .inpaint_source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty());
    let inpaint_mask_id = guide
        .inpaint_mask_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty());
    match (inpaint_source_id, inpaint_mask_id) {
        (None, None) => {}
        (Some(_), Some(_)) => {}
        _ => {
            return Err(EngineError::bad_request(
                "structural_guide.inpaint_source_asset_id and inpaint_mask_asset_id must both be set for inpaint mode",
            ));
        }
    }
    let control_id = guide.asset_id.trim();
    // DiffSynth ZImageUnit_PAIControlNet: structural map is optional. Inpaint
    // alone fills control latents with -1. Companion retouch sets asset_id to
    // the photo (same as inpaint source) — that is not a pose/canny map.
    let needs_structural_map = match inpaint_source_id {
        Some(source_id) => !control_id.is_empty() && control_id != source_id,
        None => true,
    };
    if needs_structural_map && control_id.is_empty() {
        return Err(EngineError::bad_request(
            "structural_guide.asset_id is required when inpaint is not set",
        ));
    }
    let controlnet = resolver.resolve_component(run, "controlnet", &guide.model_id)?;
    if !crate::weights::is_controlnet_safetensors_dir(&controlnet)
        && !(controlnet.is_file()
            && controlnet.extension().and_then(|v| v.to_str()) == Some("safetensors"))
    {
        return Err(EngineError::Resolve(format!(
            "structural_guide controlnet {} is not an installed safetensors controlnet at {}",
            guide.model_id,
            controlnet.display()
        )));
    }
    let (width, height) =
        danmo_contracts::parse_size(&request.size).map_err(EngineError::InvalidRequest)?;
    let inpaint = match (inpaint_source_id, inpaint_mask_id) {
        (Some(source_id), Some(mask_id)) => {
            let source_n11 =
                load_resized_n11(resolver, run, source_id, width, height, "inpaint source")?;
            let preserve_hw = load_preserve_mask(resolver, run, mask_id, width, height)?;
            Some(super::mlx::session::ControlInpaint {
                source_n11,
                preserve_hw,
            })
        }
        _ => None,
    };
    // VideoX / Diffusers inpaint demos always pass a structural map (pose/
    // canny/…). Companion retouch has none — derive Gray(source), the Union
    // 2602-trained condition. Do not reuse the photo RGB or constant mid-gray.
    let image_n11 = if needs_structural_map {
        Some(load_resized_n11(
            resolver,
            run,
            control_id,
            width,
            height,
            "control",
        )?)
    } else if let Some(inp) = inpaint.as_ref() {
        Some(rgb_n11_to_gray_n11(
            &inp.source_n11,
            width as usize,
            height as usize,
        )?)
    } else {
        None
    };
    Ok(Some(super::mlx::session::ControlGuide {
        controlnet,
        image_n11,
        scale: guide.weight,
        inpaint,
    }))
}

/// Rec.601 luma → replicated RGB in NCHW n11 (Union Gray control).
#[cfg(feature = "mlx")]
fn rgb_n11_to_gray_n11(
    source_n11: &[f32],
    width: usize,
    height: usize,
) -> Result<Vec<f32>, EngineError> {
    let plane = width * height;
    if source_n11.len() != plane * 3 {
        return Err(EngineError::bad_request(format!(
            "gray control source has {} values, expected {} for {width}x{height}",
            source_n11.len(),
            plane * 3
        )));
    }
    let mut out = vec![0f32; plane * 3];
    for i in 0..plane {
        let y = 0.299 * source_n11[i] + 0.587 * source_n11[plane + i] + 0.114 * source_n11[2 * plane + i];
        out[i] = y;
        out[plane + i] = y;
        out[2 * plane + i] = y;
    }
    Ok(out)
}

#[cfg(feature = "mlx")]
fn load_resized_n11(
    resolver: &dyn ResolvePort,
    run: &ResolvedRun,
    asset_id: &str,
    width: u32,
    height: u32,
    label: &str,
) -> Result<Vec<f32>, EngineError> {
    let (rgb, source_w, source_h) = resolver.read_source_image(run, asset_id)?;
    let rgb = if source_w == width && source_h == height {
        rgb
    } else {
        crate::common::image::resize_lanczos(&rgb, source_w, source_h, width, height).map_err(
            |error| EngineError::bad_request(format!("{label} image resize failed: {error}")),
        )?
    };
    rgb_to_n11_nchw(&rgb, width, height)
}

/// UI mask: white = repaint. Z-Image Fun ControlNet uses inverted preserve
/// (1 = keep), matching Comfy Apply Fun ControlNet.
#[cfg(feature = "mlx")]
fn load_preserve_mask(
    resolver: &dyn ResolvePort,
    run: &ResolvedRun,
    asset_id: &str,
    width: u32,
    height: u32,
) -> Result<Vec<f32>, EngineError> {
    let (rgb, source_w, source_h) = resolver.read_source_image(run, asset_id)?;
    let rgb = if source_w == width && source_h == height {
        rgb
    } else {
        crate::common::image::resize_lanczos(&rgb, source_w, source_h, width, height).map_err(
            |error| EngineError::bad_request(format!("inpaint mask resize failed: {error}")),
        )?
    };
    let w = width as usize;
    let h = height as usize;
    if rgb.len() != w * h * 3 {
        return Err(EngineError::bad_request(format!(
            "inpaint mask has {} bytes, expected {} for {width}x{height}",
            rgb.len(),
            w * h * 3
        )));
    }
    let mut preserve = vec![0f32; w * h];
    for i in 0..(w * h) {
        let r = rgb[i * 3] as f32;
        let g = rgb[i * 3 + 1] as f32;
        let b = rgb[i * 3 + 2] as f32;
        let paint = ((r + g + b) / (3.0 * 255.0)).clamp(0.0, 1.0);
        preserve[i] = 1.0 - paint;
    }
    Ok(preserve)
}

#[cfg(feature = "mlx")]
fn rgb_to_n11_nchw(rgb: &[u8], width: u32, height: u32) -> Result<Vec<f32>, EngineError> {
    let w = width as usize;
    let h = height as usize;
    if rgb.len() != w * h * 3 {
        return Err(EngineError::bad_request(format!(
            "control image has {} bytes, expected {} for {width}x{height}",
            rgb.len(),
            w * h * 3
        )));
    }
    let plane = w * h;
    let mut out = vec![0f32; plane * 3];
    for y in 0..h {
        for x in 0..w {
            for channel in 0..3 {
                let byte = rgb[(y * w + x) * 3 + channel];
                out[channel * plane + y * w + x] = (byte as f32 / 255.0) * 2.0 - 1.0;
            }
        }
    }
    Ok(out)
}
