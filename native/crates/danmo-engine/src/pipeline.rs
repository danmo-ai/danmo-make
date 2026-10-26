//! Fixed phase order: resolve → load → encode → schedule → infer → decode → persist.

use crate::families::{self, Media, Shape};
use crate::runtime::BackendKind;
use crate::weights::{self, WeightError};
use danmo_contracts::{parse_size, ImageGenerationRequest};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error(transparent)]
    Weights(#[from] WeightError),
    #[error("family {0} is not registered")]
    UnknownFamily(String),
    #[error("{family} has no ggml graph yet (shape {shape})")]
    GraphNotBuilt { family: String, shape: String },
    #[error("{0}")]
    BadRequest(String),
}

pub struct RunOutput {
    pub family: String,
    pub shape: Shape,
    pub latents: Vec<f32>,
    pub steps: u32,
    pub graph_computes: u32,
    pub width: u32,
    pub height: u32,
}

fn shape_name(shape: Shape) -> &'static str {
    match shape {
        Shape::DitPipeline => "dit_pipeline",
        Shape::JobPipeline => "job_pipeline",
        Shape::GenerationFacade => "generation_facade",
    }
}

/// Image create. Flux2 runs the reserved denoise graph. Every other registered
/// family still walks the same phases and then fails loud.
pub fn image_create(
    request: &ImageGenerationRequest,
    family: &str,
    bundle: &Path,
    backend: BackendKind,
    on_progress: impl FnMut(u32, u32),
) -> Result<RunOutput, EngineError> {
    let spec = families::spec(family).ok_or_else(|| EngineError::UnknownFamily(family.into()))?;
    if spec.media != Media::Image {
        return Err(EngineError::BadRequest(format!(
            "{family} is not an image family"
        )));
    }
    let (width, height) = parse_size(&request.size).map_err(EngineError::BadRequest)?;
    weights::assert_gguf_bundle(bundle)?;
    let steps = request.steps.unwrap_or(4);
    match spec.shape {
        Shape::DitPipeline => {
            let Some(done) = families::run_dit_image(
                family,
                bundle,
                steps,
                width,
                height,
                request.guidance.unwrap_or(1.0),
                backend,
                on_progress,
            )?
            else {
                return Err(EngineError::GraphNotBuilt {
                    family: family.into(),
                    shape: shape_name(spec.shape).into(),
                });
            };
            Ok(RunOutput {
                family: family.into(),
                shape: spec.shape,
                latents: done.latents,
                steps,
                graph_computes: done.graph_computes,
                width,
                height,
            })
        }
        shape => Err(EngineError::GraphNotBuilt {
            family: family.into(),
            shape: shape_name(shape).into(),
        }),
    }
}

pub fn video_create(family: &str, bundle: &Path) -> Result<(), EngineError> {
    route_media(family, bundle, Media::Video)
}

pub fn audio_create(family: &str, bundle: &Path) -> Result<(), EngineError> {
    route_media(family, bundle, Media::Audio)
}

pub fn upscale(family: &str, bundle: &Path) -> Result<(), EngineError> {
    let spec = families::spec(family).ok_or_else(|| EngineError::UnknownFamily(family.into()))?;
    if spec.shape != Shape::JobPipeline {
        return Err(EngineError::BadRequest(format!(
            "{family} is not an upscale job"
        )));
    }
    weights::assert_gguf_bundle(bundle)?;
    Err(EngineError::GraphNotBuilt {
        family: family.into(),
        shape: shape_name(spec.shape).into(),
    })
}

fn route_media(family: &str, bundle: &Path, media: Media) -> Result<(), EngineError> {
    let spec = families::spec(family).ok_or_else(|| EngineError::UnknownFamily(family.into()))?;
    if spec.media != media {
        return Err(EngineError::BadRequest(format!(
            "{family} does not match the requested media"
        )));
    }
    weights::assert_gguf_bundle(bundle)?;
    Err(EngineError::GraphNotBuilt {
        family: family.into(),
        shape: shape_name(spec.shape).into(),
    })
}
