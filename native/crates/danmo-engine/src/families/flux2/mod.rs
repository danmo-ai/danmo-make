mod transformer;
mod weights;

pub use transformer::Flux2Denoise;
pub use weights::Flux2Config;

use crate::weights::{assert_gguf_bundle, GgufWeights, WeightError};
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
    let ctx = crate::runtime::GgmlContext::new(backend)?;
    let config = weights::Flux2Config::smoke();
    let mut denoise = transformer::Flux2Denoise::build(&ctx, config)?;
    let loaded = GgufWeights::open_parts(&files.transformer)?;
    let dim = config.inner_dim;
    let weight = loaded.f32_exact("velocity.weight", &[dim, dim])?;
    denoise.load_f32("velocity.weight", &weight)?;
    let latent = vec![0.1f32; dim as usize];
    let latents = denoise.denoise(
        latent,
        steps,
        width,
        height,
        config.requires_sigma_shift,
        guidance,
        on_progress,
    )?;
    Ok(crate::families::DitImageRun {
        latents,
        graph_computes: denoise.graph_computes(),
    })
}
