mod transformer;
mod weights;

pub use transformer::Flux2Denoise;
pub use weights::{
    load_safetensors_f32, remap_flux2_lora_target, remap_flux2_weights, Flux2Config,
};

pub fn run_smoke(
    steps: u32,
    width: u32,
    height: u32,
    guidance: f32,
    backend: crate::runtime::BackendKind,
    on_progress: impl FnMut(u32, u32),
) -> Result<crate::families::DitImageRun, crate::runtime::RuntimeError> {
    let ctx = crate::runtime::GgmlContext::new(backend)?;
    let config = weights::Flux2Config::smoke();
    let mut denoise = transformer::Flux2Denoise::build(&ctx, config)?;
    denoise.load_weight_map(&Default::default(), config.inner_dim as usize)?;
    let latent = vec![0.1f32; config.inner_dim as usize];
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
