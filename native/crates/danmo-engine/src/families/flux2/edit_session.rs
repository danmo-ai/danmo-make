//! Flux2 rewrite. Matches BFL / diffusers `Flux2KleinPipeline` edit:
//! encode the source through the MLX VAE (`conv_out`, `quant_conv`, mean,
//! 2×2 patch, batch norm), keep those packed tokens as reference context at
//! `T = 10`, sample target noise for the output grid, and denoise the full
//! schedule while concatenating `[target | reference]` each step. Only target
//! tokens are stepped. LoRA and a guidance embedder follow the checkpoint,
//! the same as create.

use crate::contract::{Artifact, EngineOutput, EngineRequest, ExecutionContext, ResolvedRun};
use crate::error::EngineError;
use crate::families::{FacadeFamily, FacadeSession};
use crate::ports::ResolvePort;
use crate::runtime::GgmlContext;
use crate::weights::{LoraFile, WeightError};
use danmo_contracts::ImageEditRequest;
use serde_json::{json, Value};

pub struct Flux2EditFamily;

struct Prepared {
    request: ImageEditRequest,
    source: (u32, u32),
    width: u32,
    height: u32,
    reference: Vec<f32>,
    ids: Vec<i32>,
    valid: Vec<bool>,
    sigmas: Vec<f32>,
    bundle: std::path::PathBuf,
    loras: Vec<LoraFile>,
}

impl FacadeFamily for Flux2EditFamily {
    fn prepare(
        &self,
        _ctx: &GgmlContext,
        run: &ResolvedRun,
        request: &EngineRequest,
        resolver: &dyn ResolvePort,
        exec: &ExecutionContext,
    ) -> Result<Box<dyn FacadeSession>, EngineError> {
        let EngineRequest::ImageEdit(request) = request else {
            return Err(EngineError::bad_request(
                "flux2 edit received a non-edit request",
            ));
        };
        if request.operation != "rewrite" {
            return Err(WeightError::NotBuilt(format!(
                "flux2 edit operation {} is not built",
                request.operation
            ))
            .into());
        }
        if !request.reference_asset_ids.is_empty() {
            return Err(WeightError::NotBuilt(
                "flux2 edit multi reference conditioning is not built".into(),
            )
            .into());
        }
        if !request.negative_prompt.trim().is_empty() {
            return Err(WeightError::NotBuilt(
                "negative prompt is not built for this family".into(),
            )
            .into());
        }
        match request
            .scheduler
            .as_deref()
            .unwrap_or("flow_match_euler_discrete")
        {
            "flow_match_euler" | "flow_match_euler_discrete" => {}
            other => {
                return Err(WeightError::NotBuilt(format!("scheduler {other} is not built")).into())
            }
        }
        if request.prompt.trim().is_empty() {
            return Err(EngineError::bad_request("flux2 edit prompt is empty"));
        }
        if request.source_asset_id.trim().is_empty() {
            return Err(EngineError::bad_request(
                "flux2 edit source_asset_id is empty",
            ));
        }
        if let Some(value) = request.guidance {
            if !value.is_finite() || value <= 0.0 {
                return Err(EngineError::bad_request(format!(
                    "Flux2 guidance {value} is not a positive finite number"
                )));
            }
        }
        let steps = request.steps.unwrap_or(4);
        if steps < 1 {
            return Err(EngineError::bad_request(format!(
                "flux2 edit steps {steps} is below 1"
            )));
        }
        if request.seed.is_none() {
            return Err(EngineError::bad_request("Flux2 Klein edit requires a seed"));
        }
        #[cfg(not(feature = "mlx"))]
        {
            let _ = (run, resolver, exec);
            return Err(WeightError::NotBuilt(
                "flux2 image edit no longer has a ggml graph; build with the mlx feature".into(),
            )
            .into());
        }
        #[cfg(feature = "mlx")]
        {
            if !crate::weights::is_safetensors_image_bundle(&run.bundle) {
                return Err(WeightError::NotBuilt(
                    "flux2 image edit no longer has a ggml graph; install the safetensors bundle and build with the mlx feature"
                        .into(),
                )
                .into());
            }
            let (rgb, source_w, source_h) =
                resolver.read_source_image(run, &request.source_asset_id)?;
            let (rgb, width, height) =
                crate::common::image::center_crop_align(&rgb, source_w, source_h, 16)?;
            if let Some(max_edge) = super::manifest_max_edge(&run.bundle)? {
                if width > max_edge || height > max_edge {
                    return Err(WeightError::NotBuilt(format!(
                        "this bundle accepts edges up to {max_edge}, got {width}x{height}"
                    ))
                    .into());
                }
            }
            let image = crate::common::image::rgb_to_n11_planar(&rgb, width, height)?;
            let reference =
                super::mlx::session::encode_packed_image(&run.bundle, &image, width, height)?;
            let grid_w = width / 16;
            let grid_h = height / 16;
            let expect = (grid_w * grid_h * 128) as usize;
            if reference.len() != expect {
                return Err(WeightError::NotBuilt(format!(
                    "vae encode produced {} values but the DiT latent is {expect}",
                    reference.len()
                ))
                .into());
            }
            let loras = crate::pipeline::lora_files(&request.adapters)?;
            let (ids, valid) = super::mlx::session::prompt_batch(&run.bundle, &request.prompt)?;
            let sigmas = super::schedule(steps, i64::from(grid_w), i64::from(grid_h));
            if sigmas.len() != steps as usize + 1 {
                return Err(WeightError::NotBuilt(format!(
                    "scheduler returned {} sigmas for {steps} steps",
                    sigmas.len()
                ))
                .into());
            }
            if let Some(fidelity) = request.source_fidelity {
                if !fidelity.is_finite() {
                    return Err(EngineError::bad_request(format!(
                        "source_fidelity {fidelity} is not finite"
                    )));
                }
                exec.log(
                    "info",
                    &format!(
                        "flux2_edit source_fidelity={fidelity} is ignored; Klein edit uses reference conditioning (BFL/diffusers)"
                    ),
                )?;
            }
            exec.log(
                "info",
                &format!(
                    "flux2_edit mode=reference_conditioning source={source_w}x{source_h} out={width}x{height} steps={steps} refs=1 t_offset={}",
                    super::mlx::forward::REF_T_SCALE
                ),
            )?;
            Ok(Box::new(Flux2EditSession {
                prepared: Prepared {
                    request: request.clone(),
                    source: (source_w, source_h),
                    width,
                    height,
                    reference,
                    ids,
                    valid,
                    sigmas,
                    bundle: run.bundle.clone(),
                    loras,
                },
            }))
        }
    }
}

struct Flux2EditSession {
    prepared: Prepared,
}

impl FacadeSession for Flux2EditSession {
    fn infer(
        &mut self,
        _ctx: &GgmlContext,
        exec: &ExecutionContext,
    ) -> Result<EngineOutput, EngineError> {
        #[cfg(not(feature = "mlx"))]
        {
            let _ = exec;
            return Err(WeightError::NotBuilt(
                "flux2 image edit no longer has a ggml graph".into(),
            )
            .into());
        }
        #[cfg(feature = "mlx")]
        {
            let prepared = &self.prepared;
            let steps = prepared.request.steps.unwrap_or(4);
            let guidance = prepared.request.guidance.unwrap_or(1.0);
            let count = prepared.request.n.clamp(1, 8);
            let mut output = EngineOutput {
                steps,
                width: Some(prepared.width),
                height: Some(prepared.height),
                ..EngineOutput::default()
            };
            let mut items: Vec<Value> = Vec::with_capacity(count as usize);
            let grid_w = (prepared.width / 16) as i32;
            let grid_h = (prepared.height / 16) as i32;
            for index in 0..count {
                exec.check_cancelled()?;
                let seed = prepared.request.seed.unwrap_or(0) + index as i64;
                let noise = crate::runtime::mlx_executor()
                    .map_err(|error| EngineError::message(error.to_string()))?
                    .execute("flux2 edit noise", move |_, stream, _, _| {
                        super::mlx::session::packed_noise(seed, 128, grid_h, grid_w, stream)
                    })
                    .map_err(|error| EngineError::message(error.to_string()))?;
                if noise.len() != prepared.reference.len() {
                    return Err(EngineError::message(format!(
                        "edit noise has {} values, reference has {}",
                        noise.len(),
                        prepared.reference.len()
                    )));
                }
                let latents = super::mlx::session::denoise_latents(
                    &prepared.bundle,
                    &prepared.ids,
                    &prepared.valid,
                    grid_h,
                    grid_w,
                    &noise,
                    Some(&prepared.reference),
                    &prepared.sigmas,
                    prepared.request.guidance,
                    &prepared.loras,
                    exec,
                )?;
                let png = super::mlx::session::decode_png(
                    &prepared.bundle,
                    &latents,
                    grid_w,
                    grid_h,
                    prepared.width,
                    prepared.height,
                )?;
                let metadata = json!({
                    "operation": prepared.request.operation,
                    "prompt": prepared.request.prompt,
                    "title": prepared.request.title,
                    "source_asset_id": prepared.request.source_asset_id,
                    "source_width": prepared.source.0,
                    "source_height": prepared.source.1,
                    "edit_mode": "reference_conditioning",
                    "reference_t_offset": super::mlx::forward::REF_T_SCALE,
                    "source_fidelity": prepared.request.source_fidelity,
                    "steps": steps,
                    "guidance": guidance,
                    "seed": seed,
                    "batch_index": index,
                    "batch_count": count,
                });
                items.push(metadata.clone());
                let mut artifact =
                    Artifact::png(png, prepared.width as i64, prepared.height as i64, metadata);
                artifact.source_action = "edit".into();
                artifact.parent_asset_id = Some(prepared.request.source_asset_id.clone());
                artifact.relation_type = Some(edit_relation_type(&prepared.request));
                output.artifacts.push(artifact);
                output.latent_len = latents.len();
            }
            output.graph_computes = (prepared.sigmas.len().saturating_sub(1) as u32) * count;
            let first = items.first().cloned().unwrap_or_else(|| json!({}));
            let mut metadata = first;
            metadata["items"] = json!(items);
            metadata["request_metadata"] = prepared.request.metadata.clone();
            output.metadata = metadata;
            Ok(output)
        }
    }
}

fn edit_relation_type(request: &ImageEditRequest) -> String {
    if let Some(relation) = request
        .metadata
        .get("relation_type")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        return relation.to_string();
    }
    match request.operation.as_str() {
        "retouch" | "extend" | "upscale" => request.operation.clone(),
        _ => "edit".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{Action, NoopObserver};
    use crate::ports::ResolvePort;
    use crate::runtime::BackendKind;
    use serde_json::json;
    use std::path::PathBuf;

    struct StubResolver;

    impl ResolvePort for StubResolver {
        fn resolve(&self, _model: &str, _action: Action) -> Result<ResolvedRun, EngineError> {
            unreachable!()
        }

        fn resolve_component(
            &self,
            _run: &ResolvedRun,
            _role: &str,
            _model: &str,
        ) -> Result<PathBuf, EngineError> {
            unreachable!()
        }

        fn read_source_audio(
            &self,
            _run: &ResolvedRun,
            _source: &str,
        ) -> Result<Vec<u8>, EngineError> {
            unreachable!()
        }

        fn read_source_image(
            &self,
            _run: &ResolvedRun,
            _asset: &str,
        ) -> Result<(Vec<u8>, u32, u32), EngineError> {
            unreachable!()
        }
    }

    fn request() -> ImageEditRequest {
        ImageEditRequest {
            model: "flux2-klein-4b".into(),
            operation: "rewrite".into(),
            source_asset_id: "ast_src".into(),
            mask_asset_id: None,
            reference_asset_ids: Vec::new(),
            title: "edit".into(),
            prompt: "make it blue".into(),
            negative_prompt: String::new(),
            source_fidelity: None,
            size: None,
            n: 1,
            steps: Some(4),
            guidance: Some(1.0),
            seed: Some(1),
            scheduler: None,
            adapters: Vec::new(),
            metadata: json!({}),
            preview: true,
            priority: "normal".into(),
        }
    }

    fn run() -> ResolvedRun {
        ResolvedRun {
            model: "flux2-klein-4b".into(),
            version: Some("q4-mixed".into()),
            family: "flux2".into(),
            action: Action::ImageEdit,
            bundle: PathBuf::from("missing"),
            backend: BackendKind::Cpu,
        }
    }

    #[test]
    fn edit_facade_rejects_unbuilt_modes_loud() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("edit", std::sync::Arc::new(NoopObserver));
        let family = Flux2EditFamily;
        let negative = family.prepare(
            &ctx,
            &run(),
            &EngineRequest::ImageEdit(ImageEditRequest {
                negative_prompt: "blurry".into(),
                ..request()
            }),
            &StubResolver,
            &exec,
        );
        let err = match negative {
            Err(error) => error,
            Ok(_) => panic!("negative prompt was accepted"),
        };
        assert!(err.to_string().contains("negative prompt"), "{err}");
        let retouch = family.prepare(
            &ctx,
            &run(),
            &EngineRequest::ImageEdit(ImageEditRequest {
                operation: "retouch".into(),
                ..request()
            }),
            &StubResolver,
            &exec,
        );
        let err = match retouch {
            Err(error) => error,
            Ok(_) => panic!("retouch operation was accepted"),
        };
        assert!(err.to_string().contains("retouch"), "{err}");
    }

    #[test]
    fn driver_routes_flux2_edit_to_the_facade() {
        crate::session::ensure_available("flux2", Action::ImageEdit).unwrap();
        let other = crate::session::ensure_available("fibo", Action::ImageEdit);
        assert!(other.is_err());
    }
}
