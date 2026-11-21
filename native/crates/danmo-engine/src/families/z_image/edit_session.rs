//! Z-Image edit (operation `rewrite`) on the MLX session.
//!
//! The source image is aligned to 16 and center-cropped, VAE-encoded, blended
//! with noise at `sigma[init_timestep]`, and denoised from that timestep.
//! `init_timestep` is `max(1, int(steps * source_fidelity))` with the reference
//! default fidelity 0.6. The ggml DiT for this action is gone: a GGUF bundle
//! fails loud. Guidance other than 0 or 1, and a negative prompt, fail loud.
//! LoRA uses the same shared residual as create.

use super::geom;
use crate::contract::{Artifact, EngineOutput, EngineRequest, ExecutionContext, ResolvedRun};
use crate::error::EngineError;
use crate::families::{FacadeFamily, FacadeSession};
use crate::ports::ResolvePort;
use crate::runtime::GgmlContext;
use crate::weights::{LoraFile, WeightError};
use danmo_contracts::ImageEditRequest;
use serde_json::{json, Value};
use std::path::PathBuf;

pub struct ZImageEditFamily;

struct Prepared {
    request: ImageEditRequest,
    width: u32,
    height: u32,
    source: (u32, u32),
    reference: Vec<f32>,
    ids: Vec<i32>,
    sigmas: Vec<f32>,
    init: usize,
    bundle: PathBuf,
    loras: Vec<LoraFile>,
}

pub(crate) fn init_timestep(steps: u32, fidelity: f32) -> Result<usize, EngineError> {
    if !fidelity.is_finite() {
        return Err(EngineError::bad_request(format!(
            "source_fidelity {fidelity} is not finite"
        )));
    }
    let fidelity = fidelity.clamp(0.0, 1.0);
    if fidelity <= 0.0 {
        return Ok(0);
    }
    let init = ((steps as f64) * (fidelity as f64)) as usize;
    let init = init.max(1);
    if init as u32 >= steps {
        return Err(EngineError::bad_request(format!(
            "source_fidelity {fidelity} implies init_timestep {init} >= steps {steps}; no denoising steps remain"
        )));
    }
    Ok(init)
}

impl FacadeFamily for ZImageEditFamily {
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
                "z-image edit received a non-edit request",
            ));
        };
        if request.operation != "rewrite" {
            return Err(WeightError::NotBuilt(format!(
                "z-image edit operation {} is not built",
                request.operation
            ))
            .into());
        }
        if request.prompt.trim().is_empty() {
            return Err(EngineError::bad_request("z-image edit prompt is empty"));
        }
        if request.source_asset_id.trim().is_empty() {
            return Err(EngineError::bad_request(
                "z-image edit source_asset_id is empty",
            ));
        }
        if !request.reference_asset_ids.is_empty() {
            return Err(WeightError::NotBuilt(
                "z-image edit multi reference conditioning is not built".into(),
            )
            .into());
        }
        let steps = request.steps.unwrap_or(4);
        if steps < 2 {
            return Err(EngineError::bad_request(format!(
                "z-image edit steps {steps} is below 2"
            )));
        }
        let fidelity = request.source_fidelity.unwrap_or(0.6);
        let init = init_timestep(steps, fidelity)?;
        if !guidance_is_single_pass(request.guidance) {
            return Err(EngineError::bad_request(
                "MLX Z-Image edit does not apply guidance other than 0 or 1",
            ));
        }
        if !request.negative_prompt.trim().is_empty() {
            return Err(EngineError::bad_request(
                "MLX Z-Image edit does not apply a negative prompt",
            ));
        }
        let loras = crate::pipeline::lora_files(&request.adapters)?;
        if request.seed.is_none() {
            return Err(EngineError::bad_request("MLX Z-Image edit requires a seed"));
        }
        #[cfg(not(feature = "mlx"))]
        {
            let _ = (run, resolver, exec, init, loras);
            return Err(WeightError::NotBuilt(
                "z_image image edit no longer has a ggml graph; build with the mlx feature".into(),
            )
            .into());
        }
        #[cfg(feature = "mlx")]
        {
            if !super::mlx::session::is_mlx_bundle(&run.bundle) {
                return Err(WeightError::NotBuilt(
                    "z_image image edit no longer has a ggml graph; install the safetensors bundle and build with the mlx feature"
                        .into(),
                )
                .into());
            }
            let (rgb, source_w, source_h) =
                resolver.read_source_image(run, &request.source_asset_id)?;
            let (rgb, width, height) =
                crate::common::image::center_crop_align(&rgb, source_w, source_h, 16)?;
            let image = rgb_to_n11_nchw(&rgb, width, height)?;
            let reference = super::mlx::session::encode_image(&run.bundle, &image, width, height)?;
            let config = super::mlx::MlxZImageConfig::published();
            let grid = geom::token_grid(width, height, 1, config.vae_scale)?;
            let expect = (grid.latent_w * grid.latent_h * config.in_channels) as usize;
            if reference.len() != expect {
                return Err(WeightError::NotBuilt(format!(
                    "vae encode produced {} values but the DiT latent is {expect}",
                    reference.len()
                ))
                .into());
            }
            let ids = config.token_ids(&run.bundle, &request.prompt)?;
            let sigmas = super::schedule(
                super::weights::ZImageConfig::smoke(),
                steps,
                width,
                height,
                request.scheduler.as_deref(),
                None,
                false,
            )
            .map_err(EngineError::from)?;
            // schedule() only reads vae_scale and requires_sigma_shift. The
            // smoke config carries the published values for both.
            if sigmas.len() != steps as usize + 1 {
                return Err(WeightError::NotBuilt(format!(
                    "scheduler returned {} sigmas for {steps} steps",
                    sigmas.len()
                ))
                .into());
            }
            exec.log(
                "info",
                &format!(
                    "z_image_edit source={source_w}x{source_h} out={width}x{height} steps={steps} init={init}"
                ),
            )?;
            Ok(Box::new(ZImageEditSession {
                prepared: Prepared {
                    request: request.clone(),
                    width,
                    height,
                    source: (source_w, source_h),
                    reference,
                    ids,
                    sigmas,
                    init,
                    bundle: run.bundle.clone(),
                    loras,
                },
            }))
        }
    }
}

struct ZImageEditSession {
    prepared: Prepared,
}

impl FacadeSession for ZImageEditSession {
    fn infer(
        &mut self,
        _ctx: &GgmlContext,
        exec: &ExecutionContext,
    ) -> Result<EngineOutput, EngineError> {
        #[cfg(not(feature = "mlx"))]
        {
            let _ = exec;
            return Err(WeightError::NotBuilt(
                "z_image image edit no longer has a ggml graph".into(),
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
            let sigmas = &prepared.sigmas[prepared.init..];
            for index in 0..count {
                exec.check_cancelled()?;
                let seed = prepared.request.seed.unwrap_or(0) + index as i64;
                if seed < 0 {
                    return Err(EngineError::bad_request(
                        "MLX Z-Image seed must be non-negative",
                    ));
                }
                let latents = super::mlx::session::denoise_reference(
                    &prepared.bundle,
                    &prepared.ids,
                    prepared.width,
                    prepared.height,
                    seed as u64,
                    &prepared.reference,
                    sigmas,
                    &prepared.loras,
                    exec,
                )?;
                if let Some(detail) =
                    crate::common::ops::non_finite("z-image-edit latent", &latents)
                {
                    return Err(WeightError::NotBuilt(detail).into());
                }
                let png = super::mlx::session::decode_latent(
                    &prepared.bundle,
                    &latents,
                    prepared.width,
                    prepared.height,
                )?;
                let fidelity = prepared.request.source_fidelity.unwrap_or(0.6);
                let mut metadata = json!({
                    "operation": prepared.request.operation,
                    "prompt": prepared.request.prompt,
                    "negative_prompt": prepared.request.negative_prompt,
                    "title": prepared.request.title,
                    "source_asset_id": prepared.request.source_asset_id,
                    "source_width": prepared.source.0,
                    "source_height": prepared.source.1,
                    "source_fidelity": fidelity,
                    "init_timestep": prepared.init,
                    "steps": steps,
                    "guidance": guidance,
                    "seed": seed,
                    "batch_index": index,
                    "batch_count": count,
                });
                if !prepared.request.adapters.is_empty() {
                    metadata["adapters"] = json!(prepared
                        .request
                        .adapters
                        .iter()
                        .map(|adapter| {
                            json!({
                                "id": adapter.id,
                                "weight": adapter.weight,
                            })
                        })
                        .collect::<Vec<_>>());
                }
                if let Some(mask_id) = prepared
                    .request
                    .mask_asset_id
                    .as_deref()
                    .map(str::trim)
                    .filter(|id| !id.is_empty())
                {
                    metadata["mask_asset_id"] = json!(mask_id);
                }
                items.push(metadata.clone());
                let mut artifact =
                    Artifact::png(png, prepared.width as i64, prepared.height as i64, metadata);
                artifact.source_action = "edit".into();
                artifact.parent_asset_id = Some(prepared.request.source_asset_id.clone());
                artifact.relation_type = Some(edit_relation_type(&prepared.request));
                output.artifacts.push(artifact);
                output.latent_len = latents.len();
            }
            output.graph_computes = sigmas.len().saturating_sub(1) as u32;
            let first = items.first().cloned().unwrap_or_else(|| json!({}));
            let mut metadata = first;
            metadata["items"] = json!(items);
            metadata["request_metadata"] = prepared.request.metadata.clone();
            output.metadata = metadata;
            Ok(output)
        }
    }
}

/// Canvas edge label: prefer client `metadata.relation_type`, else map
/// known edit ops; rewrite/unknown stay `edit`.
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

fn guidance_is_single_pass(guidance: Option<f32>) -> bool {
    match guidance {
        None => true,
        Some(value) => value.abs() < 1e-6 || (value - 1.0).abs() < 1e-6,
    }
}

fn rgb_to_n11_nchw(rgb: &[u8], width: u32, height: u32) -> Result<Vec<f32>, WeightError> {
    let w = width as usize;
    let h = height as usize;
    if rgb.len() != w * h * 3 {
        return Err(WeightError::Shape {
            name: "rgb".into(),
            expected: vec![(w * h * 3) as i64],
            got: vec![rgb.len() as i64],
        });
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{Action, NoopObserver};
    use crate::ports::ResolvePort;
    use crate::runtime::BackendKind;
    use serde_json::json;
    use std::path::PathBuf;

    struct StubResolver {
        rgb: Vec<u8>,
        width: u32,
        height: u32,
    }

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
            asset: &str,
        ) -> Result<(Vec<u8>, u32, u32), EngineError> {
            if asset != "ast_src" {
                return Err(EngineError::Resolve(format!("unknown asset {asset}")));
            }
            Ok((self.rgb.clone(), self.width, self.height))
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "danmo-z-edit-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn request() -> ImageEditRequest {
        ImageEditRequest {
            model: "z-image".into(),
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

    fn run(dir: PathBuf) -> ResolvedRun {
        ResolvedRun {
            model: "z-image".into(),
            version: Some("q4_k".into()),
            family: "z_image".into(),
            action: Action::ImageEdit,
            bundle: dir,
            backend: BackendKind::Cpu,
        }
    }

    #[test]
    fn init_timestep_follows_the_reference_math() {
        assert_eq!(init_timestep(4, 0.6).unwrap(), 2);
        assert_eq!(init_timestep(50, 0.6).unwrap(), 30);
        assert_eq!(init_timestep(4, 0.0).unwrap(), 0);
        assert_eq!(init_timestep(4, 0.1).unwrap(), 1);
        assert!(init_timestep(4, 1.0).is_err());
    }

    #[test]
    fn edit_facade_rejects_unbuilt_modes_loud() {
        let dir = scratch("reject");
        super::super::write_edit_smoke_bundle(&dir).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("edit", std::sync::Arc::new(NoopObserver));
        let resolver = StubResolver {
            rgb: vec![0u8; 32 * 32 * 3],
            width: 32,
            height: 32,
        };
        let family = ZImageEditFamily;
        let retouch = family.prepare(
            &ctx,
            &run(dir.clone()),
            &EngineRequest::ImageEdit(ImageEditRequest {
                operation: "retouch".into(),
                ..request()
            }),
            &resolver,
            &exec,
        );
        let err = match retouch {
            Err(error) => error,
            Ok(_) => panic!("retouch operation was accepted"),
        };
        assert!(err.to_string().contains("retouch"), "{err}");
        let full = family.prepare(
            &ctx,
            &run(dir.clone()),
            &EngineRequest::ImageEdit(ImageEditRequest {
                source_fidelity: Some(1.0),
                ..request()
            }),
            &resolver,
            &exec,
        );
        let err = match full {
            Err(error) => error,
            Ok(_) => panic!("fidelity 1.0 was accepted"),
        };
        assert!(err.to_string().contains("source_fidelity"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gguf_edit_fails_loud_without_the_ggml_graph() {
        let dir = scratch("gguf");
        super::super::write_edit_smoke_bundle(&dir).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("edit", std::sync::Arc::new(NoopObserver));
        let resolver = StubResolver {
            rgb: vec![0u8; 32 * 32 * 3],
            width: 32,
            height: 32,
        };
        let err = match ZImageEditFamily.prepare(
            &ctx,
            &run(dir.clone()),
            &EngineRequest::ImageEdit(request()),
            &resolver,
            &exec,
        ) {
            Err(error) => error,
            Ok(_) => panic!("gguf edit still ran"),
        };
        assert!(
            err.to_string().contains("no longer has a ggml graph"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn driver_routes_z_image_edit_to_the_facade() {
        crate::session::ensure_available("z_image", Action::ImageEdit).unwrap();
        let other = crate::session::ensure_available("fibo", Action::ImageEdit);
        assert!(other.is_err());
    }
}
