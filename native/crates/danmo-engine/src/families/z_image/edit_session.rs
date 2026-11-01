//! Z-Image edit (operation `rewrite`): reference img2img. The source image is
//! aligned to 16 and center-cropped, VAE-encoded, blended with noise at
//! `sigma[init_timestep]`, and denoised from that timestep. `init_timestep` is
//! `max(1, int(steps * source_fidelity))` with the reference default fidelity
//! 0.6; fidelity 0 is full text-to-image at the source size.

use super::geom;
use super::transformer::ZImageDenoise;
use super::weights::{config_from_weights, weight_layout, ZImageConfig};
use crate::contract::{Artifact, EngineOutput, EngineRequest, ExecutionContext, ResolvedRun};
use crate::error::EngineError;
use crate::families::{FacadeFamily, FacadeSession};
use crate::ports::ResolvePort;
use crate::runtime::GgmlContext;
use crate::weights::{assert_gguf_bundle, load_lora_factors, GgufWeights, LoraFile, WeightError};
use danmo_contracts::ImageEditRequest;
use serde_json::{json, Value};
use std::collections::HashMap;

pub struct ZImageEditFamily;

struct Prepared {
    request: ImageEditRequest,
    config: ZImageConfig,
    source: (u32, u32),
    width: u32,
    height: u32,
    reference: Vec<f32>,
    txt: Vec<f32>,
    cap_real: i64,
    uncond: Option<Vec<f32>>,
    sigmas: Vec<f32>,
    init: usize,
    loras: Vec<LoraFile>,
}

pub struct ZImageEditSession {
    run: ResolvedRun,
    prepared: Prepared,
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
        ctx: &GgmlContext,
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
        if !request.reference_asset_ids.is_empty() {
            return Err(WeightError::NotBuilt(
                "z-image edit multi reference conditioning is not built".into(),
            )
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
        let files = assert_gguf_bundle(&run.bundle)?;
        let loaded = GgufWeights::open_parts(&files.transformer)?;
        let config = config_from_weights(&loaded)?;
        let (rgb, source_w, source_h) =
            resolver.read_source_image(run, &request.source_asset_id)?;
        let (rgb, width, height) =
            crate::common::image::center_crop_align(&rgb, source_w, source_h, 16)?;
        let steps = request.steps.unwrap_or(4);
        if steps < 2 {
            return Err(EngineError::bad_request(format!(
                "z-image edit steps {steps} is below 2"
            )));
        }
        let fidelity = request.source_fidelity.unwrap_or(0.6);
        let init = init_timestep(steps, fidelity)?;
        let guidance = request.guidance.unwrap_or(1.0);
        let n11 = crate::common::image::rgb_to_n11_planar(&rgb, width, height)?;
        let reference = super::vae::encode(ctx, &files.vae, &n11, width, height)?;
        let grid = geom::token_grid(width, height, 1, config.vae_scale)?;
        let expect = (grid.latent_w * grid.latent_h * config.in_channels) as usize;
        if reference.len() != expect {
            return Err(WeightError::NotBuilt(format!(
                "vae encode produced {} values but the DiT latent is {expect}; check the VAE latent channels against in_channels {}",
                reference.len(),
                config.in_channels
            ))
            .into());
        }
        let (txt, cap_real) = super::encode_caption(
            ctx,
            &run.bundle,
            &files.text_encoder,
            &request.prompt,
            config.cap_dim,
        )?;
        let uncond = super::encode_negative(
            ctx,
            &run.bundle,
            &files.text_encoder,
            &request.negative_prompt,
            config.cap_dim,
            cap_real,
            guidance,
        )?;
        let sigmas = super::schedule(
            config,
            steps,
            width,
            height,
            request.scheduler.as_deref(),
            None,
            false,
        )?;
        let loras = crate::pipeline::lora_files(&request.adapters)?;
        exec.log(
            "info",
            &format!(
                "z_image_edit source={source_w}x{source_h} out={width}x{height} steps={steps} init={init} guidance={guidance}"
            ),
        )?;
        Ok(Box::new(ZImageEditSession {
            run: run.clone(),
            prepared: Prepared {
                request: request.clone(),
                config,
                source: (source_w, source_h),
                width,
                height,
                reference,
                txt,
                cap_real,
                uncond,
                sigmas,
                init,
                loras,
            },
        }))
    }
}

impl FacadeSession for ZImageEditSession {
    fn infer(
        &mut self,
        ctx: &GgmlContext,
        exec: &ExecutionContext,
    ) -> Result<EngineOutput, EngineError> {
        let prepared = &self.prepared;
        let files = assert_gguf_bundle(&self.run.bundle)?;
        let loaded = GgufWeights::open_parts(&files.transformer)?;
        let layout = weight_layout(prepared.config);
        let factors = load_lora_factors(&prepared.loras, &layout)?;
        let mut types = HashMap::new();
        for (name, ne0, _) in &layout {
            let ggml_type = loaded.ggml_type_of(name)?;
            super::check_stored_type(name, ggml_type, *ne0)?;
            types.insert(name.clone(), ggml_type);
        }
        let mut denoise = ZImageDenoise::build(
            ctx,
            prepared.config,
            prepared.width,
            prepared.height,
            prepared.cap_real,
            &types,
            &factors,
            None,
        )?;
        for (name, ne0, ne1) in &layout {
            let raw = loaded.raw_exact(name, &[*ne0, *ne1])?;
            denoise.load_bytes(name, &raw)?;
        }
        for (index, factor) in factors.iter().enumerate() {
            denoise.load_f32(&format!("lora.{index}.a"), &factor.a)?;
            denoise.load_f32(&format!("lora.{index}.b"), &factor.b)?;
        }
        let grid = geom::token_grid(
            prepared.width,
            prepared.height,
            prepared.cap_real,
            prepared.config.vae_scale,
        )?;
        let steps = prepared.request.steps.unwrap_or(4);
        let guidance = prepared.request.guidance.unwrap_or(1.0);
        let sigma_blend = prepared.sigmas[prepared.init];
        let sigmas = &prepared.sigmas[prepared.init..];
        let count = prepared.request.n.clamp(1, 8);
        let mut output = EngineOutput {
            steps,
            width: Some(prepared.width),
            height: Some(prepared.height),
            ..EngineOutput::default()
        };
        let mut items: Vec<Value> = Vec::with_capacity(count as usize);
        for index in 0..count {
            exec.check_cancelled()?;
            let seed = if count > 1 {
                Some(prepared.request.seed.unwrap_or(0) + index as i64)
            } else {
                prepared.request.seed
            };
            let noise = match seed {
                Some(seed) => crate::common::ops::normal_latent(seed, denoise.latent_len()),
                None => vec![0.1f32; denoise.latent_len()],
            };
            let latent: Vec<f32> = prepared
                .reference
                .iter()
                .zip(&noise)
                .map(|(reference, noise)| (1.0 - sigma_blend) * reference + sigma_blend * noise)
                .collect();
            let mut callback_error = None;
            let result = denoise.denoise(
                latent,
                &prepared.txt,
                prepared.uncond.as_deref(),
                sigmas,
                guidance,
                None,
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
            if let Some(detail) = crate::common::ops::non_finite("z-image-edit latent", &latents) {
                return Err(WeightError::NotBuilt(detail).into());
            }
            let png = super::vae::decode(
                ctx,
                &files.vae,
                &latents,
                prepared.config.in_channels,
                grid.latent_w,
                grid.latent_h,
                prepared.width,
                prepared.height,
            )?;
            let fidelity = prepared.request.source_fidelity.unwrap_or(0.6);
            let metadata = json!({
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
            items.push(metadata.clone());
            let mut artifact =
                Artifact::png(png, prepared.width as i64, prepared.height as i64, metadata);
            artifact.source_action = "edit".into();
            artifact.parent_asset_id = Some(prepared.request.source_asset_id.clone());
            artifact.relation_type = Some("edit".into());
            output.artifacts.push(artifact);
            output.latent_len = latents.len();
        }
        output.graph_computes = denoise.graph_computes();
        let first = items.first().cloned().unwrap_or_else(|| json!({}));
        let mut metadata = first;
        metadata["items"] = json!(items);
        metadata["request_metadata"] = prepared.request.metadata.clone();
        output.metadata = metadata;
        Ok(output)
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
                return Err(EngineError::Resolve(format!(
                    "source asset {asset} is missing"
                )));
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
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn request() -> ImageEditRequest {
        ImageEditRequest {
            model: "z-image".into(),
            operation: "rewrite".into(),
            source_asset_id: "ast_src".into(),
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
        assert!(init_timestep(4, f32::NAN).is_err());
    }

    #[test]
    fn edit_facade_blends_the_reference_and_decodes_a_png() {
        let dir = scratch("facade");
        super::super::write_edit_smoke_bundle(&dir).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("edit", std::sync::Arc::new(NoopObserver));
        let resolver = StubResolver {
            rgb: vec![128u8; 32 * 32 * 3],
            width: 32,
            height: 32,
        };
        let family = ZImageEditFamily;
        let mut session = family
            .prepare(
                &ctx,
                &run(dir.clone()),
                &EngineRequest::ImageEdit(request()),
                &resolver,
                &exec,
            )
            .expect("prepare");
        let output = session.infer(&ctx, &exec).expect("infer");
        assert_eq!(output.width, Some(32));
        assert_eq!(output.height, Some(32));
        assert_eq!(output.artifacts.len(), 1);
        let artifact = &output.artifacts[0];
        assert!(artifact.bytes.starts_with(b"\x89PNG"));
        assert_eq!(artifact.parent_asset_id.as_deref(), Some("ast_src"));
        assert_eq!(artifact.relation_type.as_deref(), Some("edit"));
        assert_eq!(output.metadata["init_timestep"], json!(2));
        let fidelity = output.metadata["source_fidelity"].as_f64().unwrap();
        assert!((fidelity - 0.6).abs() < 1e-6, "{fidelity}");
        // steps 4, init 2 -> two denoise passes, one forward each at guidance 1.
        assert!(output.graph_computes >= 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn edit_facade_crops_the_source_to_sixteen() {
        let dir = scratch("crop");
        super::super::write_edit_smoke_bundle(&dir).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("edit", std::sync::Arc::new(NoopObserver));
        let resolver = StubResolver {
            rgb: vec![64u8; 40 * 34 * 3],
            width: 40,
            height: 34,
        };
        let family = ZImageEditFamily;
        let mut session = family
            .prepare(
                &ctx,
                &run(dir.clone()),
                &EngineRequest::ImageEdit(request()),
                &resolver,
                &exec,
            )
            .expect("prepare");
        let output = session.infer(&ctx, &exec).expect("infer");
        assert_eq!(output.width, Some(32));
        assert_eq!(output.height, Some(32));
        let _ = std::fs::remove_dir_all(&dir);
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
    fn driver_routes_z_image_edit_to_the_facade() {
        crate::session::ensure_available("z_image", Action::ImageEdit).unwrap();
        let other = crate::session::ensure_available("fibo", Action::ImageEdit);
        assert!(other.is_err());
    }
}
