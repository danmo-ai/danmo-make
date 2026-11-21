//! Qwen-Image edit.
//!
//! The ggml graph is gone. A safetensors bundle on an MLX build runs one of
//! two paths, chosen by the model card the host copies into `metadata`:
//!
//! - `edit_use_vl_vision` (Qwen-Image-Edit, FireRed): Qwen2.5-VL image tokens
//!   plus packed VAE reference latents. Up to `edit_max_reference_images`
//!   pictures when `edit_plus_multi_image` is set. `source_fidelity` is not
//!   part of this path.
//! - otherwise (`qwen-image` rewrite): fidelity img2img. The source is
//!   resized to the requested size, noised at `sigmas[init]`, and denoised
//!   with the create text tower. `init` is `max(1, steps * fidelity)` unless
//!   fidelity is 0.
//!
//! Lightning is a LoRA on this family; it uses the same adapter route as create.

use crate::contract::{Artifact, EngineOutput, EngineRequest, ExecutionContext, ResolvedRun};
use crate::error::EngineError;
use crate::families::{FacadeFamily, FacadeSession};
use crate::ports::ResolvePort;
use crate::runtime::GgmlContext;
use crate::weights::WeightError;
use danmo_contracts::ImageEditRequest;
use serde_json::{json, Value};

pub struct QwenImageEditFamily;

#[derive(Debug)]
struct Card {
    vl: bool,
    picture_prefix: bool,
    multi: bool,
    max_images: usize,
}

impl FacadeFamily for QwenImageEditFamily {
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
                "qwen image edit received a non-edit request",
            ));
        };
        if request.operation != "rewrite" {
            return Err(WeightError::NotBuilt(format!(
                "qwen image edit operation {} is not built",
                request.operation
            ))
            .into());
        }
        if request.prompt.trim().is_empty() {
            return Err(EngineError::bad_request("qwen image edit prompt is empty"));
        }
        if request.source_asset_id.trim().is_empty() {
            return Err(EngineError::bad_request(
                "qwen image edit source_asset_id is empty",
            ));
        }
        if request
            .mask_asset_id
            .as_deref()
            .is_some_and(|id| !id.trim().is_empty())
        {
            return Err(WeightError::NotBuilt(
                "qwen image edit inpaint mask is not built".into(),
            )
            .into());
        }
        let card = edit_card(&request.metadata)?;
        if !card.multi && !request.reference_asset_ids.is_empty() {
            return Err(WeightError::NotBuilt(
                "qwen image edit multi reference conditioning is not built".into(),
            )
            .into());
        }
        let images = 1 + request.reference_asset_ids.len();
        if images > card.max_images {
            return Err(WeightError::NotBuilt(format!(
                "qwen image edit accepts {} reference images, got {images}",
                card.max_images
            ))
            .into());
        }
        let loras = crate::pipeline::lora_files(&request.adapters)?;
        #[cfg(not(feature = "mlx"))]
        {
            let _ = (run, resolver, exec, card, loras);
            return Err(WeightError::NotBuilt(
                "qwen-image edit no longer has a ggml graph; build with the mlx feature".into(),
            )
            .into());
        }
        #[cfg(feature = "mlx")]
        {
            if !crate::weights::is_safetensors_image_bundle(&run.bundle) {
                return Err(WeightError::NotBuilt(
                    "qwen-image edit no longer has a ggml graph; install the safetensors bundle and build with the mlx feature"
                        .into(),
                )
                .into());
            }
            let mut rgb = Vec::with_capacity(images);
            let (source, source_w, source_h) =
                resolver.read_source_image(run, &request.source_asset_id)?;
            rgb.push((source, source_w, source_h));
            for id in &request.reference_asset_ids {
                rgb.push(resolver.read_source_image(run, id)?);
            }
            let (width, height) = output_size(request.size.as_deref(), source_w, source_h)?;
            let steps = request.steps.unwrap_or(if card.vl { 40 } else { 20 });
            if steps < 1 {
                return Err(EngineError::bad_request(format!(
                    "qwen image edit steps {steps} is below 1"
                )));
            }
            let guidance = request.guidance.unwrap_or(if card.vl { 4.0 } else { 1.0 });
            if !guidance.is_finite() || guidance < 1.0 {
                return Err(EngineError::bad_request(format!(
                    "Qwen-Image guidance {guidance} must be finite and >= 1"
                )));
            }
            if guidance <= 1.0 && !request.negative_prompt.trim().is_empty() {
                return Err(EngineError::bad_request(
                    "Qwen-Image negative prompt requires guidance > 1",
                ));
            }
            let seed = request.seed.ok_or_else(|| {
                EngineError::bad_request("Qwen-Image edit requires a seed")
            })?;
            if seed < 0 {
                return Err(EngineError::bad_request(
                    "Qwen-Image seed must be non-negative",
                ));
            }
            let init = if card.vl {
                if let Some(fidelity) = request.source_fidelity {
                    exec.log(
                        "info",
                        &format!(
                            "qwen_edit source_fidelity={fidelity} is ignored; VL edit uses reference latents"
                        ),
                    )?;
                }
                0
            } else {
                init_timestep(steps, request.source_fidelity.unwrap_or(0.6))?
            };
            let prepared = super::mlx::edit::prepare(
                &run.bundle,
                &rgb,
                &request.prompt,
                &request.negative_prompt,
                card.vl,
                card.picture_prefix,
                (width, height),
                steps,
                init,
                guidance,
                seed,
                request.scheduler.as_deref(),
                loras,
            )?;
            exec.log(
                "info",
                &format!(
                    "qwen_edit mode={} out={width}x{height} source={source_w}x{source_h} refs={images} steps={steps} init={init} guidance={guidance}",
                    if card.vl { "vl_reference" } else { "fidelity" }
                ),
            )?;
            Ok(Box::new(QwenEditSession {
                request: request.clone(),
                prepared,
                vl: card.vl,
            }))
        }
    }
}

struct QwenEditSession {
    request: ImageEditRequest,
    #[cfg(feature = "mlx")]
    prepared: super::mlx::edit::PreparedEdit,
    #[cfg(feature = "mlx")]
    vl: bool,
}

impl FacadeSession for QwenEditSession {
    fn infer(
        &mut self,
        _ctx: &GgmlContext,
        exec: &ExecutionContext,
    ) -> Result<EngineOutput, EngineError> {
        #[cfg(not(feature = "mlx"))]
        {
            let _ = exec;
            return Err(WeightError::NotBuilt(
                "qwen-image edit no longer has a ggml graph".into(),
            )
            .into());
        }
        #[cfg(feature = "mlx")]
        {
            let steps = self.request.steps.unwrap_or(if self.vl { 40 } else { 20 });
            let count = self.request.n.clamp(1, 8);
            let mut output = EngineOutput {
                steps,
                width: Some(self.prepared.width),
                height: Some(self.prepared.height),
                ..EngineOutput::default()
            };
            let mut items = Vec::with_capacity(count as usize);
            for index in 0..count {
                exec.check_cancelled()?;
                let mut job = self.prepared.job.clone();
                job.seed = (self.request.seed.unwrap_or(0) + index as i64) as u64;
                let bundle = self.prepared.bundle.clone();
                let text_key = super::mlx::resident::cache_key(&bundle.join("text_encoder"))
                    .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
                let dit_key = super::mlx::resident::cache_key(&bundle.join("transformer"))
                    .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
                let vae_key = super::mlx::resident::cache_key(&bundle.join("vae"))
                    .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
                let width = self.prepared.width;
                let height = self.prepared.height;
                let grid_w = job.grid_w;
                let grid_h = job.grid_h;
                let ran = job.sigmas.len().saturating_sub(1).saturating_sub(job.init) as u32;
                let exec_job = exec.clone();
                let latents = crate::runtime::mlx_executor()
                    .map_err(|error| WeightError::NotBuilt(error.to_string()))?
                    .execute("qwen_image edit denoise", move |_, stream, cache, _| {
                        super::mlx::edit::denoise(
                            cache, &text_key, &dit_key, &job, &exec_job, stream,
                        )
                    })
                    .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
                let encoded = latents.clone();
                let pixels = crate::runtime::mlx_executor()
                    .map_err(|error| WeightError::NotBuilt(error.to_string()))?
                    .execute("qwen_image edit decode", move |_, stream, cache, _| {
                        super::mlx::session::decode_pixels(
                            cache, &vae_key, &encoded, grid_w, grid_h, width, height, stream,
                        )
                    })
                    .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
                let png = crate::common::png::encode_rgb(width, height, &pixels);
                let metadata = json!({
                    "operation": self.request.operation,
                    "prompt": self.request.prompt,
                    "title": self.request.title,
                    "source_asset_id": self.request.source_asset_id,
                    "source_width": self.prepared.source.0,
                    "source_height": self.prepared.source.1,
                    "edit_mode": if self.vl { "vl_reference" } else { "fidelity" },
                    "source_fidelity": self.request.source_fidelity,
                    "steps": steps,
                    "guidance": self.prepared.job.guidance,
                    "seed": self.request.seed.unwrap_or(0) + index as i64,
                    "batch_index": index,
                    "batch_count": count,
                });
                items.push(metadata.clone());
                let mut artifact = Artifact::png(png, width as i64, height as i64, metadata);
                artifact.source_action = "edit".into();
                artifact.parent_asset_id = Some(self.request.source_asset_id.clone());
                artifact.relation_type = Some(edit_relation_type(&self.request));
                output.artifacts.push(artifact);
                output.latent_len = latents.len();
                output.graph_computes += ran;
            }
            let mut metadata = items.first().cloned().unwrap_or_else(|| json!({}));
            metadata["items"] = json!(items);
            metadata["request_metadata"] = self.request.metadata.clone();
            output.metadata = metadata;
            Ok(output)
        }
    }
}

fn edit_card(metadata: &Value) -> Result<Card, EngineError> {
    let vl = metadata
        .get("edit_use_vl_vision")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if vl {
        match metadata.get("edit_conditioning_latent_concat").and_then(Value::as_bool) {
            Some(true) => {}
            Some(false) => {
                return Err(WeightError::NotBuilt(
                    "qwen image VL edit without latent concat is not built".into(),
                )
                .into());
            }
            None => {
                return Err(EngineError::bad_request(
                    "qwen image VL edit requires latent concat (edit_conditioning_latent_concat) on the model card",
                ));
            }
        }
    }
    let max_images = metadata
        .get("edit_max_reference_images")
        .and_then(Value::as_u64)
        .unwrap_or(1) as usize;
    if max_images < 1 {
        return Err(EngineError::bad_request(
            "edit_max_reference_images must be at least 1",
        ));
    }
    Ok(Card {
        vl,
        picture_prefix: metadata
            .get("edit_use_picture_prefix")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        multi: metadata
            .get("edit_plus_multi_image")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        max_images,
    })
}

fn output_size(
    requested: Option<&str>,
    source_w: u32,
    source_h: u32,
) -> Result<(u32, u32), EngineError> {
    let Some(size) = requested.map(str::trim).filter(|size| !size.is_empty()) else {
        let width = source_w / 16 * 16;
        let height = source_h / 16 * 16;
        if width < 16 || height < 16 {
            return Err(WeightError::NotBuilt(
                "source is smaller than 16px after alignment".into(),
            )
            .into());
        }
        return Ok((width, height));
    };
    let (width, height) = danmo_contracts::parse_size(size).map_err(EngineError::InvalidRequest)?;
    if width % 16 != 0 || height % 16 != 0 || width < 16 || height < 16 {
        return Err(WeightError::NotBuilt(format!(
            "size {width}x{height} is not divisible by 16"
        ))
        .into());
    }
    Ok((width, height))
}

fn init_timestep(steps: u32, fidelity: f32) -> Result<usize, EngineError> {
    if !fidelity.is_finite() {
        return Err(EngineError::bad_request(format!(
            "source_fidelity {fidelity} is not finite"
        )));
    }
    let fidelity = fidelity.clamp(0.0, 1.0);
    if fidelity <= 0.0 {
        return Ok(0);
    }
    // `source_fidelity` is f32. Python truncates `int(steps * fidelity)` in
    // f64, so 20 * 0.35 is 7. The f32 value sits just under 7; nudge before
    // truncating so the published decimal matches.
    let init = ((steps as f64) * (fidelity as f64) + 1e-6) as usize;
    let init = init.max(1);
    if init as u32 >= steps {
        return Err(EngineError::bad_request(format!(
            "source_fidelity {fidelity} implies init_timestep {init} >= steps {steps}; no denoising steps remain"
        )));
    }
    Ok(init)
}

fn edit_relation_type(request: &ImageEditRequest) -> String {
    if let Some(relation) = request
        .metadata
        .get("relation_type")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return relation.to_string();
    }
    "edit".into()
}

#[cfg(test)]
mod tests {
    use super::{edit_card, init_timestep, output_size, QwenImageEditFamily};
    use crate::contract::{Action, EngineRequest, ExecutionContext, NoopObserver, ResolvedRun};
    use crate::families::FacadeFamily;
    use crate::ports::ResolvePort;
    use crate::runtime::{BackendKind, GgmlContext};
    use danmo_contracts::ImageEditRequest;
    use serde_json::json;
    use std::path::PathBuf;

    struct NoResolver;

    impl ResolvePort for NoResolver {
        fn resolve(
            &self,
            model: &str,
            _action: Action,
        ) -> Result<ResolvedRun, crate::error::EngineError> {
            Err(crate::error::EngineError::Resolve(format!(
                "{model} is not resolvable in this test"
            )))
        }
        fn resolve_component(
            &self,
            _run: &ResolvedRun,
            _role: &str,
            _model: &str,
        ) -> Result<PathBuf, crate::error::EngineError> {
            Err(crate::error::EngineError::Resolve(
                "no companion in test".into(),
            ))
        }
        fn read_source_audio(
            &self,
            _run: &ResolvedRun,
            _source: &str,
        ) -> Result<Vec<u8>, crate::error::EngineError> {
            Err(crate::error::EngineError::Resolve(
                "no audio in test".into(),
            ))
        }
    }

    fn request() -> ImageEditRequest {
        ImageEditRequest {
            model: "qwen-image-edit".into(),
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
            steps: Some(2),
            guidance: Some(1.0),
            seed: Some(1),
            scheduler: None,
            adapters: Vec::new(),
            metadata: serde_json::json!({}),
            preview: true,
            priority: "normal".into(),
        }
    }

    fn run() -> ResolvedRun {
        ResolvedRun {
            model: "qwen-image-edit".into(),
            version: Some("q4_k".into()),
            family: "qwen_image".into(),
            action: Action::ImageEdit,
            bundle: PathBuf::from("qwen-edit"),
            backend: BackendKind::Cpu,
        }
    }

    #[test]
    fn edit_fails_loud_without_the_ggml_graph() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("edit", std::sync::Arc::new(NoopObserver));
        let err = match QwenImageEditFamily.prepare(
            &ctx,
            &run(),
            &EngineRequest::ImageEdit(request()),
            &NoResolver,
            &exec,
        ) {
            Err(error) => error,
            Ok(_) => panic!("ggml edit still ran"),
        };
        assert!(
            err.to_string().contains("no longer has a ggml graph"),
            "{err}"
        );
    }

    #[test]
    fn edit_facade_rejects_unbuilt_modes_loud() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("edit", std::sync::Arc::new(NoopObserver));
        let retouch = QwenImageEditFamily.prepare(
            &ctx,
            &run(),
            &EngineRequest::ImageEdit(ImageEditRequest {
                operation: "retouch".into(),
                ..request()
            }),
            &NoResolver,
            &exec,
        );
        let err = match retouch {
            Err(error) => error,
            Ok(_) => panic!("retouch operation was accepted"),
        };
        assert!(err.to_string().contains("retouch"), "{err}");
        let multi = QwenImageEditFamily.prepare(
            &ctx,
            &run(),
            &EngineRequest::ImageEdit(ImageEditRequest {
                reference_asset_ids: vec!["ast_other".into()],
                ..request()
            }),
            &NoResolver,
            &exec,
        );
        let err = match multi {
            Err(error) => error,
            Ok(_) => panic!("multi reference edit was accepted"),
        };
        assert!(err.to_string().contains("multi reference"), "{err}");
    }

    #[test]
    fn vl_card_accepts_three_references_and_still_needs_a_bundle() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("edit", std::sync::Arc::new(NoopObserver));
        let err = match QwenImageEditFamily.prepare(
            &ctx,
            &run(),
            &EngineRequest::ImageEdit(ImageEditRequest {
                reference_asset_ids: vec!["ast_b".into(), "ast_c".into()],
                metadata: json!({
                    "edit_use_vl_vision": true,
                    "edit_conditioning_latent_concat": true,
                    "edit_use_picture_prefix": true,
                    "edit_plus_multi_image": true,
                    "edit_max_reference_images": 3
                }),
                ..request()
            }),
            &NoResolver,
            &exec,
        ) {
            Err(error) => error,
            Ok(_) => panic!("empty bundle still ran"),
        };
        assert!(
            err.to_string().contains("no longer has a ggml graph"),
            "{err}"
        );
        let card = edit_card(&json!({
            "edit_use_vl_vision": true,
            "edit_conditioning_latent_concat": true,
            "edit_plus_multi_image": true,
            "edit_max_reference_images": 3
        }))
        .unwrap();
        assert!(card.vl && card.multi && card.max_images == 3);
        let missing = edit_card(&json!({"edit_use_vl_vision": true}));
        assert!(missing.unwrap_err().to_string().contains("latent concat"));
    }

    #[test]
    fn fidelity_init_and_output_size_follow_the_card() {
        assert_eq!(init_timestep(20, 0.0).unwrap(), 0);
        assert_eq!(init_timestep(20, 0.35).unwrap(), 7);
        assert!(init_timestep(4, 1.0).is_err());
        assert_eq!(output_size(None, 1000, 500).unwrap(), (992, 496));
        assert_eq!(output_size(Some("1024x768"), 1000, 500).unwrap(), (1024, 768));
        assert!(output_size(Some("1000x768"), 1000, 500).is_err());
    }

    #[test]
    fn driver_routes_qwen_edit_to_the_facade() {
        crate::session::ensure_available("qwen_image", Action::ImageEdit).unwrap();
        let other = crate::session::ensure_available("fibo", Action::ImageEdit);
        assert!(other.is_err());
    }
}
