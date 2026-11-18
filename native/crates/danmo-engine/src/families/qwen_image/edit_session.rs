//! Qwen-Image edit (operation `rewrite`). Facade shape: prepare reads the
//! source image, runs the VL prompt encode and the VAE conditioning encode,
//! and infer replays the zero_cond_t DiT and decodes the PNG. Single
//! reference image only; multi-reference conditioning is not built.

use super::edit::{
    compute_edit_dimensions, preprocess_rgb, resize_rgb, rgb_to_vae, EditDimensions,
};
use super::transformer::QwenImageForward;
use super::weights::{check_layout, config_from_weights, weight_layout, QwenImageConfig};
use crate::contract::{Artifact, EngineOutput, EngineRequest, ExecutionContext, ResolvedRun};
use crate::error::EngineError;
use crate::families::{FacadeFamily, FacadeSession};
use crate::ports::ResolvePort;
use crate::runtime::GgmlContext;
use crate::weights::{assert_gguf_bundle, GgufWeights, WeightError};
use danmo_contracts::ImageEditRequest;
use serde_json::{json, Value};
use std::collections::HashMap;

pub struct QwenImageEditFamily;

struct Prepared {
    request: ImageEditRequest,
    config: QwenImageConfig,
    dims: EditDimensions,
    source: (u32, u32),
    cond_latents: Vec<f32>,
    txt: Vec<f32>,
    txt_tokens: i64,
    uncond: Option<(Vec<f32>, i64)>,
}

pub struct QwenImageEditSession {
    run: ResolvedRun,
    prepared: Prepared,
}

impl FacadeFamily for QwenImageEditFamily {
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
        if !request.reference_asset_ids.is_empty() {
            return Err(WeightError::NotBuilt(
                "qwen image edit multi reference conditioning is not built".into(),
            )
            .into());
        }
        if !request.adapters.is_empty() {
            return Err(WeightError::NotBuilt("lora inference graph is not built".into()).into());
        }
        if request.prompt.trim().is_empty() {
            return Err(EngineError::bad_request("qwen image edit prompt is empty"));
        }
        if request.source_asset_id.trim().is_empty() {
            return Err(EngineError::bad_request(
                "qwen image edit source_asset_id is empty",
            ));
        }
        let files = assert_gguf_bundle(&run.bundle)?;
        let loaded = GgufWeights::open_parts(&files.transformer)?;
        let config = config_from_weights(&loaded)?;
        check_layout(&loaded, config)?;
        if config.in_channels != 64 || config.out_channels != 16 || config.patch != 2 {
            return Err(WeightError::NotBuilt(format!(
                "qwen packed edit needs in_channels 64 out_channels 16 patch 2, got {} {} {}",
                config.in_channels, config.out_channels, config.patch
            ))
            .into());
        }
        let guidance = request.guidance.unwrap_or(4.0);
        if !guidance.is_finite() || guidance < 1.0 {
            return Err(WeightError::NotBuilt(format!(
                "qwen-image edit guidance {guidance} must be a finite value >= 1"
            ))
            .into());
        }
        let (rgb, source_w, source_h) =
            resolver.read_source_image(run, &request.source_asset_id)?;
        let size = request
            .size
            .as_deref()
            .map(danmo_contracts::parse_size)
            .transpose()
            .map_err(EngineError::InvalidRequest)?;
        let dims = compute_edit_dimensions(source_w, source_h, size)?;

        let vl_rgb = resize_rgb(&rgb, source_w, source_h, dims.vl_w, dims.vl_h);
        let vision_input = preprocess_rgb(&vl_rgb, dims.vl_w as usize, dims.vl_h as usize)?;
        let image_tokens = vision_input.grid.t * vision_input.grid.h * vision_input.grid.w / 4;
        let text_encoder = GgufWeights::open_parts(&files.text_encoder)?;
        let text_config = super::text::config_from_weights(&text_encoder)?;
        if text_config.hidden != config.joint_attention_dim {
            return Err(WeightError::NotBuilt(format!(
                "text hidden {} is not the DiT joint width {}",
                text_config.hidden, config.joint_attention_dim
            ))
            .into());
        }
        let vision_config = super::vision::config_from_weights(&text_encoder)?;
        let vision_embeds =
            super::vision::encode(ctx, &text_encoder, vision_config, &vision_input)?;
        let hidden = text_config.hidden;
        let encode_prompt = |prompt: &str| -> Result<(Vec<f32>, i64), EngineError> {
            let (ids, pads) = super::tokenizer::edit_prompt_ids(
                &run.bundle,
                prompt,
                image_tokens,
                text_config.vocab,
            )?;
            let values = super::text::encode_with_vision(
                ctx,
                &text_encoder,
                text_config,
                &ids,
                &vision_embeds,
                &pads,
            )?;
            let tokens = values.len() as i64 / hidden;
            if tokens < 1 || values.len() as i64 != hidden * tokens {
                return Err(WeightError::NotBuilt(format!(
                    "edit text embedding length {} is not a multiple of hidden {}",
                    values.len(),
                    hidden
                ))
                .into());
            }
            Ok((values, tokens))
        };
        let (txt, txt_tokens) = encode_prompt(&request.prompt)?;
        let uncond = if guidance > 1.0 {
            Some(encode_prompt(request.negative_prompt.trim())?)
        } else {
            None
        };
        let seq_txt = txt_tokens.max(uncond.as_ref().map(|(_, tokens)| *tokens).unwrap_or(0));
        let pad = |values: Vec<f32>| {
            let mut padded = vec![0.0; (hidden * seq_txt) as usize];
            padded[..values.len()].copy_from_slice(&values);
            padded
        };
        let uncond = uncond.map(|(values, tokens)| (pad(values), tokens));

        let vae_rgb = resize_rgb(&rgb, source_w, source_h, dims.vae_w, dims.vae_h);
        let vae_input = rgb_to_vae(&vae_rgb);
        let cond_latents =
            super::vae::encode_packed(ctx, &files.vae, &vae_input, dims.vae_w, dims.vae_h)?;
        let expect_cond =
            (config.in_channels * (dims.vae_w / 16) as i64 * (dims.vae_h / 16) as i64) as usize;
        if cond_latents.len() != expect_cond {
            return Err(WeightError::NotBuilt(format!(
                "qwen edit conditioning has {} values, graph expects {expect_cond}",
                cond_latents.len()
            ))
            .into());
        }
        exec.log(
            "info",
            &format!(
                "qwen_image_edit source={source_w}x{source_h} out={}x{} vl={}x{} vae={}x{} image_tokens={image_tokens}",
                dims.out_w, dims.out_h, dims.vl_w, dims.vl_h, dims.vae_w, dims.vae_h
            ),
        )?;
        Ok(Box::new(QwenImageEditSession {
            run: run.clone(),
            prepared: Prepared {
                request: request.clone(),
                config,
                dims,
                source: (source_w, source_h),
                cond_latents,
                txt: pad(txt),
                txt_tokens,
                uncond,
            },
        }))
    }
}

impl FacadeSession for QwenImageEditSession {
    fn infer(
        &mut self,
        ctx: &GgmlContext,
        exec: &ExecutionContext,
    ) -> Result<EngineOutput, EngineError> {
        let prepared = &self.prepared;
        let files = assert_gguf_bundle(&self.run.bundle)?;
        let loaded = GgufWeights::open_parts(&files.transformer)?;
        let layout = weight_layout(prepared.config);
        let mut types = HashMap::new();
        for (name, _, _) in &layout {
            let ggml_type = loaded.ggml_type_of(name)?;
            types.insert(name.clone(), ggml_type);
        }
        let grid_w = (prepared.dims.out_w / 16) as i64;
        let grid_h = (prepared.dims.out_h / 16) as i64;
        let cond_w = (prepared.dims.vae_w / 16) as i64;
        let cond_h = (prepared.dims.vae_h / 16) as i64;
        let seq_txt = prepared.txt.len() as i64 / prepared.config.joint_attention_dim;
        let mut denoise = QwenImageForward::build_edit(
            ctx,
            prepared.config,
            grid_w,
            grid_h,
            cond_w,
            cond_h,
            seq_txt,
            &types,
        )?;
        for (name, ne0, ne1) in &layout {
            let raw = loaded.raw_exact(name, &[*ne0, *ne1])?;
            denoise.load_bytes(name, &raw)?;
        }
        let steps = prepared.request.steps.unwrap_or(40);
        let sigmas = super::schedule(
            steps,
            prepared.dims.out_w,
            prepared.dims.out_h,
            prepared.config.vae_scale.max(1),
            prepared.request.scheduler.as_deref(),
            None,
            false,
        )?;
        let guidance = prepared.request.guidance.unwrap_or(4.0);
        let target = (prepared.config.in_channels * grid_w * grid_h) as usize;
        let count = prepared.request.n.clamp(1, 8);
        let uncond = prepared
            .uncond
            .as_ref()
            .map(|(values, tokens)| (values.as_slice(), *tokens));
        let mut output = EngineOutput {
            steps,
            width: Some(prepared.dims.out_w),
            height: Some(prepared.dims.out_h),
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
            let mut latent = match seed {
                Some(seed) => crate::common::ops::normal_latent(seed, target),
                None => vec![0.1f32; target],
            };
            latent.extend_from_slice(&prepared.cond_latents);
            let mut callback_error = None;
            let result = denoise.denoise(
                &latent,
                &prepared.txt,
                prepared.txt_tokens,
                uncond,
                &sigmas,
                guidance,
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
            if let Some(detail) = crate::common::ops::non_finite("qwen-image-edit latent", &latents)
            {
                return Err(WeightError::NotBuilt(detail).into());
            }
            let png = super::vae::decode_packed(
                ctx,
                &files.vae,
                &latents,
                grid_w,
                grid_h,
                prepared.dims.out_w,
                prepared.dims.out_h,
            )?;
            let metadata = json!({
                "operation": prepared.request.operation,
                "prompt": prepared.request.prompt,
                "negative_prompt": prepared.request.negative_prompt,
                "title": prepared.request.title,
                "source_asset_id": prepared.request.source_asset_id,
                "source_width": prepared.source.0,
                "source_height": prepared.source.1,
                "vl_width": prepared.dims.vl_w,
                "vl_height": prepared.dims.vl_h,
                "vae_width": prepared.dims.vae_w,
                "vae_height": prepared.dims.vae_h,
                "steps": steps,
                "guidance": guidance,
                "seed": seed,
                "batch_index": index,
                "batch_count": count,
            });
            items.push(metadata.clone());
            let mut artifact = Artifact::png(
                png,
                prepared.dims.out_w as i64,
                prepared.dims.out_h as i64,
                metadata,
            );
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
    use crate::error::EngineError;
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
            "danmo-qwen-edit-{name}-{}-{}",
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
            model: "qwen-image-edit".into(),
            operation: "rewrite".into(),
            source_asset_id: "ast_src".into(),
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
            metadata: json!({}),
            preview: true,
            priority: "normal".into(),
        }
    }

    fn run(dir: PathBuf) -> ResolvedRun {
        ResolvedRun {
            model: "qwen-image-edit".into(),
            version: Some("q4_k".into()),
            family: "qwen_image".into(),
            action: Action::ImageEdit,
            bundle: dir,
            backend: BackendKind::Cpu,
        }
    }

    #[test]
    fn edit_facade_encodes_conditions_and_decodes_a_png() {
        let dir = scratch("facade");
        super::super::write_edit_smoke_bundle(&dir).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("edit", std::sync::Arc::new(NoopObserver));
        let resolver = StubResolver {
            rgb: vec![128u8; 48 * 32 * 3],
            width: 48,
            height: 32,
        };
        let family = QwenImageEditFamily;
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
        assert_eq!(output.width, Some(48));
        assert_eq!(output.height, Some(32));
        assert_eq!(output.artifacts.len(), 1);
        let artifact = &output.artifacts[0];
        assert!(artifact.bytes.starts_with(b"\x89PNG"));
        assert_eq!(artifact.parent_asset_id.as_deref(), Some("ast_src"));
        assert_eq!(artifact.relation_type.as_deref(), Some("edit"));
        assert_eq!(artifact.source_action, "edit");
        assert!(output.graph_computes >= 2);
        assert_eq!(output.metadata["source_width"], json!(48));
        assert_eq!(output.metadata["vae_width"], json!(1248));
        let missing = family.prepare(
            &ctx,
            &run(dir.clone()),
            &EngineRequest::ImageEdit(ImageEditRequest {
                source_asset_id: "ast_missing".into(),
                ..request()
            }),
            &resolver,
            &exec,
        );
        assert!(missing.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn edit_facade_rejects_unbuilt_modes_loud() {
        let dir = scratch("reject");
        super::super::write_edit_smoke_bundle(&dir).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("edit", std::sync::Arc::new(NoopObserver));
        let resolver = StubResolver {
            rgb: vec![0u8; 48 * 32 * 3],
            width: 48,
            height: 32,
        };
        let family = QwenImageEditFamily;
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
        let multi = family.prepare(
            &ctx,
            &run(dir.clone()),
            &EngineRequest::ImageEdit(ImageEditRequest {
                reference_asset_ids: vec!["ast_other".into()],
                ..request()
            }),
            &resolver,
            &exec,
        );
        let err = match multi {
            Err(error) => error,
            Ok(_) => panic!("multi reference edit was accepted"),
        };
        assert!(err.to_string().contains("multi reference"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn driver_routes_qwen_edit_to_the_facade() {
        crate::session::ensure_available("qwen_image", Action::ImageEdit).unwrap();
        let other = crate::session::ensure_available("fibo", Action::ImageEdit);
        assert!(other.is_err());
    }
}
