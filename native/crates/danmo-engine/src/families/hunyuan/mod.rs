//! HunyuanVideo 1.5 i2v family. The jayn7 GGUF DiT (native Tencent names:
//! `double_blocks`, `img_in`, `txt_in`, `byt5_in`, `vision_in`, `time_in`,
//! `cond_type_embedding`, `final_layer`), the Tencent video VAE safetensors,
//! the Qwen2.5-VL-7B text encoder safetensors, and the FLUX.1-Redux SigLIP
//! image encoder are all in the registry, but the graphs are not built yet:
//! Qwen2.5-VL encode (chat template, hidden_states[-3], crop 108), SigLIP
//! 384px encode, the 54-block joint DiT, and the causal 3D VAE encode/decode
//! all fail loud. Wired here is the reference prepare contract (main
//! `HunyuanVideoConfig` + registry UI defaults): 720x1280, 81 frames at
//! 24fps ((num_frames - 1) % 4 == 0, pixels snap down to a multiple of
//! vae_scale 16), 50 steps, guidance 6.0 (forced to 1 on the cfg-distilled
//! checkpoint), shift 7.0. Prompts with quoted glyph text ("...", 「...」,
//! 《...》) need the byT5 glyph encoder, which has no published GGUF or
//! safetensors — those prompts fail loud.

use crate::contract::{EngineOutput, EngineRequest, ExecutionContext, ResolvedRun};
use crate::error::EngineError;
use crate::families::{FacadeFamily, FacadeSession};
use crate::ports::ResolvePort;
use crate::runtime::GgmlContext;
use crate::weights::{assert_hunyuan_bundle, WeightError};
use danmo_contracts::VideoEditRequest;

pub mod dit;
pub mod siglip;
pub mod text;
pub mod tokenizer;
pub mod vae;

/// HunyuanVideo 1.5: vae_scale 16, patch (1, 1, 1) — pixels are a multiple of 16.
const PIXEL_STEP: u32 = 16;
/// HunyuanVideo 1.5: temporal VAE scale 4, frames are 4k + 1.
const TEMPORAL_SCALE: u32 = 4;
const DEFAULT_SIZE: &str = "720x1280";
const DEFAULT_FRAMES: u32 = 81;
const DEFAULT_FPS: u32 = 24;
/// main registry UI defaults for the 720p i2v model.
const DEFAULT_STEPS: u32 = 50;
const DEFAULT_GUIDANCE: f32 = 6.0;
const DEFAULT_SHIFT: f32 = 7.0;

pub struct HunyuanVideoEditFamily;

pub struct HunyuanVideoEditSession {
    run: ResolvedRun,
    prepared: Prepared,
}

struct Prepared {
    request: VideoEditRequest,
    distilled: bool,
    width: u32,
    height: u32,
    num_frames: u32,
    fps: u32,
    steps: u32,
    guidance: f32,
    shift: f32,
    latent_frames: u32,
}

/// main `extract_glyph_texts`: quoted segments drive the byT5 glyph encoder.
pub(crate) fn has_glyph_text(prompt: &str) -> bool {
    let bytes = prompt.as_bytes();
    let mut quoted = |open: u8, close: u8| {
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == open {
                if let Some(end) = bytes[i + 1..].iter().position(|&b| b == close) {
                    if end > 0 {
                        return true;
                    }
                    i += end + 2;
                    continue;
                }
            }
            i += 1;
        }
        false
    };
    if quoted(b'"', b'"') {
        return true;
    }
    // Curly quotes and CJK brackets (multi-byte delimiters).
    for (open, close) in [("\u{201C}", "\u{201D}"), ("「", "」"), ("《", "》")] {
        if let Some(start) = prompt.find(open) {
            let rest = &prompt[start + open.len()..];
            if let Some(end) = rest.find(close) {
                if !rest[..end].trim().is_empty() {
                    return true;
                }
            }
        }
    }
    false
}

fn prepare_params(request: &VideoEditRequest, distilled: bool) -> Result<Prepared, EngineError> {
    if request.source_asset_id.trim().is_empty() {
        return Err(EngineError::bad_request(
            "hunyuan i2v (animate) requires source_asset_id",
        ));
    }
    if has_glyph_text(&request.prompt) {
        return Err(WeightError::NotBuilt(
            "quoted glyph text needs the byT5 glyph encoder; byT5 has no published \
             GGUF or safetensors (google/byt5-small ships pytorch_model.bin only), so \
             glyph conditioning is not built — remove the quoted segments"
                .into(),
        )
        .into());
    }
    let size = request.size.as_deref().unwrap_or(DEFAULT_SIZE);
    let (width, height) = danmo_contracts::parse_size(size).map_err(EngineError::InvalidRequest)?;
    let width = width - width % PIXEL_STEP;
    let height = height - height % PIXEL_STEP;
    if width == 0 || height == 0 {
        return Err(EngineError::bad_request(format!(
            "hunyuan video size {size} is too small for vae_scale=16 \
             (need width % {PIXEL_STEP} == 0, height % {PIXEL_STEP} == 0)"
        )));
    }
    let num_frames = request.num_frames.unwrap_or(DEFAULT_FRAMES);
    if num_frames == 0 || (num_frames - 1) % TEMPORAL_SCALE != 0 {
        return Err(EngineError::bad_request(format!(
            "hunyuan requires (num_frames - 1) % {TEMPORAL_SCALE} == 0; got {num_frames} \
             (valid examples: 81, 121 for temporal VAE scale {TEMPORAL_SCALE})"
        )));
    }
    let fps = request.fps.unwrap_or(DEFAULT_FPS);
    if fps == 0 {
        return Err(EngineError::bad_request("hunyuan fps must be at least 1"));
    }
    let steps = request.steps.unwrap_or(DEFAULT_STEPS);
    if steps == 0 {
        return Err(EngineError::bad_request("hunyuan steps must be at least 1"));
    }
    // The cfg-distilled checkpoint does not support CFG (registry
    // supports_guidance=false); guidance is forced to 1.
    let guidance = if distilled {
        1.0
    } else {
        let guidance = request.guidance.unwrap_or(DEFAULT_GUIDANCE);
        if !guidance.is_finite() || guidance <= 0.0 {
            return Err(EngineError::bad_request(format!(
                "hunyuan guidance {guidance} is not usable"
            )));
        }
        guidance
    };
    let shift = request.shift.unwrap_or(DEFAULT_SHIFT);
    if !shift.is_finite() || shift <= 0.0 {
        return Err(EngineError::bad_request(format!(
            "hunyuan shift {shift} is not usable"
        )));
    }
    Ok(Prepared {
        request: request.clone(),
        distilled,
        width,
        height,
        num_frames,
        fps,
        steps,
        guidance,
        shift,
        latent_frames: (num_frames - 1) / TEMPORAL_SCALE + 1,
    })
}

impl FacadeFamily for HunyuanVideoEditFamily {
    fn prepare(
        &self,
        _ctx: &GgmlContext,
        run: &ResolvedRun,
        request: &EngineRequest,
        _resolver: &dyn ResolvePort,
        exec: &ExecutionContext,
    ) -> Result<Box<dyn FacadeSession>, EngineError> {
        let EngineRequest::VideoEdit(request) = request else {
            return Err(EngineError::bad_request(
                "hunyuan video edit received a non-video-edit request",
            ));
        };
        let prepared = prepare_params(request, run.model.contains("distill"))?;
        let _bundle = assert_hunyuan_bundle(&run.bundle)?;
        exec.log(
            "info",
            &format!(
                "hunyuan i2v {} size={}x{} frames={} fps={} latent_frames={} steps={} guidance={} shift={}",
                if prepared.distilled { "distilled" } else { "dev" },
                prepared.width,
                prepared.height,
                prepared.num_frames,
                prepared.fps,
                prepared.latent_frames,
                prepared.steps,
                prepared.guidance,
                prepared.shift,
            ),
        )?;
        Ok(Box::new(HunyuanVideoEditSession {
            run: run.clone(),
            prepared,
        }))
    }
}

impl FacadeSession for HunyuanVideoEditSession {
    fn infer(
        &mut self,
        _ctx: &GgmlContext,
        _exec: &ExecutionContext,
    ) -> Result<EngineOutput, EngineError> {
        let _ = &self.prepared;
        Err(WeightError::NotBuilt(format!(
            "hunyuan i2v graph is not built yet (pending: Qwen2.5-VL text encoder, \
             SigLIP image encoder, the 54-block joint DiT, and the causal 3D VAE \
             encode/decode); bundle {} is valid",
            self.run.bundle.display()
        ))
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{Action, NoopObserver};
    use crate::runtime::BackendKind;
    use std::path::{Path, PathBuf};

    fn request() -> VideoEditRequest {
        VideoEditRequest {
            model: "hunyuan-video-1.5-i2v-720p".into(),
            source_asset_id: "ast_src".into(),
            prompt: "a cat".into(),
            negative_prompt: String::new(),
            size: None,
            steps: None,
            guidance: None,
            shift: None,
            seed: None,
            num_frames: None,
            fps: None,
            metadata: serde_json::Value::Null,
            priority: "normal".into(),
        }
    }

    #[test]
    fn prepare_applies_reference_defaults() {
        let prepared = prepare_params(&request(), false).unwrap();
        assert!(!prepared.distilled);
        assert_eq!((prepared.width, prepared.height), (720, 1280));
        assert_eq!(prepared.num_frames, 81);
        assert_eq!(prepared.latent_frames, 21);
        assert_eq!(prepared.fps, 24);
        assert_eq!(prepared.steps, 50);
        assert_eq!(prepared.guidance, 6.0);
        assert_eq!(prepared.shift, 7.0);
    }

    #[test]
    fn distilled_forces_guidance_one() {
        let mut req = request();
        req.guidance = Some(6.0);
        let prepared = prepare_params(&req, true).unwrap();
        assert!(prepared.distilled);
        assert_eq!(prepared.guidance, 1.0);
    }

    #[test]
    fn prepare_requires_the_source_asset() {
        let mut req = request();
        req.source_asset_id = "  ".into();
        assert!(prepare_params(&req, false).is_err());
    }

    #[test]
    fn glyph_prompts_fail_loud_without_byt5() {
        for prompt in [
            "a sign with \"hello\" on it",
            "a poster with 「猫」",
            "the book 《dream》",
            "curly “quoted” text",
        ] {
            let mut req = request();
            req.prompt = prompt.into();
            let err = match prepare_params(&req, false) {
                Err(err) => err,
                Ok(_) => panic!("glyph prompt {prompt:?} was accepted"),
            };
            assert!(err.to_string().contains("byT5"), "{err}");
        }
        // Empty quotes and unquoted text are fine.
        let mut req = request();
        req.prompt = "a plain prompt, no quotes".into();
        prepare_params(&req, false).unwrap();
    }

    fn params_err(req: &VideoEditRequest) -> EngineError {
        match prepare_params(req, false) {
            Err(err) => err,
            Ok(_) => panic!("prepare_params accepted {req:?}"),
        }
    }

    #[test]
    fn prepare_snaps_pixel_dims_down() {
        let mut req = request();
        req.size = Some("730x1290".into());
        let prepared = prepare_params(&req, false).unwrap();
        assert_eq!((prepared.width, prepared.height), (720, 1280));
        let mut req = request();
        req.size = Some("8x480".into());
        let err = params_err(&req);
        assert!(err.to_string().contains("too small"), "{err}");
    }

    #[test]
    fn prepare_enforces_the_frame_rule() {
        let mut req = request();
        req.num_frames = Some(80);
        let err = params_err(&req);
        assert!(err.to_string().contains("num_frames - 1) % 4"), "{err}");
        for frames in [1u32, 5, 81, 121] {
            let mut req = request();
            req.num_frames = Some(frames);
            let prepared = prepare_params(&req, false).unwrap();
            assert_eq!(prepared.latent_frames, (frames - 1) / 4 + 1);
        }
        let mut req = request();
        req.fps = Some(0);
        let _ = params_err(&req);
        let mut req = request();
        req.shift = Some(0.0);
        let _ = params_err(&req);
    }

    struct Resolver;
    impl ResolvePort for Resolver {
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
    }

    fn run(bundle: PathBuf, model: &str) -> ResolvedRun {
        ResolvedRun {
            model: model.into(),
            version: Some("q4_k".into()),
            family: "hunyuan".into(),
            action: Action::VideoEdit,
            bundle,
            backend: BackendKind::Cpu,
        }
    }

    fn write_bundle(dir: &Path, image_encoder: bool) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("hunyuanvideo1.5_720p_i2v-Q4_K_M.gguf"), []).unwrap();
        let text_encoder = dir.join("text_encoder");
        std::fs::create_dir_all(&text_encoder).unwrap();
        std::fs::write(text_encoder.join("model.safetensors"), []).unwrap();
        let vae = dir.join("vae");
        std::fs::create_dir_all(&vae).unwrap();
        std::fs::write(vae.join("diffusion_pytorch_model.safetensors"), []).unwrap();
        let tokenizer = dir.join("tokenizer");
        std::fs::create_dir_all(&tokenizer).unwrap();
        std::fs::write(tokenizer.join("tokenizer.json"), "{}").unwrap();
        let encoder_slot = if image_encoder {
            let image_encoder_dir = dir.join("image_encoder");
            std::fs::create_dir_all(&image_encoder_dir).unwrap();
            std::fs::write(image_encoder_dir.join("model.safetensors"), []).unwrap();
            r#", "image_encoder": "image_encoder/model.safetensors""#
        } else {
            ""
        };
        std::fs::write(
            dir.join("bundle.manifest.json"),
            format!(
                r#"{{"format":"gguf","quant":"q4_k","files":{{"transformer":"hunyuanvideo1.5_720p_i2v-Q4_K_M.gguf","text_encoder":"text_encoder/model.safetensors","vae":"vae/diffusion_pytorch_model.safetensors"{encoder_slot}}}}}"#
            ),
        )
        .unwrap();
    }

    fn prepare_session(dir: &Path, model: &str) -> Result<Box<dyn FacadeSession>, EngineError> {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("task", std::sync::Arc::new(NoopObserver));
        HunyuanVideoEditFamily.prepare(
            &ctx,
            &run(dir.to_path_buf(), model),
            &EngineRequest::VideoEdit(request()),
            &Resolver,
            &exec,
        )
    }

    #[test]
    fn prepare_requires_the_siglip_component() {
        let dir = std::env::temp_dir().join(format!("danmo-hy-nosiglip-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_bundle(&dir, false);
        let err = match prepare_session(&dir, "hunyuan-video-1.5-i2v-720p") {
            Err(err) => err,
            Ok(_) => panic!("prepare accepted a bundle without the image encoder"),
        };
        assert!(err.to_string().contains("image_encoder"), "{err}");
        write_bundle(&dir, true);
        prepare_session(&dir, "hunyuan-video-1.5-i2v-720p").unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn infer_fails_loud_until_the_graphs_land() {
        let dir = std::env::temp_dir().join(format!("danmo-hy-infer-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_bundle(&dir, true);
        let mut session = prepare_session(&dir, "hunyuan-video-1.5-i2v-720p").unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("task", std::sync::Arc::new(NoopObserver));
        let err = match session.infer(&ctx, &exec) {
            Err(err) => err,
            Ok(_) => panic!("infer ran on the fail-loud skeleton"),
        };
        assert!(err.to_string().contains("not built yet"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
