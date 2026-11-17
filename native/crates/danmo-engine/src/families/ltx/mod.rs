//! LTX 2.3 video family. The t2v session runs main's two-stage
//! `_generate_two_stage` (`session.rs`): Gemma-3 + connectors conditioning,
//! a guided half-resolution stage 1 (dev: four DiT forwards per step with
//! the STG block-28 and modality perturbation gates; distilled: the 8-step
//! `DISTILLED_SIGMAS` slice, one forward per step, no guidance), a spatial
//! x2 latent upscale between stages, and a fixed 3-step full-resolution
//! refine with the distilled stage-2 transformer (dev loads the
//! `transformer_stage2` component, distilled reuses its own DiT); the video
//! VAE decodes pixels and the audio VAE + vocoder decode the 48 kHz stereo
//! track that ffmpeg muxes into the MP4. Prepare contract (main `LTXConfig`
//! / generation_mlx): dev = 28 steps with CFG 3.0, distilled = 8 stage-1
//! steps with guidance forced to 1. Geometry: pixels snap down to a
//! multiple of vae_scale 32 and (num_frames - 1) % 8 == 0 (temporal VAE
//! scale 8).

use crate::contract::{EngineOutput, EngineRequest, ExecutionContext, ResolvedRun};
use crate::error::EngineError;
use crate::families::{FacadeFamily, FacadeSession};
use crate::ports::ResolvePort;
use crate::runtime::{GgmlContext, GraphBuilder, RuntimeError, Tensor};
use crate::weights::{assert_ltx_bundle, WeightError};
use danmo_contracts::VideoGenerationRequest;

pub mod audio;
pub mod connector;
pub mod dit;
pub mod gemma;
pub mod session;
pub mod upscaler;
pub mod vae;

/// Exact tanh-approximation GELU. ggml's fused `gelu` routes F32 through an
/// F16 lookup table (`GGML_GELU_FP16`), which is too imprecise for the host
/// parity checks; `0.5x(1 + tanh(u)) = x * sigmoid(2u)` keeps every step
/// exact F32 elementwise math.
pub(crate) fn gelu_tanh(b: &GraphBuilder, x: Tensor) -> Result<Tensor, RuntimeError> {
    fn nn(tensor: Tensor) -> Result<Tensor, RuntimeError> {
        if tensor.is_null() {
            Err(RuntimeError::Init)
        } else {
            Ok(tensor)
        }
    }
    const COEF: f32 = 0.044715;
    // 2 * sqrt(2 / PI).
    const ARG: f32 = 1.595_769_1;
    let cube = nn(b.mul(nn(b.sqr(x))?, x))?;
    let inner = nn(b.add(x, nn(b.scale(cube, COEF))?))?;
    nn(b.mul(x, nn(b.sigmoid(nn(b.scale(inner, ARG))?))?))
}

/// LTX 2.3: vae_scale 32, patch (1, 1, 1) — pixels are a multiple of 32.
const PIXEL_STEP: u32 = 32;
/// LTX 2.3: temporal VAE scale 8, frames are 8k + 1.
const TEMPORAL_SCALE: u32 = 8;
const DEFAULT_SIZE: &str = "512x768";
const DEFAULT_FRAMES: u32 = 97;
const DEFAULT_FPS: u32 = 24;
/// main UI default for dev (`generation_mlx` CFG default is 3.0).
const DEFAULT_DEV_STEPS: u32 = 28;
const DEFAULT_DEV_GUIDANCE: f32 = 3.0;
/// Distilled stage 1 always runs the full 8-step schedule; stage 2 adds the
/// fixed 3 refine steps (`ltx_stage2_steps`) when the two-stage session lands.
const DEFAULT_DISTILLED_STEPS: u32 = 8;

pub struct LtxVideoCreateFamily;

pub struct LtxVideoCreateSession {
    run: ResolvedRun,
    prepared: Prepared,
}

pub(crate) struct Prepared {
    pub(crate) request: VideoGenerationRequest,
    pub(crate) distilled: bool,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) num_frames: u32,
    pub(crate) fps: u32,
    pub(crate) steps: u32,
    pub(crate) guidance: f32,
    pub(crate) latent_frames: u32,
}

fn prepare_params(
    request: &VideoGenerationRequest,
    distilled: bool,
) -> Result<Prepared, EngineError> {
    let size = request.size.as_deref().unwrap_or(DEFAULT_SIZE);
    let (width, height) = danmo_contracts::parse_size(size).map_err(EngineError::InvalidRequest)?;
    let width = width - width % PIXEL_STEP;
    let height = height - height % PIXEL_STEP;
    if width == 0 || height == 0 {
        return Err(EngineError::bad_request(format!(
            "LTX video size {size} is too small for vae_scale=32 \
             (need width % {PIXEL_STEP} == 0, height % {PIXEL_STEP} == 0)"
        )));
    }
    let num_frames = request.num_frames.unwrap_or(DEFAULT_FRAMES);
    if num_frames == 0 || (num_frames - 1) % TEMPORAL_SCALE != 0 {
        return Err(EngineError::bad_request(format!(
            "LTX requires (num_frames - 1) % {TEMPORAL_SCALE} == 0; got {num_frames} \
             (valid examples: 97, 121 for temporal VAE scale {TEMPORAL_SCALE})"
        )));
    }
    let fps = request.fps.unwrap_or(DEFAULT_FPS);
    if fps == 0 {
        return Err(EngineError::bad_request("LTX fps must be at least 1"));
    }
    let steps = request.steps.unwrap_or(if distilled {
        DEFAULT_DISTILLED_STEPS
    } else {
        DEFAULT_DEV_STEPS
    });
    if steps == 0 {
        return Err(EngineError::bad_request("LTX steps must be at least 1"));
    }
    // The distilled checkpoint does not support CFG (registry
    // supports_guidance=false); main forces guidance to 1.
    let guidance = if distilled {
        1.0
    } else {
        let guidance = request.guidance.unwrap_or(DEFAULT_DEV_GUIDANCE);
        if !guidance.is_finite() || guidance <= 0.0 {
            return Err(EngineError::bad_request(format!(
                "LTX guidance {guidance} is not usable"
            )));
        }
        guidance
    };
    Ok(Prepared {
        request: request.clone(),
        distilled,
        width,
        height,
        num_frames,
        fps,
        steps,
        guidance,
        latent_frames: (num_frames - 1) / TEMPORAL_SCALE + 1,
    })
}

impl FacadeFamily for LtxVideoCreateFamily {
    fn prepare(
        &self,
        _ctx: &GgmlContext,
        run: &ResolvedRun,
        request: &EngineRequest,
        _resolver: &dyn ResolvePort,
        exec: &ExecutionContext,
    ) -> Result<Box<dyn FacadeSession>, EngineError> {
        let EngineRequest::VideoCreate(request) = request else {
            return Err(EngineError::bad_request(
                "ltx video create received a non-video-create request",
            ));
        };
        let prepared = prepare_params(request, run.model.contains("distilled"))?;
        let bundle = assert_ltx_bundle(&run.bundle)?;
        // Both variants run the two-stage session and need the spatial x2
        // latent upscaler. Dev additionally refines with the distilled
        // stage-2 transformer; distilled reuses its own DiT for stage 2.
        if bundle.upscaler.is_empty() {
            return Err(WeightError::Bundle {
                path: run.bundle.display().to_string(),
                detail: "LTX two-stage t2v needs the latent upscaler component".into(),
            }
            .into());
        }
        if !prepared.distilled && bundle.stage2_transformer.is_empty() {
            return Err(WeightError::Bundle {
                path: run.bundle.display().to_string(),
                detail: "dev LTX needs the stage-2 distilled transformer component".into(),
            }
            .into());
        }
        exec.log(
            "info",
            &format!(
                "ltx t2v {} size={}x{} frames={} fps={} latent_frames={} steps={} guidance={}",
                if prepared.distilled {
                    "distilled"
                } else {
                    "dev"
                },
                prepared.width,
                prepared.height,
                prepared.num_frames,
                prepared.fps,
                prepared.latent_frames,
                prepared.steps,
                prepared.guidance,
            ),
        )?;
        Ok(Box::new(LtxVideoCreateSession {
            run: run.clone(),
            prepared,
        }))
    }
}

impl FacadeSession for LtxVideoCreateSession {
    fn infer(
        &mut self,
        ctx: &GgmlContext,
        exec: &ExecutionContext,
    ) -> Result<EngineOutput, EngineError> {
        session::infer(&self.run, &self.prepared, ctx, exec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{Action, NoopObserver};
    use crate::runtime::BackendKind;
    use std::path::{Path, PathBuf};

    fn request() -> VideoGenerationRequest {
        VideoGenerationRequest {
            model: "ltx-2.3-dev".into(),
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
    fn prepare_applies_dev_reference_defaults() {
        let prepared = prepare_params(&request(), false).unwrap();
        assert!(!prepared.distilled);
        assert_eq!((prepared.width, prepared.height), (512, 768));
        assert_eq!(prepared.num_frames, 97);
        assert_eq!(prepared.latent_frames, 13);
        assert_eq!(prepared.fps, 24);
        assert_eq!(prepared.steps, 28);
        assert_eq!(prepared.guidance, 3.0);
    }

    #[test]
    fn prepare_applies_distilled_reference_defaults() {
        let prepared = prepare_params(&request(), true).unwrap();
        assert!(prepared.distilled);
        assert_eq!(prepared.steps, 8);
        assert_eq!(prepared.guidance, 1.0);
        // CFG is unsupported on the distilled checkpoint; a user value is
        // dropped back to 1.
        let mut req = request();
        req.guidance = Some(5.0);
        let prepared = prepare_params(&req, true).unwrap();
        assert_eq!(prepared.guidance, 1.0);
    }

    fn params_err(req: &VideoGenerationRequest) -> EngineError {
        match prepare_params(req, false) {
            Err(err) => err,
            Ok(_) => panic!("prepare_params accepted {req:?}"),
        }
    }

    #[test]
    fn prepare_snaps_pixel_dims_down() {
        let mut req = request();
        req.size = Some("520x780".into());
        let prepared = prepare_params(&req, false).unwrap();
        assert_eq!((prepared.width, prepared.height), (512, 768));
        let mut req = request();
        req.size = Some("16x480".into());
        let err = params_err(&req);
        assert!(err.to_string().contains("too small"), "{err}");
        let mut req = request();
        req.size = Some("abc".into());
        let _ = params_err(&req);
    }

    #[test]
    fn prepare_enforces_ltx_frame_rule() {
        let mut req = request();
        req.num_frames = Some(96);
        let err = params_err(&req);
        assert!(err.to_string().contains("num_frames - 1) % 8"), "{err}");
        for frames in [1u32, 9, 97, 121, 257] {
            let mut req = request();
            req.num_frames = Some(frames);
            let prepared = prepare_params(&req, false).unwrap();
            assert_eq!(prepared.latent_frames, (frames - 1) / 8 + 1);
        }
        let mut req = request();
        req.num_frames = Some(0);
        let _ = params_err(&req);
    }

    #[test]
    fn prepare_rejects_zero_fps_and_steps() {
        let mut req = request();
        req.fps = Some(0);
        let _ = params_err(&req);
        let mut req = request();
        req.steps = Some(0);
        let _ = params_err(&req);
        let mut req = request();
        req.guidance = Some(f32::NAN);
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
            version: Some("bf16".into()),
            family: "ltx".into(),
            action: Action::VideoCreate,
            bundle,
            backend: BackendKind::Cpu,
        }
    }

    fn write_bundle(dir: &Path, components: bool) {
        std::fs::create_dir_all(dir).unwrap();
        for file in [
            "ltx-2.3-22b-dev-BF16.gguf",
            "gemma-3-12b-it-BF16.gguf",
            "ltx_video_vae.safetensors",
            "ltx_audio_vae.safetensors",
            "ltx_embeddings_connectors.safetensors",
        ] {
            std::fs::write(dir.join(file), []).unwrap();
        }
        let mut extra = String::new();
        if components {
            std::fs::write(dir.join("ltx_spatial_upscaler.safetensors"), []).unwrap();
            let stage2 = dir.join("transformer_stage2");
            std::fs::create_dir_all(&stage2).unwrap();
            std::fs::write(stage2.join("ltx-2.3-22b-distilled-BF16.gguf"), []).unwrap();
            extra = r#", "upscaler": "ltx_spatial_upscaler.safetensors", "transformer_stage2": "transformer_stage2/ltx-2.3-22b-distilled-BF16.gguf""#.into();
        }
        std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(
            dir.join("bundle.manifest.json"),
            format!(
                r#"{{"format":"gguf","quant":"bf16","files":{{"transformer":"ltx-2.3-22b-dev-BF16.gguf","text_encoder":"gemma-3-12b-it-BF16.gguf","vae":"ltx_video_vae.safetensors","audio_vae":"ltx_audio_vae.safetensors","connectors":"ltx_embeddings_connectors.safetensors"{extra}}}}}"#
            ),
        )
        .unwrap();
    }

    fn prepare_session(dir: &Path, model: &str) -> Result<Box<dyn FacadeSession>, EngineError> {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("task", std::sync::Arc::new(NoopObserver));
        LtxVideoCreateFamily.prepare(
            &ctx,
            &run(dir.to_path_buf(), model),
            &EngineRequest::VideoCreate(request()),
            &Resolver,
            &exec,
        )
    }

    #[test]
    fn prepare_fails_loud_without_a_bundle() {
        let dir = std::env::temp_dir().join(format!("danmo-ltx-missing-{}", std::process::id()));
        let err = match prepare_session(&dir, "ltx-2.3-dev") {
            Err(err) => err,
            Ok(_) => panic!("prepare accepted a missing bundle"),
        };
        assert!(err.to_string().contains("directory is missing"), "{err}");
    }

    #[test]
    fn dev_prepare_requires_the_two_stage_components() {
        let dir = std::env::temp_dir().join(format!("danmo-ltx-dev-bundle-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_bundle(&dir, false);
        let err = match prepare_session(&dir, "ltx-2.3-dev") {
            Err(err) => err,
            Ok(_) => panic!("dev prepare accepted a bundle without the upscaler"),
        };
        assert!(err.to_string().contains("upscaler"), "{err}");
        write_bundle(&dir, true);
        prepare_session(&dir, "ltx-2.3-dev").unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn distilled_prepare_does_not_require_the_stage2_component() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-ltx-distilled-nostage2-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        write_bundle(&dir, true);
        // Distilled reuses its own DiT for stage 2; the manifest slot is
        // optional. Dev still requires it.
        let manifest = std::fs::read_to_string(dir.join("bundle.manifest.json")).unwrap();
        let manifest = manifest.replace(
            r#", "transformer_stage2": "transformer_stage2/ltx-2.3-22b-distilled-BF16.gguf""#,
            "",
        );
        std::fs::write(dir.join("bundle.manifest.json"), manifest).unwrap();
        prepare_session(&dir, "ltx-2.3-distilled").unwrap();
        let err = match prepare_session(&dir, "ltx-2.3-dev") {
            Err(err) => err,
            Ok(_) => panic!("dev prepare accepted a bundle without the stage-2 transformer"),
        };
        assert!(err.to_string().contains("stage-2"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn distilled_infer_attempts_real_loading() {
        let dir =
            std::env::temp_dir().join(format!("danmo-ltx-distilled-infer-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_bundle(&dir, true);
        let mut session = prepare_session(&dir, "ltx-2.3-distilled").unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("task", std::sync::Arc::new(NoopObserver));
        // The bundle holds empty placeholder files, so infer fails while
        // loading the real weights — not with the old fail-loud skeleton.
        let err = match session.infer(&ctx, &exec) {
            Err(err) => err,
            Ok(_) => panic!("infer ran on empty placeholder weights"),
        };
        assert!(!err.to_string().contains("not built yet"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn distilled_prepare_requires_the_upscaler_component() {
        let dir =
            std::env::temp_dir().join(format!("danmo-ltx-distilled-bundle-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_bundle(&dir, false);
        let err = match prepare_session(&dir, "ltx-2.3-distilled") {
            Err(err) => err,
            Ok(_) => panic!("distilled prepare accepted a bundle without the upscaler"),
        };
        assert!(err.to_string().contains("upscaler"), "{err}");
        write_bundle(&dir, true);
        prepare_session(&dir, "ltx-2.3-distilled").unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
