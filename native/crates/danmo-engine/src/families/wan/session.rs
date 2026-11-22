//! Wan t2v session (Facade). Prepare: apply the reference defaults (TI2V-5B:
//! 40 steps, CFG 5, shift 5, 81 frames at 16fps), enforce the Wan geometry
//! rules ((frames - 1) % 4 == 0, pixel dims snapped down to vae_scale x patch
//! = 32), and check the GGUF bundle + DiT layout. Infer: umT5 positive and
//! (guidance != 1) negative conditioning, one reserved DiT graph replayed per
//! step with the WanFlowUniPC sampler (standard CFG
//! `eps = neg + g * (pos - neg)`), causal 3D VAE decode, and a PNG-sequence
//! MP4 mux with no audio track. Video LoRA is not wired: the video contract
//! has no adapters field yet.

use super::dit;
use super::umt5::{Umt5Config, Umt5Encoder};
use super::vae::WanVaeDecoder;
use super::WAN_SAMPLE_NEG_PROMPT;
use crate::common::ffmpeg;
use crate::common::ops::{non_finite, normal_latent, WanFlowUniPc};
use crate::contract::{Artifact, EngineOutput, EngineRequest, ExecutionContext, ResolvedRun};
use crate::error::EngineError;
use crate::families::{FacadeFamily, FacadeSession};
use crate::ports::ResolvePort;
use crate::runtime::GgmlContext;
use crate::weights::{assert_gguf_bundle, matmul_type, type_block, GgufWeights, WeightError};
use danmo_contracts::VideoGenerationRequest;
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Wan 2.2 TI2V: vae_scale 16 x patch (1, 2, 2) snaps pixels to 32.
const PIXEL_STEP: u32 = 32;
/// Wan 2.2 TI2V-5B: temporal VAE scale 4.
const TEMPORAL_SCALE: u32 = 4;
const DEFAULT_SIZE: &str = "832x480";
const DEFAULT_FRAMES: u32 = 81;
const DEFAULT_FPS: u32 = 16;
const DEFAULT_STEPS: u32 = 40;
const DEFAULT_GUIDANCE: f32 = 5.0;
const DEFAULT_SHIFT: f32 = 5.0;

pub struct WanVideoCreateFamily;

pub struct WanVideoCreateSession {
    run: ResolvedRun,
    prepared: Prepared,
}

struct Prepared {
    request: VideoGenerationRequest,
    negative_prompt: String,
    width: u32,
    height: u32,
    num_frames: u32,
    fps: u32,
    steps: u32,
    guidance: f32,
    shift: f32,
    latent_frames: u32,
    config: dit::DitConfig,
    text_config: Umt5Config,
    text_encoder: Vec<PathBuf>,
}

fn prepare_params(request: &VideoGenerationRequest) -> Result<Prepared, EngineError> {
    let size = request.size.as_deref().unwrap_or(DEFAULT_SIZE);
    let (width, height) = danmo_contracts::parse_size(size).map_err(EngineError::InvalidRequest)?;
    // snap_wan_pixel_dims: floor each side to vae_scale * patch (32).
    let width = width - width % PIXEL_STEP;
    let height = height - height % PIXEL_STEP;
    if width == 0 || height == 0 {
        return Err(EngineError::bad_request(format!(
            "Wan video size {size} is too small for vae_scale=16 and patch_size=(1, 2, 2) \
             (need width % {PIXEL_STEP} == 0, height % {PIXEL_STEP} == 0)"
        )));
    }
    let num_frames = request.num_frames.unwrap_or(DEFAULT_FRAMES);
    if num_frames == 0 || (num_frames - 1) % TEMPORAL_SCALE != 0 {
        return Err(EngineError::bad_request(format!(
            "Wan requires (num_frames - 1) % {TEMPORAL_SCALE} == 0; got {num_frames} \
             (valid examples: 81, 49 for temporal VAE scale {TEMPORAL_SCALE})"
        )));
    }
    let fps = request.fps.unwrap_or(DEFAULT_FPS);
    if fps == 0 {
        return Err(EngineError::bad_request("Wan fps must be at least 1"));
    }
    let steps = request.steps.unwrap_or(DEFAULT_STEPS);
    if steps == 0 {
        return Err(EngineError::bad_request("Wan steps must be at least 1"));
    }
    let guidance = request.guidance.unwrap_or(DEFAULT_GUIDANCE);
    let shift = request.shift.unwrap_or(DEFAULT_SHIFT);
    if !guidance.is_finite() || !shift.is_finite() || shift <= 0.0 {
        return Err(EngineError::bad_request(format!(
            "Wan guidance {guidance} / shift {shift} are not usable"
        )));
    }
    let negative_prompt = if request.negative_prompt.trim().is_empty() {
        WAN_SAMPLE_NEG_PROMPT.to_string()
    } else {
        request.negative_prompt.clone()
    };
    Ok(Prepared {
        request: request.clone(),
        negative_prompt,
        width,
        height,
        num_frames,
        fps,
        steps,
        guidance,
        shift,
        latent_frames: (num_frames - 1) / TEMPORAL_SCALE + 1,
        config: dit::DitConfig {
            dim: 0,
            depth: 0,
            heads: 0,
            ffn_dim: 0,
            in_dim: 0,
            text_dim: 0,
            patch: (1, 2, 2),
        },
        text_config: Umt5Config {
            dim: 0,
            depth: 0,
            heads: 0,
            head_dim: 0,
            buckets: 0,
            ffn: 0,
            vocab: 0,
        },
        text_encoder: Vec::new(),
    })
}

impl FacadeFamily for WanVideoCreateFamily {
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
                "wan video create received a non-video-create request",
            ));
        };
        let mut prepared = prepare_params(request)?;
        let files = assert_gguf_bundle(&run.bundle)?;
        let dit_weights = GgufWeights::open_parts(&files.transformer)?;
        prepared.config = dit::DitConfig::infer(&dit_weights)?;
        let text_encoder = Umt5Encoder::open(&files.text_encoder)?;
        prepared.text_config = text_encoder.config();
        prepared.text_encoder = files.text_encoder.clone();
        exec.log(
            "info",
            &format!(
                "wan t2v size={}x{} frames={} fps={} latent_frames={} steps={} guidance={} shift={} dit=dim{}x{} umt5=dim{}x{}",
                prepared.width,
                prepared.height,
                prepared.num_frames,
                prepared.fps,
                prepared.latent_frames,
                prepared.steps,
                prepared.guidance,
                prepared.shift,
                prepared.config.dim,
                prepared.config.depth,
                prepared.text_config.dim,
                prepared.text_config.depth,
            ),
        )?;
        Ok(Box::new(WanVideoCreateSession {
            run: run.clone(),
            prepared,
        }))
    }
}

struct WorkDir(PathBuf);

impl Drop for WorkDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn check_stored_type(name: &str, ggml_type: u32, ne0: i64) -> Result<(), WeightError> {
    if dit::vector_param(name) {
        // Vectors are widened to F32 in the graph; any stored float works.
        return Ok(());
    }
    if !matmul_type(ggml_type) {
        return Err(WeightError::Type {
            name: name.to_string(),
            ggml_type,
        });
    }
    let block = type_block(ggml_type);
    if block > 1 && ne0 % block != 0 {
        return Err(WeightError::NotBuilt(format!(
            "{name} length {ne0} is not divisible by ggml block {block}"
        )));
    }
    Ok(())
}

fn tokenizer_json(root: &std::path::Path) -> Result<String, WeightError> {
    for candidate in [
        root.join("tokenizer.json"),
        root.join("tokenizer/tokenizer.json"),
    ] {
        if candidate.is_file() {
            return std::fs::read_to_string(&candidate).map_err(|err| WeightError::Open {
                path: candidate.display().to_string(),
                detail: err.to_string(),
            });
        }
    }
    Err(WeightError::NotBuilt(format!(
        "{} holds no tokenizer.json",
        root.display()
    )))
}

impl FacadeSession for WanVideoCreateSession {
    fn infer(
        &mut self,
        ctx: &GgmlContext,
        exec: &ExecutionContext,
    ) -> Result<EngineOutput, EngineError> {
        let p = &self.prepared;
        let config = p.config;
        let files = assert_gguf_bundle(&self.run.bundle)?;
        let tokenizer = tokenizer_json(&files.root)?;

        // ---- text conditioning (umT5, zero-padded to 512) ----
        let encoder = Umt5Encoder::open(&files.text_encoder)?;
        let (pos_txt, _) = encoder.encode(ctx, &tokenizer, &p.request.prompt)?;
        let neg_txt = if p.guidance != 1.0 {
            Some(encoder.encode(ctx, &tokenizer, &p.negative_prompt)?.0)
        } else {
            None
        };

        // ---- DiT graph ----
        let lw = (p.width / 16) as i64;
        let lh = (p.height / 16) as i64;
        let lt = p.latent_frames as i64;
        let grid = (
            lt / config.patch.0,
            lh / config.patch.1,
            lw / config.patch.2,
        );
        let seq = grid.0 * grid.1 * grid.2;
        let dit_weights = GgufWeights::open_parts(&files.transformer)?;
        let layout = dit::weight_layout(config);
        let mut types = HashMap::new();
        for (name, ne0, _) in &layout {
            let stored = dit_weights.ggml_type_of(name)?;
            check_stored_type(name, stored, *ne0)?;
            let graph_type = if dit::vector_param(name) {
                crate::runtime::TYPE_F32
            } else {
                stored
            };
            types.insert(name.clone(), graph_type);
        }
        let mut builder = ctx.builder(64 * 1024 * 1024)?;
        let out = dit::build_dit(&mut builder, config, seq, &types, &[])?;
        let mut graph = builder.finish(out)?;
        for (name, ne0, ne1) in &layout {
            if name == "patch_embedding.weight" {
                let raw = dit_weights.raw_exact(
                    name,
                    &[
                        config.patch.2,
                        config.patch.1,
                        config.patch.0,
                        config.in_dim * config.dim,
                    ],
                )?;
                graph.set_bytes(name, &raw)?;
            } else if dit::vector_param(name) {
                let data = dit_weights.f32_widened(name, &[*ne0, *ne1])?;
                graph.set_f32(name, &data)?;
            } else {
                let raw = dit_weights.raw_exact(name, &[*ne0, *ne1])?;
                graph.set_bytes(name, &raw)?;
            }
        }
        let pos = dit::rope_positions(grid);
        graph.set_i32("pos_0", &pos[0])?;
        graph.set_i32("pos_1", &pos[1])?;
        graph.set_i32("pos_2", &pos[2])?;

        // ---- UniPC denoise, standard CFG ----
        let latent_len = (config.in_dim * lt * lh * lw) as usize;
        let mut latents = normal_latent(p.request.seed.unwrap_or(0), latent_len);
        let mut solver = WanFlowUniPc::new(p.steps, p.shift);
        for i in 0..p.steps as usize {
            exec.check_cancelled()?;
            let tokens = dit::patchify(&latents, config.in_dim, lt, lh, lw, config.patch)?;
            let temb =
                dit::sinusoidal_embedding(dit::DitConfig::FREQ_DIM as usize, solver.timestep(i));
            graph.set_f32("tokens", &tokens)?;
            graph.set_f32("temb", &temb)?;
            graph.set_f32("txt", &pos_txt)?;
            let head = graph.compute()?;
            let v_pos = dit::unpatchify(&head, config.in_dim, grid, config.patch)?;
            let eps = match &neg_txt {
                Some(neg_txt) => {
                    graph.set_f32("txt", neg_txt)?;
                    let head = graph.compute()?;
                    let v_neg = dit::unpatchify(&head, config.in_dim, grid, config.patch)?;
                    v_neg
                        .iter()
                        .zip(&v_pos)
                        .map(|(u, c)| u + p.guidance * (c - u))
                        .collect()
                }
                None => v_pos,
            };
            if let Some(detail) = non_finite("wan dit output", &eps) {
                return Err(WeightError::NotBuilt(detail).into());
            }
            latents = solver
                .step(&eps, &latents)
                .map_err(|msg| EngineError::Resolve(format!("wan unipc: {msg}")))?;
            exec.progress("denoise", (i + 1) as u32, p.steps)?;
        }

        // ---- causal 3D VAE decode: [lw, lh, lt, 48] == [C, T, H, W] bytes ----
        let vae_path = files.vae.first().ok_or_else(|| WeightError::Bundle {
            path: files.root.display().to_string(),
            detail: "vae file list is empty".into(),
        })?;
        let decoder = WanVaeDecoder::open(vae_path)?;
        let decoded = decoder.decode(ctx, &latents, lw, lh, lt)?;
        let frames = decoder.spec().decode_frames(lt);
        if frames != p.num_frames as i64 {
            return Err(WeightError::NotBuilt(format!(
                "wan vae decoded {frames} frames for {} requested",
                p.num_frames
            ))
            .into());
        }

        // ---- PNG sequence -> MP4 (no audio on t2v) ----
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let work =
            std::env::temp_dir().join(format!("danmo-wan-t2v-{}-{nanos}", std::process::id()));
        let _guard = WorkDir(work.clone());
        let frames_dir = work.join("frames");
        std::fs::create_dir_all(&frames_dir)
            .map_err(|err| EngineError::Resolve(format!("create frames dir: {err}")))?;
        let (pw, ph) = (p.width as usize, p.height as usize);
        let plane = pw * ph;
        for f in 0..p.num_frames as usize {
            let mut rgb = vec![0u8; plane * 3];
            for y in 0..ph {
                for x in 0..pw {
                    for c in 0..3 {
                        let v = decoded[((c * p.num_frames as usize + f) * ph + y) * pw + x];
                        rgb[(y * pw + x) * 3 + c] =
                            ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
                    }
                }
            }
            let png = crate::common::png::encode_rgb(p.width, p.height, &rgb);
            std::fs::write(frames_dir.join(format!("frame_{:06}.png", f + 1)), png)
                .map_err(|err| EngineError::Resolve(format!("write frame: {err}")))?;
        }
        let out_path = work.join("video.mp4");
        ffmpeg::mux_png_sequence_to_mp4(
            &frames_dir.join("frame_%06d.png"),
            None,
            &out_path,
            p.fps as f64,
        )?;
        let bytes = std::fs::read(&out_path)
            .map_err(|err| EngineError::Resolve(format!("read muxed mp4: {err}")))?;
        let duration = p.num_frames as f64 / p.fps.max(1) as f64;
        let metadata = json!({
            "model": p.request.model,
            "mime_type": "video/mp4",
            "frame_count": p.num_frames,
            "fps": p.fps,
            "steps": p.steps,
            "guidance": p.guidance,
            "shift": p.shift,
            "seed": p.request.seed,
        });
        let artifact = Artifact::mp4(
            bytes,
            p.width as i64,
            p.height as i64,
            duration,
            metadata.clone(),
        );
        let mut output = EngineOutput {
            steps: p.steps,
            width: Some(p.width),
            height: Some(p.height),
            frames: Some(p.num_frames as i64),
            ..EngineOutput::default()
        };
        let mut full = metadata;
        full["request_metadata"] = p.request.metadata.clone();
        output.metadata = full;
        output.artifacts.push(artifact);
        Ok(output)
    }
}

/// Write a deterministic smoke bundle: a tiny GGUF DiT (in_dim 48 so the
/// latent channel count matches the smoke VAE), the smoke umT5 GGUF, the
/// smoke VAE safetensors, the toy tokenizer.json, and the manifest.
pub fn write_smoke_bundle(dir: &Path) -> Result<(), WeightError> {
    std::fs::create_dir_all(dir).map_err(|err| WeightError::Bundle {
        path: dir.display().to_string(),
        detail: format!("{err}"),
    })?;
    let config = dit::DitConfig {
        dim: 128,
        depth: 2,
        heads: 1,
        ffn_dim: 256,
        in_dim: 48,
        text_dim: 8,
        patch: (1, 2, 2),
    };
    dit::write_smoke(dir, config, None)?;
    super::umt5::write_smoke(dir)?;
    super::vae::write_smoke(dir)?;
    std::fs::write(
        dir.join("tokenizer.json"),
        super::umt5::smoke_tokenizer_json(),
    )
    .map_err(|err| WeightError::Bundle {
        path: dir.display().to_string(),
        detail: format!("tokenizer.json: {err}"),
    })?;
    std::fs::write(
        dir.join("bundle.manifest.json"),
        r#"{"format":"gguf","quant":"bf16","files":{"transformer":"wan-dit.gguf","text_encoder":"umt5_smoke.gguf","vae":"Wan2.2_VAE.safetensors"}}"#,
    )
    .map_err(|err| WeightError::Bundle {
        path: dir.display().to_string(),
        detail: format!("bundle.manifest.json: {err}"),
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> VideoGenerationRequest {
        VideoGenerationRequest {
            model: "wan-2.2-ti2v-5b".into(),
            prompt: "a cat".into(),
            negative_prompt: String::new(),
            size: None,
            steps: None,
            guidance: None,
            shift: None,
            seed: None,
            num_frames: None,
            fps: None,
            media_references: Vec::new(),
            adapters: Vec::new(),
            metadata: serde_json::Value::Null,
            priority: "normal".into(),
        }
    }

    #[test]
    fn prepare_applies_reference_defaults() {
        let prepared = prepare_params(&request()).unwrap();
        assert_eq!((prepared.width, prepared.height), (832, 480));
        assert_eq!(prepared.num_frames, 81);
        assert_eq!(prepared.latent_frames, 21);
        assert_eq!(prepared.fps, 16);
        assert_eq!(prepared.steps, 40);
        assert_eq!(prepared.guidance, 5.0);
        assert_eq!(prepared.shift, 5.0);
        assert_eq!(prepared.negative_prompt, WAN_SAMPLE_NEG_PROMPT);
    }

    fn params_err(req: &VideoGenerationRequest) -> EngineError {
        match prepare_params(req) {
            Err(err) => err,
            Ok(_) => panic!("prepare_params accepted {req:?}"),
        }
    }

    #[test]
    fn prepare_snaps_pixel_dims_down() {
        let mut req = request();
        req.size = Some("480x720".into());
        let prepared = prepare_params(&req).unwrap();
        assert_eq!((prepared.width, prepared.height), (480, 704));
        let mut req = request();
        req.size = Some("16x480".into());
        let err = params_err(&req);
        assert!(err.to_string().contains("too small"), "{err}");
        let mut req = request();
        req.size = Some("abc".into());
        let _ = params_err(&req);
    }

    #[test]
    fn prepare_enforces_wan_frame_rule() {
        let mut req = request();
        req.num_frames = Some(80);
        let err = params_err(&req);
        assert!(err.to_string().contains("num_frames - 1) % 4"), "{err}");
        for frames in [1u32, 5, 49, 81, 129] {
            let mut req = request();
            req.num_frames = Some(frames);
            let prepared = prepare_params(&req).unwrap();
            assert_eq!(prepared.latent_frames, (frames - 1) / 4 + 1);
        }
        let mut req = request();
        req.num_frames = Some(0);
        let _ = params_err(&req);
    }

    #[test]
    fn prepare_keeps_user_negative_prompt() {
        let mut req = request();
        req.negative_prompt = "blurry".into();
        let prepared = prepare_params(&req).unwrap();
        assert_eq!(prepared.negative_prompt, "blurry");
    }

    #[test]
    fn prepare_fails_loud_without_a_bundle() {
        use crate::contract::{Action, NoopObserver, ResolvedRun};
        use crate::runtime::BackendKind;

        let dir = std::env::temp_dir().join(format!("danmo-wan-missing-{}", std::process::id()));
        let run = ResolvedRun {
            model: "wan-2.2-ti2v-5b".into(),
            version: Some("q4_k".into()),
            family: "wan".into(),
            action: Action::VideoCreate,
            bundle: dir,
            backend: BackendKind::Cpu,
        };
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("task", std::sync::Arc::new(NoopObserver));
        let err = match WanVideoCreateFamily.prepare(
            &ctx,
            &run,
            &EngineRequest::VideoCreate(request()),
            &Resolver,
            &exec,
        ) {
            Err(err) => err,
            Ok(_) => panic!("prepare accepted a missing bundle"),
        };
        assert!(err.to_string().contains("directory is missing"), "{err}");
    }

    struct Resolver;
    impl ResolvePort for Resolver {
        fn resolve(
            &self,
            _model: &str,
            _action: crate::contract::Action,
        ) -> Result<ResolvedRun, EngineError> {
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

    #[test]
    fn t2v_runs_end_to_end_on_the_smoke_bundle() {
        use crate::contract::{Action, NoopObserver};
        use crate::runtime::BackendKind;

        if std::process::Command::new("ffmpeg")
            .arg("-version")
            .output()
            .is_err()
        {
            eprintln!("ffmpeg not installed; skipping wan t2v e2e");
            return;
        }
        let dir = std::env::temp_dir().join(format!(
            "danmo-wan-t2v-e2e-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_smoke_bundle(&dir).unwrap();
        let run = ResolvedRun {
            model: "wan-2.2-ti2v-5b".into(),
            version: Some("bf16".into()),
            family: "wan".into(),
            action: Action::VideoCreate,
            bundle: dir.clone(),
            backend: BackendKind::Cpu,
        };
        let mut req = request();
        req.size = Some("32x32".into());
        req.num_frames = Some(5);
        req.fps = Some(4);
        req.steps = Some(2);
        req.guidance = Some(2.0);
        req.seed = Some(3);
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("task", std::sync::Arc::new(NoopObserver));
        let mut session = WanVideoCreateFamily
            .prepare(
                &ctx,
                &run,
                &EngineRequest::VideoCreate(req),
                &Resolver,
                &exec,
            )
            .unwrap();
        let out = session.infer(&ctx, &exec).unwrap();
        assert_eq!(out.steps, 2);
        assert_eq!(out.width, Some(32));
        assert_eq!(out.height, Some(32));
        assert_eq!(out.frames, Some(5));
        assert_eq!(out.artifacts.len(), 1);
        let artifact = &out.artifacts[0];
        assert_eq!(artifact.mime, "video/mp4");
        assert_eq!(artifact.width, Some(32));
        assert_eq!(artifact.height, Some(32));
        assert!((artifact.duration_seconds.unwrap() - 1.25).abs() < 1e-6);
        assert!(artifact.bytes.len() > 64, "mp4 is too small");
        assert_eq!(&artifact.bytes[4..8], b"ftyp", "not an mp4");
        let _ = std::fs::remove_dir_all(&dir);
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
        req.shift = Some(f32::NAN);
        let _ = params_err(&req);
    }
}
