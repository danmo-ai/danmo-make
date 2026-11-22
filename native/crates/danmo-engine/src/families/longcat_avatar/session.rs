//! LongCat-Avatar GenerationFacade: prepare validates the request against
//! the registry contract and resolves the portrait/voice assets; infer runs
//! the whole MLX pipeline (umT5 → Whisper → Avatar DiT 8-step DMD → Wan VAE
//! → ffmpeg mux) on the process `mlx_executor`.
//!
//! `avatar_script` reuses the same action: a request whose `script_text` is
//! set (and no voice track) fails loud — the TTS family is not built, and
//! this product does not silently degrade to an audio-free run.

use crate::contract::{
    Action, Artifact, EngineOutput, EngineRequest, ExecutionContext, ResolvedRun,
};
use crate::error::EngineError;
use crate::families::{FacadeFamily, FacadeSession};
use crate::ports::ResolvePort;
use crate::runtime::GgmlContext;
use danmo_contracts::VideoAvatarRequest;
use serde_json::json;

const DEFAULT_SIZE: &str = "512x512";
const DEFAULT_FRAMES: u32 = 93;
const DEFAULT_FPS: u32 = 25;
const DEFAULT_STEPS: u32 = 8;
const SCHEDULER_SHIFT: f32 = 7.0;
const TEXT_GUIDANCE: f32 = 4.0;
const AUDIO_GUIDANCE: f32 = 4.0;
const NUM_TRAIN_TIMESTEPS: usize = 1000;
/// Whisper consumes 16 kHz mono; other PCM rates are resampled during infer.
const AUDIO_SAMPLE_RATE: u32 = 16_000;

pub struct LongcatAvatarFamily;

#[derive(Clone)]
struct Prepared {
    prompt: String,
    negative_prompt: String,
    width: u32,
    height: u32,
    frames: u32,
    fps: u32,
    steps: u32,
    seed: i64,
    portrait: (Vec<u8>, u32, u32),
    voice_wav: Vec<u8>,
    source_asset_id: String,
    audio_asset_id: String,
}

pub struct LongcatAvatarSession {
    run: ResolvedRun,
    prepared: Prepared,
    /// Original voice-track file for the final mux (resolved in prepare).
    voice_track_path: std::path::PathBuf,
}

fn prepare_params(request: &VideoAvatarRequest) -> Result<Prepared, EngineError> {
    if !request.script_text.trim().is_empty() && request.audio_asset_id.trim().is_empty() {
        // `avatar_script`: TTS synthesis is a separate family that is not
        // built yet. Fail loud instead of degrading to a silent avatar.
        return Err(EngineError::bad_request(
            "longcat_avatar script mode requires a TTS backend, which is not installed; use lip-sync mode with reference_asset_id and audio_asset_id",
        ));
    }
    if request.reference_asset_id.trim().is_empty() {
        return Err(EngineError::bad_request(
            "video avatar requires reference_asset_id (portrait image)",
        ));
    }
    if request.audio_asset_id.trim().is_empty() {
        return Err(EngineError::bad_request(
            "video avatar requires audio_asset_id (voice track)",
        ));
    }
    let size = request.size.as_deref().unwrap_or(DEFAULT_SIZE);
    let (width, height) = danmo_contracts::parse_size(size).map_err(EngineError::InvalidRequest)?;
    if width % 16 != 0 || height % 16 != 0 {
        return Err(EngineError::bad_request(format!(
            "longcat_avatar requires width/height to be multiples of 16; got {width}x{height}"
        )));
    }
    let frames = request.num_frames.unwrap_or(DEFAULT_FRAMES);
    if frames < 5 || (frames - 1) % 4 != 0 {
        return Err(EngineError::bad_request(format!(
            "longcat_avatar requires num_frames >= 5 with (num_frames - 1) % 4 == 0; got {frames}"
        )));
    }
    let fps = request.fps.unwrap_or(DEFAULT_FPS);
    if fps == 0 || fps > 60 {
        return Err(EngineError::bad_request(format!(
            "longcat_avatar fps must be in 1..=60; got {fps}"
        )));
    }
    let steps = request.steps.unwrap_or(DEFAULT_STEPS).clamp(1, 50);
    Ok(Prepared {
        prompt: request.prompt.clone(),
        negative_prompt: request.negative_prompt.clone(),
        width,
        height,
        frames,
        fps,
        steps,
        seed: request.seed.unwrap_or(42),
        portrait: (Vec::new(), 0, 0),
        voice_wav: Vec::new(),
        source_asset_id: request.reference_asset_id.clone(),
        audio_asset_id: request.audio_asset_id.clone(),
    })
}

impl FacadeFamily for LongcatAvatarFamily {
    fn prepare(
        &self,
        _ctx: &GgmlContext,
        run: &ResolvedRun,
        request: &EngineRequest,
        resolver: &dyn ResolvePort,
        exec: &ExecutionContext,
    ) -> Result<Box<dyn FacadeSession>, EngineError> {
        #[cfg(not(feature = "mlx"))]
        {
            let _ = (run, request, resolver, exec);
            return Err(EngineError::GraphNotBuilt {
                family: run.family.clone(),
                shape: "longcat_avatar requires an MLX-enabled build".into(),
            });
        }
        #[cfg(feature = "mlx")]
        {
            let EngineRequest::VideoAvatar(request) = request else {
                return Err(EngineError::bad_request(
                    "longcat_avatar accepts video.avatar requests only",
                ));
            };
            let mut prepared = prepare_params(request)?;
            prepared.portrait = resolver.read_source_image(run, &request.reference_asset_id)?;
            let source = format!("asset:{}", request.audio_asset_id.trim());
            prepared.voice_wav = resolver.read_source_audio(run, &source)?;
            let voice_track_path = resolve_audio_path(resolver, run, &request.audio_asset_id)?;
            super::config::BundleConfig::load(&run.bundle).map_err(EngineError::bad_request)?;
            exec.log(
                "info",
                &format!(
                    "longcat_avatar ati2v {}x{} frames={} fps={} steps={} seed={} shift={SCHEDULER_SHIFT}",
                    prepared.width,
                    prepared.height,
                    prepared.frames,
                    prepared.fps,
                    prepared.steps,
                    prepared.seed,
                ),
            )?;
            Ok(Box::new(LongcatAvatarSession {
                run: run.clone(),
                prepared,
                voice_track_path,
            }))
        }
    }
}

impl FacadeSession for LongcatAvatarSession {
    fn infer(
        &mut self,
        _ctx: &GgmlContext,
        exec: &ExecutionContext,
    ) -> Result<EngineOutput, EngineError> {
        #[cfg(not(feature = "mlx"))]
        {
            let _ = exec;
            Err(EngineError::GraphNotBuilt {
                family: "longcat_avatar".into(),
                shape: "longcat_avatar requires an MLX-enabled build".into(),
            })
        }
        #[cfg(feature = "mlx")]
        {
            let run = self.run.clone();
            let prepared = self.prepared.clone();
            let voice_track_path = self.voice_track_path.clone();
            let exec = exec.clone();
            crate::runtime::mlx_executor()
                .map_err(|error| EngineError::Resolve(error.to_string()))?
                .execute("longcat_avatar infer", move |_, stream, _cache, _events| {
                    Ok(infer_mlx_on_stream(
                        &run,
                        &prepared,
                        &voice_track_path,
                        &exec,
                        stream,
                    ))
                })
                .map_err(|error| EngineError::Resolve(error.to_string()))?
        }
    }
}

#[cfg(feature = "mlx")]
fn infer_mlx_on_stream(
    run: &ResolvedRun,
    prepared: &Prepared,
    voice_track_path: &std::path::Path,
    exec: &ExecutionContext,
    stream: &danmo_mlx::Stream,
) -> Result<EngineOutput, EngineError> {
    use danmo_mlx::{Array, Dtype, RandomKey};

    let mlx_err = |error: danmo_mlx::Error| EngineError::Resolve(error.to_string());
    exec.check_cancelled()?;
    let bundle =
        super::config::BundleConfig::load(&run.bundle).map_err(EngineError::bad_request)?;

    // ---- portrait: cover-resize + center crop to the request size ----
    let (rgb, sw, sh) = &prepared.portrait;
    let scale = (prepared.width as f32 / *sw as f32).max(prepared.height as f32 / *sh as f32);
    let rw = (*sw as f32 * scale).round() as u32;
    let rh = (*sh as f32 * scale).round() as u32;
    let resized = crate::common::image::resize_bicubic(rgb, *sw, *sh, rw, rh)
        .map_err(|error| EngineError::bad_request(error.to_string()))?;
    let off_x = (rw - prepared.width) / 2;
    let off_y = (rh - prepared.height) / 2;
    let mut cropped = vec![0u8; (prepared.width * prepared.height * 3) as usize];
    for y in 0..prepared.height {
        let src = ((off_y + y) * rw + off_x) as usize * 3;
        let dst = (y * prepared.width) as usize * 3;
        cropped[dst..dst + prepared.width as usize * 3]
            .copy_from_slice(&resized[src..src + prepared.width as usize * 3]);
    }
    // [1, 3, 1, H, W] in [-1, 1]
    let mut planar = vec![0f32; (prepared.width * prepared.height * 3) as usize];
    let plane = (prepared.width * prepared.height) as usize;
    for y in 0..prepared.height as usize {
        for x in 0..prepared.width as usize {
            for channel in 0..3usize {
                planar[x + y * prepared.width as usize + channel * plane] =
                    (cropped[(y * prepared.width as usize + x) * 3 + channel] as f32 / 127.5) - 1.0;
            }
        }
    }
    let image = Array::from_f32_slice(
        &planar,
        &[1, 3, 1, prepared.height as i32, prepared.width as i32],
    )
    .map_err(mlx_err)?;

    // ---- voice: PCM16 WAV → 16 kHz mono → Whisper log-mel ----
    let (pcm, sample_rate) = crate::common::wav::read_pcm16_interleaved(&prepared.voice_wav)
        .map_err(EngineError::bad_request)?;
    let mono: Vec<f32> = pcm
        .chunks(2)
        .map(|pair| (pair[0] + pair[1]) * 0.5)
        .collect();
    let mono = if sample_rate == AUDIO_SAMPLE_RATE {
        mono
    } else {
        let resampled = crate::common::wav::resample_mono(&mono, sample_rate, AUDIO_SAMPLE_RATE)
            .map_err(EngineError::bad_request)?;
        exec.log(
            "info",
            &format!(
                "longcat_avatar audio resampled: {sample_rate} Hz → {AUDIO_SAMPLE_RATE} Hz ({} → {} mono samples)",
                mono.len(),
                resampled.len()
            ),
        )?;
        resampled
    };
    let mel = super::audio::whisper_log_mel(&mono, stream).map_err(mlx_err)?;
    exec.log(
        "info",
        &format!(
            "longcat_avatar audio: {} mono samples → mel {:?}",
            mono.len(),
            mel.shape()
        ),
    )?;

    exec.phase("conditioning")?;
    // ---- text conditioning ----
    let text_encoder = super::umt5::Umt5Encoder::open(
        &run.bundle.join("text_encoder"),
        &run.bundle.join("tokenizer"),
    )
    .map_err(mlx_err)?;
    exec.log("info", "longcat_avatar umt5 loaded, encoding prompts")?;
    let positive = text_encoder
        .encode_prompt(
            if prepared.prompt.trim().is_empty() {
                " "
            } else {
                &prepared.prompt
            },
            stream,
        )
        .map_err(|e| EngineError::Resolve(format!("longcat_avatar umt5 positive prompt: {e}")))?;
    exec.log(
        "info",
        &format!(
            "longcat_avatar positive embeds {:?}",
            positive.embeds.shape()
        ),
    )?;
    let negative = text_encoder
        .encode_prompt(&prepared.negative_prompt, stream)
        .map_err(|e| EngineError::Resolve(format!("longcat_avatar umt5 negative prompt: {e}")))?;
    drop(text_encoder);

    // ---- audio conditioning ----
    let whisper = crate::common::whisper::WhisperEncoder::open(&run.bundle.join("audio_encoder"))
        .map_err(mlx_err)?;
    let groups = super::audio::encode_audio_to_groups(&whisper, &mel, stream).map_err(mlx_err)?;
    drop(whisper);
    exec.log(
        "info",
        &format!("longcat_avatar whisper groups: {:?}", groups.shape()),
    )?;
    let mut audio_embs = super::audio::build_avatar_audio_embeddings(&groups, prepared.fps, stream)
        .map_err(|e| EngineError::Resolve(format!("longcat_avatar audio embeddings: {e}")))?;
    // The DiT audio path needs T_audio = 1 + vae_scale * (T_lat_full - 1).
    // T_lat_full already includes the reference latent that replaces the first
    // noise latent in the official AI2V prepare_latents path.
    let t_lat = 1 + (prepared.frames - 1) / 4;
    let required = 1 + 4 * (t_lat as usize - 1);
    let have = audio_embs.shape()[1] as usize;
    exec.log(
        "info",
        &format!(
            "longcat_avatar audio conditioning: groups={} audio_frames={have} required={required} t_lat={t_lat}",
            groups.shape()[1]
        ),
    )?;
    audio_embs =
        super::audio::align_audio_frames(&audio_embs, required, stream).map_err(mlx_err)?;
    let uncond_audio = audio_embs.zeros_like(stream).map_err(mlx_err)?;

    // ---- reference latent ----
    let vae =
        super::vae::WanVae::open(&run.bundle.join("vae"), &bundle.vae, stream).map_err(mlx_err)?;
    let reference = vae.encode_normalized(&image, stream).map_err(mlx_err)?;

    // ---- noise + denoise ----
    exec.phase("denoise")?;
    let latent_h = prepared.height / 8;
    let latent_w = prepared.width / 8;
    let noise = RandomKey::new(prepared.seed as u64)
        .and_then(|key| {
            key.normal(
                &[1, 16, (t_lat - 1) as i32, latent_h as i32, latent_w as i32],
                Dtype::Float32,
                stream,
            )
        })
        .map_err(mlx_err)?;
    let mut latents = Array::concatenate_axis(&[reference, noise], 2, stream).map_err(mlx_err)?;

    let dit = super::transformer::AvatarDit::open(&run.bundle.join("dit"), &bundle.dit)
        .map_err(mlx_err)?;

    let mut sigmas =
        super::conditioning::dmd_distilled_sigmas(NUM_TRAIN_TIMESTEPS, DEFAULT_STEPS as usize);
    super::conditioning::apply_scheduler_shift(&mut sigmas, SCHEDULER_SHIFT);
    if prepared.steps as usize != sigmas.len() {
        return Err(EngineError::bad_request(format!(
            "longcat_avatar DMD distill fixes steps at {DEFAULT_STEPS} (got {})",
            prepared.steps
        )));
    }
    sigmas.push(0.0);

    let text_embeds_cat = Array::concatenate_axis(
        &[negative.embeds.clone(), positive.embeds.clone()],
        0,
        stream,
    )
    .map_err(mlx_err)?;
    let text_lengths = if bundle.dit.text_tokens_zero_pad {
        vec![super::umt5::TEXT_LEN, super::umt5::TEXT_LEN]
    } else {
        vec![negative.valid, positive.valid]
    };
    let text_masks =
        Array::concatenate_axis(&[negative.mask.clone(), positive.mask.clone()], 0, stream)
            .map_err(mlx_err)?;

    let total_steps = sigmas.len() - 1;
    for (index, window) in sigmas.windows(2).enumerate() {
        exec.check_cancelled()?;
        let (from, to) = (window[0], window[1]);
        let timestep_value = from * NUM_TRAIN_TIMESTEPS as f32;
        // 3-pass CFG:
        //   pass 1 batched [neg_text, pos_text] with positive audio
        //   pass 2 negative text with zero audio
        let t_arr =
            Array::from_f32_slice(&[timestep_value, timestep_value], &[2]).map_err(mlx_err)?;
        let latents_2 = Array::concatenate_axis(&[latents.clone(), latents.clone()], 0, stream)
            .map_err(mlx_err)?;
        let audio_2 = Array::concatenate_axis(&[audio_embs.clone(), audio_embs.clone()], 0, stream)
            .map_err(mlx_err)?;
        let pred_2 = dit
            .forward(
                &super::transformer::DitInputs {
                    latents: latents_2,
                    timestep: t_arr,
                    text_embeds: text_embeds_cat.clone(),
                    text_lengths: text_lengths.clone(),
                    text_masks: text_masks.clone(),
                    audio_embs: audio_2,
                    num_cond_latents: 1,
                },
                stream,
            )
            .map_err(mlx_err)?;
        let parts = pred_2.split_sections(&[1], 0, stream).map_err(mlx_err)?;
        let (uncond_text, cond) = (parts[0].clone(), parts[1].clone());
        let uncond_text_lengths = if bundle.dit.text_tokens_zero_pad {
            vec![super::umt5::TEXT_LEN]
        } else {
            vec![negative.valid]
        };
        let t_arr_1 = Array::from_f32_slice(&[timestep_value], &[1]).map_err(mlx_err)?;
        let pred_uncond = dit
            .forward(
                &super::transformer::DitInputs {
                    latents: latents.clone(),
                    timestep: t_arr_1,
                    text_embeds: negative.embeds.clone(),
                    text_lengths: uncond_text_lengths,
                    text_masks: negative.mask.clone(),
                    audio_embs: uncond_audio.clone(),
                    num_cond_latents: 1,
                },
                stream,
            )
            .map_err(mlx_err)?;
        let combined = super::conditioning::disentangled_cfg_combine(
            &cond,
            &uncond_text,
            &pred_uncond,
            TEXT_GUIDANCE,
            AUDIO_GUIDANCE,
            stream,
        )
        .map_err(mlx_err)?;
        let velocity = super::conditioning::flip_velocity(&combined, stream).map_err(mlx_err)?;
        let latent_shape = latents.shape();
        let mut cond_shape = latent_shape.clone();
        cond_shape[2] = 1;
        let cond = latents
            .slice(&[0, 0, 0, 0, 0], &cond_shape, &[1, 1, 1, 1, 1], stream)
            .map_err(mlx_err)?;
        let noise = latents
            .slice(&[0, 0, 1, 0, 0], &latent_shape, &[1, 1, 1, 1, 1], stream)
            .map_err(mlx_err)?;
        let velocity = velocity
            .slice(&[0, 0, 1, 0, 0], &latent_shape, &[1, 1, 1, 1, 1], stream)
            .map_err(mlx_err)?;
        let next_noise = super::conditioning::euler_step(&noise, &velocity, from, to, stream)
            .map_err(mlx_err)?;
        latents = Array::concatenate_axis(&[cond, next_noise], 2, stream).map_err(mlx_err)?;
        exec.progress("denoise", (index + 1) as u32, total_steps as u32)?;
    }
    drop(dit);

    // ---- decode ----
    exec.phase("decode")?;
    let denormalized = vae.denormalize(&latents, stream).map_err(mlx_err)?;
    let video = vae.decode(&denormalized, stream).map_err(mlx_err)?;
    drop(vae);
    video.eval().map_err(mlx_err)?;

    // ---- frames → PNG → MP4 (voice track muxed from the source asset) ----
    let (mut frames_rgb, frames, width, height) =
        video_to_rgb_frames(&video, prepared.frames, stream)?;
    let _ = &mut frames_rgb;
    let work = std::env::temp_dir().join(format!("danmo-longcat-avatar-{}", exec.task_id));
    std::fs::create_dir_all(&work)
        .map_err(|error| EngineError::Persist(format!("create work dir: {error}")))?;
    let cleanup = scopeguard(&work);
    for (index, frame) in frames_rgb.iter().enumerate() {
        let png = crate::common::png::encode_rgb(width, height, frame);
        std::fs::write(work.join(format!("frame_{index:06}.png")), png)
            .map_err(|error| EngineError::Persist(format!("write frame: {error}")))?;
    }
    let mp4_path = work.join("out.mp4");
    crate::common::ffmpeg::mux_png_sequence_to_mp4(
        &work.join("frame_%06d.png"),
        Some(voice_track_path),
        &mp4_path,
        prepared.fps as f64,
    )?;
    let bytes = std::fs::read(&mp4_path)
        .map_err(|error| EngineError::Persist(format!("read mp4: {error}")))?;
    drop(cleanup);
    let _ = std::fs::remove_dir_all(&work);

    let duration = frames as f64 / prepared.fps as f64;
    let metadata = json!({
        "model": run.model,
        "family": "longcat_avatar",
        "action": "avatar",
        "width": width,
        "height": height,
        "frames": frames,
        "fps": prepared.fps,
        "seed": prepared.seed,
        "steps": prepared.steps,
        "text_guidance": TEXT_GUIDANCE,
        "audio_guidance": AUDIO_GUIDANCE,
        "scheduler_shift": SCHEDULER_SHIFT,
        "reference_asset_id": prepared.source_asset_id,
        "audio_asset_id": prepared.audio_asset_id,
    });
    let mut artifact = Artifact::mp4(
        bytes,
        width as i64,
        height as i64,
        duration,
        metadata.clone(),
    );
    artifact.source_action = "avatar".into();
    artifact.parent_asset_id = Some(prepared.source_asset_id.clone());
    artifact.relation_type = Some("avatar".into());
    Ok(EngineOutput {
        family: "longcat_avatar".into(),
        action: Action::VideoAvatar.as_str().into(),
        steps: prepared.steps,
        graph_computes: prepared.steps,
        latent_len: (16 * t_lat * latent_h * latent_w) as usize,
        width: Some(width),
        height: Some(height),
        sample_rate: None,
        frames: Some(frames as i64),
        metadata,
        timings: Vec::new(),
        artifacts: vec![artifact],
    })
}

#[cfg(feature = "mlx")]
fn video_to_rgb_frames(
    video: &danmo_mlx::Array,
    want_frames: u32,
    stream: &danmo_mlx::Stream,
) -> Result<(Vec<Vec<u8>>, u32, u32, u32), EngineError> {
    let shape = video.shape();
    if shape.len() != 5 || shape[1] != 3 {
        return Err(EngineError::Resolve(format!(
            "longcat_avatar decoded video must be [1, 3, T, H, W], got {shape:?}"
        )));
    }
    let (t, h, w) = (shape[2] as u32, shape[3] as u32, shape[4] as u32);
    if t != want_frames {
        return Err(EngineError::Resolve(format!(
            "longcat_avatar decoded {t} frames, expected {want_frames}"
        )));
    }
    let values = video
        .to_vec_f32(stream)
        .map_err(|error| EngineError::Resolve(error.to_string()))?;
    let plane = (h * w) as usize;
    let mut frames = Vec::with_capacity(t as usize);
    for frame in 0..t as usize {
        let mut rgb = vec![0u8; plane * 3];
        for channel in 0..3 {
            for pixel in 0..plane {
                let value = values[((0 * 3 + channel) * t as usize + frame) * plane + pixel];
                rgb[pixel * 3 + channel] = (value * 127.5 + 127.5).round().clamp(0.0, 255.0) as u8;
            }
        }
        frames.push(rgb);
    }
    Ok((frames, t, w, h))
}

/// Muxing needs the source file, not decoded bytes. The resolve port keeps
/// asset persistence behind the host; the family must not invent path
/// layouts, so a missing path is a hard failure.
#[cfg(feature = "mlx")]
fn resolve_audio_path(
    resolver: &dyn ResolvePort,
    run: &ResolvedRun,
    audio_asset_id: &str,
) -> Result<std::path::PathBuf, EngineError> {
    resolver
        .read_source_video_path(run, audio_asset_id)
        .map_err(|_| {
            EngineError::Resolve(format!(
                "longcat_avatar cannot locate the voice track file for asset {audio_asset_id}"
            ))
        })
}

#[cfg(feature = "mlx")]
fn scopeguard(path: &std::path::Path) -> ScopeGuard {
    ScopeGuard(path.to_path_buf())
}

#[cfg(feature = "mlx")]
struct ScopeGuard(std::path::PathBuf);

#[cfg(feature = "mlx")]
impl Drop for ScopeGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
