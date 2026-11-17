//! SeedVR2 video upscale session (Facade). Prepare: validate the request,
//! resolve the source video, probe fps, and extract PPM frames with ffmpeg.
//! Infer: per `chunk_frames` chunk (tail chunks repeat the last frame up to
//! 4), preprocess every frame like the image path, tiled 3D VAE encode over
//! the time axis, one Euler step of the MM-DiT at `t = 1000`, tiled decode
//! (`t_dec > t_in` crops, `t_dec < t_in` fails), per-frame crop + color
//! correction, PNG sequence, and an ffmpeg MP4 mux that keeps the source
//! audio track when one exists.

use super::session::{crop_planar, even, load_pos_emb, pad_planar, preprocess};
use super::{color, dit, tiling, vae};
use crate::common::ffmpeg;
use crate::contract::{Artifact, EngineOutput, EngineRequest, ExecutionContext, ResolvedRun};
use crate::error::EngineError;
use crate::families::{FacadeFamily, FacadeSession};
use crate::ports::ResolvePort;
use crate::runtime::GgmlContext;
use crate::weights::{assert_upscale_bundle, GgufWeights, WeightError};
use danmo_contracts::VideoUpscaleRequest;
use serde_json::json;
use std::path::PathBuf;

const VIDEO_SUFFIXES: [&str; 5] = ["mp4", "webm", "mov", "mkv", "m4v"];

pub struct SeedVr2VideoUpscaleFamily;

struct Prepared {
    request: VideoUpscaleRequest,
    work: PathBuf,
    frames_in: PathBuf,
    n_frames: u32,
    fps: f64,
    source_path: PathBuf,
    source: (u32, u32),
    scale: u32,
    denoise: f32,
    chunk_frames: u32,
    true_w: u32,
    true_h: u32,
    pad_w: i64,
    pad_h: i64,
    pos_emb: Vec<f32>,
    txt_len: i64,
}

pub struct SeedVr2VideoUpscaleSession {
    run: ResolvedRun,
    prepared: Prepared,
}

struct WorkDir(PathBuf);

impl Drop for WorkDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

impl FacadeFamily for SeedVr2VideoUpscaleFamily {
    fn prepare(
        &self,
        _ctx: &GgmlContext,
        run: &ResolvedRun,
        request: &EngineRequest,
        resolver: &dyn ResolvePort,
        exec: &ExecutionContext,
    ) -> Result<Box<dyn FacadeSession>, EngineError> {
        let EngineRequest::VideoUpscale(request) = request else {
            return Err(EngineError::bad_request(
                "seedvr2 video upscale received a non-video-upscale request",
            ));
        };
        let scale = request.scale.unwrap_or(2);
        if scale != 2 && scale != 4 {
            return Err(EngineError::bad_request(format!(
                "seedvr2 video upscale scale must be 2 or 4, got {scale}"
            )));
        }
        // The reference treats 0 as "not set" (`denoise or 0.3`).
        let denoise = match request.denoise {
            Some(d) if d != 0.0 => d,
            _ => 0.3,
        };
        if !denoise.is_finite() {
            return Err(EngineError::bad_request(format!(
                "seedvr2 video upscale denoise {denoise} is not finite"
            )));
        }
        let denoise = denoise.clamp(0.0, 1.0);
        let chunk_frames = request
            .temporal_window
            .filter(|&v| v > 0)
            .unwrap_or(5)
            .max(4);
        let max_frames = request.max_frames.filter(|&v| v > 0).unwrap_or(300);
        if request.source_asset_id.trim().is_empty() {
            return Err(EngineError::bad_request(
                "seedvr2 video upscale source_asset_id is empty",
            ));
        }
        let source_path = resolver.read_source_video_path(run, &request.source_asset_id)?;
        let suffix = source_path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        if !VIDEO_SUFFIXES.contains(&suffix.as_str()) {
            return Err(EngineError::bad_request(format!(
                "seedvr2 video upscale expects a video file ({}); got '.{suffix}' on '{}'",
                VIDEO_SUFFIXES.join(", "),
                source_path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            )));
        }
        if !source_path.is_file() {
            return Err(EngineError::Resolve(format!(
                "source video {} is missing",
                source_path.display()
            )));
        }
        let fps = ffmpeg::probe_video_fps(&source_path)?;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let work = std::env::temp_dir().join(format!(
            "danmo-video-upscale-{}-{nanos}",
            std::process::id()
        ));
        let frames_in = work.join("frames_in");
        let n_frames = ffmpeg::extract_ppm_frames(&source_path, &frames_in, max_frames)?;
        if n_frames == 0 {
            let _ = std::fs::remove_dir_all(&work);
            return Err(EngineError::bad_request(format!(
                "no frames extracted from {}",
                source_path.display()
            )));
        }
        let first = std::fs::read(frames_in.join("frame_000001.ppm"))
            .map_err(|err| EngineError::Resolve(format!("read first extracted frame: {err}")))?;
        let (rgb, source_w, source_h) = ffmpeg::decode_ppm(&first)
            .ok_or_else(|| EngineError::Resolve("first extracted frame is not a P6 ppm".into()))?;
        drop(rgb);
        let true_w = even(source_w as u64 * scale as u64)?;
        let true_h = even(source_h as u64 * scale as u64)?;
        let (pos_emb, txt_len, txt_in) = load_pos_emb(&run.bundle)?;
        let dit_weights =
            GgufWeights::open_parts(&assert_upscale_bundle(&run.bundle)?.transformer)?;
        let config = dit::DitConfig::infer(&dit_weights, txt_len)?;
        if config.txt_in_dim != txt_in {
            return Err(WeightError::NotBuilt(format!(
                "pos_emb width {txt_in} disagrees with the DiT txt_in_dim {}",
                config.txt_in_dim
            ))
            .into());
        }
        let pad_w = (16 - true_w as i64 % 16) % 16;
        let pad_h = (16 - true_h as i64 % 16) % 16;
        exec.log(
            "info",
            &format!(
                "seedvr2_video_upscale source={source_w}x{source_h} frames={n_frames} fps={fps:.3} out={true_w}x{true_h} chunk={chunk_frames} scale={scale} denoise={denoise}"
            ),
        )?;
        Ok(Box::new(SeedVr2VideoUpscaleSession {
            run: run.clone(),
            prepared: Prepared {
                request: request.clone(),
                work,
                frames_in,
                n_frames,
                fps,
                source_path,
                source: (source_w, source_h),
                scale,
                denoise,
                chunk_frames,
                true_w,
                true_h,
                pad_w,
                pad_h,
                pos_emb,
                txt_len,
            },
        }))
    }
}

impl FacadeSession for SeedVr2VideoUpscaleSession {
    fn infer(
        &mut self,
        ctx: &GgmlContext,
        exec: &ExecutionContext,
    ) -> Result<EngineOutput, EngineError> {
        let prepared = &self.prepared;
        let _work_guard = WorkDir(prepared.work.clone());
        let frames_out = prepared.work.join("frames_out");
        std::fs::create_dir_all(&frames_out)
            .map_err(|err| EngineError::Resolve(format!("create frames_out: {err}")))?;
        let files = assert_upscale_bundle(&self.run.bundle)?;
        let dit_weights = GgufWeights::open_parts(&files.transformer)?;
        let config = dit::DitConfig::infer(&dit_weights, prepared.txt_len)?;
        let dit_loaded = dit::load(&dit_weights, &config)?;
        let vae_weights = GgufWeights::open_parts(&files.vae)?;
        let spec = vae::VaeSpec::infer(&vae_weights)?;
        if spec.latent != config.vid_channels - 17 {
            return Err(WeightError::NotBuilt(format!(
                "vae latent {} disagrees with the DiT {} channel input (16 noise + latent + mask)",
                spec.latent, config.vid_channels
            ))
            .into());
        }
        let vae_loaded = vae::load(&vae_weights, &spec)?;
        let (pw, ph) = (
            prepared.true_w as i64 + prepared.pad_w,
            prepared.true_h as i64 + prepared.pad_h,
        );
        let (lw, lh) = (pw / 8, ph / 8);
        let (tw, th) = (prepared.true_w as usize, prepared.true_h as usize);
        let plane_lc = (lw * lh) as usize;
        let latent_ch = spec.latent;
        let lc = latent_ch as usize;
        let seed_base = prepared.request.seed.unwrap_or(0);
        let n_frames = prepared.n_frames as usize;
        let chunk = prepared.chunk_frames as usize;
        let mut out_idx = 0usize;
        let mut start = 0usize;
        while start < n_frames {
            exec.check_cancelled()?;
            let end = (start + chunk).min(n_frames);
            let logical = end - start;
            let infer_len = logical.max(4);
            let t0 = std::time::Instant::now();
            // Preprocess the chunk; tail chunks repeat the last frame.
            let mut style_planes: Vec<Vec<f32>> = Vec::with_capacity(infer_len);
            for f in 0..infer_len {
                let idx = (start + f).min(end - 1) + 1;
                let path = prepared.frames_in.join(format!("frame_{idx:06}.ppm"));
                let bytes = std::fs::read(&path).map_err(|err| {
                    EngineError::Resolve(format!("read {}: {err}", path.display()))
                })?;
                let (rgb, fw, fh) = ffmpeg::decode_ppm(&bytes).ok_or_else(|| {
                    EngineError::Resolve(format!("{} is not a P6 ppm", path.display()))
                })?;
                if (fw, fh) != prepared.source {
                    return Err(EngineError::bad_request(format!(
                        "frame {idx} is {fw}x{fh}, the video opened at {}x{}",
                        prepared.source.0, prepared.source.1
                    )));
                }
                let resized = preprocess(
                    &rgb,
                    fw,
                    fh,
                    prepared.true_w,
                    prepared.true_h,
                    prepared.denoise,
                );
                style_planes.push(pad_planar(&resized, tw, th, prepared.pad_w, prepared.pad_h));
            }
            // `[C][F][H][W]` stacking, channel outer (graph input layout).
            let mut stacked = vec![0f32; 3 * infer_len * (pw * ph) as usize];
            let plane_px = (pw * ph) as usize;
            for (f, frame) in style_planes.iter().enumerate() {
                for c in 0..3 {
                    let dst = (c * infer_len + f) * plane_px;
                    stacked[dst..dst + plane_px]
                        .copy_from_slice(&frame[c * plane_px..(c + 1) * plane_px]);
                }
            }
            let latent = tiling::encode(
                ctx,
                vae::encoder_loaded(&vae_loaded),
                &spec,
                &stacked,
                pw,
                ph,
                infer_len as i64,
            )?;
            drop(stacked);
            let t_lat = vae::encode_frames_out(&spec, infer_len as i64);
            let seed = (seed_base + start as i64 * 1009) & 0x7FFF_FFFF;
            let noise = crate::common::ops::normal_latent(seed, plane_lc * t_lat as usize * lc);
            // vid input `[C][T][H][W]`: [noise; latent; mask of ones].
            let mut vid = vec![0f32; plane_lc * t_lat as usize * config.vid_channels as usize];
            let band = plane_lc * t_lat as usize;
            for c in 0..lc {
                vid[c * band..(c + 1) * band].copy_from_slice(&noise[c * band..(c + 1) * band]);
                vid[(lc + c) * band..(lc + c + 1) * band]
                    .copy_from_slice(&latent[c * band..(c + 1) * band]);
            }
            for v in &mut vid[2 * lc * band..] {
                *v = 1.0;
            }
            let velocity = dit::run(
                ctx,
                &config,
                &dit_loaded,
                &vid,
                lw,
                lh,
                t_lat,
                &prepared.pos_emb,
                1000.0,
            )?;
            drop(vid);
            let latents: Vec<f32> = noise.iter().zip(&velocity).map(|(n, v)| n - v).collect();
            if let Some(detail) = crate::common::ops::non_finite("seedvr2 video latent", &latents) {
                return Err(WeightError::NotBuilt(detail).into());
            }
            let decoded = tiling::decode(
                ctx,
                vae::decoder_loaded(&vae_loaded),
                &spec,
                &latents,
                lw,
                lh,
                t_lat,
            )?;
            let t_dec = vae::decode_frames_out(&spec, t_lat) as usize;
            if t_dec < infer_len {
                return Err(WeightError::NotBuilt(format!(
                    "seedvr2 vae decode returned {t_dec} temporal slice(s) for {infer_len} input frame(s); cannot align restored chunk"
                ))
                .into());
            }
            // Crop to the logical frames; color-correct each against its
            // preprocessed source frame.
            let out_plane = pw as usize * ph as usize;
            for f in 0..logical {
                let mut decoded_f = vec![0f32; 3 * out_plane];
                for c in 0..3 {
                    let src = (c * t_dec + f) * out_plane;
                    decoded_f[c * out_plane..(c + 1) * out_plane]
                        .copy_from_slice(&decoded[src..src + out_plane]);
                }
                let decoded_c = crop_planar(&decoded_f, pw as usize, ph as usize, tw, th);
                let style_c = crop_planar(&style_planes[f], pw as usize, ph as usize, tw, th);
                let corrected = color::correct(&decoded_c, &style_c, tw, th)?;
                let mut rgb = vec![0u8; tw * th * 3];
                for y in 0..th {
                    for x in 0..tw {
                        for c in 0..3 {
                            let unit = (corrected[x + y * tw + c * tw * th] + 1.0) * 0.5;
                            rgb[(y * tw + x) * 3 + c] =
                                (unit.clamp(0.0, 1.0) * 255.0).round() as u8;
                        }
                    }
                }
                out_idx += 1;
                let png = crate::common::png::encode_rgb(prepared.true_w, prepared.true_h, &rgb);
                std::fs::write(frames_out.join(format!("up_{out_idx:06}.png")), png)
                    .map_err(|err| EngineError::Resolve(format!("write output frame: {err}")))?;
            }
            exec.log(
                "info",
                &format!(
                    "seedvr2_video_upscale chunk start={start} end={end} frames={logical} elapsed_s={:.2}",
                    t0.elapsed().as_secs_f64()
                ),
            )?;
            exec.progress("denoise", end as u32, n_frames as u32)?;
            start = end;
        }
        let out_path = prepared.work.join("upscale.mp4");
        ffmpeg::mux_png_sequence_to_mp4(
            &frames_out.join("up_%06d.png"),
            Some(&prepared.source_path),
            &out_path,
            prepared.fps,
        )?;
        let bytes = std::fs::read(&out_path)
            .map_err(|err| EngineError::Resolve(format!("read muxed mp4: {err}")))?;
        let duration = n_frames as f64 / prepared.fps.max(1e-3);
        let metadata = json!({
            "model": prepared.request.model,
            "source_asset_id": prepared.request.source_asset_id,
            "mime_type": "video/mp4",
            "frame_count": prepared.n_frames,
            "fps": prepared.fps,
            "scale": prepared.scale,
            "denoise": prepared.denoise,
            "chunk_frames": prepared.chunk_frames,
            "seed": seed_base,
        });
        let mut artifact = Artifact::mp4(
            bytes,
            prepared.true_w as i64,
            prepared.true_h as i64,
            duration,
            metadata.clone(),
        );
        artifact.source_action = "upscale".into();
        artifact.parent_asset_id = Some(prepared.request.source_asset_id.clone());
        artifact.relation_type = Some("upscale".into());
        let mut output = EngineOutput {
            steps: 1,
            width: Some(prepared.true_w),
            height: Some(prepared.true_h),
            frames: Some(prepared.n_frames as i64),
            ..EngineOutput::default()
        };
        let mut full = metadata;
        full["request_metadata"] = prepared.request.metadata.clone();
        output.metadata = full;
        output.artifacts.push(artifact);
        Ok(output)
    }
}
