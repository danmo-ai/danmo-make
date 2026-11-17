//! SeedVR2 image upscale session (Facade). Prepare: bicubic resize so the
//! short edge scales by `scale` (optional softness pre-downscale), even
//! align, pad to 16, read the fixed positive embeddings. Infer: tiled 3D VAE
//! encode, one Euler step of the MM-DiT at `t = 1000`
//! (`latents - model_output`), tiled decode, crop, wavelet + LAB color
//! correction against the preprocessed source, PNG out.

use super::{color, dit, tiling, vae};
use crate::contract::{Artifact, EngineOutput, EngineRequest, ExecutionContext, ResolvedRun};
use crate::error::EngineError;
use crate::families::{FacadeFamily, FacadeSession};
use crate::ports::ResolvePort;
use crate::runtime::GgmlContext;
use crate::weights::{assert_upscale_bundle, f16_to_f32, GgufWeights, WeightError};
use danmo_contracts::ImageUpscaleRequest;
use serde_json::{json, Value};
use std::path::Path;

const POS_EMB_BYTES: &[u8] = include_bytes!("data/pos_emb.safetensors");

pub struct SeedVr2UpscaleFamily;

struct Prepared {
    request: ImageUpscaleRequest,
    source: (u32, u32),
    scale: u32,
    denoise: f32,
    true_w: u32,
    true_h: u32,
    style: Vec<f32>,
    pad_w: i64,
    pad_h: i64,
    pos_emb: Vec<f32>,
    txt_len: i64,
}

pub struct SeedVr2UpscaleSession {
    run: ResolvedRun,
    prepared: Prepared,
}

impl FacadeFamily for SeedVr2UpscaleFamily {
    fn prepare(
        &self,
        _ctx: &GgmlContext,
        run: &ResolvedRun,
        request: &EngineRequest,
        resolver: &dyn ResolvePort,
        exec: &ExecutionContext,
    ) -> Result<Box<dyn FacadeSession>, EngineError> {
        let EngineRequest::ImageUpscale(request) = request else {
            return Err(EngineError::bad_request(
                "seedvr2 upscale received a non-upscale request",
            ));
        };
        let scale = request.scale.unwrap_or(2);
        if scale != 2 && scale != 4 {
            return Err(EngineError::bad_request(format!(
                "seedvr2 upscale scale must be 2 or 4, got {scale}"
            )));
        }
        let denoise = request.denoise.unwrap_or(0.0);
        if !denoise.is_finite() {
            return Err(EngineError::bad_request(format!(
                "seedvr2 upscale denoise {denoise} is not finite"
            )));
        }
        let denoise = denoise.clamp(0.0, 1.0);
        if request.source_asset_id.trim().is_empty() {
            return Err(EngineError::bad_request(
                "seedvr2 upscale source_asset_id is empty",
            ));
        }
        let (rgb, source_w, source_h) =
            resolver.read_source_image(run, &request.source_asset_id)?;
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
        let resized = preprocess(&rgb, source_w, source_h, true_w, true_h, denoise);
        let pad_w = (16 - true_w as i64 % 16) % 16;
        let pad_h = (16 - true_h as i64 % 16) % 16;
        let style = pad_planar(&resized, true_w as usize, true_h as usize, pad_w, pad_h);
        exec.log(
            "info",
            &format!(
                "seedvr2_upscale source={source_w}x{source_h} out={true_w}x{true_h} padded={}x{} scale={scale} denoise={denoise}",
                true_w as i64 + pad_w,
                true_h as i64 + pad_h
            ),
        )?;
        Ok(Box::new(SeedVr2UpscaleSession {
            run: run.clone(),
            prepared: Prepared {
                request: request.clone(),
                source: (source_w, source_h),
                scale,
                denoise,
                true_w,
                true_h,
                style,
                pad_w,
                pad_h,
                pos_emb,
                txt_len,
            },
        }))
    }
}

impl FacadeSession for SeedVr2UpscaleSession {
    fn infer(
        &mut self,
        ctx: &GgmlContext,
        exec: &ExecutionContext,
    ) -> Result<EngineOutput, EngineError> {
        let prepared = &self.prepared;
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
        let latent = tiling::encode(
            ctx,
            vae::encoder_loaded(&vae_loaded),
            &spec,
            &prepared.style,
            pw,
            ph,
            1,
        )?;
        exec.progress("encode", 1, 1)?;
        let (lw, lh) = (pw / 8, ph / 8);
        let latent_ch = spec.latent;
        let plane = (lw * lh) as usize;
        let seed = prepared.request.seed.unwrap_or(0);
        let noise = crate::common::ops::normal_latent(seed, plane * latent_ch as usize);
        // vid input `[W, H, 33]`: [noise; latent; mask of ones].
        let mut vid = vec![0f32; plane * config.vid_channels as usize];
        let lc = latent_ch as usize;
        for c in 0..lc {
            vid[c * plane..(c + 1) * plane].copy_from_slice(&noise[c * plane..(c + 1) * plane]);
            vid[(lc + c) * plane..(lc + c + 1) * plane]
                .copy_from_slice(&latent[c * plane..(c + 1) * plane]);
        }
        for v in &mut vid[2 * latent_ch as usize * plane..] {
            *v = 1.0;
        }
        let velocity = dit::run(
            ctx,
            &config,
            &dit_loaded,
            &vid,
            lw,
            lh,
            1,
            &prepared.pos_emb,
            1000.0,
        )?;
        exec.progress("denoise", 1, 1)?;
        let latents: Vec<f32> = noise.iter().zip(&velocity).map(|(n, v)| n - v).collect();
        if let Some(detail) = crate::common::ops::non_finite("seedvr2 latent", &latents) {
            return Err(WeightError::NotBuilt(detail).into());
        }
        let decoded = tiling::decode(
            ctx,
            vae::decoder_loaded(&vae_loaded),
            &spec,
            &latents,
            lw,
            lh,
            1,
        )?;
        exec.progress("decode", 1, 1)?;
        let (tw, th) = (prepared.true_w as usize, prepared.true_h as usize);
        let decoded = crop_planar(&decoded, pw as usize, ph as usize, tw, th);
        let style = crop_planar(&prepared.style, pw as usize, ph as usize, tw, th);
        let corrected = color::correct(&decoded, &style, tw, th)?;
        let mut rgb = vec![0u8; tw * th * 3];
        for y in 0..th {
            for x in 0..tw {
                for c in 0..3 {
                    let unit = (corrected[x + y * tw + c * tw * th] + 1.0) * 0.5;
                    rgb[(y * tw + x) * 3 + c] = (unit.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
        let png = crate::common::png::encode_rgb(prepared.true_w, prepared.true_h, &rgb);
        let metadata = json!({
            "source_asset_id": prepared.request.source_asset_id,
            "source_width": prepared.source.0,
            "source_height": prepared.source.1,
            "scale": prepared.scale,
            "denoise": prepared.denoise,
            "seed": seed,
            "steps": 1,
            "timestep": 1000,
        });
        let mut artifact = Artifact::png(
            png,
            prepared.true_w as i64,
            prepared.true_h as i64,
            metadata.clone(),
        );
        artifact.source_action = "upscale".into();
        artifact.parent_asset_id = Some(prepared.request.source_asset_id.clone());
        artifact.relation_type = Some("upscale".into());
        let mut output = EngineOutput {
            steps: 1,
            width: Some(prepared.true_w),
            height: Some(prepared.true_h),
            latent_len: latents.len(),
            ..EngineOutput::default()
        };
        let mut full = metadata;
        full["request_metadata"] = prepared.request.metadata.clone();
        output.metadata = full;
        output.artifacts.push(artifact);
        Ok(output)
    }
}

/// `int(w * scale)` floored to even, matching the reference preprocess.
pub(crate) fn even(scaled: u64) -> Result<u32, EngineError> {
    let v = ((scaled as usize) / 2 * 2) as u32;
    if v < 16 {
        return Err(EngineError::bad_request(format!(
            "seedvr2 upscale target {v} is below 16 pixels"
        )));
    }
    Ok(v)
}

/// PIL-style separable bicubic (a = -0.5) with antialias widening on
/// downscale. `rgb` is interleaved bytes.
fn bicubic_resize(rgb: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    fn kernel(x: f32) -> f32 {
        let x = x.abs();
        if x < 1.0 {
            1.5 * x.powi(3) - 2.5 * x.powi(2) + 1.0
        } else if x < 2.0 {
            -0.5 * x.powi(3) + 2.5 * x.powi(2) - 4.0 * x + 2.0
        } else {
            0.0
        }
    }
    fn weights(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f32>)> {
        let scale = in_size as f32 / out_size as f32;
        let filter = scale.max(1.0);
        let support = 2.0 * filter;
        (0..out_size)
            .map(|o| {
                let center = (o as f32 + 0.5) * scale;
                let first = (center - support).ceil().max(0.0) as usize;
                let last = ((center + support).floor() as usize).min(in_size - 1);
                let mut ws = Vec::with_capacity(last + 1 - first);
                let mut sum = 0f32;
                for i in first..=last {
                    let w = kernel((i as f32 + 0.5 - center) / filter);
                    ws.push(w);
                    sum += w;
                }
                if sum.abs() > 1e-8 {
                    for w in &mut ws {
                        *w /= sum;
                    }
                }
                (first, ws)
            })
            .collect()
    }
    // Horizontal pass into f32, then vertical back to bytes.
    let wx = weights(sw, dw);
    let mut tmp = vec![0f32; dw * sh * 3];
    for y in 0..sh {
        for (ox, (first, ws)) in wx.iter().enumerate() {
            for c in 0..3 {
                let mut acc = 0f32;
                for (k, &w) in ws.iter().enumerate() {
                    acc += w * rgb[(y * sw + first + k) * 3 + c] as f32;
                }
                tmp[(y * dw + ox) * 3 + c] = acc;
            }
        }
    }
    let wy = weights(sh, dh);
    let mut out = vec![0u8; dw * dh * 3];
    for (oy, (first, ws)) in wy.iter().enumerate() {
        for x in 0..dw {
            for c in 0..3 {
                let mut acc = 0f32;
                for (k, &w) in ws.iter().enumerate() {
                    acc += w * tmp[((first + k) * dw + x) * 3 + c];
                }
                out[(oy * dw + x) * 3 + c] = acc.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Reference preprocess: optional softness down→up blur, then resize to the
/// true size. Returns interleaved RGB at `true_w x true_h`.
pub(crate) fn preprocess(
    rgb: &[u8],
    sw: u32,
    sh: u32,
    true_w: u32,
    true_h: u32,
    softness: f32,
) -> Vec<u8> {
    let factor = 1.0 + softness.clamp(0.0, 1.0) * 7.0;
    let (sw, sh, tw, th) = (sw as usize, sh as usize, true_w as usize, true_h as usize);
    if factor > 1.0 {
        let down_w = ((tw as f32 / factor) as usize).max(2);
        let down_h = ((th as f32 / factor) as usize).max(2);
        let down = bicubic_resize(rgb, sw, sh, down_w, down_h);
        bicubic_resize(&down, down_w, down_h, tw, th)
    } else {
        bicubic_resize(rgb, sw, sh, tw, th)
    }
}

/// Interleaved RGB to `[W, H, 3]` planar `[-1, 1]`, zero-padded right and
/// bottom by `pad_w`/`pad_h`.
pub(crate) fn pad_planar(rgb: &[u8], w: usize, h: usize, pad_w: i64, pad_h: i64) -> Vec<f32> {
    let (pw, ph) = (w + pad_w as usize, h + pad_h as usize);
    let mut out = vec![0f32; pw * ph * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let byte = rgb[(y * w + x) * 3 + c];
                out[x + y * pw + c * pw * ph] = (byte as f32 / 255.0) * 2.0 - 1.0;
            }
        }
    }
    out
}

/// Take the top-left `tw x th` window of a `[W, H, 3]` planar tensor.
pub(crate) fn crop_planar(planes: &[f32], w: usize, h: usize, tw: usize, th: usize) -> Vec<f32> {
    let mut out = vec![0f32; tw * th * 3];
    for c in 0..3 {
        for y in 0..th {
            let src = y * w + c * w * h;
            let dst = y * tw + c * tw * th;
            out[dst..dst + tw].copy_from_slice(&planes[src..src + tw]);
        }
    }
    out
}

/// Fixed positive text embeddings. A bundle-local `pos_emb.safetensors`
/// (root or `data/`) wins; otherwise the packaged default. Returns the flat
/// `[txt_in_dim, txt_len]` ggml-order floats plus both dims.
pub(crate) fn load_pos_emb(root: &Path) -> Result<(Vec<f32>, i64, i64), WeightError> {
    let candidates = [
        root.join("pos_emb.safetensors"),
        root.join("data").join("pos_emb.safetensors"),
    ];
    let bytes: Vec<u8> = match candidates.iter().find(|p| p.is_file()) {
        Some(path) => std::fs::read(path).map_err(|err| WeightError::Bundle {
            path: path.display().to_string(),
            detail: format!("pos_emb.safetensors: {err}"),
        })?,
        None => POS_EMB_BYTES.to_vec(),
    };
    let (values, shape) = read_single_safetensor(&bytes, "embedding")?;
    if shape.len() != 2 {
        return Err(WeightError::NotBuilt(format!(
            "pos_emb embedding shape {shape:?} is not [txt_len, txt_in_dim]"
        )));
    }
    Ok((values, shape[0] as i64, shape[1] as i64))
}

/// Read one F16 or F32 tensor out of a safetensors byte buffer. Returns the
/// widened floats in storage order and the PyTorch shape.
fn read_single_safetensor(bytes: &[u8], want: &str) -> Result<(Vec<f32>, Vec<u64>), WeightError> {
    let bad = |detail: String| WeightError::Bundle {
        path: "pos_emb.safetensors".into(),
        detail,
    };
    if bytes.len() < 8 {
        return Err(bad("header is shorter than 8 bytes".into()));
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header_end = 8usize
        .checked_add(n)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| bad("header extends past the end of the file".into()))?;
    let header: serde_json::Value = serde_json::from_str(
        std::str::from_utf8(&bytes[8..header_end])
            .map_err(|_| bad("header is not utf-8".into()))?
            .trim_end(),
    )
    .map_err(|err| bad(format!("header is not json: {err}")))?;
    let meta = header
        .get(want)
        .ok_or_else(|| bad(format!("no {want} tensor")))?;
    let dtype = meta
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("dtype is missing".into()))?;
    let shape: Vec<u64> = meta
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("shape is missing".into()))?
        .iter()
        .filter_map(Value::as_u64)
        .collect();
    let offsets = meta
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("data_offsets is missing".into()))?;
    let start = offsets[0].as_u64().unwrap_or(0) as usize + header_end;
    let end = offsets[1].as_u64().unwrap_or(0) as usize + header_end;
    if end > bytes.len() || end < start {
        return Err(bad("data offsets escape the file".into()));
    }
    let data = &bytes[start..end];
    let count = shape.iter().product::<u64>() as usize;
    let values = match dtype {
        "F16" => {
            if data.len() != count * 2 {
                return Err(bad("F16 data length disagrees with the shape".into()));
            }
            data.chunks_exact(2)
                .map(|pair| f16_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
                .collect()
        }
        "F32" => {
            if data.len() != count * 4 {
                return Err(bad("F32 data length disagrees with the shape".into()));
            }
            data.chunks_exact(4)
                .map(|quad| f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
                .collect()
        }
        other => return Err(bad(format!("dtype {other} is not F16 or F32"))),
    };
    Ok((values, shape))
}

/// Write a deterministic smoke bundle: tiny GGUF DiT, tiny safetensors VAE,
/// a matching tiny `pos_emb.safetensors`, and the manifest.
pub fn write_smoke_bundle(dir: &Path) -> Result<(), WeightError> {
    use crate::weights::{write_f32_nd, write_safetensors, F32Nd, SafeTensor};
    std::fs::create_dir_all(dir).map_err(|err| WeightError::Bundle {
        path: dir.display().to_string(),
        detail: format!("{err}"),
    })?;
    let config = dit::DitConfig::smoke();
    let mut names: Vec<String> = Vec::new();
    let mut dims: Vec<[i64; 4]> = Vec::new();
    let mut values: Vec<Vec<f32>> = Vec::new();
    let mut seed = 1.0f32;
    for (name, d) in dit::weight_layout(&config) {
        let n = d.iter().product::<i64>() as usize;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 + 1.0) * 0.37 + seed).sin() * 0.05)
            .collect();
        let mut shape = [1i64; 4];
        for (i, dim) in d.iter().enumerate() {
            shape[i] = *dim;
        }
        names.push(name);
        dims.push(shape);
        values.push(data);
        seed += 1.0;
    }
    let freqs: Vec<f32> = (0..config.freqs_len)
        .map(|i| 10000f32.powf(-2.0 * i as f32 / (2.0 * config.freqs_len as f32)))
        .collect();
    for (i, name) in names.iter().enumerate() {
        if name.ends_with("attn.rope.rope.freqs") {
            values[i] = freqs.clone();
        }
    }
    let tensors: Vec<F32Nd> = names
        .iter()
        .zip(&dims)
        .zip(&values)
        .map(|((name, ne), data)| F32Nd {
            name: name.as_str(),
            ne: *ne,
            data: data.as_slice(),
        })
        .collect();
    write_f32_nd(&dir.join("dit_smoke.gguf"), &tensors)?;
    vae::write_smoke_vae(dir)?;
    // Tiny pos_emb matching the smoke txt dims `[txt_len, txt_in_dim]`.
    let pos: Vec<f32> = (0..(config.txt_len * config.txt_in_dim) as usize)
        .map(|i| (i as f32 * 0.017).cos() * 0.1)
        .collect();
    let pos_bytes: Vec<u8> = pos.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_safetensors(
        &dir.join("pos_emb.safetensors"),
        &[SafeTensor {
            name: "embedding",
            dtype: "F32",
            shape: &[config.txt_len as u64, config.txt_in_dim as u64],
            data: &pos_bytes,
        }],
    )?;
    std::fs::write(
        dir.join("bundle.manifest.json"),
        r#"{"format":"gguf","quant":"bf16","files":{"transformer":"dit_smoke.gguf","vae":"ema_vae_fp16.safetensors"}}"#,
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
            "danmo-seedvr2-up-{name}-{}-{}",
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

    fn request() -> ImageUpscaleRequest {
        ImageUpscaleRequest {
            model: "seedvr2-3b".into(),
            source_asset_id: "ast_src".into(),
            scale: Some(2),
            denoise: None,
            seed: Some(7),
            metadata: json!({}),
            priority: "normal".into(),
        }
    }

    fn run(dir: PathBuf) -> ResolvedRun {
        ResolvedRun {
            model: "seedvr2-3b".into(),
            version: Some("bf16".into()),
            family: "seedvr2".into(),
            action: Action::ImageUpscale,
            bundle: dir,
            backend: BackendKind::Cpu,
        }
    }

    fn resolver() -> StubResolver {
        let mut rgb = vec![0u8; 32 * 32 * 3];
        for y in 0..32usize {
            for x in 0..32usize {
                rgb[(y * 32 + x) * 3] = (x * 8) as u8;
                rgb[(y * 32 + x) * 3 + 1] = (y * 8) as u8;
                rgb[(y * 32 + x) * 3 + 2] = 128;
            }
        }
        StubResolver {
            rgb,
            width: 32,
            height: 32,
        }
    }

    #[test]
    fn upscale_facade_runs_the_smoke_bundle_end_to_end() {
        let dir = scratch("facade");
        write_smoke_bundle(&dir).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("upscale", std::sync::Arc::new(NoopObserver));
        let family = SeedVr2UpscaleFamily;
        let mut session = family
            .prepare(
                &ctx,
                &run(dir.clone()),
                &EngineRequest::ImageUpscale(request()),
                &resolver(),
                &exec,
            )
            .expect("prepare");
        let output = session.infer(&ctx, &exec).expect("infer");
        assert_eq!(output.width, Some(64));
        assert_eq!(output.height, Some(64));
        assert_eq!(output.steps, 1);
        assert_eq!(output.artifacts.len(), 1);
        let artifact = &output.artifacts[0];
        assert!(artifact.bytes.starts_with(b"\x89PNG"));
        assert_eq!(artifact.parent_asset_id.as_deref(), Some("ast_src"));
        assert_eq!(artifact.relation_type.as_deref(), Some("upscale"));
        assert_eq!(output.metadata["scale"], json!(2));
        let (decoded, w, h) = crate::common::png::decode_rgb(&artifact.bytes).unwrap();
        assert_eq!((w, h), (64, 64));
        assert!(decoded.iter().any(|&b| b != 0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn upscale_facade_rejects_bad_input_loud() {
        let dir = scratch("reject");
        write_smoke_bundle(&dir).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("upscale", std::sync::Arc::new(NoopObserver));
        let family = SeedVr2UpscaleFamily;
        let bad_scale = family.prepare(
            &ctx,
            &run(dir.clone()),
            &EngineRequest::ImageUpscale(ImageUpscaleRequest {
                scale: Some(3),
                ..request()
            }),
            &resolver(),
            &exec,
        );
        let err = match bad_scale {
            Err(error) => error,
            Ok(_) => panic!("scale 3 was accepted"),
        };
        assert!(err.to_string().contains("scale"), "{err}");
        let missing = family.prepare(
            &ctx,
            &run(dir.clone()),
            &EngineRequest::ImageUpscale(ImageUpscaleRequest {
                source_asset_id: "ast_gone".into(),
                ..request()
            }),
            &resolver(),
            &exec,
        );
        assert!(missing.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bicubic_resize_keeps_a_constant_image_constant() {
        let rgb = vec![173u8; 8 * 8 * 3];
        let out = bicubic_resize(&rgb, 8, 8, 16, 16);
        assert!(out.iter().all(|&b| (b as i32 - 173).abs() <= 1));
        let down = bicubic_resize(&rgb, 8, 8, 4, 4);
        assert!(down.iter().all(|&b| (b as i32 - 173).abs() <= 1));
    }

    #[test]
    fn bicubic_resize_averages_a_step_edge_down() {
        // Left half black, right half white; a 2x downsample keeps the mean
        // near 128 and both far pixels near the plateaus.
        let mut rgb = vec![0u8; 8 * 8 * 3];
        for y in 0..8usize {
            for x in 4..8usize {
                for c in 0..3 {
                    rgb[(y * 8 + x) * 3 + c] = 255;
                }
            }
        }
        let out = bicubic_resize(&rgb, 8, 8, 4, 4);
        let mean: f32 = out.iter().map(|&b| b as f32).sum::<f32>() / out.len() as f32;
        assert!((mean - 128.0).abs() <= 8.0, "{mean}");
        assert!(out[(1 * 4) * 3] < 32);
        assert!(out[(1 * 4 + 3) * 3] > 223);
    }

    #[test]
    fn driver_routes_seedvr2_upscale_to_the_facade() {
        crate::session::ensure_available("seedvr2", Action::ImageUpscale).unwrap();
        let other = crate::session::ensure_available("seedvr2", Action::ImageCreate);
        assert!(other.is_err());
    }
}
