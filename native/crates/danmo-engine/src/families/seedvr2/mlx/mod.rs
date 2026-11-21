//! MLX implementation of SeedVR2 spatiotemporal video restoration.

pub mod dit;
pub mod tiling;
pub mod vae;

use std::path::Path;
use std::time::UNIX_EPOCH;

use danmo_mlx::{Array, CacheKey, CacheRevision, ModelCache, RandomKey, Result, Stream};

use crate::weights::WeightError;

pub struct Models {
    dit: dit::Dit,
    vae: vae::Vae,
    pos_emb: Vec<f32>,
    txt_len: i64,
    txt_in_dim: i64,
    stream_cache: vae::VaeStreamCache,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentReport {
    pub hit: bool,
    pub bytes: usize,
}

impl Models {
    pub fn load(
        bundle: &Path,
        pos_emb: Vec<f32>,
        txt_len: i64,
        txt_in_dim: i64,
        stream: &Stream,
    ) -> std::result::Result<Self, WeightError> {
        let transformer = [
            "seedvr2_ema_7b_fp16.safetensors",
            "seedvr2_ema_3b_fp16.safetensors",
        ]
        .into_iter()
        .map(|name| bundle.join(name))
        .find(|path| path.is_file())
        .ok_or_else(|| WeightError::Bundle {
            path: bundle.display().to_string(),
            detail: "missing seedvr2_ema_7b_fp16.safetensors or seedvr2_ema_3b_fp16.safetensors"
                .into(),
        })?;
        let vae_path = bundle.join("ema_vae_fp16.safetensors");
        if !vae_path.is_file() {
            return Err(WeightError::Bundle {
                path: vae_path.display().to_string(),
                detail: "SeedVR2 VAE is missing".into(),
            });
        }
        let dit = dit::Dit::load(&transformer, txt_len)?;
        if dit.config().txt_in_dim != txt_in_dim {
            return Err(WeightError::NotBuilt(format!(
                "pos_emb width {txt_in_dim} disagrees with DiT txt_in_dim {}",
                dit.config().txt_in_dim
            )));
        }
        let vae = vae::Vae::load(&vae_path, stream)?;
        if dit.config().vid_channels != 2 * vae.spec().latent + 1 {
            return Err(WeightError::NotBuilt(format!(
                "DiT expects {} channels but VAE has {} latent channels",
                dit.config().vid_channels,
                vae.spec().latent
            )));
        }
        Ok(Self {
            dit,
            vae,
            pos_emb,
            txt_len,
            txt_in_dim,
            stream_cache: vae::VaeStreamCache::default(),
        })
    }

    pub fn resident(
        cache: &mut ModelCache,
        bundle: &Path,
        pos_emb: Vec<f32>,
        txt_len: i64,
        txt_in_dim: i64,
        stream: &Stream,
    ) -> Result<ResidentReport> {
        let key = cache_key(bundle)?;
        if cache.get::<Models>(&key)?.is_some() {
            return Ok(ResidentReport {
                hit: true,
                bytes: cache
                    .get::<Models>(&key)?
                    .map(|models| models.nbytes())
                    .unwrap_or(0),
            });
        }
        let models = Self::load(bundle, pos_emb, txt_len, txt_in_dim, stream)
            .map_err(|error| danmo_mlx::Error::new("SeedVR2 resident", error.to_string()))?;
        let bytes = models.nbytes();
        cache.insert(key, models, bytes)?;
        Ok(ResidentReport { hit: false, bytes })
    }

    pub fn unload(bundle: &Path) -> Result<bool> {
        let id = cache_id(bundle)?;
        crate::runtime::mlx_executor()?.execute("seedvr2 unload", move |_, _, cache, _| {
            Ok(cache.unload(&id))
        })
    }

    fn nbytes(&self) -> usize {
        self.dit.nbytes() + self.vae.nbytes() + self.pos_emb.len() * std::mem::size_of::<f32>()
    }

    pub fn reset_stream(&mut self, enabled: bool) {
        self.stream_cache.reset();
        self.stream_cache.enabled = enabled;
    }

    /// Restore one `[C][T][H][W]` host tile and return decoded `[C][F][H][W]`.
    pub fn restore_chunk(
        &mut self,
        stacked: &[f32],
        w: i64,
        h: i64,
        frames: i64,
        seed: i64,
        stream: &Stream,
    ) -> Result<Vec<f32>> {
        self.restore_chunk_with_cache(stacked, w, h, frames, seed, stream, None)
    }

    pub fn restore_chunk_streaming(
        &mut self,
        stacked: &[f32],
        w: i64,
        h: i64,
        frames: i64,
        seed: i64,
        stream: &Stream,
    ) -> Result<Vec<f32>> {
        let mut cache = std::mem::take(&mut self.stream_cache);
        let result = self.restore_chunk_with_cache(
            stacked,
            w,
            h,
            frames,
            seed,
            stream,
            Some(&mut cache),
        );
        self.stream_cache = cache;
        result
    }

    fn restore_chunk_with_cache(
        &mut self,
        stacked: &[f32],
        w: i64,
        h: i64,
        frames: i64,
        seed: i64,
        stream: &Stream,
        cache: Option<&mut vae::VaeStreamCache>,
    ) -> Result<Vec<f32>> {
        let mut cache = cache;
        let latent = tiling::encode_with_cache(
            &self.vae,
            stacked,
            w,
            h,
            frames,
            stream,
            cache.as_deref_mut(),
        )?;
        latent.eval()?;
        let latent_shape = latent.shape();
        let (lw, lh, t_lat) = (
            latent_shape[4] as i64,
            latent_shape[3] as i64,
            latent_shape[2],
        );
        let dtype = latent.dtype();
        let noise =
            RandomKey::new((seed & 0x7fff_ffff) as u64)?.normal(&latent_shape, dtype, stream)?;
        let mask = latent.slice(
            &[0, 0, 0, 0, 0],
            &[1, 1, latent_shape[2], latent_shape[3], latent_shape[4]],
            &[1, 1, 1, 1, 1],
            stream,
        )?;
        let condition =
            Array::concatenate_axis(&[latent.clone(), mask.ones_like(stream)?], 1, stream)?;
        let input = Array::concatenate_axis(&[noise.clone(), condition], 1, stream)?;
        let txt = Array::from_f32_slice(
            &self.pos_emb,
            &[1, self.txt_len as i32, self.txt_in_dim as i32],
        )?
        .astype(dtype, stream)?;
        let velocity = self.dit.forward(&input, &txt, 1000.0, stream)?;
        let restored = Array::subtract(&noise, &velocity, stream)?;
        restored.eval()?;
        let decoded = tiling::decode_with_cache(
            &self.vae,
            &restored,
            lw,
            lh,
            t_lat as i64,
            stream,
            cache,
        )?;
        decoded.eval()?;
        let decoded = decoded.to_vec_f32(stream)?;
        ensure_finite(&decoded, "SeedVR2 decoded video")?;
        Ok(decoded)
    }
}

pub(crate) fn cache_key_for(bundle: &Path) -> Result<CacheKey> {
    cache_key(bundle)
}

fn cache_key(bundle: &Path) -> Result<CacheKey> {
    Ok(CacheKey {
        id: cache_id(bundle)?,
        revision: bundle_revision(bundle)?,
    })
}

fn cache_id(bundle: &Path) -> Result<String> {
    Ok(std::fs::canonicalize(bundle)
        .map_err(|error| {
            danmo_mlx::Error::new(
                "SeedVR2 resident",
                format!("{}: {error}", bundle.display()),
            )
        })?
        .display()
        .to_string())
}

fn bundle_revision(bundle: &Path) -> Result<CacheRevision> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(bundle).map_err(|error| {
        danmo_mlx::Error::new("SeedVR2 resident", format!("{}: {error}", bundle.display()))
    })? {
        let path = entry
            .map_err(|error| danmo_mlx::Error::new("SeedVR2 resident", error.to_string()))?
            .path();
        if path.extension().and_then(|value| value.to_str()) == Some("safetensors") {
            paths.push(path);
        }
    }
    paths.sort();
    let stamp = crate::weights::revision(&paths)
        .map_err(|error| danmo_mlx::Error::new("SeedVR2 resident", error.to_string()))?;
    let mut length = 0u64;
    let mut modified_ns = 0u64;
    for path in &paths {
        let metadata = std::fs::metadata(path).map_err(|error| {
            danmo_mlx::Error::new("SeedVR2 resident", format!("{}: {error}", path.display()))
        })?;
        length += metadata.len();
        modified_ns = modified_ns.max(
            metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map(|duration| duration.as_nanos() as u64)
                .unwrap_or(0),
        );
    }
    let mut mix = 0xcbf29ce484222325u64;
    for byte in stamp.as_bytes() {
        mix ^= u64::from(*byte);
        mix = mix.wrapping_mul(0x100000001b3);
    }
    Ok(CacheRevision {
        length,
        modified_ns,
        mix,
    })
}

fn ensure_finite(input: &[f32], what: &'static str) -> Result<()> {
    if input.iter().any(|value| !value.is_finite()) {
        return Err(danmo_mlx::Error::new(what, "non-finite values"));
    }
    Ok(())
}

#[cfg(all(test, feature = "mlx-metal"))]
mod tests {
    #[test]
    #[ignore = "loads the installed 7B checkpoint; set DANQING_TEST_SEEDVR2_BUNDLE"]
    fn real_seedvr2_7b_video_chunk_is_finite() {
        let bundle = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_SEEDVR2_BUNDLE").expect("DANQING_TEST_SEEDVR2_BUNDLE"),
        );
        let (pos_emb, txt_len, txt_in_dim) = super::super::session::load_pos_emb(&bundle).unwrap();
        let stream = danmo_mlx::Stream::gpu().unwrap();
        let started = std::time::Instant::now();
        let models = super::Models::load(&bundle, pos_emb, txt_len, txt_in_dim, &stream).unwrap();
        eprintln!(
            "seedvr2 model load elapsed_s={:.3}",
            started.elapsed().as_secs_f64()
        );
        let (w, h, frames) = (16, 16, 5);
        let stacked: Vec<f32> = (0..3 * frames * w * h)
            .map(|index| (index as f32 * 0.017).sin() * 0.5)
            .collect();
        let started = std::time::Instant::now();
        let latent = super::tiling::encode(&models.vae, &stacked, w, h, frames, &stream).unwrap();
        let latent_shape = latent.shape();
        let noise = danmo_mlx::RandomKey::new(42)
            .unwrap()
            .normal(&latent_shape, latent.dtype(), &stream)
            .unwrap();
        let mask = latent
            .slice(
                &[0, 0, 0, 0, 0],
                &[1, 1, latent_shape[2], latent_shape[3], latent_shape[4]],
                &[1, 1, 1, 1, 1],
                &stream,
            )
            .unwrap()
            .ones_like(&stream)
            .unwrap();
        let condition =
            danmo_mlx::Array::concatenate_axis(&[latent.clone(), mask], 1, &stream).unwrap();
        let input =
            danmo_mlx::Array::concatenate_axis(&[noise.clone(), condition], 1, &stream).unwrap();
        let txt = danmo_mlx::Array::from_f32_slice(
            &models.pos_emb,
            &[1, models.txt_len as i32, models.txt_in_dim as i32],
        )
        .unwrap()
        .astype(latent.dtype(), &stream)
        .unwrap();
        let velocity = models.dit.forward(&input, &txt, 1000.0, &stream).unwrap();
        let restored = danmo_mlx::Array::subtract(&noise, &velocity, &stream).unwrap();
        let decoded = super::tiling::decode(
            &models.vae,
            &restored,
            latent_shape[4] as i64,
            latent_shape[3] as i64,
            latent_shape[2] as i64,
            &stream,
        )
        .unwrap()
        .to_vec_f32(&stream)
        .unwrap();
        let latent = latent.to_vec_f32(&stream).unwrap();
        let noise = noise.to_vec_f32(&stream).unwrap();
        let velocity = velocity.to_vec_f32(&stream).unwrap();
        let restored = restored.to_vec_f32(&stream).unwrap();
        assert_eq!(decoded.len(), (3 * 8 * w * h) as usize);
        assert!(decoded.iter().all(|value| value.is_finite()));
        assert!(decoded.iter().any(|value| value.abs() > 1e-5));
        let mean = decoded.iter().sum::<f32>() / decoded.len() as f32;
        let variance = decoded
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f32>()
            / decoded.len() as f32;
        eprintln!(
            "seedvr2 real chunk elapsed_s={:.3} mean={mean:.6} variance={variance:.6} min={:.6} max={:.6}",
            started.elapsed().as_secs_f64(),
            decoded.iter().copied().fold(f32::INFINITY, f32::min),
            decoded.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        );
        if let Ok(reference) = std::env::var("DANQING_TEST_SEEDVR2_REFERENCE") {
            let reference: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(reference).expect("read SeedVR2 reference"),
            )
            .expect("parse SeedVR2 reference");
            for (name, actual) in [
                ("latent", &latent),
                ("noise", &noise),
                ("velocity", &velocity),
                ("restored", &restored),
                ("decoded", &decoded),
            ] {
                let expected: Vec<f32> = reference[name]
                    .as_array()
                    .expect("reference stage")
                    .iter()
                    .map(|value| value.as_f64().expect("reference float") as f32)
                    .collect();
                assert_eq!(actual.len(), expected.len(), "{name} length");
                let mut mse = 0.0f64;
                let mut max_abs = 0.0f32;
                let mut reference_energy = 0.0f64;
                for (actual, expected) in actual.iter().zip(&expected) {
                    let difference = actual - expected;
                    mse += f64::from(difference * difference);
                    max_abs = max_abs.max(difference.abs());
                    reference_energy += f64::from(expected * expected);
                }
                mse /= actual.len() as f64;
                let rel_l2 = (mse * actual.len() as f64 / reference_energy.max(1e-20)).sqrt();
                eprintln!(
                    "seedvr2 vs main {name} mse={mse:.9} max_abs={max_abs:.6} rel_l2={rel_l2:.6}"
                );
                assert!(rel_l2 < 0.01, "{name} relative L2 {rel_l2} exceeds 1%");
            }
        }
    }

    #[test]
    #[ignore = "benchmarks the installed checkpoint; set DANQING_TEST_SEEDVR2_BUNDLE"]
    fn real_seedvr2_warm_performance_and_memory() {
        let bundle = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_SEEDVR2_BUNDLE").expect("DANQING_TEST_SEEDVR2_BUNDLE"),
        );
        let size = std::env::var("DANQING_TEST_SEEDVR2_BENCH_SIZE")
            .ok()
            .and_then(|value| value.parse::<i64>().ok())
            .unwrap_or(512);
        let frames = std::env::var("DANQING_TEST_SEEDVR2_BENCH_FRAMES")
            .ok()
            .and_then(|value| value.parse::<i64>().ok())
            .unwrap_or(1);
        let passes = std::env::var("DANQING_TEST_SEEDVR2_BENCH_PASSES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(3);
        let (pos_emb, txt_len, txt_in_dim) = super::super::session::load_pos_emb(&bundle).unwrap();
        let stream = danmo_mlx::Stream::gpu().unwrap();
        let mut cache = danmo_mlx::ModelCache::new(usize::MAX);
        let started = std::time::Instant::now();
        let resident = super::Models::resident(
            &mut cache,
            &bundle,
            pos_emb,
            txt_len,
            txt_in_dim,
            &stream,
        )
        .unwrap();
        let style: Vec<f32> = (0..3 * frames * size * size)
            .map(|index| (index as f32 * 0.017).sin() * 0.5)
            .collect();
        let mut models = cache
            .get_mut::<super::Models>(&super::cache_key_for(&bundle).unwrap())
            .unwrap()
            .unwrap();
        models
            .restore_chunk(&style, size, size, frames, 42, &stream)
            .unwrap();
        let cold_seconds = started.elapsed().as_secs_f64();
        models
            .restore_chunk(&style, size, size, frames, 42, &stream)
            .unwrap();
        danmo_mlx::reset_peak_memory().unwrap();
        let mut timings = Vec::with_capacity(passes);
        for _ in 0..passes {
            let started = std::time::Instant::now();
            models
                .restore_chunk(&style, size, size, frames, 42, &stream)
                .unwrap();
            timings.push(started.elapsed().as_secs_f64());
        }
        let peak_memory = danmo_mlx::peak_memory().unwrap();
        timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let warm_p50 = timings[timings.len() / 2];
        let encode_started = std::time::Instant::now();
        let latent = super::tiling::encode(&models.vae, &style, size, size, frames, &stream).unwrap();
        latent.eval().unwrap();
        let encode_seconds = encode_started.elapsed().as_secs_f64();
        let dit_started = std::time::Instant::now();
        let latent_shape = latent.shape();
        let noise = danmo_mlx::RandomKey::new(42)
            .unwrap()
            .normal(&latent_shape, latent.dtype(), &stream)
            .unwrap();
        let mask = latent
            .slice(
                &[0, 0, 0, 0, 0],
                &[1, 1, latent_shape[2], latent_shape[3], latent_shape[4]],
                &[1, 1, 1, 1, 1],
                &stream,
            )
            .unwrap()
            .ones_like(&stream)
            .unwrap();
        let condition =
            danmo_mlx::Array::concatenate_axis(&[latent.clone(), mask], 1, &stream).unwrap();
        let input = danmo_mlx::Array::concatenate_axis(&[noise.clone(), condition], 1, &stream).unwrap();
        let txt = danmo_mlx::Array::from_f32_slice(
            &models.pos_emb,
            &[1, models.txt_len as i32, models.txt_in_dim as i32],
        )
        .unwrap()
        .astype(latent.dtype(), &stream)
        .unwrap();
        let velocity = models.dit.forward(&input, &txt, 1000.0, &stream).unwrap();
        velocity.eval().unwrap();
        let dit_seconds = dit_started.elapsed().as_secs_f64();
        let decode_started = std::time::Instant::now();
        let restored = danmo_mlx::Array::subtract(&noise, &velocity, &stream).unwrap();
        let decoded = super::tiling::decode(
            &models.vae,
            &restored,
            latent_shape[4] as i64,
            latent_shape[3] as i64,
            latent_shape[2] as i64,
            &stream,
        )
        .unwrap();
        decoded.eval().unwrap();
        let decode_seconds = decode_started.elapsed().as_secs_f64();
        let report = serde_json::json!({
            "bundle": bundle,
            "size": size,
            "frames": frames,
            "resident_hit": resident.hit,
            "cold_seconds": cold_seconds,
            "warm_p50": warm_p50,
            "warm_passes": timings,
            "encode_seconds": encode_seconds,
            "dit_seconds": dit_seconds,
            "decode_seconds": decode_seconds,
            "peak_memory": peak_memory,
        });
        eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
        if let Ok(path) = std::env::var("DANQING_TEST_SEEDVR2_BENCH_REPORT") {
            std::fs::write(path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
        }
        if let Ok(path) = std::env::var("DANQING_TEST_SEEDVR2_MAIN_BENCHMARK") {
            let main: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            let main_p50 = main["warm_p50"].as_f64().unwrap();
            let main_peak = main["peak_memory"].as_f64().unwrap();
            let latency_ratio = warm_p50 / main_p50;
            let memory_ratio = peak_memory as f64 / main_peak;
            eprintln!("latency_ratio={latency_ratio:.6} memory_ratio={memory_ratio:.6}");
            assert!(latency_ratio <= 1.05, "warm latency ratio {latency_ratio} exceeds 105%");
            assert!(memory_ratio <= 1.05, "peak memory ratio {memory_ratio} exceeds 105%");
        }
    }

    #[test]
    #[ignore = "loads the installed 7B checkpoint and a main image fixture"]
    fn real_seedvr2_image_matches_main() {
        let bundle = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_SEEDVR2_BUNDLE").expect("DANQING_TEST_SEEDVR2_BUNDLE"),
        );
        let reference: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(
                std::env::var("DANQING_TEST_SEEDVR2_IMAGE_REFERENCE")
                    .expect("DANQING_TEST_SEEDVR2_IMAGE_REFERENCE"),
            )
            .unwrap(),
        )
        .unwrap();
        let values = |name: &str| -> Vec<f32> {
            reference[name]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect()
        };
        let style = values("style");
        let (pos_emb, txt_len, txt_in_dim) = super::super::session::load_pos_emb(&bundle).unwrap();
        let stream = danmo_mlx::Stream::gpu().unwrap();
        let mut models = super::Models::load(&bundle, pos_emb, txt_len, txt_in_dim, &stream).unwrap();
        let seed = std::env::var("DANQING_TEST_SEEDVR2_SEED")
            .ok()
            .and_then(|value| value.parse::<i64>().ok())
            .unwrap_or(42);
        let decoded = models.restore_chunk(&style, 32, 32, 1, seed, &stream).unwrap();
        let corrected = super::super::color::correct_mlx(&decoded, &style, 32, 32, &stream).unwrap();
        let corrected_same_input =
            super::super::color::correct_mlx(&values("decoded"), &style, 32, 32, &stream).unwrap();
        for (name, actual, reference_name) in [
            ("decoded", decoded, "decoded"),
            ("corrected_same_input", corrected_same_input, "corrected"),
            ("corrected", corrected, "corrected"),
        ] {
            let expected = values(reference_name);
            assert_eq!(actual.len(), expected.len(), "{name} length");
            let mut mse = 0.0f64;
            let mut max_abs = 0.0f32;
            let mut energy = 0.0f64;
            for (actual, expected) in actual.iter().zip(&expected) {
                let difference = actual - expected;
                mse += f64::from(difference * difference);
                max_abs = max_abs.max(difference.abs());
                energy += f64::from(expected * expected);
            }
            mse /= actual.len() as f64;
            let rel_l2 = (mse * actual.len() as f64 / energy.max(1e-20)).sqrt();
            eprintln!("seedvr2 image {name}: mse={mse:.9} max_abs={max_abs:.6} rel_l2={rel_l2:.6}");
            let limit = if name == "corrected" { 0.02 } else { 0.01 };
            assert!(rel_l2 < limit, "{name} relative L2 {rel_l2} exceeds {limit}");
        }
    }
}
