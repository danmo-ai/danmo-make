//! Safetensors Z-Image create on the process MLX executor.
//!
//! The three component directories are admitted into `ModelCache`, then
//! encode, denoise, and decode run on that thread. Guidance 0 and 1 are the
//! single-forward path. Any other guidance, a negative prompt, LoRA, and a
//! structural guide fail instead of being dropped.

use std::path::{Path, PathBuf};

use danmo_contracts::ImageGenerationRequest;
use danmo_mlx::{Array, Dtype, Error, Stream};

use super::model::{forward_compiled_with_step_cache, CaptionCache, CompiledCompute};
use super::resident::{self, cache_key};
use super::step_cache::StepCache;
use super::MlxZImageConfig;
use crate::common::qwen3::Qwen3Config;
use crate::contract::ExecutionContext;
use crate::error::EngineError;
use crate::families::{DitDecode, DitImageSession};
use crate::runtime::GgmlContext;
use crate::weights::{MlxWeights, WeightError};

pub(crate) struct MlxSession {
    bundle: PathBuf,
    request: ImageGenerationRequest,
    ids: Vec<i32>,
    sigmas: Option<Vec<f32>>,
    width: u32,
    height: u32,
}

pub(crate) fn is_mlx_bundle(bundle: &Path) -> bool {
    crate::weights::is_safetensors_image_bundle(bundle)
}

impl MlxSession {
    pub(crate) fn prepare(
        bundle: &Path,
        request: &ImageGenerationRequest,
    ) -> Result<Self, EngineError> {
        if !is_mlx_bundle(bundle) {
            return Err(EngineError::bad_request(format!(
                "{} is not an MLX Z-Image bundle",
                bundle.display()
            )));
        }
        if !guidance_is_single_pass(request.guidance) {
            return Err(EngineError::bad_request(
                "MLX Z-Image create does not apply guidance other than 0 or 1",
            ));
        }
        if !request.negative_prompt.trim().is_empty() {
            return Err(EngineError::bad_request(
                "MLX Z-Image create does not apply a negative prompt",
            ));
        }
        if request
            .structural_guide
            .as_ref()
            .is_some_and(|guide| guide.weight > 0.0)
        {
            return Err(EngineError::bad_request(
                "MLX Z-Image create does not apply a structural guide",
            ));
        }
        let (width, height) =
            danmo_contracts::parse_size(&request.size).map_err(EngineError::InvalidRequest)?;
        Ok(Self {
            bundle: bundle.to_path_buf(),
            request: request.clone(),
            ids: Vec::new(),
            sigmas: None,
            width,
            height,
        })
    }
}

impl DitImageSession for MlxSession {
    fn encode(&mut self, _ctx: &GgmlContext) -> Result<(), EngineError> {
        let config = MlxZImageConfig::published();
        resident::load_transformer(
            &self.bundle.join("transformer"),
            config,
            crate::weights::WeightDtype::Native,
            None,
        )
        .map_err(from_mlx)?;
        resident::load_text_encoder(&self.bundle.join("text_encoder")).map_err(from_mlx)?;
        resident::load_vae(&self.bundle.join("vae")).map_err(from_mlx)?;
        self.ids = config.token_ids(&self.bundle, &self.request.prompt)?;
        Ok(())
    }

    fn schedule(&mut self) -> Result<(), EngineError> {
        if self.ids.is_empty() {
            return Err(EngineError::bad_request("encode must run before schedule"));
        }
        self.sigmas = Some(super::super::schedule(
            schedule_config(),
            self.request.steps.unwrap_or(4),
            self.width,
            self.height,
            self.request.scheduler.as_deref(),
            self.request.scheduler_shift,
            self.request.use_empirical_mu,
        )?);
        Ok(())
    }

    fn infer(&mut self, exec: &ExecutionContext) -> Result<Vec<f32>, EngineError> {
        let sigmas = self
            .sigmas
            .clone()
            .ok_or_else(|| EngineError::bad_request("schedule must run before infer"))?;
        let seed = self
            .request
            .seed
            .ok_or_else(|| EngineError::bad_request("MLX Z-Image create requires a seed"))?;
        if seed < 0 {
            return Err(EngineError::bad_request(
                "MLX Z-Image seed must be non-negative",
            ));
        }
        let text_key = cache_key(&self.bundle.join("text_encoder")).map_err(from_mlx)?;
        let dit_key = cache_key(&self.bundle.join("transformer")).map_err(from_mlx)?;
        let ids = self.ids.clone();
        let width = self.width;
        let height = self.height;
        let exec = exec.clone();
        crate::runtime::mlx_executor()
            .map_err(from_mlx)?
            .execute("z_image mlx infer", move |_, stream, cache, _| {
                denoise(
                    cache,
                    &text_key,
                    &dit_key,
                    &ids,
                    width,
                    height,
                    seed as u64,
                    &sigmas,
                    &exec,
                    stream,
                )
            })
            .map_err(from_mlx)
    }

    fn decode(
        self: Box<Self>,
        _ctx: &GgmlContext,
        latents: &[f32],
    ) -> Result<DitDecode, EngineError> {
        let vae_key = cache_key(&self.bundle.join("vae")).map_err(from_mlx)?;
        let width = self.width;
        let height = self.height;
        let latents = latents.to_vec();
        let pixels = crate::runtime::mlx_executor()
            .map_err(from_mlx)?
            .execute("z_image mlx decode", move |_, stream, cache, _| {
                decode_pixels(cache, &vae_key, &latents, width, height, stream)
            })
            .map_err(from_mlx)?;
        Ok(DitDecode {
            preview_png: crate::common::png::encode_rgb(width, height, &pixels),
            graph_computes: 1,
        })
    }
}

fn denoise(
    cache: &mut danmo_mlx::ModelCache,
    text_key: &danmo_mlx::CacheKey,
    dit_key: &danmo_mlx::CacheKey,
    ids: &[i32],
    width: u32,
    height: u32,
    seed: u64,
    sigmas: &[f32],
    exec: &ExecutionContext,
    stream: &Stream,
) -> danmo_mlx::Result<Vec<f32>> {
    let config = MlxZImageConfig::published();
    let grid = super::super::geom::token_grid(width, height, ids.len() as i64, config.vae_scale)
        .map_err(|error| Error::new("Z-Image denoise", error.to_string()))?;
    let caption = {
        let text = resident_weights(cache, text_key, "text encoder")?;
        let caption = crate::common::qwen3::mlx::encode_ids(
            text,
            text_config(),
            "model.",
            ids,
            &vec![true; ids.len()],
            &[35],
            stream,
        )?;
        caption
            .squeeze_axis(0, stream)?
            .astype(Dtype::BFloat16, stream)?
    };
    caption.eval()?;
    let weights = resident_weights(cache, dit_key, "transformer")?.clone();
    let caption_cache = CaptionCache::new(&weights, config, grid, &caption, stream)?;
    let compiled_key = danmo_mlx::CacheKey {
        id: format!(
            "{}#compiled-dit#{}x{}#cap{}",
            dit_key.id, grid.tokens_w, grid.tokens_h, grid.cap_len
        ),
        revision: dit_key.revision,
    };
    if cache.get::<CompiledCompute>(&compiled_key)?.is_none() {
        let compiled = CompiledCompute::new(&weights, config, grid, stream)?;
        let bytes = weights.nbytes();
        cache.insert(compiled_key.clone(), compiled, bytes)?;
    }
    let compiled = cache
        .get::<CompiledCompute>(&compiled_key)?
        .ok_or_else(|| {
            Error::new(
                "Z-Image denoise",
                "compiled DiT is not in the executor cache",
            )
        })?;
    let latent_h = (height as i64 / config.vae_scale) as i32;
    let latent_w = (width as i64 / config.vae_scale) as i32;
    let latent = danmo_mlx::RandomKey::new(seed)?
        .normal(&[1, 16, latent_h, latent_w], Dtype::Float32, stream)?
        .squeeze_axis(0, stream)?
        .astype(Dtype::BFloat16, stream)?;
    let mut step_cache = StepCache::z_image(sigmas.len().saturating_sub(1));
    let mut step = 0usize;
    let total = (sigmas.len().saturating_sub(1)) as u32;
    let denoised =
        super::denoise::euler_cancellable(&latent, sigmas, stream, None, |current, sigma| {
            if exec.check_cancelled().is_err() {
                return Err(Error::new(
                    "Z-Image denoise",
                    format!("cancelled at step {step}"),
                ));
            }
            exec.progress("denoise", step as u32, total)
                .map_err(|error| Error::new("Z-Image denoise", error.to_string()))?;
            let (velocity, _) = forward_compiled_with_step_cache(
                &weights,
                config,
                compiled,
                current,
                &caption_cache,
                sigma,
                &mut step_cache,
                step,
                stream,
            )?;
            step += 1;
            Ok(velocity)
        })?;
    denoised.eval()?;
    denoised.to_vec_f32(stream)
}

fn decode_pixels(
    cache: &mut danmo_mlx::ModelCache,
    vae_key: &danmo_mlx::CacheKey,
    latents: &[f32],
    width: u32,
    height: u32,
    stream: &Stream,
) -> danmo_mlx::Result<Vec<u8>> {
    let weights = resident_weights(cache, vae_key, "vae")?.clone();
    let config = MlxZImageConfig::published();
    let latent_h = (height as i64 / config.vae_scale) as i32;
    let latent_w = (width as i64 / config.vae_scale) as i32;
    let expected = 16 * latent_h * latent_w;
    if latents.len() != expected as usize {
        return Err(Error::new(
            "Z-Image decode",
            format!("latent has {} values, expected {expected}", latents.len()),
        ));
    }
    let latent = Array::from_f32_slice(latents, &[1, 16, latent_h, latent_w])?
        .astype(Dtype::BFloat16, stream)?;
    let spec = vae_spec();
    let decoder_key = danmo_mlx::CacheKey {
        id: format!("{}#compiled-vae#{latent_h}x{latent_w}", vae_key.id),
        revision: vae_key.revision,
    };
    if cache
        .get::<crate::common::vae::mlx::CompiledDecoder>(&decoder_key)?
        .is_none()
    {
        let decoder =
            crate::common::vae::mlx::CompiledDecoder::new(weights.clone(), spec, stream.clone())?;
        let bytes = weights.nbytes();
        cache.insert(decoder_key.clone(), decoder, bytes)?;
    }
    let decoder = cache
        .get::<crate::common::vae::mlx::CompiledDecoder>(&decoder_key)?
        .ok_or_else(|| {
            Error::new(
                "Z-Image decode",
                "compiled VAE is not in the executor cache",
            )
        })?;
    let decoded = decoder.apply(&latent)?;
    decoded.eval()?;
    let values = decoded.to_vec_f32(stream)?;
    let plane = (width as usize).saturating_mul(height as usize);
    if values.len() != plane * 3 {
        return Err(Error::new(
            "Z-Image decode",
            format!(
                "decoded {} values, expected {} for {width}x{height}",
                values.len(),
                plane * 3
            ),
        ));
    }
    Ok((0..plane * 3)
        .map(|index| {
            let channel = index % 3;
            let pixel = index / 3;
            let value = values[channel * plane + pixel];
            ((value + 1.0) * 127.5).clamp(0.0, 255.0).round_ties_even() as u8
        })
        .collect())
}

fn vae_spec() -> crate::common::vae::VaeSpec {
    crate::common::vae::VaeSpec {
        latent_channels: 16,
        block_out: vec![128, 256, 512, 512],
        resnets: 2,
        groups: 32,
        scaling: 0.3611,
        shift: 0.1159,
        attention: true,
    }
}

fn resident_weights<'a>(
    cache: &'a mut danmo_mlx::ModelCache,
    key: &danmo_mlx::CacheKey,
    role: &str,
) -> danmo_mlx::Result<&'a MlxWeights> {
    cache.get::<MlxWeights>(key)?.ok_or_else(|| {
        Error::new(
            "Z-Image resident",
            format!("{role} is not in the executor cache"),
        )
    })
}

fn text_config() -> Qwen3Config {
    Qwen3Config {
        vocab: 151_936,
        hidden: 2560,
        layers: 36,
        heads: 32,
        kv_heads: 8,
        head_dim: 128,
        intermediate: 9728,
        rope_theta: 1_000_000.0,
    }
}

fn schedule_config() -> super::super::weights::ZImageConfig {
    let config = MlxZImageConfig::published();
    super::super::weights::ZImageConfig {
        dim: config.dim,
        heads: config.heads,
        head_dim: config.head_dim(),
        axes: config.axes,
        layers: config.layers,
        refiners: config.refiners,
        in_channels: config.in_channels,
        cap_dim: config.cap_dim,
        ffn: config.ffn_hidden(),
        mid: config.timestep_mid,
        freq: config.frequency_dim,
        adaln: config.adaln_dim,
        rope_theta: config.rope_theta,
        t_scale: config.t_scale,
        vae_scale: config.vae_scale,
        patch: config.patch,
        requires_sigma_shift: true,
    }
}

fn from_mlx(error: Error) -> EngineError {
    let message = error.to_string();
    if message.contains("cancelled at step") {
        EngineError::Cancelled
    } else {
        WeightError::NotBuilt(message).into()
    }
}

fn guidance_is_single_pass(guidance: Option<f32>) -> bool {
    match guidance {
        None => true,
        Some(value) => value.abs() < 1e-6 || (value - 1.0).abs() < 1e-6,
    }
}

#[cfg(test)]
mod tests {
    use super::{is_mlx_bundle, MlxSession};
    use crate::contract::{ExecutionContext, NoopObserver};
    use crate::families::DitImageSession;
    use crate::runtime::{BackendKind, GgmlContext};
    use danmo_contracts::ImageGenerationRequest;
    use danmo_mlx::{Array, Stream};
    use std::path::Path;
    use std::sync::Arc;

    fn write_shard(path: &Path) {
        let values = [0.5f32; 4];
        let array = Array::from_f32_slice(&values, &[1, 4]).unwrap();
        danmo_mlx::save_safetensors(path, &[("layers.0.attention_norm1.weight", &array)], &[])
            .unwrap();
        let _ = Stream::cpu();
    }

    fn request() -> ImageGenerationRequest {
        serde_json::from_value(serde_json::json!({
            "model": "z-image-turbo",
            "prompt": "red",
            "size": "16x16",
            "steps": 2,
            "guidance": 0,
            "seed": 1
        }))
        .unwrap()
    }

    #[test]
    fn safetensors_bundle_reaches_the_mlx_text_encoder_and_keeps_weights_resident() {
        let bundle =
            std::env::temp_dir().join(format!("zimage-mlx-session-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&bundle);
        for role in ["transformer", "text_encoder", "vae"] {
            let directory = bundle.join(role);
            std::fs::create_dir_all(&directory).unwrap();
            write_shard(&directory.join("model.safetensors"));
        }
        std::fs::write(bundle.join("tokenizer.json"), r#"{"vocab":{"red":1}}"#).unwrap();
        assert!(is_mlx_bundle(&bundle));

        let mut session = MlxSession::prepare(&bundle, &request()).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        session.encode(&ctx).unwrap();
        let again = super::resident::load_transformer(
            &bundle.join("transformer"),
            super::MlxZImageConfig::published(),
            crate::weights::WeightDtype::Native,
            None,
        )
        .unwrap();
        assert!(again.hit);
        session.schedule().unwrap();
        let exec = ExecutionContext::new("mlx-session", Arc::new(NoopObserver));
        let error = session.infer(&exec).unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("embed_tokens") || message.contains("missing tensor"),
            "{message}"
        );
        assert!(!message.contains("old MLX"));
        let _ = std::fs::remove_dir_all(&bundle);
    }

    /// Real safetensors bundle through the create session. Writes a PNG.
    #[test]
    #[ignore]
    fn real_session_writes_a_png() {
        let bundle = std::path::PathBuf::from(std::env::var("DANQING_TEST_ZIMAGE_BUNDLE").unwrap());
        let output = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_ZIMAGE_SESSION_PNG")
                .unwrap_or_else(|_| "out/zimage-mlx-session.png".into()),
        );
        let width = std::env::var("DANQING_TEST_ZIMAGE_WIDTH").unwrap_or_else(|_| "768".into());
        let height = std::env::var("DANQING_TEST_ZIMAGE_HEIGHT").unwrap_or_else(|_| "1280".into());
        let steps: u32 = std::env::var("DANQING_TEST_ZIMAGE_STEPS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(9);
        let prompt = std::env::var("DANQING_TEST_ZIMAGE_PROMPT")
            .unwrap_or_else(|_| "a red cube on a white table".into());
        let request = serde_json::from_value(serde_json::json!({
            "model": "z-image-turbo",
            "prompt": prompt,
            "size": format!("{width}x{height}"),
            "steps": steps,
            "guidance": 0,
            "seed": 1,
            "scheduler": "linear"
        }))
        .unwrap();
        let mut session = MlxSession::prepare(&bundle, &request).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        session.encode(&ctx).unwrap();
        session.schedule().unwrap();
        let exec = ExecutionContext::new("mlx-session-real", Arc::new(NoopObserver));
        let latents = session.infer(&exec).unwrap();
        assert!(latents.iter().any(|value| value.abs() > 1e-4), "latent is empty");
        let decoded = Box::new(session).decode(&ctx, &latents).unwrap();
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&output, &decoded.preview_png).unwrap();
        assert_eq!(&decoded.preview_png[..8], b"\x89PNG\r\n\x1a\n");
        let distinct = decoded.preview_png.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(distinct > 8, "png has only {distinct} distinct bytes: {}", output.display());
        eprintln!("wrote {}", output.display());
    }
}
