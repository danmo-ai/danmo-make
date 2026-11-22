use std::path::{Path, PathBuf};

use danmo_contracts::ImageGenerationRequest;
use danmo_mlx::Result as MlxResult;
use danmo_mlx::{Array, Dtype, Error, RandomKey, Stream};
use serde_json::json;

use super::config::{ErnieConfig, MistralConfig};
use super::dit::{CompiledForward, Prepared};
use super::resident::{cache_key, Component};
use super::tokenizer::ErnieTokenizer;
use super::{mistral, resident, vae};
use crate::common::ops::non_finite;
use crate::contract::{ExecutionContext, ResolvedRun};
use crate::error::EngineError;
use crate::families::{DitDecode, DitImageSession};
use crate::runtime::GgmlContext;

#[derive(Clone)]
struct TextBatch {
    values: Vec<f32>,
    lens: Vec<i32>,
    tokens: i32,
    hidden: i32,
}

pub(crate) struct MlxSession {
    bundle: PathBuf,
    request: ImageGenerationRequest,
    width: u32,
    height: u32,
    config: ErnieConfig,
    text_config: MistralConfig,
    pe_config: Option<MistralConfig>,
    tokenizer: ErnieTokenizer,
    pe_tokenizer: Option<ErnieTokenizer>,
    text: Option<TextBatch>,
    revised_prompt: Option<String>,
    sigmas: Option<Vec<f32>>,
    scheduler_shift: f32,
    train_timesteps: f32,
}

impl MlxSession {
    pub(crate) fn prepare(
        run: &ResolvedRun,
        request: &ImageGenerationRequest,
    ) -> Result<Self, EngineError> {
        let bundle = &run.bundle;
        if !crate::weights::is_safetensors_image_bundle(bundle) {
            return Err(EngineError::bad_request(format!(
                "{} is not an ERNIE safetensors bundle",
                bundle.display()
            )));
        }
        let (width, height) =
            danmo_contracts::parse_size(&request.size).map_err(EngineError::InvalidRequest)?;
        if width % 16 != 0 || height % 16 != 0 {
            return Err(EngineError::bad_request(format!(
                "ERNIE size {width}x{height} must be divisible by 16"
            )));
        }
        let steps = request.steps.unwrap_or(50);
        let guidance = request.guidance.unwrap_or(4.0);
        if steps == 0 || !guidance.is_finite() || guidance < 1.0 {
            return Err(EngineError::bad_request(
                "ERNIE steps must be positive and guidance must be finite and >= 1",
            ));
        }
        if guidance <= 1.0 && !request.negative_prompt.trim().is_empty() {
            return Err(EngineError::bad_request(
                "ERNIE negative prompt requires guidance > 1",
            ));
        }
        if request.use_empirical_mu {
            return Err(EngineError::bad_request(
                "ERNIE does not support empirical-mu sigma scheduling",
            ));
        }
        match request
            .scheduler
            .as_deref()
            .unwrap_or("flow_match_euler_discrete")
        {
            "flow_match_euler" | "flow_match_euler_discrete" => {}
            other => {
                return Err(EngineError::bad_request(format!(
                    "ERNIE scheduler {other} is not built"
                )))
            }
        }
        let config =
            ErnieConfig::load(&bundle.join("transformer")).map_err(EngineError::bad_request)?;
        if config.patch != 1 || config.in_channels != 128 || config.out_channels != 128 {
            return Err(EngineError::bad_request(
                "ERNIE requires patch_size 1 and 128 input/output channels",
            ));
        }
        let text_config = MistralConfig::load_text_encoder(&bundle.join("text_encoder"))
            .map_err(EngineError::bad_request)?;
        if text_config.hidden != config.text_dim {
            return Err(EngineError::bad_request(format!(
                "ERNIE text width {} does not match DiT {}",
                text_config.hidden, config.text_dim
            )));
        }
        let tokenizer = ErnieTokenizer::load(&bundle.join("tokenizer"), text_config.vocab)
            .map_err(EngineError::bad_request)?;
        let use_pe = request.use_pe.unwrap_or(true);
        let (pe_config, pe_tokenizer) = if use_pe {
            let config =
                MistralConfig::load_pe(&bundle.join("pe")).map_err(EngineError::bad_request)?;
            let tokenizer = ErnieTokenizer::load(&bundle.join("pe_tokenizer"), config.vocab)
                .map_err(EngineError::bad_request)?;
            (Some(config), Some(tokenizer))
        } else {
            (None, None)
        };
        let scheduler = read_scheduler(bundle)?;
        let shift = scheduler.0;
        if request
            .scheduler_shift
            .is_some_and(|value| (value - shift).abs() > 1e-6)
        {
            return Err(EngineError::bad_request(format!("ERNIE uses the checkpoint scheduler shift {shift}; a custom shift is not supported")));
        }
        Ok(Self {
            bundle: bundle.clone(),
            request: request.clone(),
            width,
            height,
            config,
            text_config,
            pe_config,
            tokenizer,
            pe_tokenizer,
            text: None,
            revised_prompt: None,
            sigmas: None,
            scheduler_shift: shift,
            train_timesteps: scheduler.1,
        })
    }

    fn enhance_prompt(&self, exec: &ExecutionContext) -> Result<Option<String>, EngineError> {
        if !self.request.use_pe.unwrap_or(true) {
            return Ok(None);
        }
        let tokenizer = self
            .pe_tokenizer
            .as_ref()
            .ok_or_else(|| EngineError::bad_request("ERNIE PE tokenizer is not loaded"))?;
        let config = self
            .pe_config
            .ok_or_else(|| EngineError::bad_request("ERNIE PE config is not loaded"))?;
        let prefix = tokenizer
            .encode_pe(&self.request.prompt, self.width, self.height)
            .map_err(EngineError::bad_request)?;
        let directory = self.bundle.join("pe");
        let key = cache_key(&directory).map_err(from_mlx)?;
        let seed = self.request.seed.unwrap_or(0) as u64 ^ 0x4552_4e49_455f_5045;
        let exec = exec.clone();
        let generated = crate::runtime::mlx_executor()
            .map_err(from_mlx)?
            .execute("ernie_image prompt enhance", move |_, stream, cache, _| {
                let result = (|| {
                    let weights =
                        resident::resident(cache, &key, Component::PromptEnhancer, stream)?;
                    let mut lm_cache = None;
                    let mut logits =
                        mistral::step_logits(&weights, config, &prefix, &mut lm_cache, stream)?;
                    let mut rng = SampleRng::new(seed);
                    let mut output = Vec::new();
                    for _ in 0..2048 {
                        exec.check_cancelled()
                            .map_err(|error| Error::new("ERNIE PE", error.to_string()))?;
                        let id = sample_top_p(&logits, 0.6, 0.95, &mut rng)?;
                        if id == config.eos {
                            break;
                        }
                        output.push(id);
                        exec.progress("prompt_enhance", output.len() as u32, 2048)
                            .map_err(|error| Error::new("ERNIE PE", error.to_string()))?;
                        logits =
                            mistral::step_logits(&weights, config, &[id], &mut lm_cache, stream)?;
                    }
                    if output.is_empty() {
                        return Err(Error::new("ERNIE PE", "prompt enhancer returned no tokens"));
                    }
                    Ok(output)
                })();
                resident::unload(cache, &key);
                result
            })
            .map_err(from_mlx)?;
        let text = tokenizer
            .decode(&generated)
            .map_err(EngineError::bad_request)?;
        if text.is_empty() {
            return Err(EngineError::bad_request(
                "ERNIE prompt enhancer returned an empty prompt",
            ));
        }
        Ok(Some(text))
    }

    fn encode_text(&self, prompt: &str) -> Result<TextBatch, EngineError> {
        let positive = self
            .tokenizer
            .encode_text(prompt)
            .map_err(EngineError::bad_request)?;
        let guidance = self.request.guidance.unwrap_or(4.0);
        let negative = if guidance > 1.0 {
            Some(
                self.tokenizer
                    .encode_text(&self.request.negative_prompt)
                    .map_err(EngineError::bad_request)?,
            )
        } else {
            None
        };
        let directory = self.bundle.join("text_encoder");
        let key = cache_key(&directory).map_err(from_mlx)?;
        let config = self.text_config;
        let (positive_values, positive_len, negative_values) = crate::runtime::mlx_executor()
            .map_err(from_mlx)?
            .execute("ernie_image text encode", move |_, stream, cache, _| {
                let result = (|| {
                    let weights = resident::resident(cache, &key, Component::TextEncoder, stream)?;
                    let positive = mistral::encode(&weights, config, &positive, stream)?;
                    let positive_len = positive.shape()[1];
                    let positive_values = positive.to_vec_f32(stream)?;
                    let negative_values = if let Some(ids) = negative.as_ref() {
                        let encoded = mistral::encode(&weights, config, ids, stream)?;
                        let values = encoded.to_vec_f32(stream)?;
                        Some((values, encoded.shape()[1]))
                    } else {
                        None
                    };
                    Ok((positive_values, positive_len, negative_values))
                })();
                resident::unload(cache, &key);
                result
            })
            .map_err(from_mlx)?;
        let (negative_values, negative_len) = negative_values
            .map(|(values, len)| (Some(values), len))
            .unwrap_or((None, 0));
        Ok(pad_text(
            positive_values,
            positive_len,
            negative_values,
            negative_len,
            config.hidden,
        ))
    }
}

impl DitImageSession for MlxSession {
    fn encode(&mut self, _ctx: &GgmlContext, exec: &ExecutionContext) -> Result<(), EngineError> {
        self.revised_prompt = self.enhance_prompt(exec)?;
        let prompt = self
            .revised_prompt
            .as_deref()
            .unwrap_or(&self.request.prompt);
        self.text = Some(self.encode_text(prompt)?);
        Ok(())
    }

    fn schedule(&mut self) -> Result<(), EngineError> {
        if self.text.is_none() {
            return Err(EngineError::bad_request(
                "ERNIE encode must run before schedule",
            ));
        }
        self.sigmas = Some(
            super::super::schedule(self.request.steps.unwrap_or(50), self.scheduler_shift)
                .map_err(EngineError::bad_request)?,
        );
        Ok(())
    }

    fn infer(&mut self, exec: &ExecutionContext) -> Result<Vec<f32>, EngineError> {
        let text = self
            .text
            .clone()
            .ok_or_else(|| EngineError::bad_request("ERNIE encode must run before infer"))?;
        let sigmas = self
            .sigmas
            .clone()
            .ok_or_else(|| EngineError::bad_request("ERNIE schedule must run before infer"))?;
        let seed = self
            .request
            .seed
            .ok_or_else(|| EngineError::bad_request("ERNIE create requires a seed"))?;
        if seed < 0 {
            return Err(EngineError::bad_request("ERNIE seed must be non-negative"));
        }
        let key = cache_key(&self.bundle.join("transformer")).map_err(from_mlx)?;
        let config = self.config;
        let grid_h = (self.height / 16) as i32;
        let grid_w = (self.width / 16) as i32;
        let guidance = self.request.guidance.unwrap_or(4.0);
        let train_timesteps = self.train_timesteps;
        let exec = exec.clone();
        crate::runtime::mlx_executor()
            .map_err(from_mlx)?
            .execute("ernie_image mlx infer", move |_, stream, cache, _| {
                denoise(
                    cache,
                    &key,
                    config,
                    &text,
                    &sigmas,
                    grid_h,
                    grid_w,
                    seed as u64,
                    guidance,
                    train_timesteps,
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
        let key = cache_key(&self.bundle.join("vae")).map_err(from_mlx)?;
        let grid_h = (self.height / 16) as i32;
        let grid_w = (self.width / 16) as i32;
        let width = self.width;
        let height = self.height;
        let latents = latents.to_vec();
        let pixels = crate::runtime::mlx_executor()
            .map_err(from_mlx)?
            .execute("ernie_image mlx decode", move |_, stream, cache, _| {
                let weights = resident::resident(cache, &key, Component::Vae, stream)?;
                vae::decode(&weights, &latents, grid_h, grid_w, width, height, stream)
            })
            .map_err(from_mlx)?;
        let mut metadata = json!({"use_pe": self.request.use_pe.unwrap_or(true)});
        if let Some(prompt) = self.revised_prompt {
            metadata["revised_prompt"] = json!(prompt);
        }
        Ok(DitDecode {
            preview_png: crate::common::png::encode_rgb(width, height, &pixels),
            graph_computes: self.request.steps.unwrap_or(50),
            metadata,
        })
    }
}

fn denoise(
    cache: &mut danmo_mlx::ModelCache,
    key: &danmo_mlx::CacheKey,
    config: ErnieConfig,
    text: &TextBatch,
    sigmas: &[f32],
    grid_h: i32,
    grid_w: i32,
    seed: u64,
    guidance: f32,
    train_timesteps: f32,
    exec: &ExecutionContext,
    stream: &Stream,
) -> MlxResult<Vec<f32>> {
    if sigmas.len() < 2 {
        return Err(Error::new("ERNIE", "denoise schedule is empty"));
    }
    let weights = resident::resident(cache, key, Component::Transformer, stream)?;
    let batch = text.lens.len() as i32;
    let text_array = Array::from_f32_slice(&text.values, &[batch, text.tokens, text.hidden])?
        .astype(Dtype::BFloat16, stream)?;
    let prepared = Prepared::new(config, batch, grid_h, grid_w, &text.lens, stream)?;
    let compiled = CompiledForward::new(weights, config, text_array, prepared, stream)?;
    let mut latent = RandomKey::new(seed)?
        .normal(
            &[1, config.in_channels, grid_h, grid_w],
            Dtype::Float32,
            stream,
        )?
        .astype(Dtype::BFloat16, stream)?;
    for index in 0..sigmas.len() - 1 {
        exec.check_cancelled()
            .map_err(|error| Error::new("ERNIE", error.to_string()))?;
        let input = if guidance > 1.0 {
            Array::concatenate_axis(&[latent.clone(), latent.clone()], 0, stream)?
        } else {
            latent.clone()
        };
        let timestep = Array::from_f32_slice(
            &vec![sigmas[index] * train_timesteps; batch as usize],
            &[batch],
        )?;
        let prediction = compiled.apply(&input, &timestep)?;
        let guided = if guidance > 1.0 {
            let uncond = prediction.slice(
                &[0, 0, 0, 0],
                &[1, config.out_channels, grid_h, grid_w],
                &[1, 1, 1, 1],
                stream,
            )?;
            let cond = prediction.slice(
                &[1, 0, 0, 0],
                &[2, config.out_channels, grid_h, grid_w],
                &[1, 1, 1, 1],
                stream,
            )?;
            let scale =
                Array::from_f32_slice(&[guidance], &[])?.astype(prediction.dtype(), stream)?;
            Array::add(
                &uncond,
                &Array::multiply(&scale, &Array::subtract(&cond, &uncond, stream)?, stream)?,
                stream,
            )?
        } else {
            prediction
        };
        let finite = guided
            .isfinite_all(stream)?
            .astype(Dtype::Float32, stream)?
            .to_vec_f32(stream)?;
        if finite.first().copied() != Some(1.0) {
            return Err(Error::new(
                "ERNIE",
                format!("DiT produced non-finite values at step {}", index + 1),
            ));
        }
        let dt = Array::from_f32_slice(&[sigmas[index + 1] - sigmas[index]], &[])?;
        let sample = latent.astype(Dtype::Float32, stream)?;
        let velocity = guided.astype(Dtype::Float32, stream)?;
        latent = Array::add(&sample, &Array::multiply(&velocity, &dt, stream)?, stream)?
            .astype(guided.dtype(), stream)?;
        latent.eval()?;
        exec.progress("denoise", index as u32 + 1, (sigmas.len() - 1) as u32)
            .map_err(|error| Error::new("ERNIE", error.to_string()))?;
    }
    let values = latent.to_vec_f32(stream)?;
    if let Some(detail) = non_finite("ERNIE latent", &values) {
        return Err(Error::new("ERNIE", detail));
    }
    Ok(values)
}

fn pad_text(
    positive: Vec<f32>,
    positive_len: i32,
    negative: Option<Vec<f32>>,
    negative_len: i32,
    hidden: i32,
) -> TextBatch {
    let tokens = positive_len.max(negative_len);
    let batch = if negative.is_some() { 2 } else { 1 };
    let mut values = vec![0.0; (batch * tokens * hidden) as usize];
    if let Some(negative) = negative {
        copy_text(
            &negative,
            negative_len,
            tokens,
            hidden,
            &mut values[..(negative_len * hidden) as usize],
        );
    }
    let offset = if batch == 2 {
        (tokens * hidden) as usize
    } else {
        0
    };
    copy_text(
        &positive,
        positive_len,
        tokens,
        hidden,
        &mut values[offset..offset + (positive_len * hidden) as usize],
    );
    let lens = if batch == 2 {
        vec![negative_len, positive_len]
    } else {
        vec![positive_len]
    };
    TextBatch {
        values,
        lens,
        tokens,
        hidden,
    }
}

fn copy_text(source: &[f32], source_len: i32, _tokens: i32, hidden: i32, target: &mut [f32]) {
    let expected = (source_len * hidden) as usize;
    target.copy_from_slice(&source[..expected]);
}

fn read_scheduler(bundle: &Path) -> Result<(f32, f32), EngineError> {
    let path = bundle.join("scheduler/scheduler_config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| EngineError::bad_request(format!("{}: {error}", path.display())))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| EngineError::bad_request(format!("{}: {error}", path.display())))?;
    let shift = value.get("shift").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let train = value
        .get("num_train_timesteps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1000.0) as f32;
    if !shift.is_finite()
        || shift <= 0.0
        || train <= 0.0
        || value
            .get("use_dynamic_shifting")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    {
        return Err(EngineError::bad_request(
            "ERNIE scheduler config is unsupported",
        ));
    }
    Ok((shift, train))
}

fn sample_top_p(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    rng: &mut SampleRng,
) -> MlxResult<i32> {
    let mut scored: Vec<(i32, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .map(|(id, value)| (id as i32, value / temperature))
        .collect();
    if scored.is_empty() {
        return Err(Error::new("ERNIE PE", "logits contain no finite values"));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let max = scored[0].1;
    let mut sum = 0.0;
    for (_, value) in scored.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    let mut cumulative = 0.0;
    let mut cut = scored.len();
    for (index, (_, value)) in scored.iter().enumerate() {
        cumulative += *value / sum;
        if cumulative >= top_p {
            cut = index + 1;
            break;
        }
    }
    scored.truncate(cut.max(1));
    let renorm: f32 = scored.iter().map(|(_, value)| *value).sum();
    let draw = rng.next_f32() * renorm;
    let mut cumulative = 0.0;
    for (id, value) in &scored {
        cumulative += *value;
        if draw <= cumulative {
            return Ok(*id);
        }
    }
    Ok(scored[0].0)
}

struct SampleRng(u64);

impl SampleRng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        })
    }

    fn next_f32(&mut self) -> f32 {
        let mut value = self.0;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.0 = value;
        ((value.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / ((1u32 << 24) as f32)
    }
}

fn from_mlx(error: Error) -> EngineError {
    if error.to_string().contains("cancel") {
        EngineError::Cancelled
    } else {
        crate::weights::WeightError::NotBuilt(error.to_string()).into()
    }
}
