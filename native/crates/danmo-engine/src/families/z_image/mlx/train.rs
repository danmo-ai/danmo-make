//! Full-blood Z-Image LoRA training on the MLX executor.
//!
//! Real pipeline: dataset images → VAE latents, captions → Qwen3 hidden
//! states → caption refiner cache, then Flow-Match SFT through the actual
//! 30-block DiT. LoRA A/B (F32) ride the inference `LinearWeight` residual
//! on the frozen base; each block runs inside an `mx.checkpoint` closure so
//! backward recomputes instead of retaining 30 blocks of activations.
//!
//! Conventions are the inference ones by construction: `unified_before` /
//! `run_block` / `final_velocity` are the same functions generation calls,
//! and the loss is `train::flow_match_mse` (`v = ε − x0`, matching the
//! negated head in `unpatchify_velocity`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use danmo_contracts::{LoraTrainPreset, LoraTrainingRequest, LoraTrainingResult};
use danmo_mlx::{
    adamw_update_all, AdamWConfig, AdamWState, Array, CacheKey, Closure, Dtype, Error,
    LinearWeight, LoraResidual, ModelCache, RandomKey, Result, Stream, ValueAndGrad,
};
use serde_json::json;

use super::super::geom::{self, TokenGrid};
use super::attention::AttentionWeights;
use super::blocks::{self, BlockWeights};
use super::model::{self, CaptionCache, ForwardRope};
use super::{session, MlxZImageConfig};
use crate::error::EngineError;
use crate::train::{
    export_lora_adapters, flow_match_mse, init_lora_params, sample_discrete_sigma,
    sample_noisy_latent, FlowMatchTrainSchedule, LoraTrainParam,
};
use crate::weights::MlxWeights;

/// Attention + FFN projection slots the recipe trains, in block order.
const SLOTS: [(&str, &str); 7] = [
    ("attention.to_q", "to_q"),
    ("attention.to_k", "to_k"),
    ("attention.to_v", "to_v"),
    ("attention.to_out", "to_out"),
    ("feed_forward.w1", "w1"),
    ("feed_forward.w2", "w2"),
    ("feed_forward.w3", "w3"),
];

use crate::train::{TrainEvent, TrainProgress};

/// Everything the executor job needs, resolved on the host beforehand.
pub struct TrainJob {
    pub request: LoraTrainingRequest,
    pub preset: LoraTrainPreset,
    /// Token ids per image caption plus the empty caption (dropout variant).
    pub caption_ids: Vec<Vec<i32>>,
    pub empty_ids: Vec<i32>,
    /// Preprocessed NCHW `[-1, 1]` pixels at the recipe resolution.
    pub images: Vec<Vec<f32>>,
    pub width: u32,
    pub height: u32,
    /// Previously exported adapter to resume from (same naming as export).
    pub resume_adapter: Option<PathBuf>,
    pub work_dir: PathBuf,
    pub dit_key: CacheKey,
    pub text_key: CacheKey,
    pub vae_key: CacheKey,
}

fn mlx(error: Error) -> EngineError {
    EngineError::message(error.to_string())
}

struct TrainVariant {
    grid: TokenGrid,
    rope: ForwardRope,
    caption: CaptionCache,
}

struct TrainSample {
    latent: Array,
    captioned: TrainVariant,
    dropped: TrainVariant,
}

struct TargetModule {
    /// Graph parameter stem, e.g. `layers.3.attention.to_q`.
    name: String,
    out_dim: i32,
    in_dim: i32,
}

/// Expand the recipe into concrete target modules; also yields, per layer,
/// the slot → global param index map used to wire closure inputs.
fn target_modules(
    config: &MlxZImageConfig,
    preset: &LoraTrainPreset,
) -> Result<(Vec<TargetModule>, Vec<[Option<usize>; 7]>)> {
    let layers = if preset.lora_blocks == -1 {
        config.layers
    } else if preset.lora_blocks >= 1 && i64::from(preset.lora_blocks) <= config.layers {
        i64::from(preset.lora_blocks)
    } else {
        return Err(Error::new(
            "Z-Image train",
            format!(
                "lora_blocks {} is outside -1 or 1..={}",
                preset.lora_blocks, config.layers
            ),
        ));
    };
    let dim = config.dim as i32;
    let ffn = config.ffn_hidden() as i32;
    let mut modules = Vec::new();
    let mut slots = vec![[None; 7]; config.layers as usize];
    for layer in 0..layers {
        for (slot, (suffix, key)) in SLOTS.iter().enumerate() {
            if !preset.lora_module_keys.iter().any(|k| k == key) {
                continue;
            }
            let (out_dim, in_dim) = match *key {
                "to_q" | "to_k" | "to_v" | "to_out" => (dim, dim),
                "w1" | "w3" => (ffn, dim),
                "w2" => (dim, ffn),
                other => {
                    return Err(Error::new(
                        "Z-Image train",
                        format!("unknown lora module key {other}"),
                    ))
                }
            };
            slots[layer as usize][slot] = Some(modules.len());
            modules.push(TargetModule {
                name: format!("layers.{layer}.{suffix}"),
                out_dim,
                in_dim,
            });
        }
    }
    if modules.is_empty() {
        return Err(Error::new(
            "Z-Image train",
            "lora_module_keys selected no target modules",
        ));
    }
    Ok((modules, slots))
}

/// SplitMix64 host RNG: dataset order, dropout, and sigma sampling stay
/// reproducible from `request.seed` without consuming MLX key splits.
struct HostRng(u64);

impl HostRng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn normal(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Trainer state written next to every checkpoint. A restarted process
/// resumes at the checkpoint's step instead of re-running from zero.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
struct TrainerState {
    step: u32,
    iterations: u32,
    preset: String,
}

/// `checkpoint-500.safetensors` → `checkpoint-500.state.json`.
fn trainer_state_path(checkpoint: &Path) -> PathBuf {
    let stem = checkpoint
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "trainer".into());
    checkpoint.with_file_name(format!("{stem}.state.json"))
}

fn write_trainer_state(checkpoint: &Path, state: &TrainerState) -> Result<()> {
    let text = serde_json::to_string_pretty(state)
        .map_err(|e| Error::new("Z-Image trainer state", e.to_string()))?;
    std::fs::write(trainer_state_path(checkpoint), text)
        .map_err(|e| Error::new("Z-Image trainer state", e.to_string()))
}

/// Missing state file = pre-resume checkpoint (warm start); corrupt file is
/// a hard error — never guess the step count.
fn read_trainer_state(checkpoint: &Path) -> Result<Option<TrainerState>> {
    let path = trainer_state_path(checkpoint);
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|e| Error::new("Z-Image trainer state", e.to_string()))?;
    let state: TrainerState = serde_json::from_str(&text)
        .map_err(|e| Error::new("Z-Image trainer state", format!("{}: {e}", path.display())))?;
    Ok(Some(state))
}

/// Validate a checkpoint's trainer state against this run; returns the step
/// to resume from. Fail loud on mismatch — silently re-running or skipping
/// steps would corrupt the recipe.
fn resolve_start_step(
    state: Option<TrainerState>,
    iterations: u32,
    preset_id: &str,
) -> Result<u32> {
    let Some(state) = state else {
        return Ok(0);
    };
    if state.preset != preset_id {
        return Err(Error::new(
            "Z-Image train resume",
            format!(
                "checkpoint was saved by preset {:?} but this run uses {preset_id:?}",
                state.preset
            ),
        ));
    }
    if state.iterations != iterations {
        return Err(Error::new(
            "Z-Image train resume",
            format!(
                "checkpoint is step {}/{} but this run targets {iterations} iterations; \
                 match the recipe or start a fresh run",
                state.step, state.iterations
            ),
        ));
    }
    if state.step >= iterations {
        return Err(Error::new(
            "Z-Image train resume",
            format!(
                "checkpoint already completed {}/{} steps; nothing left to train",
                state.step, state.iterations
            ),
        ));
    }
    Ok(state.step)
}

/// Per-epoch dataset order derived from (seed, epoch): a restarted process
/// reproduces the same sequence without replaying prior steps.
fn epoch_order(seed: u64, epoch: u32, len: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..len).collect();
    let mut rng = HostRng(seed ^ 0xE0C4_A11C_5AA5_E55u64.wrapping_add(epoch as u64));
    for i in (1..len).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    order
}

/// Per-step RNG (caption dropout + sigma): independent of process boundaries.
fn step_rng(seed: u64, step: u32) -> HostRng {
    HostRng(
        seed ^ 0xD1B5_4A32_D192_ED03u64
            .wrapping_add((step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
    )
}

/// Per-step noise key (eps): same step ⇒ same noise after a restart.
fn step_noise_key(seed: u64, step: u32) -> Result<RandomKey> {
    RandomKey::new(
        seed.wrapping_add(1)
            .wrapping_add((step as u64).wrapping_mul(0x85EB_CA6B)),
    )
}

fn prepare_variant(
    dit: &MlxWeights,
    config: MlxZImageConfig,
    caption_ids: &[i32],
    text: &MlxWeights,
    width: u32,
    height: u32,
    stream: &Stream,
) -> Result<TrainVariant> {
    let grid = geom::token_grid(width, height, caption_ids.len() as i64, config.vae_scale)
        .map_err(|error| Error::new("Z-Image train grid", error.to_string()))?;
    let valid = vec![true; caption_ids.len()];
    let caption = crate::common::qwen3::mlx::encode_ids(
        text,
        session::text_config(),
        "model.",
        caption_ids,
        &valid,
        &[35],
        stream,
    )?
    .squeeze_axis(0, stream)?
    .astype(Dtype::BFloat16, stream)?;
    caption.eval()?;
    let caption = CaptionCache::new(dit, config, grid, &caption, stream)?;
    let rope = ForwardRope::new(config, grid, stream)?;
    Ok(TrainVariant {
        grid,
        rope,
        caption,
    })
}

fn prepare_samples(
    job: &TrainJob,
    dit: &MlxWeights,
    text: &MlxWeights,
    vae: &MlxWeights,
    config: MlxZImageConfig,
    stream: &Stream,
    progress: &TrainProgress,
) -> Result<Vec<TrainSample>> {
    let spec = session::vae_spec();
    let mut samples = Vec::with_capacity(job.images.len());
    for (index, pixels) in job.images.iter().enumerate() {
        let image = Array::from_f32_slice(pixels, &[1, 3, job.height as i32, job.width as i32])?;
        let latent = crate::common::vae::mlx::encode(vae, &spec, &image, stream)?;
        latent.eval()?;
        let captioned = prepare_variant(
            dit,
            config,
            &job.caption_ids[index],
            text,
            job.width,
            job.height,
            stream,
        )?;
        let dropped = prepare_variant(
            dit,
            config,
            &job.empty_ids,
            text,
            job.width,
            job.height,
            stream,
        )?;
        samples.push(TrainSample {
            latent,
            captioned,
            dropped,
        });
        progress(TrainEvent::Phase(format!(
            "encode cache {}/{} images",
            index + 1,
            job.images.len()
        )));
    }
    Ok(samples)
}

/// Block closures shared by every step of one token grid: 30 checkpointed
/// blocks with the frozen weights, rope table, and position ids captured.
struct VariantClosures {
    grid: TokenGrid,
    blocks: Vec<Closure>,
    /// Number of trained A/B factor pairs (global LoRA params × 2).
    n_train: usize,
    /// `value_and_grad` built **once** per token-grid variant: batch tensors
    /// (x0, timestep, unified, eps, train_weight) are closure inputs, not
    /// captures, so the traced 30-block graph is reused every step.
    vag: ValueAndGrad,
}

/// One transformer block wrapped in `mx.checkpoint`. Inputs:
/// `[x, timestep, a0, b0, a1, b1, ...]` — the LoRA factors must be inputs
/// (not captures) so the backward recomputation reconnects them to the
/// traced graph. `slots` maps each projection to its local input pair.
fn checkpointed_block(
    dit: &MlxWeights,
    config: MlxZImageConfig,
    layer: i64,
    slot_params: [Option<usize>; 7],
    param_scales: &[f32],
    ids: Vec<[i32; 3]>,
    rope: Array,
    stream: &Stream,
) -> Result<Closure> {
    let weights = dit.clone();
    let prefix = format!("layers.{layer}");
    let names: Vec<String> = SLOTS
        .iter()
        .map(|(suffix, _)| format!("{prefix}.{suffix}.weight"))
        .collect();
    // Local input positions in slot order (0..k), matching how the outer
    // closure packs this layer's apply inputs.
    let local: Vec<Option<usize>> = {
        let mut next = 0usize;
        slot_params
            .iter()
            .map(|slot| {
                slot.map(|_| {
                    let index = next;
                    next += 1;
                    index
                })
            })
            .collect()
    };
    let scales: Vec<f32> = slot_params
        .iter()
        .filter_map(|slot| slot.map(|global| param_scales[global]))
        .collect();
    Closure::new(stream.clone(), move |ins, call_stream| {
        let x = &ins[0];
        let timestep = &ins[1];
        let linear_at = |slot: usize| -> Result<LinearWeight> {
            let base = weights.linear(&names[slot])?;
            match local[slot] {
                Some(index) => Ok(base.with_loras(vec![LoraResidual {
                    a: ins[2 + index * 2].clone(),
                    b: ins[2 + index * 2 + 1].clone(),
                    scale: scales[index],
                }])),
                None => Ok(base.clone()),
            }
        };
        let tensor = |name: &str| weights.tensor(name);
        blocks::forward_with_table(
            x,
            BlockWeights {
                attention: AttentionWeights {
                    to_q: &linear_at(0)?,
                    to_k: &linear_at(1)?,
                    to_v: &linear_at(2)?,
                    to_out: &linear_at(3)?,
                    norm_q: tensor(&format!("{prefix}.attention.norm_q.weight"))?,
                    norm_k: tensor(&format!("{prefix}.attention.norm_k.weight"))?,
                },
                attention_norm1: tensor(&format!("{prefix}.attn_norm1.weight"))?,
                attention_norm2: tensor(&format!("{prefix}.attn_norm2.weight"))?,
                ffn_norm1: tensor(&format!("{prefix}.ffn_norm1.weight"))?,
                ffn_norm2: tensor(&format!("{prefix}.ffn_norm2.weight"))?,
                w1: &linear_at(4)?,
                w2: &linear_at(5)?,
                w3: &linear_at(6)?,
                adaln: Some((
                    weights.linear(&format!("{prefix}.adaLN_modulation.0.weight"))?,
                    tensor(&format!("{prefix}.adaLN_modulation.0.bias"))?,
                )),
            },
            config.heads as i32,
            &ids,
            config.axes.map(|value| value as i32),
            config.rope_theta,
            config.norm_eps,
            Some(timestep),
            None,
            Some(&rope),
            call_stream,
        )
    })?
    .checkpoint()
}

fn variant_closures(
    cache: &mut HashMap<i64, Rc<VariantClosures>>,
    variant: &TrainVariant,
    dit: &MlxWeights,
    config: MlxZImageConfig,
    slot_params: &[[Option<usize>; 7]],
    param_scales: &[f32],
    n_train: usize,
    stream: &Stream,
) -> Result<Rc<VariantClosures>> {
    let key = variant.grid.cap_len;
    if let Some(existing) = cache.get(&key) {
        return Ok(existing.clone());
    }
    let ids = geom::positions_image_first(variant.grid);
    let mut blocks = Vec::with_capacity(config.layers as usize);
    for layer in 0..config.layers {
        blocks.push(checkpointed_block(
            dit,
            config,
            layer,
            slot_params[layer as usize],
            param_scales,
            ids.clone(),
            variant.rope.unified.clone(),
            stream,
        )?);
    }
    let weights = dit.clone();
    let grid = variant.grid;
    let blocks_rc = Rc::new(blocks);
    let slots = Rc::new(slot_params.to_vec());
    let n = n_train;
    // Closure inputs (order):
    //   ins[0..n_train]      trained A/B factors (argnums 0..n_train)
    //   ins[n_train + 0]     x0 latent (Float32)
    //   ins[n_train + 1]     timestep embedding
    //   ins[n_train + 2]     unified tokens (pre-block)
    //   ins[n_train + 3]     eps noise (Float32)
    //   ins[n_train + 4]     train_weight scalar
    let closure = Closure::new(stream.clone(), move |ins, call_stream| {
        let x0 = &ins[n];
        let timestep = &ins[n + 1];
        let mut x = ins[n + 2].clone();
        let eps = &ins[n + 3];
        let weight = &ins[n + 4];
        for (layer, block) in blocks_rc.iter().enumerate() {
            let mut inputs = vec![x.clone(), timestep.clone()];
            for slot in slots[layer] {
                if let Some(global) = slot {
                    inputs.push(ins[global * 2].clone());
                    inputs.push(ins[global * 2 + 1].clone());
                }
            }
            x = block.apply(&inputs)?;
        }
        let velocity = model::final_velocity(
            &weights,
            config,
            grid,
            &x,
            timestep,
            call_stream,
            &mut |_, _| {},
        )?;
        let loss = flow_match_mse(&velocity, x0, eps, call_stream)?;
        Array::multiply(&loss, weight, call_stream)
    })
    .map_err(|e| Error::new("Z-Image train closure", e.to_string()))?;
    let argnums: Vec<i32> = (0..n_train as i32).collect();
    let vag = ValueAndGrad::new(closure, &argnums)
        .map_err(|e| Error::new("Z-Image train value_and_grad", e.to_string()))?;
    let built = Rc::new(VariantClosures {
        grid: variant.grid,
        blocks: Vec::new(),
        n_train,
        vag,
    });
    cache.insert(key, built.clone());
    Ok(built)
}

/// Load resume factors (and alpha) from a previously exported adapter file.
/// Goes through the shared LoRA loader (same key parsing as inference);
/// every selected module must be present in the file.
fn resume_params(path: &Path, params: &mut [LoraTrainParam], stream: &Stream) -> Result<()> {
    let layout: Vec<(String, i64, i64)> = params
        .iter()
        .map(|p| {
            (
                format!("{}.weight", p.module),
                p.in_dim as i64,
                p.out_dim as i64,
            )
        })
        .collect();
    let file = crate::weights::LoraFile {
        id: "resume".into(),
        path: path.to_path_buf(),
        weight: 1.0,
    };
    let factors = crate::weights::load_lora_factors(&[file], &layout)
        .map_err(|error| Error::new("Z-Image train resume", error.to_string()))?;
    let mut covered = vec![false; params.len()];
    for factor in &factors {
        let target = factor.target.trim_end_matches(".weight");
        let index = params
            .iter()
            .position(|p| p.module == target)
            .ok_or_else(|| {
                Error::new(
                    "Z-Image train resume",
                    format!("{} is not a selected module", factor.target),
                )
            })?;
        let param = &mut params[index];
        if factor.rank != param.rank as i64
            || factor.in_dim != param.in_dim as i64
            || factor.out_dim != param.out_dim as i64
        {
            return Err(Error::new(
                "Z-Image train resume",
                format!(
                    "{} shape mismatch: file rank {} ({}x{}), expected rank {} ({}x{})",
                    param.module,
                    factor.rank,
                    factor.out_dim,
                    factor.in_dim,
                    param.rank,
                    param.out_dim,
                    param.in_dim
                ),
            ));
        }
        param.a = Array::from_f32_slice(&factor.a, &[param.rank, param.in_dim])?;
        param.b = Array::from_f32_slice(&factor.b, &[param.out_dim, param.rank])?;
        param.scale = factor.scale;
        covered[index] = true;
    }
    let missing: Vec<&str> = params
        .iter()
        .zip(&covered)
        .filter(|(_, hit)| !*hit)
        .map(|(p, _)| p.module.as_str())
        .collect();
    if !missing.is_empty() {
        return Err(Error::new(
            "Z-Image train resume",
            format!("adapter is missing modules: {}", missing.join(", ")),
        ));
    }
    let _ = stream;
    Ok(())
}

fn append_loss_history_line(path: &Path, point: &serde_json::Value) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| Error::new("Z-Image train loss history", e.to_string()))?;
    writeln!(file, "{point}")
        .map_err(|e| Error::new("Z-Image train loss history", e.to_string()))?;
    Ok(())
}

/// Short create with the current adapter → `progress/step-{N}.png`.
/// Companion DistillPatch is not stacked here (verify UI uses the full recipe).
fn write_progress_preview(
    cache: &mut ModelCache,
    job: &TrainJob,
    adapter: &Path,
    step: u32,
    stream: &Stream,
) -> std::result::Result<String, EngineError> {
    use crate::contract::{ExecutionContext, NoopObserver};
    use crate::weights::LoraFile;
    use std::sync::Arc;

    let ids = job
        .caption_ids
        .first()
        .cloned()
        .ok_or_else(|| EngineError::message("progress preview needs at least one caption"))?;
    let steps = job.preset.inference.steps.max(1);
    let seed = job.request.seed.unwrap_or(42).max(0) as u64;
    let width = job.width;
    let height = job.height;
    let published = MlxZImageConfig::published();
    let cfg = super::super::weights::ZImageConfig {
        dim: published.dim,
        heads: published.heads,
        head_dim: published.head_dim(),
        axes: published.axes,
        layers: published.layers,
        refiners: published.refiners,
        in_channels: published.in_channels,
        cap_dim: published.cap_dim,
        ffn: published.ffn_hidden(),
        mid: published.timestep_mid,
        freq: published.frequency_dim,
        adaln: published.adaln_dim,
        rope_theta: published.rope_theta,
        t_scale: published.t_scale,
        vae_scale: published.vae_scale,
        patch: published.patch,
        requires_sigma_shift: true,
    };
    let sigmas = super::super::schedule(
        cfg,
        steps,
        width,
        height,
        Some(job.preset.inference.scheduler.as_str()),
        None,
        false,
    )
    .map_err(|e| EngineError::message(e.to_string()))?;
    let loras = vec![LoraFile {
        id: format!("train-progress-{step}"),
        path: adapter.to_path_buf(),
        weight: job.preset.inference.lora_weight,
    }];
    let exec = ExecutionContext::new(format!("train-progress-{step}"), Arc::new(NoopObserver));
    let latents = session::denoise(
        cache,
        &job.text_key,
        &job.dit_key,
        &job.vae_key,
        None,
        None,
        None,
        None,
        None,
        &ids,
        None,
        0.0,
        width,
        height,
        seed,
        None,
        &sigmas,
        &loras,
        &exec,
        stream,
    )
    .map_err(mlx)?;
    let pixels = session::decode_pixels(cache, &job.vae_key, &latents, width, height, stream)
        .map_err(mlx)?;
    let png = crate::common::png::encode_rgb(width, height, &pixels);
    let name = format!("step-{step:05}.png");
    let out = job.work_dir.join("progress").join(&name);
    std::fs::write(&out, &png).map_err(|e| EngineError::message(e.to_string()))?;
    Ok(out.display().to_string())
}

/// Run the whole training loop on the executor thread.
pub fn train_on_executor(
    job: TrainJob,
    stream: &Stream,
    cache: &mut ModelCache,
    progress: TrainProgress,
) -> std::result::Result<LoraTrainingResult, EngineError> {
    let _ = danmo_mlx::reset_peak_memory();
    let config = MlxZImageConfig::published();
    let dit = cache
        .get::<MlxWeights>(&job.dit_key)
        .map_err(mlx)?
        .ok_or_else(|| EngineError::message("Z-Image train: transformer is not resident"))?
        .clone();
    let text = cache
        .get::<MlxWeights>(&job.text_key)
        .map_err(mlx)?
        .ok_or_else(|| EngineError::message("Z-Image train: text encoder is not resident"))?
        .clone();
    let vae = cache
        .get::<MlxWeights>(&job.vae_key)
        .map_err(mlx)?
        .ok_or_else(|| EngineError::message("Z-Image train: vae is not resident"))?
        .clone();

    progress(TrainEvent::Phase(
        "encoding dataset latents and captions".into(),
    ));
    let samples =
        prepare_samples(&job, &dit, &text, &vae, config, stream, &progress).map_err(mlx)?;

    let (modules, layer_slots) = target_modules(&config, &job.preset).map_err(mlx)?;
    let rank = job.preset.lora_rank as i32;
    let seed = job.request.seed.unwrap_or(42) as u64;
    let init_key = RandomKey::new(seed).map_err(mlx)?;
    let mut params: Vec<LoraTrainParam> = Vec::with_capacity(modules.len());
    for module in &modules {
        params.push(
            init_lora_params(
                &module.name,
                module.out_dim,
                module.in_dim,
                rank,
                rank as f32,
                &init_key,
                stream,
            )
            .map_err(mlx)?,
        );
    }
    let iterations = job.preset.iterations;
    let mut start_step = 0u32;
    if let Some(path) = &job.resume_adapter {
        resume_params(path, &mut params, stream).map_err(mlx)?;
        let state = read_trainer_state(path).map_err(mlx)?;
        start_step = resolve_start_step(state.clone(), iterations, &job.preset.id).map_err(mlx)?;
        match state {
            Some(state) => progress(TrainEvent::Phase(format!(
                "resumed LoRA factors from {} (continuing at step {}/{})",
                path.display(),
                state.step,
                state.iterations
            ))),
            None => progress(TrainEvent::Phase(format!(
                "resumed LoRA factors from {} (no trainer state; warm start from step 0)",
                path.display()
            ))),
        }
    }
    let param_scales: Vec<f32> = params.iter().map(|p| p.scale).collect();

    let shift = if job.preset.id == "turbo-sft" {
        3.0
    } else {
        job.preset.train_sigma_shift
    };
    let schedule = FlowMatchTrainSchedule::z_image(1000, shift);
    let sigmoid_bias = job.preset.sigma_bias == "sigmoid";
    let adam = AdamWConfig {
        learning_rate: job.preset.learning_rate,
        ..AdamWConfig::default()
    };
    let mut states: Vec<AdamWState> = Vec::with_capacity(params.len() * 2);
    for p in &params {
        let mut sa = AdamWState::zeros_like(&p.a, stream).map_err(mlx)?;
        let mut sb = AdamWState::zeros_like(&p.b, stream).map_err(mlx)?;
        // m/v moments restart at zero (not serialized); the step counter is
        // restored so AdamW bias correction continues from the checkpoint.
        sa.step = start_step as u64;
        sb.step = start_step as u64;
        states.push(sa);
        states.push(sb);
    }

    let mut loss_history: Vec<serde_json::Value> = Vec::new();
    let mut checkpoints: Vec<String> = Vec::new();
    let adapter_dir = job.work_dir.join("adapters");
    std::fs::create_dir_all(&adapter_dir).map_err(|e| EngineError::message(e.to_string()))?;
    std::fs::create_dir_all(job.work_dir.join("progress"))
        .map_err(|e| EngineError::message(e.to_string()))?;
    let loss_path = job.work_dir.join("loss_history.jsonl");
    // Fresh run truncates; resume into a new task dir starts empty.
    if start_step == 0 {
        let _ = std::fs::remove_file(&loss_path);
    }
    let mut closure_cache: HashMap<i64, Rc<VariantClosures>> = HashMap::new();
    let mut order_cache: (u32, Vec<usize>) = (u32::MAX, Vec::new());

    progress(TrainEvent::Phase(format!(
        "training: {} iterations, rank {}, {} target modules, {} images",
        iterations,
        rank,
        params.len(),
        samples.len()
    )));

    let n_samples = samples.len() as u32;
    for step in start_step..iterations {
        // Everything below is a pure function of (seed, step): dataset pick,
        // dropout, sigma and noise reproduce exactly after a restart.
        let epoch = step / n_samples;
        if order_cache.0 != epoch {
            order_cache = (epoch, epoch_order(seed, epoch, samples.len()));
        }
        let sample = &samples[order_cache.1[(step % n_samples) as usize]];
        let mut rng = step_rng(seed, step);
        let variant = if rng.unit() < job.preset.caption_dropout {
            &sample.dropped
        } else {
            &sample.captioned
        };
        let (sigma, train_weight) = if sigmoid_bias {
            let z = rng.normal();
            ((1.0 / (1.0 + (-z).exp())).clamp(1e-3, 1.0 - 1e-3), 1.0)
        } else {
            let (s, w, _) = sample_discrete_sigma(&schedule, rng.unit());
            (s, w)
        };
        let shared = variant_closures(
            &mut closure_cache,
            variant,
            &dit,
            config,
            &layer_slots,
            &param_scales,
            params.len() * 2,
            stream,
        )
        .map_err(mlx)?;
        let noise_key = step_noise_key(seed, step).map_err(mlx)?;
        let eps = noise_key
            .normal(&sample.latent.shape(), Dtype::Float32, stream)
            .map_err(mlx)?;
        let x_t = sample_noisy_latent(&sample.latent, sigma, &eps, stream).map_err(mlx)?;
        let timestep = model::embed_timestep(&dit, config, sigma, stream).map_err(mlx)?;
        let (unified, _) = model::unified_before(
            &dit,
            config,
            variant.grid,
            &x_t,
            variant.caption.embedded(),
            &timestep,
            &variant.rope,
            None,
            stream,
            &mut |_, _| {},
        )
        .map_err(mlx)?;
        // No premature eval: unified/timestep/eps/x0 are closure inputs; the
        // traced graph consumes them lazily and one sync at the loss read
        // is enough.

        let weight_arr = Array::scalar_f32(train_weight).map_err(mlx)?;
        let n_train = shared.n_train;
        let mut flat: Vec<Array> = Vec::with_capacity(n_train);
        for p in &params {
            flat.push(p.a.clone());
            flat.push(p.b.clone());
        }
        let mut inputs: Vec<Array> = Vec::with_capacity(n_train + 5);
        inputs.extend(flat.iter().cloned());
        inputs.push(sample.latent.clone());
        inputs.push(timestep);
        inputs.push(unified);
        inputs.push(eps);
        inputs.push(weight_arr);
        let (values, grads) = shared.vag.apply(&inputs).map_err(mlx)?;
        if values.len() != 1 || grads.len() != n_train {
            return Err(EngineError::message(format!(
                "value_and_grad returned {} values / {} grads (expected 1 / {n_train})",
                values.len(),
                grads.len()
            )));
        }
        let loss = values[0].to_vec_f32(stream).map_err(mlx)?[0];
        if !loss.is_finite() {
            return Err(EngineError::message(format!(
                "non-finite loss at step {}: {loss}",
                step + 1
            )));
        }
        adamw_update_all(&mut flat, &grads, &mut states, &adam, stream).map_err(mlx)?;
        for (i, p) in params.iter_mut().enumerate() {
            p.a = flat[i * 2].clone();
            p.b = flat[i * 2 + 1].clone();
        }
        let point = json!({ "step": step + 1, "loss": loss, "sigma": sigma });
        loss_history.push(point.clone());
        append_loss_history_line(&loss_path, &point).map_err(mlx)?;
        progress(TrainEvent::Step {
            step: step + 1,
            total: iterations,
            loss,
        });

        let done = step + 1;
        let checkpoint_hit = done % job.preset.checkpoint_every.max(1) == 0 && done < iterations;
        let progress_hit = done % job.preset.progress_every.max(1) == 0;
        let mut preview_adapter: Option<PathBuf> = None;
        if checkpoint_hit {
            let path = adapter_dir.join(format!("checkpoint-{done}.safetensors"));
            export_lora_adapters(&path, &params, stream)?;
            write_trainer_state(
                &path,
                &TrainerState {
                    step: done,
                    iterations,
                    preset: job.preset.id.clone(),
                },
            )
            .map_err(mlx)?;
            checkpoints.push(path.display().to_string());
            progress(TrainEvent::Checkpoint(path.display().to_string()));
            preview_adapter = Some(path);
        }
        if progress_hit {
            let adapter = match preview_adapter {
                Some(path) => path,
                None => {
                    let path = adapter_dir.join(format!("progress-{done}.safetensors"));
                    export_lora_adapters(&path, &params, stream)?;
                    path
                }
            };
            match write_progress_preview(cache, &job, &adapter, done, stream) {
                Ok(png) => progress(TrainEvent::Phase(format!(
                    "progress preview written to {png}"
                ))),
                Err(error) => progress(TrainEvent::Phase(format!(
                    "progress preview failed (non-fatal): {error}"
                ))),
            }
        }
    }

    let final_path = adapter_dir.join("final_adapters.safetensors");
    export_lora_adapters(&final_path, &params, stream)?;
    write_trainer_state(
        &final_path,
        &TrainerState {
            step: iterations,
            iterations,
            preset: job.preset.id.clone(),
        },
    )
    .map_err(mlx)?;
    checkpoints.push(final_path.display().to_string());
    match write_progress_preview(cache, &job, &final_path, iterations, stream) {
        Ok(png) => progress(TrainEvent::Phase(format!(
            "progress preview written to {png}"
        ))),
        Err(error) => progress(TrainEvent::Phase(format!(
            "progress preview failed (non-fatal): {error}"
        ))),
    }
    let peak_gib = danmo_mlx::peak_memory()
        .map(|bytes| bytes as f64 / (1u64 << 30) as f64)
        .unwrap_or(f64::NAN);
    progress(TrainEvent::Phase(format!(
        "peak MLX memory {peak_gib:.1} GiB"
    )));

    let mid = job
        .request
        .base_model
        .split(':')
        .next()
        .unwrap_or(&job.request.base_model)
        .trim()
        .to_string();
    let config_path = adapter_dir.join("lora_config.json");
    let config_json = json!({
        "base_model": mid,
        "preset": job.preset.id,
        "lora_rank": job.preset.lora_rank,
        "lora_blocks": job.preset.lora_blocks,
        "lora_module_keys": job.preset.lora_module_keys,
        "inference": job.preset.inference,
        "train_sigma_shift": shift,
        "sigma_bias": job.preset.sigma_bias,
        "resolution": job.preset.resolution,
        "iterations": iterations,
    });
    std::fs::write(
        &config_path,
        serde_json::to_string_pretty(&config_json)
            .map_err(|e| EngineError::message(e.to_string()))?,
    )
    .map_err(|e| EngineError::message(e.to_string()))?;

    let output_name = if job.request.output_name.trim().is_empty() {
        format!("{mid}-{}", job.request.dataset_id)
    } else {
        job.request.output_name.clone()
    };
    Ok(LoraTrainingResult {
        adapter_path: final_path.display().to_string(),
        user_lora_id: String::new(),
        output_name,
        loss_history,
        needs_candidate_evaluation: true,
        checkpoints,
        inference: Some(job.preset.inference.clone()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::MlxWeights;
    use danmo_mlx::{adamw_update_all, DeviceType};
    use std::collections::BTreeMap;

    #[test]
    fn trainer_state_roundtrip_and_resume_validation() {
        let dir = std::env::temp_dir().join(format!("trainer-state-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ck = dir.join("checkpoint-500.safetensors");
        std::fs::write(&ck, b"stub").unwrap();

        // Pre-resume checkpoint (no state file) → warm start at step 0.
        assert_eq!(
            resolve_start_step(read_trainer_state(&ck).unwrap(), 2500, "turbo-sft").unwrap(),
            0
        );

        let state = TrainerState {
            step: 500,
            iterations: 2500,
            preset: "turbo-sft".into(),
        };
        write_trainer_state(&ck, &state).unwrap();
        assert_eq!(
            trainer_state_path(&ck).file_name().unwrap(),
            "checkpoint-500.state.json"
        );
        assert_eq!(read_trainer_state(&ck).unwrap(), Some(state.clone()));
        assert_eq!(
            resolve_start_step(read_trainer_state(&ck).unwrap(), 2500, "turbo-sft").unwrap(),
            500
        );

        // Fail loud: iteration count changed, preset changed, run complete,
        // corrupt state file.
        assert!(resolve_start_step(read_trainer_state(&ck).unwrap(), 1000, "turbo-sft").is_err());
        assert!(resolve_start_step(read_trainer_state(&ck).unwrap(), 2500, "base-face").is_err());
        write_trainer_state(
            &ck,
            &TrainerState {
                step: 2500,
                ..state
            },
        )
        .unwrap();
        assert!(resolve_start_step(read_trainer_state(&ck).unwrap(), 2500, "turbo-sft").is_err());
        std::fs::write(trainer_state_path(&ck), b"{not json").unwrap();
        assert!(read_trainer_state(&ck).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn step_deterministic_sampling_reproduces_after_restart() {
        // Same (seed, step) ⇒ same draws; same (seed, epoch) ⇒ same order.
        assert_eq!(epoch_order(42, 0, 9), epoch_order(42, 0, 9));
        assert_ne!(epoch_order(42, 0, 9), epoch_order(42, 1, 9));
        let mut sorted = epoch_order(42, 0, 9);
        sorted.sort_unstable();
        assert_eq!(sorted, (0..9).collect::<Vec<_>>(), "order is a permutation");

        assert_eq!(step_rng(7, 123).unit(), step_rng(7, 123).unit());
        assert_eq!(step_rng(7, 123).normal(), step_rng(7, 123).normal());
        assert_ne!(step_rng(7, 123).unit(), step_rng(7, 124).unit());

        let k1 = step_noise_key(7, 55).unwrap();
        let k2 = step_noise_key(7, 55).unwrap();
        let stream = Stream::cpu().unwrap();
        let n1 = k1.normal(&[4], Dtype::Float32, &stream).unwrap();
        let n2 = k2.normal(&[4], Dtype::Float32, &stream).unwrap();
        assert_eq!(
            n1.to_vec_f32(&stream).unwrap(),
            n2.to_vec_f32(&stream).unwrap()
        );
    }

    fn tiny_config() -> MlxZImageConfig {
        let mut config = MlxZImageConfig::published();
        config.in_channels = 2;
        config.dim = 6;
        config.heads = 1;
        config.layers = 1;
        config.refiners = 1;
        config.cap_dim = 4;
        config.adaln_dim = 6;
        config.axes = [2, 2, 2];
        config
    }

    fn add_tensor(tensors: &mut BTreeMap<String, (Vec<f32>, Vec<i32>)>, name: &str, shape: &[i64]) {
        let count = shape.iter().product::<i64>() as usize;
        let values = (0..count)
            .map(|index| ((index + name.len()) % 13) as f32 * 0.015 - 0.07)
            .collect();
        tensors.insert(
            name.to_string(),
            (values, shape.iter().map(|value| *value as i32).collect()),
        );
    }

    fn add_block(
        tensors: &mut BTreeMap<String, (Vec<f32>, Vec<i32>)>,
        prefix: &str,
        config: &MlxZImageConfig,
        modulated: bool,
    ) {
        let dim = config.dim;
        let ffn = config.ffn_hidden();
        for name in ["to_q", "to_k", "to_v", "to_out"] {
            add_tensor(
                tensors,
                &format!("{prefix}.attention.{name}.weight"),
                &[dim, dim],
            );
        }
        add_tensor(
            tensors,
            &format!("{prefix}.attention.norm_q.weight"),
            &[config.head_dim()],
        );
        add_tensor(
            tensors,
            &format!("{prefix}.attention.norm_k.weight"),
            &[config.head_dim()],
        );
        for name in ["attn_norm1", "attn_norm2", "ffn_norm1", "ffn_norm2"] {
            add_tensor(tensors, &format!("{prefix}.{name}.weight"), &[dim]);
        }
        add_tensor(
            tensors,
            &format!("{prefix}.feed_forward.w1.weight"),
            &[ffn, dim],
        );
        add_tensor(
            tensors,
            &format!("{prefix}.feed_forward.w2.weight"),
            &[dim, ffn],
        );
        add_tensor(
            tensors,
            &format!("{prefix}.feed_forward.w3.weight"),
            &[ffn, dim],
        );
        if modulated {
            add_tensor(
                tensors,
                &format!("{prefix}.adaLN_modulation.0.weight"),
                &[4 * dim, config.adaln_dim],
            );
            add_tensor(
                tensors,
                &format!("{prefix}.adaLN_modulation.0.bias"),
                &[4 * dim],
            );
        }
    }

    fn tiny_weights(config: &MlxZImageConfig) -> MlxWeights {
        let token = config.patch * config.patch * config.in_channels;
        let mut tensors = BTreeMap::new();
        add_tensor(&mut tensors, "x_embedder.weight", &[config.dim, token]);
        add_tensor(&mut tensors, "x_embedder.bias", &[config.dim]);
        add_tensor(
            &mut tensors,
            "t_embedder.linear1.weight",
            &[config.timestep_mid, config.frequency_dim],
        );
        add_tensor(
            &mut tensors,
            "t_embedder.linear1.bias",
            &[config.timestep_mid],
        );
        add_tensor(
            &mut tensors,
            "t_embedder.linear2.weight",
            &[config.adaln_dim, config.timestep_mid],
        );
        add_tensor(&mut tensors, "t_embedder.linear2.bias", &[config.adaln_dim]);
        add_tensor(&mut tensors, "cap_norm.weight", &[config.cap_dim]);
        add_tensor(
            &mut tensors,
            "cap_embedder.weight",
            &[config.dim, config.cap_dim],
        );
        add_tensor(&mut tensors, "cap_embedder.bias", &[config.dim]);
        add_block(&mut tensors, "noise_refiner.0", config, true);
        add_block(&mut tensors, "context_refiner.0", config, false);
        add_block(&mut tensors, "layers.0", config, true);
        add_tensor(
            &mut tensors,
            "final_layer.adaLN_modulation.0.weight",
            &[config.dim, config.adaln_dim],
        );
        add_tensor(
            &mut tensors,
            "final_layer.adaLN_modulation.0.bias",
            &[config.dim],
        );
        add_tensor(
            &mut tensors,
            "final_layer.linear.weight",
            &[token, config.dim],
        );
        add_tensor(&mut tensors, "final_layer.linear.bias", &[token]);
        tensors.insert(
            "x_pad_token".into(),
            (vec![0.0; config.dim as usize], vec![1, config.dim as i32]),
        );
        tensors.insert(
            "cap_pad_token".into(),
            (vec![0.0; config.dim as usize], vec![1, config.dim as i32]),
        );
        let mut arrays = BTreeMap::new();
        for (name, (values, shape)) in tensors {
            arrays.insert(name, Array::from_f32_slice(&values, &shape).unwrap());
        }
        MlxWeights::from_dense(arrays)
    }

    fn probe(device: DeviceType) {
        let config = tiny_config();
        let stream = match device {
            DeviceType::Cpu => Stream::cpu().unwrap(),
            DeviceType::Gpu => Stream::gpu().unwrap(),
        };
        let weights = tiny_weights(&config);
        let grid = geom::token_grid(16, 16, 2, config.vae_scale).unwrap();
        let caption =
            Array::from_f32_slice(&[0.1, -0.2, 0.05, 0.3, -0.15, 0.25, 0.2, -0.1], &[2, 4])
                .unwrap();
        let caption_cache = CaptionCache::new(&weights, config, grid, &caption, &stream).unwrap();
        let rope = ForwardRope::new(config, grid, &stream).unwrap();
        let x0 = Array::from_f32_slice(
            &[0.2, -0.1, 0.4, 0.05, -0.3, 0.15, 0.0, 0.25],
            &[1, 2, 2, 2],
        )
        .unwrap();

        let init_key = RandomKey::new(7).unwrap();
        let module_names = [
            "layers.0.attention.to_q",
            "layers.0.attention.to_k",
            "layers.0.attention.to_v",
            "layers.0.attention.to_out",
            "layers.0.feed_forward.w1",
            "layers.0.feed_forward.w2",
            "layers.0.feed_forward.w3",
        ];
        let ffn = config.ffn_hidden() as i32;
        let dim = config.dim as i32;
        let mut params: Vec<LoraTrainParam> = module_names
            .iter()
            .enumerate()
            .map(|(index, name)| {
                let (out_dim, in_dim) = match index {
                    0..=3 => (dim, dim),
                    4 | 6 => (ffn, dim),
                    _ => (dim, ffn),
                };
                init_lora_params(name, out_dim, in_dim, 2, 2.0, &init_key, &stream).unwrap()
            })
            .collect();
        let slots: Vec<[Option<usize>; 7]> = vec![[
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(5),
            Some(6),
        ]];
        let scales: Vec<f32> = params.iter().map(|p| p.scale).collect();
        let ids = geom::positions_image_first(grid);
        let block = std::rc::Rc::new(
            checkpointed_block(
                &weights,
                config,
                0,
                slots[0],
                &scales,
                ids,
                rope.unified.clone(),
                &stream,
            )
            .unwrap(),
        );

        let adam = AdamWConfig {
            learning_rate: 5e-2,
            ..AdamWConfig::default()
        };
        let mut states: Vec<AdamWState> = Vec::new();
        for p in &params {
            states.push(AdamWState::zeros_like(&p.a, &stream).unwrap());
            states.push(AdamWState::zeros_like(&p.b, &stream).unwrap());
        }
        // Fixed noise and sigma: a deterministic single-target optimization
        // (fresh noise per step would make the loss noisy by construction).
        let eps = RandomKey::new(11)
            .unwrap()
            .normal(&x0.shape(), Dtype::Float32, &stream)
            .unwrap();
        let mut losses = Vec::new();
        for _step in 0..12 {
            let eps = eps.clone();
            let x_t = sample_noisy_latent(&x0, 0.5, &eps, &stream).unwrap();
            let timestep = model::embed_timestep(&weights, config, 0.5, &stream).unwrap();
            let (unified, _) = model::unified_before(
                &weights,
                config,
                grid,
                &x_t,
                caption_cache.embedded(),
                &timestep,
                &rope,
                None,
                &stream,
                &mut |_, _| {},
            )
            .unwrap();
            let weights_inner = weights.clone();
            let x0_inner = x0.clone();
            let block_inner = block.clone();
            let closure = Closure::new(stream.clone(), move |ins, call_stream| {
                let mut inputs = vec![unified.clone(), timestep.clone()];
                for array in ins {
                    inputs.push(array.clone());
                }
                let x = block_inner.apply(&inputs)?;
                let velocity = model::final_velocity(
                    &weights_inner,
                    config,
                    grid,
                    &x,
                    &timestep,
                    call_stream,
                    &mut |_, _| {},
                )?;
                flow_match_mse(&velocity, &x0_inner, &eps, call_stream)
            })
            .unwrap();
            let argnums: Vec<i32> = (0..14).collect();
            let vag = ValueAndGrad::new(closure, &argnums).unwrap();
            let mut flat: Vec<Array> = Vec::new();
            for p in &params {
                flat.push(p.a.clone());
                flat.push(p.b.clone());
            }
            let (values, grads) = vag.apply(&flat).unwrap();
            assert_eq!(grads.len(), 14, "one gradient per LoRA factor");
            let loss = values[0].to_vec_f32(&stream).unwrap()[0];
            assert!(loss.is_finite(), "non-finite loss at step {_step}");
            for (index, grad) in grads.iter().enumerate() {
                let values = grad.to_vec_f32(&stream).unwrap();
                assert!(
                    values.iter().all(|value| value.is_finite()),
                    "non-finite gradient on factor {index} at step {_step}"
                );
            }
            // B factors start at zero, so their gradient is nonzero from the
            // first step; A factors pick up gradient once B moves.
            let b_grad = grads[1].to_vec_f32(&stream).unwrap();
            assert!(
                b_grad.iter().any(|value| value.abs() > 1e-8),
                "B factor gradient vanished at step {_step}"
            );
            adamw_update_all(&mut flat, &grads, &mut states, &adam, &stream).unwrap();
            for (i, p) in params.iter_mut().enumerate() {
                p.a = flat[i * 2].clone();
                p.b = flat[i * 2 + 1].clone();
            }
            losses.push(loss);
        }
        let first = losses[0];
        let last = *losses.last().unwrap();
        assert!(
            last < first * 0.9,
            "loss did not drop over 12 steps: {first} → {last} ({losses:?})"
        );
    }

    #[test]
    fn cpu_tiny_dit_lora_training_lowers_loss() {
        probe(DeviceType::Cpu);
    }

    #[cfg(feature = "mlx-metal")]
    #[test]
    fn metal_tiny_dit_lora_training_lowers_loss() {
        probe(DeviceType::Gpu);
    }
}
