//! 5Hz LM inspire / format / generate on MLX. Phase 1 rewrites caption,
//! metadata and lyrics under the optional FSM; generate adds planner codes.

use std::path::{Path, PathBuf};

use danmo_mlx::{CacheKey, CacheRevision, ModelCache, Stream};

use super::super::lm_codes::mask_planner_code_step;
use super::super::lm_expand::{LmExpandOutput, LmExpandRequest};
use super::super::lm_format::{build_lm_format_result, parse_lm_output};
use super::super::lm_fsm::{MetadataFsmSession, UserMetas, DURATION_MAX, DURATION_MIN};
use super::super::lm_prompt::{
    build_lm_prompt_uncond_with_cot, build_lm_prompt_with_cot, build_phase1_prompt, AcePrompt,
    LmMode, AUDIO_CODE_BASE, TOKEN_IM_END, TOKEN_THINK_END,
};
use super::super::lm_sample::{sample_top_k_p, XorShift64};
use super::super::lyrics::is_instrumental_lyrics;
use super::super::token_decode::Decoder;
use super::super::{LmOutput, LmRequest};
use crate::common::qwen3::mlx::{self, LmCache};
use crate::common::qwen3::{self, Qwen3Config};
use crate::weights::{MlxWeights, WeightDtype, WeightError};

const TEMPERATURE: f32 = 0.85;
const TOP_P: f32 = 0.9;
const CODES_CFG: f32 = 2.0;

pub(crate) fn expand(
    bundle: &Path,
    request: &LmExpandRequest<'_>,
    mut on_token: impl FnMut(usize, usize),
) -> Result<LmExpandOutput, WeightError> {
    let mut out = expand_limited(bundle, request, &mut on_token, None)?;
    if out.audio_code_indices.is_empty() {
        out.audio_code_indices = planner_codes(
            bundle,
            request,
            &out,
            PlannerOptions::default(),
            &mut on_token,
        )?;
    }
    Ok(out)
}

#[derive(Clone, Copy)]
struct Phase1Sampling {
    max_new_tokens: Option<usize>,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    use_fsm: bool,
}

impl Default for Phase1Sampling {
    fn default() -> Self {
        Self {
            max_new_tokens: None,
            temperature: TEMPERATURE,
            top_p: TOP_P,
            top_k: 0,
            use_fsm: true,
        }
    }
}

struct Phase1Output {
    expanded: LmExpandOutput,
    generated_ids: Vec<i32>,
    text: String,
    prompt_tokens: usize,
}

fn expand_limited(
    bundle: &Path,
    request: &LmExpandRequest<'_>,
    on_token: &mut impl FnMut(usize, usize),
    limit_override: Option<usize>,
) -> Result<LmExpandOutput, WeightError> {
    let sampling = Phase1Sampling {
        max_new_tokens: limit_override,
        ..Phase1Sampling::default()
    };
    phase1(bundle, request, LmMode::Format, sampling, on_token).map(|output| output.expanded)
}

fn phase1(
    bundle: &Path,
    request: &LmExpandRequest<'_>,
    mode: LmMode,
    sampling: Phase1Sampling,
    on_token: &mut impl FnMut(usize, usize),
) -> Result<Phase1Output, WeightError> {
    let directory = find_lm_dir(bundle, request.lm_bundle)?;
    let config = read_config(&directory)?;
    let tokenizer = std::fs::read_to_string(directory.join("tokenizer.json")).map_err(|error| {
        WeightError::Open {
            path: directory.join("tokenizer.json").display().to_string(),
            detail: error.to_string(),
        }
    })?;
    let lyrics = if request.lyrics.trim().is_empty() {
        "[Instrumental]".to_string()
    } else {
        request.lyrics.trim().to_string()
    };
    let caption = if request.caption.trim().is_empty() {
        "NO USER INPUT".to_string()
    } else {
        request.caption.trim().to_string()
    };
    let has_vocals = !is_instrumental_lyrics(&lyrics);
    let metadata_only = mode.skips_audio_codes() && has_vocals;
    let prompt = AcePrompt {
        caption: caption.clone(),
        lyrics: lyrics.clone(),
        duration: request.duration,
        bpm: request.bpm,
        keyscale: request.keyscale.to_string(),
        timesignature: request.timesignature.to_string(),
        vocal_language: request.vocal_language.to_string(),
    };
    let mut encode = |text: &str| qwen3::tokenize_plain(&tokenizer, text, config.vocab);
    let prefix = build_phase1_prompt(&mut encode, mode, &prompt)?;
    let prompt_tokens = prefix.len();
    let decoder = Decoder::from_tokenizer_json(&tokenizer)?;
    let mut fsm = if sampling.use_fsm {
        Some(
            MetadataFsmSession::open(&tokenizer, config.vocab)?.with_user_metas(UserMetas {
                bpm: (request.bpm > 0).then_some(request.bpm),
                caption: None,
                duration: (request.duration > 0.0).then_some(request.duration.round() as i32),
                keyscale: nonempty(request.keyscale),
                language: nonempty(request.vocal_language),
                timesignature: nonempty(request.timesignature),
            }),
        )
    } else {
        None
    };
    let limit = sampling
        .max_new_tokens
        .unwrap_or_else(|| max_new_tokens(request.duration, metadata_only));
    let seed = request.seed;
    let key = cache_key(&directory)?;
    let (generated, text) = crate::runtime::mlx_executor()
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?
        .execute("ace_step 5hz lm", move |_, stream, cache, _| {
            let weights = resident(cache, key, &directory, stream)?;
            let generated = sample(
                &weights,
                config,
                &prefix,
                &decoder,
                fsm.as_mut(),
                limit,
                metadata_only,
                seed,
                sampling.temperature,
                sampling.top_p,
                sampling.top_k,
                stream,
            )?;
            let text = decoder.decode(&generated);
            Ok((generated, text))
        })
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
    #[cfg(test)]
    if sampling.max_new_tokens.is_some() && mode == LmMode::Format {
        assert!(
            text.starts_with("<think>"),
            "5Hz LM format did not open the think block: {text}"
        );
    }
    on_token(generated.len(), limit.max(1));
    let meta = parse_lm_output(&text);
    let formatted = build_lm_format_result(
        &meta,
        &caption,
        &lyrics,
        (request.duration > 0.0).then_some(request.duration),
        (request.bpm > 0).then_some(request.bpm),
        request.keyscale,
        request.timesignature,
        request.vocal_language,
        has_vocals,
        mode != LmMode::Inspire && has_vocals,
    );
    Ok(Phase1Output {
        expanded: LmExpandOutput {
            caption: formatted.caption,
            lyrics: formatted.lyrics,
            bpm: formatted.bpm,
            duration: formatted.duration,
            keyscale: formatted.keyscale,
            timesignature: formatted.timesignature,
            language: formatted.language,
            audio_code_indices: if !formatted.audio_code_indices.is_empty() {
                formatted.audio_code_indices
            } else {
                request.existing_codes.unwrap_or(&[]).to_vec()
            },
            did_expand: true,
            pmi: None,
        },
        generated_ids: generated,
        text,
        prompt_tokens,
    })
}

pub(crate) fn run(
    bundle: &Path,
    request: &LmRequest<'_>,
    on_token: &mut impl FnMut(usize, usize),
) -> Result<LmOutput, WeightError> {
    let expanded_request = LmExpandRequest {
        mode: request.mode.name(),
        caption: request.caption,
        lyrics: request.lyrics,
        duration: request.duration,
        bpm: request.bpm,
        keyscale: request.keyscale,
        timesignature: request.timesignature,
        vocal_language: request.vocal_language,
        seed: request.seed,
        lm_bundle: None,
        existing_codes: None,
    };
    let phase = phase1(
        bundle,
        &expanded_request,
        request.mode,
        Phase1Sampling {
            max_new_tokens: Some(request.max_new_tokens.max(1)),
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            use_fsm: request.use_fsm,
        },
        on_token,
    )?;
    let mut expanded = phase.expanded;
    if request.mode == LmMode::Generate && expanded.audio_code_indices.is_empty() {
        expanded.audio_code_indices = planner_codes(
            bundle,
            &expanded_request,
            &expanded,
            PlannerOptions {
                cfg_scale: request.cfg_scale,
                temperature: request.temperature,
                top_p: request.top_p,
                top_k: request.top_k,
                negative_prompt: if request.negative_prompt.trim().is_empty() {
                    "NO USER INPUT"
                } else {
                    request.negative_prompt
                },
            },
            on_token,
        )?;
    }
    Ok(LmOutput {
        mode: request.mode.name().into(),
        prompt_tokens: phase.prompt_tokens,
        generated_ids: phase.generated_ids,
        text: phase.text,
        caption: expanded.caption,
        lyrics: expanded.lyrics,
        bpm: expanded.bpm,
        duration: expanded.duration,
        keyscale: expanded.keyscale,
        timesignature: expanded.timesignature,
        language: expanded.language,
        audio_code_indices: expanded.audio_code_indices,
        pmi: expanded.pmi,
    })
}

struct CodePass {
    cond: Option<LmCache>,
    uncond: Option<LmCache>,
}

#[derive(Clone, Copy)]
struct PlannerOptions<'a> {
    cfg_scale: f32,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    negative_prompt: &'a str,
}

impl Default for PlannerOptions<'static> {
    fn default() -> Self {
        Self {
            cfg_scale: CODES_CFG,
            temperature: TEMPERATURE,
            top_p: TOP_P,
            top_k: 0,
            negative_prompt: "NO USER INPUT",
        }
    }
}

/// Phase 2. Dual caches, one code per 0.2 s, stopped at the duration target.
fn planner_codes(
    bundle: &Path,
    request: &LmExpandRequest<'_>,
    formatted: &LmExpandOutput,
    options: PlannerOptions<'_>,
    on_token: &mut impl FnMut(usize, usize),
) -> Result<Vec<i32>, WeightError> {
    let directory = find_lm_dir(bundle, request.lm_bundle)?;
    let config = read_config(&directory)?;
    let tokenizer = std::fs::read_to_string(directory.join("tokenizer.json")).map_err(|error| {
        WeightError::Open {
            path: directory.join("tokenizer.json").display().to_string(),
            detail: error.to_string(),
        }
    })?;
    let duration = formatted
        .duration
        .map(|value| value as f32)
        .filter(|value| *value > 0.0)
        .unwrap_or(request.duration);
    let target = target_codes(duration)?;
    let mut prompt = AcePrompt {
        caption: formatted.caption.clone(),
        lyrics: formatted.lyrics.clone(),
        duration,
        bpm: formatted.bpm.unwrap_or(0),
        keyscale: formatted.keyscale.clone(),
        timesignature: formatted
            .timesignature
            .strip_suffix("/4")
            .unwrap_or(&formatted.timesignature)
            .to_string(),
        vocal_language: formatted.language.clone(),
    };
    let cot = super::super::lm_prompt::build_cot_yaml(&prompt);
    let mut encode = |text: &str| qwen3::tokenize_plain(&tokenizer, text, config.vocab);
    let cond_ids = build_lm_prompt_with_cot(&mut encode, &prompt, &cot)?;
    prompt.caption.clear();
    let uncond_ids =
        build_lm_prompt_uncond_with_cot(&mut encode, &prompt, options.negative_prompt)?;
    let weight_key = cache_key(&directory)?;
    let slot = CacheKey {
        id: format!("{}#planner-codes", directory.display()),
        revision: CacheRevision {
            length: weight_key.revision.length,
            modified_ns: weight_key.revision.modified_ns,
            mix: request.seed,
        },
    };
    let mut release = CodeSlot {
        key: slot.clone(),
        live: false,
    };
    let (mut cond_logits, mut uncond_logits) = crate::runtime::mlx_executor()
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?
        .execute("ace_step 5hz codes prefill", {
            let directory = directory.clone();
            let weight_key = weight_key.clone();
            let slot = slot.clone();
            let cond_ids = cond_ids.clone();
            let uncond_ids = uncond_ids.clone();
            move |_, stream, cache, _| {
                let weights = resident(cache, weight_key, &directory, stream)?;
                let mut pass = CodePass {
                    cond: None,
                    uncond: None,
                };
                let cond = mlx::step_logits(
                    &weights,
                    config,
                    "model.",
                    &cond_ids,
                    &mut pass.cond,
                    stream,
                )?;
                let uncond = mlx::step_logits(
                    &weights,
                    config,
                    "model.",
                    &uncond_ids,
                    &mut pass.uncond,
                    stream,
                )?;
                cache
                    .insert(slot, pass, 1)
                    .map_err(|error| danmo_mlx::Error::new("ACE-Step 5Hz LM", error.to_string()))?;
                Ok((cond, uncond))
            }
        })
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
    release.live = true;
    let mut rng = XorShift64::new(request.seed.wrapping_add(0xC0DE));
    let mut codes = Vec::with_capacity(target);
    while codes.len() < target {
        on_token(codes.len(), target);
        let mut combined = cfg_logits(&cond_logits, &uncond_logits, options.cfg_scale)?;
        mask_planner_code_step(&mut combined, codes.len(), target)?;
        let id = sample_top_k_p(
            &combined,
            options.temperature,
            options.top_p,
            options.top_k,
            &mut rng,
        )?;
        if id == TOKEN_IM_END {
            break;
        }
        let index = id - AUDIO_CODE_BASE;
        if !(0..super::super::fsq_quant::FSQ_CODEBOOK_SIZE).contains(&index) {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step planner sampled code {index}, outside the FSQ book"
            )));
        }
        codes.push(index);
        let step_id = id;
        let next = crate::runtime::mlx_executor()
            .map_err(|error| WeightError::NotBuilt(error.to_string()))?
            .execute("ace_step 5hz codes step", {
                let directory = directory.clone();
                let weight_key = weight_key.clone();
                let slot = slot.clone();
                move |_, stream, cache, _| {
                    let weights = resident(cache, weight_key, &directory, stream)?;
                    let pass = cache.get_mut::<CodePass>(&slot)?.ok_or_else(|| {
                        danmo_mlx::Error::new("ACE-Step 5Hz LM", "planner cache is missing")
                    })?;
                    let cond = mlx::step_logits(
                        &weights,
                        config,
                        "model.",
                        &[step_id],
                        &mut pass.cond,
                        stream,
                    )?;
                    let uncond = mlx::step_logits(
                        &weights,
                        config,
                        "model.",
                        &[step_id],
                        &mut pass.uncond,
                        stream,
                    )?;
                    Ok((cond, uncond))
                }
            })
            .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
        cond_logits = next.0;
        uncond_logits = next.1;
    }
    on_token(codes.len(), target);
    drop(release);
    if codes.len() != target {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step planner codes stopped at {} of {target}",
            codes.len()
        )));
    }
    Ok(codes)
}

struct CodeSlot {
    key: CacheKey,
    live: bool,
}

impl Drop for CodeSlot {
    fn drop(&mut self) {
        if !self.live {
            return;
        }
        self.live = false;
        let key = self.key.clone();
        let _ = crate::runtime::mlx_executor().and_then(|executor| {
            executor.execute("ace_step 5hz codes release", move |_, _, cache, _| {
                cache.insert(
                    key,
                    CodePass {
                        cond: None,
                        uncond: None,
                    },
                    1,
                )?;
                Ok(())
            })
        });
    }
}

fn target_codes(duration: f32) -> Result<usize, WeightError> {
    if !duration.is_finite() || duration <= 0.0 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step planner codes need a positive duration, got {duration}"
        )));
    }
    let effective = duration.clamp(DURATION_MIN as f32, DURATION_MAX as f32);
    let target = (effective * 5.0) as usize;
    if target < 1 {
        return Err(WeightError::NotBuilt(
            "ACE-Step planner code target is zero".into(),
        ));
    }
    Ok(target)
}

fn cfg_logits(cond: &[f32], uncond: &[f32], scale: f32) -> Result<Vec<f32>, WeightError> {
    if cond.len() != uncond.len() || cond.is_empty() {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step planner CFG logits are {} and {}",
            cond.len(),
            uncond.len()
        )));
    }
    Ok(cond
        .iter()
        .zip(uncond)
        .map(|(cond, uncond)| uncond + scale * (cond - uncond))
        .collect())
}

fn nonempty(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn max_new_tokens(duration: f32, metadata_only: bool) -> usize {
    let base = if duration > 0.0 {
        let effective = duration.clamp(DURATION_MIN as f32, DURATION_MAX as f32);
        (effective * 5.0) as usize + 500
    } else {
        1024
    };
    let cap = if metadata_only { 512 } else { 1200 };
    base.min(cap).min(4096 - 64)
}

fn sample(
    weights: &MlxWeights,
    config: Qwen3Config,
    prefix: &[i32],
    decoder: &Decoder,
    mut fsm: Option<&mut MetadataFsmSession>,
    limit: usize,
    metadata_only: bool,
    seed: u64,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    stream: &Stream,
) -> Result<Vec<i32>, danmo_mlx::Error> {
    let mut cache: Option<LmCache> = None;
    let mut logits = mlx::step_logits(weights, config, "model.", prefix, &mut cache, stream)?;
    let mut rng = XorShift64::new(seed);
    let mut generated = Vec::new();
    for _ in 0..limit {
        let piece = |id: i32| decoder.piece(id);
        if let Some(session) = fsm.as_deref_mut() {
            session
                .mask(&mut logits, &piece)
                .map_err(|error| danmo_mlx::Error::new("ACE-Step 5Hz LM", error.to_string()))?;
        }
        let id = sample_top_k_p(&logits, temperature, top_p, top_k, &mut rng)
            .map_err(|error| danmo_mlx::Error::new("ACE-Step 5Hz LM", error.to_string()))?;
        let text = decoder.piece(id);
        if let Some(session) = fsm.as_deref_mut() {
            session
                .observe(id, &text)
                .map_err(|error| danmo_mlx::Error::new("ACE-Step 5Hz LM", error.to_string()))?;
        }
        generated.push(id);
        if id == TOKEN_IM_END || (metadata_only && id == TOKEN_THINK_END) {
            break;
        }
        logits = mlx::step_logits(weights, config, "model.", &[id], &mut cache, stream)?;
    }
    Ok(generated)
}

fn find_lm_dir(bundle: &Path, override_dir: Option<&Path>) -> Result<PathBuf, WeightError> {
    if let Some(path) = override_dir.filter(|path| path.is_dir()) {
        if path.join("model.safetensors").is_file()
            || path
                .read_dir()
                .ok()
                .into_iter()
                .flatten()
                .flatten()
                .any(|entry| {
                    entry.path().extension().and_then(|ext| ext.to_str()) == Some("safetensors")
                })
        {
            return Ok(path.to_path_buf());
        }
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step 5Hz LM override {} has no safetensors",
            path.display()
        )));
    }
    for name in [
        "acestep-5Hz-lm-1.7B",
        "acestep-5Hz-lm-0.6B",
        "acestep-5Hz-lm-4B",
    ] {
        for root in [bundle, bundle.parent().unwrap_or(bundle)] {
            let path = root.join(name);
            if path.join("model.safetensors").is_file() {
                return Ok(path);
            }
        }
    }
    Err(WeightError::NotBuilt(format!(
        "ACE-Step 5Hz LM safetensors were not found under {} (set lm_expansion to off)",
        bundle.display()
    )))
}

fn read_config(directory: &Path) -> Result<Qwen3Config, WeightError> {
    let path = directory.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|error| WeightError::Open {
        path: path.display().to_string(),
        detail: error.to_string(),
    })?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|error| {
        WeightError::NotBuilt(format!("5Hz LM config.json is not json: {error}"))
    })?;
    let number = |key: &str| {
        value
            .get(key)
            .and_then(|item| item.as_i64())
            .ok_or_else(|| WeightError::NotBuilt(format!("5Hz LM config.json is missing {key}")))
    };
    let hidden = number("hidden_size")?;
    let heads = number("num_attention_heads")?;
    Ok(Qwen3Config {
        vocab: number("vocab_size")?,
        hidden,
        layers: number("num_hidden_layers")?,
        heads,
        kv_heads: number("num_key_value_heads")?,
        head_dim: value
            .get("head_dim")
            .and_then(|item| item.as_i64())
            .unwrap_or(hidden / heads),
        intermediate: number("intermediate_size")?,
        rope_theta: value
            .get("rope_theta")
            .and_then(|item| item.as_f64())
            .unwrap_or(1_000_000.0) as f32,
    })
}

fn resident(
    cache: &mut ModelCache,
    key: CacheKey,
    directory: &Path,
    stream: &Stream,
) -> Result<MlxWeights, danmo_mlx::Error> {
    if cache.get::<MlxWeights>(&key)?.is_none() {
        let weights = qwen3::mlx::load_directory(directory, WeightDtype::Native)?;
        let bytes = weights.nbytes();
        cache.insert(key.clone(), weights, bytes)?;
    }
    let _ = stream;
    cache.get::<MlxWeights>(&key)?.cloned().ok_or_else(|| {
        danmo_mlx::Error::new("ACE-Step 5Hz LM", "resident weights missing after insert")
    })
}

fn cache_key(directory: &Path) -> Result<CacheKey, WeightError> {
    let path = directory.join("model.safetensors");
    let meta = std::fs::metadata(&path).map_err(|error| WeightError::Open {
        path: path.display().to_string(),
        detail: error.to_string(),
    })?;
    let modified_ns = meta
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|time| time.as_nanos() as u64)
        .unwrap_or(0);
    Ok(CacheKey {
        id: directory.display().to_string(),
        revision: CacheRevision {
            length: meta.len(),
            modified_ns,
            mix: 1,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::lm_expand::LmExpandRequest;
    use super::expand_limited;
    use std::path::PathBuf;

    fn installed_bundle() -> Option<PathBuf> {
        let root = std::env::var("DANQING_WORKSPACE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/Users/nil.luo/Workspace/studio-workspace"));
        let bundle = root.join("models/Audio/acestep-v15-xl-sft");
        if bundle
            .join("acestep-5Hz-lm-1.7B/model.safetensors")
            .is_file()
        {
            Some(bundle)
        } else {
            None
        }
    }

    #[test]
    fn format_prefill_starts_with_think() {
        let Some(bundle) = installed_bundle() else {
            eprintln!("skip: 5Hz LM safetensors missing");
            return;
        };
        let out = expand_limited(
            &bundle,
            &LmExpandRequest {
                mode: "format",
                caption: "安静的钢琴",
                lyrics: "你好",
                duration: 10.0,
                bpm: 90,
                keyscale: "C major",
                timesignature: "4",
                vocal_language: "zh",
                seed: 7,
                lm_bundle: None,
                existing_codes: None,
            },
            &mut |_, _| (),
            Some(32),
        )
        .expect("5hz format");
        assert!(out.did_expand);
        assert!(
            out.caption.to_ascii_lowercase().contains("piano"),
            "caption={}",
            out.caption
        );
        println!("lm caption={} lyrics={}", out.caption, out.lyrics);
    }

    #[test]
    fn direct_inspire_uses_the_mlx_forward() {
        let Some(bundle) = installed_bundle() else {
            eprintln!("skip: 5Hz LM safetensors missing");
            return;
        };
        let out = super::run(
            &bundle,
            &super::super::super::LmRequest {
                mode: super::super::super::LmMode::Inspire,
                caption: "quiet piano",
                lyrics: "[Instrumental]",
                duration: 10.0,
                bpm: 90,
                keyscale: "C major",
                timesignature: "4",
                vocal_language: "en",
                max_new_tokens: 32,
                tokenizer_json: None,
                temperature: 0.85,
                top_p: 0.9,
                top_k: 0,
                seed: 7,
                use_fsm: true,
                cfg_scale: 2.0,
                negative_prompt: "",
            },
            &mut |_, _| (),
        )
        .expect("direct inspire");
        assert_eq!(out.mode, "inspire");
        assert!(out.prompt_tokens > 0 && !out.generated_ids.is_empty());
        assert!(!out.caption.trim().is_empty());
        assert!(out.audio_code_indices.is_empty());
    }

    #[test]
    #[ignore = "loads the 1.7B LM and samples one 10s code plan"]
    fn planner_codes_cover_ten_seconds() {
        let Some(bundle) = installed_bundle() else {
            eprintln!("skip: 5Hz LM safetensors missing");
            return;
        };
        let out = super::expand(
            &bundle,
            &LmExpandRequest {
                mode: "format",
                caption: "安静的钢琴",
                lyrics: "你好",
                duration: 10.0,
                bpm: 90,
                keyscale: "C major",
                timesignature: "4",
                vocal_language: "zh",
                seed: 7,
                lm_bundle: None,
                existing_codes: None,
            },
            |_, _| (),
        )
        .expect("planner codes");
        assert_eq!(
            out.audio_code_indices.len(),
            50,
            "{:?}",
            out.audio_code_indices
        );
        assert!(out
            .audio_code_indices
            .iter()
            .all(|code| (0..64_000).contains(code)));
        println!("codes {}", out.audio_code_indices.len());
    }
}
