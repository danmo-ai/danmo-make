//! ACE-Step 1.5 music DiT (GenerationFacade).
//!
//! Full text2music create (acestep.cpp `pipeline-synth`): text encode → cond
//! (lyric ‖ timbre[0] ‖ text) → silence/codes context → DiT Euler (APG CFG) →
//! VAE → WAV. Cover / cover-nofsq / repaint / lego / extract / complete on
//! audio.edit. 5Hz LM: ChatML prompts + set_rows KV greedy for inspire/format;
//! generate phase-2 constrained audio codes when metas are complete.

mod apg;
mod ca_capture;
mod cond;
mod cover;
mod create;
mod fsq;
mod fsq_quant;
mod graph;
mod lyrics;
mod lyrics_dtw;
mod lm;
mod lm_codes;
mod lm_format;
mod lm_fsm;
mod lm_prompt;
mod lm_sample;
mod repaint;
mod schedule;
mod text;
mod token_decode;
mod transformer;
mod vae;
mod vae_enc;
pub(crate) mod wav;
mod weights;

use crate::runtime::{BackendKind, GgmlContext};
use crate::weights::{GgufWeights, WeightError};
use std::path::{Path, PathBuf};
#[allow(unused_imports)]
pub use cover::{CoverMode, CoverRequest};
#[allow(unused_imports)]
pub use create::{duration_to_latent_frames, snap_latent_frames, CreateOutput, CreateRequest};
#[allow(unused_imports)]
pub use schedule::{
    sft_schedule, shift_schedule, turbo_schedule, SFT_DEFAULT_SHIFT, SFT_DEFAULT_STEPS,
    TURBO_SHIFT, TURBO_TIMESTEPS,
};
#[allow(unused_imports)]
pub use transformer::{prove_smoke_denoise, AceStepDit};
pub use weights::{
    check_layout, config_from_weights, layout_matches, weight_layout, AceStepConfig, TIME_FREQ_DIM,
};

/// Find the DiT GGUF and confirm the XL / smoke layout.
pub fn probe_layout(bundle: &Path) -> Result<AceStepConfig, WeightError> {
    let path = dit_gguf(bundle)?;
    let loaded = GgufWeights::open(&path)?;
    let config = config_from_weights(&loaded)?;
    check_layout(&loaded, config)?;
    Ok(config)
}

/// Prefer [`run_cover_edit`] with an explicit [`CoverRequest`].
pub fn run_cover(_bundle: &Path) -> Result<(), WeightError> {
    Err(WeightError::NotBuilt(
        "ACE-Step cover needs CoverRequest via audio.edit (task=cover, cover-nofsq, or repaint)".into(),
    ))
}

/// Cover / cover-nofsq / repaint: encode → optional FSQ → context → DiT → VAE.
pub fn run_cover_edit(
    bundle: &Path,
    request: &CoverRequest<'_>,
) -> Result<CreateOutput, WeightError> {
    if crate::weights::assert_gguf_bundle(bundle).is_ok() {
        return cover::run_full_cover(bundle, request);
    }
    Err(WeightError::NotBuilt(
        "ACE-Step cover needs a GGUF bundle with text encoder + VAE + FSQ weights".into(),
    ))
}

/// Alias kept for call sites that only want the no-FSQ path name.
#[allow(dead_code)]
pub fn run_cover_nofsq(
    bundle: &Path,
    request: &CoverRequest<'_>,
) -> Result<CreateOutput, WeightError> {
    run_cover_edit(bundle, request)
}

pub use lm::AceLmConfig;
pub use lm_prompt::{AcePrompt, LmMode};
#[cfg(test)]
pub use lm::write_smoke_lm;

/// 5Hz LM request (acestep.cpp `ace_lm_generate`).
pub struct LmRequest<'a> {
    pub mode: LmMode,
    pub caption: &'a str,
    pub lyrics: &'a str,
    pub duration: f32,
    pub bpm: i32,
    pub keyscale: &'a str,
    pub timesignature: &'a str,
    pub vocal_language: &'a str,
    /// Max newly sampled tokens (inspire/format phase-1).
    pub max_new_tokens: usize,
    /// Optional tokenizer.json contents; when set, prompts are BPE-encoded.
    pub tokenizer_json: Option<&'a str>,
    /// Sampling temperature (`0` = greedy). Default 0.
    pub temperature: f32,
    /// Nucleus top-p in (0, 1]. Default 1.0.
    pub top_p: f32,
    /// Top-k (`0` = disabled, `1` = greedy). Default 0.
    pub top_k: i32,
    /// RNG seed for non-greedy sampling.
    pub seed: u64,
    /// Apply MetadataFSM numeric early-block during phase-1 (needs tokenizer).
    pub use_fsm: bool,
    /// Phase-2 CFG scale (`<= 1` disables). Default 1.0.
    pub cfg_scale: f32,
    /// Negative caption for uncond CFG prompt (empty → lyrics-only uncond).
    pub negative_prompt: &'a str,
}

#[derive(Debug, Clone)]
pub struct LmOutput {
    pub mode: String,
    pub prompt_tokens: usize,
    pub generated_ids: Vec<i32>,
    pub text: String,
    pub caption: String,
    pub lyrics: String,
    pub bpm: Option<i32>,
    pub duration: Option<i32>,
    pub keyscale: String,
    pub timesignature: String,
    pub language: String,
    pub audio_code_indices: Vec<i32>,
}

/// Run inspire / format (phase-1 greedy) or generate (CoT + phase-2 codes).
pub fn run_lm(bundle: &Path, request: &LmRequest<'_>) -> Result<LmOutput, WeightError> {
    let prompt = AcePrompt {
        caption: request.caption.to_string(),
        lyrics: request.lyrics.to_string(),
        duration: request.duration,
        bpm: request.bpm,
        keyscale: request.keyscale.to_string(),
        timesignature: request.timesignature.to_string(),
        vocal_language: request.vocal_language.to_string(),
    };
    // Prefer caller tokenizer; else load from bundle/tokenizer (product layout).
    let owned_tokenizer = if request.tokenizer_json.is_none() {
        text::load_tokenizer_json(bundle).ok()
    } else {
        None
    };
    let tokenizer = request
        .tokenizer_json
        .or(owned_tokenizer.as_deref());
    let mut encode = |text: &str| -> Result<Vec<i32>, WeightError> {
        if let Some(json) = tokenizer {
            crate::common::qwen3::tokenize_plain(json, text, 200_000)
        } else {
            // Synthetic byte ids for prompt-structure tests without a tokenizer.
            Ok(text.bytes().map(|b| i32::from(b)).collect())
        }
    };
    let need_lyrics = prompt.lyrics.trim().is_empty();
    let has_metas = prompt.bpm > 0
        && prompt.duration > 0.0
        && !prompt.keyscale.is_empty()
        && !prompt.timesignature.is_empty();
    // Generate with full metas → phase-2 CoT prompt + constrained codes.
    let phase2 = request.mode == LmMode::Generate && !need_lyrics && has_metas;
    let mut prompt_ids = if phase2 {
        let cot = lm_prompt::build_cot_yaml(&prompt);
        lm_prompt::build_lm_prompt_with_cot(&mut encode, &prompt, &cot)?
    } else {
        lm_prompt::build_phase1_prompt(&mut encode, request.mode, &prompt)?
    };

    let (weights, config) = lm::open_lm_gguf(bundle)?;
    // Synthetic byte-encode (no tokenizer) can emit ChatML specials above a
    // smoke vocab — fold them so prefill stays in-range.
    if tokenizer.is_none() {
        prompt_ids = prompt_ids
            .into_iter()
            .map(|id| {
                if (id as i64) < config.vocab {
                    id
                } else {
                    (id as i64).rem_euclid(config.vocab) as i32
                }
            })
            .collect();
    }
    let ctx = GgmlContext::new(BackendKind::Cpu)
        .map_err(|err| WeightError::NotBuilt(err.to_string()))?;

    if phase2 {
        let max_codes = request
            .max_new_tokens
            .max(1)
            .min(2048)
            .min(((prompt.duration * 5.0) as usize).saturating_add(100).max(1));
        let mut uncond_ids = if request.cfg_scale > 1.0 {
            let mut enc = |text: &str| -> Result<Vec<i32>, WeightError> {
                if let Some(json) = tokenizer {
                    crate::common::qwen3::tokenize_plain(json, text, 200_000)
                } else {
                    Ok(text.bytes().map(|b| i32::from(b)).collect())
                }
            };
            let mut ids =
                lm_prompt::build_lm_prompt_uncond_with_cot(&mut enc, &prompt, request.negative_prompt)?;
            if tokenizer.is_none() {
                ids = ids
                    .into_iter()
                    .map(|id| {
                        if (id as i64) < config.vocab {
                            id
                        } else {
                            (id as i64).rem_euclid(config.vocab) as i32
                        }
                    })
                    .collect();
            }
            Some(ids)
        } else {
            None
        };
        let codes = lm_codes::sample_audio_codes_cfg(
            &ctx,
            &weights,
            config,
            &prompt_ids,
            uncond_ids.as_deref(),
            request.cfg_scale,
            max_codes,
            lm_prompt::AUDIO_CODE_BASE,
            lm_prompt::AUDIO_CODE_COUNT,
            lm_prompt::TOKEN_IM_END,
            request.temperature,
            request.top_p,
            request.top_k,
            request.seed,
        )?;
        let _ = uncond_ids.take();
        return Ok(LmOutput {
            mode: request.mode.name().into(),
            prompt_tokens: prompt_ids.len(),
            generated_ids: lm_prompt::audio_code_token_ids(&codes)?,
            text: String::new(),
            caption: prompt.caption,
            lyrics: prompt.lyrics,
            bpm: (prompt.bpm > 0).then_some(prompt.bpm),
            duration: (prompt.duration > 0.0).then_some(prompt.duration as i32),
            keyscale: prompt.keyscale,
            timesignature: prompt.timesignature,
            language: prompt.vocal_language,
            audio_code_indices: codes,
        });
    }

    let stop: Vec<i32> = [lm_prompt::TOKEN_IM_END, lm_prompt::TOKEN_THINK_END]
        .into_iter()
        .filter(|&id| (id as i64) < config.vocab)
        .collect();
    let max_new = request.max_new_tokens.max(1).min(2048);
    let generated = if request.use_fsm {
        let json = tokenizer.ok_or_else(|| {
            WeightError::NotBuilt(
                "ACE-Step LM MetadataFSM needs tokenizer.json (set use_fsm=false to skip)"
                    .into(),
            )
        })?;
        let mut fsm_session = lm_fsm::MetadataFsmSession::open(json, config.vocab)?;
        let decode_json = json.to_string();
        let decode_token = |id: i32| -> String {
            token_decode::Decoder::from_tokenizer_json(&decode_json)
                .map(|d| d.decode(&[id]))
                .unwrap_or_default()
        };
        let fsm = lm::Phase1FsmMask {
            session: &mut fsm_session,
            decode_token: &decode_token,
        };
        lm::sample_generate_fsm(
            &ctx,
            &weights,
            config,
            &prompt_ids,
            max_new,
            &stop,
            request.temperature,
            request.top_p,
            request.top_k,
            request.seed,
            Some(fsm),
        )?
    } else {
        lm::sample_generate(
            &ctx,
            &weights,
            config,
            &prompt_ids,
            max_new,
            &stop,
            request.temperature,
            request.top_p,
            request.top_k,
            request.seed,
        )?
    };
    let new_ids = generated[prompt_ids.len()..].to_vec();
    let text = if let Some(json) = tokenizer {
        token_decode::Decoder::from_tokenizer_json(json)?.decode(&new_ids)
    } else {
        String::new()
    };
    let meta = lm_format::parse_lm_output(&text);
    let formatted = lm_format::build_lm_format_result(
        &meta,
        request.caption,
        request.lyrics,
        (request.duration > 0.0).then_some(request.duration),
        (request.bpm > 0).then_some(request.bpm),
        request.keyscale,
        request.timesignature,
        request.vocal_language,
        !request.lyrics.trim().is_empty() && request.lyrics.trim() != "[Instrumental]",
        true,
    );
    Ok(LmOutput {
        mode: request.mode.name().into(),
        prompt_tokens: prompt_ids.len(),
        generated_ids: new_ids,
        text,
        caption: formatted.caption,
        lyrics: formatted.lyrics,
        bpm: formatted.bpm,
        duration: formatted.duration,
        keyscale: formatted.keyscale,
        timesignature: formatted.timesignature,
        language: formatted.language,
        audio_code_indices: formatted.audio_code_indices,
    })
}

/// Back-compat alias: inspire expand on an LM bundle.
pub fn run_lm_expand(bundle: &Path) -> Result<LmOutput, WeightError> {
    run_lm(
        bundle,
        &LmRequest {
            mode: LmMode::Inspire,
            caption: "test",
            lyrics: "",
            duration: 0.0,
            bpm: 0,
            keyscale: "",
            timesignature: "",
            vocal_language: "",
            max_new_tokens: 8,
            tokenizer_json: None,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
            use_fsm: false,
            cfg_scale: 1.0,
            negative_prompt: "",
        },
    )
}

/// Product create. A full `bundle.manifest.json` (DiT + text + VAE + tokenizer)
/// runs the acestep.cpp text2music path. A DiT-only smoke card proves Euler
/// then refuses until companions are installed.
pub fn run_create(bundle: &Path, request: &CreateRequest<'_>) -> Result<CreateOutput, WeightError> {
    if crate::weights::assert_gguf_bundle(bundle).is_ok() {
        return create::run_full_create(bundle, request);
    }
    let config = probe_layout(bundle)?;
    if config == AceStepConfig::smoke() {
        let ctx = GgmlContext::new(BackendKind::Cpu)
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        let _ = prove_smoke_denoise(&ctx, bundle, config)?;
        return Err(WeightError::NotBuilt(
            "ACE-Step needs text_encoder/, vae/, and tokenizer/ companions (install the registry components)".into(),
        ));
    }
    Err(WeightError::NotBuilt(
        "ACE-Step XL create needs a GGUF bundle with text encoder + VAE (see registry components)".into(),
    ))
}

pub(crate) fn dit_gguf(bundle: &Path) -> Result<PathBuf, WeightError> {
    if !bundle.is_dir() {
        return Err(WeightError::Bundle {
            path: bundle.display().to_string(),
            detail: "directory is missing".into(),
        });
    }
    let mut matches = Vec::new();
    let entries = std::fs::read_dir(bundle).map_err(|err| WeightError::Bundle {
        path: bundle.display().to_string(),
        detail: err.to_string(),
    })?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
            continue;
        }
        let Ok(weights) = GgufWeights::open(&path) else {
            continue;
        };
        if weights.contains("decoder.proj_in.1.weight") {
            matches.push(path);
        }
    }
    match matches.len() {
        1 => Ok(matches.remove(0)),
        0 => Err(WeightError::NotBuilt(
            "ACE-Step DiT GGUF with decoder.proj_in.1.weight is missing".into(),
        )),
        _ => Err(WeightError::NotBuilt(format!(
            "ACE-Step bundle has {} DiT GGUF files; keep one",
            matches.len()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        config_from_weights, dit_gguf, layout_matches, probe_layout, run_cover, run_create,
        weight_layout, AceStepConfig, CreateRequest, TIME_FREQ_DIM,
    };
    use crate::weights::{write_f32_nd, F32Nd, GgufWeights};
    use std::path::Path;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "danmo-ace-{name}-{}-{}",
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

    fn write_dit(dir: &Path, config: AceStepConfig) {
        let owned: Vec<(String, [i64; 4], Vec<f32>)> = weight_layout(config)
            .into_iter()
            .map(|row| {
                let n = row.ne.iter().product::<i64>() as usize;
                (row.name, row.ne, vec![0f32; n])
            })
            .collect();
        let tensors: Vec<F32Nd<'_>> = owned
            .iter()
            .map(|(name, ne, data)| F32Nd {
                name,
                ne: *ne,
                data,
            })
            .collect();
        write_f32_nd(&dir.join("acestep-v15-smoke.gguf"), &tensors).unwrap();
    }

    #[test]
    fn xl_layout_has_thirty_two_layers_and_the_acestep_names() {
        let config = AceStepConfig::xl(false);
        assert_eq!(config.hidden, 2560);
        assert_eq!(config.layers, 32);
        assert_eq!(config.intermediate, 9728);
        assert_eq!(config.heads, 32);
        assert_eq!(config.kv_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.in_channels, 192);
        assert_eq!(config.out_channels, 64);
        assert_eq!(config.patch, 2);
        assert_eq!(config.sliding_window, 128);
        assert_eq!(config.condition_dim, 2048);
        assert_eq!(TIME_FREQ_DIM, 256);
        let names: Vec<_> = weight_layout(config)
            .into_iter()
            .map(|row| row.name)
            .collect();
        assert!(names
            .iter()
            .any(|name| name == "decoder.layers.0.self_attn.q_proj.weight"));
        assert!(names
            .iter()
            .any(|name| name == "decoder.layers.31.mlp.down_proj.weight"));
        assert!(names.iter().any(|name| name == "decoder.proj_in.1.weight"));
        assert!(names
            .iter()
            .any(|name| name == "decoder.time_embed_r.time_proj.weight"));
        assert!(names.iter().all(|name| !name.starts_with("layers.")));
        assert!(!names.iter().any(|name| name.contains("null_condition")));
        assert_eq!(
            names
                .iter()
                .filter(|name| name.contains("self_attn.q_proj.weight"))
                .count(),
            32
        );
    }

    #[test]
    fn smoke_probe_accepts_the_layout() {
        let dir = scratch("smoke");
        write_dit(&dir, AceStepConfig::smoke());
        let path = dit_gguf(&dir).unwrap();
        let loaded = GgufWeights::open(&path).unwrap();
        let config = config_from_weights(&loaded).unwrap();
        assert_eq!(config, AceStepConfig::smoke());
        assert!(layout_matches(&loaded, config));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn base_2b_layout_is_refused() {
        let dir = scratch("2b");
        write_dit(&dir, AceStepConfig::base_2b());
        let err = probe_layout(&dir).unwrap_err();
        assert!(
            err.to_string().contains("2B") && err.to_string().contains("XL"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_proj_in_fails_loud() {
        let dir = scratch("empty");
        let err = probe_layout(&dir).unwrap_err();
        assert!(
            err.to_string().contains("decoder.proj_in.1.weight"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn audio_create_runs_smoke_denoise_then_refuses_companions() {
        let dir = scratch("create");
        write_dit(&dir, AceStepConfig::smoke());
        let err = crate::pipeline::audio_create(
            "ace_step",
            &dir,
            &crate::pipeline::AudioCreateRequest {
                prompt: "test",
                lyrics: "[inst]",
                duration_sec: 5.0,
                seed: Some(0),
                guidance: 1.0,
                steps: None,
                vocal_language: "en",
                audio_code_indices: None,
                timbre_latents: None,
            },
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("companions") || err.to_string().contains("text_encoder"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cover_and_lm_fail_loud_with_acestep_pointers() {
        let err = run_cover(Path::new("/tmp")).unwrap_err();
        assert!(
            err.to_string().contains("CoverRequest") || err.to_string().contains("audio.edit"),
            "{err}"
        );
        let err = super::run_lm_expand(Path::new("/tmp/no-such-ace-lm")).unwrap_err();
        assert!(
            err.to_string().contains("LM") || err.to_string().contains("missing"),
            "{err}"
        );
    }

    #[test]
    fn lm_inspire_smoke_runs_greedy_prefill() {
        let dir = scratch("lm-inspire");
        super::write_smoke_lm(&dir, super::AceLmConfig::smoke()).unwrap();
        let out = super::run_lm(
            &dir,
            &super::LmRequest {
                mode: super::LmMode::Inspire,
                caption: "lofi",
                lyrics: "",
                duration: 0.0,
                bpm: 0,
                keyscale: "",
                timesignature: "",
                vocal_language: "en",
                max_new_tokens: 3,
                tokenizer_json: None,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                seed: 0,
                use_fsm: false,
                cfg_scale: 1.0,
                negative_prompt: "",
            },
        )
        .unwrap();
        assert_eq!(out.mode, "inspire");
        assert!(out.prompt_tokens > 0);
        assert!(!out.generated_ids.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lm_generate_phase2_fails_loud_on_smoke_vocab() {
        let dir = scratch("lm-phase2");
        super::write_smoke_lm(&dir, super::AceLmConfig::smoke()).unwrap();
        let err = super::run_lm(
            &dir,
            &super::LmRequest {
                mode: super::LmMode::Generate,
                caption: "lofi beat",
                lyrics: "la la",
                duration: 10.0,
                bpm: 90,
                keyscale: "C major",
                timesignature: "4/4",
                vocal_language: "en",
                max_new_tokens: 8,
                tokenizer_json: None,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                seed: 0,
                use_fsm: false,
                cfg_scale: 1.0,
                negative_prompt: "",
            },
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("vocab") || err.to_string().contains("audio code"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_create_on_smoke_proves_denoise_then_refuses() {
        let dir = scratch("prove");
        write_dit(&dir, AceStepConfig::smoke());
        let err = run_create(
            &dir,
            &CreateRequest {
                prompt: "test",
                lyrics: "[inst]",
                duration_sec: 5.0,
                seed: Some(0),
                guidance: 1.0,
                steps: None,
                vocal_language: "en",
                bpm: None,
                keyscale: "",
                timesignature: "",
                audio_code_indices: None,
                timbre_latents: None,
            },
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("companions") || err.to_string().contains("text_encoder"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
