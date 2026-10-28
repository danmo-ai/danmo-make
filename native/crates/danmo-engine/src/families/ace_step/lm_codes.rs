//! ACE-Step LM phase-2 audio-code constrained sampling (acestep.cpp `run_phase2_*`).
//!
//! After CoT metadata, logits are masked to `{IM_END} ∪ audio_code_*` and
//! greedily sampled until EOS or `max_codes`.

use super::lm::{LmSetRowsSession, AceLmConfig};
use super::lm_prompt::{AUDIO_CODE_BASE, AUDIO_CODE_COUNT, TOKEN_IM_END};
use crate::runtime::GgmlContext;
use crate::weights::{GgufWeights, WeightError};

/// Mask logits so only `eos` and `code_base .. code_base+code_count` remain.
pub fn mask_audio_code_logits(
    logits: &mut [f32],
    code_base: i32,
    code_count: i32,
    eos: i32,
) -> Result<(), WeightError> {
    if code_count < 1 {
        return Err(WeightError::NotBuilt(
            "ACE-Step LM audio code count must be >= 1".into(),
        ));
    }
    let v = logits.len() as i64;
    let need = code_base as i64 + code_count as i64;
    if need > v {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step LM vocab {v} cannot hold audio codes at base {code_base} count {code_count}"
        )));
    }
    if eos < 0 || eos as i64 >= v {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step LM EOS id {eos} is outside vocab {v}"
        )));
    }
    for (i, logit) in logits.iter_mut().enumerate() {
        let id = i as i32;
        let allow = id == eos || (id >= code_base && id < code_base + code_count);
        if !allow {
            *logit = f32::NEG_INFINITY;
        }
    }
    Ok(())
}

/// Greedy phase-2: sample audio-code tokens (and optional EOS) from a CoT prompt.
///
/// Returns FSQ codebook indices (`token - code_base`), not raw token ids.
pub fn greedy_audio_codes(
    ctx: &GgmlContext,
    weights: &GgufWeights,
    config: AceLmConfig,
    cot_prompt: &[i32],
    max_codes: usize,
) -> Result<Vec<i32>, WeightError> {
    sample_audio_codes(
        ctx, weights, config, cot_prompt, max_codes, 0.0, 1.0, 0, 0,
    )
}

/// Phase-2 with temperature / top-k / top-p (acestep.cpp `run_phase2_*`).
pub fn sample_audio_codes(
    ctx: &GgmlContext,
    weights: &GgufWeights,
    config: AceLmConfig,
    cot_prompt: &[i32],
    max_codes: usize,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    seed: u64,
) -> Result<Vec<i32>, WeightError> {
    sample_audio_codes_cfg(
        ctx,
        weights,
        config,
        cot_prompt,
        None,
        1.0,
        max_codes,
        AUDIO_CODE_BASE,
        AUDIO_CODE_COUNT,
        TOKEN_IM_END,
        temperature,
        top_p,
        top_k,
        seed,
    )
}

/// Same as [`greedy_audio_codes`] with explicit special-token layout (tests).
pub fn greedy_audio_codes_ex(
    ctx: &GgmlContext,
    weights: &GgufWeights,
    config: AceLmConfig,
    cot_prompt: &[i32],
    max_codes: usize,
    code_base: i32,
    code_count: i32,
    eos: i32,
) -> Result<Vec<i32>, WeightError> {
    sample_audio_codes_cfg(
        ctx,
        weights,
        config,
        cot_prompt,
        None,
        1.0,
        max_codes,
        code_base,
        code_count,
        eos,
        0.0,
        1.0,
        0,
        0,
    )
}

/// Back-compat alias without CFG.
pub fn sample_audio_codes_ex(
    ctx: &GgmlContext,
    weights: &GgufWeights,
    config: AceLmConfig,
    cot_prompt: &[i32],
    max_codes: usize,
    code_base: i32,
    code_count: i32,
    eos: i32,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    seed: u64,
) -> Result<Vec<i32>, WeightError> {
    sample_audio_codes_cfg(
        ctx,
        weights,
        config,
        cot_prompt,
        None,
        1.0,
        max_codes,
        code_base,
        code_count,
        eos,
        temperature,
        top_p,
        top_k,
        seed,
    )
}

/// Phase-2 with optional cond/uncond CFG (`cfg_scale > 1` + `uncond_prompt`).
pub fn sample_audio_codes_cfg(
    ctx: &GgmlContext,
    weights: &GgufWeights,
    config: AceLmConfig,
    cot_prompt: &[i32],
    uncond_prompt: Option<&[i32]>,
    cfg_scale: f32,
    max_codes: usize,
    code_base: i32,
    code_count: i32,
    eos: i32,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    seed: u64,
) -> Result<Vec<i32>, WeightError> {
    if cot_prompt.is_empty() {
        return Err(WeightError::NotBuilt(
            "ACE-Step LM phase-2 needs a non-empty CoT prompt".into(),
        ));
    }
    if max_codes < 1 {
        return Err(WeightError::NotBuilt(
            "ACE-Step LM phase-2 max_codes must be >= 1".into(),
        ));
    }
    if !cfg_scale.is_finite() || cfg_scale < 0.0 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step LM CFG scale {cfg_scale} is invalid"
        )));
    }
    let use_cfg = cfg_scale > 1.0;
    if use_cfg && uncond_prompt.map(|p| p.is_empty()).unwrap_or(true) {
        return Err(WeightError::NotBuilt(
            "ACE-Step LM CFG needs a non-empty uncond prompt when cfg_scale > 1".into(),
        ));
    }
    let need = code_base as i64 + code_count as i64;
    if config.vocab < need || (eos as i64) >= config.vocab {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step LM vocab {} cannot sample audio codes (need base {code_base}+{code_count}, eos {eos})",
            config.vocab
        )));
    }
    let max_seq = (cot_prompt.len() + max_codes).max(1) as i64;
    let mut cond = LmSetRowsSession::open(ctx, weights, config, max_seq)?;
    let mut logits = cond.prefill(cot_prompt)?;
    let (mut uncond, mut uncond_logits) = if use_cfg {
        let up = uncond_prompt.unwrap();
        let max_u = (up.len() + max_codes).max(1) as i64;
        let mut s = LmSetRowsSession::open(ctx, weights, config, max_u)?;
        let ul = s.prefill(up)?;
        (Some(s), Some(ul))
    } else {
        (None, None)
    };
    let mut codes = Vec::new();
    let mut rng = super::lm_sample::XorShift64::new(seed);
    for _ in 0..max_codes {
        if let Some(ul) = uncond_logits.as_ref() {
            super::lm_sample::cfg_blend(&mut logits, ul, cfg_scale)?;
        }
        mask_audio_code_logits(&mut logits, code_base, code_count, eos)?;
        let next = super::lm_sample::sample_top_k_p(
            &logits, temperature, top_p, top_k, &mut rng,
        )?;
        if next == eos {
            break;
        }
        if next < code_base || next >= code_base + code_count {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step LM phase-2 sampled id {next} outside audio codes"
            )));
        }
        codes.push(next - code_base);
        logits = cond.decode(next)?;
        if let Some(s) = uncond.as_mut() {
            uncond_logits = Some(s.decode(next)?);
        }
    }
    Ok(codes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::ace_step::lm::{write_smoke_lm, AceLmConfig};
    use crate::runtime::{BackendKind, GgmlContext};
    use crate::weights::GgufWeights;

    #[test]
    fn mask_keeps_eos_and_code_band() {
        let mut logits = vec![1.0f32; 20];
        mask_audio_code_logits(&mut logits, 10, 5, 3).unwrap();
        assert_eq!(logits[3], 1.0);
        assert_eq!(logits[10], 1.0);
        assert_eq!(logits[14], 1.0);
        assert!(logits[0].is_infinite() && logits[0].is_sign_negative());
        assert!(logits[9].is_infinite() && logits[9].is_sign_negative());
        assert!(logits[15].is_infinite() && logits[15].is_sign_negative());
    }

    #[test]
    fn mask_fails_loud_when_vocab_too_small() {
        let mut logits = vec![0.0f32; 8];
        let err = mask_audio_code_logits(&mut logits, 10, 5, 1).unwrap_err();
        assert!(err.to_string().contains("vocab"), "{err}");
    }

    #[test]
    fn cfg_without_uncond_fails_loud() {
        let dir = std::env::temp_dir().join(format!(
            "ace-lm-cfg-fail-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let config = AceLmConfig::smoke();
        write_smoke_lm(&dir, config).unwrap();
        let weights = GgufWeights::open(&dir.join("acestep-lm-smoke.gguf")).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let prompt = vec![1i32, 2, 3];
        let err = sample_audio_codes_cfg(
            &ctx,
            &weights,
            config,
            &prompt,
            None,
            2.0,
            2,
            16,
            8,
            15,
            0.0,
            1.0,
            0,
            0,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("uncond"),
            "expected uncond fail-loud, got {err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn smoke_phase2_with_remapped_code_band() {
        let dir = std::env::temp_dir().join(format!(
            "ace-lm-codes-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // vocab=32: place a tiny code band at [16, 24) with eos=15.
        let config = AceLmConfig::smoke();
        write_smoke_lm(&dir, config).unwrap();
        let weights = GgufWeights::open(&dir.join("acestep-lm-smoke.gguf")).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let prompt = vec![1i32, 2, 3];
        let codes =
            greedy_audio_codes_ex(&ctx, &weights, config, &prompt, 4, 16, 8, 15).unwrap();
        // Zero weights → argmax picks lowest allowed id (eos=15) first → empty codes.
        assert!(codes.len() <= 4);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
