//! Qwen3-Embedding text tower for ACE-Step (acestep.cpp `qwen3-enc.h`).
//!
//! Caption tokens run the full causal stack. Lyric tokens use the embedding
//! table only (`qwen3_embed_lookup`). Prompts are plain BPE (no ChatML wrap).

use crate::common::ops::non_finite;
use crate::common::qwen3;
use crate::runtime::GgmlContext;
use crate::weights::{GgufWeights, WeightError};
use std::path::{Path, PathBuf};

/// acestep.cpp `DIT_INSTR_TEXT2MUSIC`.
pub const DIT_INSTR_TEXT2MUSIC: &str =
    "Fill the audio semantic mask based on the given conditions:";

/// acestep.cpp `DIT_INSTR_COVER`.
pub const DIT_INSTR_COVER: &str =
    "Generate audio semantic tokens based on the given conditions:";

/// acestep.cpp `DIT_INSTR_REPAINT`.
pub const DIT_INSTR_REPAINT: &str = "Repaint the mask area based on the given conditions:";

/// acestep.cpp `TRACK_NAMES` for lego / extract / complete.
pub const TRACK_NAMES: &[&str] = &[
    "vocals",
    "backing_vocals",
    "drums",
    "bass",
    "guitar",
    "keyboard",
    "percussion",
    "strings",
    "synth",
    "fx",
    "brass",
    "woodwinds",
];

pub const EMBED_HIDDEN: i64 = 1024;

/// Parse `"drums | bass | guitar"` (complete) or a single track; empty → `[]`.
/// Unknown names fail loud (product path; acestep.cpp only warns).
pub fn parse_track_names(track: &str) -> Result<Vec<String>, WeightError> {
    let trimmed = track.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for part in trimmed.split(" | ") {
        let name = part.trim().to_ascii_lowercase();
        if name.is_empty() {
            return Err(WeightError::NotBuilt(
                "ACE-Step track list has an empty name".into(),
            ));
        }
        if !TRACK_NAMES.iter().any(|t| *t == name) {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step track '{name}' is not a standard name ({})",
                TRACK_NAMES.join(", ")
            )));
        }
        out.push(name);
    }
    Ok(out)
}

/// acestep.cpp `dit_instr_lego` / upstream TASK_INSTRUCTIONS (track UPPERCASE).
pub fn dit_instr_lego(track: &str) -> Result<String, WeightError> {
    let names = parse_track_names(track)?;
    if names.is_empty() {
        Ok("Generate the track based on the audio context:".into())
    } else if names.len() == 1 {
        Ok(format!(
            "Generate the {} track based on the audio context:",
            names[0].to_ascii_uppercase()
        ))
    } else {
        Err(WeightError::NotBuilt(
            "ACE-Step lego expects a single track name".into(),
        ))
    }
}

/// acestep.cpp `dit_instr_extract`.
pub fn dit_instr_extract(track: &str) -> Result<String, WeightError> {
    let names = parse_track_names(track)?;
    if names.is_empty() {
        Ok("Extract the track from the audio:".into())
    } else if names.len() == 1 {
        Ok(format!(
            "Extract the {} track from the audio:",
            names[0].to_ascii_uppercase()
        ))
    } else {
        Err(WeightError::NotBuilt(
            "ACE-Step extract expects a single track name".into(),
        ))
    }
}

/// acestep.cpp `dit_instr_complete` (multi-track via `" | "`).
pub fn dit_instr_complete(track: &str) -> Result<String, WeightError> {
    let names = parse_track_names(track)?;
    if names.is_empty() {
        Ok("Complete the input track:".into())
    } else {
        let joined = names
            .iter()
            .map(|n| n.to_ascii_uppercase())
            .collect::<Vec<_>>()
            .join(" | ");
        Ok(format!("Complete the input track with {joined}:"))
    }
}

#[derive(Debug, Clone)]
pub struct PromptPair {
    pub text: String,
    pub lyric: String,
}

/// Matches `build_prompt_strings` in acestep.cpp `pipeline-synth-ops.cpp`.
pub fn build_prompt_strings(
    caption: &str,
    lyrics: &str,
    instruction: &str,
    duration_sec: f32,
    bpm: Option<i32>,
    keyscale: &str,
    timesignature: &str,
    vocal_language: &str,
) -> PromptPair {
    let bpm_b = bpm
        .filter(|v| *v > 0)
        .map(|v| v.to_string())
        .unwrap_or_else(|| "N/A".into());
    let keyscale_b = if keyscale.is_empty() {
        "N/A"
    } else {
        keyscale
    };
    let timesig_b = if timesignature.is_empty() {
        "N/A"
    } else {
        timesignature
    };
    let language_b = if vocal_language.is_empty() {
        "unknown"
    } else {
        vocal_language
    };
    let metas = format!(
        "- bpm: {bpm_b}\n- timesignature: {timesig_b}\n- keyscale: {keyscale_b}\n- duration: {} seconds\n",
        duration_sec as i32
    );
    let text = format!(
        "# Instruction\n{instruction}\n\n# Caption\n{caption}\n\n# Metas\n{metas}<|endoftext|>\n"
    );
    let lyric = format!("# Languages\n{language_b}\n\n# Lyric\n{lyrics}<|endoftext|>");
    PromptPair { text, lyric }
}

pub fn encode_caption(
    ctx: &GgmlContext,
    paths: &[PathBuf],
    tokenizer_json: &str,
    caption_prompt: &str,
) -> Result<(Vec<f32>, i64), WeightError> {
    let loaded = qwen3::Loaded::open(paths)?;
    let config = loaded.config();
    if config.hidden != EMBED_HIDDEN {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step text encoder hidden is {}, expected {EMBED_HIDDEN}",
            config.hidden
        )));
    }
    let (mut embed, seq) =
        loaded.encode_plain(ctx, tokenizer_json, caption_prompt, &[config.layers], config.hidden)?;
    let weights = GgufWeights::open_parts(paths)?;
    if let Some(gamma) = final_norm_gamma(&weights, config.hidden)? {
        embed = rms_norm_rows(&embed, &gamma, config.hidden as usize, 1e-6);
    }
    if let Some(detail) = non_finite("ACE-Step text hidden", &embed) {
        return Err(WeightError::NotBuilt(detail));
    }
    Ok((embed, seq))
}

/// Vocab lookup only (lyric path). Returns `[hidden * seq]`.
pub fn embed_lyrics(
    paths: &[PathBuf],
    tokenizer_json: &str,
    lyric_prompt: &str,
) -> Result<(Vec<f32>, i64), WeightError> {
    let (out, ids) = embed_lyrics_with_ids(paths, tokenizer_json, lyric_prompt)?;
    Ok((out, ids.len() as i64))
}

/// Same as [`embed_lyrics`] but also returns the token ids (for CA+DTW).
pub fn embed_lyrics_with_ids(
    paths: &[PathBuf],
    tokenizer_json: &str,
    lyric_prompt: &str,
) -> Result<(Vec<f32>, Vec<i32>), WeightError> {
    let loaded = qwen3::Loaded::open(paths)?;
    let config = loaded.config();
    let ids = loaded.plain_ids(tokenizer_json, lyric_prompt)?;
    if ids.is_empty() {
        return Err(WeightError::NotBuilt(
            "ACE-Step lyric prompt produced no tokens".into(),
        ));
    }
    let table = loaded.embed_table()?;
    let hidden = config.hidden as usize;
    let mut out = Vec::with_capacity(ids.len() * hidden);
    for id in &ids {
        let row = *id as usize;
        let start = row * hidden;
        let end = start + hidden;
        if end > table.len() {
            return Err(WeightError::NotBuilt(format!(
                "lyric token id {id} is outside the embedding table"
            )));
        }
        out.extend_from_slice(&table[start..end]);
    }
    Ok((out, ids))
}

/// Tokenize a plain prompt (no ChatML) — used for lyric header length.
pub fn plain_token_ids(
    paths: &[PathBuf],
    tokenizer_json: &str,
    prompt: &str,
) -> Result<Vec<i32>, WeightError> {
    let loaded = qwen3::Loaded::open(paths)?;
    loaded.plain_ids(tokenizer_json, prompt)
}

pub fn load_tokenizer_json(bundle: &Path) -> Result<String, WeightError> {
    qwen3::load_tokenizer(bundle)
}

fn final_norm_gamma(weights: &GgufWeights, hidden: i64) -> Result<Option<Vec<f32>>, WeightError> {
    for name in ["norm.weight", "model.norm.weight"] {
        if weights.contains(name) {
            return Ok(Some(weights.f32_widened(name, &[hidden, 1])?));
        }
    }
    Ok(None)
}

fn rms_norm_rows(data: &[f32], gamma: &[f32], hidden: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; data.len()];
    let rows = data.len() / hidden;
    for r in 0..rows {
        let start = r * hidden;
        let row = &data[start..start + hidden];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv = (mean_sq + eps).sqrt().recip();
        for i in 0..hidden {
            out[start + i] = row[i] * inv * gamma[i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        build_prompt_strings, dit_instr_complete, dit_instr_extract, dit_instr_lego,
        parse_track_names, DIT_INSTR_TEXT2MUSIC,
    };

    #[test]
    fn prompt_strings_match_acestep_headers() {
        let pair = build_prompt_strings(
            "lofi beat",
            "[inst]",
            DIT_INSTR_TEXT2MUSIC,
            30.0,
            Some(90),
            "C major",
            "4/4",
            "en",
        );
        assert!(pair.text.contains("# Instruction\n"));
        assert!(pair.text.contains("# Caption\nlofi beat"));
        assert!(pair.text.contains("duration: 30 seconds"));
        assert!(pair.lyric.contains("# Languages\nen"));
        assert!(pair.lyric.contains("# Lyric\n[inst]"));
        assert!(pair.text.contains("<|endoftext|>"));
    }

    #[test]
    fn track_instructions_uppercase_names() {
        assert_eq!(
            dit_instr_lego("guitar").unwrap(),
            "Generate the GUITAR track based on the audio context:"
        );
        assert_eq!(
            dit_instr_extract("vocals").unwrap(),
            "Extract the VOCALS track from the audio:"
        );
        assert_eq!(
            dit_instr_complete("drums | bass | guitar").unwrap(),
            "Complete the input track with DRUMS | BASS | GUITAR:"
        );
        assert!(parse_track_names("banjo").is_err());
        assert!(dit_instr_lego("drums | bass").is_err());
    }
}
