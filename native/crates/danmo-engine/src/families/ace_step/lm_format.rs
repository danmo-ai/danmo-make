//! ACE-Step 5Hz LM format_sample helpers (main `lm/lm_format.py`).
//!
//! Full Qwen3 LM sampling is not built yet. This module parses CoT output,
//! preserves user vocal lyrics, and extracts `<|audio_code_N|>` indices so
//! create can condition DiT on codes via FSQ detok.

use super::lyrics::{is_instrumental_lyrics, normalize_lyrics_body};
use crate::weights::WeightError;
use regex::Regex;
use std::sync::OnceLock;

/// Official rewrite instruction (format_sample system prompt).
pub const DEFAULT_LM_REWRITE_INSTRUCTION: &str =
    "Format the user's input into a more detailed and specific musical description:";

#[derive(Debug, Clone, PartialEq)]
pub struct LmFormatResult {
    pub caption: String,
    pub lyrics: String,
    pub bpm: Option<i32>,
    pub duration: Option<i32>,
    pub keyscale: String,
    pub timesignature: String,
    pub language: String,
    pub audio_codes: String,
    pub audio_code_indices: Vec<i32>,
}

fn audio_code_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"<\|audio_code_\d+\|>").expect("audio code re"))
}

fn audio_code_num_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"<\|audio_code_(\d+)\|>").expect("audio code num re"))
}

fn think_start_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)<think>\s*").expect("think start"))
}

fn think_end_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)</think>").expect("think end"))
}

pub fn parse_audio_code_indices(text: &str) -> Vec<i32> {
    audio_code_num_re()
        .captures_iter(text)
        .filter_map(|c| c.get(1)?.as_str().parse().ok())
        .collect()
}

pub fn extract_audio_codes(text: &str) -> String {
    audio_code_re()
        .find_iter(text)
        .map(|m| m.as_str())
        .collect()
}

pub fn extract_think_body(output_text: &str) -> (String, String) {
    let mut text = output_text.to_string();
    if let Some(m) = audio_code_re().find(&text) {
        text.truncate(m.start());
    }
    if let Some(start) = think_start_re().find(&text) {
        let body = &text[start.end()..];
        if let Some(end) = think_end_re().find(body) {
            return (
                body[..end.start()].trim().to_string(),
                body[end.end()..].trim().to_string(),
            );
        }
        return (body.trim().to_string(), String::new());
    }
    if let Some(end) = think_end_re().find(&text) {
        return (
            text[..end.start()].trim().to_string(),
            text[end.end()..].trim().to_string(),
        );
    }
    (text.trim().to_string(), String::new())
}

/// Parse CoT reasoning key:value lines into a flat meta map.
pub fn parse_reasoning_block(text: &str) -> std::collections::HashMap<String, String> {
    let mut metadata = std::collections::HashMap::new();
    if text.trim().is_empty() {
        return metadata;
    }
    let mut current_key: Option<String> = None;
    let mut value_lines: Vec<String> = Vec::new();

    let flush = |key: &mut Option<String>,
                 values: &mut Vec<String>,
                 metadata: &mut std::collections::HashMap<String, String>| {
        let Some(k) = key.take() else {
            values.clear();
            return;
        };
        if values.is_empty() {
            return;
        }
        let value = values.join("\n").trim().to_string();
        values.clear();
        metadata.insert(k, value);
    };

    for line in text.split('\n') {
        let stripped = line.trim();
        if stripped.is_empty() || stripped.starts_with('<') {
            continue;
        }
        let first = line.chars().next();
        if first.is_some_and(|c| !c.is_whitespace()) && line.contains(':') {
            flush(&mut current_key, &mut value_lines, &mut metadata);
            let (key, rest) = line.split_once(':').unwrap();
            current_key = Some(key.trim().to_ascii_lowercase());
            let rest = rest.trim();
            if !rest.is_empty() {
                value_lines.push(rest.to_string());
            }
        } else if line.starts_with([' ', '\t']) && current_key.is_some() {
            value_lines.push(line.to_string());
        }
    }
    flush(&mut current_key, &mut value_lines, &mut metadata);
    metadata
}

pub fn parse_lm_output(output_text: &str) -> std::collections::HashMap<String, String> {
    let (reasoning, tail) = extract_think_body(output_text);
    let mut meta = parse_reasoning_block(&reasoning);
    let codes = extract_audio_codes(output_text);
    if !codes.is_empty() {
        meta.insert("audio_codes".into(), codes.clone());
        let indices = parse_audio_code_indices(&codes);
        meta.insert(
            "audio_code_indices".into(),
            indices
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    if !tail.is_empty() {
        meta.insert("lyrics_tail".into(), normalize_lyrics_body(&tail));
    }
    meta
}

/// Assemble format_sample result; `preserve_user_lyrics` keeps non-instrumental input.
pub fn build_lm_format_result(
    meta: &std::collections::HashMap<String, String>,
    caption_in: &str,
    lyrics_in: &str,
    duration: Option<f32>,
    bpm: Option<i32>,
    keyscale: &str,
    timesignature: &str,
    language: &str,
    want_vocals: bool,
    preserve_user_lyrics: bool,
) -> LmFormatResult {
    let out_caption = meta
        .get("caption")
        .cloned()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| caption_in.to_string());
    let lm_lyrics = normalize_lyrics_body(meta.get("lyrics_tail").map(|s| s.as_str()).unwrap_or(""));
    let lyrics_in_norm = normalize_lyrics_body(lyrics_in);
    let user_wants_vocals = want_vocals || !is_instrumental_lyrics(lyrics_in);
    let out_lyrics = if preserve_user_lyrics
        && !lyrics_in_norm.is_empty()
        && !is_instrumental_lyrics(&lyrics_in_norm)
    {
        lyrics_in_norm.clone()
    } else if user_wants_vocals && is_instrumental_lyrics(&lm_lyrics) {
        if !lyrics_in_norm.is_empty() && !is_instrumental_lyrics(&lyrics_in_norm) {
            lyrics_in_norm
        } else {
            String::new()
        }
    } else if !lm_lyrics.is_empty() {
        lm_lyrics
    } else {
        lyrics_in_norm
    };

    let out_bpm = meta
        .get("bpm")
        .and_then(|v| v.split_whitespace().next()?.parse::<f32>().ok())
        .map(|v| v as i32)
        .or(bpm);
    let out_duration = meta
        .get("duration")
        .and_then(|v| {
            let digits: String = v.chars().filter(|c| c.is_ascii_digit()).collect();
            digits.parse::<i32>().ok()
        })
        .or_else(|| duration.map(|d| d.round() as i32));

    let audio_codes = meta.get("audio_codes").cloned().unwrap_or_default();
    let audio_code_indices = if let Some(csv) = meta.get("audio_code_indices") {
        csv.split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect()
    } else {
        parse_audio_code_indices(&audio_codes)
    };

    LmFormatResult {
        caption: out_caption,
        lyrics: out_lyrics,
        bpm: out_bpm,
        duration: out_duration,
        keyscale: meta
            .get("keyscale")
            .cloned()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| keyscale.to_string()),
        timesignature: meta
            .get("timesignature")
            .cloned()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| timesignature.to_string()),
        language: meta
            .get("language")
            .cloned()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                if language.is_empty() {
                    "en".into()
                } else {
                    language.into()
                }
            }),
        audio_codes,
        audio_code_indices,
    }
}

/// Validate codebook indices for FSQ detok (`[0, 64000)`).
pub fn validate_audio_code_indices(codes: &[i32]) -> Result<(), WeightError> {
    if codes.is_empty() {
        return Err(WeightError::NotBuilt(
            "ACE-Step audio_code_indices must be non-empty".into(),
        ));
    }
    for &c in codes {
        if !(0..super::fsq_quant::FSQ_CODEBOOK_SIZE).contains(&c) {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step audio code {c} is outside [0, {})",
                super::fsq_quant::FSQ_CODEBOOK_SIZE
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_audio_code_tags() {
        let text = "<|audio_code_12|><|audio_code_64000|>";
        assert_eq!(parse_audio_code_indices(text), vec![12, 64000]);
        assert_eq!(extract_audio_codes(text), text);
    }

    #[test]
    fn preserve_user_lyrics_keeps_verses() {
        let mut meta = std::collections::HashMap::new();
        meta.insert("caption".into(), "jazz trio".into());
        meta.insert("lyrics_tail".into(), "[Verse]\nrewritten".into());
        meta.insert("bpm".into(), "120".into());
        let got = build_lm_format_result(
            &meta,
            "jazz",
            "[Verse]\nhello world\n",
            Some(30.0),
            None,
            "",
            "",
            "en",
            true,
            true,
        );
        assert_eq!(got.caption, "jazz trio");
        assert_eq!(got.lyrics, "[Verse]\nhello world");
        assert_eq!(got.bpm, Some(120));
    }

    #[test]
    fn parse_cot_reasoning_block() {
        let text = "<think>\nbpm: 96\ncaption: soft pop\n</think>\n[Verse]\nline\n<|audio_code_7|>";
        let meta = parse_lm_output(text);
        assert_eq!(meta.get("bpm").map(String::as_str), Some("96"));
        assert_eq!(meta.get("caption").map(String::as_str), Some("soft pop"));
        assert_eq!(meta.get("lyrics_tail").map(String::as_str), Some("[Verse]\nline"));
        assert_eq!(meta.get("audio_code_indices").map(String::as_str), Some("7"));
        let _ = DEFAULT_LM_REWRITE_INSTRUCTION;
    }
}
