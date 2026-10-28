//! ACE-Step 5Hz LM ChatML prompt builders (acestep.cpp `prompt.h` + `task-types.h`).
//!
//! Builds token-id sequences for inspire / format / generate / phase-2-with-CoT.
//! Special tokens are injected as raw IDs; text spans use plain BPE encode.

use crate::weights::WeightError;

/// `<|im_start|>`
pub const TOKEN_IM_START: i32 = 151_644;
/// `<|im_end|>`
pub const TOKEN_IM_END: i32 = 151_645;
/// `<think>`
pub const TOKEN_THINK: i32 = 151_667;
/// `</think>`
pub const TOKEN_THINK_END: i32 = 151_668;
/// First `<|audio_code_0|>` id; codes are `AUDIO_CODE_BASE + index` (0..=65535).
pub const AUDIO_CODE_BASE: i32 = 151_669;
pub const AUDIO_CODE_COUNT: i32 = 65_535;

/// acestep.cpp `LM_INSTRUCTION` (generate / codes).
pub const LM_INSTRUCTION: &str =
    "Generate audio semantic tokens based on the given conditions:";
/// acestep.cpp `LM_INSPIRE_INSTRUCTION`.
pub const LM_INSPIRE_INSTRUCTION: &str =
    "Expand the user's input into a more detailed and specific musical description:";
/// acestep.cpp `LM_FORMAT_INSTRUCTION`.
pub const LM_FORMAT_INSTRUCTION: &str =
    "Format the user's input into a more detailed and specific musical description:";
/// acestep.cpp `LM_UNDERSTAND_INSTRUCTION`.
pub const LM_UNDERSTAND_INSTRUCTION: &str =
    "Understand the given musical conditions and describe the audio semantics accordingly:";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LmMode {
    /// Full: metadata + lyrics + audio codes.
    Generate,
    /// Short query → metadata + lyrics (no codes).
    Inspire,
    /// Caption + lyrics → refined metadata + lyrics (no codes).
    Format,
}

impl LmMode {
    pub fn parse(raw: &str) -> Result<Self, WeightError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "generate" => Ok(Self::Generate),
            "inspire" => Ok(Self::Inspire),
            "format" => Ok(Self::Format),
            other => Err(WeightError::NotBuilt(format!(
                "ACE-Step LM mode '{other}' is unknown (use generate, inspire, or format)"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Generate => "generate",
            Self::Inspire => "inspire",
            Self::Format => "format",
        }
    }

    pub fn skips_audio_codes(self) -> bool {
        matches!(self, Self::Inspire | Self::Format)
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AcePrompt {
    pub caption: String,
    pub lyrics: String,
    pub duration: f32,
    pub bpm: i32,
    pub keyscale: String,
    pub timesignature: String,
    pub vocal_language: String,
}

type EncodeFn<'a> = dyn FnMut(&str) -> Result<Vec<i32>, WeightError> + 'a;

fn append_text(ids: &mut Vec<i32>, encode: &mut EncodeFn<'_>, text: &str) -> Result<(), WeightError> {
    if text.is_empty() {
        return Ok(());
    }
    ids.extend(encode(text)?);
    Ok(())
}

/// Qwen3 chat: system / user / assistant (open).
pub fn build_custom_prompt(
    encode: &mut EncodeFn<'_>,
    system: &str,
    user: &str,
) -> Result<Vec<i32>, WeightError> {
    let mut ids = Vec::new();
    ids.push(TOKEN_IM_START);
    append_text(&mut ids, encode, &format!("system\n{system}\n"))?;
    ids.push(TOKEN_IM_END);
    append_text(&mut ids, encode, "\n")?;
    ids.push(TOKEN_IM_START);
    append_text(&mut ids, encode, &format!("user\n{user}\n"))?;
    ids.push(TOKEN_IM_END);
    append_text(&mut ids, encode, "\n")?;
    ids.push(TOKEN_IM_START);
    append_text(&mut ids, encode, "assistant\n")?;
    Ok(ids)
}

/// Generate-mode prompt (no CoT injected yet).
pub fn build_lm_prompt(
    encode: &mut EncodeFn<'_>,
    prompt: &AcePrompt,
) -> Result<Vec<i32>, WeightError> {
    let mut ids = Vec::new();
    ids.push(TOKEN_IM_START);
    append_text(
        &mut ids,
        encode,
        &format!("system\n# Instruction\n{LM_INSTRUCTION}\n\n"),
    )?;
    ids.push(TOKEN_IM_END);
    append_text(&mut ids, encode, "\n")?;
    ids.push(TOKEN_IM_START);
    append_text(
        &mut ids,
        encode,
        &format!(
            "user\n# Caption\n{}\n\n# Lyric\n{}\n",
            prompt.caption, prompt.lyrics
        ),
    )?;
    ids.push(TOKEN_IM_END);
    append_text(&mut ids, encode, "\n")?;
    ids.push(TOKEN_IM_START);
    append_text(&mut ids, encode, "assistant\n")?;
    Ok(ids)
}

/// Inspire-mode prompt (short query → expand).
pub fn build_inspire_prompt(
    encode: &mut EncodeFn<'_>,
    prompt: &AcePrompt,
) -> Result<Vec<i32>, WeightError> {
    let sys = format!("# Instruction\n{LM_INSPIRE_INSTRUCTION}\n");
    let mut user = prompt.caption.clone();
    if prompt.lyrics.trim() == "[Instrumental]" {
        user.push_str("\n\ninstrumental: true");
    } else {
        user.push_str("\n\ninstrumental: false");
    }
    if !prompt.vocal_language.is_empty() && prompt.vocal_language != "unknown" {
        user.push_str(&format!("\nlanguage: {}", prompt.vocal_language));
    }
    build_custom_prompt(encode, &sys, &user)
}

/// Format-mode prompt (caption + lyrics → rewrite).
pub fn build_format_prompt(
    encode: &mut EncodeFn<'_>,
    prompt: &AcePrompt,
) -> Result<Vec<i32>, WeightError> {
    let sys = format!("# Instruction\n{LM_FORMAT_INSTRUCTION}\n");
    let user = format!(
        "# Caption\n{}\n\n# Lyric\n{}",
        prompt.caption, prompt.lyrics
    );
    build_custom_prompt(encode, &sys, &user)
}

/// Phase-2 prompt with forced CoT YAML already in the assistant turn.
pub fn build_lm_prompt_with_cot(
    encode: &mut EncodeFn<'_>,
    prompt: &AcePrompt,
    cot_yaml: &str,
) -> Result<Vec<i32>, WeightError> {
    let mut ids = build_lm_prompt(encode, prompt)?;
    ids.push(TOKEN_THINK);
    append_text(&mut ids, encode, &format!("\n{cot_yaml}"))?;
    ids.push(TOKEN_THINK_END);
    append_text(&mut ids, encode, "\n\n")?;
    Ok(ids)
}

/// CoT YAML matching Python `yaml.dump(sort_keys=True)` field order used upstream.
pub fn build_cot_yaml(prompt: &AcePrompt) -> String {
    let mut yaml = String::new();
    if prompt.bpm > 0 {
        yaml.push_str(&format!("bpm: {}\n", prompt.bpm));
    }
    if !prompt.caption.is_empty() {
        yaml.push_str(&yaml_wrap("caption", &prompt.caption));
    }
    if prompt.duration > 0.0 {
        yaml.push_str(&format!("duration: {}\n", prompt.duration as i32));
    }
    if !prompt.keyscale.is_empty() {
        yaml.push_str(&format!("keyscale: {}\n", prompt.keyscale));
    }
    if !prompt.vocal_language.is_empty() {
        yaml.push_str(&format!("language: {}\n", prompt.vocal_language));
    }
    if !prompt.timesignature.is_empty() {
        yaml.push_str(&format!("timesignature: {}\n", prompt.timesignature));
    }
    yaml
}

fn yaml_wrap(key: &str, val: &str) -> String {
    let mut result = format!("{key}:");
    let mut col = key.len() + 1;
    for word in val.split(' ') {
        if word.is_empty() {
            continue;
        }
        if col > 80 {
            result.push_str("\n  ");
            col = 2;
        } else {
            result.push(' ');
            col += 1;
        }
        result.push_str(word);
        col += word.len();
    }
    result.push('\n');
    result
}

/// Select phase-1 prompt tokens for the given LM mode (acestep.cpp `ace_lm_generate`).
pub fn build_phase1_prompt(
    encode: &mut EncodeFn<'_>,
    mode: LmMode,
    prompt: &AcePrompt,
) -> Result<Vec<i32>, WeightError> {
    match mode {
        LmMode::Inspire => build_inspire_prompt(encode, prompt),
        LmMode::Format => build_format_prompt(encode, prompt),
        LmMode::Generate => {
            // Generate with empty lyrics uses inspire instruction (pipeline-lm.cpp).
            if prompt.lyrics.trim().is_empty() {
                build_inspire_prompt(encode, prompt)
            } else {
                build_lm_prompt(encode, prompt)
            }
        }
    }
}

/// Map FSQ codebook indices to LM audio-code token ids.
pub fn audio_code_token_ids(codes: &[i32]) -> Result<Vec<i32>, WeightError> {
    let mut out = Vec::with_capacity(codes.len());
    for &c in codes {
        if c < 0 || c > AUDIO_CODE_COUNT {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step audio code {c} is outside 0..={AUDIO_CODE_COUNT}"
            )));
        }
        out.push(AUDIO_CODE_BASE + c);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_encode(text: &str) -> Result<Vec<i32>, WeightError> {
        // One synthetic id per UTF-8 byte so structure is visible in asserts.
        Ok(text.bytes().map(|b| b as i32).collect())
    }

    #[test]
    fn mode_parse_and_skips_codes() {
        assert_eq!(LmMode::parse("inspire").unwrap(), LmMode::Inspire);
        assert!(LmMode::Inspire.skips_audio_codes());
        assert!(!LmMode::Generate.skips_audio_codes());
        assert!(LmMode::parse("codes").is_err());
    }

    #[test]
    fn custom_prompt_chatml_frame() {
        let mut enc = |t: &str| fake_encode(t);
        let ids = build_custom_prompt(&mut enc, "sys", "usr").unwrap();
        assert_eq!(ids[0], TOKEN_IM_START);
        assert!(ids.contains(&TOKEN_IM_END));
        // Three im_start turns: system, user, assistant
        assert_eq!(ids.iter().filter(|&&t| t == TOKEN_IM_START).count(), 3);
    }

    #[test]
    fn inspire_and_format_include_instructions() {
        let prompt = AcePrompt {
            caption: "lofi".into(),
            lyrics: "[Instrumental]".into(),
            vocal_language: "en".into(),
            ..AcePrompt::default()
        };
        let mut enc = |t: &str| fake_encode(t);
        let ids = build_inspire_prompt(&mut enc, &prompt).unwrap();
        let flat: Vec<u8> = ids
            .iter()
            .filter(|&&t| t < 256)
            .map(|&t| t as u8)
            .collect();
        let s = String::from_utf8_lossy(&flat);
        assert!(s.contains("Expand the user's input"));
        assert!(s.contains("instrumental: true"));
        assert!(s.contains("language: en"));

        let mut enc = |t: &str| fake_encode(t);
        let ids = build_format_prompt(&mut enc, &prompt).unwrap();
        let flat: Vec<u8> = ids
            .iter()
            .filter(|&&t| t < 256)
            .map(|&t| t as u8)
            .collect();
        let s = String::from_utf8_lossy(&flat);
        assert!(s.contains("Format the user's input"));
        assert!(s.contains("# Lyric"));
    }

    #[test]
    fn cot_yaml_and_phase2_prompt() {
        let prompt = AcePrompt {
            caption: "soft pop ballad with piano".into(),
            lyrics: "[Verse]\nhi".into(),
            duration: 30.0,
            bpm: 96,
            keyscale: "C major".into(),
            timesignature: "4/4".into(),
            vocal_language: "en".into(),
        };
        let yaml = build_cot_yaml(&prompt);
        assert!(yaml.starts_with("bpm: 96\n"));
        assert!(yaml.contains("caption: soft pop"));
        assert!(yaml.contains("duration: 30\n"));
        let mut enc = |t: &str| fake_encode(t);
        let ids = build_lm_prompt_with_cot(&mut enc, &prompt, &yaml).unwrap();
        assert!(ids.contains(&TOKEN_THINK));
        assert!(ids.contains(&TOKEN_THINK_END));
        let think_at = ids.iter().position(|&t| t == TOKEN_THINK).unwrap();
        let end_at = ids.iter().position(|&t| t == TOKEN_THINK_END).unwrap();
        assert!(think_at < end_at);
    }

    #[test]
    fn audio_code_tokens_offset_from_base() {
        assert_eq!(
            audio_code_token_ids(&[0, 12, 65535]).unwrap(),
            vec![AUDIO_CODE_BASE, AUDIO_CODE_BASE + 12, AUDIO_CODE_BASE + 65535]
        );
        assert!(audio_code_token_ids(&[-1]).is_err());
        assert!(audio_code_token_ids(&[65536]).is_err());
    }
}
