//! ACE-Step LM MetadataFSM (main `constrained_lm.py` / acestep.cpp
//! `metadata-fsm.h`).
//!
//! Forces CoT field-name phrases (`bpm:`, `duration:`, …) token-by-token and
//! early-blocks bpm/duration digit continuations. Caption / keyscale /
//! language value trees are still free-form.

use crate::weights::WeightError;
use std::collections::HashMap;

pub const BPM_MIN: i32 = 30;
pub const BPM_MAX: i32 = 300;
pub const DURATION_MIN: i32 = 10;
pub const DURATION_MAX: i32 = 600;
pub const VALID_TIME_SIGNATURES: [i32; 4] = [2, 3, 4, 6];

pub const VALID_LANGUAGES: &[&str] = &[
    "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "fa", "fi",
    "fr", "he", "hi", "hr", "ht", "hu", "id", "is", "it", "ja", "ko", "la", "lt",
    "ms", "ne", "nl", "no", "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv",
    "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh", "unknown",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsmState {
    ThinkTag,
    NewlineAfterThink,
    BpmName,
    BpmValue,
    NewlineAfterBpm,
    CaptionName,
    CaptionValue,
    DurationName,
    DurationValue,
    NewlineAfterDuration,
    KeyscaleName,
    KeyscaleValue,
    NewlineAfterKeyscale,
    LanguageName,
    LanguageValue,
    TimesigName,
    TimesigValue,
    NewlineAfterTimesig,
    ThinkEndTag,
    CodesGeneration,
    Completed,
}

#[derive(Debug, Clone, Default)]
pub struct UserMetas {
    pub bpm: Option<i32>,
    pub caption: Option<String>,
    pub duration: Option<i32>,
    pub keyscale: Option<String>,
    pub language: Option<String>,
    pub timesignature: Option<String>,
}

/// One forced BPE/special token sequence (field name, think tag, newline).
#[derive(Debug, Clone)]
pub struct ForcedPhrase {
    pub tokens: Vec<i32>,
    pub pos: usize,
}

impl ForcedPhrase {
    pub fn from_ids(tokens: Vec<i32>) -> Result<Self, WeightError> {
        if tokens.is_empty() {
            return Err(WeightError::NotBuilt(
                "ACE-Step LM FSM forced phrase is empty".into(),
            ));
        }
        Ok(Self { tokens, pos: 0 })
    }

    pub fn from_text(
        tokenizer_json: &str,
        text: &str,
        vocab: i64,
    ) -> Result<Self, WeightError> {
        let tokens = crate::common::qwen3::tokenize_plain(tokenizer_json, text, vocab)?;
        Self::from_ids(tokens)
    }

    pub fn done(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    pub fn next_id(&self) -> Option<i32> {
        self.tokens.get(self.pos).copied()
    }

    pub fn mask_next(&self, logits: &mut [f32]) -> Result<(), WeightError> {
        let Some(id) = self.next_id() else {
            return Err(WeightError::NotBuilt(
                "ACE-Step LM FSM forced phrase is already complete".into(),
            ));
        };
        if id < 0 || id as usize >= logits.len() {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step LM FSM forced token {id} is outside vocab {}",
                logits.len()
            )));
        }
        for logit in logits.iter_mut() {
            *logit = f32::NEG_INFINITY;
        }
        logits[id as usize] = 0.0;
        Ok(())
    }

    /// Advance after sampling `id`. Returns true when the phrase completes.
    pub fn observe(&mut self, id: i32) -> Result<bool, WeightError> {
        let Some(expected) = self.next_id() else {
            return Err(WeightError::NotBuilt(
                "ACE-Step LM FSM forced phrase observe after complete".into(),
            ));
        };
        if id != expected {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step LM FSM expected token {expected}, got {id}"
            )));
        }
        self.pos += 1;
        Ok(self.done())
    }

    pub fn reset(&mut self) {
        self.pos = 0;
    }
}

/// Pre-tokenized fixed strings for NAME / think states.
#[derive(Debug, Clone)]
pub struct FsmLexicon {
    pub think: ForcedPhrase,
    pub newline: ForcedPhrase,
    pub bpm_name: ForcedPhrase,
    pub caption_name: ForcedPhrase,
    pub duration_name: ForcedPhrase,
    pub keyscale_name: ForcedPhrase,
    pub language_name: ForcedPhrase,
    pub timesig_name: ForcedPhrase,
    pub think_end: ForcedPhrase,
    pub digit_ids: [i32; 10],
    pub newline_id: i32,
}

impl FsmLexicon {
    pub fn from_tokenizer(tokenizer_json: &str, vocab: i64) -> Result<Self, WeightError> {
        let think_id = super::lm_prompt::TOKEN_THINK;
        let think_end_id = super::lm_prompt::TOKEN_THINK_END;
        if think_id as i64 >= vocab || think_end_id as i64 >= vocab {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step LM FSM think specials need vocab > {think_end_id}, got {vocab}"
            )));
        }
        let (digit_ids, newline_id) = resolve_digit_newline_ids(tokenizer_json, vocab)?;
        Ok(Self {
            think: ForcedPhrase::from_ids(vec![think_id])?,
            newline: ForcedPhrase::from_ids(vec![newline_id])?,
            bpm_name: ForcedPhrase::from_text(tokenizer_json, "bpm:", vocab)?,
            caption_name: ForcedPhrase::from_text(tokenizer_json, "caption:", vocab)?,
            duration_name: ForcedPhrase::from_text(tokenizer_json, "duration:", vocab)?,
            keyscale_name: ForcedPhrase::from_text(tokenizer_json, "keyscale:", vocab)?,
            language_name: ForcedPhrase::from_text(tokenizer_json, "language:", vocab)?,
            timesig_name: ForcedPhrase::from_text(tokenizer_json, "timesignature:", vocab)?,
            think_end: ForcedPhrase::from_ids(vec![think_end_id])?,
            digit_ids,
            newline_id,
        })
    }
}

/// Phase-1 metadata FSM (field order matches upstream CoT YAML).
#[derive(Debug, Clone)]
pub struct MetadataFsm {
    pub state: FsmState,
    pub value_buf: String,
    pub skip_caption: bool,
    pub skip_language: bool,
    pub skip_genres: bool,
    pub user: UserMetas,
    pub max_duration: i32,
    /// Active forced phrase for NAME / think states.
    pub phrase: Option<ForcedPhrase>,
}

impl Default for MetadataFsm {
    fn default() -> Self {
        Self {
            state: FsmState::ThinkTag,
            value_buf: String::new(),
            skip_caption: false,
            skip_language: false,
            skip_genres: true,
            user: UserMetas::default(),
            max_duration: DURATION_MAX,
            phrase: None,
        }
    }
}

impl MetadataFsm {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_user_metas(mut self, user: UserMetas) -> Self {
        self.user = user;
        self
    }

    /// Reset to the start of a CoT block and arm `<think>`.
    pub fn reset(&mut self, lex: &FsmLexicon) {
        self.state = FsmState::ThinkTag;
        self.value_buf.clear();
        self.phrase = Some(lex.think.clone());
    }

    pub fn in_codes(&self) -> bool {
        matches!(self.state, FsmState::CodesGeneration | FsmState::Completed)
    }

    fn next_field_after(&self, current: &str) -> FsmState {
        let order = [
            "bpm",
            "caption",
            "duration",
            "genres",
            "keyscale",
            "language",
            "timesignature",
        ];
        let map: HashMap<&str, FsmState> = HashMap::from([
            ("bpm", FsmState::BpmName),
            ("caption", FsmState::CaptionName),
            ("duration", FsmState::DurationName),
            ("keyscale", FsmState::KeyscaleName),
            ("language", FsmState::LanguageName),
            ("timesignature", FsmState::TimesigName),
        ]);
        let start = order.iter().position(|f| *f == current).map(|i| i + 1).unwrap_or(0);
        for field in &order[start..] {
            if *field == "genres" && self.skip_genres {
                continue;
            }
            if *field == "caption" && self.skip_caption {
                continue;
            }
            if *field == "language" && self.skip_language {
                continue;
            }
            if let Some(st) = map.get(field) {
                return *st;
            }
        }
        FsmState::ThinkEndTag
    }

    fn arm_phrase(&mut self, lex: &FsmLexicon) {
        let mut phrase = match self.state {
            FsmState::ThinkTag => Some(lex.think.clone()),
            FsmState::NewlineAfterThink
            | FsmState::NewlineAfterBpm
            | FsmState::NewlineAfterDuration
            | FsmState::NewlineAfterKeyscale
            | FsmState::NewlineAfterTimesig => Some(lex.newline.clone()),
            FsmState::BpmName => Some(lex.bpm_name.clone()),
            FsmState::CaptionName => Some(lex.caption_name.clone()),
            FsmState::DurationName => Some(lex.duration_name.clone()),
            FsmState::KeyscaleName => Some(lex.keyscale_name.clone()),
            FsmState::LanguageName => Some(lex.language_name.clone()),
            FsmState::TimesigName => Some(lex.timesig_name.clone()),
            FsmState::ThinkEndTag => Some(lex.think_end.clone()),
            _ => None,
        };
        if let Some(p) = phrase.as_mut() {
            p.reset();
        }
        self.phrase = phrase;
    }

    /// Mask logits for the current state (forced phrase or numeric value).
    pub fn mask_logits(
        &self,
        logits: &mut [f32],
        lex: &FsmLexicon,
        decoded_value: &str,
    ) -> Result<bool, WeightError> {
        if let Some(phrase) = &self.phrase {
            if !phrase.done() {
                phrase.mask_next(logits)?;
                return Ok(true);
            }
        }
        match self.state {
            FsmState::BpmValue => {
                Self::mask_numeric_value(
                    logits,
                    decoded_value,
                    BPM_MIN,
                    BPM_MAX,
                    &lex.digit_ids,
                    lex.newline_id,
                )?;
                Ok(true)
            }
            FsmState::DurationValue => {
                let max_v = self.max_duration.min(DURATION_MAX);
                Self::mask_numeric_value(
                    logits,
                    decoded_value,
                    DURATION_MIN,
                    max_v,
                    &lex.digit_ids,
                    lex.newline_id,
                )?;
                Ok(true)
            }
            FsmState::TimesigValue => {
                Self::mask_timesig_value(logits, decoded_value, &lex.digit_ids, lex.newline_id)?;
                Ok(true)
            }
            FsmState::CodesGeneration | FsmState::Completed => Ok(false),
            // Caption / keyscale / language values stay free-form for now.
            _ => Ok(false),
        }
    }

    /// Observe a sampled token and advance state / phrase progress.
    pub fn observe(
        &mut self,
        id: i32,
        lex: &FsmLexicon,
        token_text: &str,
    ) -> Result<(), WeightError> {
        if let Some(phrase) = self.phrase.as_mut() {
            if !phrase.done() {
                let done = phrase.observe(id)?;
                if done {
                    self.phrase = None;
                    self.advance_after_phrase(lex);
                }
                return Ok(());
            }
        }
        match self.state {
            FsmState::BpmValue | FsmState::DurationValue | FsmState::TimesigValue => {
                if id == lex.newline_id {
                    self.value_buf.clear();
                    let next = match self.state {
                        FsmState::BpmValue => self.next_field_after("bpm"),
                        FsmState::DurationValue => self.next_field_after("duration"),
                        FsmState::TimesigValue => FsmState::ThinkEndTag,
                        _ => unreachable!(),
                    };
                    self.state = next;
                    self.arm_phrase(lex);
                } else {
                    self.value_buf.push_str(token_text);
                }
            }
            FsmState::CaptionValue | FsmState::KeyscaleValue | FsmState::LanguageValue => {
                // Free-form until newline ends the field.
                if id == lex.newline_id || token_text.contains('\n') {
                    self.value_buf.clear();
                    let next = match self.state {
                        FsmState::CaptionValue => self.next_field_after("caption"),
                        FsmState::KeyscaleValue => self.next_field_after("keyscale"),
                        FsmState::LanguageValue => self.next_field_after("language"),
                        _ => unreachable!(),
                    };
                    self.state = next;
                    self.arm_phrase(lex);
                } else {
                    self.value_buf.push_str(token_text);
                }
            }
            FsmState::CodesGeneration => {
                self.state = FsmState::Completed;
            }
            _ => {}
        }
        Ok(())
    }

    fn advance_after_phrase(&mut self, lex: &FsmLexicon) {
        self.state = match self.state {
            FsmState::ThinkTag => FsmState::NewlineAfterThink,
            FsmState::NewlineAfterThink => FsmState::BpmName,
            FsmState::BpmName => FsmState::BpmValue,
            FsmState::CaptionName => FsmState::CaptionValue,
            FsmState::DurationName => FsmState::DurationValue,
            FsmState::KeyscaleName => FsmState::KeyscaleValue,
            FsmState::LanguageName => FsmState::LanguageValue,
            FsmState::TimesigName => FsmState::TimesigValue,
            FsmState::ThinkEndTag => FsmState::CodesGeneration,
            FsmState::NewlineAfterBpm => self.next_field_after("bpm"),
            FsmState::NewlineAfterDuration => self.next_field_after("duration"),
            FsmState::NewlineAfterKeyscale => self.next_field_after("keyscale"),
            FsmState::NewlineAfterTimesig => FsmState::ThinkEndTag,
            other => other,
        };
        self.value_buf.clear();
        self.arm_phrase(lex);
    }

    fn mask_timesig_value(
        logits: &mut [f32],
        so_far: &str,
        digit_ids: &[i32; 10],
        newline_id: i32,
    ) -> Result<(), WeightError> {
        for logit in logits.iter_mut() {
            *logit = f32::NEG_INFINITY;
        }
        if so_far.is_empty() {
            for &d in &VALID_TIME_SIGNATURES {
                let id = digit_ids[d as usize] as usize;
                logits[id] = 0.0;
            }
        } else if is_valid_timesignature(so_far) {
            logits[newline_id as usize] = 0.0;
        }
        Ok(())
    }

    /// Early-block digit continuations that cannot lead to a value in
    /// `[min_v, max_v]`. `digit_ids[d]` is the token id for decimal digit `d`.
    /// Allowed: newline (end value) and any digit that still has a valid
    /// extension. Returns whether at least one digit remained allowed.
    pub fn mask_numeric_value(
        logits: &mut [f32],
        so_far: &str,
        min_v: i32,
        max_v: i32,
        digit_ids: &[i32; 10],
        newline_id: i32,
    ) -> Result<bool, WeightError> {
        if min_v > max_v || min_v < 0 {
            return Err(WeightError::NotBuilt(
                "ACE-Step LM FSM numeric range is invalid".into(),
            ));
        }
        let v = logits.len() as i64;
        for &id in digit_ids.iter().chain(std::iter::once(&newline_id)) {
            if id < 0 || id as i64 >= v {
                return Err(WeightError::NotBuilt(format!(
                    "ACE-Step LM FSM token id {id} is outside vocab {v}"
                )));
            }
        }
        // Default: block everything, then re-enable valid digits + newline.
        for logit in logits.iter_mut() {
            *logit = f32::NEG_INFINITY;
        }
        // Empty prefix: prefer digits that can start a valid number.
        let mut any_digit = false;
        for d in 0i32..10 {
            let next = format!("{so_far}{d}");
            if numeric_prefix_viable(&next, min_v, max_v) {
                let id = digit_ids[d as usize] as usize;
                logits[id] = 0.0;
                any_digit = true;
            }
        }
        if !so_far.is_empty() {
            if let Ok(val) = so_far.parse::<i32>() {
                if val >= min_v && val <= max_v {
                    logits[newline_id as usize] = 0.0;
                }
            }
        }
        Ok(any_digit)
    }
}

/// Owns lexicon + FSM for one phase-1 generate call.
pub struct MetadataFsmSession {
    pub lex: FsmLexicon,
    pub fsm: MetadataFsm,
}

impl MetadataFsmSession {
    pub fn open(tokenizer_json: &str, vocab: i64) -> Result<Self, WeightError> {
        let lex = FsmLexicon::from_tokenizer(tokenizer_json, vocab)?;
        let mut fsm = MetadataFsm::new();
        fsm.reset(&lex);
        Ok(Self { lex, fsm })
    }

    pub fn mask(&self, logits: &mut [f32]) -> Result<bool, WeightError> {
        self.fsm
            .mask_logits(logits, &self.lex, &self.fsm.value_buf)
    }

    pub fn observe(&mut self, id: i32, token_text: &str) -> Result<(), WeightError> {
        self.fsm.observe(id, &self.lex, token_text)
    }
}

/// True if `prefix` is a decimal prefix of some integer in `[min_v, max_v]`.
pub fn numeric_prefix_viable(prefix: &str, min_v: i32, max_v: i32) -> bool {
    if prefix.is_empty() {
        return true;
    }
    if !prefix.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    // Reject leading zeros. Lone "0" cannot grow into bpm/duration ranges (>0).
    if prefix.starts_with('0') {
        return false;
    }
    let Ok(val) = prefix.parse::<i32>() else {
        return false;
    };
    // Still typing: allow if some extension lands in range.
    // Lower bound: val * 10^k for remaining digits must be able to reach min_v.
    // Upper bound: the number formed by padding with 9s must be >= min and
    // the number itself must not already exceed max_v.
    if val > max_v {
        return false;
    }
    let digits_left_max = {
        let max_digits = max_v.to_string().len();
        max_digits.saturating_sub(prefix.len())
    };
    // Maximum reachable by appending digits.
    let mut hi = val;
    for _ in 0..digits_left_max {
        hi = hi.saturating_mul(10).saturating_add(9);
    }
    // Minimum reachable by appending digits (append zeros).
    let mut lo = val;
    for _ in 0..digits_left_max {
        lo = lo.saturating_mul(10);
    }
    // Also consider stopping now if already in range (caller enables newline).
    hi >= min_v && lo <= max_v
}

pub fn is_valid_language(lang: &str) -> bool {
    VALID_LANGUAGES.iter().any(|l| *l == lang)
}

pub fn is_valid_timesignature(raw: &str) -> bool {
    raw.parse::<i32>()
        .ok()
        .is_some_and(|n| VALID_TIME_SIGNATURES.contains(&n))
}

/// Resolve single-digit and newline token ids via plain BPE encode.
pub fn resolve_digit_newline_ids(
    tokenizer_json: &str,
    vocab: i64,
) -> Result<([i32; 10], i32), WeightError> {
    let mut digits = [0i32; 10];
    for d in 0..10 {
        let ids = crate::common::qwen3::tokenize_plain(tokenizer_json, &d.to_string(), vocab)?;
        if ids.len() != 1 {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step LM FSM digit {d} did not encode to a single token (got {})",
                ids.len()
            )));
        }
        digits[d] = ids[0];
    }
    let nl = crate::common::qwen3::tokenize_plain(tokenizer_json, "\n", vocab)?;
    if nl.len() != 1 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step LM FSM newline did not encode to a single token (got {})",
            nl.len()
        )));
    }
    Ok((digits, nl[0]))
}

/// If `decoded` ends inside a bpm or duration value, mask logits to valid digits.
/// Returns true when a mask was applied.
pub fn maybe_mask_numeric_field(
    logits: &mut [f32],
    decoded: &str,
    digit_ids: &[i32; 10],
    newline_id: i32,
    max_duration: i32,
) -> Result<bool, WeightError> {
    if let Some(prefix) = open_numeric_field(decoded, "bpm: ") {
        MetadataFsm::mask_numeric_value(
            logits, prefix, BPM_MIN, BPM_MAX, digit_ids, newline_id,
        )?;
        return Ok(true);
    }
    if let Some(prefix) = open_numeric_field(decoded, "duration: ") {
        let max_v = max_duration.min(DURATION_MAX);
        MetadataFsm::mask_numeric_value(
            logits, prefix, DURATION_MIN, max_v, digit_ids, newline_id,
        )?;
        return Ok(true);
    }
    Ok(false)
}

fn open_numeric_field<'a>(decoded: &'a str, field: &str) -> Option<&'a str> {
    let idx = decoded.rfind(field)?;
    let after = &decoded[idx + field.len()..];
    if after.contains('\n') {
        return None;
    }
    if !after.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(after)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bpm_prefixes() {
        assert!(numeric_prefix_viable("", BPM_MIN, BPM_MAX));
        assert!(numeric_prefix_viable("1", BPM_MIN, BPM_MAX));
        assert!(numeric_prefix_viable("12", BPM_MIN, BPM_MAX));
        assert!(numeric_prefix_viable("120", BPM_MIN, BPM_MAX));
        assert!(!numeric_prefix_viable("301", BPM_MIN, BPM_MAX));
        assert!(!numeric_prefix_viable("0", BPM_MIN, BPM_MAX));
        assert!(!numeric_prefix_viable("02", BPM_MIN, BPM_MAX));
    }

    #[test]
    fn mask_bpm_allows_valid_digits_and_newline() {
        let mut logits = vec![1.0f32; 20];
        let digits = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
        let newline = 5;
        MetadataFsm::mask_numeric_value(&mut logits, "12", BPM_MIN, BPM_MAX, &digits, newline)
            .unwrap();
        // 12 < BPM_MIN so newline stays blocked; digit 0 → 120 is allowed.
        assert!(logits[newline as usize].is_infinite());
        assert_eq!(logits[digits[0] as usize], 0.0);
        MetadataFsm::mask_numeric_value(&mut logits, "120", BPM_MIN, BPM_MAX, &digits, newline)
            .unwrap();
        assert_eq!(logits[newline as usize], 0.0);
        assert!(logits[digits[1] as usize].is_infinite()); // 1201 > max
    }

    #[test]
    fn language_and_timesig_tables() {
        assert!(is_valid_language("en"));
        assert!(is_valid_language("zh"));
        assert!(!is_valid_language("xx"));
        assert!(is_valid_timesignature("4"));
        assert!(!is_valid_timesignature("5"));
    }

    #[test]
    fn open_numeric_detects_bpm_prefix() {
        assert_eq!(open_numeric_field("<think>\nbpm: 12", "bpm: "), Some("12"));
        assert_eq!(open_numeric_field("<think>\nbpm: 12\n", "bpm: "), None);
        assert_eq!(
            open_numeric_field("<think>\nbpm: 120\nduration: 3", "duration: "),
            Some("3")
        );
    }

    #[test]
    fn maybe_mask_applies_on_open_bpm() {
        let mut logits = vec![1.0f32; 20];
        let digits = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
        let applied =
            maybe_mask_numeric_field(&mut logits, "bpm: 12", &digits, 5, DURATION_MAX).unwrap();
        assert!(applied);
        assert!(logits[5].is_infinite()); // 12 < BPM_MIN → no newline
        assert_eq!(logits[10], 0.0); // digit 0 → 120
    }

    #[test]
    fn forced_phrase_masks_and_advances() {
        let mut phrase = ForcedPhrase::from_ids(vec![7, 8, 9]).unwrap();
        let mut logits = vec![1.0f32; 16];
        phrase.mask_next(&mut logits).unwrap();
        assert_eq!(logits[7], 0.0);
        assert!(logits[0].is_infinite());
        assert!(!phrase.observe(7).unwrap());
        phrase.mask_next(&mut logits).unwrap();
        assert_eq!(logits[8], 0.0);
        assert!(!phrase.observe(8).unwrap());
        assert!(phrase.observe(9).unwrap());
        assert!(phrase.done());
    }

    #[test]
    fn fsm_forces_think_then_bpm_name() {
        let newline_id = 5;
        let digit_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
        let lex = FsmLexicon {
            think: ForcedPhrase::from_ids(vec![100]).unwrap(),
            newline: ForcedPhrase::from_ids(vec![newline_id]).unwrap(),
            bpm_name: ForcedPhrase::from_ids(vec![20, 21]).unwrap(),
            caption_name: ForcedPhrase::from_ids(vec![30]).unwrap(),
            duration_name: ForcedPhrase::from_ids(vec![40]).unwrap(),
            keyscale_name: ForcedPhrase::from_ids(vec![50]).unwrap(),
            language_name: ForcedPhrase::from_ids(vec![60]).unwrap(),
            timesig_name: ForcedPhrase::from_ids(vec![70]).unwrap(),
            think_end: ForcedPhrase::from_ids(vec![80]).unwrap(),
            digit_ids,
            newline_id,
        };
        let mut fsm = MetadataFsm::new();
        fsm.skip_caption = true;
        fsm.skip_language = true;
        fsm.reset(&lex);
        let mut logits = vec![1.0f32; 128];
        assert!(fsm.mask_logits(&mut logits, &lex, "").unwrap());
        assert_eq!(logits[100], 0.0);
        fsm.observe(100, &lex, "").unwrap();
        assert_eq!(fsm.state, FsmState::NewlineAfterThink);
        fsm.observe(newline_id, &lex, "\n").unwrap();
        assert_eq!(fsm.state, FsmState::BpmName);
        fsm.mask_logits(&mut logits, &lex, "").unwrap();
        assert_eq!(logits[20], 0.0);
        fsm.observe(20, &lex, "bp").unwrap();
        fsm.observe(21, &lex, "m:").unwrap();
        assert_eq!(fsm.state, FsmState::BpmValue);
    }
}
