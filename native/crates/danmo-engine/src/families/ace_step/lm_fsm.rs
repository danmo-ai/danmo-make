//! ACE-Step LM MetadataFSM scaffold (main `constrained_lm.py` / acestep.cpp
//! `metadata-fsm.h`).
//!
//! Full BPE token-set masking needs a tokenizer vocab map. This module ships
//! the state machine, metadata constants, and numeric early-block helpers so
//! phase-1 can constrain bpm/duration once digit token ids are known.

use crate::weights::WeightError;

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

/// Phase-1 metadata FSM (field order matches upstream CoT YAML).
#[derive(Debug, Clone)]
pub struct MetadataFsm {
    pub state: FsmState,
    pub value_buf: String,
    pub skip_caption: bool,
    pub skip_language: bool,
    pub user: UserMetas,
    pub max_duration: i32,
}

impl Default for MetadataFsm {
    fn default() -> Self {
        Self {
            state: FsmState::ThinkTag,
            value_buf: String::new(),
            skip_caption: false,
            skip_language: false,
            user: UserMetas::default(),
            max_duration: DURATION_MAX,
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

    /// Reset to the start of a CoT block.
    pub fn reset(&mut self) {
        self.state = FsmState::ThinkTag;
        self.value_buf.clear();
    }

    pub fn in_codes(&self) -> bool {
        matches!(self.state, FsmState::CodesGeneration | FsmState::Completed)
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
}
