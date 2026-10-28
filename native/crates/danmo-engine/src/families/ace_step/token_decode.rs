//! Reverse BPE decode for Qwen lyric tokens (needed by CA+DTW line grouping).

use crate::weights::WeightError;
use serde_json::Value;
use std::collections::HashMap;

/// Qwen `<|endoftext|>` id (shared with lyric prompts).
pub const QWEN_ENDOFTEXT_ID: i32 = 151_643;

pub const LYRIC_HEADER_TEMPLATE: &str = "# Languages\n{language}\n\n# Lyric\n";

pub struct Decoder {
    id_to_token: HashMap<i32, String>,
}

impl Decoder {
    pub fn from_tokenizer_json(tokenizer_json: &str) -> Result<Self, WeightError> {
        let value: Value = serde_json::from_str(tokenizer_json)
            .map_err(|err| WeightError::NotBuilt(format!("tokenizer.json is not json: {err}")))?;
        let mut id_to_token = HashMap::new();
        if let Some(vocab) = value
            .pointer("/model/vocab")
            .and_then(|v| v.as_object())
            .or_else(|| value.get("vocab").and_then(|v| v.as_object()))
        {
            for (tok, id) in vocab {
                if let Some(id) = id.as_i64().or_else(|| id.as_u64().map(|u| u as i64)) {
                    id_to_token.insert(id as i32, tok.clone());
                }
            }
        }
        if let Some(added) = value.get("added_tokens").and_then(|v| v.as_array()) {
            for item in added {
                let id = item
                    .get("id")
                    .and_then(|v| v.as_i64().or_else(|| v.as_u64().map(|u| u as i64)));
                let content = item.get("content").and_then(|v| v.as_str());
                if let (Some(id), Some(content)) = (id, content) {
                    id_to_token.insert(id as i32, content.to_string());
                }
            }
        }
        if id_to_token.is_empty() {
            return Err(WeightError::NotBuilt(
                "ACE-Step tokenizer has no vocab for lyric decode".into(),
            ));
        }
        Ok(Self { id_to_token })
    }

    pub fn decode(&self, ids: &[i32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(tok) = self.id_to_token.get(&id) else {
                continue;
            };
            if tok.starts_with("<|") && tok.ends_with("|>") {
                continue;
            }
            if let Some(hex) = tok.strip_prefix("<0x").and_then(|s| s.strip_suffix('>')) {
                if let Ok(b) = u8::from_str_radix(hex, 16) {
                    bytes.push(b);
                }
                continue;
            }
            for ch in tok.chars() {
                if ch == 'Ġ' || ch == '▁' {
                    bytes.push(b' ');
                } else {
                    let mut buf = [0u8; 4];
                    for b in ch.encode_utf8(&mut buf).as_bytes() {
                        bytes.push(*b);
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Pure lyric token span inside the lyric prompt (header stripped, endoftext dropped).
pub fn pure_lyric_range(
    lyric_token_ids: &[i32],
    header_token_count: usize,
) -> Option<(usize, usize)> {
    if lyric_token_ids.is_empty() {
        return None;
    }
    let start = header_token_count.min(lyric_token_ids.len());
    let end = lyric_token_ids
        .iter()
        .position(|&id| id == QWEN_ENDOFTEXT_ID)
        .unwrap_or(lyric_token_ids.len());
    if end <= start {
        None
    } else {
        Some((start, end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_range_strips_header_and_eot() {
        let ids = [1, 2, 3, 4, QWEN_ENDOFTEXT_ID, 9];
        assert_eq!(pure_lyric_range(&ids, 2), Some((2, 4)));
        assert!(pure_lyric_range(&ids, 4).is_none());
    }
}
