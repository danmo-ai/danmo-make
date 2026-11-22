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

    /// Raw vocab piece. Byte-level BPE (`Ġ`, `Ċ`, …) is mapped back to bytes so
    /// the FSM sees real spaces and newlines. Special tokens stay in the string;
    /// [`decode`] drops `<|...|>`.
    pub fn piece(&self, id: i32) -> String {
        self.id_to_token
            .get(&id)
            .map(|tok| {
                let mut bytes = Vec::new();
                append_token_bytes(tok, &mut bytes);
                String::from_utf8_lossy(&bytes).into_owned()
            })
            .unwrap_or_default()
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
            append_token_bytes(tok, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

fn append_token_bytes(tok: &str, bytes: &mut Vec<u8>) {
    if let Some(hex) = tok
        .strip_prefix("<0x")
        .and_then(|suffix| suffix.strip_suffix('>'))
    {
        if let Ok(byte) = u8::from_str_radix(hex, 16) {
            bytes.push(byte);
            return;
        }
    }
    for ch in tok.chars() {
        if ch == '▁' {
            bytes.push(b' ');
            continue;
        }
        if let Some(byte) = gpt2_byte(ch) {
            bytes.push(byte);
        } else {
            let mut buf = [0u8; 4];
            bytes.extend(ch.encode_utf8(&mut buf).as_bytes());
        }
    }
}

/// Inverse of the GPT-2 byte-to-unicode map. Printable bytes stay themselves;
/// the rest (`\n` → `Ċ`, space → `Ġ`) use the extra codepoints.
fn gpt2_byte(ch: char) -> Option<u8> {
    use std::sync::OnceLock;
    static MAP: OnceLock<[Option<u8>; 512]> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut table = [None; 512];
        let mut direct = Vec::with_capacity(256);
        direct.extend(b'!'..=b'~');
        direct.extend(0xA1..=0xAC);
        direct.extend(0xAE..=0xFF);
        for byte in direct.iter().copied() {
            table[byte as usize] = Some(byte);
        }
        let mut extra = 0usize;
        for byte in 0..=255u16 {
            if direct.contains(&(byte as u8)) {
                continue;
            }
            let point = 256 + extra;
            if point < table.len() {
                table[point] = Some(byte as u8);
            }
            extra += 1;
        }
        table
    });
    let point = ch as usize;
    map.get(point).copied().flatten()
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

    #[test]
    fn gpt2_maps_newline_and_space() {
        assert_eq!(gpt2_byte('Ċ'), Some(b'\n'));
        assert_eq!(gpt2_byte('Ġ'), Some(b' '));
        assert_eq!(gpt2_byte('A'), Some(b'A'));
    }
}
