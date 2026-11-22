//! Gemma 4 tokenizer and LTX dialogue normalization.
//!
//! The published tokenizer is a Hugging Face BPE with SentencePiece `▁`
//! normalization and byte fallback. The shared BPE owns merge ranking; this
//! module owns Gemma's normalization, left padding, and BOS policy.

use crate::common::bpe::encode_byte_fallback;
use crate::weights::WeightError;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenBatch {
    pub ids: Vec<i32>,
    pub attention_mask: Vec<i32>,
}

pub struct GemmaTokenizer {
    vocab: HashMap<String, i32>,
    ranks: HashMap<(String, String), usize>,
    specials: Vec<(String, i32)>,
    vocab_size: i64,
}

impl GemmaTokenizer {
    pub fn load(root: &Path) -> Result<Self, WeightError> {
        let path = root.join("tokenizer.json");
        let text = std::fs::read_to_string(&path).map_err(|error| WeightError::Open {
            path: path.display().to_string(),
            detail: error.to_string(),
        })?;
        let value: Value = serde_json::from_str(&text)
            .map_err(|error| WeightError::NotBuilt(format!("Gemma tokenizer.json: {error}")))?;
        let model = value
            .get("model")
            .and_then(Value::as_object)
            .ok_or_else(|| WeightError::NotBuilt("Gemma tokenizer has no model".into()))?;
        if model.get("type").and_then(Value::as_str) != Some("BPE")
            || model.get("byte_fallback").and_then(Value::as_bool) != Some(true)
        {
            return Err(WeightError::NotBuilt(
                "LTX 2.5 requires Gemma BPE with byte_fallback=true".into(),
            ));
        }
        let vocab_value = model
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or_else(|| WeightError::NotBuilt("Gemma tokenizer has no vocab".into()))?;
        let mut vocab = HashMap::with_capacity(vocab_value.len());
        for (token, id) in vocab_value {
            let id = id.as_i64().ok_or_else(|| {
                WeightError::NotBuilt(format!("Gemma token id for {token:?} is not an integer"))
            })?;
            if id < 0 || id > i32::MAX as i64 {
                return Err(WeightError::NotBuilt(format!(
                    "Gemma token id {id} for {token:?} is out of range"
                )));
            }
            vocab.insert(token.clone(), id as i32);
        }
        let vocab_size = vocab
            .values()
            .map(|id| i64::from(*id) + 1)
            .max()
            .ok_or_else(|| WeightError::NotBuilt("Gemma tokenizer vocab is empty".into()))?;
        let merges = model
            .get("merges")
            .and_then(Value::as_array)
            .ok_or_else(|| WeightError::NotBuilt("Gemma tokenizer has no merges".into()))?;
        let mut ranks = HashMap::with_capacity(merges.len());
        for (rank, merge) in merges.iter().enumerate() {
            let (left, right) = match merge {
                Value::String(pair) => {
                    let mut parts = pair.split(' ');
                    let left = parts.next().unwrap_or_default();
                    let right = parts.next().unwrap_or_default();
                    if parts.next().is_some() || left.is_empty() || right.is_empty() {
                        return Err(WeightError::NotBuilt(format!(
                            "Gemma tokenizer merge {pair:?} is not a pair"
                        )));
                    }
                    (left.to_string(), right.to_string())
                }
                Value::Array(pair) if pair.len() == 2 => {
                    let left = pair[0].as_str().ok_or_else(|| {
                        WeightError::NotBuilt("Gemma tokenizer merge left is not text".into())
                    })?;
                    let right = pair[1].as_str().ok_or_else(|| {
                        WeightError::NotBuilt("Gemma tokenizer merge right is not text".into())
                    })?;
                    (left.to_string(), right.to_string())
                }
                _ => {
                    return Err(WeightError::NotBuilt(format!(
                        "Gemma tokenizer merge at rank {rank} is not a pair"
                    )));
                }
            };
            if ranks.insert((left, right), rank).is_some() {
                return Err(WeightError::NotBuilt(format!(
                    "Gemma tokenizer repeats merge rank {rank}"
                )));
            }
        }
        let mut specials = Vec::new();
        if let Some(added) = value.get("added_tokens").and_then(Value::as_array) {
            for token in added {
                if !token
                    .get("special")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    continue;
                }
                let content = token
                    .get("content")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        WeightError::NotBuilt("Gemma special token has no content".into())
                    })?;
                let id = token.get("id").and_then(Value::as_i64).ok_or_else(|| {
                    WeightError::NotBuilt(format!("Gemma special token {content} has no id"))
                })?;
                let vocab_id = vocab.get(content).copied().ok_or_else(|| {
                    WeightError::NotBuilt(format!("Gemma special token {content} is not in vocab"))
                })?;
                if i64::from(vocab_id) != id {
                    return Err(WeightError::NotBuilt(format!(
                        "Gemma special token {content} has conflicting ids {id} and {vocab_id}"
                    )));
                }
                specials.push((content.to_string(), vocab_id));
            }
        }
        specials.sort_by_key(|(token, _)| std::cmp::Reverse(token.len()));
        Ok(Self {
            vocab,
            ranks,
            specials,
            vocab_size,
        })
    }

    pub fn encode(&self, prompt: &str, max_length: usize) -> Result<TokenBatch, WeightError> {
        if max_length < 2 {
            return Err(WeightError::NotBuilt(
                "Gemma max_length must leave room for BOS and one token".into(),
            ));
        }
        let normalized = normalize_dialogue(prompt.trim());
        let normalized = if normalized.starts_with(' ') {
            normalized
        } else {
            format!(" {normalized}")
        };
        let normalized = normalized.replace(' ', "▁");
        let mut ids = vec![2];
        ids.extend(encode_byte_fallback(
            &normalized,
            &self.vocab,
            &self.ranks,
            &self.specials,
            self.vocab_size,
        )?);
        ids.truncate(max_length);
        let valid = ids.len();
        let padding = max_length - valid;
        let mut padded_ids = vec![0; padding];
        padded_ids.extend(ids);
        let mut padded_mask = vec![0; padding];
        padded_mask.extend(vec![1; valid]);
        Ok(TokenBatch {
            ids: padded_ids,
            attention_mask: padded_mask,
        })
    }
}

/// Opening dialogue quotes carry a SentencePiece boundary. Closing quotes stay
/// attached to the spoken text; newlines inside a spoken span become spaces.
pub fn normalize_dialogue(text: &str) -> String {
    let mut output = String::with_capacity(text.len() + 8);
    let mut in_quote = false;
    for ch in text.chars() {
        let ch = match ch {
            '“' | '”' | '「' | '」' | '『' | '』' => '"',
            other => other,
        };
        if ch == '"' {
            if !in_quote {
                while output.chars().last().is_some_and(char::is_whitespace) {
                    output.pop();
                }
                output.push(' ');
                in_quote = true;
            } else {
                in_quote = false;
            }
            output.push('"');
            continue;
        }
        if in_quote && matches!(ch, '\r' | '\n') {
            if !output.ends_with(' ') {
                output.push(' ');
            }
            continue;
        }
        output.push(ch);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::normalize_dialogue;

    #[test]
    fn dialogue_quotes_keep_the_training_word_boundary() {
        assert_eq!(
            normalize_dialogue("他说：“你好\n世界”"),
            "他说： \"你好 世界\""
        );
        assert_eq!(normalize_dialogue("say \"hello\""), "say \"hello\"");
        assert_eq!(normalize_dialogue("say\n\"hello\""), "say \"hello\"");
    }
}
