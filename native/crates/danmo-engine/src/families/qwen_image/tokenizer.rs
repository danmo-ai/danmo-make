//! Qwen2 byte-level tokenizer for the image prompt template.
//!
//! Files are `tokenizer/vocab.json` and `tokenizer/merges.txt`, plus
//! `tokenizer/tokenizer_config.json` for `added_tokens_decoder`. The filled
//! template is tokenized as plain text. ChatML from the config is not applied.
//! The result is truncated to [`MAX_LENGTH`]. The 34-token prefix drop happens
//! after the text tower, on the hidden states.

use super::text::{MAX_LENGTH, PROMPT_TEMPLATE};
use crate::common::bpe::{self, byte_token};
use crate::weights::WeightError;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

pub fn prompt_ids(bundle: &Path, prompt: &str, vocab_size: i64) -> Result<Vec<i32>, WeightError> {
    let root = bundle.join("tokenizer");
    let vocab_path = root.join("vocab.json");
    let merges_path = root.join("merges.txt");
    if !vocab_path.is_file() || !merges_path.is_file() {
        return Err(WeightError::NotBuilt(
            "Qwen2.5-VL tokenizer is missing tokenizer/vocab.json or tokenizer/merges.txt".into(),
        ));
    }
    let vocab_text = std::fs::read_to_string(&vocab_path).map_err(|err| WeightError::Open {
        path: vocab_path.display().to_string(),
        detail: err.to_string(),
    })?;
    let vocab_json: Value = serde_json::from_str(&vocab_text)
        .map_err(|err| WeightError::NotBuilt(format!("tokenizer vocab.json is not json: {err}")))?;
    let vocab = vocab_map(&vocab_json)?;
    let merges_text = std::fs::read_to_string(&merges_path).map_err(|err| WeightError::Open {
        path: merges_path.display().to_string(),
        detail: err.to_string(),
    })?;
    let ranks = merge_ranks(&merges_text)?;
    let specials = added_specials(&root.join("tokenizer_config.json"), &vocab)?;
    for marker in ["<|im_start|>", "<|im_end|>"] {
        if !specials.iter().any(|(token, _)| token == marker) {
            return Err(WeightError::NotBuilt(format!(
                "Qwen2.5-VL tokenizer is missing {marker}"
            )));
        }
    }
    let filled = PROMPT_TEMPLATE.replacen("{}", prompt, 1);
    let mut ids = bpe::encode(&filled, &vocab, &ranks, &specials, vocab_size)?;
    if ids.len() > MAX_LENGTH {
        ids.truncate(MAX_LENGTH);
    }
    Ok(ids)
}

fn vocab_map(value: &Value) -> Result<HashMap<String, i32>, WeightError> {
    let Some(object) = value.as_object() else {
        return Err(WeightError::NotBuilt(
            "tokenizer vocab.json is not an object".into(),
        ));
    };
    let mut vocab = HashMap::with_capacity(object.len());
    for (token, id) in object {
        let Some(id) = id.as_i64() else {
            return Err(WeightError::NotBuilt(format!(
                "tokenizer id for {token} is not an integer"
            )));
        };
        if id < 0 || id > i32::MAX as i64 {
            return Err(WeightError::NotBuilt(format!(
                "tokenizer id {id} for {token} does not fit"
            )));
        }
        vocab.insert(token.clone(), id as i32);
    }
    Ok(vocab)
}

fn merge_ranks(text: &str) -> Result<HashMap<(String, String), usize>, WeightError> {
    let mut ranks = HashMap::new();
    for line in text.lines() {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split(' ');
        let Some(left) = parts.next() else {
            continue;
        };
        let Some(right) = parts.next() else {
            return Err(WeightError::NotBuilt(format!(
                "tokenizer merge {line} is not a pair"
            )));
        };
        if parts.next().is_some() || left.is_empty() || right.is_empty() {
            return Err(WeightError::NotBuilt(format!(
                "tokenizer merge {line} is not a pair"
            )));
        }
        let rank = ranks.len();
        if ranks
            .insert((left.to_string(), right.to_string()), rank)
            .is_some()
        {
            return Err(WeightError::NotBuilt(
                "tokenizer merge list repeats a pair".into(),
            ));
        }
    }
    Ok(ranks)
}

fn added_specials(
    path: &Path,
    vocab: &HashMap<String, i32>,
) -> Result<Vec<(String, i32)>, WeightError> {
    if !path.is_file() {
        return Err(WeightError::NotBuilt(
            "Qwen2.5-VL tokenizer is missing tokenizer/tokenizer_config.json".into(),
        ));
    }
    let text = std::fs::read_to_string(path).map_err(|err| WeightError::Open {
        path: path.display().to_string(),
        detail: err.to_string(),
    })?;
    let config: Value = serde_json::from_str(&text).map_err(|err| {
        WeightError::NotBuilt(format!("tokenizer_config.json is not json: {err}"))
    })?;
    let Some(added) = config
        .get("added_tokens_decoder")
        .and_then(|item| item.as_object())
    else {
        return Err(WeightError::NotBuilt(
            "tokenizer_config.json has no added_tokens_decoder".into(),
        ));
    };
    let mut specials = Vec::new();
    for (key, row) in added {
        let special = row
            .get("special")
            .and_then(|item| item.as_bool())
            .unwrap_or(true);
        if !special {
            continue;
        }
        let Some(content) = row.get("content").and_then(|item| item.as_str()) else {
            return Err(WeightError::NotBuilt(
                "tokenizer added token has no content".into(),
            ));
        };
        let id = key.parse::<i64>().map_err(|_| {
            WeightError::NotBuilt(format!("tokenizer added token id {key} is not an integer"))
        })?;
        let Some(vocab_id) = vocab.get(content).copied() else {
            return Err(WeightError::NotBuilt(format!(
                "tokenizer has no vocab id for {content}"
            )));
        };
        if i64::from(vocab_id) != id {
            return Err(WeightError::NotBuilt(format!(
                "tokenizer id for {content} is {vocab_id} in vocab and {id} in added_tokens_decoder"
            )));
        }
        specials.push((content.to_string(), vocab_id));
    }
    specials.sort_by_key(|(token, _)| std::cmp::Reverse(token.len()));
    Ok(specials)
}

/// Byte-level vocab for tests. Ids are the raw bytes. Specials sit above 255.
fn test_vocab() -> (String, String, String) {
    let mut vocab = String::from("{");
    for byte in 0u16..256 {
        if byte > 0 {
            vocab.push(',');
        }
        let token = byte_token(byte as u8).to_string();
        vocab.push_str(&serde_json::to_string(&token).unwrap());
        vocab.push(':');
        vocab.push_str(&byte.to_string());
    }
    vocab.push_str(",\"<|im_start|>\":256,\"<|im_end|>\":257}");
    let config = r#"{
  "added_tokens_decoder": {
    "256": {"content": "<|im_start|>", "special": true},
    "257": {"content": "<|im_end|>", "special": true}
  }
}"#;
    (vocab, "#version: 0.2\n".to_string(), config.to_string())
}

#[cfg(test)]
mod tests {
    use super::super::text::PROMPT_TEMPLATE;
    use super::prompt_ids;
    use crate::common::bpe::byte_token;
    use std::path::Path;

    fn write_tok(dir: &Path, merges: &str) {
        let tok = dir.join("tokenizer");
        std::fs::create_dir_all(&tok).unwrap();
        let (vocab, _header, config) = super::test_vocab();
        std::fs::write(tok.join("vocab.json"), vocab).unwrap();
        std::fs::write(tok.join("merges.txt"), merges).unwrap();
        std::fs::write(tok.join("tokenizer_config.json"), config).unwrap();
    }

    #[test]
    fn specials_stay_whole_and_a_merge_joins_the_prompt() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-qwen-tok-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_tok(&dir, "#version: 0.2\n");
        let plain = prompt_ids(&dir, "hi", 258).unwrap();
        assert!(plain.contains(&256), "{plain:?}");
        assert!(plain.contains(&257), "{plain:?}");
        assert!(plain
            .windows(2)
            .all(|pair| pair != [i32::from(b'<'), i32::from(b'|')]));
        let pair = format!("{} {}", byte_token(b'h'), byte_token(b'i'));
        let merged = format!("{}{}", byte_token(b'h'), byte_token(b'i'));
        let tok = dir.join("tokenizer");
        let mut vocab: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(tok.join("vocab.json")).unwrap())
                .unwrap();
        vocab[&merged] = serde_json::json!(300);
        std::fs::write(tok.join("vocab.json"), vocab.to_string()).unwrap();
        std::fs::write(tok.join("merges.txt"), format!("#version: 0.2\n{pair}\n")).unwrap();
        let err = prompt_ids(&dir, "hi", 258).unwrap_err();
        assert!(err.to_string().contains("outside vocab"), "{err}");
        let merged_ids = prompt_ids(&dir, "hi", 301).unwrap();
        assert_eq!(merged_ids.iter().filter(|id| **id == 300).count(), 2);
        assert_eq!(merged_ids.len() + 2, plain.len());
        let truncated = prompt_ids(&dir, &"a".repeat(2000), 301).unwrap();
        assert_eq!(truncated.len(), super::super::text::MAX_LENGTH);
        assert!(PROMPT_TEMPLATE.contains("{}"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
