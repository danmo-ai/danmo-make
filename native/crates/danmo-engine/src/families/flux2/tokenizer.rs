//! Qwen3 byte-level BPE. This is the tokenizer `main` loads with
//! `AutoTokenizer` for Z-Image and Flux2 Klein.
//!
//! The chat string is the single user turn those encoders pass to
//! `apply_chat_template(..., add_generation_prompt=True)`. It is not a Jinja
//! interpreter. A template that is not ChatML fails. Z-Image leaves thinking
//! on. Flux2 passes `enable_thinking=False`, which appends the empty think
//! block only when the template mentions that flag.

use crate::weights::WeightError;
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;

const MAX_TEXT_TOKENS: usize = 512;
const QWEN_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

static PIECE_CONTRACTION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?i:'s|'t|'re|'ve|'m|'ll|'d)").unwrap());
static PIECE_LETTERS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[^\r\n\p{L}\p{N}]?\p{L}+").unwrap());
static PIECE_NUMBER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\p{N}").unwrap());
static PIECE_PUNCT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^ ?[^\s\p{L}\p{N}]+[\r\n]*").unwrap());
static PIECE_NEWLINE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*[\r\n]+").unwrap());
static PIECE_SPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s+").unwrap());

pub(crate) fn load_tokenizer(bundle: &Path) -> Result<String, WeightError> {
    let path = if bundle.join("tokenizer.json").is_file() {
        bundle.join("tokenizer.json")
    } else {
        bundle.join("tokenizer").join("tokenizer.json")
    };
    let json = std::fs::read_to_string(&path).map_err(|err| WeightError::Open {
        path: path.display().to_string(),
        detail: err.to_string(),
    })?;
    let config_path = path.with_file_name("tokenizer_config.json");
    if !config_path.is_file() {
        return Ok(json);
    }
    let config = std::fs::read_to_string(&config_path).map_err(|err| WeightError::Open {
        path: config_path.display().to_string(),
        detail: err.to_string(),
    })?;
    merge_chat_template(&json, &config)
}

pub(super) fn tokenize(
    json: &str,
    prompt: &str,
    vocab_size: i64,
    thinking: bool,
) -> Result<Vec<i32>, WeightError> {
    let value: Value = serde_json::from_str(json)
        .map_err(|err| WeightError::NotBuilt(format!("tokenizer.json is not json: {err}")))?;
    if value.get("encoding").and_then(|item| item.as_str()) == Some("bytes") {
        return tokenize_bytes(prompt, vocab_size);
    }
    if value.get("model").is_some() {
        return tokenize_bpe(&value, prompt, vocab_size, thinking);
    }
    let Some(vocab) = value.get("vocab").and_then(|item| item.as_object()) else {
        return Err(WeightError::NotBuilt("tokenizer has no vocab".into()));
    };
    let words: Vec<&str> = prompt
        .split_whitespace()
        .filter(|word| !word.is_empty())
        .collect();
    if words.is_empty() {
        return Err(WeightError::NotBuilt("prompt has no tokens".into()));
    }
    let mut ids = Vec::with_capacity(words.len());
    for word in words {
        let Some(id) = vocab.get(word).and_then(|item| item.as_i64()) else {
            return Err(WeightError::NotBuilt(format!(
                "tokenizer has no id for {word}"
            )));
        };
        push_id(&mut ids, id, vocab_size, word)?;
    }
    Ok(ids)
}

fn merge_chat_template(tokenizer_json: &str, config_json: &str) -> Result<String, WeightError> {
    let mut tokenizer: Value = serde_json::from_str(tokenizer_json)
        .map_err(|err| WeightError::NotBuilt(format!("tokenizer.json is not json: {err}")))?;
    let config: Value = serde_json::from_str(config_json).map_err(|err| {
        WeightError::NotBuilt(format!("tokenizer_config.json is not json: {err}"))
    })?;
    let Some(template) = config.get("chat_template") else {
        return Ok(tokenizer_json.to_string());
    };
    if template.is_null() {
        return Ok(tokenizer_json.to_string());
    }
    let Some(object) = tokenizer.as_object_mut() else {
        return Err(WeightError::NotBuilt(
            "tokenizer.json is not an object".into(),
        ));
    };
    object.insert("chat_template".into(), template.clone());
    Ok(tokenizer.to_string())
}

fn tokenize_bpe(
    value: &Value,
    prompt: &str,
    vocab_size: i64,
    thinking: bool,
) -> Result<Vec<i32>, WeightError> {
    if prompt.is_empty() {
        return Err(WeightError::NotBuilt("prompt has no tokens".into()));
    }
    let model = value
        .get("model")
        .ok_or_else(|| WeightError::NotBuilt("tokenizer model is missing".into()))?;
    let model_type = model
        .get("type")
        .and_then(|item| item.as_str())
        .unwrap_or("");
    if model_type != "BPE" {
        return Err(WeightError::NotBuilt(format!(
            "tokenizer model {model_type} is not built"
        )));
    }
    if value.get("normalizer").is_some_and(|item| !item.is_null()) {
        return Err(WeightError::NotBuilt(
            "tokenizer normalizer is not built".into(),
        ));
    }
    if model.get("byte_fallback").and_then(|item| item.as_bool()) == Some(true) {
        return Err(WeightError::NotBuilt(
            "tokenizer byte fallback is not built".into(),
        ));
    }
    if model.get("ignore_merges").and_then(|item| item.as_bool()) == Some(true) {
        return Err(WeightError::NotBuilt(
            "tokenizer ignore_merges is not built".into(),
        ));
    }
    assert_qwen_pretokenizer(value.get("pre_tokenizer"))?;
    let vocab = vocab_map(model)?;
    let ranks = merge_ranks(model)?;
    let specials = special_tokens(&vocab, value.get("added_tokens"))?;
    let template = chat_template(value)?;
    let text = match template {
        Some(template) => chat_text(prompt, thinking, template.contains("enable_thinking")),
        None => prompt.to_string(),
    };
    let mut ids = Vec::new();
    for piece in split_specials(&text, &specials) {
        match piece {
            Piece::Special(token) => {
                let id = specials
                    .iter()
                    .find(|(name, _)| name == token)
                    .map(|(_, id)| i64::from(*id))
                    .ok_or_else(|| {
                        WeightError::NotBuilt(format!("tokenizer has no id for {token}"))
                    })?;
                push_id(&mut ids, id, vocab_size, token)?;
            }
            Piece::Plain(plain) => {
                if plain.is_empty() {
                    continue;
                }
                for word in pretokenize(plain)? {
                    let symbols = byte_symbols(&word);
                    for token in bpe(symbols, &ranks) {
                        let Some(id) = vocab.get(&token).copied() else {
                            return Err(WeightError::NotBuilt(format!(
                                "tokenizer has no id for {token}"
                            )));
                        };
                        push_id(&mut ids, i64::from(id), vocab_size, &token)?;
                    }
                }
            }
        }
    }
    if ids.is_empty() {
        return Err(WeightError::NotBuilt("prompt has no tokens".into()));
    }
    if ids.len() > MAX_TEXT_TOKENS {
        return Err(WeightError::NotBuilt(format!(
            "prompt tokenized to {} tokens, the text encoder accepts {MAX_TEXT_TOKENS}",
            ids.len()
        )));
    }
    Ok(ids)
}

fn chat_template(value: &Value) -> Result<Option<&str>, WeightError> {
    let Some(template) = value.get("chat_template") else {
        return Ok(None);
    };
    if template.is_null() {
        return Ok(None);
    }
    let Some(template) = template.as_str() else {
        return Err(WeightError::NotBuilt("chat template is not built".into()));
    };
    if !template.contains("<|im_start|>") || !template.contains("<|im_end|>") {
        return Err(WeightError::NotBuilt("chat template is not built".into()));
    }
    Ok(Some(template))
}

fn chat_text(prompt: &str, thinking: bool, template_has_thinking: bool) -> String {
    let mut text = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    if !thinking && template_has_thinking {
        text.push_str("<think>\n\n</think>\n\n");
    }
    text
}

fn assert_qwen_pretokenizer(pre: Option<&Value>) -> Result<(), WeightError> {
    let Some(pre) = pre else {
        return Err(WeightError::NotBuilt(
            "tokenizer pre-tokenizer is not the Qwen byte-level split".into(),
        ));
    };
    if pre.get("type").and_then(|item| item.as_str()) != Some("Sequence") {
        return Err(WeightError::NotBuilt(
            "tokenizer pre-tokenizer is not the Qwen byte-level split".into(),
        ));
    }
    let Some(steps) = pre.get("pretokenizers").and_then(|item| item.as_array()) else {
        return Err(WeightError::NotBuilt(
            "tokenizer pre-tokenizer is not the Qwen byte-level split".into(),
        ));
    };
    if steps.len() != 2 {
        return Err(WeightError::NotBuilt(
            "tokenizer pre-tokenizer is not the Qwen byte-level split".into(),
        ));
    }
    if steps[0].get("type").and_then(|item| item.as_str()) != Some("Split") {
        return Err(WeightError::NotBuilt(
            "tokenizer pre-tokenizer is not the Qwen byte-level split".into(),
        ));
    }
    let pattern = steps[0]
        .get("pattern")
        .and_then(regex_pattern)
        .unwrap_or("");
    if pattern != QWEN_SPLIT {
        return Err(WeightError::NotBuilt(
            "tokenizer pre-tokenizer is not the Qwen byte-level split".into(),
        ));
    }
    if steps[1].get("type").and_then(|item| item.as_str()) != Some("ByteLevel") {
        return Err(WeightError::NotBuilt(
            "tokenizer pre-tokenizer is not the Qwen byte-level split".into(),
        ));
    }
    if steps[1]
        .get("add_prefix_space")
        .and_then(|item| item.as_bool())
        == Some(true)
    {
        return Err(WeightError::NotBuilt(
            "tokenizer pre-tokenizer adds a prefix space".into(),
        ));
    }
    Ok(())
}

fn regex_pattern(value: &Value) -> Option<&str> {
    value
        .as_str()
        .or_else(|| value.get("Regex").and_then(|item| item.as_str()))
        .or_else(|| value.get("String").and_then(|item| item.as_str()))
}

fn vocab_map(model: &Value) -> Result<HashMap<String, i32>, WeightError> {
    let Some(vocab) = model.get("vocab").and_then(|item| item.as_object()) else {
        return Err(WeightError::NotBuilt("tokenizer has no vocab".into()));
    };
    let mut out = HashMap::with_capacity(vocab.len());
    for (token, id) in vocab {
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
        out.insert(token.clone(), id as i32);
    }
    Ok(out)
}

fn merge_ranks(model: &Value) -> Result<HashMap<(String, String), usize>, WeightError> {
    let Some(merges) = model.get("merges").and_then(|item| item.as_array()) else {
        return Err(WeightError::NotBuilt("tokenizer has no merges".into()));
    };
    let mut ranks = HashMap::with_capacity(merges.len());
    for (rank, item) in merges.iter().enumerate() {
        let pair = if let Some(text) = item.as_str() {
            let Some((left, right)) = text.split_once(' ') else {
                return Err(WeightError::NotBuilt(format!(
                    "tokenizer merge {text} is not a pair"
                )));
            };
            if right.contains(' ') {
                return Err(WeightError::NotBuilt(format!(
                    "tokenizer merge {text} is not a pair"
                )));
            }
            (left.to_string(), right.to_string())
        } else if let Some(pair) = item.as_array() {
            if pair.len() != 2 {
                return Err(WeightError::NotBuilt(
                    "tokenizer merge is not a pair".into(),
                ));
            }
            let left = pair[0]
                .as_str()
                .ok_or_else(|| WeightError::NotBuilt("tokenizer merge is not a pair".into()))?;
            let right = pair[1]
                .as_str()
                .ok_or_else(|| WeightError::NotBuilt("tokenizer merge is not a pair".into()))?;
            (left.to_string(), right.to_string())
        } else {
            return Err(WeightError::NotBuilt(
                "tokenizer merge is not a pair".into(),
            ));
        };
        if ranks.insert(pair, rank).is_some() {
            return Err(WeightError::NotBuilt(
                "tokenizer merge list repeats a pair".into(),
            ));
        }
    }
    Ok(ranks)
}

fn special_tokens(
    vocab: &HashMap<String, i32>,
    added: Option<&Value>,
) -> Result<Vec<(String, i32)>, WeightError> {
    let mut out = Vec::new();
    if let Some(added) = added {
        let Some(rows) = added.as_array() else {
            return Err(WeightError::NotBuilt(
                "tokenizer added_tokens is not a list".into(),
            ));
        };
        for row in rows {
            if row.get("special").and_then(|item| item.as_bool()) != Some(true) {
                continue;
            }
            let Some(content) = row.get("content").and_then(|item| item.as_str()) else {
                return Err(WeightError::NotBuilt(
                    "tokenizer added token has no content".into(),
                ));
            };
            let Some(id) = row.get("id").and_then(|item| item.as_i64()) else {
                return Err(WeightError::NotBuilt(format!(
                    "tokenizer added token {content} has no id"
                )));
            };
            if id < 0 || id > i32::MAX as i64 {
                return Err(WeightError::NotBuilt(format!(
                    "tokenizer id {id} for {content} does not fit"
                )));
            }
            if let Some(vocab_id) = vocab.get(content) {
                if i64::from(*vocab_id) != id {
                    return Err(WeightError::NotBuilt(format!(
                        "tokenizer id for {content} is {vocab_id} in vocab and {id} in added_tokens"
                    )));
                }
            }
            out.push((content.to_string(), id as i32));
        }
    }
    for name in ["<|im_start|>", "<|im_end|>", "<think>", "</think>"] {
        if out.iter().any(|(token, _)| token == name) {
            continue;
        }
        if let Some(id) = vocab.get(name) {
            out.push((name.to_string(), *id));
        }
    }
    out.sort_by_key(|(token, _)| std::cmp::Reverse(token.len()));
    Ok(out)
}

enum Piece<'a> {
    Special(&'a str),
    Plain(&'a str),
}

fn split_specials<'a>(text: &'a str, specials: &'a [(String, i32)]) -> Vec<Piece<'a>> {
    let mut out = Vec::new();
    let mut plain_start = 0;
    let mut index = 0;
    while index < text.len() {
        let found = specials
            .iter()
            .find(|(token, _)| !token.is_empty() && text[index..].starts_with(token.as_str()));
        if let Some((token, _)) = found {
            if index > plain_start {
                out.push(Piece::Plain(&text[plain_start..index]));
            }
            out.push(Piece::Special(token.as_str()));
            index += token.len();
            plain_start = index;
        } else {
            let step = text[index..].chars().next().map_or(1, char::len_utf8);
            index += step;
        }
    }
    if plain_start < text.len() {
        out.push(Piece::Plain(&text[plain_start..]));
    }
    out
}

fn pretokenize(text: &str) -> Result<Vec<String>, WeightError> {
    let mut index = 0;
    let mut out = Vec::new();
    while index < text.len() {
        let Some(len) = next_piece(&text[index..]) else {
            let shown = text[index..].chars().next().unwrap_or('\0');
            return Err(WeightError::NotBuilt(format!(
                "pre-tokenizer stopped at {shown:?}"
            )));
        };
        if len == 0 || index + len > text.len() {
            return Err(WeightError::NotBuilt("pre-tokenizer stopped".into()));
        }
        out.push(text[index..index + len].to_string());
        index += len;
    }
    Ok(out)
}

fn next_piece(text: &str) -> Option<usize> {
    if let Some(found) = PIECE_CONTRACTION.find(text) {
        return Some(found.end());
    }
    if let Some(found) = PIECE_LETTERS.find(text) {
        return Some(found.end());
    }
    if let Some(found) = PIECE_NUMBER.find(text) {
        return Some(found.end());
    }
    if let Some(found) = PIECE_PUNCT.find(text) {
        return Some(found.end());
    }
    if let Some(found) = PIECE_NEWLINE.find(text) {
        return Some(found.end());
    }
    if let Some(len) = trailing_space(text) {
        return Some(len);
    }
    PIECE_SPACE.find(text).map(|found| found.end())
}

/// `\s+(?!\S)`: a whitespace run that ends at the string, or all but the last
/// whitespace character when a non-whitespace character follows.
fn trailing_space(text: &str) -> Option<usize> {
    let found = PIECE_SPACE.find(text)?;
    let end = found.end();
    if end == 0 {
        return None;
    }
    if end == text.len() {
        return Some(end);
    }
    let ws = &text[..end];
    let mut ends = Vec::new();
    for (index, ch) in ws.char_indices() {
        ends.push(index + ch.len_utf8());
    }
    if ends.len() >= 2 {
        Some(ends[ends.len() - 2])
    } else {
        None
    }
}

fn byte_symbols(word: &str) -> Vec<String> {
    let map = byte_to_unicode();
    word.as_bytes()
        .iter()
        .map(|byte| map[*byte as usize].to_string())
        .collect()
}

fn bpe(mut symbols: Vec<String>, ranks: &HashMap<(String, String), usize>) -> Vec<String> {
    if symbols.len() < 2 {
        return symbols;
    }
    loop {
        let mut best_rank = usize::MAX;
        let mut best_index = None;
        for index in 0..symbols.len() - 1 {
            if let Some(&rank) = ranks.get(&(symbols[index].clone(), symbols[index + 1].clone())) {
                if rank < best_rank {
                    best_rank = rank;
                    best_index = Some(index);
                }
            }
        }
        let Some(index) = best_index else {
            break;
        };
        let merged = format!("{}{}", symbols[index], symbols[index + 1]);
        symbols.splice(index..index + 2, std::iter::once(merged));
    }
    symbols
}

fn byte_to_unicode() -> [char; 256] {
    let mut visible = Vec::with_capacity(256);
    visible.extend(u32::from(b'!')..=u32::from(b'~'));
    visible.extend(0xA1..=0xAC);
    visible.extend(0xAE..=0xFF);
    let mut mapped = visible.clone();
    let mut extra = 0u32;
    for byte in 0..256u32 {
        if !visible.contains(&byte) {
            visible.push(byte);
            mapped.push(256 + extra);
            extra += 1;
        }
    }
    let mut out = ['\0'; 256];
    for (byte, ch) in visible.into_iter().zip(mapped) {
        out[byte as usize] =
            char::from_u32(ch).expect("byte map stays inside unicode scalar values");
    }
    out
}

fn push_id(ids: &mut Vec<i32>, id: i64, vocab_size: i64, token: &str) -> Result<(), WeightError> {
    if id < 0 || id >= vocab_size {
        return Err(WeightError::NotBuilt(format!(
            "tokenizer id {id} for {token} is outside vocab {vocab_size}"
        )));
    }
    ids.push(id as i32);
    Ok(())
}

/// One token per UTF-8 byte. Any non-empty prompt maps into a 256-id table.
fn tokenize_bytes(prompt: &str, vocab_size: i64) -> Result<Vec<i32>, WeightError> {
    if vocab_size < 256 {
        return Err(WeightError::NotBuilt(format!(
            "byte tokenizer needs vocab >= 256, weights have {vocab_size}"
        )));
    }
    let bytes = prompt.as_bytes();
    if bytes.is_empty() {
        return Err(WeightError::NotBuilt("prompt has no tokens".into()));
    }
    const MAX_BYTES: usize = 1024;
    if bytes.len() > MAX_BYTES {
        return Err(WeightError::NotBuilt(format!(
            "byte tokenizer accepts at most {MAX_BYTES} bytes, prompt is {}",
            bytes.len()
        )));
    }
    Ok(bytes.iter().map(|byte| i32::from(*byte)).collect())
}

#[cfg(test)]
mod tests {
    use super::{chat_text, load_tokenizer, pretokenize, tokenize};
    use crate::families::flux2::text::{encode_penultimate, encoder_layout, Qwen3Config};
    use crate::runtime::{BackendKind, GgmlContext};
    use crate::weights::{write_f32_tensors, F32Tensor};

    #[test]
    fn qwen_pieces_match_the_chat_regex() {
        let cases = [
            ("a cat", &["a", " cat"][..]),
            ("hello", &["hello"][..]),
            ("hello ", &["hello", " "][..]),
            ("a  cat", &["a", " ", " cat"][..]),
            ("it's", &["it", "'s"][..]),
            ("It's", &["It", "'s"][..]),
            ("123", &["1", "2", "3"][..]),
            ("a-b", &["a", "-b"][..]),
            ("hi\nthere", &["hi", "\n", "there"][..]),
            ("  x", &[" ", " x"][..]),
            ("a cat!", &["a", " cat", "!"][..]),
            ("  ", &["  "][..]),
            ("hello  world\n", &["hello", " ", " world", "\n"][..]),
            (" \n ", &[" \n", " "][..]),
            ("a\tb", &["a", "\tb"][..]),
            ("一只猫", &["一只猫"][..]),
        ];
        for (text, expect) in cases {
            let got = pretokenize(text).unwrap();
            assert_eq!(got, expect, "{text:?}");
            assert_eq!(got.concat(), text);
        }
    }

    #[test]
    fn chat_wrap_and_one_merge_change_the_ids() {
        let mut json = qwen_tokenizer(false);
        let thinking = tokenize(&json, "hi", 260, true).unwrap();
        assert_eq!(thinking, hi_ids(false));
        let quiet = tokenize(&json, "hi", 262, false).unwrap();
        assert_eq!(quiet, hi_ids(true));
        let no_flag = qwen_tokenizer_template(false);
        let quiet_same = tokenize(&no_flag, "hi", 260, false).unwrap();
        assert_eq!(quiet_same, hi_ids(false));
        json = qwen_tokenizer(true);
        let merged = tokenize(&json, "hi", 262, true).unwrap();
        assert_eq!(merged, hi_ids_merged());
        let other = tokenize(&json, "cat", 262, true).unwrap();
        assert_ne!(merged, other);
        let err = tokenize(
            r#"{"model":{"type":"Unigram","vocab":{},"merges":[]}}"#,
            "hi",
            8,
            true,
        )
        .unwrap_err();
        assert!(err.to_string().contains("model Unigram"), "{err}");
        let err = tokenize(&llama_template(), "hi", 260, true).unwrap_err();
        assert!(
            err.to_string().contains("chat template is not built"),
            "{err}"
        );
    }

    #[test]
    fn config_template_reaches_the_penultimate_tower() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-qwen-chat-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut enc = Qwen3Config::smoke();
        enc.vocab = 260;
        let owned: Vec<(String, i64, i64, Vec<f32>)> = encoder_layout(enc)
            .into_iter()
            .map(|(name, ne0, ne1)| {
                let n = (ne0 as usize) * (ne1 as usize);
                let data = if name.ends_with("norm.weight") || name.ends_with("layernorm.weight") {
                    vec![1.0; n]
                } else if name == "embed_tokens.weight" {
                    embed_rows(ne0, ne1)
                } else {
                    (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.02).collect()
                };
                (name, ne0, ne1, data)
            })
            .collect();
        let tensors: Vec<F32Tensor<'_>> = owned
            .iter()
            .map(|(name, ne0, ne1, data)| F32Tensor {
                name,
                ne0: *ne0,
                ne1: *ne1,
                data,
            })
            .collect();
        write_f32_tensors(&dir.join("text_encoder.gguf"), &tensors).unwrap();
        std::fs::write(dir.join("tokenizer.json"), tokenizer_without_template()).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template":"<|im_start|>user\n{{ content }}<|im_end|>\n{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}"}"#,
        )
        .unwrap();
        let loaded = load_tokenizer(&dir).unwrap();
        assert!(loaded.contains("enable_thinking"), "{loaded}");
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let paths = vec![dir.join("text_encoder.gguf")];
        let (cat, cat_len) = encode_penultimate(&ctx, &paths, &loaded, "cat", 8).unwrap();
        let (dog, _) = encode_penultimate(&ctx, &paths, &loaded, "dog", 8).unwrap();
        assert!(cat_len > 2, "chat wrap collapsed to {cat_len}");
        assert_ne!(cat, dog);
        assert!(cat.iter().all(|v| v.is_finite()));
        let long = "a".repeat(600);
        let err = tokenize(&loaded, &long, 260, true).unwrap_err();
        assert!(err.to_string().contains("accepts 512"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn embed_rows(hidden: i64, vocab: i64) -> Vec<f32> {
        let hidden = hidden as usize;
        let mut data = vec![0f32; hidden * (vocab as usize)];
        for token in 0..vocab as usize {
            data[token % hidden + token * hidden] = 0.05 + (token as f32) * 0.001;
        }
        data
    }

    fn hi_ids(with_think: bool) -> Vec<i32> {
        let mut ids = Vec::new();
        ids.push(256);
        ids.extend(b"user".iter().map(|byte| i32::from(*byte)));
        ids.push(i32::from(b'\n'));
        ids.extend(b"hi".iter().map(|byte| i32::from(*byte)));
        ids.push(257);
        ids.push(i32::from(b'\n'));
        ids.push(256);
        ids.extend(b"assistant".iter().map(|byte| i32::from(*byte)));
        ids.push(i32::from(b'\n'));
        if with_think {
            ids.push(258);
            ids.push(i32::from(b'\n'));
            ids.push(i32::from(b'\n'));
            ids.push(259);
            ids.push(i32::from(b'\n'));
            ids.push(i32::from(b'\n'));
        }
        ids
    }

    fn hi_ids_merged() -> Vec<i32> {
        let mut ids = hi_ids(false);
        let hi = ids.iter().position(|id| *id == i32::from(b'h')).unwrap();
        assert_eq!(ids[hi + 1], i32::from(b'i'));
        ids.splice(hi..hi + 2, std::iter::once(260));
        ids
    }

    fn qwen_tokenizer(merge_hi: bool) -> String {
        tokenizer_json(merge_hi, true)
    }

    fn qwen_tokenizer_template(enable_thinking_flag: bool) -> String {
        tokenizer_json(false, enable_thinking_flag)
    }

    fn tokenizer_without_template() -> String {
        let mut vocab = String::from("{");
        let map = super::byte_to_unicode();
        for byte in 0..256 {
            if byte > 0 {
                vocab.push(',');
            }
            let token = map[byte].to_string();
            vocab.push_str(&serde_json::to_string(&token).unwrap());
            vocab.push(':');
            vocab.push_str(&byte.to_string());
        }
        for (token, id) in [
            ("<|im_start|>", 256),
            ("<|im_end|>", 257),
            ("<think>", 258),
            ("</think>", 259),
        ] {
            vocab.push(',');
            vocab.push_str(&serde_json::to_string(token).unwrap());
            vocab.push(':');
            vocab.push_str(&id.to_string());
        }
        vocab.push('}');
        format!(
            r#"{{"pre_tokenizer":{{"type":"Sequence","pretokenizers":[{{"type":"Split","pattern":{{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{{L}}\\p{{N}}]?\\p{{L}}+|\\p{{N}}| ?[^\\s\\p{{L}}\\p{{N}}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}},"behavior":"Isolated"}},{{"type":"ByteLevel","add_prefix_space":false,"use_regex":false}}]}},"added_tokens":[{{"id":256,"content":"<|im_start|>","special":true}},{{"id":257,"content":"<|im_end|>","special":true}},{{"id":258,"content":"<think>","special":true}},{{"id":259,"content":"</think>","special":true}}],"model":{{"type":"BPE","vocab":{vocab},"merges":[]}}}}"#
        )
    }

    fn tokenizer_json(merge_hi: bool, thinking_flag: bool) -> String {
        let mut raw = tokenizer_without_template();
        if merge_hi {
            raw = raw.replace("\"merges\":[]", "\"merges\":[\"h i\"],\"vocab_pad\":0");
            raw = raw.replace("\"</think>\":259}", "\"</think>\":259,\"hi\":260}");
        }
        let template = if thinking_flag {
            "<|im_start|>user\\n{{ content }}<|im_end|>\\n{% if enable_thinking is defined and enable_thinking is false %}<think>\\n\\n</think>\\n\\n{% endif %}"
        } else {
            "<|im_start|>user\\n{{ content }}<|im_end|>"
        };
        raw.pop();
        format!("{raw},\"chat_template\":\"{template}\"}}")
    }

    fn llama_template() -> String {
        let mut raw = tokenizer_without_template();
        raw.pop();
        format!("{raw},\"chat_template\":\"<|start_header_id|>user\"}}")
    }

    #[test]
    fn chat_string_matches_the_single_user_turn() {
        assert_eq!(
            chat_text("hi", true, true),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            chat_text("hi", false, true),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(
            chat_text("hi", false, false),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
