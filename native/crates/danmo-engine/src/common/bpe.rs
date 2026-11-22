//! Byte-level BPE. The caller supplies the vocab, merge ranks, and specials.
//! Chat templates and length caps stay with the caller.

use crate::weights::WeightError;
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

static PIECE_CONTRACTION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?i:'s|'t|'re|'ve|'m|'ll|'d)").unwrap());
static PIECE_LETTERS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[^\r\n\p{L}\p{N}]?\p{L}+").unwrap());
static PIECE_NUMBER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\p{N}").unwrap());
static PIECE_PUNCT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^ ?[^\s\p{L}\p{N}]+[\r\n]*").unwrap());
static PIECE_NEWLINE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*[\r\n]+").unwrap());
static PIECE_SPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s+").unwrap());

pub(crate) fn encode(
    text: &str,
    vocab: &HashMap<String, i32>,
    ranks: &HashMap<(String, String), usize>,
    specials: &[(String, i32)],
    vocab_size: i64,
) -> Result<Vec<i32>, WeightError> {
    let mut ids = Vec::new();
    for piece in split_specials(text, specials) {
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
                    for token in merge(symbols, ranks) {
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
    Ok(ids)
}

/// SentencePiece-style BPE over Unicode scalar symbols with `<0xXX>` fallback.
/// Normalization and pre-tokenization remain the model family's responsibility.
pub(crate) fn encode_byte_fallback(
    text: &str,
    vocab: &HashMap<String, i32>,
    ranks: &HashMap<(String, String), usize>,
    specials: &[(String, i32)],
    vocab_size: i64,
) -> Result<Vec<i32>, WeightError> {
    let mut ids = Vec::new();
    for piece in split_specials(text, specials) {
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
                let symbols = plain.chars().map(|ch| ch.to_string()).collect();
                for token in merge(symbols, ranks) {
                    if let Some(id) = vocab.get(&token).copied() {
                        push_id(&mut ids, i64::from(id), vocab_size, &token)?;
                        continue;
                    }
                    for byte in token.as_bytes() {
                        let fallback = format!("<0x{byte:02X}>");
                        let id = vocab.get(&fallback).copied().ok_or_else(|| {
                            WeightError::NotBuilt(format!(
                                "tokenizer has neither {token:?} nor byte fallback {fallback}"
                            ))
                        })?;
                        push_id(&mut ids, i64::from(id), vocab_size, &fallback)?;
                    }
                }
            }
        }
    }
    if ids.is_empty() {
        return Err(WeightError::NotBuilt("prompt has no tokens".into()));
    }
    Ok(ids)
}

pub(crate) fn pretokenize(text: &str) -> Result<Vec<String>, WeightError> {
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

fn merge(mut symbols: Vec<String>, ranks: &HashMap<(String, String), usize>) -> Vec<String> {
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

pub(crate) fn byte_token(byte: u8) -> char {
    byte_to_unicode()[byte as usize]
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

pub(crate) fn push_id(
    ids: &mut Vec<i32>,
    id: i64,
    vocab_size: i64,
    token: &str,
) -> Result<(), WeightError> {
    if id < 0 || id >= vocab_size {
        return Err(WeightError::NotBuilt(format!(
            "tokenizer id {id} for {token} is outside vocab {vocab_size}"
        )));
    }
    ids.push(id as i32);
    Ok(())
}

/// Qwen2-style byte-level BPE files: `vocab.json` + `merges.txt` +
/// `tokenizer_config.json` (`added_tokens_decoder` for the specials).
pub(crate) struct Qwen2Tokenizer {
    pub vocab: HashMap<String, i32>,
    pub ranks: HashMap<(String, String), usize>,
    pub specials: Vec<(String, i32)>,
}

impl Qwen2Tokenizer {
    pub(crate) fn load(root: &std::path::Path) -> Result<Self, WeightError> {
        let vocab_path = root.join("vocab.json");
        let merges_path = root.join("merges.txt");
        if !vocab_path.is_file() || !merges_path.is_file() {
            return Err(WeightError::NotBuilt(format!(
                "tokenizer is missing vocab.json or merges.txt under {}",
                root.display()
            )));
        }
        let vocab_text = std::fs::read_to_string(&vocab_path).map_err(|err| WeightError::Open {
            path: vocab_path.display().to_string(),
            detail: err.to_string(),
        })?;
        let vocab_json: serde_json::Value = serde_json::from_str(&vocab_text).map_err(|err| {
            WeightError::NotBuilt(format!("tokenizer vocab.json is not json: {err}"))
        })?;
        let mut vocab = vocab_map(&vocab_json)?;
        let merges_text =
            std::fs::read_to_string(&merges_path).map_err(|err| WeightError::Open {
                path: merges_path.display().to_string(),
                detail: err.to_string(),
            })?;
        let ranks = merge_ranks(&merges_text)?;
        let specials = added_specials(&root.join("tokenizer_config.json"), &mut vocab)?;
        Ok(Self {
            vocab,
            ranks,
            specials,
        })
    }

    pub(crate) fn encode(&self, text: &str, vocab_size: i64) -> Result<Vec<i32>, WeightError> {
        encode(text, &self.vocab, &self.ranks, &self.specials, vocab_size)
    }
}

fn vocab_map(value: &serde_json::Value) -> Result<HashMap<String, i32>, WeightError> {
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
    path: &std::path::Path,
    vocab: &mut HashMap<String, i32>,
) -> Result<Vec<(String, i32)>, WeightError> {
    if !path.is_file() {
        return Err(WeightError::NotBuilt(format!(
            "tokenizer is missing {}",
            path.display()
        )));
    }
    let text = std::fs::read_to_string(path).map_err(|err| WeightError::Open {
        path: path.display().to_string(),
        detail: err.to_string(),
    })?;
    let config: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
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
        let vocab_id = match vocab.get(content).copied() {
            Some(vocab_id) => {
                if i64::from(vocab_id) != id {
                    return Err(WeightError::NotBuilt(format!(
                        "tokenizer id for {content} is {vocab_id} in vocab and {id} in added_tokens_decoder"
                    )));
                }
                vocab_id
            }
            None => {
                let Some(vocab_id) = i32::try_from(id).ok() else {
                    return Err(WeightError::NotBuilt(format!(
                        "tokenizer added token id {id} for {content} does not fit"
                    )));
                };
                if vocab.values().any(|existing| *existing == vocab_id) {
                    return Err(WeightError::NotBuilt(format!(
                        "tokenizer added token {content} reuses id {vocab_id}"
                    )));
                }
                vocab.insert(content.to_string(), vocab_id);
                vocab_id
            }
        };
        specials.push((content.to_string(), vocab_id));
    }
    specials.sort_by_key(|(token, _)| std::cmp::Reverse(token.len()));
    Ok(specials)
}
