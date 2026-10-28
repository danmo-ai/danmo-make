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
