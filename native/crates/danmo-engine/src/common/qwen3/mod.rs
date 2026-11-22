//! Qwen3 prompt tower. Tensor names are `layers.*` and `embed_tokens.weight`,
//! or the same names under the `model.` prefix when that prefix is in the file.
//! `blk.*` names are refused.
//!
//! The caller chooses which 1-based hidden states to concatenate and whether
//! the chat string leaves thinking on. One selected layer returns that state.
//! The concatenated width must equal the `width` argument.

mod graph;
#[cfg(feature = "mlx")]
pub(crate) mod mlx;
mod session;
mod tokenizer;

pub(crate) use session::{encoder_layout, is_rms, tower_present, Loaded, Qwen3Config};
pub(crate) use tokenizer::{encode_plain, load_tokenizer};
#[allow(unused_imports)]
pub(crate) use tokenizer::{parse_tokenizer_json, tokenize_plain, tokenize_plain_parsed};
#[cfg(feature = "mlx")]
pub(crate) use tokenizer::{tokenize, tokenize_chat_empty};
