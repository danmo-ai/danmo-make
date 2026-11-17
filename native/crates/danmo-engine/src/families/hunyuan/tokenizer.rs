//! HunyuanVideo 1.5 prompt tokenization: main `render_qwen_chat_messages`
//! (system + user + generation prompt, **no** `<|im_end|>`) over the fixed
//! video system message, truncated to `MAX_LENGTH` = crop_start 108 +
//! mllm_max_length 1000. The 108-token crop happens after the text tower.

use super::text::{MAX_LENGTH, SYSTEM_MESSAGE};
use crate::common::bpe::Qwen2Tokenizer;
use crate::weights::WeightError;
use std::path::Path;

/// main `_format_text_input` + `render_qwen_chat_messages`. Note the Qwen
/// blocks carry no `<|im_end|>` in this render.
pub fn render_chat(prompt: &str) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM_MESSAGE}\n<|im_start|>user\n{prompt}\n<|im_start|>assistant\n"
    )
}

/// Rendered chat → token ids, truncated to `MAX_LENGTH` (1108).
pub fn prompt_ids(bundle: &Path, prompt: &str, vocab_size: i64) -> Result<Vec<i32>, WeightError> {
    let tokenizer = Qwen2Tokenizer::load(&bundle.join("tokenizer"))?;
    let mut ids = tokenizer.encode(&render_chat(prompt), vocab_size)?;
    if ids.len() > MAX_LENGTH {
        ids.truncate(MAX_LENGTH);
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::{prompt_ids, render_chat, MAX_LENGTH};
    use crate::common::bpe::byte_token;
    use crate::families::hunyuan::text::SYSTEM_MESSAGE;
    use std::path::Path;

    fn write_tok(dir: &Path) {
        let tok = dir.join("tokenizer");
        std::fs::create_dir_all(&tok).unwrap();
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
        std::fs::write(tok.join("vocab.json"), vocab).unwrap();
        std::fs::write(tok.join("merges.txt"), "#version: 0.2\n").unwrap();
        std::fs::write(
            tok.join("tokenizer_config.json"),
            r#"{"added_tokens_decoder": {
                "256": {"content": "<|im_start|>", "special": true},
                "257": {"content": "<|im_end|>", "special": true}
            }}"#,
        )
        .unwrap();
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "danmo-hy-tok-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn chat_render_has_no_im_end() {
        let rendered = render_chat("a cat");
        let expected = format!(
            "<|im_start|>system\n{SYSTEM_MESSAGE}\n<|im_start|>user\na cat\n<|im_start|>assistant\n"
        );
        assert_eq!(render_chat("a cat"), expected);
        assert!(!expected.contains("<|im_end|>"));
    }

    #[test]
    fn prompt_ids_keep_specials_whole_and_truncate() {
        let dir = temp_dir("encode");
        write_tok(&dir);
        let ids = prompt_ids(&dir, "a cat", 258).unwrap();
        assert_eq!(ids[0], 256, "{ids:?}");
        assert!(ids
            .windows(2)
            .all(|pair| pair != [i32::from(b'<'), i32::from(b'|')]));
        let long = prompt_ids(&dir, &"a".repeat(3000), 258).unwrap();
        assert_eq!(long.len(), MAX_LENGTH);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
