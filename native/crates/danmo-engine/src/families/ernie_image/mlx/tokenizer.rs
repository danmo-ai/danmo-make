use std::path::Path;

use tokenizers::Tokenizer;

const MAX_TEXT_TOKENS: usize = 2048;
const SYSTEM_PROMPT: &str = "你是一个专业的文生图 Prompt 增强助手。你将收到用户的简短图片描述及目标生成分辨率，请据此扩写为一段内容丰富、细节充分的视觉描述，以帮助文生图模型生成高质量的图片。仅输出增强后的描述，不要包含任何解释或前缀。";

pub struct ErnieTokenizer {
    tokenizer: Tokenizer,
    vocab: usize,
}

impl ErnieTokenizer {
    pub fn load(directory: &Path, vocab: i32) -> Result<Self, String> {
        let path = directory.join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        if vocab < 1 {
            return Err("ERNIE tokenizer vocab must be positive".into());
        }
        Ok(Self {
            tokenizer,
            vocab: vocab as usize,
        })
    }

    pub fn encode_text(&self, prompt: &str) -> Result<Vec<i32>, String> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|error| error.to_string())?;
        let mut ids = encoding.get_ids().to_vec();
        ids.truncate(MAX_TEXT_TOKENS);
        self.convert(ids)
    }

    pub fn encode_pe(&self, prompt: &str, width: u32, height: u32) -> Result<Vec<i32>, String> {
        let content = serde_json::to_string(&serde_json::json!({
            "prompt": prompt,
            "width": width,
            "height": height,
        }))
        .map_err(|error| error.to_string())?;
        let text =
            format!("<s>[SYSTEM_PROMPT]{SYSTEM_PROMPT}[/SYSTEM_PROMPT][INST]{content}[/INST]");
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| error.to_string())?;
        self.convert(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[i32]) -> Result<String, String> {
        let ids: Vec<u32> = ids
            .iter()
            .copied()
            .map(|id| u32::try_from(id).map_err(|_| format!("negative token id {id}")))
            .collect::<Result<_, _>>()?;
        self.tokenizer
            .decode(&ids, true)
            .map(|text| text.trim().to_string())
            .map_err(|error| error.to_string())
    }

    fn convert(&self, ids: Vec<u32>) -> Result<Vec<i32>, String> {
        if ids.is_empty() {
            return Err("ERNIE prompt produced no tokens".into());
        }
        ids.into_iter()
            .map(|id| {
                let id = id as usize;
                if id >= self.vocab || id > i32::MAX as usize {
                    Err(format!(
                        "ERNIE token id {id} is outside vocab {}",
                        self.vocab
                    ))
                } else {
                    Ok(id as i32)
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::ErnieTokenizer;

    #[test]
    #[ignore = "loads the official 16 MB ERNIE tokenizers"]
    fn official_text_and_pe_templates_round_trip() {
        let root = PathBuf::from(std::env::var("DANQING_TEST_ERNIE_BUNDLE").unwrap());
        let text = ErnieTokenizer::load(&root.join("tokenizer"), 131072).unwrap();
        let ids = text.encode_text("一只黑白相间的中华田园犬").unwrap();
        assert_eq!(ids[0], 1);
        assert_eq!(text.decode(&ids).unwrap(), "一只黑白相间的中华田园犬");
        let pe = ErnieTokenizer::load(&root.join("pe_tokenizer"), 131072).unwrap();
        let prefix = pe.encode_pe("一个女孩在海边", 1024, 1024).unwrap();
        assert!(prefix.starts_with(&[1, 17]));
        assert!(prefix.contains(&18));
        assert!(prefix.contains(&3));
        assert_eq!(prefix.last().copied(), Some(4));
    }
}
