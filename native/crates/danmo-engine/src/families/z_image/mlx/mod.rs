mod attention;
mod blocks;
mod control;
mod denoise;
mod feed_forward;
#[cfg(test)]
mod host;
mod model;
mod modulation;
pub(crate) mod resident;
pub(crate) mod session;
mod step_cache;
mod timestep;
pub(crate) mod train;
mod weights;

use crate::weights::WeightError;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MlxZImageConfig {
    pub in_channels: i64,
    pub dim: i64,
    pub heads: i64,
    pub layers: i64,
    pub refiners: i64,
    pub cap_dim: i64,
    pub vocab: i64,
    pub max_seq_len: usize,
    pub frequency_dim: i64,
    pub timestep_mid: i64,
    pub adaln_dim: i64,
    pub axes: [i64; 3],
    pub rope_theta: f32,
    pub patch: i64,
    pub t_scale: f32,
    pub norm_eps: f32,
    pub vae_scale: i64,
}

impl MlxZImageConfig {
    pub fn published() -> Self {
        Self {
            in_channels: 16,
            dim: 3840,
            heads: 30,
            layers: 30,
            refiners: 2,
            cap_dim: 2560,
            vocab: 151_936,
            max_seq_len: 512,
            frequency_dim: 256,
            timestep_mid: 1024,
            adaln_dim: 256,
            axes: [32, 48, 48],
            rope_theta: 256.0,
            patch: 2,
            t_scale: 1000.0,
            norm_eps: 1e-5,
            vae_scale: 8,
        }
    }

    pub fn head_dim(self) -> i64 {
        self.dim / self.heads
    }

    pub fn ffn_hidden(self) -> i64 {
        (self.dim / 3) * 8
    }

    pub fn token_ids(self, bundle: &Path, prompt: &str) -> Result<Vec<i32>, WeightError> {
        let tokenizer = crate::common::qwen3::load_tokenizer(bundle)?;
        let ids = crate::common::qwen3::tokenize(&tokenizer, prompt, self.vocab, true)?;
        if ids.len() > self.max_seq_len {
            return Err(WeightError::NotBuilt(format!(
                "prompt tokenized to {} tokens, maximum is {}",
                ids.len(),
                self.max_seq_len
            )));
        }
        Ok(ids)
    }

    /// Empty-user ChatML ids for CFG unconditional / caption dropout.
    /// Diffusers encodes `negative_prompt=""` through the same template.
    pub fn empty_token_ids(self, bundle: &Path) -> Result<Vec<i32>, WeightError> {
        let tokenizer = crate::common::qwen3::load_tokenizer(bundle)?;
        let ids = crate::common::qwen3::tokenize_chat_empty(&tokenizer, self.vocab, true)?;
        if ids.len() > self.max_seq_len {
            return Err(WeightError::NotBuilt(format!(
                "empty caption tokenized to {} tokens, maximum is {}",
                ids.len(),
                self.max_seq_len
            )));
        }
        if ids.is_empty() {
            return Err(WeightError::NotBuilt(
                "empty caption produced no tokens".into(),
            ));
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::MlxZImageConfig;

    #[test]
    fn published_config_matches_main() {
        let config = MlxZImageConfig::published();
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.ffn_hidden(), 10_240);
        assert_eq!(config.axes.iter().sum::<i64>(), 128);
        assert_eq!(config.patch * config.patch * config.in_channels, 64);
        assert_eq!(config.cap_dim, 2560);
        assert_eq!(config.vae_scale, 8);
        assert_eq!(config.frequency_dim, 256);
        assert_eq!(config.timestep_mid, 1024);
        assert_eq!(config.adaln_dim, 256);
    }

    #[test]
    #[ignore]
    fn real_bundle_token_ids_match_the_shared_qwen3_tokenizer() {
        let root = std::path::PathBuf::from(std::env::var("DANQING_TEST_ZIMAGE_BUNDLE").unwrap());
        let ids = MlxZImageConfig::published()
            .token_ids(&root, "a red cube")
            .unwrap();
        assert_eq!(
            ids,
            vec![151644, 872, 198, 64, 2518, 23739, 151645, 198, 151644, 77091, 198]
        );
    }
}
