//! Z-Image component weights kept in the process MLX executor cache.
//!
//! A safetensors bundle admits its transformer, text encoder, and VAE here.
//! The second load of an unchanged directory does not read the shards again.
//! GGUF bundles never enter this cache.

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use danmo_mlx::{CacheKey, CacheRevision, Error, Result};

use super::model::load_directory;
use super::MlxZImageConfig;
use crate::weights::{MlxWeights, WeightDtype};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentReport {
    pub hit: bool,
    pub bytes: usize,
    pub tensors: usize,
}

pub fn load_transformer(
    directory: &Path,
    config: MlxZImageConfig,
    dtype: WeightDtype,
    quantization: Option<danmo_mlx::AffineQuantization>,
) -> Result<ResidentReport> {
    admit(directory, "z_image resident transformer", move |path, _| {
        load_directory(path, config, dtype, quantization)
    })
}

pub fn load_text_encoder(directory: &Path) -> Result<ResidentReport> {
    admit(
        directory,
        "z_image resident text encoder",
        |path, stream| {
            let weights = crate::common::qwen3::mlx::load_directory(path, WeightDtype::Native)?;
            weights.materialize_embedding("model.embed_tokens.weight", stream)
        },
    )
}

pub fn load_vae(directory: &Path) -> Result<ResidentReport> {
    admit(directory, "z_image resident vae", |path, _| {
        crate::common::vae::mlx::load(path, WeightDtype::Native)
    })
}

pub(crate) fn cache_key(directory: &Path) -> Result<CacheKey> {
    Ok(CacheKey {
        id: cache_id(directory)?,
        revision: directory_revision(directory)?,
    })
}

fn admit(
    directory: &Path,
    operation: &'static str,
    load: impl FnOnce(&Path, &danmo_mlx::Stream) -> Result<MlxWeights> + Send + 'static,
) -> Result<ResidentReport> {
    let key = cache_key(directory)?;
    crate::runtime::mlx_executor()?.execute(operation, move |_, stream, cache, _| {
        if let Some(weights) = cache.get::<MlxWeights>(&key)? {
            return Ok(ResidentReport {
                hit: true,
                bytes: weights.nbytes(),
                tensors: weights.names().count(),
            });
        }
        let weights = load(Path::new(&key.id), stream)?;
        let bytes = weights.nbytes();
        let tensors = weights.names().count();
        cache.insert(key, weights, bytes)?;
        Ok(ResidentReport {
            hit: false,
            bytes,
            tensors,
        })
    })
}

pub fn unload_transformer(directory: &Path) -> Result<bool> {
    let id = cache_id(directory)?;
    crate::runtime::mlx_executor()?.execute("z_image unload transformer", move |_, _, cache, _| {
        Ok(cache.unload(&id))
    })
}

fn cache_id(directory: &Path) -> Result<String> {
    Ok(std::fs::canonicalize(directory)
        .map_err(|error| {
            Error::new(
                "Z-Image resident",
                format!("{}: {error}", directory.display()),
            )
        })?
        .display()
        .to_string())
}

fn directory_revision(directory: &Path) -> Result<CacheRevision> {
    let paths = shard_paths(directory)?;
    let stamp = crate::weights::revision(&paths)
        .map_err(|error| Error::new("Z-Image resident", error.to_string()))?;
    let mut length = 0u64;
    let mut modified_ns = 0u64;
    for path in &paths {
        let metadata = std::fs::metadata(path).map_err(|error| {
            Error::new("Z-Image resident", format!("{}: {error}", path.display()))
        })?;
        length += metadata.len();
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0);
        modified_ns = modified_ns.max(modified);
    }
    Ok(CacheRevision {
        length,
        modified_ns,
        mix: mix_stamp(&stamp),
    })
}

fn shard_paths(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(directory).map_err(|error| {
        Error::new(
            "Z-Image resident",
            format!("{}: {error}", directory.display()),
        )
    })? {
        let path = entry
            .map_err(|error| Error::new("Z-Image resident", error.to_string()))?
            .path();
        if path.extension().and_then(|value| value.to_str()) == Some("safetensors") {
            paths.push(path);
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err(Error::new(
            "Z-Image resident",
            format!("no safetensors shards in {}", directory.display()),
        ));
    }
    Ok(paths)
}

fn mix_stamp(stamp: &str) -> u64 {
    let mut mix = 0xcbf29ce484222325u64;
    for byte in stamp.as_bytes() {
        mix ^= u64::from(*byte);
        mix = mix.wrapping_mul(0x100000001b3);
    }
    mix
}

#[cfg(test)]
mod tests {
    use super::super::MlxZImageConfig;
    use super::{load_transformer, unload_transformer};
    use crate::weights::WeightDtype;
    use danmo_mlx::{Array, Stream};
    use std::path::Path;

    fn write_shard(path: &Path, width: i32) {
        let stream = Stream::cpu().unwrap();
        let values = vec![0.25f32; width as usize];
        let array = Array::from_f32_slice(&values, &[1, width]).unwrap();
        danmo_mlx::save_safetensors(path, &[("layers.0.attention_norm1.weight", &array)], &[])
            .unwrap();
        let _ = stream;
    }

    #[test]
    fn resident_transformer_reuses_an_unchanged_directory_and_drops_on_unload() {
        let directory =
            std::env::temp_dir().join(format!("zimage-resident-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let shard = directory.join("model.safetensors");
        write_shard(&shard, 4);

        let first = load_transformer(
            &directory,
            MlxZImageConfig::published(),
            WeightDtype::Native,
            None,
        )
        .unwrap();
        assert!(!first.hit);
        assert_eq!(first.tensors, 1);
        assert!(first.bytes > 0);

        let second = load_transformer(
            &directory,
            MlxZImageConfig::published(),
            WeightDtype::Native,
            None,
        )
        .unwrap();
        assert!(second.hit);
        assert_eq!(second.bytes, first.bytes);
        assert_eq!(second.tensors, 1);

        write_shard(&shard, 16);
        let changed = load_transformer(
            &directory,
            MlxZImageConfig::published(),
            WeightDtype::Native,
            None,
        )
        .unwrap();
        assert!(!changed.hit);
        assert_ne!(changed.bytes, first.bytes);

        assert!(unload_transformer(&directory).unwrap());
        let reloaded = load_transformer(
            &directory,
            MlxZImageConfig::published(),
            WeightDtype::Native,
            None,
        )
        .unwrap();
        assert!(!reloaded.hit);
        let _ = std::fs::remove_dir_all(&directory);
    }
}
