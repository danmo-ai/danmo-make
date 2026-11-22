use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use danmo_mlx::{CacheKey, CacheRevision, Error, ModelCache, Result};

use super::mistral;
use crate::common::vae;
use crate::weights::{MlxWeights, WeightDtype};

#[derive(Clone, Copy)]
pub enum Component {
    Transformer,
    TextEncoder,
    PromptEnhancer,
    Vae,
}

pub fn resident(
    cache: &mut ModelCache,
    key: &CacheKey,
    component: Component,
    stream: &danmo_mlx::Stream,
) -> Result<MlxWeights> {
    if cache.get::<MlxWeights>(key)?.is_none() {
        let path = Path::new(&key.id);
        let weights = match component {
            Component::Transformer => {
                let weights = crate::weights::load_mlx_directory(path, WeightDtype::Native, None)?;
                super::config::ErnieConfig::load(path)
                    .map_err(|error| Error::new("ERNIE transformer", error))?;
                weights
            }
            Component::TextEncoder => mistral::load_text_encoder(path, stream)?,
            Component::PromptEnhancer => mistral::load_pe(path, stream)?,
            Component::Vae => vae::mlx::load(path, WeightDtype::Native)?,
        };
        let bytes = weights.nbytes();
        cache.insert(key.clone(), weights, bytes)?;
    }
    cache
        .get::<MlxWeights>(key)?
        .cloned()
        .ok_or_else(|| Error::new("ERNIE resident", "component cache insertion failed"))
}

pub fn cache_key(directory: &Path) -> Result<CacheKey> {
    Ok(CacheKey {
        id: cache_id(directory)?,
        revision: directory_revision(directory)?,
    })
}

pub fn unload(cache: &mut ModelCache, key: &CacheKey) {
    cache.unload(&key.id);
}

fn cache_id(directory: &Path) -> Result<String> {
    Ok(std::fs::canonicalize(directory)
        .map_err(|error| {
            Error::new(
                "ERNIE resident",
                format!("{}: {error}", directory.display()),
            )
        })?
        .display()
        .to_string())
}

fn directory_revision(directory: &Path) -> Result<CacheRevision> {
    let paths = shard_paths(directory)?;
    let stamp = crate::weights::revision(&paths)
        .map_err(|error| Error::new("ERNIE resident", error.to_string()))?;
    let mut length = 0u64;
    let mut modified_ns = 0u64;
    for path in &paths {
        let metadata = std::fs::metadata(path).map_err(|error| {
            Error::new("ERNIE resident", format!("{}: {error}", path.display()))
        })?;
        length += metadata.len();
        modified_ns = modified_ns.max(
            metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map(|duration| duration.as_nanos() as u64)
                .unwrap_or(0),
        );
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
            "ERNIE resident",
            format!("{}: {error}", directory.display()),
        )
    })? {
        let path = entry
            .map_err(|error| Error::new("ERNIE resident", error.to_string()))?
            .path();
        if path.extension().and_then(|value| value.to_str()) == Some("safetensors") {
            paths.push(path);
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err(Error::new(
            "ERNIE resident",
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
