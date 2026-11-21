//! Klein component weights kept on the process MLX executor.

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use danmo_mlx::{CacheKey, CacheRevision, Error, Result};

use super::forward;
use crate::weights::{MlxWeights, WeightDtype};

pub(super) fn load_transformer(directory: &Path) -> Result<()> {
    admit(directory, "flux2 resident transformer", |path, _| {
        let weights = crate::weights::load_mlx_directory(path, WeightDtype::Native, None)?;
        let map = super::super::checkpoint::remap_diffusers_names(weights.names()).map_err(
            |error| Error::new("Flux2 Klein", error),
        )?;
        let weights = if map.iter().any(|(source, target)| source != target) {
            weights.rename(&map)?
        } else {
            weights
        };
        forward::assert_contract(&weights)?;
        Ok(weights)
    })
}

pub(super) fn load_text_encoder(directory: &Path) -> Result<()> {
    admit(directory, "flux2 resident text encoder", |path, stream| {
        let weights = crate::common::qwen3::mlx::load_directory(path, WeightDtype::Native)?;
        weights.materialize_embedding("model.embed_tokens.weight", stream)
    })
}

pub(super) fn load_vae(directory: &Path) -> Result<()> {
    admit(directory, "flux2 resident vae", |path, _| {
        crate::common::vae::mlx::load(path, WeightDtype::Native)
    })
}

pub(super) fn cache_key(directory: &Path) -> Result<CacheKey> {
    Ok(CacheKey {
        id: cache_id(directory)?,
        revision: directory_revision(directory)?,
    })
}

fn admit(
    directory: &Path,
    operation: &'static str,
    load: impl FnOnce(&Path, &danmo_mlx::Stream) -> Result<MlxWeights> + Send + 'static,
) -> Result<()> {
    let key = cache_key(directory)?;
    crate::runtime::mlx_executor()?.execute(operation, move |_, stream, cache, _| {
        if cache.get::<MlxWeights>(&key)?.is_some() {
            return Ok(());
        }
        let weights = load(Path::new(&key.id), stream)?;
        let bytes = weights.nbytes();
        cache.insert(key, weights, bytes)?;
        Ok(())
    })
}

fn cache_id(directory: &Path) -> Result<String> {
    Ok(std::fs::canonicalize(directory)
        .map_err(|error| {
            Error::new(
                "Flux2 resident",
                format!("{}: {error}", directory.display()),
            )
        })?
        .display()
        .to_string())
}

fn directory_revision(directory: &Path) -> Result<CacheRevision> {
    let paths = shard_paths(directory)?;
    let stamp = crate::weights::revision(&paths)
        .map_err(|error| Error::new("Flux2 resident", error.to_string()))?;
    let mut length = 0u64;
    let mut modified_ns = 0u64;
    for path in &paths {
        let metadata = std::fs::metadata(path).map_err(|error| {
            Error::new("Flux2 resident", format!("{}: {error}", path.display()))
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
            "Flux2 resident",
            format!("{}: {error}", directory.display()),
        )
    })? {
        let path = entry
            .map_err(|error| Error::new("Flux2 resident", error.to_string()))?
            .path();
        if path.extension().and_then(|value| value.to_str()) == Some("safetensors") {
            paths.push(path);
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err(Error::new(
            "Flux2 resident",
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
