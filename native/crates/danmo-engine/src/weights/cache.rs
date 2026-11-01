use super::{GgufWeights, WeightError};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

const MAX_ENTRIES: usize = 1;
const TTL: Duration = Duration::from_secs(30 * 60);

struct Entry {
    key: String,
    weights: Arc<GgufWeights>,
    last_used: Instant,
}

#[derive(Default)]
struct Cache {
    entries: VecDeque<Entry>,
}

impl Cache {
    fn get(&mut self, key: &str) -> Option<Arc<GgufWeights>> {
        let index = self.entries.iter().position(|entry| entry.key == key)?;
        let expired = self.entries[index].last_used.elapsed() > TTL;
        let entry = self.entries.remove(index)?;
        if expired {
            return None;
        }
        let weights = entry.weights.clone();
        self.entries.push_back(Entry {
            key: key.into(),
            weights: weights.clone(),
            last_used: Instant::now(),
        });
        Some(weights)
    }

    fn put(&mut self, key: String, weights: Arc<GgufWeights>) -> Arc<GgufWeights> {
        self.entries.retain(|entry| entry.key != key);
        while self.entries.len() >= MAX_ENTRIES {
            self.entries.pop_front();
        }
        self.entries.push_back(Entry {
            key,
            weights: weights.clone(),
            last_used: Instant::now(),
        });
        weights
    }

    fn purge(&mut self) {
        self.entries
            .retain(|entry| entry.last_used.elapsed() <= TTL);
    }
}

fn cache() -> &'static Mutex<Cache> {
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Cache::default()))
}

pub fn open(path: &Path) -> Result<Arc<GgufWeights>, WeightError> {
    open_parts(std::slice::from_ref(&path.to_path_buf()))
}

pub fn open_parts(paths: &[PathBuf]) -> Result<Arc<GgufWeights>, WeightError> {
    open_parts_with(cache(), paths)
}

fn open_parts_with(
    cache: &Mutex<Cache>,
    paths: &[PathBuf],
) -> Result<Arc<GgufWeights>, WeightError> {
    let key = revision(paths)?;
    let mut guard = cache.lock().expect("weight cache");
    guard.purge();
    if let Some(weights) = guard.get(&key) {
        return Ok(weights);
    }
    drop(guard);
    let weights = Arc::new(if paths.len() == 1 {
        GgufWeights::open_uncached(&paths[0])?
    } else {
        GgufWeights::open_parts_uncached(paths)?
    });
    Ok(cache.lock().expect("weight cache").put(key, weights))
}

pub fn revision(paths: &[PathBuf]) -> Result<String, WeightError> {
    let mut key = String::new();
    for path in paths {
        let metadata = std::fs::metadata(path).map_err(|error| WeightError::Open {
            path: path.display().to_string(),
            detail: error.to_string(),
        })?;
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        key.push_str(&format!(
            "{}:{}:{}:{};",
            path.display(),
            metadata.len(),
            modified,
            key.len()
        ));
    }
    Ok(key)
}

pub fn resident_bytes() -> usize {
    cache()
        .lock()
        .expect("weight cache")
        .entries
        .iter()
        .map(|entry| entry.weights.mapped_bytes())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::write_f32_gguf;

    #[test]
    fn mmap_cache_reuses_unchanged_file_and_evicts_on_replace() {
        let cache = Mutex::new(Cache::default());
        let dir = std::env::temp_dir().join(format!("danmo-weight-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let first = dir.join("first.gguf");
        let second = dir.join("second.gguf");
        write_f32_gguf(&first, "w", 1, 1, &[1.0]).unwrap();
        write_f32_gguf(&second, "w", 1, 1, &[2.0]).unwrap();
        let a = open_parts_with(&cache, std::slice::from_ref(&first)).unwrap();
        let b = open_parts_with(&cache, std::slice::from_ref(&first)).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        let _ = open_parts_with(&cache, std::slice::from_ref(&second)).unwrap();
        assert_eq!(cache.lock().unwrap().entries.len(), 1);
        let c = open_parts_with(&cache, std::slice::from_ref(&first)).unwrap();
        assert!(!Arc::ptr_eq(&a, &c));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
