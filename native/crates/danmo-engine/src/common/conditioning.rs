use crate::runtime::BackendKind;
use crate::weights::WeightError;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

const CAPACITY: usize = 4;

#[derive(PartialEq, Eq)]
struct Key {
    revision: String,
    backend: BackendKind,
    layers: Vec<i64>,
    width: i64,
    ids: Vec<i32>,
}

struct Entry {
    key: Key,
    values: Vec<f32>,
}

#[derive(Default)]
struct Cache {
    entries: VecDeque<Entry>,
    hits: u64,
    misses: u64,
}

fn cache() -> &'static Mutex<Cache> {
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Cache::default()))
}

pub fn get_or_compute(
    paths: &[PathBuf],
    backend: BackendKind,
    layers: &[i64],
    width: i64,
    ids: &[i32],
    compute: impl FnOnce() -> Result<Vec<f32>, WeightError>,
) -> Result<Vec<f32>, WeightError> {
    let key = Key {
        revision: crate::weights::revision(paths)?,
        backend,
        layers: layers.to_vec(),
        width,
        ids: ids.to_vec(),
    };
    {
        let mut cache = cache().lock().expect("conditioning cache");
        if let Some(index) = cache.entries.iter().position(|entry| entry.key == key) {
            let entry = cache.entries.remove(index).expect("cache entry");
            let values = entry.values.clone();
            cache.entries.push_front(entry);
            cache.hits += 1;
            return Ok(values);
        }
        cache.misses += 1;
    }
    let values = compute()?;
    let mut cache = cache().lock().expect("conditioning cache");
    while cache.entries.len() >= CAPACITY {
        cache.entries.pop_back();
    }
    cache.entries.push_front(Entry { key, values });
    Ok(cache.entries[0].values.clone())
}

pub fn resident_bytes() -> usize {
    cache()
        .lock()
        .expect("conditioning cache")
        .entries
        .iter()
        .map(|entry| entry.values.len() * std::mem::size_of::<f32>())
        .sum()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConditioningStats {
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
}

pub fn stats() -> ConditioningStats {
    let cache = cache().lock().expect("conditioning cache");
    ConditioningStats {
        entries: cache.entries.len(),
        hits: cache.hits,
        misses: cache.misses,
    }
}

#[cfg(test)]
pub fn clear() {
    let mut cache = cache().lock().expect("conditioning cache");
    cache.entries.clear();
    cache.hits = 0;
    cache.misses = 0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_reuses_embedding_and_invalidates_on_file_revision() {
        clear();
        let dir = std::env::temp_dir().join(format!("danmo-conditioning-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("encoder.gguf");
        std::fs::write(&path, b"one").unwrap();
        let paths = vec![path.clone()];
        let calls = std::cell::Cell::new(0);
        let mut compute = || {
            calls.set(calls.get() + 1);
            Ok(vec![1.0, 2.0])
        };
        let first =
            get_or_compute(&paths, BackendKind::Cpu, &[2], 2, &[4, 5], &mut compute).unwrap();
        let second =
            get_or_compute(&paths, BackendKind::Cpu, &[2], 2, &[4, 5], &mut compute).unwrap();
        assert_eq!(first, second);
        assert_eq!(calls.get(), 1);
        assert_eq!(stats().hits, 1);
        assert_eq!(stats().misses, 1);
        assert!(resident_bytes() > 0);
        std::fs::write(&path, b"two").unwrap();
        let _ = get_or_compute(&paths, BackendKind::Cpu, &[2], 2, &[4, 5], &mut compute).unwrap();
        assert_eq!(calls.get(), 2);
        assert_eq!(stats().misses, 2);
        clear();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
