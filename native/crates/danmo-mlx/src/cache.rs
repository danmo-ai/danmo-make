//! Resident model slots owned by the MLX executor thread.
//!
//! Keys are an opaque id plus a content revision. This crate does not know
//! family ids. Values stay on the executor thread (`Box<dyn Any>`, not
//! `Send`). Callers report resident bytes; the cache does not inspect them.

use std::any::Any;
use std::collections::{HashMap, VecDeque};

use crate::{Error, Result};

/// File identity that invalidates a resident model. Callers fill `length`
/// and `modified_ns` from the bytes on disk. `mix` is an extra stamp for a
/// directory of shards (names and per-file lengths) so a rename still misses.
/// Plain data, so it may cross threads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheRevision {
    pub length: u64,
    pub modified_ns: u64,
    pub mix: u64,
}

/// One resident slot. `id` is an opaque path or bundle stamp, not a family id.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub id: String,
    pub revision: CacheRevision,
}

struct Slot {
    revision: CacheRevision,
    bytes: usize,
    value: Box<dyn Any>,
}

/// LRU cache with a byte budget. A single entry larger than the budget is
/// rejected and the cache is left unchanged. Eviction only drops other slots
/// to make a legal insert fit, or to honor `set_budget`.
pub struct ModelCache {
    budget_bytes: usize,
    used_bytes: usize,
    order: VecDeque<String>,
    slots: HashMap<String, Slot>,
}

impl ModelCache {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            used_bytes: 0,
            order: VecDeque::new(),
            slots: HashMap::new(),
        }
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Borrow a resident value when the revision matches. A mismatch drops
    /// the stale slot: it must not be served for this key. A type mismatch
    /// fails; it is not reported as a miss.
    pub fn get<T: Any>(&mut self, key: &CacheKey) -> Result<Option<&T>> {
        if !self.fresh(key) {
            return Ok(None);
        }
        self.touch(&key.id);
        match self
            .slots
            .get(&key.id)
            .map(|slot| slot.value.downcast_ref::<T>())
        {
            Some(Some(value)) => Ok(Some(value)),
            Some(None) => Err(Error::new(
                "ModelCache::get",
                format!("slot {} is resident with a different type", key.id),
            )),
            None => Ok(None),
        }
    }

    pub fn get_mut<T: Any>(&mut self, key: &CacheKey) -> Result<Option<&mut T>> {
        if !self.fresh(key) {
            return Ok(None);
        }
        self.touch(&key.id);
        match self
            .slots
            .get_mut(&key.id)
            .map(|slot| slot.value.downcast_mut::<T>())
        {
            Some(Some(value)) => Ok(Some(value)),
            Some(None) => Err(Error::new(
                "ModelCache::get",
                format!("slot {} is resident with a different type", key.id),
            )),
            None => Ok(None),
        }
    }

    /// Insert or replace the slot for `key.id`. Fails without mutation when
    /// `bytes` alone exceeds the budget.
    pub fn insert<T: Any>(&mut self, key: CacheKey, value: T, bytes: usize) -> Result<()> {
        if bytes > self.budget_bytes {
            return Err(Error::new(
                "ModelCache::insert",
                format!(
                    "entry {} is {bytes} bytes, over the {} byte budget",
                    key.id, self.budget_bytes
                ),
            ));
        }
        let old = self.slots.get(&key.id).map(|slot| slot.bytes).unwrap_or(0);
        let mut used = self.used_bytes - old;
        let mut victims = Vec::new();
        let mut remaining: VecDeque<String> = self
            .order
            .iter()
            .filter(|id| id.as_str() != key.id)
            .cloned()
            .collect();
        while used + bytes > self.budget_bytes {
            let Some(victim) = remaining.pop_front() else {
                return Err(Error::new(
                    "ModelCache::insert",
                    format!(
                        "entry {} of {bytes} bytes does not fit in the {} byte budget",
                        key.id, self.budget_bytes
                    ),
                ));
            };
            let victim_bytes = self
                .slots
                .get(&victim)
                .map(|slot| slot.bytes)
                .ok_or_else(|| {
                    Error::new(
                        "ModelCache::insert",
                        format!("lru slot {victim} is missing"),
                    )
                })?;
            used -= victim_bytes;
            victims.push(victim);
        }
        for victim in victims {
            self.remove_slot(&victim);
        }
        if self.slots.contains_key(&key.id) {
            self.remove_slot(&key.id);
        }
        self.used_bytes += bytes;
        self.order.push_back(key.id.clone());
        self.slots.insert(
            key.id,
            Slot {
                revision: key.revision,
                bytes,
                value: Box::new(value),
            },
        );
        Ok(())
    }

    pub fn unload(&mut self, id: &str) -> bool {
        self.remove_slot(id)
    }

    pub fn clear(&mut self) {
        self.slots.clear();
        self.order.clear();
        self.used_bytes = 0;
    }

    /// Shrink or grow the budget. A smaller budget evicts least-recent slots
    /// until the resident set fits, including a single slot that is now over
    /// the new budget.
    pub fn set_budget(&mut self, budget_bytes: usize) {
        self.budget_bytes = budget_bytes;
        while self.used_bytes > self.budget_bytes {
            let Some(oldest) = self.order.front().cloned() else {
                break;
            };
            self.remove_slot(&oldest);
        }
    }

    fn fresh(&mut self, key: &CacheKey) -> bool {
        match self.slots.get(&key.id) {
            Some(slot) if slot.revision == key.revision => true,
            Some(_) => {
                self.remove_slot(&key.id);
                false
            }
            None => false,
        }
    }

    fn touch(&mut self, id: &str) {
        if let Some(index) = self.order.iter().position(|item| item == id) {
            let id = self.order.remove(index).expect("lru index");
            self.order.push_back(id);
        }
    }

    fn remove_slot(&mut self, id: &str) -> bool {
        let Some(slot) = self.slots.remove(id) else {
            return false;
        };
        self.used_bytes -= slot.bytes;
        if let Some(index) = self.order.iter().position(|item| item == id) {
            self.order.remove(index);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::{CacheKey, CacheRevision, ModelCache};

    fn key(id: &str, modified_ns: u64) -> CacheKey {
        CacheKey {
            id: id.to_string(),
            revision: CacheRevision {
                length: 10,
                modified_ns,
                mix: 0,
            },
        }
    }

    #[test]
    fn revision_mismatch_drops_the_stale_slot() {
        let mut cache = ModelCache::new(100);
        cache
            .insert(key("weights", 1), "v1".to_string(), 40)
            .unwrap();
        assert!(cache.get::<String>(&key("weights", 2)).unwrap().is_none());
        assert_eq!(cache.used_bytes(), 0);
        assert!(cache.get::<String>(&key("weights", 1)).unwrap().is_none());
    }

    #[test]
    fn oversized_insert_leaves_the_cache_unchanged() {
        let mut cache = ModelCache::new(50);
        cache
            .insert(key("keep", 1), "stay".to_string(), 20)
            .unwrap();
        let error = cache
            .insert(key("huge", 1), "nope".to_string(), 80)
            .unwrap_err();
        assert!(error.to_string().contains("over the 50 byte budget"));
        assert_eq!(
            cache
                .get::<String>(&key("keep", 1))
                .unwrap()
                .map(String::as_str),
            Some("stay")
        );
        assert_eq!(cache.used_bytes(), 20);
    }

    #[test]
    fn insert_evicts_least_recent_until_the_budget_fits() {
        let mut cache = ModelCache::new(100);
        cache.insert(key("a", 1), "a".to_string(), 30).unwrap();
        cache.insert(key("b", 1), "b".to_string(), 30).unwrap();
        cache.get::<String>(&key("a", 1)).unwrap();
        cache.insert(key("c", 1), "c".to_string(), 50).unwrap();
        assert!(cache.get::<String>(&key("b", 1)).unwrap().is_none());
        assert_eq!(
            cache
                .get::<String>(&key("a", 1))
                .unwrap()
                .map(String::as_str),
            Some("a")
        );
        assert_eq!(
            cache
                .get::<String>(&key("c", 1))
                .unwrap()
                .map(String::as_str),
            Some("c")
        );
        assert_eq!(cache.used_bytes(), 80);
    }

    #[test]
    fn unload_and_set_budget_free_resident_bytes() {
        let mut cache = ModelCache::new(100);
        cache.insert(key("a", 1), "a".to_string(), 60).unwrap();
        cache.insert(key("b", 1), "b".to_string(), 30).unwrap();
        assert!(cache.unload("a"));
        assert_eq!(cache.used_bytes(), 30);
        cache.set_budget(10);
        assert!(cache.is_empty());
        assert_eq!(cache.budget_bytes(), 10);
    }
}
