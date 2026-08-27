//! Bounded LRU cache for already-decoded streamed layer representations.
//!
//! Entries arrive with live `layer:{index}:*` MemoryBudget charges. Successful
//! insertion atomically renames those charges to `cache:layer:{index}:*`;
//! eviction and clear release the exact renamed charges.

use std::collections::{HashMap, VecDeque};

use ramforge_core::memory::MemoryBudget;

#[derive(Debug)]
struct CacheEntry<T> {
    value: T,
    bytes: u64,
    charge_names: Vec<String>,
}

#[derive(Debug)]
pub(crate) enum InsertOutcome<T> {
    Cached { evictions: usize },
    Skipped { value: T, evictions: usize },
}

#[derive(Debug)]
pub(crate) struct LayerCache<T> {
    capacity_bytes: u64,
    used_bytes: u64,
    entries: HashMap<usize, CacheEntry<T>>,
    /// Front is most recently used; back is least recently used.
    lru: VecDeque<usize>,
}

impl<T> LayerCache<T> {
    pub(crate) fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            entries: HashMap::new(),
            lru: VecDeque::new(),
        }
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    pub(crate) fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    #[cfg(test)]
    pub(crate) fn contains(&self, layer_index: usize) -> bool {
        self.entries.contains_key(&layer_index)
    }

    pub(crate) fn with_entry<R>(
        &mut self,
        layer_index: usize,
        use_entry: impl FnOnce(&T) -> R,
    ) -> Option<R> {
        if !self.entries.contains_key(&layer_index) {
            return None;
        }
        self.touch(layer_index);
        self.entries
            .get(&layer_index)
            .map(|entry| use_entry(&entry.value))
    }

    pub(crate) fn insert_loaded(
        &mut self,
        layer_index: usize,
        value: T,
        bytes: u64,
        budget: &mut MemoryBudget,
    ) -> Result<InsertOutcome<T>, String> {
        if bytes == 0 || bytes > self.capacity_bytes {
            return Ok(InsertOutcome::Skipped { value, evictions: 0 });
        }
        let active_prefix = format!("layer:{}:", layer_index);
        let active_charge_bytes = budget
            .allocations()
            .iter()
            .filter(|(name, _)| name.starts_with(&active_prefix))
            .try_fold(0u64, |total, (_, bytes)| total.checked_add(*bytes));
        if active_charge_bytes != Some(bytes) {
            return Ok(InsertOutcome::Skipped { value, evictions: 0 });
        }

        let mut evictions = 0usize;
        if self.entries.contains_key(&layer_index) {
            self.evict_layer(layer_index, budget)?;
            evictions += 1;
        }
        while self.used_bytes.saturating_add(bytes) > self.capacity_bytes {
            if !self.evict_lru(budget)? {
                return Ok(InsertOutcome::Skipped { value, evictions });
            }
            evictions += 1;
        }

        let new_used_bytes = self
            .used_bytes
            .checked_add(bytes)
            .ok_or_else(|| "layer cache byte accounting overflow".to_string())?;
        let cache_prefix = format!("cache:layer:{}:", layer_index);
        let charge_names = match budget.rename_prefix(&active_prefix, &cache_prefix) {
            Ok(names) => names,
            Err(error) => {
                // The value is about to be dropped. Release any still-active
                // layer charges so an internal rename error cannot leak budget.
                release_budget_prefix(budget, &active_prefix);
                return Err(format!(
                    "failed to convert layer {} charges to cache residency: {}",
                    layer_index, error
                ));
            }
        };

        self.used_bytes = new_used_bytes;
        self.lru.push_front(layer_index);
        self.entries.insert(
            layer_index,
            CacheEntry {
                value,
                bytes,
                charge_names,
            },
        );
        debug_assert!(self.used_bytes <= self.capacity_bytes);
        Ok(InsertOutcome::Cached { evictions })
    }

    /// Evict LRU entries until `required_bytes` can be allocated in the shared
    /// budget, or until the cache is empty. Returns the eviction count.
    pub(crate) fn evict_until_available(
        &mut self,
        budget: &mut MemoryBudget,
        required_bytes: u64,
    ) -> Result<usize, String> {
        let mut evictions = 0usize;
        while !budget.can_allocate(required_bytes) {
            if !self.evict_lru(budget)? {
                break;
            }
            evictions += 1;
        }
        Ok(evictions)
    }

    pub(crate) fn clear(&mut self, budget: &mut MemoryBudget) -> Result<usize, String> {
        let mut evictions = 0usize;
        while self.evict_lru(budget)? {
            evictions += 1;
        }
        Ok(evictions)
    }

    fn touch(&mut self, layer_index: usize) {
        self.lru.retain(|index| *index != layer_index);
        self.lru.push_front(layer_index);
    }

    fn evict_lru(&mut self, budget: &mut MemoryBudget) -> Result<bool, String> {
        let Some(&layer_index) = self.lru.back() else {
            return Ok(false);
        };
        self.release_entry(layer_index, budget)?;
        self.lru.pop_back();
        Ok(true)
    }

    fn evict_layer(
        &mut self,
        layer_index: usize,
        budget: &mut MemoryBudget,
    ) -> Result<(), String> {
        self.release_entry(layer_index, budget)?;
        self.lru.retain(|index| *index != layer_index);
        Ok(())
    }

    fn release_entry(
        &mut self,
        layer_index: usize,
        budget: &mut MemoryBudget,
    ) -> Result<(), String> {
        let entry = self
            .entries
            .get(&layer_index)
            .ok_or_else(|| format!("cached layer {} missing during eviction", layer_index))?;
        for charge_name in &entry.charge_names {
            if budget.get(charge_name).is_none() {
                return Err(format!(
                    "cached layer {} charge '{}' missing before eviction",
                    layer_index, charge_name
                ));
            }
        }
        let entry = self
            .entries
            .remove(&layer_index)
            .expect("cached layer validated above");
        for charge_name in &entry.charge_names {
            budget.release(charge_name).map_err(|error| {
                format!(
                    "failed to release cached layer {} charge '{}': {}",
                    layer_index, charge_name, error
                )
            })?;
        }
        self.used_bytes = self
            .used_bytes
            .checked_sub(entry.bytes)
            .ok_or_else(|| "layer cache byte accounting underflow".to_string())?;
        Ok(())
    }
}

fn release_budget_prefix(budget: &mut MemoryBudget, prefix: &str) {
    let names: Vec<String> = budget
        .allocations()
        .keys()
        .filter(|name| name.starts_with(prefix))
        .cloned()
        .collect();
    for name in names {
        let _ = budget.release(&name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn charge_layer(budget: &mut MemoryBudget, index: usize, bytes: u64) {
        budget
            .allocate(format!("layer:{}:weight", index), bytes)
            .unwrap();
    }

    #[test]
    fn test_cache_hit_and_lru_eviction_release_budget() {
        let mut budget = MemoryBudget::new(1000).unwrap();
        let mut cache = LayerCache::new(200);
        charge_layer(&mut budget, 0, 100);
        assert!(matches!(
            cache.insert_loaded(0, "zero", 100, &mut budget).unwrap(),
            InsertOutcome::Cached { .. }
        ));
        charge_layer(&mut budget, 1, 100);
        cache.insert_loaded(1, "one", 100, &mut budget).unwrap();

        assert_eq!(cache.with_entry(0, |value| *value), Some("zero"));
        charge_layer(&mut budget, 2, 100);
        let outcome = cache.insert_loaded(2, "two", 100, &mut budget).unwrap();
        assert!(matches!(outcome, InsertOutcome::Cached { evictions: 1 }));
        assert!(cache.contains(0));
        assert!(!cache.contains(1));
        assert!(cache.contains(2));
        assert!(budget.get("cache:layer:1:weight").is_none());
        assert_eq!(cache.used_bytes(), 200);
        assert!(cache.used_bytes() <= cache.capacity_bytes());
    }

    #[test]
    fn test_oversized_insert_is_skipped_without_accounting_corruption() {
        let mut budget = MemoryBudget::new(1000).unwrap();
        let mut cache = LayerCache::new(100);
        charge_layer(&mut budget, 3, 200);
        let before = budget.used_bytes();
        let outcome = cache.insert_loaded(3, "large", 200, &mut budget).unwrap();
        assert!(matches!(outcome, InsertOutcome::Skipped { value: "large", evictions: 0 }));
        assert_eq!(budget.used_bytes(), before);
        assert_eq!(budget.get("layer:3:weight"), Some(200));
        assert_eq!(cache.used_bytes(), 0);
    }

    #[test]
    fn test_headroom_eviction_and_clear_release_exact_charges() {
        let mut budget = MemoryBudget::new(300).unwrap();
        let mut cache = LayerCache::new(200);
        for index in 0..2 {
            charge_layer(&mut budget, index, 100);
            cache
                .insert_loaded(index, index, 100, &mut budget)
                .unwrap();
        }
        assert_eq!(budget.used_bytes(), 200);
        assert_eq!(cache.evict_until_available(&mut budget, 200).unwrap(), 1);
        assert_eq!(budget.used_bytes(), 100);
        assert_eq!(cache.entry_count(), 1);
        assert_eq!(cache.clear(&mut budget).unwrap(), 1);
        assert_eq!(budget.used_bytes(), 0);
        assert_eq!(cache.entry_count(), 0);
        assert_eq!(cache.used_bytes(), 0);
    }
}
