//! Strict bounded LRU tensor cache
//!
//! This cache is explicitly controlled by RAMforge and tracks exact byte costs.
//! It never silently exceeds its configured capacity. It is designed to be
//! reusable for tensor data, blocks, layers, MoE experts, etc.

use std::collections::{HashMap, VecDeque};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

use crate::error::CacheError;
use crate::memory::MemoryBudget;

/// Statistics for cache operations
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub current_bytes: u64,
    pub capacity_bytes: u64,
    pub num_entries: usize,
}

/// Entry in the cache
#[derive(Debug, Clone)]
struct CacheEntry {
    data: Vec<u8>,
    size: u64,
    // For LRU we track last access order via external structure
}

/// Strict bounded LRU cache
///
/// - `capacity` is max bytes
/// - Every entry has an exact checked byte cost derived from `data.len()`
/// - Insertion respects capacity, evicting LRU entries when needed
/// - Oversized entry (larger than capacity) fails with `CacheError::TooLarge`
/// - Never exceeds capacity
#[derive(Debug)]
pub struct BoundedCache {
    capacity: u64,
    used: u64,
    entries: HashMap<String, CacheEntry>,
    // LRU order: front = most recently used, back = least recently used
    lru: VecDeque<String>,
    stats: CacheStats,
}

impl BoundedCache {
    /// Create a new cache with given capacity in bytes
    pub fn new(capacity_bytes: u64) -> Result<Self, CacheError> {
        if capacity_bytes == 0 {
            return Err(CacheError::General(
                "cache capacity must be > 0".to_string(),
            ));
        }
        Ok(Self {
            capacity: capacity_bytes,
            used: 0,
            entries: HashMap::new(),
            lru: VecDeque::new(),
            stats: CacheStats {
                capacity_bytes,
                ..Default::default()
            },
        })
    }

    pub fn capacity_bytes(&self) -> u64 {
        self.capacity
    }

    pub fn current_bytes(&self) -> u64 {
        self.used
    }

    pub fn available_bytes(&self) -> u64 {
        self.capacity
            .checked_sub(self.used)
            .expect("BoundedCache invariant violated: used bytes exceed capacity")
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Get a reference to cached data, updating LRU order and stats
    ///
    /// Returns None on miss (increments misses), Some on hit (increments hits)
    pub fn get(&mut self, key: &str) -> Option<&Vec<u8>> {
        if self.entries.contains_key(key) {
            // Update LRU: move to front
            self.lru.retain(|k| k != key);
            self.lru.push_front(key.to_string());
            self.stats.hits += 1;
            self.entries.get(key).map(|e| &e.data)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Get without updating stats (for internal use)
    pub fn get_without_stats(&self, key: &str) -> Option<&Vec<u8>> {
        self.entries.get(key).map(|e| &e.data)
    }

    /// Insert data into cache
    ///
    /// - If key already exists, it is replaced (old size freed)
    /// - If entry size > capacity, returns `TooLarge` error (policy: fail clearly)
    /// - Evicts LRU entries until enough space
    /// - Never exceeds capacity
    pub fn insert(&mut self, key: String, data: Vec<u8>) -> Result<(), CacheError> {
        let size = u64::try_from(data.len())
            .map_err(|_| CacheError::General("cache entry length exceeds u64".to_string()))?;

        if size > self.capacity {
            return Err(CacheError::TooLarge {
                size,
                capacity: self.capacity,
            });
        }

        // If key exists, remove old entry first
        if let Some(old) = self.entries.remove(&key) {
            let old_size = old.size;
            drop(old.data);
            self.used = self
                .used
                .checked_sub(old_size)
                .expect("BoundedCache accounting underflow while replacing entry");
            self.lru.retain(|k| k != &key);
        }

        // Evict until enough space.
        while self
            .used
            .checked_add(size)
            .expect("BoundedCache accounting overflow while inserting")
            > self.capacity
        {
            if let Some(lru_key) = self.lru.pop_back() {
                if let Some(entry) = self.entries.remove(&lru_key) {
                    let entry_size = entry.size;
                    drop(entry.data);
                    self.used = self
                        .used
                        .checked_sub(entry_size)
                        .expect("BoundedCache accounting underflow while evicting");
                    self.stats.evictions += 1;
                }
            } else {
                return Err(CacheError::General(
                    "cache LRU is empty while usage exceeds capacity".to_string(),
                ));
            }
        }

        if self
            .used
            .checked_add(size)
            .map_or(true, |used| used > self.capacity)
        {
            return Err(CacheError::General(
                "cache accounting cannot fit entry after eviction".to_string(),
            ));
        }

        // Insert.
        self.lru.push_front(key.clone());
        self.entries.insert(key, CacheEntry { data, size });
        self.used = self
            .used
            .checked_add(size)
            .expect("BoundedCache accounting overflow while inserting");

        self.update_stats();
        debug_assert!(self.used <= self.capacity);

        Ok(())
    }

    /// Remove an entry
    pub fn remove(&mut self, key: &str) -> Option<Vec<u8>> {
        if let Some(entry) = self.entries.remove(key) {
            let entry_size = entry.size;
            let data = entry.data;
            self.used = self
                .used
                .checked_sub(entry_size)
                .expect("BoundedCache accounting underflow while removing entry");
            self.lru.retain(|k| k != key);
            self.update_stats();
            Some(data)
        } else {
            None
        }
    }

    // ---------- Budget-charged operations ----------
    //
    // Every byte stored through these operations is also charged to the
    // supplied `MemoryBudget` under the name `cache:<key>`, so cached data
    // can never grow into an unaccounted second memory pool. The budget is
    // the authoritative limit; `capacity` remains a secondary hard cap.

    /// Budget-charged insert whose builder runs only after the cache charge is
    /// reserved. This is the managed entry point for reads or decodes that
    /// would otherwise allocate a `Vec<u8>` before accounting it.
    ///
    /// Evicts LRU entries until both the cache capacity and the memory budget
    /// have room (each eviction drops its data before releasing its charge).
    /// The builder's returned buffer must have exactly `size` bytes.
    pub fn insert_budgeted_with<F>(
        &mut self,
        budget: &mut MemoryBudget,
        key: String,
        size: u64,
        build: F,
    ) -> Result<bool, CacheError>
    where
        F: FnOnce() -> Result<Vec<u8>, CacheError>,
    {
        if size == 0 {
            return Err(CacheError::General(
                "budgeted cache entries must have positive size".to_string(),
            ));
        }
        if size > self.capacity {
            return Err(CacheError::TooLarge {
                size,
                capacity: self.capacity,
            });
        }

        // Replacing an existing key: drop its old data before releasing the
        // corresponding budget charge.
        if let Some(old) = self.entries.remove(&key) {
            let old_size = old.size;
            drop(old.data);
            self.used = self
                .used
                .checked_sub(old_size)
                .expect("BoundedCache accounting underflow while replacing budgeted entry");
            self.lru.retain(|k| k != &key);
            budget
                .release(&Self::budget_key(&key))
                .unwrap_or_else(|error| panic!("failed to release replaced cache charge: {error}"));
        }

        // Evict until capacity has room (releasing budget charges only after
        // each evicted buffer has been dropped).
        while self
            .used
            .checked_add(size)
            .expect("BoundedCache accounting overflow while inserting budgeted entry")
            > self.capacity
        {
            if !self.evict_lru_budgeted(budget) {
                return Err(CacheError::General(
                    "cache LRU is empty while usage exceeds capacity".to_string(),
                ));
            }
        }
        if self
            .used
            .checked_add(size)
            .map_or(true, |used| used > self.capacity)
        {
            return Err(CacheError::General(
                "cache accounting cannot fit entry after eviction".to_string(),
            ));
        }

        // Evict further until the budget has room.
        while !budget.can_allocate(size) {
            if !self.evict_lru_budgeted(budget) {
                // Nothing left to evict and budget still full: skip building
                // the entry so no unaccounted cache allocation is created.
                return Ok(false);
            }
        }

        budget
            .allocate(Self::budget_key(&key), size)
            .map_err(|e| CacheError::General(format!("budget charge failed: {}", e)))?;
        let data = match catch_unwind(AssertUnwindSafe(build)) {
            Ok(Ok(data)) => data,
            Ok(Err(error)) => {
                budget
                    .release(&Self::budget_key(&key))
                    .unwrap_or_else(|release_error| {
                        panic!("failed to roll back cache build charge: {release_error}")
                    });
                return Err(error);
            }
            Err(payload) => {
                budget
                    .release(&Self::budget_key(&key))
                    .unwrap_or_else(|release_error| {
                        panic!(
                            "failed to roll back cache build charge during unwind: {release_error}"
                        )
                    });
                resume_unwind(payload);
            }
        };
        let actual_size = match u64::try_from(data.len()) {
            Ok(actual_size) => actual_size,
            Err(_) => {
                drop(data);
                budget
                    .release(&Self::budget_key(&key))
                    .unwrap_or_else(|error| {
                        panic!("failed to release oversized cache build charge: {error}")
                    });
                return Err(CacheError::General(
                    "cache entry length exceeds u64".to_string(),
                ));
            }
        };
        if actual_size != size {
            drop(data);
            budget
                .release(&Self::budget_key(&key))
                .unwrap_or_else(|error| {
                    panic!("failed to release mismatched cache charge: {error}")
                });
            return Err(CacheError::General(format!(
                "cache builder returned {} bytes, expected {}",
                actual_size, size
            )));
        }

        self.lru.push_front(key.clone());
        self.entries.insert(key, CacheEntry { data, size });
        self.used = self
            .used
            .checked_add(size)
            .expect("BoundedCache accounting overflow after budgeted insert");
        self.update_stats();
        debug_assert!(self.used <= self.capacity);
        Ok(true)
    }

    /// Budget-charged insert of an already-created buffer. Prefer
    /// [`Self::insert_budgeted_with`] when the buffer can be built after the
    /// reservation; this compatibility form cannot retroactively protect the
    /// allocation performed by its caller.
    pub fn insert_budgeted(
        &mut self,
        budget: &mut MemoryBudget,
        key: String,
        data: Vec<u8>,
    ) -> Result<bool, CacheError> {
        let size = u64::try_from(data.len())
            .map_err(|_| CacheError::General("cache entry length exceeds u64".to_string()))?;
        self.insert_budgeted_with(budget, key, size, || Ok(data))
    }

    /// Remove an entry and release its budget charge.
    ///
    /// This legacy API returns an owned `Vec` after the cache charge is
    /// released, so the returned copy is outside `MemoryBudget` accounting.
    /// Managed callers that need lifetime coverage should retain a caller-owned
    /// output charge and use the cache only as the source of its bytes.
    pub fn remove_budgeted(&mut self, budget: &mut MemoryBudget, key: &str) -> Option<Vec<u8>> {
        let removed = self.remove(key);
        let data = removed?;
        // Preserve the legacy return type while ensuring the cache-owned
        // allocation is dropped before its charge is released. The returned
        // clone is caller-owned and intentionally outside this budget ledger.
        let mut returned = Vec::new();
        if returned.try_reserve_exact(data.len()).is_err() {
            drop(data);
            budget
                .release(&Self::budget_key(key))
                .unwrap_or_else(|error| {
                    panic!("failed to release removed cache charge: {error}")
                });
            return None;
        }
        returned.extend_from_slice(&data);
        drop(data);
        budget
            .release(&Self::budget_key(key))
            .unwrap_or_else(|error| panic!("failed to release removed cache charge: {error}"));
        Some(returned)
    }

    /// Clear the cache and release all budget charges after dropping the
    /// corresponding cached buffers.
    pub fn clear_budgeted(&mut self, budget: &mut MemoryBudget) {
        let entries = std::mem::take(&mut self.entries);
        self.lru.clear();
        self.used = 0;
        for (key, entry) in entries {
            drop(entry.data);
            budget
                .release(&Self::budget_key(&key))
                .unwrap_or_else(|error| panic!("failed to release cleared cache charge: {error}"));
        }
        self.update_stats();
    }

    fn budget_key(key: &str) -> String {
        format!("cache:{}", key)
    }

    /// Evict the least-recently-used entry, releasing its budget charge.
    /// Returns false if the cache was already empty.
    fn evict_lru_budgeted(&mut self, budget: &mut MemoryBudget) -> bool {
        if let Some(lru_key) = self.lru.pop_back() {
            if let Some(entry) = self.entries.remove(&lru_key) {
                let entry_size = entry.size;
                drop(entry.data);
                self.used = self
                    .used
                    .checked_sub(entry_size)
                    .expect("BoundedCache accounting underflow while budgeted eviction");
                self.stats.evictions += 1;
                budget
                    .release(&Self::budget_key(&lru_key))
                    .unwrap_or_else(|error| {
                        panic!("failed to release evicted cache charge: {error}")
                    });
                self.update_stats();
                return true;
            }
        }
        false
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
        self.used = 0;
        self.update_stats();
    }

    pub fn stats(&self) -> CacheStats {
        let mut s = self.stats.clone();
        s.current_bytes = self.used;
        s.capacity_bytes = self.capacity;
        s.num_entries = self.entries.len();
        s
    }

    fn update_stats(&mut self) {
        self.stats.current_bytes = self.used;
        self.stats.num_entries = self.entries.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn test_insert_and_hit() {
        let mut cache = BoundedCache::new(100).unwrap();
        cache.insert("a".to_string(), vec![0u8; 10]).unwrap();
        assert_eq!(cache.current_bytes(), 10);
        assert_eq!(cache.len(), 1);
        // Hit
        assert!(cache.get("a").is_some());
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 0);
        // Miss
        assert!(cache.get("b").is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = BoundedCache::new(30).unwrap();
        cache.insert("a".to_string(), vec![0u8; 10]).unwrap(); // used 10
        cache.insert("b".to_string(), vec![0u8; 10]).unwrap(); // used 20
        cache.insert("c".to_string(), vec![0u8; 10]).unwrap(); // used 30
        assert_eq!(cache.len(), 3);

        // Access a to make it MRU, order now a,c,b? Actually after inserts order: c front, b, a back
        // Let's access a -> a becomes front, order a,c,b, LRU is b
        cache.get("a");
        // Insert d (10 bytes) -> should evict b (LRU)
        cache.insert("d".to_string(), vec![0u8; 10]).unwrap();
        assert_eq!(cache.len(), 3);
        assert!(cache.contains("a"));
        assert!(!cache.contains("b"), "b should have been evicted");
        assert!(cache.contains("c"));
        assert!(cache.contains("d"));
        assert_eq!(cache.stats().evictions, 1);
        assert!(cache.current_bytes() <= cache.capacity_bytes());
    }

    #[test]
    fn test_usage_never_exceeds_capacity() {
        let mut cache = BoundedCache::new(25).unwrap();
        for i in 0..10 {
            let key = format!("k{}", i);
            cache.insert(key, vec![0u8; 10]).unwrap();
            assert!(cache.current_bytes() <= cache.capacity_bytes());
        }
        // After many inserts, only 2 entries should remain (20 bytes)
        assert!(cache.current_bytes() <= 25);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_oversized_entry() {
        let mut cache = BoundedCache::new(10).unwrap();
        let result = cache.insert("big".to_string(), vec![0u8; 20]);
        match result {
            Err(CacheError::TooLarge {
                size: 20,
                capacity: 10,
            }) => {}
            _ => panic!("expected TooLarge error"),
        }
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.current_bytes(), 0);
    }

    #[test]
    fn test_replace_existing() {
        let mut cache = BoundedCache::new(30).unwrap();
        cache.insert("a".to_string(), vec![0u8; 10]).unwrap();
        assert_eq!(cache.current_bytes(), 10);
        cache.insert("a".to_string(), vec![0u8; 20]).unwrap();
        assert_eq!(cache.current_bytes(), 20);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_budgeted_insert_charges_budget() {
        let mut budget = MemoryBudget::new(1000).unwrap();
        let mut cache = BoundedCache::new(100).unwrap();
        assert!(cache
            .insert_budgeted(&mut budget, "a".to_string(), vec![0u8; 40])
            .unwrap());
        assert_eq!(cache.current_bytes(), 40);
        assert_eq!(
            budget.used_bytes(),
            40,
            "cached bytes must be budget-charged"
        );
        assert_eq!(budget.get("cache:a"), Some(40));
        assert!(cache
            .insert_budgeted(&mut budget, "b".to_string(), vec![0u8; 30])
            .unwrap());
        assert_eq!(budget.used_bytes(), 70);
        assert!(cache.remove_budgeted(&mut budget, "a").is_some());
        assert_eq!(
            budget.used_bytes(),
            30,
            "removal must release budget charge"
        );
    }

    #[test]
    fn test_budgeted_eviction_releases_budget() {
        let mut budget = MemoryBudget::new(1000).unwrap();
        let mut cache = BoundedCache::new(30).unwrap();
        cache
            .insert_budgeted(&mut budget, "a".into(), vec![0u8; 15])
            .unwrap();
        cache
            .insert_budgeted(&mut budget, "b".into(), vec![0u8; 15])
            .unwrap();
        assert_eq!(budget.used_bytes(), 30);
        // forces capacity-driven eviction of "a"
        cache
            .insert_budgeted(&mut budget, "c".into(), vec![0u8; 15])
            .unwrap();
        assert!(!cache.contains("a"));
        assert_eq!(
            budget.used_bytes(),
            30,
            "evicted entry must release its charge"
        );
        assert!(budget.get("cache:a").is_none());
    }

    #[test]
    fn test_budgeted_insert_skips_when_budget_full() {
        let mut budget = MemoryBudget::new(50).unwrap();
        budget.allocate("other", 45).unwrap();
        let mut cache = BoundedCache::new(100).unwrap();
        let built = Cell::new(false);
        let cached = cache
            .insert_budgeted_with(&mut budget, "x".to_string(), 10, || {
                built.set(true);
                Ok(vec![0u8; 10])
            })
            .unwrap();
        assert!(!cached, "must skip caching when budget cannot fit the entry");
        assert!(!built.get(), "builder must run only after reservation");
        assert!(!cache.contains("x"));
        assert_eq!(budget.used_bytes(), 45);
    }

    #[test]
    fn test_budgeted_builder_failure_rolls_back_reservation() {
        let mut budget = MemoryBudget::new(100).unwrap();
        budget.allocate("other", 25).unwrap();
        let mut cache = BoundedCache::new(100).unwrap();
        let result = cache.insert_budgeted_with(&mut budget, "failed".into(), 50, || {
            Err(CacheError::General("synthetic read failure".to_string()))
        });
        assert!(result.is_err());
        assert_eq!(budget.used_bytes(), 25);
        assert!(budget.get("cache:failed").is_none());
        assert!(!cache.contains("failed"));
    }

    #[test]
    fn test_budgeted_builder_panic_rolls_back_reservation() {
        let mut budget = MemoryBudget::new(1000).unwrap();
        let mut cache = BoundedCache::new(100).unwrap();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = cache.insert_budgeted_with(&mut budget, "panic".into(), 50, || {
                panic!("synthetic cache-builder panic");
            });
        }));
        assert!(panic.is_err());
        assert_eq!(budget.used_bytes(), 0);
        assert!(budget.get("cache:panic").is_none());
        assert!(cache.is_empty());
    }

    #[test]
    fn test_budgeted_builder_size_mismatch_rolls_back_reservation() {
        let mut budget = MemoryBudget::new(1000).unwrap();
        budget.allocate("other", 25).unwrap();
        let mut cache = BoundedCache::new(100).unwrap();
        let result = cache.insert_budgeted_with(&mut budget, "mismatch".into(), 50, || {
            Ok(vec![0u8; 49])
        });
        assert!(result.is_err());
        assert_eq!(budget.used_bytes(), 25);
        assert!(budget.get("cache:mismatch").is_none());
        assert!(cache.is_empty());
    }

    #[test]
    fn test_clear_budgeted_releases_everything() {
        let mut budget = MemoryBudget::new(1000).unwrap();
        let mut cache = BoundedCache::new(100).unwrap();
        cache
            .insert_budgeted(&mut budget, "a".into(), vec![0u8; 10])
            .unwrap();
        cache
            .insert_budgeted(&mut budget, "b".into(), vec![0u8; 20])
            .unwrap();
        assert_eq!(budget.used_bytes(), 30);
        cache.clear_budgeted(&mut budget);
        assert_eq!(budget.used_bytes(), 0);
        assert_eq!(cache.current_bytes(), 0);
    }

    #[test]
    fn test_large_data_source_incremental_access() {
        // Simulate a data source larger than cache
        // Create synthetic file of 1MB, but cache only 100KB
        // Demonstrate that we can open and access incrementally
        let mut cache = BoundedCache::new(100 * 1024).unwrap();
        // Simulate reading 10 chunks of 100KB each from a 1MB source
        for i in 0..10 {
            let key = format!("chunk{}", i);
            let data = vec![i as u8; 100 * 1024];
            // Before insert, cache may need to evict
            cache.insert(key, data).unwrap();
            assert!(cache.current_bytes() <= cache.capacity_bytes());
        }
        // Only 1 chunk remains due to capacity
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.stats().evictions, 9);
    }
}
