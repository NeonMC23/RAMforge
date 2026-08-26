//! Strict bounded LRU tensor cache
//!
//! This cache is explicitly controlled by RAMforge and tracks exact byte costs.
//! It never silently exceeds its configured capacity. It is designed to be
//! reusable for tensor data, blocks, layers, MoE experts, etc.

use std::collections::{HashMap, VecDeque};

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
/// - Every entry has exact byte cost (data.len() as u64)
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
        self.capacity.saturating_sub(self.used)
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
        let size = data.len() as u64;

        if size > self.capacity {
            return Err(CacheError::TooLarge {
                size,
                capacity: self.capacity,
            });
        }

        // If key exists, remove old entry first
        if let Some(old) = self.entries.remove(&key) {
            self.used = self.used.saturating_sub(old.size);
            self.lru.retain(|k| k != &key);
        }

        // Evict until enough space
        while self.used + size > self.capacity {
            if let Some(lru_key) = self.lru.pop_back() {
                if let Some(entry) = self.entries.remove(&lru_key) {
                    self.used = self.used.saturating_sub(entry.size);
                    self.stats.evictions += 1;
                }
            } else {
                // Should not happen if size <= capacity, but break to avoid infinite loop
                break;
            }
        }

        // Insert
        self.lru.push_front(key.clone());
        self.entries.insert(key, CacheEntry { data, size });
        self.used += size;

        self.update_stats();
        debug_assert!(self.used <= self.capacity);

        Ok(())
    }

    /// Remove an entry
    pub fn remove(&mut self, key: &str) -> Option<Vec<u8>> {
        if let Some(entry) = self.entries.remove(key) {
            self.used = self.used.saturating_sub(entry.size);
            self.lru.retain(|k| k != key);
            self.update_stats();
            Some(entry.data)
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

    /// Budget-charged insert.
    ///
    /// Evicts LRU entries until both the cache capacity and the memory
    /// budget have room (each eviction releases its budget charge).
    ///
    /// Returns:
    /// - `Ok(true)` if the entry was cached (and charged to the budget)
    /// - `Ok(false)` if the budget cannot fit the entry even after evicting
    ///   everything (the entry is simply not cached; callers can fall back
    ///   to reading from disk)
    /// - `Err(CacheError::TooLarge)` if the entry exceeds the cache capacity
    pub fn insert_budgeted(
        &mut self,
        budget: &mut MemoryBudget,
        key: String,
        data: Vec<u8>,
    ) -> Result<bool, CacheError> {
        let size = data.len() as u64;
        if size > self.capacity {
            return Err(CacheError::TooLarge {
                size,
                capacity: self.capacity,
            });
        }

        // Replacing an existing key: release its old budget charge first.
        if let Some(old) = self.entries.remove(&key) {
            self.used = self.used.saturating_sub(old.size);
            self.lru.retain(|k| k != &key);
            let _ = budget.release(&Self::budget_key(&key));
        }

        // Evict until capacity has room (releasing budget charges).
        while self.used + size > self.capacity {
            if !self.evict_lru_budgeted(budget) {
                break;
            }
        }

        // Evict further until the budget has room.
        while !budget.can_allocate(size) {
            if !self.evict_lru_budgeted(budget) {
                // Nothing left to evict and budget still full: skip caching
                // instead of failing – streaming must keep working.
                return Ok(false);
            }
        }

        budget
            .allocate(Self::budget_key(&key), size)
            .map_err(|e| CacheError::General(format!("budget charge failed: {}", e)))?;

        self.lru.push_front(key.clone());
        self.entries.insert(key, CacheEntry { data, size });
        self.used += size;
        self.update_stats();
        debug_assert!(self.used <= self.capacity);
        Ok(true)
    }

    /// Remove an entry and release its budget charge.
    pub fn remove_budgeted(&mut self, budget: &mut MemoryBudget, key: &str) -> Option<Vec<u8>> {
        let removed = self.remove(key);
        if removed.is_some() {
            let _ = budget.release(&Self::budget_key(key));
        }
        removed
    }

    /// Clear the cache and release all budget charges.
    pub fn clear_budgeted(&mut self, budget: &mut MemoryBudget) {
        let keys: Vec<String> = self.entries.keys().cloned().collect();
        for k in keys {
            let _ = budget.release(&Self::budget_key(&k));
        }
        self.clear();
    }

    fn budget_key(key: &str) -> String {
        format!("cache:{}", key)
    }

    /// Evict the least-recently-used entry, releasing its budget charge.
    /// Returns false if the cache was already empty.
    fn evict_lru_budgeted(&mut self, budget: &mut MemoryBudget) -> bool {
        if let Some(lru_key) = self.lru.pop_back() {
            if let Some(entry) = self.entries.remove(&lru_key) {
                self.used = self.used.saturating_sub(entry.size);
                self.stats.evictions += 1;
                let _ = budget.release(&Self::budget_key(&lru_key));
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
        let cached = cache
            .insert_budgeted(&mut budget, "x".to_string(), vec![0u8; 10])
            .unwrap();
        assert!(
            !cached,
            "must skip caching when budget cannot fit the entry"
        );
        assert!(!cache.contains("x"));
        assert_eq!(budget.used_bytes(), 45);
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
