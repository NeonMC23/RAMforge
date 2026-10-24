//! Residency observability for out-of-core layer streaming.
//!
//! These execution counters complement the optional timing profiler. Exact
//! allocation authority remains with `MemoryBudget`.

#[derive(Debug, Clone, Default)]
pub struct ResidencyStats {
    /// Total model weight bytes (sum of all tensors)
    pub total_model_weight_bytes: u64,
    /// Current resident layer weight bytes
    pub current_resident_layer_bytes: u64,
    /// Peak resident layer weight bytes observed
    pub peak_resident_layer_bytes: u64,
    /// Number of layer loads
    pub num_layer_loads: u64,
    /// Number of active streamed layers whose charges were released.
    pub num_layer_releases: u64,
    /// Cumulative number of active layers transitioned into cache residency.
    pub num_layer_cached: u64,
    /// Current number of cached streamed layers.
    pub current_cached_layer_count: u64,
    /// Peak number of cached streamed layers.
    pub peak_cached_layer_count: u64,
    /// Current bytes retained by the streamed-layer cache.
    pub current_cached_bytes: u64,
    /// Peak bytes retained by the streamed-layer cache.
    pub peak_cached_bytes: u64,
    /// Cache lookups served by a resident layer.
    pub cache_hits: u64,
    /// Cache lookups that required a load.
    pub cache_misses: u64,
    /// Layer entries evicted to satisfy cache capacity or budget pressure.
    pub cache_evictions: u64,
    /// Peak RAMforge-managed bytes (budget used)
    pub peak_managed_bytes: u64,
    /// Current managed bytes
    pub current_managed_bytes: u64,
}

impl ResidencyStats {
    pub fn new(total_model_weight_bytes: u64) -> Self {
        Self {
            total_model_weight_bytes,
            ..Default::default()
        }
    }

    pub fn on_layer_load(&mut self, layer_bytes: u64, current_managed: u64) {
        self.num_layer_loads += 1;
        self.current_resident_layer_bytes = layer_bytes;
        if layer_bytes > self.peak_resident_layer_bytes {
            self.peak_resident_layer_bytes = layer_bytes;
        }
        self.current_managed_bytes = current_managed;
        if current_managed > self.peak_managed_bytes {
            self.peak_managed_bytes = current_managed;
        }
    }

    pub fn on_layer_release(&mut self, current_managed: u64) {
        self.num_layer_releases += 1;
        self.on_layer_inactive(current_managed);
    }

    /// The active layer became cache-resident rather than being released.
    pub fn on_layer_cached(&mut self, current_managed: u64) {
        self.num_layer_cached += 1;
        self.on_layer_inactive(current_managed);
    }

    pub fn on_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    pub fn on_cache_miss(&mut self) {
        self.cache_misses += 1;
    }

    pub fn on_cache_evictions(&mut self, count: usize) {
        self.cache_evictions = self.cache_evictions.saturating_add(count as u64);
    }

    /// Record the cache's exact current state after a hit, insertion, eviction,
    /// or clear. The counters are distinct from the cumulative placement count.
    pub fn on_cache_state(&mut self, entries: usize, bytes: u64) {
        self.current_cached_layer_count = entries as u64;
        self.current_cached_bytes = bytes;
        if self.current_cached_layer_count > self.peak_cached_layer_count {
            self.peak_cached_layer_count = self.current_cached_layer_count;
        }
        if bytes > self.peak_cached_bytes {
            self.peak_cached_bytes = bytes;
        }
    }

    fn on_layer_inactive(&mut self, current_managed: u64) {
        self.current_resident_layer_bytes = 0;
        self.current_managed_bytes = current_managed;
        if current_managed > self.peak_managed_bytes {
            self.peak_managed_bytes = current_managed;
        }
    }

    pub fn update_managed(&mut self, current_managed: u64) {
        self.current_managed_bytes = current_managed;
        if current_managed > self.peak_managed_bytes {
            self.peak_managed_bytes = current_managed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residency_tracking() {
        let mut stats = ResidencyStats::new(1000);
        stats.on_layer_load(100, 500);
        assert_eq!(stats.current_resident_layer_bytes, 100);
        assert_eq!(stats.peak_resident_layer_bytes, 100);
        assert_eq!(stats.num_layer_loads, 1);
        stats.on_layer_load(200, 600);
        assert_eq!(stats.peak_resident_layer_bytes, 200);
        assert_eq!(stats.num_layer_loads, 2);
        stats.on_layer_release(400);
        assert_eq!(stats.current_resident_layer_bytes, 0);
        assert_eq!(stats.num_layer_releases, 1);
        stats.on_layer_cached(550);
        assert_eq!(stats.num_layer_cached, 1);
        assert_eq!(stats.peak_managed_bytes, 600);
    }

    #[test]
    fn test_cache_state_is_current_and_peak_not_cumulative() {
        let mut stats = ResidencyStats::new(1000);
        stats.on_cache_miss();
        stats.on_cache_state(1, 100);
        stats.on_cache_hit();
        stats.on_cache_state(1, 100);
        stats.on_cache_state(3, 300);
        stats.on_cache_evictions(2);
        stats.on_cache_state(1, 100);

        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_evictions, 2);
        assert_eq!(stats.current_cached_layer_count, 1);
        assert_eq!(stats.peak_cached_layer_count, 3);
        assert_eq!(stats.current_cached_bytes, 100);
        assert_eq!(stats.peak_cached_bytes, 300);
    }
}
