//! RAMforge Runtime – Milestone 3: First Real CPU Inference
//!
//! This crate builds on `ramforge-core` to provide:
//! - Memory budget enforcement
//! - File-backed tensor data source
//! - Strict bounded LRU cache with explicit accounting
//! - Planning logic for `ramforge plan`
//! - Real CPU inference for LLaMA architecture (F32/F16)
//!
//! RAMforge-managed memory is defined as memory explicitly tracked via
//! `MemoryBudget`. It does NOT include total process RSS or OS page cache.

pub mod backend;
pub mod inference;
pub mod kv_cache;
pub mod layer;
pub mod model;
pub mod ops;
pub mod plan;
pub mod residency;
pub mod sampling;
pub mod streaming_model;

use ramforge_core::{
    cache::BoundedCache, datasource::GgufDataSource, memory::MemoryBudget, GgufModel,
};

pub use ramforge_core::{
    CacheError, CacheStats, DataSourceError, GgufError, MemoryError, ParseSizeError,
};
pub use plan::{plan_model, PlanResult};

/// Runtime that owns a data source, budget, and cache
#[derive(Debug)]
pub struct Runtime {
    pub data_source: GgufDataSource,
    pub budget: MemoryBudget,
    pub cache: BoundedCache,
}

impl Runtime {
    pub fn new<P: AsRef<std::path::Path>>(
        model_path: P,
        ram_budget_bytes: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let data_source = GgufDataSource::open(&model_path)?;
        let mut budget = MemoryBudget::new(ram_budget_bytes)?;

        let cache_capacity = (ram_budget_bytes as f64 * 0.8) as u64;
        let cache_capacity = cache_capacity.max(1024 * 1024).min(ram_budget_bytes.saturating_sub(1024 * 1024));
        let overhead = (ram_budget_bytes as f64 * 0.1) as u64;

        budget.allocate("tensor_cache", cache_capacity)?;
        if overhead > 0 && budget.can_allocate(overhead) {
            budget.allocate("runtime_overhead", overhead)?;
        }

        let cache = BoundedCache::new(cache_capacity)?;

        Ok(Self {
            data_source,
            budget,
            cache,
        })
    }

    pub fn get_tensor(&mut self, name: &str) -> Result<Vec<u8>, DataSourceError> {
        if let Some(data) = self.cache.get(name) {
            return Ok(data.clone());
        }
        let data = self.data_source.read_tensor(name)?;
        match self.cache.insert(name.to_string(), data.clone()) {
            Ok(()) => {},
            Err(ramforge_core::CacheError::TooLarge { .. }) => {},
            Err(e) => {
                return Err(DataSourceError::General(format!("cache insert failed: {}", e)));
            }
        }
        Ok(data)
    }

    pub fn model(&self) -> &GgufModel {
        self.data_source.model()
    }

    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }
}
