//! RAMforge runtime for budgeted, profiled out-of-core inference.
//!
//! This crate builds on `ramforge-core` to provide:
//! - Memory budget enforcement (RAII-style scoped temp reservations)
//! - File-backed tensor data source
//! - Bounded LRU cache whose contents are charged to the budget
//! - Planning logic for `ramforge plan`
//! - Explicit discovery → Planner → compiler → runtime orchestration
//! - Versioned, bounded ExecutionPlan persistence and compatibility validation
//! - CPU inference for llama/qwen2 (F32/F16/BF16 + ggml quant formats),
//!   out-of-core layer streaming with compact quantized residency
//! - SIMD (AVX2) + rayon-threaded F32 matvec hot path
//!
//! RAMforge-managed memory is defined as memory explicitly tracked via
//! `MemoryBudget`. It does NOT include total process RSS or OS page cache.

pub(crate) mod accounting;
pub mod backend;
pub mod calibration;
pub mod discovery;
pub mod inference;
pub mod kv_cache;
pub mod layer;
pub(crate) mod layer_cache;
pub(crate) mod layer_read;
pub mod memory_report;
pub mod model;
pub mod ops;
pub mod orchestration;
pub mod persistent;
pub mod plan;
pub mod plan_compiler;
pub mod plan_persistence;
pub mod planner;
pub mod profile;
pub mod residency;
pub mod runtime_config;
pub mod sampling;
pub mod simd;
pub mod streaming_model;
pub mod support;

use ramforge_core::{
    cache::BoundedCache, datasource::GgufDataSource, memory::MemoryBudget, GgufModel,
};

pub use ramforge_core::{
    CacheError, CacheStats, DataSourceError, GgufError, MemoryError, ParseSizeError,
};
pub use orchestration::{
    OrchestratedRuntime, OrchestrationError, OrchestrationRequest, PlanningSession,
    RuntimeOrchestrator,
};
pub use plan::{plan_model, PlanResult};
pub use plan_compiler::{PlanCompilationContext, PlanCompilationError, PlanCompiler};
pub use plan_persistence::{
    PersistedExecutionPlan, PersistedPlanCompatibilityContext, PersistedPlanValidationError,
    PlanPersistenceError, PERSISTED_EXECUTION_PLAN_SCHEMA_VERSION,
};
pub use runtime_config::{RuntimeConfig, RuntimeConfigError, RuntimeExecutionDevice};

/// Basic raw-tensor cache runtime retained as a public utility.
///
/// This is separate from `InferenceEngine` and its generation-local,
/// decoded-layer cache; it is not the model execution planner.
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
        let budget = MemoryBudget::new(ram_budget_bytes)?;

        // The cache capacity is a hard bound; its contents are charged to the
        // budget per entry via `insert_budgeted` – no double-counted capacity
        // pre-reservation.
        let cache_capacity = (ram_budget_bytes as f64 * 0.8) as u64;
        let cache_capacity = cache_capacity.max(1024 * 1024).min(ram_budget_bytes.saturating_sub(1024 * 1024));
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
        // Budget-charged insert: if the budget has no room even after LRU
        // eviction, the entry is simply not cached (Ok(false)) and the
        // caller still gets the data it asked for.
        match self
            .cache
            .insert_budgeted(&mut self.budget, name.to_string(), data.clone())
        {
            Ok(_) => {},
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
