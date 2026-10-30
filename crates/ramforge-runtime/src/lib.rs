//! RAMforge runtime for budgeted, profiled out-of-core inference.
//!
//! This crate builds on `ramforge-core` to provide:
//! - RAMforge-managed allocation accounting (scoped temporary reservations)
//! - File-backed tensor data source
//! - Bounded LRU cache whose contents are charged to the budget
//! - Planning logic for `ramforge plan`
//! - Explicit discovery → Planner → compiler → runtime orchestration
//! - Versioned, bounded ExecutionPlan persistence and compatibility validation
//! - CPU inference for llama/qwen2 (F32/F16/BF16 + ggml quant formats),
//!   out-of-core layer streaming with compact quantized residency
//! - SIMD (AVX2) + rayon-threaded F32 matvec hot path
//! - Static ForgeCore device discovery, selection, and capability diagnostics
//!
//! RAMforge-managed memory is defined as memory explicitly tracked via
//! `MemoryBudget`. It does NOT include total process RSS or OS page cache.

pub(crate) mod accounting;
pub mod backend;
pub mod cgroup;
pub mod calibration;
pub(crate) mod compute_dispatch;
pub mod discovery;
pub mod execution;
pub mod execution_profile;
pub mod forgecore;
pub mod graph_manifest;
pub mod inference;
pub mod kv_cache;
pub mod layer;
pub(crate) mod layer_cache;
pub(crate) mod layer_read;
pub mod memory_ledger;
pub mod memory_report;
pub mod model;
pub mod model_executor;
pub mod ops;
pub mod orchestration;
pub mod page_residency;
pub mod persistent;
pub mod plan;
pub mod plan_compiler;
pub mod plan_persistence;
pub mod planner;
pub mod profile;
pub mod residency;
pub mod residency_engine;
pub mod resource_lifetime;
pub mod runtime_config;
pub mod sampling;
pub mod simd;
pub mod streaming_model;
pub mod support;

use ramforge_core::{
    cache::BoundedCache, datasource::GgufDataSource, memory::MemoryBudget, GgufModel,
};

pub use crate::execution::{
    CapabilityState, DeviceCapabilities, DeviceInfo, DeviceKind, DeviceRegistry, DeviceSelection,
    DeviceSelectionError, ExecutionConfig, ExecutionConfigError, SelectedDevice,
};
pub use crate::forgecore::{
    ForgeCoreAdapterError, ForgeCoreContextOptions, ForgeCoreLogits, ForgeCoreModel,
    ForgeCoreModelMetadata, ForgeCoreModelOptions, ForgeCoreSession,
};
pub use forge_core::{PageReleaseResult, TensorMappingInfo};
pub use orchestration::{
    OrchestratedRuntime, OrchestrationError, OrchestrationRequest, PlanningSession,
    RuntimeOrchestrator,
};
pub use plan::{plan_model, PlanResult};
pub use plan_compiler::{PlanCompilationContext, PlanCompilationError, PlanCompiler};
pub use plan_persistence::{
    PersistedExecutionPlan, PersistedPlanCompatibilityContext, PersistedPlanValidationError,
    PlanCompatibilityReason, PlanCompatibilityReport, PlanPersistenceError,
    PERSISTED_EXECUTION_PLAN_SCHEMA_VERSION,
};
pub use ramforge_core::{
    CacheError, CacheStats, DataSourceError, GgufError, MemoryError, ParseSizeError,
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
    fn copy_returned_bytes(data: &[u8], name: &str) -> Result<Vec<u8>, DataSourceError> {
        let mut returned = Vec::new();
        returned.try_reserve_exact(data.len()).map_err(|error| {
            DataSourceError::General(format!(
                "failed to reserve compatibility tensor copy '{}': {}",
                name, error
            ))
        })?;
        returned.extend_from_slice(data);
        Ok(returned)
    }

    pub fn new<P: AsRef<std::path::Path>>(
        model_path: P,
        ram_budget_bytes: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let data_source = GgufDataSource::open(&model_path)?;
        let budget = MemoryBudget::new(ram_budget_bytes)?;

        // The cache capacity is a hard bound; its contents are charged to the
        // budget per entry via `insert_budgeted` – no double-counted capacity
        // pre-reservation. Keep the 80% policy in integer arithmetic so a
        // byte-size conversion cannot silently round or overflow.
        let cache_capacity_80 = (ram_budget_bytes / 5)
            .checked_mul(4)
            .and_then(|base| base.checked_add((ram_budget_bytes % 5) * 4 / 5))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "cache capacity arithmetic overflow",
                )
            })?;
        let cache_headroom = ram_budget_bytes.saturating_sub(1024 * 1024);
        let cache_capacity = cache_capacity_80
            .max(1024 * 1024)
            .min(cache_headroom);
        let cache = BoundedCache::new(cache_capacity)?;

        Ok(Self {
            data_source,
            budget,
            cache,
        })
    }

    /// Read a tensor, using the raw cache when possible.
    ///
    /// This compatibility API returns a caller-owned `Vec` clone. The cache
    /// charge covers only the cached copy and is not retained for the returned
    /// buffer; managed inference paths use caller-owned buffers with an
    /// enclosing charge instead.
    pub fn get_tensor(&mut self, name: &str) -> Result<Vec<u8>, DataSourceError> {
        if let Some(data) = self.cache.get(name) {
            return Self::copy_returned_bytes(data, name);
        }
        let file_bytes = self
            .data_source
            .get_descriptor(name)?
            .byte_length
            .ok_or_else(|| {
                DataSourceError::General(format!(
                    "cannot cache tensor '{}': byte length is unknown",
                    name
                ))
            })?;
        // The builder runs only after the cache charge is live. If the entry is
        // too large or the budget cannot fit it, read once as an explicitly
        // unmanaged compatibility fallback.
        let data_source = &self.data_source;
        let cache = &mut self.cache;
        let budget = &mut self.budget;
        let insertion = cache.insert_budgeted_with(
            budget,
            name.to_string(),
            file_bytes,
            || {
                data_source
                    .read_tensor(name)
                    .map_err(|error| ramforge_core::CacheError::General(error.to_string()))
            },
        );
        match insertion {
            Ok(true) => {
                let data = cache.get(name).ok_or_else(|| {
                    DataSourceError::General(format!(
                        "cache reported insertion for '{}' but the entry was missing",
                        name
                    ))
                })?;
                Self::copy_returned_bytes(data, name)
            },
            Ok(false) | Err(ramforge_core::CacheError::TooLarge { .. }) => data_source.read_tensor(name),
            Err(error) => Err(DataSourceError::General(format!(
                "cache insert failed: {}",
                error
            ))),
        }
    }

    pub fn model(&self) -> &GgufModel {
        self.data_source.model()
    }

    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }
}
