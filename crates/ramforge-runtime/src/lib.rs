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
pub mod benchmark_runner;
pub mod capacity;
pub mod context_config;
pub mod dense_residency;
pub mod host_staging;
pub mod hw_calibration;
pub mod cgroup;
pub mod calibration;
pub mod calibration_compatibility;
pub(crate) mod compute_dispatch;
pub mod discovery;
pub mod execution;
pub mod execution_candidate;
pub mod execution_profile;
pub mod forgecore;
pub mod gpu_observability;
pub mod graph_manifest;
pub mod inference;
pub mod inference_service;
pub mod kv_cache;
pub mod layer;
pub(crate) mod layer_cache;
pub(crate) mod layer_read;
pub mod memory_hierarchy;
pub mod memory_ledger;
pub mod memory_report;
pub mod model;
pub mod machine_profile;
pub mod model_architecture;
pub mod model_executor;
pub mod model_requirements;
pub mod moe_routing;
pub mod model_storage;
pub mod observation_contract;
pub mod ops;
pub mod optimization_loop;
pub mod orchestration;
pub mod page_residency;
pub mod persistent;
pub mod plan;
pub mod plan_compiler;
pub mod plan_persistence;
pub mod planner;
pub mod prediction;
pub mod prediction_planner;
pub mod prediction_sim;
pub mod adaptive;
pub use adaptive::{
    applied_report_from_observation, assess_budget, baseline_strategy, build_calibration_report,
    build_plan, compare_experiments, environment_changes, generate_candidates, record_invalidation,
    resolve_compatibility, revalidate_plan, select_best, AdaptivePlan, AdaptiveStrategy,
    AppliedPlanReport, BestSelection, BudgetClass, BudgetComponent, CalibrationExperiment,
    CalibrationReport, CandidateComparison, CandidateLimits, CandidateSet, CandidateStatus,
    CompatibilityReport, CompatibilityRequest, Evidence, EvidenceSource, ExecutionEnvironment,
    ExecutionFeature, ExperimentStatus, FeatureSupport, FieldObservation, FieldState, Invalidation,
    InvalidationReason, MeasurementSummary, ObjectivePreference, PlanDecision, PlanStatus,
    PlanValidation, ResourceBudget, RuntimeObservation, StrategyAxis, StrategyCandidate,
    SupportClass, ADAPTIVE_PLAN_SCHEMA_VERSION, ADAPTIVE_SCHEMA_VERSION,
    CAPABILITY_SCHEMA_VERSION, EXECUTION_ENVIRONMENT_SCHEMA_VERSION,
    RESOURCE_BUDGET_SCHEMA_VERSION,
};
pub mod profiling;
pub use profiling::{
    hardware_profile, model_profile, performance_identity, recommend, workload_from_plan,
    AdaptiveRecommendation, HardwareProfile, MeasuredSummary, ModelProfile, PerformanceIdentity,
    PerformanceProfileStore, PerformanceRecord, PlacementAdvice, QuantizationBucket,
    ResidencyAdvice, SpeculativeAdvice, WorkloadProfile, PROFILING_SCHEMA_VERSION,
};
pub mod profile;
pub mod residency;
pub mod residency_engine;
pub mod resource_lifetime;
pub mod runtime_calibration;
pub mod runtime_config;
pub mod runtime_measurement;
pub mod residency_prediction;
pub mod runtime_plan;
pub mod scheduler_policy;
pub mod sampling;
pub mod service_server;
pub mod simd;
pub mod speculative;
pub mod staging_plan;
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
    ForgeCoreAdapterError, ForgeCoreContextConfiguration, ForgeCoreContextOptions, ForgeCoreLogits,
    ForgeCoreModel, ForgeCoreModelMetadata, ForgeCoreModelOptions, ForgeCoreSession,
};
pub use forge_core::{
    ExpertRoutingObservation, ExpertRoutingOptions, ExpertRoutingSummary, NodeDispatchBackend,
    NodeDispatchEvent, NodeDispatchOptions, NodeDispatchSource, NodeDispatchSummary,
    NodeSamplingStrategy, PageReleaseResult, PageSample, PageSampleStatus, TensorMappingInfo,
    MAX_NODE_DISPATCH_EVENTS,
};
pub use benchmark_runner::{
    run_benchmark, BenchmarkConfig, BenchmarkReport, BenchmarkRun, BENCHMARK_SCHEMA_VERSION,
};
pub use context_config::{ContextConfig, FlashAttentionMode, KvCacheType};
pub use dense_residency::{decide_dense_residency, DenseResidencyDecision};
pub use capacity::{expert_accounting, ExpertAccounting};
pub use calibration_compatibility::{
    CalibrationCompatibility, CompatibilityClass, CompatibilityTarget, DimensionResult,
    DimensionStatus, EvidenceUse, CALIBRATION_COMPATIBILITY_SCHEMA_VERSION,
    MAX_CALIBRATION_AGE_MS,
};
pub use execution_candidate::{
    candidate_plans, compare_plans, compare_plans_with_calibration, compare_plans_with_evidence,
    evaluate_plan, evaluate_plan_with_calibration, CandidateRequest, ExecutionPlanCandidate,
    Feasibility, MappingStrategy, PlanBackend, PlanComparison, PlanEvaluation, PlanProfile,
    PlanReason, EXECUTION_CANDIDATE_SCHEMA_VERSION,
};
pub use machine_profile::{
    collect_machine_profile, BackendInventory, CpuRuntimeSettings, Freshness, MachineProfile,
    MachineProfileConfig, ObservationState, Observed, Provenance, RegisteredBackend,
    StorageBenchmarkConfig, StorageProfile, MACHINE_PROFILE_SCHEMA_VERSION,
};
pub use forge_core::staging;
pub use gpu_observability::{
    collect_observability, BufferObservation, CounterObservation, DeviceObservation,
    GpuObservability, LayerStorageObservation, MemoryObservation, MemoryObservationState,
    ObservationScope, ObservationSource, StagingCounters, StagingObservation, StagingState,
    TransferObservation, GPU_OBSERVABILITY_SCHEMA_VERSION,
};
pub use memory_hierarchy::{
    build_memory_hierarchy, build_placement_model, build_placement_model_for_plan,
    evaluate_hierarchy_feasibility, ComponentPlacement, DeviceMemory, HierarchyFeasibility,
    MemoryCategory, MemoryCategoryObservation, MemoryHierarchy, ObservationQuality, PlacementModel,
    PlacementState, MEMORY_HIERARCHY_SCHEMA_VERSION,
};
pub use model_architecture::{inspect_architecture, ArchitectureKind, ModelArchitecture};
pub use moe_routing::{
    aggregate_routing, expert_byte_ranges, expert_layout, ExpertByteRange, ExpertLayerLayout,
    ExpertLayout, LayerRoutingReport, PrefetchAssessment, ResidencySimulation, RoutingAggregate,
    RoutingObservationRecord, SimulationPolicy, MOE_ROUTING_SCHEMA_VERSION,
};
pub use model_storage::{
    assess_storage, copy_model_verified, CopyReport, DestinationAssessment, StorageRecommendation,
    MODEL_STORAGE_SCHEMA_VERSION,
};
pub use model_requirements::{
    model_file_fingerprint, model_requirements, ModelRequirementConfig, ModelRequirements,
    RequirementBytes, RequirementCertainty, MODEL_REQUIREMENTS_SCHEMA_VERSION,
};
pub use optimization_loop::{
    run_profile_optimization, OptimizationConfig, OptimizationRecommendation, OptimizationReport,
    ProfileMeasurement, OPTIMIZATION_SCHEMA_VERSION,
};
pub use observation_contract::{
    provenance_for_status, ContractError, DispatchEvidence, ObservationProvenance,
    ObservationStrategy, PageAccount, SchedulerObservation, SourceAttribution,
    MAX_CONTRACT_DISPATCHES, SCHEDULER_OBSERVATION_SCHEMA_VERSION,
};
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
pub use runtime_calibration::{
    build_attribution, build_calibration, CalibrationEvidence, EstimateComparison, MachineIdentity,
    ModelIdentity, ObservationConditions, RuntimeCalibration, RuntimeConfiguration,
    KV_TYPE_SOURCE_ADAPTER_DEFAULT, RUNTIME_CALIBRATION_SCHEMA_VERSION,
};
pub use runtime_measurement::{
    capture_snapshot, BackendMemorySnapshot, LifecycleDeltas, LifecycleRecorder, LifecycleStage,
    MemoryComponent, ModelMappingMemory, ProcessMemorySnapshot, RuntimeAttribution,
    RuntimeSnapshot, SystemMemorySnapshot, MAX_RUNTIME_SNAPSHOTS,
};
pub use inference_service::{
    GenerationOptions, GenerationReport, GenerationStats, InferenceService,
    PipelineCommitReport, ServiceCapabilities, ServiceError, ServiceState, ServiceStatus,
    StagedLayerStatus, StagingLifecycleStatus, INFERENCE_SERVICE_API_VERSION,
};
pub use service_server::{serve_stdio, ServiceServerError};
pub use host_staging::{
    HostStagingMode, HostStagingReport, PinnedPool, PooledPinnedStats, ReadBatching,
};
pub use hw_calibration::{
    default_calibration_path, run_calibration, CalibrationIdentity, CalibrationRecord,
    CalibrationStatus, HW_CALIBRATION_SCHEMA_VERSION,
};
pub use residency_prediction::{
    PredictionConfidence, PredictionStatus, PredictionStrategy, ResidencyPrediction,
    TransitionPredictor,
};
pub use scheduler_policy::{
    AdaptiveController, DecisionConfidence, DecisionContext, StrategyDecision, TransferStrategy,
};
pub use speculative::{
    draft_vocabulary_compatible, generate_speculative, SpeculativeConfig, SpeculativeOutput,
    SpeculativeRound, SpeculativeStats,
};
pub use staging_plan::{
    SelectionPolicy, StagingLayerPlan, StagingPlan, StagingPlanState, StagingRequest,
    DEFAULT_STAGING_SAFETY_MARGIN_BYTES, STAGING_PLAN_SCHEMA_VERSION,
};
pub use runtime_plan::{
    CustomRuntimeRequest, EffectiveRuntimeConfig, ExecutionMode, FieldStatus, PlanField,
    RuntimePlan, RuntimeProfileKind, RUNTIME_PLAN_SCHEMA_VERSION,
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
