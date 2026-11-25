//! Transport-independent operational inference service.
//!
//! The service owns one model/context at a time, executes through forgeCORE,
//! streams decoded text through a caller callback, and supports cooperative
//! cancellation between token decodes. It is not an inference implementation.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::context_config::{FlashAttentionMode, KvCacheType};
use crate::forgecore::{
    ForgeCoreContextOptions, ForgeCoreModel, ForgeCoreModelOptions, ForgeCoreSession,
};
use crate::gpu_observability::{
    collect_observability, GpuObservability, StagingCounters, StagingObservation, StagingState,
};
use crate::host_staging::{
    HostStagingMode, HostStagingReport, HostStagingStatus, PinnedPool, ReadBatching,
};
use crate::memory_report::{process_io, process_memory_split, process_rss, ProcessIo};
use crate::residency_engine::process_faults;
use crate::runtime_plan::RuntimePlan;
use crate::sampling::Sampler;
use crate::staging::{Engine, StagingError};
use crate::staging_plan::{StagingLayerPlan, StagingPlan, StagingPlanState, StagingRequest};
use crate::ExpertRoutingOptions;
use ramforge_core::{discover_model_set, Tokenizer};

pub const INFERENCE_SERVICE_API_VERSION: u32 = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceState {
    Empty,
    ModelLoaded,
    Ready,
    Generating,
    Failed,
    Terminated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceCapabilities {
    pub api_version: u32,
    pub backend: String,
    pub device: String,
    pub streaming: bool,
    pub cooperative_cancellation: bool,
    pub multiple_sessions: bool,
    pub concurrent_generation: bool,
    pub max_context_tokens: Option<u32>,
    pub execution_modes: Vec<String>,
    pub gpu_execution: bool,
    pub notes: Vec<String>,
}

impl ServiceCapabilities {
    pub fn for_plan(plan: &RuntimePlan) -> Self {
        Self {
            api_version: INFERENCE_SERVICE_API_VERSION,
            backend: plan.effective.backend.clone(),
            device: plan.effective.device_id.clone(),
            streaming: true,
            cooperative_cancellation: true,
            multiple_sessions: false,
            concurrent_generation: false,
            max_context_tokens: Some(plan.effective.context_tokens),
            execution_modes: {
                let registry = crate::DeviceRegistry::discover();
                if registry
                    .devices()
                    .iter()
                    .any(|device| !device.is_cpu() && device.selectable)
                {
                    vec!["cpu".to_string(), "gpu".to_string(), "hybrid".to_string()]
                } else {
                    vec!["cpu".to_string()]
                }
            },
            gpu_execution: crate::DeviceRegistry::discover()
                .devices()
                .iter()
                .any(|device| !device.is_cpu() && device.selectable),
            notes: vec![
                "One session at a time; cancellation is checked between token decodes.".to_string(),
                "No ownership or eviction authorization is exposed.".to_string(),
                "max_context_tokens is the planned request; the native context rounds n_ctx up (GGML_PAD in llama-context.cpp) and status reports the effective value."
                    .to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StagedLayerStatus {
    pub layer: u32,
    pub tensor_count: u32,
    pub device_bytes: u64,
    pub graph_substitutions: u64,
    pub promoted_from_standby: bool,
}

/// Native staged-residency lifecycle snapshot. Counters are native facts, not
/// RAMforge policy estimates. `reuse_count` stays zero while capacity reuse is
/// unsupported.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StagingLifecycleStatus {
    pub graph_phase: String,
    pub graph_invalidated: bool,
    pub resident_layers: Vec<u32>,
    pub resident_bytes: u64,
    pub pending_layers: u32,
    pub pending_bytes: u64,
    pub pool_bytes: u64,
    pub pool_limit_bytes: u64,
    pub pool_buffers: u64,
    pub commit_count: u64,
    pub standby_bytes: u64,
    pub standby_limit_bytes: u64,
    pub standby_layers: u64,
    pub standby_hits: u64,
    pub allocation_count: u64,
    pub release_count: u64,
    pub reuse_count: u64,
    pub staging_failures: u64,
    pub graph_invalidation_count: u64,
    pub graph_rebuild_count: u64,
    pub scheduler_reset_count: u64,
    pub graph_build_ns: u64,
    pub last_graph_build_ns: u64,
    pub invalidation_failures: u64,
    pub quiescence_wait_ns: u64,
    pub last_quiescence_wait_ns: u64,
    pub eviction_supported: bool,
    pub allocation_reuse_supported: bool,
    pub transfer_compute_overlap_supported: bool,
}

/// Result of committing a prepared pipeline replacement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineCommitReport {
    pub layers: Vec<u32>,
    pub transfer_bytes: u64,
    pub transfer_duration_ns: u64,
    pub transfer_wait_ns: u64,
    pub submit_ns: u64,
    pub invalidation_ns: u64,
    pub commit_ns: u64,
    pub transfer_in_flight_at_commit: bool,
    pub host_staging: HostStagingReport,
    pub host_read_wait_ns: Option<u64>,
    pub partial: bool,
    pub preserved_layers: Vec<u32>,
    pub new_layers: Vec<u32>,
    pub removed_layers: Vec<u32>,
    pub preserved_bytes: u64,
    pub new_bytes: u64,
    pub removed_bytes: u64,
    pub promoted_layers: Vec<u32>,
    pub promoted_bytes: u64,
}

struct PendingPipeline {
    plan: StagingPlan,
    layers: Vec<u32>,
    host: Option<crate::staging::StagingBuffer>,
    submit_ns: u64,
    host_staging: HostStagingReport,
    read_job: Option<crate::host_staging::HostReadJob>,
    device_ptrs: Vec<u64>,
    sizes: Vec<u64>,
    partial: bool,
    preserved_layers: Vec<u32>,
    new_layers: Vec<u32>,
    remove_layers: Vec<u32>,
    preserved_bytes: u64,
    new_bytes: u64,
    removed_bytes: u64,
    promoted_layers: Vec<u32>,
    promoted_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefetchStatus {
    pub prepared: u64,
    pub committed: u64,
    pub aborted: u64,
    pub bytes_prefetched: u64,
    pub bytes_committed: u64,
    pub bytes_wasted: u64,
    pub bytes_preserved: u64,
    pub bytes_from_standby: u64,
    pub hit_rate_permille: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceStatus {
    pub api_version: u32,
    pub state: ServiceState,
    pub model_path: String,
    pub effective_context_tokens: Option<u32>,
    pub effective_batch_tokens: Option<u32>,
    pub effective_ubatch_tokens: Option<u32>,
    pub effective_threads: Option<u32>,
    pub effective_threads_batch: Option<u32>,
    pub effective_sequence_max: Option<u32>,
    /// Requested KV/flash values forwarded to the native context. Native KV
    /// dtype getters are unavailable at the pinned revision.
    pub requested_kv_type_k: Option<String>,
    pub requested_kv_type_v: Option<String>,
    pub requested_flash_attn: Option<String>,
    pub kv_types_native_readback: Option<bool>,
    pub architecture: Option<crate::model_architecture::ModelArchitecture>,
    pub execution_mode: crate::runtime_plan::ExecutionMode,
    pub device_id: String,
    pub requested_gpu_layers: Option<u32>,
    pub effective_gpu_layers: Option<u32>,
    pub gpu_transformer_layers: Option<u32>,
    pub cpu_transformer_layers: Option<u32>,
    pub output_device_type: Option<String>,
    pub staged_layers: Vec<StagedLayerStatus>,
    pub staging_plan: Option<StagingPlan>,
    pub staging_lifecycle: Option<StagingLifecycleStatus>,
    pub host_staging: HostStagingStatus,
    pub prefetch: PrefetchStatus,
    pub pipeline_pending: bool,
    pub model_instance_id: Option<u64>,
    pub context_id: Option<u64>,
    pub rss_bytes: Option<u64>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationOptions {
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_tokens: 32,
            temperature: 0.0,
            top_k: None,
            top_p: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenPhase {
    Prompt,
    Decode,
}

/// Incremental I/O and fault cost for one native decode call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenIo {
    pub phase: TokenPhase,
    pub token_index: u32,
    pub elapsed_ns: u64,
    pub storage_read_bytes: Option<u64>,
    pub rchar_bytes: Option<u64>,
    pub read_syscalls: Option<u64>,
    pub major_faults: Option<u64>,
    pub minor_faults: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationStats {
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub prompt_processing_ns: u64,
    pub time_to_first_token_ns: Option<u64>,
    pub decode_ns: Option<u64>,
    pub total_generation_ns: u64,
    pub prompt_tokens_per_second: f64,
    pub decode_tokens_per_second: Option<f64>,
    pub generation_tokens_per_second: f64,
    pub stop_reason: String,
    pub rss_before_bytes: Option<u64>,
    pub rss_after_bytes: Option<u64>,
    /// Storage bytes actually fetched during prompt processing and decode.
    /// This excludes page-cache hits; `rchar_bytes` includes them.
    pub storage_read_bytes: Option<u64>,
    pub rchar_bytes: Option<u64>,
    pub read_syscalls: Option<u64>,
    pub major_faults: Option<u64>,
    pub minor_faults: Option<u64>,
    pub pss_bytes: Option<u64>,
    pub pss_file_bytes: Option<u64>,
    pub pss_anon_bytes: Option<u64>,
    pub anonymous_bytes: Option<u64>,
    pub swap_bytes: Option<u64>,
    pub token_io: Vec<TokenIo>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationReport {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub stats: GenerationStats,
    pub effective_context_tokens: u32,
    pub effective_batch_tokens: u32,
    pub effective_ubatch_tokens: u32,
    pub effective_threads: u32,
    pub effective_sequence_max: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceError {
    InvalidState {
        expected: String,
        actual: ServiceState,
    },
    Configuration(String),
    Model(String),
    Tokenizer(String),
    Context(String),
    Execution(String),
    UnsupportedExecution(String),
    Replacement {
        message: String,
        rolled_back: bool,
    },
    Cancelled,
    Sink(String),
}

impl std::fmt::Display for ServiceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidState { expected, actual } => {
                write!(
                    formatter,
                    "invalid service state: expected {expected}, got {actual:?}"
                )
            }
            Self::Configuration(message) => write!(formatter, "configuration: {message}"),
            Self::Model(message) => write!(formatter, "model: {message}"),
            Self::Tokenizer(message) => write!(formatter, "tokenizer: {message}"),
            Self::Context(message) => write!(formatter, "context: {message}"),
            Self::Execution(message) => write!(formatter, "execution: {message}"),
            Self::UnsupportedExecution(message) => {
                write!(formatter, "unsupported execution: {message}")
            }
            Self::Replacement {
                message,
                rolled_back,
            } => {
                if *rolled_back {
                    write!(formatter, "replacement failed and rolled back: {message}")
                } else {
                    write!(
                        formatter,
                        "replacement failed and rollback failed: {message}"
                    )
                }
            }
            Self::Cancelled => formatter.write_str("generation cancelled"),
            Self::Sink(message) => write!(formatter, "stream sink: {message}"),
        }
    }
}

impl std::error::Error for ServiceError {}

pub struct InferenceService {
    plan: RuntimePlan,
    state: ServiceState,
    model: Option<ForgeCoreModel>,
    session: Option<ForgeCoreSession>,
    tokenizer: Option<Tokenizer>,
    errors: Vec<String>,
    staged_layers: Vec<u32>,
    staging_request: StagingRequest,
    staging_plan: Option<StagingPlan>,
    staging_engine: Option<Engine>,
    staging_status: Vec<StagedLayerStatus>,
    staging_pool_limit_bytes: u64,
    staging_standby_limit_bytes: u64,
    selection_policy: crate::staging_plan::SelectionPolicy,
    host_staging_mode: HostStagingMode,
    read_batching: ReadBatching,
    host_read_async: bool,
    partial_residency: bool,
    storage_concurrency: usize,
    storage_max_outstanding_bytes: u64,
    pinned_pool: PinnedPool,
    last_host_staging: Option<HostStagingReport>,
    prefetch_prepared: u64,
    prefetch_committed: u64,
    prefetch_aborted: u64,
    prefetch_bytes: u64,
    prefetch_committed_bytes: u64,
    prefetch_wasted_bytes: u64,
    prefetch_preserved_bytes: u64,
    prefetch_standby_bytes: u64,
    tensor_mappings: Option<HashMap<String, forge_core::TensorMappingInfo>>,
    pending_pipeline: Option<PendingPipeline>,
    routing_options: Option<ExpertRoutingOptions>,
}

impl InferenceService {
    pub fn new(plan: RuntimePlan) -> Result<Self, ServiceError> {
        plan.validate().map_err(ServiceError::Configuration)?;
        Ok(Self {
            plan,
            state: ServiceState::Empty,
            model: None,
            session: None,
            tokenizer: None,
            errors: Vec::new(),
            staged_layers: Vec::new(),
            staging_request: StagingRequest::default(),
            staging_plan: None,
            staging_engine: None,
            staging_status: Vec::new(),
            staging_pool_limit_bytes: 0,
            staging_standby_limit_bytes: 0,
            selection_policy: crate::staging_plan::SelectionPolicy::Ascending,
            host_staging_mode: HostStagingMode::Copy,
            read_batching: ReadBatching::Coalesced,
            host_read_async: false,
            partial_residency: false,
            storage_concurrency: 1,
            storage_max_outstanding_bytes: 0,
            pinned_pool: PinnedPool::new(0),
            last_host_staging: None,
            prefetch_prepared: 0,
            prefetch_committed: 0,
            prefetch_aborted: 0,
            prefetch_bytes: 0,
            prefetch_committed_bytes: 0,
            prefetch_wasted_bytes: 0,
            prefetch_preserved_bytes: 0,
            prefetch_standby_bytes: 0,
            tensor_mappings: None,
            pending_pipeline: None,
            routing_options: None,
        })
    }

    pub fn plan(&self) -> &RuntimePlan {
        &self.plan
    }

    pub fn capabilities(&self) -> ServiceCapabilities {
        ServiceCapabilities::for_plan(&self.plan)
    }

    pub fn state(&self) -> ServiceState {
        self.state
    }

    /// Enable explicit graph-integrated staging for the listed layers.
    /// This is separate from normal CPU/GPU plan modes and never falls back.
    pub fn with_staged_layers(self, layers: Vec<u32>) -> Self {
        self.with_staging_request(StagingRequest {
            layers,
            ..StagingRequest::default()
        })
    }

    pub fn with_staging_pool_limit(mut self, bytes: u64) -> Self {
        self.staging_pool_limit_bytes = bytes;
        self
    }

    pub fn with_staging_standby_limit(mut self, bytes: u64) -> Self {
        self.staging_standby_limit_bytes = bytes;
        self
    }

    pub fn with_selection_policy(mut self, policy: crate::staging_plan::SelectionPolicy) -> Self {
        self.selection_policy = policy;
        self
    }

    pub fn with_host_staging(
        mut self,
        mode: HostStagingMode,
        batching: ReadBatching,
        pinned_pool_bytes: u64,
    ) -> Self {
        self.host_staging_mode = mode;
        self.read_batching = batching;
        self.pinned_pool = PinnedPool::new(pinned_pool_bytes);
        self
    }

    pub fn with_async_host_read(mut self, enabled: bool) -> Self {
        self.host_read_async = enabled;
        self
    }

    pub fn with_partial_residency(mut self, enabled: bool) -> Self {
        self.partial_residency = enabled;
        self
    }

    pub fn with_storage_pipeline(mut self, concurrency: usize, max_outstanding_bytes: u64) -> Self {
        self.storage_concurrency = concurrency.max(1);
        self.storage_max_outstanding_bytes = max_outstanding_bytes;
        self
    }

    /// Build a plan without mutating residency or touching CUDA state.
    pub fn preview_staging_plan(
        &self,
        request: StagingRequest,
    ) -> Result<StagingPlan, ServiceError> {
        let registry = crate::DeviceRegistry::discover();
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| ServiceError::InvalidState {
                expected: "context ready".to_string(),
                actual: self.state,
            })?;
        self.build_staging_plan_for(session, &registry, &request)
    }

    pub fn with_staging_request(mut self, mut request: StagingRequest) -> Self {
        request.layers.sort_unstable();
        request.layers.dedup();
        self.staged_layers = request.layers.clone();
        self.staging_request = request;
        self
    }

    /// Enable read-only expert-routing observation for subsequent sessions.
    /// Requires a CPU plan; the native context rejects GPU execution.
    pub fn with_routing_observation(mut self, options: ExpertRoutingOptions) -> Self {
        self.routing_options = Some(options);
        self
    }

    pub fn routing_observations(&self) -> Vec<crate::ExpertRoutingObservation> {
        self.session
            .as_ref()
            .map(|session| session.routing_observations())
            .unwrap_or_default()
    }

    pub fn routing_summary(&self) -> Option<crate::ExpertRoutingSummary> {
        self.session
            .as_ref()
            .and_then(|session| session.routing_summary())
    }

    pub fn staged_layers(&self) -> &[u32] {
        &self.staged_layers
    }

    pub fn status(&self) -> ServiceStatus {
        let model_id = self
            .model
            .as_ref()
            .and_then(|model| model.lifetime_state().ok())
            .map(|state| state.instance_id);
        let context_id = self
            .session
            .as_ref()
            .and_then(|session| session.lifetime_state().ok())
            .map(|state| state.context_id);
        ServiceStatus {
            api_version: INFERENCE_SERVICE_API_VERSION,
            state: self.state,
            model_path: self.plan.model_path.clone(),
            effective_context_tokens: self.session.as_ref().map(ForgeCoreSession::context_length),
            effective_batch_tokens: self
                .session
                .as_ref()
                .map(ForgeCoreSession::effective_batch_tokens),
            effective_ubatch_tokens: self
                .session
                .as_ref()
                .map(ForgeCoreSession::effective_ubatch_tokens),
            effective_threads: self
                .session
                .as_ref()
                .map(ForgeCoreSession::effective_threads),
            effective_threads_batch: self
                .session
                .as_ref()
                .map(ForgeCoreSession::effective_threads_batch),
            effective_sequence_max: self
                .session
                .as_ref()
                .map(ForgeCoreSession::effective_sequence_max),
            requested_kv_type_k: self
                .session
                .as_ref()
                .map(|session| session.context_options().kv_type_k.label().to_string()),
            requested_kv_type_v: self
                .session
                .as_ref()
                .map(|session| session.context_options().kv_type_v.label().to_string()),
            requested_flash_attn: self
                .session
                .as_ref()
                .map(|session| session.context_options().flash_attn.label().to_string()),
            kv_types_native_readback: self.session.as_ref().map(|session| {
                session
                    .effective_context_configuration()
                    .kv_types_native_readback
            }),
            architecture: self.plan.architecture.clone(),
            execution_mode: self.plan.effective.execution_mode,
            device_id: self.plan.effective.device_id.clone(),
            requested_gpu_layers: self.plan.requested_gpu_layers,
            effective_gpu_layers: self
                .model
                .as_ref()
                .map(|model| model.metadata().effective_gpu_layers),
            gpu_transformer_layers: self
                .model
                .as_ref()
                .map(|model| model.metadata().gpu_transformer_layers),
            cpu_transformer_layers: self
                .model
                .as_ref()
                .map(|model| model.metadata().cpu_transformer_layers),
            output_device_type: self
                .model
                .as_ref()
                .map(|model| model.metadata().output_device_type.clone()),
            staged_layers: self.current_staged_status(),
            staging_plan: self.staging_plan.clone(),
            staging_lifecycle: self.staging_lifecycle(),
            host_staging: HostStagingStatus {
                mode: self.host_staging_mode,
                batching: self.read_batching,
                last: self.last_host_staging.clone(),
                pool: self.pinned_pool.stats(),
            },
            prefetch: PrefetchStatus {
                prepared: self.prefetch_prepared,
                committed: self.prefetch_committed,
                aborted: self.prefetch_aborted,
                bytes_prefetched: self.prefetch_bytes,
                bytes_committed: self.prefetch_committed_bytes,
                bytes_wasted: self.prefetch_wasted_bytes,
                bytes_preserved: self.prefetch_preserved_bytes,
                bytes_from_standby: self.prefetch_standby_bytes,
                hit_rate_permille: (self.prefetch_prepared > 0).then(|| {
                    ((self.prefetch_committed.saturating_mul(1000)) / self.prefetch_prepared) as u32
                }),
            },
            pipeline_pending: self.pending_pipeline.is_some(),
            model_instance_id: model_id,
            context_id,
            rss_bytes: process_rss(),
            errors: self.errors.clone(),
        }
    }

    /// Point-in-time device and native buffer observability. Device values are
    /// available without a session; model/context/compute buffers require one.
    pub fn memory_observability(&self) -> Result<GpuObservability, ServiceError> {
        let registry = crate::DeviceRegistry::discover();
        let mut observability =
            collect_observability(&registry, self.session.as_ref(), self.model.as_ref())
                .map_err(ServiceError::Execution)?;
        if let Some(engine) = self.staging_engine.as_ref() {
            let stats = engine
                .stats()
                .map_err(|error| ServiceError::Execution(error.to_string()))?;
            observability.staging = StagingObservation {
                schema_version: 1,
                state: if stats.in_flight_operations > 0 {
                    StagingState::InFlight
                } else {
                    StagingState::Idle
                },
                reason: "staging engine attached; graph-integrated layer mode".to_string(),
                counters: Some(StagingCounters {
                    allocation_requests: stats.allocation_requests,
                    allocation_successes: stats.allocation_successes,
                    allocation_failures: stats.allocation_failures,
                    released: stats.released,
                    current_bytes: stats.current_bytes,
                    high_water_bytes: stats.high_water_bytes,
                    transfer_requests: stats.transfer_requests,
                    transfer_completed: stats.transfer_completed,
                    transfer_failed: stats.transfer_failed,
                    transfer_bytes: stats.transfer_bytes,
                    submit_ns: stats.submit_ns,
                    wait_ns: stats.wait_ns,
                    last_duration_ns: stats.last_duration_ns,
                    total_duration_ns: stats.total_duration_ns,
                    in_flight_operations: stats.in_flight_operations,
                    cancellation_requests: stats.cancellation_requests,
                }),
                sampled_at_unix_ms: observability.sampled_at_unix_ms,
                collection_sequence: observability.collection_sequence,
            };
        }
        Ok(observability)
    }

    /// Load the model and build the tokenizer. The context is created separately.
    pub fn load(&mut self) -> Result<(), ServiceError> {
        match self.state {
            ServiceState::Empty | ServiceState::Terminated => {}
            _ => {
                return Err(ServiceError::InvalidState {
                    expected: "empty or terminated".to_string(),
                    actual: self.state,
                })
            }
        }
        let path = PathBuf::from(&self.plan.model_path);
        let model_set =
            discover_model_set(&path).map_err(|error| ServiceError::Model(error.to_string()))?;
        let gguf = model_set.merged_model();
        let tokenizer = Tokenizer::from_gguf(&gguf).map_err(ServiceError::Tokenizer)?;
        let registry = crate::DeviceRegistry::discover();
        let requested_device =
            if self.plan.effective.execution_mode == crate::runtime_plan::ExecutionMode::Cpu {
                crate::DeviceSelection::Cpu
            } else {
                crate::DeviceSelection::by_id(self.plan.effective.device_id.clone())
            };
        let selected = registry.select(&requested_device).map_err(|error| {
            ServiceError::UnsupportedExecution(format!(
                "requested device {} is not selectable: {error}",
                self.plan.effective.device_id
            ))
        })?;
        let execution = crate::ExecutionConfig {
            device: requested_device,
            cpu_thread_count: self.plan.effective.threads as usize,
            context_length: Some(self.plan.effective.context_tokens),
            check_tensors: false,
        };
        let mut staging_engine = None;
        let mut extra_device_index = None;
        if !self.staging_request.layers.is_empty() || self.staging_request.auto {
            let gpu = registry
                .devices()
                .iter()
                .find(|device| !device.is_cpu() && device.selectable)
                .ok_or_else(|| {
                    ServiceError::UnsupportedExecution(
                        "staged layers require a selectable CUDA device".to_string(),
                    )
                })?;
            let engine = Engine::new(0).map_err(|error| match error {
                StagingError::Unsupported(reason) => ServiceError::UnsupportedExecution(format!(
                    "CUDA staging unavailable: {reason}"
                )),
                other => ServiceError::Execution(other.to_string()),
            })?;
            extra_device_index = gpu.registry_index;
            staging_engine = Some(engine);
        }
        let mut options = ForgeCoreModelOptions {
            use_extra_bufts: self.plan.effective.weight_repacking,
            gpu_layers: self.plan.effective.gpu_layers,
            gpu_device_index: if self.plan.effective.gpu_layers > 0 {
                selected.info().registry_index
            } else {
                None
            },
            extra_device_index,
            ..ForgeCoreModelOptions::default()
        };
        if self.plan.effective.load_mode == "mmap-no-prefetch" {
            options = options.with_no_prefetch();
        }
        let model = ForgeCoreModel::load_for_execution_with_options(
            model_set.primary_path(),
            execution,
            &registry,
            options,
        )
        .map_err(|error| ServiceError::Model(error.to_string()))?;
        let effective_gpu_layers = model.metadata().effective_gpu_layers;
        if self.plan.effective.gpu_layers > 0 && effective_gpu_layers == 0 {
            return Err(ServiceError::UnsupportedExecution(format!(
                "GPU layers were requested but native placement readback reports zero offloaded layers; refusing to claim GPU execution (requested={}, effective={effective_gpu_layers})",
                self.plan.effective.gpu_layers
            )));
        }
        if self.plan.effective.gpu_layers == 0 && effective_gpu_layers != 0 {
            return Err(ServiceError::UnsupportedExecution(format!(
                "CPU plan unexpectedly reports {effective_gpu_layers} offloaded layers"
            )));
        }
        if self.plan.effective.gpu_layers > 0
            && effective_gpu_layers < self.plan.effective.gpu_layers
        {
            self.errors.push(format!(
                "requested {} GPU layers but native placement reports {effective_gpu_layers}",
                self.plan.effective.gpu_layers
            ));
        }
        self.tokenizer = Some(tokenizer);
        self.model = Some(model);
        self.staging_engine = staging_engine;
        self.staging_status.clear();
        self.state = ServiceState::ModelLoaded;
        Ok(())
    }

    /// Reuse the current context when it is already ready. Context creation
    /// allocates KV/compute buffers and re-stages configured layers; repeated
    /// generation requests must not pay that cost when the plan is unchanged.
    pub fn ensure_session(&mut self) -> Result<(), ServiceError> {
        if self.session.is_some() {
            if self.state == ServiceState::Ready {
                return Ok(());
            }
            return Err(ServiceError::InvalidState {
                expected: "ready".to_string(),
                actual: self.state,
            });
        }
        self.create_session()
    }

    /// Create the single reusable context for this service.
    pub fn create_session(&mut self) -> Result<(), ServiceError> {
        if self.state != ServiceState::ModelLoaded && self.state != ServiceState::Ready {
            return Err(ServiceError::InvalidState {
                expected: "model_loaded or ready".to_string(),
                actual: self.state,
            });
        }
        // A pending double buffer owns device allocations and an in-flight
        // transfer; complete and discard it before dropping the context.
        let _ = self.abort_staged_pipeline();
        // Drop the previous context before opening the replacement so two
        // contexts are not held simultaneously during a benchmark reset.
        self.session = None;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ServiceError::InvalidState {
                expected: "model loaded".to_string(),
                actual: self.state,
            })?;
        let session = model
            .open_context(ForgeCoreContextOptions {
                n_ctx: Some(self.plan.effective.context_tokens),
                n_threads: self.plan.effective.threads as usize,
                n_threads_batch: Some(self.plan.effective.threads_batch as usize),
                n_batch: Some(self.plan.effective.batch_tokens),
                n_ubatch: Some(self.plan.effective.ubatch_tokens),
                n_seq_max: Some(self.plan.effective.sequence_max),
                kv_type_k: KvCacheType::parse(&self.plan.effective.kv_type_k)
                    .map_err(ServiceError::Configuration)?,
                kv_type_v: KvCacheType::parse(&self.plan.effective.kv_type_v)
                    .map_err(ServiceError::Configuration)?,
                flash_attn: FlashAttentionMode::parse(&self.plan.effective.flash_attn)
                    .map_err(ServiceError::Configuration)?,
                offload_kqv: Some(self.plan.effective.offload_kqv),
                op_offload: Some(self.plan.effective.op_offload),
                trace_max_events: None,
                routing: self.routing_options,
            })
            .map_err(|error| ServiceError::Context(error.to_string()))?;
        let effective_context = session.context_length();
        if effective_context == 0 {
            return Err(ServiceError::Context(
                "native context reported zero length".to_string(),
            ));
        }
        self.session = Some(session);
        if self.staging_pool_limit_bytes > 0 {
            let limit = self
                .staging_pool_limit_bytes
                .min(self.staging_request.budget_bytes.unwrap_or(u64::MAX));
            self.session
                .as_ref()
                .expect("session assigned")
                .set_staging_pool_limit(limit)
                .map_err(|error| ServiceError::Execution(error.to_string()))?;
        }
        if self.staging_standby_limit_bytes > 0 {
            let limit = self
                .staging_standby_limit_bytes
                .min(self.staging_request.budget_bytes.unwrap_or(u64::MAX));
            self.session
                .as_ref()
                .expect("session assigned")
                .set_staging_standby_limit(limit)
                .map_err(|error| ServiceError::Execution(error.to_string()))?;
        }
        self.state = ServiceState::Ready;
        if let Err(error) = self.stage_configured_layers() {
            self.session = None;
            self.state = ServiceState::ModelLoaded;
            return Err(error);
        }
        Ok(())
    }

    fn stage_configured_layers(&mut self) -> Result<(), ServiceError> {
        if self.staging_request.layers.is_empty() && !self.staging_request.auto {
            return Ok(());
        }
        self.ensure_tensor_mappings()?;
        let registry = crate::DeviceRegistry::discover();
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| ServiceError::InvalidState {
                expected: "context ready".to_string(),
                actual: self.state,
            })?;
        let plan = self.build_staging_plan(session, &registry)?;
        self.staging_plan = Some(plan.clone());
        if plan.state == StagingPlanState::Rejected
            || plan.state == StagingPlanState::Unknown
            || plan.accepted_layers.is_empty()
        {
            return Err(ServiceError::Configuration(format!(
                "staged layer plan rejected: {}",
                serde_json::to_string(&plan)
                    .unwrap_or_else(|_| "plan serialization failed".to_string())
            )));
        }
        self.staged_layers = plan.accepted_layers.clone();
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ServiceError::InvalidState {
                expected: "model loaded".to_string(),
                actual: self.state,
            })?;
        let engine = self.staging_engine.as_ref().ok_or_else(|| {
            ServiceError::Execution("staging engine is not initialized".to_string())
        })?;
        let mode = self.host_staging_mode;
        let batching = self.read_batching;
        let mappings = self.tensor_mappings.as_ref().expect("mappings cached");
        let mut statuses = Vec::with_capacity(self.staged_layers.len());
        let mut aggregate: Option<HostStagingReport> = None;
        for layer in &self.staged_layers {
            let (status, report) = stage_layer_data(
                model,
                session,
                engine,
                &mut self.pinned_pool,
                mappings,
                mode,
                batching,
                self.storage_concurrency,
                self.storage_max_outstanding_bytes,
                *layer,
            )?;
            if let Some(total) = aggregate.as_mut() {
                total.accumulate(&report);
            } else {
                aggregate = Some(report);
            }
            statuses.push(status);
        }
        self.last_host_staging = aggregate.clone();
        if let (Some(plan), Some(report)) = (self.staging_plan.as_mut(), aggregate) {
            plan.host_staging = Some(report);
        }
        self.staging_status = statuses;
        Ok(())
    }

    /// Prepare a double-buffered pipeline replacement. Allocates the complete
    /// next selection in the native pending slot and submits one asynchronous
    /// H2D batch on the staging stream. Returns before any transfer completes;
    /// the caller may execute inference on the active selection while the batch
    /// runs. Commit after execution completion.
    pub fn prepare_staged_pipeline(
        &mut self,
        request: StagingRequest,
    ) -> Result<StagingPlan, ServiceError> {
        if self.state != ServiceState::Ready {
            return Err(ServiceError::InvalidState {
                expected: "ready".to_string(),
                actual: self.state,
            });
        }
        if self.pending_pipeline.is_some() {
            return Err(ServiceError::Configuration(
                "a pipeline replacement is already pending".to_string(),
            ));
        }
        self.ensure_tensor_mappings()?;
        let registry = crate::DeviceRegistry::discover();
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| ServiceError::InvalidState {
                expected: "context ready".to_string(),
                actual: self.state,
            })?;
        let mut plan = self.build_staging_plan_for(session, &registry, &request)?;
        let state = session.staging_state().map_err(|error| {
            ServiceError::Execution(format!("native staging state failed: {error}"))
        })?;
        let partial = self.partial_residency;
        let stage_layers = if partial {
            plan.newly_staged_layers.clone()
        } else {
            plan.accepted_layers.clone()
        };
        let preserved_layers = if partial {
            plan.kept_layers.clone()
        } else {
            Vec::new()
        };
        let remove_layers = if partial {
            plan.evicted_layers.clone()
        } else {
            Vec::new()
        };
        let bytes_for = |layer: u32| {
            plan.layers
                .iter()
                .find(|entry| entry.layer == layer)
                .and_then(|entry| entry.bytes)
                .unwrap_or(0)
        };
        let preserved_bytes: u64 = preserved_layers.iter().map(|layer| bytes_for(*layer)).sum();
        let standby_count = session
            .standby_layer_count()
            .map_err(|error| ServiceError::Execution(error.to_string()))?;
        let mut standby_layers = Vec::with_capacity(standby_count as usize);
        for index in 0..standby_count {
            standby_layers.push(
                session
                    .standby_layer_at(index)
                    .map_err(|error| ServiceError::Execution(error.to_string()))?
                    .layer,
            );
        }
        let standby_candidates: Vec<u32> = stage_layers
            .iter()
            .copied()
            .filter(|layer| standby_layers.contains(layer))
            .collect();
        let standby_candidate_bytes: u64 = standby_candidates
            .iter()
            .map(|layer| bytes_for(*layer))
            .sum();
        plan.standby_candidates = standby_candidates.clone();
        plan.standby_candidate_bytes = standby_candidate_bytes;
        let pending_bytes = if partial {
            plan.required_new_bytes
                .saturating_sub(standby_candidate_bytes)
        } else {
            plan.planned_bytes
        };
        let removed_bytes: u64 = remove_layers.iter().map(|layer| bytes_for(*layer)).sum();
        let required = state
            .resident_bytes
            .saturating_add(state.standby_bytes)
            .saturating_add(pending_bytes)
            .saturating_add(state.pool_bytes);
        let fits = plan
            .effective_budget_bytes
            .map(|budget| required <= budget)
            .unwrap_or(false);
        plan.pipeline = true;
        plan.pipeline_partial = partial;
        plan.pipeline_active_bytes = state.resident_bytes;
        plan.pipeline_pending_bytes = pending_bytes;
        plan.pipeline_preserved_bytes = preserved_bytes;
        plan.pipeline_new_bytes = pending_bytes;
        plan.pipeline_removed_bytes = removed_bytes;
        plan.pipeline_pool_bytes = state.pool_bytes;
        plan.pipeline_required_bytes = required;
        plan.pipeline_fits = fits;
        if !fits
            || plan.state == StagingPlanState::Rejected
            || plan.state == StagingPlanState::Unknown
            || plan.accepted_layers.is_empty()
        {
            plan.state = StagingPlanState::Rejected;
            plan.notes.push(
                "pipeline requires active + pending + pool bytes within the effective budget"
                    .to_string(),
            );
            self.staging_plan = Some(plan.clone());
            return Err(ServiceError::Configuration(format!(
                "staged pipeline plan rejected: {}",
                serde_json::to_string(&plan)
                    .unwrap_or_else(|_| "plan serialization failed".to_string())
            )));
        }

        // Bound retained evicted buffers before any commit. The limit cannot
        // exceed the effective budget.
        let pool_limit = self
            .staging_pool_limit_bytes
            .min(plan.effective_budget_bytes.unwrap_or(0));
        session
            .set_staging_pool_limit(pool_limit)
            .map_err(|error| ServiceError::Execution(format!("pool limit failed: {error}")))?;

        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ServiceError::InvalidState {
                expected: "model loaded".to_string(),
                actual: self.state,
            })?;
        let engine = self.staging_engine.as_ref().ok_or_else(|| {
            ServiceError::Execution("staging engine is not initialized".to_string())
        })?;
        let mappings = self.tensor_mappings.as_ref().expect("mappings cached");

        let mut stage_error = None;
        let mut promoted_layers = Vec::new();
        let mut promoted_bytes = 0u64;
        for layer in &stage_layers {
            #[cfg(debug_assertions)]
            {
                if std::env::var("RAMFORGE_TEST_FAIL_PENDING_LAYER")
                    .ok()
                    .and_then(|value| value.parse::<u32>().ok())
                    == Some(*layer)
                {
                    stage_error = Some(format!("injected pending stage failure (layer {layer})"));
                    break;
                }
            }
            if let Err(error) = session.stage_layer_pending_cuda(*layer) {
                stage_error = Some(format!("pending stage layer {layer}: {error}"));
                break;
            }
            if let Ok(info) = session.pending_layer_info(*layer) {
                if info.promoted_from_standby {
                    promoted_layers.push(*layer);
                    promoted_bytes = promoted_bytes.saturating_add(info.device_bytes);
                }
            }
        }
        if let Some(message) = stage_error {
            let _ = session.discard_pending_cuda();
            return Err(ServiceError::Execution(message));
        }

        let mut tensors = Vec::new();
        for layer in &stage_layers {
            if promoted_layers.contains(layer) {
                continue;
            }
            let count = session
                .pending_tensor_count(*layer)
                .map_err(|error| ServiceError::Execution(error.to_string()))?;
            for index in 0..count {
                tensors.push(
                    session
                        .pending_tensor_info(*layer, index)
                        .map_err(|error| ServiceError::Execution(error.to_string()))?,
                );
            }
        }
        if tensors.is_empty() {
            let report = HostStagingReport {
                mode: self.host_staging_mode,
                batching: self.read_batching,
                tensors: 0,
                bytes_read: 0,
                read_groups: 0,
                coalesced_groups: 0,
                read_requests: 0,
                read_concurrency: self.storage_concurrency as u32,
                max_outstanding_bytes: 0,
                read_ns: 0,
                prepare_ns: 0,
                pinned_capacity: 0,
                pinned_reused: false,
                fallback_tensors: 0,
            };
            plan.host_staging = Some(report.clone());
            plan.standby_reused_layers = promoted_layers.clone();
            plan.standby_reused_bytes = promoted_bytes;
            self.last_host_staging = Some(report.clone());
            self.prefetch_prepared += 1;
            self.prefetch_bytes = self.prefetch_bytes.saturating_add(promoted_bytes);
            self.pending_pipeline = Some(PendingPipeline {
                plan: plan.clone(),
                layers: plan.accepted_layers.clone(),
                host: None,
                submit_ns: 0,
                host_staging: report,
                read_job: None,
                device_ptrs: Vec::new(),
                sizes: Vec::new(),
                partial,
                preserved_layers: preserved_layers.clone(),
                new_layers: promoted_layers.clone(),
                remove_layers: remove_layers.clone(),
                preserved_bytes,
                new_bytes: promoted_bytes,
                removed_bytes,
                promoted_layers,
                promoted_bytes,
            });
            self.staging_plan = Some(plan.clone());
            return Ok(plan);
        }

        let total: u64 = tensors.iter().map(|tensor| tensor.bytes).sum();
        let device_ptrs: Vec<u64> = tensors.iter().map(|tensor| tensor.device_ptr).collect();
        let sizes: Vec<u64> = tensors.iter().map(|tensor| tensor.bytes).collect();
        if self.host_read_async {
            let prepare_started = Instant::now();
            let (host, pinned_reused) = self.pinned_pool.acquire(engine, total)?;
            let capacity = host
                .info()
                .map_err(|error| ServiceError::Execution(error.to_string()))?
                .capacity;
            let job = match crate::host_staging::HostReadJob::prepare(
                model,
                &tensors,
                mappings,
                host,
                self.storage_concurrency,
                self.storage_max_outstanding_bytes,
            ) {
                Ok(job) => job,
                Err(error) => {
                    let _ = session.discard_pending_cuda();
                    return Err(error);
                }
            };
            let report = HostStagingReport {
                mode: HostStagingMode::Direct,
                batching: ReadBatching::Coalesced,
                tensors: tensors.len() as u64,
                bytes_read: total,
                read_groups: 0,
                coalesced_groups: 0,
                read_requests: 0,
                read_concurrency: self.storage_concurrency as u32,
                max_outstanding_bytes: 0,
                read_ns: 0,
                prepare_ns: duration_ns(prepare_started.elapsed()),
                pinned_capacity: capacity,
                pinned_reused,
                fallback_tensors: 0,
            };
            plan.host_staging = Some(report.clone());
            self.last_host_staging = Some(report.clone());
            self.prefetch_prepared += 1;
            self.prefetch_bytes = self.prefetch_bytes.saturating_add(total);
            self.pending_pipeline = Some(PendingPipeline {
                plan: plan.clone(),
                layers: plan.accepted_layers.clone(),
                host: None,
                submit_ns: 0,
                host_staging: report,
                read_job: Some(job),
                device_ptrs,
                sizes,
                partial,
                preserved_layers: preserved_layers.clone(),
                new_layers: stage_layers.clone(),
                remove_layers: remove_layers.clone(),
                preserved_bytes,
                new_bytes: total,
                removed_bytes,
                promoted_layers: promoted_layers.clone(),
                promoted_bytes,
            });
            self.staging_plan = Some(plan.clone());
            return Ok(plan);
        }
        let prepare_started = Instant::now();
        let (host, pinned_reused) = self.pinned_pool.acquire(engine, total)?;
        let (offsets, mut host_report) = match crate::host_staging::read_tensors_into_host(
            model,
            &tensors,
            &host,
            self.host_staging_mode,
            self.read_batching,
            mappings,
            self.storage_concurrency,
            self.storage_max_outstanding_bytes,
        ) {
            Ok(result) => result,
            Err(error) => {
                self.pinned_pool.release(host);
                return Err(error);
            }
        };
        host_report.prepare_ns = duration_ns(prepare_started.elapsed());
        host_report.pinned_reused = pinned_reused;
        let submit_started = Instant::now();
        let submit_result =
            engine.submit_batch_to_device_ptrs(&host, &offsets, &device_ptrs, &sizes);
        let submit_ns = duration_ns(submit_started.elapsed());
        if let Err(error) = submit_result {
            self.pinned_pool.release(host);
            let _ = session.discard_pending_cuda();
            return Err(ServiceError::Execution(format!(
                "pipeline transfer submission failed: {error}"
            )));
        }
        // Ensure the DMA has started before the caller begins compute. This is
        // the anchor for overlap accounting; it does not wait for completion.
        engine.wait_started().map_err(|error| {
            ServiceError::Execution(format!("pipeline transfer start failed: {error}"))
        })?;

        let host_staging = host_report;
        plan.host_staging = Some(host_staging.clone());
        plan.standby_reused_layers = promoted_layers.clone();
        plan.standby_reused_bytes = promoted_bytes;
        self.last_host_staging = Some(host_staging.clone());
        self.prefetch_prepared += 1;
        self.prefetch_bytes = self.prefetch_bytes.saturating_add(total);
        self.pending_pipeline = Some(PendingPipeline {
            plan: plan.clone(),
            layers: plan.accepted_layers.clone(),
            host: Some(host),
            submit_ns,
            host_staging,
            read_job: None,
            device_ptrs,
            sizes,
            partial,
            preserved_layers,
            new_layers: stage_layers,
            remove_layers,
            preserved_bytes,
            new_bytes: total,
            removed_bytes,
            promoted_layers,
            promoted_bytes,
        });
        self.staging_plan = Some(plan.clone());
        Ok(plan)
    }

    /// Wait for the pending pipeline transfer, invalidate the active graph and
    /// activate the pending selection. The old active buffers are recycled into
    /// the bounded pool. On failure before commit the active selection is kept.
    pub fn commit_staged_pipeline(&mut self) -> Result<PipelineCommitReport, ServiceError> {
        let mut pending =
            self.pending_pipeline
                .take()
                .ok_or_else(|| ServiceError::InvalidState {
                    expected: "pending pipeline".to_string(),
                    actual: self.state,
                })?;
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| ServiceError::InvalidState {
                expected: "context ready".to_string(),
                actual: self.state,
            })?;
        let engine = self.staging_engine.as_ref().ok_or_else(|| {
            ServiceError::Execution("staging engine is not initialized".to_string())
        })?;
        let mut async_host = None;
        let mut host_staging = pending.host_staging.clone();
        let mut host_read_wait_ns = None;
        let has_pending_transfer = pending.host.is_some() || pending.read_job.is_some();
        let (transfer, transfer_in_flight_at_commit, transfer_wait_ns) = if !has_pending_transfer {
            (crate::staging::TransferReport::default(), false, 0)
        } else if let Some(job) = pending.read_job.take() {
            let read_started = Instant::now();
            let (host, offsets, report, _) = match job.finish() {
                Ok(result) => result,
                Err((error, host)) => {
                    if let Some(host) = host {
                        self.pinned_pool.release(host);
                    } else {
                        self.state = ServiceState::Failed;
                    }
                    let _ = session.discard_pending_cuda();
                    return Err(ServiceError::Execution(format!(
                        "pipeline host read failed: {error}"
                    )));
                }
            };
            host_read_wait_ns = Some(duration_ns(read_started.elapsed()));
            host_staging = report;
            if let Err(error) = engine.submit_batch_to_device_ptrs(
                &host,
                &offsets,
                &pending.device_ptrs,
                &pending.sizes,
            ) {
                self.pinned_pool.release(host);
                let _ = session.discard_pending_cuda();
                return Err(ServiceError::Execution(format!(
                    "pipeline transfer submission failed: {error}"
                )));
            }
            let wait_started = Instant::now();
            let transfer = match engine.wait() {
                Ok(report) => report,
                Err(error) => {
                    self.pinned_pool.release(host);
                    let _ = session.discard_pending_cuda();
                    return Err(ServiceError::Execution(format!(
                        "pipeline transfer wait failed: {error}"
                    )));
                }
            };
            async_host = Some(host);
            (transfer, false, duration_ns(wait_started.elapsed()))
        } else {
            match engine.query() {
                Ok(Some(report)) => (report, false, 0),
                Ok(None) => {
                    let wait_started = Instant::now();
                    match engine.wait() {
                        Ok(report) => (report, true, duration_ns(wait_started.elapsed())),
                        Err(error) => {
                            let _ = session.discard_pending_cuda();
                            return Err(ServiceError::Execution(format!(
                                "pipeline transfer wait failed: {error}"
                            )));
                        }
                    }
                }
                Err(error) => {
                    let _ = session.discard_pending_cuda();
                    return Err(ServiceError::Execution(format!(
                        "pipeline transfer query failed: {error}"
                    )));
                }
            }
        };
        let invalidation_started = Instant::now();
        session.invalidate_graph().map_err(|error| {
            ServiceError::Execution(format!("pipeline invalidation failed: {error}"))
        })?;
        let invalidation_ns = duration_ns(invalidation_started.elapsed());
        let commit_started = Instant::now();
        if pending.partial {
            session
                .commit_pending_delta_cuda(&pending.remove_layers)
                .map_err(|error| {
                    ServiceError::Execution(format!("pipeline delta commit failed: {error}"))
                })?;
        } else {
            session.commit_pending_cuda().map_err(|error| {
                ServiceError::Execution(format!("pipeline commit failed: {error}"))
            })?;
        }
        let commit_ns = duration_ns(commit_started.elapsed());
        let layers = pending.layers.clone();
        if let Some(host) = pending.host.take() {
            self.pinned_pool.release(host);
        }
        if let Some(host) = async_host {
            self.pinned_pool.release(host);
        }
        let statuses = statuses_for_layers(session, &layers);
        self.staged_layers = layers.clone();
        self.staging_status = statuses;
        self.staging_plan = Some(pending.plan.clone());
        self.prefetch_committed += 1;
        self.prefetch_committed_bytes = self
            .prefetch_committed_bytes
            .saturating_add(host_staging.bytes_read);
        self.prefetch_preserved_bytes = self
            .prefetch_preserved_bytes
            .saturating_add(pending.preserved_bytes);
        self.prefetch_standby_bytes = self
            .prefetch_standby_bytes
            .saturating_add(pending.promoted_bytes);
        Ok(PipelineCommitReport {
            layers,
            transfer_bytes: transfer.bytes,
            transfer_duration_ns: transfer.duration_ns,
            transfer_wait_ns,
            submit_ns: pending.submit_ns,
            invalidation_ns,
            commit_ns,
            transfer_in_flight_at_commit,
            host_staging,
            host_read_wait_ns,
            partial: pending.partial,
            preserved_layers: pending.preserved_layers.clone(),
            new_layers: pending.new_layers.clone(),
            removed_layers: pending.remove_layers.clone(),
            preserved_bytes: pending.preserved_bytes,
            new_bytes: pending.new_bytes,
            removed_bytes: pending.removed_bytes,
            promoted_layers: pending.promoted_layers.clone(),
            promoted_bytes: pending.promoted_bytes,
        })
    }

    /// Wait for and discard a prepared pipeline selection without changing the
    /// active selection. Safe to call when no pipeline is pending.
    pub fn abort_staged_pipeline(&mut self) -> Result<(), ServiceError> {
        let Some(mut pending) = self.pending_pipeline.take() else {
            return Ok(());
        };
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| ServiceError::InvalidState {
                expected: "context ready".to_string(),
                actual: self.state,
            })?;
        if let Some(engine) = self.staging_engine.as_ref() {
            let _ = engine.wait();
        }
        self.prefetch_aborted += 1;
        self.prefetch_wasted_bytes = self
            .prefetch_wasted_bytes
            .saturating_add(pending.host_staging.bytes_read);
        if let Some(job) = pending.read_job.take() {
            match job.finish() {
                Ok((host, _, _, _)) => self.pinned_pool.release(host),
                Err((_, Some(host))) => self.pinned_pool.release(host),
                Err((_, None)) => self.state = ServiceState::Failed,
            }
        }
        if let Some(host) = pending.host.take() {
            self.pinned_pool.release(host);
        }
        session
            .discard_pending_cuda()
            .map_err(|error| ServiceError::Execution(error.to_string()))
    }

    pub fn has_pending_pipeline(&self) -> bool {
        self.pending_pipeline.is_some()
    }

    /// Transactional same-context replacement of the staged layer selection.
    ///
    /// The sequence is: plan against current residency, invalidate the graph
    /// and scheduler (quiescing outstanding work), evict removed layers, stage
    /// added layers, then leave the context invalidated so the next decode
    /// rebuilds against the new selection. The old selection is restored if a
    /// stage fails. If restoration also fails the service fails closed.
    pub fn replace_staged_layers(
        &mut self,
        request: StagingRequest,
    ) -> Result<StagingPlan, ServiceError> {
        if self.state != ServiceState::Ready {
            return Err(ServiceError::InvalidState {
                expected: "ready".to_string(),
                actual: self.state,
            });
        }
        self.ensure_tensor_mappings()?;
        let registry = crate::DeviceRegistry::discover();
        let mut plan = {
            let session = self
                .session
                .as_ref()
                .ok_or_else(|| ServiceError::InvalidState {
                    expected: "context ready".to_string(),
                    actual: self.state,
                })?;
            self.build_staging_plan_for(session, &registry, &request)?
        };
        self.staging_plan = Some(plan.clone());
        if plan.state == StagingPlanState::Rejected || plan.state == StagingPlanState::Unknown {
            return Err(ServiceError::Configuration(format!(
                "staged replacement rejected: {}",
                serde_json::to_string(&plan)
                    .unwrap_or_else(|_| "plan serialization failed".to_string())
            )));
        }
        let old_layers = plan.current_resident_layers.clone();
        if plan.evicted_layers.is_empty() && plan.newly_staged_layers.is_empty() {
            self.staged_layers = plan.accepted_layers.clone();
            self.staging_request = request;
            self.staging_status = self.current_staged_status();
            return Ok(plan);
        }

        type ReplacementOutcome = (
            Vec<StagedLayerStatus>,
            u64,
            u64,
            u64,
            u64,
            Option<HostStagingReport>,
        );
        let outcome: Result<ReplacementOutcome, (String, bool, Vec<StagedLayerStatus>)> = {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| ServiceError::InvalidState {
                    expected: "model loaded".to_string(),
                    actual: self.state,
                })?;
            let session = self
                .session
                .as_ref()
                .ok_or_else(|| ServiceError::InvalidState {
                    expected: "context ready".to_string(),
                    actual: self.state,
                })?;
            let engine = self.staging_engine.as_ref().ok_or_else(|| {
                ServiceError::Execution("staging engine is not initialized".to_string())
            })?;
            let mode = self.host_staging_mode;
            let batching = self.read_batching;
            let mappings = self.tensor_mappings.as_ref().expect("mappings cached");
            let pool = &mut self.pinned_pool;

            let total_started = Instant::now();
            let mut failure: Option<String> = None;

            let invalidation_started = Instant::now();
            if let Err(error) = session.invalidate_graph() {
                failure = Some(format!("graph invalidation failed: {error}"));
            }
            let invalidation_ns = duration_ns(invalidation_started.elapsed());

            let mut staged_now: Vec<u32> = Vec::new();
            let eviction_started = Instant::now();
            if failure.is_none() {
                for layer in &plan.evicted_layers {
                    if let Err(error) = session.evict_layer_cuda(*layer) {
                        failure = Some(format!("evict layer {layer}: {error}"));
                        break;
                    }
                }
            }
            let eviction_ns = duration_ns(eviction_started.elapsed());

            let staging_started = Instant::now();
            let mut host_aggregate: Option<HostStagingReport> = None;
            if failure.is_none() {
                for layer in &plan.newly_staged_layers {
                    match stage_layer_data(
                        model,
                        session,
                        engine,
                        pool,
                        mappings,
                        mode,
                        batching,
                        self.storage_concurrency,
                        self.storage_max_outstanding_bytes,
                        *layer,
                    ) {
                        Ok((_, report)) => {
                            if let Some(total) = host_aggregate.as_mut() {
                                total.accumulate(&report);
                            } else {
                                host_aggregate = Some(report);
                            }
                            staged_now.push(*layer);
                        }
                        Err(error) => {
                            failure = Some(format!("stage layer {layer}: {error}"));
                            break;
                        }
                    }
                }
            }
            let staging_ns = duration_ns(staging_started.elapsed());
            let total_ns = duration_ns(total_started.elapsed());

            match failure {
                None => Ok((
                    statuses_for_layers(session, &plan.accepted_layers),
                    invalidation_ns,
                    eviction_ns,
                    staging_ns,
                    total_ns,
                    host_aggregate,
                )),
                Some(message) => {
                    // Roll back to the previous selection: release partial new
                    // layers first, then re-stage every missing old layer.
                    for layer in &staged_now {
                        let _ = session.evict_layer_cuda(*layer);
                    }
                    let mut rollback_errors = Vec::new();
                    for layer in &old_layers {
                        if session.staged_layer_info(*layer).is_ok() {
                            continue;
                        }
                        if let Err(error) = stage_layer_data(
                            model,
                            session,
                            engine,
                            pool,
                            mappings,
                            mode,
                            batching,
                            self.storage_concurrency,
                            self.storage_max_outstanding_bytes,
                            *layer,
                        ) {
                            rollback_errors.push(format!("restage layer {layer}: {error}"));
                        }
                    }
                    let restored = statuses_for_layers(session, &old_layers);
                    if rollback_errors.is_empty() {
                        Err((message, true, restored))
                    } else {
                        Err((
                            format!("{message}; rollback failed: {}", rollback_errors.join("; ")),
                            false,
                            restored,
                        ))
                    }
                }
            }
        };

        match outcome {
            Ok((statuses, invalidation_ns, eviction_ns, staging_ns, total_ns, host_aggregate)) => {
                plan.host_staging = host_aggregate.clone();
                self.last_host_staging = host_aggregate;
                plan.replacement_invalidation_ns = Some(invalidation_ns);
                plan.replacement_eviction_ns = Some(eviction_ns);
                plan.replacement_staging_ns = Some(staging_ns);
                plan.replacement_total_ns = Some(total_ns);
                self.staged_layers = plan.accepted_layers.clone();
                self.staging_request = request;
                self.staging_status = statuses;
                self.staging_plan = Some(plan.clone());
                Ok(plan)
            }
            Err((message, true, restored)) => {
                self.staged_layers = old_layers;
                self.staging_status = restored;
                Err(ServiceError::Replacement {
                    message,
                    rolled_back: true,
                })
            }
            Err((message, false, _)) => {
                self.staged_layers.clear();
                self.staging_status.clear();
                self.state = ServiceState::Failed;
                self.errors.push(message.clone());
                Err(ServiceError::Replacement {
                    message,
                    rolled_back: false,
                })
            }
        }
    }

    fn ensure_tensor_mappings(&mut self) -> Result<(), ServiceError> {
        if self.tensor_mappings.is_some() {
            return Ok(());
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ServiceError::InvalidState {
                expected: "model loaded".to_string(),
                actual: self.state,
            })?;
        let mappings = model
            .tensor_mappings()
            .map_err(|error| ServiceError::Execution(error.to_string()))?;
        self.tensor_mappings = Some(
            mappings
                .into_iter()
                .map(|mapping| (mapping.name.clone(), mapping))
                .collect(),
        );
        Ok(())
    }

    fn build_staging_plan(
        &self,
        session: &crate::ForgeCoreSession,
        registry: &crate::DeviceRegistry,
    ) -> Result<StagingPlan, ServiceError> {
        self.build_staging_plan_for(session, registry, &self.staging_request)
    }

    fn build_staging_plan_for(
        &self,
        session: &crate::ForgeCoreSession,
        registry: &crate::DeviceRegistry,
        request: &StagingRequest,
    ) -> Result<StagingPlan, ServiceError> {
        let layer_count = session.model().metadata().layer_count;
        let observed_free = registry
            .devices()
            .iter()
            .find(|device| !device.is_cpu() && device.selectable)
            .and_then(|device| device.memory_free.bytes());
        let margin = request.safety_margin_bytes;
        let device_limit = observed_free.map(|free| free.saturating_sub(margin));
        let effective_budget = match (request.budget_bytes, device_limit) {
            (Some(configured), Some(device)) => Some(configured.min(device)),
            (Some(configured), None) => Some(configured),
            (None, Some(device)) => Some(device),
            (None, None) => None,
        };

        let current: Vec<(u32, u64)> = (0..session.staged_layer_count().unwrap_or(0))
            .filter_map(|index| {
                session
                    .staged_layer_at(index)
                    .ok()
                    .map(|info| (info.layer, info.device_bytes))
            })
            .collect();
        let current_layers: Vec<u32> = current.iter().map(|(layer, _)| *layer).collect();
        let current_bytes: u64 = current.iter().map(|(_, bytes)| *bytes).sum();

        let Some(effective_budget) = effective_budget else {
            return Ok(StagingPlan {
                schema_version: crate::staging_plan::STAGING_PLAN_SCHEMA_VERSION,
                requested_layers: request.layers.clone(),
                auto_selected: request.auto,
                selection_policy: self.selection_policy,
                configured_budget_bytes: request.budget_bytes,
                effective_budget_bytes: None,
                observed_free_device_bytes: observed_free,
                safety_margin_bytes: margin,
                existing_staged_bytes: current_bytes,
                planned_bytes: 0,
                available_bytes_after_plan: None,
                current_resident_layers: current_layers,
                current_resident_bytes: current_bytes,
                kept_layers: Vec::new(),
                evicted_layers: Vec::new(),
                newly_staged_layers: Vec::new(),
                reclaimable_bytes: 0,
                required_new_bytes: 0,
                temporary_headroom_bytes: 0,
                pipeline: false,
                pipeline_active_bytes: 0,
                pipeline_pending_bytes: 0,
                pipeline_preserved_bytes: 0,
                pipeline_new_bytes: 0,
                pipeline_removed_bytes: 0,
                pipeline_partial: false,
                standby_candidates: Vec::new(),
                standby_candidate_bytes: 0,
                standby_reused_layers: Vec::new(),
                standby_reused_bytes: 0,
                pipeline_pool_bytes: 0,
                pipeline_required_bytes: 0,
                pipeline_fits: false,
                host_staging: None,
                replacement_invalidation_ns: None,
                replacement_eviction_ns: None,
                replacement_staging_ns: None,
                replacement_total_ns: None,
                layers: Vec::new(),
                accepted_layers: Vec::new(),
                rejected_layers: Vec::new(),
                eviction_supported: true,
                allocation_reuse_supported: true,
                transfer_compute_overlap_supported: false,
                state: StagingPlanState::Unknown,
                notes: vec![
                    "configured budget and free device observation are unavailable".to_string(),
                ],
            });
        };

        let candidates: Vec<u32> = if request.auto {
            let ordered: Vec<u32> = (0..layer_count)
                // Static GPU placement already owns these layers; staging them
                // would allocate a duplicate device copy and waste VRAM.
                .filter(|layer| session.model().layer_is_cpu(*layer).unwrap_or(true))
                .collect();
            crate::staging_plan::order_candidate_layers(
                self.selection_policy,
                ordered,
                |layer| session.staged_layer_bytes(layer).ok(),
                |layer| current_layers.contains(&layer),
            )
        } else {
            for layer in &request.layers {
                if !session.model().layer_is_cpu(*layer).unwrap_or(true) {
                    return Err(ServiceError::Configuration(format!(
                        "layer {layer} is already offloaded by static GPU placement; staging it would duplicate device residency"
                    )));
                }
            }
            request.layers.clone()
        };
        let mut layers = Vec::new();
        let mut accepted_layers = Vec::new();
        let mut rejected_layers = Vec::new();
        let mut planned_bytes = 0u64;
        let mut any_unknown = false;
        for layer in candidates {
            let Ok(bytes) = session.staged_layer_bytes(layer) else {
                any_unknown = true;
                rejected_layers.push(layer);
                layers.push(StagingLayerPlan {
                    layer,
                    bytes: None,
                    accepted: false,
                    reason: Some("native staging preflight unavailable".to_string()),
                });
                continue;
            };
            let fits = planned_bytes.saturating_add(bytes) <= effective_budget;
            if fits {
                planned_bytes = planned_bytes.saturating_add(bytes);
                accepted_layers.push(layer);
                layers.push(StagingLayerPlan {
                    layer,
                    bytes: Some(bytes),
                    accepted: true,
                    reason: None,
                });
            } else {
                rejected_layers.push(layer);
                layers.push(StagingLayerPlan {
                    layer,
                    bytes: Some(bytes),
                    accepted: false,
                    reason: Some("exceeds effective staged-VRAM budget".to_string()),
                });
            }
        }
        if request.auto {
            layers.sort_by_key(|entry| entry.layer);
            accepted_layers.sort_unstable();
        }
        if !request.auto && !rejected_layers.is_empty() {
            planned_bytes = 0;
            for layer_plan in &mut layers {
                if layer_plan.accepted {
                    layer_plan.accepted = false;
                    layer_plan.reason =
                        Some("explicit staging request rejected as a set".to_string());
                    rejected_layers.push(layer_plan.layer);
                }
            }
            accepted_layers.clear();
        }
        let state = if any_unknown {
            StagingPlanState::Unknown
        } else if accepted_layers.is_empty() {
            StagingPlanState::Rejected
        } else if rejected_layers.is_empty() {
            StagingPlanState::Exact
        } else {
            StagingPlanState::Partial
        };

        let bytes_for = |layer: u32| {
            layers
                .iter()
                .find(|entry| entry.layer == layer)
                .and_then(|entry| entry.bytes)
        };
        let kept_layers: Vec<u32> = accepted_layers
            .iter()
            .copied()
            .filter(|layer| current_layers.contains(layer))
            .collect();
        let evicted_layers: Vec<u32> = current_layers
            .iter()
            .copied()
            .filter(|layer| !accepted_layers.contains(layer))
            .collect();
        let newly_staged_layers: Vec<u32> = accepted_layers
            .iter()
            .copied()
            .filter(|layer| !current_layers.contains(layer))
            .collect();
        let reclaimable_bytes: u64 = current
            .iter()
            .filter(|(layer, _)| evicted_layers.contains(layer))
            .map(|(_, bytes)| *bytes)
            .sum();
        let required_new_bytes: u64 = newly_staged_layers
            .iter()
            .filter_map(|layer| bytes_for(*layer))
            .sum();

        let mut notes = vec![
            "budget applies only to RAMforge-owned staged CUDA allocations".to_string(),
            "observed free device memory is not a reservation against other processes"
                .to_string(),
            "replacement uses invalidate -> evict -> allocate; old allocations are released before new ones are created"
                .to_string(),
            "evicted buffers may be retained in a bounded pool and reused; reuse requires invalidation"
                .to_string(),
            "graph substitutions for the new selection are reported after the next execution"
                .to_string(),
        ];
        if !evicted_layers.is_empty() {
            notes.push(
                "temporary headroom is zero for this sequence because new allocations occur after eviction"
                    .to_string(),
            );
        }

        Ok(StagingPlan {
            schema_version: crate::staging_plan::STAGING_PLAN_SCHEMA_VERSION,
            requested_layers: request.layers.clone(),
            auto_selected: request.auto,
            selection_policy: self.selection_policy,
            configured_budget_bytes: request.budget_bytes,
            effective_budget_bytes: Some(effective_budget),
            observed_free_device_bytes: observed_free,
            safety_margin_bytes: margin,
            existing_staged_bytes: current_bytes,
            planned_bytes,
            available_bytes_after_plan: Some(effective_budget.saturating_sub(planned_bytes)),
            current_resident_layers: current_layers,
            current_resident_bytes: current_bytes,
            kept_layers,
            evicted_layers,
            newly_staged_layers,
            reclaimable_bytes,
            required_new_bytes,
            temporary_headroom_bytes: 0,
            pipeline: false,
            pipeline_active_bytes: 0,
            pipeline_pending_bytes: 0,
            pipeline_preserved_bytes: 0,
            pipeline_new_bytes: 0,
            pipeline_removed_bytes: 0,
            pipeline_partial: false,
            standby_candidates: Vec::new(),
            standby_candidate_bytes: 0,
            standby_reused_layers: Vec::new(),
            standby_reused_bytes: 0,
            pipeline_pool_bytes: 0,
            pipeline_required_bytes: 0,
            pipeline_fits: false,
            host_staging: None,
            replacement_invalidation_ns: None,
            replacement_eviction_ns: None,
            replacement_staging_ns: None,
            replacement_total_ns: None,
            layers,
            accepted_layers,
            rejected_layers,
            eviction_supported: true,
            allocation_reuse_supported: true,
            transfer_compute_overlap_supported: false,
            state,
            notes,
        })
    }

    fn current_staged_status(&self) -> Vec<StagedLayerStatus> {
        if self.staged_layers.is_empty() {
            return Vec::new();
        }
        if let Some(session) = self.session.as_ref() {
            return self
                .staged_layers
                .iter()
                .filter_map(|layer| {
                    session
                        .staged_layer_info(*layer)
                        .ok()
                        .map(|info| StagedLayerStatus {
                            layer: *layer,
                            tensor_count: info.tensor_count,
                            device_bytes: info.device_bytes,
                            graph_substitutions: info.graph_substitutions,
                            promoted_from_standby: info.promoted_from_standby,
                        })
                })
                .collect();
        }
        self.staging_status.clone()
    }

    /// Native lifecycle/counter snapshot for the live context, if any.
    pub fn staging_lifecycle(&self) -> Option<StagingLifecycleStatus> {
        let session = self.session.as_ref()?;
        let state = session.staging_state().ok()?;
        let resident_layers = (0..state.resident_layers)
            .filter_map(|index| session.staged_layer_at(index).ok().map(|info| info.layer))
            .collect();
        Some(StagingLifecycleStatus {
            graph_phase: state.graph_phase.name().to_string(),
            graph_invalidated: state.graph_invalidated,
            resident_layers,
            resident_bytes: state.resident_bytes,
            pending_layers: state.pending_layers,
            pending_bytes: state.pending_bytes,
            pool_bytes: state.pool_bytes,
            pool_limit_bytes: state.pool_limit_bytes,
            pool_buffers: state.pool_buffers,
            commit_count: state.commit_count,
            standby_bytes: state.standby_bytes,
            standby_limit_bytes: state.standby_limit_bytes,
            standby_layers: state.standby_layers,
            standby_hits: state.standby_hits,
            allocation_count: state.allocation_count,
            release_count: state.release_count,
            reuse_count: state.reuse_count,
            staging_failures: state.staging_failures,
            graph_invalidation_count: state.graph_invalidation_count,
            graph_rebuild_count: state.graph_rebuild_count,
            scheduler_reset_count: state.scheduler_reset_count,
            graph_build_ns: state.graph_build_ns,
            last_graph_build_ns: state.last_graph_build_ns,
            invalidation_failures: state.invalidation_failures,
            quiescence_wait_ns: state.quiescence_wait_ns,
            last_quiescence_wait_ns: state.last_quiescence_wait_ns,
            eviction_supported: state.eviction_supported,
            allocation_reuse_supported: state.allocation_reuse_supported,
            transfer_compute_overlap_supported: state.transfer_compute_overlap_supported,
        })
    }

    /// Read the file-backed tensors of the given layers once so later staging
    /// comparisons are not biased by cold page-cache reads. Returns bytes read.
    pub fn warm_model_layers(&self, layers: &[u32]) -> Result<u64, ServiceError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ServiceError::InvalidState {
                expected: "model loaded".to_string(),
                actual: self.state,
            })?;
        let storage = model
            .tensor_storage()
            .map_err(|error| ServiceError::Execution(error.to_string()))?;
        let mut total = 0u64;
        for tensor in storage {
            let Some(layer) = tensor.layer else {
                continue;
            };
            if !layers.contains(&layer) || !tensor.file_backed {
                continue;
            }
            let length = usize::try_from(tensor.tensor_bytes)
                .map_err(|_| ServiceError::Execution("tensor is too large".to_string()))?;
            let mut bytes = vec![0u8; length];
            model
                .read_tensor_bytes(&tensor.name, 0, &mut bytes)
                .map_err(|error| ServiceError::Execution(error.to_string()))?;
            total = total.saturating_add(tensor.tensor_bytes);
        }
        Ok(total)
    }

    /// Generate tokens, streaming decoded text into `sink`.
    ///
    /// Cancellation is cooperative and checked between decodes. The service
    /// never switches model, backend, context, or plan during generation.
    pub fn generate<F>(
        &mut self,
        options: &GenerationOptions,
        cancel: &AtomicBool,
        mut sink: F,
    ) -> Result<GenerationReport, ServiceError>
    where
        F: FnMut(&str) -> Result<(), String>,
    {
        if self.state != ServiceState::Ready {
            return Err(ServiceError::InvalidState {
                expected: "ready".to_string(),
                actual: self.state,
            });
        }
        if options.prompt.is_empty() {
            return Err(ServiceError::Configuration("prompt is empty".to_string()));
        }
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| ServiceError::InvalidState {
                expected: "tokenizer loaded".to_string(),
                actual: self.state,
            })?;
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| ServiceError::InvalidState {
                expected: "session ready".to_string(),
                actual: self.state,
            })?;
        let prompt_tokens = tokenizer.encode(&options.prompt, true);
        if prompt_tokens.is_empty() {
            return Err(ServiceError::Tokenizer(
                "prompt tokenization produced no tokens".to_string(),
            ));
        }
        if prompt_tokens.len() >= session.context_length() as usize {
            return Err(ServiceError::Configuration(format!(
                "prompt has {} tokens but context length is {}",
                prompt_tokens.len(),
                session.context_length()
            )));
        }
        // Generation is repeatable within one live context: clear only the KV
        // cache, never staged weight allocations.
        session.clear_kv();
        self.state = ServiceState::Generating;
        let rss_before = process_rss();
        let io_before = process_io();
        let (minor_faults_before, major_faults_before) = process_faults();
        let mut token_io = Vec::new();
        let prompt_started = Instant::now();
        let prompt_io_before = process_io();
        let prompt_faults_before = process_faults();
        let mut logits = match session.decode_tokens(&prompt_tokens, 0) {
            Ok(logits) => logits,
            Err(error) => {
                self.state = ServiceState::Ready;
                return Err(ServiceError::Execution(error.to_string()));
            }
        };
        session.synchronize();
        let prompt_ns = duration_ns(prompt_started.elapsed());
        token_io.push(token_io_sample(
            TokenPhase::Prompt,
            0,
            prompt_started.elapsed(),
            prompt_io_before,
            process_io(),
            prompt_faults_before,
            process_faults(),
        ));
        let sampler = Sampler::new(options.temperature, options.top_k, options.top_p);
        let eos = tokenizer.eos_id;
        let mut decoder = tokenizer.decoder();
        let mut token_ids = Vec::new();
        let mut text = String::new();
        let mut first_token: Option<Duration> = None;
        let mut decode_started: Option<Instant> = None;
        let mut stop_reason = if options.max_tokens == 0 {
            "max_tokens"
        } else {
            "context_length"
        };
        let max_tokens = usize::try_from(options.max_tokens).unwrap_or(usize::MAX);
        for _ in 0..max_tokens {
            if cancel.load(Ordering::Relaxed) {
                stop_reason = "cancelled";
                break;
            }
            let current_length = prompt_tokens.len().saturating_add(token_ids.len());
            if current_length >= session.context_length() as usize {
                stop_reason = "context_length";
                break;
            }
            let next = sampler.sample(logits.values());
            if Some(next) == eos {
                stop_reason = "eos";
                break;
            }
            token_ids.push(next);
            if first_token.is_none() {
                first_token = Some(prompt_started.elapsed());
                decode_started = Some(Instant::now());
            }
            let piece = decoder.push(next);
            if !piece.is_empty() {
                text.push_str(&piece);
                if let Err(error) = sink(&piece) {
                    self.state = ServiceState::Ready;
                    return Err(ServiceError::Sink(error));
                }
            }
            if token_ids.len() >= max_tokens {
                stop_reason = "max_tokens";
                break;
            }
            let next_position = prompt_tokens
                .len()
                .checked_add(token_ids.len())
                .and_then(|length| length.checked_sub(1))
                .ok_or_else(|| ServiceError::Execution("token position overflow".to_string()))?;
            let next_position = u32::try_from(next_position)
                .map_err(|_| ServiceError::Execution("token position exceeds u32".to_string()))?;
            if cancel.load(Ordering::Relaxed) {
                stop_reason = "cancelled";
                break;
            }
            let decode_io_before = process_io();
            let decode_faults_before = process_faults();
            let decode_token_started = Instant::now();
            logits = match session.decode_tokens(&[next], next_position) {
                Ok(logits) => logits,
                Err(error) => {
                    self.state = ServiceState::Ready;
                    return Err(ServiceError::Execution(error.to_string()));
                }
            };
            session.synchronize();
            token_io.push(token_io_sample(
                TokenPhase::Decode,
                token_ids.len() as u32,
                decode_token_started.elapsed(),
                decode_io_before,
                process_io(),
                decode_faults_before,
                process_faults(),
            ));
        }
        text.push_str(&decoder.finish());
        let total_ns = duration_ns(prompt_started.elapsed());
        let decode_ns = decode_started.map(|started| duration_ns(started.elapsed()));
        let generated = token_ids.len() as u64;
        let decode_tokens = generated.saturating_sub(1);
        let io_after = process_io();
        let (minor_faults_after, major_faults_after) = process_faults();
        let memory_split = process_memory_split();
        let counter_delta =
            |before: Option<u64>, after: Option<u64>| Some(after?.saturating_sub(before?));
        let stats = GenerationStats {
            prompt_tokens: prompt_tokens.len() as u64,
            generated_tokens: generated,
            prompt_processing_ns: prompt_ns,
            time_to_first_token_ns: first_token.map(duration_ns),
            decode_ns,
            total_generation_ns: total_ns,
            prompt_tokens_per_second: rate(prompt_tokens.len() as u64, prompt_ns),
            decode_tokens_per_second: decode_ns.map(|ns| rate(decode_tokens, ns)),
            generation_tokens_per_second: rate(generated, total_ns),
            stop_reason: stop_reason.to_string(),
            rss_before_bytes: rss_before,
            rss_after_bytes: process_rss(),
            storage_read_bytes: counter_delta(
                io_before.as_ref().map(|io| io.read_bytes),
                io_after.as_ref().map(|io| io.read_bytes),
            ),
            rchar_bytes: counter_delta(
                io_before.as_ref().map(|io| io.rchar_bytes),
                io_after.as_ref().map(|io| io.rchar_bytes),
            ),
            read_syscalls: counter_delta(
                io_before.as_ref().map(|io| io.read_syscalls),
                io_after.as_ref().map(|io| io.read_syscalls),
            ),
            major_faults: counter_delta(major_faults_before, major_faults_after),
            minor_faults: counter_delta(minor_faults_before, minor_faults_after),
            pss_bytes: memory_split.map(|split| split.pss_bytes),
            pss_file_bytes: memory_split.and_then(|split| split.pss_file_bytes),
            pss_anon_bytes: memory_split.and_then(|split| split.pss_anon_bytes),
            anonymous_bytes: memory_split.map(|split| split.anonymous_bytes),
            swap_bytes: memory_split.map(|split| split.swap_bytes),
            token_io,
        };
        let report = GenerationReport {
            text,
            token_ids,
            stats,
            effective_context_tokens: session.context_length(),
            effective_batch_tokens: session.effective_batch_tokens(),
            effective_ubatch_tokens: session.effective_ubatch_tokens(),
            effective_threads: session.effective_threads(),
            effective_sequence_max: session.effective_sequence_max(),
        };
        self.state = ServiceState::Ready;
        if stop_reason == "cancelled" {
            return Err(ServiceError::Cancelled);
        }
        Ok(report)
    }

    /// Drop the context and model; the service can be loaded again.
    pub fn unload(&mut self) {
        let _ = self.abort_staged_pipeline();
        self.pinned_pool.clear();
        self.session = None;
        self.model = None;
        self.tokenizer = None;
        self.staging_status.clear();
        self.staging_plan = None;
        self.state = ServiceState::Terminated;
    }
}

impl Drop for InferenceService {
    fn drop(&mut self) {
        // Never drop a live context while a pipeline transfer or pending slot
        // may still reference device memory.
        let _ = self.abort_staged_pipeline();
    }
}

fn statuses_for_layers(
    session: &crate::ForgeCoreSession,
    layers: &[u32],
) -> Vec<StagedLayerStatus> {
    layers
        .iter()
        .filter_map(|layer| {
            session
                .staged_layer_info(*layer)
                .ok()
                .map(|info| StagedLayerStatus {
                    layer: *layer,
                    tensor_count: info.tensor_count,
                    device_bytes: info.device_bytes,
                    graph_substitutions: info.graph_substitutions,
                    promoted_from_standby: info.promoted_from_standby,
                })
        })
        .collect()
}

/// Allocate one native staged layer, copy its weights from the model and return
/// the live native metadata. On error the caller is responsible for releasing a
/// partially staged layer (native allocation is reclaimed by the context even if
/// the transfer fails).
#[allow(clippy::too_many_arguments)]
fn stage_layer_data(
    model: &ForgeCoreModel,
    session: &crate::ForgeCoreSession,
    engine: &Engine,
    pool: &mut PinnedPool,
    mappings: &HashMap<String, forge_core::TensorMappingInfo>,
    mode: HostStagingMode,
    batching: ReadBatching,
    concurrency: usize,
    max_outstanding_bytes: u64,
    layer: u32,
) -> Result<(StagedLayerStatus, HostStagingReport), ServiceError> {
    #[cfg(debug_assertions)]
    {
        if std::env::var("RAMFORGE_TEST_FAIL_STAGE_ALL").as_deref() == Ok("1") {
            return Err(ServiceError::Execution(
                "injected staging failure (all layers)".to_string(),
            ));
        }
        if std::env::var("RAMFORGE_TEST_FAIL_STAGE_LAYER")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            == Some(layer)
        {
            return Err(ServiceError::Execution(format!(
                "injected staging failure (layer {layer})"
            )));
        }
    }
    session
        .stage_layer_cuda(layer)
        .map_err(|error| ServiceError::Execution(error.to_string()))?;
    let count = session
        .staged_tensor_count(layer)
        .map_err(|error| ServiceError::Execution(error.to_string()))?;
    let mut tensors = Vec::with_capacity(count as usize);
    for index in 0..count {
        tensors.push(
            session
                .staged_tensor_info(layer, index)
                .map_err(|error| ServiceError::Execution(error.to_string()))?,
        );
    }
    let total: u64 = tensors.iter().map(|tensor| tensor.bytes).sum();
    let prepare_started = Instant::now();
    let (host, pinned_reused) = pool.acquire(engine, total)?;
    let (offsets, mut report) = match crate::host_staging::read_tensors_into_host(
        model,
        &tensors,
        &host,
        mode,
        batching,
        mappings,
        concurrency,
        max_outstanding_bytes,
    ) {
        Ok(result) => result,
        Err(error) => {
            pool.release(host);
            return Err(error);
        }
    };
    report.prepare_ns = duration_ns(prepare_started.elapsed());
    report.pinned_reused = pinned_reused;
    let device_ptrs: Vec<u64> = tensors.iter().map(|tensor| tensor.device_ptr).collect();
    let sizes: Vec<u64> = tensors.iter().map(|tensor| tensor.bytes).collect();
    if let Err(error) = engine.submit_batch_to_device_ptrs(&host, &offsets, &device_ptrs, &sizes) {
        pool.release(host);
        return Err(ServiceError::Execution(format!(
            "layer {layer} H2D submission failed: {error}"
        )));
    }
    if let Err(error) = engine.wait() {
        pool.release(host);
        return Err(ServiceError::Execution(format!(
            "layer {layer} H2D wait failed: {error}"
        )));
    }
    pool.release(host);
    let info = session
        .staged_layer_info(layer)
        .map_err(|error| ServiceError::Execution(error.to_string()))?;
    Ok((
        StagedLayerStatus {
            layer,
            tensor_count: info.tensor_count,
            device_bytes: info.device_bytes,
            graph_substitutions: info.graph_substitutions,
            promoted_from_standby: info.promoted_from_standby,
        },
        report,
    ))
}

pub(crate) fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn token_io_sample(
    phase: TokenPhase,
    token_index: u32,
    elapsed: Duration,
    io_before: Option<ProcessIo>,
    io_after: Option<ProcessIo>,
    faults_before: (Option<u64>, Option<u64>),
    faults_after: (Option<u64>, Option<u64>),
) -> TokenIo {
    let delta = |before: Option<u64>, after: Option<u64>| Some(after?.saturating_sub(before?));
    TokenIo {
        phase,
        token_index,
        elapsed_ns: duration_ns(elapsed),
        storage_read_bytes: delta(
            io_before.as_ref().map(|io| io.read_bytes),
            io_after.as_ref().map(|io| io.read_bytes),
        ),
        rchar_bytes: delta(
            io_before.as_ref().map(|io| io.rchar_bytes),
            io_after.as_ref().map(|io| io.rchar_bytes),
        ),
        read_syscalls: delta(
            io_before.as_ref().map(|io| io.read_syscalls),
            io_after.as_ref().map(|io| io.read_syscalls),
        ),
        major_faults: delta(faults_before.1, faults_after.1),
        minor_faults: delta(faults_before.0, faults_after.0),
    }
}

fn rate(tokens: u64, ns: u64) -> f64 {
    if ns == 0 {
        0.0
    } else {
        tokens as f64 * 1_000_000_000.0 / ns as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine_profile::{collect_machine_profile, MachineProfileConfig};
    use crate::model_requirements::RequirementBytes;
    use crate::runtime_plan::{CustomRuntimeRequest, RuntimeProfileKind};

    fn plan() -> RuntimePlan {
        let machine = collect_machine_profile(&MachineProfileConfig::default());
        let requirements = crate::model_requirements::ModelRequirements {
            schema_version: crate::model_requirements::MODEL_REQUIREMENTS_SCHEMA_VERSION,
            model_fingerprint: Some("fnv:test".to_string()),
            model_name: Some("test".to_string()),
            architecture: Some("llama".to_string()),
            architecture_info: None,
            quantization: Some("q4".to_string()),
            tensor_count: 1,
            file_bytes: RequirementBytes::exact(100, "test"),
            tensor_payload_bytes: RequirementBytes::exact(100, "test"),
            fully_resident_weight_bytes: RequirementBytes::exact(100, "test"),
            kv_cache_bytes: RequirementBytes::estimated(10, "test"),
            context_tokens: 128,
            batch_tokens: 1,
            model_max_context_tokens: Some(128),
            resident_weights_plus_kv_bytes: RequirementBytes::estimated(110, "test"),
            mapped_weights_plus_kv_bytes: RequirementBytes::estimated(10, "test"),
            unknown_requirements: Vec::new(),
            assumptions: Vec::new(),
        };
        RuntimePlan::resolve(
            &machine,
            &requirements,
            "/definitely/not/a/model.gguf",
            RuntimeProfileKind::Balanced,
            &CustomRuntimeRequest::default(),
        )
        .unwrap()
    }

    #[test]
    fn service_state_and_capabilities_are_explicit() {
        let service = InferenceService::new(plan()).expect("service");
        assert_eq!(service.state(), ServiceState::Empty);
        let capabilities = service.capabilities();
        assert_eq!(capabilities.api_version, INFERENCE_SERVICE_API_VERSION);
        assert!(capabilities.streaming);
        assert!(!capabilities.concurrent_generation);
        assert_eq!(service.status().state, ServiceState::Empty);
    }

    #[test]
    fn generation_requires_ready_state_and_nonempty_prompt() {
        let mut service = InferenceService::new(plan()).expect("service");
        let cancel = AtomicBool::new(false);
        let error = service
            .generate(&GenerationOptions::default(), &cancel, |_| Ok(()))
            .unwrap_err();
        assert!(matches!(error, ServiceError::InvalidState { .. }));
        let error = service
            .generate(
                &GenerationOptions {
                    prompt: "hi".to_string(),
                    ..GenerationOptions::default()
                },
                &cancel,
                |_| Ok(()),
            )
            .unwrap_err();
        assert!(matches!(error, ServiceError::InvalidState { .. }));
    }

    #[test]
    fn gpu_plan_is_rejected_without_silent_cpu_fallback() {
        let mut plan = plan();
        plan.effective.execution_mode = crate::runtime_plan::ExecutionMode::Hybrid;
        plan.effective.gpu_layers = 1;
        plan.effective.backend = "hybrid".to_string();
        let mut service = InferenceService::new(plan).expect("service");
        let error = service.load().unwrap_err();
        assert!(matches!(
            error,
            ServiceError::UnsupportedExecution(_)
                | ServiceError::Configuration(_)
                | ServiceError::Model(_)
        ));
        assert_eq!(service.state(), ServiceState::Empty);
    }

    #[test]
    fn model_load_failure_is_typed_and_leaves_state_empty() {
        let mut service = InferenceService::new(plan()).expect("service");
        let error = service.load().unwrap_err();
        assert!(matches!(error, ServiceError::Model(_)));
        assert_eq!(service.state(), ServiceState::Empty);
    }
}
