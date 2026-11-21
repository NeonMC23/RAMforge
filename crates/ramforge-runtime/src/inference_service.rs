//! Transport-independent operational inference service.
//!
//! The service owns one model/context at a time, executes through forgeCORE,
//! streams decoded text through a caller callback, and supports cooperative
//! cancellation between token decodes. It is not an inference implementation.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::forgecore::{
    ForgeCoreContextOptions, ForgeCoreModel, ForgeCoreModelOptions, ForgeCoreSession,
};
use crate::gpu_observability::{
    collect_observability, GpuObservability, StagingCounters, StagingObservation, StagingState,
};
use crate::memory_report::process_rss;
use crate::runtime_plan::RuntimePlan;
use crate::sampling::Sampler;
use crate::staging::{Engine, MemoryType, StagingError};
use crate::staging_plan::{StagingLayerPlan, StagingPlan, StagingPlanState, StagingRequest};
use ramforge_core::{discover_model_set, Tokenizer};

pub const INFERENCE_SERVICE_API_VERSION: u32 = 1;

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

    pub fn with_staging_request(mut self, mut request: StagingRequest) -> Self {
        request.layers.sort_unstable();
        request.layers.dedup();
        self.staged_layers = request.layers.clone();
        self.staging_request = request;
        self
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
            if self.plan.effective.execution_mode != crate::runtime_plan::ExecutionMode::Cpu {
                return Err(ServiceError::UnsupportedExecution(
                    "staged layers require a CPU plan; GPU/hybrid plans use normal placement"
                        .to_string(),
                ));
            }
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

    /// Create the single reusable context for this service.
    pub fn create_session(&mut self) -> Result<(), ServiceError> {
        if self.state != ServiceState::ModelLoaded && self.state != ServiceState::Ready {
            return Err(ServiceError::InvalidState {
                expected: "model_loaded or ready".to_string(),
                actual: self.state,
            });
        }
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
                trace_max_events: None,
            })
            .map_err(|error| ServiceError::Context(error.to_string()))?;
        let effective_context = session.context_length();
        if effective_context == 0 {
            return Err(ServiceError::Context(
                "native context reported zero length".to_string(),
            ));
        }
        self.session = Some(session);
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
        let mut statuses = Vec::with_capacity(self.staged_layers.len());
        for layer in &self.staged_layers {
            session
                .stage_layer_cuda(*layer)
                .map_err(|error| ServiceError::Execution(error.to_string()))?;
            let count = session
                .staged_tensor_count(*layer)
                .map_err(|error| ServiceError::Execution(error.to_string()))?;
            let mut tensors = Vec::with_capacity(count as usize);
            for index in 0..count {
                tensors.push(
                    session
                        .staged_tensor_info(*layer, index)
                        .map_err(|error| ServiceError::Execution(error.to_string()))?,
                );
            }
            let total: u64 = tensors.iter().map(|tensor| tensor.bytes).sum();
            let mut host = engine
                .allocate(MemoryType::PinnedHost, total)
                .map_err(|error| ServiceError::Execution(error.to_string()))?;
            let mut offset = 0u64;
            for tensor in &tensors {
                let length = usize::try_from(tensor.bytes).map_err(|_| {
                    ServiceError::Execution("staged tensor is too large".to_string())
                })?;
                let mut bytes = vec![0u8; length];
                model
                    .read_tensor_bytes(&tensor.name, 0, &mut bytes)
                    .map_err(|error| ServiceError::Execution(error.to_string()))?;
                host.write_host(offset, &bytes)
                    .map_err(|error| ServiceError::Execution(error.to_string()))?;
                engine
                    .submit_to_device_ptr(&host, offset, tensor.device_ptr, tensor.bytes)
                    .map_err(|error| ServiceError::Execution(error.to_string()))?;
                engine
                    .wait()
                    .map_err(|error| ServiceError::Execution(error.to_string()))?;
                offset += tensor.bytes;
            }
            host.release()
                .map_err(|error| ServiceError::Execution(error.to_string()))?;
            let info = session
                .staged_layer_info(*layer)
                .map_err(|error| ServiceError::Execution(error.to_string()))?;
            statuses.push(StagedLayerStatus {
                layer: *layer,
                tensor_count: info.tensor_count,
                device_bytes: info.device_bytes,
                graph_substitutions: info.graph_substitutions,
            });
        }
        self.staging_status = statuses;
        Ok(())
    }

    fn build_staging_plan(
        &self,
        session: &crate::ForgeCoreSession,
        registry: &crate::DeviceRegistry,
    ) -> Result<StagingPlan, ServiceError> {
        let layer_count = session.model().metadata().layer_count;
        let observed_free = registry
            .devices()
            .iter()
            .find(|device| !device.is_cpu() && device.selectable)
            .and_then(|device| device.memory_free.bytes());
        let margin = self.staging_request.safety_margin_bytes;
        let device_limit = observed_free.map(|free| free.saturating_sub(margin));
        let effective_budget = match (self.staging_request.budget_bytes, device_limit) {
            (Some(configured), Some(device)) => Some(configured.min(device)),
            (Some(configured), None) => Some(configured),
            (None, Some(device)) => Some(device),
            (None, None) => None,
        };
        let Some(effective_budget) = effective_budget else {
            return Ok(StagingPlan {
                schema_version: crate::staging_plan::STAGING_PLAN_SCHEMA_VERSION,
                requested_layers: self.staging_request.layers.clone(),
                auto_selected: self.staging_request.auto,
                configured_budget_bytes: self.staging_request.budget_bytes,
                effective_budget_bytes: None,
                observed_free_device_bytes: observed_free,
                safety_margin_bytes: margin,
                existing_staged_bytes: 0,
                planned_bytes: 0,
                available_bytes_after_plan: None,
                layers: Vec::new(),
                accepted_layers: Vec::new(),
                rejected_layers: Vec::new(),
                eviction_supported: false,
                allocation_reuse_supported: false,
                transfer_compute_overlap_supported: false,
                state: StagingPlanState::Unknown,
                notes: vec![
                    "configured budget and free device observation are unavailable".to_string(),
                ],
            });
        };
        let candidates: Vec<u32> = if self.staging_request.auto {
            (0..layer_count).collect()
        } else {
            self.staging_request.layers.clone()
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
        if !self.staging_request.auto && !rejected_layers.is_empty() {
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
        Ok(StagingPlan {
            schema_version: crate::staging_plan::STAGING_PLAN_SCHEMA_VERSION,
            requested_layers: self.staging_request.layers.clone(),
            auto_selected: self.staging_request.auto,
            configured_budget_bytes: self.staging_request.budget_bytes,
            effective_budget_bytes: Some(effective_budget),
            observed_free_device_bytes: observed_free,
            safety_margin_bytes: margin,
            existing_staged_bytes: 0,
            planned_bytes,
            available_bytes_after_plan: Some(effective_budget.saturating_sub(planned_bytes)),
            layers,
            accepted_layers,
            rejected_layers,
            eviction_supported: false,
            allocation_reuse_supported: false,
            transfer_compute_overlap_supported: false,
            state,
            notes: vec![
                "budget applies only to RAMforge-owned staged CUDA allocations".to_string(),
                "observed free device memory is not a reservation against other processes"
                    .to_string(),
                "staged layers are static for the context lifetime; eviction/reuse unsupported"
                    .to_string(),
                "graph substitutions are reported after execution".to_string(),
            ],
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
                        })
                })
                .collect();
        }
        self.staging_status.clone()
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
        self.state = ServiceState::Generating;
        let rss_before = process_rss();
        let prompt_started = Instant::now();
        let mut logits = match session.decode_tokens(&prompt_tokens, 0) {
            Ok(logits) => logits,
            Err(error) => {
                self.state = ServiceState::Ready;
                return Err(ServiceError::Execution(error.to_string()));
            }
        };
        session.synchronize();
        let prompt_ns = duration_ns(prompt_started.elapsed());
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
            logits = match session.decode_tokens(&[next], next_position) {
                Ok(logits) => logits,
                Err(error) => {
                    self.state = ServiceState::Ready;
                    return Err(ServiceError::Execution(error.to_string()));
                }
            };
            session.synchronize();
        }
        text.push_str(&decoder.finish());
        let total_ns = duration_ns(prompt_started.elapsed());
        let decode_ns = decode_started.map(|started| duration_ns(started.elapsed()));
        let generated = token_ids.len() as u64;
        let decode_tokens = generated.saturating_sub(1);
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
        self.session = None;
        self.model = None;
        self.tokenizer = None;
        self.staging_status.clear();
        self.staging_plan = None;
        self.state = ServiceState::Terminated;
    }
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
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
