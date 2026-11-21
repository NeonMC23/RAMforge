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
use crate::memory_report::process_rss;
use crate::runtime_plan::RuntimePlan;
use crate::sampling::Sampler;
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
            model_instance_id: model_id,
            context_id,
            rss_bytes: process_rss(),
            errors: self.errors.clone(),
        }
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
        let execution = crate::ExecutionConfig {
            device: crate::DeviceSelection::Cpu,
            cpu_thread_count: self.plan.effective.threads as usize,
            context_length: Some(self.plan.effective.context_tokens),
            check_tensors: false,
        };
        let mut options = ForgeCoreModelOptions {
            use_extra_bufts: self.plan.effective.weight_repacking,
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
        self.tokenizer = Some(tokenizer);
        self.model = Some(model);
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
        Ok(())
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
    fn model_load_failure_is_typed_and_leaves_state_empty() {
        let mut service = InferenceService::new(plan()).expect("service");
        let error = service.load().unwrap_err();
        assert!(matches!(error, ServiceError::Model(_)));
        assert_eq!(service.state(), ServiceState::Empty);
    }
}
