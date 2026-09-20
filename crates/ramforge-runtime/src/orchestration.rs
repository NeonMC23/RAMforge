//! Explicit high-level sequencing from discovery through runtime construction.
//!
//! This layer owns order and error staging only. Strategy and resource choices
//! remain in `Planner`; `PlanCompiler` remains the only translation from an
//! `ExecutionPlan` to a concrete `RuntimeConfig`.

use std::fmt;
use std::path::Path;

use ramforge_core::datasource::GgufDataSource;
use ramforge_core::{DataSourceError, ModelInfo};

use crate::discovery::{
    MachineDiscovery, MachineDiscoveryError, StorageDiscovery, StorageDiscoveryError,
};
use crate::inference::InferenceEngine;
use crate::plan::{plan_model, PlanResult};
use crate::plan_compiler::{PlanCompilationContext, PlanCompilationError, PlanCompiler};
use crate::planner::{
    CalibrationResult, CapabilitySet, ExecutionPlan, MachineProfile, ModelProfile, Planner,
    PlannerError, StorageProfile, UserProfile, PROFILE_SCHEMA_VERSION,
};
use crate::runtime_config::RuntimeConfig;

#[derive(Debug, Clone, Copy)]
pub struct OrchestrationRequest<'a> {
    pub model_path: &'a Path,
    pub user_profile: &'a UserProfile,
    pub calibration: Option<&'a CalibrationResult>,
}

impl<'a> OrchestrationRequest<'a> {
    pub fn new(model_path: &'a Path, user_profile: &'a UserProfile) -> Self {
        Self {
            model_path,
            user_profile,
            calibration: None,
        }
    }

    pub fn with_calibration(mut self, calibration: &'a CalibrationResult) -> Self {
        self.calibration = Some(calibration);
        self
    }
}

/// Planning-only facts retained for interactive inspection, optional
/// calibration, Planner invocation, and explicit compilation. No model weights,
/// runtime allocations, or inference state are owned here.
#[derive(Debug)]
pub struct PlanningSession {
    pub machine: MachineProfile,
    pub model: ModelProfile,
    pub model_info: ModelInfo,
    pub storage: StorageProfile,
    pub capabilities: CapabilitySet,
    pub static_plan: PlanResult,
    data_source: Option<GgufDataSource>,
}

impl PlanningSession {
    pub fn create_plan(
        &self,
        user_profile: &UserProfile,
        calibration: Option<&CalibrationResult>,
    ) -> Result<ExecutionPlan, OrchestrationError> {
        Planner
            .plan(
                user_profile,
                &self.machine,
                &self.model,
                &self.storage,
                &self.capabilities,
                &self.static_plan,
                calibration,
            )
            .map_err(OrchestrationError::Planning)
    }

    pub fn compile_plan(
        &self,
        execution_plan: &ExecutionPlan,
    ) -> Result<RuntimeConfig, OrchestrationError> {
        compile_runtime_config(
            execution_plan,
            &self.machine,
            &self.model,
            &self.storage,
            &self.capabilities,
            &self.static_plan,
        )
    }

    /// Compile and activate only this already analyzed plan. Compilation occurs
    /// before ownership of the retained datasource is transferred.
    pub fn activate_plan(
        &mut self,
        execution_plan: &ExecutionPlan,
    ) -> Result<InferenceEngine, OrchestrationError> {
        let runtime_config = self.compile_plan(execution_plan)?;
        let data_source = self
            .data_source
            .take()
            .ok_or(OrchestrationError::RuntimeSourceUnavailable)?;
        construct_runtime(data_source, runtime_config)
    }

    pub fn compilation_context(&self) -> PlanCompilationContext<'_> {
        PlanCompilationContext {
            machine: &self.machine,
            model: &self.model,
            storage: &self.storage,
            capabilities: &self.capabilities,
            static_plan: &self.static_plan,
        }
    }
}

#[derive(Debug)]
pub struct OrchestratedRuntime {
    pub execution_plan: ExecutionPlan,
    pub engine: InferenceEngine,
}

impl OrchestratedRuntime {
    pub fn runtime_config(&self) -> &RuntimeConfig {
        self.engine.runtime_config()
    }
}

#[derive(Debug)]
pub enum OrchestrationError {
    StorageDiscovery(StorageDiscoveryError),
    ModelInspection(DataSourceError),
    ModelProfileConstruction(PlannerError),
    StaticPlanning(String),
    MachineDiscovery(MachineDiscoveryError),
    CapabilityConstruction { binding: &'static str },
    Planning(PlannerError),
    PlanCompilation(PlanCompilationError),
    RuntimeSourceUnavailable,
    RuntimeConstruction(String),
}

impl fmt::Display for OrchestrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StorageDiscovery(error) => write!(formatter, "storage discovery failed: {error}"),
            Self::ModelInspection(error) => write!(formatter, "GGUF inspection failed: {error}"),
            Self::ModelProfileConstruction(error) => {
                write!(formatter, "model profile construction failed: {error}")
            }
            Self::StaticPlanning(error) => {
                write!(formatter, "static execution planning failed: {error}")
            }
            Self::MachineDiscovery(error) => write!(formatter, "machine discovery failed: {error}"),
            Self::CapabilityConstruction { binding } => {
                write!(formatter, "capability construction failed: {binding}")
            }
            Self::Planning(error) => write!(formatter, "deterministic planning failed: {error}"),
            Self::PlanCompilation(error) => write!(formatter, "plan compilation failed: {error}"),
            Self::RuntimeSourceUnavailable => {
                write!(
                    formatter,
                    "validated model datasource is no longer available"
                )
            }
            Self::RuntimeConstruction(error) => {
                write!(formatter, "runtime construction failed: {error}")
            }
        }
    }
}

impl std::error::Error for OrchestrationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::StorageDiscovery(error) => Some(error),
            Self::ModelInspection(error) => Some(error),
            Self::ModelProfileConstruction(error) | Self::Planning(error) => Some(error),
            Self::MachineDiscovery(error) => Some(error),
            Self::PlanCompilation(error) => Some(error),
            Self::StaticPlanning(_)
            | Self::CapabilityConstruction { .. }
            | Self::RuntimeSourceUnavailable
            | Self::RuntimeConstruction(_) => None,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RuntimeOrchestrator;

impl RuntimeOrchestrator {
    /// Inspect and discover everything needed for planning without loading
    /// model weights or constructing inference state.
    pub fn analyze(
        &self,
        model_path: &Path,
        ram_budget_bytes: u64,
    ) -> Result<PlanningSession, OrchestrationError> {
        self.analyze_retained(model_path, ram_budget_bytes)
    }

    pub fn orchestrate(
        &self,
        request: OrchestrationRequest<'_>,
    ) -> Result<OrchestratedRuntime, OrchestrationError> {
        let mut session =
            self.analyze_retained(request.model_path, request.user_profile.ram_budget_bytes)?;
        let execution_plan = session.create_plan(request.user_profile, request.calibration)?;
        let engine = session.activate_plan(&execution_plan)?;

        Ok(OrchestratedRuntime {
            execution_plan,
            engine,
        })
    }

    fn analyze_retained(
        &self,
        model_path: &Path,
        ram_budget_bytes: u64,
    ) -> Result<PlanningSession, OrchestrationError> {
        let storage = StorageDiscovery
            .discover(model_path)
            .map_err(OrchestrationError::StorageDiscovery)?;

        // GgufDataSource parses metadata/descriptors and opens the retained
        // synchronized handle exactly once. The full orchestration path moves
        // the same value into the engine; planning-only callers drop it.
        let data_source = GgufDataSource::open(&storage.model_path)
            .map_err(OrchestrationError::ModelInspection)?;
        let model_info = data_source.model().info();
        let model = ModelProfile::from_gguf(data_source.model());
        model
            .validate()
            .map_err(OrchestrationError::ModelProfileConstruction)?;
        let static_plan = plan_model(data_source.model(), ram_budget_bytes)
            .map_err(OrchestrationError::StaticPlanning)?;

        let machine = MachineDiscovery
            .discover()
            .map_err(OrchestrationError::MachineDiscovery)?;
        let planner = Planner;
        let capabilities =
            construct_capabilities(&planner, &machine, &model, &storage, &static_plan)?;
        Ok(PlanningSession {
            machine,
            model,
            model_info,
            storage,
            capabilities,
            static_plan,
            data_source: Some(data_source),
        })
    }
}

fn construct_capabilities(
    planner: &Planner,
    machine: &MachineProfile,
    model: &ModelProfile,
    storage: &StorageProfile,
    static_plan: &PlanResult,
) -> Result<CapabilitySet, OrchestrationError> {
    let capabilities = planner.derive_capabilities(machine, model, storage, static_plan);
    for (matches, binding) in [
        (
            capabilities.schema_version == PROFILE_SCHEMA_VERSION,
            "schema version",
        ),
        (
            capabilities.machine_fingerprint == machine.fingerprint(),
            "machine fingerprint",
        ),
        (
            capabilities.model_fingerprint == model.identity.descriptor_fingerprint,
            "model fingerprint",
        ),
        (
            capabilities.storage_fingerprint == storage.fingerprint(),
            "storage fingerprint",
        ),
    ] {
        if !matches {
            return Err(OrchestrationError::CapabilityConstruction { binding });
        }
    }
    Ok(capabilities)
}

fn construct_runtime(
    data_source: GgufDataSource,
    runtime_config: RuntimeConfig,
) -> Result<InferenceEngine, OrchestrationError> {
    InferenceEngine::from_data_source_with_runtime_config(data_source, runtime_config)
        .map_err(OrchestrationError::RuntimeConstruction)
}

fn compile_runtime_config(
    execution_plan: &ExecutionPlan,
    machine: &MachineProfile,
    model: &ModelProfile,
    storage: &StorageProfile,
    capabilities: &CapabilitySet,
    static_plan: &PlanResult,
) -> Result<RuntimeConfig, OrchestrationError> {
    PlanCompiler
        .compile(
            execution_plan,
            PlanCompilationContext {
                machine,
                model,
                storage,
                capabilities,
                static_plan,
            },
        )
        .map_err(OrchestrationError::PlanCompilation)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::{tempdir, NamedTempFile};

    use super::*;
    use crate::backend::ComputeBackend as _;
    use crate::inference::tests::create_tiny_llama_gguf;
    #[cfg(target_os = "linux")]
    use crate::planner::{
        observation_quality_basis_points, AdvancedOverrides, CalibrationLevel,
        FeasibilityRejection, GpuPreference, MeasurementSample, ObservationMetric,
        ObservationStatus, ObservationStatusReason, ObservationUnit, PerformanceObservation,
        PlanReasonCode, StrategyId, CALIBRATION_PLAN_VERSION, CALIBRATION_RULESET_VERSION,
    };

    const TEST_RAM_BYTES: u64 = 10_000;

    fn user_profile() -> UserProfile {
        let mut user = UserProfile::new(
            TEST_RAM_BYTES,
            crate::planner::OperatingMode::BalancedNormal,
        );
        user.cpu_thread_limit = Some(1);
        user
    }

    #[cfg(target_os = "linux")]
    struct PreparedPlan {
        data_source: GgufDataSource,
        machine: MachineProfile,
        model: ModelProfile,
        storage: StorageProfile,
        capabilities: CapabilitySet,
        static_plan: PlanResult,
        execution_plan: ExecutionPlan,
    }

    #[cfg(target_os = "linux")]
    fn prepare_plan(model_path: &Path, user: &UserProfile) -> PreparedPlan {
        let storage = StorageDiscovery.discover(model_path).unwrap();
        let data_source = GgufDataSource::open(&storage.model_path).unwrap();
        let model = ModelProfile::from_gguf(data_source.model());
        let static_plan = plan_model(data_source.model(), user.ram_budget_bytes).unwrap();
        let machine = MachineDiscovery.discover().unwrap();
        let planner = Planner;
        let capabilities = planner.derive_capabilities(&machine, &model, &storage, &static_plan);
        let execution_plan = planner
            .plan(
                user,
                &machine,
                &model,
                &storage,
                &capabilities,
                &static_plan,
                None,
            )
            .unwrap();
        PreparedPlan {
            data_source,
            machine,
            model,
            storage,
            capabilities,
            static_plan,
            execution_plan,
        }
    }

    #[cfg(target_os = "linux")]
    fn matching_calibration(model_path: &Path) -> CalibrationResult {
        let storage = StorageDiscovery.discover(model_path).unwrap();
        let data_source = GgufDataSource::open(&storage.model_path).unwrap();
        let model = ModelProfile::from_gguf(data_source.model());
        let machine = MachineDiscovery.discover().unwrap();
        let sample = MeasurementSample {
            elapsed_ns: 1_000_000_000,
            units_processed: 1_000_000,
        };
        CalibrationResult {
            schema_version: PROFILE_SCHEMA_VERSION,
            identifier: "orchestration-calibration".to_string(),
            level: CalibrationLevel::Quick,
            machine_fingerprint: machine.fingerprint(),
            model_fingerprint: Some(model.identity.descriptor_fingerprint),
            storage_fingerprint: Some(storage.fingerprint()),
            calibration_plan_identifier: "orchestration-calibration-plan".to_string(),
            calibration_plan_version: CALIBRATION_PLAN_VERSION,
            calibration_ruleset_version: CALIBRATION_RULESET_VERSION,
            observations: vec![PerformanceObservation {
                identifier: "orchestration-sequential-read".to_string(),
                metric: ObservationMetric::SequentialReadThroughput,
                unit: ObservationUnit::BytesPerSecond,
                value: Some(1_000_000),
                status: ObservationStatus::Measured,
                status_reason: None,
                strategy: None,
                quantization_format: None,
                calibration_level: CalibrationLevel::Quick,
                workload_bytes: 1_000_000,
                workload_elements: 0,
                measurement_duration_ns: sample.elapsed_ns,
                sample_count: 1,
                raw_samples: vec![sample],
                timestamp_unix_seconds: None,
                quality_basis_points: observation_quality_basis_points(&[sample]),
            }],
        }
    }

    #[test]
    fn test_orchestration_request_is_minimal_and_calibration_is_optional() {
        let user = user_profile();
        let request = OrchestrationRequest::new(Path::new("/tmp/model.gguf"), &user);
        assert_eq!(request.model_path, Path::new("/tmp/model.gguf"));
        assert!(request.calibration.is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_planning_only_analysis_exposes_real_profiles_without_runtime_state() {
        let model_file = create_tiny_llama_gguf();
        let session = RuntimeOrchestrator
            .analyze(model_file.path(), TEST_RAM_BYTES)
            .unwrap();
        assert_eq!(session.model.tensor_count, 11);
        assert_eq!(session.model_info.embedding_length, Some(8));
        assert_eq!(session.model_info.vocab_size, Some(16));
        assert_eq!(
            session.storage.file_size_bytes,
            Some(session.model.file_size_bytes)
        );
        assert_eq!(
            session.capabilities.model_fingerprint,
            session.model.identity.descriptor_fingerprint
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_plan_activation_compiles_before_transferring_runtime_source() {
        let model_file = create_tiny_llama_gguf();
        let user = user_profile();
        let mut session = RuntimeOrchestrator
            .analyze(model_file.path(), user.ram_budget_bytes)
            .unwrap();
        let plan = session.create_plan(&user, None).unwrap();
        let mut rejected = plan.clone();
        rejected.cpu_thread_count = 0;
        assert!(matches!(
            session.activate_plan(&rejected),
            Err(OrchestrationError::PlanCompilation(
                crate::plan_compiler::PlanCompilationError::ZeroThreadCount
            ))
        ));

        let mut engine = session.activate_plan(&plan).unwrap();
        assert_eq!(
            engine.runtime_config().cpu_thread_count,
            plan.cpu_thread_count
        );
        assert_eq!(engine.budget.total_bytes(), plan.ram_budget_bytes);
        let (tokens, generated_text) = engine
            .generate("hello", 1, &crate::sampling::Sampler::greedy())
            .unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(generated_text, engine.tokenizer.decode(&tokens));
        drop(engine);
        assert!(matches!(
            session.activate_plan(&plan),
            Err(OrchestrationError::RuntimeSourceUnavailable)
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_successful_orchestration_without_calibration_materializes_plan_decisions() {
        let model_file = create_tiny_llama_gguf();
        let user = user_profile();
        let orchestrated = RuntimeOrchestrator
            .orchestrate(OrchestrationRequest::new(model_file.path(), &user))
            .unwrap();
        let plan = &orchestrated.execution_plan;
        let config = orchestrated.runtime_config();

        assert!(plan.calibration.calibration_identifier.is_none());
        assert_eq!(config.cpu_thread_count, plan.cpu_thread_count);
        assert_eq!(config.ram_budget_bytes, plan.ram_budget_bytes);
        assert_eq!(config.layer_cache_enabled, plan.layer_cache.enabled);
        assert_eq!(
            config.layer_cache_capacity_bytes,
            plan.layer_cache.capacity_bytes
        );
        assert_eq!(
            config.read_coalescing_enabled,
            plan.io.read_coalescing_enabled
        );
        assert_eq!(
            config.grouped_read_buffer_reuse_enabled,
            plan.io.grouped_read_buffer_reuse_enabled
        );
        assert_eq!(
            orchestrated.engine.backend.num_threads(),
            plan.cpu_thread_count
        );
        assert_eq!(
            orchestrated.engine.budget.total_bytes(),
            plan.ram_budget_bytes
        );
        assert_eq!(
            orchestrated.engine.model.layer_cache_capacity_bytes(),
            plan.layer_cache.capacity_bytes
        );
        assert_eq!(
            orchestrated.engine.model.read_coalescing_enabled(),
            plan.io.read_coalescing_enabled
        );
        assert_eq!(
            orchestrated
                .engine
                .model
                .grouped_read_buffer_reuse_enabled(),
            plan.io.grouped_read_buffer_reuse_enabled
        );
        assert_eq!(
            orchestrated.engine.data_source.model().path.as_path(),
            plan.binding.model_path.as_path()
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_successful_orchestration_uses_matching_calibration() {
        let model_file = create_tiny_llama_gguf();
        let mut user = user_profile();
        user.calibration_level = CalibrationLevel::Quick;
        let calibration = matching_calibration(model_file.path());
        calibration.validate().unwrap();

        let orchestrated = RuntimeOrchestrator
            .orchestrate(
                OrchestrationRequest::new(model_file.path(), &user).with_calibration(&calibration),
            )
            .unwrap();
        assert_eq!(
            orchestrated
                .execution_plan
                .calibration
                .calibration_identifier
                .as_deref(),
            Some("orchestration-calibration")
        );
        assert_eq!(
            orchestrated
                .execution_plan
                .calibration
                .observation_identifiers,
            vec!["orchestration-sequential-read".to_string()]
        );
        assert!(orchestrated
            .execution_plan
            .cost
            .calibrated_read_time_ns_per_forward
            .is_some());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_calibration_binding_mismatch_is_not_silently_reused() {
        let model_file = create_tiny_llama_gguf();
        let mut user = user_profile();
        user.calibration_level = CalibrationLevel::Quick;
        let mut calibration = matching_calibration(model_file.path());
        let fingerprint = calibration.model_fingerprint.unwrap();
        calibration.model_fingerprint = Some(fingerprint.wrapping_add(1).max(1));
        calibration.validate().unwrap();

        let orchestrated = RuntimeOrchestrator
            .orchestrate(
                OrchestrationRequest::new(model_file.path(), &user).with_calibration(&calibration),
            )
            .unwrap();
        assert!(orchestrated
            .execution_plan
            .calibration
            .calibration_identifier
            .is_none());
        assert!(orchestrated
            .execution_plan
            .calibration
            .observation_identifiers
            .is_empty());
        assert!(orchestrated
            .execution_plan
            .reasons
            .iter()
            .any(|reason| { reason.code == PlanReasonCode::CalibrationNotApplicable }));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_non_measured_calibration_observations_do_not_block_or_influence_runtime() {
        let model_file = create_tiny_llama_gguf();
        let mut user = user_profile();
        user.calibration_level = CalibrationLevel::Quick;

        for (status, reason) in [
            (
                ObservationStatus::Unavailable,
                ObservationStatusReason::CapabilityUnavailable,
            ),
            (
                ObservationStatus::Skipped,
                ObservationStatusReason::TestNotImplemented,
            ),
            (
                ObservationStatus::Failed,
                ObservationStatusReason::IoFailure,
            ),
        ] {
            let mut calibration = matching_calibration(model_file.path());
            let observation = &mut calibration.observations[0];
            observation.value = None;
            observation.status = status;
            observation.status_reason = Some(reason);
            observation.measurement_duration_ns = 0;
            observation.sample_count = 0;
            observation.raw_samples.clear();
            observation.quality_basis_points = 0;
            calibration.validate().unwrap();

            let orchestrated = RuntimeOrchestrator
                .orchestrate(
                    OrchestrationRequest::new(model_file.path(), &user)
                        .with_calibration(&calibration),
                )
                .unwrap();
            assert!(orchestrated
                .execution_plan
                .calibration
                .observation_identifiers
                .is_empty());
            assert!(orchestrated
                .execution_plan
                .cost
                .calibrated_read_time_ns_per_forward
                .is_none());
        }
    }

    #[test]
    fn test_storage_discovery_failure_is_preserved() {
        let directory = tempdir().unwrap();
        let missing = directory.path().join("missing.gguf");
        let user = user_profile();
        let error = RuntimeOrchestrator
            .orchestrate(OrchestrationRequest::new(&missing, &user))
            .unwrap_err();
        assert!(matches!(
            error,
            OrchestrationError::StorageDiscovery(StorageDiscoveryError::InvalidPath)
        ));
    }

    #[test]
    fn test_model_inspection_failure_is_preserved() {
        let mut invalid = NamedTempFile::new().unwrap();
        invalid.write_all(b"not a GGUF file").unwrap();
        invalid.flush().unwrap();
        let user = user_profile();
        let error = RuntimeOrchestrator
            .orchestrate(OrchestrationRequest::new(invalid.path(), &user))
            .unwrap_err();
        assert!(matches!(error, OrchestrationError::ModelInspection(_)));
    }

    #[test]
    fn test_static_planning_failure_is_preserved() {
        let model_file = create_tiny_llama_gguf();
        let user = UserProfile::new(0, crate::planner::OperatingMode::BalancedNormal);
        let error = RuntimeOrchestrator
            .orchestrate(OrchestrationRequest::new(model_file.path(), &user))
            .unwrap_err();
        assert!(matches!(error, OrchestrationError::StaticPlanning(_)));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_planner_rejection_is_preserved() {
        let model_file = create_tiny_llama_gguf();
        let mut user = user_profile();
        user.advanced = AdvancedOverrides {
            thread_count: Some(1),
            ..AdvancedOverrides::default()
        };
        let error = RuntimeOrchestrator
            .orchestrate(OrchestrationRequest::new(model_file.path(), &user))
            .unwrap_err();
        assert!(matches!(
            error,
            OrchestrationError::Planning(PlannerError::Infeasible(
                FeasibilityRejection::AdvancedOverrideRequiresAdvancedMode
            ))
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_required_unsupported_gpu_is_not_downgraded() {
        let model_file = create_tiny_llama_gguf();
        let mut user = user_profile();
        user.gpu_preference = GpuPreference::Require;
        let error = RuntimeOrchestrator
            .orchestrate(OrchestrationRequest::new(model_file.path(), &user))
            .unwrap_err();
        assert!(matches!(
            error,
            OrchestrationError::Planning(PlannerError::Infeasible(
                FeasibilityRejection::GpuRequiredButUnavailable
            ))
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_compiler_rejection_is_preserved() {
        let model_file = create_tiny_llama_gguf();
        let user = user_profile();
        let mut prepared = prepare_plan(model_file.path(), &user);
        assert!(prepared.execution_plan.layer_cache.enabled);
        assert!(prepared.execution_plan.layer_cache.capacity_bytes > 0);
        prepared.execution_plan.layer_cache.enabled = false;
        let error = compile_runtime_config(
            &prepared.execution_plan,
            &prepared.machine,
            &prepared.model,
            &prepared.storage,
            &prepared.capabilities,
            &prepared.static_plan,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            OrchestrationError::PlanCompilation(PlanCompilationError::LayerCacheStateInconsistent)
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_compiler_unsupported_gpu_rejection_is_preserved() {
        let model_file = create_tiny_llama_gguf();
        let user = user_profile();
        let mut prepared = prepare_plan(model_file.path(), &user);
        prepared.execution_plan.strategy = StrategyId::GpuLayerOffload;
        prepared.execution_plan.gpu.preference = GpuPreference::Prefer;
        prepared.execution_plan.gpu.enabled = true;
        prepared.execution_plan.gpu.selected_device_id = Some("synthetic-gpu".to_string());
        let error = compile_runtime_config(
            &prepared.execution_plan,
            &prepared.machine,
            &prepared.model,
            &prepared.storage,
            &prepared.capabilities,
            &prepared.static_plan,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            OrchestrationError::PlanCompilation(PlanCompilationError::UnsupportedGpuExecution)
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_runtime_construction_failure_is_preserved() {
        let model_file = create_tiny_llama_gguf();
        let user = user_profile();
        let prepared = prepare_plan(model_file.path(), &user);
        let runtime_config = compile_runtime_config(
            &prepared.execution_plan,
            &prepared.machine,
            &prepared.model,
            &prepared.storage,
            &prepared.capabilities,
            &prepared.static_plan,
        )
        .unwrap();
        model_file.as_file().set_len(0).unwrap();

        let error = construct_runtime(prepared.data_source, runtime_config).unwrap_err();
        assert!(matches!(error, OrchestrationError::RuntimeConstruction(_)));
    }

    #[test]
    fn test_legacy_inference_construction_remains_available() {
        let model_file = create_tiny_llama_gguf();
        let engine =
            InferenceEngine::new(model_file.path().to_str().unwrap(), TEST_RAM_BYTES).unwrap();
        assert_eq!(engine.budget.total_bytes(), TEST_RAM_BYTES);
    }
}
