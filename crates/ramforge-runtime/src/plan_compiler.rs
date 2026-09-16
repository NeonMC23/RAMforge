//! Validation and materialization boundary between Planner output and runtime.
//!
//! The compiler consumes Planner contracts, verifies that their concrete
//! execution decisions still fit the supplied machine/model/storage context,
//! and emits a Planner-free `RuntimeConfig`. It does not choose strategies or
//! alter requested values.

use std::fmt;

use crate::plan::PlanResult;
use crate::planner::{
    CapabilitySet, ExecutionPlan, GpuPreference, MachineProfile, ModelProfile, StorageProfile,
    StrategyId, EXECUTION_PLAN_SCHEMA_VERSION, PLANNER_RULESET_VERSION,
};
use crate::runtime_config::{RuntimeConfig, RuntimeConfigError};

#[derive(Debug, Clone, Copy)]
pub struct PlanCompilationContext<'a> {
    pub machine: &'a MachineProfile,
    pub model: &'a ModelProfile,
    pub storage: &'a StorageProfile,
    pub capabilities: &'a CapabilitySet,
    pub static_plan: &'a PlanResult,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanCompilationError {
    UnsupportedSchemaVersion {
        found: u32,
        supported: u32,
    },
    UnsupportedRulesetVersion {
        found: u32,
        supported: u32,
    },
    InvalidContext {
        component: &'static str,
    },
    BindingMismatch {
        binding: &'static str,
    },
    StaticPlanMismatch {
        field: &'static str,
    },
    ZeroThreadCount,
    ThreadCountExceedsLimit {
        requested: usize,
        limit: usize,
    },
    ZeroRamBudget,
    RamBudgetExceedsTotalRam {
        requested: u64,
        limit: u64,
    },
    RamBudgetExceedsAvailableRam {
        requested: u64,
        limit: u64,
    },
    StorageUnavailable,
    CpuExecutionUnavailable,
    ModelExecutionUnavailable,
    ExecutionPreflightUnavailable,
    ManagedMemoryLowerBoundExceedsBudget {
        lower_bound_bytes: u64,
        ram_budget_bytes: u64,
    },
    ReservationMismatch {
        field: &'static str,
    },
    LayerCacheStateInconsistent,
    LayerCacheUnavailable,
    LayerCacheCapacityExceedsLimit {
        requested: u64,
        limit: u64,
    },
    LayerCacheCapacityCannotFitLayer {
        requested: u64,
        minimum: u64,
    },
    ReadCoalescingUnavailable,
    GroupedReadBufferReuseUnavailable,
    InconsistentGpuDecision,
    UnsupportedGpuExecution,
    InvalidRuntimeConfig(RuntimeConfigError),
}

impl fmt::Display for PlanCompilationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found, supported } => write!(
                formatter,
                "execution plan schema version {found} is unsupported; expected {supported}"
            ),
            Self::UnsupportedRulesetVersion { found, supported } => write!(
                formatter,
                "planner ruleset version {found} is unsupported; expected {supported}"
            ),
            Self::InvalidContext { component } => {
                write!(formatter, "invalid plan compilation context: {component}")
            }
            Self::BindingMismatch { binding } => {
                write!(formatter, "execution plan binding mismatch: {binding}")
            }
            Self::StaticPlanMismatch { field } => {
                write!(formatter, "static execution plan mismatch: {field}")
            }
            Self::ZeroThreadCount => write!(formatter, "execution plan has zero CPU threads"),
            Self::ThreadCountExceedsLimit { requested, limit } => write!(
                formatter,
                "execution plan requests {requested} CPU threads but the limit is {limit}"
            ),
            Self::ZeroRamBudget => write!(formatter, "execution plan has a zero RAM budget"),
            Self::RamBudgetExceedsTotalRam { requested, limit } => write!(
                formatter,
                "execution plan RAM budget {requested} exceeds total RAM {limit}"
            ),
            Self::RamBudgetExceedsAvailableRam { requested, limit } => write!(
                formatter,
                "execution plan RAM budget {requested} exceeds available RAM {limit}"
            ),
            Self::StorageUnavailable => {
                write!(formatter, "execution plan requires unavailable storage execution")
            }
            Self::CpuExecutionUnavailable => {
                write!(formatter, "execution plan requires unavailable CPU execution")
            }
            Self::ModelExecutionUnavailable => {
                write!(formatter, "execution plan model is not executable by this runtime")
            }
            Self::ExecutionPreflightUnavailable => {
                write!(formatter, "runtime execution preflight is unavailable")
            }
            Self::ManagedMemoryLowerBoundExceedsBudget {
                lower_bound_bytes,
                ram_budget_bytes,
            } => write!(
                formatter,
                "managed-memory lower bound {lower_bound_bytes} exceeds RAM budget {ram_budget_bytes}"
            ),
            Self::ReservationMismatch { field } => {
                write!(formatter, "execution plan reservation mismatch: {field}")
            }
            Self::LayerCacheStateInconsistent => {
                write!(formatter, "execution plan layer-cache state is inconsistent")
            }
            Self::LayerCacheUnavailable => {
                write!(formatter, "execution plan requests an unavailable layer cache")
            }
            Self::LayerCacheCapacityExceedsLimit { requested, limit } => write!(
                formatter,
                "execution plan layer-cache capacity {requested} exceeds limit {limit}"
            ),
            Self::LayerCacheCapacityCannotFitLayer { requested, minimum } => write!(
                formatter,
                "execution plan layer-cache capacity {requested} cannot fit the minimum layer size {minimum}"
            ),
            Self::ReadCoalescingUnavailable => {
                write!(formatter, "execution plan requests unavailable read coalescing")
            }
            Self::GroupedReadBufferReuseUnavailable => write!(
                formatter,
                "execution plan requests unavailable grouped read buffer reuse"
            ),
            Self::InconsistentGpuDecision => {
                write!(formatter, "execution plan GPU decision is inconsistent")
            }
            Self::UnsupportedGpuExecution => {
                write!(formatter, "GPU execution is not implemented by this runtime")
            }
            Self::InvalidRuntimeConfig(error) => {
                write!(formatter, "compiled runtime configuration is invalid: {error}")
            }
        }
    }
}

impl std::error::Error for PlanCompilationError {}

#[derive(Debug, Default, Clone, Copy)]
pub struct PlanCompiler;

impl PlanCompiler {
    /// Validate only execution safety, bindings, capabilities, and exact
    /// materialization constraints. Planner mode, reasons, calibration
    /// provenance, and cost annotations are deliberately not reinterpreted.
    pub fn compile(
        &self,
        plan: &ExecutionPlan,
        context: PlanCompilationContext<'_>,
    ) -> Result<RuntimeConfig, PlanCompilationError> {
        validate_plan_header(plan)?;
        validate_concrete_decisions(plan)?;
        validate_context(context)?;
        validate_bindings(plan, context)?;
        let execution = validate_static_plan(plan, context)?;
        validate_resource_limits(plan, context)?;
        validate_reservations(plan, execution)?;
        validate_cache(plan, context.capabilities, execution)?;
        validate_io(plan, context.capabilities)?;

        RuntimeConfig::new(
            plan.cpu_thread_count,
            plan.ram_budget_bytes,
            plan.layer_cache.enabled,
            plan.layer_cache.capacity_bytes,
            plan.io.read_coalescing_enabled,
            plan.io.grouped_read_buffer_reuse_enabled,
        )
        .map_err(PlanCompilationError::InvalidRuntimeConfig)
    }
}

fn validate_plan_header(plan: &ExecutionPlan) -> Result<(), PlanCompilationError> {
    if plan.schema_version != EXECUTION_PLAN_SCHEMA_VERSION {
        return Err(PlanCompilationError::UnsupportedSchemaVersion {
            found: plan.schema_version,
            supported: EXECUTION_PLAN_SCHEMA_VERSION,
        });
    }
    if plan.ruleset_version != PLANNER_RULESET_VERSION {
        return Err(PlanCompilationError::UnsupportedRulesetVersion {
            found: plan.ruleset_version,
            supported: PLANNER_RULESET_VERSION,
        });
    }
    Ok(())
}

fn validate_concrete_decisions(plan: &ExecutionPlan) -> Result<(), PlanCompilationError> {
    if plan.cpu_thread_count == 0 {
        return Err(PlanCompilationError::ZeroThreadCount);
    }
    if plan.ram_budget_bytes == 0 {
        return Err(PlanCompilationError::ZeroRamBudget);
    }
    if plan.layer_cache.enabled != (plan.layer_cache.capacity_bytes > 0) {
        return Err(PlanCompilationError::LayerCacheStateInconsistent);
    }
    match plan.strategy {
        StrategyId::CpuLayerStreaming => {
            if plan.gpu.enabled
                || plan.gpu.selected_device_id.is_some()
                || plan.gpu.preference == GpuPreference::Require
            {
                return Err(PlanCompilationError::InconsistentGpuDecision);
            }
        }
        StrategyId::GpuLayerOffload => {
            if !plan.gpu.enabled
                || plan.gpu.selected_device_id.is_none()
                || plan.gpu.preference == GpuPreference::Disabled
            {
                return Err(PlanCompilationError::InconsistentGpuDecision);
            }
            return Err(PlanCompilationError::UnsupportedGpuExecution);
        }
    }
    Ok(())
}

fn validate_context(context: PlanCompilationContext<'_>) -> Result<(), PlanCompilationError> {
    context
        .machine
        .validate()
        .map_err(|_| PlanCompilationError::InvalidContext {
            component: "machine profile",
        })?;
    context
        .model
        .validate()
        .map_err(|_| PlanCompilationError::InvalidContext {
            component: "model profile",
        })?;
    context
        .storage
        .validate()
        .map_err(|_| PlanCompilationError::InvalidContext {
            component: "storage profile",
        })?;

    let derived = CapabilitySet::derive(
        context.machine,
        context.model,
        context.storage,
        context.static_plan,
    );
    if context.capabilities != &derived {
        return Err(PlanCompilationError::InvalidContext {
            component: "capability set does not match current profiles and static plan",
        });
    }
    Ok(())
}

fn validate_bindings(
    plan: &ExecutionPlan,
    context: PlanCompilationContext<'_>,
) -> Result<(), PlanCompilationError> {
    if plan.binding.machine_fingerprint != context.machine.fingerprint() {
        return Err(PlanCompilationError::BindingMismatch {
            binding: "machine fingerprint",
        });
    }
    if plan.binding.model_descriptor_fingerprint
        != context.model.identity.descriptor_fingerprint
    {
        return Err(PlanCompilationError::BindingMismatch {
            binding: "model descriptor fingerprint",
        });
    }
    if plan.binding.storage_fingerprint != context.storage.fingerprint() {
        return Err(PlanCompilationError::BindingMismatch {
            binding: "storage fingerprint",
        });
    }
    if plan.binding.model_path.as_path() != context.storage.model_path.as_path() {
        return Err(PlanCompilationError::BindingMismatch {
            binding: "model path",
        });
    }
    Ok(())
}

fn validate_static_plan<'a>(
    plan: &ExecutionPlan,
    context: PlanCompilationContext<'a>,
) -> Result<&'a crate::plan::ExecutionMemoryPlan, PlanCompilationError> {
    if context.static_plan.ram_requested != plan.ram_budget_bytes {
        return Err(PlanCompilationError::StaticPlanMismatch {
            field: "RAM budget",
        });
    }
    if context.static_plan.file_size != context.model.file_size_bytes {
        return Err(PlanCompilationError::StaticPlanMismatch {
            field: "model file size",
        });
    }
    if context.static_plan.tensor_count != context.model.tensor_count {
        return Err(PlanCompilationError::StaticPlanMismatch {
            field: "tensor count",
        });
    }
    if context.static_plan.architecture != context.model.architecture {
        return Err(PlanCompilationError::StaticPlanMismatch {
            field: "architecture",
        });
    }
    if context.storage.file_size_bytes != Some(context.model.file_size_bytes) {
        return Err(PlanCompilationError::StaticPlanMismatch {
            field: "storage/model file size",
        });
    }
    if context.static_plan.execution_memory.is_some()
        && context.static_plan.execution_preflight_error.is_some()
    {
        return Err(PlanCompilationError::StaticPlanMismatch {
            field: "execution preflight state",
        });
    }
    context
        .static_plan
        .execution_memory
        .as_ref()
        .ok_or(PlanCompilationError::ExecutionPreflightUnavailable)
}

fn validate_resource_limits(
    plan: &ExecutionPlan,
    context: PlanCompilationContext<'_>,
) -> Result<(), PlanCompilationError> {
    if plan.cpu_thread_count > context.machine.logical_cpu_cores {
        return Err(PlanCompilationError::ThreadCountExceedsLimit {
            requested: plan.cpu_thread_count,
            limit: context.machine.logical_cpu_cores,
        });
    }
    let total_ram = context.machine.total_ram_bytes.ok_or(
        PlanCompilationError::InvalidContext {
            component: "machine total RAM is unavailable",
        },
    )?;
    let available_ram = context.machine.available_ram_bytes.ok_or(
        PlanCompilationError::InvalidContext {
            component: "machine available RAM is unavailable",
        },
    )?;
    if plan.ram_budget_bytes > total_ram {
        return Err(PlanCompilationError::RamBudgetExceedsTotalRam {
            requested: plan.ram_budget_bytes,
            limit: total_ram,
        });
    }
    if plan.reservations.available_ram_bytes_at_planning > total_ram {
        return Err(PlanCompilationError::ReservationMismatch {
            field: "available RAM at planning exceeds total RAM",
        });
    }
    if plan.ram_budget_bytes > available_ram {
        return Err(PlanCompilationError::RamBudgetExceedsAvailableRam {
            requested: plan.ram_budget_bytes,
            limit: available_ram,
        });
    }
    if !context.capabilities.storage_usable {
        return Err(PlanCompilationError::StorageUnavailable);
    }
    if !context.capabilities.cpu_backend_usable {
        return Err(PlanCompilationError::CpuExecutionUnavailable);
    }
    if !context.capabilities.model_execution_compatible {
        return Err(PlanCompilationError::ModelExecutionUnavailable);
    }
    Ok(())
}

fn validate_reservations(
    plan: &ExecutionPlan,
    execution: &crate::plan::ExecutionMemoryPlan,
) -> Result<(), PlanCompilationError> {
    if !execution.layer_streaming_lower_bound_fits
        || execution.managed_lower_bound_bytes > plan.ram_budget_bytes
    {
        return Err(PlanCompilationError::ManagedMemoryLowerBoundExceedsBudget {
            lower_bound_bytes: execution.managed_lower_bound_bytes,
            ram_budget_bytes: plan.ram_budget_bytes,
        });
    }
    if plan.reservations.available_ram_bytes_at_planning < plan.ram_budget_bytes {
        return Err(PlanCompilationError::ReservationMismatch {
            field: "available RAM at planning",
        });
    }
    for (matches, field) in [
        (
            plan.reservations.persistent_resident_bytes
                == execution.persistent_resident_bytes,
            "persistent resident bytes",
        ),
        (
            plan.reservations.persistent_startup_peak_bytes
                == execution.persistent_startup_peak_bytes,
            "persistent startup peak bytes",
        ),
        (
            plan.reservations.largest_layer_load_peak_bytes
                == execution.largest_layer_load_peak_bytes,
            "largest layer load peak bytes",
        ),
        (
            plan.reservations.managed_lower_bound_bytes
                == execution.managed_lower_bound_bytes,
            "managed-memory lower bound",
        ),
    ] {
        if !matches {
            return Err(PlanCompilationError::ReservationMismatch { field });
        }
    }
    let remaining = plan
        .ram_budget_bytes
        .checked_sub(plan.reservations.managed_lower_bound_bytes)
        .ok_or(PlanCompilationError::ManagedMemoryLowerBoundExceedsBudget {
            lower_bound_bytes: plan.reservations.managed_lower_bound_bytes,
            ram_budget_bytes: plan.ram_budget_bytes,
        })?;
    if plan.reservations.remaining_budget_after_lower_bound_bytes != remaining {
        return Err(PlanCompilationError::ReservationMismatch {
            field: "remaining budget after lower bound",
        });
    }
    if execution.layer_cache_capacity_bytes != remaining {
        return Err(PlanCompilationError::StaticPlanMismatch {
            field: "layer-cache capacity/lower-bound relationship",
        });
    }
    Ok(())
}

fn validate_cache(
    plan: &ExecutionPlan,
    capabilities: &CapabilitySet,
    execution: &crate::plan::ExecutionMemoryPlan,
) -> Result<(), PlanCompilationError> {
    if !plan.layer_cache.enabled {
        return Ok(());
    }
    if !capabilities.layer_cache_possible {
        return Err(PlanCompilationError::LayerCacheUnavailable);
    }
    if plan.layer_cache.capacity_bytes > execution.layer_cache_capacity_bytes {
        return Err(PlanCompilationError::LayerCacheCapacityExceedsLimit {
            requested: plan.layer_cache.capacity_bytes,
            limit: execution.layer_cache_capacity_bytes,
        });
    }
    if plan.layer_cache.capacity_bytes < execution.min_layer_resident_bytes {
        return Err(PlanCompilationError::LayerCacheCapacityCannotFitLayer {
            requested: plan.layer_cache.capacity_bytes,
            minimum: execution.min_layer_resident_bytes,
        });
    }
    Ok(())
}

fn validate_io(
    plan: &ExecutionPlan,
    capabilities: &CapabilitySet,
) -> Result<(), PlanCompilationError> {
    if plan.io.read_coalescing_enabled && !capabilities.read_coalescing_possible {
        return Err(PlanCompilationError::ReadCoalescingUnavailable);
    }
    if plan.io.grouped_read_buffer_reuse_enabled
        && !capabilities.grouped_read_buffer_reuse_possible
    {
        return Err(PlanCompilationError::GroupedReadBufferReuseUnavailable);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    use ramforge_core::memory::MemoryBudget;

    use super::*;
    use crate::plan::ExecutionMemoryPlan;
    use crate::planner::{
        Availability, CalibrationProvenance, CostEstimate, DiscoveryState, GpuDecision,
        IoDecision, LayerCacheDecision, ModelExecutionCompatibility, ModelIdentity, OperatingMode,
        PlanBinding, ResourceReservations, StorageKind, StoragePathState, TensorFormatProfile,
        PROFILE_SCHEMA_VERSION,
    };

    struct CompilerFixture {
        plan: ExecutionPlan,
        machine: MachineProfile,
        model: ModelProfile,
        storage: StorageProfile,
        capabilities: CapabilitySet,
        static_plan: PlanResult,
    }

    impl CompilerFixture {
        fn new() -> Self {
            let machine = MachineProfile {
                schema_version: PROFILE_SCHEMA_VERSION,
                os: "linux".to_string(),
                architecture: "x86_64".to_string(),
                kernel_version: None,
                cpu_vendor: Some("test-vendor".to_string()),
                cpu_model: Some("test-cpu".to_string()),
                physical_cpu_cores: Some(4),
                logical_cpu_cores: 8,
                cpu_features: BTreeSet::new(),
                total_ram_bytes: Some(20_000),
                available_ram_bytes: Some(16_000),
                cpu_backend: Availability {
                    detected: true,
                    supported: true,
                    usable: true,
                },
                gpu_inventory_state: DiscoveryState::Unavailable,
                gpus: Vec::new(),
            };
            let model = ModelProfile {
                schema_version: PROFILE_SCHEMA_VERSION,
                identity: ModelIdentity {
                    display_name: Some("compiler-fixture".to_string()),
                    descriptor_fingerprint: 17,
                },
                architecture: Some("llama".to_string()),
                context_size: Some(64),
                layer_count: Some(1),
                tensor_count: 1,
                total_tensor_elements: Some(1_250),
                file_size_bytes: 5_000,
                tensor_data_bytes: Some(5_000),
                tensor_formats: vec![TensorFormatProfile {
                    format: "F32".to_string(),
                    tensor_count: 1,
                    element_count: Some(1_250),
                    byte_count: Some(5_000),
                    runtime_supported: true,
                }],
                execution_compatibility: ModelExecutionCompatibility::Executable,
                has_coalescible_layer_reads: true,
                has_reusable_grouped_read_candidate: true,
            };
            let storage = StorageProfile {
                schema_version: PROFILE_SCHEMA_VERSION,
                model_path: PathBuf::from("/tmp/compiler-fixture.gguf"),
                path_state: StoragePathState::Ready,
                readable: true,
                regular_file: true,
                file_size_bytes: Some(5_000),
                filesystem_type: Some("testfs".to_string()),
                filesystem_id: Some("testfs:1".to_string()),
                device_id: Some("test-device".to_string()),
                kind: StorageKind::Local,
                seekable: true,
            };
            let execution = ExecutionMemoryPlan {
                block_count: 1,
                resident_persistent_count: 1,
                streamed_persistent_count: 0,
                persistent_resident_bytes: 1_000,
                persistent_startup_peak_bytes: 1_200,
                largest_layer_index: 0,
                largest_layer_tensor_count: 1,
                largest_layer_resident_bytes: 2_000,
                largest_layer_load_peak_bytes: 3_000,
                managed_lower_bound_bytes: 4_000,
                layer_cache_capacity_bytes: 6_000,
                max_complete_cached_layers: 3,
                min_layer_resident_bytes: 2_000,
                logical_tensor_reads_per_forward: 1,
                estimated_physical_reads_per_forward: 1,
                logical_tensor_bytes_per_forward: 5_000,
                estimated_physical_bytes_per_forward: 5_000,
                estimated_gap_bytes_per_forward: 0,
                layer_streaming_lower_bound_fits: true,
            };
            let static_plan = PlanResult {
                file_size: 5_000,
                architecture: Some("llama".to_string()),
                tensor_count: 1,
                ram_requested: 10_000,
                cache_capacity: 5_000,
                overhead_reserved: 0,
                available: 10_000,
                fits_in_ram: true,
                file_backed_needed: 0,
                total_tensor_bytes: Some(5_000),
                execution_memory: Some(execution),
                execution_preflight_error: None,
                budget: MemoryBudget::new(10_000).unwrap(),
            };
            let capabilities = CapabilitySet::derive(&machine, &model, &storage, &static_plan);
            let plan = ExecutionPlan {
                schema_version: EXECUTION_PLAN_SCHEMA_VERSION,
                ruleset_version: PLANNER_RULESET_VERSION,
                binding: PlanBinding {
                    model_descriptor_fingerprint: model.identity.descriptor_fingerprint,
                    machine_fingerprint: machine.fingerprint(),
                    storage_fingerprint: storage.fingerprint(),
                    model_path: storage.model_path.clone(),
                },
                mode: OperatingMode::BalancedNormal,
                strategy: StrategyId::CpuLayerStreaming,
                ram_budget_bytes: 10_000,
                cpu_thread_count: 4,
                layer_cache: LayerCacheDecision {
                    enabled: true,
                    capacity_bytes: 2_000,
                },
                io: IoDecision {
                    read_coalescing_enabled: true,
                    grouped_read_buffer_reuse_enabled: true,
                },
                gpu: GpuDecision {
                    preference: GpuPreference::Disabled,
                    enabled: false,
                    selected_device_id: None,
                },
                reservations: ResourceReservations {
                    available_ram_bytes_at_planning: 16_000,
                    persistent_resident_bytes: 1_000,
                    persistent_startup_peak_bytes: 1_200,
                    largest_layer_load_peak_bytes: 3_000,
                    managed_lower_bound_bytes: 4_000,
                    remaining_budget_after_lower_bound_bytes: 6_000,
                },
                cost: CostEstimate {
                    physical_read_bytes_per_forward: 5_000,
                    calibrated_read_time_ns_per_forward: None,
                    observed_strategy_latency_ns: None,
                },
                calibration: CalibrationProvenance {
                    calibration_identifier: None,
                    calibration_plan_identifier: None,
                    calibration_plan_version: None,
                    calibration_ruleset_version: None,
                    observation_identifiers: Vec::new(),
                },
                reasons: Vec::new(),
            };
            Self {
                plan,
                machine,
                model,
                storage,
                capabilities,
                static_plan,
            }
        }

        fn context(&self) -> PlanCompilationContext<'_> {
            PlanCompilationContext {
                machine: &self.machine,
                model: &self.model,
                storage: &self.storage,
                capabilities: &self.capabilities,
                static_plan: &self.static_plan,
            }
        }

        fn refresh_capabilities(&mut self) {
            self.capabilities = CapabilitySet::derive(
                &self.machine,
                &self.model,
                &self.storage,
                &self.static_plan,
            );
        }
    }

    #[test]
    fn test_valid_plan_compiles_without_changing_values() {
        let fixture = CompilerFixture::new();
        let config = PlanCompiler
            .compile(&fixture.plan, fixture.context())
            .unwrap();
        assert_eq!(config.cpu_thread_count, fixture.plan.cpu_thread_count);
        assert_eq!(config.ram_budget_bytes, fixture.plan.ram_budget_bytes);
        assert_eq!(
            config.layer_cache_capacity_bytes,
            fixture.plan.layer_cache.capacity_bytes
        );
    }

    #[test]
    fn test_zero_thread_count_is_rejected() {
        let mut fixture = CompilerFixture::new();
        fixture.plan.cpu_thread_count = 0;
        assert_eq!(
            PlanCompiler
                .compile(&fixture.plan, fixture.context())
                .unwrap_err(),
            PlanCompilationError::ZeroThreadCount
        );
    }

    #[test]
    fn test_excessive_thread_count_is_rejected() {
        let mut fixture = CompilerFixture::new();
        fixture.plan.cpu_thread_count = 9;
        assert_eq!(
            PlanCompiler
                .compile(&fixture.plan, fixture.context())
                .unwrap_err(),
            PlanCompilationError::ThreadCountExceedsLimit {
                requested: 9,
                limit: 8,
            }
        );
    }

    #[test]
    fn test_zero_ram_budget_is_rejected() {
        let mut fixture = CompilerFixture::new();
        fixture.plan.ram_budget_bytes = 0;
        assert_eq!(
            PlanCompiler
                .compile(&fixture.plan, fixture.context())
                .unwrap_err(),
            PlanCompilationError::ZeroRamBudget
        );
    }

    #[test]
    fn test_ram_budget_above_available_limit_is_rejected() {
        let mut fixture = CompilerFixture::new();
        fixture.machine.available_ram_bytes = Some(9_999);
        assert_eq!(
            PlanCompiler
                .compile(&fixture.plan, fixture.context())
                .unwrap_err(),
            PlanCompilationError::RamBudgetExceedsAvailableRam {
                requested: 10_000,
                limit: 9_999,
            }
        );
    }

    #[test]
    fn test_cache_settings_are_preserved_exactly() {
        let mut fixture = CompilerFixture::new();
        fixture.plan.layer_cache.capacity_bytes = 3_000;
        let config = PlanCompiler
            .compile(&fixture.plan, fixture.context())
            .unwrap();
        assert!(config.layer_cache_enabled);
        assert_eq!(config.layer_cache_capacity_bytes, 3_000);
    }

    #[test]
    fn test_cache_capacity_above_static_limit_is_rejected() {
        let mut fixture = CompilerFixture::new();
        fixture.plan.layer_cache.capacity_bytes = 6_001;
        assert_eq!(
            PlanCompiler
                .compile(&fixture.plan, fixture.context())
                .unwrap_err(),
            PlanCompilationError::LayerCacheCapacityExceedsLimit {
                requested: 6_001,
                limit: 6_000,
            }
        );
    }

    #[test]
    fn test_read_coalescing_setting_is_preserved_exactly() {
        let mut fixture = CompilerFixture::new();
        fixture.plan.io.read_coalescing_enabled = false;
        let config = PlanCompiler
            .compile(&fixture.plan, fixture.context())
            .unwrap();
        assert!(!config.read_coalescing_enabled);
        assert!(config.grouped_read_buffer_reuse_enabled);
    }

    #[test]
    fn test_grouped_buffer_setting_is_preserved_exactly() {
        let fixture = CompilerFixture::new();
        let config = PlanCompiler
            .compile(&fixture.plan, fixture.context())
            .unwrap();
        assert!(config.read_coalescing_enabled);
        assert!(config.grouped_read_buffer_reuse_enabled);
    }

    #[test]
    fn test_unsupported_gpu_execution_is_rejected() {
        let mut fixture = CompilerFixture::new();
        fixture.plan.strategy = StrategyId::GpuLayerOffload;
        fixture.plan.gpu.preference = GpuPreference::Prefer;
        fixture.plan.gpu.enabled = true;
        fixture.plan.gpu.selected_device_id = Some("gpu0".to_string());
        assert_eq!(
            PlanCompiler
                .compile(&fixture.plan, fixture.context())
                .unwrap_err(),
            PlanCompilationError::UnsupportedGpuExecution
        );
    }

    #[test]
    fn test_internally_inconsistent_plan_is_rejected() {
        let mut fixture = CompilerFixture::new();
        fixture
            .plan
            .reservations
            .remaining_budget_after_lower_bound_bytes = 5_999;
        assert_eq!(
            PlanCompiler
                .compile(&fixture.plan, fixture.context())
                .unwrap_err(),
            PlanCompilationError::ReservationMismatch {
                field: "remaining budget after lower bound",
            }
        );
    }

    #[test]
    fn test_inconsistent_cache_state_is_rejected() {
        let mut fixture = CompilerFixture::new();
        fixture.plan.layer_cache.enabled = false;
        assert_eq!(
            PlanCompiler
                .compile(&fixture.plan, fixture.context())
                .unwrap_err(),
            PlanCompilationError::LayerCacheStateInconsistent
        );
    }

    #[test]
    fn test_read_coalescing_requires_current_capability() {
        let mut fixture = CompilerFixture::new();
        fixture.model.has_coalescible_layer_reads = false;
        fixture.refresh_capabilities();
        assert_eq!(
            PlanCompiler
                .compile(&fixture.plan, fixture.context())
                .unwrap_err(),
            PlanCompilationError::ReadCoalescingUnavailable
        );
    }

    #[test]
    fn test_grouped_buffer_reuse_requires_current_capability() {
        let mut fixture = CompilerFixture::new();
        fixture.model.has_reusable_grouped_read_candidate = false;
        fixture.refresh_capabilities();
        assert_eq!(
            PlanCompiler
                .compile(&fixture.plan, fixture.context())
                .unwrap_err(),
            PlanCompilationError::GroupedReadBufferReuseUnavailable
        );
    }
}
