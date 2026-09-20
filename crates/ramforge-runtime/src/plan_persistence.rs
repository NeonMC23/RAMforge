//! Deterministic, bounded persistence for planner-produced execution plans.
//!
//! The persisted form contains identities, decisions, reservations, costs, and
//! calibration provenance only. Model paths, weights, tensor payloads, runtime
//! allocations, caches, and inference state are deliberately excluded.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::plan_compiler::{
    validate_concrete_decisions, PlanCompilationContext, PlanCompilationError, PlanCompiler,
};
use crate::planner::{
    CalibrationProvenance, CalibrationResult, CostEstimate, ExecutionPlan, GpuDecision,
    GpuPreference, IoDecision, LayerCacheDecision, OperatingMode, PlanBinding, PlanCompatibility,
    PlanReason, PlanReasonCode, ResourceReservations, StrategyId,
};

pub const PERSISTED_EXECUTION_PLAN_SCHEMA_VERSION: u32 = 1;

const PERSISTED_PLAN_MAGIC: &[u8; 8] = b"RAMFPLN\0";
const MAX_PERSISTED_PLAN_BYTES: usize = 1024 * 1024;
const MAX_STRING_BYTES: usize = 64 * 1024;
const MAX_LIST_ITEMS: usize = 1024;

#[derive(Debug)]
pub enum PlanPersistenceError {
    Io(std::io::Error),
    TooLarge {
        bytes: usize,
        maximum: usize,
    },
    InvalidMagic,
    UnsupportedPersistenceSchema {
        found: u32,
        supported: u32,
    },
    Truncated {
        field: &'static str,
    },
    TrailingBytes {
        bytes: usize,
    },
    InvalidBoolean {
        field: &'static str,
        value: u8,
    },
    InvalidEnum {
        field: &'static str,
        value: u8,
    },
    InvalidUtf8 {
        field: &'static str,
    },
    LimitExceeded {
        field: &'static str,
        value: usize,
        maximum: usize,
    },
    NumericOverflow {
        field: &'static str,
    },
    InvalidPlan {
        field: &'static str,
    },
}

impl fmt::Display for PlanPersistenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "persisted plan I/O failed: {error}"),
            Self::TooLarge { bytes, maximum } => write!(
                formatter,
                "persisted plan is {bytes} bytes; maximum is {maximum} bytes"
            ),
            Self::InvalidMagic => write!(formatter, "invalid persisted execution plan magic"),
            Self::UnsupportedPersistenceSchema { found, supported } => write!(
                formatter,
                "persisted plan schema {found} is unsupported; expected {supported}"
            ),
            Self::Truncated { field } => {
                write!(formatter, "persisted plan is truncated at {field}")
            }
            Self::TrailingBytes { bytes } => {
                write!(formatter, "persisted plan has {bytes} trailing bytes")
            }
            Self::InvalidBoolean { field, value } => {
                write!(
                    formatter,
                    "persisted plan has invalid boolean {value} for {field}"
                )
            }
            Self::InvalidEnum { field, value } => {
                write!(
                    formatter,
                    "persisted plan has invalid enum tag {value} for {field}"
                )
            }
            Self::InvalidUtf8 { field } => {
                write!(formatter, "persisted plan has invalid UTF-8 for {field}")
            }
            Self::LimitExceeded {
                field,
                value,
                maximum,
            } => write!(
                formatter,
                "persisted plan {field} value {value} exceeds maximum {maximum}"
            ),
            Self::NumericOverflow { field } => {
                write!(formatter, "persisted plan numeric value overflows {field}")
            }
            Self::InvalidPlan { field } => {
                write!(formatter, "persisted execution plan is invalid: {field}")
            }
        }
    }
}

impl std::error::Error for PlanPersistenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for PlanPersistenceError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Debug)]
pub enum PersistedPlanValidationError {
    Persistence(PlanPersistenceError),
    Compilation(PlanCompilationError),
}

impl fmt::Display for PersistedPlanValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Persistence(error) => {
                write!(formatter, "persisted plan validation failed: {error}")
            }
            Self::Compilation(error) => {
                write!(formatter, "persisted plan compilation failed: {error}")
            }
        }
    }
}

impl std::error::Error for PersistedPlanValidationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Persistence(error) => Some(error),
            Self::Compilation(error) => Some(error),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedExecutionPlan {
    pub persistence_schema_version: u32,
    pub execution_schema_version: u32,
    pub planner_ruleset_version: u32,
    pub model_descriptor_fingerprint: u64,
    pub source_machine_fingerprint: u64,
    pub source_storage_fingerprint: u64,
    pub mode: OperatingMode,
    pub strategy: StrategyId,
    pub ram_budget_bytes: u64,
    pub cpu_thread_count: usize,
    pub layer_cache: LayerCacheDecision,
    pub io: IoDecision,
    pub gpu: GpuDecision,
    pub reservations: ResourceReservations,
    pub cost: CostEstimate,
    pub calibration: CalibrationProvenance,
    pub reasons: Vec<PlanReason>,
}

#[derive(Debug, Clone, Copy)]
pub struct PersistedPlanCompatibilityContext<'a> {
    pub compilation: PlanCompilationContext<'a>,
    pub calibration: Option<&'a CalibrationResult>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanCompatibilityReason {
    PersistedPlanInvalid,
    ModelFingerprintMismatch,
    PlanCompilationFailed(PlanCompilationError),
    CalibrationNotApplied,
    CalibrationMissing,
    CalibrationInvalid,
    CalibrationSourceMachineChanged,
    CalibrationSourceStorageChanged,
    CalibrationProvenanceMismatch,
    CalibrationBindingMismatch,
    CalibrationObservationUnavailable,
}

impl fmt::Display for PlanCompatibilityReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PersistedPlanInvalid => write!(formatter, "persisted plan structure is invalid"),
            Self::ModelFingerprintMismatch => write!(formatter, "model fingerprint does not match"),
            Self::PlanCompilationFailed(error) => {
                write!(formatter, "PlanCompiler rejected the plan: {error}")
            }
            Self::CalibrationNotApplied => write!(
                formatter,
                "calibration was requested but no observation was applied"
            ),
            Self::CalibrationMissing => {
                write!(formatter, "matching calibration result is not available")
            }
            Self::CalibrationInvalid => write!(formatter, "current calibration result is invalid"),
            Self::CalibrationSourceMachineChanged => write!(
                formatter,
                "calibrated source machine differs from the current machine"
            ),
            Self::CalibrationSourceStorageChanged => write!(
                formatter,
                "calibrated source storage differs from the current storage"
            ),
            Self::CalibrationProvenanceMismatch => {
                write!(formatter, "calibration provenance no longer matches")
            }
            Self::CalibrationBindingMismatch => write!(
                formatter,
                "calibration environment binding no longer matches"
            ),
            Self::CalibrationObservationUnavailable => write!(
                formatter,
                "an applied calibration observation is missing or no longer measured"
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanCompatibilityReport {
    pub state: PlanCompatibility,
    pub reasons: Vec<PlanCompatibilityReason>,
}

impl PersistedExecutionPlan {
    pub fn from_execution_plan(plan: &ExecutionPlan) -> Result<Self, PlanPersistenceError> {
        let persisted = Self {
            persistence_schema_version: PERSISTED_EXECUTION_PLAN_SCHEMA_VERSION,
            execution_schema_version: plan.schema_version,
            planner_ruleset_version: plan.ruleset_version,
            model_descriptor_fingerprint: plan.binding.model_descriptor_fingerprint,
            source_machine_fingerprint: plan.binding.machine_fingerprint,
            source_storage_fingerprint: plan.binding.storage_fingerprint,
            mode: plan.mode,
            strategy: plan.strategy,
            ram_budget_bytes: plan.ram_budget_bytes,
            cpu_thread_count: plan.cpu_thread_count,
            layer_cache: plan.layer_cache.clone(),
            io: plan.io.clone(),
            gpu: plan.gpu.clone(),
            reservations: plan.reservations.clone(),
            cost: plan.cost.clone(),
            calibration: plan.calibration.clone(),
            reasons: plan.reasons.clone(),
        };
        persisted.validate_structure()?;
        Ok(persisted)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, PlanPersistenceError> {
        self.validate_structure()?;
        self.encode_unchecked()
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PlanPersistenceError> {
        if bytes.len() > MAX_PERSISTED_PLAN_BYTES {
            return Err(PlanPersistenceError::TooLarge {
                bytes: bytes.len(),
                maximum: MAX_PERSISTED_PLAN_BYTES,
            });
        }
        let mut reader = PlanReader::new(bytes);
        if reader.take(PERSISTED_PLAN_MAGIC.len(), "magic")? != &PERSISTED_PLAN_MAGIC[..] {
            return Err(PlanPersistenceError::InvalidMagic);
        }
        let persistence_schema_version = reader.u32("persistence schema version")?;
        if persistence_schema_version != PERSISTED_EXECUTION_PLAN_SCHEMA_VERSION {
            return Err(PlanPersistenceError::UnsupportedPersistenceSchema {
                found: persistence_schema_version,
                supported: PERSISTED_EXECUTION_PLAN_SCHEMA_VERSION,
            });
        }

        // Unknown execution/planner versions remain readable so compatibility
        // evaluation can deterministically classify them as Incompatible. Only
        // an unknown persistence schema prevents decoding the representation.
        let execution_schema_version = reader.u32("execution schema version")?;
        let planner_ruleset_version = reader.u32("planner ruleset version")?;
        let model_descriptor_fingerprint = reader.u64("model fingerprint")?;
        let source_machine_fingerprint = reader.u64("source machine fingerprint")?;
        let source_storage_fingerprint = reader.u64("source storage fingerprint")?;
        let mode = decode_mode(reader.u8("operating mode")?)?;
        let strategy = decode_strategy(reader.u8("execution strategy")?)?;
        let ram_budget_bytes = reader.u64("RAM budget")?;
        let cpu_thread_count_u64 = reader.u64("CPU thread count")?;
        let cpu_thread_count = usize::try_from(cpu_thread_count_u64).map_err(|_| {
            PlanPersistenceError::NumericOverflow {
                field: "CPU thread count",
            }
        })?;
        let layer_cache = LayerCacheDecision {
            enabled: reader.boolean("layer cache enabled")?,
            capacity_bytes: reader.u64("layer cache capacity")?,
        };
        let io = IoDecision {
            read_coalescing_enabled: reader.boolean("read coalescing enabled")?,
            grouped_read_buffer_reuse_enabled: reader
                .boolean("grouped read buffer reuse enabled")?,
        };
        let gpu = GpuDecision {
            preference: decode_gpu_preference(reader.u8("GPU preference")?)?,
            enabled: reader.boolean("GPU enabled")?,
            selected_device_id: reader.optional_string("GPU device identifier")?,
        };
        let reservations = ResourceReservations {
            available_ram_bytes_at_planning: reader.u64("available RAM at planning")?,
            persistent_resident_bytes: reader.u64("persistent resident bytes")?,
            persistent_startup_peak_bytes: reader.u64("persistent startup peak bytes")?,
            largest_layer_load_peak_bytes: reader.u64("largest layer load peak bytes")?,
            managed_lower_bound_bytes: reader.u64("managed-memory lower bound")?,
            remaining_budget_after_lower_bound_bytes: reader
                .u64("remaining budget after lower bound")?,
        };
        let cost = CostEstimate {
            physical_read_bytes_per_forward: reader.u64("physical read bytes per forward")?,
            calibrated_read_time_ns_per_forward: reader.optional_u64("calibrated read time")?,
            observed_strategy_latency_ns: reader.optional_u64("observed strategy latency")?,
        };
        let calibration = CalibrationProvenance {
            calibration_identifier: reader.optional_string("calibration identifier")?,
            calibration_plan_identifier: reader.optional_string("calibration plan identifier")?,
            calibration_plan_version: reader.optional_u32("calibration plan version")?,
            calibration_ruleset_version: reader.optional_u32("calibration ruleset version")?,
            observation_identifiers: reader.string_list("calibration observation identifiers")?,
        };
        let reason_count = reader.list_count("plan reasons")?;
        let mut reasons = Vec::with_capacity(reason_count);
        for _ in 0..reason_count {
            reasons.push(PlanReason {
                code: decode_reason(reader.u8("plan reason code")?)?,
                value: reader.optional_u64("plan reason value")?,
            });
        }
        if reader.remaining() != 0 {
            return Err(PlanPersistenceError::TrailingBytes {
                bytes: reader.remaining(),
            });
        }

        let persisted = Self {
            persistence_schema_version,
            execution_schema_version,
            planner_ruleset_version,
            model_descriptor_fingerprint,
            source_machine_fingerprint,
            source_storage_fingerprint,
            mode,
            strategy,
            ram_budget_bytes,
            cpu_thread_count,
            layer_cache,
            io,
            gpu,
            reservations,
            cost,
            calibration,
            reasons,
        };
        persisted.validate_structure()?;
        Ok(persisted)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), PlanPersistenceError> {
        let bytes = self.to_bytes()?;
        let mut file = File::create(path)?;
        file.write_all(&bytes)?;
        file.flush()?;
        Ok(())
    }

    /// Save only when the destination does not already exist.
    pub fn save_new(&self, path: impl AsRef<Path>) -> Result<(), PlanPersistenceError> {
        let bytes = self.to_bytes()?;
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(&bytes)?;
        file.flush()?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, PlanPersistenceError> {
        let file = File::open(path)?;
        let mut limited = file.take((MAX_PERSISTED_PLAN_BYTES + 1) as u64);
        let mut bytes = Vec::new();
        limited.read_to_end(&mut bytes)?;
        if bytes.len() > MAX_PERSISTED_PLAN_BYTES {
            return Err(PlanPersistenceError::TooLarge {
                bytes: bytes.len(),
                maximum: MAX_PERSISTED_PLAN_BYTES,
            });
        }
        Self::from_bytes(&bytes)
    }

    /// Re-materialize against the current environment and invoke the normal
    /// compiler boundary. This is the only API that yields an executable
    /// RuntimeConfig from persisted decisions.
    pub fn compile_runtime_config(
        &self,
        context: PlanCompilationContext<'_>,
    ) -> Result<crate::runtime_config::RuntimeConfig, PersistedPlanValidationError> {
        self.validate_structure()
            .map_err(PersistedPlanValidationError::Persistence)?;
        if self.model_descriptor_fingerprint != context.model.identity.descriptor_fingerprint {
            return Err(PersistedPlanValidationError::Persistence(
                PlanPersistenceError::InvalidPlan {
                    field: "model fingerprint mismatch",
                },
            ));
        }
        let candidate = self.materialize_for_context(context);
        PlanCompiler
            .compile(&candidate, context)
            .map_err(PersistedPlanValidationError::Compilation)
    }

    /// Materialize an in-memory ExecutionPlan for review and explicit compiler
    /// validation. This does not validate runtime compatibility and never
    /// yields RuntimeConfig; callers must still invoke PlanCompiler.
    pub fn materialize_for_validation(
        &self,
        context: PlanCompilationContext<'_>,
    ) -> Result<ExecutionPlan, PlanPersistenceError> {
        self.validate_structure()?;
        Ok(self.materialize_for_context(context))
    }

    pub fn compatibility_report(
        &self,
        context: PersistedPlanCompatibilityContext<'_>,
    ) -> PlanCompatibilityReport {
        if self.validate_structure().is_err() {
            return PlanCompatibilityReport {
                state: PlanCompatibility::Incompatible,
                reasons: vec![PlanCompatibilityReason::PersistedPlanInvalid],
            };
        }
        if self.model_descriptor_fingerprint
            != context.compilation.model.identity.descriptor_fingerprint
        {
            return PlanCompatibilityReport {
                state: PlanCompatibility::Incompatible,
                reasons: vec![PlanCompatibilityReason::ModelFingerprintMismatch],
            };
        }
        let candidate = self.materialize_for_context(context.compilation);
        if let Err(error) = PlanCompiler.compile(&candidate, context.compilation) {
            return PlanCompatibilityReport {
                state: PlanCompatibility::Incompatible,
                reasons: vec![PlanCompatibilityReason::PlanCompilationFailed(error)],
            };
        }
        let reasons = self.recalibration_reasons(context);
        PlanCompatibilityReport {
            state: if reasons.is_empty() {
                PlanCompatibility::Valid
            } else {
                PlanCompatibility::CompatibleRecalibrationRecommended
            },
            reasons,
        }
    }

    pub fn validate_compatibility(
        &self,
        context: PersistedPlanCompatibilityContext<'_>,
    ) -> PlanCompatibility {
        self.compatibility_report(context).state
    }

    fn validate_structure(&self) -> Result<(), PlanPersistenceError> {
        if self.persistence_schema_version != PERSISTED_EXECUTION_PLAN_SCHEMA_VERSION {
            return Err(PlanPersistenceError::UnsupportedPersistenceSchema {
                found: self.persistence_schema_version,
                supported: PERSISTED_EXECUTION_PLAN_SCHEMA_VERSION,
            });
        }
        if self.execution_schema_version == 0 {
            return Err(PlanPersistenceError::InvalidPlan {
                field: "missing execution schema version",
            });
        }
        if self.planner_ruleset_version == 0 {
            return Err(PlanPersistenceError::InvalidPlan {
                field: "missing planner ruleset version",
            });
        }
        if self.model_descriptor_fingerprint == 0
            || self.source_machine_fingerprint == 0
            || self.source_storage_fingerprint == 0
        {
            return Err(PlanPersistenceError::InvalidPlan {
                field: "missing or invalid fingerprint",
            });
        }
        if self.reservations.available_ram_bytes_at_planning < self.ram_budget_bytes {
            return Err(PlanPersistenceError::InvalidPlan {
                field: "available RAM at planning",
            });
        }
        let remaining = self
            .ram_budget_bytes
            .checked_sub(self.reservations.managed_lower_bound_bytes)
            .ok_or(PlanPersistenceError::InvalidPlan {
                field: "managed-memory lower bound exceeds RAM budget",
            })?;
        if self.reservations.remaining_budget_after_lower_bound_bytes != remaining {
            return Err(PlanPersistenceError::InvalidPlan {
                field: "remaining budget after lower bound",
            });
        }
        if self.layer_cache.capacity_bytes > remaining {
            return Err(PlanPersistenceError::InvalidPlan {
                field: "layer-cache capacity exceeds remaining budget",
            });
        }
        if self
            .gpu
            .selected_device_id
            .as_ref()
            .is_some_and(|identifier| identifier.is_empty())
        {
            return Err(PlanPersistenceError::InvalidPlan {
                field: "empty GPU device identifier",
            });
        }

        self.validate_calibration_provenance()?;
        self.validate_reasons()?;
        let source_plan = self.materialize(
            self.source_machine_fingerprint,
            self.source_storage_fingerprint,
            PathBuf::new(),
            None,
        );
        match validate_concrete_decisions(&source_plan) {
            Ok(()) | Err(PlanCompilationError::UnsupportedGpuExecution) => Ok(()),
            Err(_) => Err(PlanPersistenceError::InvalidPlan {
                field: "inconsistent concrete execution decisions",
            }),
        }
    }

    fn validate_calibration_provenance(&self) -> Result<(), PlanPersistenceError> {
        let header_fields = [
            self.calibration.calibration_identifier.is_some(),
            self.calibration.calibration_plan_identifier.is_some(),
            self.calibration.calibration_plan_version.is_some(),
            self.calibration.calibration_ruleset_version.is_some(),
        ];
        let header_present = header_fields.iter().any(|present| *present);
        if header_present && !header_fields.iter().all(|present| *present) {
            return Err(PlanPersistenceError::InvalidPlan {
                field: "partial calibration provenance",
            });
        }
        if self
            .calibration
            .calibration_identifier
            .as_ref()
            .is_some_and(|identifier| identifier.is_empty() || identifier.len() > MAX_STRING_BYTES)
            || self
                .calibration
                .calibration_plan_identifier
                .as_ref()
                .is_some_and(|identifier| {
                    identifier.is_empty() || identifier.len() > MAX_STRING_BYTES
                })
            || self.calibration.calibration_plan_version == Some(0)
            || self.calibration.calibration_ruleset_version == Some(0)
        {
            return Err(PlanPersistenceError::InvalidPlan {
                field: "invalid calibration provenance",
            });
        }
        if !header_present && !self.calibration.observation_identifiers.is_empty() {
            return Err(PlanPersistenceError::InvalidPlan {
                field: "calibration observations without provenance",
            });
        }
        if self.calibration.observation_identifiers.len() > MAX_LIST_ITEMS {
            return Err(PlanPersistenceError::LimitExceeded {
                field: "calibration observation count",
                value: self.calibration.observation_identifiers.len(),
                maximum: MAX_LIST_ITEMS,
            });
        }
        let mut seen = Vec::new();
        for identifier in &self.calibration.observation_identifiers {
            if identifier.is_empty() || identifier.len() > MAX_STRING_BYTES {
                return Err(PlanPersistenceError::InvalidPlan {
                    field: "invalid calibration observation identifier",
                });
            }
            if seen.contains(identifier) {
                return Err(PlanPersistenceError::InvalidPlan {
                    field: "duplicate calibration observation identifier",
                });
            }
            seen.push(identifier.clone());
        }
        Ok(())
    }

    fn validate_reasons(&self) -> Result<(), PlanPersistenceError> {
        if self.reasons.is_empty() || self.reasons.len() > MAX_LIST_ITEMS {
            return Err(PlanPersistenceError::InvalidPlan {
                field: "missing or excessive plan reasons",
            });
        }
        let mut codes = Vec::new();
        let mut calibration_reason = None;
        for reason in &self.reasons {
            if codes.contains(&reason.code) {
                return Err(PlanPersistenceError::InvalidPlan {
                    field: "duplicate plan reason code",
                });
            }
            codes.push(reason.code);
            if matches!(
                reason.code,
                PlanReasonCode::CalibrationObservationsApplied
                    | PlanReasonCode::CalibrationNotRequested
                    | PlanReasonCode::CalibrationNotApplicable
            ) {
                if calibration_reason.replace(reason.code).is_some() {
                    return Err(PlanPersistenceError::InvalidPlan {
                        field: "multiple calibration reason codes",
                    });
                }
            }
        }
        let calibration_reason = calibration_reason.ok_or(PlanPersistenceError::InvalidPlan {
            field: "missing calibration reason code",
        })?;
        let header_present = self.calibration.calibration_identifier.is_some();
        let observations_present = !self.calibration.observation_identifiers.is_empty();
        let valid = match calibration_reason {
            PlanReasonCode::CalibrationNotRequested => !header_present && !observations_present,
            PlanReasonCode::CalibrationNotApplicable => !observations_present,
            PlanReasonCode::CalibrationObservationsApplied => {
                header_present && observations_present
            }
            _ => false,
        };
        if !valid {
            return Err(PlanPersistenceError::InvalidPlan {
                field: "calibration reason/provenance mismatch",
            });
        }
        Ok(())
    }

    /// Build a temporary compiler input bound to current facts. Only
    /// environment identity, current model path, and the volatile available-RAM
    /// snapshot are replaced; every concrete planner decision is preserved.
    fn materialize_for_context(&self, context: PlanCompilationContext<'_>) -> ExecutionPlan {
        self.materialize(
            context.machine.fingerprint(),
            context.storage.fingerprint(),
            context.storage.model_path.clone(),
            context.machine.available_ram_bytes,
        )
    }

    fn materialize(
        &self,
        machine_fingerprint: u64,
        storage_fingerprint: u64,
        model_path: PathBuf,
        available_ram_bytes: Option<u64>,
    ) -> ExecutionPlan {
        let mut reservations = self.reservations.clone();
        if let Some(available) = available_ram_bytes {
            reservations.available_ram_bytes_at_planning = available;
        }
        ExecutionPlan {
            schema_version: self.execution_schema_version,
            ruleset_version: self.planner_ruleset_version,
            binding: PlanBinding {
                model_descriptor_fingerprint: self.model_descriptor_fingerprint,
                machine_fingerprint,
                storage_fingerprint,
                model_path,
            },
            mode: self.mode,
            strategy: self.strategy,
            ram_budget_bytes: self.ram_budget_bytes,
            cpu_thread_count: self.cpu_thread_count,
            layer_cache: self.layer_cache.clone(),
            io: self.io.clone(),
            gpu: self.gpu.clone(),
            reservations,
            cost: self.cost.clone(),
            calibration: self.calibration.clone(),
            reasons: self.reasons.clone(),
        }
    }

    fn recalibration_reasons(
        &self,
        context: PersistedPlanCompatibilityContext<'_>,
    ) -> Vec<PlanCompatibilityReason> {
        match self.calibration_reason() {
            Some(PlanReasonCode::CalibrationNotRequested) => Vec::new(),
            Some(PlanReasonCode::CalibrationNotApplicable) => {
                vec![PlanCompatibilityReason::CalibrationNotApplied]
            }
            Some(PlanReasonCode::CalibrationObservationsApplied) => self
                .calibration_mismatch_reason(context)
                .into_iter()
                .collect(),
            _ => vec![PlanCompatibilityReason::CalibrationNotApplied],
        }
    }

    fn calibration_reason(&self) -> Option<PlanReasonCode> {
        self.reasons.iter().find_map(|reason| {
            matches!(
                reason.code,
                PlanReasonCode::CalibrationObservationsApplied
                    | PlanReasonCode::CalibrationNotRequested
                    | PlanReasonCode::CalibrationNotApplicable
            )
            .then_some(reason.code)
        })
    }

    fn calibration_mismatch_reason(
        &self,
        context: PersistedPlanCompatibilityContext<'_>,
    ) -> Option<PlanCompatibilityReason> {
        let Some(calibration) = context.calibration else {
            return Some(PlanCompatibilityReason::CalibrationMissing);
        };
        if calibration.validate().is_err() {
            return Some(PlanCompatibilityReason::CalibrationInvalid);
        }
        if self.source_machine_fingerprint != context.compilation.machine.fingerprint() {
            return Some(PlanCompatibilityReason::CalibrationSourceMachineChanged);
        }
        if self.source_storage_fingerprint != context.compilation.storage.fingerprint() {
            return Some(PlanCompatibilityReason::CalibrationSourceStorageChanged);
        }
        if calibration.identifier.as_str()
            != self
                .calibration
                .calibration_identifier
                .as_deref()
                .unwrap_or("")
            || Some(calibration.calibration_plan_identifier.as_str())
                != self.calibration.calibration_plan_identifier.as_deref()
            || Some(calibration.calibration_plan_version)
                != self.calibration.calibration_plan_version
            || Some(calibration.calibration_ruleset_version)
                != self.calibration.calibration_ruleset_version
        {
            return Some(PlanCompatibilityReason::CalibrationProvenanceMismatch);
        }
        if calibration.machine_fingerprint != context.compilation.machine.fingerprint()
            || calibration.model_fingerprint.is_some_and(|fingerprint| {
                fingerprint != context.compilation.model.identity.descriptor_fingerprint
            })
            || calibration
                .storage_fingerprint
                .is_some_and(|fingerprint| fingerprint != context.compilation.storage.fingerprint())
        {
            return Some(PlanCompatibilityReason::CalibrationBindingMismatch);
        }
        if self
            .calibration
            .observation_identifiers
            .iter()
            .any(|identifier| {
                !calibration.observations.iter().any(|observation| {
                    observation.identifier.as_str() == identifier.as_str()
                        && observation.status == crate::planner::ObservationStatus::Measured
                })
            })
        {
            return Some(PlanCompatibilityReason::CalibrationObservationUnavailable);
        }
        None
    }

    fn encode_unchecked(&self) -> Result<Vec<u8>, PlanPersistenceError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(PERSISTED_PLAN_MAGIC);
        put_u32(&mut bytes, self.persistence_schema_version);
        put_u32(&mut bytes, self.execution_schema_version);
        put_u32(&mut bytes, self.planner_ruleset_version);
        put_u64(&mut bytes, self.model_descriptor_fingerprint);
        put_u64(&mut bytes, self.source_machine_fingerprint);
        put_u64(&mut bytes, self.source_storage_fingerprint);
        bytes.push(encode_mode(self.mode));
        bytes.push(encode_strategy(self.strategy));
        put_u64(&mut bytes, self.ram_budget_bytes);
        put_u64(
            &mut bytes,
            u64::try_from(self.cpu_thread_count).map_err(|_| {
                PlanPersistenceError::NumericOverflow {
                    field: "CPU thread count",
                }
            })?,
        );
        put_bool(&mut bytes, self.layer_cache.enabled);
        put_u64(&mut bytes, self.layer_cache.capacity_bytes);
        put_bool(&mut bytes, self.io.read_coalescing_enabled);
        put_bool(&mut bytes, self.io.grouped_read_buffer_reuse_enabled);
        bytes.push(encode_gpu_preference(self.gpu.preference));
        put_bool(&mut bytes, self.gpu.enabled);
        put_optional_string(
            &mut bytes,
            self.gpu.selected_device_id.as_deref(),
            "GPU device identifier",
        )?;
        put_u64(
            &mut bytes,
            self.reservations.available_ram_bytes_at_planning,
        );
        put_u64(&mut bytes, self.reservations.persistent_resident_bytes);
        put_u64(&mut bytes, self.reservations.persistent_startup_peak_bytes);
        put_u64(&mut bytes, self.reservations.largest_layer_load_peak_bytes);
        put_u64(&mut bytes, self.reservations.managed_lower_bound_bytes);
        put_u64(
            &mut bytes,
            self.reservations.remaining_budget_after_lower_bound_bytes,
        );
        put_u64(&mut bytes, self.cost.physical_read_bytes_per_forward);
        put_optional_u64(&mut bytes, self.cost.calibrated_read_time_ns_per_forward);
        put_optional_u64(&mut bytes, self.cost.observed_strategy_latency_ns);
        put_optional_string(
            &mut bytes,
            self.calibration.calibration_identifier.as_deref(),
            "calibration identifier",
        )?;
        put_optional_string(
            &mut bytes,
            self.calibration.calibration_plan_identifier.as_deref(),
            "calibration plan identifier",
        )?;
        put_optional_u32(&mut bytes, self.calibration.calibration_plan_version);
        put_optional_u32(&mut bytes, self.calibration.calibration_ruleset_version);
        put_string_list(
            &mut bytes,
            &self.calibration.observation_identifiers,
            "calibration observation identifiers",
        )?;
        put_list_count(&mut bytes, self.reasons.len(), "plan reasons")?;
        for reason in &self.reasons {
            bytes.push(encode_reason(reason.code));
            put_optional_u64(&mut bytes, reason.value);
        }
        if bytes.len() > MAX_PERSISTED_PLAN_BYTES {
            return Err(PlanPersistenceError::TooLarge {
                bytes: bytes.len(),
                maximum: MAX_PERSISTED_PLAN_BYTES,
            });
        }
        Ok(bytes)
    }
}

struct PlanReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> PlanReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.position)
    }

    fn take(
        &mut self,
        length: usize,
        field: &'static str,
    ) -> Result<&'a [u8], PlanPersistenceError> {
        let end =
            self.position
                .checked_add(length)
                .ok_or(PlanPersistenceError::NumericOverflow {
                    field: "persisted plan cursor",
                })?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or(PlanPersistenceError::Truncated { field })?;
        self.position = end;
        Ok(value)
    }

    fn u8(&mut self, field: &'static str) -> Result<u8, PlanPersistenceError> {
        Ok(self.take(1, field)?[0])
    }

    fn u32(&mut self, field: &'static str) -> Result<u32, PlanPersistenceError> {
        let bytes: [u8; 4] = self
            .take(4, field)?
            .try_into()
            .expect("four-byte slice validated above");
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self, field: &'static str) -> Result<u64, PlanPersistenceError> {
        let bytes: [u8; 8] = self
            .take(8, field)?
            .try_into()
            .expect("eight-byte slice validated above");
        Ok(u64::from_le_bytes(bytes))
    }

    fn boolean(&mut self, field: &'static str) -> Result<bool, PlanPersistenceError> {
        let value = self.u8(field)?;
        match value {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(PlanPersistenceError::InvalidBoolean { field, value }),
        }
    }

    fn optional_u32(&mut self, field: &'static str) -> Result<Option<u32>, PlanPersistenceError> {
        if self.boolean(field)? {
            self.u32(field).map(Some)
        } else {
            Ok(None)
        }
    }

    fn optional_u64(&mut self, field: &'static str) -> Result<Option<u64>, PlanPersistenceError> {
        if self.boolean(field)? {
            self.u64(field).map(Some)
        } else {
            Ok(None)
        }
    }

    fn string(&mut self, field: &'static str) -> Result<String, PlanPersistenceError> {
        let length = self.u32(field)? as usize;
        if length > MAX_STRING_BYTES {
            return Err(PlanPersistenceError::LimitExceeded {
                field,
                value: length,
                maximum: MAX_STRING_BYTES,
            });
        }
        let bytes = self.take(length, field)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| PlanPersistenceError::InvalidUtf8 { field })
    }

    fn optional_string(
        &mut self,
        field: &'static str,
    ) -> Result<Option<String>, PlanPersistenceError> {
        if self.boolean(field)? {
            self.string(field).map(Some)
        } else {
            Ok(None)
        }
    }

    fn list_count(&mut self, field: &'static str) -> Result<usize, PlanPersistenceError> {
        let count = self.u32(field)? as usize;
        if count > MAX_LIST_ITEMS {
            return Err(PlanPersistenceError::LimitExceeded {
                field,
                value: count,
                maximum: MAX_LIST_ITEMS,
            });
        }
        Ok(count)
    }

    fn string_list(&mut self, field: &'static str) -> Result<Vec<String>, PlanPersistenceError> {
        let count = self.list_count(field)?;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.string(field)?);
        }
        Ok(values)
    }
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_bool(bytes: &mut Vec<u8>, value: bool) {
    bytes.push(u8::from(value));
}

fn put_optional_u32(bytes: &mut Vec<u8>, value: Option<u32>) {
    put_bool(bytes, value.is_some());
    if let Some(value) = value {
        put_u32(bytes, value);
    }
}

fn put_optional_u64(bytes: &mut Vec<u8>, value: Option<u64>) {
    put_bool(bytes, value.is_some());
    if let Some(value) = value {
        put_u64(bytes, value);
    }
}

fn put_string(
    bytes: &mut Vec<u8>,
    value: &str,
    field: &'static str,
) -> Result<(), PlanPersistenceError> {
    if value.len() > MAX_STRING_BYTES {
        return Err(PlanPersistenceError::LimitExceeded {
            field,
            value: value.len(),
            maximum: MAX_STRING_BYTES,
        });
    }
    let length =
        u32::try_from(value.len()).map_err(|_| PlanPersistenceError::NumericOverflow { field })?;
    put_u32(bytes, length);
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}

fn put_optional_string(
    bytes: &mut Vec<u8>,
    value: Option<&str>,
    field: &'static str,
) -> Result<(), PlanPersistenceError> {
    put_bool(bytes, value.is_some());
    if let Some(value) = value {
        put_string(bytes, value, field)?;
    }
    Ok(())
}

fn put_list_count(
    bytes: &mut Vec<u8>,
    count: usize,
    field: &'static str,
) -> Result<(), PlanPersistenceError> {
    if count > MAX_LIST_ITEMS {
        return Err(PlanPersistenceError::LimitExceeded {
            field,
            value: count,
            maximum: MAX_LIST_ITEMS,
        });
    }
    let count =
        u32::try_from(count).map_err(|_| PlanPersistenceError::NumericOverflow { field })?;
    put_u32(bytes, count);
    Ok(())
}

fn put_string_list(
    bytes: &mut Vec<u8>,
    values: &[String],
    field: &'static str,
) -> Result<(), PlanPersistenceError> {
    put_list_count(bytes, values.len(), field)?;
    for value in values {
        put_string(bytes, value, field)?;
    }
    Ok(())
}

fn encode_mode(value: OperatingMode) -> u8 {
    match value {
        OperatingMode::BackgroundConstrained => 0,
        OperatingMode::BalancedNormal => 1,
        OperatingMode::MaximumPerformance => 2,
        OperatingMode::Advanced => 3,
    }
}

fn decode_mode(value: u8) -> Result<OperatingMode, PlanPersistenceError> {
    match value {
        0 => Ok(OperatingMode::BackgroundConstrained),
        1 => Ok(OperatingMode::BalancedNormal),
        2 => Ok(OperatingMode::MaximumPerformance),
        3 => Ok(OperatingMode::Advanced),
        _ => Err(PlanPersistenceError::InvalidEnum {
            field: "operating mode",
            value,
        }),
    }
}

fn encode_strategy(value: StrategyId) -> u8 {
    match value {
        StrategyId::CpuLayerStreaming => 0,
        StrategyId::GpuLayerOffload => 1,
    }
}

fn decode_strategy(value: u8) -> Result<StrategyId, PlanPersistenceError> {
    match value {
        0 => Ok(StrategyId::CpuLayerStreaming),
        1 => Ok(StrategyId::GpuLayerOffload),
        _ => Err(PlanPersistenceError::InvalidEnum {
            field: "execution strategy",
            value,
        }),
    }
}

fn encode_gpu_preference(value: GpuPreference) -> u8 {
    match value {
        GpuPreference::Disabled => 0,
        GpuPreference::Prefer => 1,
        GpuPreference::Require => 2,
    }
}

fn decode_gpu_preference(value: u8) -> Result<GpuPreference, PlanPersistenceError> {
    match value {
        0 => Ok(GpuPreference::Disabled),
        1 => Ok(GpuPreference::Prefer),
        2 => Ok(GpuPreference::Require),
        _ => Err(PlanPersistenceError::InvalidEnum {
            field: "GPU preference",
            value,
        }),
    }
}

fn encode_reason(value: PlanReasonCode) -> u8 {
    match value {
        PlanReasonCode::UserRamBudgetPreserved => 0,
        PlanReasonCode::ThreadCountSelectedByPreset => 1,
        PlanReasonCode::ThreadCountSelectedByAdvancedOverride => 2,
        PlanReasonCode::CpuStrategySelected => 3,
        PlanReasonCode::GpuStrategySelected => 4,
        PlanReasonCode::GpuDisabledByUser => 5,
        PlanReasonCode::GpuUnavailable => 6,
        PlanReasonCode::LayerCacheEnabledWithinStaticCapacity => 7,
        PlanReasonCode::LayerCacheDisabledByPreset => 8,
        PlanReasonCode::LayerCacheDisabledBecauseNoCompleteLayerFits => 9,
        PlanReasonCode::ReadCoalescingEnabled => 10,
        PlanReasonCode::ReadCoalescingDisabled => 11,
        PlanReasonCode::GroupedReadBufferReuseEnabled => 12,
        PlanReasonCode::GroupedReadBufferReuseDisabled => 13,
        PlanReasonCode::CalibrationObservationsApplied => 14,
        PlanReasonCode::CalibrationNotRequested => 15,
        PlanReasonCode::CalibrationNotApplicable => 16,
    }
}

fn decode_reason(value: u8) -> Result<PlanReasonCode, PlanPersistenceError> {
    match value {
        0 => Ok(PlanReasonCode::UserRamBudgetPreserved),
        1 => Ok(PlanReasonCode::ThreadCountSelectedByPreset),
        2 => Ok(PlanReasonCode::ThreadCountSelectedByAdvancedOverride),
        3 => Ok(PlanReasonCode::CpuStrategySelected),
        4 => Ok(PlanReasonCode::GpuStrategySelected),
        5 => Ok(PlanReasonCode::GpuDisabledByUser),
        6 => Ok(PlanReasonCode::GpuUnavailable),
        7 => Ok(PlanReasonCode::LayerCacheEnabledWithinStaticCapacity),
        8 => Ok(PlanReasonCode::LayerCacheDisabledByPreset),
        9 => Ok(PlanReasonCode::LayerCacheDisabledBecauseNoCompleteLayerFits),
        10 => Ok(PlanReasonCode::ReadCoalescingEnabled),
        11 => Ok(PlanReasonCode::ReadCoalescingDisabled),
        12 => Ok(PlanReasonCode::GroupedReadBufferReuseEnabled),
        13 => Ok(PlanReasonCode::GroupedReadBufferReuseDisabled),
        14 => Ok(PlanReasonCode::CalibrationObservationsApplied),
        15 => Ok(PlanReasonCode::CalibrationNotRequested),
        16 => Ok(PlanReasonCode::CalibrationNotApplicable),
        _ => Err(PlanPersistenceError::InvalidEnum {
            field: "plan reason code",
            value,
        }),
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::plan_compiler::tests::CompilerFixture;
    use crate::planner::{
        observation_quality_basis_points, CalibrationLevel, MeasurementSample, ObservationMetric,
        ObservationStatus, ObservationUnit, PerformanceObservation, CALIBRATION_PLAN_VERSION,
        CALIBRATION_RULESET_VERSION, EXECUTION_PLAN_SCHEMA_VERSION, PLANNER_RULESET_VERSION,
        PROFILE_SCHEMA_VERSION,
    };

    fn fixture() -> CompilerFixture {
        let mut fixture = CompilerFixture::new();
        fixture.plan.reasons = vec![
            PlanReason {
                code: PlanReasonCode::UserRamBudgetPreserved,
                value: Some(fixture.plan.ram_budget_bytes),
            },
            PlanReason {
                code: PlanReasonCode::ThreadCountSelectedByPreset,
                value: Some(fixture.plan.cpu_thread_count as u64),
            },
            PlanReason {
                code: PlanReasonCode::CpuStrategySelected,
                value: None,
            },
            PlanReason {
                code: PlanReasonCode::LayerCacheEnabledWithinStaticCapacity,
                value: Some(fixture.plan.layer_cache.capacity_bytes),
            },
            PlanReason {
                code: PlanReasonCode::ReadCoalescingEnabled,
                value: None,
            },
            PlanReason {
                code: PlanReasonCode::GroupedReadBufferReuseEnabled,
                value: None,
            },
            PlanReason {
                code: PlanReasonCode::CalibrationNotRequested,
                value: Some(0),
            },
        ];
        fixture
    }

    fn persisted(fixture: &CompilerFixture) -> PersistedExecutionPlan {
        PersistedExecutionPlan::from_execution_plan(&fixture.plan).unwrap()
    }

    fn compatibility_context<'a>(
        fixture: &'a CompilerFixture,
        calibration: Option<&'a CalibrationResult>,
    ) -> PersistedPlanCompatibilityContext<'a> {
        PersistedPlanCompatibilityContext {
            compilation: fixture.context(),
            calibration,
        }
    }

    fn matching_calibration(fixture: &CompilerFixture) -> CalibrationResult {
        let sample = MeasurementSample {
            elapsed_ns: 1_000_000_000,
            units_processed: 1_000_000,
        };
        CalibrationResult {
            schema_version: PROFILE_SCHEMA_VERSION,
            identifier: "persisted-calibration".to_string(),
            level: CalibrationLevel::Quick,
            machine_fingerprint: fixture.machine.fingerprint(),
            model_fingerprint: Some(fixture.model.identity.descriptor_fingerprint),
            storage_fingerprint: Some(fixture.storage.fingerprint()),
            calibration_plan_identifier: "persisted-calibration-plan".to_string(),
            calibration_plan_version: CALIBRATION_PLAN_VERSION,
            calibration_ruleset_version: CALIBRATION_RULESET_VERSION,
            observations: vec![PerformanceObservation {
                identifier: "persisted-read".to_string(),
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

    fn apply_calibration_provenance(
        fixture: &mut CompilerFixture,
        calibration: &CalibrationResult,
    ) {
        fixture
            .plan
            .reasons
            .retain(|reason| reason.code != PlanReasonCode::CalibrationNotRequested);
        fixture.plan.reasons.push(PlanReason {
            code: PlanReasonCode::CalibrationObservationsApplied,
            value: Some(1),
        });
        fixture.plan.calibration = CalibrationProvenance {
            calibration_identifier: Some(calibration.identifier.clone()),
            calibration_plan_identifier: Some(calibration.calibration_plan_identifier.clone()),
            calibration_plan_version: Some(calibration.calibration_plan_version),
            calibration_ruleset_version: Some(calibration.calibration_ruleset_version),
            observation_identifiers: vec!["persisted-read".to_string()],
        };
        fixture.plan.cost.calibrated_read_time_ns_per_forward = Some(5_000_000);
    }

    #[test]
    fn test_valid_plan_round_trip_preserves_relevant_contract() {
        let fixture = fixture();
        let persisted = persisted(&fixture);
        let loaded = PersistedExecutionPlan::from_bytes(&persisted.to_bytes().unwrap()).unwrap();

        assert_eq!(loaded, persisted);
        assert_eq!(loaded.execution_schema_version, fixture.plan.schema_version);
        assert_eq!(loaded.planner_ruleset_version, fixture.plan.ruleset_version);
        assert_eq!(loaded.strategy, fixture.plan.strategy);
        assert_eq!(loaded.ram_budget_bytes, fixture.plan.ram_budget_bytes);
        assert_eq!(loaded.cpu_thread_count, fixture.plan.cpu_thread_count);
        assert_eq!(loaded.layer_cache, fixture.plan.layer_cache);
        assert_eq!(loaded.io, fixture.plan.io);
        assert_eq!(loaded.gpu, fixture.plan.gpu);
        assert_eq!(loaded.reservations, fixture.plan.reservations);
        assert_eq!(loaded.calibration, fixture.plan.calibration);
    }

    #[test]
    fn test_serialized_output_and_file_round_trip_are_deterministic() {
        let fixture = fixture();
        let persisted = persisted(&fixture);
        let first = persisted.to_bytes().unwrap();
        let second = persisted.to_bytes().unwrap();
        assert_eq!(first, second);

        let directory = TempDir::new().unwrap();
        let path = directory.path().join("plan.rfp");
        persisted.save_new(&path).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), first);
        let loaded = PersistedExecutionPlan::load(&path).unwrap();
        assert_eq!(loaded, persisted);
        assert_eq!(
            loaded.validate_compatibility(compatibility_context(&fixture, None)),
            PlanCompatibility::Valid
        );
        let runtime_config = loaded.compile_runtime_config(fixture.context()).unwrap();
        assert_eq!(
            runtime_config.cpu_thread_count,
            fixture.plan.cpu_thread_count
        );
        assert!(matches!(
            persisted.save_new(&path),
            Err(PlanPersistenceError::Io(ref error))
                if error.kind() == std::io::ErrorKind::AlreadyExists
        ));
    }

    #[test]
    fn test_malformed_truncated_and_trailing_input_are_rejected() {
        let fixture = fixture();
        let bytes = persisted(&fixture).to_bytes().unwrap();

        let mut invalid_magic = bytes.clone();
        invalid_magic[0] ^= 0xff;
        assert!(matches!(
            PersistedExecutionPlan::from_bytes(&invalid_magic),
            Err(PlanPersistenceError::InvalidMagic)
        ));
        assert!(matches!(
            PersistedExecutionPlan::from_bytes(&bytes[..24]),
            Err(PlanPersistenceError::Truncated { .. })
        ));
        let mut trailing = bytes;
        trailing.push(0);
        assert!(matches!(
            PersistedExecutionPlan::from_bytes(&trailing),
            Err(PlanPersistenceError::TrailingBytes { bytes: 1 })
        ));
        let oversized = vec![0; MAX_PERSISTED_PLAN_BYTES + 1];
        assert!(matches!(
            PersistedExecutionPlan::from_bytes(&oversized),
            Err(PlanPersistenceError::TooLarge { .. })
        ));
    }

    #[test]
    fn test_unsupported_persistence_schema_is_rejected() {
        let fixture = fixture();
        let mut bytes = persisted(&fixture).to_bytes().unwrap();
        bytes[8..12].copy_from_slice(&(PERSISTED_EXECUTION_PLAN_SCHEMA_VERSION + 1).to_le_bytes());
        assert!(matches!(
            PersistedExecutionPlan::from_bytes(&bytes),
            Err(PlanPersistenceError::UnsupportedPersistenceSchema { .. })
        ));
    }

    #[test]
    fn test_invalid_enum_and_state_combinations_are_rejected() {
        let fixture = fixture();
        let persisted = persisted(&fixture);
        let mut invalid_enum = persisted.to_bytes().unwrap();
        invalid_enum[44] = 0xff;
        assert!(matches!(
            PersistedExecutionPlan::from_bytes(&invalid_enum),
            Err(PlanPersistenceError::InvalidEnum {
                field: "operating mode",
                ..
            })
        ));

        let mut invalid_state = persisted;
        invalid_state.gpu.enabled = true;
        let invalid_state = invalid_state.encode_unchecked().unwrap();
        assert!(matches!(
            PersistedExecutionPlan::from_bytes(&invalid_state),
            Err(PlanPersistenceError::InvalidPlan { .. })
        ));
    }

    #[test]
    fn test_missing_required_identity_is_rejected() {
        let fixture = fixture();
        let mut persisted = persisted(&fixture);
        persisted.model_descriptor_fingerprint = 0;
        assert!(matches!(
            persisted.to_bytes(),
            Err(PlanPersistenceError::InvalidPlan {
                field: "missing or invalid fingerprint"
            })
        ));
    }

    #[test]
    fn test_same_model_machine_and_storage_are_valid() {
        let fixture = fixture();
        let loaded =
            PersistedExecutionPlan::from_bytes(&persisted(&fixture).to_bytes().unwrap()).unwrap();
        assert_eq!(
            loaded.validate_compatibility(compatibility_context(&fixture, None)),
            PlanCompatibility::Valid
        );
        let runtime_config = loaded.compile_runtime_config(fixture.context()).unwrap();
        assert_eq!(
            runtime_config.cpu_thread_count,
            fixture.plan.cpu_thread_count
        );
        assert_eq!(
            runtime_config.ram_budget_bytes,
            fixture.plan.ram_budget_bytes
        );
        assert_eq!(
            runtime_config.layer_cache_capacity_bytes,
            fixture.plan.layer_cache.capacity_bytes
        );
    }

    #[test]
    fn test_cross_machine_load_revalidates_concrete_thread_requirements() {
        let source = fixture();
        let bytes = persisted(&source).to_bytes().unwrap();
        let loaded = PersistedExecutionPlan::from_bytes(&bytes).unwrap();

        let mut compatible_machine = fixture();
        compatible_machine.machine.cpu_model = Some("different-compatible-cpu".to_string());
        compatible_machine.refresh_capabilities();
        assert_ne!(
            loaded.source_machine_fingerprint,
            compatible_machine.machine.fingerprint()
        );
        assert_eq!(
            loaded.validate_compatibility(compatibility_context(&compatible_machine, None)),
            PlanCompatibility::Valid
        );

        let mut insufficient_machine = fixture();
        insufficient_machine.machine.logical_cpu_cores = 2;
        insufficient_machine.refresh_capabilities();
        assert_eq!(
            loaded.validate_compatibility(compatibility_context(&insufficient_machine, None)),
            PlanCompatibility::Incompatible
        );
        assert_eq!(loaded.cpu_thread_count, source.plan.cpu_thread_count);
    }

    #[test]
    fn test_changed_storage_is_valid_only_when_required_capabilities_remain() {
        let source = fixture();
        let loaded =
            PersistedExecutionPlan::from_bytes(&persisted(&source).to_bytes().unwrap()).unwrap();

        let mut compatible_storage = fixture();
        compatible_storage.storage.device_id = Some("different-device".to_string());
        compatible_storage.refresh_capabilities();
        assert_ne!(
            loaded.source_storage_fingerprint,
            compatible_storage.storage.fingerprint()
        );
        assert_eq!(
            loaded.validate_compatibility(compatibility_context(&compatible_storage, None)),
            PlanCompatibility::Valid
        );

        compatible_storage.storage.seekable = false;
        compatible_storage.refresh_capabilities();
        assert_eq!(
            loaded.validate_compatibility(compatibility_context(&compatible_storage, None)),
            PlanCompatibility::Incompatible
        );
    }

    #[test]
    fn test_model_fingerprint_mismatch_is_incompatible() {
        let source = fixture();
        let loaded = persisted(&source);
        let mut current = fixture();
        current.model.identity.descriptor_fingerprint = current
            .model
            .identity
            .descriptor_fingerprint
            .wrapping_add(1)
            .max(1);
        current.refresh_capabilities();
        let report = loaded.compatibility_report(compatibility_context(&current, None));
        assert_eq!(report.state, PlanCompatibility::Incompatible);
        assert_eq!(
            report.reasons,
            vec![PlanCompatibilityReason::ModelFingerprintMismatch]
        );
    }

    #[test]
    fn test_unsupported_execution_schema_or_ruleset_is_incompatible() {
        let fixture = fixture();
        for (schema, ruleset) in [
            (EXECUTION_PLAN_SCHEMA_VERSION + 1, PLANNER_RULESET_VERSION),
            (EXECUTION_PLAN_SCHEMA_VERSION, PLANNER_RULESET_VERSION + 1),
        ] {
            let mut persisted = persisted(&fixture);
            persisted.execution_schema_version = schema;
            persisted.planner_ruleset_version = ruleset;
            let loaded =
                PersistedExecutionPlan::from_bytes(&persisted.to_bytes().unwrap()).unwrap();
            assert_eq!(
                loaded.validate_compatibility(compatibility_context(&fixture, None)),
                PlanCompatibility::Incompatible
            );
        }
    }

    #[test]
    fn test_matching_calibration_is_valid_and_stale_calibration_recommends_refresh() {
        let mut fixture = fixture();
        let calibration = matching_calibration(&fixture);
        calibration.validate().unwrap();
        apply_calibration_provenance(&mut fixture, &calibration);
        let loaded = persisted(&fixture);

        assert_eq!(
            loaded.validate_compatibility(compatibility_context(&fixture, Some(&calibration),)),
            PlanCompatibility::Valid
        );

        let mut stale = calibration.clone();
        stale.machine_fingerprint = stale.machine_fingerprint.wrapping_add(1).max(1);
        stale.validate().unwrap();
        let stale_report =
            loaded.compatibility_report(compatibility_context(&fixture, Some(&stale)));
        assert_eq!(
            stale_report.state,
            PlanCompatibility::CompatibleRecalibrationRecommended
        );
        assert_eq!(
            stale_report.reasons,
            vec![PlanCompatibilityReason::CalibrationBindingMismatch]
        );
    }

    #[test]
    fn test_missing_calibration_recommends_refresh_when_execution_remains_valid() {
        let mut calibrated_fixture = fixture();
        let calibration = matching_calibration(&calibrated_fixture);
        apply_calibration_provenance(&mut calibrated_fixture, &calibration);
        let loaded = persisted(&calibrated_fixture);
        let missing_report =
            loaded.compatibility_report(compatibility_context(&calibrated_fixture, None));
        assert_eq!(
            missing_report.state,
            PlanCompatibility::CompatibleRecalibrationRecommended
        );
        assert_eq!(
            missing_report.reasons,
            vec![PlanCompatibilityReason::CalibrationMissing]
        );

        let mut requested_without_result = fixture();
        requested_without_result.plan.calibration = CalibrationProvenance {
            calibration_identifier: None,
            calibration_plan_identifier: None,
            calibration_plan_version: None,
            calibration_ruleset_version: None,
            observation_identifiers: Vec::new(),
        };
        requested_without_result
            .plan
            .reasons
            .retain(|reason| reason.code != PlanReasonCode::CalibrationNotRequested);
        requested_without_result.plan.reasons.push(PlanReason {
            code: PlanReasonCode::CalibrationNotApplicable,
            value: Some(0),
        });
        assert_eq!(
            persisted(&requested_without_result)
                .validate_compatibility(compatibility_context(&requested_without_result, None),),
            PlanCompatibility::CompatibleRecalibrationRecommended
        );
    }

    #[test]
    fn test_hard_compiler_invariant_failure_is_incompatible() {
        let fixture = fixture();
        let mut persisted = persisted(&fixture);
        persisted.layer_cache.capacity_bytes = 1_000;
        let loaded = PersistedExecutionPlan::from_bytes(&persisted.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            loaded.compile_runtime_config(fixture.context()),
            Err(PersistedPlanValidationError::Compilation(
                PlanCompilationError::LayerCacheCapacityCannotFitLayer { .. }
            ))
        ));
        assert_eq!(
            loaded.validate_compatibility(compatibility_context(&fixture, None)),
            PlanCompatibility::Incompatible
        );
    }

    #[test]
    fn test_gpu_execution_requirement_is_loadable_but_incompatible() {
        let fixture = fixture();
        let mut persisted = persisted(&fixture);
        persisted.strategy = StrategyId::GpuLayerOffload;
        persisted.gpu = GpuDecision {
            preference: GpuPreference::Prefer,
            enabled: true,
            selected_device_id: Some("gpu0".to_string()),
        };
        let loaded = PersistedExecutionPlan::from_bytes(&persisted.to_bytes().unwrap()).unwrap();
        assert_eq!(
            loaded.validate_compatibility(compatibility_context(&fixture, None)),
            PlanCompatibility::Incompatible
        );
    }

    #[test]
    fn test_persistence_excludes_paths_payloads_and_runtime_state() {
        let fixture = fixture();
        let bytes = persisted(&fixture).to_bytes().unwrap();
        let path = fixture.plan.binding.model_path.to_string_lossy();
        assert!(!bytes
            .windows(path.len())
            .any(|window| window == path.as_bytes()));
        let forbidden_values: &[&[u8]] = &[b"weight:", b"cache:layer:", b"GGUF"];
        for &forbidden in forbidden_values {
            assert!(!bytes
                .windows(forbidden.len())
                .any(|window| window == forbidden));
        }
    }
}
