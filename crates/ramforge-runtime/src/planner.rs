//! Deterministic planning contracts and explicit policy selection.
//!
//! This module orchestrates parsed model facts and the existing static planning
//! report. It performs no hardware detection, calibration execution, tensor I/O,
//! model execution, persistence, or UI rendering.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::PathBuf;

use ramforge_core::GgufModel;

use crate::layer::group_layers;
use crate::layer_read::build_layer_read_plan;
use crate::plan::{ExecutionMemoryPlan, PlanResult};
use crate::support::{architecture_capability, ggml_type_supported_for_inference, RunSupport};

pub const PROFILE_SCHEMA_VERSION: u32 = 1;
pub const EXECUTION_PLAN_SCHEMA_VERSION: u32 = 1;
pub const PLANNER_RULESET_VERSION: u32 = 1;
pub const CALIBRATION_PLAN_VERSION: u32 = 1;
pub const CALIBRATION_RULESET_VERSION: u32 = 1;

// GPU hardware may be represented in profiles, but RAMforge has no GPU
// execution backend yet. Capability derivation must not imply otherwise.
const GPU_EXECUTION_IMPLEMENTED: bool = false;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatingMode {
    BackgroundConstrained,
    BalancedNormal,
    MaximumPerformance,
    Advanced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPreference {
    Disabled,
    Prefer,
    Require,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationLevel {
    None,
    Quick,
    Standard,
    Thorough,
}

impl CalibrationLevel {
    pub(crate) const fn rank(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Quick => 1,
            Self::Standard => 2,
            Self::Thorough => 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AdvancedOverrides {
    pub thread_count: Option<usize>,
    pub layer_cache_enabled: Option<bool>,
    pub layer_cache_capacity_bytes: Option<u64>,
    pub read_coalescing_enabled: Option<bool>,
    pub grouped_read_buffer_reuse_enabled: Option<bool>,
    pub gpu_device_id: Option<String>,
}

impl AdvancedOverrides {
    fn is_empty(&self) -> bool {
        self.thread_count.is_none()
            && self.layer_cache_enabled.is_none()
            && self.layer_cache_capacity_bytes.is_none()
            && self.read_coalescing_enabled.is_none()
            && self.grouped_read_buffer_reuse_enabled.is_none()
            && self.gpu_device_id.is_none()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserProfile {
    pub schema_version: u32,
    pub ram_budget_bytes: u64,
    pub mode: OperatingMode,
    /// Upper bound for planner-selected CPU threads. `None` means the detected
    /// logical CPU count remains the upper bound.
    pub cpu_thread_limit: Option<usize>,
    pub gpu_preference: GpuPreference,
    pub calibration_level: CalibrationLevel,
    pub advanced: AdvancedOverrides,
}

impl UserProfile {
    pub fn new(ram_budget_bytes: u64, mode: OperatingMode) -> Self {
        Self {
            schema_version: PROFILE_SCHEMA_VERSION,
            ram_budget_bytes,
            mode,
            cpu_thread_limit: None,
            gpu_preference: GpuPreference::Disabled,
            calibration_level: CalibrationLevel::None,
            advanced: AdvancedOverrides::default(),
        }
    }

    pub fn validate(&self) -> Result<(), PlannerError> {
        if self.schema_version != PROFILE_SCHEMA_VERSION {
            return Err(PlannerError::invalid("UserProfile", "schema_version"));
        }
        if self.ram_budget_bytes == 0 {
            return Err(PlannerError::invalid("UserProfile", "ram_budget_bytes"));
        }
        if self.cpu_thread_limit == Some(0) {
            return Err(PlannerError::invalid("UserProfile", "cpu_thread_limit"));
        }
        if self.mode != OperatingMode::Advanced && !self.advanced.is_empty() {
            return Err(PlannerError::Infeasible(
                FeasibilityRejection::AdvancedOverrideRequiresAdvancedMode,
            ));
        }
        if self.advanced.thread_count == Some(0) {
            return Err(PlannerError::invalid(
                "UserProfile",
                "advanced.thread_count",
            ));
        }
        if self.advanced.layer_cache_capacity_bytes == Some(0) {
            return Err(PlannerError::invalid(
                "UserProfile",
                "advanced.layer_cache_capacity_bytes",
            ));
        }
        if self
            .advanced
            .layer_cache_capacity_bytes
            .is_some_and(|bytes| bytes > self.ram_budget_bytes)
        {
            return Err(PlannerError::invalid(
                "UserProfile",
                "advanced.layer_cache_capacity_bytes",
            ));
        }
        if self.advanced.layer_cache_enabled == Some(false)
            && self.advanced.layer_cache_capacity_bytes.is_some()
        {
            return Err(PlannerError::invalid(
                "UserProfile",
                "disabled layer cache has a capacity override",
            ));
        }
        if self.advanced.gpu_device_id.is_some()
            && self.gpu_preference == GpuPreference::Disabled
        {
            return Err(PlannerError::invalid(
                "UserProfile",
                "GPU device selected while GPU use is disabled",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CpuFeature {
    Avx2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Availability {
    pub detected: bool,
    pub supported: bool,
    pub usable: bool,
}

impl Availability {
    pub const fn unavailable() -> Self {
        Self {
            detected: false,
            supported: false,
            usable: false,
        }
    }

    fn validate(self, profile: &'static str) -> Result<(), PlannerError> {
        if self.supported && !self.detected {
            return Err(PlannerError::invalid(profile, "supported but not detected"));
        }
        if self.usable && !self.supported {
            return Err(PlannerError::invalid(profile, "usable but not supported"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDeviceProfile {
    pub identifier: String,
    pub name: Option<String>,
    pub dedicated_memory_bytes: Option<u64>,
    pub backend: Availability,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineProfile {
    pub schema_version: u32,
    pub os: String,
    pub architecture: String,
    pub physical_cpu_cores: Option<usize>,
    pub logical_cpu_cores: usize,
    pub cpu_features: BTreeSet<CpuFeature>,
    pub total_ram_bytes: u64,
    pub available_ram_bytes: u64,
    pub cpu_backend: Availability,
    pub gpus: Vec<GpuDeviceProfile>,
}

impl MachineProfile {
    pub fn validate(&self) -> Result<(), PlannerError> {
        if self.schema_version != PROFILE_SCHEMA_VERSION {
            return Err(PlannerError::invalid("MachineProfile", "schema_version"));
        }
        if self.os.is_empty() || self.architecture.is_empty() {
            return Err(PlannerError::invalid("MachineProfile", "platform"));
        }
        if self.logical_cpu_cores == 0
            || self.physical_cpu_cores == Some(0)
            || self
                .physical_cpu_cores
                .is_some_and(|cores| cores > self.logical_cpu_cores)
        {
            return Err(PlannerError::invalid("MachineProfile", "cpu core count"));
        }
        if self.total_ram_bytes == 0 || self.available_ram_bytes > self.total_ram_bytes {
            return Err(PlannerError::invalid("MachineProfile", "RAM values"));
        }
        self.cpu_backend.validate("MachineProfile.cpu_backend")?;
        let mut identifiers = BTreeSet::new();
        for gpu in &self.gpus {
            if gpu.identifier.is_empty() || !identifiers.insert(gpu.identifier.as_str()) {
                return Err(PlannerError::invalid(
                    "MachineProfile",
                    "GPU identifier",
                ));
            }
            gpu.backend.validate("MachineProfile.gpu.backend")?;
        }
        Ok(())
    }

    pub fn fingerprint(&self) -> u64 {
        let mut hash = fingerprint_start();
        fingerprint_str(&mut hash, &self.os);
        fingerprint_str(&mut hash, &self.architecture);
        fingerprint_u64(&mut hash, self.physical_cpu_cores.unwrap_or(0) as u64);
        fingerprint_u64(&mut hash, self.logical_cpu_cores as u64);
        fingerprint_u64(&mut hash, self.total_ram_bytes);
        for feature in &self.cpu_features {
            fingerprint_u64(&mut hash, *feature as u64 + 1);
        }
        fingerprint_availability(&mut hash, self.cpu_backend);
        let mut gpus: Vec<&GpuDeviceProfile> = self.gpus.iter().collect();
        gpus.sort_by(|left, right| left.identifier.cmp(&right.identifier));
        for gpu in gpus {
            fingerprint_str(&mut hash, &gpu.identifier);
            fingerprint_str(&mut hash, gpu.name.as_deref().unwrap_or(""));
            fingerprint_u64(&mut hash, gpu.dedicated_memory_bytes.unwrap_or(0));
            fingerprint_availability(&mut hash, gpu.backend);
        }
        hash
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelExecutionCompatibility {
    Executable,
    InspectPlanOnly,
    UnknownArchitecture,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelIdentity {
    pub display_name: Option<String>,
    /// Stable descriptor/metadata fingerprint. This is an identity aid, not a
    /// cryptographic digest of model weights.
    pub descriptor_fingerprint: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorFormatProfile {
    pub format: String,
    pub tensor_count: usize,
    pub element_count: Option<u64>,
    pub byte_count: Option<u64>,
    pub runtime_supported: bool,
}

#[derive(Debug)]
struct FormatAccumulator {
    tensor_count: usize,
    element_count: Option<u64>,
    byte_count: Option<u64>,
    runtime_supported: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelProfile {
    pub schema_version: u32,
    pub identity: ModelIdentity,
    pub architecture: Option<String>,
    pub context_size: Option<u64>,
    pub layer_count: Option<u64>,
    pub tensor_count: usize,
    pub total_tensor_elements: Option<u64>,
    pub file_size_bytes: u64,
    pub tensor_data_bytes: Option<u64>,
    pub tensor_formats: Vec<TensorFormatProfile>,
    pub execution_compatibility: ModelExecutionCompatibility,
    pub has_coalescible_layer_reads: bool,
    pub has_reusable_grouped_read_candidate: bool,
}

impl ModelProfile {
    pub fn from_gguf(model: &GgufModel) -> Self {
        let info = model.info();
        let architecture = info.architecture.clone();
        let execution_compatibility = architecture
            .as_deref()
            .and_then(architecture_capability)
            .map(|capability| {
                if capability.run == RunSupport::Supported {
                    ModelExecutionCompatibility::Executable
                } else {
                    ModelExecutionCompatibility::InspectPlanOnly
                }
            })
            .unwrap_or(ModelExecutionCompatibility::UnknownArchitecture);

        let mut formats: BTreeMap<String, FormatAccumulator> = BTreeMap::new();
        let mut total_tensor_elements = Some(0u64);
        for tensor in &model.tensors {
            let entry = formats
                .entry(tensor.ggml_type.name())
                .or_insert_with(|| FormatAccumulator {
                    tensor_count: 0,
                    element_count: Some(0),
                    byte_count: Some(0),
                    runtime_supported: ggml_type_supported_for_inference(tensor.ggml_type),
                });
            entry.tensor_count += 1;
            entry.element_count = entry
                .element_count
                .and_then(|total| total.checked_add(tensor.num_elements));
            entry.byte_count = match (entry.byte_count, tensor.byte_length) {
                (Some(total), Some(bytes)) => total.checked_add(bytes),
                _ => None,
            };
            total_tensor_elements = total_tensor_elements
                .and_then(|total| total.checked_add(tensor.num_elements));
        }
        let tensor_formats = formats
            .into_iter()
            .map(|(format, accumulated)| TensorFormatProfile {
                format,
                tensor_count: accumulated.tensor_count,
                element_count: accumulated.element_count,
                byte_count: accumulated.byte_count,
                runtime_supported: accumulated.runtime_supported,
            })
            .collect();

        let mut has_coalescible_layer_reads = false;
        let mut has_reusable_grouped_read_candidate = false;
        if let Some(layer_count) = info
            .block_count
            .and_then(|count| usize::try_from(count).ok())
        {
            for layer in group_layers(model, layer_count) {
                if let Ok(plan) = build_layer_read_plan(&layer.tensors) {
                    has_coalescible_layer_reads |=
                        plan.ranges.iter().any(|range| range.tensors.len() > 1);
                    has_reusable_grouped_read_candidate |=
                        plan.reusable_group_buffer_bytes().is_some();
                }
            }
        }

        Self {
            schema_version: PROFILE_SCHEMA_VERSION,
            identity: ModelIdentity {
                display_name: info.name,
                descriptor_fingerprint: model_descriptor_fingerprint(model),
            },
            architecture,
            context_size: info.context_length,
            layer_count: info.block_count,
            tensor_count: model.tensors.len(),
            total_tensor_elements,
            file_size_bytes: model.file_size,
            tensor_data_bytes: model.total_tensor_bytes(),
            tensor_formats,
            execution_compatibility,
            has_coalescible_layer_reads,
            has_reusable_grouped_read_candidate,
        }
    }

    pub fn validate(&self) -> Result<(), PlannerError> {
        if self.schema_version != PROFILE_SCHEMA_VERSION {
            return Err(PlannerError::invalid("ModelProfile", "schema_version"));
        }
        if self.identity.descriptor_fingerprint == 0 {
            return Err(PlannerError::invalid(
                "ModelProfile",
                "descriptor_fingerprint",
            ));
        }
        let mut format_names = BTreeSet::new();
        let represented_tensors = self.tensor_formats.iter().try_fold(
            0usize,
            |total, format| {
                if format.format.is_empty()
                    || format.tensor_count == 0
                    || !format_names.insert(format.format.as_str())
                {
                    return None;
                }
                total.checked_add(format.tensor_count)
            },
        );
        if represented_tensors != Some(self.tensor_count) {
            return Err(PlannerError::invalid(
                "ModelProfile",
                "tensor distribution",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageKind {
    Unknown,
    Local,
    Network,
    Removable,
    MemoryBacked,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageProfile {
    pub schema_version: u32,
    pub model_path: PathBuf,
    pub filesystem_id: Option<String>,
    pub device_id: Option<String>,
    pub kind: StorageKind,
    pub seekable: bool,
}

impl StorageProfile {
    pub fn for_model_path(model_path: impl Into<PathBuf>) -> Self {
        Self {
            schema_version: PROFILE_SCHEMA_VERSION,
            model_path: model_path.into(),
            filesystem_id: None,
            device_id: None,
            kind: StorageKind::Unknown,
            seekable: true,
        }
    }

    pub fn validate(&self) -> Result<(), PlannerError> {
        if self.schema_version != PROFILE_SCHEMA_VERSION {
            return Err(PlannerError::invalid("StorageProfile", "schema_version"));
        }
        if self.model_path.as_os_str().is_empty() {
            return Err(PlannerError::invalid("StorageProfile", "model_path"));
        }
        Ok(())
    }

    pub fn fingerprint(&self) -> u64 {
        let mut hash = fingerprint_start();
        fingerprint_str(&mut hash, &self.model_path.to_string_lossy());
        fingerprint_str(&mut hash, self.filesystem_id.as_deref().unwrap_or(""));
        fingerprint_str(&mut hash, self.device_id.as_deref().unwrap_or(""));
        fingerprint_u64(&mut hash, self.kind as u64);
        fingerprint_u64(&mut hash, self.seekable as u64);
        hash
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyId {
    CpuLayerStreaming,
    GpuLayerOffload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrategyDefinition {
    pub id: StrategyId,
    pub runtime_available: bool,
    pub requires_gpu: bool,
}

pub const STRATEGY_REGISTRY: &[StrategyDefinition] = &[
    StrategyDefinition {
        id: StrategyId::CpuLayerStreaming,
        runtime_available: true,
        requires_gpu: false,
    },
    StrategyDefinition {
        id: StrategyId::GpuLayerOffload,
        runtime_available: GPU_EXECUTION_IMPLEMENTED,
        requires_gpu: true,
    },
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilitySet {
    pub schema_version: u32,
    pub machine_fingerprint: u64,
    pub model_fingerprint: u64,
    pub storage_fingerprint: u64,
    pub storage_usable: bool,
    pub cpu_backend_usable: bool,
    pub cpu_simd_avx2: bool,
    pub gpu_backend_usable: bool,
    pub model_execution_compatible: bool,
    pub layer_cache_possible: bool,
    pub read_coalescing_possible: bool,
    pub grouped_read_buffer_reuse_possible: bool,
    pub supported_tensor_formats: Vec<String>,
    pub unsupported_tensor_formats: Vec<String>,
}

impl CapabilitySet {
    pub fn derive(
        machine: &MachineProfile,
        model: &ModelProfile,
        storage: &StorageProfile,
        static_plan: &PlanResult,
    ) -> Self {
        let execution = static_plan.execution_memory.as_ref();
        let layer_cache_possible = execution.is_some_and(|memory| {
            memory.min_layer_resident_bytes > 0
                && memory.layer_cache_capacity_bytes >= memory.min_layer_resident_bytes
        });
        let gpu_hardware_usable = machine.gpus.iter().any(|gpu| gpu.backend.usable);
        let cpu_strategy_available = STRATEGY_REGISTRY.iter().any(|strategy| {
            strategy.id == StrategyId::CpuLayerStreaming
                && !strategy.requires_gpu
                && strategy.runtime_available
        });
        let gpu_strategy_available = STRATEGY_REGISTRY.iter().any(|strategy| {
            strategy.id == StrategyId::GpuLayerOffload
                && strategy.requires_gpu
                && strategy.runtime_available
        });
        let mut supported_tensor_formats = Vec::new();
        let mut unsupported_tensor_formats = Vec::new();
        for format in &model.tensor_formats {
            if format.runtime_supported {
                supported_tensor_formats.push(format.format.clone());
            } else {
                unsupported_tensor_formats.push(format.format.clone());
            }
        }

        Self {
            schema_version: PROFILE_SCHEMA_VERSION,
            machine_fingerprint: machine.fingerprint(),
            model_fingerprint: model.identity.descriptor_fingerprint,
            storage_fingerprint: storage.fingerprint(),
            storage_usable: storage.seekable,
            cpu_backend_usable: machine.cpu_backend.usable && cpu_strategy_available,
            cpu_simd_avx2: machine.cpu_backend.usable
                && cpu_strategy_available
                && machine.cpu_features.contains(&CpuFeature::Avx2),
            gpu_backend_usable: gpu_hardware_usable && gpu_strategy_available,
            model_execution_compatible: model.execution_compatibility
                == ModelExecutionCompatibility::Executable
                && execution.is_some(),
            layer_cache_possible,
            read_coalescing_possible: execution.is_some()
                && storage.seekable
                && model.has_coalescible_layer_reads,
            grouped_read_buffer_reuse_possible: execution.is_some()
                && storage.seekable
                && model.has_reusable_grouped_read_candidate,
            supported_tensor_formats,
            unsupported_tensor_formats,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationMetric {
    CpuFloatThroughput,
    SequentialReadThroughput,
    RandomReadLatency,
    MemoryBandwidth,
    QuantizedDecodeThroughput,
    StrategyLatency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationUnit {
    BytesPerSecond,
    ElementsPerSecond,
    Nanoseconds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationStatus {
    Measured,
    Unavailable,
    Skipped,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationStatusReason {
    CapabilityUnavailable,
    DependencyUnavailable,
    ResourceLimitExceeded,
    TimeLimitReached,
    TestNotImplemented,
    IoFailure,
    InvalidWorkload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeasurementSample {
    pub elapsed_ns: u64,
    pub units_processed: u64,
}

/// Derive observation quality from the relative spread of per-sample rates.
/// Identical rates produce 10,000 basis points; a spread equal to the maximum
/// observed rate produces zero. No statistical confidence is implied.
pub fn observation_quality_basis_points(samples: &[MeasurementSample]) -> u16 {
    if samples.is_empty()
        || samples
            .iter()
            .any(|sample| sample.elapsed_ns == 0 || sample.units_processed == 0)
    {
        return 0;
    }
    let mut minimum_rate = u128::MAX;
    let mut maximum_rate = 0u128;
    for sample in samples {
        let rate = sample.units_processed as u128 * 1_000_000_000u128
            / sample.elapsed_ns as u128;
        minimum_rate = minimum_rate.min(rate);
        maximum_rate = maximum_rate.max(rate);
    }
    if maximum_rate == 0 {
        return 0;
    }
    let spread_basis_points =
        ((maximum_rate - minimum_rate) * 10_000u128 / maximum_rate).min(10_000);
    (10_000u128 - spread_basis_points) as u16
}

pub fn aggregate_observation_value(
    unit: ObservationUnit,
    samples: &[MeasurementSample],
) -> Option<u64> {
    if samples.is_empty()
        || samples
            .iter()
            .any(|sample| sample.elapsed_ns == 0 || sample.units_processed == 0)
    {
        return None;
    }
    let elapsed_ns: u128 = samples.iter().map(|sample| sample.elapsed_ns as u128).sum();
    let units: u128 = samples
        .iter()
        .map(|sample| sample.units_processed as u128)
        .sum();
    let value = match unit {
        ObservationUnit::BytesPerSecond | ObservationUnit::ElementsPerSecond => {
            units.checked_mul(1_000_000_000u128)? / elapsed_ns
        }
        ObservationUnit::Nanoseconds => elapsed_ns / units,
    };
    Some(value.min(u64::MAX as u128) as u64).filter(|value| *value > 0)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerformanceObservation {
    pub identifier: String,
    pub metric: ObservationMetric,
    pub unit: ObservationUnit,
    /// Measured aggregate value in `unit`; absent for non-measured states.
    pub value: Option<u64>,
    pub status: ObservationStatus,
    pub status_reason: Option<ObservationStatusReason>,
    pub strategy: Option<StrategyId>,
    pub quantization_format: Option<String>,
    pub calibration_level: CalibrationLevel,
    pub workload_bytes: u64,
    pub workload_elements: u64,
    pub measurement_duration_ns: u64,
    pub sample_count: u32,
    pub raw_samples: Vec<MeasurementSample>,
    pub timestamp_unix_seconds: Option<u64>,
    /// Sample-spread quality in basis points, calculated from raw samples.
    pub quality_basis_points: u16,
}

impl PerformanceObservation {
    pub fn validate(&self) -> Result<(), PlannerError> {
        if self.identifier.is_empty()
            || self.calibration_level == CalibrationLevel::None
            || self.quality_basis_points > 10_000
        {
            return Err(PlannerError::invalid(
                "PerformanceObservation",
                "identity/level/quality",
            ));
        }
        let expected_unit = match self.metric {
            ObservationMetric::SequentialReadThroughput
            | ObservationMetric::MemoryBandwidth => ObservationUnit::BytesPerSecond,
            ObservationMetric::CpuFloatThroughput
            | ObservationMetric::QuantizedDecodeThroughput => ObservationUnit::ElementsPerSecond,
            ObservationMetric::RandomReadLatency | ObservationMetric::StrategyLatency => {
                ObservationUnit::Nanoseconds
            }
        };
        if self.unit != expected_unit {
            return Err(PlannerError::invalid(
                "PerformanceObservation",
                "metric/unit mismatch",
            ));
        }
        if self.metric == ObservationMetric::StrategyLatency && self.strategy.is_none() {
            return Err(PlannerError::invalid(
                "PerformanceObservation",
                "strategy latency without strategy",
            ));
        }
        if self.metric == ObservationMetric::QuantizedDecodeThroughput
            && self.quantization_format.is_none()
        {
            return Err(PlannerError::invalid(
                "PerformanceObservation",
                "decode throughput without quantization format",
            ));
        }

        match self.status {
            ObservationStatus::Measured => {
                if !self.value.is_some_and(|value| value > 0)
                    || self.measurement_duration_ns == 0
                    || self.sample_count == 0
                    || self.raw_samples.len() != self.sample_count as usize
                    || self.status_reason.is_some()
                    || self
                        .raw_samples
                        .iter()
                        .any(|sample| sample.elapsed_ns == 0 || sample.units_processed == 0)
                    || self
                        .raw_samples
                        .iter()
                        .try_fold(0u64, |total, sample| {
                            total.checked_add(sample.elapsed_ns)
                        })
                        != Some(self.measurement_duration_ns)
                    || self.quality_basis_points
                        != observation_quality_basis_points(&self.raw_samples)
                    || self.value
                        != aggregate_observation_value(self.unit, &self.raw_samples)
                {
                    return Err(PlannerError::invalid(
                        "PerformanceObservation",
                        "measured state",
                    ));
                }
            }
            ObservationStatus::Unavailable
            | ObservationStatus::Skipped
            | ObservationStatus::Failed => {
                if self.value.is_some()
                    || self.measurement_duration_ns != 0
                    || self.sample_count != 0
                    || !self.raw_samples.is_empty()
                    || self.status_reason.is_none()
                    || self.quality_basis_points != 0
                {
                    return Err(PlannerError::invalid(
                        "PerformanceObservation",
                        "non-measured state",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationResult {
    pub schema_version: u32,
    pub identifier: String,
    pub level: CalibrationLevel,
    pub machine_fingerprint: u64,
    pub model_fingerprint: Option<u64>,
    pub storage_fingerprint: Option<u64>,
    pub calibration_plan_identifier: String,
    pub calibration_plan_version: u32,
    pub calibration_ruleset_version: u32,
    pub observations: Vec<PerformanceObservation>,
}

impl CalibrationResult {
    pub fn validate(&self) -> Result<(), PlannerError> {
        if self.schema_version != PROFILE_SCHEMA_VERSION
            || self.identifier.is_empty()
            || self.level == CalibrationLevel::None
            || self.machine_fingerprint == 0
            || self.model_fingerprint == Some(0)
            || self.storage_fingerprint == Some(0)
            || self.calibration_plan_identifier.is_empty()
            || self.calibration_plan_version != CALIBRATION_PLAN_VERSION
            || self.calibration_ruleset_version != CALIBRATION_RULESET_VERSION
        {
            return Err(PlannerError::invalid(
                "CalibrationResult",
                "identity/schema/level",
            ));
        }
        let mut identifiers = BTreeSet::new();
        for observation in &self.observations {
            observation.validate()?;
            if observation.calibration_level != self.level {
                return Err(PlannerError::invalid(
                    "CalibrationResult",
                    "observation calibration level mismatch",
                ));
            }
            if !identifiers.insert(observation.identifier.as_str()) {
                return Err(PlannerError::invalid(
                    "CalibrationResult",
                    "duplicate observation identifier",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresetPolicy {
    pub mode: OperatingMode,
    pub thread_fraction_numerator: usize,
    pub thread_fraction_denominator: usize,
    /// Fraction of the statically available layer-cache capacity, in basis points.
    pub layer_cache_capacity_basis_points: u16,
    pub read_coalescing: bool,
    pub grouped_read_buffer_reuse: bool,
    pub allow_gpu: bool,
}

impl OperatingMode {
    pub const fn policy(self) -> PresetPolicy {
        match self {
            Self::BackgroundConstrained => PresetPolicy {
                mode: self,
                thread_fraction_numerator: 1,
                thread_fraction_denominator: 2,
                layer_cache_capacity_basis_points: 0,
                read_coalescing: true,
                grouped_read_buffer_reuse: false,
                allow_gpu: false,
            },
            Self::BalancedNormal => PresetPolicy {
                mode: self,
                thread_fraction_numerator: 3,
                thread_fraction_denominator: 4,
                layer_cache_capacity_basis_points: 5_000,
                read_coalescing: true,
                grouped_read_buffer_reuse: true,
                allow_gpu: true,
            },
            Self::MaximumPerformance => PresetPolicy {
                mode: self,
                thread_fraction_numerator: 1,
                thread_fraction_denominator: 1,
                layer_cache_capacity_basis_points: 10_000,
                read_coalescing: true,
                grouped_read_buffer_reuse: true,
                allow_gpu: true,
            },
            Self::Advanced => PresetPolicy {
                mode: self,
                thread_fraction_numerator: 3,
                thread_fraction_denominator: 4,
                layer_cache_capacity_basis_points: 5_000,
                read_coalescing: true,
                grouped_read_buffer_reuse: true,
                allow_gpu: true,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanReasonCode {
    UserRamBudgetPreserved,
    ThreadCountSelectedByPreset,
    ThreadCountSelectedByAdvancedOverride,
    CpuStrategySelected,
    GpuStrategySelected,
    GpuDisabledByUser,
    GpuUnavailable,
    LayerCacheEnabledWithinStaticCapacity,
    LayerCacheDisabledByPreset,
    LayerCacheDisabledBecauseNoCompleteLayerFits,
    ReadCoalescingEnabled,
    ReadCoalescingDisabled,
    GroupedReadBufferReuseEnabled,
    GroupedReadBufferReuseDisabled,
    CalibrationObservationsApplied,
    CalibrationNotRequested,
    CalibrationNotApplicable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanReason {
    pub code: PlanReasonCode,
    pub value: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerCacheDecision {
    pub enabled: bool,
    pub capacity_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IoDecision {
    pub read_coalescing_enabled: bool,
    pub grouped_read_buffer_reuse_enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDecision {
    pub preference: GpuPreference,
    pub enabled: bool,
    pub selected_device_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceReservations {
    pub available_ram_bytes_at_planning: u64,
    pub persistent_resident_bytes: u64,
    pub persistent_startup_peak_bytes: u64,
    pub largest_layer_load_peak_bytes: u64,
    pub managed_lower_bound_bytes: u64,
    pub remaining_budget_after_lower_bound_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CostEstimate {
    pub physical_read_bytes_per_forward: u64,
    pub calibrated_read_time_ns_per_forward: Option<u64>,
    pub observed_strategy_latency_ns: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationProvenance {
    pub calibration_identifier: Option<String>,
    pub calibration_plan_identifier: Option<String>,
    pub calibration_plan_version: Option<u32>,
    pub calibration_ruleset_version: Option<u32>,
    pub observation_identifiers: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanBinding {
    pub model_descriptor_fingerprint: u64,
    pub machine_fingerprint: u64,
    pub storage_fingerprint: u64,
    pub model_path: PathBuf,
}

/// Versioned planner output. It contains identifiers, decisions, and numeric
/// reservations only; model weights and runtime-owned allocations are excluded.
/// A persistent serialization format is intentionally not defined yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionPlan {
    pub schema_version: u32,
    pub ruleset_version: u32,
    pub binding: PlanBinding,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanCompatibility {
    Valid,
    CompatibleRecalibrationRecommended,
    Incompatible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeasibilityRejection {
    AdvancedOverrideRequiresAdvancedMode,
    RamBudgetExceedsTotalRam,
    RamBudgetExceedsAvailableRam,
    StorageNotUsable,
    CpuBackendUnavailable,
    ModelNotExecutable,
    ExecutionPreflightUnavailable,
    StaticPlanDoesNotMatchProfiles,
    ManagedMemoryLowerBoundExceedsBudget,
    GpuRequiredButUnavailable,
    RequestedThreadCountUnavailable,
    LayerCacheRequiredButUnavailable,
    LayerCacheCapacityExceedsStaticCapacity,
    LayerCacheCapacityCannotFitCompleteLayer,
    ReadCoalescingRequiredButUnavailable,
    GroupedReadBufferReuseRequiredButUnavailable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlannerError {
    InvalidProfile {
        profile: &'static str,
        field: &'static str,
    },
    Infeasible(FeasibilityRejection),
}

impl PlannerError {
    fn invalid(profile: &'static str, field: &'static str) -> Self {
        Self::InvalidProfile { profile, field }
    }
}

impl fmt::Display for PlannerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProfile { profile, field } => {
                write!(formatter, "invalid {profile}: {field}")
            }
            Self::Infeasible(reason) => write!(formatter, "infeasible plan: {reason:?}"),
        }
    }
}

impl std::error::Error for PlannerError {}

#[derive(Debug, Default, Clone, Copy)]
pub struct Planner;

impl Planner {
    pub fn derive_capabilities(
        &self,
        machine: &MachineProfile,
        model: &ModelProfile,
        storage: &StorageProfile,
        static_plan: &PlanResult,
    ) -> CapabilitySet {
        CapabilitySet::derive(machine, model, storage, static_plan)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn plan(
        &self,
        user: &UserProfile,
        machine: &MachineProfile,
        model: &ModelProfile,
        storage: &StorageProfile,
        capabilities: &CapabilitySet,
        static_plan: &PlanResult,
        calibration: Option<&CalibrationResult>,
    ) -> Result<ExecutionPlan, PlannerError> {
        user.validate()?;
        machine.validate()?;
        model.validate()?;
        storage.validate()?;
        if let Some(calibration) = calibration {
            calibration.validate()?;
        }

        let machine_fingerprint = machine.fingerprint();
        let derived_capabilities = CapabilitySet::derive(machine, model, storage, static_plan);
        if capabilities != &derived_capabilities {
            return Err(PlannerError::Infeasible(
                FeasibilityRejection::StaticPlanDoesNotMatchProfiles,
            ));
        }
        if static_plan.ram_requested != user.ram_budget_bytes
            || static_plan.file_size != model.file_size_bytes
            || static_plan.tensor_count != model.tensor_count
            || static_plan.architecture != model.architecture
        {
            return Err(PlannerError::Infeasible(
                FeasibilityRejection::StaticPlanDoesNotMatchProfiles,
            ));
        }
        if user.ram_budget_bytes > machine.total_ram_bytes {
            return Err(PlannerError::Infeasible(
                FeasibilityRejection::RamBudgetExceedsTotalRam,
            ));
        }
        if user.ram_budget_bytes > machine.available_ram_bytes {
            return Err(PlannerError::Infeasible(
                FeasibilityRejection::RamBudgetExceedsAvailableRam,
            ));
        }
        if !capabilities.storage_usable {
            return Err(PlannerError::Infeasible(
                FeasibilityRejection::StorageNotUsable,
            ));
        }
        if !capabilities.cpu_backend_usable {
            return Err(PlannerError::Infeasible(
                FeasibilityRejection::CpuBackendUnavailable,
            ));
        }
        if !capabilities.model_execution_compatible {
            return Err(PlannerError::Infeasible(
                FeasibilityRejection::ModelNotExecutable,
            ));
        }
        let execution = static_plan.execution_memory.as_ref().ok_or(
            PlannerError::Infeasible(FeasibilityRejection::ExecutionPreflightUnavailable),
        )?;
        if !execution.layer_streaming_lower_bound_fits
            || execution.managed_lower_bound_bytes > user.ram_budget_bytes
        {
            return Err(PlannerError::Infeasible(
                FeasibilityRejection::ManagedMemoryLowerBoundExceedsBudget,
            ));
        }

        let policy = user.mode.policy();
        let thread_limit = machine
            .logical_cpu_cores
            .min(user.cpu_thread_limit.unwrap_or(machine.logical_cpu_cores));
        let (cpu_thread_count, thread_reason) =
            if let Some(requested) = user.advanced.thread_count {
                if requested > thread_limit {
                    return Err(PlannerError::Infeasible(
                        FeasibilityRejection::RequestedThreadCountUnavailable,
                    ));
                }
                (
                    requested,
                    PlanReasonCode::ThreadCountSelectedByAdvancedOverride,
                )
            } else {
                let selected = thread_limit
                    .saturating_mul(policy.thread_fraction_numerator)
                    / policy.thread_fraction_denominator;
                (
                    selected.max(1),
                    PlanReasonCode::ThreadCountSelectedByPreset,
                )
            };

        if user.gpu_preference == GpuPreference::Require
            && (!policy.allow_gpu || !capabilities.gpu_backend_usable)
        {
            return Err(PlannerError::Infeasible(
                FeasibilityRejection::GpuRequiredButUnavailable,
            ));
        }
        if user.advanced.gpu_device_id.is_some() && !capabilities.gpu_backend_usable {
            return Err(PlannerError::Infeasible(
                FeasibilityRejection::GpuRequiredButUnavailable,
            ));
        }
        let gpu_enabled = policy.allow_gpu
            && user.gpu_preference != GpuPreference::Disabled
            && capabilities.gpu_backend_usable;
        let selected_device_id = if gpu_enabled {
            select_gpu(machine, user.advanced.gpu_device_id.as_deref())?
        } else {
            None
        };
        let strategy = if gpu_enabled {
            StrategyId::GpuLayerOffload
        } else {
            StrategyId::CpuLayerStreaming
        };

        let explicit_cache = user.advanced.layer_cache_enabled;
        let explicit_cache_capacity = user.advanced.layer_cache_capacity_bytes;
        let cache_is_required = explicit_cache == Some(true) || explicit_cache_capacity.is_some();
        let wants_cache = explicit_cache.unwrap_or(policy.layer_cache_capacity_basis_points > 0);
        if cache_is_required && !capabilities.layer_cache_possible {
            return Err(PlannerError::Infeasible(
                FeasibilityRejection::LayerCacheRequiredButUnavailable,
            ));
        }
        let maximum_cache_capacity = execution.layer_cache_capacity_bytes;
        let requested_cache_capacity = if !wants_cache || !capabilities.layer_cache_possible {
            0
        } else if let Some(capacity) = explicit_cache_capacity {
            if capacity > maximum_cache_capacity {
                return Err(PlannerError::Infeasible(
                    FeasibilityRejection::LayerCacheCapacityExceedsStaticCapacity,
                ));
            }
            capacity
        } else {
            ((maximum_cache_capacity as u128
                * policy.layer_cache_capacity_basis_points as u128)
                / 10_000) as u64
        };
        if cache_is_required
            && requested_cache_capacity < execution.min_layer_resident_bytes
        {
            return Err(PlannerError::Infeasible(
                FeasibilityRejection::LayerCacheCapacityCannotFitCompleteLayer,
            ));
        }
        let cache_enabled = requested_cache_capacity >= execution.min_layer_resident_bytes
            && execution.min_layer_resident_bytes > 0;
        let layer_cache = LayerCacheDecision {
            enabled: cache_enabled,
            capacity_bytes: if cache_enabled {
                requested_cache_capacity
            } else {
                0
            },
        };

        let read_coalescing = resolve_capability_choice(
            user.advanced.read_coalescing_enabled,
            policy.read_coalescing,
            capabilities.read_coalescing_possible,
            FeasibilityRejection::ReadCoalescingRequiredButUnavailable,
        )?;
        let grouped_read_buffer_reuse = resolve_capability_choice(
            user.advanced.grouped_read_buffer_reuse_enabled,
            policy.grouped_read_buffer_reuse,
            capabilities.grouped_read_buffer_reuse_possible,
            FeasibilityRejection::GroupedReadBufferReuseRequiredButUnavailable,
        )?;

        let storage_fingerprint = storage.fingerprint();
        let (cost, calibration_provenance, calibration_reason) = estimate_cost(
            user,
            machine_fingerprint,
            model.identity.descriptor_fingerprint,
            storage_fingerprint,
            strategy,
            execution,
            calibration,
        );

        let mut reasons = vec![
            PlanReason {
                code: PlanReasonCode::UserRamBudgetPreserved,
                value: Some(user.ram_budget_bytes),
            },
            PlanReason {
                code: thread_reason,
                value: Some(cpu_thread_count as u64),
            },
            PlanReason {
                code: if gpu_enabled {
                    PlanReasonCode::GpuStrategySelected
                } else {
                    PlanReasonCode::CpuStrategySelected
                },
                value: None,
            },
        ];
        if !gpu_enabled {
            reasons.push(PlanReason {
                code: if user.gpu_preference == GpuPreference::Disabled {
                    PlanReasonCode::GpuDisabledByUser
                } else {
                    PlanReasonCode::GpuUnavailable
                },
                value: None,
            });
        }
        reasons.extend([
            PlanReason {
                code: if layer_cache.enabled {
                    PlanReasonCode::LayerCacheEnabledWithinStaticCapacity
                } else if !wants_cache {
                    PlanReasonCode::LayerCacheDisabledByPreset
                } else {
                    PlanReasonCode::LayerCacheDisabledBecauseNoCompleteLayerFits
                },
                value: Some(layer_cache.capacity_bytes),
            },
            PlanReason {
                code: if read_coalescing {
                    PlanReasonCode::ReadCoalescingEnabled
                } else {
                    PlanReasonCode::ReadCoalescingDisabled
                },
                value: None,
            },
            PlanReason {
                code: if grouped_read_buffer_reuse {
                    PlanReasonCode::GroupedReadBufferReuseEnabled
                } else {
                    PlanReasonCode::GroupedReadBufferReuseDisabled
                },
                value: None,
            },
            PlanReason {
                code: calibration_reason,
                value: Some(calibration_provenance.observation_identifiers.len() as u64),
            },
        ]);

        Ok(ExecutionPlan {
            schema_version: EXECUTION_PLAN_SCHEMA_VERSION,
            ruleset_version: PLANNER_RULESET_VERSION,
            binding: PlanBinding {
                model_descriptor_fingerprint: model.identity.descriptor_fingerprint,
                machine_fingerprint,
                storage_fingerprint,
                model_path: storage.model_path.clone(),
            },
            mode: user.mode,
            strategy,
            ram_budget_bytes: user.ram_budget_bytes,
            cpu_thread_count,
            layer_cache,
            io: IoDecision {
                read_coalescing_enabled: read_coalescing,
                grouped_read_buffer_reuse_enabled: grouped_read_buffer_reuse,
            },
            gpu: GpuDecision {
                preference: user.gpu_preference,
                enabled: gpu_enabled,
                selected_device_id,
            },
            reservations: ResourceReservations {
                available_ram_bytes_at_planning: machine.available_ram_bytes,
                persistent_resident_bytes: execution.persistent_resident_bytes,
                persistent_startup_peak_bytes: execution.persistent_startup_peak_bytes,
                largest_layer_load_peak_bytes: execution.largest_layer_load_peak_bytes,
                managed_lower_bound_bytes: execution.managed_lower_bound_bytes,
                remaining_budget_after_lower_bound_bytes: user
                    .ram_budget_bytes
                    .saturating_sub(execution.managed_lower_bound_bytes),
            },
            cost,
            calibration: calibration_provenance,
            reasons,
        })
    }
}

fn resolve_capability_choice(
    explicit: Option<bool>,
    policy_default: bool,
    available: bool,
    rejection: FeasibilityRejection,
) -> Result<bool, PlannerError> {
    if explicit == Some(true) && !available {
        return Err(PlannerError::Infeasible(rejection));
    }
    Ok(explicit.unwrap_or(policy_default) && available)
}

fn select_gpu(
    machine: &MachineProfile,
    requested_identifier: Option<&str>,
) -> Result<Option<String>, PlannerError> {
    let mut usable: Vec<&GpuDeviceProfile> = machine
        .gpus
        .iter()
        .filter(|gpu| gpu.backend.usable)
        .collect();
    usable.sort_by(|left, right| left.identifier.cmp(&right.identifier));
    if let Some(identifier) = requested_identifier {
        return usable
            .into_iter()
            .find(|gpu| gpu.identifier.as_str() == identifier)
            .map(|gpu| Some(gpu.identifier.clone()))
            .ok_or(PlannerError::Infeasible(
                FeasibilityRejection::GpuRequiredButUnavailable,
            ));
    }
    Ok(usable.first().map(|gpu| gpu.identifier.clone()))
}

fn estimate_cost(
    user: &UserProfile,
    machine_fingerprint: u64,
    model_fingerprint: u64,
    storage_fingerprint: u64,
    strategy: StrategyId,
    execution: &ExecutionMemoryPlan,
    calibration: Option<&CalibrationResult>,
) -> (CostEstimate, CalibrationProvenance, PlanReasonCode) {
    let mut estimate = CostEstimate {
        physical_read_bytes_per_forward: execution.estimated_physical_bytes_per_forward,
        calibrated_read_time_ns_per_forward: None,
        observed_strategy_latency_ns: None,
    };
    let mut provenance = CalibrationProvenance {
        calibration_identifier: None,
        calibration_plan_identifier: None,
        calibration_plan_version: None,
        calibration_ruleset_version: None,
        observation_identifiers: Vec::new(),
    };
    if user.calibration_level == CalibrationLevel::None {
        return (
            estimate,
            provenance,
            PlanReasonCode::CalibrationNotRequested,
        );
    }
    let Some(calibration) = calibration else {
        return (
            estimate,
            provenance,
            PlanReasonCode::CalibrationNotApplicable,
        );
    };
    if calibration.level.rank() < user.calibration_level.rank()
        || calibration.machine_fingerprint != machine_fingerprint
        || calibration
            .model_fingerprint
            .is_some_and(|fingerprint| fingerprint != model_fingerprint)
        || calibration
            .storage_fingerprint
            .is_some_and(|fingerprint| fingerprint != storage_fingerprint)
    {
        return (
            estimate,
            provenance,
            PlanReasonCode::CalibrationNotApplicable,
        );
    }

    provenance.calibration_identifier = Some(calibration.identifier.clone());
    provenance.calibration_plan_identifier =
        Some(calibration.calibration_plan_identifier.clone());
    provenance.calibration_plan_version = Some(calibration.calibration_plan_version);
    provenance.calibration_ruleset_version = Some(calibration.calibration_ruleset_version);
    if let Some(read) = best_observation(
        calibration,
        ObservationMetric::SequentialReadThroughput,
        None,
    ) {
        estimate.calibrated_read_time_ns_per_forward = Some(ceil_rate_duration_ns(
            execution.estimated_physical_bytes_per_forward,
            read.value.expect("measured observation validated above"),
        ));
        provenance
            .observation_identifiers
            .push(read.identifier.clone());
    }
    if let Some(latency) = best_observation(
        calibration,
        ObservationMetric::StrategyLatency,
        Some(strategy),
    ) {
        estimate.observed_strategy_latency_ns =
            Some(latency.value.expect("measured observation validated above"));
        provenance
            .observation_identifiers
            .push(latency.identifier.clone());
    }
    provenance.observation_identifiers.sort();
    let reason = if provenance.observation_identifiers.is_empty() {
        PlanReasonCode::CalibrationNotApplicable
    } else {
        PlanReasonCode::CalibrationObservationsApplied
    };
    (estimate, provenance, reason)
}

fn best_observation(
    calibration: &CalibrationResult,
    metric: ObservationMetric,
    strategy: Option<StrategyId>,
) -> Option<&PerformanceObservation> {
    calibration
        .observations
        .iter()
        .filter(|observation| {
            observation.status == ObservationStatus::Measured
                && observation.metric == metric
                && observation.strategy == strategy
        })
        .max_by(|left, right| {
            left.quality_basis_points
                .cmp(&right.quality_basis_points)
                .then(left.sample_count.cmp(&right.sample_count))
                .then(
                    left.measurement_duration_ns
                        .cmp(&right.measurement_duration_ns),
                )
                // Lexicographically smaller identifiers win the final tie.
                .then_with(|| reverse_identifier_order(&left.identifier, &right.identifier))
        })
}

fn reverse_identifier_order(left: &str, right: &str) -> Ordering {
    right.cmp(left)
}

fn ceil_rate_duration_ns(units: u64, units_per_second: u64) -> u64 {
    let numerator = units as u128 * 1_000_000_000u128;
    let denominator = units_per_second as u128;
    let value = numerator.div_ceil(denominator);
    value.min(u64::MAX as u128) as u64
}

fn model_descriptor_fingerprint(model: &GgufModel) -> u64 {
    let mut hash = fingerprint_start();
    fingerprint_u64(&mut hash, model.version as u64);
    fingerprint_u64(&mut hash, model.file_size);
    fingerprint_u64(&mut hash, model.alignment);
    let info = model.info();
    fingerprint_str(&mut hash, info.architecture.as_deref().unwrap_or(""));
    fingerprint_str(&mut hash, info.name.as_deref().unwrap_or(""));
    fingerprint_str(&mut hash, info.tokenizer_model.as_deref().unwrap_or(""));
    for value in [
        info.context_length,
        info.embedding_length,
        info.block_count,
        info.head_count,
        info.head_count_kv,
        info.expert_count,
        info.expert_used_count,
        info.file_type.map(u64::from),
        info.vocab_size.map(|value| value as u64),
    ] {
        fingerprint_u64(&mut hash, value.unwrap_or(u64::MAX));
    }
    for tensor in &model.tensors {
        fingerprint_str(&mut hash, &tensor.name);
        fingerprint_u64(&mut hash, tensor.ggml_type.as_u32() as u64);
        fingerprint_u64(&mut hash, tensor.offset);
        fingerprint_u64(&mut hash, tensor.byte_length.unwrap_or(u64::MAX));
        fingerprint_u64(&mut hash, tensor.num_elements);
        for dimension in &tensor.dimensions {
            fingerprint_u64(&mut hash, *dimension);
        }
    }
    hash
}

fn fingerprint_start() -> u64 {
    0xcbf29ce484222325
}

fn fingerprint_u64(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        fingerprint_byte(hash, byte);
    }
}

fn fingerprint_str(hash: &mut u64, value: &str) {
    fingerprint_u64(hash, value.len() as u64);
    for byte in value.as_bytes() {
        fingerprint_byte(hash, *byte);
    }
}

fn fingerprint_availability(hash: &mut u64, availability: Availability) {
    fingerprint_u64(hash, availability.detected as u64);
    fingerprint_u64(hash, availability.supported as u64);
    fingerprint_u64(hash, availability.usable as u64);
}

fn fingerprint_byte(hash: &mut u64, byte: u8) {
    *hash ^= byte as u64;
    *hash = (*hash).wrapping_mul(0x100000001b3);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use ramforge_core::{GgmlType, MetadataValue, TensorDescriptor};

    fn tiny_model(architecture: &str) -> GgufModel {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            MetadataValue::String(architecture.to_string()),
        );
        for (key, value) in [
            ("llama.vocab_size", 16),
            ("llama.context_length", 64),
            ("llama.embedding_length", 8),
            ("llama.block_count", 1),
            ("llama.feed_forward_length", 16),
            ("llama.attention.head_count", 2),
            ("llama.attention.head_count_kv", 2),
        ] {
            metadata.insert(key.to_string(), MetadataValue::UInt32(value));
        }
        let definitions = [
            ("token_embd.weight", vec![8, 16]),
            ("output_norm.weight", vec![8]),
            ("blk.0.attn_norm.weight", vec![8]),
            ("blk.0.attn_q.weight", vec![8, 8]),
            ("blk.0.attn_k.weight", vec![8, 8]),
            ("blk.0.attn_v.weight", vec![8, 8]),
            ("blk.0.attn_output.weight", vec![8, 8]),
            ("blk.0.ffn_norm.weight", vec![8]),
            ("blk.0.ffn_gate.weight", vec![8, 16]),
            ("blk.0.ffn_up.weight", vec![8, 16]),
            ("blk.0.ffn_down.weight", vec![16, 8]),
        ];
        let tensors = definitions
            .into_iter()
            .scan(0u64, |offset, (name, dimensions)| {
                let num_elements = dimensions.iter().product::<u64>();
                let byte_length = num_elements * 4;
                let descriptor = TensorDescriptor {
                    name: name.to_string(),
                    dimensions,
                    ggml_type: GgmlType::F32,
                    offset: *offset,
                    file_offset: *offset,
                    byte_length: Some(byte_length),
                    num_elements,
                };
                *offset += byte_length;
                Some(descriptor)
            })
            .collect::<Vec<_>>();
        let file_size = tensors
            .iter()
            .filter_map(|tensor| tensor.byte_length)
            .sum();
        GgufModel {
            path: PathBuf::from("/models/tiny.gguf"),
            file_size,
            version: 3,
            metadata,
            tensors,
            alignment: 32,
            data_start_offset: 0,
        }
    }

    fn machine_profile() -> MachineProfile {
        let mut cpu_features = BTreeSet::new();
        cpu_features.insert(CpuFeature::Avx2);
        MachineProfile {
            schema_version: PROFILE_SCHEMA_VERSION,
            os: "linux".to_string(),
            architecture: "x86_64".to_string(),
            physical_cpu_cores: Some(4),
            logical_cpu_cores: 8,
            cpu_features,
            total_ram_bytes: 32_000,
            available_ram_bytes: 24_000,
            cpu_backend: Availability {
                detected: true,
                supported: true,
                usable: true,
            },
            gpus: vec![GpuDeviceProfile {
                identifier: "gpu0".to_string(),
                name: Some("detected-only".to_string()),
                dedicated_memory_bytes: Some(8_000),
                backend: Availability {
                    detected: true,
                    supported: false,
                    usable: false,
                },
            }],
        }
    }

    fn storage_profile() -> StorageProfile {
        StorageProfile::for_model_path("/models/tiny.gguf")
    }

    fn profiles_and_static_plan(
        mode: OperatingMode,
    ) -> (UserProfile, MachineProfile, ModelProfile, StorageProfile, PlanResult) {
        let model = tiny_model("llama");
        let user = UserProfile::new(10_000, mode);
        let machine = machine_profile();
        let model_profile = ModelProfile::from_gguf(&model);
        let storage = storage_profile();
        let static_plan = crate::plan::plan_model(&model, user.ram_budget_bytes).unwrap();
        (user, machine, model_profile, storage, static_plan)
    }

    #[test]
    fn test_profile_construction_and_model_distribution_are_deterministic() {
        let model = tiny_model("llama");
        let profile = ModelProfile::from_gguf(&model);
        profile.validate().unwrap();
        assert_eq!(profile.architecture.as_deref(), Some("llama"));
        assert_eq!(profile.tensor_count, 11);
        assert_eq!(profile.tensor_formats.len(), 1);
        assert_eq!(profile.tensor_formats[0].format, "F32");
        assert_eq!(profile.tensor_formats[0].tensor_count, 11);
        assert!(profile.tensor_formats[0].runtime_supported);
        assert!(profile.has_coalescible_layer_reads);
        assert!(!profile.has_reusable_grouped_read_candidate);

        let mut relocated = model.clone();
        relocated.path = PathBuf::from("/other/location/tiny.gguf");
        assert_eq!(
            profile.identity.descriptor_fingerprint,
            ModelProfile::from_gguf(&relocated)
                .identity
                .descriptor_fingerprint
        );
    }

    #[test]
    fn test_qwen35_profile_remains_inspect_plan_only() {
        let profile = ModelProfile::from_gguf(&tiny_model("qwen35"));
        assert_eq!(
            profile.execution_compatibility,
            ModelExecutionCompatibility::InspectPlanOnly
        );
    }

    #[test]
    fn test_machine_availability_distinguishes_detected_supported_and_usable() {
        let machine = machine_profile();
        machine.validate().unwrap();
        assert!(machine.gpus[0].backend.detected);
        assert!(!machine.gpus[0].backend.supported);
        assert!(!machine.gpus[0].backend.usable);

        let mut invalid = machine;
        invalid.gpus[0].backend.usable = true;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_capability_derivation_is_independent_from_calibration() {
        let (user, machine, model, storage, static_plan) =
            profiles_and_static_plan(OperatingMode::BalancedNormal);
        let planner = Planner;
        let capabilities = planner.derive_capabilities(&machine, &model, &storage, &static_plan);
        assert!(capabilities.storage_usable);
        assert!(capabilities.cpu_backend_usable);
        assert!(capabilities.cpu_simd_avx2);
        assert!(!capabilities.gpu_backend_usable);
        assert!(capabilities.model_execution_compatible);
        assert!(capabilities.layer_cache_possible);
        assert!(capabilities.read_coalescing_possible);
        assert!(!capabilities.grouped_read_buffer_reuse_possible);
        assert_eq!(
            capabilities.supported_tensor_formats,
            vec!["F32".to_string()]
        );
        assert!(capabilities.unsupported_tensor_formats.is_empty());
        assert_eq!(static_plan.ram_requested, user.ram_budget_bytes);
    }

    #[test]
    fn test_preset_policies_are_explicit_and_bounded() {
        let background = OperatingMode::BackgroundConstrained.policy();
        let balanced = OperatingMode::BalancedNormal.policy();
        let maximum = OperatingMode::MaximumPerformance.policy();
        let advanced = OperatingMode::Advanced.policy();
        assert_eq!(
            (
                background.thread_fraction_numerator,
                background.thread_fraction_denominator
            ),
            (1, 2)
        );
        assert_eq!(background.layer_cache_capacity_basis_points, 0);
        assert!(!background.grouped_read_buffer_reuse);
        assert_eq!(
            (
                balanced.thread_fraction_numerator,
                balanced.thread_fraction_denominator
            ),
            (3, 4)
        );
        assert_eq!(balanced.layer_cache_capacity_basis_points, 5_000);
        assert_eq!(maximum.layer_cache_capacity_basis_points, 10_000);
        assert!(maximum.allow_gpu);
        assert_eq!(
            advanced.layer_cache_capacity_basis_points,
            balanced.layer_cache_capacity_basis_points
        );
    }

    #[test]
    fn test_deterministic_strategy_and_execution_plan_construction() {
        let (user, machine, model, storage, static_plan) =
            profiles_and_static_plan(OperatingMode::BalancedNormal);
        let planner = Planner;
        let capabilities = planner.derive_capabilities(&machine, &model, &storage, &static_plan);
        let first = planner
            .plan(
                &user,
                &machine,
                &model,
                &storage,
                &capabilities,
                &static_plan,
                None,
            )
            .unwrap();
        let second = planner
            .plan(
                &user,
                &machine,
                &model,
                &storage,
                &capabilities,
                &static_plan,
                None,
            )
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.strategy, StrategyId::CpuLayerStreaming);
        assert_eq!(first.ram_budget_bytes, user.ram_budget_bytes);
        assert_eq!(first.cpu_thread_count, 6);
        assert!(first.layer_cache.enabled);
        assert!(first.layer_cache.capacity_bytes <= 6_832);
        assert!(first.io.read_coalescing_enabled);
        assert!(!first.io.grouped_read_buffer_reuse_enabled);
        assert!(!first.gpu.enabled);
        assert_eq!(first.binding.model_path, storage.model_path);
        assert_eq!(first.binding.storage_fingerprint, storage.fingerprint());
        assert!(first.reasons.iter().any(|reason| {
            reason.code == PlanReasonCode::LayerCacheEnabledWithinStaticCapacity
        }));
    }

    #[test]
    fn test_maximum_performance_respects_explicit_resource_limits() {
        let (mut user, machine, model, storage, static_plan) =
            profiles_and_static_plan(OperatingMode::MaximumPerformance);
        user.cpu_thread_limit = Some(2);
        user.gpu_preference = GpuPreference::Disabled;
        let planner = Planner;
        let capabilities = planner.derive_capabilities(&machine, &model, &storage, &static_plan);
        let plan = planner
            .plan(
                &user,
                &machine,
                &model,
                &storage,
                &capabilities,
                &static_plan,
                None,
            )
            .unwrap();
        assert_eq!(plan.ram_budget_bytes, user.ram_budget_bytes);
        assert_eq!(plan.cpu_thread_count, 2);
        assert!(!plan.gpu.enabled);
        assert!(plan.layer_cache.capacity_bytes <= 6_832);
    }

    #[test]
    fn test_feasibility_rejects_unusable_storage() {
        let (user, machine, model, mut storage, static_plan) =
            profiles_and_static_plan(OperatingMode::BalancedNormal);
        storage.seekable = false;
        let planner = Planner;
        let capabilities =
            planner.derive_capabilities(&machine, &model, &storage, &static_plan);
        assert_eq!(
            planner
                .plan(
                    &user,
                    &machine,
                    &model,
                    &storage,
                    &capabilities,
                    &static_plan,
                    None,
                )
                .unwrap_err(),
            PlannerError::Infeasible(FeasibilityRejection::StorageNotUsable)
        );
    }

    #[test]
    fn test_feasibility_rejects_budget_above_available_ram() {
        let (user, mut machine, model, storage, static_plan) =
            profiles_and_static_plan(OperatingMode::BalancedNormal);
        machine.available_ram_bytes = user.ram_budget_bytes - 1;
        let planner = Planner;
        let capabilities = planner.derive_capabilities(&machine, &model, &storage, &static_plan);
        assert_eq!(
            planner
                .plan(
                    &user,
                    &machine,
                    &model,
                    &storage,
                    &capabilities,
                    &static_plan,
                    None,
                )
                .unwrap_err(),
            PlannerError::Infeasible(FeasibilityRejection::RamBudgetExceedsAvailableRam)
        );
    }

    #[test]
    fn test_feasibility_rejects_required_gpu_and_non_executable_model() {
        let (mut user, machine, model, storage, static_plan) =
            profiles_and_static_plan(OperatingMode::BalancedNormal);
        user.gpu_preference = GpuPreference::Require;
        let planner = Planner;
        let capabilities = planner.derive_capabilities(&machine, &model, &storage, &static_plan);
        assert_eq!(
            planner
                .plan(
                    &user,
                    &machine,
                    &model,
                    &storage,
                    &capabilities,
                    &static_plan,
                    None,
                )
                .unwrap_err(),
            PlannerError::Infeasible(FeasibilityRejection::GpuRequiredButUnavailable)
        );

        let unsupported_model = tiny_model("qwen35");
        let unsupported_profile = ModelProfile::from_gguf(&unsupported_model);
        let unsupported_plan =
            crate::plan::plan_model(&unsupported_model, user.ram_budget_bytes).unwrap();
        let unsupported_capabilities =
            planner.derive_capabilities(
                &machine,
                &unsupported_profile,
                &storage,
                &unsupported_plan,
            );
        user.gpu_preference = GpuPreference::Disabled;
        assert_eq!(
            planner
                .plan(
                    &user,
                    &machine,
                    &unsupported_profile,
                    &storage,
                    &unsupported_capabilities,
                    &unsupported_plan,
                    None,
                )
                .unwrap_err(),
            PlannerError::Infeasible(FeasibilityRejection::ModelNotExecutable)
        );
    }

    #[test]
    fn test_advanced_overrides_preserve_explicit_user_constraints() {
        let (mut user, machine, model, storage, static_plan) =
            profiles_and_static_plan(OperatingMode::Advanced);
        user.cpu_thread_limit = Some(6);
        user.advanced.thread_count = Some(3);
        user.advanced.layer_cache_enabled = Some(true);
        user.advanced.layer_cache_capacity_bytes = Some(3_000);
        user.advanced.read_coalescing_enabled = Some(false);
        user.advanced.grouped_read_buffer_reuse_enabled = Some(false);
        let planner = Planner;
        let capabilities = planner.derive_capabilities(&machine, &model, &storage, &static_plan);
        let plan = planner
            .plan(
                &user,
                &machine,
                &model,
                &storage,
                &capabilities,
                &static_plan,
                None,
            )
            .unwrap();
        assert_eq!(plan.cpu_thread_count, 3);
        assert_eq!(plan.layer_cache.capacity_bytes, 3_000);
        assert!(!plan.io.read_coalescing_enabled);
        assert!(!plan.io.grouped_read_buffer_reuse_enabled);
    }

    #[test]
    fn test_calibration_observations_are_measured_facts_not_capabilities() {
        let (mut user, machine, model, storage, static_plan) =
            profiles_and_static_plan(OperatingMode::BalancedNormal);
        user.calibration_level = CalibrationLevel::Standard;
        let planner = Planner;
        let capabilities = planner.derive_capabilities(&machine, &model, &storage, &static_plan);
        let capabilities_before = capabilities.clone();
        let calibration = CalibrationResult {
            schema_version: PROFILE_SCHEMA_VERSION,
            identifier: "calibration-1".to_string(),
            level: CalibrationLevel::Standard,
            machine_fingerprint: machine.fingerprint(),
            model_fingerprint: Some(model.identity.descriptor_fingerprint),
            storage_fingerprint: Some(storage.fingerprint()),
            calibration_plan_identifier: "calibration-plan-1".to_string(),
            calibration_plan_version: CALIBRATION_PLAN_VERSION,
            calibration_ruleset_version: CALIBRATION_RULESET_VERSION,
            observations: vec![
                PerformanceObservation {
                    identifier: "sequential-read".to_string(),
                    metric: ObservationMetric::SequentialReadThroughput,
                    unit: ObservationUnit::BytesPerSecond,
                    value: Some(1_000_000),
                    status: ObservationStatus::Measured,
                    status_reason: None,
                    strategy: None,
                    quantization_format: None,
                    calibration_level: CalibrationLevel::Standard,
                    workload_bytes: 8_000_000,
                    workload_elements: 0,
                    measurement_duration_ns: 8_000_000_000,
                    sample_count: 4,
                    raw_samples: vec![
                        MeasurementSample {
                            elapsed_ns: 2_000_000_000,
                            units_processed: 2_000_000,
                        };
                        4
                    ],
                    timestamp_unix_seconds: Some(1_700_000_000),
                    quality_basis_points: 10_000,
                },
                PerformanceObservation {
                    identifier: "strategy-latency".to_string(),
                    metric: ObservationMetric::StrategyLatency,
                    unit: ObservationUnit::Nanoseconds,
                    value: Some(5_000_000),
                    status: ObservationStatus::Measured,
                    status_reason: None,
                    strategy: Some(StrategyId::CpuLayerStreaming),
                    quantization_format: None,
                    calibration_level: CalibrationLevel::Standard,
                    workload_bytes: 0,
                    workload_elements: 4,
                    measurement_duration_ns: 20_000_000,
                    sample_count: 4,
                    raw_samples: vec![
                        MeasurementSample {
                            elapsed_ns: 5_000_000,
                            units_processed: 1,
                        };
                        4
                    ],
                    timestamp_unix_seconds: Some(1_700_000_001),
                    quality_basis_points: 10_000,
                },
            ],
        };
        calibration.validate().unwrap();
        let uncalibrated = planner
            .plan(
                &user,
                &machine,
                &model,
                &storage,
                &capabilities,
                &static_plan,
                None,
            )
            .unwrap();
        let plan = planner
            .plan(
                &user,
                &machine,
                &model,
                &storage,
                &capabilities,
                &static_plan,
                Some(&calibration),
            )
            .unwrap();
        let repeated = planner
            .plan(
                &user,
                &machine,
                &model,
                &storage,
                &capabilities,
                &static_plan,
                Some(&calibration),
            )
            .unwrap();
        assert_eq!(plan, repeated);
        assert_eq!(capabilities, capabilities_before);
        assert_eq!(plan.strategy, uncalibrated.strategy);
        assert_eq!(plan.cpu_thread_count, uncalibrated.cpu_thread_count);
        assert_eq!(plan.layer_cache, uncalibrated.layer_cache);
        assert_eq!(plan.io, uncalibrated.io);
        assert_eq!(plan.gpu, uncalibrated.gpu);
        assert_eq!(plan.reservations, uncalibrated.reservations);
        assert_eq!(plan.strategy, StrategyId::CpuLayerStreaming);
        assert_eq!(
            plan.cost.calibrated_read_time_ns_per_forward,
            Some(2_624_000)
        );
        assert_eq!(plan.cost.observed_strategy_latency_ns, Some(5_000_000));
        assert_eq!(
            plan.calibration.calibration_plan_identifier.as_deref(),
            Some("calibration-plan-1")
        );
        assert_eq!(
            plan.calibration.observation_identifiers,
            vec![
                "sequential-read".to_string(),
                "strategy-latency".to_string()
            ]
        );
    }

    #[test]
    fn test_non_advanced_profile_rejects_advanced_override() {
        let mut user = UserProfile::new(10_000, OperatingMode::BalancedNormal);
        user.advanced.read_coalescing_enabled = Some(false);
        assert_eq!(
            user.validate().unwrap_err(),
            PlannerError::Infeasible(
                FeasibilityRejection::AdvancedOverrideRequiresAdvancedMode
            )
        );
    }

    #[test]
    fn test_observation_aggregation_and_quality_are_mathematical() {
        let samples = [
            MeasurementSample {
                elapsed_ns: 100,
                units_processed: 100,
            },
            MeasurementSample {
                elapsed_ns: 200,
                units_processed: 100,
            },
        ];
        assert_eq!(observation_quality_basis_points(&samples), 5_000);
        assert_eq!(
            aggregate_observation_value(ObservationUnit::ElementsPerSecond, &samples),
            Some(666_666_666)
        );
    }

    #[test]
    fn test_plan_compatibility_states_are_explicit() {
        assert_ne!(
            PlanCompatibility::Valid,
            PlanCompatibility::CompatibleRecalibrationRecommended
        );
        assert_ne!(
            PlanCompatibility::CompatibleRecalibrationRecommended,
            PlanCompatibility::Incompatible
        );
    }
}
