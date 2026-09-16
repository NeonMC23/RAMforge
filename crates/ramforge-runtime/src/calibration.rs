//! Explicit, bounded local calibration plans and measurements.
//!
//! Calibration records measured facts. It does not select execution strategies,
//! mutate capabilities, change policies, or modify Planner rules.

use std::collections::BTreeSet;
use std::fmt;
use std::fs::File;
use std::hint::black_box;
use std::io::{Read, Seek, SeekFrom};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use ramforge_core::memory::MemoryBudget;
use ramforge_core::quant::{
    dequantize_row_q4_0, dequantize_row_q4_k, dequantize_row_q8_0, BLOCK_SIZE_Q4_0,
    BLOCK_SIZE_Q4_K, BLOCK_SIZE_Q8_0, QK4_0, QK8_0, QK_K,
};
use ramforge_core::MemoryError;

pub use crate::planner::{CALIBRATION_PLAN_VERSION, CALIBRATION_RULESET_VERSION};
use crate::planner::{
    aggregate_observation_value, observation_quality_basis_points, CalibrationLevel,
    CalibrationResult, CapabilitySet, MachineProfile, MeasurementSample, ModelProfile,
    ObservationMetric, ObservationStatus, ObservationStatusReason, ObservationUnit,
    PerformanceObservation, PlannerError, StorageProfile, StrategyId, UserProfile,
    PROFILE_SCHEMA_VERSION,
};
use crate::simd::dot_f32_scalar;

const MAX_ABSOLUTE_MEMORY_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ABSOLUTE_STORAGE_READ_BYTES: u64 = 16 * 1024 * 1024;
const CALIBRATION_METADATA_BYTES: u64 = 4 * 1024;
const MAX_ABSOLUTE_TOTAL_DURATION_NS: u64 = 120_000_000_000;
const MAX_ABSOLUTE_TASK_DURATION_NS: u64 = 30_000_000_000;
const MAX_ABSOLUTE_SAMPLES: u32 = 16;
const MAX_ABSOLUTE_ITERATIONS: u32 = 1_024;
const MAX_ABSOLUTE_TASK_BYTES_PROCESSED: u64 = 1024 * 1024 * 1024;
const MAX_ABSOLUTE_TASK_ELEMENTS_PROCESSED: u64 = 100_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationQuantization {
    Q4_0,
    Q8_0,
    Q4K,
}

impl CalibrationQuantization {
    pub const fn format_name(self) -> &'static str {
        match self {
            Self::Q4_0 => "Q4_0",
            Self::Q8_0 => "Q8_0",
            Self::Q4K => "Q4_K",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationTestKind {
    CpuFloatDot,
    MemoryCopyBandwidth,
    SequentialStorageRead,
    StorageReadLatency,
    QuantizedRowDecode(CalibrationQuantization),
    StrategyLatency(StrategyId),
}

impl CalibrationTestKind {
    pub fn stable_identifier(self) -> String {
        match self {
            Self::CpuFloatDot => "cpu.float-dot.v1".to_string(),
            Self::MemoryCopyBandwidth => "memory.copy-bandwidth.v1".to_string(),
            Self::SequentialStorageRead => "storage.sequential-read.v1".to_string(),
            Self::StorageReadLatency => "storage.read-latency.v1".to_string(),
            Self::QuantizedRowDecode(format) => {
                format!("decode.{}.v1", format.format_name().to_ascii_lowercase())
            }
            Self::StrategyLatency(StrategyId::CpuLayerStreaming) => {
                "strategy.cpu-layer-streaming.v1".to_string()
            }
            Self::StrategyLatency(StrategyId::GpuLayerOffload) => {
                "strategy.gpu-layer-offload.v1".to_string()
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationPurpose {
    CpuRuntimeBaseline,
    MemoryBandwidth,
    StorageThroughput,
    StorageLatency,
    QuantizedDecodeThroughput,
    StrategyComparison,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationRequirement {
    CpuBackend,
    SeekableStorage,
    ModelExecution,
    QuantizationFormat(String),
    Strategy(StrategyId),
}

/// Bounded calibration working set. Returned observation metadata is
/// caller-owned, like other API results, and is not retained by the runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpectedResourceUsage {
    pub memory_bytes: u64,
    /// Maximum storage bytes touched by one bounded measurement operation.
    pub storage_read_bytes: u64,
    pub cpu_threads: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalibrationWorkBudget {
    pub warmup_iterations: u32,
    pub iterations_per_sample: u32,
    pub sample_count: u32,
    pub elements_per_iteration: u64,
    pub bytes_per_iteration: u64,
    pub max_duration_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationTask {
    pub identifier: String,
    pub kind: CalibrationTestKind,
    pub purpose: CalibrationPurpose,
    pub requirements: Vec<CalibrationRequirement>,
    pub expected_resources: ExpectedResourceUsage,
    pub model_fingerprint: Option<u64>,
    pub storage_fingerprint: Option<u64>,
    pub work: CalibrationWorkBudget,
}

impl CalibrationTask {
    pub fn validate(&self) -> Result<(), CalibrationError> {
        if self.identifier != self.kind.stable_identifier() {
            return Err(CalibrationError::InvalidPlan("unstable task identifier"));
        }
        if self.expected_resources.cpu_threads != 1
            || self.expected_resources.memory_bytes == 0
            || self.work.iterations_per_sample == 0
            || self.work.sample_count == 0
            || self.work.max_duration_ns == 0
            || (self.work.elements_per_iteration == 0 && self.work.bytes_per_iteration == 0)
            || self.work.warmup_iterations > MAX_ABSOLUTE_ITERATIONS
            || self.work.sample_count > MAX_ABSOLUTE_SAMPLES
            || self.work.iterations_per_sample > MAX_ABSOLUTE_ITERATIONS
            || self.expected_resources.memory_bytes > MAX_ABSOLUTE_MEMORY_BYTES
            || self.expected_resources.storage_read_bytes > MAX_ABSOLUTE_STORAGE_READ_BYTES
            || self.work.max_duration_ns > MAX_ABSOLUTE_TASK_DURATION_NS
            || !work_volume_is_bounded(self.work)
        {
            return Err(CalibrationError::InvalidPlan("unbounded task resources"));
        }
        match self.kind {
            CalibrationTestKind::SequentialStorageRead | CalibrationTestKind::StorageReadLatency
                if self.storage_fingerprint.is_none() =>
            {
                return Err(CalibrationError::InvalidPlan(
                    "storage measurement lacks storage dependency",
                ))
            }
            CalibrationTestKind::QuantizedRowDecode(_)
            | CalibrationTestKind::StrategyLatency(_)
                if self.model_fingerprint.is_none() =>
            {
                return Err(CalibrationError::InvalidPlan(
                    "model measurement lacks model dependency",
                ))
            }
            _ => {}
        }
        Ok(())
    }
}

fn work_volume_is_bounded(work: CalibrationWorkBudget) -> bool {
    let measured_iterations = (work.iterations_per_sample as u64)
        .checked_mul(work.sample_count as u64);
    let total_iterations = measured_iterations
        .and_then(|iterations| iterations.checked_add(work.warmup_iterations as u64));
    let Some(total_iterations) = total_iterations else {
        return false;
    };
    let bytes = work.bytes_per_iteration.checked_mul(total_iterations);
    let elements = work.elements_per_iteration.checked_mul(total_iterations);
    bytes.is_some_and(|value| value <= MAX_ABSOLUTE_TASK_BYTES_PROCESSED)
        && elements.is_some_and(|value| value <= MAX_ABSOLUTE_TASK_ELEMENTS_PROCESSED)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalibrationLimits {
    pub max_memory_bytes: u64,
    pub max_storage_read_bytes: u64,
    pub max_total_duration_ns: u64,
    pub max_task_duration_ns: u64,
    pub max_samples: u32,
    pub max_iterations_per_sample: u32,
}

impl CalibrationLimits {
    pub fn conservative(user: &UserProfile) -> Self {
        let memory = (user.ram_budget_bytes / 32)
            .max(64 * 1024)
            .min(8 * 1024 * 1024)
            .min(user.ram_budget_bytes);
        Self {
            max_memory_bytes: memory,
            max_storage_read_bytes: 1024 * 1024,
            max_total_duration_ns: 15_000_000_000,
            max_task_duration_ns: 3_000_000_000,
            max_samples: 7,
            max_iterations_per_sample: 64,
        }
    }

    pub fn validate(&self, user: &UserProfile) -> Result<(), CalibrationError> {
        if self.max_memory_bytes == 0
            || self.max_memory_bytes > user.ram_budget_bytes
            || self.max_memory_bytes > MAX_ABSOLUTE_MEMORY_BYTES
            || self.max_storage_read_bytes == 0
            || self.max_storage_read_bytes > MAX_ABSOLUTE_STORAGE_READ_BYTES
            || self.max_total_duration_ns == 0
            || self.max_total_duration_ns > MAX_ABSOLUTE_TOTAL_DURATION_NS
            || self.max_task_duration_ns == 0
            || self.max_task_duration_ns > self.max_total_duration_ns
            || self.max_task_duration_ns > MAX_ABSOLUTE_TASK_DURATION_NS
            || self.max_samples == 0
            || self.max_samples > MAX_ABSOLUTE_SAMPLES
            || self.max_iterations_per_sample == 0
            || self.max_iterations_per_sample > MAX_ABSOLUTE_ITERATIONS
        {
            return Err(CalibrationError::InvalidPlan("invalid calibration limits"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationPlan {
    pub schema_version: u32,
    pub plan_version: u32,
    pub ruleset_version: u32,
    pub identifier: String,
    pub level: CalibrationLevel,
    pub machine_fingerprint: u64,
    pub model_fingerprint: u64,
    pub storage_fingerprint: u64,
    pub limits: CalibrationLimits,
    pub tasks: Vec<CalibrationTask>,
}

impl CalibrationPlan {
    pub fn build(
        user: &UserProfile,
        machine: &MachineProfile,
        model: &ModelProfile,
        storage: &StorageProfile,
        capabilities: &CapabilitySet,
        limits: CalibrationLimits,
    ) -> Result<Self, CalibrationError> {
        user.validate()?;
        machine.validate()?;
        model.validate()?;
        storage.validate()?;
        limits.validate(user)?;
        if storage
            .file_size_bytes
            .is_some_and(|bytes| bytes != model.file_size_bytes)
        {
            return Err(CalibrationError::InvalidPlan(
                "storage file size does not match model profile",
            ));
        }
        let available_ram_bytes = machine.available_ram_bytes.ok_or(
            CalibrationError::InvalidPlan("available machine RAM is unknown"),
        )?;
        if limits.max_memory_bytes > available_ram_bytes {
            return Err(CalibrationError::InvalidPlan(
                "calibration memory exceeds available machine RAM",
            ));
        }
        let machine_fingerprint = machine.fingerprint();
        let model_fingerprint = model.identity.descriptor_fingerprint;
        let storage_fingerprint = storage.fingerprint();
        if capabilities.schema_version != PROFILE_SCHEMA_VERSION
            || capabilities.machine_fingerprint != machine_fingerprint
            || capabilities.model_fingerprint != model_fingerprint
            || capabilities.storage_fingerprint != storage_fingerprint
        {
            return Err(CalibrationError::InvalidPlan(
                "capabilities do not match profiles",
            ));
        }

        let mut tasks = Vec::new();
        if user.calibration_level != CalibrationLevel::None {
            tasks.push(cpu_task(user.calibration_level, limits));
            tasks.push(storage_task(user.calibration_level, storage_fingerprint, limits));
        }
        if level_at_least(user.calibration_level, CalibrationLevel::Standard) {
            tasks.push(memory_task(user.calibration_level, limits));
            let relevant = relevant_quantizations(model);
            if let Some(format) = relevant.first() {
                tasks.push(quantized_task(
                    user.calibration_level,
                    *format,
                    model_fingerprint,
                    limits,
                ));
            }
        }
        if user.calibration_level == CalibrationLevel::Thorough {
            for format in relevant_quantizations(model).into_iter().skip(1).take(2) {
                tasks.push(quantized_task(
                    user.calibration_level,
                    format,
                    model_fingerprint,
                    limits,
                ));
            }
        }

        let mut plan = Self {
            schema_version: PROFILE_SCHEMA_VERSION,
            plan_version: CALIBRATION_PLAN_VERSION,
            ruleset_version: CALIBRATION_RULESET_VERSION,
            identifier: String::new(),
            level: user.calibration_level,
            machine_fingerprint,
            model_fingerprint,
            storage_fingerprint,
            limits,
            tasks,
        };
        plan.identifier = calibration_plan_identifier(&plan);
        plan.validate()?;
        Ok(plan)
    }

    pub fn request_strategy_measurement(
        &mut self,
        strategy: StrategyId,
    ) -> Result<(), CalibrationError> {
        if self.level != CalibrationLevel::Thorough {
            return Err(CalibrationError::InvalidPlan(
                "strategy measurements require Thorough calibration",
            ));
        }
        let task = strategy_task(
            self.level,
            strategy,
            self.model_fingerprint,
            self.limits,
        );
        if self
            .tasks
            .iter()
            .any(|existing| existing.identifier.as_str() == task.identifier.as_str())
        {
            return Err(CalibrationError::InvalidPlan(
                "duplicate strategy measurement",
            ));
        }
        self.tasks.push(task);
        self.identifier = calibration_plan_identifier(self);
        self.validate()
    }

    pub fn validate(&self) -> Result<(), CalibrationError> {
        if self.schema_version != PROFILE_SCHEMA_VERSION
            || self.plan_version != CALIBRATION_PLAN_VERSION
            || self.ruleset_version != CALIBRATION_RULESET_VERSION
            || self.identifier.is_empty()
            || self.machine_fingerprint == 0
            || self.model_fingerprint == 0
            || self.storage_fingerprint == 0
        {
            return Err(CalibrationError::InvalidPlan("invalid calibration identity"));
        }
        if self.level == CalibrationLevel::None && !self.tasks.is_empty() {
            return Err(CalibrationError::InvalidPlan(
                "none-level plan contains tasks",
            ));
        }
        let mut identifiers = BTreeSet::new();
        for task in &self.tasks {
            task.validate()?;
            if !identifiers.insert(task.identifier.as_str()) {
                return Err(CalibrationError::InvalidPlan("duplicate calibration task"));
            }
        }
        if self.identifier != calibration_plan_identifier(self) {
            return Err(CalibrationError::InvalidPlan(
                "calibration plan identifier mismatch",
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum CalibrationError {
    InvalidPlan(&'static str),
    Planner(PlannerError),
    Budget(MemoryError),
}

impl From<PlannerError> for CalibrationError {
    fn from(error: PlannerError) -> Self {
        Self::Planner(error)
    }
}

impl From<MemoryError> for CalibrationError {
    fn from(error: MemoryError) -> Self {
        Self::Budget(error)
    }
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlan(reason) => write!(formatter, "invalid calibration plan: {reason}"),
            Self::Planner(error) => write!(formatter, "invalid calibration profile: {error}"),
            Self::Budget(error) => write!(formatter, "calibration budget error: {error}"),
        }
    }
}

impl std::error::Error for CalibrationError {}

#[derive(Debug, Default, Clone, Copy)]
pub struct CalibrationRunner;

impl CalibrationRunner {
    pub fn run(
        &self,
        plan: &CalibrationPlan,
        user: &UserProfile,
        machine: &MachineProfile,
        model: &ModelProfile,
        storage: &StorageProfile,
        capabilities: &CapabilitySet,
    ) -> Result<CalibrationResult, CalibrationError> {
        plan.validate()?;
        user.validate()?;
        machine.validate()?;
        model.validate()?;
        storage.validate()?;
        plan.limits.validate(user)?;
        let available_ram_bytes = machine.available_ram_bytes.ok_or(
            CalibrationError::InvalidPlan("current available RAM is unknown"),
        )?;
        if plan.limits.max_memory_bytes > available_ram_bytes {
            return Err(CalibrationError::InvalidPlan(
                "calibration memory exceeds current available RAM",
            ));
        }
        if plan.level == CalibrationLevel::None {
            return Err(CalibrationError::InvalidPlan(
                "none-level calibration cannot be executed",
            ));
        }
        if plan.level != user.calibration_level
            || plan.machine_fingerprint != machine.fingerprint()
            || plan.model_fingerprint != model.identity.descriptor_fingerprint
            || plan.storage_fingerprint != storage.fingerprint()
            || capabilities.schema_version != PROFILE_SCHEMA_VERSION
            || capabilities.machine_fingerprint != plan.machine_fingerprint
            || capabilities.model_fingerprint != plan.model_fingerprint
            || capabilities.storage_fingerprint != plan.storage_fingerprint
        {
            return Err(CalibrationError::InvalidPlan(
                "calibration inputs do not match the plan",
            ));
        }

        let mut budget = MemoryBudget::new(user.ram_budget_bytes)?;
        let calibration_started = Instant::now();
        let mut observations = Vec::with_capacity(plan.tasks.len());
        for task in &plan.tasks {
            let elapsed_ns = duration_ns(&calibration_started);
            let remaining_ns = plan.limits.max_total_duration_ns.saturating_sub(elapsed_ns);
            if remaining_ns < task.work.max_duration_ns {
                observations.push(non_measured_observation(
                    task,
                    plan.level,
                    ObservationStatus::Skipped,
                    ObservationStatusReason::TimeLimitReached,
                ));
                continue;
            }
            if task
                .model_fingerprint
                .is_some_and(|fingerprint| fingerprint != plan.model_fingerprint)
                || task
                    .storage_fingerprint
                    .is_some_and(|fingerprint| fingerprint != plan.storage_fingerprint)
            {
                observations.push(non_measured_observation(
                    task,
                    plan.level,
                    ObservationStatus::Unavailable,
                    ObservationStatusReason::DependencyUnavailable,
                ));
                continue;
            }
            if let Some(reason) = unavailable_requirement(task, capabilities) {
                observations.push(non_measured_observation(
                    task,
                    plan.level,
                    ObservationStatus::Unavailable,
                    reason,
                ));
                continue;
            }
            if task.expected_resources.memory_bytes > plan.limits.max_memory_bytes
                || task.expected_resources.storage_read_bytes
                    > plan.limits.max_storage_read_bytes
                || task.work.sample_count > plan.limits.max_samples
                || task.work.warmup_iterations > plan.limits.max_iterations_per_sample
                || task.work.iterations_per_sample > plan.limits.max_iterations_per_sample
                || task.work.max_duration_ns > plan.limits.max_task_duration_ns
            {
                observations.push(non_measured_observation(
                    task,
                    plan.level,
                    ObservationStatus::Skipped,
                    ObservationStatusReason::ResourceLimitExceeded,
                ));
                continue;
            }

            let charge_name = format!("calibration:{}", task.identifier);
            let charge_bytes = task.expected_resources.memory_bytes.max(1);
            let observation = budget.with_temp(
                &charge_name,
                charge_bytes,
                |_budget| Ok::<_, CalibrationError>(execute_task(task, plan.level, storage)),
            )?;
            observations.push(observation);
        }
        debug_assert_eq!(budget.used_bytes(), 0);
        let timestamp_unix_seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .map(|duration| duration.as_secs());
        for observation in &mut observations {
            observation.timestamp_unix_seconds = timestamp_unix_seconds;
        }

        let result = CalibrationResult {
            schema_version: PROFILE_SCHEMA_VERSION,
            identifier: format!("calibration-result:{}", plan.identifier),
            level: plan.level,
            machine_fingerprint: plan.machine_fingerprint,
            model_fingerprint: Some(plan.model_fingerprint),
            storage_fingerprint: Some(plan.storage_fingerprint),
            calibration_plan_identifier: plan.identifier.clone(),
            calibration_plan_version: plan.plan_version,
            calibration_ruleset_version: plan.ruleset_version,
            observations,
        };
        result.validate()?;
        Ok(result)
    }
}

fn execute_task(
    task: &CalibrationTask,
    level: CalibrationLevel,
    storage: &StorageProfile,
) -> PerformanceObservation {
    let measured = match task.kind {
        CalibrationTestKind::CpuFloatDot => measure_cpu_float(task),
        CalibrationTestKind::MemoryCopyBandwidth => measure_memory_copy(task),
        CalibrationTestKind::SequentialStorageRead => measure_storage(task, storage),
        CalibrationTestKind::QuantizedRowDecode(format) => measure_quantized(task, format),
        CalibrationTestKind::StorageReadLatency | CalibrationTestKind::StrategyLatency(_) => {
            return non_measured_observation(
                task,
                level,
                ObservationStatus::Skipped,
                ObservationStatusReason::TestNotImplemented,
            )
        }
    };
    match measured {
        Ok(samples) => measured_observation(task, level, samples),
        Err(reason @ ObservationStatusReason::DependencyUnavailable) => non_measured_observation(
            task,
            level,
            ObservationStatus::Unavailable,
            reason,
        ),
        Err(reason @ ObservationStatusReason::TimeLimitReached) => non_measured_observation(
            task,
            level,
            ObservationStatus::Skipped,
            reason,
        ),
        Err(reason) => non_measured_observation(
            task,
            level,
            ObservationStatus::Failed,
            reason,
        ),
    }
}

fn measure_cpu_float(
    task: &CalibrationTask,
) -> Result<Vec<MeasurementSample>, ObservationStatusReason> {
    let elements = usize::try_from(task.work.elements_per_iteration)
        .map_err(|_| ObservationStatusReason::InvalidWorkload)?;
    if elements == 0 {
        return Err(ObservationStatusReason::InvalidWorkload);
    }
    let left: Vec<f32> = (0..elements)
        .map(|index| (index % 97) as f32 * 0.03125 - 1.0)
        .collect();
    let right: Vec<f32> = (0..elements)
        .map(|index| (index % 89) as f32 * 0.015625 + 0.25)
        .collect();
    measure_repeated(task, task.work.elements_per_iteration, || {
        black_box(dot_f32_scalar(black_box(&left), black_box(&right)));
        Ok(())
    })
}

fn measure_memory_copy(
    task: &CalibrationTask,
) -> Result<Vec<MeasurementSample>, ObservationStatusReason> {
    let bytes = usize::try_from(task.work.bytes_per_iteration)
        .map_err(|_| ObservationStatusReason::InvalidWorkload)?;
    if bytes == 0 {
        return Err(ObservationStatusReason::InvalidWorkload);
    }
    let source: Vec<u8> = (0..bytes).map(|index| (index as u8).wrapping_mul(31)).collect();
    let mut destination = vec![0u8; bytes];
    measure_repeated(task, task.work.bytes_per_iteration, || {
        destination.copy_from_slice(black_box(&source));
        black_box(&destination);
        Ok(())
    })
}

fn measure_storage(
    task: &CalibrationTask,
    storage: &StorageProfile,
) -> Result<Vec<MeasurementSample>, ObservationStatusReason> {
    let mut file = match File::open(&storage.model_path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(ObservationStatusReason::DependencyUnavailable)
        }
        Err(_) => return Err(ObservationStatusReason::IoFailure),
    };
    let file_bytes = file
        .metadata()
        .map_err(|_| ObservationStatusReason::IoFailure)?
        .len();
    if storage
        .file_size_bytes
        .is_some_and(|expected| expected != file_bytes)
    {
        return Err(ObservationStatusReason::DependencyUnavailable);
    }
    let bytes_to_read = task.work.bytes_per_iteration.min(file_bytes);
    let length = usize::try_from(bytes_to_read)
        .map_err(|_| ObservationStatusReason::InvalidWorkload)?;
    if length == 0 {
        return Err(ObservationStatusReason::DependencyUnavailable);
    }
    let mut buffer = vec![0u8; length];
    measure_repeated(task, bytes_to_read, || {
        file.seek(SeekFrom::Start(0))
            .map_err(|_| ObservationStatusReason::IoFailure)?;
        file.read_exact(&mut buffer)
            .map_err(|_| ObservationStatusReason::IoFailure)?;
        black_box(&buffer);
        Ok(())
    })
}

fn measure_quantized(
    task: &CalibrationTask,
    format: CalibrationQuantization,
) -> Result<Vec<MeasurementSample>, ObservationStatusReason> {
    let elements = usize::try_from(task.work.elements_per_iteration)
        .map_err(|_| ObservationStatusReason::InvalidWorkload)?;
    let encoded = make_quantized_row(format, elements)
        .ok_or(ObservationStatusReason::InvalidWorkload)?;
    let mut output = vec![0.0f32; elements];
    measure_repeated(task, task.work.elements_per_iteration, || {
        let result = match format {
            CalibrationQuantization::Q4_0 => {
                dequantize_row_q4_0(black_box(&encoded), elements, &mut output)
            }
            CalibrationQuantization::Q8_0 => {
                dequantize_row_q8_0(black_box(&encoded), elements, &mut output)
            }
            CalibrationQuantization::Q4K => {
                dequantize_row_q4_k(black_box(&encoded), elements, &mut output)
            }
        };
        result.map_err(|_| ObservationStatusReason::InvalidWorkload)?;
        black_box(&output);
        Ok(())
    })
}

fn measure_repeated(
    task: &CalibrationTask,
    units_per_iteration: u64,
    mut operation: impl FnMut() -> Result<(), ObservationStatusReason>,
) -> Result<Vec<MeasurementSample>, ObservationStatusReason> {
    let task_started = Instant::now();
    for _ in 0..task.work.warmup_iterations {
        operation()?;
        if duration_ns(&task_started) >= task.work.max_duration_ns {
            return Err(ObservationStatusReason::TimeLimitReached);
        }
    }

    let units_per_sample = units_per_iteration
        .checked_mul(task.work.iterations_per_sample as u64)
        .ok_or(ObservationStatusReason::InvalidWorkload)?;
    let mut samples = Vec::with_capacity(task.work.sample_count as usize);
    for _ in 0..task.work.sample_count {
        let sample_started = Instant::now();
        for _ in 0..task.work.iterations_per_sample {
            operation()?;
        }
        let elapsed_ns = duration_ns(&sample_started);
        if elapsed_ns == 0 {
            return Err(ObservationStatusReason::InvalidWorkload);
        }
        samples.push(MeasurementSample {
            elapsed_ns,
            units_processed: units_per_sample,
        });
        if duration_ns(&task_started) >= task.work.max_duration_ns {
            break;
        }
    }
    if samples.is_empty() {
        Err(ObservationStatusReason::TimeLimitReached)
    } else {
        Ok(samples)
    }
}

fn measured_observation(
    task: &CalibrationTask,
    level: CalibrationLevel,
    samples: Vec<MeasurementSample>,
) -> PerformanceObservation {
    let measurement_duration_ns = samples
        .iter()
        .fold(0u64, |total, sample| total.saturating_add(sample.elapsed_ns));
    let total_units = samples
        .iter()
        .fold(0u64, |total, sample| total.saturating_add(sample.units_processed));
    let (metric, unit, strategy, quantization_format) = observation_shape(task.kind);
    let value = aggregate_observation_value(unit, &samples);
    let (workload_bytes, workload_elements) = match unit {
        ObservationUnit::BytesPerSecond => (total_units, 0),
        ObservationUnit::ElementsPerSecond => (0, total_units),
        ObservationUnit::Nanoseconds => (task.work.bytes_per_iteration, total_units),
    };
    let Some(value) = value else {
        return non_measured_observation(
            task,
            level,
            ObservationStatus::Failed,
            ObservationStatusReason::InvalidWorkload,
        );
    };
    let quality_basis_points = observation_quality_basis_points(&samples);
    PerformanceObservation {
        identifier: task.identifier.clone(),
        metric,
        unit,
        value: Some(value),
        status: ObservationStatus::Measured,
        status_reason: None,
        strategy,
        quantization_format,
        calibration_level: level,
        workload_bytes,
        workload_elements,
        measurement_duration_ns,
        sample_count: samples.len() as u32,
        raw_samples: samples,
        timestamp_unix_seconds: None,
        quality_basis_points,
    }
}

fn non_measured_observation(
    task: &CalibrationTask,
    level: CalibrationLevel,
    status: ObservationStatus,
    reason: ObservationStatusReason,
) -> PerformanceObservation {
    let (metric, unit, strategy, quantization_format) = observation_shape(task.kind);
    PerformanceObservation {
        identifier: task.identifier.clone(),
        metric,
        unit,
        value: None,
        status,
        status_reason: Some(reason),
        strategy,
        quantization_format,
        calibration_level: level,
        workload_bytes: task.work.bytes_per_iteration,
        workload_elements: task.work.elements_per_iteration,
        measurement_duration_ns: 0,
        sample_count: 0,
        raw_samples: Vec::new(),
        timestamp_unix_seconds: None,
        quality_basis_points: 0,
    }
}

fn observation_shape(
    kind: CalibrationTestKind,
) -> (
    ObservationMetric,
    ObservationUnit,
    Option<StrategyId>,
    Option<String>,
) {
    match kind {
        CalibrationTestKind::CpuFloatDot => (
            ObservationMetric::CpuFloatThroughput,
            ObservationUnit::ElementsPerSecond,
            None,
            None,
        ),
        CalibrationTestKind::MemoryCopyBandwidth => (
            ObservationMetric::MemoryBandwidth,
            ObservationUnit::BytesPerSecond,
            None,
            None,
        ),
        CalibrationTestKind::SequentialStorageRead => (
            ObservationMetric::SequentialReadThroughput,
            ObservationUnit::BytesPerSecond,
            None,
            None,
        ),
        CalibrationTestKind::StorageReadLatency => (
            ObservationMetric::RandomReadLatency,
            ObservationUnit::Nanoseconds,
            None,
            None,
        ),
        CalibrationTestKind::QuantizedRowDecode(format) => (
            ObservationMetric::QuantizedDecodeThroughput,
            ObservationUnit::ElementsPerSecond,
            None,
            Some(format.format_name().to_string()),
        ),
        CalibrationTestKind::StrategyLatency(strategy) => (
            ObservationMetric::StrategyLatency,
            ObservationUnit::Nanoseconds,
            Some(strategy),
            None,
        ),
    }
}

fn unavailable_requirement(
    task: &CalibrationTask,
    capabilities: &CapabilitySet,
) -> Option<ObservationStatusReason> {
    for requirement in &task.requirements {
        let available = match requirement {
            CalibrationRequirement::CpuBackend => capabilities.cpu_backend_usable,
            CalibrationRequirement::SeekableStorage => capabilities.storage_usable,
            CalibrationRequirement::ModelExecution => capabilities.model_execution_compatible,
            CalibrationRequirement::QuantizationFormat(format) => capabilities
                .supported_tensor_formats
                .iter()
                .any(|supported| supported == format),
            CalibrationRequirement::Strategy(StrategyId::CpuLayerStreaming) => {
                capabilities.cpu_backend_usable && capabilities.model_execution_compatible
            }
            CalibrationRequirement::Strategy(StrategyId::GpuLayerOffload) => {
                capabilities.gpu_backend_usable && capabilities.model_execution_compatible
            }
        };
        if !available {
            return Some(ObservationStatusReason::CapabilityUnavailable);
        }
    }
    None
}

fn cpu_task(level: CalibrationLevel, limits: CalibrationLimits) -> CalibrationTask {
    let (samples, iterations) = level_work(level);
    CalibrationTask {
        identifier: CalibrationTestKind::CpuFloatDot.stable_identifier(),
        kind: CalibrationTestKind::CpuFloatDot,
        purpose: CalibrationPurpose::CpuRuntimeBaseline,
        requirements: vec![CalibrationRequirement::CpuBackend],
        expected_resources: ExpectedResourceUsage {
            memory_bytes: 2 * 4_096 * std::mem::size_of::<f32>() as u64
                + CALIBRATION_METADATA_BYTES,
            storage_read_bytes: 0,
            cpu_threads: 1,
        },
        model_fingerprint: None,
        storage_fingerprint: None,
        work: CalibrationWorkBudget {
            warmup_iterations: 4,
            iterations_per_sample: iterations,
            sample_count: samples,
            elements_per_iteration: 4_096,
            bytes_per_iteration: 0,
            max_duration_ns: limits.max_task_duration_ns,
        },
    }
}

fn memory_task(level: CalibrationLevel, limits: CalibrationLimits) -> CalibrationTask {
    let (samples, iterations) = level_work(level);
    let available_per_buffer = limits
        .max_memory_bytes
        .saturating_sub(CALIBRATION_METADATA_BYTES)
        / 2;
    let bytes = if available_per_buffer >= 4 * 1024 {
        available_per_buffer.min(1024 * 1024)
    } else {
        1024 * 1024
    };
    CalibrationTask {
        identifier: CalibrationTestKind::MemoryCopyBandwidth.stable_identifier(),
        kind: CalibrationTestKind::MemoryCopyBandwidth,
        purpose: CalibrationPurpose::MemoryBandwidth,
        requirements: vec![CalibrationRequirement::CpuBackend],
        expected_resources: ExpectedResourceUsage {
            memory_bytes: 2 * bytes + CALIBRATION_METADATA_BYTES,
            storage_read_bytes: 0,
            cpu_threads: 1,
        },
        model_fingerprint: None,
        storage_fingerprint: None,
        work: CalibrationWorkBudget {
            warmup_iterations: 2,
            iterations_per_sample: iterations,
            sample_count: samples,
            elements_per_iteration: 0,
            bytes_per_iteration: bytes,
            max_duration_ns: limits.max_task_duration_ns,
        },
    }
}

fn storage_task(
    level: CalibrationLevel,
    storage_fingerprint: u64,
    limits: CalibrationLimits,
) -> CalibrationTask {
    let (samples, _) = level_work(level);
    let available_buffer = limits
        .max_memory_bytes
        .saturating_sub(CALIBRATION_METADATA_BYTES);
    let bytes = if available_buffer >= 4 * 1024 {
        limits
            .max_storage_read_bytes
            .min(1024 * 1024)
            .min(available_buffer)
    } else {
        limits.max_storage_read_bytes.min(1024 * 1024)
    };
    CalibrationTask {
        identifier: CalibrationTestKind::SequentialStorageRead.stable_identifier(),
        kind: CalibrationTestKind::SequentialStorageRead,
        purpose: CalibrationPurpose::StorageThroughput,
        requirements: vec![CalibrationRequirement::SeekableStorage],
        expected_resources: ExpectedResourceUsage {
            memory_bytes: bytes + CALIBRATION_METADATA_BYTES,
            storage_read_bytes: bytes,
            cpu_threads: 1,
        },
        model_fingerprint: None,
        storage_fingerprint: Some(storage_fingerprint),
        work: CalibrationWorkBudget {
            warmup_iterations: 1,
            iterations_per_sample: 1,
            sample_count: samples,
            elements_per_iteration: 0,
            bytes_per_iteration: bytes,
            max_duration_ns: limits.max_task_duration_ns,
        },
    }
}

fn quantized_task(
    level: CalibrationLevel,
    format: CalibrationQuantization,
    model_fingerprint: u64,
    limits: CalibrationLimits,
) -> CalibrationTask {
    let (samples, iterations) = level_work(level);
    let elements = 2_048u64;
    let encoded_bytes = match format {
        CalibrationQuantization::Q4_0 => elements / QK4_0 as u64 * BLOCK_SIZE_Q4_0 as u64,
        CalibrationQuantization::Q8_0 => elements / QK8_0 as u64 * BLOCK_SIZE_Q8_0 as u64,
        CalibrationQuantization::Q4K => elements / QK_K as u64 * BLOCK_SIZE_Q4_K as u64,
    };
    CalibrationTask {
        identifier: CalibrationTestKind::QuantizedRowDecode(format).stable_identifier(),
        kind: CalibrationTestKind::QuantizedRowDecode(format),
        purpose: CalibrationPurpose::QuantizedDecodeThroughput,
        requirements: vec![
            CalibrationRequirement::CpuBackend,
            CalibrationRequirement::QuantizationFormat(format.format_name().to_string()),
        ],
        expected_resources: ExpectedResourceUsage {
            memory_bytes: encoded_bytes
                + elements * std::mem::size_of::<f32>() as u64
                + CALIBRATION_METADATA_BYTES,
            storage_read_bytes: 0,
            cpu_threads: 1,
        },
        model_fingerprint: Some(model_fingerprint),
        storage_fingerprint: None,
        work: CalibrationWorkBudget {
            warmup_iterations: 2,
            iterations_per_sample: iterations,
            sample_count: samples,
            elements_per_iteration: elements,
            bytes_per_iteration: encoded_bytes,
            max_duration_ns: limits.max_task_duration_ns,
        },
    }
}

fn strategy_task(
    level: CalibrationLevel,
    strategy: StrategyId,
    model_fingerprint: u64,
    limits: CalibrationLimits,
) -> CalibrationTask {
    let kind = CalibrationTestKind::StrategyLatency(strategy);
    let mut requirements = vec![
        CalibrationRequirement::ModelExecution,
        CalibrationRequirement::Strategy(strategy),
    ];
    if strategy == StrategyId::CpuLayerStreaming {
        requirements.push(CalibrationRequirement::CpuBackend);
    }
    CalibrationTask {
        identifier: kind.stable_identifier(),
        kind,
        purpose: CalibrationPurpose::StrategyComparison,
        requirements,
        expected_resources: ExpectedResourceUsage {
            memory_bytes: CALIBRATION_METADATA_BYTES,
            storage_read_bytes: 0,
            cpu_threads: 1,
        },
        model_fingerprint: Some(model_fingerprint),
        storage_fingerprint: None,
        work: CalibrationWorkBudget {
            warmup_iterations: 0,
            iterations_per_sample: 1,
            sample_count: 1,
            elements_per_iteration: 1,
            bytes_per_iteration: 0,
            max_duration_ns: limits.max_task_duration_ns,
        },
    }
}

fn relevant_quantizations(model: &ModelProfile) -> Vec<CalibrationQuantization> {
    [
        CalibrationQuantization::Q4K,
        CalibrationQuantization::Q8_0,
        CalibrationQuantization::Q4_0,
    ]
    .into_iter()
    .filter(|format| {
        model
            .tensor_formats
            .iter()
            .any(|profile| profile.format.as_str() == format.format_name())
    })
    .collect()
}

fn level_work(level: CalibrationLevel) -> (u32, u32) {
    match level {
        CalibrationLevel::None => (0, 0),
        CalibrationLevel::Quick => (3, 16),
        CalibrationLevel::Standard => (5, 32),
        CalibrationLevel::Thorough => (7, 64),
    }
}

fn level_at_least(level: CalibrationLevel, minimum: CalibrationLevel) -> bool {
    level.rank() >= minimum.rank()
}

fn make_quantized_row(format: CalibrationQuantization, elements: usize) -> Option<Vec<u8>> {
    let (elements_per_block, block) = match format {
        CalibrationQuantization::Q4_0 => {
            let mut block = Vec::with_capacity(BLOCK_SIZE_Q4_0);
            block.extend_from_slice(&0x3800u16.to_le_bytes());
            for index in 0..16 {
                block.push((index as u8).wrapping_mul(29).wrapping_add(17));
            }
            (QK4_0, block)
        }
        CalibrationQuantization::Q8_0 => {
            let mut block = Vec::with_capacity(BLOCK_SIZE_Q8_0);
            block.extend_from_slice(&0x3800u16.to_le_bytes());
            for index in 0..32 {
                let value = (index as i16 * 17 % 255) - 127;
                block.push(value as i8 as u8);
            }
            (QK8_0, block)
        }
        CalibrationQuantization::Q4K => {
            let mut block = Vec::with_capacity(BLOCK_SIZE_Q4_K);
            block.extend_from_slice(&0x3800u16.to_le_bytes());
            block.extend_from_slice(&0x3000u16.to_le_bytes());
            for index in 0..12 {
                block.push((index as u8).wrapping_mul(11).wrapping_add(3));
            }
            for index in 0..128 {
                block.push((index as u8).wrapping_mul(37).wrapping_add(19));
            }
            (QK_K, block)
        }
    };
    if elements == 0 || elements % elements_per_block != 0 {
        return None;
    }
    let mut row = Vec::with_capacity(elements / elements_per_block * block.len());
    for _ in 0..elements / elements_per_block {
        row.extend_from_slice(&block);
    }
    Some(row)
}

fn duration_ns(started: &Instant) -> u64 {
    started.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn calibration_plan_identifier(plan: &CalibrationPlan) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    hash_value(&mut hash, plan.plan_version as u64);
    hash_value(&mut hash, plan.ruleset_version as u64);
    hash_value(&mut hash, plan.level as u64);
    hash_value(&mut hash, plan.machine_fingerprint);
    hash_value(&mut hash, plan.model_fingerprint);
    hash_value(&mut hash, plan.storage_fingerprint);
    hash_value(&mut hash, plan.limits.max_memory_bytes);
    hash_value(&mut hash, plan.limits.max_storage_read_bytes);
    hash_value(&mut hash, plan.limits.max_total_duration_ns);
    hash_value(&mut hash, plan.limits.max_task_duration_ns);
    hash_value(&mut hash, plan.limits.max_samples as u64);
    hash_value(&mut hash, plan.limits.max_iterations_per_sample as u64);
    for task in &plan.tasks {
        hash_bytes(&mut hash, task.identifier.as_bytes());
        hash_value(&mut hash, task.purpose as u64);
        hash_value(&mut hash, task.expected_resources.memory_bytes);
        hash_value(&mut hash, task.expected_resources.storage_read_bytes);
        hash_value(&mut hash, task.expected_resources.cpu_threads as u64);
        hash_value(&mut hash, task.model_fingerprint.unwrap_or(0));
        hash_value(&mut hash, task.storage_fingerprint.unwrap_or(0));
        hash_value(&mut hash, task.work.warmup_iterations as u64);
        hash_value(&mut hash, task.work.iterations_per_sample as u64);
        hash_value(&mut hash, task.work.sample_count as u64);
        hash_value(&mut hash, task.work.elements_per_iteration);
        hash_value(&mut hash, task.work.bytes_per_iteration);
        hash_value(&mut hash, task.work.max_duration_ns);
        for requirement in &task.requirements {
            hash_requirement(&mut hash, requirement);
        }
    }
    format!("calibration-plan-v{}-{hash:016x}", plan.plan_version)
}

fn hash_requirement(hash: &mut u64, requirement: &CalibrationRequirement) {
    match requirement {
        CalibrationRequirement::CpuBackend => hash_value(hash, 1),
        CalibrationRequirement::SeekableStorage => hash_value(hash, 2),
        CalibrationRequirement::ModelExecution => hash_value(hash, 3),
        CalibrationRequirement::QuantizationFormat(format) => {
            hash_value(hash, 4);
            hash_bytes(hash, format.as_bytes());
        }
        CalibrationRequirement::Strategy(strategy) => {
            hash_value(hash, 5);
            hash_value(hash, *strategy as u64);
        }
    }
}

fn hash_value(hash: &mut u64, value: u64) {
    hash_bytes(hash, &value.to_le_bytes());
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= *byte as u64;
        *hash = (*hash).wrapping_mul(0x100000001b3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::io::Write;
    use std::path::PathBuf;

    use crate::planner::{
        AdvancedOverrides, Availability, CpuFeature, DiscoveryState, GpuPreference,
        ModelExecutionCompatibility, ModelIdentity, OperatingMode, StoragePathState,
        TensorFormatProfile,
    };
    use tempfile::NamedTempFile;

    fn profiles(
        level: CalibrationLevel,
        path: PathBuf,
    ) -> (
        UserProfile,
        MachineProfile,
        ModelProfile,
        StorageProfile,
        CapabilitySet,
    ) {
        let mut user = UserProfile::new(8 * 1024 * 1024, OperatingMode::BalancedNormal);
        user.calibration_level = level;
        user.gpu_preference = GpuPreference::Disabled;
        user.advanced = AdvancedOverrides::default();

        let mut cpu_features = BTreeSet::new();
        cpu_features.insert(CpuFeature::Avx2);
        let machine = MachineProfile {
            schema_version: PROFILE_SCHEMA_VERSION,
            os: "linux".to_string(),
            architecture: "x86_64".to_string(),
            kernel_version: Some("test-kernel".to_string()),
            cpu_vendor: Some("test-vendor".to_string()),
            cpu_model: Some("test-cpu".to_string()),
            physical_cpu_cores: Some(2),
            logical_cpu_cores: 4,
            cpu_features,
            total_ram_bytes: Some(16 * 1024 * 1024),
            available_ram_bytes: Some(12 * 1024 * 1024),
            cpu_backend: Availability {
                detected: true,
                supported: true,
                usable: true,
            },
            gpu_inventory_state: DiscoveryState::Complete,
            gpus: Vec::new(),
        };
        let model = ModelProfile {
            schema_version: PROFILE_SCHEMA_VERSION,
            identity: ModelIdentity {
                display_name: Some("synthetic".to_string()),
                descriptor_fingerprint: 11,
            },
            architecture: Some("llama".to_string()),
            context_size: Some(64),
            layer_count: Some(1),
            tensor_count: 2,
            total_tensor_elements: Some(4_096),
            file_size_bytes: 4_096,
            tensor_data_bytes: Some(4_096),
            tensor_formats: vec![
                TensorFormatProfile {
                    format: "Q4_K".to_string(),
                    tensor_count: 1,
                    element_count: Some(2_048),
                    byte_count: Some(1_152),
                    runtime_supported: true,
                },
                TensorFormatProfile {
                    format: "Q8_0".to_string(),
                    tensor_count: 1,
                    element_count: Some(2_048),
                    byte_count: Some(2_176),
                    runtime_supported: true,
                },
            ],
            execution_compatibility: ModelExecutionCompatibility::Executable,
            has_coalescible_layer_reads: true,
            has_reusable_grouped_read_candidate: false,
        };
        let storage = StorageProfile {
            schema_version: PROFILE_SCHEMA_VERSION,
            model_path: path,
            path_state: StoragePathState::Ready,
            readable: true,
            regular_file: true,
            file_size_bytes: Some(4_096),
            filesystem_type: Some("testfs".to_string()),
            filesystem_id: Some("test-fs".to_string()),
            device_id: Some("test-device".to_string()),
            kind: crate::planner::StorageKind::Local,
            seekable: true,
        };
        let capabilities = CapabilitySet {
            schema_version: PROFILE_SCHEMA_VERSION,
            machine_fingerprint: machine.fingerprint(),
            model_fingerprint: model.identity.descriptor_fingerprint,
            storage_fingerprint: storage.fingerprint(),
            storage_usable: true,
            cpu_backend_usable: true,
            cpu_simd_avx2: true,
            gpu_backend_usable: false,
            model_execution_compatible: true,
            layer_cache_possible: true,
            read_coalescing_possible: true,
            grouped_read_buffer_reuse_possible: false,
            supported_tensor_formats: vec!["Q4_K".to_string(), "Q8_0".to_string()],
            unsupported_tensor_formats: Vec::new(),
        };
        (user, machine, model, storage, capabilities)
    }

    fn test_limits() -> CalibrationLimits {
        CalibrationLimits {
            max_memory_bytes: 4 * 1024 * 1024,
            max_storage_read_bytes: 64 * 1024,
            max_total_duration_ns: 5_000_000_000,
            max_task_duration_ns: 1_000_000_000,
            max_samples: 7,
            max_iterations_per_sample: 64,
        }
    }

    #[test]
    fn test_calibration_plan_level_mapping_is_explicit() {
        let path = PathBuf::from("/missing/model.gguf");
        let (mut user, machine, model, storage, capabilities) =
            profiles(CalibrationLevel::None, path);
        let none = CalibrationPlan::build(
            &user,
            &machine,
            &model,
            &storage,
            &capabilities,
            test_limits(),
        )
        .unwrap();
        assert!(none.tasks.is_empty());

        user.calibration_level = CalibrationLevel::Quick;
        let quick = CalibrationPlan::build(
            &user,
            &machine,
            &model,
            &storage,
            &capabilities,
            test_limits(),
        )
        .unwrap();
        assert_eq!(quick.tasks.len(), 2);

        user.calibration_level = CalibrationLevel::Standard;
        let standard = CalibrationPlan::build(
            &user,
            &machine,
            &model,
            &storage,
            &capabilities,
            test_limits(),
        )
        .unwrap();
        assert_eq!(standard.tasks.len(), 4);
        assert!(matches!(
            standard.tasks[3].kind,
            CalibrationTestKind::QuantizedRowDecode(CalibrationQuantization::Q4K)
        ));

        user.calibration_level = CalibrationLevel::Thorough;
        let thorough = CalibrationPlan::build(
            &user,
            &machine,
            &model,
            &storage,
            &capabilities,
            test_limits(),
        )
        .unwrap();
        assert_eq!(thorough.tasks.len(), 5);
    }

    #[test]
    fn test_calibration_identity_is_deterministic_and_versioned() {
        let (user, machine, model, storage, capabilities) = profiles(
            CalibrationLevel::Standard,
            PathBuf::from("/missing/model.gguf"),
        );
        let first = CalibrationPlan::build(
            &user,
            &machine,
            &model,
            &storage,
            &capabilities,
            test_limits(),
        )
        .unwrap();
        let second = CalibrationPlan::build(
            &user,
            &machine,
            &model,
            &storage,
            &capabilities,
            test_limits(),
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.plan_version, CALIBRATION_PLAN_VERSION);
        assert_eq!(first.ruleset_version, CALIBRATION_RULESET_VERSION);
    }

    #[test]
    fn test_runner_returns_measured_and_unavailable_states() {
        let (user, machine, model, mut storage, mut capabilities) = profiles(
            CalibrationLevel::Quick,
            PathBuf::from("/missing/model.gguf"),
        );
        storage.seekable = false;
        capabilities.storage_fingerprint = storage.fingerprint();
        capabilities.storage_usable = false;
        let plan = CalibrationPlan::build(
            &user,
            &machine,
            &model,
            &storage,
            &capabilities,
            test_limits(),
        )
        .unwrap();
        let result = CalibrationRunner
            .run(&plan, &user, &machine, &model, &storage, &capabilities)
            .unwrap();
        assert_eq!(result.calibration_plan_identifier, plan.identifier);
        assert_eq!(result.calibration_plan_version, CALIBRATION_PLAN_VERSION);
        assert_eq!(result.calibration_ruleset_version, CALIBRATION_RULESET_VERSION);
        assert_eq!(result.observations.len(), 2);
        assert_eq!(result.observations[0].status, ObservationStatus::Measured);
        assert_eq!(
            result.observations[1].status,
            ObservationStatus::Unavailable
        );
        assert_eq!(
            result.observations[1].status_reason,
            Some(ObservationStatusReason::CapabilityUnavailable)
        );
    }

    #[test]
    fn test_runner_enforces_resource_limits_with_skipped_state() {
        let (user, machine, model, storage, capabilities) = profiles(
            CalibrationLevel::Standard,
            PathBuf::from("/missing/model.gguf"),
        );
        let mut limits = test_limits();
        limits.max_memory_bytes = 1;
        let plan = CalibrationPlan::build(
            &user,
            &machine,
            &model,
            &storage,
            &capabilities,
            limits,
        )
        .unwrap();
        let result = CalibrationRunner
            .run(&plan, &user, &machine, &model, &storage, &capabilities)
            .unwrap();
        assert!(result.observations.iter().any(|observation| {
            observation.status == ObservationStatus::Skipped
                && observation.status_reason
                    == Some(ObservationStatusReason::ResourceLimitExceeded)
        }));
    }

    #[test]
    fn test_storage_measurement_is_bounded_and_structurally_valid() {
        let mut file = NamedTempFile::new().unwrap();
        let data = vec![0xA5; 4 * 1024];
        file.write_all(&data).unwrap();
        file.flush().unwrap();
        let (user, machine, model, storage, capabilities) = profiles(
            CalibrationLevel::Quick,
            file.path().to_path_buf(),
        );
        let plan = CalibrationPlan::build(
            &user,
            &machine,
            &model,
            &storage,
            &capabilities,
            test_limits(),
        )
        .unwrap();
        let result = CalibrationRunner
            .run(&plan, &user, &machine, &model, &storage, &capabilities)
            .unwrap();
        let storage_observation = result
            .observations
            .iter()
            .find(|observation| {
                observation.metric == ObservationMetric::SequentialReadThroughput
            })
            .unwrap();
        assert_eq!(storage_observation.status, ObservationStatus::Measured);
        assert!(storage_observation.value.is_some());
        assert!(storage_observation.sample_count > 0);
        assert!(storage_observation.workload_bytes <= 64 * 1024 * 8 * 7);
        storage_observation.validate().unwrap();
    }

    #[test]
    fn test_observation_states_are_explicit_and_validated() {
        let task = cpu_task(CalibrationLevel::Quick, test_limits());
        for (status, reason) in [
            (
                ObservationStatus::Unavailable,
                ObservationStatusReason::CapabilityUnavailable,
            ),
            (
                ObservationStatus::Skipped,
                ObservationStatusReason::ResourceLimitExceeded,
            ),
            (
                ObservationStatus::Failed,
                ObservationStatusReason::InvalidWorkload,
            ),
        ] {
            non_measured_observation(&task, CalibrationLevel::Quick, status, reason)
                .validate()
                .unwrap();
        }
    }

    #[test]
    fn test_strategy_measurement_is_explicitly_skipped_until_implemented() {
        let (user, machine, model, storage, capabilities) = profiles(
            CalibrationLevel::Thorough,
            PathBuf::from("/missing/model.gguf"),
        );
        let mut plan = CalibrationPlan::build(
            &user,
            &machine,
            &model,
            &storage,
            &capabilities,
            test_limits(),
        )
        .unwrap();
        plan.tasks.clear();
        plan.identifier = calibration_plan_identifier(&plan);
        plan.request_strategy_measurement(StrategyId::CpuLayerStreaming)
            .unwrap();
        let result = CalibrationRunner
            .run(&plan, &user, &machine, &model, &storage, &capabilities)
            .unwrap();
        let observation = result.observations.last().unwrap();
        assert_eq!(observation.status, ObservationStatus::Skipped);
        assert_eq!(
            observation.status_reason,
            Some(ObservationStatusReason::TestNotImplemented)
        );
    }
}
