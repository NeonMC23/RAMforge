//! Adaptive strategy generation, bounded calibration, and multi-objective
//! comparison.
//!
//! The engine does not execute anything. It produces explainable candidates and
//! interprets measured benchmark reports. Execution remains in the CLI/application
//! layer, which must still require explicit confirmation before applying a plan.

use serde::{Deserialize, Serialize};

use crate::adaptive::capability::{
    resolve_compatibility, CompatibilityReport, CompatibilityRequest, SupportClass,
};
use crate::adaptive::resource::{assess_budget, BudgetClass, ResourceBudget};
use crate::benchmark_runner::BenchmarkReport;
use crate::model_requirements::ModelRequirements;
use crate::profiling::{now_unix_ms, HardwareProfile, ModelProfile, WorkloadProfile};

pub const ADAPTIVE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrategyAxis {
    Baseline,
    ThreadCount,
    KvQuantization,
    StagedResidency,
    GpuPlacement,
    Speculative,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdaptiveStrategy {
    pub schema_version: u32,
    pub id: String,
    pub label: String,
    pub axis: StrategyAxis,
    /// `cpu`, `gpu`, or `hybrid`.
    pub execution_mode: String,
    pub device_id: String,
    pub gpu_layers: u32,
    pub threads: u32,
    pub context_tokens: u32,
    pub batch_tokens: u32,
    pub ubatch_tokens: u32,
    pub kv_type_k: String,
    pub kv_type_v: String,
    pub load_mode: String,
    /// `off`, `auto`, or `staged`.
    pub residency: String,
    pub staged_auto: bool,
    pub staged_layers: Vec<u32>,
    pub staged_budget_bytes: Option<u64>,
    pub draft_model_path: Option<String>,
    pub draft_tokens: Option<u32>,
    pub speculative: bool,
}

impl AdaptiveStrategy {
    pub fn key(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
            self.execution_mode,
            self.device_id,
            self.gpu_layers,
            self.threads,
            self.context_tokens,
            self.batch_tokens,
            self.ubatch_tokens,
            self.kv_type_k,
            self.kv_type_v,
            self.load_mode,
            self.residency,
            self.staged_auto,
            self.draft_tokens.unwrap_or(0),
            self.draft_model_path.as_deref().unwrap_or(""),
        )
    }

    pub fn with_identity(mut self) -> Self {
        self.id = self.key();
        self
    }

    pub fn is_cpu_target(&self) -> bool {
        self.gpu_layers == 0
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != ADAPTIVE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported strategy schema {}",
                self.schema_version
            ));
        }
        if self.threads == 0 {
            return Err("strategy threads must be greater than zero".to_string());
        }
        if self.context_tokens == 0 || self.batch_tokens == 0 || self.ubatch_tokens == 0 {
            return Err("strategy context and batch values must be greater than zero".to_string());
        }
        if self.batch_tokens > self.context_tokens {
            return Err("strategy batch exceeds context".to_string());
        }
        if self.ubatch_tokens > self.batch_tokens {
            return Err("strategy micro-batch exceeds batch".to_string());
        }
        if self.speculative && self.draft_tokens.unwrap_or(0) == 0 {
            return Err("speculative strategy has no draft length".to_string());
        }
        if self.speculative && self.draft_model_path.is_none() {
            return Err("speculative strategy has no draft model".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateStatus {
    Eligible,
    EligibleWithRisk { reason: String },
    Rejected { reason: String },
    Unsupported { reason: String },
    InsufficientInformation { reason: String },
}

impl CandidateStatus {
    pub fn is_eligible(&self) -> bool {
        matches!(self, Self::Eligible | Self::EligibleWithRisk { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StrategyCandidate {
    pub strategy: AdaptiveStrategy,
    pub status: CandidateStatus,
    pub compatibility: CompatibilityReport,
    pub budget: ResourceBudget,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateLimits {
    pub max_candidates: usize,
    pub max_experiments: usize,
    pub min_runs_for_selection: u32,
    pub min_improvement_bps: i64,
    pub max_variability_bps: u32,
}

impl Default for CandidateLimits {
    fn default() -> Self {
        Self {
            max_candidates: 8,
            max_experiments: 3,
            min_runs_for_selection: 2,
            min_improvement_bps: 500,
            max_variability_bps: 1_000,
        }
    }
}

impl CandidateLimits {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_candidates == 0 {
            return Err("max_candidates must be greater than zero".to_string());
        }
        if self.max_experiments == 0 {
            return Err("max_experiments must be greater than zero".to_string());
        }
        if self.max_experiments > 10 {
            return Err("max_experiments must be at most 10".to_string());
        }
        if self.min_runs_for_selection == 0 {
            return Err("min_runs_for_selection must be greater than zero".to_string());
        }
        if self.min_improvement_bps < 0 {
            return Err("min_improvement_bps must not be negative".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateSet {
    pub schema_version: u32,
    pub baseline_id: String,
    pub candidates: Vec<StrategyCandidate>,
    pub notes: Vec<String>,
}

/// Build a baseline strategy from a workload profile.
pub fn baseline_strategy(workload: &WorkloadProfile) -> AdaptiveStrategy {
    let execution_mode = if workload.gpu_layers == 0 {
        "cpu"
    } else if workload.residency == "staged" {
        "hybrid"
    } else {
        "gpu"
    };
    let mut strategy = AdaptiveStrategy {
        schema_version: ADAPTIVE_SCHEMA_VERSION,
        id: String::new(),
        label: "baseline".to_string(),
        axis: StrategyAxis::Baseline,
        execution_mode: execution_mode.to_string(),
        device_id: workload.device_id.clone(),
        gpu_layers: workload.gpu_layers,
        threads: workload.threads,
        context_tokens: workload.context_tokens,
        batch_tokens: workload.batch_tokens,
        ubatch_tokens: workload.ubatch_tokens,
        kv_type_k: workload.kv_type_k.clone(),
        kv_type_v: workload.kv_type_v.clone(),
        load_mode: workload.load_mode.clone(),
        residency: workload.residency.clone(),
        staged_auto: workload.residency == "auto" || workload.residency == "staged",
        staged_layers: Vec::new(),
        staged_budget_bytes: None,
        draft_model_path: None,
        draft_tokens: workload.draft_tokens,
        speculative: workload.draft_tokens.is_some(),
    };
    strategy.id = strategy.key();
    strategy
}

fn strategy_workload(base: &WorkloadProfile, strategy: &AdaptiveStrategy) -> WorkloadProfile {
    let mut workload = base.clone();
    workload.context_tokens = strategy.context_tokens;
    workload.batch_tokens = strategy.batch_tokens;
    workload.ubatch_tokens = strategy.ubatch_tokens;
    workload.threads = strategy.threads;
    workload.gpu_layers = strategy.gpu_layers;
    workload.kv_type_k = strategy.kv_type_k.clone();
    workload.kv_type_v = strategy.kv_type_v.clone();
    workload.load_mode = strategy.load_mode.clone();
    workload.device_id = strategy.device_id.clone();
    workload.residency = strategy.residency.clone();
    workload.draft_tokens = strategy.draft_tokens;
    workload.backend = strategy.execution_mode.clone();
    workload
}

fn candidate(
    base: &WorkloadProfile,
    model: &ModelProfile,
    hardware: &HardwareProfile,
    requirements: Option<&ModelRequirements>,
    strategy: AdaptiveStrategy,
    draft_compatible: Option<bool>,
    draft_bytes: Option<u64>,
) -> StrategyCandidate {
    let strategy = strategy.with_identity();
    let workload = strategy_workload(base, &strategy);
    let request = CompatibilityRequest::from_workload(&workload, draft_compatible);
    let compatibility = resolve_compatibility(model, hardware, &workload, &request);
    let budget = assess_budget(model, hardware, &workload, requirements, draft_bytes);
    let status = if compatibility.overall == SupportClass::Unsupported {
        CandidateStatus::Unsupported {
            reason: compatibility
                .reasons
                .first()
                .cloned()
                .unwrap_or_else(|| "a required feature is unsupported".to_string()),
        }
    } else if compatibility.overall == SupportClass::InsufficientInformation {
        CandidateStatus::InsufficientInformation {
            reason: compatibility
                .reasons
                .first()
                .cloned()
                .unwrap_or_else(|| "compatibility is not resolved".to_string()),
        }
    } else if budget.admission == BudgetClass::Infeasible {
        CandidateStatus::Rejected {
            reason: budget
                .reasons
                .first()
                .cloned()
                .unwrap_or_else(|| "estimated resource budget is infeasible".to_string()),
        }
    } else if budget.admission == BudgetClass::Risky {
        CandidateStatus::EligibleWithRisk {
            reason: budget
                .reasons
                .first()
                .cloned()
                .unwrap_or_else(|| "estimated resource budget is risky".to_string()),
        }
    } else if budget.admission == BudgetClass::Unknown && strategy.axis != StrategyAxis::Baseline {
        CandidateStatus::InsufficientInformation {
            reason: "resource admission is unknown; only the baseline may proceed".to_string(),
        }
    } else {
        CandidateStatus::Eligible
    };
    StrategyCandidate {
        strategy,
        status,
        compatibility,
        budget,
    }
}

/// Generate a small, explainable candidate set around `baseline`. Variants are
/// added only when they differ meaningfully from the baseline.
#[allow(clippy::too_many_arguments)]
pub fn generate_candidates(
    model: &ModelProfile,
    hardware: &HardwareProfile,
    workload: &WorkloadProfile,
    baseline: &AdaptiveStrategy,
    requirements: Option<&ModelRequirements>,
    draft_compatible: Option<bool>,
    draft_bytes: Option<u64>,
    limits: &CandidateLimits,
) -> CandidateSet {
    let mut notes = Vec::new();
    if let Err(error) = limits.validate() {
        notes.push(format!("candidate limits rejected: {error}"));
        return CandidateSet {
            schema_version: ADAPTIVE_SCHEMA_VERSION,
            baseline_id: baseline.id.clone(),
            candidates: Vec::new(),
            notes,
        };
    }
    let base_workload = strategy_workload(workload, baseline);
    let mut candidates = vec![candidate(
        &base_workload,
        model,
        hardware,
        requirements,
        baseline.clone(),
        draft_compatible,
        draft_bytes,
    )];

    let mut push = |strategy: AdaptiveStrategy| {
        let candidate = candidate(
            &base_workload,
            model,
            hardware,
            requirements,
            strategy,
            draft_compatible,
            draft_bytes,
        );
        if !candidates
            .iter()
            .any(|existing| existing.strategy.id == candidate.strategy.id)
        {
            candidates.push(candidate);
        }
    };

    // Speculative variants come first for CPU-target experiments: this is the
    // historically measured general win when the target is oversized.
    if draft_compatible == Some(true) && baseline.is_cpu_target() {
        if let Some(draft_path) = baseline.draft_model_path.clone() {
            for draft_tokens in [4u32, 8, 12] {
                let mut strategy = baseline.clone();
                strategy.label = format!("speculative-k{draft_tokens}");
                strategy.axis = StrategyAxis::Speculative;
                strategy.speculative = true;
                strategy.draft_tokens = Some(draft_tokens);
                strategy.draft_model_path = Some(draft_path.clone());
                push(strategy);
            }
        }
    }

    if let Some(logical) = hardware.logical_cpus {
        if logical > baseline.threads {
            let mut strategy = baseline.clone();
            strategy.label = format!("threads-{logical}");
            strategy.axis = StrategyAxis::ThreadCount;
            strategy.threads = logical;
            push(strategy);
        }
    }

    if baseline.kv_type_k.eq_ignore_ascii_case("f16") {
        let mut strategy = baseline.clone();
        strategy.label = "kv-q8_0".to_string();
        strategy.axis = StrategyAxis::KvQuantization;
        strategy.kv_type_k = "q8_0".to_string();
        strategy.kv_type_v = "q8_0".to_string();
        push(strategy);
    }

    let gpu = hardware.gpu_available();
    if gpu && baseline.gpu_layers == 0 && baseline.residency != "staged" {
        let mut strategy = baseline.clone();
        strategy.label = "staged-residency".to_string();
        strategy.axis = StrategyAxis::StagedResidency;
        strategy.execution_mode = "hybrid".to_string();
        strategy.residency = "staged".to_string();
        strategy.staged_auto = true;
        push(strategy);
    }

    if gpu && baseline.gpu_layers == 0 {
        if let Some(layers) = model.layer_count {
            let free = hardware.max_free_vram_bytes().unwrap_or(0);
            let per_layer = model.file_bytes.checked_div(layers).unwrap_or(0);
            let fit = free
                .saturating_sub(256 * 1024 * 1024)
                .checked_div(per_layer)
                .unwrap_or(0);
            if fit > 0 {
                let mut strategy = baseline.clone();
                strategy.label = format!("gpu-{}-layers", fit.min(layers));
                strategy.axis = StrategyAxis::GpuPlacement;
                strategy.execution_mode = "gpu".to_string();
                strategy.device_id = hardware
                    .gpus
                    .iter()
                    .find(|gpu| gpu.selectable)
                    .map(|gpu| gpu.id.clone())
                    .unwrap_or_else(|| "auto".to_string());
                strategy.gpu_layers = fit.min(layers) as u32;
                push(strategy);
            }
        }
    }

    drop(push);
    candidates.truncate(limits.max_candidates);
    notes.push(format!(
        "{} candidates generated; {} experiments allowed",
        candidates.len(),
        limits.max_experiments
    ));
    CandidateSet {
        schema_version: ADAPTIVE_SCHEMA_VERSION,
        baseline_id: baseline.id.clone(),
        candidates,
        notes,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasurementSummary {
    pub generation_tokens_per_second: f64,
    pub min_generation_tokens_per_second: f64,
    pub max_generation_tokens_per_second: f64,
    pub decode_tokens_per_second: Option<f64>,
    pub prompt_tokens_per_second: Option<f64>,
    pub time_to_first_token_ns: Option<u64>,
    pub pss_bytes: Option<u64>,
    pub acceptance_rate_bps: Option<u32>,
    pub runs: u32,
    pub condition: String,
    pub variability_bps: Option<u32>,
}

impl MeasurementSummary {
    pub fn from_report(report: &BenchmarkReport) -> Self {
        let variability_bps = if report.median_generation_tokens_per_second > 0.0 {
            Some(
                ((report.max_generation_tokens_per_second
                    - report.min_generation_tokens_per_second)
                    / report.median_generation_tokens_per_second
                    * 10_000.0)
                    .round()
                    .max(0.0) as u32,
            )
        } else {
            None
        };
        Self {
            generation_tokens_per_second: report.median_generation_tokens_per_second,
            min_generation_tokens_per_second: report.min_generation_tokens_per_second,
            max_generation_tokens_per_second: report.max_generation_tokens_per_second,
            decode_tokens_per_second: report.median_decode_tokens_per_second,
            prompt_tokens_per_second: report.median_prompt_tokens_per_second,
            time_to_first_token_ns: report.median_time_to_first_token_ns,
            pss_bytes: report.runs_detail.last().and_then(|run| run.pss_bytes),
            acceptance_rate_bps: report
                .runs_detail
                .last()
                .and_then(|run| run.speculative.as_ref())
                .map(|stats| stats.acceptance_rate_bps),
            runs: report.runs,
            condition: report.condition.clone(),
            variability_bps,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentStatus {
    Completed,
    Inconclusive,
    Failed,
    Interrupted,
    Unsupported,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationExperiment {
    pub strategy: AdaptiveStrategy,
    pub status: ExperimentStatus,
    pub measurement: Option<MeasurementSummary>,
    pub error: Option<String>,
    pub notes: Vec<String>,
}

impl CalibrationExperiment {
    pub fn completed(strategy: AdaptiveStrategy, report: &BenchmarkReport) -> Self {
        Self {
            strategy,
            status: ExperimentStatus::Completed,
            measurement: Some(MeasurementSummary::from_report(report)),
            error: None,
            notes: Vec::new(),
        }
    }

    pub fn not_run(
        strategy: AdaptiveStrategy,
        status: ExperimentStatus,
        error: impl Into<String>,
    ) -> Self {
        Self {
            strategy,
            status,
            measurement: None,
            error: Some(error.into()),
            notes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectivePreference {
    Throughput,
    Balanced,
    Latency,
    Memory,
}

impl ObjectivePreference {
    pub fn label(self) -> &'static str {
        match self {
            Self::Throughput => "throughput",
            Self::Balanced => "balanced",
            Self::Latency => "latency",
            Self::Memory => "memory",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateComparison {
    pub baseline_id: String,
    pub strategy_id: String,
    pub throughput_gain_bps: Option<i64>,
    pub ttft_change_bps: Option<i64>,
    pub memory_change_bps: Option<i64>,
    pub significant: bool,
    pub tradeoffs: Vec<String>,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BestSelection {
    pub strategy_id: String,
    pub confidence: String,
    pub improvement_bps: i64,
    pub tradeoffs: Vec<String>,
    pub reasons: Vec<String>,
    pub measurement: MeasurementSummary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub schema_version: u32,
    pub model_fingerprint: Option<String>,
    pub hardware_fingerprint: String,
    pub environment_fingerprint: String,
    pub baseline_id: Option<String>,
    pub experiments: Vec<CalibrationExperiment>,
    pub comparisons: Vec<CandidateComparison>,
    pub best: Option<BestSelection>,
    pub preference: ObjectivePreference,
    pub limits: CandidateLimits,
    pub notes: Vec<String>,
    pub limitations: Vec<String>,
    pub created_at_unix_ms: Option<u64>,
}

impl CalibrationReport {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != ADAPTIVE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported calibration schema {}",
                self.schema_version
            ));
        }
        if self.experiments.is_empty() {
            return Err("calibration report has no experiments".to_string());
        }
        self.limits.validate()?;
        for experiment in &self.experiments {
            experiment.strategy.validate()?;
            if experiment.status == ExperimentStatus::Completed && experiment.measurement.is_none()
            {
                return Err("completed experiment has no measurement".to_string());
            }
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(|error| error.to_string())
    }
}

fn gain_bps(candidate: f64, baseline: f64) -> Option<i64> {
    if baseline <= 0.0 {
        None
    } else {
        Some(((candidate - baseline) / baseline * 10_000.0).round() as i64)
    }
}

pub fn compare_experiments(
    baseline: &CalibrationExperiment,
    candidate: &CalibrationExperiment,
    limits: &CandidateLimits,
) -> CandidateComparison {
    let mut reasons = Vec::new();
    let mut tradeoffs = Vec::new();
    let baseline_measurement = baseline.measurement.as_ref();
    let candidate_measurement = candidate.measurement.as_ref();
    let (Some(base), Some(candidate_measurement)) = (baseline_measurement, candidate_measurement)
    else {
        return CandidateComparison {
            baseline_id: baseline.strategy.id.clone(),
            strategy_id: candidate.strategy.id.clone(),
            throughput_gain_bps: None,
            ttft_change_bps: None,
            memory_change_bps: None,
            significant: false,
            tradeoffs: vec!["a measurement is unavailable".to_string()],
            reasons,
        };
    };

    let throughput_gain_bps = gain_bps(
        candidate_measurement.generation_tokens_per_second,
        base.generation_tokens_per_second,
    );
    let ttft_change_bps = match (
        base.time_to_first_token_ns,
        candidate_measurement.time_to_first_token_ns,
    ) {
        (Some(base), Some(candidate)) if base > 0 => {
            Some(((candidate as f64 - base as f64) / base as f64 * 10_000.0).round() as i64)
        }
        _ => None,
    };
    let memory_change_bps = match (base.pss_bytes, candidate_measurement.pss_bytes) {
        (Some(base), Some(candidate)) if base > 0 => {
            Some(((candidate as f64 - base as f64) / base as f64 * 10_000.0).round() as i64)
        }
        _ => None,
    };

    let stable = candidate_measurement
        .variability_bps
        .zip(base.variability_bps)
        .is_some_and(|(candidate, base)| {
            candidate <= limits.max_variability_bps && base <= limits.max_variability_bps
        });
    let separated = candidate_measurement.min_generation_tokens_per_second
        >= base.max_generation_tokens_per_second;
    let improvement = throughput_gain_bps.unwrap_or(0);
    let enough_runs = candidate_measurement.runs >= limits.min_runs_for_selection;
    let significant =
        improvement >= limits.min_improvement_bps && enough_runs && (separated || stable);
    if improvement < limits.min_improvement_bps {
        reasons.push(format!(
            "throughput gain {improvement} bps is below the {} bps threshold",
            limits.min_improvement_bps
        ));
    }
    if !separated && !stable {
        reasons.push("measurement ranges overlap and variability is high".to_string());
    }
    if !enough_runs {
        reasons.push(format!(
            "{} runs is below the {} run minimum",
            candidate_measurement.runs, limits.min_runs_for_selection
        ));
    }
    if let Some(ttft) = ttft_change_bps {
        if ttft > 2_000 {
            tradeoffs.push(format!("time-to-first-token worsens by {ttft} bps"));
        }
    }
    if let Some(memory) = memory_change_bps {
        if memory > 1_000 {
            tradeoffs.push(format!("process PSS grows by {memory} bps"));
        }
    }
    if candidate_measurement.acceptance_rate_bps.is_some()
        && candidate_measurement.acceptance_rate_bps.unwrap_or(0) < 2_000
    {
        tradeoffs.push(format!(
            "speculative acceptance is low ({:?} bps)",
            candidate_measurement.acceptance_rate_bps
        ));
    }
    if memory_change_bps.is_none() {
        tradeoffs.push("PSS was not observed for the candidate".to_string());
    }

    CandidateComparison {
        baseline_id: baseline.strategy.id.clone(),
        strategy_id: candidate.strategy.id.clone(),
        throughput_gain_bps,
        ttft_change_bps,
        memory_change_bps,
        significant,
        tradeoffs,
        reasons,
    }
}

/// Select a winner only from comparable, sufficiently repeated measurements.
/// Returns `None` and explains why when the evidence is inconclusive.
pub fn select_best(
    experiments: &[CalibrationExperiment],
    baseline_id: &str,
    preference: ObjectivePreference,
    limits: &CandidateLimits,
) -> (Option<BestSelection>, Vec<CandidateComparison>, Vec<String>) {
    let baseline = experiments
        .iter()
        .find(|experiment| experiment.strategy.id == baseline_id);
    let Some(baseline) = baseline else {
        return (
            None,
            Vec::new(),
            vec!["baseline experiment is missing; no winner can be selected".to_string()],
        );
    };
    let Some(base) = baseline.measurement.as_ref() else {
        return (
            None,
            Vec::new(),
            vec!["baseline has no completed measurement; no winner can be selected".to_string()],
        );
    };
    let mut comparisons = Vec::new();
    let mut notes = Vec::new();
    let mut viable_indices: Vec<usize> = Vec::new();

    for experiment in experiments {
        if experiment.strategy.id == baseline_id {
            continue;
        }
        if experiment.status != ExperimentStatus::Completed {
            notes.push(format!(
                "{} was not considered: {:?}",
                experiment.strategy.label, experiment.status
            ));
            continue;
        }
        let comparison = compare_experiments(baseline, experiment, limits);
        if comparison.significant {
            viable_indices.push(comparisons.len());
        }
        comparisons.push(comparison);
    }

    let viable: Vec<(&CalibrationExperiment, &CandidateComparison)> = viable_indices
        .iter()
        .map(|index| {
            let comparison = &comparisons[*index];
            let experiment = experiments
                .iter()
                .find(|experiment| experiment.strategy.id == comparison.strategy_id)
                .expect("comparison references an existing experiment");
            (experiment, comparison)
        })
        .collect();

    if viable.is_empty() {
        notes.push(format!(
            "no candidate cleared the {} bps improvement threshold with sufficient runs",
            limits.min_improvement_bps
        ));
        return (None, comparisons, notes);
    }

    let score = |experiment: &CalibrationExperiment| -> f64 {
        experiment
            .measurement
            .as_ref()
            .map(|measurement| measurement.generation_tokens_per_second)
            .unwrap_or(0.0)
    };
    let chosen = match preference {
        ObjectivePreference::Throughput => viable
            .iter()
            .max_by(|left, right| score(left.0).total_cmp(&score(right.0))),
        ObjectivePreference::Balanced => viable
            .iter()
            .filter(|(_experiment, comparison)| {
                comparison.ttft_change_bps.unwrap_or(0) <= 2_500
                    && comparison.memory_change_bps.unwrap_or(0) <= 2_000
            })
            .max_by(|left, right| score(left.0).total_cmp(&score(right.0)))
            .or_else(|| {
                notes.push(
                    "balanced preference rejected every improvement because of TTFT or memory tradeoffs"
                        .to_string(),
                );
                None
            }),
        ObjectivePreference::Latency => viable
            .iter()
            .filter(|(experiment, _)| {
                score(experiment) >= base.generation_tokens_per_second * 0.95
            })
            .min_by_key(|(experiment, _)| {
                experiment
                    .measurement
                    .as_ref()
                    .and_then(|measurement| measurement.time_to_first_token_ns)
                    .unwrap_or(u64::MAX)
            }),
        ObjectivePreference::Memory => viable
            .iter()
            .filter(|(experiment, _)| {
                score(experiment) >= base.generation_tokens_per_second * 0.95
            })
            .min_by_key(|(experiment, _)| {
                experiment
                    .measurement
                    .as_ref()
                    .and_then(|measurement| measurement.pss_bytes)
                    .unwrap_or(u64::MAX)
            }),
    };

    let Some((chosen, comparison)) = chosen else {
        return (None, comparisons, notes);
    };
    let measurement = chosen.measurement.as_ref().unwrap();
    let stable = measurement
        .variability_bps
        .is_some_and(|variability| variability <= limits.max_variability_bps)
        && measurement.runs >= limits.min_runs_for_selection;
    let confidence = if stable {
        "measured".to_string()
    } else {
        "low-confidence".to_string()
    };
    if measurement.runs < limits.min_runs_for_selection {
        notes.push("winner is below the minimum run count".to_string());
    }
    let selection = BestSelection {
        strategy_id: chosen.strategy.id.clone(),
        confidence,
        improvement_bps: comparison.throughput_gain_bps.unwrap_or(0),
        tradeoffs: comparison.tradeoffs.clone(),
        reasons: vec![format!(
            "selected by {} preference from {} significant candidate(s)",
            preference.label(),
            viable.len()
        )],
        measurement: measurement.clone(),
    };
    (Some(selection), comparisons, notes)
}

pub fn build_calibration_report(
    model: &ModelProfile,
    hardware: &HardwareProfile,
    environment_fingerprint: impl Into<String>,
    baseline_id: &str,
    experiments: Vec<CalibrationExperiment>,
    preference: ObjectivePreference,
    limits: CandidateLimits,
) -> CalibrationReport {
    let (best, comparisons, notes) = select_best(&experiments, baseline_id, preference, &limits);
    CalibrationReport {
        schema_version: ADAPTIVE_SCHEMA_VERSION,
        model_fingerprint: model.fingerprint.clone(),
        hardware_fingerprint: hardware.fingerprint.clone(),
        environment_fingerprint: environment_fingerprint.into(),
        baseline_id: Some(baseline_id.to_string()),
        experiments,
        comparisons,
        best,
        preference,
        limits,
        notes,
        limitations: vec![
            "short benchmarks are not universal performance guarantees".to_string(),
            "calibration measures the requested workload only; results are not reused for a different model, environment, or workload".to_string(),
            "budgets remain advisory and are not OS-enforced".to_string(),
        ],
        created_at_unix_ms: now_unix_ms(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive::environment::ExecutionEnvironment;
    use crate::model_architecture::ArchitectureKind;

    fn model() -> ModelProfile {
        ModelProfile {
            schema_version: 2,
            fingerprint: Some("model".to_string()),
            path: None,
            architecture: Some("qwen2".to_string()),
            kind: ArchitectureKind::Dense,
            parameters: Some(1),
            file_bytes: 2 << 30,
            tensor_payload_bytes: Some(2 << 30),
            tensor_count: 1,
            layer_count: Some(24),
            embedding_length: Some(128),
            head_count: Some(4),
            head_count_kv: Some(4),
            context_max: Some(4096),
            vocab_size: Some(100),
            tokenizer_model: Some("bpe".to_string()),
            expert_count: None,
            expert_used_count: None,
            quantization: Some("mostly-q4_k_m".to_string()),
            quantization_mix: Vec::new(),
            legacy_execution_supported: true,
            forgecore_execution_supported: true,
            unsupported_tensor_formats: Vec::new(),
            routed_expert_bytes: None,
            estimated_active_weight_bytes: None,
            notes: Vec::new(),
        }
    }

    fn hardware(gpu: bool, logical: u32) -> HardwareProfile {
        HardwareProfile {
            schema_version: 2,
            fingerprint: "hw".to_string(),
            cpu_model: Some("cpu".to_string()),
            physical_cores: Some(6),
            logical_cpus: Some(logical),
            total_ram_bytes: Some(16 << 30),
            available_ram_bytes: Some(8 << 30),
            os: "linux".to_string(),
            kernel: None,
            gpus: if gpu {
                vec![crate::profiling::GpuProfile {
                    id: "gpu0".to_string(),
                    backend: "cuda".to_string(),
                    name: "gpu".to_string(),
                    total_bytes: Some(4 << 30),
                    free_bytes: Some(4 << 30),
                    selectable: true,
                }]
            } else {
                Vec::new()
            },
            backends: vec!["cpu".to_string()],
            storage_kind: None,
            storage_read_bytes_per_second: None,
            cgroup_limit_bytes: None,
            execution_environment: ExecutionEnvironment {
                schema_version: 1,
                os: "linux".to_string(),
                kernel: None,
                ramforge_version: "0.1.0".to_string(),
                native_contract_version: 1,
                native_revision: None,
                backend_build: "cpu".to_string(),
                gpu_driver_version: None,
                cpu_feature_fingerprint: None,
                fingerprint: "0000000000000000".to_string(),
            },
            notes: Vec::new(),
        }
    }

    fn workload() -> WorkloadProfile {
        WorkloadProfile {
            context_tokens: 256,
            batch_tokens: 64,
            ubatch_tokens: 64,
            sequence_max: 1,
            threads: 6,
            gpu_layers: 0,
            kv_type_k: "f16".to_string(),
            kv_type_v: "f16".to_string(),
            load_mode: "mmap".to_string(),
            backend: "cpu".to_string(),
            device_id: "cpu".to_string(),
            residency: "off".to_string(),
            draft_model_fingerprint: None,
            draft_tokens: None,
            max_tokens: Some(8),
            ram_budget_bytes: Some(10 << 30),
        }
    }

    fn experiment(id: &str, tps: f64, min: f64, max: f64, ttft: u64) -> CalibrationExperiment {
        CalibrationExperiment {
            strategy: AdaptiveStrategy {
                schema_version: ADAPTIVE_SCHEMA_VERSION,
                id: id.to_string(),
                label: id.to_string(),
                axis: StrategyAxis::Baseline,
                execution_mode: "cpu".to_string(),
                device_id: "cpu".to_string(),
                gpu_layers: 0,
                threads: 6,
                context_tokens: 256,
                batch_tokens: 64,
                ubatch_tokens: 64,
                kv_type_k: "f16".to_string(),
                kv_type_v: "f16".to_string(),
                load_mode: "mmap".to_string(),
                residency: "off".to_string(),
                staged_auto: false,
                staged_layers: Vec::new(),
                staged_budget_bytes: None,
                draft_model_path: None,
                draft_tokens: None,
                speculative: false,
            },
            status: ExperimentStatus::Completed,
            measurement: Some(MeasurementSummary {
                generation_tokens_per_second: tps,
                min_generation_tokens_per_second: min,
                max_generation_tokens_per_second: max,
                decode_tokens_per_second: Some(tps),
                prompt_tokens_per_second: Some(100.0),
                time_to_first_token_ns: Some(ttft),
                pss_bytes: Some(1 << 30),
                acceptance_rate_bps: None,
                runs: 2,
                condition: "warm".to_string(),
                variability_bps: Some(500),
            }),
            error: None,
            notes: Vec::new(),
        }
    }

    #[test]
    fn baseline_is_always_generated() {
        let limits = CandidateLimits::default();
        let base = baseline_strategy(&workload());
        let set = generate_candidates(
            &model(),
            &hardware(false, 12),
            &workload(),
            &base,
            None,
            None,
            None,
            &limits,
        );
        assert_eq!(set.candidates[0].strategy.id, set.baseline_id);
        assert!(set
            .candidates
            .iter()
            .any(|candidate| candidate.strategy.axis == StrategyAxis::ThreadCount));
    }

    #[test]
    fn gpu_candidate_is_rejected_without_a_device() {
        let limits = CandidateLimits::default();
        let base = baseline_strategy(&workload());
        let set = generate_candidates(
            &model(),
            &hardware(false, 12),
            &workload(),
            &base,
            None,
            None,
            None,
            &limits,
        );
        assert!(!set
            .candidates
            .iter()
            .any(|candidate| candidate.strategy.axis == StrategyAxis::GpuPlacement));
    }

    #[test]
    fn gpu_candidate_appears_with_a_device() {
        let limits = CandidateLimits::default();
        let base = baseline_strategy(&workload());
        let set = generate_candidates(
            &model(),
            &hardware(true, 12),
            &workload(),
            &base,
            None,
            None,
            None,
            &limits,
        );
        assert!(set
            .candidates
            .iter()
            .any(|candidate| candidate.strategy.axis == StrategyAxis::GpuPlacement));
    }

    #[test]
    fn selection_requires_a_significant_improvement() {
        let limits = CandidateLimits::default();
        let experiments = vec![
            experiment("baseline", 1.0, 0.95, 1.05, 1_000),
            experiment("small-gain", 1.02, 0.97, 1.07, 1_000),
        ];
        let (best, _, notes) = select_best(
            &experiments,
            "baseline",
            ObjectivePreference::Throughput,
            &limits,
        );
        assert!(best.is_none());
        assert!(notes.iter().any(|note| note.contains("threshold")));
    }

    #[test]
    fn selection_accepts_separated_improvement() {
        let limits = CandidateLimits::default();
        let experiments = vec![
            experiment("baseline", 1.0, 0.95, 1.05, 1_000),
            experiment("better", 1.5, 1.4, 1.6, 1_100),
        ];
        let (best, _, _) = select_best(
            &experiments,
            "baseline",
            ObjectivePreference::Throughput,
            &limits,
        );
        let best = best.expect("significant candidate should win");
        assert_eq!(best.strategy_id, "better");
        assert_eq!(best.confidence, "measured");
        assert!(best.improvement_bps >= 5_000);
    }

    #[test]
    fn latency_preference_can_choose_the_faster_ttft() {
        let limits = CandidateLimits::default();
        let experiments = vec![
            experiment("baseline", 1.0, 0.95, 1.05, 5_000),
            experiment("fast-first-token", 1.2, 1.15, 1.25, 1_000),
        ];
        let (best, _, _) = select_best(
            &experiments,
            "baseline",
            ObjectivePreference::Latency,
            &limits,
        );
        assert_eq!(best.unwrap().strategy_id, "fast-first-token");
    }

    #[test]
    fn report_validation_rejects_completed_without_measurement() {
        let mut experiments = vec![experiment("baseline", 1.0, 0.9, 1.1, 1_000)];
        experiments[0].measurement = None;
        let report = build_calibration_report(
            &model(),
            &hardware(false, 12),
            "env",
            "baseline",
            experiments,
            ObjectivePreference::Balanced,
            CandidateLimits::default(),
        );
        assert!(report.validate().is_err());
    }
}
