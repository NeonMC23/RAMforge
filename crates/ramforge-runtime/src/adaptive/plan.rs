//! Explainable execution plans with explicit confirmation and readback.
//!
//! A recommendation is not a plan, a plan is not applied, and a successful
//! configuration request is not proof of effective execution. This module keeps
//! those stages distinct and records exactly which settings were verified by
//! native readback or runtime evidence.

use serde::{Deserialize, Serialize};

use crate::adaptive::capability::CompatibilityReport;
use crate::adaptive::engine::{AdaptiveStrategy, MeasurementSummary};
use crate::adaptive::environment::ExecutionEnvironment;
use crate::adaptive::resource::ResourceBudget;
use crate::benchmark_runner::BenchmarkReport;
use crate::profiling::{fnv_hex, now_unix_ms, HardwareProfile, ModelProfile, WorkloadProfile};

pub const ADAPTIVE_PLAN_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    Proposed,
    Confirmed,
    Applied,
    Verified,
    PartiallyApplied,
    Rejected,
    Stale,
    Failed,
}

impl PlanStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Proposed => "proposed",
            Self::Confirmed => "confirmed",
            Self::Applied => "applied",
            Self::Verified => "verified",
            Self::PartiallyApplied => "partially-applied",
            Self::Rejected => "rejected",
            Self::Stale => "stale",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Evidence {
    GgufMetadata,
    MachineObservation,
    NativeBackend,
    SupportRegistry,
    Measurement,
    HeuristicEstimate,
    UserOverride,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanDecision {
    pub decision: String,
    pub value: String,
    pub evidence: Vec<Evidence>,
    pub explanation: String,
    pub confidence: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdaptivePlan {
    pub schema_version: u32,
    pub plan_id: String,
    pub created_at_unix_ms: Option<u64>,
    pub model_path: String,
    pub model_fingerprint: Option<String>,
    pub hardware_fingerprint: String,
    pub environment: ExecutionEnvironment,
    pub workload: WorkloadProfile,
    pub strategy: AdaptiveStrategy,
    pub draft_model_path: Option<String>,
    pub draft_model_fingerprint: Option<String>,
    pub budget: ResourceBudget,
    pub compatibility: CompatibilityReport,
    pub decisions: Vec<PlanDecision>,
    pub risks: Vec<String>,
    pub unknowns: Vec<String>,
    /// This build never applies a plan without `--confirm`.
    pub requires_confirmation: bool,
    pub status: PlanStatus,
    pub performance_profile: Option<String>,
    pub prompt: String,
    pub max_tokens: u32,
    pub runs: u32,
    pub warmup_runs: u32,
}

impl AdaptivePlan {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != ADAPTIVE_PLAN_SCHEMA_VERSION {
            return Err(format!(
                "unsupported adaptive plan schema {}",
                self.schema_version
            ));
        }
        self.environment.validate()?;
        self.strategy.validate()?;
        self.budget.validate()?;
        self.compatibility.validate()?;
        if !self.requires_confirmation {
            return Err("adaptive plans must require explicit confirmation".to_string());
        }
        if self.plan_id.len() != 16 || !self.plan_id.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err("adaptive plan id is malformed".to_string());
        }
        if self.prompt.trim().is_empty() {
            return Err("adaptive plan prompt is empty".to_string());
        }
        if self.runs == 0 {
            return Err("adaptive plan has no measured runs".to_string());
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(|error| error.to_string())
    }
}

fn plan_identity(
    model: &ModelProfile,
    hardware: &HardwareProfile,
    environment: &ExecutionEnvironment,
    strategy: &AdaptiveStrategy,
    workload: &WorkloadProfile,
) -> String {
    let mut text = String::new();
    for part in [
        model.fingerprint.clone().unwrap_or_default(),
        hardware.fingerprint.clone(),
        environment.fingerprint.clone(),
        strategy.key(),
        workload.context_tokens.to_string(),
        workload.max_tokens.unwrap_or(0).to_string(),
    ] {
        text.push_str(&part);
        text.push('\u{1f}');
    }
    fnv_hex(text.as_bytes())
}

#[allow(clippy::too_many_arguments)]
pub fn build_plan(
    model: &ModelProfile,
    hardware: &HardwareProfile,
    environment: &ExecutionEnvironment,
    workload: &WorkloadProfile,
    strategy: &AdaptiveStrategy,
    budget: &ResourceBudget,
    compatibility: &CompatibilityReport,
    draft_model_path: Option<String>,
    draft_model_fingerprint: Option<String>,
    performance_profile: Option<String>,
    prompt: String,
    max_tokens: u32,
    runs: u32,
    warmup_runs: u32,
) -> AdaptivePlan {
    let plan_id = plan_identity(model, hardware, environment, strategy, workload);
    let mut decisions = vec![
        PlanDecision {
            decision: "execution-placement".to_string(),
            value: strategy.execution_mode.clone(),
            evidence: vec![Evidence::MachineObservation, Evidence::NativeBackend],
            explanation: format!(
                "backend={} device={} gpu_layers={}; effective placement is confirmed after load",
                strategy.execution_mode, strategy.device_id, strategy.gpu_layers
            ),
            confidence: "measured-readback-required".to_string(),
        },
        PlanDecision {
            decision: "cpu-threads".to_string(),
            value: strategy.threads.to_string(),
            evidence: vec![Evidence::MachineObservation],
            explanation:
                "selected from observed CPU topology; native readback confirms the effective count"
                    .to_string(),
            confidence: "observed-topology".to_string(),
        },
        PlanDecision {
            decision: "kv-cache-type".to_string(),
            value: format!("K={} V={}", strategy.kv_type_k, strategy.kv_type_v),
            evidence: vec![Evidence::NativeBackend],
            explanation:
                "forwarded to the native context; no KV dtype getter at the pinned revision"
                    .to_string(),
            confidence: "forwarded-only".to_string(),
        },
        PlanDecision {
            decision: "residency".to_string(),
            value: strategy.residency.clone(),
            evidence: vec![Evidence::NativeBackend, Evidence::HeuristicEstimate],
            explanation: if strategy.staged_auto {
                "bounded staging is requested; staged layers are reported after generation"
                    .to_string()
            } else {
                "no staged residency requested".to_string()
            },
            confidence: "measured-readback-required".to_string(),
        },
        PlanDecision {
            decision: "memory-policy".to_string(),
            value: format!("load={} repack={}", strategy.load_mode, !strategy.no_repack),
            evidence: vec![Evidence::Measurement, Evidence::HeuristicEstimate],
            explanation: if strategy.no_repack {
                "repacking and whole-file prefaulting are disabled for a constrained model; measured to reduce PSS and target-forward time on this machine".to_string()
            } else {
                "default repacking and prefetch are retained; the model has comfortable memory headroom".to_string()
            },
            confidence: "measured-policy".to_string(),
        },
        PlanDecision {
            decision: "resource-admission".to_string(),
            value: budget.admission.label().to_string(),
            evidence: vec![Evidence::GgufMetadata, Evidence::MachineObservation],
            explanation: budget
                .reasons
                .first()
                .cloned()
                .unwrap_or_else(|| "no budget reason recorded".to_string()),
            confidence: "advisory-budget".to_string(),
        },
    ];
    if strategy.speculative {
        decisions.push(PlanDecision {
            decision: "speculative".to_string(),
            value: format!(
                "greedy K={} draft={}",
                strategy.draft_tokens.unwrap_or(0),
                draft_model_path.as_deref().unwrap_or("unavailable")
            ),
            evidence: vec![Evidence::SupportRegistry, Evidence::GgufMetadata],
            explanation:
                "greedy-only speculative decoding; exact token ids are verified against the target"
                    .to_string(),
            confidence: "measured-acceptance-required".to_string(),
        });
    }
    let mut risks = Vec::new();
    if compatibility.overall != crate::adaptive::capability::SupportClass::FullySupported {
        risks.push(format!(
            "compatibility is {}: {}",
            compatibility.overall.label(),
            compatibility
                .reasons
                .first()
                .cloned()
                .unwrap_or_else(|| "see compatibility report".to_string())
        ));
    }
    if budget.admission == crate::adaptive::resource::BudgetClass::Risky {
        risks.push("estimated resource budget is risky".to_string());
    }
    if strategy.staged_auto {
        risks.push("staged residency can add storage reads and latency".to_string());
    }
    if strategy.speculative {
        risks.push(
            "speculative decoding raises TTFT while waiting for target verification".to_string(),
        );
    }
    if strategy.kv_type_k != "f16" || strategy.kv_type_v != "f16" {
        risks.push("quantized KV cache trades numerical fidelity for memory".to_string());
    }
    let unknowns = budget
        .unknowns
        .iter()
        .chain(compatibility.unknowns.iter())
        .cloned()
        .collect();

    AdaptivePlan {
        schema_version: ADAPTIVE_PLAN_SCHEMA_VERSION,
        plan_id,
        created_at_unix_ms: now_unix_ms(),
        model_path: model.path.clone().unwrap_or_default(),
        model_fingerprint: model.fingerprint.clone(),
        hardware_fingerprint: hardware.fingerprint.clone(),
        environment: environment.clone(),
        workload: workload.clone(),
        strategy: strategy.clone(),
        draft_model_path,
        draft_model_fingerprint,
        budget: budget.clone(),
        compatibility: compatibility.clone(),
        decisions,
        risks,
        unknowns,
        requires_confirmation: true,
        status: PlanStatus::Proposed,
        performance_profile,
        prompt,
        max_tokens,
        runs,
        warmup_runs,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanValidation {
    pub valid: bool,
    pub stale: bool,
    pub reasons: Vec<String>,
    pub environment_changes: Vec<String>,
}

/// Revalidate a serialized plan against the current model, machine, and
/// execution environment. The caller must refuse application when this fails.
pub fn revalidate_plan(
    plan: &AdaptivePlan,
    model: &ModelProfile,
    hardware: &HardwareProfile,
    environment: &ExecutionEnvironment,
) -> PlanValidation {
    let mut reasons = Vec::new();
    if let Err(error) = plan.validate() {
        reasons.push(format!("plan is invalid: {error}"));
    }
    if plan.model_fingerprint != model.fingerprint {
        reasons.push(format!(
            "model changed: plan {:?} vs current {:?}",
            plan.model_fingerprint, model.fingerprint
        ));
    }
    if plan.hardware_fingerprint != hardware.fingerprint {
        reasons.push(format!(
            "hardware changed: plan {} vs current {}",
            plan.hardware_fingerprint, hardware.fingerprint
        ));
    }
    let environment_changes =
        crate::adaptive::environment::environment_changes(&plan.environment, environment);
    if plan.environment.fingerprint != environment.fingerprint {
        reasons.push("execution environment changed".to_string());
    }
    PlanValidation {
        valid: reasons.is_empty(),
        stale: !plan.model_fingerprint.eq(&model.fingerprint)
            || plan.hardware_fingerprint != hardware.fingerprint
            || plan.environment.fingerprint != environment.fingerprint,
        reasons,
        environment_changes,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldState {
    Requested,
    /// The native configuration request was accepted, but there is no getter
    /// to verify the effective value.
    Accepted,
    Applied,
    Verified,
    Unsupported,
    Rejected,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldObservation {
    pub field: String,
    pub requested: String,
    pub effective: Option<String>,
    pub state: FieldState,
    pub evidence: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceComparison {
    pub host_estimate_bytes: Option<u64>,
    pub host_observed_pss_bytes: Option<u64>,
    /// Signed error in basis points relative to the estimate. Negative means
    /// the estimate was conservative.
    pub host_error_bps: Option<i64>,
    pub vram_estimate_bytes: Option<u64>,
    pub vram_observed_bytes: Option<u64>,
    pub vram_error_bps: Option<i64>,
    pub storage_read_bytes: Option<u64>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppliedPlanReport {
    pub schema_version: u32,
    pub plan_id: String,
    pub applied_at_unix_ms: Option<u64>,
    pub status: PlanStatus,
    pub model_fingerprint: Option<String>,
    pub hardware_fingerprint: String,
    pub environment_fingerprint: String,
    pub fields: Vec<FieldObservation>,
    pub measurement: Option<MeasurementSummary>,
    pub resource_comparison: Option<ResourceComparison>,
    pub notes: Vec<String>,
    pub limitations: Vec<String>,
}

impl AppliedPlanReport {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != ADAPTIVE_PLAN_SCHEMA_VERSION {
            return Err(format!(
                "unsupported applied-report schema {}",
                self.schema_version
            ));
        }
        if self.plan_id.len() != 16 || !self.plan_id.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err("applied report plan id is malformed".to_string());
        }
        if self.fields.is_empty() {
            return Err("applied report has no field observations".to_string());
        }
        Ok(())
    }
}

/// Runtime facts extracted from a benchmark run or a future service readback.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeObservation {
    pub generation_tokens_per_second: f64,
    pub time_to_first_token_ns: Option<u64>,
    pub acceptance_rate_bps: Option<u32>,
    pub effective_execution_mode: Option<String>,
    pub effective_gpu_layers: Option<u32>,
    pub effective_threads: Option<u32>,
    pub effective_context_tokens: Option<u32>,
    pub effective_batch_tokens: Option<u32>,
    pub effective_ubatch_tokens: Option<u32>,
    pub output_device_type: Option<String>,
    pub staged_layer_count: usize,
    pub draft_model_present: bool,
    pub speculative_observed: bool,
    pub kv_forwarded: bool,
    #[serde(default)]
    pub pss_bytes: Option<u64>,
    #[serde(default)]
    pub staged_vram_bytes: Option<u64>,
    #[serde(default)]
    pub storage_read_bytes: Option<u64>,
    pub warnings: Vec<String>,
}

impl RuntimeObservation {
    pub fn from_benchmark(report: &BenchmarkReport) -> Self {
        let speculative = report
            .runs_detail
            .last()
            .and_then(|run| run.speculative.as_ref());
        Self {
            generation_tokens_per_second: report.median_generation_tokens_per_second,
            time_to_first_token_ns: report.median_time_to_first_token_ns,
            acceptance_rate_bps: speculative.map(|stats| stats.acceptance_rate_bps),
            effective_execution_mode: Some(
                format!("{:?}", report.plan.effective.execution_mode).to_ascii_lowercase(),
            ),
            effective_gpu_layers: report.effective_gpu_layers,
            effective_threads: Some(report.plan.effective.threads),
            effective_context_tokens: Some(report.plan.effective.context_tokens),
            effective_batch_tokens: Some(report.plan.effective.batch_tokens),
            effective_ubatch_tokens: Some(report.plan.effective.ubatch_tokens),
            output_device_type: report.output_device_type.clone(),
            staged_layer_count: report.staged_layers.len(),
            draft_model_present: report.draft_model.is_some(),
            speculative_observed: speculative.is_some(),
            kv_forwarded: true,
            pss_bytes: report.runs_detail.last().and_then(|run| run.pss_bytes),
            staged_vram_bytes: report
                .staging_lifecycle
                .as_ref()
                .map(|lifecycle| lifecycle.resident_bytes),
            storage_read_bytes: report
                .runs_detail
                .last()
                .and_then(|run| run.storage_read_bytes),
            warnings: report.limitations.clone(),
        }
    }
}

/// Build the applied/verified report. Every requested field is explicitly
/// classified; a missing observation never becomes a success claim.
pub fn applied_report_from_observation(
    plan: &AdaptivePlan,
    observation: &RuntimeObservation,
) -> AppliedPlanReport {
    let mut fields = Vec::new();
    let strategy = &plan.strategy;

    let mode_state = match observation.effective_execution_mode.as_deref() {
        Some(mode) if mode == strategy.execution_mode => FieldState::Verified,
        // A staged plan is hybrid in resource terms (CPU target plus GPU
        // staged layers), while the target runtime plan correctly reports a
        // CPU execution mode. Both facts must be verified together.
        Some("cpu")
            if strategy.execution_mode == "hybrid"
                && strategy.staged_auto
                && observation.staged_layer_count > 0 =>
        {
            FieldState::Verified
        }
        Some(_) => FieldState::Applied,
        None => FieldState::Unknown,
    };
    fields.push(FieldObservation {
        field: "execution_mode".to_string(),
        requested: strategy.execution_mode.clone(),
        effective: observation.effective_execution_mode.clone(),
        state: mode_state,
        evidence: Some(if strategy.execution_mode == "hybrid" {
            "CPU target plan plus verified staged GPU layers".to_string()
        } else {
            "runtime plan and native device readback".to_string()
        }),
    });

    let gpu_state = if strategy.gpu_layers == 0 {
        match observation.effective_gpu_layers {
            Some(0) => FieldState::Verified,
            Some(_) => FieldState::Rejected,
            None => FieldState::Unknown,
        }
    } else {
        match observation.effective_gpu_layers {
            Some(effective) if effective == strategy.gpu_layers => FieldState::Verified,
            Some(effective) if effective > 0 => FieldState::Applied,
            Some(_) => FieldState::Rejected,
            None => FieldState::Unknown,
        }
    };
    fields.push(FieldObservation {
        field: "gpu_layers".to_string(),
        requested: strategy.gpu_layers.to_string(),
        effective: observation
            .effective_gpu_layers
            .map(|value| value.to_string()),
        state: gpu_state,
        evidence: Some("native placement readback after model load".to_string()),
    });

    let mut compare_u32 = |field: &str, requested: u32, effective: Option<u32>| {
        let state = match effective {
            Some(value) if value == requested => FieldState::Verified,
            Some(_) => FieldState::Applied,
            None => FieldState::Unknown,
        };
        fields.push(FieldObservation {
            field: field.to_string(),
            requested: requested.to_string(),
            effective: effective.map(|value| value.to_string()),
            state,
            evidence: Some("effective runtime plan after resolution".to_string()),
        });
    };
    compare_u32(
        "context_tokens",
        strategy.context_tokens,
        observation.effective_context_tokens,
    );
    compare_u32(
        "batch_tokens",
        strategy.batch_tokens,
        observation.effective_batch_tokens,
    );
    compare_u32(
        "ubatch_tokens",
        strategy.ubatch_tokens,
        observation.effective_ubatch_tokens,
    );
    compare_u32("threads", strategy.threads, observation.effective_threads);

    let kv_state = if observation.kv_forwarded {
        FieldState::Accepted
    } else {
        FieldState::Unknown
    };
    fields.push(FieldObservation {
        field: "kv_type".to_string(),
        requested: format!("K={} V={}", strategy.kv_type_k, strategy.kv_type_v),
        effective: observation
            .kv_forwarded
            .then(|| format!("K={} V={}", strategy.kv_type_k, strategy.kv_type_v)),
        state: kv_state,
        evidence: Some(
            "forwarded to the native context; no native KV dtype getter at the pinned revision"
                .to_string(),
        ),
    });

    let device_state = if observation.output_device_type.is_some() {
        FieldState::Verified
    } else {
        FieldState::Unknown
    };
    fields.push(FieldObservation {
        field: "device".to_string(),
        requested: strategy.device_id.clone(),
        effective: observation.output_device_type.clone(),
        state: device_state,
        evidence: Some("native output device type".to_string()),
    });

    if strategy.staged_auto || strategy.residency == "staged" {
        let state = if observation.staged_layer_count > 0 {
            FieldState::Verified
        } else {
            FieldState::Unknown
        };
        fields.push(FieldObservation {
            field: "staged_layers".to_string(),
            requested: "bounded staged residency".to_string(),
            effective: Some(format!(
                "{} staged layer(s)",
                observation.staged_layer_count
            )),
            state,
            evidence: Some("staged layer readback after generation".to_string()),
        });
    }

    if strategy.speculative {
        let state = if observation.draft_model_present && observation.speculative_observed {
            FieldState::Verified
        } else if observation.draft_model_present {
            FieldState::Applied
        } else {
            FieldState::Rejected
        };
        fields.push(FieldObservation {
            field: "speculative".to_string(),
            requested: format!("greedy K={}", strategy.draft_tokens.unwrap_or(0)),
            effective: observation
                .acceptance_rate_bps
                .map(|bps| format!("draft present; acceptance={bps} bps")),
            state,
            evidence: Some("speculative round statistics".to_string()),
        });
    }

    let any_verified = fields
        .iter()
        .any(|field| matches!(field.state, FieldState::Verified | FieldState::Applied));
    let any_failed = fields
        .iter()
        .any(|field| matches!(field.state, FieldState::Rejected | FieldState::Unsupported));
    let all_verified_or_accepted = fields
        .iter()
        .all(|field| matches!(field.state, FieldState::Verified | FieldState::Accepted));
    let status = if any_failed && !any_verified {
        PlanStatus::Rejected
    } else if any_failed || !all_verified_or_accepted {
        PlanStatus::PartiallyApplied
    } else {
        PlanStatus::Verified
    };

    let measurement = MeasurementSummary {
        generation_tokens_per_second: observation.generation_tokens_per_second,
        min_generation_tokens_per_second: observation.generation_tokens_per_second,
        max_generation_tokens_per_second: observation.generation_tokens_per_second,
        decode_tokens_per_second: None,
        prompt_tokens_per_second: None,
        time_to_first_token_ns: observation.time_to_first_token_ns,
        pss_bytes: None,
        acceptance_rate_bps: observation.acceptance_rate_bps,
        target_forward_ns: None,
        draft_forward_ns: None,
        cpu_busy_bps: None,
        staged_vram_bytes: None,
        runs: 1,
        condition: "plan-application-verification".to_string(),
        variability_bps: None,
    };

    let staged_window = plan
        .budget
        .components
        .iter()
        .find(|component| component.name == "staged-vram-window")
        .and_then(|component| component.bytes);
    let error_bps = |estimate: Option<u64>, observed: Option<u64>| match (estimate, observed) {
        (Some(estimate), Some(observed)) if estimate > 0 => {
            Some(((observed as f64 - estimate as f64) / estimate as f64 * 10_000.0).round() as i64)
        }
        _ => None,
    };
    let host_error_bps = error_bps(plan.budget.host_required_bytes, observation.pss_bytes);
    let vram_error_bps = error_bps(staged_window, observation.staged_vram_bytes);
    let mut resource_notes = Vec::new();
    if host_error_bps.is_some_and(|error| error < -2_000) {
        resource_notes.push(
            "host estimate is more than 20% above observed PSS; the budget is conservative"
                .to_string(),
        );
    }
    if host_error_bps.is_some_and(|error| error > 1_000) {
        resource_notes.push(
            "observed PSS exceeded the host estimate; tighten admission or increase overhead"
                .to_string(),
        );
    }
    if observation.pss_bytes.is_none() {
        resource_notes.push("PSS was not observed".to_string());
    }
    if observation.staged_vram_bytes.is_none() && strategy.staged_auto {
        resource_notes.push("staged VRAM bytes were not observed".to_string());
    }
    let resource_comparison = ResourceComparison {
        host_estimate_bytes: plan.budget.host_required_bytes,
        host_observed_pss_bytes: observation.pss_bytes,
        host_error_bps,
        vram_estimate_bytes: staged_window,
        vram_observed_bytes: observation.staged_vram_bytes,
        vram_error_bps,
        storage_read_bytes: observation.storage_read_bytes,
        notes: resource_notes,
    };

    AppliedPlanReport {
        schema_version: ADAPTIVE_PLAN_SCHEMA_VERSION,
        plan_id: plan.plan_id.clone(),
        applied_at_unix_ms: now_unix_ms(),
        status,
        model_fingerprint: plan.model_fingerprint.clone(),
        hardware_fingerprint: plan.hardware_fingerprint.clone(),
        environment_fingerprint: plan.environment.fingerprint.clone(),
        fields,
        measurement: Some(measurement),
        resource_comparison: Some(resource_comparison),
        notes: observation.warnings.clone(),
        limitations: vec![
            "KV dtype is forwarded-only; the pinned native revision has no getter".to_string(),
            "verification is scoped to the current model, environment, and workload".to_string(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive::capability::{resolve_compatibility, CompatibilityRequest};
    use crate::adaptive::engine::baseline_strategy;
    use crate::adaptive::environment::ExecutionEnvironment;
    use crate::adaptive::resource::assess_budget;
    use crate::model_architecture::ArchitectureKind;

    fn environment(fingerprint: &str) -> ExecutionEnvironment {
        ExecutionEnvironment {
            schema_version: 1,
            os: "linux".to_string(),
            kernel: Some("kernel".to_string()),
            ramforge_version: "0.1.0".to_string(),
            native_contract_version: 1,
            native_revision: None,
            backend_build: "cpu".to_string(),
            gpu_driver_version: None,
            cpu_feature_fingerprint: None,
            fingerprint: fingerprint.to_string(),
        }
    }

    fn model(fingerprint: &str) -> ModelProfile {
        ModelProfile {
            schema_version: 2,
            fingerprint: Some(fingerprint.to_string()),
            path: Some("/tmp/model.gguf".to_string()),
            architecture: Some("qwen2".to_string()),
            kind: ArchitectureKind::Dense,
            parameters: Some(1),
            file_bytes: 1024,
            tensor_payload_bytes: Some(1024),
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

    fn hardware(fingerprint: &str) -> HardwareProfile {
        HardwareProfile {
            schema_version: 2,
            fingerprint: fingerprint.to_string(),
            cpu_model: Some("cpu".to_string()),
            physical_cores: Some(6),
            logical_cpus: Some(12),
            total_ram_bytes: Some(16 << 30),
            available_ram_bytes: Some(8 << 30),
            os: "linux".to_string(),
            kernel: None,
            gpus: Vec::new(),
            backends: vec!["cpu".to_string()],
            storage_kind: None,
            storage_read_bytes_per_second: None,
            cgroup_limit_bytes: None,
            execution_environment: environment("0000000000000000"),
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
            no_repack: false,
            backend: "cpu".to_string(),
            device_id: "cpu".to_string(),
            residency: "off".to_string(),
            draft_model_fingerprint: None,
            draft_tokens: None,
            max_tokens: Some(8),
            ram_budget_bytes: Some(10 << 30),
        }
    }

    fn plan() -> AdaptivePlan {
        let model = model("model");
        let hardware = hardware("hw");
        let environment = environment("0123456789abcdef");
        let workload = workload();
        let strategy = baseline_strategy(&workload);
        let request = CompatibilityRequest::from_workload(&workload, None);
        let compatibility = resolve_compatibility(&model, &hardware, &workload, &request);
        let budget = assess_budget(&model, &hardware, &workload, None, None);
        build_plan(
            &model,
            &hardware,
            &environment,
            &workload,
            &strategy,
            &budget,
            &compatibility,
            None,
            None,
            None,
            "prompt".to_string(),
            8,
            2,
            0,
        )
    }

    fn observation() -> RuntimeObservation {
        RuntimeObservation {
            generation_tokens_per_second: 1.0,
            time_to_first_token_ns: Some(1_000),
            acceptance_rate_bps: None,
            effective_execution_mode: Some("cpu".to_string()),
            effective_gpu_layers: Some(0),
            effective_threads: Some(6),
            effective_context_tokens: Some(256),
            effective_batch_tokens: Some(64),
            effective_ubatch_tokens: Some(64),
            output_device_type: Some("CPU".to_string()),
            staged_layer_count: 0,
            draft_model_present: false,
            speculative_observed: false,
            kv_forwarded: true,
            pss_bytes: Some(12 << 30),
            staged_vram_bytes: Some(3 << 30),
            storage_read_bytes: Some(64 << 30),
            warnings: Vec::new(),
        }
    }

    #[test]
    fn plan_requires_confirmation_and_validates() {
        let plan = plan();
        assert!(plan.requires_confirmation);
        assert_eq!(plan.status, PlanStatus::Proposed);
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn changed_model_or_environment_makes_the_plan_stale() {
        let plan = plan();
        let validation = revalidate_plan(
            &plan,
            &model("different"),
            &hardware("hw"),
            &environment("0123456789abcdef"),
        );
        assert!(!validation.valid);
        assert!(validation.stale);
        let validation = revalidate_plan(
            &plan,
            &model("model"),
            &hardware("hw"),
            &environment("fedcba9876543210"),
        );
        assert!(!validation.valid);
        assert!(validation.stale);
        assert!(validation
            .reasons
            .iter()
            .any(|reason| reason.contains("environment changed")));
    }

    #[test]
    fn matching_identity_passes_revalidation() {
        let plan = plan();
        let validation = revalidate_plan(
            &plan,
            &model("model"),
            &hardware("hw"),
            &environment("0123456789abcdef"),
        );
        assert!(validation.valid, "{:?}", validation.reasons);
        assert!(!validation.stale);
    }

    #[test]
    fn applied_report_verifies_readback_fields() {
        let mut plan = plan();
        plan.budget.host_required_bytes = Some(10 << 30);
        let report = applied_report_from_observation(&plan, &observation());
        assert_eq!(report.status, PlanStatus::Verified);
        assert!(report
            .fields
            .iter()
            .any(|field| field.field == "gpu_layers" && field.state == FieldState::Verified));
        assert!(report
            .fields
            .iter()
            .any(|field| field.field == "kv_type" && field.state == FieldState::Accepted));
        let comparison = report.resource_comparison.as_ref().expect("comparison");
        assert!(comparison.host_error_bps.is_some());
        assert_eq!(comparison.host_observed_pss_bytes, Some(12 << 30));
        assert!(report.validate().is_ok());
    }

    #[test]
    fn staged_hybrid_plan_verifies_cpu_target_plus_gpu_layers() {
        let mut plan = plan();
        plan.strategy.execution_mode = "hybrid".to_string();
        plan.strategy.residency = "staged".to_string();
        plan.strategy.staged_auto = true;
        let mut observation = observation();
        observation.effective_execution_mode = Some("cpu".to_string());
        observation.staged_layer_count = 11;
        let report = applied_report_from_observation(&plan, &observation);
        assert_eq!(report.status, PlanStatus::Verified);
        assert!(report.fields.iter().any(|field| {
            field.field == "execution_mode" && field.state == FieldState::Verified
        }));
        assert!(report.fields.iter().any(|field| {
            field.field == "staged_layers" && field.state == FieldState::Verified
        }));
    }

    #[test]
    fn requested_gpu_that_did_not_apply_is_not_a_success() {
        let mut plan = plan();
        plan.strategy.gpu_layers = 32;
        plan.strategy.execution_mode = "gpu".to_string();
        let mut observation = observation();
        observation.effective_execution_mode = Some("gpu".to_string());
        observation.effective_gpu_layers = Some(0);
        let report = applied_report_from_observation(&plan, &observation);
        assert_eq!(report.status, PlanStatus::PartiallyApplied);
        assert!(report
            .fields
            .iter()
            .any(|field| field.field == "gpu_layers" && field.state == FieldState::Rejected));
    }
}
