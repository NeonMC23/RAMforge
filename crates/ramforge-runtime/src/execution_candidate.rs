//! Candidate execution plans and conservative feasibility analysis.
//!
//! This is not a scheduler. It constructs inspectable plans, evaluates only the
//! constraints supported by observed data, and preserves missing information.

use serde::{Deserialize, Serialize};

use crate::machine_profile::MachineProfile;
use crate::model_requirements::{ModelRequirements, RequirementBytes};
use crate::runtime_calibration::CalibrationEvidence;

pub const EXECUTION_CANDIDATE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanProfile {
    Performance,
    Balanced,
    MemoryConstrained,
    MaximumModelSize,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MappingStrategy {
    FileBacked,
    ResidentWeights,
    Streaming,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanBackend {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Feasibility {
    Feasible,
    FeasibleWithConstraints,
    ConditionallyFeasible,
    Infeasible,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanReason {
    pub code: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionPlanCandidate {
    pub schema_version: u32,
    pub id: String,
    pub profile: PlanProfile,
    pub backend: PlanBackend,
    pub device_id: Option<String>,
    pub cpu_threads: u32,
    pub context_tokens: u32,
    pub batch_tokens: u32,
    pub ram_budget_bytes: Option<u64>,
    pub vram_budget_bytes: Option<u64>,
    pub mapping: MappingStrategy,
    pub assumptions: Vec<String>,
    pub unknown_information: Vec<String>,
}

impl ExecutionPlanCandidate {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != EXECUTION_CANDIDATE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported candidate schema {}",
                self.schema_version
            ));
        }
        if self.cpu_threads == 0 {
            return Err("cpu_threads must be greater than zero".to_string());
        }
        if self.context_tokens == 0 {
            return Err("context_tokens must be greater than zero".to_string());
        }
        if self.batch_tokens == 0 {
            return Err("batch_tokens must be greater than zero".to_string());
        }
        if self.batch_tokens > self.context_tokens {
            return Err("batch_tokens cannot exceed context_tokens".to_string());
        }
        if self.ram_budget_bytes == Some(0) {
            return Err("ram_budget_bytes cannot be zero".to_string());
        }
        if self.vram_budget_bytes == Some(0) {
            return Err("vram_budget_bytes cannot be zero".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanEvaluation {
    pub plan_id: String,
    pub profile: PlanProfile,
    pub feasibility: Feasibility,
    pub estimated_ram: RequirementBytes,
    pub estimated_vram: RequirementBytes,
    pub reasons: Vec<PlanReason>,
    pub unknown_information: Vec<String>,
    pub storage_dependency: bool,
}

impl PlanEvaluation {
    pub fn is_known_feasible(&self) -> bool {
        matches!(
            self.feasibility,
            Feasibility::Feasible | Feasibility::FeasibleWithConstraints
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanComparison {
    pub schema_version: u32,
    pub dimensions: Vec<String>,
    pub evaluations: Vec<PlanEvaluation>,
    pub note: String,
}

impl PlanComparison {
    pub fn to_json(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(|error| error.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateRequest {
    pub context_tokens: u32,
    pub batch_tokens: u32,
    pub ram_budget_bytes: Option<u64>,
    pub vram_budget_bytes: Option<u64>,
    pub selected_threads: Option<u32>,
}

impl Default for CandidateRequest {
    fn default() -> Self {
        Self {
            context_tokens: 2048,
            batch_tokens: 1,
            ram_budget_bytes: None,
            vram_budget_bytes: None,
            selected_threads: None,
        }
    }
}

/// Construct the initial, non-automatic candidate plan set.
pub fn candidate_plans(
    machine: &MachineProfile,
    requirements: &ModelRequirements,
    request: CandidateRequest,
) -> Vec<ExecutionPlanCandidate> {
    let available = machine
        .cpu
        .available_threads
        .as_ref()
        .copied()
        .unwrap_or(1)
        .max(1);
    let threads = request
        .selected_threads
        .unwrap_or(available)
        .min(available)
        .max(1);
    let gpu = machine
        .devices
        .iter()
        .find(|device| device.selectable && device.kind != "cpu");
    let context = request
        .context_tokens
        .min(
            requirements
                .model_max_context_tokens
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(u32::MAX),
        )
        .max(1);
    let batch = request.batch_tokens.min(context).max(1);
    let mut plans = Vec::new();
    plans.push(ExecutionPlanCandidate {
        schema_version: EXECUTION_CANDIDATE_SCHEMA_VERSION,
        id: "performance".to_string(),
        profile: PlanProfile::Performance,
        backend: if gpu.is_some() {
            PlanBackend::Gpu
        } else {
            PlanBackend::Cpu
        },
        device_id: gpu.map(|device| device.id.clone()),
        cpu_threads: threads,
        context_tokens: context,
        batch_tokens: batch,
        ram_budget_bytes: request.ram_budget_bytes,
        vram_budget_bytes: request.vram_budget_bytes,
        mapping: MappingStrategy::ResidentWeights,
        assumptions: vec![
            "prefers resident weights and a selected device where available".to_string(),
        ],
        unknown_information: Vec::new(),
    });
    plans.push(ExecutionPlanCandidate {
        schema_version: EXECUTION_CANDIDATE_SCHEMA_VERSION,
        id: "balanced".to_string(),
        profile: PlanProfile::Balanced,
        backend: PlanBackend::Cpu,
        device_id: None,
        cpu_threads: threads,
        context_tokens: context,
        batch_tokens: batch,
        ram_budget_bytes: request.ram_budget_bytes,
        vram_budget_bytes: request.vram_budget_bytes,
        mapping: MappingStrategy::FileBacked,
        assumptions: vec![
            "keeps weights file-backed and leaves reclaim to the operating system".to_string(),
        ],
        unknown_information: vec![
            "runtime activations and graph scratch are not modeled".to_string()
        ],
    });
    plans.push(ExecutionPlanCandidate {
        schema_version: EXECUTION_CANDIDATE_SCHEMA_VERSION,
        id: "memory-constrained".to_string(),
        profile: PlanProfile::MemoryConstrained,
        backend: PlanBackend::Cpu,
        device_id: None,
        cpu_threads: threads.clamp(1, 4),
        context_tokens: context.clamp(1, 2048),
        batch_tokens: 1,
        ram_budget_bytes: request.ram_budget_bytes,
        vram_budget_bytes: None,
        mapping: MappingStrategy::Streaming,
        assumptions: vec![
            "streams tensor data and minimizes resident weights".to_string(),
            "does not claim a complete memory bound without runtime overhead data".to_string(),
        ],
        unknown_information: vec![
            "staging and activation workspace sizes are not modeled".to_string()
        ],
    });
    plans.push(ExecutionPlanCandidate {
        schema_version: EXECUTION_CANDIDATE_SCHEMA_VERSION,
        id: "maximum-model-size".to_string(),
        profile: PlanProfile::MaximumModelSize,
        backend: PlanBackend::Cpu,
        device_id: None,
        cpu_threads: threads.clamp(1, 2),
        context_tokens: context,
        batch_tokens: 1,
        ram_budget_bytes: request.ram_budget_bytes,
        vram_budget_bytes: None,
        mapping: MappingStrategy::FileBacked,
        assumptions: vec!["prioritizes keeping the model file mapped over latency".to_string()],
        unknown_information: vec!["no latency or throughput prediction is made".to_string()],
    });
    plans
}

pub fn evaluate_plan(
    machine: &MachineProfile,
    requirements: &ModelRequirements,
    plan: &ExecutionPlanCandidate,
) -> PlanEvaluation {
    evaluate_plan_internal(machine, requirements, plan, None)
}

/// Evaluate with optional calibration evidence. Evidence is applied only when
/// the model fingerprint, backend, context, and batch all match.
pub fn evaluate_plan_with_calibration(
    machine: &MachineProfile,
    requirements: &ModelRequirements,
    plan: &ExecutionPlanCandidate,
    calibration: Option<&CalibrationEvidence>,
) -> PlanEvaluation {
    evaluate_plan_internal(machine, requirements, plan, calibration)
}

fn evaluate_plan_internal(
    machine: &MachineProfile,
    requirements: &ModelRequirements,
    plan: &ExecutionPlanCandidate,
    calibration: Option<&CalibrationEvidence>,
) -> PlanEvaluation {
    let mut reasons = Vec::new();
    let mut unknown = plan.unknown_information.clone();
    if let Err(error) = plan.validate() {
        return PlanEvaluation {
            plan_id: plan.id.clone(),
            profile: plan.profile,
            feasibility: Feasibility::Infeasible,
            estimated_ram: RequirementBytes::unknown("invalid plan"),
            estimated_vram: RequirementBytes::unknown("invalid plan"),
            reasons: vec![PlanReason {
                code: "invalid_plan".to_string(),
                detail: error,
            }],
            unknown_information: unknown,
            storage_dependency: plan.mapping != MappingStrategy::ResidentWeights,
        };
    }
    if let Some(max_context) = requirements.model_max_context_tokens {
        if u64::from(plan.context_tokens) > max_context {
            reasons.push(PlanReason {
                code: "context_exceeds_model".to_string(),
                detail: format!(
                    "plan context {} exceeds model maximum {max_context}",
                    plan.context_tokens
                ),
            });
        }
    }
    if let Some(limit) = machine.cpu.user_thread_limit.as_ref() {
        if plan.cpu_threads > *limit {
            reasons.push(PlanReason {
                code: "thread_limit_exceeded".to_string(),
                detail: format!(
                    "plan requests {} threads but user limit is {limit}",
                    plan.cpu_threads
                ),
            });
        }
    }
    if let Some(available) = machine.cpu.available_threads.as_ref() {
        if plan.cpu_threads > *available {
            reasons.push(PlanReason {
                code: "threads_oversubscribed".to_string(),
                detail: format!(
                    "plan requests {} threads but only {available} are process-visible",
                    plan.cpu_threads
                ),
            });
        }
    }
    let hard_infeasible = reasons.iter().any(|reason| {
        matches!(
            reason.code.as_str(),
            "context_exceeds_model" | "invalid_plan"
        )
    });
    if hard_infeasible {
        return PlanEvaluation {
            plan_id: plan.id.clone(),
            profile: plan.profile,
            feasibility: Feasibility::Infeasible,
            estimated_ram: estimate_ram(requirements, plan.mapping),
            estimated_vram: estimate_vram(requirements, plan),
            reasons,
            unknown_information: unknown,
            storage_dependency: plan.mapping != MappingStrategy::ResidentWeights,
        };
    }

    let feasibility;
    let mut estimated_ram = estimate_ram(requirements, plan.mapping);
    let estimated_vram = estimate_vram(requirements, plan);
    if let Some(evidence) = calibration {
        if evidence.is_comparable(requirements, plan) {
            if evidence.observed_runtime_overhead.bytes.is_some() {
                estimated_ram = estimated_ram.checked_add(&evidence.observed_runtime_overhead);
                reasons.push(PlanReason {
                    code: "calibration_applied".to_string(),
                    detail: "compatible observed runtime overhead added conservatively".to_string(),
                });
            } else {
                unknown.push("calibration runtime overhead is unavailable".to_string());
            }
        } else {
            reasons.push(PlanReason {
                code: "calibration_incomparable".to_string(),
                detail: evidence
                    .reason
                    .clone()
                    .unwrap_or_else(|| "model/config/backend does not match".to_string()),
            });
        }
    }
    match plan.backend {
        PlanBackend::Cpu => {
            if let Some(budget) = plan.ram_budget_bytes {
                if let Some(required) = estimated_ram.bytes {
                    if required > budget {
                        reasons.push(PlanReason {
                            code: "ram_requirement_exceeds_budget".to_string(),
                            detail: format!(
                                "estimated RAM requirement {required} exceeds budget {budget}"
                            ),
                        });
                        feasibility = Feasibility::Infeasible;
                    } else {
                        feasibility = match plan.mapping {
                            MappingStrategy::ResidentWeights => {
                                Feasibility::FeasibleWithConstraints
                            }
                            MappingStrategy::FileBacked | MappingStrategy::Streaming => {
                                reasons.push(PlanReason {
                                    code: "mapped_weights_not_resident".to_string(),
                                    detail:
                                        "file-backed/streamed weights are not assumed resident; fit is conditional"
                                            .to_string(),
                                });
                                Feasibility::ConditionallyFeasible
                            }
                        };
                    }
                } else {
                    unknown.push("RAM requirement is not fully known".to_string());
                    feasibility = Feasibility::Unknown;
                }
            } else {
                unknown.push("no RAM budget specified".to_string());
                feasibility = Feasibility::Unknown;
            }
        }
        PlanBackend::Gpu => {
            let device = plan
                .device_id
                .as_ref()
                .and_then(|id| machine.devices.iter().find(|device| &device.id == id));
            let Some(device) = device else {
                reasons.push(PlanReason {
                    code: "no_gpu_device".to_string(),
                    detail: "GPU plan has no matching observed device".to_string(),
                });
                return PlanEvaluation {
                    plan_id: plan.id.clone(),
                    profile: plan.profile,
                    feasibility: if machine.devices.iter().any(|device| device.kind != "cpu") {
                        Feasibility::Unknown
                    } else {
                        Feasibility::Infeasible
                    },
                    estimated_ram,
                    estimated_vram,
                    reasons,
                    unknown_information: unknown,
                    storage_dependency: plan.mapping != MappingStrategy::ResidentWeights,
                };
            };
            if !device.selectable {
                reasons.push(PlanReason {
                    code: "gpu_not_selectable".to_string(),
                    detail: "device is present but not selectable by the current adapter"
                        .to_string(),
                });
                feasibility = Feasibility::Infeasible;
            } else if !matches!(
                device.memory_free_bytes.state,
                crate::machine_profile::ObservationState::Known
            ) {
                unknown.push("free VRAM is unavailable or unsupported".to_string());
                feasibility = Feasibility::Unknown;
            } else {
                let free = device.memory_free_bytes.as_ref().copied().unwrap_or(0);
                if let Some(required) = estimated_vram.bytes {
                    if let Some(budget) = plan.vram_budget_bytes {
                        if required > budget {
                            reasons.push(PlanReason {
                                code: "vram_requirement_exceeds_budget".to_string(),
                                detail: format!(
                                    "estimated VRAM requirement {required} exceeds budget {budget}"
                                ),
                            });
                            feasibility = Feasibility::Infeasible;
                            return PlanEvaluation {
                                plan_id: plan.id.clone(),
                                profile: plan.profile,
                                feasibility,
                                estimated_ram,
                                estimated_vram,
                                reasons,
                                unknown_information: unknown,
                                storage_dependency: plan.mapping
                                    != MappingStrategy::ResidentWeights,
                            };
                        }
                    }
                    if required > free {
                        reasons.push(PlanReason {
                            code: "vram_requirement_exceeds_free".to_string(),
                            detail: format!(
                                "estimated VRAM requirement {required} exceeds observed free {free}"
                            ),
                        });
                        feasibility = Feasibility::Infeasible;
                    } else {
                        feasibility = Feasibility::ConditionallyFeasible;
                        reasons.push(PlanReason {
                            code: "vram_observation_is_point_in_time".to_string(),
                            detail: "free VRAM is not proof of exclusive availability".to_string(),
                        });
                    }
                } else {
                    unknown.push("VRAM requirement is not fully known".to_string());
                    feasibility = Feasibility::Unknown;
                }
            }
        }
    }
    PlanEvaluation {
        plan_id: plan.id.clone(),
        profile: plan.profile,
        feasibility,
        estimated_ram,
        estimated_vram,
        reasons,
        unknown_information: unknown,
        storage_dependency: plan.mapping != MappingStrategy::ResidentWeights,
    }
}

fn estimate_ram(requirements: &ModelRequirements, mapping: MappingStrategy) -> RequirementBytes {
    match mapping {
        MappingStrategy::ResidentWeights => requirements.resident_weights_plus_kv_bytes.clone(),
        MappingStrategy::FileBacked => requirements.mapped_weights_plus_kv_bytes.clone(),
        MappingStrategy::Streaming => {
            if requirements.kv_cache_bytes.bytes.is_some() {
                RequirementBytes::estimated(
                    requirements.kv_cache_bytes.bytes.unwrap_or(0),
                    "streaming keeps weights out of resident memory but excludes staging/activation workspace",
                )
            } else {
                RequirementBytes::unknown("KV estimate unavailable for streaming plan")
            }
        }
    }
}

fn estimate_vram(
    requirements: &ModelRequirements,
    plan: &ExecutionPlanCandidate,
) -> RequirementBytes {
    match plan.backend {
        PlanBackend::Cpu => RequirementBytes::exact(0, "CPU backend does not use VRAM"),
        PlanBackend::Gpu => match plan.mapping {
            MappingStrategy::ResidentWeights => requirements.resident_weights_plus_kv_bytes.clone(),
            MappingStrategy::FileBacked | MappingStrategy::Streaming => {
                RequirementBytes::unknown("GPU KV, staging, and activation placement not modeled")
            }
        },
    }
}

pub fn compare_plans(
    machine: &MachineProfile,
    requirements: &ModelRequirements,
    plans: &[ExecutionPlanCandidate],
) -> PlanComparison {
    compare_plans_internal(machine, requirements, plans, None)
}

pub fn compare_plans_with_calibration(
    machine: &MachineProfile,
    requirements: &ModelRequirements,
    plans: &[ExecutionPlanCandidate],
    calibration: Option<&CalibrationEvidence>,
) -> PlanComparison {
    compare_plans_internal(machine, requirements, plans, calibration)
}

/// Compare plans with per-plan evidence. Each entry is applied only through
/// the same conservative `evaluate_plan_internal` rules.
pub fn compare_plans_with_evidence(
    machine: &MachineProfile,
    requirements: &ModelRequirements,
    plans: &[ExecutionPlanCandidate],
    evidence: &[Option<CalibrationEvidence>],
) -> PlanComparison {
    PlanComparison {
        schema_version: EXECUTION_CANDIDATE_SCHEMA_VERSION,
        dimensions: comparison_dimensions(),
        evaluations: plans
            .iter()
            .enumerate()
            .map(|(index, plan)| {
                evaluate_plan_internal(
                    machine,
                    requirements,
                    plan,
                    evidence.get(index).and_then(|value| value.as_ref()),
                )
            })
            .collect(),
        note: "No single score is produced. Weights or ranking assumptions are not hidden."
            .to_string(),
    }
}

fn comparison_dimensions() -> Vec<String> {
    vec![
        "feasibility".to_string(),
        "estimated_ram".to_string(),
        "estimated_vram".to_string(),
        "ram_budget".to_string(),
        "vram_budget".to_string(),
        "cpu_threads".to_string(),
        "context_tokens".to_string(),
        "batch_tokens".to_string(),
        "mapping".to_string(),
        "storage_dependency".to_string(),
        "unknown_information".to_string(),
        "reasons".to_string(),
    ]
}

fn compare_plans_internal(
    machine: &MachineProfile,
    requirements: &ModelRequirements,
    plans: &[ExecutionPlanCandidate],
    calibration: Option<&CalibrationEvidence>,
) -> PlanComparison {
    PlanComparison {
        schema_version: EXECUTION_CANDIDATE_SCHEMA_VERSION,
        dimensions: comparison_dimensions(),
        evaluations: plans
            .iter()
            .map(|plan| evaluate_plan_internal(machine, requirements, plan, calibration))
            .collect(),
        note: "No single score is produced. Weights or ranking assumptions are not hidden."
            .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine_profile::{collect_machine_profile, MachineProfileConfig, Observed};

    fn requirements(weights: u64, kv: u64, max_context: Option<u64>) -> ModelRequirements {
        ModelRequirements {
            schema_version: crate::model_requirements::MODEL_REQUIREMENTS_SCHEMA_VERSION,
            model_fingerprint: None,
            model_name: Some("test".to_string()),
            architecture: Some("llama".to_string()),
            architecture_info: None,
            quantization: Some("mostly-q4_k".to_string()),
            tensor_count: 1,
            file_bytes: RequirementBytes::exact(weights, "test"),
            tensor_payload_bytes: RequirementBytes::exact(weights, "test"),
            fully_resident_weight_bytes: RequirementBytes::exact(weights, "test"),
            kv_cache_bytes: RequirementBytes::estimated(kv, "test"),
            context_tokens: 512,
            batch_tokens: 1,
            model_max_context_tokens: max_context,
            resident_weights_plus_kv_bytes: RequirementBytes::estimated(
                weights.saturating_add(kv),
                "test",
            ),
            mapped_weights_plus_kv_bytes: RequirementBytes::estimated(kv, "test mapped"),
            unknown_requirements: Vec::new(),
            assumptions: Vec::new(),
        }
    }

    fn profile() -> MachineProfile {
        let mut profile = collect_machine_profile(&MachineProfileConfig::default());
        profile.cpu.available_threads = Observed::known(
            8,
            crate::machine_profile::Provenance::Configured,
            crate::machine_profile::Freshness::Dynamic,
            1,
        );
        profile.cpu.user_thread_limit = Observed::unavailable("none", 1);
        profile
    }

    fn cpu_plan(mapping: MappingStrategy, budget: Option<u64>) -> ExecutionPlanCandidate {
        ExecutionPlanCandidate {
            schema_version: EXECUTION_CANDIDATE_SCHEMA_VERSION,
            id: "test".to_string(),
            profile: PlanProfile::Custom,
            backend: PlanBackend::Cpu,
            device_id: None,
            cpu_threads: 4,
            context_tokens: 512,
            batch_tokens: 1,
            ram_budget_bytes: budget,
            vram_budget_bytes: None,
            mapping,
            assumptions: Vec::new(),
            unknown_information: Vec::new(),
        }
    }

    #[test]
    fn resident_plan_is_constrained_feasible_when_budget_covers_estimate() {
        let evaluation = evaluate_plan(
            &profile(),
            &requirements(1000, 200, Some(4096)),
            &cpu_plan(MappingStrategy::ResidentWeights, Some(2000)),
        );
        assert_eq!(evaluation.feasibility, Feasibility::FeasibleWithConstraints);
    }

    #[test]
    fn resident_plan_is_infeasible_when_budget_is_too_small() {
        let evaluation = evaluate_plan(
            &profile(),
            &requirements(3000, 200, Some(4096)),
            &cpu_plan(MappingStrategy::ResidentWeights, Some(2000)),
        );
        assert_eq!(evaluation.feasibility, Feasibility::Infeasible);
    }

    #[test]
    fn file_backed_plan_is_not_presented_as_definitely_feasible() {
        let evaluation = evaluate_plan(
            &profile(),
            &requirements(100_000, 200, Some(4096)),
            &cpu_plan(MappingStrategy::FileBacked, Some(1000)),
        );
        assert_eq!(evaluation.feasibility, Feasibility::ConditionallyFeasible);
        assert!(evaluation
            .reasons
            .iter()
            .any(|reason| reason.code == "mapped_weights_not_resident"));
    }

    #[test]
    fn missing_budget_is_unknown_not_feasible() {
        let evaluation = evaluate_plan(
            &profile(),
            &requirements(1000, 200, None),
            &cpu_plan(MappingStrategy::ResidentWeights, None),
        );
        assert_eq!(evaluation.feasibility, Feasibility::Unknown);
    }

    #[test]
    fn context_above_model_limit_is_infeasible() {
        let mut plan = cpu_plan(MappingStrategy::FileBacked, Some(10_000));
        plan.context_tokens = 8192;
        let evaluation = evaluate_plan(&profile(), &requirements(1000, 200, Some(4096)), &plan);
        assert_eq!(evaluation.feasibility, Feasibility::Infeasible);
    }

    #[test]
    fn missing_gpu_does_not_affect_cpu_plans() {
        let mut machine = profile();
        machine.devices.clear();
        let cpu = evaluate_plan(
            &machine,
            &requirements(1000, 200, Some(4096)),
            &cpu_plan(MappingStrategy::FileBacked, Some(10_000)),
        );
        assert_ne!(cpu.feasibility, Feasibility::Infeasible);
        let mut gpu_plan = cpu_plan(MappingStrategy::ResidentWeights, Some(10_000));
        gpu_plan.backend = PlanBackend::Gpu;
        gpu_plan.device_id = Some("missing".to_string());
        let gpu = evaluate_plan(&machine, &requirements(1000, 200, Some(4096)), &gpu_plan);
        assert_eq!(gpu.feasibility, Feasibility::Infeasible);
    }

    #[test]
    fn compatible_calibration_can_change_feasibility_conservatively() {
        use crate::runtime_calibration::CalibrationEvidence;
        let requirements = requirements(1000, 200, Some(4096));
        let plan = cpu_plan(MappingStrategy::ResidentWeights, Some(1500));
        let baseline = evaluate_plan(&profile(), &requirements, &plan);
        assert_eq!(baseline.feasibility, Feasibility::FeasibleWithConstraints);
        let mut calibrated_requirements = requirements.clone();
        calibrated_requirements.model_fingerprint = Some("fnv:test".to_string());
        let evidence = CalibrationEvidence {
            model_fingerprint: Some("fnv:test".to_string()),
            context_tokens: Some(512),
            batch_tokens: Some(1),
            backend: "cpu".to_string(),
            observed_runtime_overhead: RequirementBytes::estimated(600, "test overhead"),
            compatibility: crate::calibration_compatibility::CompatibilityClass::Exact,
            valid: true,
            reason: None,
        };
        let calibrated = evaluate_plan_with_calibration(
            &profile(),
            &calibrated_requirements,
            &plan,
            Some(&evidence),
        );
        assert_eq!(calibrated.feasibility, Feasibility::Infeasible);
        assert!(calibrated
            .reasons
            .iter()
            .any(|reason| reason.code == "calibration_applied"));
        let mismatched = CalibrationEvidence {
            context_tokens: Some(256),
            ..evidence
        };
        let result = evaluate_plan_with_calibration(
            &profile(),
            &calibrated_requirements,
            &plan,
            Some(&mismatched),
        );
        assert!(result
            .reasons
            .iter()
            .any(|reason| reason.code == "calibration_incomparable"));
    }

    #[test]
    fn vram_budget_is_enforced_for_resident_gpu_plans() {
        let mut machine = profile();
        let mut gpu = machine.devices[0].clone();
        gpu.id = "gpu-test".to_string();
        gpu.kind = "gpu".to_string();
        gpu.selectable = true;
        gpu.memory_total_bytes = Observed::known(
            2000,
            crate::machine_profile::Provenance::Configured,
            crate::machine_profile::Freshness::Stable,
            1,
        );
        gpu.memory_free_bytes = Observed::known(
            1000,
            crate::machine_profile::Provenance::Configured,
            crate::machine_profile::Freshness::Dynamic,
            1,
        );
        machine.devices.push(gpu);
        let mut plan = cpu_plan(MappingStrategy::ResidentWeights, Some(10_000));
        plan.backend = PlanBackend::Gpu;
        plan.device_id = Some("gpu-test".to_string());
        plan.vram_budget_bytes = Some(100);
        let evaluation = evaluate_plan(&machine, &requirements(1000, 200, Some(4096)), &plan);
        assert_eq!(evaluation.feasibility, Feasibility::Infeasible);
        assert!(evaluation
            .reasons
            .iter()
            .any(|reason| reason.code == "vram_requirement_exceeds_budget"));
    }

    #[test]
    fn candidate_plans_are_distinct_and_comparable() {
        let machine = profile();
        let requirements = requirements(1000, 200, Some(4096));
        let plans = candidate_plans(
            &machine,
            &requirements,
            CandidateRequest {
                ram_budget_bytes: Some(10_000),
                ..CandidateRequest::default()
            },
        );
        assert_eq!(plans.len(), 4);
        assert!(plans.iter().all(|plan| plan.validate().is_ok()));
        let profiles: Vec<_> = plans.iter().map(|plan| plan.profile).collect();
        assert!(profiles.windows(2).all(|pair| pair[0] != pair[1]));
        let comparison = compare_plans(&machine, &requirements, &plans);
        assert_eq!(comparison.evaluations.len(), plans.len());
        assert_eq!(
            comparison
                .evaluations
                .iter()
                .map(|evaluation| evaluation.plan_id.clone())
                .collect::<Vec<_>>(),
            plans.iter().map(|plan| plan.id.clone()).collect::<Vec<_>>()
        );
        assert!(comparison.note.contains("No single score"));
    }

    #[test]
    fn comparison_serializes_deterministically() {
        let machine = profile();
        let requirements = requirements(1000, 200, Some(4096));
        let plans = candidate_plans(
            &machine,
            &requirements,
            CandidateRequest {
                ram_budget_bytes: Some(10_000),
                ..CandidateRequest::default()
            },
        );
        let comparison = compare_plans(&machine, &requirements, &plans);
        let first = comparison.to_json().unwrap();
        let second = comparison.to_json().unwrap();
        assert_eq!(first, second);
        let decoded: PlanComparison = serde_json::from_value(first).unwrap();
        assert_eq!(decoded, comparison);
    }
}
