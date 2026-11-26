//! Conservative resource budgeting and admission classification.
//!
//! File size is not runtime memory. KV cache, compute buffers, staging and the
//! draft model are separate contributions, and missing metadata stays unknown.
//! Budgets are planning limits; they are not OS-enforced isolation.

use serde::{Deserialize, Serialize};

use crate::model_requirements::{ModelRequirements, RequirementCertainty};
use crate::profiling::{HardwareProfile, ModelProfile, WorkloadProfile};

pub const RESOURCE_BUDGET_SCHEMA_VERSION: u32 = 1;

/// Conservative per-process allowance for backend/context/compute buffers that
/// GGUF metadata does not describe. Not a measured peak.
pub const DEFAULT_RUNTIME_OVERHEAD_BYTES: u64 = 512 * 1024 * 1024;
pub const DEFAULT_VRAM_OVERHEAD_BYTES: u64 = 256 * 1024 * 1024;
pub const RISK_FRACTION_BPS: u64 = 8000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetClass {
    WithinBudget,
    WithinBudgetWithAssumptions,
    Risky,
    Infeasible,
    Unknown,
}

impl BudgetClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::WithinBudget => "within-budget",
            Self::WithinBudgetWithAssumptions => "within-budget-with-assumptions",
            Self::Risky => "risky",
            Self::Infeasible => "infeasible",
            Self::Unknown => "unknown",
        }
    }

    fn severity(self) -> u8 {
        match self {
            Self::WithinBudget => 0,
            Self::WithinBudgetWithAssumptions => 1,
            Self::Risky => 2,
            Self::Unknown => 3,
            Self::Infeasible => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetComponent {
    pub name: String,
    pub bytes: Option<u64>,
    pub certainty: RequirementCertainty,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceBudget {
    pub schema_version: u32,
    /// Advisory planning budget supplied by the caller.
    pub ram_budget_bytes: Option<u64>,
    pub available_ram_bytes: Option<u64>,
    pub total_ram_bytes: Option<u64>,
    pub cgroup_limit_bytes: Option<u64>,
    pub free_vram_bytes: Option<u64>,
    pub host_required_bytes: Option<u64>,
    pub vram_required_bytes: Option<u64>,
    pub components: Vec<BudgetComponent>,
    pub admission: BudgetClass,
    /// Always false: RAMforge does not enforce this budget at the OS level.
    pub enforceable: bool,
    pub reasons: Vec<String>,
    pub assumptions: Vec<String>,
    pub unknowns: Vec<String>,
}

impl ResourceBudget {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != RESOURCE_BUDGET_SCHEMA_VERSION {
            return Err(format!(
                "unsupported resource budget schema {}",
                self.schema_version
            ));
        }
        if self.enforceable {
            return Err("resource budgets are always advisory in this build".to_string());
        }
        if self.components.is_empty() {
            return Err("resource budget has no components".to_string());
        }
        for component in &self.components {
            match (component.certainty, component.bytes) {
                (RequirementCertainty::Unknown, None) => {}
                (RequirementCertainty::Unknown, Some(_)) => {
                    return Err(format!("{} is unknown but has bytes", component.name))
                }
                (_, Some(_)) => {}
                (_, None) => return Err(format!("{} has no byte value", component.name)),
            }
        }
        Ok(())
    }
}

/// Build a conservative budget for the proposed workload.
pub fn assess_budget(
    model: &ModelProfile,
    hardware: &HardwareProfile,
    workload: &WorkloadProfile,
    requirements: Option<&ModelRequirements>,
    draft_bytes: Option<u64>,
) -> ResourceBudget {
    let mut components = Vec::new();
    let mut reasons = Vec::new();
    let mut assumptions = Vec::new();
    let mut unknowns = Vec::new();

    let model_bytes = model.file_bytes;
    components.push(BudgetComponent {
        name: "model-file".to_string(),
        bytes: Some(model_bytes),
        certainty: RequirementCertainty::Exact,
        detail: "GGUF file size; mapped or read depending on load mode".to_string(),
    });

    let kv = requirements.map(|requirements| {
        (
            requirements.kv_cache_bytes.bytes,
            requirements.kv_cache_bytes.certainty,
            requirements.kv_cache_bytes.detail.clone(),
        )
    });
    let kv_bytes = kv.as_ref().and_then(|(bytes, _, _)| *bytes);
    let kv_certainty = kv
        .as_ref()
        .map(|(_, certainty, _)| *certainty)
        .unwrap_or(RequirementCertainty::Unknown);
    components.push(BudgetComponent {
        name: "kv-cache".to_string(),
        bytes: kv_bytes,
        certainty: kv_certainty,
        detail: kv
            .as_ref()
            .and_then(|(_, _, detail)| detail.clone())
            .unwrap_or_else(|| "architecture metadata is insufficient to size KV".to_string()),
    });
    if kv_bytes.is_none() {
        unknowns.push("KV cache bytes are unknown".to_string());
    }

    components.push(BudgetComponent {
        name: "runtime-overhead".to_string(),
        bytes: Some(DEFAULT_RUNTIME_OVERHEAD_BYTES),
        certainty: RequirementCertainty::Estimated,
        detail: "fixed conservative allowance for backend, context, and compute buffers"
            .to_string(),
    });
    assumptions
        .push("runtime overhead is a fixed conservative estimate, not a measured peak".to_string());

    let draft = draft_bytes;
    if draft.is_some() {
        components.push(BudgetComponent {
            name: "draft-model".to_string(),
            bytes: draft,
            certainty: RequirementCertainty::Estimated,
            detail: "draft GGUF file bytes; draft context/compute overhead is not included"
                .to_string(),
        });
    }

    let available = hardware.available_ram_bytes;
    let staged = workload.residency == "staged";
    let resident_model_bytes = if staged {
        let window = workload
            .ram_budget_bytes
            .or(available)
            .unwrap_or(model_bytes);
        model_bytes.min(window)
    } else {
        model_bytes
    };
    if staged {
        components.push(BudgetComponent {
            name: "resident-model-window".to_string(),
            bytes: Some(resident_model_bytes),
            certainty: RequirementCertainty::Estimated,
            detail:
                "model bytes expected resident under staged residency; the remainder is fetched from storage"
                    .to_string(),
        });
        assumptions.push(
            "staged residency does not require the whole file resident; decode cost depends on measured storage throughput"
                .to_string(),
        );
    }
    let host_required = match (kv_bytes, draft) {
        (Some(kv), draft) => resident_model_bytes
            .checked_add(kv)
            .and_then(|value| value.checked_add(DEFAULT_RUNTIME_OVERHEAD_BYTES))
            .and_then(|value| value.checked_add(draft.unwrap_or(0))),
        (None, _) => None,
    };

    let layers = model.layer_count.unwrap_or(0);
    let gpu_share_bps = if workload.gpu_layers == 0 || layers == 0 {
        0
    } else {
        u64::from(workload.gpu_layers)
            .saturating_mul(10_000)
            .saturating_div(layers)
            .min(10_000)
    };
    let free_vram = hardware.max_free_vram_bytes();
    let vram_required = if gpu_share_bps == 0 {
        Some(0)
    } else {
        match kv_bytes {
            Some(kv) => model_bytes
                .saturating_mul(gpu_share_bps)
                .saturating_div(10_000)
                .checked_add(kv.saturating_mul(gpu_share_bps).saturating_div(10_000))
                .and_then(|value| value.checked_add(DEFAULT_VRAM_OVERHEAD_BYTES)),
            None => None,
        }
    };
    if gpu_share_bps > 0 {
        components.push(BudgetComponent {
            name: "vram-required".to_string(),
            bytes: vram_required,
            certainty: if vram_required.is_some() {
                RequirementCertainty::Estimated
            } else {
                RequirementCertainty::Unknown
            },
            detail: format!(
                "model and KV scaled to {gpu_share_bps} bps of layers plus a fixed {DEFAULT_VRAM_OVERHEAD_BYTES}-byte allowance"
            ),
        });
        if vram_required.is_none() {
            unknowns.push("VRAM requirement is unknown because KV bytes are unknown".to_string());
        }
    }

    let mut admission = BudgetClass::WithinBudget;
    let mut raise = |class: BudgetClass| {
        if class.severity() > admission.severity() {
            admission = class;
        }
    };

    match (available, host_required) {
        (None, _) => {
            raise(BudgetClass::Unknown);
            reasons.push("available system RAM is unobservable".to_string());
        }
        (Some(available), Some(required)) => {
            let non_reclaimable = if staged {
                let kv = kv_bytes.unwrap_or(0);
                kv.saturating_add(DEFAULT_RUNTIME_OVERHEAD_BYTES)
                    .saturating_add(draft.unwrap_or(0))
            } else {
                0
            };
            if staged && non_reclaimable > available {
                raise(BudgetClass::Infeasible);
                reasons.push(format!(
                    "staged non-reclaimable requirement (KV plus overhead) {} exceeds available RAM {}",
                    non_reclaimable, available
                ));
            } else if staged && required > available {
                raise(BudgetClass::Risky);
                reasons.push(format!(
                    "staged resident window plus KV and overhead ({required}) exceeds available RAM {available}; file-backed pages must be reclaimed during execution"
                ));
            } else if !staged && required > available {
                raise(BudgetClass::Infeasible);
                reasons.push(format!(
                    "estimated host requirement {} exceeds available RAM {}",
                    required, available
                ));
            } else if required.saturating_mul(10_000) > available.saturating_mul(RISK_FRACTION_BPS)
            {
                raise(BudgetClass::Risky);
                reasons.push(format!(
                    "estimated host requirement {} uses more than 80% of available RAM {}",
                    required, available
                ));
            } else if staged {
                raise(BudgetClass::WithinBudgetWithAssumptions);
                reasons.push(format!(
                    "staged resident window plus KV and overhead ({required}) fits available RAM {available}; the model file is streamed"
                ));
            } else {
                raise(BudgetClass::WithinBudgetWithAssumptions);
                reasons.push(format!(
                    "estimated host requirement {} fits available RAM {}",
                    required, available
                ));
            }
        }
        (Some(available), None) => {
            let comfortable = model_bytes
                .saturating_add(DEFAULT_RUNTIME_OVERHEAD_BYTES)
                .saturating_add(draft.unwrap_or(0))
                .saturating_mul(10_000)
                <= available.saturating_mul(6_000);
            if comfortable {
                raise(BudgetClass::WithinBudgetWithAssumptions);
                reasons.push(
                    "model plus overhead fits 60% of available RAM even though KV is unknown"
                        .to_string(),
                );
            } else {
                raise(BudgetClass::Unknown);
                reasons.push(
                    "KV is unknown and the model does not leave a comfortable RAM margin"
                        .to_string(),
                );
            }
        }
    }

    if let Some(limit) = hardware.cgroup_bytes() {
        let compared = if staged {
            kv_bytes
                .unwrap_or(0)
                .saturating_add(DEFAULT_RUNTIME_OVERHEAD_BYTES)
                .saturating_add(draft.unwrap_or(0))
        } else {
            host_required.unwrap_or(0)
        };
        if compared > limit {
            raise(BudgetClass::Infeasible);
            reasons.push(format!(
                "estimated requirement exceeds cgroup limit {limit}"
            ));
        }
    }

    if gpu_share_bps > 0 {
        match (free_vram, vram_required) {
            (Some(free), Some(required)) => {
                if required > free {
                    raise(BudgetClass::Infeasible);
                    reasons.push(format!(
                        "estimated VRAM requirement {} exceeds free VRAM {}",
                        required, free
                    ));
                } else if required.saturating_mul(10_000) > free.saturating_mul(RISK_FRACTION_BPS) {
                    raise(BudgetClass::Risky);
                    reasons.push(format!(
                        "estimated VRAM requirement {} uses more than 80% of free VRAM {}",
                        required, free
                    ));
                } else {
                    raise(BudgetClass::WithinBudgetWithAssumptions);
                }
            }
            (None, _) => {
                raise(BudgetClass::Unknown);
                reasons.push("free VRAM is unobservable".to_string());
            }
            (_, None) => {
                raise(BudgetClass::Unknown);
            }
        }
    } else if staged {
        match free_vram {
            Some(free) if free > DEFAULT_VRAM_OVERHEAD_BYTES => {
                let window = free - DEFAULT_VRAM_OVERHEAD_BYTES;
                components.push(BudgetComponent {
                    name: "staged-vram-window".to_string(),
                    bytes: Some(window),
                    certainty: RequirementCertainty::Estimated,
                    detail: "free VRAM minus a fixed safety allowance; the staging policy may use this window"
                        .to_string(),
                });
                assumptions.push(format!(
                    "staged residency may use up to {window} of free VRAM minus the runtime margin"
                ));
            }
            Some(_) => {
                raise(BudgetClass::Infeasible);
                reasons.push(
                    "staged residency requires more free VRAM than the safety allowance"
                        .to_string(),
                );
            }
            None => {
                raise(BudgetClass::Unknown);
                reasons
                    .push("staged residency requested but free VRAM is unobservable".to_string());
            }
        }
    }

    if let Some(budget) = workload.ram_budget_bytes {
        assumptions.push(format!(
            "caller planning budget {} is advisory and not enforced",
            budget
        ));
        let compared = if staged {
            resident_model_bytes
        } else {
            host_required.unwrap_or(0)
        };
        if compared > budget {
            reasons.push(format!(
                "estimated resident requirement {} exceeds the advisory planning budget {}",
                compared, budget
            ));
            raise(BudgetClass::Risky);
        }
    }

    if kv_certainty == RequirementCertainty::Estimated {
        assumptions.push("KV bytes are estimated from GGUF architecture metadata".to_string());
    }

    ResourceBudget {
        schema_version: RESOURCE_BUDGET_SCHEMA_VERSION,
        ram_budget_bytes: workload.ram_budget_bytes,
        available_ram_bytes: available,
        total_ram_bytes: hardware.total_ram_bytes,
        cgroup_limit_bytes: hardware.cgroup_bytes(),
        free_vram_bytes: free_vram,
        host_required_bytes: host_required,
        vram_required_bytes: vram_required,
        components,
        admission,
        enforceable: false,
        reasons,
        assumptions,
        unknowns,
    }
}

impl HardwareProfile {
    /// Observed cgroup memory limit, when the machine profile exposed one.
    pub fn cgroup_bytes(&self) -> Option<u64> {
        self.cgroup_limit_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive::environment::ExecutionEnvironment;
    use crate::model_architecture::ArchitectureKind;
    use crate::model_requirements::RequirementBytes;

    fn model(file_bytes: u64, layers: u64) -> ModelProfile {
        ModelProfile {
            schema_version: 2,
            fingerprint: Some("model".to_string()),
            path: None,
            architecture: Some("qwen2".to_string()),
            kind: ArchitectureKind::Dense,
            parameters: Some(1),
            file_bytes,
            tensor_payload_bytes: Some(file_bytes),
            tensor_count: 1,
            layer_count: Some(layers),
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

    fn hardware(
        available: Option<u64>,
        free_vram: Option<u64>,
        cgroup: Option<u64>,
    ) -> HardwareProfile {
        HardwareProfile {
            schema_version: 2,
            fingerprint: "hw".to_string(),
            cpu_model: Some("cpu".to_string()),
            physical_cores: Some(6),
            logical_cpus: Some(12),
            total_ram_bytes: Some(16 << 30),
            available_ram_bytes: available,
            os: "linux".to_string(),
            kernel: None,
            gpus: free_vram.map_or_else(Vec::new, |free| {
                vec![crate::profiling::GpuProfile {
                    id: "gpu0".to_string(),
                    backend: "cuda".to_string(),
                    name: "gpu".to_string(),
                    total_bytes: Some(4 << 30),
                    free_bytes: Some(free),
                    selectable: true,
                }]
            }),
            backends: vec!["cpu".to_string()],
            storage_kind: None,
            storage_read_bytes_per_second: None,
            cgroup_limit_bytes: cgroup,
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

    fn workload(gpu_layers: u32, staged: bool) -> WorkloadProfile {
        WorkloadProfile {
            context_tokens: 256,
            batch_tokens: 64,
            ubatch_tokens: 64,
            sequence_max: 1,
            threads: 6,
            gpu_layers,
            kv_type_k: "f16".to_string(),
            kv_type_v: "f16".to_string(),
            load_mode: "mmap".to_string(),
            backend: "cpu".to_string(),
            device_id: if gpu_layers > 0 { "gpu0" } else { "cpu" }.to_string(),
            residency: if staged { "staged" } else { "off" }.to_string(),
            draft_model_fingerprint: None,
            draft_tokens: None,
            max_tokens: Some(8),
            ram_budget_bytes: Some(10 << 30),
        }
    }

    fn requirements(kv: Option<u64>) -> ModelRequirements {
        ModelRequirements {
            schema_version: 1,
            model_fingerprint: Some("model".to_string()),
            model_name: None,
            architecture: Some("qwen2".to_string()),
            architecture_info: None,
            quantization: None,
            tensor_count: 1,
            file_bytes: RequirementBytes::exact(1024, "file"),
            tensor_payload_bytes: RequirementBytes::exact(1024, "payload"),
            fully_resident_weight_bytes: RequirementBytes::exact(1024, "resident"),
            kv_cache_bytes: kv.map_or_else(
                || RequirementBytes::unknown("missing KV heads"),
                |bytes| RequirementBytes::estimated(bytes, "metadata KV estimate"),
            ),
            context_tokens: 256,
            batch_tokens: 64,
            model_max_context_tokens: Some(4096),
            resident_weights_plus_kv_bytes: RequirementBytes::unknown("test"),
            mapped_weights_plus_kv_bytes: RequirementBytes::unknown("test"),
            unknown_requirements: Vec::new(),
            assumptions: Vec::new(),
        }
    }

    #[test]
    fn small_model_fits_with_assumptions() {
        let budget = assess_budget(
            &model(100 * 1024 * 1024, 24),
            &hardware(Some(8 << 30), None, None),
            &workload(0, false),
            Some(&requirements(Some(64 * 1024 * 1024))),
            None,
        );
        assert_eq!(budget.admission, BudgetClass::WithinBudgetWithAssumptions);
        assert!(!budget.enforceable);
        assert!(budget.validate().is_ok());
    }

    #[test]
    fn oversized_host_requirement_is_infeasible() {
        let budget = assess_budget(
            &model(20 << 30, 64),
            &hardware(Some(8 << 30), Some(4 << 30), None),
            &workload(0, false),
            Some(&requirements(Some(1 << 30))),
            None,
        );
        assert_eq!(budget.admission, BudgetClass::Infeasible);
    }

    #[test]
    fn unknown_kv_does_not_become_a_precise_claim() {
        let budget = assess_budget(
            &model(13 << 30, 44),
            &hardware(Some(8 << 30), Some(4 << 30), None),
            &workload(0, false),
            Some(&requirements(None)),
            None,
        );
        assert!(budget.host_required_bytes.is_none());
        assert_eq!(budget.admission, BudgetClass::Unknown);
        assert!(budget.unknowns.iter().any(|value| value.contains("KV")));
    }

    #[test]
    fn gpu_requirement_over_free_vram_is_infeasible() {
        let budget = assess_budget(
            &model(20 << 30, 64),
            &hardware(Some(8 << 30), Some(2 << 30), None),
            &workload(64, false),
            Some(&requirements(Some(1 << 30))),
            None,
        );
        assert_eq!(budget.admission, BudgetClass::Infeasible);
    }

    #[test]
    fn staged_residency_is_not_infeasible_merely_because_the_file_is_large() {
        let budget = assess_budget(
            &model(20 << 30, 64),
            &hardware(Some(8 << 30), Some(4 << 30), None),
            &workload(0, true),
            Some(&requirements(Some(1 << 30))),
            None,
        );
        assert_eq!(budget.admission, BudgetClass::Risky);
        assert!(budget
            .reasons
            .iter()
            .any(|reason| reason.contains("staged resident window")));
        assert!(budget
            .components
            .iter()
            .any(|component| component.name == "resident-model-window"));
    }

    #[test]
    fn cgroup_limit_participates_in_admission() {
        let budget = assess_budget(
            &model(2 << 30, 24),
            &hardware(Some(8 << 30), None, Some(1 << 30)),
            &workload(0, false),
            Some(&requirements(Some(64 * 1024 * 1024))),
            None,
        );
        assert_eq!(budget.admission, BudgetClass::Infeasible);
        assert!(budget
            .reasons
            .iter()
            .any(|reason| reason.contains("cgroup")));
    }
}
