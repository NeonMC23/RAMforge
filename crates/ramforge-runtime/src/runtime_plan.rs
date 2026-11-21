//! Versioned operational execution-plan contract.
//!
//! Plans are resolved into a concrete CPU runtime configuration and explicitly
//! report which fields were applied, constrained, advisory, unsupported, or
//! unknown. Nothing in this module authorizes reclamation or automatic profile
//! switching.

use serde::{Deserialize, Serialize};

use crate::machine_profile::MachineProfile;
use crate::model_requirements::ModelRequirements;

pub const RUNTIME_PLAN_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeProfileKind {
    MaximumPerformance,
    Balanced,
    Background,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldStatus {
    Applied,
    Constrained,
    Advisory,
    Unsupported,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanField {
    pub field: String,
    pub status: FieldStatus,
    pub requested: Option<String>,
    pub effective: Option<String>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectiveRuntimeConfig {
    pub backend: String,
    pub device_id: String,
    pub context_tokens: u32,
    pub batch_tokens: u32,
    pub ubatch_tokens: u32,
    pub sequence_max: u32,
    pub threads: u32,
    pub threads_batch: u32,
    pub load_mode: String,
    pub weight_repacking: bool,
    pub kv_type_k: String,
    pub kv_type_v: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePlan {
    pub schema_version: u32,
    pub profile: RuntimeProfileKind,
    pub model_path: String,
    pub model_fingerprint: Option<String>,
    pub architecture: Option<crate::model_architecture::ModelArchitecture>,
    pub effective: EffectiveRuntimeConfig,
    pub fields: Vec<PlanField>,
    pub warnings: Vec<String>,
    pub observation_only: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CustomRuntimeRequest {
    pub context_tokens: Option<u32>,
    pub batch_tokens: Option<u32>,
    pub threads: Option<u32>,
    pub load_mode: Option<String>,
    pub no_repack: Option<bool>,
}

impl RuntimePlan {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != RUNTIME_PLAN_SCHEMA_VERSION {
            return Err(format!(
                "unsupported runtime plan schema {}",
                self.schema_version
            ));
        }
        if self.effective.context_tokens == 0
            || self.effective.batch_tokens == 0
            || self.effective.ubatch_tokens == 0
            || self.effective.threads == 0
        {
            return Err("effective runtime values must be greater than zero".to_string());
        }
        if self.effective.batch_tokens > self.effective.context_tokens {
            return Err("effective batch exceeds effective context".to_string());
        }
        if self.effective.sequence_max == 0 {
            return Err("effective sequence count must be greater than zero".to_string());
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(|error| error.to_string())
    }
}

fn push(
    fields: &mut Vec<PlanField>,
    field: &str,
    status: FieldStatus,
    requested: Option<String>,
    effective: Option<String>,
    reason: Option<String>,
) {
    fields.push(PlanField {
        field: field.to_string(),
        status,
        requested,
        effective,
        reason,
    });
}

impl RuntimePlan {
    /// Resolve a profile request against observed machine/model constraints.
    /// This does not load the model; the service reads back native values later.
    pub fn resolve(
        machine: &MachineProfile,
        requirements: &ModelRequirements,
        model_path: &str,
        profile: RuntimeProfileKind,
        custom: &CustomRuntimeRequest,
    ) -> Result<Self, String> {
        if model_path.is_empty() {
            return Err("model path is empty".to_string());
        }
        let available = machine
            .cpu
            .available_threads
            .as_ref()
            .copied()
            .unwrap_or(1)
            .max(1);
        let model_max = requirements
            .model_max_context_tokens
            .and_then(|value| u32::try_from(value).ok());
        let requested_context = custom
            .context_tokens
            .or(match profile {
                RuntimeProfileKind::MaximumPerformance => model_max.map(|value| value.min(8192)),
                RuntimeProfileKind::Balanced => model_max.map(|value| value.min(4096)),
                RuntimeProfileKind::Background => model_max.map(|value| value.min(2048)),
                RuntimeProfileKind::Custom => model_max,
            })
            .unwrap_or(2048)
            .max(1);
        let context = model_max
            .map(|max| requested_context.min(max))
            .unwrap_or(requested_context);
        let requested_batch = custom.batch_tokens.unwrap_or(1).max(1);
        let batch = requested_batch.min(context);
        let requested_threads = custom.threads.unwrap_or(match profile {
            RuntimeProfileKind::MaximumPerformance => available,
            RuntimeProfileKind::Balanced => available.saturating_sub(1).max(1),
            RuntimeProfileKind::Background => 1,
            RuntimeProfileKind::Custom => available,
        });
        let threads = requested_threads.min(available).max(1);
        let threads_batch = threads;
        let load_mode = custom.load_mode.clone().unwrap_or_else(|| match profile {
            RuntimeProfileKind::MaximumPerformance => "mmap".to_string(),
            RuntimeProfileKind::Balanced => "mmap".to_string(),
            RuntimeProfileKind::Background => "mmap-no-prefetch".to_string(),
            RuntimeProfileKind::Custom => "mmap".to_string(),
        });
        let no_repack = custom.no_repack.unwrap_or(matches!(
            profile,
            RuntimeProfileKind::Balanced | RuntimeProfileKind::Background
        ));
        let weight_repacking = !no_repack;

        let mut fields = Vec::new();
        push(
            &mut fields,
            "backend",
            FieldStatus::Applied,
            Some("cpu".to_string()),
            Some("cpu".to_string()),
            Some("the verified ForgeCore adapter currently executes CPU only".to_string()),
        );
        push(
            &mut fields,
            "device",
            FieldStatus::Applied,
            Some("cpu".to_string()),
            Some("cpu".to_string()),
            None,
        );
        push(
            &mut fields,
            "context_tokens",
            if context == requested_context {
                FieldStatus::Applied
            } else {
                FieldStatus::Constrained
            },
            Some(requested_context.to_string()),
            Some(context.to_string()),
            (context != requested_context)
                .then(|| "requested context exceeds model maximum".to_string()),
        );
        push(
            &mut fields,
            "batch_tokens",
            if batch == requested_batch {
                FieldStatus::Applied
            } else {
                FieldStatus::Constrained
            },
            Some(requested_batch.to_string()),
            Some(batch.to_string()),
            (batch != requested_batch).then(|| "batch clamped to context".to_string()),
        );
        push(
            &mut fields,
            "threads",
            if threads == requested_threads {
                FieldStatus::Applied
            } else {
                FieldStatus::Constrained
            },
            Some(requested_threads.to_string()),
            Some(threads.to_string()),
            (threads != requested_threads)
                .then(|| "thread count exceeds process-visible CPUs".to_string()),
        );
        push(
            &mut fields,
            "load_mode",
            FieldStatus::Applied,
            None,
            Some(load_mode.clone()),
            None,
        );
        push(
            &mut fields,
            "weight_repacking",
            FieldStatus::Applied,
            None,
            Some(weight_repacking.to_string()),
            None,
        );
        push(
            &mut fields,
            "gpu_offload",
            FieldStatus::Unsupported,
            Some("auto".to_string()),
            Some("cpu".to_string()),
            Some("no selectable GPU path is enabled by the current adapter".to_string()),
        );
        push(
            &mut fields,
            "ram_budget",
            FieldStatus::Advisory,
            Some("user-provided or profile-derived".to_string()),
            None,
            Some("no hard cgroup enforcement is implied by a plan".to_string()),
        );
        push(
            &mut fields,
            "kv_type",
            FieldStatus::Unknown,
            Some("f16".to_string()),
            Some("f16".to_string()),
            Some("adapter default; native effective KV type getter unavailable".to_string()),
        );
        let mut warnings = Vec::new();
        if machine.cpu.hardware_physical_cores.as_ref().is_none() {
            warnings
                .push("physical CPU topology unavailable; using logical availability".to_string());
        }
        if machine.memory.cgroup.limit_bytes.as_ref().is_none() {
            warnings
                .push("no finite cgroup memory limit observed; RAM budget is advisory".to_string());
        }
        if machine.devices.iter().any(|device| device.kind != "cpu") {
            warnings
                .push("non-CPU devices exist but are not selectable by this adapter".to_string());
        }
        let plan = Self {
            schema_version: RUNTIME_PLAN_SCHEMA_VERSION,
            profile,
            model_path: model_path.to_string(),
            model_fingerprint: requirements.model_fingerprint.clone(),
            architecture: requirements.architecture_info.clone(),
            effective: EffectiveRuntimeConfig {
                backend: "cpu".to_string(),
                device_id: "cpu".to_string(),
                context_tokens: context,
                batch_tokens: batch,
                ubatch_tokens: context,
                sequence_max: 1,
                threads,
                threads_batch,
                load_mode,
                weight_repacking,
                kv_type_k: "f16".to_string(),
                kv_type_v: "f16".to_string(),
            },
            fields,
            warnings,
            observation_only: true,
        };
        plan.validate()?;
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine_profile::{collect_machine_profile, MachineProfileConfig};
    use crate::model_requirements::RequirementBytes;

    fn requirements(max_context: Option<u64>) -> ModelRequirements {
        ModelRequirements {
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
            context_tokens: 256,
            batch_tokens: 1,
            model_max_context_tokens: max_context,
            resident_weights_plus_kv_bytes: RequirementBytes::estimated(110, "test"),
            mapped_weights_plus_kv_bytes: RequirementBytes::estimated(10, "test"),
            unknown_requirements: Vec::new(),
            assumptions: Vec::new(),
        }
    }

    #[test]
    fn profiles_resolve_distinct_effective_configurations() {
        let machine = collect_machine_profile(&MachineProfileConfig::default());
        let requirements = requirements(Some(4096));
        let maximum = RuntimePlan::resolve(
            &machine,
            &requirements,
            "/m",
            RuntimeProfileKind::MaximumPerformance,
            &CustomRuntimeRequest::default(),
        )
        .unwrap();
        let background = RuntimePlan::resolve(
            &machine,
            &requirements,
            "/m",
            RuntimeProfileKind::Background,
            &CustomRuntimeRequest::default(),
        )
        .unwrap();
        assert!(maximum.effective.threads >= background.effective.threads);
        assert_eq!(background.effective.load_mode, "mmap-no-prefetch");
        assert!(!background.effective.weight_repacking);
        assert_eq!(maximum.effective.context_tokens, 4096);
    }

    #[test]
    fn custom_values_are_clamped_and_marked_constrained() {
        let machine = collect_machine_profile(&MachineProfileConfig::default());
        let requirements = requirements(Some(512));
        let plan = RuntimePlan::resolve(
            &machine,
            &requirements,
            "/m",
            RuntimeProfileKind::Custom,
            &CustomRuntimeRequest {
                context_tokens: Some(4096),
                batch_tokens: Some(2048),
                threads: Some(9999),
                ..CustomRuntimeRequest::default()
            },
        )
        .unwrap();
        assert_eq!(plan.effective.context_tokens, 512);
        assert_eq!(plan.effective.batch_tokens, 512);
        assert!(plan.fields.iter().any(|field| {
            field.field == "context_tokens" && field.status == FieldStatus::Constrained
        }));
        assert!(plan.fields.iter().any(|field| {
            field.field == "gpu_offload" && field.status == FieldStatus::Unsupported
        }));
    }

    #[test]
    fn plan_serializes_and_validates() {
        let machine = collect_machine_profile(&MachineProfileConfig::default());
        let plan = RuntimePlan::resolve(
            &machine,
            &requirements(None),
            "/m",
            RuntimeProfileKind::Balanced,
            &CustomRuntimeRequest::default(),
        )
        .unwrap();
        let json = serde_json::to_string(&plan).unwrap();
        let decoded: RuntimePlan = serde_json::from_str(&json).unwrap();
        assert_eq!(plan, decoded);
        decoded.validate().unwrap();
    }
}
