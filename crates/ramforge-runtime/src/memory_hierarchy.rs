//! Memory hierarchy and explicit placement model.
//!
//! Categories overlap and are never added into a fabricated process total.
//! Placement is only marked confirmed where the selected plan/backend provides
//! evidence; otherwise it remains unknown.

use serde::{Deserialize, Serialize};

use crate::execution_candidate::Feasibility;
use crate::machine_profile::MachineProfile;
use crate::model_requirements::{ModelRequirements, RequirementBytes, RequirementCertainty};
use crate::runtime_plan::{ExecutionMode, RuntimePlan};

pub const MEMORY_HIERARCHY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryCategory {
    HostRam,
    DeviceMemory,
    DeviceMemoryAvailable,
    FileBackedMapping,
    AnonymousProcess,
    KvCache,
    ComputeGraph,
    BackendBuffers,
    Staging,
    StorageBacked,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationQuality {
    Observed,
    Estimated,
    Unknown,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryCategoryObservation {
    pub category: MemoryCategory,
    pub bytes: Option<u64>,
    pub quality: ObservationQuality,
    pub provenance: String,
    pub overlapping: bool,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementState {
    GpuResident,
    HostResident,
    FileBackedMapped,
    StorageBacked,
    Staged,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentPlacement {
    pub component: String,
    pub placement: PlacementState,
    pub confirmed: bool,
    pub evidence: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceMemory {
    pub id: String,
    pub kind: String,
    pub total_bytes: Option<u64>,
    pub free_bytes: Option<u64>,
    pub selectable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryHierarchy {
    pub schema_version: u32,
    pub host_total_bytes: Option<u64>,
    pub host_available_bytes: Option<u64>,
    pub cgroup_limit_bytes: Option<u64>,
    pub configured_budget_bytes: Option<u64>,
    pub storage_capacity_bytes: Option<u64>,
    pub storage_available_bytes: Option<u64>,
    pub devices: Vec<DeviceMemory>,
    pub categories: Vec<MemoryCategoryObservation>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacementModel {
    pub profile: String,
    pub components: Vec<ComponentPlacement>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HierarchyFeasibility {
    pub status: Feasibility,
    pub host_requirement: RequirementBytes,
    pub device_requirement: RequirementBytes,
    pub storage_requirement: RequirementBytes,
    pub reasons: Vec<String>,
    pub unknown_information: Vec<String>,
}

impl MemoryHierarchy {
    pub fn to_json(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(|error| error.to_string())
    }
}

impl PlacementModel {
    pub fn to_json(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(|error| error.to_string())
    }
}

pub fn build_memory_hierarchy(machine: &MachineProfile) -> MemoryHierarchy {
    let devices = machine
        .devices
        .iter()
        .map(|device| DeviceMemory {
            id: device.id.clone(),
            kind: device.kind.clone(),
            total_bytes: device.memory_total_bytes.as_ref().copied(),
            free_bytes: device.memory_free_bytes.as_ref().copied(),
            selectable: device.selectable,
        })
        .collect();
    MemoryHierarchy {
        schema_version: MEMORY_HIERARCHY_SCHEMA_VERSION,
        host_total_bytes: machine.memory.physical_total_bytes.as_ref().copied(),
        host_available_bytes: machine.memory.os_available_bytes.as_ref().copied(),
        cgroup_limit_bytes: machine.memory.cgroup.limit_bytes.as_ref().copied(),
        configured_budget_bytes: machine.memory.configured_budget_bytes.as_ref().copied(),
        storage_capacity_bytes: machine
            .storage
            .first()
            .and_then(|storage| storage.capacity_bytes.as_ref().copied()),
        storage_available_bytes: machine
            .storage
            .first()
            .and_then(|storage| storage.available_bytes.as_ref().copied()),
        devices,
        categories: vec![
            MemoryCategoryObservation {
                category: MemoryCategory::HostRam,
                bytes: machine.memory.physical_total_bytes.as_ref().copied(),
                quality: observation_quality(
                    machine.memory.physical_total_bytes.as_ref().is_some(),
                ),
                provenance: "OS MemTotal".to_string(),
                overlapping: false,
                detail: None,
            },
            {
                let mut non_cpu = machine
                    .devices
                    .iter()
                    .filter(|device| device.kind != "cpu")
                    .filter_map(|device| device.memory_total_bytes.as_ref().copied());
                let (bytes, quality, detail) = match (non_cpu.next(), non_cpu.next()) {
                    (Some(total), None) => (
                        Some(total),
                        ObservationQuality::Observed,
                        "single non-CPU device; per-device value is reported separately"
                            .to_string(),
                    ),
                    (Some(_), Some(_)) => (
                        None,
                        ObservationQuality::Unknown,
                        "multiple non-CPU devices; no universal GPU total is synthesized"
                            .to_string(),
                    ),
                    _ => (
                        None,
                        ObservationQuality::Unsupported,
                        "no non-CPU device is present in the backend registry".to_string(),
                    ),
                };
                MemoryCategoryObservation {
                    category: MemoryCategory::DeviceMemory,
                    bytes,
                    quality,
                    provenance: "native device snapshot".to_string(),
                    overlapping: false,
                    detail: Some(detail),
                }
            },
            {
                let mut non_cpu = machine
                    .devices
                    .iter()
                    .filter(|device| device.kind != "cpu")
                    .filter_map(|device| device.memory_free_bytes.as_ref().copied());
                let (bytes, quality, detail) = match (non_cpu.next(), non_cpu.next()) {
                    (Some(free), None) => (
                        Some(free),
                        ObservationQuality::Observed,
                        "single non-CPU device; a point-in-time free value is not a reservation"
                            .to_string(),
                    ),
                    (Some(_), Some(_)) => (
                        None,
                        ObservationQuality::Unknown,
                        "multiple non-CPU devices; per-device free values are listed separately"
                            .to_string(),
                    ),
                    _ => (
                        None,
                        ObservationQuality::Unsupported,
                        "no non-CPU device is present in the backend registry".to_string(),
                    ),
                };
                MemoryCategoryObservation {
                    category: MemoryCategory::DeviceMemoryAvailable,
                    bytes,
                    quality,
                    provenance: "native device snapshot".to_string(),
                    overlapping: true,
                    detail: Some(detail),
                }
            },
            MemoryCategoryObservation {
                category: MemoryCategory::AnonymousProcess,
                bytes: machine.memory.process_rss_bytes.as_ref().copied(),
                quality: observation_quality(machine.memory.process_rss_bytes.as_ref().is_some()),
                provenance: "process RSS (not anonymous-only)".to_string(),
                overlapping: true,
                detail: Some(
                    "RSS is a process total, not an anonymous allocation category".to_string(),
                ),
            },
            MemoryCategoryObservation {
                category: MemoryCategory::FileBackedMapping,
                bytes: None,
                quality: ObservationQuality::Unknown,
                provenance: "not sampled by this hierarchy builder".to_string(),
                overlapping: true,
                detail: Some(
                    "mapping size is known only after load; resident mapped pages are not tensor use"
                        .to_string(),
                ),
            },
            MemoryCategoryObservation {
                category: MemoryCategory::KvCache,
                bytes: None,
                quality: ObservationQuality::Unknown,
                provenance: "derived estimate in model requirements".to_string(),
                overlapping: false,
                detail: Some(
                    "the estimate depends on context, batch, KV type, and architecture; it is not a native readback"
                        .to_string(),
                ),
            },
            MemoryCategoryObservation {
                category: MemoryCategory::ComputeGraph,
                bytes: None,
                quality: ObservationQuality::Unsupported,
                provenance: "native graph allocation is not exposed".to_string(),
                overlapping: true,
                detail: Some(
                    "graph and compute workspace may be allocated and freed per decode".to_string(),
                ),
            },
            MemoryCategoryObservation {
                category: MemoryCategory::BackendBuffers,
                bytes: None,
                quality: ObservationQuality::Unsupported,
                provenance: "native backend buffer accounting is not exposed".to_string(),
                overlapping: true,
                detail: Some(
                    "CPU repacking and backend copies may add buffers not visible to RSS attribution"
                        .to_string(),
                ),
            },
            MemoryCategoryObservation {
                category: MemoryCategory::Staging,
                bytes: None,
                quality: ObservationQuality::Unsupported,
                provenance: "no tensor staging buffer is exposed".to_string(),
                overlapping: true,
                detail: Some(
                    "page-level prefetch plans exist separately; no caller-owned tensor/layer staging buffer is implemented"
                        .to_string(),
                ),
            },
            MemoryCategoryObservation {
                category: MemoryCategory::StorageBacked,
                bytes: machine
                    .storage
                    .first()
                    .and_then(|storage| storage.available_bytes.as_ref().copied()),
                quality: observation_quality(!machine.storage.is_empty()),
                provenance: "filesystem statvfs".to_string(),
                overlapping: false,
                detail: Some("storage availability is not model residency".to_string()),
            },
        ],
        notes: vec![
            "Categories overlap and must not be summed.".to_string(),
            "Resident model pages are not equivalent to tensor use.".to_string(),
            "Compute, graph, backend, and staging bytes are not universally exposed.".to_string(),
            "Free device memory is a point-in-time observation, not a reservation for this process."
                .to_string(),
            "Storage totals refer to the first observed filesystem; other filesystems are listed in the machine profile."
                .to_string(),
        ],
    }
}

fn estimate_gpu_weight_bytes(tensor_bytes: u64, layer_count: u64, gpu_layers: u32) -> u64 {
    let total_with_output = layer_count.saturating_add(1);
    let layers = u64::from(gpu_layers).min(total_with_output);
    (u128::from(tensor_bytes) * u128::from(layers) / u128::from(total_with_output))
        .try_into()
        .unwrap_or(u64::MAX)
}

fn single_device_free_bytes(machine: &MachineProfile) -> Option<u64> {
    let mut devices = machine
        .devices
        .iter()
        .filter(|device| device.kind != "cpu")
        .filter_map(|device| device.memory_free_bytes.as_ref().copied());
    let first = devices.next()?;
    devices.next().is_none().then_some(first)
}

fn observation_quality(known: bool) -> ObservationQuality {
    if known {
        ObservationQuality::Observed
    } else {
        ObservationQuality::Unknown
    }
}

pub fn build_placement_model(
    requirements: &ModelRequirements,
    plan: &RuntimePlan,
) -> PlacementModel {
    build_placement_model_for_plan(plan, requirements.file_bytes.bytes)
}

/// Build placement from a plan alone. Used by the service, which does not
/// retain the full model-requirements snapshot.
pub fn build_placement_model_for_plan(
    plan: &RuntimePlan,
    model_bytes: Option<u64>,
) -> PlacementModel {
    let file_backed = plan.effective.load_mode.starts_with("mmap");
    let mut components = vec![
        ComponentPlacement {
            component: "model_weights".to_string(),
            placement: if file_backed && !plan.effective.weight_repacking {
                PlacementState::FileBackedMapped
            } else {
                PlacementState::Unknown
            },
            confirmed: file_backed && !plan.effective.weight_repacking,
            evidence: if file_backed {
                "plan selects mmap; repacked or backend copies are not independently observed"
                    .to_string()
            } else {
                "no mapping evidence".to_string()
            },
        },
        ComponentPlacement {
            component: "kv_cache".to_string(),
            placement: if plan.effective.backend == "cpu" {
                PlacementState::HostResident
            } else {
                PlacementState::Unknown
            },
            confirmed: plan.effective.backend == "cpu",
            evidence: "context backend is CPU; exact KV allocation not exposed".to_string(),
        },
        ComponentPlacement {
            component: "compute_graph".to_string(),
            placement: PlacementState::Unknown,
            confirmed: false,
            evidence: "native compute/graph allocation is not exposed".to_string(),
        },
        ComponentPlacement {
            component: "backend_buffers".to_string(),
            placement: PlacementState::Unknown,
            confirmed: false,
            evidence: "backend buffer placement is not exposed".to_string(),
        },
        ComponentPlacement {
            component: "staging".to_string(),
            placement: PlacementState::Unknown,
            confirmed: false,
            evidence: "no controlled staging path is enabled".to_string(),
        },
    ];
    if plan.effective.execution_mode != ExecutionMode::Cpu {
        components.push(ComponentPlacement {
            component: "gpu_offload".to_string(),
            placement: PlacementState::GpuResident,
            confirmed: false,
            evidence: format!(
                "plan requests {} offloaded layers; native placement readback is not captured by this builder",
                plan.effective.gpu_layers
            ),
        });
    }
    PlacementModel {
        profile: format!("{:?}", plan.profile).to_ascii_lowercase(),
        components,
        notes: vec![
            "Only confirmed placements are evidence-backed.".to_string(),
            "File-backed mapping is not proof of resident or physically accessed weights."
                .to_string(),
            format!("model file bytes: {model_bytes:?}"),
        ],
    }
}

pub fn evaluate_hierarchy_feasibility(
    machine: &MachineProfile,
    requirements: &ModelRequirements,
    plan: &RuntimePlan,
) -> HierarchyFeasibility {
    let mut reasons = Vec::new();
    let mut unknown = Vec::new();
    let kv = requirements.kv_cache_bytes.clone();
    let mut host = kv.clone();
    if plan.effective.weight_repacking {
        host = host.checked_add(&requirements.fully_resident_weight_bytes);
        reasons.push(
            "repacking may create anonymous weight buffers; estimate uses tensor payload size"
                .to_string(),
        );
    }
    unknown.push("compute/graph/backend/staging allocations are not modeled".to_string());
    if plan.effective.load_mode.starts_with("mmap") {
        reasons.push(
            "file-backed weights may exceed the host budget and refault; the host requirement is KV-only"
                .to_string(),
        );
    }
    if machine.storage.len() > 1 {
        unknown.push(
            "storage feasibility uses the first observed filesystem, not necessarily the model filesystem"
                .to_string(),
        );
    }
    let device = if plan.effective.execution_mode == ExecutionMode::Cpu {
        RequirementBytes::exact(0, "CPU backend")
    } else {
        match (
            requirements.tensor_payload_bytes.bytes,
            plan.architecture
                .as_ref()
                .and_then(|architecture| architecture.layer_count),
        ) {
            (Some(tensor_bytes), Some(layer_count)) if layer_count > 0 => {
                let bytes =
                    estimate_gpu_weight_bytes(tensor_bytes, layer_count, plan.effective.gpu_layers);
                reasons.push(
                    "GPU weight estimate assumes uniform bytes per layer and excludes KV, compute, graph, and backend buffers"
                        .to_string(),
                );
                RequirementBytes::estimated(bytes, "uniform per-layer weight estimate")
            }
            _ => {
                unknown.push(
                    "GPU weight estimate needs tensor payload and architecture layer count"
                        .to_string(),
                );
                RequirementBytes::unknown("GPU allocation is not verified")
            }
        }
    };
    let storage = requirements.file_bytes.clone();
    let budget = machine
        .memory
        .cgroup
        .limit_bytes
        .as_ref()
        .copied()
        .or_else(|| machine.memory.configured_budget_bytes.as_ref().copied());
    let mut status;
    if plan.effective.execution_mode != ExecutionMode::Cpu {
        status = Feasibility::Unknown;
        reasons.push(
            "GPU execution is planning-only in this adapter; native allocation and placement are not verified"
                .to_string(),
        );
        if let (Some(free), Some(required)) = (single_device_free_bytes(machine), device.bytes) {
            if required > free {
                status = Feasibility::Infeasible;
                reasons.push(format!(
                    "estimated GPU weight requirement {required} exceeds observed free device memory {free}"
                ));
            } else {
                status = Feasibility::ConditionallyFeasible;
                reasons.push(
                    "estimated GPU weights fit observed free device memory, but KV, compute, graph, and backend buffers are unmodeled"
                        .to_string(),
                );
            }
        } else {
            unknown.push("free device memory or GPU requirement is unavailable".to_string());
        }
    } else if let Some(budget) = budget {
        if let Some(required) = host.bytes {
            if required > budget {
                status = Feasibility::Infeasible;
                reasons.push(format!(
                    "estimated host requirement {required} exceeds budget {budget}"
                ));
            } else {
                status = Feasibility::ConditionallyFeasible;
                reasons.push(
                    "host requirement is within budget but excludes unknown runtime allocations"
                        .to_string(),
                );
            }
        } else {
            status = Feasibility::Unknown;
        }
    } else {
        status = Feasibility::Unknown;
        unknown.push("no finite cgroup limit or configured budget".to_string());
    }
    if let (Some(available), Some(required)) = (
        machine
            .storage
            .first()
            .and_then(|storage| storage.available_bytes.as_ref().copied()),
        storage.bytes,
    ) {
        if required > available {
            status = Feasibility::Infeasible;
            reasons.push(format!(
                "model file requires {required} bytes but storage has {available} available"
            ));
        }
    } else {
        unknown.push("storage availability or model file size is unknown".to_string());
    }
    if host.certainty != RequirementCertainty::Exact {
        reasons.push("host requirement contains estimates".to_string());
    }
    HierarchyFeasibility {
        status,
        host_requirement: host,
        device_requirement: device,
        storage_requirement: storage,
        reasons,
        unknown_information: unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine_profile::{collect_machine_profile, MachineProfileConfig};
    use crate::runtime_plan::{CustomRuntimeRequest, RuntimeProfileKind};

    fn requirements() -> ModelRequirements {
        ModelRequirements {
            schema_version: crate::model_requirements::MODEL_REQUIREMENTS_SCHEMA_VERSION,
            model_fingerprint: Some("fnv:test".to_string()),
            model_name: Some("test".to_string()),
            architecture: Some("llama".to_string()),
            architecture_info: None,
            quantization: Some("q4".to_string()),
            tensor_count: 1,
            file_bytes: RequirementBytes::exact(1000, "test"),
            tensor_payload_bytes: RequirementBytes::exact(900, "test"),
            fully_resident_weight_bytes: RequirementBytes::exact(900, "test"),
            kv_cache_bytes: RequirementBytes::estimated(100, "test"),
            context_tokens: 128,
            batch_tokens: 1,
            model_max_context_tokens: Some(128),
            resident_weights_plus_kv_bytes: RequirementBytes::estimated(1000, "test"),
            mapped_weights_plus_kv_bytes: RequirementBytes::estimated(100, "test"),
            unknown_requirements: Vec::new(),
            assumptions: Vec::new(),
        }
    }

    #[test]
    fn hierarchy_is_explicit_and_non_double_counting() {
        let machine = collect_machine_profile(&MachineProfileConfig::default());
        let hierarchy = build_memory_hierarchy(&machine);
        assert_eq!(hierarchy.schema_version, 1);
        assert!(hierarchy
            .categories
            .iter()
            .any(|category| category.category == MemoryCategory::HostRam));
        assert!(hierarchy
            .notes
            .iter()
            .any(|note| note.contains("must not be summed")));
    }

    #[test]
    fn placement_does_not_confirm_unobserved_backend_memory() {
        let machine = collect_machine_profile(&MachineProfileConfig::default());
        let plan = RuntimePlan::resolve(
            &machine,
            &requirements(),
            "/m",
            RuntimeProfileKind::Balanced,
            &CustomRuntimeRequest::default(),
        )
        .unwrap();
        let placement = build_placement_model(&requirements(), &plan);
        assert!(placement
            .components
            .iter()
            .any(|component| component.component == "compute_graph" && !component.confirmed));
        assert!(placement
            .components
            .iter()
            .any(|component| component.component == "model_weights"));
    }

    #[test]
    fn gpu_weight_estimate_is_uniform_per_layer_and_capped() {
        assert_eq!(estimate_gpu_weight_bytes(1_000, 4, 0), 0);
        assert_eq!(estimate_gpu_weight_bytes(1_000, 4, 2), 400);
        assert_eq!(estimate_gpu_weight_bytes(1_000, 4, 5), 1_000);
        assert_eq!(estimate_gpu_weight_bytes(1_000, 4, 99), 1_000);
    }

    #[test]
    fn storage_shortfall_is_infeasible() {
        let machine = collect_machine_profile(&MachineProfileConfig::default());
        let plan = RuntimePlan::resolve(
            &machine,
            &requirements(),
            "/m",
            RuntimeProfileKind::Balanced,
            &CustomRuntimeRequest::default(),
        )
        .unwrap();
        let result = evaluate_hierarchy_feasibility(&machine, &requirements(), &plan);
        assert_ne!(result.status, Feasibility::Feasible);
        assert!(!result.unknown_information.is_empty());
    }
}
