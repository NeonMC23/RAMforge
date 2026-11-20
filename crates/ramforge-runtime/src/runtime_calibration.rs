//! Versioned runtime calibration and estimate-versus-observation analysis.
//!
//! Calibration is evidence, not a promise. Incomparable configurations remain
//! explicitly invalid and are never silently reused.

use serde::{Deserialize, Serialize};

use crate::machine_profile::{Freshness, MachineProfile, Observed, Provenance};
use crate::model_requirements::{ModelRequirements, RequirementBytes, RequirementCertainty};
use crate::runtime_measurement::{
    LifecycleDeltas, LifecycleStage, MemoryComponent, RuntimeAttribution, RuntimeSnapshot,
};

pub const RUNTIME_CALIBRATION_SCHEMA_VERSION: u32 = 1;
pub const KV_TYPE_SOURCE_ADAPTER_DEFAULT: &str =
    "adapter default; native effective-type getter unavailable";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineIdentity {
    pub profile_schema: u32,
    pub cpu_architecture: Option<String>,
    pub cpu_model: Option<String>,
    pub physical_ram_bytes: Option<u64>,
    pub device_count: usize,
    pub device_classes: Vec<String>,
    pub os: String,
    pub kernel: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub name: Option<String>,
    pub file_size_bytes: u64,
    pub tensor_count: usize,
    pub architecture: Option<String>,
    pub quantization: Option<String>,
    /// Non-cryptographic bounded header fingerprint. Never a local path.
    pub fingerprint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeConfiguration {
    pub backend: String,
    pub device_id: Option<String>,
    pub load_mode: Option<String>,
    pub weight_repacking: Option<String>,
    pub requested_context_tokens: Option<u32>,
    pub effective_context_tokens: Option<u32>,
    pub requested_batch_tokens: Option<u32>,
    pub effective_batch_tokens: Option<u32>,
    pub effective_ubatch_tokens: Option<u32>,
    pub effective_sequence_max: Option<u32>,
    pub effective_threads: Option<u32>,
    pub effective_threads_batch: Option<u32>,
    pub kv_type_k: String,
    pub kv_type_v: String,
    pub kv_type_source: String,
    pub state_serialization_size_bytes: Option<u64>,
    pub native_contract_version: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EstimateComparison {
    pub label: String,
    pub predicted: RequirementBytes,
    pub observed: Observed<u64>,
    pub absolute_difference: Option<i128>,
    pub relative_difference: Option<f64>,
    pub valid: bool,
    pub reason: Option<String>,
}

impl EstimateComparison {
    pub fn new(
        label: impl Into<String>,
        predicted: RequirementBytes,
        observed: Observed<u64>,
        valid: bool,
        reason: Option<String>,
    ) -> Self {
        let predicted_bytes = predicted.bytes;
        let observed_bytes = observed.value;
        let absolute_difference = predicted_bytes
            .zip(observed_bytes)
            .map(|(predicted, observed)| i128::from(observed) - i128::from(predicted));
        let relative_difference = match (predicted_bytes, observed_bytes) {
            (Some(0), Some(_)) => None,
            (Some(predicted), Some(observed)) => {
                Some((observed as f64 - predicted as f64) / predicted as f64)
            }
            _ => None,
        };
        Self {
            label: label.into(),
            predicted,
            observed,
            absolute_difference,
            relative_difference,
            valid,
            reason,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        self.predicted.validate()?;
        self.observed
            .validate()
            .map_err(|error| error.to_string())?;
        if self.valid && (self.predicted.bytes.is_none() || self.observed.value.is_none()) {
            return Err("valid comparison requires predicted and observed values".to_string());
        }
        Ok(())
    }
}

/// Compact evidence supplied to a plan evaluator. `valid` is deliberately
/// conservative: incompatible or unknown configurations must not be applied.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationEvidence {
    pub model_fingerprint: Option<String>,
    pub context_tokens: Option<u32>,
    pub batch_tokens: Option<u32>,
    pub backend: String,
    pub observed_runtime_overhead: RequirementBytes,
    pub compatibility: crate::calibration_compatibility::CompatibilityClass,
    pub valid: bool,
    pub reason: Option<String>,
}

impl CalibrationEvidence {
    pub fn is_comparable(
        &self,
        requirements: &ModelRequirements,
        plan: &crate::execution_candidate::ExecutionPlanCandidate,
    ) -> bool {
        self.valid
            && self.compatibility == crate::calibration_compatibility::CompatibilityClass::Exact
            && self.model_fingerprint.is_some()
            && self.model_fingerprint == requirements.model_fingerprint
            && self.context_tokens == Some(plan.context_tokens)
            && self.batch_tokens == Some(plan.batch_tokens)
            && self.backend == plan_backend_label(plan.backend)
    }
}

fn plan_backend_label(backend: crate::execution_candidate::PlanBackend) -> String {
    match backend {
        crate::execution_candidate::PlanBackend::Cpu => "cpu".to_string(),
        crate::execution_candidate::PlanBackend::Gpu => "gpu".to_string(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationConditions {
    pub cache_condition: String,
    pub cache_condition_known: bool,
    pub cgroup_limit_bytes: Observed<u64>,
    pub pressure_class: String,
    pub baseline_rss_bytes: Observed<u64>,
    pub snapshot_count: usize,
    pub missing_stages: Vec<String>,
    pub measurement_complete: bool,
}

impl ObservationConditions {
    pub fn validate(&self) -> Result<(), String> {
        self.cgroup_limit_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        self.baseline_rss_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeCalibration {
    pub schema_version: u32,
    pub observation_only: bool,
    pub captured_at_unix_ms: Option<u64>,
    pub machine: MachineIdentity,
    pub model: ModelIdentity,
    pub runtime: RuntimeConfiguration,
    pub predicted: ModelRequirements,
    pub snapshots: Vec<RuntimeSnapshot>,
    pub deltas: LifecycleDeltas,
    pub comparisons: Vec<EstimateComparison>,
    pub attribution: RuntimeAttribution,
    pub observation: ObservationConditions,
    pub limitations: Vec<String>,
}

impl RuntimeCalibration {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != RUNTIME_CALIBRATION_SCHEMA_VERSION {
            return Err(format!(
                "unsupported calibration schema {}",
                self.schema_version
            ));
        }
        if !self.observation_only {
            return Err("calibration must remain observation-only".to_string());
        }
        if self.snapshots.is_empty() {
            return Err("calibration has no snapshots".to_string());
        }
        self.predicted.validate()?;
        self.attribution.validate()?;
        self.observation.validate()?;
        let mut previous = None;
        for snapshot in &self.snapshots {
            snapshot.validate()?;
            if let Some(previous) = previous {
                if snapshot.sequence <= previous {
                    return Err("snapshot sequence must be strictly increasing".to_string());
                }
            }
            previous = Some(snapshot.sequence);
        }
        for comparison in &self.comparisons {
            comparison.validate()?;
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(|error| error.to_string())
    }

    pub fn from_json(value: serde_json::Value) -> Result<Self, String> {
        let calibration: Self = serde_json::from_value(value).map_err(|error| error.to_string())?;
        calibration.validate()?;
        Ok(calibration)
    }

    pub fn redacted(&self) -> Self {
        self.clone()
    }

    /// Produce plan-comparison evidence only when the measurement is bounded
    /// and attributable enough to be applied conservatively.
    pub fn evidence(&self) -> CalibrationEvidence {
        let overhead = observed_runtime_overhead(&self.deltas, &self.predicted.kv_cache_bytes);
        CalibrationEvidence {
            model_fingerprint: self.predicted.model_fingerprint.clone(),
            context_tokens: self.runtime.effective_context_tokens,
            batch_tokens: self.runtime.requested_batch_tokens,
            backend: self.runtime.backend.clone(),
            observed_runtime_overhead: overhead,
            compatibility:
                crate::calibration_compatibility::CompatibilityClass::InsufficientEvidence,
            valid: false,
            reason: Some(
                "evidence requires compatibility evaluation against a concrete target".to_string(),
            ),
        }
    }
}

/// Build a conservative attribution view from lifecycle snapshots. Components
/// intentionally overlap and are not summed. Exact backend/KV attribution is
/// not available from the pinned runtime.
pub fn build_attribution(
    snapshots: &[RuntimeSnapshot],
    requirements: &ModelRequirements,
) -> RuntimeAttribution {
    let latest = snapshots
        .iter()
        .find(|snapshot| snapshot.stage == LifecycleStage::DecodeSteady)
        .or_else(|| snapshots.last());
    let loaded = snapshots
        .iter()
        .find(|snapshot| snapshot.stage == LifecycleStage::ModelLoaded);
    let os = |value: Option<u64>, detail: &str| MemoryComponent {
        component: detail.to_string(),
        bytes: value,
        certainty: if value.is_some() { "exact" } else { "unknown" }.to_string(),
        provenance: "operating_system".to_string(),
        method: "/proc/self/status, smaps_rollup, or smaps".to_string(),
        overlapping: true,
        detail: None,
    };
    let unknown = |name: &str, detail: &str| MemoryComponent {
        component: name.to_string(),
        bytes: None,
        certainty: "unknown".to_string(),
        provenance: "native API unavailable".to_string(),
        method: "not exposed by the pinned runtime".to_string(),
        overlapping: true,
        detail: Some(detail.to_string()),
    };
    let mut components = vec![
        os(
            latest.and_then(|snapshot| snapshot.process.rss_bytes.as_ref().copied()),
            "process_rss",
        ),
        os(
            latest.and_then(|snapshot| snapshot.process.anonymous_bytes.as_ref().copied()),
            "anonymous_resident",
        ),
        os(
            latest.and_then(|snapshot| snapshot.process.file_backed_bytes.as_ref().copied()),
            "file_backed_resident_pss",
        ),
        os(
            loaded
                .and_then(|snapshot| snapshot.model_mapping.as_ref())
                .map(|mapping| mapping.resident_bytes),
            "model_mapping_resident",
        ),
    ];
    components.push(MemoryComponent {
        component: "kv_cache_estimate".to_string(),
        bytes: requirements.kv_cache_bytes.bytes,
        certainty: match requirements.kv_cache_bytes.certainty {
            RequirementCertainty::Exact => "exact",
            RequirementCertainty::Estimated => "estimated",
            RequirementCertainty::Unknown => "unknown",
        }
        .to_string(),
        provenance: requirements.kv_cache_bytes.provenance.clone(),
        method: "architecture-derived estimate from GGUF metadata and context".to_string(),
        overlapping: true,
        detail: requirements.kv_cache_bytes.detail.clone(),
    });
    components.extend([
        unknown(
            "compute_buffers",
            "native compute buffer allocation is not exposed",
        ),
        unknown(
            "graph_activations",
            "graph and activation memory are not exposed",
        ),
        unknown(
            "backend_buffers",
            "backend buffer allocation is not exposed",
        ),
        unknown(
            "staging_buffers",
            "staging workspace is not exposed for this path",
        ),
    ]);
    RuntimeAttribution {
        components,
        unattributed_bytes: None,
        note: "Process RSS contains all listed categories; components overlap and must not be summed. No exact unattributed amount can be derived safely."
            .to_string(),
        limitations: vec![
            "KV cache is an estimate; the pinned API exposes no exact allocation counter.".to_string(),
            "Backend, graph, activation, and staging allocations are unobservable.".to_string(),
            "Model mapping residency is not tensor use.".to_string(),
            "RSS deltas are not component allocations.".to_string(),
        ],
    }
}

fn observation_conditions(snapshots: &[RuntimeSnapshot]) -> ObservationConditions {
    let required = [
        LifecycleStage::Baseline,
        LifecycleStage::ModelLoaded,
        LifecycleStage::ContextCreated,
        LifecycleStage::PromptProcessed,
        LifecycleStage::DecodeSteady,
        LifecycleStage::ContextDestroyed,
        LifecycleStage::ModelUnloaded,
        LifecycleStage::Final,
    ];
    let missing_stages: Vec<String> = required
        .iter()
        .filter(|stage| !snapshots.iter().any(|snapshot| snapshot.stage == **stage))
        .map(|stage| format!("{stage:?}").to_ascii_lowercase())
        .collect();
    let last = snapshots.last();
    let pressure_class = last
        .and_then(|snapshot| snapshot.system.pressure.as_ref())
        .map(|pressure| {
            let some = pressure.some_avg10.unwrap_or(0);
            if some <= 100 {
                "low"
            } else if some <= 500 {
                "elevated"
            } else {
                "high"
            }
        })
        .unwrap_or("unknown")
        .to_string();
    let baseline_rss = snapshots
        .iter()
        .find(|snapshot| snapshot.stage == LifecycleStage::Baseline)
        .map(|snapshot| snapshot.process.rss_bytes.clone())
        .unwrap_or_else(|| Observed::unavailable("baseline RSS unavailable", 0));
    let cgroup_limit = last
        .map(|snapshot| snapshot.system.cgroup_limit_bytes.clone())
        .unwrap_or_else(|| Observed::unavailable("cgroup limit unavailable", 0));
    let measurement_complete = missing_stages.is_empty();
    ObservationConditions {
        cache_condition: "warm-uncontrolled".to_string(),
        cache_condition_known: true,
        cgroup_limit_bytes: cgroup_limit,
        pressure_class,
        baseline_rss_bytes: baseline_rss,
        snapshot_count: snapshots.len(),
        missing_stages,
        measurement_complete,
    }
}

fn observed_runtime_overhead(
    deltas: &LifecycleDeltas,
    kv_estimate: &RequirementBytes,
) -> RequirementBytes {
    let context = deltas.context_create_rss_bytes;
    let prompt = deltas.prompt_rss_bytes;
    let decode = deltas.decode_rss_bytes;
    let Some(context) = context else {
        return RequirementBytes::unknown("context-creation RSS delta unavailable");
    };
    let prompt = prompt.unwrap_or(0);
    let decode = decode.unwrap_or(0);
    let observed = context
        .max(0)
        .checked_add(prompt.max(0))
        .and_then(|value| value.checked_add(decode.max(0)))
        .and_then(|value| u64::try_from(value).ok());
    let Some(observed) = observed else {
        return RequirementBytes::unknown("runtime RSS delta overflow or unavailable");
    };
    let non_kv = kv_estimate
        .bytes
        .map(|kv| observed.saturating_sub(kv))
        .unwrap_or(observed);
    RequirementBytes {
        bytes: Some(non_kv),
        certainty: RequirementCertainty::Estimated,
        provenance: "observed lifecycle RSS delta minus estimated KV".to_string(),
        detail: Some(
            "context/prompt/decode RSS deltas include backend buffers and page-cache effects; \
             the KV estimate is subtracted once, but overlap remains uncertain"
                .to_string(),
        ),
    }
}

pub fn build_calibration(
    machine: &MachineProfile,
    requirements: &ModelRequirements,
    runtime: RuntimeConfiguration,
    snapshots: Vec<RuntimeSnapshot>,
    attribution: RuntimeAttribution,
) -> RuntimeCalibration {
    let deltas = LifecycleDeltas::from_snapshots(&snapshots);
    let loaded_mapping = snapshots
        .iter()
        .find(|snapshot| snapshot.stage == LifecycleStage::ModelLoaded)
        .and_then(|snapshot| snapshot.model_mapping.clone());
    let observed_mapping = loaded_mapping
        .as_ref()
        .map(|mapping| {
            Observed::known(
                mapping.resident_bytes,
                Provenance::OperatingSystem,
                Freshness::Dynamic,
                snapshots
                    .iter()
                    .find(|snapshot| snapshot.stage == LifecycleStage::ModelLoaded)
                    .map(|snapshot| snapshot.sequence)
                    .unwrap_or(0),
            )
        })
        .unwrap_or_else(|| Observed::unavailable("model mapping residency unavailable", 0));
    let mut comparisons = Vec::new();
    comparisons.push(EstimateComparison::new(
        "tensor_payload_vs_loaded_mapping_resident",
        requirements.tensor_payload_bytes.clone(),
        observed_mapping,
        loaded_mapping.is_some(),
        Some("mapping RSS is resident pages, not total allocation or tensor use".to_string()),
    ));
    let observed_mapped = loaded_mapping
        .as_ref()
        .map(|mapping| {
            Observed::known(
                mapping.mapped_bytes,
                Provenance::OperatingSystem,
                Freshness::Dynamic,
                snapshots
                    .iter()
                    .find(|snapshot| snapshot.stage == LifecycleStage::ModelLoaded)
                    .map(|snapshot| snapshot.sequence)
                    .unwrap_or(0),
            )
        })
        .unwrap_or_else(|| Observed::unavailable("model mapping size unavailable", 0));
    comparisons.push(EstimateComparison::new(
        "gguf_file_bytes_vs_mapping_size",
        requirements.file_bytes.clone(),
        observed_mapped,
        loaded_mapping.is_some(),
        Some("mapping size is file-backed virtual range size, not resident memory".to_string()),
    ));
    let context_delta = deltas
        .context_create_rss_bytes
        .and_then(|value| u64::try_from(value.max(0)).ok());
    let observed_context = context_delta
        .map(|value| {
            Observed::known(
                value,
                Provenance::OperatingSystem,
                Freshness::Dynamic,
                snapshots
                    .last()
                    .map(|snapshot| snapshot.sequence)
                    .unwrap_or(0),
            )
        })
        .unwrap_or_else(|| Observed::unavailable("context RSS delta unavailable", 0));
    comparisons.push(EstimateComparison::new(
        "kv_estimate_vs_context_create_rss_delta",
        requirements.kv_cache_bytes.clone(),
        observed_context,
        false,
        Some(
            "context creation also allocates compute buffers and graph memory; RSS delta is not \
             a KV-cache measurement"
                .to_string(),
        ),
    ));
    let model_load_delta = deltas
        .model_load_rss_bytes
        .and_then(|value| u64::try_from(value.max(0)).ok());
    let observed_load = model_load_delta
        .map(|value| {
            Observed::known(
                value,
                Provenance::OperatingSystem,
                Freshness::Dynamic,
                snapshots
                    .iter()
                    .find(|snapshot| snapshot.stage == LifecycleStage::ModelLoaded)
                    .map(|snapshot| snapshot.sequence)
                    .unwrap_or(0),
            )
        })
        .unwrap_or_else(|| Observed::unavailable("model-load RSS delta unavailable", 0));
    comparisons.push(EstimateComparison::new(
        "resident_weights_plus_kv_vs_model_load_rss_delta",
        requirements.resident_weights_plus_kv_bytes.clone(),
        observed_load,
        false,
        Some(
            "model-load RSS may exclude or include file-backed pages and backend buffers; it is \
             not a component allocation"
                .to_string(),
        ),
    ));
    let observation = observation_conditions(&snapshots);
    RuntimeCalibration {
        schema_version: RUNTIME_CALIBRATION_SCHEMA_VERSION,
        observation_only: true,
        captured_at_unix_ms: snapshots
            .last()
            .and_then(|snapshot| snapshot.captured_at_unix_ms),
        machine: MachineIdentity {
            profile_schema: machine.schema_version,
            cpu_architecture: machine.cpu.architecture.as_ref().cloned(),
            cpu_model: machine.cpu.model.as_ref().cloned(),
            physical_ram_bytes: machine.memory.physical_total_bytes.as_ref().copied(),
            device_count: machine.devices.len(),
            device_classes: {
                let mut classes: Vec<String> = machine
                    .devices
                    .iter()
                    .map(|device| device.kind.clone())
                    .collect();
                classes.sort();
                classes.dedup();
                classes
            },
            os: machine.os.clone(),
            kernel: machine.kernel.clone(),
        },
        model: ModelIdentity {
            name: requirements.model_name.clone(),
            file_size_bytes: requirements.file_bytes.bytes.unwrap_or(0),
            tensor_count: requirements.tensor_count,
            architecture: requirements.architecture.clone(),
            quantization: requirements.quantization.clone(),
            fingerprint: requirements.model_fingerprint.clone(),
        },
        runtime,
        predicted: requirements.clone(),
        snapshots,
        deltas,
        comparisons,
        attribution,
        observation,
        limitations: vec![
            "Lifecycle RSS deltas include page-cache, allocator, and backend effects.".to_string(),
            "KV cache and graph/backend allocations are not exactly attributable.".to_string(),
            "Calibration is valid only for matching model/context/batch configuration.".to_string(),
            "No observation authorizes eviction, release, or prefetch.".to_string(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine_profile::{collect_machine_profile, MachineProfileConfig};
    use crate::runtime_measurement::{capture_snapshot, LifecycleStage};

    fn requirements() -> ModelRequirements {
        ModelRequirements {
            schema_version: crate::model_requirements::MODEL_REQUIREMENTS_SCHEMA_VERSION,
            model_fingerprint: Some("fnv1a64:test:size=100".to_string()),
            model_name: Some("test".to_string()),
            architecture: Some("llama".to_string()),
            quantization: Some("mostly-q4_k".to_string()),
            tensor_count: 1,
            file_bytes: RequirementBytes::exact(100, "test"),
            tensor_payload_bytes: RequirementBytes::exact(100, "test"),
            fully_resident_weight_bytes: RequirementBytes::exact(100, "test"),
            kv_cache_bytes: RequirementBytes::estimated(50, "test"),
            context_tokens: 128,
            batch_tokens: 1,
            model_max_context_tokens: Some(512),
            resident_weights_plus_kv_bytes: RequirementBytes::estimated(150, "test"),
            mapped_weights_plus_kv_bytes: RequirementBytes::estimated(50, "test"),
            unknown_requirements: Vec::new(),
            assumptions: Vec::new(),
        }
    }

    fn attribution() -> RuntimeAttribution {
        RuntimeAttribution {
            components: Vec::new(),
            unattributed_bytes: None,
            note: "test".to_string(),
            limitations: Vec::new(),
        }
    }

    fn runtime() -> RuntimeConfiguration {
        RuntimeConfiguration {
            backend: "cpu".to_string(),
            device_id: None,
            load_mode: Some("mmap".to_string()),
            weight_repacking: None,
            requested_context_tokens: Some(128),
            effective_context_tokens: Some(128),
            requested_batch_tokens: Some(1),
            effective_batch_tokens: Some(1),
            effective_ubatch_tokens: Some(1),
            effective_sequence_max: Some(1),
            effective_threads: Some(1),
            effective_threads_batch: Some(1),
            kv_type_k: "f16".to_string(),
            kv_type_v: "f16".to_string(),
            kv_type_source: "adapter default; native getter unavailable".to_string(),
            state_serialization_size_bytes: None,
            native_contract_version: Some(1),
        }
    }

    fn snapshots() -> Vec<RuntimeSnapshot> {
        let stages = [
            LifecycleStage::Baseline,
            LifecycleStage::ModelLoaded,
            LifecycleStage::ContextCreated,
            LifecycleStage::PromptProcessed,
            LifecycleStage::DecodeSteady,
        ];
        stages
            .into_iter()
            .enumerate()
            .map(|(index, stage)| {
                capture_snapshot(index as u64 + 1, stage, None, Some(1024), None, None)
            })
            .collect()
    }

    #[test]
    fn calibration_round_trips_and_validates() {
        let machine = collect_machine_profile(&MachineProfileConfig::default());
        let calibration = build_calibration(
            &machine,
            &requirements(),
            runtime(),
            snapshots(),
            attribution(),
        );
        calibration.validate().expect("calibration");
        let json = calibration.to_json().expect("json");
        let decoded = RuntimeCalibration::from_json(json).expect("decode");
        assert_eq!(calibration, decoded);
    }

    #[test]
    fn unchecked_evidence_is_not_applicable() {
        let machine = collect_machine_profile(&MachineProfileConfig::default());
        let calibration = build_calibration(
            &machine,
            &requirements(),
            runtime(),
            snapshots(),
            attribution(),
        );
        let evidence = calibration.evidence();
        assert!(!evidence.valid);
        let requirements = requirements();
        let mut plan = crate::execution_candidate::ExecutionPlanCandidate {
            schema_version: crate::execution_candidate::EXECUTION_CANDIDATE_SCHEMA_VERSION,
            id: "test".to_string(),
            profile: crate::execution_candidate::PlanProfile::Custom,
            backend: crate::execution_candidate::PlanBackend::Cpu,
            device_id: None,
            cpu_threads: 1,
            context_tokens: 128,
            batch_tokens: 1,
            ram_budget_bytes: Some(1024),
            vram_budget_bytes: None,
            mapping: crate::execution_candidate::MappingStrategy::ResidentWeights,
            assumptions: Vec::new(),
            unknown_information: Vec::new(),
        };
        assert!(!evidence.is_comparable(&requirements, &plan));
        plan.context_tokens = 256;
        assert!(!evidence.is_comparable(&requirements, &plan));
    }

    #[test]
    fn comparison_marks_incomparable_values_invalid() {
        let comparison = EstimateComparison::new(
            "kv_vs_rss",
            RequirementBytes::estimated(50, "estimate"),
            Observed::unavailable("missing", 1),
            false,
            Some("missing observation".to_string()),
        );
        assert!(!comparison.valid);
        comparison.validate().expect("comparison");
        assert_eq!(comparison.relative_difference, None);
    }

    #[test]
    fn zero_prediction_does_not_emit_relative_percentage() {
        let comparison = EstimateComparison::new(
            "zero",
            RequirementBytes::exact(0, "zero"),
            Observed::known(10, Provenance::OperatingSystem, Freshness::Dynamic, 1),
            true,
            None,
        );
        assert_eq!(comparison.relative_difference, None);
        assert_eq!(comparison.absolute_difference, Some(10));
    }
}
