//! Explicit calibration compatibility and evidence-use policy.
//!
//! A matching model fingerprint and runtime context are necessary but not
//! sufficient. Machine identity, backend/device, load mode, repacking, effective
//! execution configuration, observation conditions, and freshness are all
//! considered. Only exact matches may influence feasibility.

use serde::{Deserialize, Serialize};

use crate::machine_profile::MachineProfile;
use crate::model_requirements::ModelRequirements;
use crate::runtime_calibration::{
    CalibrationEvidence, ObservationConditions, RuntimeCalibration, RuntimeConfiguration,
};

pub const CALIBRATION_COMPATIBILITY_SCHEMA_VERSION: u32 = 1;
/// Conservative default: calibration older than this is not applied.
pub const MAX_CALIBRATION_AGE_MS: u64 = 7 * 24 * 60 * 60 * 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompatibilityClass {
    Exact,
    CompatibleWithConstraints,
    Incomparable,
    Incompatible,
    InsufficientEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceUse {
    Apply,
    DisplayOnly,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DimensionStatus {
    Match,
    Constraint,
    Mismatch,
    Missing,
    Stale,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DimensionResult {
    pub dimension: String,
    pub status: DimensionStatus,
    pub calibration_value: Option<String>,
    pub current_value: Option<String>,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationCompatibility {
    pub schema_version: u32,
    pub classification: CompatibilityClass,
    pub evidence_use: EvidenceUse,
    pub dimensions: Vec<DimensionResult>,
    pub matched: Vec<String>,
    pub mismatched: Vec<String>,
    pub missing: Vec<String>,
    pub limitations: Vec<String>,
    pub reasons: Vec<String>,
}

impl CalibrationCompatibility {
    pub fn can_apply(&self) -> bool {
        self.evidence_use == EvidenceUse::Apply
    }
}

#[derive(Debug, Clone)]
pub struct CompatibilityTarget {
    pub machine: MachineProfile,
    pub requirements: ModelRequirements,
    pub runtime: RuntimeConfiguration,
    pub observation: ObservationConditions,
    pub now_unix_ms: Option<u64>,
}

impl RuntimeCalibration {
    /// Compare this calibration with a current execution target.
    pub fn compatibility(&self, target: &CompatibilityTarget) -> CalibrationCompatibility {
        let mut builder = CompatibilityBuilder::default();

        builder.compare_required_string(
            "model.fingerprint",
            self.model.fingerprint.as_deref(),
            target.requirements.model_fingerprint.as_deref(),
        );
        builder.compare_u64(
            "model.file_size",
            Some(self.model.file_size_bytes),
            target.requirements.file_bytes.bytes,
        );
        builder.compare_usize(
            "model.tensor_count",
            Some(self.model.tensor_count),
            Some(target.requirements.tensor_count),
        );
        builder.compare_required_string(
            "model.architecture",
            self.model.architecture.as_deref(),
            target.requirements.architecture.as_deref(),
        );
        builder.compare_required_string(
            "model.quantization",
            self.model.quantization.as_deref(),
            target.requirements.quantization.as_deref(),
        );

        builder.compare_required_string(
            "machine.os",
            Some(self.machine.os.as_str()),
            Some(target.machine.os.as_str()),
        );
        builder.compare_required_string(
            "machine.kernel",
            self.machine.kernel.as_deref(),
            target.machine.kernel.as_deref(),
        );
        builder.compare_required_string(
            "machine.cpu_architecture",
            self.machine.cpu_architecture.as_deref(),
            target.machine.cpu.architecture.as_ref().map(String::as_str),
        );
        builder.compare_required_string(
            "machine.cpu_model",
            self.machine.cpu_model.as_deref(),
            target.machine.cpu.model.as_ref().map(String::as_str),
        );
        builder.compare_u64(
            "machine.physical_ram",
            self.machine.physical_ram_bytes,
            target.machine.memory.physical_total_bytes.as_ref().copied(),
        );
        builder.compare_usize(
            "machine.device_count",
            Some(self.machine.device_count),
            Some(target.machine.devices.len()),
        );
        let current_classes = {
            let mut classes: Vec<String> = target
                .machine
                .devices
                .iter()
                .map(|device| device.kind.clone())
                .collect();
            classes.sort();
            classes.dedup();
            classes
        };
        builder.compare_vec(
            "machine.device_classes",
            Some(&self.machine.device_classes),
            Some(&current_classes),
        );

        builder.compare_required_string(
            "backend",
            Some(self.runtime.backend.as_str()),
            Some(target.runtime.backend.as_str()),
        );
        builder.compare_optional_string(
            "device_id",
            self.runtime.device_id.as_deref(),
            target.runtime.device_id.as_deref(),
        );
        builder.compare_optional_string(
            "runtime.native_contract_version",
            self.runtime
                .native_contract_version
                .map(|value| value.to_string())
                .as_deref(),
            target
                .runtime
                .native_contract_version
                .map(|value| value.to_string())
                .as_deref(),
        );
        builder.compare_optional_string(
            "runtime.load_mode",
            self.runtime.load_mode.as_deref(),
            target.runtime.load_mode.as_deref(),
        );
        builder.compare_optional_string(
            "runtime.weight_repacking",
            self.runtime.weight_repacking.as_deref(),
            target.runtime.weight_repacking.as_deref(),
        );
        builder.compare_optional_u32(
            "runtime.requested_context",
            self.runtime.requested_context_tokens,
            target.runtime.requested_context_tokens,
        );
        builder.compare_optional_u32(
            "runtime.effective_context",
            self.runtime.effective_context_tokens,
            target.runtime.effective_context_tokens,
        );
        builder.compare_optional_u32(
            "runtime.requested_batch",
            self.runtime.requested_batch_tokens,
            target.runtime.requested_batch_tokens,
        );
        builder.compare_optional_u32(
            "runtime.effective_batch",
            self.runtime.effective_batch_tokens,
            target.runtime.effective_batch_tokens,
        );
        builder.compare_optional_u32(
            "runtime.effective_ubatch",
            self.runtime.effective_ubatch_tokens,
            target.runtime.effective_ubatch_tokens,
        );
        builder.compare_optional_u32(
            "runtime.effective_sequences",
            self.runtime.effective_sequence_max,
            target.runtime.effective_sequence_max,
        );
        builder.compare_optional_u32(
            "runtime.effective_threads",
            self.runtime.effective_threads,
            target.runtime.effective_threads,
        );
        builder.compare_optional_u32(
            "runtime.effective_threads_batch",
            self.runtime.effective_threads_batch,
            target.runtime.effective_threads_batch,
        );
        builder.compare_required_string(
            "runtime.kv_type_k",
            Some(self.runtime.kv_type_k.as_str()),
            Some(target.runtime.kv_type_k.as_str()),
        );
        builder.compare_required_string(
            "runtime.kv_type_v",
            Some(self.runtime.kv_type_v.as_str()),
            Some(target.runtime.kv_type_v.as_str()),
        );
        builder.compare_required_string(
            "runtime.kv_type_source",
            Some(self.runtime.kv_type_source.as_str()),
            Some(target.runtime.kv_type_source.as_str()),
        );

        builder.compare_observation(self, target);
        builder.finish()
    }

    /// Evidence for plan evaluation. Only exact compatibility is applied.
    pub fn evidence_for(
        &self,
        target: &CompatibilityTarget,
    ) -> (CalibrationEvidence, CalibrationCompatibility) {
        let compatibility = self.compatibility(target);
        let mut evidence = self.evidence();
        evidence.valid = compatibility.can_apply();
        evidence.compatibility = compatibility.classification;
        evidence.reason = if evidence.valid {
            None
        } else {
            Some(format!(
                "calibration is {:?}, evidence use {:?}",
                compatibility.classification, compatibility.evidence_use
            ))
        };
        (evidence, compatibility)
    }
}

#[derive(Default)]
struct CompatibilityBuilder {
    dimensions: Vec<DimensionResult>,
}

impl CompatibilityBuilder {
    fn push(
        &mut self,
        dimension: &str,
        status: DimensionStatus,
        calibration: Option<String>,
        current: Option<String>,
        detail: Option<String>,
    ) {
        self.dimensions.push(DimensionResult {
            dimension: dimension.to_string(),
            status,
            calibration_value: calibration,
            current_value: current,
            detail,
        });
    }

    fn compare_required_string(
        &mut self,
        name: &str,
        calibration: Option<&str>,
        current: Option<&str>,
    ) {
        match (calibration, current) {
            (Some(calibration), Some(current)) if calibration == current => self.push(
                name,
                DimensionStatus::Match,
                Some(calibration.to_string()),
                Some(current.to_string()),
                None,
            ),
            (Some(calibration), Some(current)) => self.push(
                name,
                DimensionStatus::Mismatch,
                Some(calibration.to_string()),
                Some(current.to_string()),
                None,
            ),
            _ => self.push(
                name,
                DimensionStatus::Missing,
                calibration.map(str::to_string),
                current.map(str::to_string),
                Some("one or both values unavailable".to_string()),
            ),
        }
    }

    fn compare_optional_string(
        &mut self,
        name: &str,
        calibration: Option<&str>,
        current: Option<&str>,
    ) {
        match (calibration, current) {
            (None, None) => self.push(
                name,
                DimensionStatus::Missing,
                None,
                None,
                Some("both values unavailable; cannot establish compatibility".to_string()),
            ),
            _ => self.compare_required_string(name, calibration, current),
        }
    }

    fn compare_u64(&mut self, name: &str, calibration: Option<u64>, current: Option<u64>) {
        self.compare_required_string(
            name,
            calibration.map(|value| value.to_string()).as_deref(),
            current.map(|value| value.to_string()).as_deref(),
        );
    }

    fn compare_usize(&mut self, name: &str, calibration: Option<usize>, current: Option<usize>) {
        self.compare_required_string(
            name,
            calibration.map(|value| value.to_string()).as_deref(),
            current.map(|value| value.to_string()).as_deref(),
        );
    }

    fn compare_optional_u32(&mut self, name: &str, calibration: Option<u32>, current: Option<u32>) {
        match (calibration, current) {
            (None, None) => self.push(
                name,
                DimensionStatus::Missing,
                None,
                None,
                Some("both effective values unavailable".to_string()),
            ),
            _ => self.compare_u64(name, calibration.map(u64::from), current.map(u64::from)),
        }
    }

    fn compare_vec(
        &mut self,
        name: &str,
        calibration: Option<&Vec<String>>,
        current: Option<&Vec<String>>,
    ) {
        self.compare_required_string(
            name,
            calibration.map(|value| value.join(",")).as_deref(),
            current.map(|value| value.join(",")).as_deref(),
        );
    }

    fn compare_observation(
        &mut self,
        calibration: &RuntimeCalibration,
        target: &CompatibilityTarget,
    ) {
        match (
            calibration.observation.cache_condition_known,
            target.observation.cache_condition_known,
        ) {
            (true, true) => self.compare_required_string(
                "observation.cache_condition",
                Some(calibration.observation.cache_condition.as_str()),
                Some(target.observation.cache_condition.as_str()),
            ),
            _ => self.push(
                "observation.cache_condition",
                DimensionStatus::Missing,
                Some(calibration.observation.cache_condition.clone()),
                Some(target.observation.cache_condition.clone()),
                Some("cache condition is unknown for one or both records".to_string()),
            ),
        }
        match (
            calibration.observation.pressure_class.as_str(),
            target.observation.pressure_class.as_str(),
        ) {
            (left, right) if left == right => self.push(
                "observation.pressure_class",
                DimensionStatus::Match,
                Some(left.to_string()),
                Some(right.to_string()),
                None,
            ),
            (left, right) => self.push(
                "observation.pressure_class",
                DimensionStatus::Constraint,
                Some(left.to_string()),
                Some(right.to_string()),
                Some("memory pressure is dynamic; evidence is display-only".to_string()),
            ),
        }
        let calibration_cgroup = calibration.observation.cgroup_limit_bytes.as_ref().copied();
        let current_cgroup = target.observation.cgroup_limit_bytes.as_ref().copied();
        match (calibration_cgroup, current_cgroup) {
            (None, None) => self.push(
                "observation.cgroup_limit",
                DimensionStatus::Match,
                None,
                None,
                Some("both cgroup limits unavailable; no cgroup-specific claim".to_string()),
            ),
            _ => self.compare_u64(
                "observation.cgroup_limit",
                calibration_cgroup,
                current_cgroup,
            ),
        }
        let age = target
            .now_unix_ms
            .zip(calibration.captured_at_unix_ms)
            .map(|(now, then)| now.saturating_sub(then));
        match age {
            Some(age) if age <= MAX_CALIBRATION_AGE_MS => self.push(
                "observation.freshness",
                DimensionStatus::Match,
                calibration
                    .captured_at_unix_ms
                    .map(|value| value.to_string()),
                target.now_unix_ms.map(|value| value.to_string()),
                Some(format!("age_ms={age}")),
            ),
            Some(age) => self.push(
                "observation.freshness",
                DimensionStatus::Stale,
                calibration
                    .captured_at_unix_ms
                    .map(|value| value.to_string()),
                target.now_unix_ms.map(|value| value.to_string()),
                Some(format!("age_ms={age} exceeds {MAX_CALIBRATION_AGE_MS}")),
            ),
            None => self.push(
                "observation.freshness",
                DimensionStatus::Missing,
                None,
                None,
                Some("timestamp unavailable".to_string()),
            ),
        }
        self.push(
            "observation.measurement_complete",
            if calibration.observation.measurement_complete
                && target.observation.measurement_complete
            {
                DimensionStatus::Match
            } else {
                DimensionStatus::Missing
            },
            Some(calibration.observation.measurement_complete.to_string()),
            Some(target.observation.measurement_complete.to_string()),
            if calibration.observation.missing_stages.is_empty() {
                None
            } else {
                Some(format!(
                    "calibration missing stages: {}",
                    calibration.observation.missing_stages.join(",")
                ))
            },
        );
    }

    fn finish(self) -> CalibrationCompatibility {
        let matched: Vec<String> = self
            .dimensions
            .iter()
            .filter(|dimension| dimension.status == DimensionStatus::Match)
            .map(|dimension| dimension.dimension.clone())
            .collect();
        let mismatched: Vec<String> = self
            .dimensions
            .iter()
            .filter(|dimension| dimension.status == DimensionStatus::Mismatch)
            .map(|dimension| dimension.dimension.clone())
            .collect();
        let missing: Vec<String> = self
            .dimensions
            .iter()
            .filter(|dimension| {
                matches!(
                    dimension.status,
                    DimensionStatus::Missing | DimensionStatus::Stale
                )
            })
            .map(|dimension| dimension.dimension.clone())
            .collect();
        let constraints: Vec<String> = self
            .dimensions
            .iter()
            .filter(|dimension| dimension.status == DimensionStatus::Constraint)
            .map(|dimension| dimension.dimension.clone())
            .collect();
        let runtime_or_model_mismatch = mismatched.iter().any(|name| {
            name.starts_with("model.")
                || name.starts_with("runtime.")
                || name == "backend"
                || name == "device_id"
                || name == "observation.cache_condition"
                || name == "observation.cgroup_limit"
        });
        let machine_mismatch = mismatched.iter().any(|name| name.starts_with("machine."));
        let classification = if runtime_or_model_mismatch {
            CompatibilityClass::Incompatible
        } else if machine_mismatch {
            CompatibilityClass::Incomparable
        } else if !mismatched.is_empty() {
            CompatibilityClass::Incompatible
        } else if !missing.is_empty() {
            CompatibilityClass::InsufficientEvidence
        } else if !constraints.is_empty() {
            CompatibilityClass::CompatibleWithConstraints
        } else {
            CompatibilityClass::Exact
        };
        let evidence_use = match classification {
            CompatibilityClass::Exact => EvidenceUse::Apply,
            CompatibilityClass::CompatibleWithConstraints
            | CompatibilityClass::InsufficientEvidence => EvidenceUse::DisplayOnly,
            CompatibilityClass::Incompatible | CompatibilityClass::Incomparable => {
                EvidenceUse::Reject
            }
        };
        let reasons = if mismatched.is_empty() && missing.is_empty() && constraints.is_empty() {
            vec!["all tracked dimensions match".to_string()]
        } else {
            let mut reasons = Vec::new();
            if !mismatched.is_empty() {
                reasons.push(format!("mismatched dimensions: {}", mismatched.join(", ")));
            }
            if !missing.is_empty() {
                reasons.push(format!(
                    "insufficient or stale dimensions: {}",
                    missing.join(", ")
                ));
            }
            if !constraints.is_empty() {
                reasons.push(format!(
                    "non-blocking observation constraints: {}",
                    constraints.join(", ")
                ));
            }
            reasons
        };
        CalibrationCompatibility {
            schema_version: CALIBRATION_COMPATIBILITY_SCHEMA_VERSION,
            classification,
            evidence_use,
            dimensions: self.dimensions,
            matched,
            mismatched,
            missing,
            limitations: vec![
                "Only exact compatibility applies calibration evidence.".to_string(),
                "Constraint-only or incomplete evidence is display-only.".to_string(),
                "Calibration does not authorize eviction or release.".to_string(),
            ],
            reasons,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine_profile::{collect_machine_profile, MachineProfileConfig, Observed};
    use crate::model_requirements::RequirementBytes;
    use crate::runtime_measurement::LifecycleStage;

    fn requirements() -> ModelRequirements {
        ModelRequirements {
            schema_version: crate::model_requirements::MODEL_REQUIREMENTS_SCHEMA_VERSION,
            model_fingerprint: Some("fnv1a64:test:size=100".to_string()),
            model_name: Some("test".to_string()),
            architecture: Some("llama".to_string()),
            architecture_info: None,
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

    fn runtime() -> RuntimeConfiguration {
        RuntimeConfiguration {
            backend: "cpu".to_string(),
            device_id: Some("cpu".to_string()),
            load_mode: Some("mmap".to_string()),
            weight_repacking: Some("disabled".to_string()),
            requested_context_tokens: Some(128),
            effective_context_tokens: Some(128),
            requested_batch_tokens: Some(1),
            effective_batch_tokens: Some(128),
            effective_ubatch_tokens: Some(128),
            effective_sequence_max: Some(1),
            effective_threads: Some(1),
            effective_threads_batch: Some(1),
            kv_type_k: "f16".to_string(),
            kv_type_v: "f16".to_string(),
            kv_type_source: "adapter-default".to_string(),
            state_serialization_size_bytes: None,
            native_contract_version: Some(1),
        }
    }

    fn conditions() -> ObservationConditions {
        ObservationConditions {
            cache_condition: "warm-uncontrolled".to_string(),
            cache_condition_known: true,
            cgroup_limit_bytes: Observed::unavailable("unlimited", 1),
            pressure_class: "low".to_string(),
            baseline_rss_bytes: Observed::known(
                1024,
                crate::machine_profile::Provenance::OperatingSystem,
                crate::machine_profile::Freshness::Dynamic,
                1,
            ),
            snapshot_count: 9,
            missing_stages: Vec::new(),
            measurement_complete: true,
        }
    }

    fn machine() -> crate::machine_profile::MachineProfile {
        static MACHINE: std::sync::OnceLock<crate::machine_profile::MachineProfile> =
            std::sync::OnceLock::new();
        MACHINE
            .get_or_init(|| collect_machine_profile(&MachineProfileConfig::default()))
            .clone()
    }

    fn target() -> CompatibilityTarget {
        CompatibilityTarget {
            machine: machine(),
            requirements: requirements(),
            runtime: runtime(),
            observation: conditions(),
            now_unix_ms: Some(1000),
        }
    }

    fn calibration() -> RuntimeCalibration {
        use crate::runtime_calibration::build_calibration;
        use crate::runtime_measurement::LifecycleRecorder;
        let machine = machine();
        let mut recorder = LifecycleRecorder::new(None, None);
        for stage in [
            LifecycleStage::Baseline,
            LifecycleStage::ModelLoaded,
            LifecycleStage::ContextCreated,
            LifecycleStage::PromptProcessed,
            LifecycleStage::DecodeSteady,
            LifecycleStage::ContextDestroyed,
            LifecycleStage::ModelUnloaded,
            LifecycleStage::Final,
        ] {
            recorder.record(stage);
        }
        let attribution =
            crate::runtime_calibration::build_attribution(recorder.snapshots(), &requirements());
        let mut calibration = build_calibration(
            &machine,
            &requirements(),
            runtime(),
            recorder.snapshots().to_vec(),
            attribution,
        );
        calibration.captured_at_unix_ms = Some(500);
        calibration
    }

    #[test]
    fn exact_match_applies_evidence() {
        let calibration = calibration();
        let compatibility = calibration.compatibility(&target());
        assert_eq!(compatibility.classification, CompatibilityClass::Exact);
        assert_eq!(compatibility.evidence_use, EvidenceUse::Apply);
        assert!(compatibility.mismatched.is_empty());
    }

    #[test]
    fn different_model_is_incompatible_and_rejected() {
        let calibration = calibration();
        let mut target = target();
        target.requirements.model_fingerprint = Some("different".to_string());
        let compatibility = calibration.compatibility(&target);
        assert_eq!(
            compatibility.classification,
            CompatibilityClass::Incompatible
        );
        assert_eq!(compatibility.evidence_use, EvidenceUse::Reject);
        assert!(compatibility
            .mismatched
            .iter()
            .any(|name| name == "model.fingerprint"));
    }

    #[test]
    fn different_effective_context_is_incompatible() {
        let calibration = calibration();
        let mut target = target();
        target.runtime.effective_context_tokens = Some(256);
        let compatibility = calibration.compatibility(&target);
        assert_eq!(
            compatibility.classification,
            CompatibilityClass::Incompatible
        );
        assert!(compatibility
            .mismatched
            .iter()
            .any(|name| name == "runtime.effective_context"));
    }

    #[test]
    fn different_pressure_is_constraint_only_and_not_applied() {
        let calibration = calibration();
        let mut target = target();
        target.observation.pressure_class = "high".to_string();
        let compatibility = calibration.compatibility(&target);
        assert_eq!(
            compatibility.classification,
            CompatibilityClass::CompatibleWithConstraints
        );
        assert_eq!(compatibility.evidence_use, EvidenceUse::DisplayOnly);
        let (evidence, _) = calibration.evidence_for(&target);
        assert!(!evidence.valid);
    }

    #[test]
    fn unknown_cache_condition_is_insufficient_evidence() {
        let calibration = calibration();
        let mut target = target();
        target.observation.cache_condition_known = false;
        target.observation.cache_condition = "unknown".to_string();
        let compatibility = calibration.compatibility(&target);
        assert_eq!(
            compatibility.classification,
            CompatibilityClass::InsufficientEvidence
        );
        assert_eq!(compatibility.evidence_use, EvidenceUse::DisplayOnly);
    }

    #[test]
    fn stale_calibration_is_not_applied() {
        let calibration = calibration();
        let mut target = target();
        target.now_unix_ms =
            Some(calibration.captured_at_unix_ms.unwrap_or(0) + MAX_CALIBRATION_AGE_MS + 1);
        let compatibility = calibration.compatibility(&target);
        assert_eq!(
            compatibility.classification,
            CompatibilityClass::InsufficientEvidence
        );
        assert_eq!(compatibility.evidence_use, EvidenceUse::DisplayOnly);
        assert!(compatibility
            .missing
            .iter()
            .any(|name| name == "observation.freshness"));
    }

    #[test]
    fn different_machine_is_incomparable_not_applied() {
        let calibration = calibration();
        let mut target = target();
        target.machine.cpu.model = Observed::known(
            "Different CPU".to_string(),
            crate::machine_profile::Provenance::OperatingSystem,
            crate::machine_profile::Freshness::Stable,
            1,
        );
        let compatibility = calibration.compatibility(&target);
        assert_eq!(
            compatibility.classification,
            CompatibilityClass::Incomparable
        );
        assert_eq!(compatibility.evidence_use, EvidenceUse::Reject);
    }

    #[test]
    fn missing_fingerprint_is_insufficient_evidence() {
        let calibration = calibration();
        let mut target = target();
        target.requirements.model_fingerprint = None;
        let compatibility = calibration.compatibility(&target);
        assert_eq!(
            compatibility.classification,
            CompatibilityClass::InsufficientEvidence
        );
        assert_eq!(compatibility.evidence_use, EvidenceUse::DisplayOnly);
    }

    #[test]
    fn load_mode_and_repacking_mismatches_are_incompatible() {
        let calibration = calibration();
        let mut first = target();
        first.runtime.load_mode = Some("mmap-no-prefetch".to_string());
        assert_eq!(
            calibration.compatibility(&first).classification,
            CompatibilityClass::Incompatible
        );
        let mut second = target();
        second.runtime.weight_repacking = Some("enabled".to_string());
        assert_eq!(
            calibration.compatibility(&second).classification,
            CompatibilityClass::Incompatible
        );
    }

    #[test]
    fn micro_batch_sequence_and_thread_mismatches_are_incompatible() {
        let calibration = calibration();
        for mutate in [
            |target: &mut CompatibilityTarget| target.runtime.effective_ubatch_tokens = Some(64),
            |target: &mut CompatibilityTarget| target.runtime.effective_sequence_max = Some(2),
            |target: &mut CompatibilityTarget| target.runtime.effective_threads = Some(2),
            |target: &mut CompatibilityTarget| target.runtime.effective_batch_tokens = Some(64),
            |target: &mut CompatibilityTarget| target.runtime.requested_batch_tokens = Some(2),
        ] {
            let mut target = target();
            mutate(&mut target);
            assert_eq!(
                calibration.compatibility(&target).classification,
                CompatibilityClass::Incompatible
            );
        }
    }

    #[test]
    fn incomplete_calibration_is_insufficient_evidence() {
        let mut calibration = calibration();
        calibration.observation.measurement_complete = false;
        calibration.observation.missing_stages = vec!["decode_steady".to_string()];
        let compatibility = calibration.compatibility(&target());
        assert_eq!(
            compatibility.classification,
            CompatibilityClass::InsufficientEvidence
        );
        assert_eq!(compatibility.evidence_use, EvidenceUse::DisplayOnly);
    }

    #[test]
    fn incompatible_evidence_does_not_change_plan_estimate() {
        use crate::execution_candidate::{
            evaluate_plan, evaluate_plan_with_calibration, ExecutionPlanCandidate, MappingStrategy,
            PlanBackend, PlanProfile, EXECUTION_CANDIDATE_SCHEMA_VERSION,
        };
        let calibration = calibration();
        let mut target = target();
        target.requirements.model_fingerprint = Some("different".to_string());
        let (evidence, compatibility) = calibration.evidence_for(&target);
        assert_eq!(
            compatibility.classification,
            CompatibilityClass::Incompatible
        );
        let machine = collect_machine_profile(&MachineProfileConfig::default());
        let plan = ExecutionPlanCandidate {
            schema_version: EXECUTION_CANDIDATE_SCHEMA_VERSION,
            id: "test".to_string(),
            profile: PlanProfile::Custom,
            backend: PlanBackend::Cpu,
            device_id: None,
            cpu_threads: 1,
            context_tokens: 128,
            batch_tokens: 1,
            ram_budget_bytes: Some(1_000_000),
            vram_budget_bytes: None,
            mapping: MappingStrategy::ResidentWeights,
            assumptions: Vec::new(),
            unknown_information: Vec::new(),
        };
        let base = evaluate_plan(&machine, &target.requirements, &plan);
        let calibrated =
            evaluate_plan_with_calibration(&machine, &target.requirements, &plan, Some(&evidence));
        assert_eq!(base.estimated_ram, calibrated.estimated_ram);
        assert_eq!(base.feasibility, calibrated.feasibility);
        assert!(calibrated
            .reasons
            .iter()
            .any(|reason| reason.code == "calibration_incomparable"));
    }

    #[test]
    fn no_calibration_preserves_uncalibrated_path() {
        let machine = collect_machine_profile(&MachineProfileConfig::default());
        let plan = crate::execution_candidate::ExecutionPlanCandidate {
            schema_version: crate::execution_candidate::EXECUTION_CANDIDATE_SCHEMA_VERSION,
            id: "test".to_string(),
            profile: crate::execution_candidate::PlanProfile::Custom,
            backend: crate::execution_candidate::PlanBackend::Cpu,
            device_id: None,
            cpu_threads: 1,
            context_tokens: 128,
            batch_tokens: 1,
            ram_budget_bytes: Some(1_000_000),
            vram_budget_bytes: None,
            mapping: crate::execution_candidate::MappingStrategy::ResidentWeights,
            assumptions: Vec::new(),
            unknown_information: Vec::new(),
        };
        let baseline = crate::execution_candidate::evaluate_plan(&machine, &requirements(), &plan);
        let uncalibrated = crate::execution_candidate::evaluate_plan_with_calibration(
            &machine,
            &requirements(),
            &plan,
            None,
        );
        assert_eq!(baseline.feasibility, uncalibrated.feasibility);
        assert_eq!(baseline.estimated_ram, uncalibrated.estimated_ram);
    }
}
