//! Stable execution-environment identity and performance-record invalidation.
//!
//! The hardware fingerprint covers physical facts. This module adds the
//! software facts that materially change measured execution: RAMforge version,
//! native residency contract, registered backends, GPU driver, and CPU feature
//! identity. Volatile values (free RAM/VRAM, load average) are excluded.
//!
//! A changed fingerprint does not prove that old measurements are wrong; it
//! only means they must not be reused automatically.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::machine_profile::MachineProfile;
use crate::profiling::{fnv_hex, PerformanceRecord, PROFILING_SCHEMA_VERSION};

pub const EXECUTION_ENVIRONMENT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionEnvironment {
    pub schema_version: u32,
    pub os: String,
    pub kernel: Option<String>,
    pub ramforge_version: String,
    pub native_contract_version: u32,
    /// Optional revision supplied by the build/launcher environment. The
    /// pinned native checkout is not read or modified at runtime.
    pub native_revision: Option<String>,
    /// Registered backend names plus the native GPU-offload capability bit.
    pub backend_build: String,
    /// Parsed from `/proc/driver/nvidia/version` when available.
    pub gpu_driver_version: Option<String>,
    /// Stable hash of the CPU feature list.
    pub cpu_feature_fingerprint: Option<String>,
    pub fingerprint: String,
}

impl ExecutionEnvironment {
    pub fn collect(machine: &MachineProfile) -> Self {
        let ramforge_version = env!("CARGO_PKG_VERSION").to_string();
        let native_contract_version = crate::forgecore::ForgeCoreModel::native_contract_version();
        let native_revision = std::env::var("RAMFORGE_NATIVE_REVISION").ok();
        let mut backends: Vec<String> = machine
            .backend_inventory
            .registered
            .iter()
            .map(|backend| backend.name.clone())
            .collect();
        backends.sort();
        backends.dedup();
        let backend_build = format!(
            "{}|gpu_offload={}",
            backends.join(","),
            machine.backend_inventory.gpu_offload_available
        );
        let gpu_driver_version = read_nvidia_driver_version();
        let cpu_features = machine.cpu.features.as_ref().cloned().unwrap_or_default();
        let cpu_feature_fingerprint = if cpu_features.is_empty() {
            None
        } else {
            Some(fnv_hex(cpu_features.join(",").as_bytes()))
        };
        let mut identity = String::new();
        for part in [
            machine.os.clone(),
            machine.kernel.clone().unwrap_or_default(),
            ramforge_version.clone(),
            native_contract_version.to_string(),
            native_revision.clone().unwrap_or_default(),
            backend_build.clone(),
            gpu_driver_version.clone().unwrap_or_default(),
            cpu_feature_fingerprint.clone().unwrap_or_default(),
        ] {
            identity.push_str(&part);
            identity.push('\u{1f}');
        }
        Self {
            schema_version: EXECUTION_ENVIRONMENT_SCHEMA_VERSION,
            os: machine.os.clone(),
            kernel: machine.kernel.clone(),
            ramforge_version,
            native_contract_version,
            native_revision,
            backend_build,
            gpu_driver_version,
            cpu_feature_fingerprint,
            fingerprint: fnv_hex(identity.as_bytes()),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != EXECUTION_ENVIRONMENT_SCHEMA_VERSION {
            return Err(format!(
                "unsupported execution environment schema {}",
                self.schema_version
            ));
        }
        if self.ramforge_version.trim().is_empty() {
            return Err("execution environment has no RAMforge version".to_string());
        }
        if self.fingerprint.len() != 16 || !self.fingerprint.chars().all(|c| c.is_ascii_hexdigit())
        {
            return Err("execution environment fingerprint is malformed".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvalidationReason {
    ModelChanged,
    HardwareChanged,
    EnvironmentChanged,
    WorkloadChanged,
    SchemaChanged,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Invalidation {
    pub reusable: bool,
    pub reasons: Vec<InvalidationReason>,
    pub details: Vec<String>,
}

impl Invalidation {
    pub fn reusable() -> Self {
        Self {
            reusable: true,
            reasons: Vec::new(),
            details: Vec::new(),
        }
    }
}

/// Compare a persisted record with the current execution identity. A record is
/// reusable only when the model, hardware, and schema all still match. Workload
/// equality is checked separately by exact identity key because a different
/// workload does not falsify the measurement, it only makes it non-exact.
pub fn record_invalidation(
    record: &PerformanceRecord,
    model_fingerprint: Option<&str>,
    hardware_fingerprint: &str,
) -> Invalidation {
    let mut reasons = Vec::new();
    let mut details = Vec::new();

    if record.identity.model_fingerprint.as_deref() != model_fingerprint {
        reasons.push(InvalidationReason::ModelChanged);
        details.push(format!(
            "record model {:?} != current model {:?}",
            record.identity.model_fingerprint, model_fingerprint
        ));
    }
    if record.identity.hardware_fingerprint != hardware_fingerprint {
        reasons.push(InvalidationReason::HardwareChanged);
        details.push(format!(
            "record hardware {} != current hardware {}",
            record.identity.hardware_fingerprint, hardware_fingerprint
        ));
    }
    if record.identity.runtime_schema != PROFILING_SCHEMA_VERSION {
        reasons.push(InvalidationReason::SchemaChanged);
        details.push(format!(
            "record schema {} != current schema {}",
            record.identity.runtime_schema, PROFILING_SCHEMA_VERSION
        ));
    }

    Invalidation {
        reusable: reasons.is_empty(),
        reasons,
        details,
    }
}

/// Explain which environment components differ. Intended for the `--json`
/// plan/calibration outputs, not for automatic mutation.
pub fn environment_changes(
    previous: &ExecutionEnvironment,
    current: &ExecutionEnvironment,
) -> Vec<String> {
    let mut changes = Vec::new();
    if previous.os != current.os {
        changes.push(format!("os {} -> {}", previous.os, current.os));
    }
    if previous.kernel != current.kernel {
        changes.push(format!(
            "kernel {:?} -> {:?}",
            previous.kernel, current.kernel
        ));
    }
    if previous.ramforge_version != current.ramforge_version {
        changes.push(format!(
            "ramforge {} -> {}",
            previous.ramforge_version, current.ramforge_version
        ));
    }
    if previous.native_contract_version != current.native_contract_version {
        changes.push(format!(
            "native contract {} -> {}",
            previous.native_contract_version, current.native_contract_version
        ));
    }
    if previous.native_revision != current.native_revision {
        changes.push(format!(
            "native revision {:?} -> {:?}",
            previous.native_revision, current.native_revision
        ));
    }
    if previous.backend_build != current.backend_build {
        changes.push(format!(
            "backends {} -> {}",
            previous.backend_build, current.backend_build
        ));
    }
    if previous.gpu_driver_version != current.gpu_driver_version {
        changes.push(format!(
            "gpu driver {:?} -> {:?}",
            previous.gpu_driver_version, current.gpu_driver_version
        ));
    }
    if previous.cpu_feature_fingerprint != current.cpu_feature_fingerprint {
        changes.push("cpu feature identity changed".to_string());
    }
    changes
}

fn read_nvidia_driver_version() -> Option<String> {
    // Read-only and bounded. No device or driver state is changed.
    let path = Path::new("/proc/driver/nvidia/version");
    let text = std::fs::read_to_string(path).ok()?;
    parse_nvidia_driver_version(&text)
}

fn parse_nvidia_driver_version(text: &str) -> Option<String> {
    for line in text.lines() {
        if let Some(index) = line.find("Kernel Module") {
            let rest = line[index + "Kernel Module".len()..].trim();
            let version = rest.split_whitespace().next()?;
            if version.chars().next().is_some_and(|c| c.is_ascii_digit())
                && version.chars().any(|c| c == '.')
            {
                return Some(version.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn environment(driver: Option<&str>) -> ExecutionEnvironment {
        let mut identity = String::new();
        for part in [
            "linux".to_string(),
            "kernel".to_string(),
            "0.1.0".to_string(),
            "1".to_string(),
            String::new(),
            "cpu|gpu_offload=false".to_string(),
            driver.unwrap_or_default().to_string(),
            "ffff".to_string(),
        ] {
            identity.push_str(&part);
            identity.push('\u{1f}');
        }
        ExecutionEnvironment {
            schema_version: EXECUTION_ENVIRONMENT_SCHEMA_VERSION,
            os: "linux".to_string(),
            kernel: Some("kernel".to_string()),
            ramforge_version: "0.1.0".to_string(),
            native_contract_version: 1,
            native_revision: None,
            backend_build: "cpu|gpu_offload=false".to_string(),
            gpu_driver_version: driver.map(str::to_string),
            cpu_feature_fingerprint: Some("ffff".to_string()),
            fingerprint: fnv_hex(identity.as_bytes()),
        }
    }

    #[test]
    fn parses_nvidia_driver_version() {
        let text = "NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.95.05  Fri Sep 26 19:00:00 UTC 2025\nGCC version:  gcc version 15.2.1\n";
        assert_eq!(
            parse_nvidia_driver_version(text).as_deref(),
            Some("580.95.05")
        );
    }

    #[test]
    fn missing_driver_is_not_invented() {
        assert_eq!(parse_nvidia_driver_version("nothing here"), None);
    }

    #[test]
    fn environment_change_detects_driver_update() {
        let before = environment(Some("580.95.05"));
        let after = environment(Some("581.01.02"));
        let changes = environment_changes(&before, &after);
        assert_eq!(changes.len(), 1);
        assert!(changes[0].contains("gpu driver"));
    }

    #[test]
    fn environment_validation_rejects_malformed_fingerprint() {
        let mut env = environment(None);
        env.fingerprint = "not-a-hash".to_string();
        assert!(env.validate().is_err());
    }

    #[test]
    fn record_invalidation_flags_model_hardware_and_schema() {
        let record = PerformanceRecord {
            identity: crate::profiling::PerformanceIdentity {
                model_fingerprint: Some("model".to_string()),
                architecture: Some("llama".to_string()),
                quantization: Some("mostly-q4_k_m".to_string()),
                hardware_fingerprint: "hw".to_string(),
                backend: "cpu".to_string(),
                device_id: "cpu".to_string(),
                workload: crate::profiling::WorkloadProfile {
                    context_tokens: 128,
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
                    max_tokens: Some(4),
                    ram_budget_bytes: None,
                },
                runtime_schema: 1,
            },
            captured_at_unix_ms: Some(1),
            generation_tokens_per_second: 1.0,
            decode_tokens_per_second: None,
            prompt_tokens_per_second: None,
            time_to_first_token_ns: None,
            total_generation_ns: 1,
            emitted_tokens: 4,
            target_forwards: None,
            accepted_tokens: None,
            draft_tokens_generated: None,
            acceptance_rate_bps: None,
            storage_read_bytes: None,
            pss_bytes: None,
            condition: "warm".to_string(),
            limitations: Vec::new(),
        };
        let invalidation = record_invalidation(&record, Some("model"), "hw");
        assert!(!invalidation.reusable);
        assert!(invalidation
            .reasons
            .contains(&InvalidationReason::SchemaChanged));

        let invalidation = record_invalidation(&record, Some("other"), "other-hw");
        assert!(invalidation
            .reasons
            .contains(&InvalidationReason::ModelChanged));
        assert!(invalidation
            .reasons
            .contains(&InvalidationReason::HardwareChanged));
    }
}
