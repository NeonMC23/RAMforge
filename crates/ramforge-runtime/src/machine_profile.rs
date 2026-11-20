//! Explicit, serializable machine resource profiles.
//!
//! Every field is an [`Observed`] value so unknown, unavailable, unsupported,
//! and failed information cannot be confused with zero. Relatively stable
//! hardware facts and dynamic observations are marked separately.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub mod cpu;
pub mod memory;
pub mod storage;

pub use cpu::{collect_cpu_profile, CpuProfile, CpuRuntimeSettings};
pub use memory::{collect_memory_profile, MemoryProfile};
pub use storage::{
    collect_storage_profile, run_storage_read_benchmark, StorageBenchmarkConfig,
    StorageBenchmarkReport, StorageProfile,
};

/// Current exported machine-profile schema.
pub const MACHINE_PROFILE_SCHEMA_VERSION: u32 = 1;

/// Where a value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    OperatingSystem,
    NativeBackend,
    Configured,
    Measured,
    Derived,
    NotAvailable,
}

/// Whether information describes stable hardware or a dynamic observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Freshness {
    Stable,
    Dynamic,
    Unknown,
}

/// Explicit observation state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationState {
    Known,
    Unavailable,
    Unsupported,
    Failed,
}

/// A value that is only meaningful when `state == Known`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Observed<T> {
    pub state: ObservationState,
    pub value: Option<T>,
    pub provenance: Provenance,
    pub freshness: Freshness,
    pub sequence: u64,
    pub detail: Option<String>,
}

impl<T> Observed<T> {
    pub fn known(value: T, provenance: Provenance, freshness: Freshness, sequence: u64) -> Self {
        Self {
            state: ObservationState::Known,
            value: Some(value),
            provenance,
            freshness,
            sequence,
            detail: None,
        }
    }

    pub fn unavailable(reason: impl Into<String>, sequence: u64) -> Self {
        Self {
            state: ObservationState::Unavailable,
            value: None,
            provenance: Provenance::NotAvailable,
            freshness: Freshness::Unknown,
            sequence,
            detail: Some(reason.into()),
        }
    }

    pub fn unsupported(reason: impl Into<String>, sequence: u64) -> Self {
        Self {
            state: ObservationState::Unsupported,
            value: None,
            provenance: Provenance::NotAvailable,
            freshness: Freshness::Unknown,
            sequence,
            detail: Some(reason.into()),
        }
    }

    pub fn failed(reason: impl Into<String>, sequence: u64) -> Self {
        Self {
            state: ObservationState::Failed,
            value: None,
            provenance: Provenance::NotAvailable,
            freshness: Freshness::Unknown,
            sequence,
            detail: Some(reason.into()),
        }
    }

    pub fn derived(
        value: T,
        detail: impl Into<String>,
        freshness: Freshness,
        sequence: u64,
    ) -> Self {
        Self {
            state: ObservationState::Known,
            value: Some(value),
            provenance: Provenance::Derived,
            freshness,
            sequence,
            detail: Some(detail.into()),
        }
    }

    pub fn as_ref(&self) -> Option<&T> {
        self.value.as_ref()
    }

    pub fn is_known(&self) -> bool {
        self.state == ObservationState::Known && self.value.is_some()
    }

    /// A known state must contain a value, and every other state must not.
    pub fn validate(&self) -> Result<(), ProfileError> {
        match (self.state, self.value.is_some()) {
            (ObservationState::Known, true) => Ok(()),
            (ObservationState::Known, false) => Err(ProfileError::Invariant(
                "known observation has no value".to_string(),
            )),
            (_, false) => Ok(()),
            (_, true) => Err(ProfileError::Invariant(
                "non-known observation carries a value".to_string(),
            )),
        }
    }
}

/// A backend/device snapshot captured from forgeCORE.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceProfile {
    pub id: String,
    pub backend_id: String,
    pub kind: String,
    pub name: String,
    pub description: String,
    pub available: bool,
    pub selectable: bool,
    pub memory_total_bytes: Observed<u64>,
    pub memory_free_bytes: Observed<u64>,
    pub backend_initialization: String,
    pub model_execution: String,
    pub static_model_offload: String,
    pub memory_reporting: String,
    pub cpu_thread_configuration: String,
    pub unavailable_reason: Option<String>,
}

/// Backend aggregate derived from the device snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendProfile {
    pub id: String,
    pub device_count: usize,
    pub cpu_devices: usize,
    pub gpu_devices: usize,
    pub selectable_devices: usize,
}

/// Complete machine profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineProfile {
    pub schema_version: u32,
    pub collection_sequence: u64,
    pub collected_at_unix_ms: Option<u64>,
    pub hostname_redacted: bool,
    pub cpu: CpuProfile,
    pub memory: MemoryProfile,
    pub storage: Vec<StorageProfile>,
    pub devices: Vec<DeviceProfile>,
    pub backends: Vec<BackendProfile>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileError {
    Invariant(String),
    UnsupportedSchema(u32),
    Serialization(String),
}

impl std::fmt::Display for ProfileError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invariant(message) => write!(formatter, "machine profile invariant: {message}"),
            Self::UnsupportedSchema(version) => {
                write!(formatter, "unsupported machine profile schema {version}")
            }
            Self::Serialization(message) => {
                write!(formatter, "machine profile serialization failed: {message}")
            }
        }
    }
}

impl std::error::Error for ProfileError {}

impl MachineProfile {
    pub fn validate(&self) -> Result<(), ProfileError> {
        if self.schema_version != MACHINE_PROFILE_SCHEMA_VERSION {
            return Err(ProfileError::UnsupportedSchema(self.schema_version));
        }
        self.cpu.validate()?;
        self.memory.validate()?;
        for storage in &self.storage {
            storage.validate()?;
        }
        for device in &self.devices {
            device.memory_total_bytes.validate()?;
            device.memory_free_bytes.validate()?;
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<serde_json::Value, ProfileError> {
        serde_json::to_value(self).map_err(|error| ProfileError::Serialization(error.to_string()))
    }

    pub fn from_json(value: serde_json::Value) -> Result<Self, ProfileError> {
        let profile: Self = serde_json::from_value(value)
            .map_err(|error| ProfileError::Serialization(error.to_string()))?;
        profile.validate()?;
        Ok(profile)
    }

    /// Redact local paths before exporting a profile. Device and filesystem
    /// identities remain available; private mount/path strings do not.
    pub fn redacted(&self) -> Self {
        let mut profile = self.clone();
        profile.hostname_redacted = true;
        profile.storage = profile
            .storage
            .iter()
            .map(StorageProfile::redacted)
            .collect();
        profile.memory.cgroup.path = Observed::unavailable(
            "redacted for export",
            profile.memory.cgroup.path.sequence,
        );
        profile
    }
}

/// Configuration for profile collection. All optional work is bounded.
#[derive(Debug, Clone, Default)]
pub struct MachineProfileConfig {
    pub model_path: Option<PathBuf>,
    pub storage_paths: Vec<PathBuf>,
    pub cpu: CpuRuntimeSettings,
    pub ram_budget_bytes: Option<u64>,
    pub include_storage_benchmark: bool,
    pub storage_benchmark: StorageBenchmarkConfig,
}

/// Collect a profile. Missing optional resources never panic; they become
/// explicit unavailable/unsupported observations.
pub fn collect_machine_profile(config: &MachineProfileConfig) -> MachineProfile {
    let sequence = next_profile_sequence();
    let cpu = collect_cpu_profile(&config.cpu, sequence);
    let memory = collect_memory_profile(config.ram_budget_bytes, sequence);
    let devices = collect_devices(sequence);
    let backends = collect_backends(&devices);
    let mut storage = Vec::new();
    let mut paths = config.storage_paths.clone();
    if let Some(model) = &config.model_path {
        if !paths.iter().any(|path| path == model) {
            paths.push(model.clone());
        }
    }
    let mut seen_paths = std::collections::BTreeSet::new();
    for path in paths {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        if !seen_paths.insert(canonical) {
            continue;
        }
        let mut profile = collect_storage_profile(&path, sequence);
        if config.include_storage_benchmark {
            if let Some(parent) = profile.local_path.as_deref().map(PathBuf::from) {
                let directory = parent
                    .parent()
                    .map(PathBuf::from)
                    .unwrap_or_else(std::env::temp_dir);
                if let Ok(benchmark) =
                    run_storage_read_benchmark(&directory, config.storage_benchmark)
                {
                    profile.benchmark = Some(benchmark);
                }
            }
        }
        storage.push(profile);
    }
    let mut limitations = vec![
        "Resource values are point-in-time observations, not guarantees.".to_string(),
        "Device memory free is not proof of exclusive availability.".to_string(),
        "Storage throughput is not page-cache throughput.".to_string(),
        "Unknown, unavailable, unsupported, and failed values are not zero.".to_string(),
    ];
    if storage.is_empty() {
        limitations.push("No storage path was inspected.".to_string());
    }
    MachineProfile {
        schema_version: MACHINE_PROFILE_SCHEMA_VERSION,
        collection_sequence: sequence,
        collected_at_unix_ms: unix_millis(),
        hostname_redacted: false,
        cpu,
        memory,
        storage,
        devices,
        backends,
        limitations,
    }
}

fn collect_devices(sequence: u64) -> Vec<DeviceProfile> {
    crate::execution::DeviceRegistry::discover()
        .devices()
        .iter()
        .map(|device| device_profile(device, sequence))
        .collect()
}

fn device_profile(device: &crate::execution::DeviceInfo, sequence: u64) -> DeviceProfile {
    use crate::execution::CapabilityState;

    let label = |state: CapabilityState| match state {
        CapabilityState::Supported => "supported".to_string(),
        CapabilityState::Unsupported => "unsupported".to_string(),
        CapabilityState::Unknown => "unknown".to_string(),
    };
    let memory_total = device
        .memory_total
        .bytes()
        .map(|bytes| {
            Observed::known(
                bytes,
                Provenance::NativeBackend,
                Freshness::Stable,
                sequence,
            )
        })
        .unwrap_or_else(|| {
            Observed::unavailable("native device did not report capacity", sequence)
        });
    let memory_free = device
        .memory_free
        .bytes()
        .map(|bytes| {
            Observed::known(
                bytes,
                Provenance::NativeBackend,
                Freshness::Dynamic,
                sequence,
            )
        })
        .unwrap_or_else(|| {
            Observed::unavailable("native device did not report free memory", sequence)
        });
    DeviceProfile {
        id: device.id.to_string(),
        backend_id: device.backend_id.to_string(),
        kind: device.kind.to_string(),
        name: device.name.clone(),
        description: device.description.clone(),
        available: device.available,
        selectable: device.selectable,
        memory_total_bytes: memory_total,
        memory_free_bytes: memory_free,
        backend_initialization: label(device.capabilities.backend_initialization),
        model_execution: label(device.capabilities.model_execution),
        static_model_offload: label(device.capabilities.static_model_offload),
        memory_reporting: label(device.capabilities.memory_reporting),
        cpu_thread_configuration: label(device.capabilities.cpu_thread_configuration),
        unavailable_reason: device.selection_unavailable_reason.clone(),
    }
}

fn collect_backends(devices: &[DeviceProfile]) -> Vec<BackendProfile> {
    let mut ids: Vec<&str> = devices
        .iter()
        .map(|device| device.backend_id.as_str())
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids.into_iter()
        .map(|id| {
            let matching: Vec<_> = devices
                .iter()
                .filter(|device| device.backend_id == id)
                .collect();
            BackendProfile {
                id: id.to_string(),
                device_count: matching.len(),
                cpu_devices: matching
                    .iter()
                    .filter(|device| device.kind == "cpu")
                    .count(),
                gpu_devices: matching
                    .iter()
                    .filter(|device| matches!(device.kind.as_str(), "gpu" | "igpu" | "accelerator"))
                    .count(),
                selectable_devices: matching.iter().filter(|device| device.selectable).count(),
            }
        })
        .collect()
}

fn next_profile_sequence() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    SEQUENCE.fetch_add(1, Ordering::Relaxed).saturating_add(1)
}

fn unix_millis() -> Option<u64> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observed_state_enforces_value_invariant() {
        let known = Observed::known(7u64, Provenance::OperatingSystem, Freshness::Stable, 1);
        assert!(known.validate().is_ok());
        let mut broken = known.clone();
        broken.value = None;
        assert!(broken.validate().is_err());
        let mut broken = Observed::<u64>::unavailable("missing", 1);
        broken.value = Some(0);
        assert!(broken.validate().is_err());
    }

    #[test]
    fn unknown_is_not_zero() {
        let value = Observed::<u64>::unavailable("not reported", 1);
        assert_eq!(value.as_ref(), None);
        assert!(!value.is_known());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn real_linux_profile_is_bounded_and_explicit() {
        let config = MachineProfileConfig {
            storage_paths: vec![std::env::temp_dir()],
            cpu: CpuRuntimeSettings {
                selected_threads: Some(4),
                user_thread_limit: Some(8),
            },
            ram_budget_bytes: Some(256 * 1024 * 1024),
            ..MachineProfileConfig::default()
        };
        let profile = collect_machine_profile(&config);
        profile.validate().expect("real profile");
        assert!(profile.cpu.architecture.is_known());
        assert!(profile.cpu.available_threads.is_known());
        assert_eq!(profile.cpu.selected_threads.as_ref(), Some(&4));
        assert_eq!(profile.cpu.user_thread_limit.as_ref(), Some(&8));
        assert!(profile.memory.physical_total_bytes.is_known());
        assert_eq!(
            profile.memory.configured_budget_bytes.as_ref(),
            Some(&(256 * 1024 * 1024))
        );
        assert!(!profile.storage.is_empty());
        assert!(profile.storage[0].filesystem.is_known());
        assert!(profile
            .storage
            .iter()
            .all(|storage| storage.capacity_bytes.is_known()));
        assert!(profile
            .devices
            .iter()
            .all(|device| device.memory_total_bytes.validate().is_ok()));
    }

    #[test]
    fn profile_round_trips_and_redacts_paths() {
        let config = MachineProfileConfig {
            storage_paths: vec![std::env::temp_dir()],
            ..MachineProfileConfig::default()
        };
        let profile = collect_machine_profile(&config);
        profile.validate().expect("profile");
        let json = profile.to_json().expect("json");
        let decoded = MachineProfile::from_json(json).expect("decode");
        assert_eq!(profile, decoded);
        let redacted = profile.redacted();
        assert!(redacted.hostname_redacted);
        assert!(redacted
            .storage
            .iter()
            .all(|storage| storage.local_path.is_none()));
        redacted.validate().expect("redacted profile");
    }
}
