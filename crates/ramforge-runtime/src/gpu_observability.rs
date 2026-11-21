//! Versioned GPU memory and allocation observability.
//!
//! Values are point-in-time snapshots from the native backend, derived from
//! those snapshots, or explicitly unsupported. Categories overlap and are never
//! summed into a fabricated process total.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::execution::DeviceRegistry;
use crate::forgecore::{ForgeCoreModel, ForgeCoreSession};

pub const GPU_OBSERVABILITY_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationScope {
    Device,
    Process,
    Model,
    Context,
    Buffer,
    Session,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryObservationState {
    Measured,
    Estimated,
    Unsupported,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationSource {
    NativeBackend,
    OperatingSystem,
    Derived,
    Configuration,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryObservation {
    pub category: String,
    pub scope: ObservationScope,
    pub state: MemoryObservationState,
    pub bytes: Option<u64>,
    pub device_id: Option<String>,
    pub source: ObservationSource,
    pub sampled_at_unix_ms: Option<u64>,
    pub collection_sequence: u64,
    /// True when this category can overlap another listed category.
    pub overlapping: bool,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceObservation {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub total_bytes: Option<u64>,
    pub free_bytes: Option<u64>,
    pub used_bytes: Option<u64>,
    pub state: MemoryObservationState,
    pub sampled_at_unix_ms: Option<u64>,
    pub collection_sequence: u64,
}

/// One native buffer-type entry. Model/context/compute are disjoint within the
/// entry. Entries are grouped by backend buffer type and may share a device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferObservation {
    pub buffer_type: String,
    pub device_name: String,
    pub model_bytes: u64,
    pub context_bytes: u64,
    pub compute_bytes: u64,
    pub total_bytes: u64,
    pub overlapping: bool,
}

/// Counter support is reported explicitly. `count`/`bytes` are `None` when the
/// native backend does not expose event counters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CounterObservation {
    pub state: MemoryObservationState,
    pub count: Option<u64>,
    pub bytes: Option<u64>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferObservation {
    pub state: MemoryObservationState,
    pub host_to_device_count: Option<u64>,
    pub host_to_device_bytes: Option<u64>,
    pub device_to_host_count: Option<u64>,
    pub device_to_host_bytes: Option<u64>,
    pub duration_ns: Option<u64>,
    pub reason: String,
}

type LayerStorageKey = (Option<u32>, String, String, bool);
type LayerStorageTotals = (u64, u64, u64);

/// Read-only tensor storage aggregated by layer, device, and buffer type.
/// This is the granularity needed to plan placement; it does not authorize
/// staging or independent buffer release.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerStorageObservation {
    pub layer: Option<u32>,
    pub device_name: String,
    pub buffer_type: String,
    pub tensor_count: u64,
    pub bytes: u64,
    pub file_backed_bytes: u64,
    pub host: bool,
    pub overlapping: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuObservability {
    pub schema_version: u32,
    pub sampled_at_unix_ms: Option<u64>,
    pub collection_sequence: u64,
    pub devices: Vec<DeviceObservation>,
    pub buffers: Vec<BufferObservation>,
    pub layers: Vec<LayerStorageObservation>,
    pub observations: Vec<MemoryObservation>,
    pub allocation_counters: CounterObservation,
    pub deallocation_counters: CounterObservation,
    pub transfers: TransferObservation,
    pub notes: Vec<String>,
}

impl GpuObservability {
    pub fn to_json(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(|error| error.to_string())
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != GPU_OBSERVABILITY_SCHEMA_VERSION {
            return Err(format!(
                "unsupported GPU observability schema {}",
                self.schema_version
            ));
        }
        if self.devices.is_empty() {
            return Err("GPU observability has no device observation".to_string());
        }
        for observation in &self.observations {
            if observation.state == MemoryObservationState::Measured && observation.bytes.is_none()
            {
                return Err(format!(
                    "measured observation '{}' has no byte value",
                    observation.category
                ));
            }
            if observation.state != MemoryObservationState::Measured && observation.bytes.is_some()
            {
                return Err(format!(
                    "non-measured observation '{}' carries a byte value",
                    observation.category
                ));
            }
        }
        Ok(())
    }
}

pub fn collect_observability(
    registry: &DeviceRegistry,
    session: Option<&ForgeCoreSession>,
    model: Option<&ForgeCoreModel>,
) -> Result<GpuObservability, String> {
    let sequence = next_sequence();
    let sampled_at = now_ms();
    let mut devices = Vec::new();
    let mut observations = Vec::new();
    for device in registry.devices() {
        let total = device.memory_total.bytes();
        let free = device.memory_free.bytes();
        let used = match (total, free) {
            (Some(total), Some(free)) => Some(total.saturating_sub(free)),
            _ => None,
        };
        let state = if total.is_some() || free.is_some() {
            MemoryObservationState::Measured
        } else {
            MemoryObservationState::Unsupported
        };
        devices.push(DeviceObservation {
            id: device.id.to_string(),
            name: device.name.clone(),
            kind: device.kind.to_string(),
            total_bytes: total,
            free_bytes: free,
            used_bytes: used,
            state,
            sampled_at_unix_ms: sampled_at,
            collection_sequence: sequence,
        });
        push_observation(
            &mut observations,
            "device_total",
            ObservationScope::Device,
            total,
            Some(device.id.to_string()),
            ObservationSource::NativeBackend,
            sampled_at,
            sequence,
            true,
            None,
        );
        push_observation(
            &mut observations,
            "device_free",
            ObservationScope::Device,
            free,
            Some(device.id.to_string()),
            ObservationSource::NativeBackend,
            sampled_at,
            sequence,
            true,
            Some("point-in-time free memory; not a reservation".to_string()),
        );
        push_observation(
            &mut observations,
            "device_used",
            ObservationScope::Device,
            used,
            Some(device.id.to_string()),
            ObservationSource::Derived,
            sampled_at,
            sequence,
            true,
            Some("derived as total minus free; includes other processes".to_string()),
        );
    }

    let mut buffers = Vec::new();
    let breakdown = if let Some(session) = session {
        Some(
            session
                .memory_breakdown()
                .map_err(|error| error.to_string())?,
        )
    } else {
        model
            .map(|model| model.memory_breakdown().map_err(|error| error.to_string()))
            .transpose()?
    };
    if let Some(breakdown) = breakdown {
        for entry in breakdown {
            let total = entry.total_bytes();
            let device_id = devices
                .iter()
                .find(|device| device.name == entry.device_name)
                .map(|device| device.id.clone());
            buffers.push(BufferObservation {
                buffer_type: entry.buffer_type.clone(),
                device_name: entry.device_name.clone(),
                model_bytes: entry.model_bytes,
                context_bytes: entry.context_bytes,
                compute_bytes: entry.compute_bytes,
                total_bytes: total,
                overlapping: true,
            });
            if entry.model_bytes > 0 {
                push_observation(
                    &mut observations,
                    "model_weights",
                    ObservationScope::Buffer,
                    Some(entry.model_bytes),
                    device_id.clone(),
                    ObservationSource::NativeBackend,
                    sampled_at,
                    sequence,
                    true,
                    Some(format!("buffer={}", entry.buffer_type)),
                );
            }
            if entry.context_bytes > 0 {
                push_observation(
                    &mut observations,
                    "kv_cache",
                    ObservationScope::Context,
                    Some(entry.context_bytes),
                    device_id.clone(),
                    ObservationSource::NativeBackend,
                    sampled_at,
                    sequence,
                    true,
                    Some(format!("buffer={}", entry.buffer_type)),
                );
            }
            if entry.compute_bytes > 0 {
                push_observation(
                    &mut observations,
                    "compute_graph",
                    ObservationScope::Buffer,
                    Some(entry.compute_bytes),
                    device_id.clone(),
                    ObservationSource::NativeBackend,
                    sampled_at,
                    sequence,
                    true,
                    Some(format!("buffer={}", entry.buffer_type)),
                );
            }
            let category = if entry.buffer_type.contains("CPU_Mapped") {
                Some("cpu_mapped")
            } else if entry.buffer_type.contains("CUDA_Host") {
                Some("host_pinned")
            } else if entry.buffer_type.contains("CUDA") {
                Some("backend_buffer")
            } else if entry.buffer_type.contains("CPU") {
                Some("cpu_buffer")
            } else {
                None
            };
            if let Some(category) = category {
                push_observation(
                    &mut observations,
                    category,
                    ObservationScope::Buffer,
                    Some(total),
                    device_id,
                    ObservationSource::NativeBackend,
                    sampled_at,
                    sequence,
                    true,
                    Some(format!("buffer={}", entry.buffer_type)),
                );
            }
        }
        push_observation(
            &mut observations,
            "session_buffer_bytes",
            ObservationScope::Session,
            Some(buffers.iter().map(|buffer| buffer.total_bytes).sum()),
            None,
            ObservationSource::Derived,
            sampled_at,
            sequence,
            true,
            Some("sum of native buffer-type entries; grouped by buffer type".to_string()),
        );
    }

    let mut layers: Vec<LayerStorageObservation> = Vec::new();
    if let Some(model) = session.map(|session| session.model()).or(model) {
        let storage = model.tensor_storage().map_err(|error| error.to_string())?;
        let mut aggregated: BTreeMap<LayerStorageKey, LayerStorageTotals> = BTreeMap::new();
        for tensor in storage {
            let key = (
                tensor.layer,
                tensor.device_name.clone(),
                tensor.buffer_type.clone(),
                tensor.is_host,
            );
            let entry = aggregated.entry(key).or_insert((0, 0, 0));
            entry.0 += 1;
            entry.1 = entry.1.saturating_add(tensor.tensor_bytes);
            if tensor.file_backed {
                entry.2 = entry.2.saturating_add(tensor.tensor_bytes);
            }
        }
        for ((layer, device_name, buffer_type, host), (count, bytes, file_backed)) in aggregated {
            layers.push(LayerStorageObservation {
                layer,
                device_name,
                buffer_type,
                tensor_count: count,
                bytes,
                file_backed_bytes: file_backed,
                host,
                overlapping: true,
            });
        }
        push_observation(
            &mut observations,
            "session_tensor_bytes",
            ObservationScope::Session,
            Some(layers.iter().map(|layer| layer.bytes).sum()),
            None,
            ObservationSource::Derived,
            sampled_at,
            sequence,
            true,
            Some(
                "sum of read-only tensor storage entries; grouped by layer/device/buffer"
                    .to_string(),
            ),
        );
    }

    // Explicit unsupported or unknown measurements.
    for (category, scope, detail) in [
        (
            "process_attributed_device_memory",
            ObservationScope::Process,
            "NVML process-level attribution is not integrated",
        ),
        (
            "driver_overhead",
            ObservationScope::Device,
            "driver and other-process usage is not separable from device used",
        ),
        (
            "peak_vram",
            ObservationScope::Session,
            "no native peak-VRAM sampler is exposed",
        ),
        (
            "synchronization_duration",
            ObservationScope::Session,
            "no per-synchronization native counter is exposed",
        ),
    ] {
        push_observation(
            &mut observations,
            category,
            scope,
            None,
            None,
            ObservationSource::NativeBackend,
            sampled_at,
            sequence,
            true,
            Some(detail.to_string()),
        );
    }

    Ok(GpuObservability {
        schema_version: GPU_OBSERVABILITY_SCHEMA_VERSION,
        sampled_at_unix_ms: sampled_at,
        collection_sequence: sequence,
        devices,
        buffers,
        layers,
        observations,
        allocation_counters: CounterObservation {
            state: MemoryObservationState::Unsupported,
            count: None,
            bytes: None,
            reason:
                "native backend exposes no allocation event counter; buffer totals are snapshots"
                    .to_string(),
        },
        deallocation_counters: CounterObservation {
            state: MemoryObservationState::Unsupported,
            count: None,
            bytes: None,
            reason: "native backend exposes no deallocation event counter".to_string(),
        },
        transfers: TransferObservation {
            state: MemoryObservationState::Unsupported,
            host_to_device_count: None,
            host_to_device_bytes: None,
            device_to_host_count: None,
            device_to_host_bytes: None,
            duration_ns: None,
            reason: "no transfer instrumentation is installed in this milestone".to_string(),
        },
        notes: vec![
            "Categories overlap; do not sum them into a process total.".to_string(),
            "Buffer model/context/compute values are disjoint within one native entry.".to_string(),
            "Device used includes driver and other-process usage.".to_string(),
            "Free device memory is a point-in-time observation, not a reservation.".to_string(),
        ],
    })
}

#[allow(clippy::too_many_arguments)]
fn push_observation(
    observations: &mut Vec<MemoryObservation>,
    category: &str,
    scope: ObservationScope,
    bytes: Option<u64>,
    device_id: Option<String>,
    source: ObservationSource,
    sampled_at: Option<u64>,
    sequence: u64,
    overlapping: bool,
    detail: Option<String>,
) {
    let state = match (bytes, category) {
        (Some(_), _) => MemoryObservationState::Measured,
        (None, "process_attributed_device_memory")
        | (None, "driver_overhead")
        | (None, "peak_vram")
        | (None, "synchronization_duration") => MemoryObservationState::Unsupported,
        (None, _) => MemoryObservationState::Unknown,
    };
    observations.push(MemoryObservation {
        category: category.to_string(),
        scope,
        state,
        bytes,
        device_id,
        source,
        sampled_at_unix_ms: sampled_at,
        collection_sequence: sequence,
        overlapping,
        detail,
    });
}

fn next_sequence() -> u64 {
    static SEQUENCE: AtomicU64 = AtomicU64::new(1);
    SEQUENCE.fetch_add(1, Ordering::Relaxed)
}

fn now_ms() -> Option<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_only_collection_is_explicit_and_nonzero_free_is_not_assumed() {
        let observability =
            collect_observability(&DeviceRegistry::cpu_only(), None, None).expect("observability");
        observability.validate().expect("valid schema");
        assert_eq!(observability.schema_version, 2);
        assert!(!observability.devices.is_empty());
        assert_eq!(
            observability.allocation_counters.state,
            MemoryObservationState::Unsupported
        );
        assert_eq!(
            observability.transfers.state,
            MemoryObservationState::Unsupported
        );
        assert!(observability
            .observations
            .iter()
            .any(
                |observation| observation.category == "process_attributed_device_memory"
                    && observation.state == MemoryObservationState::Unsupported
            ));
        assert!(observability.buffers.is_empty());
        assert!(observability.layers.is_empty());
        let json = observability.to_json().expect("json");
        let decoded: GpuObservability = serde_json::from_value(json).expect("round trip");
        assert_eq!(decoded, observability);
    }
}
