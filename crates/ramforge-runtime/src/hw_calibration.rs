//! Bounded hardware/model calibration for transfer and scheduling policy.
//!
//! Calibration never loads a model or allocates more than a few hundred MiB of
//! synthetic device/host memory. Records are explicit files and are rejected
//! when the hardware or model identity does not match.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::staging::{Engine, MemoryType, StagingError};

pub const HW_CALIBRATION_SCHEMA_VERSION: u32 = 1;
pub const HW_CALIBRATION_TOOL_VERSION: u32 = 1;
const MIB: u64 = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationIdentity {
    pub device_name: String,
    pub device_index: u32,
    pub driver_version: i32,
    pub native_backend: String,
    pub model_path: String,
    pub model_size_bytes: u64,
    pub model_modified_unix_ms: Option<u64>,
    pub calibration_tool_version: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransferSample {
    pub memory_type: String,
    pub payload_bytes: u64,
    pub duration_ns: u64,
    pub submit_ns: u64,
    pub bandwidth_gbps: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkSample {
    pub payload_bytes: u64,
    pub chunk_bytes: u64,
    pub batches: u64,
    pub duration_ns: u64,
    pub bandwidth_gbps: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationSample {
    pub memory_type: String,
    pub bytes: u64,
    pub create_ns: u64,
    pub release_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationRecord {
    pub schema_version: u32,
    pub created_unix_ms: u64,
    pub identity: CalibrationIdentity,
    pub pinned: Vec<TransferSample>,
    pub pageable: Vec<TransferSample>,
    pub chunks: Vec<ChunkSample>,
    pub allocations: Vec<AllocationSample>,
    pub event_overhead_ns: u64,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationStatus {
    Missing,
    IdentityMismatch,
    Stale,
    Valid,
}

impl CalibrationRecord {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|error| error.to_string())?;
        let record: Self = serde_json::from_str(&text).map_err(|error| error.to_string())?;
        if record.schema_version != HW_CALIBRATION_SCHEMA_VERSION {
            return Err(format!(
                "calibration schema {} is not supported (expected {})",
                record.schema_version, HW_CALIBRATION_SCHEMA_VERSION
            ));
        }
        Ok(record)
    }

    pub fn save(&self, path: &Path, force: bool) -> Result<(), String> {
        if path.exists() && !force {
            return Err(format!(
                "calibration file {} already exists; pass --force to replace it",
                path.display()
            ));
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|error| error.to_string())?;
            }
        }
        let text = serde_json::to_string_pretty(self).map_err(|error| error.to_string())?;
        fs::write(path, text).map_err(|error| error.to_string())
    }

    pub fn status_for(&self, identity: &CalibrationIdentity) -> CalibrationStatus {
        if self.identity.device_name != identity.device_name
            || self.identity.device_index != identity.device_index
            || self.identity.driver_version != identity.driver_version
            || self.identity.native_backend != identity.native_backend
            || self.identity.model_path != identity.model_path
            || self.identity.model_size_bytes != identity.model_size_bytes
            || self.identity.model_modified_unix_ms != identity.model_modified_unix_ms
            || self.identity.calibration_tool_version != identity.calibration_tool_version
        {
            return CalibrationStatus::IdentityMismatch;
        }
        CalibrationStatus::Valid
    }

    /// Effective H2D bandwidth for a payload, preferring the measured sample at
    /// or above the requested size and otherwise the closest smaller sample.
    pub fn pinned_bandwidth_for(&self, payload_bytes: u64) -> Option<f64> {
        bandwidth_for(&self.pinned, payload_bytes)
    }

    pub fn pageable_bandwidth_for(&self, payload_bytes: u64) -> Option<f64> {
        bandwidth_for(&self.pageable, payload_bytes)
    }

    pub fn to_json(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(|error| error.to_string())
    }
}

fn bandwidth_for(samples: &[TransferSample], payload_bytes: u64) -> Option<f64> {
    let mut best: Option<&TransferSample> = None;
    for sample in samples {
        if sample.payload_bytes >= payload_bytes
            && best
                .map(|current| sample.payload_bytes < current.payload_bytes)
                .unwrap_or(true)
        {
            best = Some(sample);
        }
    }
    if best.is_none() {
        best = samples.iter().max_by_key(|sample| sample.payload_bytes);
    }
    best.map(|sample| sample.bandwidth_gbps)
}

pub fn identity_for_model(
    engine: &Engine,
    device_index: u32,
    model_path: &Path,
) -> Result<CalibrationIdentity, String> {
    let info = engine.info().map_err(|error| error.to_string())?;
    let metadata = fs::metadata(model_path).map_err(|error| error.to_string())?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis() as u64);
    Ok(CalibrationIdentity {
        device_name: info.device_name(),
        device_index,
        driver_version: info.driver_version,
        native_backend: "cuda".to_string(),
        model_path: fs::canonicalize(model_path)
            .unwrap_or_else(|_| model_path.to_path_buf())
            .to_string_lossy()
            .into_owned(),
        model_size_bytes: metadata.len(),
        model_modified_unix_ms: modified,
        calibration_tool_version: HW_CALIBRATION_TOOL_VERSION,
    })
}

/// Run a bounded transfer/calibration sweep. Maximum synthetic allocation is
/// one 128 MiB device buffer plus one 128 MiB host buffer.
pub fn run_calibration(
    model_path: &Path,
    device_index: u32,
    out_path: &Path,
    force: bool,
) -> Result<CalibrationRecord, String> {
    let engine = Engine::new(device_index).map_err(|error| match error {
        StagingError::Unsupported(reason) => format!("CUDA staging unavailable: {reason}"),
        other => other.to_string(),
    })?;
    let identity = identity_for_model(&engine, device_index, model_path)?;

    let sizes = [
        64 * 1024,
        256 * 1024,
        MIB,
        4 * MIB,
        16 * MIB,
        64 * MIB,
        128 * MIB,
    ];
    let mut pinned = Vec::new();
    let mut pageable = Vec::new();
    for size in sizes {
        let iterations = if size >= 64 * MIB { 2 } else { 3 };
        pinned.extend(measure_transfers(
            &engine,
            MemoryType::PinnedHost,
            size,
            iterations,
        )?);
        pageable.extend(measure_transfers(
            &engine,
            MemoryType::PageableHost,
            size,
            iterations,
        )?);
    }

    let chunks = measure_chunks(&engine, 64 * MIB, &[4 * MIB, 16 * MIB, 64 * MIB])?;
    let allocations = measure_allocations(&engine)?;
    let event_overhead_ns = pinned
        .iter()
        .filter(|sample| sample.payload_bytes == 64 * 1024)
        .map(|sample| sample.duration_ns)
        .min()
        .unwrap_or(0);

    let record = CalibrationRecord {
        schema_version: HW_CALIBRATION_SCHEMA_VERSION,
        created_unix_ms: now_unix_ms(),
        identity,
        pinned,
        pageable,
        chunks,
        allocations,
        event_overhead_ns,
        notes: vec![
            "synthetic transfer calibration; no model or inference allocation is performed"
                .to_string(),
            "bandwidth uses CUDA-event duration, not host submission time".to_string(),
            "chunk samples wait after each batch and therefore measure sequential chunk cost"
                .to_string(),
            "records are rejected when device, driver, model identity or tool version differs"
                .to_string(),
        ],
    };
    record.save(out_path, force)?;
    Ok(record)
}

fn measure_transfers(
    engine: &Engine,
    memory_type: MemoryType,
    bytes: u64,
    iterations: u32,
) -> Result<Vec<TransferSample>, String> {
    let mut host = engine
        .allocate(memory_type, bytes)
        .map_err(|error| error.to_string())?;
    let mut device = engine
        .allocate(MemoryType::Device, bytes)
        .map_err(|error| error.to_string())?;
    let zeros = vec![0u8; usize::try_from(bytes).map_err(|error| error.to_string())?];
    host.write_host(0, &zeros)
        .map_err(|error| error.to_string())?;
    let device_ptr = device.info().map_err(|error| error.to_string())?.device_ptr;
    let mut samples = Vec::new();
    for _ in 0..iterations {
        engine
            .submit_batch_to_device_ptrs(&host, &[0], &[device_ptr], &[bytes])
            .map_err(|error| error.to_string())?;
        let report = engine.wait().map_err(|error| error.to_string())?;
        samples.push(TransferSample {
            memory_type: match memory_type {
                MemoryType::PinnedHost => "pinned_host".to_string(),
                MemoryType::PageableHost => "pageable_host".to_string(),
                MemoryType::Device => "device".to_string(),
            },
            payload_bytes: bytes,
            duration_ns: report.duration_ns,
            submit_ns: report.submit_ns,
            bandwidth_gbps: bandwidth(bytes, report.duration_ns),
        });
    }
    let _ = host.release();
    let _ = device.release();
    Ok(samples)
}

fn measure_chunks(
    engine: &Engine,
    payload_bytes: u64,
    chunk_sizes: &[u64],
) -> Result<Vec<ChunkSample>, String> {
    let mut host = engine
        .allocate(MemoryType::PinnedHost, payload_bytes)
        .map_err(|error| error.to_string())?;
    let mut device = engine
        .allocate(MemoryType::Device, payload_bytes)
        .map_err(|error| error.to_string())?;
    let zeros = vec![0u8; usize::try_from(payload_bytes).map_err(|error| error.to_string())?];
    host.write_host(0, &zeros)
        .map_err(|error| error.to_string())?;
    let base = device.info().map_err(|error| error.to_string())?.device_ptr;
    let mut samples = Vec::new();
    for &chunk in chunk_sizes {
        let started = Instant::now();
        let mut offset = 0;
        let mut batches = 0;
        while offset < payload_bytes {
            let bytes = chunk.min(payload_bytes - offset);
            engine
                .submit_batch_to_device_ptrs(&host, &[offset], &[base + offset], &[bytes])
                .map_err(|error| error.to_string())?;
            engine.wait().map_err(|error| error.to_string())?;
            offset += bytes;
            batches += 1;
        }
        let elapsed = started.elapsed().as_nanos() as u64;
        samples.push(ChunkSample {
            payload_bytes,
            chunk_bytes: chunk,
            batches,
            duration_ns: elapsed,
            bandwidth_gbps: bandwidth(payload_bytes, elapsed),
        });
    }
    let _ = host.release();
    let _ = device.release();
    Ok(samples)
}

fn measure_allocations(engine: &Engine) -> Result<Vec<AllocationSample>, String> {
    let mut samples = Vec::new();
    for memory_type in [MemoryType::PinnedHost, MemoryType::Device] {
        for _ in 0..5 {
            let start = Instant::now();
            let mut buffer = engine
                .allocate(memory_type, 16 * MIB)
                .map_err(|error| error.to_string())?;
            let create_ns = start.elapsed().as_nanos() as u64;
            let release_start = Instant::now();
            buffer.release().map_err(|error| error.to_string())?;
            samples.push(AllocationSample {
                memory_type: match memory_type {
                    MemoryType::PinnedHost => "pinned_host".to_string(),
                    MemoryType::PageableHost => "pageable_host".to_string(),
                    MemoryType::Device => "device".to_string(),
                },
                bytes: 16 * MIB,
                create_ns,
                release_ns: release_start.elapsed().as_nanos() as u64,
            });
        }
    }
    Ok(samples)
}

fn bandwidth(bytes: u64, duration_ns: u64) -> f64 {
    if bytes == 0 || duration_ns == 0 {
        0.0
    } else {
        bytes as f64 / (duration_ns as f64 / 1e9) / 1e9
    }
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

pub fn default_calibration_path(model_path: &Path) -> PathBuf {
    let mut path = model_path.to_path_buf();
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());
    path.set_file_name(format!("{name}.ramforge-calibration.json"));
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> CalibrationIdentity {
        CalibrationIdentity {
            device_name: "GPU".to_string(),
            device_index: 0,
            driver_version: 1,
            native_backend: "cuda".to_string(),
            model_path: "/m/model.gguf".to_string(),
            model_size_bytes: 10,
            model_modified_unix_ms: Some(5),
            calibration_tool_version: HW_CALIBRATION_TOOL_VERSION,
        }
    }

    fn record() -> CalibrationRecord {
        CalibrationRecord {
            schema_version: HW_CALIBRATION_SCHEMA_VERSION,
            created_unix_ms: 1,
            identity: identity(),
            pinned: vec![TransferSample {
                memory_type: "pinned_host".to_string(),
                payload_bytes: 4 * MIB,
                duration_ns: 1_000_000,
                submit_ns: 10_000,
                bandwidth_gbps: 4.0,
            }],
            pageable: vec![],
            chunks: vec![],
            allocations: vec![],
            event_overhead_ns: 1000,
            notes: vec![],
        }
    }

    #[test]
    fn identity_mismatch_is_rejected() {
        let record = record();
        assert_eq!(record.status_for(&identity()), CalibrationStatus::Valid);
        let mut other = identity();
        other.model_size_bytes += 1;
        assert_eq!(
            record.status_for(&other),
            CalibrationStatus::IdentityMismatch
        );
    }

    #[test]
    fn bandwidth_uses_measured_size() {
        let record = record();
        assert_eq!(record.pinned_bandwidth_for(2 * MIB), Some(4.0));
        assert_eq!(record.pinned_bandwidth_for(8 * MIB), Some(4.0));
    }

    #[test]
    fn serialization_roundtrip_preserves_identity() {
        let record = record();
        let text = serde_json::to_string(&record).unwrap();
        let decoded: CalibrationRecord = serde_json::from_str(&text).unwrap();
        assert_eq!(decoded.identity, record.identity);
        assert_eq!(decoded.pinned[0].payload_bytes, 4 * MIB);
    }
}
