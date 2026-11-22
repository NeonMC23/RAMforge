//! Host-side staging: direct file reads into pinned memory, tensor-range
//! coalescing, and a bounded reusable pinned buffer pool.

use std::collections::HashMap;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::thread::JoinHandle;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::forgecore::ForgeCoreModel;
use crate::staging::{Engine, MemoryType, StagingBuffer};
use crate::ServiceError;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostStagingMode {
    #[default]
    Copy,
    Direct,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadBatching {
    Tensor,
    #[default]
    Coalesced,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostStagingReport {
    pub mode: HostStagingMode,
    pub batching: ReadBatching,
    pub tensors: u64,
    pub bytes_read: u64,
    pub read_groups: u64,
    pub coalesced_groups: u64,
    pub read_ns: u64,
    pub prepare_ns: u64,
    pub pinned_capacity: u64,
    pub pinned_reused: bool,
    pub fallback_tensors: u64,
}

impl HostStagingReport {
    pub fn accumulate(&mut self, other: &HostStagingReport) {
        self.tensors += other.tensors;
        self.bytes_read += other.bytes_read;
        self.read_groups += other.read_groups;
        self.coalesced_groups += other.coalesced_groups;
        self.read_ns += other.read_ns;
        self.prepare_ns += other.prepare_ns;
        self.pinned_capacity = self.pinned_capacity.max(other.pinned_capacity);
        self.pinned_reused |= other.pinned_reused;
        self.fallback_tensors += other.fallback_tensors;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorReadSpec {
    name: String,
    file_path: Option<String>,
    file_offset: u64,
    bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReadGroup {
    file_path: Option<String>,
    file_offset: u64,
    bytes: u64,
    members: Vec<usize>,
    host_offset: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostStagingStatus {
    pub mode: HostStagingMode,
    pub batching: ReadBatching,
    pub last: Option<HostStagingReport>,
    pub pool: PooledPinnedStats,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PooledPinnedStats {
    pub limit_bytes: u64,
    pub idle_bytes: u64,
    pub idle_buffers: u64,
    pub allocations: u64,
    pub reuses: u64,
    pub peak_in_use_bytes: u64,
}

struct PooledBuffer {
    buffer: StagingBuffer,
    capacity: u64,
}

/// Bounded pool for pinned host buffers. Idle buffers are retained only up to
/// the configured limit; larger or excess buffers are released.
pub struct PinnedPool {
    idle: Vec<PooledBuffer>,
    limit_bytes: u64,
    idle_bytes: u64,
    allocations: u64,
    reuses: u64,
    in_use_bytes: u64,
    peak_in_use_bytes: u64,
}

impl PinnedPool {
    pub fn new(limit_bytes: u64) -> Self {
        Self {
            idle: Vec::new(),
            limit_bytes,
            idle_bytes: 0,
            allocations: 0,
            reuses: 0,
            in_use_bytes: 0,
            peak_in_use_bytes: 0,
        }
    }

    pub fn limit_bytes(&self) -> u64 {
        self.limit_bytes
    }

    pub fn acquire(
        &mut self,
        engine: &Engine,
        bytes: u64,
    ) -> Result<(StagingBuffer, bool), ServiceError> {
        let mut best: Option<usize> = None;
        for (index, pooled) in self.idle.iter().enumerate() {
            if pooled.capacity >= bytes
                && best
                    .map(|current| pooled.capacity < self.idle[current].capacity)
                    .unwrap_or(true)
            {
                best = Some(index);
            }
        }
        if let Some(index) = best {
            let pooled = self.idle.remove(index);
            self.idle_bytes = self.idle_bytes.saturating_sub(pooled.capacity);
            self.reuses += 1;
            self.in_use_bytes = self.in_use_bytes.saturating_add(pooled.capacity);
            self.peak_in_use_bytes = self.peak_in_use_bytes.max(self.in_use_bytes);
            return Ok((pooled.buffer, true));
        }
        let buffer = engine
            .allocate(MemoryType::PinnedHost, bytes)
            .map_err(|error| ServiceError::Execution(error.to_string()))?;
        self.allocations += 1;
        self.in_use_bytes = self.in_use_bytes.saturating_add(bytes);
        self.peak_in_use_bytes = self.peak_in_use_bytes.max(self.in_use_bytes);
        Ok((buffer, false))
    }

    pub fn release(&mut self, mut buffer: StagingBuffer) {
        let capacity = buffer.info().map(|info| info.capacity).unwrap_or(0);
        self.in_use_bytes = self.in_use_bytes.saturating_sub(capacity);
        if self.limit_bytes > 0 && capacity <= self.limit_bytes.saturating_sub(self.idle_bytes) {
            self.idle_bytes = self.idle_bytes.saturating_add(capacity);
            self.idle.push(PooledBuffer { buffer, capacity });
        } else {
            let _ = buffer.release();
        }
    }

    pub fn clear(&mut self) {
        for pooled in self.idle.drain(..) {
            let mut buffer = pooled.buffer;
            let _ = buffer.release();
        }
        self.idle_bytes = 0;
    }

    pub fn stats(&self) -> PooledPinnedStats {
        PooledPinnedStats {
            limit_bytes: self.limit_bytes,
            idle_bytes: self.idle_bytes,
            idle_buffers: self.idle.len() as u64,
            allocations: self.allocations,
            reuses: self.reuses,
            peak_in_use_bytes: self.peak_in_use_bytes,
        }
    }
}

/// Build read groups and per-tensor host offsets. Direct+coalesced groups only
/// exact adjacent ranges in the same file; gaps are never over-read.
fn build_groups(specs: &[TensorReadSpec], batching: ReadBatching) -> (Vec<ReadGroup>, Vec<u64>) {
    let mut host_offsets = vec![0u64; specs.len()];
    let mut groups = Vec::new();
    let mut host_cursor = 0u64;

    if batching == ReadBatching::Tensor {
        for (index, spec) in specs.iter().enumerate() {
            host_offsets[index] = host_cursor;
            groups.push(ReadGroup {
                file_path: spec.file_path.clone(),
                file_offset: spec.file_offset,
                bytes: spec.bytes,
                members: vec![index],
                host_offset: host_cursor,
            });
            host_cursor += spec.bytes;
        }
        return (groups, host_offsets);
    }

    let mut file_indices: Vec<usize> = specs
        .iter()
        .enumerate()
        .filter(|(_, spec)| spec.file_path.is_some())
        .map(|(index, _)| index)
        .collect();
    file_indices.sort_by_key(|index| {
        (
            specs[*index].file_path.clone().unwrap_or_default(),
            specs[*index].file_offset,
            *index,
        )
    });

    let mut current: Option<ReadGroup> = None;
    for index in file_indices {
        let spec = &specs[index];
        let path = spec.file_path.clone().unwrap_or_default();
        let can_append = current.as_ref().map(|group| {
            group.file_path.as_deref() == Some(path.as_str())
                && group.file_offset + group.bytes == spec.file_offset
        });
        if can_append == Some(true) {
            let group = current.as_mut().expect("group exists");
            group.bytes += spec.bytes;
            group.members.push(index);
        } else {
            if let Some(group) = current.take() {
                groups.push(group);
            }
            current = Some(ReadGroup {
                file_path: Some(path),
                file_offset: spec.file_offset,
                bytes: spec.bytes,
                members: vec![index],
                host_offset: 0,
            });
        }
    }
    if let Some(group) = current {
        groups.push(group);
    }

    for group in &mut groups {
        group.host_offset = host_cursor;
        for member in &group.members {
            let spec = &specs[*member];
            host_offsets[*member] = host_cursor + (spec.file_offset - group.file_offset);
        }
        host_cursor += group.bytes;
    }

    // Non-file-backed tensors are staged through the model read API.
    for (index, spec) in specs.iter().enumerate() {
        if spec.file_path.is_some() {
            continue;
        }
        host_offsets[index] = host_cursor;
        groups.push(ReadGroup {
            file_path: None,
            file_offset: 0,
            bytes: spec.bytes,
            members: vec![index],
            host_offset: host_cursor,
        });
        host_cursor += spec.bytes;
    }

    (groups, host_offsets)
}

fn build_specs(
    _model: &ForgeCoreModel,
    tensors: &[forge_core::StagedTensorInfo],
    mode: HostStagingMode,
    mappings: &HashMap<String, forge_core::TensorMappingInfo>,
) -> Vec<TensorReadSpec> {
    tensors
        .iter()
        .map(|tensor| {
            let mapping = mappings.get(tensor.name.as_str());
            let direct = mode == HostStagingMode::Direct
                && mapping
                    .map(|mapping| {
                        mapping.file_backed
                            && mapping.file_path.is_some()
                            && mapping.size == tensor.bytes
                    })
                    .unwrap_or(false);
            TensorReadSpec {
                name: tensor.name.clone(),
                file_path: direct
                    .then(|| mapping.and_then(|mapping| mapping.file_path.clone()))
                    .flatten(),
                file_offset: if direct {
                    mapping.map(|mapping| mapping.offset).unwrap_or(0)
                } else {
                    0
                },
                bytes: tensor.bytes,
            }
        })
        .collect()
}

fn read_file_groups(host: &StagingBuffer, groups: &[ReadGroup]) -> Result<(), String> {
    let mut files: HashMap<String, File> = HashMap::new();
    for group in groups {
        let Some(path) = &group.file_path else {
            return Err("asynchronous direct read requires file-backed tensors".to_string());
        };
        if !files.contains_key(path) {
            let file = File::open(Path::new(path))
                .map_err(|error| format!("open {path} failed: {error}"))?;
            files.insert(path.clone(), file);
        }
        let file = files.get(path).expect("file opened");
        host.read_fd(
            group.host_offset,
            file.as_raw_fd(),
            group.file_offset,
            group.bytes,
        )
        .map_err(|error| format!("direct read of {path} failed: {error}"))?;
    }
    Ok(())
}

/// A host read running on a worker thread. The pinned buffer is owned by the
/// worker until `finish`.
type HostReadValue = (StagingBuffer, Vec<u64>, HostStagingReport);
type HostReadFailure = (String, StagingBuffer);
type HostReadOutcome = Result<HostReadValue, HostReadFailure>;
type HostReadFinish =
    Result<(StagingBuffer, Vec<u64>, HostStagingReport, u64), (String, Option<StagingBuffer>)>;

pub struct HostReadJob {
    handle: JoinHandle<HostReadOutcome>,
    started: Instant,
}

impl HostReadJob {
    pub fn prepare(
        model: &ForgeCoreModel,
        tensors: &[forge_core::StagedTensorInfo],
        mappings: &HashMap<String, forge_core::TensorMappingInfo>,
        host: StagingBuffer,
    ) -> Result<Self, ServiceError> {
        let specs = build_specs(model, tensors, HostStagingMode::Direct, mappings);
        if specs.iter().any(|spec| spec.file_path.is_none()) {
            return Err(ServiceError::Execution(
                "asynchronous host read requires file-backed staged tensors".to_string(),
            ));
        }
        let total: u64 = specs.iter().map(|spec| spec.bytes).sum();
        let capacity = host
            .info()
            .map_err(|error| ServiceError::Execution(error.to_string()))?
            .capacity;
        if total > capacity {
            return Err(ServiceError::Execution(format!(
                "pinned staging buffer capacity {capacity} is smaller than {total} bytes"
            )));
        }
        let (groups, offsets) = build_groups(&specs, ReadBatching::Coalesced);
        let tensors_count = tensors.len() as u64;
        let coalesced_groups = groups
            .iter()
            .filter(|group| group.members.len() > 1)
            .count() as u64;
        let read_groups = groups.len() as u64;
        let started = Instant::now();
        let handle = std::thread::spawn(move || {
            let read_started = Instant::now();
            let result = catch_unwind(AssertUnwindSafe(|| read_file_groups(&host, &groups)));
            match result {
                Ok(Ok(())) => {
                    let read_ns = crate::inference_service::duration_ns(read_started.elapsed());
                    let report = HostStagingReport {
                        mode: HostStagingMode::Direct,
                        batching: ReadBatching::Coalesced,
                        tensors: tensors_count,
                        bytes_read: total,
                        read_groups,
                        coalesced_groups,
                        read_ns,
                        prepare_ns: read_ns,
                        pinned_capacity: capacity,
                        pinned_reused: false,
                        fallback_tensors: 0,
                    };
                    Ok((host, offsets, report))
                }
                Ok(Err(error)) => Err((error, host)),
                Err(_) => Err(("host read worker panicked".to_string(), host)),
            }
        });
        Ok(Self { handle, started })
    }

    /// Wait for the worker and return the filled buffer, offsets, report and
    /// host wait duration. On error the pinned buffer is returned for recycling.
    pub fn finish(self) -> HostReadFinish {
        match self.handle.join() {
            Ok(Ok((host, offsets, report))) => {
                let wait_ns = self.started.elapsed().as_nanos() as u64;
                Ok((host, offsets, report, wait_ns))
            }
            Ok(Err((error, host))) => Err((error, Some(host))),
            Err(_) => Err(("host read worker join failed".to_string(), None)),
        }
    }
}

/// Read the selected staged tensors into a pinned host buffer.
pub fn read_tensors_into_host(
    model: &ForgeCoreModel,
    tensors: &[forge_core::StagedTensorInfo],
    host: &StagingBuffer,
    mode: HostStagingMode,
    batching: ReadBatching,
    mappings: &HashMap<String, forge_core::TensorMappingInfo>,
) -> Result<(Vec<u64>, HostStagingReport), ServiceError> {
    let specs = build_specs(model, tensors, mode, mappings);

    let capacity = host
        .info()
        .map_err(|error| ServiceError::Execution(error.to_string()))?
        .capacity;
    let total: u64 = specs.iter().map(|spec| spec.bytes).sum();
    if total > capacity {
        return Err(ServiceError::Execution(format!(
            "pinned staging buffer capacity {capacity} is smaller than {total} bytes"
        )));
    }

    let (groups, host_offsets) = build_groups(&specs, batching);
    let coalesced_groups = groups
        .iter()
        .filter(|group| group.members.len() > 1)
        .count() as u64;
    let read_started = Instant::now();
    let mut files: HashMap<String, File> = HashMap::new();
    let mut fallback_tensors = 0u64;
    for group in &groups {
        if let Some(path) = &group.file_path {
            if !files.contains_key(path) {
                let file = File::open(Path::new(path)).map_err(|error| {
                    ServiceError::Execution(format!("open {path} failed: {error}"))
                })?;
                files.insert(path.clone(), file);
            }
            let file = files.get(path).expect("file opened");
            host.read_fd(
                group.host_offset,
                file.as_raw_fd(),
                group.file_offset,
                group.bytes,
            )
            .map_err(|error| {
                ServiceError::Execution(format!("direct read of {path} failed: {error}"))
            })?;
        } else {
            for member in &group.members {
                let tensor = &tensors[*member];
                let length = usize::try_from(tensor.bytes).map_err(|_| {
                    ServiceError::Execution("staged tensor is too large".to_string())
                })?;
                let mut bytes = vec![0u8; length];
                model
                    .read_tensor_bytes(&tensor.name, 0, &mut bytes)
                    .map_err(|error| ServiceError::Execution(error.to_string()))?;
                host.write_host(host_offsets[*member], &bytes)
                    .map_err(|error| ServiceError::Execution(error.to_string()))?;
                fallback_tensors += 1;
            }
        }
    }
    let read_ns = crate::inference_service::duration_ns(read_started.elapsed());
    let report = HostStagingReport {
        mode,
        batching,
        tensors: tensors.len() as u64,
        bytes_read: total,
        read_groups: groups.len() as u64,
        coalesced_groups,
        read_ns,
        prepare_ns: read_ns,
        pinned_capacity: capacity,
        pinned_reused: false,
        fallback_tensors,
    };
    Ok((host_offsets, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(path: Option<&str>, offset: u64, bytes: u64) -> TensorReadSpec {
        TensorReadSpec {
            name: format!("t{offset}"),
            file_path: path.map(str::to_string),
            file_offset: offset,
            bytes,
        }
    }

    #[test]
    fn coalesces_exact_adjacent_ranges() {
        let specs = vec![
            spec(Some("/a"), 100, 10),
            spec(Some("/a"), 110, 20),
            spec(Some("/a"), 140, 5),
        ];
        let (groups, offsets) = build_groups(&specs, ReadBatching::Coalesced);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].bytes, 30);
        assert_eq!(groups[1].bytes, 5);
        assert_eq!(offsets, vec![0, 10, 30]);
    }

    #[test]
    fn does_not_coalesce_gaps_or_files() {
        let specs = vec![
            spec(Some("/a"), 100, 10),
            spec(Some("/a"), 120, 10),
            spec(Some("/b"), 0, 10),
        ];
        let (groups, offsets) = build_groups(&specs, ReadBatching::Coalesced);
        assert_eq!(groups.len(), 3);
        assert_eq!(offsets, vec![0, 10, 20]);
    }

    #[test]
    fn tensor_batching_preserves_input_order() {
        let specs = vec![spec(Some("/a"), 100, 10), spec(None, 0, 20)];
        let (groups, offsets) = build_groups(&specs, ReadBatching::Tensor);
        assert_eq!(groups.len(), 2);
        assert_eq!(offsets, vec![0, 10]);
        assert!(groups[1].file_path.is_none());
    }
}
