//! Bounded runtime lifecycle memory observation.
//!
//! Snapshots are taken only at explicit lifecycle boundaries. Process totals,
//! system/cgroup values, mapping residency, and backend values remain separate;
//! no category is fabricated or summed into a fake total.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::machine_profile::memory::MemoryProfile;
use crate::machine_profile::{collect_memory_profile, Freshness, Observed, Provenance};

pub const MAX_RUNTIME_SNAPSHOTS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleStage {
    Baseline,
    ModelInspected,
    ModelLoaded,
    ContextCreated,
    PromptProcessed,
    DecodeSteady,
    ContextDestroyed,
    ModelUnloaded,
    Final,
}

impl LifecycleStage {
    pub fn rank(self) -> u8 {
        match self {
            Self::Baseline => 0,
            Self::ModelInspected => 1,
            Self::ModelLoaded => 2,
            Self::ContextCreated => 3,
            Self::PromptProcessed => 4,
            Self::DecodeSteady => 5,
            Self::ContextDestroyed => 6,
            Self::ModelUnloaded => 7,
            Self::Final => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessMemorySnapshot {
    pub rss_bytes: Observed<u64>,
    pub peak_rss_bytes: Observed<u64>,
    pub virtual_bytes: Observed<u64>,
    pub anonymous_bytes: Observed<u64>,
    pub file_backed_bytes: Observed<u64>,
    pub shared_bytes: Observed<u64>,
    pub private_bytes: Observed<u64>,
    pub swap_bytes: Observed<u64>,
    pub locked_bytes: Observed<u64>,
    pub thread_count: Observed<u64>,
    pub mapping_count: Observed<u64>,
    pub mapping_bytes: Observed<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemMemorySnapshot {
    pub physical_total_bytes: Observed<u64>,
    pub os_available_bytes: Observed<u64>,
    pub cgroup_limit_bytes: Observed<u64>,
    pub cgroup_current_bytes: Observed<u64>,
    pub cgroup_peak_bytes: Observed<u64>,
    pub cgroup_anon_bytes: Observed<u64>,
    pub cgroup_file_bytes: Observed<u64>,
    pub cgroup_file_mapped_bytes: Observed<u64>,
    pub cgroup_swap_bytes: Observed<u64>,
    pub pressure: Observed<crate::machine_profile::memory::MemoryPressure>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelMappingMemory {
    pub mapped_bytes: u64,
    pub resident_bytes: u64,
    pub pss_bytes: u64,
    pub private_bytes: u64,
    pub shared_bytes: u64,
    pub mapping_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendMemorySnapshot {
    pub device_id: String,
    pub backend_id: String,
    pub memory_total_bytes: Observed<u64>,
    pub memory_free_bytes: Observed<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
    pub sequence: u64,
    pub stage: LifecycleStage,
    pub captured_at_unix_ms: Option<u64>,
    pub model_instance_id: Option<u64>,
    pub context_id: Option<u64>,
    pub process: ProcessMemorySnapshot,
    pub system: SystemMemorySnapshot,
    pub model_mapping: Option<ModelMappingMemory>,
    pub backend: Vec<BackendMemorySnapshot>,
}

impl RuntimeSnapshot {
    pub fn validate(&self) -> Result<(), String> {
        self.process
            .rss_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        self.process
            .peak_rss_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        self.process
            .virtual_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        self.process
            .anonymous_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        self.process
            .file_backed_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        self.process
            .shared_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        self.process
            .private_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        self.process
            .swap_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        self.process
            .locked_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        self.process
            .thread_count
            .validate()
            .map_err(|e| e.to_string())?;
        self.process
            .mapping_count
            .validate()
            .map_err(|e| e.to_string())?;
        self.process
            .mapping_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        self.system
            .physical_total_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        self.system
            .os_available_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        self.system
            .cgroup_limit_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        self.system
            .cgroup_current_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        self.system
            .cgroup_peak_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        self.system
            .cgroup_anon_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        self.system
            .cgroup_file_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        self.system
            .cgroup_file_mapped_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        self.system
            .cgroup_swap_bytes
            .validate()
            .map_err(|e| e.to_string())?;
        self.system.pressure.validate().map_err(|e| e.to_string())?;
        for device in &self.backend {
            device
                .memory_total_bytes
                .validate()
                .map_err(|e| e.to_string())?;
            device
                .memory_free_bytes
                .validate()
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

/// Bounded recorder for explicit lifecycle stages.
#[derive(Debug)]
pub struct LifecycleRecorder {
    model_path: Option<PathBuf>,
    ram_budget_bytes: Option<u64>,
    next_sequence: u64,
    snapshots: Vec<RuntimeSnapshot>,
    dropped_snapshots: u64,
    model_instance_id: Option<u64>,
    context_id: Option<u64>,
}

impl LifecycleRecorder {
    pub fn new(model_path: Option<PathBuf>, ram_budget_bytes: Option<u64>) -> Self {
        Self {
            model_path: model_path.map(|path| path.canonicalize().unwrap_or(path)),
            ram_budget_bytes,
            next_sequence: 1,
            snapshots: Vec::new(),
            dropped_snapshots: 0,
            model_instance_id: None,
            context_id: None,
        }
    }

    pub fn set_model_instance_id(&mut self, id: u64) {
        self.model_instance_id = Some(id);
    }

    pub fn set_context_id(&mut self, id: u64) {
        self.context_id = Some(id);
    }

    pub fn clear_context_id(&mut self) {
        self.context_id = None;
    }

    pub fn record(&mut self, stage: LifecycleStage) -> RuntimeSnapshot {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        let snapshot = capture_snapshot(
            sequence,
            stage,
            self.model_path.as_deref(),
            self.ram_budget_bytes,
            self.model_instance_id,
            self.context_id,
        );
        if self.snapshots.len() >= MAX_RUNTIME_SNAPSHOTS {
            self.snapshots.remove(0);
            self.dropped_snapshots = self.dropped_snapshots.saturating_add(1);
        }
        self.snapshots.push(snapshot.clone());
        snapshot
    }

    pub fn snapshots(&self) -> &[RuntimeSnapshot] {
        &self.snapshots
    }

    pub fn dropped_snapshots(&self) -> u64 {
        self.dropped_snapshots
    }
}

pub fn capture_snapshot(
    sequence: u64,
    stage: LifecycleStage,
    model_path: Option<&Path>,
    ram_budget_bytes: Option<u64>,
    model_instance_id: Option<u64>,
    context_id: Option<u64>,
) -> RuntimeSnapshot {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let rollup = std::fs::read_to_string("/proc/self/smaps_rollup").unwrap_or_default();
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    let (mapping_count, mapping_bytes) = mapping_totals(&maps);
    let memory = collect_memory_profile(ram_budget_bytes, sequence);
    RuntimeSnapshot {
        sequence,
        stage,
        captured_at_unix_ms: unix_millis(),
        model_instance_id,
        context_id,
        process: ProcessMemorySnapshot {
            rss_bytes: kib_field(&status, "VmRSS:", sequence),
            peak_rss_bytes: kib_field(&status, "VmHWM:", sequence),
            virtual_bytes: kib_field(&status, "VmSize:", sequence),
            anonymous_bytes: kib_field(&rollup, "Anonymous:", sequence),
            file_backed_bytes: kib_field(&rollup, "Pss_File:", sequence),
            shared_bytes: kib_field(&rollup, "Shared_Clean:", sequence),
            private_bytes: kib_field(&rollup, "Private_Dirty:", sequence),
            swap_bytes: kib_field(&status, "VmSwap:", sequence),
            locked_bytes: kib_field(&rollup, "Locked:", sequence),
            thread_count: status
                .lines()
                .find_map(|line| line.strip_prefix("Threads:"))
                .and_then(|value| value.trim().parse::<u64>().ok())
                .map(|value| {
                    Observed::known(
                        value,
                        Provenance::OperatingSystem,
                        Freshness::Dynamic,
                        sequence,
                    )
                })
                .unwrap_or_else(|| Observed::unavailable("Threads field unavailable", sequence)),
            mapping_count: mapping_count
                .map(|value| {
                    Observed::known(
                        value,
                        Provenance::OperatingSystem,
                        Freshness::Dynamic,
                        sequence,
                    )
                })
                .unwrap_or_else(|| Observed::unavailable("mapping count unavailable", sequence)),
            mapping_bytes: mapping_bytes
                .map(|value| {
                    Observed::known(
                        value,
                        Provenance::OperatingSystem,
                        Freshness::Dynamic,
                        sequence,
                    )
                })
                .unwrap_or_else(|| Observed::unavailable("mapping bytes unavailable", sequence)),
        },
        system: system_snapshot(&memory),
        model_mapping: model_path.and_then(scan_model_mapping),
        backend: backend_snapshots(sequence),
    }
}

fn system_snapshot(memory: &MemoryProfile) -> SystemMemorySnapshot {
    SystemMemorySnapshot {
        physical_total_bytes: memory.physical_total_bytes.clone(),
        os_available_bytes: memory.os_available_bytes.clone(),
        cgroup_limit_bytes: memory.cgroup.limit_bytes.clone(),
        cgroup_current_bytes: memory.cgroup.current_bytes.clone(),
        cgroup_peak_bytes: memory.cgroup.peak_bytes.clone(),
        cgroup_anon_bytes: memory.cgroup.anon_bytes.clone(),
        cgroup_file_bytes: memory.cgroup.file_bytes.clone(),
        cgroup_file_mapped_bytes: memory.cgroup.file_mapped_bytes.clone(),
        cgroup_swap_bytes: memory.cgroup.swap_current_bytes.clone(),
        pressure: memory.pressure.clone(),
    }
}

fn kib_field(text: &str, key: &str, sequence: u64) -> Observed<u64> {
    text.lines()
        .find(|line| line.starts_with(key))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u64>().ok())
        .and_then(|value| value.checked_mul(1024))
        .map(|value| {
            Observed::known(
                value,
                Provenance::OperatingSystem,
                Freshness::Dynamic,
                sequence,
            )
        })
        .unwrap_or_else(|| Observed::unavailable(format!("{key} unavailable"), sequence))
}

fn mapping_totals(maps: &str) -> (Option<u64>, Option<u64>) {
    let mut count = 0u64;
    let mut bytes = 0u64;
    for line in maps.lines() {
        let Some(range) = line.split_whitespace().next() else {
            continue;
        };
        let Some((start, end)) = range.split_once('-') else {
            continue;
        };
        let (Ok(start), Ok(end)) = (u64::from_str_radix(start, 16), u64::from_str_radix(end, 16))
        else {
            return (None, None);
        };
        count = count.saturating_add(1);
        bytes = bytes.saturating_add(end.saturating_sub(start));
    }
    if count == 0 {
        (None, None)
    } else {
        (Some(count), Some(bytes))
    }
}

fn scan_model_mapping(path: &Path) -> Option<ModelMappingMemory> {
    let target = path.to_string_lossy();
    let target_name = path.file_name().map(|name| name.to_string_lossy());
    let text = std::fs::read_to_string("/proc/self/smaps").ok()?;
    let mut matched = false;
    let mut mapped_bytes = 0u64;
    let mut resident_bytes = 0u64;
    let mut pss_bytes = 0u64;
    let mut private_bytes = 0u64;
    let mut shared_bytes = 0u64;
    let mut mapping_count = 0u64;
    let mut current_matches = false;
    for line in text.lines() {
        if let Some((range, rest)) = line.split_once(' ') {
            if let Some((start, end)) = range.split_once('-') {
                if let (Ok(start), Ok(end)) =
                    (u64::from_str_radix(start, 16), u64::from_str_radix(end, 16))
                {
                    current_matches = rest.splitn(5, ' ').nth(4).is_some_and(|mapped_path| {
                        let mapped_path = mapped_path
                            .trim_start()
                            .trim_end_matches(" (deleted)")
                            .replace("\\040", " ")
                            .replace("\\011", "\t")
                            .replace("\\012", "\n")
                            .replace("\\134", "\\");
                        mapped_path == target
                            || std::path::Path::new(&mapped_path)
                                .file_name()
                                .and_then(|name| name.to_str())
                                == target_name.as_deref()
                    });
                    if current_matches {
                        matched = true;
                        mapping_count = mapping_count.saturating_add(1);
                        mapped_bytes = mapped_bytes.saturating_add(end.saturating_sub(start));
                    }
                    continue;
                }
            }
        }
        if !current_matches {
            continue;
        }
        if let Some((key, value)) = line.split_once(':') {
            let value = value
                .split_whitespace()
                .next()
                .and_then(|value| value.parse::<u64>().ok())
                .and_then(|value| value.checked_mul(1024));
            match (key, value) {
                ("Rss", Some(value)) => resident_bytes = resident_bytes.saturating_add(value),
                ("Pss", Some(value)) => pss_bytes = pss_bytes.saturating_add(value),
                ("Private_Dirty", Some(value)) | ("Private_Clean", Some(value)) => {
                    private_bytes = private_bytes.saturating_add(value)
                }
                ("Shared_Clean", Some(value)) | ("Shared_Dirty", Some(value)) => {
                    shared_bytes = shared_bytes.saturating_add(value)
                }
                _ => {}
            }
        }
    }
    matched.then_some(ModelMappingMemory {
        mapped_bytes,
        resident_bytes,
        pss_bytes,
        private_bytes,
        shared_bytes,
        mapping_count,
    })
}

fn backend_snapshots(sequence: u64) -> Vec<BackendMemorySnapshot> {
    crate::execution::DeviceRegistry::discover()
        .devices()
        .iter()
        .map(|device| BackendMemorySnapshot {
            device_id: device.id.to_string(),
            backend_id: device.backend_id.to_string(),
            memory_total_bytes: device
                .memory_total
                .bytes()
                .map(|value| {
                    Observed::known(
                        value,
                        Provenance::NativeBackend,
                        Freshness::Stable,
                        sequence,
                    )
                })
                .unwrap_or_else(|| Observed::unavailable("device total unavailable", sequence)),
            memory_free_bytes: device
                .memory_free
                .bytes()
                .map(|value| {
                    Observed::known(
                        value,
                        Provenance::NativeBackend,
                        Freshness::Dynamic,
                        sequence,
                    )
                })
                .unwrap_or_else(|| {
                    Observed::unavailable("device free memory unavailable", sequence)
                }),
        })
        .collect()
}

fn unix_millis() -> Option<u64> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
}

/// RSS and mapping deltas between lifecycle stages. Negative deltas are valid.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleDeltas {
    pub model_load_rss_bytes: Option<i128>,
    pub context_create_rss_bytes: Option<i128>,
    pub prompt_rss_bytes: Option<i128>,
    pub decode_rss_bytes: Option<i128>,
    pub context_destroy_rss_bytes: Option<i128>,
    pub model_unload_rss_bytes: Option<i128>,
    pub model_load_mapping_rss_bytes: Option<i128>,
    pub context_create_mapping_rss_bytes: Option<i128>,
}

impl LifecycleDeltas {
    pub fn from_snapshots(snapshots: &[RuntimeSnapshot]) -> Self {
        let find = |stage| snapshots.iter().find(|snapshot| snapshot.stage == stage);
        let rss = |snapshot: &RuntimeSnapshot| snapshot.process.rss_bytes.as_ref().copied();
        let mapping = |snapshot: &RuntimeSnapshot| {
            snapshot
                .model_mapping
                .as_ref()
                .map(|mapping| mapping.resident_bytes)
        };
        let delta =
            |left: Option<u64>, right: Option<u64>| Some(i128::from(right?) - i128::from(left?));
        let baseline = find(LifecycleStage::Baseline);
        let loaded = find(LifecycleStage::ModelLoaded);
        let context = find(LifecycleStage::ContextCreated);
        let prompt = find(LifecycleStage::PromptProcessed);
        let decode = find(LifecycleStage::DecodeSteady);
        let destroyed = find(LifecycleStage::ContextDestroyed);
        let unloaded = find(LifecycleStage::ModelUnloaded);
        Self {
            model_load_rss_bytes: delta(baseline.and_then(rss), loaded.and_then(rss)),
            context_create_rss_bytes: delta(loaded.and_then(rss), context.and_then(rss)),
            prompt_rss_bytes: delta(context.and_then(rss), prompt.and_then(rss)),
            decode_rss_bytes: delta(prompt.and_then(rss), decode.and_then(rss)),
            context_destroy_rss_bytes: delta(decode.and_then(rss), destroyed.and_then(rss)),
            model_unload_rss_bytes: delta(destroyed.and_then(rss), unloaded.and_then(rss)),
            model_load_mapping_rss_bytes: delta(
                baseline.and_then(mapping),
                loaded.and_then(mapping),
            ),
            context_create_mapping_rss_bytes: delta(
                loaded.and_then(mapping),
                context.and_then(mapping),
            ),
        }
    }
}

/// One observed or estimated memory component. `overlapping` components must
/// never be summed into a fabricated process total.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryComponent {
    pub component: String,
    pub bytes: Option<u64>,
    pub certainty: String,
    pub provenance: String,
    pub method: String,
    pub overlapping: bool,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeAttribution {
    pub components: Vec<MemoryComponent>,
    pub unattributed_bytes: Option<u64>,
    pub note: String,
    pub limitations: Vec<String>,
}

impl RuntimeAttribution {
    pub fn validate(&self) -> Result<(), String> {
        for component in &self.components {
            match (component.bytes, component.certainty.as_str()) {
                (Some(_), "unknown") => return Err("unknown component carries bytes".to_string()),
                (None, "exact") | (None, "estimated") => {
                    return Err("numeric component has no bytes".to_string())
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_totals_parse_proc_ranges() {
        let maps = "00400000-00452000 r-xp 00000000 00:00 0 /bin/test\n7f0000000000-7f0000001000 rw-p 00000000 00:00 0 [heap]\n";
        let (count, bytes) = mapping_totals(maps);
        assert_eq!(count, Some(2));
        assert_eq!(bytes, Some(0x52000 + 0x1000));
    }

    #[test]
    fn lifecycle_sequence_is_bounded_and_ordered() {
        let mut recorder = LifecycleRecorder::new(None, Some(1024));
        let first = recorder.record(LifecycleStage::Baseline);
        let second = recorder.record(LifecycleStage::ModelLoaded);
        assert!(second.sequence > first.sequence);
        assert!(second.stage.rank() > first.stage.rank());
        assert_eq!(recorder.snapshots().len(), 2);
        for _ in 0..MAX_RUNTIME_SNAPSHOTS + 2 {
            recorder.record(LifecycleStage::Final);
        }
        assert!(recorder.snapshots().len() <= MAX_RUNTIME_SNAPSHOTS);
        assert!(recorder.dropped_snapshots() > 0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn snapshot_keeps_unknown_distinct_from_zero() {
        let snapshot = capture_snapshot(1, LifecycleStage::Baseline, None, Some(1024), None, None);
        snapshot.validate().expect("snapshot");
        assert!(snapshot.process.rss_bytes.is_known());
        assert!(snapshot.system.physical_total_bytes.is_known());
        assert_eq!(snapshot.process.swap_bytes.as_ref().copied(), Some(0));
    }

    #[test]
    fn deltas_preserve_negative_values() {
        let mut recorder = LifecycleRecorder::new(None, None);
        let first = recorder.record(LifecycleStage::Baseline);
        let mut second = first.clone();
        second.stage = LifecycleStage::ModelLoaded;
        if let Some(rss) = second.process.rss_bytes.value {
            second.process.rss_bytes.value = Some(rss.saturating_sub(1024));
        }
        let deltas = LifecycleDeltas::from_snapshots(&[first, second]);
        assert!(deltas.model_load_rss_bytes.unwrap_or(0) <= 0);
    }
}
