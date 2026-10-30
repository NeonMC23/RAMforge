//! Model-scoped memory ledger with explicit provenance.
//!
//! Categories are intentionally not summed into a single "memory" number.
//! Logical weights, mapped pages, resident pages, process RSS, and cgroup
//! accounting answer different questions.

use crate::residency_engine::{
    InventorySummary, ModelGeneration, ModelId, ResidencySnapshot, StagingPlan, StagingResult,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementKind {
    ExactMeasured,
    NativeLogical,
    Estimated,
    Unknown,
    Requested,
    NativeAccepted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryCategory {
    FileBackedLogical,
    FileBackedResident,
    AnonymousWeights,
    StagingRequested,
    StagingAccepted,
    ProcessRss,
    ProcessRssPeak,
    MappingPss,
    CgroupCurrent,
    CgroupPeak,
    KvCache,
    GraphScratch,
    Runtime,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerEntry {
    pub category: MemoryCategory,
    pub bytes: Option<u64>,
    pub kind: MeasurementKind,
    pub source: String,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reconciliation {
    pub mapped_virtual_bytes: u64,
    pub mapped_resident_bytes: Option<u64>,
    pub process_rss_bytes: Option<u64>,
    pub cgroup_current_bytes: Option<u64>,
    pub cgroup_peak_bytes: Option<u64>,
    pub rss_minus_mapping: Option<i128>,
    pub cgroup_minus_rss: Option<i128>,
    pub unknown_components: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LedgerError {
    Overflow,
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overflow => formatter.write_str("memory ledger byte total overflowed"),
        }
    }
}

impl std::error::Error for LedgerError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryLedger {
    pub model_id: ModelId,
    pub generation: ModelGeneration,
    pub entries: Vec<LedgerEntry>,
    pub unknown_components: Vec<String>,
    pub warnings: Vec<String>,
}

impl MemoryLedger {
    pub fn from_inventory(
        model_id: ModelId,
        generation: ModelGeneration,
        inventory: &InventorySummary,
        snapshot: &ResidencySnapshot,
        staging: Option<&StagingPlan>,
        staging_result: Option<&StagingResult>,
    ) -> Self {
        let mut entries = Vec::new();
        let mut unknown = Vec::new();
        let mut warnings = Vec::new();

        entries.push(entry(
            MemoryCategory::FileBackedLogical,
            Some(inventory.logical_file_backed_bytes),
            MeasurementKind::NativeLogical,
            "forgeCORE tensor mapping metadata",
            "sum of file-backed tensor payload bytes; not resident bytes",
        ));
        entries.push(entry(
            MemoryCategory::AnonymousWeights,
            Some(inventory.logical_anonymous_bytes),
            MeasurementKind::NativeLogical,
            "forgeCORE tensor mapping metadata",
            "anonymous/repacked tensor payloads; not file-backed",
        ));
        entries.push(entry(
            MemoryCategory::FileBackedResident,
            snapshot.mapping_rss_bytes,
            if snapshot.mapping_rss_bytes.is_some() {
                MeasurementKind::ExactMeasured
            } else {
                MeasurementKind::Unknown
            },
            "/proc/self/smaps",
            "resident process pages in model file mappings",
        ));
        entries.push(entry(
            MemoryCategory::MappingPss,
            snapshot.mapping_pss_bytes,
            if snapshot.mapping_pss_bytes.is_some() {
                MeasurementKind::ExactMeasured
            } else {
                MeasurementKind::Unknown
            },
            "/proc/self/smaps",
            "PSS share of model mappings",
        ));
        entries.push(entry(
            MemoryCategory::ProcessRss,
            snapshot.process_rss_bytes,
            if snapshot.process_rss_bytes.is_some() {
                MeasurementKind::ExactMeasured
            } else {
                MeasurementKind::Unknown
            },
            "/proc/self/status",
            "whole-process RSS",
        ));
        entries.push(entry(
            MemoryCategory::ProcessRssPeak,
            snapshot.process_rss_peak_bytes,
            if snapshot.process_rss_peak_bytes.is_some() {
                MeasurementKind::ExactMeasured
            } else {
                MeasurementKind::Unknown
            },
            "/proc/self/status VmHWM",
            "process RSS high-water mark",
        ));
        entries.push(entry(
            MemoryCategory::CgroupCurrent,
            snapshot.cgroup_current_bytes,
            if snapshot.cgroup_current_bytes.is_some() {
                MeasurementKind::ExactMeasured
            } else {
                MeasurementKind::Unknown
            },
            "cgroup v2 memory.current",
            "charged cgroup memory, not all host page cache",
        ));
        entries.push(entry(
            MemoryCategory::CgroupPeak,
            snapshot.cgroup_peak_bytes,
            if snapshot.cgroup_peak_bytes.is_some() {
                MeasurementKind::ExactMeasured
            } else {
                MeasurementKind::Unknown
            },
            "cgroup v2 memory.peak",
            "charged cgroup high-water mark",
        ));

        if let Some(staging) = staging {
            entries.push(entry(
                MemoryCategory::StagingRequested,
                Some(staging.selected_bytes),
                MeasurementKind::Requested,
                "RAMforge staging plan",
                "page bytes requested for prefetch advice",
            ));
        }
        if let Some(result) = staging_result {
            entries.push(entry(
                MemoryCategory::StagingAccepted,
                Some(result.native_requested_bytes),
                MeasurementKind::NativeAccepted,
                "forgeCORE prefetch result",
                "bytes accepted by the native prefetch hint; not observed residency",
            ));
        }

        if inventory.logical_file_backed_bytes == 0 && inventory.file_backed_tensors > 0 {
            warnings
                .push("file-backed tensor count is nonzero but logical bytes are zero".to_string());
        }
        for (category, reason) in [
            (
                MemoryCategory::KvCache,
                "KV cache bytes are unavailable from the pinned C API",
            ),
            (
                MemoryCategory::GraphScratch,
                "graph/scratch sizes are not exposed",
            ),
            (
                MemoryCategory::Runtime,
                "allocator/runtime overhead is not attributed natively",
            ),
        ] {
            entries.push(entry(
                category,
                None,
                MeasurementKind::Unknown,
                "unavailable",
                reason,
            ));
            unknown.push(reason.to_string());
        }

        Self {
            model_id,
            generation,
            entries,
            unknown_components: unknown,
            warnings,
        }
    }

    pub fn reconciliation(&self, mapped_virtual_bytes: u64) -> Reconciliation {
        let resident = self.value(MemoryCategory::FileBackedResident);
        let rss = self.value(MemoryCategory::ProcessRss);
        let cgroup = self.value(MemoryCategory::CgroupCurrent);
        let peak = self.value(MemoryCategory::CgroupPeak);
        let rss_minus_mapping = resident
            .zip(rss)
            .map(|(mapping, rss)| rss as i128 - mapping as i128);
        let cgroup_minus_rss = cgroup
            .zip(rss)
            .map(|(cgroup, rss)| cgroup as i128 - rss as i128);
        let mut notes = Vec::new();
        if rss_minus_mapping.is_some() {
            notes.push("RSS - mapping resident includes KV, scratch, allocator, runtime, and other process pages".to_string());
        }
        if cgroup_minus_rss.is_some() {
            notes.push("cgroup.current can differ from RSS because of charged page cache and kernel accounting".to_string());
        }
        notes.push("mapped virtual bytes are not resident bytes".to_string());
        Reconciliation {
            mapped_virtual_bytes,
            mapped_resident_bytes: resident,
            process_rss_bytes: rss,
            cgroup_current_bytes: cgroup,
            cgroup_peak_bytes: peak,
            rss_minus_mapping,
            cgroup_minus_rss,
            unknown_components: self.unknown_components.clone(),
            notes,
        }
    }

    pub fn value(&self, category: MemoryCategory) -> Option<u64> {
        self.entries
            .iter()
            .find(|entry| entry.category == category)
            .and_then(|entry| entry.bytes)
    }

    /// Sum only requested categories with checked arithmetic. Categories that
    /// represent different measurements should never be passed together.
    pub fn checked_total(&self, categories: &[MemoryCategory]) -> Result<u64, LedgerError> {
        let mut total = 0u64;
        for entry in &self.entries {
            if !categories.contains(&entry.category) {
                continue;
            }
            if let Some(bytes) = entry.bytes {
                total = total.checked_add(bytes).ok_or(LedgerError::Overflow)?;
            }
        }
        Ok(total)
    }

    pub fn kind(&self, category: MemoryCategory) -> Option<MeasurementKind> {
        self.entries
            .iter()
            .find(|entry| entry.category == category)
            .map(|entry| entry.kind)
    }
}

fn entry(
    category: MemoryCategory,
    bytes: Option<u64>,
    kind: MeasurementKind,
    source: &str,
    note: &str,
) -> LedgerEntry {
    LedgerEntry {
        category,
        bytes,
        kind,
        source: source.to_string(),
        note: note.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residency_engine::ResidencySnapshot;

    fn inventory() -> InventorySummary {
        InventorySummary {
            tensors: 2,
            file_backed_tensors: 1,
            anonymous_tensors: 1,
            mappings: 1,
            logical_file_backed_bytes: 100,
            logical_anonymous_bytes: 50,
            page_bytes: 4096,
            shared_page_bytes: 0,
            unknown_tensors: 1,
        }
    }

    fn snapshot() -> ResidencySnapshot {
        ResidencySnapshot {
            process_rss_bytes: Some(300),
            process_rss_peak_bytes: Some(350),
            mapping_rss_bytes: Some(200),
            mapping_pss_bytes: Some(180),
            cgroup_current_bytes: Some(500),
            cgroup_peak_bytes: Some(600),
            cgroup_events: None,
            process_minor_faults: None,
            process_major_faults: None,
        }
    }

    #[test]
    fn ledger_keeps_categories_distinct() {
        let ledger = MemoryLedger::from_inventory(
            ModelId("m".to_string()),
            ModelGeneration(1),
            &inventory(),
            &snapshot(),
            None,
            None,
        );
        assert_eq!(ledger.value(MemoryCategory::FileBackedLogical), Some(100));
        assert_eq!(ledger.value(MemoryCategory::AnonymousWeights), Some(50));
        assert_eq!(ledger.value(MemoryCategory::FileBackedResident), Some(200));
        assert_eq!(ledger.value(MemoryCategory::ProcessRss), Some(300));
        assert_eq!(ledger.value(MemoryCategory::KvCache), None);
        assert_eq!(
            ledger.kind(MemoryCategory::KvCache),
            Some(MeasurementKind::Unknown)
        );
        let reconciliation = ledger.reconciliation(4096);
        assert_eq!(reconciliation.rss_minus_mapping, Some(100));
        assert_eq!(reconciliation.cgroup_minus_rss, Some(200));
        assert!(reconciliation.unknown_components.len() >= 3);
    }

    #[test]
    fn checked_totals_reject_overflow() {
        let mut ledger = MemoryLedger::from_inventory(
            ModelId("m".to_string()),
            ModelGeneration(1),
            &inventory(),
            &snapshot(),
            None,
            None,
        );
        ledger.entries.push(entry(
            MemoryCategory::FileBackedLogical,
            Some(u64::MAX),
            MeasurementKind::NativeLogical,
            "test",
            "overflow fixture",
        ));
        ledger.entries.push(entry(
            MemoryCategory::FileBackedLogical,
            Some(1),
            MeasurementKind::NativeLogical,
            "test",
            "overflow fixture",
        ));
        assert_eq!(
            ledger.checked_total(&[MemoryCategory::FileBackedLogical]),
            Err(LedgerError::Overflow)
        );
    }

    #[test]
    fn checked_totals_do_not_mix_categories() {
        let ledger = MemoryLedger::from_inventory(
            ModelId("m".to_string()),
            ModelGeneration(1),
            &inventory(),
            &snapshot(),
            None,
            None,
        );
        assert_eq!(
            ledger
                .checked_total(&[MemoryCategory::FileBackedLogical])
                .expect("total"),
            100
        );
        assert_eq!(
            ledger
                .checked_total(&[MemoryCategory::FileBackedResident])
                .expect("total"),
            200
        );
    }

    #[test]
    fn missing_measurements_remain_unknown() {
        let ledger = MemoryLedger::from_inventory(
            ModelId("m".to_string()),
            ModelGeneration(1),
            &inventory(),
            &ResidencySnapshot::default(),
            None,
            None,
        );
        assert_eq!(ledger.value(MemoryCategory::ProcessRss), None);
        assert_eq!(
            ledger.kind(MemoryCategory::ProcessRss),
            Some(MeasurementKind::Unknown)
        );
        assert_eq!(ledger.reconciliation(4096).rss_minus_mapping, None);
    }
}
