//! Runtime memory visibility.
//!
//! RAMforge-managed bytes, process RSS, and system memory are intentionally
//! reported as separate concepts. MemoryBudget does not control OS page cache.

use ramforge_core::{
    memory::MemoryBudget,
    resource::{ByteMeasurement, ResourceBudgets},
};

use crate::cgroup::CgroupStats;
use crate::execution::DeviceInfo;
use crate::page_residency::MappingResidency;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryReport {
    pub ramforge_current_bytes: u64,
    pub ramforge_peak_bytes: u64,
    pub ramforge_budget_bytes: u64,
    pub process_rss_bytes: Option<u64>,
    pub system_total_bytes: Option<u64>,
    pub system_available_bytes: Option<u64>,
}

impl MemoryReport {
    pub fn collect(budget: &MemoryBudget) -> Self {
        let (system_total_bytes, system_available_bytes) = system_memory();
        Self {
            ramforge_current_bytes: budget.used_bytes(),
            ramforge_peak_bytes: budget.peak_used_bytes(),
            ramforge_budget_bytes: budget.total_bytes(),
            process_rss_bytes: process_rss(),
            system_total_bytes,
            system_available_bytes,
        }
    }

    /// Expose the configured RAMforge host budget in the Phase-3 domain-aware
    /// representation.  RSS and OS page cache remain separate facts and are
    /// not inserted as allocations.
    pub fn resource_budgets(&self) -> ResourceBudgets {
        ResourceBudgets::host_ram(self.ramforge_budget_bytes)
    }
}

/// How the requested budget is enforced, if at all.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BudgetEnforcement {
    /// The value is recorded and used for planning, but the OS does not cap it.
    #[default]
    DiagnosticOnly,
    /// A cgroup v2 `memory.max` hard limit was active for the measured process.
    CgroupV2,
}

impl BudgetEnforcement {
    pub const fn label(self) -> &'static str {
        match self {
            Self::DiagnosticOnly => "diagnostic-only",
            Self::CgroupV2 => "cgroup-v2-memory.max",
        }
    }
}

/// Memory facts for the ForgeCore/libllama execution path.
///
/// This report deliberately separates facts that are often conflated:
///
/// - `logical_model_bytes` is the sum of GGUF tensor payload lengths read by
///   RAMforge's inspector. It is a file/logical size, not an allocation.
/// - `native_model_bytes` is libllama's own `size_bytes()` accounting.
/// - `model_residency` is the measured RSS/PSS of process mappings backed by
///   the model file. llama.cpp uses a file-backed mmap by default, so these
///   pages are reclaimable rather than an anonymous copy of the weights.
/// - RSS and peak-RSS values are measured process facts from
///   `/proc/self/status`; they include the loader, page cache effects, Rust
///   heap, and everything else in the process.
/// - `cgroup` is present only when the run used an explicit cgroup v2 limit.
///   `memory.peak` is the kernel's charged-memory peak for the cgroup.
/// - device totals/free bytes are native snapshots exposed by forgeCORE.
///
/// A mmap-backed model is not proof that RSS is bounded. The kernel reclaims
/// clean file pages under pressure, but the pinned C API exposes no per-layer
/// eviction hook. A cgroup limit is a hard process-group cap; RAMforge does not
/// perform layer-at-a-time staging on the ForgeCore path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgeCoreMemoryReport {
    pub requested_budget_bytes: u64,
    pub enforcement: BudgetEnforcement,
    pub logical_model_bytes: Option<u64>,
    pub native_model_bytes: u64,
    pub load_mode: Option<String>,
    pub weight_repacking: Option<String>,
    pub rss_before_load_bytes: Option<u64>,
    pub rss_after_load_bytes: Option<u64>,
    pub rss_after_decode_bytes: Option<u64>,
    pub rss_peak_bytes: Option<u64>,
    pub model_residency: Option<MappingResidency>,
    pub cgroup: Option<CgroupStats>,
    pub system_total_bytes: Option<u64>,
    pub system_available_bytes: Option<u64>,
    pub device_memory_total: ByteMeasurement,
    pub device_memory_free: ByteMeasurement,
}

impl ForgeCoreMemoryReport {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        requested_budget_bytes: u64,
        logical_model_bytes: Option<u64>,
        native_model_bytes: u64,
        rss_before_load_bytes: Option<u64>,
        rss_after_load_bytes: Option<u64>,
        rss_after_decode_bytes: Option<u64>,
        device: &DeviceInfo,
    ) -> Self {
        let (system_total_bytes, system_available_bytes) = system_memory();
        Self {
            requested_budget_bytes,
            enforcement: BudgetEnforcement::DiagnosticOnly,
            logical_model_bytes,
            native_model_bytes,
            load_mode: None,
            weight_repacking: None,
            rss_before_load_bytes,
            rss_after_load_bytes,
            rss_after_decode_bytes,
            rss_peak_bytes: process_rss_peak(),
            model_residency: None,
            cgroup: None,
            system_total_bytes,
            system_available_bytes,
            device_memory_total: device.memory_total.clone(),
            device_memory_free: device.memory_free.clone(),
        }
    }

    /// Attach an active cgroup v2 limit and its measured counters.
    pub fn with_cgroup(mut self, stats: CgroupStats) -> Self {
        self.enforcement = BudgetEnforcement::CgroupV2;
        self.cgroup = Some(stats);
        self
    }

    /// Attach measured residency for the model file mapping.
    pub fn with_model_residency(mut self, residency: MappingResidency) -> Self {
        self.model_residency = Some(residency);
        self
    }

    /// Record the requested native load mode.
    pub fn with_load_mode(mut self, load_mode: impl Into<String>) -> Self {
        self.load_mode = Some(load_mode.into());
        self
    }

    /// Record whether native CPU weight repacking was enabled.
    pub fn with_weight_repacking(mut self, label: impl Into<String>) -> Self {
        self.weight_repacking = Some(label.into());
        self
    }

    /// True only when a real cgroup v2 limit was active for this run.
    pub const fn enforces_budget(&self) -> bool {
        matches!(self.enforcement, BudgetEnforcement::CgroupV2)
    }

    /// True when the kernel reported a cgroup limit or OOM event.
    pub fn cgroup_exceeded(&self) -> bool {
        self.cgroup.as_ref().is_some_and(CgroupStats::exceeded)
    }

    /// True when the logical GGUF payload is larger than the requested budget.
    /// This is a planning fact, not evidence that the process exceeded it.
    pub fn logical_model_exceeds_budget(&self) -> Option<bool> {
        self.logical_model_bytes
            .map(|logical| logical > self.requested_budget_bytes)
    }

    /// Measured RSS change from before load to after load.
    pub fn rss_delta_after_load_bytes(&self) -> Option<i128> {
        rss_delta(self.rss_before_load_bytes, self.rss_after_load_bytes)
    }

    /// Measured RSS change from before load to after decode/generation.
    pub fn rss_delta_after_decode_bytes(&self) -> Option<i128> {
        rss_delta(self.rss_before_load_bytes, self.rss_after_decode_bytes)
    }

    /// Native memory categories that the pinned C API does not expose.
    pub const fn unexposed_categories(&self) -> &'static [&'static str] {
        &[
            "KV cache bytes (only the cgroup total is visible)",
            "graph scratch/workspace",
            "staging buffers",
        ]
    }
}

fn rss_delta(before: Option<u64>, after: Option<u64>) -> Option<i128> {
    Some(i128::from(after?) - i128::from(before?))
}

#[cfg(target_os = "linux")]
pub fn process_rss() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    parse_kib_field(&status, "VmRSS:")
}

#[cfg(not(target_os = "linux"))]
pub fn process_rss() -> Option<u64> {
    None
}

/// Peak resident set size for this process (`VmHWM`).
#[cfg(target_os = "linux")]
pub fn process_rss_peak() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    parse_kib_field(&status, "VmHWM:")
}

#[cfg(not(target_os = "linux"))]
pub fn process_rss_peak() -> Option<u64> {
    None
}

#[cfg(target_os = "linux")]
fn system_memory() -> (Option<u64>, Option<u64>) {
    let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") else {
        return (None, None);
    };
    (
        parse_kib_field(&meminfo, "MemTotal:"),
        parse_kib_field(&meminfo, "MemAvailable:"),
    )
}

#[cfg(not(target_os = "linux"))]
fn system_memory() -> (Option<u64>, Option<u64>) {
    (None, None)
}

fn parse_kib_field(input: &str, key: &str) -> Option<u64> {
    let line = input.lines().find(|line| line.starts_with(key))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    kib.checked_mul(1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_proc_memory_fields() {
        let input =
            "MemTotal:       16384 kB\nMemAvailable:    4096 kB\nVmRSS:             512 kB\n";
        assert_eq!(parse_kib_field(input, "MemTotal:"), Some(16 * 1024 * 1024));
        assert_eq!(
            parse_kib_field(input, "MemAvailable:"),
            Some(4 * 1024 * 1024)
        );
        assert_eq!(parse_kib_field(input, "VmRSS:"), Some(512 * 1024));
    }

    #[test]
    fn test_memory_report_keeps_managed_memory_separate() {
        let mut budget = MemoryBudget::new(1024).unwrap();
        budget.allocate("test", 256).unwrap();
        let report = MemoryReport::collect(&budget);
        assert_eq!(report.ramforge_current_bytes, 256);
        assert_eq!(report.ramforge_peak_bytes, 256);
        assert_eq!(report.ramforge_budget_bytes, 1024);
        assert_eq!(report.resource_budgets().host_ram_bytes(), Some(1024));
    }

    #[test]
    fn forgecore_report_keeps_logical_native_and_rss_separate() {
        let registry = crate::execution::DeviceRegistry::cpu_only();
        let device = &registry.devices()[0];
        let report = ForgeCoreMemoryReport::new(
            100,
            Some(250),
            240,
            Some(1_000),
            Some(1_500),
            Some(1_800),
            device,
        );
        assert!(!report.enforces_budget());
        assert_eq!(report.logical_model_exceeds_budget(), Some(true));
        assert_eq!(report.rss_delta_after_load_bytes(), Some(500));
        assert_eq!(report.rss_delta_after_decode_bytes(), Some(800));
        assert_eq!(report.logical_model_bytes, Some(250));
        assert_eq!(report.native_model_bytes, 240);
        assert!(report
            .unexposed_categories()
            .iter()
            .any(|item| item.starts_with("KV cache")));
    }

    #[test]
    fn cgroup_attachment_is_the_only_enforcement_claim() {
        let registry = crate::execution::DeviceRegistry::cpu_only();
        let device = &registry.devices()[0];
        let report = ForgeCoreMemoryReport::new(100, Some(250), 240, None, None, None, device);
        assert_eq!(report.enforcement, BudgetEnforcement::DiagnosticOnly);
        assert!(!report.enforces_budget());

        let report = report.with_cgroup(crate::cgroup::CgroupStats {
            limit_bytes: Some(100),
            peak_bytes: Some(90),
            events: Some(crate::cgroup::CgroupEvents {
                max: 1,
                ..Default::default()
            }),
            ..Default::default()
        });
        assert_eq!(report.enforcement, BudgetEnforcement::CgroupV2);
        assert!(report.enforces_budget());
        assert!(report.cgroup_exceeded());
    }

    #[test]
    fn forgecore_report_does_not_invent_missing_facts() {
        let registry = crate::execution::DeviceRegistry::cpu_only();
        let device = &registry.devices()[0];
        let report = ForgeCoreMemoryReport::new(100, None, 240, None, None, None, device);
        assert_eq!(report.logical_model_exceeds_budget(), None);
        assert_eq!(report.rss_delta_after_load_bytes(), None);
        assert_eq!(report.rss_delta_after_decode_bytes(), None);
    }
}
