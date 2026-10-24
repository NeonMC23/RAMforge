//! Runtime memory visibility.
//!
//! RAMforge-managed bytes, process RSS, and system memory are intentionally
//! reported as separate concepts. MemoryBudget does not control OS page cache.

use ramforge_core::{
    memory::MemoryBudget,
    resource::{ByteMeasurement, ResourceBudgets},
};

use crate::execution::DeviceInfo;

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

/// Memory facts for the ForgeCore/libllama execution path.
///
/// This report deliberately separates facts that are often conflated:
///
/// - `logical_model_bytes` is the sum of GGUF tensor payload lengths read by
///   RAMforge's inspector. It is a file/logical size, not an allocation.
/// - `native_model_bytes` is libllama's own `size_bytes()` accounting.
/// - RSS values are measured process facts from `/proc/self/status`; they
///   include the loader, page cache effects, Rust heap, and everything else in
///   the process. They are not an attribution of model residency.
/// - device totals/free bytes are native snapshots exposed by forgeCORE.
///
/// The pinned llama.cpp C API does not expose KV-cache, graph-scratch, staging,
/// or mapped-page residency bytes. Those fields are therefore absent rather
/// than estimated. The requested RAM budget is recorded but is **not**
/// enforced on this backend: libllama loads the model eagerly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgeCoreMemoryReport {
    pub requested_budget_bytes: u64,
    pub logical_model_bytes: Option<u64>,
    pub native_model_bytes: u64,
    pub rss_before_load_bytes: Option<u64>,
    pub rss_after_load_bytes: Option<u64>,
    pub rss_after_decode_bytes: Option<u64>,
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
            logical_model_bytes,
            native_model_bytes,
            rss_before_load_bytes,
            rss_after_load_bytes,
            rss_after_decode_bytes,
            system_total_bytes,
            system_available_bytes,
            device_memory_total: device.memory_total.clone(),
            device_memory_free: device.memory_free.clone(),
        }
    }

    /// The ForgeCore backend does not enforce RAMforge's RAM budget.
    pub const fn enforces_budget(&self) -> bool {
        false
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
            "KV cache",
            "graph scratch/workspace",
            "staging buffers",
            "mapped-page residency",
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
        assert!(report.unexposed_categories().contains(&"KV cache"));
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
