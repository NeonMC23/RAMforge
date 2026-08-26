//! Runtime memory visibility.
//!
//! RAMforge-managed bytes, process RSS, and system memory are intentionally
//! reported as separate concepts. MemoryBudget does not control OS page cache.

use ramforge_core::memory::MemoryBudget;

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
}

#[cfg(target_os = "linux")]
fn process_rss() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    parse_kib_field(&status, "VmRSS:")
}

#[cfg(not(target_os = "linux"))]
fn process_rss() -> Option<u64> {
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
        let input = "MemTotal:       16384 kB\nMemAvailable:    4096 kB\nVmRSS:             512 kB\n";
        assert_eq!(parse_kib_field(input, "MemTotal:"), Some(16 * 1024 * 1024));
        assert_eq!(parse_kib_field(input, "MemAvailable:"), Some(4 * 1024 * 1024));
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
    }
}
