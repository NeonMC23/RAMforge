//! RAM, process, cgroup, and pressure observations.
//!
//! These categories remain separate: host capacity, OS availability, process
//! usage, RAMforge budgets, and cgroup limits are not interchangeable.

use serde::{Deserialize, Serialize};

use super::{Observed, ProfileError};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPressure {
    /// `some avg10` in basis points of a percent (100 = 1.00%).
    pub some_avg10: Option<u32>,
    pub some_avg60: Option<u32>,
    pub some_avg300: Option<u32>,
    pub full_avg10: Option<u32>,
    pub full_avg60: Option<u32>,
    pub full_avg300: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CgroupMemoryProfile {
    pub path: Observed<String>,
    pub limit_bytes: Observed<u64>,
    pub current_bytes: Observed<u64>,
    pub peak_bytes: Observed<u64>,
    pub anon_bytes: Observed<u64>,
    pub file_bytes: Observed<u64>,
    pub file_mapped_bytes: Observed<u64>,
    pub shmem_bytes: Observed<u64>,
    pub swap_current_bytes: Observed<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryProfile {
    pub physical_total_bytes: Observed<u64>,
    pub os_available_bytes: Observed<u64>,
    pub swap_total_bytes: Observed<u64>,
    pub swap_free_bytes: Observed<u64>,
    pub process_rss_bytes: Observed<u64>,
    pub process_rss_peak_bytes: Observed<u64>,
    pub process_virtual_bytes: Observed<u64>,
    pub configured_budget_bytes: Observed<u64>,
    pub cgroup: CgroupMemoryProfile,
    pub pressure: Observed<MemoryPressure>,
}

impl MemoryProfile {
    pub fn validate(&self) -> Result<(), ProfileError> {
        self.physical_total_bytes.validate()?;
        self.os_available_bytes.validate()?;
        self.swap_total_bytes.validate()?;
        self.swap_free_bytes.validate()?;
        self.process_rss_bytes.validate()?;
        self.process_rss_peak_bytes.validate()?;
        self.process_virtual_bytes.validate()?;
        self.configured_budget_bytes.validate()?;
        self.pressure.validate()?;
        self.cgroup.path.validate()?;
        self.cgroup.limit_bytes.validate()?;
        self.cgroup.current_bytes.validate()?;
        self.cgroup.peak_bytes.validate()?;
        self.cgroup.anon_bytes.validate()?;
        self.cgroup.file_bytes.validate()?;
        self.cgroup.file_mapped_bytes.validate()?;
        self.cgroup.shmem_bytes.validate()?;
        self.cgroup.swap_current_bytes.validate()?;
        Ok(())
    }
}

pub(crate) fn self_cgroup_path() -> Option<String> {
    platform::self_cgroup_path()
}

pub fn collect_memory_profile(
    configured_budget_bytes: Option<u64>,
    sequence: u64,
) -> MemoryProfile {
    platform::collect(configured_budget_bytes, sequence)
}

#[cfg(target_os = "linux")]
mod platform {
    use super::super::{Freshness, Observed, Provenance};
    use super::*;

    pub(super) fn collect(configured_budget: Option<u64>, sequence: u64) -> MemoryProfile {
        let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let field = |text: &str, name: &str| parse_kib(text, name);
        let os = |value: Option<u64>, detail: &str, dynamic: bool| {
            value
                .map(|value| {
                    Observed::known(
                        value,
                        Provenance::OperatingSystem,
                        if dynamic {
                            Freshness::Dynamic
                        } else {
                            Freshness::Stable
                        },
                        sequence,
                    )
                })
                .unwrap_or_else(|| Observed::unavailable(detail.to_string(), sequence))
        };
        let configured = configured_budget
            .map(|value| {
                Observed::known(value, Provenance::Configured, Freshness::Dynamic, sequence)
            })
            .unwrap_or_else(|| Observed::unavailable("no RAMforge budget configured", sequence));
        let cgroup = collect_cgroup(sequence);
        let pressure = read_pressure(sequence);
        MemoryProfile {
            physical_total_bytes: os(field(&meminfo, "MemTotal:"), "MemTotal unavailable", false),
            os_available_bytes: os(
                field(&meminfo, "MemAvailable:"),
                "MemAvailable unavailable",
                true,
            ),
            swap_total_bytes: os(
                field(&meminfo, "SwapTotal:"),
                "SwapTotal unavailable",
                false,
            ),
            swap_free_bytes: os(field(&meminfo, "SwapFree:"), "SwapFree unavailable", true),
            process_rss_bytes: os(field(&status, "VmRSS:"), "VmRSS unavailable", true),
            process_rss_peak_bytes: os(field(&status, "VmHWM:"), "VmHWM unavailable", true),
            process_virtual_bytes: os(field(&status, "VmSize:"), "VmSize unavailable", true),
            configured_budget_bytes: configured,
            cgroup,
            pressure,
        }
    }

    fn collect_cgroup(sequence: u64) -> CgroupMemoryProfile {
        let path = self_cgroup_path();
        let directory = path
            .as_ref()
            .map(|path| std::path::Path::new("/sys/fs/cgroup").join(path.trim_start_matches('/')));
        let read = |name: &str| {
            directory
                .as_ref()
                .and_then(|directory| std::fs::read_to_string(directory.join(name)).ok())
                .map(|value| value.trim().to_string())
        };
        let observed = |value: Option<String>, detail: &str| {
            value
                .and_then(|value| value.parse::<u64>().ok())
                .map(|value| {
                    Observed::known(
                        value,
                        Provenance::OperatingSystem,
                        Freshness::Dynamic,
                        sequence,
                    )
                })
                .unwrap_or_else(|| Observed::unavailable(detail.to_string(), sequence))
        };
        let stat = read("memory.stat").unwrap_or_default();
        let stat_field = |name: &str| {
            stat.lines()
                .find_map(|line| line.strip_prefix(name)?.trim().parse::<u64>().ok())
        };
        CgroupMemoryProfile {
            path: path
                .map(|path| {
                    Observed::known(
                        path,
                        Provenance::OperatingSystem,
                        Freshness::Stable,
                        sequence,
                    )
                })
                .unwrap_or_else(|| {
                    Observed::unavailable("cgroup v2 self path unavailable", sequence)
                }),
            limit_bytes: observed(read("memory.max"), "cgroup memory.max unavailable"),
            current_bytes: observed(read("memory.current"), "cgroup memory.current unavailable"),
            peak_bytes: observed(read("memory.peak"), "cgroup memory.peak unavailable"),
            anon_bytes: observed(
                stat_field("anon").map(|value| value.to_string()),
                "cgroup anon usage unavailable",
            ),
            file_bytes: observed(
                stat_field("file").map(|value| value.to_string()),
                "cgroup file usage unavailable",
            ),
            file_mapped_bytes: observed(
                stat_field("file_mapped").map(|value| value.to_string()),
                "cgroup file_mapped usage unavailable",
            ),
            shmem_bytes: observed(
                stat_field("shmem").map(|value| value.to_string()),
                "cgroup shmem usage unavailable",
            ),
            swap_current_bytes: observed(
                read("memory.swap.current"),
                "cgroup memory.swap.current unavailable",
            ),
        }
    }

    fn read_pressure(sequence: u64) -> Observed<MemoryPressure> {
        let content = std::fs::read_to_string("/proc/pressure/memory")
            .or_else(|_| std::fs::read_to_string("/sys/fs/cgroup/memory.pressure"))
            .ok();
        let Some(content) = content else {
            return Observed::unavailable("memory pressure interface unavailable", sequence);
        };
        let mut pressure = MemoryPressure {
            some_avg10: None,
            some_avg60: None,
            some_avg300: None,
            full_avg10: None,
            full_avg60: None,
            full_avg300: None,
        };
        for line in content.lines() {
            let mut fields = line.split_whitespace();
            let Some(kind) = fields.next() else { continue };
            for field in fields {
                let Some((name, value)) = field.split_once('=') else {
                    continue;
                };
                let basis_points = value
                    .parse::<f64>()
                    .ok()
                    .and_then(|value| u32::try_from((value * 100.0).round() as u128).ok());
                match (kind, name) {
                    ("some", "avg10") => pressure.some_avg10 = basis_points,
                    ("some", "avg60") => pressure.some_avg60 = basis_points,
                    ("some", "avg300") => pressure.some_avg300 = basis_points,
                    ("full", "avg10") => pressure.full_avg10 = basis_points,
                    ("full", "avg60") => pressure.full_avg60 = basis_points,
                    ("full", "avg300") => pressure.full_avg300 = basis_points,
                    _ => {}
                }
            }
        }
        if pressure.some_avg10.is_none()
            && pressure.full_avg10.is_none()
            && pressure.some_avg300.is_none()
            && pressure.full_avg300.is_none()
        {
            return Observed::unavailable("memory pressure values unavailable", sequence);
        }
        Observed::known(
            pressure,
            Provenance::OperatingSystem,
            Freshness::Dynamic,
            sequence,
        )
    }

    fn parse_kib(text: &str, key: &str) -> Option<u64> {
        let line = text.lines().find(|line| line.starts_with(key))?;
        let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
        kib.checked_mul(1024)
    }

    pub(crate) fn self_cgroup_path() -> Option<String> {
        let text = std::fs::read_to_string("/proc/self/cgroup").ok()?;
        text.lines()
            .find_map(|line| line.strip_prefix("0::"))
            .map(str::to_string)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn meminfo_parser_does_not_invent_values() {
            let text = "MemTotal: 1024 kB\nMemAvailable: 512 kB\n";
            assert_eq!(parse_kib(text, "MemTotal:"), Some(1024 * 1024));
            assert_eq!(parse_kib(text, "MemAvailable:"), Some(512 * 1024));
            assert_eq!(parse_kib(text, "SwapTotal:"), None);
        }

        #[test]
        fn pressure_parser_uses_basis_points() {
            let content = "some avg10=1.25 avg60=0.00 avg300=0.10 total=1\nfull avg10=0.50 avg60=0.00 avg300=0.00 total=0\n";
            let mut pressure = MemoryPressure {
                some_avg10: None,
                some_avg60: None,
                some_avg300: None,
                full_avg10: None,
                full_avg60: None,
                full_avg300: None,
            };
            for line in content.lines() {
                let mut fields = line.split_whitespace();
                let kind = fields.next().unwrap();
                for field in fields {
                    let (name, value) = field.split_once('=').unwrap();
                    let value = (value.parse::<f64>().unwrap() * 100.0).round() as u32;
                    match (kind, name) {
                        ("some", "avg10") => pressure.some_avg10 = Some(value),
                        ("full", "avg10") => pressure.full_avg10 = Some(value),
                        _ => {}
                    }
                }
            }
            assert_eq!(pressure.some_avg10, Some(125));
            assert_eq!(pressure.full_avg10, Some(50));
        }

        #[test]
        fn cgroup_path_parser_handles_namespaced_paths() {
            let text = "0::/user.slice/session.scope\n";
            let path = text
                .lines()
                .find_map(|line| line.strip_prefix("0::"))
                .map(str::to_string);
            assert_eq!(path.as_deref(), Some("/user.slice/session.scope"));
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod platform {
    use super::super::{Observed, Provenance};
    use super::*;

    pub(super) fn collect(configured_budget: Option<u64>, sequence: u64) -> MemoryProfile {
        let unsupported = |name: &str| {
            Observed::unsupported(format!("{name} unsupported on this platform"), sequence)
        };
        let configured = configured_budget
            .map(|value| {
                Observed::known(
                    value,
                    Provenance::Configured,
                    super::super::Freshness::Dynamic,
                    sequence,
                )
            })
            .unwrap_or_else(|| Observed::unavailable("no RAMforge budget configured", sequence));
        MemoryProfile {
            physical_total_bytes: unsupported("physical RAM"),
            os_available_bytes: unsupported("available RAM"),
            swap_total_bytes: unsupported("swap"),
            swap_free_bytes: unsupported("swap availability"),
            process_rss_bytes: unsupported("process RSS"),
            process_rss_peak_bytes: unsupported("process peak RSS"),
            process_virtual_bytes: unsupported("process virtual memory"),
            configured_budget_bytes: configured,
            cgroup: CgroupMemoryProfile {
                path: unsupported("cgroup path"),
                limit_bytes: unsupported("cgroup limit"),
                current_bytes: unsupported("cgroup usage"),
                peak_bytes: unsupported("cgroup peak"),
                anon_bytes: unsupported("cgroup anon"),
                file_bytes: unsupported("cgroup file"),
                file_mapped_bytes: unsupported("cgroup mapped"),
                shmem_bytes: unsupported("cgroup shmem"),
                swap_current_bytes: unsupported("cgroup swap"),
            },
            pressure: unsupported("memory pressure"),
        }
    }
}
