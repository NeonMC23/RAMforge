//! CPU capability and topology observation.
//!
//! Linux values come from `/proc`, `/sys`, and standard process APIs. Missing
//! fields remain explicit; logical CPUs are never treated as compute capacity.

use serde::{Deserialize, Serialize};

use super::{Observed, ProfileError};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuRuntimeSettings {
    pub selected_threads: Option<u32>,
    pub user_thread_limit: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuSimdCapabilities {
    pub sse2: Observed<bool>,
    pub sse4_2: Observed<bool>,
    pub avx: Observed<bool>,
    pub avx2: Observed<bool>,
    pub avx512f: Observed<bool>,
    pub fma: Observed<bool>,
    pub neon: Observed<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuCache {
    pub level: u32,
    pub kind: String,
    pub size_bytes: u64,
    pub shared_cpu_list: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuFrequency {
    pub min_mhz: Option<u64>,
    pub max_mhz: Option<u64>,
    pub current_mhz: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuQuota {
    pub quota_us: u64,
    pub period_us: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuProfile {
    pub architecture: Observed<String>,
    pub vendor: Observed<String>,
    pub model: Observed<String>,
    pub hardware_logical_cpus: Observed<u32>,
    pub hardware_physical_cores: Observed<u32>,
    pub available_threads: Observed<u32>,
    pub affinity_cpus: Observed<Vec<u32>>,
    pub user_thread_limit: Observed<u32>,
    pub selected_threads: Observed<u32>,
    pub features: Observed<Vec<String>>,
    pub simd: CpuSimdCapabilities,
    pub numa_nodes: Observed<Vec<u32>>,
    pub caches: Observed<Vec<CpuCache>>,
    pub frequency: Observed<CpuFrequency>,
    pub cgroup_cpu_quota: Observed<CpuQuota>,
}

impl CpuProfile {
    pub fn validate(&self) -> Result<(), ProfileError> {
        self.architecture.validate()?;
        self.vendor.validate()?;
        self.model.validate()?;
        self.hardware_logical_cpus.validate()?;
        self.hardware_physical_cores.validate()?;
        self.available_threads.validate()?;
        self.affinity_cpus.validate()?;
        self.user_thread_limit.validate()?;
        self.selected_threads.validate()?;
        self.features.validate()?;
        self.simd.sse2.validate()?;
        self.simd.sse4_2.validate()?;
        self.simd.avx.validate()?;
        self.simd.avx2.validate()?;
        self.simd.avx512f.validate()?;
        self.simd.fma.validate()?;
        self.simd.neon.validate()?;
        self.numa_nodes.validate()?;
        self.caches.validate()?;
        self.frequency.validate()?;
        self.cgroup_cpu_quota.validate()?;
        if let Some(threads) = self.selected_threads.as_ref() {
            if *threads == 0 {
                return Err(ProfileError::Invariant(
                    "selected CPU threads must be greater than zero".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
struct CpuInfo {
    vendor: Option<String>,
    model: Option<String>,
    features: Vec<String>,
    processors: u32,
    physical_cores: Option<u32>,
    mhz: Option<u64>,
}

pub fn collect_cpu_profile(settings: &CpuRuntimeSettings, sequence: u64) -> CpuProfile {
    platform::collect(settings, sequence)
}

#[cfg(target_os = "linux")]
mod platform {
    use std::collections::BTreeSet;

    use super::super::{Freshness, Observed, Provenance};
    use super::*;

    pub(super) fn collect(settings: &CpuRuntimeSettings, sequence: u64) -> CpuProfile {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo")
            .map(|text| parse_cpuinfo(&text))
            .unwrap_or_default();
        let architecture = Observed::known(
            std::env::consts::ARCH.to_string(),
            Provenance::OperatingSystem,
            Freshness::Stable,
            sequence,
        );
        let vendor = cpuinfo
            .vendor
            .clone()
            .map(|value| known_stable(value, sequence))
            .unwrap_or_else(|| Observed::unavailable("vendor not reported", sequence));
        let model = cpuinfo
            .model
            .clone()
            .map(|value| known_stable(value, sequence))
            .unwrap_or_else(|| Observed::unavailable("model not reported", sequence));
        let hardware_logical = if cpuinfo.processors > 0 {
            known_stable(cpuinfo.processors, sequence)
        } else {
            Observed::unavailable("processor count not reported", sequence)
        };
        let physical = cpuinfo
            .physical_cores
            .map(|value| known_stable(value, sequence))
            .unwrap_or_else(|| {
                Observed::unavailable("physical core topology not reported", sequence)
            });
        let available = std::thread::available_parallelism()
            .ok()
            .and_then(|value| u32::try_from(value.get()).ok())
            .map(|value| {
                Observed::known(
                    value,
                    Provenance::OperatingSystem,
                    Freshness::Dynamic,
                    sequence,
                )
            })
            .unwrap_or_else(|| {
                Observed::unavailable("available parallelism unavailable", sequence)
            });
        let affinity = parse_cpu_list_field(
            &std::fs::read_to_string("/proc/self/status").unwrap_or_default(),
            "Cpus_allowed_list:",
        )
        .map(|value| {
            Observed::known(
                value,
                Provenance::OperatingSystem,
                Freshness::Dynamic,
                sequence,
            )
        })
        .unwrap_or_else(|| Observed::unavailable("CPU affinity list unavailable", sequence));
        let features = if cpuinfo.features.is_empty() {
            Observed::unavailable("CPU feature flags not reported", sequence)
        } else {
            let mut features = cpuinfo.features.clone();
            features.sort();
            features.dedup();
            Observed::known(
                features,
                Provenance::OperatingSystem,
                Freshness::Stable,
                sequence,
            )
        };
        let simd = simd_capabilities(&cpuinfo.features, sequence);
        let numa_nodes = read_cpu_list("/sys/devices/system/node/online")
            .map(|value| known_stable(value, sequence))
            .unwrap_or_else(|| Observed::unavailable("NUMA topology unavailable", sequence));
        let caches = read_caches(sequence);
        let frequency = read_frequency(&cpuinfo, sequence);
        let cgroup_cpu_quota = read_cgroup_quota(sequence);
        let selected_threads = settings
            .selected_threads
            .map(|value| {
                Observed::known(value, Provenance::Configured, Freshness::Dynamic, sequence)
            })
            .unwrap_or_else(|| {
                Observed::unavailable("no runtime thread selection recorded", sequence)
            });
        let user_thread_limit = settings
            .user_thread_limit
            .map(|value| {
                Observed::known(value, Provenance::Configured, Freshness::Dynamic, sequence)
            })
            .unwrap_or_else(|| Observed::unavailable("no user thread limit configured", sequence));
        CpuProfile {
            architecture,
            vendor,
            model,
            hardware_logical_cpus: hardware_logical,
            hardware_physical_cores: physical,
            available_threads: available,
            affinity_cpus: affinity,
            user_thread_limit,
            selected_threads,
            features,
            simd,
            numa_nodes,
            caches,
            frequency,
            cgroup_cpu_quota,
        }
    }

    fn known_stable<T>(value: T, sequence: u64) -> Observed<T> {
        Observed::known(
            value,
            Provenance::OperatingSystem,
            Freshness::Stable,
            sequence,
        )
    }

    fn simd_capabilities(features: &[String], sequence: u64) -> CpuSimdCapabilities {
        let has = |name: &str| {
            if features.is_empty() {
                Observed::unavailable("feature flags unavailable", sequence)
            } else {
                Observed::known(
                    features.iter().any(|feature| feature == name),
                    Provenance::OperatingSystem,
                    Freshness::Stable,
                    sequence,
                )
            }
        };
        CpuSimdCapabilities {
            sse2: has("sse2"),
            sse4_2: has("sse4_2"),
            avx: has("avx"),
            avx2: has("avx2"),
            avx512f: has("avx512f"),
            fma: has("fma"),
            neon: {
                #[cfg(target_arch = "aarch64")]
                {
                    Observed::known(
                        cfg!(target_feature = "neon"),
                        Provenance::OperatingSystem,
                        Freshness::Stable,
                        sequence,
                    )
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    Observed::known(
                        false,
                        Provenance::OperatingSystem,
                        Freshness::Stable,
                        sequence,
                    )
                }
            },
        }
    }

    fn read_caches(sequence: u64) -> Observed<Vec<CpuCache>> {
        let base = "/sys/devices/system/cpu/cpu0/cache";
        let Ok(entries) = std::fs::read_dir(base) else {
            return Observed::unavailable("CPU cache sysfs unavailable", sequence);
        };
        let mut caches = Vec::new();
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with("index") {
                continue;
            }
            let path = entry.path();
            let level = read_trim(path.join("level")).and_then(|value| value.parse().ok());
            let kind = read_trim(path.join("type"));
            let size = read_trim(path.join("size")).and_then(|value| parse_size_bytes(&value));
            if let (Some(level), Some(kind), Some(size)) = (level, kind, size) {
                caches.push(CpuCache {
                    level,
                    kind,
                    size_bytes: size,
                    shared_cpu_list: read_trim(path.join("shared_cpu_list")),
                });
            }
        }
        if caches.is_empty() {
            Observed::unavailable("CPU cache details unavailable", sequence)
        } else {
            caches.sort_by_key(|cache| (cache.level, cache.kind.clone()));
            known_stable(caches, sequence)
        }
    }

    fn read_frequency(cpuinfo: &CpuInfo, sequence: u64) -> Observed<CpuFrequency> {
        let base = "/sys/devices/system/cpu/cpu0/cpufreq";
        let min_mhz = read_trim(format!("{base}/cpuinfo_min_freq"))
            .and_then(|value| value.parse::<u64>().ok())
            .map(|khz| khz / 1000);
        let max_mhz = read_trim(format!("{base}/cpuinfo_max_freq"))
            .and_then(|value| value.parse::<u64>().ok())
            .map(|khz| khz / 1000);
        let current_mhz = read_trim(format!("{base}/scaling_cur_freq"))
            .and_then(|value| value.parse::<u64>().ok())
            .map(|khz| khz / 1000)
            .or(cpuinfo.mhz);
        if min_mhz.is_none() && max_mhz.is_none() && current_mhz.is_none() {
            Observed::unavailable("CPU frequency details unavailable", sequence)
        } else {
            Observed::known(
                CpuFrequency {
                    min_mhz,
                    max_mhz,
                    current_mhz,
                },
                Provenance::OperatingSystem,
                Freshness::Dynamic,
                sequence,
            )
        }
    }

    fn read_cgroup_quota(sequence: u64) -> Observed<CpuQuota> {
        let Some(path) = super::super::memory::self_cgroup_path() else {
            return Observed::unavailable("cgroup v2 path unavailable", sequence);
        };
        let content = std::fs::read_to_string(
            std::path::Path::new("/sys/fs/cgroup")
                .join(path.trim_start_matches('/'))
                .join("cpu.max"),
        )
        .ok();
        let Some(content) = content else {
            return Observed::unavailable("cgroup CPU quota unavailable", sequence);
        };
        let mut fields = content.split_whitespace();
        let quota = fields.next().unwrap_or("max");
        let period = fields.next().and_then(|value| value.parse::<u64>().ok());
        if quota == "max" {
            return Observed::unavailable("cgroup CPU quota is unlimited", sequence);
        }
        let quota = quota.parse::<u64>().ok();
        match (quota, period) {
            (Some(quota_us), Some(period_us)) if period_us > 0 => Observed::known(
                CpuQuota {
                    quota_us,
                    period_us,
                },
                Provenance::OperatingSystem,
                Freshness::Dynamic,
                sequence,
            ),
            _ => Observed::unavailable("cgroup CPU quota is malformed or unlimited", sequence),
        }
    }

    fn read_trim(path: impl AsRef<std::path::Path>) -> Option<String> {
        std::fs::read_to_string(path)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    }

    fn read_cpu_list(path: &str) -> Option<Vec<u32>> {
        let text = std::fs::read_to_string(path).ok()?;
        parse_cpu_list(&text)
    }

    fn parse_cpu_list(text: &str) -> Option<Vec<u32>> {
        let mut out = Vec::new();
        for part in text.trim().split(',') {
            if part.is_empty() {
                continue;
            }
            if let Some((start, end)) = part.split_once('-') {
                let start: u32 = start.parse().ok()?;
                let end: u32 = end.parse().ok()?;
                if end < start || end.saturating_sub(start) > 4096 {
                    return None;
                }
                out.extend(start..=end);
            } else {
                out.push(part.parse().ok()?);
            }
        }
        Some(out)
    }

    fn parse_cpu_list_field(text: &str, field: &str) -> Option<Vec<u32>> {
        let value = text
            .lines()
            .find(|line| line.starts_with(field))?
            .split_once(':')?
            .1
            .trim();
        parse_cpu_list(value)
    }

    fn parse_size_bytes(value: &str) -> Option<u64> {
        let value = value.trim();
        let number: u64 = value
            .trim_end_matches(|character: char| character.is_ascii_alphabetic())
            .parse()
            .ok()?;
        match value.chars().last()?.to_ascii_uppercase() {
            'K' => number.checked_mul(1024),
            'M' => number.checked_mul(1024 * 1024),
            'G' => number.checked_mul(1024 * 1024 * 1024),
            _ => Some(number),
        }
    }

    fn parse_cpuinfo(text: &str) -> CpuInfo {
        let mut info = CpuInfo::default();
        let mut vendors = BTreeSet::new();
        let mut models = BTreeSet::new();
        let mut cores = BTreeSet::new();
        for block in text.split("\n\n") {
            let mut physical_id = None;
            let mut core_id = None;
            for line in block.lines() {
                let Some((key, value)) = line.split_once(':') else {
                    continue;
                };
                let key = key.trim();
                let value = value.trim();
                match key {
                    "processor" => info.processors = info.processors.saturating_add(1),
                    "vendor_id" => {
                        vendors.insert(value.to_string());
                    }
                    "model name" => {
                        models.insert(value.to_string());
                    }
                    "flags" | "Features" => {
                        info.features = value.split_whitespace().map(str::to_string).collect();
                    }
                    "physical id" => physical_id = value.parse::<u32>().ok(),
                    "core id" => core_id = value.parse::<u32>().ok(),
                    "cpu MHz" => {
                        info.mhz = value
                            .parse::<f64>()
                            .ok()
                            .and_then(|mhz| u64::try_from(mhz.round() as u128).ok())
                            .or(info.mhz)
                    }
                    _ => {}
                }
            }
            if let (Some(physical), Some(core)) = (physical_id, core_id) {
                cores.insert((physical, core));
            }
        }
        info.vendor = vendors.into_iter().next();
        info.model = models.into_iter().next();
        if !cores.is_empty() {
            info.physical_cores = u32::try_from(cores.len()).ok();
        }
        info
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn cpuinfo_parser_keeps_unknown_fields_unknown() {
            let info =
                parse_cpuinfo("processor : 0\nvendor_id : TestVendor\nmodel name : Test CPU\n");
            assert_eq!(info.processors, 1);
            assert_eq!(info.vendor.as_deref(), Some("TestVendor"));
            assert!(info.features.is_empty());
            assert_eq!(info.physical_cores, None);
        }

        #[test]
        fn cpuinfo_parser_counts_unique_physical_cores() {
            let text = "processor : 0\nphysical id : 0\ncore id : 0\n\nprocessor : 1\nphysical id : 0\ncore id : 1\n\nprocessor : 2\nphysical id : 0\ncore id : 0\n";
            let info = parse_cpuinfo(text);
            assert_eq!(info.processors, 3);
            assert_eq!(info.physical_cores, Some(2));
        }

        #[test]
        fn cpu_list_parser_handles_ranges_and_single_cpus() {
            assert_eq!(
                parse_cpu_list("0-3,8,10-11"),
                Some(vec![0, 1, 2, 3, 8, 10, 11])
            );
            assert_eq!(parse_cpu_list("garbage"), None);
        }

        #[test]
        fn cache_size_parser_is_explicit() {
            assert_eq!(parse_size_bytes("32K"), Some(32 * 1024));
            assert_eq!(parse_size_bytes("2M"), Some(2 * 1024 * 1024));
            assert_eq!(parse_size_bytes("512"), Some(512));
            assert_eq!(parse_size_bytes("bad"), None);
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod platform {
    use super::super::{Freshness, Observed, Provenance};
    use super::*;

    pub(super) fn collect(settings: &CpuRuntimeSettings, sequence: u64) -> CpuProfile {
        let architecture = Observed::known(
            std::env::consts::ARCH.to_string(),
            Provenance::OperatingSystem,
            Freshness::Stable,
            sequence,
        );
        let available = std::thread::available_parallelism()
            .ok()
            .and_then(|value| u32::try_from(value.get()).ok())
            .map(|value| {
                Observed::known(
                    value,
                    Provenance::OperatingSystem,
                    Freshness::Dynamic,
                    sequence,
                )
            })
            .unwrap_or_else(|| {
                Observed::unavailable("available parallelism unavailable", sequence)
            });
        let unsupported = |name: &str| {
            Observed::unsupported(format!("{name} unsupported on this platform"), sequence)
        };
        let configured = |value: Option<u32>, detail: &str| {
            value
                .map(|value| {
                    Observed::known(value, Provenance::Configured, Freshness::Dynamic, sequence)
                })
                .unwrap_or_else(|| Observed::unavailable(detail.to_string(), sequence))
        };
        CpuProfile {
            architecture,
            vendor: unsupported("CPU vendor"),
            model: unsupported("CPU model"),
            hardware_logical_cpus: unsupported("hardware logical CPU count"),
            hardware_physical_cores: unsupported("physical core count"),
            available_threads: available,
            affinity_cpus: unsupported("CPU affinity"),
            user_thread_limit: configured(
                settings.user_thread_limit,
                "no user thread limit configured",
            ),
            selected_threads: configured(settings.selected_threads, "no runtime thread selection"),
            features: unsupported("CPU feature flags"),
            simd: CpuSimdCapabilities {
                sse2: unsupported("sse2"),
                sse4_2: unsupported("sse4.2"),
                avx: unsupported("avx"),
                avx2: unsupported("avx2"),
                avx512f: unsupported("avx512f"),
                fma: unsupported("fma"),
                neon: unsupported("neon"),
            },
            numa_nodes: unsupported("NUMA topology"),
            caches: unsupported("CPU caches"),
            frequency: unsupported("CPU frequency"),
            cgroup_cpu_quota: unsupported("cgroup CPU quota"),
        }
    }
}
