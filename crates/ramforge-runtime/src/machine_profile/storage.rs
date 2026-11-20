//! Filesystem, backing-device, and bounded read-benchmark observation.
//!
//! Cache state is never inferred from read timing. A file-scoped cache drop is
//! only a request; reports label the result as unverified.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::{Observed, ProfileError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageKind {
    Rotational,
    SolidState,
    Nvme,
    Network,
    MemoryBacked,
    Virtual,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageLocality {
    Local,
    Network,
    MemoryBacked,
    Virtual,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageProfile {
    pub local_path: Option<String>,
    pub mount_point: Observed<String>,
    pub filesystem: Observed<String>,
    pub filesystem_magic: Observed<u64>,
    pub device_major_minor: Observed<String>,
    pub device_name: Observed<String>,
    pub storage_kind: Observed<StorageKind>,
    pub locality: Observed<StorageLocality>,
    pub read_only: Observed<bool>,
    pub capacity_bytes: Observed<u64>,
    pub available_bytes: Observed<u64>,
    pub benchmark: Option<StorageBenchmarkReport>,
}

impl StorageProfile {
    pub fn validate(&self) -> Result<(), ProfileError> {
        self.mount_point.validate()?;
        self.filesystem.validate()?;
        self.filesystem_magic.validate()?;
        self.device_major_minor.validate()?;
        self.device_name.validate()?;
        self.storage_kind.validate()?;
        self.locality.validate()?;
        self.read_only.validate()?;
        self.capacity_bytes.validate()?;
        self.available_bytes.validate()?;
        if let Some(benchmark) = &self.benchmark {
            benchmark.validate()?;
        }
        Ok(())
    }

    /// Remove local path and mount-point strings for export.
    pub fn redacted(&self) -> Self {
        let mut profile = self.clone();
        profile.local_path = None;
        profile.mount_point =
            Observed::unavailable("redacted for export", profile.mount_point.sequence);
        profile
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageBenchmarkConfig {
    pub file_size_bytes: u64,
    pub block_size_bytes: usize,
    pub runs: u32,
}

impl Default for StorageBenchmarkConfig {
    fn default() -> Self {
        Self {
            file_size_bytes: 8 * 1024 * 1024,
            block_size_bytes: 64 * 1024,
            runs: 3,
        }
    }
}

impl StorageBenchmarkConfig {
    pub fn validate(self) -> Result<Self, String> {
        if !(4096..=64 * 1024 * 1024).contains(&self.file_size_bytes) {
            return Err("benchmark file size must be in 4 KiB..=64 MiB".to_string());
        }
        if !(512..=1024 * 1024).contains(&self.block_size_bytes) {
            return Err("benchmark block size must be in 512 B..=1 MiB".to_string());
        }
        if !(1..=10).contains(&self.runs) {
            return Err("benchmark runs must be in 1..=10".to_string());
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageBenchmarkSample {
    pub elapsed_ns: u64,
    pub bytes: u64,
    pub bytes_per_second: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageBenchmarkReport {
    pub file_size_bytes: u64,
    pub block_size_bytes: usize,
    pub runs: u32,
    pub fadvise_request_ok: bool,
    pub cache_condition: String,
    pub physical_claim: bool,
    pub samples: Vec<StorageBenchmarkSample>,
    pub median_bytes_per_second: u64,
    pub min_bytes_per_second: u64,
    pub max_bytes_per_second: u64,
    pub notes: Vec<String>,
}

impl StorageBenchmarkReport {
    fn validate(&self) -> Result<(), ProfileError> {
        if self.samples.is_empty() {
            return Err(ProfileError::Invariant(
                "storage benchmark has no samples".to_string(),
            ));
        }
        if self.physical_claim {
            return Err(ProfileError::Invariant(
                "storage benchmark must not claim physical device performance".to_string(),
            ));
        }
        Ok(())
    }
}

pub fn collect_storage_profile(path: &Path, sequence: u64) -> StorageProfile {
    platform::collect(path, sequence)
}

pub fn run_storage_read_benchmark(
    directory: &Path,
    config: StorageBenchmarkConfig,
) -> Result<StorageBenchmarkReport, String> {
    let config = config.validate()?;
    platform::benchmark(directory, config)
}

#[cfg(unix)]
mod platform {
    use std::ffi::CString;
    use std::fs::{File, OpenOptions};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::os::fd::AsRawFd;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::super::{Freshness, Observed, Provenance};
    use super::*;

    pub(super) fn collect(path: &Path, sequence: u64) -> StorageProfile {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let metadata = std::fs::metadata(&canonical).ok();
        let device = metadata.as_ref().map(|metadata| {
            use std::os::unix::fs::MetadataExt;
            metadata.dev()
        });
        let statvfs = statvfs(&canonical);
        let statfs = statfs(&canonical);
        let mount = mount_for_path(&canonical);
        let source_device = mount
            .as_ref()
            .and_then(|mount| source_device(&mount.source));
        let (major, minor) = match source_device {
            Some((major, minor)) => (Some(major), Some(minor)),
            None => (
                device.map(|device| libc::major(device)),
                device.map(|device| libc::minor(device)),
            ),
        };
        let filesystem = mount
            .as_ref()
            .map(|mount| mount.filesystem.clone())
            .or_else(|| statfs.as_ref().map(|stat| filesystem_name(stat.0)))
            .unwrap_or_else(|| "unknown".to_string());
        let device_name = major
            .zip(minor)
            .and_then(|(major, minor)| sysfs_device_name(major, minor));
        let rotational = major
            .zip(minor)
            .and_then(|(major, minor)| sysfs_rotational(major, minor));
        let kind = classify_storage(
            &filesystem,
            device_name.as_deref(),
            rotational,
            mount
                .as_ref()
                .is_some_and(|mount| mount.super_options.split(',').any(|option| option == "ssd")),
        );
        let locality = classify_locality(&filesystem, kind);
        let os_stable = |value: Option<String>| {
            value
                .map(|value| {
                    Observed::known(
                        value,
                        Provenance::OperatingSystem,
                        Freshness::Stable,
                        sequence,
                    )
                })
                .unwrap_or_else(|| {
                    Observed::unavailable("filesystem metadata unavailable", sequence)
                })
        };
        let os_dynamic = |value: Option<u64>| {
            value
                .map(|value| {
                    Observed::known(
                        value,
                        Provenance::OperatingSystem,
                        Freshness::Dynamic,
                        sequence,
                    )
                })
                .unwrap_or_else(|| {
                    Observed::unavailable("filesystem capacity unavailable", sequence)
                })
        };
        StorageProfile {
            local_path: Some(canonical.to_string_lossy().to_string()),
            mount_point: os_stable(mount.as_ref().map(|mount| mount.mount_point.clone())),
            filesystem: os_stable(Some(filesystem.clone())),
            filesystem_magic: statfs
                .as_ref()
                .and_then(|stat| u64::try_from(stat.0).ok())
                .map(|value| {
                    Observed::known(
                        value,
                        Provenance::OperatingSystem,
                        Freshness::Stable,
                        sequence,
                    )
                })
                .unwrap_or_else(|| Observed::unavailable("statfs magic unavailable", sequence)),
            device_major_minor: major
                .zip(minor)
                .map(|(major, minor)| {
                    Observed::known(
                        format!("{major}:{minor}"),
                        Provenance::OperatingSystem,
                        Freshness::Stable,
                        sequence,
                    )
                })
                .unwrap_or_else(|| Observed::unavailable("device identity unavailable", sequence)),
            device_name: os_stable(device_name),
            storage_kind: Observed::known(
                kind,
                if kind == StorageKind::Unknown {
                    Provenance::NotAvailable
                } else {
                    Provenance::OperatingSystem
                },
                Freshness::Stable,
                sequence,
            ),
            locality: Observed::known(locality, Provenance::Derived, Freshness::Stable, sequence),
            read_only: statvfs
                .as_ref()
                .map(|stat| {
                    Observed::known(
                        stat.read_only,
                        Provenance::OperatingSystem,
                        Freshness::Stable,
                        sequence,
                    )
                })
                .unwrap_or_else(|| Observed::unavailable("read-only state unavailable", sequence)),
            capacity_bytes: os_dynamic(statvfs.as_ref().map(|stat| stat.capacity_bytes)),
            available_bytes: os_dynamic(statvfs.as_ref().map(|stat| stat.available_bytes)),
            benchmark: None,
        }
    }

    struct StatVfs {
        capacity_bytes: u64,
        available_bytes: u64,
        read_only: bool,
    }

    fn statvfs(path: &Path) -> Option<StatVfs> {
        let path = CString::new(path.as_os_str().as_encoded_bytes()).ok()?;
        let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
        // SAFETY: path is NUL-terminated and stat is a valid out pointer.
        if unsafe { libc::statvfs(path.as_ptr(), &mut stat) } != 0 {
            return None;
        }
        let unit = if stat.f_frsize > 0 {
            stat.f_frsize
        } else {
            stat.f_bsize
        };
        Some(StatVfs {
            capacity_bytes: stat.f_blocks.checked_mul(unit)?,
            available_bytes: stat.f_bavail.checked_mul(unit)?,
            read_only: stat.f_flag & libc::ST_RDONLY != 0,
        })
    }

    fn statfs(path: &Path) -> Option<(libc::c_long, String)> {
        let path = CString::new(path.as_os_str().as_encoded_bytes()).ok()?;
        let mut stat: libc::statfs = unsafe { std::mem::zeroed() };
        // SAFETY: path is NUL-terminated and stat is a valid out pointer.
        if unsafe { libc::statfs(path.as_ptr(), &mut stat) } != 0 {
            return None;
        }
        Some((
            stat.f_type as libc::c_long,
            filesystem_name(stat.f_type as libc::c_long),
        ))
    }

    fn filesystem_name(magic: libc::c_long) -> String {
        match magic as u64 {
            0xEF53 => "ext4-or-ext2-ext3".to_string(),
            0x9123_683E => "btrfs".to_string(),
            0x5846_5342 => "xfs".to_string(),
            0x0102_1994 => "tmpfs".to_string(),
            0x8584_58F6 => "ramfs".to_string(),
            0x794C_7630 => "overlayfs".to_string(),
            0x6969 => "nfs".to_string(),
            0xFF53_4D4F => "cifs".to_string(),
            0x6573_5546 => "fuse".to_string(),
            other => format!("unknown-magic-{other:#x}"),
        }
    }

    struct MountInfo {
        mount_point: String,
        filesystem: String,
        source: String,
        super_options: String,
    }

    fn mount_for_path(path: &Path) -> Option<MountInfo> {
        let text = std::fs::read_to_string("/proc/self/mountinfo").ok()?;
        let path = path.to_string_lossy();
        let mut best: Option<(usize, MountInfo)> = None;
        for line in text.lines() {
            let Some((before, after)) = line.split_once(" - ") else {
                continue;
            };
            let before_fields: Vec<&str> = before.split_whitespace().collect();
            let after_fields: Vec<&str> = after.split_whitespace().collect();
            if before_fields.len() < 5 || after_fields.is_empty() {
                continue;
            }
            let mount_point = unescape_mount(before_fields[4]);
            if mount_point == "/"
                || path == mount_point
                || path
                    .strip_prefix(&mount_point)
                    .is_some_and(|rest| rest.starts_with('/'))
            {
                let length = mount_point.len();
                if best.as_ref().is_none_or(|(best, _)| length >= *best) {
                    best = Some((
                        length,
                        MountInfo {
                            mount_point,
                            filesystem: after_fields[0].to_string(),
                            source: after_fields
                                .get(1)
                                .copied()
                                .unwrap_or("unknown")
                                .to_string(),
                            super_options: after_fields.get(2).copied().unwrap_or("").to_string(),
                        },
                    ));
                }
            }
        }
        best.map(|(_, mount)| mount)
    }

    fn unescape_mount(value: &str) -> String {
        value
            .replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\012", "\n")
            .replace("\\134", "\\")
    }

    fn sysfs_device_name(major: libc::c_uint, minor: libc::c_uint) -> Option<String> {
        let link = std::fs::read_link(format!("/sys/dev/block/{major}:{minor}")).ok()?;
        link.file_name()
            .map(|name| name.to_string_lossy().to_string())
    }

    fn sysfs_rotational(major: libc::c_uint, minor: libc::c_uint) -> Option<bool> {
        let link = std::fs::read_link(format!("/sys/dev/block/{major}:{minor}")).ok()?;
        let mut directory: PathBuf =
            PathBuf::from("/sys/dev/block").join(format!("{major}:{minor}"));
        if !directory.exists() {
            directory = link;
        }
        for _ in 0..4 {
            let candidate = directory.join("queue/rotational");
            if let Ok(value) = std::fs::read_to_string(&candidate) {
                return value.trim().parse::<u8>().ok().map(|value| value == 1);
            }
            directory = directory.parent()?.to_path_buf();
        }
        None
    }

    fn source_device(source: &str) -> Option<(libc::c_uint, libc::c_uint)> {
        if !source.starts_with("/dev/") {
            return None;
        }
        use std::os::unix::fs::MetadataExt;
        let metadata = std::fs::metadata(source).ok()?;
        let device = if metadata.rdev() != 0 {
            metadata.rdev()
        } else {
            metadata.dev()
        };
        Some((libc::major(device), libc::minor(device)))
    }

    fn classify_storage(
        filesystem: &str,
        device_name: Option<&str>,
        rotational: Option<bool>,
        super_options_ssd: bool,
    ) -> StorageKind {
        if matches!(filesystem, "tmpfs" | "ramfs") {
            return StorageKind::MemoryBacked;
        }
        if matches!(filesystem, "nfs" | "cifs" | "fuse" | "9p") {
            return StorageKind::Network;
        }
        if filesystem == "overlayfs" {
            return StorageKind::Virtual;
        }
        if device_name.is_some_and(|name| name.starts_with("nvme")) {
            return StorageKind::Nvme;
        }
        match rotational {
            Some(true) => StorageKind::Rotational,
            Some(false) => StorageKind::SolidState,
            None if super_options_ssd => StorageKind::SolidState,
            None => StorageKind::Unknown,
        }
    }

    fn classify_locality(filesystem: &str, kind: StorageKind) -> StorageLocality {
        match kind {
            StorageKind::MemoryBacked => StorageLocality::MemoryBacked,
            StorageKind::Network => StorageLocality::Network,
            StorageKind::Virtual => StorageLocality::Virtual,
            _ if filesystem == "unknown" => StorageLocality::Unknown,
            _ => StorageLocality::Local,
        }
    }

    pub(super) fn benchmark(
        directory: &Path,
        config: StorageBenchmarkConfig,
    ) -> Result<StorageBenchmarkReport, String> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        std::fs::create_dir_all(directory).map_err(|error| error.to_string())?;
        let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = directory.join(format!(
            "forge-storage-bench-{}-{counter}.bin",
            std::process::id()
        ));
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| error.to_string())?;
        let result = (|| {
            let block = vec![0xA5u8; config.block_size_bytes];
            let mut written = 0u64;
            while written < config.file_size_bytes {
                let remaining = config.file_size_bytes - written;
                let take = remaining.min(config.block_size_bytes as u64) as usize;
                file.write_all(&block[..take])
                    .map_err(|error| error.to_string())?;
                written += take as u64;
            }
            file.sync_all().map_err(|error| error.to_string())?;
            let storage = collect(directory, 1);
            let fadvise_ok = request_cache_drop(&file);
            let mut samples = Vec::new();
            let mut buffer = vec![0u8; config.block_size_bytes];
            for _ in 0..config.runs {
                file.seek(SeekFrom::Start(0))
                    .map_err(|error| error.to_string())?;
                if fadvise_ok {
                    let _ = request_cache_drop(&file);
                }
                let started = std::time::Instant::now();
                let mut read = 0u64;
                while read < config.file_size_bytes {
                    let remaining = config.file_size_bytes - read;
                    let take = remaining.min(config.block_size_bytes as u64) as usize;
                    file.read_exact(&mut buffer[..take])
                        .map_err(|error| error.to_string())?;
                    read += take as u64;
                }
                let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                let bytes_per_second = config
                    .file_size_bytes
                    .checked_mul(1_000_000_000)
                    .and_then(|value| value.checked_div(elapsed_ns))
                    .unwrap_or(0);
                samples.push(StorageBenchmarkSample {
                    elapsed_ns,
                    bytes: config.file_size_bytes,
                    bytes_per_second,
                });
            }
            let mut rates: Vec<u64> = samples
                .iter()
                .map(|sample| sample.bytes_per_second)
                .collect();
            rates.sort_unstable();
            let median = rates[rates.len() / 2];
            let (min, max) = (rates[0], rates[rates.len() - 1]);
            let cache_condition = match storage.storage_kind.value {
                Some(StorageKind::MemoryBacked) => "memory-backed-uncontrolled",
                _ if fadvise_ok => "file-cache-drop-requested-unverified",
                _ => "warm-or-unknown",
            }
            .to_string();
            Ok(StorageBenchmarkReport {
                file_size_bytes: config.file_size_bytes,
                block_size_bytes: config.block_size_bytes,
                runs: config.runs,
                fadvise_request_ok: fadvise_ok,
                cache_condition,
                physical_claim: false,
                samples,
                median_bytes_per_second: median,
                min_bytes_per_second: min,
                max_bytes_per_second: max,
                notes: vec![
                    "This is not a physical-device benchmark unless cache state was independently \
                     verified."
                        .to_string(),
                    "The file-scoped cache-drop request is advisory.".to_string(),
                ],
            })
        })();
        let _ = std::fs::remove_file(&path);
        result
    }

    fn request_cache_drop(file: &File) -> bool {
        // File-scoped advisory request only.
        unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) == 0 }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn mountinfo_unescapes_spaces() {
            assert_eq!(unescape_mount("/mnt/with\\040space"), "/mnt/with space");
        }

        #[test]
        fn root_mount_matches_absolute_paths() {
            let mount = mount_for_path(std::path::Path::new("/var/tmp")).expect("root mount");
            assert_eq!(mount.mount_point, "/");
        }

        #[test]
        fn storage_classification_does_not_guess_nvme() {
            assert_eq!(
                classify_storage("unknown-magic-0x1", None, None, false),
                StorageKind::Unknown
            );
            assert_eq!(
                classify_storage("xfs", Some("nvme0n1p1"), Some(false), false),
                StorageKind::Nvme
            );
            assert_eq!(
                classify_storage("ext4-or-ext2-ext3", Some("sda1"), Some(true), false),
                StorageKind::Rotational
            );
            assert_eq!(
                classify_storage("tmpfs", None, None, false),
                StorageKind::MemoryBacked
            );
        }

        #[test]
        fn benchmark_is_bounded_and_labeled() {
            let config = StorageBenchmarkConfig {
                file_size_bytes: 64 * 1024,
                block_size_bytes: 4096,
                runs: 1,
            };
            let directory =
                std::env::temp_dir().join(format!("ramforge-storage-test-{}", std::process::id()));
            let report = benchmark(&directory, config).expect("benchmark");
            assert_eq!(report.samples.len(), 1);
            assert_eq!(report.samples[0].bytes, 64 * 1024);
            assert!(!report.physical_claim);
            assert!(
                report.cache_condition.contains("unverified")
                    || report.cache_condition.contains("uncontrolled")
                    || report.cache_condition.contains("unknown")
            );
            let _ = std::fs::remove_dir_all(directory);
        }

        #[test]
        fn invalid_benchmark_config_is_rejected() {
            assert!(StorageBenchmarkConfig {
                file_size_bytes: 1,
                block_size_bytes: 4096,
                runs: 1,
            }
            .validate()
            .is_err());
            assert!(StorageBenchmarkConfig {
                file_size_bytes: 4096,
                block_size_bytes: 0,
                runs: 1,
            }
            .validate()
            .is_err());
        }
    }
}

#[cfg(not(unix))]
mod platform {
    use std::path::Path;

    use super::super::{Observed, Provenance};
    use super::*;

    pub(super) fn collect(_path: &Path, sequence: u64) -> StorageProfile {
        let unsupported = |name: &str| {
            Observed::unsupported(format!("{name} unsupported on this platform"), sequence)
        };
        StorageProfile {
            local_path: None,
            mount_point: unsupported("mount point"),
            filesystem: unsupported("filesystem"),
            filesystem_magic: unsupported("filesystem magic"),
            device_major_minor: unsupported("device identity"),
            device_name: unsupported("device name"),
            storage_kind: unsupported("storage kind"),
            locality: unsupported("storage locality"),
            read_only: unsupported("read-only state"),
            capacity_bytes: unsupported("storage capacity"),
            available_bytes: unsupported("storage availability"),
            benchmark: None,
        }
    }

    pub(super) fn benchmark(
        _directory: &Path,
        _config: StorageBenchmarkConfig,
    ) -> Result<StorageBenchmarkReport, String> {
        Err("storage benchmarks require a Unix platform".to_string())
    }
}
