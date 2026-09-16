//! Read-only local machine and storage discovery.
//!
//! Discovery reports facts only. Capability interpretation remains in the
//! Planner, and performance measurement remains in calibration.

use std::collections::BTreeSet;
use std::fmt;
use std::fs::{self, File, Metadata};
use std::io::{self, Seek};
use std::path::{Path, PathBuf};

use crate::planner::{
    Availability, CpuFeature, DiscoveryState, GpuDeviceProfile, MachineProfile, StorageKind,
    StoragePathState, StorageProfile, PROFILE_SCHEMA_VERSION,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MachineDiscoveryError {
    LogicalCpuUnavailable,
    InvalidDetectedFacts,
}

impl fmt::Display for MachineDiscoveryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LogicalCpuUnavailable => write!(formatter, "logical CPU count is unavailable"),
            Self::InvalidDetectedFacts => write!(formatter, "detected machine facts are invalid"),
        }
    }
}

impl std::error::Error for MachineDiscoveryError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageDiscoveryError {
    InvalidPath,
    PermissionDenied,
    NotRegularFile,
    Unavailable,
}

impl fmt::Display for StorageDiscoveryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPath => write!(formatter, "storage path is invalid or unavailable"),
            Self::PermissionDenied => write!(formatter, "storage path permission denied"),
            Self::NotRegularFile => write!(formatter, "storage path is not a regular file"),
            Self::Unavailable => write!(formatter, "storage information is unavailable"),
        }
    }
}

impl std::error::Error for StorageDiscoveryError {}

#[derive(Debug, Default, Clone, Copy)]
pub struct MachineDiscovery;

impl MachineDiscovery {
    pub fn discover(&self) -> Result<MachineProfile, MachineDiscoveryError> {
        let logical_cpu_cores = std::thread::available_parallelism()
            .map(|count| count.get())
            .map_err(|_| MachineDiscoveryError::LogicalCpuUnavailable)?;
        let cpu_info = discover_cpu_info();
        let (total_ram_bytes, available_ram_bytes) = discover_memory();
        let (gpu_inventory_state, gpus) = discover_gpus();
        let facts = MachineFacts {
            os: std::env::consts::OS.to_string(),
            architecture: std::env::consts::ARCH.to_string(),
            kernel_version: discover_kernel_version(),
            cpu_vendor: cpu_info.vendor,
            cpu_model: cpu_info.model,
            physical_cpu_cores: cpu_info.physical_cores,
            logical_cpu_cores,
            cpu_features: discover_cpu_features(),
            total_ram_bytes,
            available_ram_bytes,
            gpu_inventory_state,
            gpus,
        };
        machine_profile_from_facts(facts)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct StorageDiscovery;

impl StorageDiscovery {
    pub fn discover(
        &self,
        model_path: impl AsRef<Path>,
    ) -> Result<StorageProfile, StorageDiscoveryError> {
        let path = model_path.as_ref();
        if path.as_os_str().is_empty() {
            return Err(StorageDiscoveryError::InvalidPath);
        }
        let canonical = fs::canonicalize(path).map_err(map_storage_error)?;
        let metadata = fs::metadata(&canonical).map_err(map_storage_error)?;
        if !metadata.is_file() {
            return Err(StorageDiscoveryError::NotRegularFile);
        }
        let mut file = File::open(&canonical).map_err(map_storage_error)?;
        let seekable = file.stream_position().is_ok();
        let details = discover_storage_details(&canonical, &metadata);
        let profile = StorageProfile {
            schema_version: PROFILE_SCHEMA_VERSION,
            model_path: canonical,
            path_state: StoragePathState::Ready,
            readable: true,
            regular_file: true,
            file_size_bytes: Some(metadata.len()),
            filesystem_type: details.filesystem_type,
            filesystem_id: details.filesystem_id,
            device_id: details.device_id,
            kind: details.kind,
            seekable,
        };
        profile
            .validate()
            .map_err(|_| StorageDiscoveryError::Unavailable)?;
        Ok(profile)
    }
}

#[derive(Debug)]
struct MachineFacts {
    os: String,
    architecture: String,
    kernel_version: Option<String>,
    cpu_vendor: Option<String>,
    cpu_model: Option<String>,
    physical_cpu_cores: Option<usize>,
    logical_cpu_cores: usize,
    cpu_features: BTreeSet<CpuFeature>,
    total_ram_bytes: Option<u64>,
    available_ram_bytes: Option<u64>,
    gpu_inventory_state: DiscoveryState,
    gpus: Vec<GpuDeviceProfile>,
}

fn machine_profile_from_facts(
    facts: MachineFacts,
) -> Result<MachineProfile, MachineDiscoveryError> {
    let profile = MachineProfile {
        schema_version: PROFILE_SCHEMA_VERSION,
        os: facts.os,
        architecture: facts.architecture,
        kernel_version: facts.kernel_version,
        cpu_vendor: facts.cpu_vendor,
        cpu_model: facts.cpu_model,
        physical_cpu_cores: facts.physical_cpu_cores,
        logical_cpu_cores: facts.logical_cpu_cores,
        cpu_features: facts.cpu_features,
        total_ram_bytes: facts.total_ram_bytes,
        available_ram_bytes: facts.available_ram_bytes,
        cpu_backend: Availability {
            detected: true,
            supported: true,
            usable: true,
        },
        gpu_inventory_state: facts.gpu_inventory_state,
        gpus: facts.gpus,
    };
    profile
        .validate()
        .map_err(|_| MachineDiscoveryError::InvalidDetectedFacts)?;
    Ok(profile)
}

#[derive(Debug, Default)]
struct CpuInfo {
    vendor: Option<String>,
    model: Option<String>,
    physical_cores: Option<usize>,
}

fn discover_cpu_features() -> BTreeSet<CpuFeature> {
    let mut features = BTreeSet::new();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx2") {
        features.insert(CpuFeature::Avx2);
    }
    features
}

#[cfg(target_os = "linux")]
fn discover_cpu_info() -> CpuInfo {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .map(|contents| parse_linux_cpuinfo(&contents))
        .unwrap_or_default()
}

#[cfg(not(target_os = "linux"))]
fn discover_cpu_info() -> CpuInfo {
    CpuInfo::default()
}

#[cfg(any(target_os = "linux", test))]
fn parse_linux_cpuinfo(contents: &str) -> CpuInfo {
    let mut vendor = None;
    let mut model = None;
    let mut physical_cores = BTreeSet::new();
    let mut processor_sections = 0usize;
    let mut complete_topology = true;

    for section in contents.split("\n\n") {
        let mut processor = false;
        let mut physical_id = None;
        let mut core_id = None;
        for line in section.lines() {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let key = key.trim();
            let value = value.trim();
            match key {
                "processor" => processor = true,
                "physical id" => physical_id = value.parse::<u32>().ok(),
                "core id" => core_id = value.parse::<u32>().ok(),
                "vendor_id" | "CPU implementer" if vendor.is_none() && !value.is_empty() => {
                    vendor = Some(value.to_string())
                }
                "model name" | "Hardware" if model.is_none() && !value.is_empty() => {
                    model = Some(value.to_string())
                }
                _ => {}
            }
        }
        if processor {
            processor_sections += 1;
            if let (Some(package), Some(core)) = (physical_id, core_id) {
                physical_cores.insert((package, core));
            } else {
                complete_topology = false;
            }
        }
    }

    CpuInfo {
        vendor,
        model,
        physical_cores: (processor_sections > 0 && complete_topology)
            .then_some(physical_cores.len())
            .filter(|count| *count > 0),
    }
}

#[cfg(target_os = "linux")]
fn discover_memory() -> (Option<u64>, Option<u64>) {
    let Some(contents) = fs::read_to_string("/proc/meminfo").ok() else {
        return (None, None);
    };
    parse_linux_meminfo(&contents)
}

#[cfg(not(target_os = "linux"))]
fn discover_memory() -> (Option<u64>, Option<u64>) {
    (None, None)
}

#[cfg(any(target_os = "linux", test))]
fn parse_linux_meminfo(contents: &str) -> (Option<u64>, Option<u64>) {
    let mut total = None;
    let mut available = None;
    for line in contents.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let Some(kibibytes) = rest
            .split_whitespace()
            .next()
            .and_then(|value| value.parse::<u64>().ok())
        else {
            continue;
        };
        let bytes = kibibytes.checked_mul(1024);
        match key {
            "MemTotal" => total = bytes,
            "MemAvailable" => available = bytes,
            _ => {}
        }
    }
    let Some(total_bytes) = total else {
        return (None, None);
    };
    (
        Some(total_bytes),
        available.filter(|available_bytes| *available_bytes <= total_bytes),
    )
}

#[cfg(target_os = "linux")]
fn discover_kernel_version() -> Option<String> {
    read_trimmed("/proc/sys/kernel/osrelease")
}

#[cfg(not(target_os = "linux"))]
fn discover_kernel_version() -> Option<String> {
    None
}

#[cfg(target_os = "linux")]
fn discover_gpus() -> (DiscoveryState, Vec<GpuDeviceProfile>) {
    let entries = match fs::read_dir("/sys/class/drm") {
        Ok(entries) => entries,
        Err(_) => return (DiscoveryState::Unavailable, Vec::new()),
    };
    let mut state = DiscoveryState::Complete;
    let mut gpus = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => {
                state = DiscoveryState::Partial;
                continue;
            }
        };
        let card = entry.file_name().to_string_lossy().into_owned();
        if !is_drm_card_name(&card) {
            continue;
        }
        let device_path = entry.path().join("device");
        if !device_path.exists() {
            state = DiscoveryState::Partial;
            continue;
        }
        let canonical_device = fs::canonicalize(&device_path).ok();
        let slot = canonical_device
            .as_deref()
            .and_then(|path| path.file_name())
            .map(|value| value.to_string_lossy().into_owned());
        let vendor_id = read_trimmed(device_path.join("vendor"));
        let device_id = read_trimmed(device_path.join("device"));
        if vendor_id.is_none() || device_id.is_none() {
            state = DiscoveryState::Partial;
        }
        let vendor = vendor_id.as_deref().and_then(gpu_vendor_name);
        let name = read_trimmed(device_path.join("product_name"));
        let dedicated_memory_bytes = read_trimmed(device_path.join("mem_info_vram_total"))
            .and_then(|value| value.parse::<u64>().ok());
        let identifier = format!(
            "linux-drm:{}:{}:{}",
            slot.as_deref().unwrap_or(&card),
            vendor_id.as_deref().unwrap_or("unknown-vendor"),
            device_id.as_deref().unwrap_or("unknown-device")
        );
        if gpus
            .iter()
            .any(|gpu: &GpuDeviceProfile| gpu.identifier.as_str() == identifier.as_str())
        {
            continue;
        }
        gpus.push(GpuDeviceProfile {
            identifier,
            vendor,
            device_id,
            name,
            dedicated_memory_bytes,
            backend: Availability {
                detected: true,
                supported: false,
                usable: false,
            },
        });
    }
    gpus.sort_by(|left, right| left.identifier.cmp(&right.identifier));
    (state, gpus)
}

#[cfg(not(target_os = "linux"))]
fn discover_gpus() -> (DiscoveryState, Vec<GpuDeviceProfile>) {
    (DiscoveryState::Unavailable, Vec::new())
}

#[cfg(any(target_os = "linux", test))]
fn is_drm_card_name(name: &str) -> bool {
    name.strip_prefix("card").is_some_and(|suffix| {
        !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
    })
}

#[cfg(any(target_os = "linux", test))]
fn gpu_vendor_name(identifier: &str) -> Option<String> {
    match identifier.trim().to_ascii_lowercase().as_str() {
        "0x10de" => Some("NVIDIA".to_string()),
        "0x1002" => Some("AMD".to_string()),
        "0x8086" => Some("Intel".to_string()),
        _ => None,
    }
}

#[derive(Debug)]
struct StorageDetails {
    filesystem_type: Option<String>,
    filesystem_id: Option<String>,
    device_id: Option<String>,
    kind: StorageKind,
}

#[cfg(target_os = "linux")]
fn discover_storage_details(path: &Path, metadata: &Metadata) -> StorageDetails {
    use std::os::unix::fs::MetadataExt;

    let fallback_device_id = Some(format!("linux-dev:{:x}", metadata.dev()));
    let mount = fs::read_to_string("/proc/self/mountinfo")
        .ok()
        .and_then(|contents| find_linux_mount(&contents, path));
    let Some(mount) = mount else {
        return StorageDetails {
            filesystem_type: None,
            filesystem_id: None,
            device_id: fallback_device_id,
            kind: StorageKind::Unknown,
        };
    };
    let kind = classify_linux_storage(&mount.filesystem_type, &mount.major_minor);
    StorageDetails {
        filesystem_type: Some(mount.filesystem_type.clone()),
        filesystem_id: Some(format!("{}:{}", mount.filesystem_type, mount.major_minor)),
        device_id: Some(format!("linux-dev:{}", mount.major_minor)),
        kind,
    }
}

#[cfg(not(target_os = "linux"))]
fn discover_storage_details(_path: &Path, _metadata: &Metadata) -> StorageDetails {
    StorageDetails {
        filesystem_type: None,
        filesystem_id: None,
        device_id: None,
        kind: StorageKind::Unknown,
    }
}

#[cfg(any(target_os = "linux", test))]
#[derive(Debug)]
struct LinuxMount {
    mount_point: PathBuf,
    major_minor: String,
    filesystem_type: String,
}

#[cfg(any(target_os = "linux", test))]
fn find_linux_mount(contents: &str, path: &Path) -> Option<LinuxMount> {
    let mut best = None;
    for line in contents.lines() {
        let Some((left, right)) = line.split_once(" - ") else {
            continue;
        };
        let left_fields: Vec<&str> = left.split_whitespace().collect();
        let right_fields: Vec<&str> = right.split_whitespace().collect();
        if left_fields.len() < 5 || right_fields.is_empty() {
            continue;
        }
        let mount_point = PathBuf::from(decode_mount_field(left_fields[4]));
        if !path.starts_with(&mount_point) {
            continue;
        }
        let candidate = LinuxMount {
            mount_point,
            major_minor: left_fields[2].to_string(),
            filesystem_type: right_fields[0].to_string(),
        };
        let is_better_match = match &best {
            Some(current) => {
                candidate.mount_point.as_os_str().len() > current.mount_point.as_os_str().len()
            }
            None => true,
        };
        if is_better_match {
            best = Some(candidate);
        }
    }
    best
}

#[cfg(any(target_os = "linux", test))]
fn decode_mount_field(value: &str) -> String {
    value
        .replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}

#[cfg(any(target_os = "linux", test))]
fn classify_linux_storage(filesystem_type: &str, major_minor: &str) -> StorageKind {
    if is_network_filesystem(filesystem_type) {
        return StorageKind::Network;
    }
    if matches!(filesystem_type, "tmpfs" | "ramfs") {
        return StorageKind::MemoryBacked;
    }
    let sysfs_path = PathBuf::from("/sys/dev/block").join(major_minor);
    let Ok(device_path) = fs::canonicalize(sysfs_path) else {
        return StorageKind::Unknown;
    };
    let removable = device_path.ancestors().take(8).find_map(|ancestor| {
        match read_trimmed(ancestor.join("removable")).as_deref() {
            Some("0") => Some(false),
            Some("1") => Some(true),
            _ => None,
        }
    });
    let subsystem = device_path.ancestors().take(8).find_map(|ancestor| {
        fs::canonicalize(ancestor.join("device/subsystem"))
            .ok()
            .and_then(|path| path.file_name().map(|name| name.to_string_lossy().into_owned()))
    });
    let rotational = device_path.ancestors().take(8).find_map(|ancestor| {
        match read_trimmed(ancestor.join("queue/rotational")).as_deref() {
            Some("0") => Some(false),
            Some("1") => Some(true),
            _ => None,
        }
    });
    classify_block_storage(removable, subsystem.as_deref(), rotational)
}

#[cfg(any(target_os = "linux", test))]
fn classify_block_storage(
    removable: Option<bool>,
    subsystem: Option<&str>,
    rotational: Option<bool>,
) -> StorageKind {
    if removable == Some(true) {
        StorageKind::Removable
    } else if subsystem == Some("nvme") {
        StorageKind::Nvme
    } else if rotational == Some(false) {
        StorageKind::SolidState
    } else if rotational == Some(true) {
        StorageKind::Rotational
    } else {
        StorageKind::Local
    }
}

#[cfg(any(target_os = "linux", test))]
fn is_network_filesystem(filesystem_type: &str) -> bool {
    matches!(
        filesystem_type,
        "nfs"
            | "nfs4"
            | "cifs"
            | "smb3"
            | "9p"
            | "ceph"
            | "glusterfs"
            | "fuse.sshfs"
    )
}

#[cfg(any(target_os = "linux", test))]
fn read_trimmed(path: impl AsRef<Path>) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn map_storage_error(error: io::Error) -> StorageDiscoveryError {
    match error.kind() {
        io::ErrorKind::NotFound | io::ErrorKind::InvalidInput => {
            StorageDiscoveryError::InvalidPath
        }
        io::ErrorKind::PermissionDenied => StorageDiscoveryError::PermissionDenied,
        _ => StorageDiscoveryError::Unavailable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    use tempfile::{NamedTempFile, TempDir};

    fn synthetic_machine(available_ram_bytes: Option<u64>) -> MachineProfile {
        let mut cpu_features = BTreeSet::new();
        cpu_features.insert(CpuFeature::Avx2);
        machine_profile_from_facts(MachineFacts {
            os: "linux".to_string(),
            architecture: "x86_64".to_string(),
            kernel_version: Some("test-kernel".to_string()),
            cpu_vendor: Some("GenuineTest".to_string()),
            cpu_model: Some("Test CPU".to_string()),
            physical_cpu_cores: Some(2),
            logical_cpu_cores: 4,
            cpu_features,
            total_ram_bytes: Some(16 * 1024 * 1024),
            available_ram_bytes,
            gpu_inventory_state: DiscoveryState::Complete,
            gpus: vec![
                GpuDeviceProfile {
                    identifier: "gpu-a".to_string(),
                    vendor: Some("Vendor A".to_string()),
                    device_id: Some("device-a".to_string()),
                    name: Some("GPU A".to_string()),
                    dedicated_memory_bytes: Some(1024),
                    backend: Availability {
                        detected: true,
                        supported: false,
                        usable: false,
                    },
                },
                GpuDeviceProfile {
                    identifier: "gpu-b".to_string(),
                    vendor: Some("Vendor B".to_string()),
                    device_id: Some("device-b".to_string()),
                    name: None,
                    dedicated_memory_bytes: None,
                    backend: Availability {
                        detected: true,
                        supported: false,
                        usable: false,
                    },
                },
            ],
        })
        .unwrap()
    }

    #[test]
    fn test_machine_profile_from_synthetic_facts_is_deterministic() {
        let first = synthetic_machine(Some(12 * 1024 * 1024));
        let second = synthetic_machine(Some(12 * 1024 * 1024));
        assert_eq!(first, second);
        assert_eq!(first.architecture, "x86_64");
        assert_eq!(first.physical_cpu_cores, Some(2));
        assert_eq!(first.logical_cpu_cores, 4);
        assert!(first.cpu_features.contains(&CpuFeature::Avx2));
        assert_eq!(first.gpus.len(), 2);
        assert!(first.gpus.iter().all(|gpu| gpu.backend.detected));
        assert!(first.gpus.iter().all(|gpu| !gpu.backend.supported));
        assert!(first.gpus.iter().all(|gpu| !gpu.backend.usable));
    }

    #[test]
    fn test_machine_fingerprint_excludes_available_ram_and_kernel_version() {
        let first = synthetic_machine(Some(12 * 1024 * 1024));
        let mut second = synthetic_machine(Some(4 * 1024 * 1024));
        second.kernel_version = Some("updated-kernel".to_string());
        assert_eq!(first.fingerprint(), second.fingerprint());
    }

    #[test]
    fn test_machine_profile_allows_unknown_memory_and_gpu_inventory() {
        let profile = machine_profile_from_facts(MachineFacts {
            os: "other".to_string(),
            architecture: "unknown-arch".to_string(),
            kernel_version: None,
            cpu_vendor: None,
            cpu_model: None,
            physical_cpu_cores: None,
            logical_cpu_cores: 1,
            cpu_features: BTreeSet::new(),
            total_ram_bytes: None,
            available_ram_bytes: None,
            gpu_inventory_state: DiscoveryState::Unavailable,
            gpus: Vec::new(),
        })
        .unwrap();
        assert_eq!(profile.total_ram_bytes, None);
        assert_eq!(profile.available_ram_bytes, None);
        assert_eq!(profile.gpu_inventory_state, DiscoveryState::Unavailable);
    }

    #[test]
    fn test_linux_cpu_and_memory_parsers_use_explicit_fields() {
        let cpuinfo = concat!(
            "processor: 0\nphysical id: 0\ncore id: 0\nvendor_id: V\nmodel name: M\n\n",
            "processor: 1\nphysical id: 0\ncore id: 0\n\n",
            "processor: 2\nphysical id: 0\ncore id: 1\n\n",
            "processor: 3\nphysical id: 0\ncore id: 1\n",
        );
        let cpu = parse_linux_cpuinfo(cpuinfo);
        assert_eq!(cpu.vendor.as_deref(), Some("V"));
        assert_eq!(cpu.model.as_deref(), Some("M"));
        assert_eq!(cpu.physical_cores, Some(2));

        let memory = parse_linux_meminfo("MemTotal: 16384 kB\nMemAvailable: 8192 kB\n");
        assert_eq!(memory, (Some(16 * 1024 * 1024), Some(8 * 1024 * 1024)));
        assert_eq!(parse_linux_meminfo("MemAvailable: 8 kB\n"), (None, None));
    }

    #[test]
    fn test_storage_discovery_accepts_small_regular_file() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&[0xA5; 64]).unwrap();
        file.flush().unwrap();
        let profile = StorageDiscovery.discover(file.path()).unwrap();
        assert_eq!(profile.path_state, StoragePathState::Ready);
        assert!(profile.readable);
        assert!(profile.regular_file);
        assert_eq!(profile.file_size_bytes, Some(64));
        assert!(profile.seekable);
        assert!(profile.model_path.is_absolute());
    }

    #[test]
    fn test_storage_discovery_rejects_invalid_path_and_directory() {
        let missing_parent = TempDir::new().unwrap();
        let missing = missing_parent.path().join("missing-model.gguf");
        assert_eq!(
            StorageDiscovery.discover(missing).unwrap_err(),
            StorageDiscoveryError::InvalidPath
        );
        let directory = TempDir::new().unwrap();
        assert_eq!(
            StorageDiscovery.discover(directory.path()).unwrap_err(),
            StorageDiscoveryError::NotRegularFile
        );
        assert_eq!(
            map_storage_error(io::Error::from(io::ErrorKind::PermissionDenied)),
            StorageDiscoveryError::PermissionDenied
        );
    }

    #[test]
    fn test_storage_fingerprint_omits_path_but_tracks_storage_identity() {
        let mut first = StorageProfile {
            schema_version: PROFILE_SCHEMA_VERSION,
            model_path: PathBuf::from("/home/alice/model.gguf"),
            path_state: StoragePathState::Ready,
            readable: true,
            regular_file: true,
            file_size_bytes: Some(100),
            filesystem_type: Some("ext4".to_string()),
            filesystem_id: Some("ext4:8:1".to_string()),
            device_id: Some("linux-dev:8:1".to_string()),
            kind: StorageKind::SolidState,
            seekable: true,
        };
        let mut second = first.clone();
        second.model_path = PathBuf::from("/home/bob/renamed.gguf");
        second.file_size_bytes = Some(200);
        assert_eq!(first.fingerprint(), second.fingerprint());
        first.device_id = Some("linux-dev:8:2".to_string());
        assert_ne!(first.fingerprint(), second.fingerprint());
    }

    #[test]
    fn test_mount_parser_prefers_longest_matching_mount() {
        let mountinfo = concat!(
            "1 0 8:1 / / rw - ext4 /dev/root rw\n",
            "2 1 0:42 / /models rw - nfs server:/models rw\n",
        );
        let mount = find_linux_mount(mountinfo, Path::new("/models/model.gguf")).unwrap();
        assert_eq!(mount.mount_point, PathBuf::from("/models"));
        assert_eq!(mount.major_minor, "0:42");
        assert_eq!(mount.filesystem_type, "nfs");
        assert_eq!(
            classify_linux_storage(&mount.filesystem_type, &mount.major_minor),
            StorageKind::Network
        );
    }

    #[test]
    fn test_block_storage_classification_uses_explicit_device_facts() {
        assert_eq!(
            classify_block_storage(Some(true), Some("nvme"), Some(false)),
            StorageKind::Removable
        );
        assert_eq!(
            classify_block_storage(Some(false), Some("nvme"), Some(false)),
            StorageKind::Nvme
        );
        assert_eq!(
            classify_block_storage(Some(false), None, Some(false)),
            StorageKind::SolidState
        );
        assert_eq!(
            classify_block_storage(Some(false), None, Some(true)),
            StorageKind::Rotational
        );
        assert_eq!(
            classify_block_storage(None, None, None),
            StorageKind::Local
        );
    }

    #[test]
    fn test_drm_card_name_and_vendor_parsing_are_conservative() {
        assert!(is_drm_card_name("card0"));
        assert!(is_drm_card_name("card12"));
        assert!(!is_drm_card_name("card0-DP-1"));
        assert!(!is_drm_card_name("renderD128"));
        assert_eq!(gpu_vendor_name("0x10de").as_deref(), Some("NVIDIA"));
        assert_eq!(gpu_vendor_name("0xffff"), None);
    }
}
