//! Measured resident pages for a file-backed model mapping.
//!
//! This is observation only. It does not evict pages or claim that RSS is
//! capped. The kernel owns reclaim for clean file-backed mappings.

use std::path::{Path, PathBuf};

/// Resident accounting for mappings whose backing path matches the model file.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MappingResidency {
    pub mapping_count: u64,
    /// Virtual bytes in the matching mappings.
    pub mapped_bytes: u64,
    /// Resident set bytes across the matching mappings.
    pub rss_bytes: u64,
    /// Proportional set bytes, which apportions shared pages.
    pub pss_bytes: u64,
    pub shared_clean_bytes: u64,
    pub private_clean_bytes: u64,
    pub private_dirty_bytes: u64,
    pub anonymous_bytes: u64,
}

impl MappingResidency {
    pub fn is_present(&self) -> bool {
        self.mapping_count > 0
    }

    /// Resident share of the virtual mapping, in bytes per byte.
    pub fn resident_fraction(&self) -> Option<f64> {
        (self.mapped_bytes > 0).then(|| self.rss_bytes as f64 / self.mapped_bytes as f64)
    }
}

/// Measure current resident pages for `model_path` in this process.
#[cfg(target_os = "linux")]
pub fn model_mapping_residency(model_path: &Path) -> Option<MappingResidency> {
    let smaps = std::fs::read_to_string("/proc/self/smaps").ok()?;
    let target = std::fs::canonicalize(model_path).ok()?;
    Some(parse_smaps(&smaps, target.to_str()?))
}

#[cfg(not(target_os = "linux"))]
pub fn model_mapping_residency(_model_path: &Path) -> Option<MappingResidency> {
    None
}

/// Drop clean page-cache pages for a file so a subsequent cold run is charged
/// for the pages it faults. This is a hint, not a guarantee: mapped or dirty
/// pages may remain.
#[cfg(unix)]
pub fn evict_file_cache(path: &Path) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;
    let file = std::fs::File::open(path)?;
    let result = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    if result != 0 {
        return Err(std::io::Error::from_raw_os_error(result));
    }
    Ok(())
}

#[cfg(not(unix))]
pub fn evict_file_cache(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

/// Parse smaps text and sum mappings backed by `target`.
pub fn parse_smaps(input: &str, target: &str) -> MappingResidency {
    let mut residency = MappingResidency::default();
    let mut in_target = false;

    for line in input.lines() {
        if is_mapping_header(line) {
            in_target = mapping_path(line).is_some_and(|path| path == target);
            if in_target {
                residency.mapping_count += 1;
            }
            continue;
        }
        if !in_target {
            continue;
        }
        let Some((key, bytes)) = smaps_field(line) else {
            continue;
        };
        match key {
            "Size" => residency.mapped_bytes += bytes,
            "Rss" => residency.rss_bytes += bytes,
            "Pss" => residency.pss_bytes += bytes,
            "Shared_Clean" => residency.shared_clean_bytes += bytes,
            "Private_Clean" => residency.private_clean_bytes += bytes,
            "Private_Dirty" => residency.private_dirty_bytes += bytes,
            "Anonymous" => residency.anonymous_bytes += bytes,
            _ => {}
        }
    }
    residency
}

fn is_mapping_header(line: &str) -> bool {
    let first = line.split_whitespace().next().unwrap_or("");
    first.contains('-') && first.chars().all(|ch| ch.is_ascii_hexdigit() || ch == '-')
}

fn mapping_path(line: &str) -> Option<&str> {
    // Mapping headers have five fields before the path. The path itself may
    // contain spaces, so take the remaining text instead of the next token.
    let mut rest = line;
    for _ in 0..5 {
        let trimmed = rest.trim_start();
        let boundary = trimmed.find(char::is_whitespace)?;
        rest = &trimmed[boundary..];
    }
    let path = rest.trim_start();
    if !path.starts_with('/') {
        return None;
    }
    Some(path.strip_suffix(" (deleted)").unwrap_or(path))
}

fn smaps_field(line: &str) -> Option<(&str, u64)> {
    let (key, rest) = line.split_once(':')?;
    let value = rest.split_whitespace().next()?.parse::<u64>().ok()?;
    Some((key, value.saturating_mul(1024)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
7f0000000000-7f0000100000 r--p 00000000 08:01 42 /models/m.gguf
Size:             1024 kB
KernelPageSize:      4 kB
Rss:               512 kB
Pss:               500 kB
Shared_Clean:      100 kB
Private_Clean:     400 kB
Private_Dirty:      12 kB
Anonymous:           0 kB

7f0000100000-7f0000200000 rw-p 00000000 00:00 0
Size:              512 kB
Rss:               200 kB
Pss:               200 kB

7f0000200000-7f0000300000 r--p 00000000 08:01 42 /models/m.gguf (deleted)
Size:              256 kB
Rss:               128 kB
Pss:               128 kB
";

    #[test]
    fn sums_only_matching_file_mappings() {
        let residency = parse_smaps(SAMPLE, "/models/m.gguf");
        assert_eq!(residency.mapping_count, 2);
        assert_eq!(residency.mapped_bytes, (1024 + 256) * 1024);
        assert_eq!(residency.rss_bytes, (512 + 128) * 1024);
        assert_eq!(residency.pss_bytes, (500 + 128) * 1024);
        assert_eq!(residency.shared_clean_bytes, 100 * 1024);
        assert_eq!(residency.private_clean_bytes, 400 * 1024);
        assert_eq!(residency.private_dirty_bytes, 12 * 1024);
        assert_eq!(residency.anonymous_bytes, 0);
        assert_eq!(residency.resident_fraction(), Some(640.0 / 1280.0));
    }

    #[test]
    fn absent_file_reports_zero_mappings() {
        let residency = parse_smaps(SAMPLE, "/models/other.gguf");
        assert!(!residency.is_present());
        assert_eq!(residency.resident_fraction(), None);
    }

    #[test]
    fn keeps_spaces_in_mapping_paths() {
        let input = "7f0-7f1 r--p 0 08:01 42 /models/with space/m.gguf\nSize: 8 kB\nRss: 4 kB\n";
        let residency = parse_smaps(input, "/models/with space/m.gguf");
        assert_eq!(residency.mapping_count, 1);
        assert_eq!(residency.rss_bytes, 4 * 1024);
    }

    #[test]
    fn ignores_non_file_mapping_headers() {
        let residency = parse_smaps(
            "7f0000000000-7f0000100000 rw-p 00000000 00:00 0\nSize: 4 kB\nRss: 4 kB\n",
            "/models/m.gguf",
        );
        assert_eq!(residency, MappingResidency::default());
    }
}

/// Bounded per-page residency observation configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageObservationConfig {
    /// Maximum pages inspected in one snapshot.
    pub max_pages: usize,
    /// Maximum resources included in one snapshot.
    pub max_resources: usize,
}

impl Default for PageObservationConfig {
    fn default() -> Self {
        Self {
            max_pages: 4096,
            max_resources: 512,
        }
    }
}

/// One resource page range to inspect. `mapping_path` is the file backing the
/// native mapping; it is not a pointer or a virtual address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageRange {
    pub name: String,
    pub tensor_index: i32,
    pub generation: u64,
    pub mapping_index: Option<u32>,
    pub mapping_path: Option<PathBuf>,
    pub page_start: u64,
    pub page_end: u64,
    pub page_size: u64,
    pub mapping_size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourcePageResidency {
    pub name: String,
    pub tensor_index: i32,
    pub generation: u64,
    pub mapping_index: Option<u32>,
    pub total_pages: u64,
    pub inspected_pages: u64,
    pub resident_pages: u64,
    pub nonresident_pages: u64,
    pub unknown_pages: u64,
    pub sampled: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageResidencySnapshot {
    pub supported: bool,
    pub source: &'static str,
    pub page_size: Option<u64>,
    pub inspected_pages: u64,
    pub total_pages: u64,
    pub resident_pages: u64,
    pub nonresident_pages: u64,
    pub unknown_pages: u64,
    pub unique_resident_pages: u64,
    pub unique_nonresident_pages: u64,
    /// Unknown pages cannot be safely deduplicated without inspecting their
    /// page keys; `None` means the unique unknown count is unavailable.
    pub unique_unknown_pages: Option<u64>,
    pub sampled: bool,
    pub resources: Vec<ResourcePageResidency>,
    pub elapsed_ns: u128,
    pub unsupported_reason: Option<String>,
}

impl PageResidencySnapshot {
    pub fn unsupported(reason: impl Into<String>) -> Self {
        Self {
            supported: false,
            source: "unavailable",
            page_size: None,
            inspected_pages: 0,
            total_pages: 0,
            resident_pages: 0,
            nonresident_pages: 0,
            unknown_pages: 0,
            unique_resident_pages: 0,
            unique_nonresident_pages: 0,
            unique_unknown_pages: None,
            sampled: false,
            resources: Vec::new(),
            elapsed_ns: 0,
            unsupported_reason: Some(reason.into()),
        }
    }
}

/// Observe resident pages for bounded resource ranges using Linux `mincore`.
/// This never faults pages in, never evicts, and never exposes addresses.
#[cfg(target_os = "linux")]
pub fn observe_page_residency(
    ranges: &[PageRange],
    config: PageObservationConfig,
) -> PageResidencySnapshot {
    use std::collections::BTreeMap;
    use std::time::Instant;

    let started = Instant::now();
    let maps = match std::fs::read_to_string("/proc/self/maps") {
        Ok(maps) => maps,
        Err(error) => {
            return PageResidencySnapshot::unsupported(format!("/proc/self/maps: {error}"));
        }
    };
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return PageResidencySnapshot::unsupported("page size unavailable");
    }
    let page_size = page_size as u64;
    let mut bases: BTreeMap<PathBuf, u64> = BTreeMap::new();
    for line in maps.lines() {
        let Some((start, offset, path)) = parse_maps_header(line) else {
            continue;
        };
        let Some(base) = start.checked_sub(offset) else {
            continue;
        };
        bases
            .entry(PathBuf::from(path))
            .and_modify(|existing| *existing = (*existing).min(base))
            .or_insert(base);
    }

    let mut snapshot = PageResidencySnapshot {
        supported: true,
        source: "linux mincore(/proc/self/maps)",
        page_size: Some(page_size),
        inspected_pages: 0,
        total_pages: 0,
        resident_pages: 0,
        nonresident_pages: 0,
        unknown_pages: 0,
        unique_resident_pages: 0,
        unique_nonresident_pages: 0,
        unique_unknown_pages: None,
        sampled: false,
        resources: Vec::new(),
        elapsed_ns: 0,
        unsupported_reason: None,
    };
    let mut unique: BTreeMap<(PathBuf, u64), bool> = BTreeMap::new();
    for range in ranges.iter().take(config.max_resources) {
        let total_pages = if range.page_size == 0 || range.page_end <= range.page_start {
            0
        } else {
            (range.page_end - range.page_start).div_ceil(range.page_size)
        };
        snapshot.total_pages = snapshot.total_pages.saturating_add(total_pages);
        let Some(mapping_index) = range.mapping_index else {
            snapshot.resources.push(ResourcePageResidency {
                name: range.name.clone(),
                tensor_index: range.tensor_index,
                generation: range.generation,
                mapping_index: None,
                total_pages,
                inspected_pages: 0,
                resident_pages: 0,
                nonresident_pages: 0,
                unknown_pages: total_pages,
                sampled: false,
                error: Some("resource has no file-backed mapping".to_string()),
            });
            snapshot.unknown_pages = snapshot.unknown_pages.saturating_add(total_pages);
            continue;
        };
        let Some(path) = range.mapping_path.as_ref() else {
            snapshot.resources.push(ResourcePageResidency {
                name: range.name.clone(),
                tensor_index: range.tensor_index,
                generation: range.generation,
                mapping_index: Some(mapping_index),
                total_pages,
                inspected_pages: 0,
                resident_pages: 0,
                nonresident_pages: 0,
                unknown_pages: total_pages,
                sampled: false,
                error: Some("mapping shard path unavailable".to_string()),
            });
            snapshot.unknown_pages = snapshot.unknown_pages.saturating_add(total_pages);
            continue;
        };
        let Some(base) = bases.get(path) else {
            snapshot.resources.push(ResourcePageResidency {
                name: range.name.clone(),
                tensor_index: range.tensor_index,
                generation: range.generation,
                mapping_index: Some(mapping_index),
                total_pages,
                inspected_pages: 0,
                resident_pages: 0,
                nonresident_pages: 0,
                unknown_pages: total_pages,
                sampled: false,
                error: Some("mapping is not present in /proc/self/maps".to_string()),
            });
            snapshot.unknown_pages = snapshot.unknown_pages.saturating_add(total_pages);
            continue;
        };
        let remaining_budget = config
            .max_pages
            .saturating_sub(snapshot.inspected_pages as usize);
        let inspect_pages = (total_pages as usize).min(remaining_budget);
        let mut resource = ResourcePageResidency {
            name: range.name.clone(),
            tensor_index: range.tensor_index,
            generation: range.generation,
            mapping_index: Some(mapping_index),
            total_pages,
            inspected_pages: 0,
            resident_pages: 0,
            nonresident_pages: 0,
            unknown_pages: 0,
            sampled: inspect_pages < total_pages as usize,
            error: None,
        };
        if inspect_pages == 0 {
            resource.unknown_pages = total_pages;
            snapshot.unknown_pages = snapshot.unknown_pages.saturating_add(total_pages);
            snapshot.resources.push(resource);
            snapshot.sampled = true;
            continue;
        }
        let length = (inspect_pages as u64).saturating_mul(page_size);
        let address = match base.checked_add(range.page_start) {
            Some(address) => address,
            None => {
                resource.unknown_pages = total_pages;
                resource.error = Some("page address overflow".to_string());
                snapshot.unknown_pages = snapshot.unknown_pages.saturating_add(total_pages);
                snapshot.resources.push(resource);
                continue;
            }
        };
        let mut vector = vec![0u8; inspect_pages];
        let rc = unsafe {
            libc::mincore(
                address as *mut libc::c_void,
                length as usize,
                vector.as_mut_ptr(),
            )
        };
        if rc != 0 {
            let errno = std::io::Error::last_os_error();
            resource.unknown_pages = total_pages;
            resource.error = Some(format!("mincore: {errno}"));
            snapshot.unknown_pages = snapshot.unknown_pages.saturating_add(total_pages);
            snapshot.resources.push(resource);
            continue;
        }
        resource.inspected_pages = inspect_pages as u64;
        for (index, status) in vector.iter().enumerate() {
            if status & 1 != 0 {
                resource.resident_pages += 1;
                snapshot.resident_pages += 1;
                unique.insert(
                    (path.clone(), range.page_start / page_size + index as u64),
                    true,
                );
            } else {
                resource.nonresident_pages += 1;
                snapshot.nonresident_pages += 1;
                unique.insert(
                    (path.clone(), range.page_start / page_size + index as u64),
                    false,
                );
            }
        }
        resource.unknown_pages = total_pages.saturating_sub(inspect_pages as u64);
        snapshot.inspected_pages = snapshot
            .inspected_pages
            .saturating_add(inspect_pages as u64);
        snapshot.unknown_pages = snapshot
            .unknown_pages
            .saturating_add(resource.unknown_pages);
        snapshot.sampled |= resource.sampled;
        snapshot.resources.push(resource);
    }
    // Resources omitted by max_resources are unknown, never zero.
    if ranges.len() > config.max_resources {
        let omitted = ranges.len() - config.max_resources;
        snapshot.unknown_pages = snapshot
            .unknown_pages
            .saturating_add(omitted as u64 * page_size);
        snapshot.sampled = true;
    }
    for resident in unique.values() {
        if *resident {
            snapshot.unique_resident_pages += 1;
        } else {
            snapshot.unique_nonresident_pages += 1;
        }
    }
    snapshot.elapsed_ns = started.elapsed().as_nanos();
    snapshot
}

#[cfg(not(target_os = "linux"))]
pub fn observe_page_residency(
    _ranges: &[PageRange],
    _config: PageObservationConfig,
) -> PageResidencySnapshot {
    PageResidencySnapshot::unsupported("mincore observation is Linux-only")
}

#[cfg(target_os = "linux")]
fn parse_maps_header(line: &str) -> Option<(u64, u64, &str)> {
    let mut rest = line;
    let mut fields = [""; 5];
    for field in &mut fields {
        let trimmed = rest.trim_start();
        let boundary = trimmed.find(char::is_whitespace)?;
        *field = &trimmed[..boundary];
        rest = &trimmed[boundary..];
    }
    let (start, end) = fields[0].split_once('-')?;
    let start = u64::from_str_radix(start, 16).ok()?;
    let _end = u64::from_str_radix(end, 16).ok()?;
    let offset = u64::from_str_radix(fields[2], 16).ok()?;
    let path = rest.trim_start();
    if !path.starts_with('/') {
        return None;
    }
    Some((
        start,
        offset,
        path.strip_suffix(" (deleted)").unwrap_or(path),
    ))
}

#[cfg(test)]
mod observation_tests {
    use super::*;

    fn page_size() -> u64 {
        unsafe { libc::sysconf(libc::_SC_PAGESIZE) as u64 }
    }

    #[test]
    fn maps_header_parsing_handles_offsets_and_deleted_paths() {
        let line = "7f0000000000-7f0000100000 r--p 00000000 08:01 42 /tmp/m.gguf";
        let (start, offset, path) = parse_maps_header(line).expect("header");
        assert_eq!(start, 0x7f0000000000);
        assert_eq!(offset, 0);
        assert_eq!(path, "/tmp/m.gguf");
        let deleted = "7f0000000000-7f0000100000 r--p 00001000 08:01 42 /tmp/m.gguf (deleted)";
        assert_eq!(
            parse_maps_header(deleted).expect("deleted").2,
            "/tmp/m.gguf"
        );
    }

    #[test]
    fn zero_or_reversed_ranges_are_unknown_not_resident() {
        let path = std::env::temp_dir().join(format!(
            "ramforge-pages-{}-{}.bin",
            std::process::id(),
            "empty"
        ));
        std::fs::write(&path, b"x").expect("file");
        let snapshot = observe_page_residency(
            &[PageRange {
                name: "x".to_string(),
                tensor_index: 0,
                generation: 1,
                mapping_index: Some(0),
                mapping_path: Some(path.clone()),
                page_start: 4096,
                page_end: 4096,
                page_size: 4096,
                mapping_size: 4096,
            }],
            PageObservationConfig::default(),
        );
        let _ = std::fs::remove_file(&path);
        assert!(snapshot.supported);
        assert_eq!(snapshot.resources[0].total_pages, 0);
        assert_eq!(snapshot.resources[0].resident_pages, 0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn mincore_is_bounded_and_does_not_invent_unknown_zeroes() {
        use std::os::unix::io::AsRawFd;
        let page = page_size();
        let path = std::env::temp_dir().join(format!(
            "ramforge-pages-{}-{}.bin",
            std::process::id(),
            "mincore"
        ));
        std::fs::write(&path, vec![1u8; (page * 2) as usize]).expect("file");
        let canonical = path.canonicalize().expect("canonical");
        let file = std::fs::File::open(&canonical).expect("open");
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                (page * 2) as usize,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        assert_ne!(address, libc::MAP_FAILED);
        // Touch the first page through the mapping.
        unsafe { std::ptr::read_volatile(address.cast::<u8>()) };
        let snapshot = observe_page_residency(
            &[PageRange {
                name: "x".to_string(),
                tensor_index: 0,
                generation: 1,
                mapping_index: Some(0),
                mapping_path: Some(canonical.clone()),
                page_start: 0,
                page_end: page * 2,
                page_size: page,
                mapping_size: page * 2,
            }],
            PageObservationConfig {
                max_pages: 1,
                max_resources: 1,
            },
        );
        unsafe { libc::munmap(address, (page * 2) as usize) };
        let _ = std::fs::remove_file(&path);
        assert!(snapshot.supported);
        assert_eq!(snapshot.inspected_pages, 1);
        assert_eq!(snapshot.resources[0].inspected_pages, 1);
        assert_eq!(snapshot.resources[0].unknown_pages, 1);
        assert!(snapshot.sampled);
        assert_eq!(
            snapshot.unique_resident_pages + snapshot.unique_nonresident_pages,
            1
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchResidencyClass {
    AlreadyResident,
    BecameResident,
    RemainedNonResident,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchBenefitClass {
    /// Only assigned when a measured physical/performance benefit exists.
    ClearlyUseful,
    /// Residency changed and the tensor was observed, but no per-resource
    /// fault or performance attribution is available.
    PotentiallyUseful,
    /// The pages were already resident before the request.
    Redundant,
    /// The request was accepted/attempted but no useful access/residency
    /// effect was observed.
    Wasted,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefetchResourceCorrelation {
    pub name: String,
    pub before_resident_pages: u64,
    pub before_total_pages: u64,
    pub before_unknown_pages: u64,
    pub after_resident_pages: u64,
    pub after_total_pages: u64,
    pub after_unknown_pages: u64,
    pub class: PrefetchResidencyClass,
    pub benefit: PrefetchBenefitClass,
    pub access_observed: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrefetchCorrelationReport {
    pub resources: Vec<PrefetchResourceCorrelation>,
    pub already_resident: usize,
    pub became_resident: usize,
    pub remained_nonresident: usize,
    pub unknown: usize,
    pub access_observed: usize,
    pub access_not_observed: usize,
    pub access_unknown: usize,
    pub clearly_useful: usize,
    pub potentially_useful: usize,
    pub redundant: usize,
    pub wasted: usize,
    pub benefit_unknown: usize,
    pub notes: Vec<String>,
}

/// Correlate selected prefetch resource names with before/after page snapshots.
/// A page is never called newly resident if it was already resident before the
/// request. Per-resource page observations are sampled and may be incomplete.
pub fn correlate_prefetch_residency(
    selected: &[String],
    before: &PageResidencySnapshot,
    after: &PageResidencySnapshot,
    observed_access: Option<&std::collections::BTreeSet<String>>,
) -> PrefetchCorrelationReport {
    let mut report = PrefetchCorrelationReport::default();
    for name in selected {
        let before_resource = before
            .resources
            .iter()
            .find(|resource| resource.name == *name);
        let after_resource = after
            .resources
            .iter()
            .find(|resource| resource.name == *name);
        let (
            class,
            before_resident,
            before_total,
            before_unknown,
            after_resident,
            after_total,
            after_unknown,
        ) = match (before_resource, after_resource) {
            (Some(before), Some(after)) => {
                let before_resident = before.resident_pages;
                let before_total = before.total_pages;
                let before_unknown = before.unknown_pages;
                let after_resident = after.resident_pages;
                let after_total = after.total_pages;
                let after_unknown = after.unknown_pages;
                let class = if before_total == 0 || before_unknown > 0 || after_unknown > 0 {
                    PrefetchResidencyClass::Unknown
                } else if before_resident == before_total {
                    PrefetchResidencyClass::AlreadyResident
                } else if after_resident > before_resident {
                    PrefetchResidencyClass::BecameResident
                } else {
                    PrefetchResidencyClass::RemainedNonResident
                };
                (
                    class,
                    before_resident,
                    before_total,
                    before_unknown,
                    after_resident,
                    after_total,
                    after_unknown,
                )
            }
            _ => (PrefetchResidencyClass::Unknown, 0, 0, 0, 0, 0, 0),
        };
        let access_observed = observed_access.map(|names| names.contains(name));
        match class {
            PrefetchResidencyClass::AlreadyResident => report.already_resident += 1,
            PrefetchResidencyClass::BecameResident => report.became_resident += 1,
            PrefetchResidencyClass::RemainedNonResident => report.remained_nonresident += 1,
            PrefetchResidencyClass::Unknown => report.unknown += 1,
        }
        match access_observed {
            Some(true) => report.access_observed += 1,
            Some(false) => report.access_not_observed += 1,
            None => report.access_unknown += 1,
        }
        let benefit = match (class, access_observed) {
            (PrefetchResidencyClass::AlreadyResident, _) => PrefetchBenefitClass::Redundant,
            (PrefetchResidencyClass::BecameResident, Some(true)) => {
                PrefetchBenefitClass::PotentiallyUseful
            }
            (PrefetchResidencyClass::BecameResident, Some(false)) => PrefetchBenefitClass::Wasted,
            (PrefetchResidencyClass::RemainedNonResident, _) => PrefetchBenefitClass::Wasted,
            _ => PrefetchBenefitClass::Unknown,
        };
        match benefit {
            PrefetchBenefitClass::ClearlyUseful => report.clearly_useful += 1,
            PrefetchBenefitClass::PotentiallyUseful => report.potentially_useful += 1,
            PrefetchBenefitClass::Redundant => report.redundant += 1,
            PrefetchBenefitClass::Wasted => report.wasted += 1,
            PrefetchBenefitClass::Unknown => report.benefit_unknown += 1,
        }
        report.resources.push(PrefetchResourceCorrelation {
            name: name.clone(),
            before_resident_pages: before_resident,
            before_total_pages: before_total,
            before_unknown_pages: before_unknown,
            after_resident_pages: after_resident,
            after_total_pages: after_total,
            after_unknown_pages: after_unknown,
            class,
            benefit,
            access_observed,
        });
    }
    if before.sampled || after.sampled {
        report.notes.push(
            "page observations are sampled; unknown pages are not treated as nonresident"
                .to_string(),
        );
    }
    if !before.supported || !after.supported {
        report
            .notes
            .push("page residency observation is unsupported or incomplete".to_string());
    }
    report
}

#[cfg(test)]
mod correlation_tests {
    use super::*;

    fn snapshot(resources: Vec<ResourcePageResidency>, sampled: bool) -> PageResidencySnapshot {
        PageResidencySnapshot {
            supported: true,
            source: "test",
            page_size: Some(4096),
            inspected_pages: 3,
            total_pages: 3,
            resident_pages: 2,
            nonresident_pages: 1,
            unknown_pages: 0,
            unique_resident_pages: 2,
            unique_nonresident_pages: 1,
            unique_unknown_pages: None,
            sampled,
            resources,
            elapsed_ns: 0,
            unsupported_reason: None,
        }
    }

    fn resource(name: &str, resident: u64, total: u64, unknown: u64) -> ResourcePageResidency {
        ResourcePageResidency {
            name: name.to_string(),
            tensor_index: 0,
            generation: 1,
            mapping_index: Some(0),
            total_pages: total,
            inspected_pages: total.saturating_sub(unknown),
            resident_pages: resident,
            nonresident_pages: total.saturating_sub(resident).saturating_sub(unknown),
            unknown_pages: unknown,
            sampled: unknown > 0,
            error: None,
        }
    }

    #[test]
    fn classification_does_not_claim_already_resident_or_unknown_as_benefit() {
        let before = snapshot(
            vec![
                resource("a", 2, 2, 0),
                resource("b", 0, 1, 0),
                resource("c", 0, 1, 1),
            ],
            false,
        );
        let after = snapshot(
            vec![
                resource("a", 2, 2, 0),
                resource("b", 1, 1, 0),
                resource("c", 1, 1, 1),
            ],
            false,
        );
        let observed = std::collections::BTreeSet::from(["a".to_string(), "b".to_string()]);
        let report = correlate_prefetch_residency(
            &["a".to_string(), "b".to_string(), "c".to_string()],
            &before,
            &after,
            Some(&observed),
        );
        assert_eq!(report.already_resident, 1);
        assert_eq!(report.became_resident, 1);
        assert_eq!(report.unknown, 1);
        assert_eq!(report.access_observed, 2);
        assert_eq!(report.access_not_observed, 1);
        assert_eq!(
            report.resources[0].class,
            PrefetchResidencyClass::AlreadyResident
        );
        assert_eq!(
            report.resources[1].class,
            PrefetchResidencyClass::BecameResident
        );
        assert_eq!(report.resources[2].class, PrefetchResidencyClass::Unknown);
        assert_eq!(report.redundant, 1);
        assert_eq!(report.potentially_useful, 1);
        assert_eq!(report.benefit_unknown, 1);
    }
}
