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
    /// Maximum unique pages inspected in one snapshot.
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
    /// True when the resource was selected by the current prediction plan.
    pub targeted: bool,
    pub error: Option<String>,
}

/// Coverage of a targeted observation. Targeted and general coverage refer to
/// different populations and must not be combined into a single misleading
/// percentage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetedObservationCoverage {
    pub selected_resources: usize,
    pub selected_resources_fully_observed: usize,
    pub selected_resources_partially_observed: usize,
    pub selected_resources_not_observed: usize,
    /// Unique pages in selected resources.
    pub targeted_pages_available: u64,
    pub targeted_pages_sampled: u64,
    pub targeted_pages_unknown: u64,
    /// Unique pages in non-selected resources.
    pub general_pages_available: u64,
    pub general_pages_sampled: u64,
    pub general_pages_unknown: u64,
    pub targeted_coverage_bps: Option<u16>,
    pub general_coverage_bps: Option<u16>,
    pub page_budget: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageResidencySnapshot {
    pub supported: bool,
    pub source: &'static str,
    pub page_size: Option<u64>,
    /// Unique pages physically inspected.
    pub inspected_pages: u64,
    /// Unique pages in the observed population, where metadata permits their
    /// enumeration.
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
    /// Present only for targeted observation. `None` means this was a general
    /// observation with no selected-prefetch population.
    pub targeted: Option<TargetedObservationCoverage>,
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
            targeted: None,
            elapsed_ns: 0,
            unsupported_reason: Some(reason.into()),
        }
    }
}

/// Observe resident pages for bounded resource ranges using Linux `mincore`.
/// This never faults pages in, never evicts, and never exposes addresses.
pub fn observe_page_residency(
    ranges: &[PageRange],
    config: PageObservationConfig,
) -> PageResidencySnapshot {
    observe_page_residency_internal(ranges, &std::collections::BTreeSet::new(), config, false)
}

/// Observe selected prefetch resources first, then use remaining budget for
/// broader mapping coverage. The selected population and the general
/// population are reported separately.
pub fn observe_page_residency_targeted(
    ranges: &[PageRange],
    selected: &std::collections::BTreeSet<String>,
    config: PageObservationConfig,
) -> PageResidencySnapshot {
    observe_page_residency_internal(ranges, selected, config, true)
}

#[cfg(target_os = "linux")]
fn page_count(range: &PageRange) -> u64 {
    if range.page_size == 0 || range.page_end <= range.page_start {
        0
    } else {
        (range.page_end - range.page_start).div_ceil(range.page_size)
    }
}

#[cfg(target_os = "linux")]
fn page_interval(range: &PageRange) -> Option<(u64, u64)> {
    if range.page_size == 0 || range.page_end <= range.page_start || range.mapping_index.is_none() {
        return None;
    }
    let start = range.page_start / range.page_size;
    let end = range.page_end.div_ceil(range.page_size);
    (end > start).then_some((start, end))
}

#[cfg(target_os = "linux")]
fn add_interval(
    intervals: &mut std::collections::BTreeMap<u32, Vec<(u64, u64)>>,
    range: &PageRange,
) {
    let (Some(mapping_index), Some((start, end))) = (range.mapping_index, page_interval(range))
    else {
        return;
    };
    intervals
        .entry(mapping_index)
        .or_default()
        .push((start, end));
}

#[cfg(target_os = "linux")]
fn interval_union_count(intervals: &mut std::collections::BTreeMap<u32, Vec<(u64, u64)>>) -> u64 {
    let mut total = 0u64;
    for values in intervals.values_mut() {
        values.sort_unstable();
        let Some((mut first, mut last)) = values.first().copied() else {
            continue;
        };
        for &(next_first, next_last) in &values[1..] {
            if next_first > last {
                total = total.saturating_add(last - first);
                first = next_first;
                last = next_last;
            } else {
                last = last.max(next_last);
            }
        }
        total = total.saturating_add(last - first);
    }
    total
}

#[cfg(target_os = "linux")]
fn interval_contains(
    intervals: &std::collections::BTreeMap<u32, Vec<(u64, u64)>>,
    mapping_index: u32,
    page: u64,
) -> bool {
    intervals.get(&mapping_index).is_some_and(|values| {
        values
            .iter()
            .any(|&(start, end)| page >= start && page < end)
    })
}

#[cfg(target_os = "linux")]
fn observe_page_residency_internal(
    ranges: &[PageRange],
    selected: &std::collections::BTreeSet<String>,
    config: PageObservationConfig,
    targeted_mode: bool,
) -> PageResidencySnapshot {
    use std::collections::BTreeMap;
    use std::time::Instant;

    let started = Instant::now();
    let maps = match std::fs::read_to_string("/proc/self/maps") {
        Ok(maps) => maps,
        Err(error) => {
            return PageResidencySnapshot::unsupported(format!("/proc/self/maps: {error}"))
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

    // Selected resources are ordered first. Within each group, inventory order
    // is preserved, making sampling deterministic.
    let mut order: Vec<usize> = (0..ranges.len()).collect();
    order.sort_by_key(|index| {
        let is_target = selected.contains(&ranges[*index].name);
        (!is_target, *index)
    });
    let processed_count = order.len().min(config.max_resources);
    let processed = &order[..processed_count];
    let omitted = &order[processed_count..];

    let total_for = page_count;
    let available_for = |range: &PageRange| {
        range.mapping_index.is_some()
            && range.mapping_path.is_some()
            && range
                .mapping_path
                .as_ref()
                .is_some_and(|path| bases.contains_key(path))
    };

    // Interval arithmetic gives the true unique-page population without
    // allocating one key per page. This matters for multi-hundred-thousand
    // page Qwen mappings and keeps a small sample small.
    let mut all_intervals: BTreeMap<u32, Vec<(u64, u64)>> = BTreeMap::new();
    let mut target_intervals: BTreeMap<u32, Vec<(u64, u64)>> = BTreeMap::new();
    let mut unavailable_unknown_pages = 0u64;
    for index in processed.iter().chain(omitted.iter()) {
        let range = &ranges[*index];
        let total = total_for(range);
        if !available_for(range) {
            unavailable_unknown_pages = unavailable_unknown_pages.saturating_add(total);
            continue;
        }
        add_interval(&mut all_intervals, range);
        if selected.contains(&range.name) {
            add_interval(&mut target_intervals, range);
        }
    }
    let all_unique_pages = interval_union_count(&mut all_intervals);
    let targeted_available = interval_union_count(&mut target_intervals);

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
        targeted: None,
        elapsed_ns: 0,
        unsupported_reason: None,
    };
    let mut observed: BTreeMap<(u32, u64), bool> = BTreeMap::new();

    for index in processed {
        let range = &ranges[*index];
        let total_pages = total_for(range);
        let is_targeted = selected.contains(&range.name);
        let mut resource = ResourcePageResidency {
            name: range.name.clone(),
            tensor_index: range.tensor_index,
            generation: range.generation,
            mapping_index: range.mapping_index,
            total_pages,
            inspected_pages: 0,
            resident_pages: 0,
            nonresident_pages: 0,
            unknown_pages: 0,
            sampled: false,
            targeted: is_targeted,
            error: None,
        };
        if total_pages == 0 {
            snapshot.resources.push(resource);
            continue;
        }
        let Some(mapping_index) = range.mapping_index else {
            resource.unknown_pages = total_pages;
            resource.error = Some("resource has no file-backed mapping".to_string());
            snapshot.resources.push(resource);
            continue;
        };
        let Some(path) = range.mapping_path.as_ref() else {
            resource.unknown_pages = total_pages;
            resource.error = Some("mapping shard path unavailable".to_string());
            snapshot.resources.push(resource);
            continue;
        };
        let Some(base) = bases.get(path).copied() else {
            resource.unknown_pages = total_pages;
            resource.error = Some("mapping is not present in /proc/self/maps".to_string());
            snapshot.resources.push(resource);
            continue;
        };
        let base = match base.checked_add(range.page_start) {
            Some(base) => base,
            None => {
                resource.unknown_pages = total_pages;
                resource.error = Some("page address overflow".to_string());
                snapshot.resources.push(resource);
                continue;
            }
        };
        let mut page_index = 0u64;
        while page_index < total_pages {
            let page = range.page_start / range.page_size + page_index;
            let key = (mapping_index, page);
            if let Some(resident) = observed.get(&key).copied() {
                resource.inspected_pages += 1;
                if resident {
                    resource.resident_pages += 1;
                } else {
                    resource.nonresident_pages += 1;
                }
                page_index += 1;
                continue;
            }
            let budget_left = config.max_pages.saturating_sub(observed.len());
            if budget_left == 0 {
                break;
            }
            let run_start = page_index;
            let mut run_len = 0u64;
            while page_index < total_pages && run_len < budget_left as u64 {
                let run_page = range.page_start / range.page_size + page_index;
                if observed.contains_key(&(mapping_index, run_page)) {
                    break;
                }
                page_index += 1;
                run_len += 1;
            }
            if run_len == 0 {
                break;
            }
            let address = match base.checked_add(run_start.saturating_mul(page_size)) {
                Some(address) => address,
                None => {
                    resource.error = Some("page address overflow".to_string());
                    break;
                }
            };
            let mut vector = vec![0u8; run_len as usize];
            let rc = unsafe {
                libc::mincore(
                    address as *mut libc::c_void,
                    run_len.saturating_mul(page_size) as usize,
                    vector.as_mut_ptr(),
                )
            };
            if rc != 0 {
                resource.error = Some(format!("mincore: {}", std::io::Error::last_os_error()));
                break;
            }
            for (offset, status) in vector.iter().enumerate() {
                let resident = status & 1 != 0;
                let page = range.page_start / range.page_size + run_start + offset as u64;
                observed.insert((mapping_index, page), resident);
                if resident {
                    resource.resident_pages += 1;
                } else {
                    resource.nonresident_pages += 1;
                }
            }
            resource.inspected_pages += run_len;
        }
        resource.unknown_pages = resource
            .total_pages
            .saturating_sub(resource.inspected_pages);
        resource.sampled = resource.inspected_pages < resource.total_pages;
        snapshot.sampled |= resource.sampled;
        snapshot.resources.push(resource);
    }

    for index in omitted {
        let range = &ranges[*index];
        let total_pages = total_for(range);
        snapshot.resources.push(ResourcePageResidency {
            name: range.name.clone(),
            tensor_index: range.tensor_index,
            generation: range.generation,
            mapping_index: range.mapping_index,
            total_pages,
            inspected_pages: 0,
            resident_pages: 0,
            nonresident_pages: 0,
            unknown_pages: total_pages,
            sampled: total_pages > 0,
            targeted: selected.contains(&range.name),
            error: Some("resource omitted by max_resources".to_string()),
        });
        snapshot.sampled |= total_pages > 0;
    }

    snapshot.inspected_pages = observed.len() as u64;
    snapshot.total_pages = all_unique_pages.saturating_add(unavailable_unknown_pages);
    snapshot.resident_pages = observed.values().filter(|resident| **resident).count() as u64;
    snapshot.nonresident_pages = observed.values().filter(|resident| !**resident).count() as u64;
    snapshot.unique_resident_pages = snapshot.resident_pages;
    snapshot.unique_nonresident_pages = snapshot.nonresident_pages;
    let unknown_unique = all_unique_pages.saturating_sub(snapshot.inspected_pages);
    snapshot.unique_unknown_pages = Some(unknown_unique);
    snapshot.unknown_pages = unknown_unique.saturating_add(unavailable_unknown_pages);
    snapshot.sampled |= snapshot.inspected_pages < snapshot.total_pages;

    if targeted_mode {
        let targeted_sampled = observed
            .keys()
            .filter(|(mapping, page)| interval_contains(&target_intervals, *mapping, *page))
            .count() as u64;
        let general_sampled = snapshot.inspected_pages.saturating_sub(targeted_sampled);
        let mut selected_totals: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        for resource in &snapshot.resources {
            if resource.targeted {
                let entry = selected_totals
                    .entry(resource.name.clone())
                    .or_insert((0, 0));
                entry.0 = entry.0.saturating_add(resource.total_pages);
                entry.1 = entry.1.saturating_add(resource.unknown_pages);
            }
        }
        let fully = selected_totals
            .values()
            .filter(|(total, unknown)| *total == 0 || *unknown == 0)
            .count();
        let not_observed = selected_totals
            .values()
            .filter(|(total, unknown)| *total > 0 && *unknown == *total)
            .count()
            + selected.len().saturating_sub(selected_totals.len());
        let partial = selected_totals
            .values()
            .filter(|(total, unknown)| *total > 0 && *unknown > 0 && *unknown < *total)
            .count();
        let bps = |sampled: u64, available: u64| {
            (available > 0).then(|| ((sampled * 10_000) / available) as u16)
        };
        snapshot.targeted = Some(TargetedObservationCoverage {
            selected_resources: selected.len(),
            selected_resources_fully_observed: fully,
            selected_resources_partially_observed: partial,
            selected_resources_not_observed: not_observed,
            targeted_pages_available: targeted_available,
            targeted_pages_sampled: targeted_sampled,
            targeted_pages_unknown: targeted_available.saturating_sub(targeted_sampled),
            general_pages_available: all_unique_pages.saturating_sub(targeted_available),
            general_pages_sampled: general_sampled,
            general_pages_unknown: all_unique_pages
                .saturating_sub(targeted_available)
                .saturating_sub(general_sampled),
            targeted_coverage_bps: bps(targeted_sampled, targeted_available),
            general_coverage_bps: bps(
                general_sampled,
                all_unique_pages.saturating_sub(targeted_available),
            ),
            page_budget: config.max_pages,
        });
    }

    snapshot.elapsed_ns = started.elapsed().as_nanos();
    snapshot
}

#[cfg(not(target_os = "linux"))]
fn observe_page_residency_internal(
    _ranges: &[PageRange],
    _selected: &std::collections::BTreeSet<String>,
    _config: PageObservationConfig,
    _targeted_mode: bool,
) -> PageResidencySnapshot {
    PageResidencySnapshot::unsupported("mincore observation is Linux-only")
}

/// Deprecated compatibility helper retained for callers that only need a
/// general snapshot. It performs no page faults and remains bounded.
#[cfg(not(target_os = "linux"))]
pub fn observe_page_residency_targeted(
    _ranges: &[PageRange],
    _selected: &std::collections::BTreeSet<String>,
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
    use std::collections::BTreeSet;

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
    fn mapped_file(label: &str, pages: u64) -> (PathBuf, *mut libc::c_void) {
        use std::os::unix::io::AsRawFd;
        let page = page_size();
        let path = std::env::temp_dir().join(format!(
            "ramforge-pages-{}-{}.bin",
            std::process::id(),
            label
        ));
        std::fs::write(&path, vec![1u8; (page * pages) as usize]).expect("file");
        let canonical = path.canonicalize().expect("canonical");
        let file = std::fs::File::open(&canonical).expect("open");
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                (page * pages) as usize,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        assert_ne!(address, libc::MAP_FAILED);
        (canonical, address)
    }

    #[cfg(target_os = "linux")]
    fn range(name: &str, path: &Path, start_page: u64, pages: u64) -> PageRange {
        PageRange {
            name: name.to_string(),
            tensor_index: 0,
            generation: 1,
            mapping_index: Some(0),
            mapping_path: Some(path.to_path_buf()),
            page_start: start_page * page_size(),
            page_end: (start_page + pages) * page_size(),
            page_size: page_size(),
            mapping_size: pages * page_size(),
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn targeted_resources_are_sampled_before_general_ranges() {
        let (path, address) = mapped_file("targeted", 3);
        unsafe { std::ptr::read_volatile(address.cast::<u8>()) };
        let ranges = [
            range("general", &path, 0, 1),
            range("selected", &path, 1, 1),
        ];
        let selected = BTreeSet::from(["selected".to_string()]);
        let snapshot = observe_page_residency_targeted(
            &ranges,
            &selected,
            PageObservationConfig {
                max_pages: 1,
                max_resources: 2,
            },
        );
        unsafe { libc::munmap(address, (page_size() * 3) as usize) };
        let _ = std::fs::remove_file(&path);
        assert_eq!(snapshot.resources[0].name, "selected");
        assert_eq!(snapshot.resources[0].inspected_pages, 1);
        assert_eq!(snapshot.resources[1].name, "general");
        assert_eq!(snapshot.resources[1].inspected_pages, 0);
        assert_eq!(snapshot.resources[1].unknown_pages, 1);
        let coverage = snapshot.targeted.expect("targeted coverage");
        assert_eq!(coverage.selected_resources, 1);
        assert_eq!(coverage.selected_resources_fully_observed, 1);
        assert_eq!(coverage.targeted_pages_sampled, 1);
        assert_eq!(coverage.general_pages_sampled, 0);
        assert_eq!(coverage.page_budget, 1);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn selected_name_without_a_range_is_not_observed() {
        let (path, address) = mapped_file("missing-selected", 1);
        let ranges = [range("general", &path, 0, 1)];
        let selected = BTreeSet::from(["ghost".to_string()]);
        let snapshot = observe_page_residency_targeted(
            &ranges,
            &selected,
            PageObservationConfig {
                max_pages: 1,
                max_resources: 1,
            },
        );
        unsafe { libc::munmap(address, page_size() as usize) };
        let _ = std::fs::remove_file(&path);
        let coverage = snapshot.targeted.expect("coverage");
        assert_eq!(coverage.selected_resources, 1);
        assert_eq!(coverage.selected_resources_not_observed, 1);
        assert_eq!(coverage.selected_resources_fully_observed, 0);
        assert_eq!(coverage.general_pages_sampled, 1);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn general_sampling_uses_only_remaining_budget() {
        let (path, address) = mapped_file("remaining", 3);
        let ranges = [
            range("b", &path, 1, 1),
            range("a", &path, 0, 1),
            range("c", &path, 2, 1),
        ];
        let selected = BTreeSet::from(["a".to_string()]);
        let snapshot = observe_page_residency_targeted(
            &ranges,
            &selected,
            PageObservationConfig {
                max_pages: 2,
                max_resources: 3,
            },
        );
        unsafe { libc::munmap(address, (page_size() * 3) as usize) };
        let _ = std::fs::remove_file(&path);
        let by_name = |name: &str| {
            snapshot
                .resources
                .iter()
                .find(|resource| resource.name == name)
                .expect("resource")
        };
        assert_eq!(by_name("a").inspected_pages, 1);
        assert_eq!(by_name("b").inspected_pages, 1);
        assert_eq!(by_name("c").inspected_pages, 0);
        assert_eq!(by_name("c").unknown_pages, 1);
        let coverage = snapshot.targeted.expect("coverage");
        assert_eq!(coverage.targeted_pages_sampled, 1);
        assert_eq!(coverage.general_pages_sampled, 1);
        assert_eq!(coverage.general_pages_unknown, 1);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn overlap_does_not_consume_page_budget_twice() {
        let (path, address) = mapped_file("overlap", 2);
        unsafe { std::ptr::read_volatile(address.cast::<u8>()) };
        let ranges = [range("a", &path, 0, 2), range("b", &path, 0, 2)];
        let selected = BTreeSet::from(["a".to_string(), "b".to_string()]);
        let snapshot = observe_page_residency_targeted(
            &ranges,
            &selected,
            PageObservationConfig {
                max_pages: 2,
                max_resources: 2,
            },
        );
        unsafe { libc::munmap(address, (page_size() * 2) as usize) };
        let _ = std::fs::remove_file(&path);
        assert_eq!(snapshot.inspected_pages, 2);
        assert_eq!(snapshot.total_pages, 2);
        assert_eq!(
            snapshot.unique_resident_pages + snapshot.unique_nonresident_pages,
            2
        );
        let coverage = snapshot.targeted.expect("coverage");
        assert_eq!(coverage.targeted_pages_available, 2);
        assert_eq!(coverage.targeted_pages_sampled, 2);
        assert_eq!(coverage.selected_resources_fully_observed, 2);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn selected_resource_larger_than_budget_is_partial_not_complete() {
        let (path, address) = mapped_file("partial", 4);
        unsafe { std::ptr::read_volatile(address.cast::<u8>()) };
        let ranges = [range("selected", &path, 0, 4)];
        let selected = BTreeSet::from(["selected".to_string()]);
        let snapshot = observe_page_residency_targeted(
            &ranges,
            &selected,
            PageObservationConfig {
                max_pages: 2,
                max_resources: 1,
            },
        );
        unsafe { libc::munmap(address, (page_size() * 4) as usize) };
        let _ = std::fs::remove_file(&path);
        let resource = &snapshot.resources[0];
        assert_eq!(resource.inspected_pages, 2);
        assert_eq!(resource.unknown_pages, 2);
        assert!(resource.sampled);
        let coverage = snapshot.targeted.expect("coverage");
        assert_eq!(coverage.selected_resources_partially_observed, 1);
        assert_eq!(coverage.selected_resources_fully_observed, 0);
        assert_eq!(coverage.targeted_pages_unknown, 2);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn zero_budget_and_missing_mappings_are_unknown() {
        let (path, address) = mapped_file("zero", 1);
        let ranges = [
            range("selected", &path, 0, 1),
            PageRange {
                name: "missing".to_string(),
                tensor_index: 1,
                generation: 1,
                mapping_index: None,
                mapping_path: None,
                page_start: 0,
                page_end: 4096,
                page_size: 4096,
                mapping_size: 4096,
            },
        ];
        let selected = BTreeSet::from(["selected".to_string(), "missing".to_string()]);
        let snapshot = observe_page_residency_targeted(
            &ranges,
            &selected,
            PageObservationConfig {
                max_pages: 0,
                max_resources: 2,
            },
        );
        unsafe { libc::munmap(address, page_size() as usize) };
        let _ = std::fs::remove_file(&path);
        assert_eq!(snapshot.inspected_pages, 0);
        assert_eq!(snapshot.unknown_pages, 2);
        let coverage = snapshot.targeted.expect("coverage");
        assert_eq!(coverage.selected_resources_not_observed, 2);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn mincore_is_bounded_and_does_not_invent_unknown_zeroes() {
        let (path, address) = mapped_file("mincore", 2);
        unsafe { std::ptr::read_volatile(address.cast::<u8>()) };
        let snapshot = observe_page_residency(
            &[range("x", &path, 0, 2)],
            PageObservationConfig {
                max_pages: 1,
                max_resources: 1,
            },
        );
        unsafe { libc::munmap(address, (page_size() * 2) as usize) };
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
    /// Staging was attempted but the selected pages demonstrably remained
    /// nonresident.
    Ineffective,
    /// The pages became resident but were not accessed within the observation
    /// window.
    Wasted,
    /// The pages became resident but a before-access observation showed they
    /// were nonresident again before the access.
    EvictedBeforeUse,
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
    pub ineffective: usize,
    pub wasted: usize,
    pub evicted_before_use: usize,
    pub benefit_unknown: usize,
    /// False unless a safe pre-access snapshot was actually captured.
    pub before_access_observation_available: bool,
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
            (PrefetchResidencyClass::RemainedNonResident, _) => PrefetchBenefitClass::Ineffective,
            _ => PrefetchBenefitClass::Unknown,
        };
        match benefit {
            PrefetchBenefitClass::ClearlyUseful => report.clearly_useful += 1,
            PrefetchBenefitClass::PotentiallyUseful => report.potentially_useful += 1,
            PrefetchBenefitClass::Redundant => report.redundant += 1,
            PrefetchBenefitClass::Ineffective => report.ineffective += 1,
            PrefetchBenefitClass::Wasted => report.wasted += 1,
            PrefetchBenefitClass::EvictedBeforeUse => report.evicted_before_use += 1,
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
    report.notes.push(
        "before-access residency was not observed; evicted-before-use remains unknown".to_string(),
    );
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
            targeted: None,
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
            targeted: false,
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
                resource("d", 0, 1, 0),
            ],
            false,
        );
        let after = snapshot(
            vec![
                resource("a", 2, 2, 0),
                resource("b", 1, 1, 0),
                resource("c", 1, 1, 1),
                resource("d", 0, 1, 0),
            ],
            false,
        );
        let observed = std::collections::BTreeSet::from(["a".to_string(), "b".to_string()]);
        let report = correlate_prefetch_residency(
            &[
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            &before,
            &after,
            Some(&observed),
        );
        assert_eq!(report.already_resident, 1);
        assert_eq!(report.became_resident, 1);
        assert_eq!(report.unknown, 1);
        assert_eq!(report.access_observed, 2);
        assert_eq!(report.access_not_observed, 2);
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
        assert_eq!(report.ineffective, 1);
        assert!(!report.before_access_observation_available);
        assert!(report
            .notes
            .iter()
            .any(|note| note.contains("before-access")));
    }
}
