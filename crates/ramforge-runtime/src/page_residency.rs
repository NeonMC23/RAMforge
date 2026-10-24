//! Measured resident pages for a file-backed model mapping.
//!
//! This is observation only. It does not evict pages or claim that RSS is
//! capped. The kernel owns reclaim for clean file-backed mappings.

use std::path::Path;

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
