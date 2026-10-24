//! Unprivileged cgroup v2 memory-budget enforcement.
//!
//! The Linux kernel can cap charged memory for a process group. File-backed
//! pages that belong to a model mapping are reclaimable, so a limit smaller
//! than the model file can still run, at the cost of page faults and disk I/O.
//! This module only creates a child cgroup below the caller's own delegated
//! subtree; it never modifies global limits or swap configuration.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const CGROUP_ROOT: &str = "/sys/fs/cgroup";
static NONCE: AtomicU64 = AtomicU64::new(0);

/// Errors from the cgroup v2 budget path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CgroupError {
    /// The host does not expose cgroup v2 at the expected mount.
    Unsupported(String),
    /// No delegated ancestor accepts a child cgroup with the memory controller.
    NoDelegatedParent(String),
    /// The requested limit is too small to be meaningful.
    InvalidLimit(u64),
    /// A filesystem operation failed.
    Io(String),
}

impl std::fmt::Display for CgroupError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(message) => write!(formatter, "cgroup v2 unavailable: {message}"),
            Self::NoDelegatedParent(message) => {
                write!(formatter, "no delegated cgroup v2 parent: {message}")
            }
            Self::InvalidLimit(bytes) => {
                write!(formatter, "cgroup memory limit {bytes} bytes is too small")
            }
            Self::Io(message) => write!(formatter, "cgroup I/O error: {message}"),
        }
    }
}

impl std::error::Error for CgroupError {}

/// Parsed `memory.events` counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CgroupEvents {
    pub low: u64,
    pub high: u64,
    pub max: u64,
    pub oom: u64,
    pub oom_kill: u64,
    pub oom_group_kill: u64,
}

/// Measured cgroup facts after a run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CgroupStats {
    pub limit_bytes: Option<u64>,
    pub peak_bytes: Option<u64>,
    pub current_bytes: Option<u64>,
    pub swap_peak_bytes: Option<u64>,
    pub events: Option<CgroupEvents>,
    pub anon_bytes: Option<u64>,
    pub file_bytes: Option<u64>,
    pub shmem_bytes: Option<u64>,
    pub file_mapped_bytes: Option<u64>,
    pub kernel_bytes: Option<u64>,
    pub pgfault: Option<u64>,
    pub pgmajfault: Option<u64>,
}

impl CgroupStats {
    /// True when the kernel reported a limit breach or an OOM event.
    pub fn exceeded(&self) -> bool {
        self.events
            .is_some_and(|events| events.max > 0 || events.oom > 0 || events.oom_kill > 0)
    }
}

/// A child cgroup with `memory.max` and `memory.swap.max=0`.
#[derive(Debug)]
pub struct MemoryCgroup {
    path: PathBuf,
    limit_bytes: u64,
}

impl MemoryCgroup {
    /// Create a child cgroup under the deepest usable delegated ancestor.
    pub fn create(limit_bytes: u64) -> Result<Self, CgroupError> {
        if limit_bytes < 1024 * 1024 {
            return Err(CgroupError::InvalidLimit(limit_bytes));
        }
        let root = Path::new(CGROUP_ROOT);
        if !root.join("cgroup.controllers").exists() {
            return Err(CgroupError::Unsupported(format!(
                "{CGROUP_ROOT} is not a cgroup v2 mount"
            )));
        }
        let self_path =
            read_self_cgroup_path().map_err(|error| CgroupError::Io(error.to_string()))?;
        let parent = discover_delegated_parent(root, &self_path)?;
        Self::create_in(&parent, limit_bytes)
    }

    fn create_in(parent: &Path, limit_bytes: u64) -> Result<Self, CgroupError> {
        let name = unique_name();
        let path = parent.join(name);
        fs::create_dir(&path)
            .map_err(|error| CgroupError::Io(format!("create {}: {error}", path.display())))?;

        let result = (|| -> Result<(), CgroupError> {
            write_limit(&path.join("memory.max"), limit_bytes)?;
            let swap = path.join("memory.swap.max");
            if swap.exists() {
                write_limit(&swap, 0)?;
            }
            let oom_group = path.join("memory.oom.group");
            if oom_group.exists() {
                fs::write(&oom_group, b"1").map_err(|error| {
                    CgroupError::Io(format!("write {}: {error}", oom_group.display()))
                })?;
            }
            Ok(())
        })();

        if let Err(error) = result {
            let _ = fs::remove_dir(&path);
            return Err(error);
        }

        Ok(Self { path, limit_bytes })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn limit_bytes(&self) -> u64 {
        self.limit_bytes
    }

    /// Path a process writes its own PID to in order to join this cgroup.
    pub fn procs_path(&self) -> PathBuf {
        self.path.join("cgroup.procs")
    }

    /// Move a process into this cgroup.
    pub fn attach_pid(&self, pid: u32) -> io::Result<()> {
        fs::write(self.procs_path(), pid.to_string())
    }

    /// Ask the kernel to kill every process in the cgroup.
    pub fn kill(&self) -> io::Result<()> {
        let kill = self.path.join("cgroup.kill");
        if kill.exists() {
            fs::write(kill, b"1")
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "cgroup.kill is unavailable",
            ))
        }
    }

    /// Read the measured facts while the cgroup still exists.
    pub fn stats(&self) -> io::Result<CgroupStats> {
        stats_at(&self.path)
    }

    /// Remove the cgroup after its processes have exited.
    pub fn remove(self) -> io::Result<()> {
        remove_best_effort(&self.path)
    }
}

impl Drop for MemoryCgroup {
    fn drop(&mut self) {
        let _ = remove_best_effort(&self.path);
    }
}

/// Read the cgroup v2 path of the current process.
pub fn read_self_cgroup_path() -> io::Result<String> {
    let text = fs::read_to_string("/proc/self/cgroup")?;
    parse_self_cgroup_path(&text)
        .map(ToOwned::to_owned)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no cgroup v2 entry"))
}

/// Filesystem path of the current process's cgroup v2 directory.
pub fn self_cgroup_path() -> io::Result<PathBuf> {
    let path = read_self_cgroup_path()?;
    Ok(Path::new(CGROUP_ROOT).join(path.trim_start_matches('/')))
}

/// Read measured facts from an existing cgroup directory.
pub fn stats_at(path: &Path) -> io::Result<CgroupStats> {
    let stat = fs::read_to_string(path.join("memory.stat")).ok();
    let events = fs::read_to_string(path.join("memory.events"))
        .ok()
        .map(|text| parse_memory_events(&text));
    Ok(CgroupStats {
        limit_bytes: read_u64(path.join("memory.max")).filter(|value| *value > 0),
        peak_bytes: read_u64(path.join("memory.peak")),
        current_bytes: read_u64(path.join("memory.current")),
        swap_peak_bytes: read_u64(path.join("memory.swap.peak")),
        events,
        anon_bytes: stat.as_deref().and_then(|text| stat_value(text, "anon")),
        file_bytes: stat.as_deref().and_then(|text| stat_value(text, "file")),
        shmem_bytes: stat.as_deref().and_then(|text| stat_value(text, "shmem")),
        file_mapped_bytes: stat
            .as_deref()
            .and_then(|text| stat_value(text, "file_mapped")),
        kernel_bytes: stat.as_deref().and_then(|text| stat_value(text, "kernel")),
        pgfault: stat.as_deref().and_then(|text| stat_value(text, "pgfault")),
        pgmajfault: stat
            .as_deref()
            .and_then(|text| stat_value(text, "pgmajfault")),
    })
}

/// Parse the `0::/path` line from `/proc/self/cgroup`.
pub fn parse_self_cgroup_path(input: &str) -> Option<&str> {
    input
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .map(str::trim)
        .filter(|path| path.starts_with('/'))
}

/// Find the deepest ancestor of `self_path` that accepts a memory child cgroup.
pub fn discover_delegated_parent(root: &Path, self_path: &str) -> Result<PathBuf, CgroupError> {
    let mut candidate = root.join(self_path.trim_start_matches('/'));
    while candidate != root && candidate.starts_with(root) {
        if is_usable_parent(&candidate) {
            return Ok(candidate);
        }
        if !candidate.pop() {
            break;
        }
    }
    Err(CgroupError::NoDelegatedParent(format!(
        "walked from {} to {}",
        self_path,
        root.display()
    )))
}

fn is_usable_parent(candidate: &Path) -> bool {
    let controllers = fs::read_to_string(candidate.join("cgroup.controllers")).unwrap_or_default();
    if !controllers.split_whitespace().any(|item| item == "memory") {
        return false;
    }
    let subtree = fs::read_to_string(candidate.join("cgroup.subtree_control")).unwrap_or_default();
    if !subtree.split_whitespace().any(|item| item == "memory") {
        return false;
    }
    // A directory that cannot accept children is not a usable parent.
    let probe = candidate.join(unique_name());
    match fs::create_dir(&probe) {
        Ok(()) => {
            let _ = fs::remove_dir(&probe);
            true
        }
        Err(_) => false,
    }
}

fn unique_name() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let serial = NONCE.fetch_add(1, Ordering::Relaxed);
    format!(
        "ramforge-{}-{}-{}",
        std::process::id(),
        serial,
        nanos % 1_000_000
    )
}

fn write_limit(path: &Path, value: u64) -> Result<(), CgroupError> {
    fs::write(path, value.to_string())
        .map_err(|error| CgroupError::Io(format!("write {}: {error}", path.display())))
}

fn read_u64(path: PathBuf) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn stat_value(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let mut parts = line.split_whitespace();
        if parts.next() != Some(key) {
            return None;
        }
        parts.next()?.parse().ok()
    })
}

/// Parse `memory.events` into counters.
pub fn parse_memory_events(input: &str) -> CgroupEvents {
    let mut events = CgroupEvents::default();
    for line in input.lines() {
        let mut parts = line.split_whitespace();
        let Some(key) = parts.next() else { continue };
        let Some(value) = parts.next().and_then(|value| value.parse().ok()) else {
            continue;
        };
        match key {
            "low" => events.low = value,
            "high" => events.high = value,
            "max" => events.max = value,
            "oom" => events.oom = value,
            "oom_kill" => events.oom_kill = value,
            "oom_group_kill" => events.oom_group_kill = value,
            _ => {}
        }
    }
    events
}

fn remove_best_effort(path: &Path) -> io::Result<()> {
    // The kernel may need a moment to release the final reference.
    let mut last = None;
    for _ in 0..20 {
        match fs::remove_dir(path) {
            Ok(()) => return Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                last = Some(error);
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
    Err(last.unwrap_or_else(|| io::Error::other("cgroup removal failed")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn parses_cgroup_v2_path() {
        let input = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/x.scope\n";
        assert_eq!(
            parse_self_cgroup_path(input),
            Some("/user.slice/user-1000.slice/user@1000.service/app.slice/x.scope")
        );
        assert_eq!(parse_self_cgroup_path("1:name=systemd:/legacy\n"), None);
    }

    #[test]
    fn parses_memory_events() {
        let events = parse_memory_events(
            "low 1\nhigh 2\nmax 3\noom 4\noom_kill 5\noom_group_kill 6\nextra 7\n",
        );
        assert_eq!(
            events,
            CgroupEvents {
                low: 1,
                high: 2,
                max: 3,
                oom: 4,
                oom_kill: 5,
                oom_group_kill: 6,
            }
        );
        assert!(CgroupStats {
            events: Some(events),
            ..Default::default()
        }
        .exceeded());
    }

    #[test]
    fn parses_selected_memory_stat_fields() {
        let text = "anon 100\nfile 200\nkernel 300\npgfault 400\npgmajfault 5\n";
        assert_eq!(stat_value(text, "anon"), Some(100));
        assert_eq!(stat_value(text, "file"), Some(200));
        assert_eq!(stat_value(text, "kernel"), Some(300));
        assert_eq!(stat_value(text, "pgfault"), Some(400));
        assert_eq!(stat_value(text, "pgmajfault"), Some(5));
        assert_eq!(stat_value(text, "missing"), None);
    }

    #[test]
    fn discovers_deepest_usable_parent_in_a_fake_tree() {
        let root = tempfile::tempdir().unwrap();
        let root_path = root.path();
        let leaf = root_path.join("user.slice/user-1000.slice/app.slice/leaf.scope");
        fs::create_dir_all(&leaf).unwrap();
        fs::write(root_path.join("cgroup.controllers"), "memory\n").unwrap();
        fs::write(root_path.join("cgroup.subtree_control"), "memory\n").unwrap();
        fs::write(root_path.join("user.slice/cgroup.controllers"), "memory\n").unwrap();
        fs::write(
            root_path.join("user.slice/cgroup.subtree_control"),
            "memory\n",
        )
        .unwrap();
        let app = root_path.join("user.slice/user-1000.slice/app.slice");
        fs::write(app.join("cgroup.controllers"), "memory\n").unwrap();
        fs::write(app.join("cgroup.subtree_control"), "memory\n").unwrap();
        fs::write(leaf.join("cgroup.controllers"), "memory\n").unwrap();
        fs::write(leaf.join("cgroup.subtree_control"), "").unwrap();

        let found = discover_delegated_parent(
            root_path,
            "/user.slice/user-1000.slice/app.slice/leaf.scope",
        )
        .unwrap();
        assert_eq!(found, app);
    }

    #[test]
    fn rejects_tiny_limits_before_touching_the_kernel() {
        assert_eq!(
            MemoryCgroup::create(1024).unwrap_err(),
            CgroupError::InvalidLimit(1024)
        );
    }
}
