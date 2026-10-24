//! Re-execute the current command under a cgroup v2 memory limit.
//!
//! The outer process creates the cgroup and supervises the child. The child
//! is the same binary with `RAMFORGE_SUPERVISED=1`; it runs the normal command
//! path and reports its own measured state. The outer process prints the
//! authoritative post-exit peak, events, and exit status.

use std::ffi::CString;
use std::process::{Command, ExitStatus};
use std::time::{Duration, Instant};

use ramforge_runtime::cgroup::{
    self_cgroup_path, stats_at, CgroupError, CgroupStats, MemoryCgroup,
};

pub const SUPERVISED_ENV: &str = "RAMFORGE_SUPERVISED";
pub const BUDGET_ENV: &str = "RAMFORGE_ENFORCED_BUDGET_BYTES";
pub const CGROUP_ENV: &str = "RAMFORGE_CGROUP_DIR";

/// Outcome of a supervised run.
#[derive(Debug, Clone)]
pub struct SupervisionOutcome {
    pub exit_code: i32,
    pub timed_out: bool,
    pub wall: Duration,
    pub stats: CgroupStats,
    pub exit_status: ExitStatus,
    pub cgroup_path: std::path::PathBuf,
}

impl SupervisionOutcome {
    pub fn oom_killed(&self) -> bool {
        self.stats
            .events
            .is_some_and(|events| events.oom_kill > 0 || events.oom_group_kill > 0)
    }
}

/// True when this process is already the supervised child.
pub fn is_supervised() -> bool {
    std::env::var_os(SUPERVISED_ENV).is_some()
}

/// Budget bytes exported by the supervisor for the child.
pub fn enforced_budget_bytes() -> Option<u64> {
    std::env::var(BUDGET_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
}

/// Measured cgroup facts for the current process, when it is in a cgroup.
pub fn current_cgroup_stats() -> Option<CgroupStats> {
    let path = self_cgroup_path().ok()?;
    stats_at(&path).ok()
}

/// Create a limited cgroup, re-exec this command under it, and wait.
#[cfg(unix)]
pub fn run_supervised(budget_bytes: u64, timeout: Duration) -> Result<SupervisionOutcome, String> {
    if is_supervised() {
        return Err("refusing to nest a supervised run".to_string());
    }
    let cgroup = MemoryCgroup::create(budget_bytes).map_err(format_cgroup_error)?;
    let procs = cgroup.procs_path();
    #[cfg(unix)]
    let procs_bytes = {
        use std::os::unix::ffi::OsStrExt;
        procs.as_os_str().as_bytes()
    };
    let procs_c =
        CString::new(procs_bytes).map_err(|_| "cgroup.procs path contains a NUL".to_string())?;

    let executable = std::env::current_exe()
        .map_err(|error| format!("cannot locate current executable: {error}"))?;
    let mut command = Command::new(executable);
    command.args(std::env::args_os().skip(1));
    command.env(SUPERVISED_ENV, "1");
    command.env(BUDGET_ENV, budget_bytes.to_string());
    command.env(CGROUP_ENV, cgroup.path());

    // Move the child into the cgroup before exec, so no model or runtime
    // allocation happens outside the measured limit.
    //
    // SAFETY: the closure only calls async-signal-safe libc functions and
    // writes a PID into a pre-opened path. It allocates nothing.
    unsafe {
        use std::os::unix::process::CommandExt;
        command.pre_exec(move || attach_self_to_cgroup(&procs_c));
    }

    let started = Instant::now();
    let mut child = command
        .spawn()
        .map_err(|error| format!("failed to start supervised command: {error}"))?;

    let mut timed_out = false;
    let exit_status = loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("failed to poll supervised command: {error}"))?
        {
            break status;
        }
        if started.elapsed() >= timeout {
            timed_out = true;
            let _ = cgroup.kill();
            break wait_after_kill(&mut child)?;
        }
        std::thread::sleep(Duration::from_millis(50));
    };

    let stats = cgroup.stats().unwrap_or_default();
    let cgroup_path = cgroup.path().to_path_buf();
    let _ = cgroup.remove();
    let exit_code = exit_code(&exit_status, timed_out);
    Ok(SupervisionOutcome {
        exit_code,
        timed_out,
        wall: started.elapsed(),
        stats,
        exit_status,
        cgroup_path,
    })
}

#[cfg(not(unix))]
pub fn run_supervised(
    _budget_bytes: u64,
    _timeout: Duration,
) -> Result<SupervisionOutcome, String> {
    Err("cgroup supervision is only implemented on Unix".to_string())
}

#[cfg(unix)]
fn attach_self_to_cgroup(procs: &CString) -> std::io::Result<()> {
    let pid = unsafe { libc::getpid() };
    let mut digits = [0u8; 24];
    let mut index = digits.len();
    let mut value = pid as u32;
    loop {
        index -= 1;
        digits[index] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    index -= 1;
    digits[index] = b'\n';

    let fd = unsafe { libc::open(procs.as_ptr(), libc::O_WRONLY | libc::O_CLOEXEC) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let bytes = &digits[index..];
    let written = unsafe {
        libc::write(
            fd,
            bytes.as_ptr().cast::<libc::c_void>(),
            bytes.len() as libc::size_t,
        )
    };
    let close_result = unsafe { libc::close(fd) };
    if written != bytes.len() as isize {
        return Err(std::io::Error::last_os_error());
    }
    if close_result != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(unix)]
fn wait_after_kill(child: &mut std::process::Child) -> Result<ExitStatus, String> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("failed to poll after cgroup kill: {error}"))?
        {
            return Ok(status);
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            return child
                .wait()
                .map_err(|error| format!("failed to reap supervised command: {error}"));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn exit_code(status: &ExitStatus, timed_out: bool) -> i32 {
    if timed_out {
        return 124;
    }
    if let Some(code) = status.code() {
        return code;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        status.signal().map_or(1, |signal| 128 + signal)
    }
    #[cfg(not(unix))]
    {
        1
    }
}

fn format_cgroup_error(error: CgroupError) -> String {
    match error {
        CgroupError::Unsupported(detail) => format!(
            "--enforce-ram requires cgroup v2: {detail}. Run without --enforce-ram for diagnostic-only reporting"
        ),
        CgroupError::NoDelegatedParent(detail) => format!(
            "--enforce-ram could not find a delegated cgroup v2 parent ({detail}). The user manager must delegate the memory controller"
        ),
        other => other.to_string(),
    }
}

/// Print the authoritative enforcement report after the child exits.
pub fn print_report(outcome: &SupervisionOutcome, budget_bytes: u64) {
    let stats = &outcome.stats;
    eprintln!("Memory enforcement: cgroup v2");
    eprintln!(
        "Requested budget: {} bytes (limit {})",
        budget_bytes,
        stats
            .limit_bytes
            .map(|bytes| format!("{bytes} bytes"))
            .unwrap_or_else(|| "unavailable".to_string())
    );
    eprintln!(
        "Peak charged: {}",
        stats
            .peak_bytes
            .map(|bytes| format!("{bytes} bytes"))
            .unwrap_or_else(|| "unavailable".to_string())
    );
    eprintln!(
        "Peak swap: {}",
        stats
            .swap_peak_bytes
            .map(|bytes| format!("{bytes} bytes"))
            .unwrap_or_else(|| "unavailable".to_string())
    );
    if let Some(events) = stats.events {
        eprintln!(
            "Limit events: max={} high={} oom={} oom_kill={} oom_group_kill={}",
            events.max, events.high, events.oom, events.oom_kill, events.oom_group_kill
        );
    }
    eprintln!(
        "Page faults: {} (major {})",
        stats
            .pgfault
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unavailable".to_string()),
        stats
            .pgmajfault
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unavailable".to_string())
    );
    eprintln!("Wall: {:.3} s", outcome.wall.as_secs_f64());
    if outcome.timed_out {
        eprintln!("Result: timeout; cgroup killed");
    } else if outcome.oom_killed() {
        eprintln!("Result: cgroup OOM kill; the requested budget was not sufficient");
    } else {
        eprintln!("Result: exit {}", outcome.exit_code);
    }
    eprintln!("Cgroup: {}", outcome.cgroup_path.display());
}

/// The cgroup path from the current environment, for child diagnostics.
pub fn current_cgroup_dir() -> Option<std::path::PathBuf> {
    std::env::var_os(CGROUP_ENV).map(std::path::PathBuf::from)
}

/// Read the current child's cgroup facts for its own report.
pub fn child_report_stats() -> Option<CgroupStats> {
    if !is_supervised() {
        return None;
    }
    current_cgroup_stats()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_code_maps_signals_and_timeouts() {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            let signal = ExitStatus::from_raw(9);
            assert_eq!(exit_code(&signal, false), 137);
        }
        let ok = Command::new("true").status().unwrap();
        assert_eq!(exit_code(&ok, false), 0);
        assert_eq!(exit_code(&ok, true), 124);
    }

    #[test]
    fn cgroup_error_message_names_the_flag() {
        let message = format_cgroup_error(CgroupError::Unsupported("missing".to_string()));
        assert!(message.contains("--enforce-ram"));
        assert!(message.contains("diagnostic-only"));
    }
}
