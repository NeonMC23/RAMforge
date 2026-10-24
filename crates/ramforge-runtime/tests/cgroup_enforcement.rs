//! Process-level cgroup v2 enforcement checks.
//!
//! These tests create a real child cgroup below the caller's delegated user
//! subtree. They are ignored by default because they need cgroup v2 memory
//! delegation and `python3`; run them explicitly on a supported host.

#![cfg(target_os = "linux")]

use std::process::Command;

use ramforge_runtime::cgroup::MemoryCgroup;

fn spawn_in_cgroup(cgroup: &MemoryCgroup, bytes: usize) -> std::process::ExitStatus {
    let procs = cgroup.procs_path();
    let script = format!(
        "echo $$ > '{}'; exec python3 -c 'a=bytearray({bytes}); print(len(a))'",
        procs.display()
    );
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(script)
        .spawn()
        .expect("spawn child");
    child.wait().expect("wait child")
}

#[test]
#[ignore = "requires delegated cgroup v2 memory controller and python3"]
fn child_above_the_limit_is_oom_killed() {
    let cgroup = match MemoryCgroup::create(32 * 1024 * 1024) {
        Ok(cgroup) => cgroup,
        Err(error) => {
            eprintln!("skipping: {error}");
            return;
        }
    };
    let status = spawn_in_cgroup(&cgroup, 64 * 1024 * 1024);
    let stats = cgroup.stats().expect("read stats");
    let events = stats.events.expect("events");
    eprintln!("status={status:?} stats={stats:?}");

    assert!(!status.success(), "allocation above limit must not succeed");
    assert!(
        events.oom_kill > 0 || events.oom_group_kill > 0,
        "kernel must report an OOM kill: {events:?}"
    );
    let _ = cgroup.remove();
}

#[test]
#[ignore = "requires delegated cgroup v2 memory controller and python3"]
fn child_below_the_limit_succeeds_and_stays_under_the_cap() {
    let limit = 256 * 1024 * 1024;
    let cgroup = match MemoryCgroup::create(limit) {
        Ok(cgroup) => cgroup,
        Err(error) => {
            eprintln!("skipping: {error}");
            return;
        }
    };
    let status = spawn_in_cgroup(&cgroup, 32 * 1024 * 1024);
    let stats = cgroup.stats().expect("read stats");
    eprintln!("status={status:?} stats={stats:?}");

    assert!(status.success(), "allocation below limit should succeed");
    assert_eq!(stats.limit_bytes, Some(limit));
    assert!(
        stats.peak_bytes.is_some_and(|peak| peak <= limit),
        "charged peak must not exceed the limit: {stats:?}"
    );
    assert!(
        stats
            .events
            .is_none_or(|events| events.oom_kill == 0 && events.oom_group_kill == 0),
        "small allocation must not OOM: {stats:?}"
    );
    let _ = cgroup.remove();
}
