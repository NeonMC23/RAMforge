//! CLI-level checks for the enforced-budget surface.
//!
//! Real-model checks are ignored by default and use paths from the
//! environment. They run the built `ramforge` binary, so they also exercise
//! the supervisor re-exec path.

use std::process::Command;

fn ramforge() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ramforge"))
}

#[test]
fn help_lists_memory_control_flags() {
    let output = ramforge().args(["run", "--help"]).output().unwrap();
    assert!(output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    for flag in [
        "--enforce-ram",
        "--enforce-timeout",
        "--ctx",
        "--no-repack",
        "--no-prefetch",
        "--release-mapping-pages",
        "--inspect-layer",
        "--release-layer",
    ] {
        assert!(text.contains(flag), "help must list {flag}: {text}");
    }
}

#[test]
fn zero_enforcement_timeout_is_rejected_before_any_model_access() {
    let output = ramforge()
        .args([
            "run",
            "/nonexistent/model.gguf",
            "--ram",
            "64M",
            "--prompt",
            "hi",
            "--enforce-ram",
            "--enforce-timeout",
            "0",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let text = String::from_utf8_lossy(&output.stderr);
    assert!(
        text.contains("--enforce-timeout must be greater than zero"),
        "unexpected stderr: {text}"
    );
}

#[test]
fn no_prefetch_is_rejected_for_the_legacy_backend() {
    let output = ramforge()
        .args([
            "run",
            "/nonexistent/model.gguf",
            "--ram",
            "64M",
            "--prompt",
            "hi",
            "--no-prefetch",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let text = String::from_utf8_lossy(&output.stderr);
    assert!(
        text.contains("--no-prefetch is only supported with --backend forgecore"),
        "unexpected stderr: {text}"
    );
}

#[test]
fn release_mapping_pages_is_rejected_for_the_legacy_backend() {
    let output = ramforge()
        .args([
            "run",
            "/nonexistent/model.gguf",
            "--ram",
            "64M",
            "--prompt",
            "hi",
            "--release-mapping-pages",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let text = String::from_utf8_lossy(&output.stderr);
    assert!(
        text.contains("--release-mapping-pages is only supported with --backend forgecore"),
        "unexpected stderr: {text}"
    );
}

#[test]
fn layer_residency_flags_are_rejected_for_the_legacy_backend() {
    let output = ramforge()
        .args([
            "run",
            "/nonexistent/model.gguf",
            "--ram",
            "64M",
            "--prompt",
            "hi",
            "--release-layer",
            "0",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let text = String::from_utf8_lossy(&output.stderr);
    assert!(
        text.contains(
            "--inspect-layer and --release-layer are only supported with --backend forgecore"
        ),
        "unexpected stderr: {text}"
    );
}

#[test]
#[ignore = "requires cgroup v2 delegation, a real model, and libggml"]
fn enforced_real_model_run_reports_the_cgroup_peak() {
    let model = std::env::var("FORGE_TEST_MODEL").expect("set FORGE_TEST_MODEL");
    let output = ramforge()
        .args([
            "run",
            &model,
            "--ram",
            "512M",
            "--enforce-ram",
            "--ctx",
            "256",
            "--no-repack",
            "--prompt",
            "The capital of France is",
            "--max-tokens",
            "1",
            "--temperature",
            "0",
            "--backend",
            "forgecore",
            "--memory-report",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "enforced run failed: {stderr}");
    for expected in [
        "Memory enforcement: cgroup v2",
        "Peak charged:",
        "CPU weight repacking: disabled",
        "Result: exit 0",
    ] {
        assert!(
            stderr.contains(expected),
            "missing {expected:?} in: {stderr}"
        );
    }
}

#[test]
#[ignore = "requires cgroup v2 delegation, a slow-loading real model, and libggml"]
fn enforced_timeout_kills_the_cgroup_and_returns_124() {
    let model = std::env::var("FORGE_TIMEOUT_MODEL")
        .or_else(|_| std::env::var("FORGE_OOM_MODEL"))
        .expect("set FORGE_TIMEOUT_MODEL or FORGE_OOM_MODEL");
    let output = ramforge()
        .args([
            "run",
            &model,
            "--ram",
            "512M",
            "--enforce-ram",
            "--enforce-timeout",
            "1",
            "--ctx",
            "256",
            "--no-repack",
            "--prompt",
            "The capital of France is",
            "--max-tokens",
            "1",
            "--temperature",
            "0",
            "--backend",
            "forgecore",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(124), "stderr: {stderr}");
    assert!(
        stderr.contains("Result: timeout; cgroup killed"),
        "timeout must be reported: {stderr}"
    );
}

#[test]
#[ignore = "requires cgroup v2 delegation, a real model, and libggml"]
fn release_mapping_pages_reports_zero_after_advice() {
    let model = std::env::var("FORGE_TEST_MODEL").expect("set FORGE_TEST_MODEL");
    let output = ramforge()
        .args([
            "run",
            &model,
            "--ram",
            "192M",
            "--enforce-ram",
            "--ctx",
            "256",
            "--no-repack",
            "--release-mapping-pages",
            "--prompt",
            "The capital of France is",
            "--max-tokens",
            "1",
            "--temperature",
            "0",
            "--backend",
            "forgecore",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "run failed: {stderr}");
    assert!(
        stderr.contains("Mapping page release:") && stderr.contains("resident_after=0 bytes"),
        "release report missing: {stderr}"
    );
}

#[test]
#[ignore = "requires cgroup v2 delegation, a real model, and libggml"]
fn release_layer_reports_selected_tensor_ranges() {
    let model = std::env::var("FORGE_TEST_MODEL").expect("set FORGE_TEST_MODEL");
    let output = ramforge()
        .args([
            "run",
            &model,
            "--ram",
            "192M",
            "--enforce-ram",
            "--ctx",
            "256",
            "--no-repack",
            "--release-layer",
            "0",
            "--prompt",
            "The capital of France is",
            "--max-tokens",
            "1",
            "--temperature",
            "0",
            "--backend",
            "forgecore",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "run failed: {stderr}");
    for expected in [
        "Layer 0 mapping metadata:",
        "Layer 0 release:",
        "released=",
        "skipped_shared=",
    ] {
        assert!(stderr.contains(expected), "missing {expected:?}: {stderr}");
    }
}

#[test]
#[ignore = "requires cgroup v2 delegation, a model larger than the limit, and libggml"]
fn enforced_oom_is_a_failure_not_a_success() {
    let model = std::env::var("FORGE_OOM_MODEL").expect("set FORGE_OOM_MODEL");
    let output = ramforge()
        .args([
            "run",
            &model,
            "--ram",
            "128M",
            "--enforce-ram",
            "--ctx",
            "256",
            "--no-repack",
            "--prompt",
            "The capital of France is",
            "--max-tokens",
            "1",
            "--temperature",
            "0",
            "--backend",
            "forgecore",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(137), "stderr: {stderr}");
    assert!(
        stderr.contains("cgroup OOM kill") || stderr.contains("oom_kill="),
        "OOM must be reported: {stderr}"
    );
}
