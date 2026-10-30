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
        "--residency-policy",
        "--residency-target",
        "--residency-reserve",
        "--residency-report",
        "--residency-report-json",
        "--residency-profile-load",
        "--residency-profile-save",
        "--residency-prefetch",
        "--graph-trace",
        "--graph-trace-events",
        "--graph-trace-detail",
    ] {
        assert!(text.contains(flag), "help must list {flag}: {text}");
    }
}

#[test]
fn profile_command_inspects_versioned_profile() {
    let path =
        std::env::temp_dir().join(format!("ramforge-cli-profile-{}.json", std::process::id()));
    let profile = serde_json::json!({
        "schema_version": 1,
        "kind": "ramforge.execution_profile",
        "identity": {
            "model_path": "/tmp/reference.gguf",
            "model_size_bytes": 123,
            "model_modified_unix_ns": null,
            "inventory_generation": 1,
            "native_contract_version": 1
        },
        "profile": {
            "version": 1,
            "model_id": "/tmp/reference.gguf",
            "generation": 1,
            "complete_observations": 1,
            "incomplete_observations": 0,
            "last_node_count": 1,
            "last_tensor_count": 1,
            "access": {"blk.0.weight": {"first_order": 0, "last_order": 1, "observed_consumers": 1}},
            "invalidated_reason": null
        }
    });
    std::fs::write(&path, serde_json::to_vec(&profile).unwrap()).unwrap();
    let output = ramforge()
        .args(["profile", path.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&path);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "profile failed: {stdout}");
    let value: serde_json::Value = serde_json::from_str(&stdout).expect("profile JSON");
    assert_eq!(value["valid"], true);
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["entries"], 1);
    assert_eq!(value["model_identity_validated"], false);
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
fn residency_policy_is_rejected_for_the_legacy_backend() {
    let output = ramforge()
        .args([
            "run",
            "/nonexistent/model.gguf",
            "--ram",
            "64M",
            "--prompt",
            "hi",
            "--residency-policy",
            "observe",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let text = String::from_utf8_lossy(&output.stderr);
    assert!(
        text.contains("residency policy, target, reserve, prefetch, and report controls are only supported with --backend forgecore"),
        "unexpected stderr: {text}"
    );
}

#[test]
fn residency_target_requires_policy_or_report() {
    let output = ramforge()
        .args([
            "run",
            "/nonexistent/model.gguf",
            "--ram",
            "64M",
            "--prompt",
            "hi",
            "--residency-target",
            "32M",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let text = String::from_utf8_lossy(&output.stderr);
    assert!(
        text.contains("require --residency-policy"),
        "unexpected stderr: {text}"
    );
}

#[test]
fn residency_reserve_cannot_exceed_target() {
    let output = ramforge()
        .args([
            "run",
            "/nonexistent/model.gguf",
            "--ram",
            "64M",
            "--prompt",
            "hi",
            "--residency-policy",
            "observe",
            "--residency-target",
            "32M",
            "--residency-reserve",
            "64M",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let text = String::from_utf8_lossy(&output.stderr);
    assert!(
        text.contains("exceeds --residency-target"),
        "unexpected stderr: {text}"
    );
}

#[test]
fn graph_trace_is_rejected_for_the_legacy_backend() {
    let output = ramforge()
        .args([
            "run",
            "/nonexistent/model.gguf",
            "--ram",
            "64M",
            "--prompt",
            "hi",
            "--graph-trace",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let text = String::from_utf8_lossy(&output.stderr);
    assert!(
        text.contains("--graph-trace is only supported with --backend forgecore"),
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
#[ignore = "requires cgroup v2 delegation, a real model, and libggml"]
fn bounded_prefetch_is_reported_separately_from_residency() {
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
            "--no-prefetch",
            "--residency-prefetch",
            "16M",
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
        stderr.contains("Residency staging plan:"),
        "missing plan: {stderr}"
    );
    assert!(
        stderr.contains("native_requested=") && stderr.contains("mapping_before="),
        "missing staging result: {stderr}"
    );
}

#[test]
#[ignore = "requires cgroup v2 delegation, a real model, and libggml"]
fn graph_trace_reports_manifest_without_claiming_eviction() {
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
            "--graph-trace",
            "--graph-trace-events",
            "256",
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
        stderr.contains("Graph trace summary:"),
        "missing trace: {stderr}"
    );
    assert!(
        stderr.contains("Graph manifest:"),
        "missing manifest: {stderr}"
    );
    assert!(
        stderr.contains("intra_decode_safe=false"),
        "trace must not claim intra-decode safety: {stderr}"
    );
}

#[test]
#[ignore = "requires cgroup v2 delegation, a real model, and libggml"]
fn observe_policy_reports_without_applying() {
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
            "--residency-policy",
            "observe",
            "--residency-target",
            "64M",
            "--residency-report",
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
        stderr.contains("Residency report:"),
        "missing report: {stderr}"
    );
    assert!(
        stderr.contains("Application: none"),
        "observe applied: {stderr}"
    );
}

#[test]
#[ignore = "requires cgroup v2 delegation, a real model, and libggml"]
fn json_report_is_versioned_and_keeps_measurements_distinct() {
    let model = std::env::var("FORGE_TEST_MODEL").expect("set FORGE_TEST_MODEL");
    let path = std::env::temp_dir().join(format!(
        "ramforge-residency-report-{}.json",
        std::process::id()
    ));
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
            "--residency-policy",
            "observe",
            "--residency-target",
            "64M",
            "--residency-report-json",
            path.to_str().unwrap(),
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
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("json report must exist"))
            .expect("valid json");
    let _ = std::fs::remove_file(&path);
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["kind"], "ramforge.residency.operation_report");
    assert!(report["ledger"]["entries"].is_array());
    assert!(report["reconciliation"]["mapped_virtual_bytes"].is_u64());
    assert!(report["measurements"]["before"]["process_rss_bytes"].is_u64());
    assert_eq!(report["application"], serde_json::Value::Null);
    assert_eq!(report["native_lifetime"]["contract_version"], 1);
    assert!(report["native_lifetime"]["model_instance_id"].is_u64());
    assert_eq!(report["native_lifetime"]["quiescent"], true);
    assert!(report["native_lifetime"]["context_id"].is_u64());
    assert!(report["resource_lifetime_states"].is_object());
    assert!(report["admission"]["eligible_reclaimable_bytes"].is_u64());
    assert_eq!(report["admission"]["native_contract_version"], 1);
    assert!(report["profile_source"].is_string());
    assert!(report["timing"]["ttft_ns"].is_u64());
    assert!(report["timing"]["output_fingerprint"].is_string());
    assert!(report["ledger"]["unknown_components"]
        .as_array()
        .is_some_and(|components| !components.is_empty()));
}

#[test]
#[ignore = "requires cgroup v2 delegation, a real model, and libggml"]
fn profile_save_then_validate_has_matching_model_identity() {
    let model = std::env::var("FORGE_TEST_MODEL").expect("set FORGE_TEST_MODEL");
    let path =
        std::env::temp_dir().join(format!("ramforge-profile-save-{}.json", std::process::id()));
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
            "--graph-trace",
            "--graph-trace-events",
            "16384",
            "--residency-policy",
            "reuse-aware",
            "--residency-target",
            "128M",
            "--residency-profile-save",
            path.to_str().unwrap(),
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
    assert!(output.status.success(), "save run failed: {stderr}");
    assert!(
        stderr.contains("Residency profile saved:"),
        "missing save: {stderr}"
    );

    let validate = ramforge()
        .args([
            "profile",
            path.to_str().unwrap(),
            "--model",
            &model,
            "--json",
        ])
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&path);
    let stdout = String::from_utf8_lossy(&validate.stdout);
    assert!(validate.status.success(), "validate failed: {stdout}");
    let value: serde_json::Value = serde_json::from_str(&stdout).expect("validate JSON");
    assert_eq!(value["valid"], true);
    assert_eq!(value["model_identity_validated"], true);
}

#[test]
#[ignore = "requires cgroup v2 delegation, a real model, and libggml"]
fn reuse_aware_reports_profile_without_intra_decode_claims() {
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
            "--graph-trace",
            "--residency-policy",
            "reuse-aware",
            "--residency-target",
            "64M",
            "--residency-report",
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
        stderr.contains("Execution profile:"),
        "missing profile: {stderr}"
    );
    assert!(stderr.contains("profile_used=true") || stderr.contains("Execution profile:"));
    assert!(
        stderr.contains("intra_decode_safe=false"),
        "unsafe claim: {stderr}"
    );
}

#[test]
#[ignore = "requires cgroup v2 delegation, a real model, and libggml"]
fn active_policy_applies_release_at_post_operation_boundary() {
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
            "--residency-policy",
            "active",
            "--residency-target",
            "32M",
            "--residency-report",
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
        stderr.contains("Residency report:"),
        "missing report: {stderr}"
    );
    assert!(
        stderr.contains("Application:"),
        "missing application: {stderr}"
    );
    assert!(
        stderr.contains("native: released="),
        "no native release: {stderr}"
    );
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
