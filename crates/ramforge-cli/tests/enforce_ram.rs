//! CLI-level checks for the enforced-budget surface.
//!
//! Real-model checks are ignored by default and use paths from the
//! environment. They run the built `ramforge` binary, so they also exercise
//! the supervisor re-exec path.

use std::process::Command;

use ramforge_runtime::SchedulerObservation;

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
        "--prediction",
        "--prediction-min-confidence",
        "--prediction-window",
        "--prediction-speculative",
        "--prediction-max-inflight",
        "--prediction-history-load",
        "--prediction-history-save",
        "--residency-observe",
        "--residency-observe-pages",
        "--residency-prefetch",
        "--graph-trace",
        "--graph-trace-events",
        "--graph-trace-detail",
        "--node-dispatch-observe",
        "--node-dispatch-events",
        "--node-dispatch-sample-pages",
        "--node-dispatch-sample-budget",
        "--node-dispatch-sample-strategy",
        "--node-dispatch-cache-capacity",
        "--node-dispatch-cache-ttl",
        "--node-dispatch-force-fresh",
        "--node-dispatch-detail",
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
fn prediction_requires_profile_and_rejects_bad_bounds() {
    let output = ramforge()
        .args([
            "run",
            "/nonexistent/model.gguf",
            "--ram",
            "64M",
            "--prompt",
            "hi",
            "--backend",
            "forgecore",
            "--prediction",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("--prediction requires --residency-profile-load"));

    let output = ramforge()
        .args([
            "run",
            "/nonexistent/model.gguf",
            "--ram",
            "64M",
            "--prompt",
            "hi",
            "--backend",
            "forgecore",
            "--prediction",
            "--residency-profile-load",
            "/tmp/profile.json",
            "--prediction-min-confidence",
            "101",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("min confidence must be between 0 and 10000"));
}

#[test]
fn simulate_command_is_deterministic_and_budget_safe() {
    let first = ramforge()
        .args(["simulate", "--budget", "12K", "--window", "8K", "--json"])
        .output()
        .unwrap();
    let second = ramforge()
        .args(["simulate", "--budget", "12K", "--window", "8K", "--json"])
        .output()
        .unwrap();
    assert!(first.status.success());
    assert_eq!(first.stdout, second.stdout);
    let value: serde_json::Value = serde_json::from_slice(&first.stdout).expect("simulation JSON");
    assert_eq!(value["simulation_only"], true);
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["scenarios"][0]["scenario"], "useful");
    assert_eq!(value["scenarios"][0]["predictive"]["budget_violations"], 0);
    assert!(
        value["scenarios"][0]["predictive"]["net_benefit_ns"]
            .as_i64()
            .unwrap()
            > 0
    );
    assert_eq!(value["scenarios"][0]["predictive"]["improved"], true);
    let all = ramforge()
        .args([
            "simulate",
            "--budget",
            "12K",
            "--window",
            "8K",
            "--scenario",
            "all",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(all.status.success());
    let all_value: serde_json::Value = serde_json::from_slice(&all.stdout).expect("all scenarios");
    assert_eq!(all_value["scenarios"].as_array().unwrap().len(), 16);
    let observation = ramforge()
        .args([
            "simulate",
            "--budget",
            "12K",
            "--window",
            "8K",
            "--scenario",
            "observation-overhead",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(observation.status.success());
    let observation_value: serde_json::Value =
        serde_json::from_slice(&observation.stdout).expect("observation simulation");
    let predictive = &observation_value["scenarios"][0]["predictive"];
    assert_eq!(predictive["observation_mode"], "targeted");
    assert_eq!(predictive["observation_page_budget"], 4096);
    assert!(predictive["targeted_pages_sampled"].as_u64().unwrap() > 0);
    assert!(predictive["observation_cost_ns"].as_u64().unwrap() > 0);
    assert_eq!(predictive["before_access_observation_available"], false);
    assert_eq!(predictive["improved"], false);
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
    assert!(report["process_faults"]["after_minor"].is_u64());
    assert!(
        report["execution_timing"]["end_to_end_ns"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(report["execution_timing"]["inference_ns"].as_u64().unwrap() > 0);
    assert!(report["residency_observation"]["before"].is_null());
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
fn prediction_staging_is_bounded_and_reports_feedback() {
    let model = std::env::var("FORGE_TEST_MODEL").expect("set FORGE_TEST_MODEL");
    let profile_path = std::env::temp_dir().join(format!(
        "ramforge-prediction-profile-{}.json",
        std::process::id()
    ));
    let history_path = std::env::temp_dir().join(format!(
        "ramforge-prediction-history-{}.json",
        std::process::id()
    ));
    let report_path = std::env::temp_dir().join(format!(
        "ramforge-prediction-report-{}.json",
        std::process::id()
    ));
    let save = ramforge()
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
            profile_path.to_str().unwrap(),
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
    assert!(
        save.status.success(),
        "profile save failed: {}",
        String::from_utf8_lossy(&save.stderr)
    );

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
            "observe",
            "--residency-target",
            "128M",
            "--residency-profile-load",
            profile_path.to_str().unwrap(),
            "--prediction",
            "--prediction-min-confidence",
            "40",
            "--prediction-window",
            "64M",
            "--prediction-speculative",
            "16M",
            "--prediction-history-save",
            history_path.to_str().unwrap(),
            "--residency-observe",
            "--residency-observe-pages",
            "32768",
            "--residency-report-json",
            report_path.to_str().unwrap(),
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
    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&report_path).unwrap_or_else(|_| panic!("missing report: {stderr}")),
    )
    .expect("prediction report JSON");
    let _ = std::fs::remove_file(&profile_path);
    let _ = std::fs::remove_file(&history_path);
    let _ = std::fs::remove_file(&report_path);
    assert!(output.status.success(), "prediction run failed: {stderr}");
    let prediction = &report["prediction"];
    assert_eq!(prediction["source"], "Profile");
    assert!(prediction["counts"]["predictions"].as_u64().unwrap() > 0);
    let requested = prediction["plan"]["requested_bytes"].as_u64().unwrap();
    assert!(requested <= 64 * 1024 * 1024);
    assert_eq!(prediction["plan"]["release_authorized"], false);
    assert!(
        prediction["staging"]["native_accepted_bytes"]
            .as_u64()
            .unwrap()
            <= requested
    );
    assert!(prediction["history"]["total_predictions"].as_u64().unwrap() > 0);
    let observation = &report["residency_observation"];
    assert_eq!(observation["before"]["supported"], true);
    assert_eq!(observation["before"]["sampled"], false);
    assert!(observation["before"]["resident_pages"].as_u64().unwrap() > 0);
    let targeted = &observation["before"]["targeted"];
    assert!(targeted["selected_resources"].as_u64().unwrap() > 0);
    assert!(targeted["targeted_pages_sampled"].as_u64().unwrap() > 0);
    assert_eq!(targeted["page_budget"], 32768);
    assert_eq!(
        targeted["selected_resources_fully_observed"],
        targeted["selected_resources"]
    );
    assert_eq!(
        observation["prefetch_correlation"]["before_access_observation_available"],
        false
    );
    assert!(
        observation["prefetch_correlation"]["already_resident"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(
        report["execution_timing"]["end_to_end_ns"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(report["execution_timing"]["inference_ns"].as_u64().unwrap() > 0);
    assert!(
        report["execution_timing"]["observation_ns"]
            .as_u64()
            .unwrap()
            > 0
    );
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

#[test]
#[ignore = "requires FORGE_TEST_MODEL and the native CPU observer build"]
fn node_dispatch_observer_reports_diagnostic_events() {
    let model = std::env::var("FORGE_TEST_MODEL").expect("set FORGE_TEST_MODEL");
    let path = std::env::temp_dir().join(format!(
        "ramforge-node-dispatch-{}.json",
        std::process::id()
    ));
    let output = ramforge()
        .args([
            "run",
            &model,
            "--ram",
            "192M",
            "--ctx",
            "256",
            "--no-repack",
            "--no-prefetch",
            "--backend",
            "forgecore",
            "--node-dispatch-observe",
            "--node-dispatch-events",
            "4096",
            "--node-dispatch-sample-pages",
            "64",
            "--residency-report-json",
            path.to_str().unwrap(),
            "--prompt",
            "The capital of France is",
            "--max-tokens",
            "1",
            "--temperature",
            "0",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&path).unwrap_or_else(|_| panic!("missing report: {stderr}")),
    )
    .expect("node dispatch report");
    let _ = std::fs::remove_file(&path);
    assert!(
        output.status.success(),
        "node dispatch run failed: {stderr}"
    );
    let dispatch = &report["node_dispatch"];
    assert_eq!(dispatch["summary"]["dropped_events"], 0);
    assert!(dispatch["summary"]["events"].as_u64().unwrap() > 0);
    assert!(!dispatch["events"].as_array().unwrap().is_empty());
    assert!(dispatch["attributed_events"].as_u64().unwrap() > 0);
    assert!(dispatch["unattributed_events"].as_u64().unwrap() > 0);
    assert!(dispatch["matched_resources"].as_u64().unwrap() > 0);
    assert_eq!(dispatch["identity_mismatches"], 0);
    assert_eq!(dispatch["events"][0]["backend"], "cpu");
    assert!(dispatch["events"][0]["node_count"].as_u64().unwrap() > 0);
    assert!(dispatch["sampled_sources"].as_u64().unwrap() > 0);
    assert!(dispatch["requested_pages"].as_u64().unwrap() > 0);
    assert_eq!(
        dispatch["resident_pages"].as_u64().unwrap()
            + dispatch["nonresident_pages"].as_u64().unwrap()
            + dispatch["unknown_pages"].as_u64().unwrap(),
        dispatch["sampled_pages"].as_u64().unwrap()
    );
    assert!(dispatch["truncated_samples"].as_u64().unwrap() > 0);
    let sample = &dispatch["events"][0]["dispatch_sample"];
    assert!(sample["mapping_generation"].as_u64().unwrap() > 0);
    assert!(sample["mapping_id"].as_u64().unwrap() < 1_000_000);
    assert!(sample["fresh_sampled_pages"].as_u64().unwrap() > 0);
    assert!(dispatch["cache_skips"].as_u64().is_some());
    assert_eq!(dispatch["sampling_enabled"], true);
    assert_eq!(
        dispatch["requested_pages"].as_u64().unwrap(),
        dispatch["sampled_pages"].as_u64().unwrap()
            + dispatch["skipped_pages"].as_u64().unwrap()
            + dispatch["budget_excluded_pages"].as_u64().unwrap()
            + dispatch["unsampled_pages"].as_u64().unwrap()
    );
    let contract: SchedulerObservation =
        serde_json::from_value(dispatch["scheduler_observation"].clone())
            .expect("scheduler observation contract");
    contract.validate().expect("contract invariants");
    assert!(contract.observation_only);
    assert_eq!(contract.schema_version, 1);
    assert!(contract.sampling_enabled);
    for evidence in &contract.dispatches {
        if let Some(sample) = &evidence.sample {
            sample.validate().expect("dispatch account");
        }
        for source in &evidence.sources {
            source.validate().expect("source attribution");
        }
    }
    assert_eq!(
        dispatch["semantics"],
        "CPU node dispatch start; not a tensor read or residency proof"
    );
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL and a native build"]
fn machine_profile_reports_resources_requirements_and_plans() {
    let model = std::env::var("FORGE_TEST_MODEL").expect("set FORGE_TEST_MODEL");
    let output = ramforge()
        .args([
            "machine",
            "--model",
            &model,
            "--ram",
            "1G",
            "--context",
            "2048",
            "--storage",
            "/var/tmp",
            "--benchmark-storage",
            "--benchmark-size",
            "64K",
            "--benchmark-block",
            "4K",
            "--benchmark-runs",
            "1",
            "--json",
            "--export",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "machine profile failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).expect("json");
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["kind"], "ramforge.machine_profile");
    let profile = &report["profile"];
    assert!(profile["hostname_redacted"].as_bool().unwrap());
    assert!(profile["storage"]
        .as_array()
        .unwrap()
        .iter()
        .all(|storage| storage["local_path"].is_null()));
    assert_eq!(profile["memory"]["cgroup"]["path"]["state"], "unavailable");
    assert!(profile["cpu"]["architecture"]["value"].is_string());
    assert!(
        profile["cpu"]["available_threads"]["value"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(
        profile["memory"]["physical_total_bytes"]["value"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(!profile["storage"].as_array().unwrap().is_empty());
    let benchmark = profile["storage"]
        .as_array()
        .unwrap()
        .iter()
        .find_map(|storage| storage["benchmark"].as_object())
        .expect("storage benchmark");
    assert_eq!(benchmark["physical_claim"], false);
    assert!(benchmark["samples"].as_array().unwrap().len() == 1);

    let readiness = &report["execution_readiness"];
    assert_eq!(readiness["cpu_execution"], true);
    assert_eq!(readiness["selected_device"], "cpu");
    assert!(readiness["registered_backends"].is_array());
    let inventory = &report["profile"]["backend_inventory"];
    assert!(!inventory["registered"].as_array().unwrap().is_empty());
    assert_eq!(
        inventory["gpu_offload_available"],
        readiness["gpu_offload_available"]
    );
    if readiness["gpu_offload_available"] == false {
        assert_eq!(readiness["gpu_execution"], false);
    } else {
        let selectable_gpu = readiness["gpu_devices"]
            .as_array()
            .unwrap()
            .iter()
            .any(|device| device["selectable"] == true);
        assert_eq!(readiness["gpu_execution"], selectable_gpu);
    }

    let hierarchy = &report["memory_hierarchy"];
    assert_eq!(hierarchy["schema_version"], 2);
    assert!(hierarchy["sampled_at_unix_ms"].as_u64().is_some());
    assert!(hierarchy["host_total_bytes"].as_u64().unwrap() > 0);
    let categories = hierarchy["categories"].as_array().unwrap();
    assert!(categories
        .iter()
        .any(|category| category["category"] == "host_ram"));
    assert!(hierarchy["notes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|note| note.as_str().unwrap().contains("must not be summed")));

    let requirements = &report["model_requirements"];
    let architecture = &requirements["architecture_info"];
    assert!(matches!(
        architecture["kind"].as_str(),
        Some("dense" | "mixture_of_experts" | "unknown")
    ));
    assert_eq!(architecture["provenance"], "GGUF metadata");
    assert!(
        requirements["tensor_payload_bytes"]["bytes"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(requirements["tensor_payload_bytes"]["certainty"], "exact");
    assert_eq!(requirements["kv_cache_bytes"]["certainty"], "estimated");
    let plans = report["candidate_plans"].as_array().unwrap();
    assert_eq!(plans.len(), 4);
    let plan_hierarchy = report["plan_hierarchy"].as_array().unwrap();
    assert_eq!(plan_hierarchy.len(), plans.len());
    for entry in plan_hierarchy {
        assert!(entry["requested"]["backend"].is_string());
        assert!(entry["placement"]["components"].is_array());
        assert!(entry["feasibility"]["status"].is_string());
    }
    let streaming = plan_hierarchy
        .iter()
        .find(|entry| entry["plan_id"] == "memory-constrained")
        .expect("streaming candidate");
    assert_eq!(streaming["requested"]["mapping"], "streaming");
    assert_eq!(streaming["feasibility"]["status"], "unknown");
    let weights = streaming["placement"]["components"]
        .as_array()
        .unwrap()
        .iter()
        .find(|component| component["component"] == "model_weights")
        .expect("weights placement");
    assert_eq!(weights["placement"], "unknown");
    assert_eq!(weights["confirmed"], false);
    let comparison = &report["plan_comparison"];
    assert_eq!(
        comparison["evaluations"].as_array().unwrap().len(),
        plans.len()
    );
    assert!(comparison["note"]
        .as_str()
        .unwrap()
        .contains("No single score"));
    for evaluation in comparison["evaluations"].as_array().unwrap() {
        assert!(evaluation["feasibility"].is_string());
        assert!(evaluation["estimated_ram"]["bytes"].is_u64());
    }
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL and a native build"]
fn machine_profile_does_not_claim_a_small_budget_fits_resident_weights() {
    let model = std::env::var("FORGE_TEST_MODEL").expect("set FORGE_TEST_MODEL");
    let output = ramforge()
        .args(["machine", "--model", &model, "--ram", "1M", "--json"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).expect("json");
    let evaluations = report["plan_comparison"]["evaluations"]
        .as_array()
        .expect("evaluations");
    let performance = evaluations
        .iter()
        .find(|evaluation| evaluation["plan_id"] == "performance")
        .expect("performance plan");
    assert_eq!(performance["feasibility"], "infeasible");
    let balanced = evaluations
        .iter()
        .find(|evaluation| evaluation["plan_id"] == "balanced")
        .expect("balanced plan");
    assert_ne!(balanced["feasibility"], "feasible");
    assert_ne!(balanced["feasibility"], "feasible_with_constraints");
}

#[test]
fn root_help_lists_machine_command() {
    let output = ramforge().arg("--help").output().unwrap();
    assert!(output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(
        text.contains("machine"),
        "root help must list machine: {text}"
    );
    let output = ramforge().args(["machine", "--help"]).output().unwrap();
    assert!(output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    for flag in [
        "--model",
        "--storage",
        "--ram",
        "--context",
        "--batch",
        "--threads",
        "--no-prefetch",
        "--no-repack",
        "--ubatch",
        "--sequences",
        "--calibration",
        "--benchmark-storage",
        "--benchmark-size",
        "--benchmark-runs",
        "--json",
        "--export",
        "--calibration",
    ] {
        assert!(text.contains(flag), "machine help must list {flag}: {text}");
    }
    let output = ramforge().args(["calibrate", "--help"]).output().unwrap();
    assert!(output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    for flag in [
        "--ram",
        "--context",
        "--threads",
        "--max-tokens",
        "--json",
        "--output",
        "--no-repack",
        "--no-prefetch",
    ] {
        assert!(
            text.contains(flag),
            "calibrate help must list {flag}: {text}"
        );
    }
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL and a native build"]
fn calibration_records_lifecycle_and_comparisons() {
    let model = std::env::var("FORGE_TEST_MODEL").expect("set FORGE_TEST_MODEL");
    let path =
        std::env::temp_dir().join(format!("ramforge-calibration-{}.json", std::process::id()));
    let output = ramforge()
        .args([
            "calibrate",
            &model,
            "--ram",
            "1G",
            "--context",
            "256",
            "--threads",
            "4",
            "--max-tokens",
            "2",
            "--no-repack",
            "--json",
            "--output",
            path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "calibration failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let calibration: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("calibration output")).expect("json");
    assert_eq!(calibration["schema_version"], 1);
    assert_eq!(calibration["observation_only"], true);
    let snapshots = calibration["snapshots"].as_array().unwrap();
    assert!(snapshots.len() >= 8);
    let stages: Vec<_> = snapshots
        .iter()
        .map(|snapshot| snapshot["stage"].as_str().unwrap())
        .collect();
    for required in [
        "baseline",
        "model_loaded",
        "context_created",
        "prompt_processed",
        "decode_steady",
        "context_destroyed",
        "model_unloaded",
        "final",
    ] {
        assert!(
            stages.contains(&required),
            "missing stage {required}: {stages:?}"
        );
    }
    assert!(calibration["deltas"]["model_load_rss_bytes"]
        .as_i64()
        .is_some());
    assert!(calibration["deltas"]["context_create_rss_bytes"]
        .as_i64()
        .is_some());
    assert!(calibration["predicted"]["model_fingerprint"]
        .as_str()
        .unwrap()
        .starts_with("fnv1a64:"));
    assert_eq!(calibration["runtime"]["effective_context_tokens"], 256);
    assert_eq!(calibration["runtime"]["effective_threads"], 4);
    let comparisons = calibration["comparisons"].as_array().unwrap();
    assert!(comparisons.iter().any(|comparison| {
        comparison["label"] == "tensor_payload_vs_loaded_mapping_resident"
            && comparison["valid"] == true
    }));
    assert!(calibration["attribution"]["components"]
        .as_array()
        .unwrap()
        .iter()
        .any(|component| component["component"] == "process_rss"));
    assert!(calibration["attribution"]["note"]
        .as_str()
        .unwrap()
        .contains("must not be summed"));
    let serialized = calibration.to_string();
    assert!(!serialized.contains(&model));

    let plan_output = ramforge()
        .args([
            "machine",
            "--model",
            &model,
            "--ram",
            "200M",
            "--context",
            "256",
            "--batch",
            "1",
            "--threads",
            "4",
            "--ubatch",
            "256",
            "--sequences",
            "1",
            "--no-repack",
            "--calibration",
            path.to_str().unwrap(),
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        plan_output.status.success(),
        "calibrated plan failed: {}",
        String::from_utf8_lossy(&plan_output.stderr)
    );
    let plan: serde_json::Value = serde_json::from_slice(&plan_output.stdout).expect("plan json");
    assert!(plan["plan_comparison"]["evaluations"]
        .as_array()
        .unwrap()
        .iter()
        .any(|evaluation| evaluation["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reason| reason["code"] == "calibration_applied")));
    assert!(plan["calibration_compatibility"]
        .as_array()
        .unwrap()
        .iter()
        .any(|entry| entry["result"]["classification"] == "exact"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn root_help_lists_operational_commands() {
    let output = ramforge().arg("--help").output().unwrap();
    assert!(output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    for command in [
        "benchmark",
        "optimize",
        "serve",
        "storage",
        "calibrate",
        "machine",
    ] {
        assert!(
            text.contains(command),
            "root help must list {command}: {text}"
        );
    }
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL and a native build"]
fn benchmark_reports_measured_generation_throughput() {
    let model = std::env::var("FORGE_TEST_MODEL").expect("set FORGE_TEST_MODEL");
    let output = ramforge()
        .args([
            "benchmark",
            &model,
            "--profile",
            "balanced",
            "--ram",
            "1G",
            "--runs",
            "1",
            "--warmup",
            "0",
            "--max-tokens",
            "8",
            "--context",
            "256",
            "--threads",
            "2",
            "--no-repack",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "benchmark failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).expect("json");
    assert_eq!(report["schema_version"], 10);
    assert_eq!(report["plan"]["effective"]["backend"], "cpu");
    assert_eq!(report["effective_gpu_layers"], 0);
    assert_eq!(report["device_id"], "cpu");
    assert!(report["output_device_type"].is_string());
    assert_eq!(report["plan"]["effective"]["execution_mode"], "cpu");
    assert_eq!(report["plan"]["effective"]["gpu_layers"], 0);
    assert!(report["plan"]["effective"]["threads"].as_u64().unwrap() > 0);
    assert_eq!(report["plan"]["effective"]["context_tokens"], 256);
    assert!(
        report["median_generation_tokens_per_second"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert!(report["median_decode_tokens_per_second"].as_f64().is_some());
    let median = report["median_generation_tokens_per_second"]
        .as_f64()
        .unwrap();
    let p25 = report["p25_generation_tokens_per_second"].as_f64().unwrap();
    let p75 = report["p75_generation_tokens_per_second"].as_f64().unwrap();
    let iqr = report["iqr_generation_tokens_per_second"].as_f64().unwrap();
    assert!(p25 <= median && median <= p75);
    assert!((p75 - p25 - iqr).abs() < 1e-9);
    assert!(report["model_load_ns"].as_u64().unwrap() > 0);
    let memory = &report["memory_after_generation"];
    assert_eq!(memory["schema_version"], 3);
    assert!(!memory["layers"].as_array().unwrap().is_empty());
    assert!(!memory["buffers"].as_array().unwrap().is_empty());
    assert_eq!(
        memory["transfers"]["state"], "unsupported",
        "transfers must not be fabricated"
    );
    assert!(report["memory_after_unload"]["buffers"]
        .as_array()
        .unwrap()
        .is_empty());
    assert_eq!(report["runs"], 1);
    assert_eq!(report["runs_detail"].as_array().unwrap().len(), 1);
    assert!(report["limitations"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value.as_str().unwrap().contains("warm")));
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL and a native build"]
fn serve_streams_tokens_and_reports_status() {
    use std::io::Write;
    use std::process::Stdio;

    let model = std::env::var("FORGE_TEST_MODEL").expect("set FORGE_TEST_MODEL");
    let mut child = ramforge()
        .args([
            "serve",
            &model,
            "--profile",
            "balanced",
            "--ram",
            "1G",
            "--context",
            "128",
            "--threads",
            "2",
            "--no-repack",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("serve");
    {
        let stdin = child.stdin.as_mut().expect("stdin");
        writeln!(stdin, r#"{{"id":1,"command":"status"}}"#).unwrap();
        writeln!(
            stdin,
            r#"{{"id":2,"command":"generate","prompt":"test","max_tokens":2}}"#
        )
        .unwrap();
        writeln!(stdin, r#"{{"id":3,"command":"architecture"}}"#).unwrap();
        writeln!(stdin, r#"{{"id":4,"command":"hierarchy"}}"#).unwrap();
        writeln!(stdin, r#"{{"id":5,"command":"memory"}}"#).unwrap();
    }
    drop(child.stdin.take());
    let output = child.wait_with_output().expect("serve output");
    assert!(
        output.status.success(),
        "serve failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let lines: Vec<serde_json::Value> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    let status = lines
        .iter()
        .find(|line| line["event"] == "result" && line["id"] == 1)
        .expect("status result");
    assert_eq!(status["status"]["execution_mode"], "cpu");
    assert_eq!(status["status"]["effective_gpu_layers"], 0);
    assert!(lines
        .iter()
        .any(|line| line["event"] == "token" && line["id"] == 2));
    assert!(lines
        .iter()
        .any(|line| line["event"] == "done" && line["id"] == 2));
    let architecture = lines
        .iter()
        .find(|line| line["event"] == "result" && line["id"] == 3)
        .expect("architecture result");
    assert!(matches!(
        architecture["architecture"]["kind"].as_str(),
        Some("dense" | "mixture_of_experts" | "unknown")
    ));
    let hierarchy = lines
        .iter()
        .find(|line| line["event"] == "result" && line["id"] == 4)
        .expect("hierarchy result");
    assert_eq!(hierarchy["memory_hierarchy"]["schema_version"], 2);
    assert!(hierarchy["memory_hierarchy"]["sampled_at_unix_ms"]
        .as_u64()
        .is_some());
    assert!(hierarchy["placement"]["components"].is_array());
    assert!(hierarchy["feasibility"]["status"].is_string());
    assert!(hierarchy["note"]
        .as_str()
        .unwrap()
        .contains("fresh OS observation"));
    let memory = lines
        .iter()
        .find(|line| line["event"] == "result" && line["id"] == 5)
        .expect("memory result");
    assert_eq!(memory["gpu_observability"]["schema_version"], 3);
    assert!(!memory["gpu_observability"]["layers"]
        .as_array()
        .unwrap()
        .is_empty());
    assert_eq!(
        memory["gpu_observability"]["transfers"]["state"],
        "unsupported"
    );
    assert!(!memory["gpu_observability"]["devices"]
        .as_array()
        .unwrap()
        .is_empty());
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL and a CPU-only native build"]
fn gpu_request_fails_explicitly_when_backend_is_unavailable() {
    let model = std::env::var("FORGE_TEST_MODEL").expect("set FORGE_TEST_MODEL");
    let output = ramforge()
        .args([
            "benchmark",
            &model,
            "--profile",
            "balanced",
            "--ram",
            "1G",
            "--runs",
            "1",
            "--warmup",
            "0",
            "--max-tokens",
            "4",
            "--context",
            "128",
            "--device",
            "auto",
            "--gpu-layers",
            "4",
        ])
        .output()
        .unwrap();
    if output.status.success() {
        println!("SKIP: a GPU execution path is available on this machine");
        return;
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("GPU execution requested")
            || stderr.contains("GPU execution requested but unavailable"),
        "unexpected GPU failure: {stderr}"
    );
    assert!(
        stderr.contains("no GPU backend") || stderr.contains("GPU execution unavailable"),
        "failure must explain backend unavailability: {stderr}"
    );
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL and a CUDA native build"]
fn staged_layer_inference_consumes_graph_substitutions() {
    let model = std::env::var("FORGE_TEST_MODEL").expect("set FORGE_TEST_MODEL");
    let machine = ramforge()
        .args(["machine", "--model", &model, "--json"])
        .output()
        .unwrap();
    assert!(machine.status.success());
    let profile: serde_json::Value = serde_json::from_slice(&machine.stdout).expect("json");
    if profile["execution_readiness"]["gpu_execution"] != true {
        println!("SKIP: no selectable CUDA device");
        return;
    }
    let output = ramforge()
        .args([
            "benchmark",
            &model,
            "--profile",
            "balanced",
            "--ram",
            "1G",
            "--runs",
            "1",
            "--warmup",
            "1",
            "--max-tokens",
            "8",
            "--context",
            "128",
            "--threads",
            "2",
            "--no-repack",
            "--staged-layer",
            "0",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "staged benchmark failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).expect("json");
    assert_eq!(report["schema_version"], 10);
    let staged = report["staged_layers"].as_array().expect("staged");
    assert_eq!(staged.len(), 1);
    assert_eq!(staged[0]["layer"], 0);
    assert_eq!(report["staging_plan"]["state"], "exact");
    assert_eq!(report["staging_plan"]["eviction_supported"], false);
    assert_eq!(
        report["staging_plan"]["transfer_compute_overlap_supported"],
        false
    );
    assert!(staged[0]["tensor_count"].as_u64().unwrap() > 0);
    assert!(
        staged[0]["graph_substitutions"].as_u64().unwrap() >= 9,
        "graph did not substitute staged weights: {report}"
    );
    assert!(
        report["runs_detail"][0]["generated_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );
    let counters = &report["memory_after_generation"]["staging"]["counters"];
    assert!(counters["transfer_bytes"].as_u64().unwrap() > 0);
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL and a CUDA native build"]
fn multi_layer_staging_respects_budget_and_auto_selection() {
    let model = std::env::var("FORGE_TEST_MODEL").expect("set FORGE_TEST_MODEL");
    let machine = ramforge()
        .args(["machine", "--model", &model, "--json"])
        .output()
        .unwrap();
    assert!(machine.status.success());
    let profile: serde_json::Value = serde_json::from_slice(&machine.stdout).expect("json");
    if profile["execution_readiness"]["gpu_execution"] != true {
        println!("SKIP: no selectable CUDA device");
        return;
    }
    let output = ramforge()
        .args([
            "benchmark",
            &model,
            "--profile",
            "balanced",
            "--ram",
            "1G",
            "--runs",
            "1",
            "--warmup",
            "1",
            "--max-tokens",
            "4",
            "--context",
            "128",
            "--threads",
            "2",
            "--no-repack",
            "--staged-layer",
            "0",
            "--staged-layer",
            "1",
            "--staged-budget",
            "20M",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "multi-layer staging failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).expect("json");
    let staged = report["staged_layers"].as_array().unwrap();
    assert_eq!(staged.len(), 2);
    for layer in staged {
        assert!(layer["graph_substitutions"].as_u64().unwrap() >= 9);
    }
    assert_eq!(report["staging_plan"]["state"], "exact");
    assert_eq!(
        report["staging_plan"]["accepted_layers"]
            .as_array()
            .unwrap()
            .len(),
        2
    );

    let rejected = ramforge()
        .args([
            "benchmark",
            &model,
            "--runs",
            "1",
            "--warmup",
            "0",
            "--max-tokens",
            "2",
            "--context",
            "128",
            "--threads",
            "2",
            "--no-repack",
            "--staged-layer",
            "0",
            "--staged-layer",
            "1",
            "--staged-budget",
            "1M",
        ])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    let stderr = String::from_utf8_lossy(&rejected.stderr);
    assert!(
        stderr.contains("staged layer plan rejected") && stderr.contains("rejected"),
        "budget rejection must be explicit: {stderr}"
    );

    let auto = ramforge()
        .args([
            "benchmark",
            &model,
            "--runs",
            "1",
            "--warmup",
            "0",
            "--max-tokens",
            "2",
            "--context",
            "128",
            "--threads",
            "2",
            "--no-repack",
            "--staged-auto",
            "--staged-budget",
            "10M",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(auto.status.success());
    let report: serde_json::Value = serde_json::from_slice(&auto.stdout).expect("json");
    assert!(report["staging_plan"]["auto_selected"].as_bool().unwrap());
    assert!(
        report["staging_plan"]["accepted_layers"]
            .as_array()
            .unwrap()
            .len()
            >= 2
    );
    assert!(matches!(
        report["staging_plan"]["state"].as_str(),
        Some("partial" | "exact")
    ));
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL and a CUDA native build"]
fn cuda_benchmark_reports_native_placement_and_memory() {
    let model = std::env::var("FORGE_TEST_MODEL").expect("set FORGE_TEST_MODEL");
    let machine = ramforge()
        .args(["machine", "--model", &model, "--json"])
        .output()
        .unwrap();
    assert!(machine.status.success());
    let profile: serde_json::Value = serde_json::from_slice(&machine.stdout).expect("json");
    if profile["execution_readiness"]["gpu_execution"] != true {
        println!("SKIP: no selectable CUDA device");
        return;
    }
    let output = ramforge()
        .args([
            "benchmark",
            &model,
            "--profile",
            "balanced",
            "--ram",
            "1G",
            "--runs",
            "1",
            "--warmup",
            "0",
            "--max-tokens",
            "8",
            "--context",
            "128",
            "--threads",
            "2",
            "--no-repack",
            "--device",
            "auto",
            "--gpu-layers",
            "8",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "CUDA benchmark failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).expect("json");
    assert_eq!(report["schema_version"], 10);
    assert_ne!(report["device_id"], "cpu");
    assert_eq!(report["requested_gpu_layers"], 8);
    let effective = report["effective_gpu_layers"].as_u64().expect("readback");
    assert!(effective >= 1, "no effective GPU layers: {report}");
    assert!(report["gpu_transformer_layers"].as_u64().unwrap() >= 1);
    assert_eq!(report["output_device_type"], "gpu");
    let after_load = report["device_memory_free_after_load_bytes"]
        .as_u64()
        .unwrap();
    let after_unload = report["device_memory_free_after_unload_bytes"]
        .as_u64()
        .unwrap();
    assert!(
        after_unload > after_load,
        "unload did not release VRAM: {after_load} -> {after_unload}"
    );
    let memory = &report["memory_after_generation"];
    assert_eq!(memory["schema_version"], 3);
    assert!(memory["layers"].as_array().unwrap().iter().any(|layer| {
        layer["device_name"]
            .as_str()
            .unwrap_or("")
            .contains("CUDA0")
            && layer["bytes"].as_u64().unwrap_or(0) > 0
    }));
    assert!(memory["buffers"].as_array().unwrap().iter().any(|buffer| {
        buffer["device_name"]
            .as_str()
            .unwrap_or("")
            .contains("CUDA0")
            && buffer["model_bytes"].as_u64().unwrap_or(0) > 0
    }));
    assert!(memory["observations"]
        .as_array()
        .unwrap()
        .iter()
        .any(|observation| observation["category"] == "kv_cache"));
    assert!(
        report["runs_detail"][0]["generated_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL"]
fn storage_copy_is_verified_and_preserves_source() {
    let model = std::env::var("FORGE_TEST_MODEL").expect("set FORGE_TEST_MODEL");
    let destination =
        std::env::temp_dir().join(format!("ramforge-storage-copy-{}.gguf", std::process::id()));
    let output = ramforge()
        .args([
            "storage",
            &model,
            "--copy",
            "--destination",
            destination.to_str().unwrap(),
            "--yes",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "storage copy failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).expect("json");
    assert_eq!(report["verified"], true);
    assert_eq!(report["source_preserved"], true);
    assert_eq!(
        std::fs::metadata(&model).unwrap().len(),
        std::fs::metadata(&destination).unwrap().len()
    );
    let _ = std::fs::remove_file(destination);
}
