//! Manual resource-profile and candidate-plan overhead benchmark.
//!
//! Warm, local, bounded. This is not a multi-machine validation claim.

use std::path::PathBuf;
use std::time::Instant;

use ramforge_runtime::{
    candidate_plans, collect_machine_profile, compare_plans, model_requirements, CandidateRequest,
    CpuRuntimeSettings, MachineProfileConfig, ModelRequirementConfig, StorageBenchmarkConfig,
};

fn median(values: &mut [u64]) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

#[test]
#[ignore = "manual resource-profile benchmark"]
fn resource_profile_benchmark() {
    let runs = std::env::var("PROFILE_BENCH_RUNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(20)
        .max(3);
    let directory =
        std::env::temp_dir().join(format!("ramforge-resource-bench-{}", std::process::id()));
    let config = MachineProfileConfig {
        storage_paths: vec![directory.clone()],
        cpu: CpuRuntimeSettings {
            selected_threads: Some(4),
            user_thread_limit: None,
        },
        ram_budget_bytes: Some(1024 * 1024 * 1024),
        include_storage_benchmark: false,
        storage_benchmark: StorageBenchmarkConfig::default(),
        ..MachineProfileConfig::default()
    };
    let model_set = std::env::var_os("FORGE_TEST_MODEL")
        .map(PathBuf::from)
        .map(|path| ramforge_core::discover_model_set(&path).expect("model set"));
    let mut collect_ns = Vec::with_capacity(runs);
    let mut serialize_ns = Vec::with_capacity(runs);
    let mut requirement_ns = Vec::with_capacity(runs);
    let mut plan_ns = Vec::with_capacity(runs);
    for _ in 0..runs {
        let started = Instant::now();
        let profile = collect_machine_profile(&config);
        collect_ns.push(started.elapsed().as_nanos() as u64);
        profile.validate().expect("profile");
        let started = Instant::now();
        let json = profile.to_json().expect("json");
        let decoded = ramforge_runtime::MachineProfile::from_json(json).expect("decode");
        serialize_ns.push(started.elapsed().as_nanos() as u64);
        assert_eq!(profile, decoded);
        if let Some(model_set) = &model_set {
            let model = model_set.merged_model();
            let started = Instant::now();
            let requirements = model_requirements(
                &model,
                ModelRequirementConfig {
                    context_tokens: 2048,
                    batch_tokens: 1,
                    kv_bytes_per_element: 2,
                },
            );
            requirement_ns.push(started.elapsed().as_nanos() as u64);
            requirements.validate().expect("requirements");
            let started = Instant::now();
            let plans = candidate_plans(
                &profile,
                &requirements,
                CandidateRequest {
                    context_tokens: 2048,
                    batch_tokens: 1,
                    ram_budget_bytes: Some(1024 * 1024 * 1024),
                    vram_budget_bytes: None,
                    selected_threads: Some(4),
                },
            );
            let comparison = compare_plans(&profile, &requirements, &plans);
            comparison.to_json().expect("comparison");
            plan_ns.push(started.elapsed().as_nanos() as u64);
        }
    }
    let _ = std::fs::remove_dir_all(&directory);
    let result = serde_json::json!({
        "kind": "ramforge.resource_profile.benchmark",
        "runs": runs,
        "condition": "warm-local-uncontrolled",
        "cpu_threads": 4,
        "ram_budget_bytes": 1024u64 * 1024 * 1024,
        "storage_benchmark": false,
        "model_included": model_set.is_some(),
        "collect_ns": {"median": median(&mut collect_ns.clone()), "min": collect_ns.iter().min(), "max": collect_ns.iter().max()},
        "serialize_roundtrip_ns": {"median": median(&mut serialize_ns.clone()), "min": serialize_ns.iter().min(), "max": serialize_ns.iter().max()},
        "requirements_ns": if requirement_ns.is_empty() { serde_json::Value::Null } else { serde_json::json!({"median": median(&mut requirement_ns.clone()), "min": requirement_ns.iter().min(), "max": requirement_ns.iter().max()}) },
        "plans_and_comparison_ns": if plan_ns.is_empty() { serde_json::Value::Null } else { serde_json::json!({"median": median(&mut plan_ns.clone()), "min": plan_ns.iter().min(), "max": plan_ns.iter().max()}) },
        "limitations": ["single-machine warm observation", "not a performance guarantee", "storage benchmark excluded"],
    });
    eprintln!("{}", serde_json::to_string(&result).expect("result"));
}
