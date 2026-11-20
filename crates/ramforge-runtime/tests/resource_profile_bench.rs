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

#[test]
#[ignore = "manual lifecycle calibration benchmark"]
fn lifecycle_calibration_benchmark() {
    use ramforge_runtime::{
        build_attribution, build_calibration, CompatibilityTarget, LifecycleRecorder,
        LifecycleStage, ModelRequirements, RequirementBytes, RuntimeCalibration,
        RuntimeConfiguration,
    };

    let runs = std::env::var("CALIBRATION_BENCH_RUNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10)
        .max(3);
    let machine = collect_machine_profile(&MachineProfileConfig::default());
    let requirements = ModelRequirements {
        schema_version: ramforge_runtime::MODEL_REQUIREMENTS_SCHEMA_VERSION,
        model_fingerprint: Some("benchmark-fixture".to_string()),
        model_name: Some("fixture".to_string()),
        architecture: Some("llama".to_string()),
        quantization: Some("fixture".to_string()),
        tensor_count: 1,
        file_bytes: RequirementBytes::exact(1024, "fixture"),
        tensor_payload_bytes: RequirementBytes::exact(1024, "fixture"),
        fully_resident_weight_bytes: RequirementBytes::exact(1024, "fixture"),
        kv_cache_bytes: RequirementBytes::estimated(256, "fixture"),
        context_tokens: 256,
        batch_tokens: 1,
        model_max_context_tokens: Some(256),
        resident_weights_plus_kv_bytes: RequirementBytes::estimated(1280, "fixture"),
        mapped_weights_plus_kv_bytes: RequirementBytes::estimated(256, "fixture"),
        unknown_requirements: Vec::new(),
        assumptions: Vec::new(),
    };
    let runtime = RuntimeConfiguration {
        backend: "cpu".to_string(),
        device_id: Some("cpu".to_string()),
        load_mode: Some("mmap".to_string()),
        weight_repacking: Some("disabled".to_string()),
        requested_context_tokens: Some(256),
        effective_context_tokens: Some(256),
        requested_batch_tokens: Some(1),
        effective_batch_tokens: Some(1),
        effective_ubatch_tokens: Some(1),
        effective_sequence_max: Some(1),
        effective_threads: Some(1),
        effective_threads_batch: Some(1),
        kv_type_k: "f16".to_string(),
        kv_type_v: "f16".to_string(),
        kv_type_source: "fixture".to_string(),
        state_serialization_size_bytes: None,
        native_contract_version: Some(1),
    };
    let stages = [
        LifecycleStage::Baseline,
        LifecycleStage::ModelInspected,
        LifecycleStage::ModelLoaded,
        LifecycleStage::ContextCreated,
        LifecycleStage::PromptProcessed,
        LifecycleStage::DecodeSteady,
        LifecycleStage::ContextDestroyed,
        LifecycleStage::ModelUnloaded,
        LifecycleStage::Final,
    ];
    let mut snapshot_ns = Vec::with_capacity(runs);
    let mut calibration_ns = Vec::with_capacity(runs);
    let mut serialization_ns = Vec::with_capacity(runs);
    let mut compatibility_ns = Vec::with_capacity(runs);
    for _ in 0..runs {
        let started = Instant::now();
        let mut recorder = LifecycleRecorder::new(None, Some(1024 * 1024));
        for stage in stages {
            recorder.record(stage);
        }
        snapshot_ns.push(started.elapsed().as_nanos() as u64);
        let attribution = build_attribution(recorder.snapshots(), &requirements);
        let started = Instant::now();
        let calibration = build_calibration(
            &machine,
            &requirements,
            runtime.clone(),
            recorder.snapshots().to_vec(),
            attribution,
        );
        calibration.validate().expect("calibration");
        calibration_ns.push(started.elapsed().as_nanos() as u64);
        let started = Instant::now();
        let json = calibration.to_json().expect("json");
        let decoded = RuntimeCalibration::from_json(json).expect("decode");
        assert_eq!(decoded.schema_version, 1);
        serialization_ns.push(started.elapsed().as_nanos() as u64);
        let target = CompatibilityTarget {
            machine: machine.clone(),
            requirements: requirements.clone(),
            runtime: runtime.clone(),
            observation: calibration.observation.clone(),
            now_unix_ms: calibration.captured_at_unix_ms.map(|value| value + 1),
        };
        let started = Instant::now();
        for _ in 0..100 {
            let compatibility = decoded.compatibility(&target);
            assert_eq!(
                compatibility.classification,
                ramforge_runtime::CompatibilityClass::Exact
            );
        }
        compatibility_ns.push(started.elapsed().as_nanos() as u64 / 100);
    }
    let result = serde_json::json!({
        "kind": "ramforge.lifecycle_calibration.benchmark",
        "runs": runs,
        "stages_per_run": stages.len(),
        "condition": "warm-local-uncontrolled",
        "snapshot_ns": {"median": median(&mut snapshot_ns.clone()), "min": snapshot_ns.iter().min(), "max": snapshot_ns.iter().max()},
        "build_calibration_ns": {"median": median(&mut calibration_ns.clone()), "min": calibration_ns.iter().min(), "max": calibration_ns.iter().max()},
        "serialize_roundtrip_ns": {"median": median(&mut serialization_ns.clone()), "min": serialization_ns.iter().min(), "max": serialization_ns.iter().max()},
        "compatibility_ns": {"median": median(&mut compatibility_ns.clone()), "min": compatibility_ns.iter().min(), "max": compatibility_ns.iter().max()},
        "limitations": ["synthetic fixture requirements", "single-machine warm observation", "not a performance guarantee"],
    });
    eprintln!("{}", serde_json::to_string(&result).expect("result"));
}
