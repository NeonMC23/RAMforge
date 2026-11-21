//! Real CPU inference-service validation. Requires FORGE_TEST_MODEL and native.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use ramforge_runtime::{
    collect_machine_profile, model_requirements, CustomRuntimeRequest, GenerationOptions,
    InferenceService, MachineProfileConfig, ModelRequirementConfig, RuntimePlan,
    RuntimeProfileKind, ServiceError,
};

fn fixture() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("FORGE_TEST_MODEL")?);
    path.is_file().then_some(path)
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL and a native build"]
fn service_executes_streams_and_reports_throughput() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let model_set = ramforge_core::discover_model_set(&path).expect("model set");
    let gguf = model_set.merged_model();
    let requirements = model_requirements(
        &gguf,
        ModelRequirementConfig {
            context_tokens: 128,
            batch_tokens: 1,
            kv_bytes_per_element: 2,
        },
    );
    let machine = collect_machine_profile(&MachineProfileConfig {
        model_path: Some(path.clone()),
        ram_budget_bytes: Some(1024 * 1024 * 1024),
        ..MachineProfileConfig::default()
    });
    let plan = RuntimePlan::resolve(
        &machine,
        &requirements,
        path.to_string_lossy().as_ref(),
        RuntimeProfileKind::Custom,
        &CustomRuntimeRequest {
            context_tokens: Some(128),
            batch_tokens: Some(1),
            threads: Some(2),
            load_mode: Some("mmap".to_string()),
            no_repack: Some(true),
            device: None,
            gpu_layers: None,
        },
    )
    .expect("plan");
    let mut service = InferenceService::new(plan).expect("service");
    service.load().expect("load");
    service.create_session().expect("session");
    let status = service.status();
    assert!(status.effective_context_tokens.is_some());
    assert!(status.effective_batch_tokens.is_some());
    let cancel = AtomicBool::new(false);
    let mut pieces = String::new();
    let report = service
        .generate(
            &GenerationOptions {
                prompt: "The capital of France is".to_string(),
                max_tokens: 2,
                temperature: 0.0,
                top_k: None,
                top_p: None,
            },
            &cancel,
            |piece| {
                pieces.push_str(piece);
                Ok(())
            },
        )
        .expect("generation");
    assert!(report.stats.prompt_tokens > 0);
    assert!(report.stats.generated_tokens > 0);
    assert!(report.stats.generation_tokens_per_second > 0.0);
    assert!(report.stats.time_to_first_token_ns.is_some());
    assert_eq!(report.text, pieces);
    eprintln!(
        "service_generation prompt_tokens={} generated={} ttft_ns={:?} decode_ns={:?} decode_tps={:?}",
        report.stats.prompt_tokens,
        report.stats.generated_tokens,
        report.stats.time_to_first_token_ns,
        report.stats.decode_ns,
        report.stats.decode_tokens_per_second
    );
    service.unload();
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL and a native build"]
fn service_cancellation_is_explicit() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let model_set = ramforge_core::discover_model_set(&path).expect("model set");
    let gguf = model_set.merged_model();
    let requirements = model_requirements(
        &gguf,
        ModelRequirementConfig {
            context_tokens: 128,
            batch_tokens: 1,
            kv_bytes_per_element: 2,
        },
    );
    let machine = collect_machine_profile(&MachineProfileConfig::default());
    let plan = RuntimePlan::resolve(
        &machine,
        &requirements,
        path.to_string_lossy().as_ref(),
        RuntimeProfileKind::Custom,
        &CustomRuntimeRequest {
            context_tokens: Some(128),
            batch_tokens: Some(1),
            threads: Some(1),
            load_mode: Some("mmap".to_string()),
            no_repack: Some(true),
            device: None,
            gpu_layers: None,
        },
    )
    .expect("plan");
    let mut service = InferenceService::new(plan).expect("service");
    service.load().expect("load");
    service.create_session().expect("session");
    let cancel = AtomicBool::new(true);
    let error = service
        .generate(
            &GenerationOptions {
                prompt: "test".to_string(),
                max_tokens: 2,
                ..GenerationOptions::default()
            },
            &cancel,
            |_| Ok(()),
        )
        .unwrap_err();
    assert!(matches!(error, ServiceError::Cancelled));
    let after_cancel = service
        .memory_observability()
        .expect("post-cancel snapshot");
    assert!(!after_cancel.buffers.is_empty());
    service.unload();
    let after_unload = service
        .memory_observability()
        .expect("post-unload snapshot");
    assert!(after_unload.buffers.is_empty());
}

fn gpu_plan(path: &std::path::Path, gpu_layers: u32) -> Option<RuntimePlan> {
    let model_set = ramforge_core::discover_model_set(path).expect("model set");
    let gguf = model_set.merged_model();
    let requirements = model_requirements(
        &gguf,
        ModelRequirementConfig {
            context_tokens: 128,
            batch_tokens: 1,
            kv_bytes_per_element: 2,
        },
    );
    let machine = collect_machine_profile(&MachineProfileConfig::default());
    if !machine
        .devices
        .iter()
        .any(|device| device.kind != "cpu" && device.selectable)
    {
        eprintln!("SKIP: no selectable CUDA device");
        return None;
    }
    Some(
        RuntimePlan::resolve(
            &machine,
            &requirements,
            path.to_string_lossy().as_ref(),
            RuntimeProfileKind::Custom,
            &CustomRuntimeRequest {
                context_tokens: Some(128),
                batch_tokens: Some(1),
                threads: Some(2),
                load_mode: Some("mmap".to_string()),
                no_repack: Some(true),
                device: Some("auto".to_string()),
                gpu_layers: Some(gpu_layers),
            },
        )
        .expect("GPU plan"),
    )
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL and a CUDA native build"]
fn service_runs_on_cuda_with_native_placement() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let Some(plan) = gpu_plan(&path, 8) else {
        return;
    };
    let expected_device = plan.effective.device_id.clone();
    let mut service = InferenceService::new(plan).expect("service");
    service.load().expect("GPU model load");
    let load_only = service
        .memory_observability()
        .expect("load-only observability");
    eprintln!("load_only buffers: {:?}", load_only.buffers);
    assert!(load_only
        .buffers
        .iter()
        .any(|buffer| buffer.model_bytes > 0));
    assert!(load_only
        .buffers
        .iter()
        .all(|buffer| buffer.context_bytes == 0));
    service.create_session().expect("session");
    let status = service.status();
    assert_ne!(status.execution_mode, ramforge_runtime::ExecutionMode::Cpu);
    assert_eq!(status.requested_gpu_layers, Some(8));
    let effective = status.effective_gpu_layers.expect("readback");
    assert!(effective >= 1, "expected GPU layers, got {effective}");
    assert!(status.gpu_transformer_layers.unwrap() >= 1);
    assert_eq!(status.output_device_type.as_deref(), Some("gpu"));
    assert_eq!(
        status.device_id, expected_device,
        "status device must match the plan"
    );
    let observability = service
        .memory_observability()
        .expect("memory observability");
    observability.validate().expect("schema");
    assert!(observability
        .devices
        .iter()
        .any(|device| device.kind != "cpu" && device.used_bytes.is_some()));
    assert!(observability
        .buffers
        .iter()
        .any(|buffer| { buffer.device_name.contains("CUDA0") && buffer.model_bytes > 0 }));
    assert!(observability
        .observations
        .iter()
        .any(|observation| observation.category == "kv_cache"));
    assert_eq!(
        observability.transfers.state,
        ramforge_runtime::MemoryObservationState::Unsupported
    );
    let cancel = AtomicBool::new(false);
    let report = service
        .generate(
            &GenerationOptions {
                prompt: "The capital of France is".to_string(),
                max_tokens: 4,
                temperature: 0.0,
                top_k: None,
                top_p: None,
            },
            &cancel,
            |_| Ok(()),
        )
        .expect("GPU generation");
    assert!(report.stats.generated_tokens > 0);
    assert!(report.stats.generation_tokens_per_second > 0.0);
    eprintln!(
        "cuda_generation requested=8 effective={} gpu_transformer={} tps={:.2}",
        effective,
        status.gpu_transformer_layers.unwrap(),
        report.stats.generation_tokens_per_second
    );
    service.unload();
    assert!(service.status().effective_gpu_layers.is_none());
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL and a CUDA native build"]
fn service_cancellation_works_on_cuda() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let Some(plan) = gpu_plan(&path, 8) else {
        return;
    };
    let mut service = InferenceService::new(plan).expect("service");
    service.load().expect("GPU model load");
    service.create_session().expect("session");
    let cancel = AtomicBool::new(false);
    let error = service
        .generate(
            &GenerationOptions {
                prompt: "Write a detailed essay about the history of computing.".to_string(),
                max_tokens: 512,
                ..GenerationOptions::default()
            },
            &cancel,
            |_| {
                cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            },
        )
        .unwrap_err();
    assert!(matches!(error, ServiceError::Cancelled));
    service.unload();
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL and a CUDA native build"]
fn repeated_gpu_load_unload_does_not_grow_buffers_unboundedly() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let Some(plan) = gpu_plan(&path, 8) else {
        return;
    };
    let mut baseline_free: Option<u64> = None;
    let mut previous_unload_free: Option<u64> = None;
    for cycle in 0..3 {
        let mut service = InferenceService::new(plan.clone()).expect("service");
        service.load().expect("load");
        service.create_session().expect("session");
        let loaded = service.memory_observability().expect("loaded snapshot");
        let cuda_device = loaded
            .devices
            .iter()
            .find(|device| device.kind != "cpu")
            .expect("CUDA device");
        let loaded_free = cuda_device.free_bytes.expect("free");
        baseline_free.get_or_insert(cuda_device.total_bytes.expect("total"));
        assert!(loaded.buffers.iter().any(|buffer| buffer.model_bytes > 0));
        assert!(loaded
            .observations
            .iter()
            .any(|observation| observation.category == "compute_graph"));
        service.unload();
        let unloaded = service.memory_observability().expect("unloaded snapshot");
        assert!(unloaded.buffers.is_empty());
        let unloaded_free = unloaded
            .devices
            .iter()
            .find(|device| device.kind != "cpu")
            .and_then(|device| device.free_bytes)
            .expect("free");
        assert!(
            unloaded_free > loaded_free,
            "cycle {cycle}: unload did not release VRAM ({loaded_free} -> {unloaded_free})"
        );
        if let Some(previous) = previous_unload_free {
            let difference = previous.abs_diff(unloaded_free);
            assert!(
                difference < 256 * 1024 * 1024,
                "cycle {cycle}: free memory drift {difference} bytes"
            );
            eprintln!("cycle {cycle}: unload free={unloaded_free}, drift={difference}");
        }
        previous_unload_free = Some(unloaded_free);
    }
    assert!(baseline_free.is_some());
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL and a CUDA native build"]
fn observability_sampling_overhead_is_bounded() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let Some(plan) = gpu_plan(&path, 8) else {
        return;
    };
    let mut service = InferenceService::new(plan).expect("service");
    service.load().expect("load");
    service.create_session().expect("session");
    let started = std::time::Instant::now();
    for _ in 0..100 {
        service.memory_observability().expect("snapshot");
    }
    let elapsed = started.elapsed();
    let average = elapsed / 100;
    eprintln!("observability: 100 snapshots in {elapsed:?}, average {average:?}");
    assert!(
        average < std::time::Duration::from_millis(5),
        "observability snapshot too slow: {average:?}"
    );
    service.unload();
}
