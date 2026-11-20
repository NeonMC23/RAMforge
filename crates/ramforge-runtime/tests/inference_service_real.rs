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
    service.unload();
}
