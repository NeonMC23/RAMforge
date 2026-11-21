//! Same-context staged layer lifecycle through the public service API.
//!
//! Requires a physical CUDA device and `FORGE_TEST_MODEL`. Ignored by default.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Mutex, MutexGuard};

use ramforge_core::discover_model_set;
use ramforge_runtime::machine_profile::{
    collect_machine_profile, CpuRuntimeSettings, MachineProfileConfig,
};
use ramforge_runtime::model_requirements::{model_requirements, ModelRequirementConfig};
use ramforge_runtime::runtime_plan::{CustomRuntimeRequest, RuntimePlan};
use ramforge_runtime::{
    GenerationOptions, InferenceService, RuntimeProfileKind, ServiceError, StagingRequest,
    DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
};

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_lock() -> MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn fixture() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("FORGE_TEST_MODEL")?);
    path.is_file().then_some(path)
}

fn plan_for(path: &std::path::Path) -> RuntimePlan {
    let model_set = discover_model_set(path).expect("model set");
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
        model_path: Some(path.to_path_buf()),
        ram_budget_bytes: Some(1024 * 1024 * 1024),
        cpu: CpuRuntimeSettings {
            selected_threads: Some(4),
            user_thread_limit: None,
        },
        ..MachineProfileConfig::default()
    });
    RuntimePlan::resolve(
        &machine,
        &requirements,
        path.to_string_lossy().as_ref(),
        RuntimeProfileKind::Balanced,
        &CustomRuntimeRequest {
            context_tokens: Some(128),
            batch_tokens: Some(1),
            threads: Some(4),
            load_mode: Some("mmap-no-prefetch".to_string()),
            no_repack: Some(true),
            device: Some("cpu".to_string()),
            gpu_layers: Some(0),
        },
    )
    .expect("runtime plan")
}

fn generate(service: &mut InferenceService) -> String {
    let cancel = AtomicBool::new(false);
    service
        .generate(
            &GenerationOptions {
                prompt: "The capital of France is".to_string(),
                max_tokens: 4,
                ..Default::default()
            },
            &cancel,
            |_| Ok(()),
        )
        .expect("generation")
        .text
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL, a CUDA build, and a physical GPU"]
fn rejected_replacement_preserves_old_selection_and_context() {
    let _guard = env_lock();
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let plan = plan_for(&path);
    let initial = StagingRequest {
        layers: vec![0, 1],
        budget_bytes: Some(20 * 1024 * 1024),
        auto: false,
        safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
    };
    let mut service = InferenceService::new(plan)
        .expect("service")
        .with_staging_request(initial);
    service.load().expect("load");
    service.create_session().expect("session");
    let first = generate(&mut service);
    assert!(!first.is_empty());

    let layer_count = 32;
    let oversized = StagingRequest {
        layers: (0..layer_count).collect(),
        budget_bytes: Some(1024 * 1024),
        auto: false,
        safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
    };
    let error = service
        .replace_staged_layers(oversized)
        .expect_err("oversized explicit replacement must be rejected");
    assert!(
        matches!(error, ServiceError::Configuration(_)),
        "unexpected error: {error}"
    );

    let status = service.status();
    let resident: Vec<u32> = status
        .staged_layers
        .iter()
        .map(|layer| layer.layer)
        .collect();
    assert_eq!(
        resident,
        vec![0, 1],
        "rejected replacement mutated residency"
    );
    let lifecycle = status.staging_lifecycle.expect("lifecycle");
    assert_eq!(lifecycle.graph_invalidation_count, 0);
    let second = generate(&mut service);
    assert!(
        !second.is_empty(),
        "context unusable after rejected replacement"
    );

    let replacement = StagingRequest {
        layers: vec![1, 2],
        budget_bytes: Some(20 * 1024 * 1024),
        auto: false,
        safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
    };
    let plan = service
        .replace_staged_layers(replacement)
        .expect("valid replacement");
    assert_eq!(plan.evicted_layers, vec![0]);
    assert_eq!(plan.kept_layers, vec![1]);
    assert_eq!(plan.newly_staged_layers, vec![2]);
    assert!(plan.replacement_invalidation_ns.is_some());
    assert!(plan.replacement_staging_ns.is_some());
    assert!(plan.replacement_total_ns.is_some());

    let status = service.status();
    let resident: Vec<u32> = status
        .staged_layers
        .iter()
        .map(|layer| layer.layer)
        .collect();
    assert_eq!(resident, vec![1, 2]);
    let lifecycle = status.staging_lifecycle.expect("lifecycle");
    assert_eq!(lifecycle.graph_invalidation_count, 1);
    assert_eq!(lifecycle.release_count, 1);
    assert_eq!(lifecycle.reuse_count, 0);
    assert!(lifecycle.eviction_supported);
    assert!(!lifecycle.allocation_reuse_supported);

    let third = generate(&mut service);
    assert!(!third.is_empty(), "context unusable after replacement");
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL, a CUDA build, and a physical GPU"]
fn staged_failure_rolls_back_to_previous_selection() {
    let _guard = env_lock();
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let plan = plan_for(&path);
    let initial = StagingRequest {
        layers: vec![0, 1],
        budget_bytes: Some(20 * 1024 * 1024),
        auto: false,
        safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
    };
    let mut service = InferenceService::new(plan)
        .expect("service")
        .with_staging_request(initial);
    service.load().expect("load");
    service.create_session().expect("session");
    assert!(!generate(&mut service).is_empty());

    std::env::set_var("RAMFORGE_TEST_FAIL_STAGE_LAYER", "2");
    let replacement = StagingRequest {
        layers: vec![1, 2],
        budget_bytes: Some(20 * 1024 * 1024),
        auto: false,
        safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
    };
    let error = service
        .replace_staged_layers(replacement)
        .expect_err("injected stage failure must be reported");
    std::env::remove_var("RAMFORGE_TEST_FAIL_STAGE_LAYER");
    assert!(
        matches!(
            error,
            ServiceError::Replacement {
                rolled_back: true,
                ..
            }
        ),
        "expected rolled_back replacement error, got {error}"
    );

    let status = service.status();
    let resident: Vec<u32> = status
        .staged_layers
        .iter()
        .map(|layer| layer.layer)
        .collect();
    assert_eq!(
        resident,
        vec![0, 1],
        "rollback did not restore the old selection"
    );
    let lifecycle = status.staging_lifecycle.expect("lifecycle");
    assert_eq!(lifecycle.graph_invalidation_count, 1);
    assert_eq!(lifecycle.release_count, 1);
    assert!(
        !generate(&mut service).is_empty(),
        "context unusable after rollback"
    );
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL, a CUDA build, and a physical GPU"]
fn failed_rollback_fails_closed_and_rejects_generation() {
    let _guard = env_lock();
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let plan = plan_for(&path);
    let initial = StagingRequest {
        layers: vec![0, 1],
        budget_bytes: Some(20 * 1024 * 1024),
        auto: false,
        safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
    };
    let mut service = InferenceService::new(plan)
        .expect("service")
        .with_staging_request(initial);
    service.load().expect("load");
    service.create_session().expect("session");
    assert!(!generate(&mut service).is_empty());

    std::env::set_var("RAMFORGE_TEST_FAIL_STAGE_ALL", "1");
    let replacement = StagingRequest {
        layers: vec![1, 2],
        budget_bytes: Some(20 * 1024 * 1024),
        auto: false,
        safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
    };
    let error = service
        .replace_staged_layers(replacement)
        .expect_err("injected failure must be reported");
    std::env::remove_var("RAMFORGE_TEST_FAIL_STAGE_ALL");
    assert!(
        matches!(
            error,
            ServiceError::Replacement {
                rolled_back: false,
                ..
            }
        ),
        "expected unrecovered replacement error, got {error}"
    );
    assert_eq!(
        service.status().state,
        ramforge_runtime::ServiceState::Failed
    );
    let cancel = AtomicBool::new(false);
    let generate_error = service
        .generate(
            &GenerationOptions {
                prompt: "The capital of France is".to_string(),
                max_tokens: 1,
                ..Default::default()
            },
            &cancel,
            |_| Ok(()),
        )
        .expect_err("failed service must reject generation");
    assert!(matches!(generate_error, ServiceError::InvalidState { .. }));
}
