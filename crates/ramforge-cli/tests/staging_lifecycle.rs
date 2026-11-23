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
    assert!(lifecycle.allocation_reuse_supported);

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

#[test]
#[ignore = "requires FORGE_TEST_MODEL, a CUDA build, and a physical GPU"]
fn pipeline_reuses_pool_and_commits_pending_selection() {
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
        .with_staging_pool_limit(20 * 1024 * 1024)
        .with_staging_request(initial);
    service.load().expect("load");
    service.create_session().expect("session");
    assert!(!generate(&mut service).is_empty());

    let selections = [vec![1, 2], vec![0, 1], vec![1, 2]];
    for target in selections {
        let prepared = service
            .prepare_staged_pipeline(StagingRequest {
                layers: target.clone(),
                budget_bytes: Some(20 * 1024 * 1024),
                auto: false,
                safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
            })
            .expect("prepare pipeline");
        assert!(prepared.pipeline);
        assert!(prepared.pipeline_fits);
        assert!(prepared.pipeline_required_bytes > prepared.pipeline_pending_bytes);
        let text = generate(&mut service);
        assert!(!text.is_empty());
        let commit = service.commit_staged_pipeline().expect("commit pipeline");
        assert_eq!(commit.layers, target);
        assert!(commit.transfer_duration_ns > 0);
        let status = service.status();
        let resident: Vec<u32> = status
            .staged_layers
            .iter()
            .map(|layer| layer.layer)
            .collect();
        assert_eq!(resident, target);
        assert!(!status.pipeline_pending);
    }
    let lifecycle = service.status().staging_lifecycle.expect("lifecycle");
    assert_eq!(lifecycle.allocation_count, 4);
    assert!(lifecycle.reuse_count >= 4);
    assert_eq!(lifecycle.commit_count, 3);
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL, a CUDA build, and a physical GPU"]
fn pipeline_rejects_two_slots_over_budget_and_abort_preserves_active() {
    let _guard = env_lock();
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let plan = plan_for(&path);
    let initial = StagingRequest {
        layers: vec![0, 1],
        budget_bytes: Some(8 * 1024 * 1024),
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

    let error = service
        .prepare_staged_pipeline(StagingRequest {
            layers: vec![1, 2],
            budget_bytes: Some(8 * 1024 * 1024),
            auto: false,
            safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
        })
        .expect_err("two slots must not exceed the budget");
    assert!(
        matches!(error, ServiceError::Configuration(_)),
        "unexpected: {error}"
    );
    let status = service.status();
    assert!(!status.pipeline_pending);
    let resident: Vec<u32> = status
        .staged_layers
        .iter()
        .map(|layer| layer.layer)
        .collect();
    assert_eq!(resident, vec![0, 1]);
    assert!(!generate(&mut service).is_empty());

    // A fitting service can prepare and then abort without changing residency.
    let plan = plan_for(&path);
    let mut aborting = InferenceService::new(plan)
        .expect("service")
        .with_staging_pool_limit(20 * 1024 * 1024)
        .with_staging_request(StagingRequest {
            layers: vec![0, 1],
            budget_bytes: Some(20 * 1024 * 1024),
            auto: false,
            safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
        });
    aborting.load().expect("load");
    aborting.create_session().expect("session");
    assert!(!generate(&mut aborting).is_empty());
    aborting
        .prepare_staged_pipeline(StagingRequest {
            layers: vec![1, 2],
            budget_bytes: Some(20 * 1024 * 1024),
            auto: false,
            safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
        })
        .expect("prepare");
    assert!(aborting.status().pipeline_pending);
    aborting.abort_staged_pipeline().expect("abort");
    let status = aborting.status();
    assert!(!status.pipeline_pending);
    let resident: Vec<u32> = status
        .staged_layers
        .iter()
        .map(|layer| layer.layer)
        .collect();
    assert_eq!(resident, vec![0, 1]);
    assert!(!generate(&mut aborting).is_empty());
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL, a CUDA build, and a physical GPU"]
fn async_host_read_abort_and_commit_preserve_lifecycle() {
    let _guard = env_lock();
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let plan = plan_for(&path);
    let mut service = InferenceService::new(plan)
        .expect("service")
        .with_host_staging(
            ramforge_runtime::HostStagingMode::Direct,
            ramforge_runtime::ReadBatching::Coalesced,
            32 * 1024 * 1024,
        )
        .with_async_host_read(true)
        .with_staging_request(StagingRequest {
            layers: vec![0, 1],
            budget_bytes: Some(20 * 1024 * 1024),
            auto: false,
            safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
        });
    service.load().expect("load");
    service.create_session().expect("session");
    assert!(!generate(&mut service).is_empty());

    let request = || StagingRequest {
        layers: vec![1, 2],
        budget_bytes: Some(20 * 1024 * 1024),
        auto: false,
        safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
    };

    service
        .prepare_staged_pipeline(request())
        .expect("prepare async host read");
    assert!(service.status().pipeline_pending);
    service.abort_staged_pipeline().expect("abort host read");
    let status = service.status();
    assert!(!status.pipeline_pending);
    let resident: Vec<u32> = status
        .staged_layers
        .iter()
        .map(|layer| layer.layer)
        .collect();
    assert_eq!(resident, vec![0, 1]);
    assert!(!generate(&mut service).is_empty());

    service
        .prepare_staged_pipeline(request())
        .expect("prepare async host read");
    assert!(!generate(&mut service).is_empty());
    let commit = service.commit_staged_pipeline().expect("commit host read");
    assert!(commit.host_staging.bytes_read > 0);
    let prefetch = service.status().prefetch;
    assert_eq!(prefetch.committed, 1);
    assert_eq!(prefetch.aborted, 1);
    assert_eq!(prefetch.hit_rate_permille, Some(500));
    let resident: Vec<u32> = service
        .status()
        .staged_layers
        .iter()
        .map(|layer| layer.layer)
        .collect();
    assert_eq!(resident, vec![1, 2]);
    assert!(!generate(&mut service).is_empty());
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL, a CUDA build, and a physical GPU"]
fn partial_residency_stages_only_delta_and_preserves_shared_layers() {
    let _guard = env_lock();
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let plan = plan_for(&path);
    let mut service = InferenceService::new(plan)
        .expect("service")
        .with_host_staging(
            ramforge_runtime::HostStagingMode::Direct,
            ramforge_runtime::ReadBatching::Coalesced,
            32 * 1024 * 1024,
        )
        .with_async_host_read(false)
        .with_partial_residency(true)
        .with_staging_request(StagingRequest {
            layers: vec![0, 1],
            budget_bytes: Some(20 * 1024 * 1024),
            auto: false,
            safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
        });
    service.load().expect("load");
    service.create_session().expect("session");
    assert!(!generate(&mut service).is_empty());

    let prepared = service
        .prepare_staged_pipeline(StagingRequest {
            layers: vec![1, 2],
            budget_bytes: Some(20 * 1024 * 1024),
            auto: false,
            safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
        })
        .expect("prepare partial");
    assert!(prepared.pipeline_partial);
    assert_eq!(prepared.pipeline_preserved_bytes, 2_598_964);
    assert_eq!(prepared.pipeline_new_bytes, 2_329_268);
    let host = prepared.host_staging.expect("host report");
    assert!(
        host.bytes_read < 2_598_964,
        "partial read staged shared data"
    );
    assert!(!generate(&mut service).is_empty());
    let commit = service.commit_staged_pipeline().expect("commit partial");
    assert!(commit.partial);
    assert_eq!(commit.preserved_layers, vec![1]);
    assert_eq!(commit.new_layers, vec![2]);
    assert_eq!(commit.removed_layers, vec![0]);
    assert!(commit.preserved_bytes > 0);
    assert!(commit.transfer_bytes < 2_598_964);
    let status = service.status();
    let resident: Vec<u32> = status
        .staged_layers
        .iter()
        .map(|layer| layer.layer)
        .collect();
    assert_eq!(resident, vec![1, 2]);
    assert!(status.prefetch.bytes_preserved > 0);
    assert!(!generate(&mut service).is_empty());

    // Removal-only delta: no storage or transfer should be needed.
    service
        .prepare_staged_pipeline(StagingRequest {
            layers: vec![1],
            budget_bytes: Some(20 * 1024 * 1024),
            auto: false,
            safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
        })
        .expect("prepare removal-only");
    let commit = service
        .commit_staged_pipeline()
        .expect("commit removal-only");
    assert!(commit.new_layers.is_empty());
    assert_eq!(commit.removed_layers, vec![2]);
    assert_eq!(commit.transfer_bytes, 0);
    let resident: Vec<u32> = service
        .status()
        .staged_layers
        .iter()
        .map(|layer| layer.layer)
        .collect();
    assert_eq!(resident, vec![1]);
    assert!(!generate(&mut service).is_empty());
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL, a CUDA build, and a physical GPU"]
fn failed_partial_prepare_preserves_active_residency() {
    let _guard = env_lock();
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let plan = plan_for(&path);
    let mut service = InferenceService::new(plan)
        .expect("service")
        .with_partial_residency(true)
        .with_staging_request(StagingRequest {
            layers: vec![0, 1],
            budget_bytes: Some(20 * 1024 * 1024),
            auto: false,
            safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
        });
    service.load().expect("load");
    service.create_session().expect("session");
    assert!(!generate(&mut service).is_empty());

    std::env::set_var("RAMFORGE_TEST_FAIL_PENDING_LAYER", "3");
    let error = service
        .prepare_staged_pipeline(StagingRequest {
            layers: vec![2, 3],
            budget_bytes: Some(20 * 1024 * 1024),
            auto: false,
            safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
        })
        .expect_err("injected pending failure must be reported");
    std::env::remove_var("RAMFORGE_TEST_FAIL_PENDING_LAYER");
    assert!(error.to_string().contains("injected pending stage failure"));
    let status = service.status();
    assert!(!status.pipeline_pending);
    let resident: Vec<u32> = status
        .staged_layers
        .iter()
        .map(|layer| layer.layer)
        .collect();
    assert_eq!(resident, vec![0, 1]);
    assert!(!generate(&mut service).is_empty());

    service
        .prepare_staged_pipeline(StagingRequest {
            layers: vec![1, 2],
            budget_bytes: Some(20 * 1024 * 1024),
            auto: false,
            safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
        })
        .expect("prepare after failure");
    let second = service
        .prepare_staged_pipeline(StagingRequest {
            layers: vec![3],
            budget_bytes: Some(20 * 1024 * 1024),
            auto: false,
            safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
        })
        .expect_err("concurrent transaction must be rejected");
    assert!(second.to_string().contains("already pending"));
    service.abort_staged_pipeline().expect("abort");
    let resident: Vec<u32> = service
        .status()
        .staged_layers
        .iter()
        .map(|layer| layer.layer)
        .collect();
    assert_eq!(resident, vec![0, 1]);
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL, a CUDA build, and a physical GPU"]
fn standby_cache_makes_alternating_transitions_zero_read() {
    let _guard = env_lock();
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let plan = plan_for(&path);
    let mut service = InferenceService::new(plan)
        .expect("service")
        .with_host_staging(
            ramforge_runtime::HostStagingMode::Direct,
            ramforge_runtime::ReadBatching::Coalesced,
            32 * 1024 * 1024,
        )
        .with_staging_standby_limit(20 * 1024 * 1024)
        .with_partial_residency(true)
        .with_staging_request(StagingRequest {
            layers: vec![0, 1],
            budget_bytes: Some(20 * 1024 * 1024),
            auto: false,
            safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
        });
    service.load().expect("load");
    service.create_session().expect("session");
    assert!(!generate(&mut service).is_empty());

    let prepare = |service: &mut InferenceService, layers: Vec<u32>| {
        service
            .prepare_staged_pipeline(StagingRequest {
                layers,
                budget_bytes: Some(20 * 1024 * 1024),
                auto: false,
                safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
            })
            .expect("prepare")
    };

    let first = prepare(&mut service, vec![1, 2]);
    assert!(first.host_staging.expect("host").bytes_read > 0);
    assert!(!generate(&mut service).is_empty());
    let first_commit = service.commit_staged_pipeline().expect("commit 1");
    assert_eq!(first_commit.new_layers, vec![2]);

    let second = prepare(&mut service, vec![0, 1]);
    assert_eq!(second.host_staging.expect("host").bytes_read, 0);
    assert_eq!(second.standby_reused_layers, vec![0]);
    assert!(!generate(&mut service).is_empty());
    let second_commit = service.commit_staged_pipeline().expect("commit 2");
    assert_eq!(second_commit.transfer_bytes, 0);
    assert_eq!(second_commit.promoted_layers, vec![0]);
    assert!(second_commit.promoted_bytes > 0);
    let status = service.status();
    let resident: Vec<u32> = status
        .staged_layers
        .iter()
        .map(|layer| layer.layer)
        .collect();
    assert_eq!(resident, vec![0, 1]);
    assert!(status.prefetch.bytes_from_standby > 0);
    assert!(status.staging_lifecycle.expect("lifecycle").standby_hits >= 1);
    assert!(!generate(&mut service).is_empty());
}
