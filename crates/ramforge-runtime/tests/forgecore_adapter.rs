//! Focused ForgeCore adapter coverage.
//!
//! The native-backed tests use ForgeCore's deterministic tiny GGUF fixture
//! through FORGE_TEST_MODEL. The binary is deliberately not stored in this
//! workspace; see forgeCORE/docs/NATIVE.md and the Phase-2 handoff notes.

use std::path::PathBuf;

use ramforge_runtime::{
    DeviceRegistry, DeviceSelection, ExecutionConfig, ForgeCoreAdapterError,
    ForgeCoreContextOptions, ForgeCoreModel, ForgeCoreModelMetadata,
};

fn fixture() -> Option<PathBuf> {
    match std::env::var("FORGE_TEST_MODEL") {
        Ok(path) => Some(PathBuf::from(path)),
        Err(_) => {
            println!("SKIP: FORGE_TEST_MODEL not set (use ForgeCore's tiny GGUF fixture)");
            None
        }
    }
}

fn load_fixture() -> Option<(ForgeCoreModel, ForgeCoreModelMetadata)> {
    let path = fixture()?;
    let model = ForgeCoreModel::load(&path).expect("deterministic ForgeCore fixture must load");
    let metadata = model.metadata().clone();
    Some((model, metadata))
}

#[test]
fn forgecore_selection_errors_are_propagated_without_fallback() {
    let missing_id = ExecutionConfig::new(
        DeviceSelection::by_id("missing-device"),
        1,
        None,
        false,
    )
    .expect("selection syntax is valid");
    let error = ForgeCoreModel::load_for_execution(
        "/definitely/not/a/model.gguf",
        missing_id,
        &DeviceRegistry::cpu_only(),
    )
    .unwrap_err();
    assert!(matches!(error, ForgeCoreAdapterError::DeviceSelection(_)));

    let no_cpu = ExecutionConfig::new(DeviceSelection::Cpu, 1, None, false)
        .expect("CPU selection is valid");
    let error = ForgeCoreModel::load_for_execution(
        "/definitely/not/a/model.gguf",
        no_cpu,
        &DeviceRegistry::from_devices(Vec::new()),
    )
    .unwrap_err();
    assert!(matches!(error, ForgeCoreAdapterError::DeviceSelection(_)));
}

#[test]
fn forgecore_model_and_context_use_native_metadata() {
    let Some((model, metadata)) = load_fixture() else {
        return;
    };
    assert_eq!(metadata.n_params, 1176);
    assert_eq!(metadata.vocab_size, 32);
    assert_eq!(metadata.size_bytes, 4704);
    assert_eq!(metadata.context_length, 64);
    assert_eq!(metadata.embedding_length, 8);
    assert_eq!(metadata.layer_count, 1);
    assert_eq!(metadata.head_count, 2);
    assert_eq!(metadata.kv_head_count, 2);
    assert!(metadata.description.contains("llama"));

    let session = model
        .open_context(ForgeCoreContextOptions::default())
        .expect("default CPU context must open");
    // ForgeCore exposes the effective native length, including its 256-token
    // padding; model metadata above retains the training length (64).
    assert_eq!(session.context_length(), 256);
    assert_eq!(session.vocab_size(), 32);
}

#[test]
fn forgecore_batch_decode_returns_owned_logits_for_last_position() {
    let Some((model, _)) = load_fixture() else {
        return;
    };
    let mut session = model
        .open_context(ForgeCoreContextOptions::default())
        .expect("context");
    let logits = session
        .decode_tokens(&[1, 5, 6, 7], 0)
        .expect("tiny batch must decode");
    assert_eq!(logits.token_index(), 3);
    assert_eq!(logits.n_vocab(), 32);
    assert!(logits.values().iter().all(|value| value.is_finite()));
    assert!(logits.values().iter().any(|value| *value != 0.0));
}

#[test]
fn forgecore_adapter_translates_model_batch_and_decode_errors() {
    let missing = ForgeCoreModel::load("/definitely/not/a/model.gguf").unwrap_err();
    assert!(matches!(missing, ForgeCoreAdapterError::Model(_)));

    let Some((model, _)) = load_fixture() else {
        return;
    };
    let invalid_context = model
        .open_context(ForgeCoreContextOptions {
            n_ctx: None,
            n_threads: 0,
        })
        .unwrap_err();
    assert!(matches!(
        invalid_context,
        ForgeCoreAdapterError::Configuration(_)
    ));

    let mut session = model
        .open_context(ForgeCoreContextOptions::default())
        .expect("context");

    let empty = session.decode_tokens(&[], 0).unwrap_err();
    assert!(matches!(empty, ForgeCoreAdapterError::Batch(_)));

    let out_of_vocab = session.decode_tokens(&[32], 0).unwrap_err();
    assert!(matches!(out_of_vocab, ForgeCoreAdapterError::Batch(_)));

    session
        .decode_tokens(&[1, 5], 0)
        .expect("valid batch for logits error checks");
    let unflagged = session.logits(0).unwrap_err();
    assert!(matches!(unflagged, ForgeCoreAdapterError::Logits(_)));
    let beyond = session.logits(99).unwrap_err();
    assert!(matches!(beyond, ForgeCoreAdapterError::Logits(_)));

    let mut constrained = model
        .open_context(ForgeCoreContextOptions {
            n_ctx: Some(1),
            n_threads: 1,
        })
        .expect("one-token context");
    constrained
        .decode_token(1, 0)
        .expect("first token in one-token context");
    let native_decode_error = constrained.decode_token(2, 0).unwrap_err();
    assert!(matches!(
        native_decode_error,
        ForgeCoreAdapterError::Decode(_)
    ));
}

#[test]
fn forgecore_boundary_keeps_native_ffi_out_of_ramforge() {
    let manifest = include_str!("../Cargo.toml");
    let adapter = include_str!("../src/forgecore.rs");
    assert!(manifest.contains("forge-core"));
    assert!(!manifest.contains("forge-sys"));
    assert!(!adapter.contains("forge_sys"));
    assert!(!adapter.contains("extern \"C\""));
}
