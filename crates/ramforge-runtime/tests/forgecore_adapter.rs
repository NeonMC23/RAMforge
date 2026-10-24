//! Focused ForgeCore adapter coverage.
//!
//! The native-backed tests use ForgeCore's deterministic tiny GGUF fixture
//! through FORGE_TEST_MODEL. The binary is deliberately not stored in this
//! workspace; see forgeCORE/docs/NATIVE.md and the Phase-2 handoff notes.

use std::path::PathBuf;

use ramforge_core::parse_gguf_file;
use ramforge_runtime::memory_report::{process_rss, ForgeCoreMemoryReport};
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
fn forgecore_contexts_keep_independent_native_state_across_drops() {
    let Some((model, metadata)) = load_fixture() else {
        return;
    };
    let vocab = usize::try_from(metadata.vocab_size).expect("vocab size fits usize");
    let mut first = model
        .open_context(ForgeCoreContextOptions::default())
        .expect("first context");
    let mut second = model
        .open_context(ForgeCoreContextOptions::default())
        .expect("second context");

    assert_eq!(
        first
            .decode_tokens(&[1, 5], 0)
            .expect("first decode")
            .n_vocab(),
        vocab
    );
    let second_logits = second.decode_tokens(&[1, 6], 0).expect("second decode");
    assert_eq!(second_logits.n_vocab(), vocab);
    assert!(second_logits.values().iter().all(|value| value.is_finite()));

    // A session owns a shared model handle and its own native context. Dropping
    // the first context must not invalidate the second or the model wrapper.
    drop(first);
    let after_drop = second
        .decode_tokens(&[7], 2)
        .expect("second context survives first drop");
    assert_eq!(after_drop.token_index(), 0);
    assert_eq!(after_drop.n_vocab(), vocab);

    // Owned logits stay valid after the session that produced them is dropped.
    drop(second);
    assert_eq!(second_logits.values().len(), vocab);

    let mut third = model
        .open_context(ForgeCoreContextOptions::default())
        .expect("model remains usable after context drops");
    assert_eq!(
        third
            .decode_tokens(&[1, 8], 0)
            .expect("third decode")
            .n_vocab(),
        vocab
    );
}

#[test]
fn forgecore_memory_report_uses_real_native_facts() {
    let Some(path) = fixture() else {
        return;
    };
    let rss_before_load = process_rss();
    let model = ForgeCoreModel::load(&path).expect("fixture load");
    let rss_after_load = process_rss();
    let metadata = model.metadata().clone();
    let gguf = parse_gguf_file(&path).expect("RAMforge GGUF parse");
    let logical = gguf
        .total_tensor_bytes()
        .expect("fixture tensor sizes are known");
    let report = ForgeCoreMemoryReport::new(
        logical.saturating_sub(1),
        Some(logical),
        metadata.size_bytes,
        rss_before_load,
        rss_after_load,
        process_rss(),
        model.device(),
    );

    assert!(!report.enforces_budget());
    assert_eq!(report.logical_model_bytes, Some(logical));
    assert_eq!(report.native_model_bytes, metadata.size_bytes);
    assert_eq!(report.logical_model_exceeds_budget(), Some(true));
    assert!(logical > 0);

    #[cfg(target_os = "linux")]
    {
        assert!(report.rss_before_load_bytes.is_some());
        assert!(report.rss_after_load_bytes.is_some());
        assert!(report.rss_delta_after_load_bytes().is_some());
    }
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

/// Opt-in GPT-NeoX/Pythia execution through RAMforge's safe adapter.
///
/// This is the integration boundary that RAMforge's `--backend forgecore`
/// uses. The legacy executor rejects `gptneox` intentionally; the ForgeCore
/// backend must load, decode, and generate without falling back or
/// approximating the architecture.
#[test]
#[ignore = "requires FORGE_GPTNEOX_MODEL=<gptneox gguf>; opt-in real-model test"]
fn forgecore_gptneox_generates_tokens_through_the_adapter() {
    let Ok(path) = std::env::var("FORGE_GPTNEOX_MODEL") else {
        println!("SKIP: FORGE_GPTNEOX_MODEL not set (opt-in GPT-NeoX fixture)");
        return;
    };
    let path = PathBuf::from(path);
    let model = ForgeCoreModel::load(&path).expect("GPT-NeoX fixture must load");
    let metadata = model.metadata().clone();
    assert!(
        metadata
            .description
            .to_ascii_lowercase()
            .contains("gptneox"),
        "fixture must be a GPT-NeoX model, got: {}",
        metadata.description
    );
    assert!(metadata.layer_count > 0);
    assert!(metadata.vocab_size > 0);

    let gguf = parse_gguf_file(&path).expect("RAMforge GGUF inspection");
    let tokenizer = ramforge_core::Tokenizer::from_gguf(&gguf).expect("RAMforge tokenizer");
    let prompt = tokenizer.encode("The capital of France is", true);
    assert!(!prompt.is_empty());
    assert_eq!(tokenizer.vocab_size(), metadata.vocab_size as usize);
    println!("ramforge gptneox prompt ids: {prompt:?}");

    let mut session = model
        .open_context(ForgeCoreContextOptions::default())
        .expect("GPT-NeoX context");
    let mut logits = session
        .decode_tokens(&prompt, 0)
        .expect("GPT-NeoX prompt decode");
    assert!(logits.values().iter().all(|value| value.is_finite()));

    let vocab = metadata.vocab_size as usize;
    let mut generated = Vec::new();
    let start_position = prompt.len();
    for position in start_position..start_position + 4 {
        let mut best = 0usize;
        for (index, value) in logits.values().iter().enumerate().skip(1) {
            if *value > logits.values()[best] {
                best = index;
            }
        }
        assert!(best < vocab);
        generated.push(best as u32);
        logits = session
            .decode_token(
                best as u32,
                u32::try_from(position).expect("generated position fits u32"),
            )
            .expect("GPT-NeoX generated-token decode");
    }
    assert_eq!(generated.len(), 4);
    println!("forgecore gptneox generated ids: {generated:?}");
}
