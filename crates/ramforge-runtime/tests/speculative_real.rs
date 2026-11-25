//! Opt-in real-model speculative correctness test.
//!
//! Requires `FORGE_TEST_MODEL` (target) and `FORGE_DRAFT_MODEL` (compatible
//! draft). The test compares greedy target output with greedy speculative
//! output token-for-token.

use std::path::PathBuf;

use ramforge_core::{discover_model_set, Tokenizer};
use ramforge_runtime::{
    draft_vocabulary_compatible, generate_speculative, ForgeCoreContextOptions, ForgeCoreModel,
    SpeculativeConfig,
};

fn env_model(name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os(name)?);
    path.is_file().then_some(path)
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL and FORGE_DRAFT_MODEL compatible GGUF files"]
fn speculative_greedy_matches_ordinary_target_tokens() {
    let Some(target_path) = env_model("FORGE_TEST_MODEL") else {
        return;
    };
    let Some(draft_path) = env_model("FORGE_DRAFT_MODEL") else {
        return;
    };
    let target_set = discover_model_set(&target_path).expect("target set");
    let draft_set = discover_model_set(&draft_path).expect("draft set");
    let target_gguf = target_set.merged_model();
    let draft_gguf = draft_set.merged_model();
    draft_vocabulary_compatible(&draft_gguf, &target_gguf).expect("compatible vocabularies");

    let tokenizer = Tokenizer::from_gguf(&target_gguf).expect("tokenizer");
    let prompt = tokenizer.encode("The capital of France is", true);
    let target_model = ForgeCoreModel::load(target_set.primary_path()).expect("target load");
    let draft_model = ForgeCoreModel::load(&draft_path).expect("draft load");
    let context_options = |outputs: u32| ForgeCoreContextOptions {
        n_ctx: Some(256),
        n_threads: 4,
        n_batch: Some(64),
        n_ubatch: Some(64),
        n_outputs_max: Some(outputs),
        n_outputs_max_per_seq: Some(outputs),
        ..ForgeCoreContextOptions::default()
    };

    let mut reference = target_model
        .open_context(context_options(1))
        .expect("reference context");
    let mut logits = reference.decode_tokens(&prompt, 0).expect("prompt");
    let mut expected = Vec::new();
    for position in prompt.len() as u32..prompt.len() as u32 + 6 {
        let token = logits
            .values()
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index as u32)
            .expect("logits");
        if Some(token) == tokenizer.eos_id {
            break;
        }
        expected.push(token);
        logits = reference.decode_token(token, position).expect("decode");
    }

    let mut target = target_model
        .open_context(context_options(5))
        .expect("target context");
    let mut draft = draft_model
        .open_context(context_options(5))
        .expect("draft context");
    let output = generate_speculative(
        &mut target,
        &mut draft,
        &prompt,
        SpeculativeConfig {
            max_draft_tokens: 4,
            max_tokens: 6,
        },
        tokenizer.eos_id,
        |_| Ok(()),
    )
    .expect("speculative");
    assert_eq!(output.token_ids, expected);
    assert!(output.stats.target_forwards >= 1);
    assert!(output.stats.emitted_tokens == expected.len() as u64);
}
