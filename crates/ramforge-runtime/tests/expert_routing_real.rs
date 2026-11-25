//! Opt-in real-model routing observation through the RAMforge adapter.
//!
//! Requires `FORGE_TEST_MODEL=<MoE gguf>` and a CPU-capable native build.

use ramforge_runtime::{
    expert_layout, ExpertRoutingOptions, ForgeCoreContextOptions, ForgeCoreModel,
};

#[test]
#[ignore = "requires FORGE_TEST_MODEL=<MoE gguf>; opt-in real-model adapter routing test"]
fn adapter_forwards_routing_observation_without_changing_decode() {
    let Some(path) = std::env::var_os("FORGE_TEST_MODEL").map(std::path::PathBuf::from) else {
        return;
    };
    if !path.is_file() {
        return;
    }
    let model = ForgeCoreModel::load(&path).expect("model load");
    let metadata = model.metadata().clone();
    let mut session = model
        .open_context(ForgeCoreContextOptions {
            n_ctx: Some(128),
            n_threads: 4,
            n_batch: Some(64),
            n_ubatch: Some(64),
            routing: Some(ExpertRoutingOptions {
                max_observations: 4096,
                expert_count: 64,
                expert_used_count: 6,
            }),
            ..ForgeCoreContextOptions::default()
        })
        .expect("routing session");
    let mut logits = session.decode_tokens(&[1, 2], 0).expect("prefill");
    for position in 2..5u32 {
        let next = logits
            .values()
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index as u32)
            .expect("logits");
        logits = session.decode_token(next, position).expect("decode");
    }
    let observations = session.routing_observations();
    let summary = session.routing_summary().expect("summary");
    assert!(summary.complete_decode_observations > 0);
    assert!(observations.iter().all(|observation| {
        observation.selected_experts.len() == 6 && observation.invalid_indices == 0
    }));
    assert_eq!(logits.values().len(), metadata.vocab_size as usize);

    let set = ramforge_core::discover_model_set(&path).expect("model set");
    let layout = expert_layout(&set.merged_model());
    assert!(layout.total_routed_expert_bytes > 0);
    assert!(layout
        .layers
        .iter()
        .any(|layer| layer.per_expert_bytes.is_some()));
}
