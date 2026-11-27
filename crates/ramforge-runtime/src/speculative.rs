//! Greedy speculative decoding with multi-token target verification.
//!
//! The target model verifies a batch of draft candidates in one forward and
//! returns logits for every position. The first mismatch becomes the target
//! fallback. The fallback is carried as the first token of the next verification
//! batch, so a round costs one target forward rather than a separate fallback
//! decode.
//!
//! This module is deliberately restricted to greedy sampling. Exact
//! distribution-preserving speculative sampling for temperature/top-k/top-p is
//! not implemented; callers must reject those configurations.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::forgecore::{ForgeCoreAdapterError, ForgeCoreLogits, ForgeCoreSession};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeConfig {
    pub max_draft_tokens: u32,
    pub max_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculativeRound {
    pub round: u32,
    pub candidates: u32,
    pub accepted: u32,
    pub rejected: u32,
    pub fallback: u32,
    pub target_evaluated: u32,
    pub target_ns: u64,
    pub draft_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculativeStats {
    pub prompt_tokens: u64,
    pub emitted_tokens: u64,
    pub rounds: u64,
    pub draft_tokens_generated: u64,
    pub target_positions_evaluated: u64,
    pub accepted_tokens: u64,
    pub rejected_tokens: u64,
    pub fallback_tokens: u64,
    pub target_only_tokens: u64,
    pub target_forwards: u64,
    pub acceptance_rate_bps: u32,
    pub avg_accepted_per_round_x100: u32,
    pub target_forward_ns: u64,
    pub draft_forward_ns: u64,
    pub used_target_only_fallback: bool,
    pub prompt_ns: u64,
    pub first_token_ns: Option<u64>,
    pub total_ns: u64,
    pub stop_reason: String,
    pub rounds_detail: Vec<SpeculativeRound>,
    /// Number of candidate positions tested at each draft index. Length is
    /// `max_draft_tokens`; late rounds may test fewer.
    #[serde(default)]
    pub position_attempts: Vec<u64>,
    /// Number of candidates accepted at each draft index.
    #[serde(default)]
    pub position_accepts: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculativeOutput {
    pub token_ids: Vec<u32>,
    pub stats: SpeculativeStats,
}

fn argmax(values: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_value = values.first().copied().unwrap_or(f32::NEG_INFINITY);
    for (index, value) in values.iter().enumerate().skip(1) {
        if *value > best_value {
            best = index;
            best_value = *value;
        }
    }
    best as u32
}

/// Pure acceptance logic shared by the decoder and unit tests.
///
/// `target_logits` has one entry per candidate when there is no pending token,
/// or one extra entry for the pending token at index 0. `prompt_logits` predicts
/// the token after the already-decoded context and is used only when there is no
/// pending token.
fn verify_candidates(
    candidates: &[u32],
    has_pending: bool,
    prompt_logits: &[f32],
    target_logits: &[&[f32]],
) -> (u32, u32) {
    let pending_offset = usize::from(has_pending);
    let mut accepted = 0u32;
    let fallback: u32;
    let mut previous: &[f32] = if has_pending {
        target_logits.first().copied().unwrap_or(&[])
    } else {
        prompt_logits
    };
    for (index, candidate) in candidates.iter().enumerate() {
        let target = argmax(previous);
        if *candidate == target {
            accepted += 1;
            previous = target_logits
                .get(index + pending_offset)
                .copied()
                .unwrap_or(&[]);
        } else {
            fallback = target;
            return (accepted, fallback);
        }
    }
    fallback = target_logits.last().copied().map(argmax).unwrap_or(0);
    (accepted, fallback)
}

/// Run one greedy speculative generation.
///
/// `prompt_tokens` are decoded into both contexts at position 0. The sessions
/// must be opened with `n_outputs_max_per_seq >= max_draft_tokens + 1` and a KV
/// dtype supported by the backend. The draft vocabulary must be a compatible
/// subset of the target vocabulary; this is validated by the caller from GGUF
/// metadata.
pub fn generate_speculative<F>(
    target: &mut ForgeCoreSession,
    draft: &mut ForgeCoreSession,
    prompt_tokens: &[u32],
    config: SpeculativeConfig,
    eos: Option<u32>,
    mut on_token: F,
) -> Result<SpeculativeOutput, ForgeCoreAdapterError>
where
    F: FnMut(u32) -> Result<(), ForgeCoreAdapterError>,
{
    if prompt_tokens.is_empty() {
        return Err(ForgeCoreAdapterError::Configuration(
            "speculative prompt is empty".to_string(),
        ));
    }
    if config.max_draft_tokens == 0 {
        return Err(ForgeCoreAdapterError::Configuration(
            "max draft tokens must be greater than zero".to_string(),
        ));
    }
    if config.max_tokens == 0 {
        return Err(ForgeCoreAdapterError::Configuration(
            "max tokens must be greater than zero".to_string(),
        ));
    }
    let target_batch = target.effective_batch_tokens();
    let draft_batch = draft.effective_batch_tokens();
    if config.max_draft_tokens.saturating_add(1) > target_batch {
        return Err(ForgeCoreAdapterError::Configuration(format!(
            "draft length {} plus pending token exceeds target n_batch {}",
            config.max_draft_tokens, target_batch
        )));
    }
    if config.max_draft_tokens > draft_batch {
        return Err(ForgeCoreAdapterError::Configuration(format!(
            "draft length {} exceeds draft n_batch {}",
            config.max_draft_tokens, draft_batch
        )));
    }
    let max_context = target.context_length().min(draft.context_length());
    if prompt_tokens.len() as u32 >= max_context {
        return Err(ForgeCoreAdapterError::Configuration(format!(
            "prompt has {} tokens but effective context is {}",
            prompt_tokens.len(),
            max_context
        )));
    }

    let generation_started = Instant::now();
    let draft_vocab = draft.vocab_size();
    if prompt_tokens.iter().any(|token| *token >= draft_vocab) {
        return Err(ForgeCoreAdapterError::Configuration(
            "prompt contains tokens outside the draft vocabulary; use ordinary decoding"
                .to_string(),
        ));
    }
    let prompt_started = Instant::now();
    let prompt_logits = target.decode_tokens(prompt_tokens, 0)?;
    let mut draft_logits = draft.decode_tokens(prompt_tokens, 0)?;
    let prompt_ns = prompt_started.elapsed().as_nanos() as u64;

    let mut token_ids = Vec::new();
    let mut rounds_detail = Vec::new();
    let mut stats = SpeculativeStats {
        prompt_tokens: prompt_tokens.len() as u64,
        emitted_tokens: 0,
        rounds: 0,
        draft_tokens_generated: 0,
        target_positions_evaluated: 0,
        accepted_tokens: 0,
        rejected_tokens: 0,
        fallback_tokens: 0,
        target_only_tokens: 0,
        target_forwards: 0,
        acceptance_rate_bps: 0,
        avg_accepted_per_round_x100: 0,
        target_forward_ns: 0,
        draft_forward_ns: 0,
        used_target_only_fallback: false,
        prompt_ns,
        first_token_ns: None,
        total_ns: 0,
        stop_reason: String::new(),
        rounds_detail: Vec::new(),
        position_attempts: vec![0; config.max_draft_tokens as usize],
        position_accepts: vec![0; config.max_draft_tokens as usize],
    };

    let mut target_position = prompt_tokens.len() as u32;
    let mut draft_position = prompt_tokens.len() as u32;
    let mut pending: Option<u32> = None;
    let mut prompt_target_logits: Option<ForgeCoreLogits> = Some(prompt_logits);
    let mut hit_eos = false;

    while token_ids.len() < config.max_tokens as usize && target_position < max_context {
        if token_ids.len() >= config.max_tokens as usize {
            break;
        }
        let pending_for_target = pending.take();
        if let Some(token) = pending_for_target {
            if token >= draft_vocab {
                // The target emitted a token that the draft cannot represent.
                // Finish with ordinary greedy target decoding without losing
                // the already-emitted prefix.
                stats.used_target_only_fallback = true;
                let started = Instant::now();
                let mut logits = target.decode_token(token, target_position)?;
                stats.target_forward_ns = stats
                    .target_forward_ns
                    .saturating_add(started.elapsed().as_nanos() as u64);
                stats.target_forwards += 1;
                stats.target_positions_evaluated += 1;
                target_position += 1;
                while token_ids.len() < config.max_tokens as usize && target_position < max_context
                {
                    let next = argmax(logits.values());
                    if Some(next) == eos {
                        hit_eos = true;
                        break;
                    }
                    if stats.first_token_ns.is_none() {
                        stats.first_token_ns = Some(generation_started.elapsed().as_nanos() as u64);
                    }
                    on_token(next)?;
                    token_ids.push(next);
                    stats.target_only_tokens += 1;
                    if token_ids.len() >= config.max_tokens as usize {
                        break;
                    }
                    let started = Instant::now();
                    logits = target.decode_token(next, target_position)?;
                    stats.target_forward_ns = stats
                        .target_forward_ns
                        .saturating_add(started.elapsed().as_nanos() as u64);
                    stats.target_forwards += 1;
                    stats.target_positions_evaluated += 1;
                    target_position += 1;
                }
                break;
            }
            let started = Instant::now();
            draft_logits = draft.decode_token(token, draft_position)?;
            stats.draft_forward_ns = stats
                .draft_forward_ns
                .saturating_add(started.elapsed().as_nanos() as u64);
            draft_position += 1;
        }
        let candidate_start = target_position + u32::from(pending_for_target.is_some());

        let remaining_context = max_context.saturating_sub(candidate_start);
        let remaining_output = config
            .max_tokens
            .saturating_sub(token_ids.len() as u32)
            .saturating_sub(1);
        let draft_limit = config
            .max_draft_tokens
            .min(remaining_context)
            .min(remaining_output.max(1));
        let mut candidates = Vec::new();
        let mut drafted_ns = 0u64;
        for _ in 0..draft_limit {
            let token = argmax(draft_logits.values());
            candidates.push(token);
            let started = Instant::now();
            draft_logits = draft.decode_token(token, draft_position)?;
            drafted_ns = drafted_ns.saturating_add(started.elapsed().as_nanos() as u64);
            draft_position += 1;
            if Some(token) == eos {
                break;
            }
        }
        stats.draft_forward_ns = stats.draft_forward_ns.saturating_add(drafted_ns);
        if candidates.is_empty() {
            break;
        }

        let mut batch =
            Vec::with_capacity(candidates.len() + usize::from(pending_for_target.is_some()));
        if let Some(token) = pending_for_target {
            batch.push(token);
        }
        batch.extend_from_slice(&candidates);
        let started = Instant::now();
        let verified = target.decode_tokens_all_logits(&batch, target_position)?;
        let target_ns = started.elapsed().as_nanos() as u64;
        stats.target_forward_ns = stats.target_forward_ns.saturating_add(target_ns);
        stats.target_forwards += 1;
        stats.target_positions_evaluated = stats
            .target_positions_evaluated
            .saturating_add(batch.len() as u64);
        stats.draft_tokens_generated = stats
            .draft_tokens_generated
            .saturating_add(candidates.len() as u64);

        let prompt_values = prompt_target_logits
            .as_ref()
            .map(|logits| logits.values())
            .unwrap_or(&[]);
        let verified_values: Vec<&[f32]> = verified.iter().map(|logits| logits.values()).collect();
        let (accepted, fallback) = verify_candidates(
            &candidates,
            pending_for_target.is_some(),
            prompt_values,
            &verified_values,
        );

        for (position, _) in candidates.iter().enumerate() {
            if let Some(attempts) = stats.position_attempts.get_mut(position) {
                *attempts += 1;
            }
            if position < accepted as usize {
                if let Some(accepts) = stats.position_accepts.get_mut(position) {
                    *accepts += 1;
                }
            }
        }

        let mut round_stop = false;
        for &token in &candidates[..accepted as usize] {
            if token_ids.len() >= config.max_tokens as usize {
                round_stop = true;
                break;
            }
            if Some(token) == eos {
                round_stop = true;
                hit_eos = true;
                break;
            }
            if stats.first_token_ns.is_none() {
                stats.first_token_ns = Some(generation_started.elapsed().as_nanos() as u64);
            }
            on_token(token)?;
            token_ids.push(token);
            stats.accepted_tokens += 1;
        }
        stats.rejected_tokens += u64::from(candidates.len() as u32 - accepted);

        let keep_end = candidate_start + accepted;
        if keep_end < candidate_start + candidates.len() as u32 {
            target.remove_kv_range(0, keep_end, Some(candidate_start + candidates.len() as u32))?;
            draft.remove_kv_range(0, keep_end, Some(candidate_start + candidates.len() as u32))?;
        }

        let mut fallback_emitted = false;
        if !round_stop && token_ids.len() < config.max_tokens as usize {
            if Some(fallback) == eos {
                round_stop = true;
                hit_eos = true;
            } else {
                if stats.first_token_ns.is_none() {
                    stats.first_token_ns = Some(generation_started.elapsed().as_nanos() as u64);
                }
                on_token(fallback)?;
                token_ids.push(fallback);
                stats.fallback_tokens += 1;
                fallback_emitted = true;
            }
        }

        rounds_detail.push(SpeculativeRound {
            round: stats.rounds as u32,
            candidates: candidates.len() as u32,
            accepted,
            rejected: candidates.len() as u32 - accepted,
            fallback: u32::from(fallback_emitted),
            target_evaluated: batch.len() as u32,
            target_ns,
            draft_ns: drafted_ns,
        });
        stats.rounds += 1;

        if round_stop {
            break;
        }
        if fallback_emitted {
            pending = Some(fallback);
            target_position = keep_end;
            draft_position = keep_end;
        } else {
            break;
        }
        prompt_target_logits = None;
    }

    stats.stop_reason = if hit_eos {
        "eos"
    } else if token_ids.len() >= config.max_tokens as usize {
        "max_tokens"
    } else {
        "context_length"
    }
    .to_string();
    stats.emitted_tokens = token_ids.len() as u64;
    stats.total_ns = generation_started.elapsed().as_nanos() as u64;
    stats.acceptance_rate_bps = if stats.draft_tokens_generated == 0 {
        0
    } else {
        u32::try_from(
            stats
                .accepted_tokens
                .saturating_mul(10_000)
                .checked_div(stats.draft_tokens_generated)
                .unwrap_or(0),
        )
        .unwrap_or(u32::MAX)
    };
    stats.avg_accepted_per_round_x100 = if stats.rounds == 0 {
        0
    } else {
        u32::try_from(
            stats
                .accepted_tokens
                .saturating_mul(100)
                .checked_div(stats.rounds)
                .unwrap_or(0),
        )
        .unwrap_or(u32::MAX)
    };
    stats.rounds_detail = rounds_detail;
    Ok(SpeculativeOutput { token_ids, stats })
}

/// Validate that the draft tokenizer is a safe prefix of the target tokenizer.
///
/// Speculative verification compares raw token ids, so a draft upcast or a
/// different vocabulary would silently produce wrong candidates. The draft token
/// list and merges must match the target exactly over the draft's length; extra
/// target-only tokens are allowed because the target fallback can still emit
/// them.
pub fn draft_vocabulary_compatible(
    draft: &ramforge_core::GgufModel,
    target: &ramforge_core::GgufModel,
) -> Result<(), String> {
    let draft_meta = &draft.metadata;
    let target_meta = &target.metadata;
    for key in ["tokenizer.ggml.model", "tokenizer.ggml.pre"] {
        let a = metadata_string(draft_meta, key);
        let b = metadata_string(target_meta, key);
        if a != b {
            return Err(format!(
                "draft/target tokenizer mismatch for {key}: {:?} vs {:?}",
                a, b
            ));
        }
    }
    for key in ["tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"] {
        let a = metadata_u64(draft_meta, key);
        let b = metadata_u64(target_meta, key);
        if a != b {
            return Err(format!(
                "draft/target special-token mismatch for {key}: {:?} vs {:?}",
                a, b
            ));
        }
    }
    let draft_tokens = metadata_strings(draft_meta, "tokenizer.ggml.tokens")
        .ok_or_else(|| "draft tokenizer token list is unavailable".to_string())?;
    let target_tokens = metadata_strings(target_meta, "tokenizer.ggml.tokens")
        .ok_or_else(|| "target tokenizer token list is unavailable".to_string())?;
    if draft_tokens.len() > target_tokens.len() {
        return Err(format!(
            "draft vocabulary {} is larger than target vocabulary {}",
            draft_tokens.len(),
            target_tokens.len()
        ));
    }
    if !prefix_matches(&draft_tokens, &target_tokens) {
        return Err("draft token list is not an exact prefix of the target token list".to_string());
    }
    if let (Some(draft_merges), Some(target_merges)) = (
        metadata_strings(draft_meta, "tokenizer.ggml.merges"),
        metadata_strings(target_meta, "tokenizer.ggml.merges"),
    ) {
        if draft_merges.len() > target_merges.len()
            || !prefix_matches(&draft_merges, &target_merges)
        {
            return Err(
                "draft merge list is not an exact prefix of the target merge list".to_string(),
            );
        }
    }
    if let (Some(draft_types), Some(target_types)) = (
        metadata_u32s(draft_meta, "tokenizer.ggml.token_type"),
        metadata_u32s(target_meta, "tokenizer.ggml.token_type"),
    ) {
        if draft_types.len() > target_types.len() || !prefix_matches(&draft_types, &target_types) {
            return Err(
                "draft token-type list is not an exact prefix of the target list".to_string(),
            );
        }
    }
    Ok(())
}

fn prefix_matches<T: PartialEq>(draft: &[T], target: &[T]) -> bool {
    target.starts_with(draft)
}

fn metadata_string(
    metadata: &std::collections::BTreeMap<String, ramforge_core::MetadataValue>,
    key: &str,
) -> Option<String> {
    match metadata.get(key)? {
        ramforge_core::MetadataValue::String(value) => Some(value.clone()),
        _ => None,
    }
}

fn metadata_u64(
    metadata: &std::collections::BTreeMap<String, ramforge_core::MetadataValue>,
    key: &str,
) -> Option<u64> {
    match metadata.get(key)? {
        ramforge_core::MetadataValue::UInt64(value) => Some(*value),
        ramforge_core::MetadataValue::UInt32(value) => Some(u64::from(*value)),
        ramforge_core::MetadataValue::Int64(value) => u64::try_from(*value).ok(),
        ramforge_core::MetadataValue::Int32(value) => u64::try_from(*value).ok(),
        _ => None,
    }
}

fn metadata_strings(
    metadata: &std::collections::BTreeMap<String, ramforge_core::MetadataValue>,
    key: &str,
) -> Option<Vec<String>> {
    match metadata.get(key)? {
        ramforge_core::MetadataValue::Array(array) => array
            .values
            .iter()
            .map(|value| match value {
                ramforge_core::MetadataValue::String(value) => Some(value.clone()),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

fn metadata_u32s(
    metadata: &std::collections::BTreeMap<String, ramforge_core::MetadataValue>,
    key: &str,
) -> Option<Vec<u32>> {
    match metadata.get(key)? {
        ramforge_core::MetadataValue::Array(array) => array
            .values
            .iter()
            .map(|value| match value {
                ramforge_core::MetadataValue::UInt32(value) => Some(*value),
                ramforge_core::MetadataValue::Int32(value) => u32::try_from(*value).ok(),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_compatibility_is_exact() {
        assert!(prefix_matches(&["a", "b"], &["a", "b", "c"]));
        assert!(!prefix_matches(&["a", "c"], &["a", "b", "c"]));
        assert!(!prefix_matches(&["a"], &[]));
    }

    #[test]
    fn argmax_uses_first_maximum() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0]), 0);
    }

    #[test]
    fn verifies_all_candidates_and_returns_next_fallback() {
        // prompt predicts 2; after 2 the target predicts 0; after 0 it predicts 1.
        let prompt = [0.0, 0.0, 1.0];
        let l0 = [2.0, 0.0, 0.0];
        let l1 = [0.0, 1.0, 0.0];
        let logits: Vec<&[f32]> = vec![&l0, &l1];
        let (accepted, fallback) = verify_candidates(&[2, 0], false, &prompt, &logits);
        assert_eq!(accepted, 2);
        assert_eq!(fallback, 1); // argmax of l1
    }

    #[test]
    fn verifies_partial_acceptance() {
        // prompt predicts 2; after candidate 2 the target predicts 2 again,
        // so candidate 1 at the next position is rejected.
        let prompt = [0.0, 0.0, 1.0];
        let l0 = [0.0, 0.0, 2.0];
        let logits: Vec<&[f32]> = vec![&l0];
        let (accepted, fallback) = verify_candidates(&[2, 1], false, &prompt, &logits);
        assert_eq!(accepted, 1);
        assert_eq!(fallback, 2); // target argmax after candidate 2
    }

    #[test]
    fn pending_token_shifts_logits_by_one() {
        // l0 is after the pending token and predicts draft 1; l1 is after 1
        // and predicts draft 2. The fallback after 2 is the argmax of l1.
        let l0 = [0.0, 1.0, 0.0];
        let l1 = [0.0, 0.0, 1.0];
        let logits: Vec<&[f32]> = vec![&l0, &l1];
        let (accepted, fallback) = verify_candidates(&[1, 2], true, &[], &logits);
        assert_eq!(accepted, 2);
        assert_eq!(fallback, 2);
    }
}
