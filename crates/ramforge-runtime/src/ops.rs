//! Model-independent operation adapters.
//!
//! The numerical definitions live in `ramforge_core::compute`, which has no
//! storage, streaming, budget, or runtime dependencies. This module preserves
//! the runtime's historical infallible operation API and translates the
//! validated reference-core results at the model boundary.

/// Apply half-split llama/Qwen RoPE to Q and K.
pub fn apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    freq_base: f32,
) {
    ramforge_core::compute::rope_reference(
        q,
        k,
        pos,
        head_dim,
        n_heads,
        n_kv_heads,
        freq_base,
    )
    .expect("model RoPE tensors must satisfy their declared dimensions");
}

/// Compute single-token causal attention over cached history plus current K/V.
///
/// This compatibility adapter keeps the runtime's existing explicit operation
/// signature; the storage-independent core receives the same values through
/// `AttentionConfig`.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &[f32],
    k_hist: &[f32],
    v_hist: &[f32],
    k_new: &[f32],
    v_new: &[f32],
    hist_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    ramforge_core::compute::attention_reference(
        q,
        k_hist,
        v_hist,
        k_new,
        v_new,
        ramforge_core::compute::AttentionConfig::new(hist_len, n_heads, n_kv_heads, head_dim),
    )
    .expect("model attention tensors must satisfy their declared dimensions")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_rope_adapter_preserves_reference_identity_at_position_zero() {
        let mut q = vec![1.0, 2.0, 3.0, 4.0];
        let mut k = q.clone();
        apply_rope(&mut q, &mut k, 0, 4, 1, 1, 10_000.0);
        assert_eq!(q, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(k, q);
    }

    #[test]
    fn runtime_attention_adapter_uses_reference_gqa_mapping() {
        let result = attention(
            &[1.0, 0.0, 0.0, 1.0],
            &[],
            &[],
            &[1.0, 0.0],
            &[2.0, 3.0],
            0,
            2,
            1,
            2,
        );
        assert_eq!(result, [2.0, 3.0, 2.0, 3.0]);
    }
}
