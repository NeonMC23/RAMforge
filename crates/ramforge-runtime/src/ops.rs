//! Core ops for transformer: RoPE, attention

/// Apply RoPE to query and key vectors
///
/// q and k are [n_heads * head_dim] or [head_dim] for single head.
/// We apply RoPE per head with position `pos` and base `freq_base`.
///
/// Convention (M6.1 fix): the llama/qwen2 "half-split" rotary scheme used
/// by HF Transformers and llama.cpp (rope NORMAL / GPT-NeoX style):
/// for each head, element `j` rotates with element `j + head_dim/2`,
/// with theta_j = pos * freq_base^(-2j/head_dim). This replaces the
/// earlier GPT-J/interleaved adjacent-pair convention, which is *not*
/// what llama/qwen2 GGUF weights are trained/converted for.
pub fn apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    freq_base: f32,
) {
    // For each head in q
    for head in 0..n_heads {
        let offset = head * head_dim;
        rope_single(&mut q[offset..offset + head_dim], pos, freq_base);
    }
    for head in 0..n_kv_heads {
        let offset = head * head_dim;
        rope_single(&mut k[offset..offset + head_dim], pos, freq_base);
    }
}

/// Half-split (llama/qwen2) RoPE for one head of `dim` elements:
/// pairs (x[j], x[j + dim/2]) are rotated by theta_j. Position 0 is the
/// identity. Even/odd `dim` (dim must be even) leaves no unpaired slots.
fn rope_single(x: &mut [f32], pos: usize, freq_base: f32) {
    let dim = x.len();
    let half = dim / 2;
    for j in 0..half {
        let theta = freq_base.powf(-2.0f32 * (j as f32) / (dim as f32)) * (pos as f32);
        let cos = theta.cos();
        let sin = theta.sin();
        let x0 = x[j];
        let x1 = x[j + half];
        x[j] = x0 * cos - x1 * sin;
        x[j + half] = x0 * sin + x1 * cos;
    }
}

/// Compute attention for a single token over `hist` cached K/V plus the
/// current token's K/V — without materializing a concatenated copy.
///
/// - `q`: `[n_heads * head_dim]`
/// - `k_hist`/`v_hist`: cached prefix, `[hist_len * n_kv_heads * head_dim]` flattened
/// - `k_new`/`v_new`: current token, `[n_kv_heads * head_dim]` each
///
/// Position `p < hist_len` reads from the cache slices; position `hist_len`
/// reads the current-token vectors. The only allocation is the `scores`
/// vector (`hist_len + 1` per head) plus the output.
///
/// Returns output `[n_heads * head_dim]`.
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
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
    let kv_dim = n_kv_heads * head_dim;
    let total = hist_len + 1;
    debug_assert_eq!(k_new.len(), kv_dim);
    debug_assert_eq!(v_new.len(), kv_dim);
    debug_assert!(k_hist.len() >= hist_len * kv_dim);
    debug_assert!(v_hist.len() >= hist_len * kv_dim);

    // K/V accessor for position `pos` without copying the cache prefix.
    let k_at = |pos: usize| -> &[f32] {
        if pos < hist_len {
            let off = pos * kv_dim;
            &k_hist[off..off + kv_dim]
        } else {
            k_new
        }
    };
    let v_at = |pos: usize| -> &[f32] {
        if pos < hist_len {
            let off = pos * kv_dim;
            &v_hist[off..off + kv_dim]
        } else {
            v_new
        }
    };

    let mut output = vec![0.0f32; n_heads * head_dim];

    for h in 0..n_heads {
        let q_offset = h * head_dim;
        let q_head = &q[q_offset..q_offset + head_dim];

        let kv_h = if n_kv_heads == 0 {
            0
        } else {
            h * n_kv_heads / n_heads
        };

        let mut scores = vec![0.0f32; total];
        for pos in 0..total {
            let k_pos = k_at(pos);
            let k_head = &k_pos[kv_h * head_dim..kv_h * head_dim + head_dim];
            let mut dot = 0.0f32;
            for i in 0..head_dim {
                dot += q_head[i] * k_head[i];
            }
            dot /= (head_dim as f32).sqrt();
            scores[pos] = dot;
        }

        // Softmax
        let max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        for s in scores.iter_mut() {
            *s /= sum;
        }

        // Weighted sum of V
        let out_offset = h * head_dim;
        for pos in 0..total {
            let v_pos = v_at(pos);
            let v_head = &v_pos[kv_h * head_dim..kv_h * head_dim + head_dim];
            let weight = scores[pos];
            for i in 0..head_dim {
                output[out_offset + i] += weight * v_head[i];
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_position_zero_is_identity() {
        // Position 0 must be the identity under the half-split convention
        // (theta_j = 0 for all j) for multiple head sizes.
        for dim in [4usize, 8, 16] {
            let orig: Vec<f32> = (0..dim).map(|i| 0.25 + 0.5 * i as f32).collect();
            let mut q = orig.clone();
            let mut k = orig.clone();
            apply_rope(&mut q, &mut k, 0, dim, 1, 1, 10000.0);
            assert_eq!(q, orig, "dim {}", dim);
            assert_eq!(k, orig, "dim {}", dim);
        }
    }

    #[test]
    fn test_rope_half_split_convention_nonzero_position() {
        // Half-split (llama/qwen2): element j pairs with element j + dim/2.
        // With position != 0 and a non-symmetric vector, this must equal the
        // manual half-split formula and MUST NOT equal the GPT-J/interleaved
        // adjacent-pair formula.
        let dim = 8usize;
        let half = dim / 2;
        let base = 10000.0f32;
        let pos = 3usize;
        let orig: Vec<f32> = (0..dim).map(|i| 0.1 + 0.3 * i as f32).collect();

        // Manual half-split reference.
        let mut half_split = orig.clone();
        for j in 0..half {
            let theta = base.powf(-2.0f32 * (j as f32) / (dim as f32)) * (pos as f32);
            let (c, s) = (theta.cos(), theta.sin());
            let (a, b) = (orig[j], orig[j + half]);
            half_split[j] = a * c - b * s;
            half_split[j + half] = a * s + b * c;
        }

        // Manual adjacent-pair (GPT-J/interleaved) reference – the OLD,
        // incorrect convention for llama/qwen2.
        let mut adjacent = orig.clone();
        for i in (0..dim).step_by(2) {
            let theta = base.powf(-2.0f32 * (i as f32 / 2.0) / (dim as f32)) * (pos as f32);
            let (c, s) = (theta.cos(), theta.sin());
            let (a, b) = (orig[i], orig[i + 1]);
            adjacent[i] = a * c - b * s;
            adjacent[i + 1] = a * s + b * c;
        }

        let mut q = orig.clone();
        let mut k = orig.clone();
        apply_rope(&mut q, &mut k, pos, dim, 1, 1, base);

        for i in 0..dim {
            assert!(
                (q[i] - half_split[i]).abs() < 1e-5,
                "q[{}] = {}, half-split reference {}",
                i,
                q[i],
                half_split[i]
            );
            assert!((k[i] - half_split[i]).abs() < 1e-5);
        }
        // Guard against regression to the adjacent convention: at least one
        // element must differ meaningfully between the two schemes.
        let max_diff = (0..dim)
            .map(|i| (q[i] - adjacent[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 0.05,
            "implementation must differ from the interleaved convention (max diff {})",
            max_diff
        );
    }

    #[test]
    fn test_rope_pairing_discriminator_sparse_vector() {
        // Sparse activations at two partner positions make the pairing
        // observable directly: only (x[1], x[1 + dim/2]) may interact.
        let dim = 8usize;
        let half = dim / 2;
        let base = 10000.0f32;
        let pos = 1usize;

        let mut x = vec![0.0f32; dim];
        x[1] = 1.0;
        x[1 + half] = 2.0;

        let theta = base.powf(-2.0f32 / (dim as f32)); // j = 1
        let (c, s) = (theta.cos(), theta.sin());

        let mut q = x.clone();
        let mut k = x.clone();
        apply_rope(&mut q, &mut k, pos, dim, 1, 1, base);

        // Half-split expects: q[1] = 1*c - 2*s ; q[1+half] = 1*s + 2*c.
        assert!((q[1] - (1.0 * c - 2.0 * s)).abs() < 1e-6);
        assert!((q[1 + half] - (1.0 * s + 2.0 * c)).abs() < 1e-6);
        // Every other slot stays zero (no adjacent leakage).
        for (i, &v) in q.iter().enumerate() {
            if i != 1 && i != 1 + half {
                assert_eq!(v, 0.0, "leakage at {}", i);
            }
        }
    }

    #[test]
    fn test_rope_multi_head_uses_per_head_block() {
        // Two heads of dim 4: each head's rotation must be confined to its
        // own head_dim block with the same theta schedule.
        let head_dim = 4usize;
        let base = 10000.0f32;
        let pos = 5usize;
        let mut q = vec![0.0f32; 8];
        q[0] = 1.0; // head 0, j=0
        q[head_dim] = 1.0; // head 1, j=0
        let mut k = q.clone();
        apply_rope(&mut q, &mut k, pos, head_dim, 2, 2, base);

        let theta = pos as f32; // base^0 * pos
        let (c, s) = (theta.cos(), theta.sin());
        // head0: (0, 0+2)
        assert!((q[0] - c).abs() < 1e-5);
        assert!((q[2] - s).abs() < 1e-5);
        // head1: (4, 4+2)
        assert!((q[4] - c).abs() < 1e-5);
        assert!((q[6] - s).abs() < 1e-5);
        assert!(q[1].abs() < 1e-6 && q[3].abs() < 1e-6);
    }

    #[test]
    fn test_attention_simple() {
        // Single head, head_dim 2, single (new) position, no history
        let q = vec![1.0, 0.0];
        let k_hist: Vec<f32> = Vec::new();
        let v_hist: Vec<f32> = Vec::new();
        let k_new = vec![1.0, 0.0];
        let v_new = vec![5.0, 6.0];
        let out = attention(&q, &k_hist, &v_hist, &k_new, &v_new, 0, 1, 1, 2);
        // Q dot K =1, softmax single =1, output = V
        assert_eq!(out, vec![5.0, 6.0]);
    }

    #[test]
    fn test_attention_with_history_no_copy() {
        // 1 head, head_dim 1, hist_len 2 + current => compare against a naive
        // concatenated reference implementation.
        let q = vec![1.0f32];
        // history positions: k=2.0 v=10.0 ; k=3.0 v=20.0
        let k_hist = vec![2.0f32, 3.0];
        let v_hist = vec![10.0f32, 20.0];
        let k_new = vec![4.0f32]; // current
        let v_new = vec![30.0f32];

        let out = attention(&q, &k_hist, &v_hist, &k_new, &v_new, 2, 1, 1, 1);

        // Naive reference over the concatenated prefix.
        let ks = [2.0f32, 3.0, 4.0];
        let vs = [10.0f32, 20.0, 30.0];
        let mut scores = [0.0f32; 3];
        for (p, s) in scores.iter_mut().enumerate() {
            *s = (q[0] * ks[p]) / 1.0f32.sqrt();
        }
        let max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        let mut expected = 0.0f32;
        for p in 0..3 {
            expected += (scores[p] / sum) * vs[p];
        }
        assert!(
            (out[0] - expected).abs() < 1e-5,
            "out={} expected={}",
            out[0],
            expected
        );
    }
}
