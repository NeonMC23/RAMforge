//! Core ops for transformer: RoPE, attention

/// Apply RoPE to query and key vectors
///
/// q and k are [n_heads * head_dim] or [head_dim] for single head
/// We apply RoPE per head with position `pos` and base `freq_base`
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

fn rope_single(x: &mut [f32], pos: usize, freq_base: f32) {
    let dim = x.len();
    // RoPE rotates pairs
    for i in (0..dim).step_by(2) {
        let theta = freq_base.powf(-2.0 * (i as f32 / 2.0) / (dim as f32)) * (pos as f32);
        let cos = theta.cos();
        let sin = theta.sin();
        let x0 = x[i];
        let x1 = x[i + 1];
        x[i] = x0 * cos - x1 * sin;
        x[i + 1] = x0 * sin + x1 * cos;
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
    fn test_rope() {
        let mut q = vec![1.0, 0.0, 1.0, 0.0];
        let mut k = vec![1.0, 0.0, 1.0, 0.0];
        apply_rope(&mut q, &mut k, 0, 4, 1, 1, 10000.0);
        // At pos 0, theta=0, cos=1, sin=0, so unchanged
        assert!((q[0] - 1.0).abs() < 1e-5);
        assert!((q[1] - 0.0).abs() < 1e-5);
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
        assert!((out[0] - expected).abs() < 1e-5, "out={} expected={}", out[0], expected);
    }
}
