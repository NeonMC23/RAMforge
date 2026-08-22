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

/// Compute attention for single token with KV cache
///
/// q: [n_heads * head_dim]
/// k_cache: [seq_len * n_kv_heads * head_dim] flattened, seq_len is current length
/// v_cache: same
/// Returns output [n_heads * head_dim]
#[allow(clippy::needless_range_loop)]
pub fn attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; n_heads * head_dim];

    // For each head
    for h in 0..n_heads {
        let q_offset = h * head_dim;
        let q_head = &q[q_offset..q_offset + head_dim];

        // Determine which KV head corresponds (for GQA)
        let kv_h = if n_kv_heads == 0 {
            0
        } else {
            h * n_kv_heads / n_heads
        };
        // Compute scores for each position
        let mut scores = vec![0.0f32; seq_len];
        for pos in 0..seq_len {
            let k_offset = (pos * n_kv_heads + kv_h) * head_dim;
            let k_head = &k_cache[k_offset..k_offset + head_dim];
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
        for pos in 0..seq_len {
            let v_offset = (pos * n_kv_heads + kv_h) * head_dim;
            let v_head = &v_cache[v_offset..v_offset + head_dim];
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
        // Single head, head_dim 2, seq_len 1
        let q = vec![1.0, 0.0];
        let k_cache = vec![1.0, 0.0];
        let v_cache = vec![5.0, 6.0];
        let out = attention(&q, &k_cache, &v_cache, 1, 1, 1, 2);
        // Q dot K =1, softmax single =1, output = V
        assert_eq!(out, vec![5.0, 6.0]);
    }
}
