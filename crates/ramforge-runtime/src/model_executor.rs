//! Model execution layer for the dense Llama/Qwen2 transformer path.
//!
//! `ModelExecutor` receives already-materialized layer weights and activation
//! buffers. It does not load GGUF data, own a `MemoryBudget`, decide cache
//! residency, or perform generation. Those concerns remain in
//! `StreamingLlamaModel` and `InferenceEngine`.
//!
//! The execution order is deliberately explicit:
//!
//! ```text
//! input
//! -> attention RMSNorm
//! -> Q/K/V projections and Qwen biases
//! -> RoPE
//! -> cached/current GQA attention
//! -> attention output projection
//! -> attention residual
//! -> FFN RMSNorm
//! -> gate/up projections
//! -> SiLU(gate) * up
//! -> down projection
//! -> FFN residual
//! ```

use crate::backend::ComputeBackend;
use crate::compute_dispatch::{matvec_backend, tensor_f32_view};
use crate::kv_cache::KvCache;
use crate::model::LlamaConfig;
use crate::profile::Profiler;
use ramforge_core::tensor::TensorData;

#[derive(Debug, Clone)]
pub struct StreamingLayerWeights {
    pub attn_norm: TensorData,
    pub attn_q: TensorData,
    pub attn_k: TensorData,
    pub attn_v: TensorData,
    pub attn_output: TensorData,
    pub ffn_norm: TensorData,
    pub ffn_gate: TensorData,
    pub ffn_up: TensorData,
    pub ffn_down: TensorData,
    /// Optional qwen2-style Q/K/V biases. Biases are applied after projection
    /// and before RoPE/KV-cache insertion.
    pub attn_q_bias: Option<TensorData>,
    pub attn_k_bias: Option<TensorData>,
    pub attn_v_bias: Option<TensorData>,
}

impl StreamingLayerWeights {
    /// Checked resident byte total for all owned layer tensor payloads.
    pub fn try_total_resident_bytes(&self) -> Result<u64, String> {
        let tensors = [
            &self.attn_norm,
            &self.attn_q,
            &self.attn_k,
            &self.attn_v,
            &self.attn_output,
            &self.ffn_norm,
            &self.ffn_gate,
            &self.ffn_up,
            &self.ffn_down,
        ];
        let mut total = 0u64;
        for tensor in tensors {
            total = total
                .checked_add(u64::try_from(tensor.resident_bytes()).map_err(|_| {
                    "layer tensor resident size exceeds u64".to_string()
                })?)
                .ok_or_else(|| "layer resident byte total overflow".to_string())?;
        }
        for bias in [&self.attn_q_bias, &self.attn_k_bias, &self.attn_v_bias]
            .into_iter()
            .flatten()
        {
            total = total
                .checked_add(u64::try_from(bias.resident_bytes()).map_err(|_| {
                    "layer bias resident size exceeds u64".to_string()
                })?)
                .ok_or_else(|| "layer resident byte total overflow".to_string())?;
        }
        Ok(total)
    }

    /// Compatibility accessor for callers that cannot propagate a sizing
    /// error. New accounting paths should use `try_total_resident_bytes`.
    pub fn total_resident_bytes(&self) -> u64 {
        self.try_total_resident_bytes()
            .expect("layer resident byte total must be representable")
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ModelExecutor;

impl ModelExecutor {
    /// Execute one complete transformer block over caller-owned activation
    /// scratch. K/V for this token is returned through `k_tmp`/`v_tmp`; the
    /// caller commits it to every layer's cache only after the complete layer
    /// sequence has consumed the previous history.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_layer<B: ComputeBackend>(
        layer: &StreamingLayerWeights,
        layer_idx: usize,
        pos: usize,
        kv_cache: &KvCache,
        backend: &B,
        cfg: &LlamaConfig,
        profiler: &Profiler,
        hidden: &mut [f32],
        tmp: &mut [f32],
        q_tmp: &mut [f32],
        k_tmp: &mut [f32],
        v_tmp: &mut [f32],
        attn_proj: &mut [f32],
        gate: &mut [f32],
        up: &mut [f32],
        gate_silu: &mut [f32],
        gate_up: &mut [f32],
        ffn_out: &mut [f32],
    ) -> Result<(), String> {
        let n_embd = cfg.embedding_length;
        let n_heads = cfg.head_count;
        let n_kv_heads = cfg.head_count_kv;
        let head_dim = cfg.head_dim;
        let q_dim = n_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "query activation dimension overflow".to_string())?;
        let kv_dim = n_kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "KV activation dimension overflow".to_string())?;

        if hidden.len() != n_embd
            || tmp.len() != n_embd
            || attn_proj.len() != n_embd
            || ffn_out.len() != n_embd
            || q_tmp.len() != q_dim
            || k_tmp.len() != kv_dim
            || v_tmp.len() != kv_dim
        {
            return Err(format!(
                "layer {} activation shape mismatch for embedding {}, heads {}, kv heads {}, head_dim {}",
                layer_idx, n_embd, n_heads, n_kv_heads, head_dim
            ));
        }

        // Attention normalization and projections.
        let dequant_started = profiler.start();
        let attn_norm_f32 = tensor_f32_view(&layer.attn_norm)
            .map_err(|e| format!("failed to decode attn_norm of layer {}: {}", layer_idx, e))?;
        profiler.record_since(
            crate::profile::ProfileEvent::Dequantization,
            dequant_started,
        );
        backend.rmsnorm(hidden, attn_norm_f32.as_ref(), cfg.rms_eps, tmp);

        matvec_backend(backend, profiler, &layer.attn_q, tmp, q_tmp)?;
        matvec_backend(backend, profiler, &layer.attn_k, tmp, k_tmp)?;
        matvec_backend(backend, profiler, &layer.attn_v, tmp, v_tmp)?;

        match (&layer.attn_q_bias, &layer.attn_k_bias, &layer.attn_v_bias) {
            (None, None, None) => {}
            (Some(bq), Some(bk), Some(bv)) => {
                let dequant_started = profiler.start();
                let bq = tensor_f32_view(bq).map_err(|e| {
                    format!("failed to decode attn_q.bias of layer {}: {}", layer_idx, e)
                })?;
                let bk = tensor_f32_view(bk).map_err(|e| {
                    format!("failed to decode attn_k.bias of layer {}: {}", layer_idx, e)
                })?;
                let bv = tensor_f32_view(bv).map_err(|e| {
                    format!("failed to decode attn_v.bias of layer {}: {}", layer_idx, e)
                })?;
                profiler.record_since(
                    crate::profile::ProfileEvent::Dequantization,
                    dequant_started,
                );
                for (value, bias) in q_tmp.iter_mut().zip(bq.iter()) {
                    *value += *bias;
                }
                for (value, bias) in k_tmp.iter_mut().zip(bk.iter()) {
                    *value += *bias;
                }
                for (value, bias) in v_tmp.iter_mut().zip(bv.iter()) {
                    *value += *bias;
                }
            }
            _ => {
                return Err(format!(
                    "internal error: incomplete Q/K/V bias set survived load of layer {}",
                    layer_idx
                ));
            }
        }

        crate::ops::apply_rope(
            q_tmp,
            k_tmp,
            pos,
            head_dim,
            n_heads,
            n_kv_heads,
            cfg.rope_freq_base,
        );

        let hist_len = kv_cache.seq_len();
        let k_history = kv_cache.get_k_range(layer_idx, 0, hist_len)?;
        let v_history = kv_cache.get_v_range(layer_idx, 0, hist_len)?;
        let attn_out = crate::ops::attention_checked(
            q_tmp, k_history, v_history, k_tmp, v_tmp, hist_len, n_heads, n_kv_heads, head_dim,
        )?;

        matvec_backend(backend, profiler, &layer.attn_output, &attn_out, attn_proj)?;
        backend.add(hidden, attn_proj, tmp);
        hidden.copy_from_slice(tmp);

        // FFN normalization, SwiGLU, down projection, and final residual.
        let dequant_started = profiler.start();
        let ffn_norm_f32 = tensor_f32_view(&layer.ffn_norm)
            .map_err(|e| format!("failed to decode ffn_norm of layer {}: {}", layer_idx, e))?;
        profiler.record_since(
            crate::profile::ProfileEvent::Dequantization,
            dequant_started,
        );
        backend.rmsnorm(hidden, ffn_norm_f32.as_ref(), cfg.rms_eps, tmp);

        matvec_backend(backend, profiler, &layer.ffn_gate, tmp, gate)?;
        matvec_backend(backend, profiler, &layer.ffn_up, tmp, up)?;
        backend.silu(gate, gate_silu);
        backend.mul(gate_silu, up, gate_up);
        matvec_backend(backend, profiler, &layer.ffn_down, gate_up, ffn_out)?;
        backend.add(hidden, ffn_out, tmp);
        hidden.copy_from_slice(tmp);

        Ok(())
    }
}
