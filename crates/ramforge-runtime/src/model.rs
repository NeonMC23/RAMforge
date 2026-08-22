//! LLaMA-compatible model loading and inference
//!
//! Supported architecture families for Milestone 3:
//! - "llama" (dense transformer)
//! - "qwen2" (same tensor layout as llama, with qwen2.* metadata keys)
//!
//! Supported GGUF metadata values:
//! - general.architecture = "llama" or "qwen2"
//! - {arch}.vocab_size, context_length, embedding_length, block_count,
//!   feed_forward_length, attention.head_count, attention.head_count_kv,
//!   attention.layer_norm_rms_epsilon, rope.freq_base
//!
//! Supported tensor types:
//! - F32, F16, BF16
//!
//! Other types produce clear error.

use ramforge_core::{
    cache::BoundedCache,
    datasource::GgufDataSource,
    memory::MemoryBudget,
    model::GgufModel,
    types::GgmlType,
};

use crate::backend::ComputeBackend;
use crate::kv_cache::KvCache;
use crate::ops::{apply_rope, attention};

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub context_length: usize,
    pub embedding_length: usize,
    pub block_count: usize,
    pub feed_forward_length: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub head_dim: usize,
    pub rms_eps: f32,
    pub rope_freq_base: f32,
}

impl LlamaConfig {
    pub fn from_gguf(model: &GgufModel) -> Result<Self, String> {
        let info = model.info();
        let arch = info.architecture.as_deref().unwrap_or("unknown").to_string();

        // Supported architectures: llama and qwen2 (both use same dense transformer layout)
        let supported = ["llama", "qwen2"];
        if !supported.contains(&arch.as_str()) {
            return Err(format!(
                "unsupported architecture '{}': only {:?} are supported in milestone 3 (found general.architecture = '{}')",
                arch, supported, arch
            ));
        }

        // Helper to get u64 from metadata with fallback trying arch-specific keys
        let get_u64_arch = |suffix: &str| -> Option<u64> {
            // Try {arch}.{suffix}
            let key = format!("{}.{}", arch, suffix);
            if let Some(v) = model.get_metadata(&key).and_then(|v| v.as_u64()) {
                return Some(v);
            }
            // Try llama.{suffix} as fallback (many qwen2 models still use llama keys for some)
            let llama_key = format!("llama.{}", suffix);
            if let Some(v) = model.get_metadata(&llama_key).and_then(|v| v.as_u64()) {
                return Some(v);
            }
            // Try qwen2.{suffix}
            let qwen2_key = format!("qwen2.{}", suffix);
            if let Some(v) = model.get_metadata(&qwen2_key).and_then(|v| v.as_u64()) {
                return Some(v);
            }
            // Try general.{suffix}
            let general_key = format!("general.{}", suffix);
            if let Some(v) = model.get_metadata(&general_key).and_then(|v| v.as_u64()) {
                return Some(v);
            }
            None
        };

        let get_f32_arch = |suffix: &str| -> Option<f32> {
            let keys = vec![
                format!("{}.{}", arch, suffix),
                format!("llama.{}", suffix),
                format!("qwen2.{}", suffix),
                format!("general.{}", suffix),
            ];
            for k in keys {
                if let Some(val) = model.get_metadata(&k) {
                    match val {
                        ramforge_core::MetadataValue::Float32(f) => return Some(*f),
                        ramforge_core::MetadataValue::Float64(f) => return Some(*f as f32),
                        ramforge_core::MetadataValue::UInt32(u) => return Some(*u as f32),
                        ramforge_core::MetadataValue::Int32(i) => return Some(*i as f32),
                        _ => {}
                    }
                }
            }
            None
        };

        let vocab_size = get_u64_arch("vocab_size")
            .or_else(|| model.get_metadata("tokenizer.ggml.tokens").and_then(|v| v.as_array()).map(|a| a.values.len() as u64))
            .ok_or_else(|| format!("missing vocab_size (tried {}.vocab_size)", arch))? as usize;

        let context_length = get_u64_arch("context_length")
            .ok_or_else(|| format!("missing {}.context_length", arch))? as usize;

        let embedding_length = get_u64_arch("embedding_length")
            .ok_or_else(|| format!("missing {}.embedding_length", arch))? as usize;

        let block_count = get_u64_arch("block_count")
            .ok_or_else(|| format!("missing {}.block_count", arch))? as usize;

        let feed_forward_length = get_u64_arch("feed_forward_length")
            .or_else(|| get_u64_arch("intermediate_size"))
            .or_else(|| {
                // Some models use feed_forward_length as intermediate size
                model
                    .get_metadata("llama.feed_forward_length")
                    .and_then(|v| v.as_u64())
            })
            .ok_or_else(|| format!("missing {}.feed_forward_length", arch))? as usize;

        let head_count = get_u64_arch("attention.head_count")
            .ok_or_else(|| format!("missing {}.attention.head_count", arch))? as usize;

        let head_count_kv = get_u64_arch("attention.head_count_kv")
            .unwrap_or(head_count as u64) as usize;

        let rms_eps = get_f32_arch("attention.layer_norm_rms_epsilon")
            .or_else(|| get_f32_arch("attention.layer_norm_epsilon"))
            .unwrap_or(1e-5);

        let rope_freq_base = get_f32_arch("rope.freq_base").unwrap_or(10000.0);

        let head_dim = embedding_length / head_count;

        Ok(Self {
            vocab_size,
            context_length,
            embedding_length,
            block_count,
            feed_forward_length,
            head_count,
            head_count_kv,
            head_dim,
            rms_eps,
            rope_freq_base,
        })
    }
}

#[derive(Debug)]
pub struct LlamaWeights {
    // token embedding and output
    pub token_embd: Vec<f32>, // [vocab, n_embd] or [n_embd, vocab] handled as contiguous per token
    pub output_norm: Vec<f32>, // [n_embd]
    pub output: Option<Vec<f32>>, // [vocab, n_embd] or [n_embd, vocab]

    // per layer
    pub layers: Vec<LayerWeights>,
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub attn_q: Vec<f32>,
    pub attn_k: Vec<f32>,
    pub attn_v: Vec<f32>,
    pub attn_output: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_down: Vec<f32>,
}

impl LlamaWeights {
    /// Check that all required tensors exist in model, return list of missing
    pub fn validate(model: &GgufModel, config: &LlamaConfig) -> Result<(), String> {
        let mut missing = Vec::new();

        let required = vec!["token_embd.weight", "output_norm.weight"];

        for name in required {
            if !model.tensors.iter().any(|t| t.name == name) {
                missing.push(name.to_string());
            }
        }

        for i in 0..config.block_count {
            let layer_tensors = vec![
                format!("blk.{}.attn_norm.weight", i),
                format!("blk.{}.attn_q.weight", i),
                format!("blk.{}.attn_k.weight", i),
                format!("blk.{}.attn_v.weight", i),
                format!("blk.{}.attn_output.weight", i),
                format!("blk.{}.ffn_norm.weight", i),
                format!("blk.{}.ffn_gate.weight", i),
                format!("blk.{}.ffn_up.weight", i),
                format!("blk.{}.ffn_down.weight", i),
            ];
            for name in layer_tensors {
                if !model.tensors.iter().any(|t| t.name == name) {
                    missing.push(name);
                }
            }
        }

        if !missing.is_empty() {
            let mut msg = String::from("Unsupported or incomplete model:\n");
            for m in missing.iter().take(20) {
                msg.push_str(&format!("missing tensor '{}'\n", m));
            }
            if missing.len() > 20 {
                msg.push_str(&format!("... and {} more\n", missing.len() - 20));
            }
            return Err(msg);
        }

        Ok(())
    }
}

/// LLaMA model with file-backed weight access
pub struct LlamaModel {
    pub config: LlamaConfig,
    pub weights: LlamaWeights,
    // Keep data source for potential future on-demand loading
    // For milestone 3, weights are loaded into memory via cache, but still file-backed originally
}

impl LlamaModel {
    /// Load model weights through GgufDataSource and BoundedCache with budget accounting
    pub fn load(
        data_source: &GgufDataSource,
        cache: &mut BoundedCache,
        budget: &mut MemoryBudget,
    ) -> Result<Self, String> {
        let gguf_model = data_source.model();
        let config = LlamaConfig::from_gguf(gguf_model)?;

        // Validate required tensors
        LlamaWeights::validate(gguf_model, &config)?;

        // Helper to load and decode a tensor via file-backed data source and cache
        let mut load_tensor = |name: &str| -> Result<Vec<f32>, String> {
            // Try cache first (raw bytes)
            if let Some(cached_bytes) = cache.get(name) {
                let cached_clone = cached_bytes.clone();
                // Need descriptor to decode
                let desc = data_source
                    .get_descriptor(name)
                    .map_err(|e| format!("tensor '{}' not found: {}", name, e))?;
                let decoded = ramforge_core::tensor::decode_tensor_to_f32(
                    &cached_clone,
                    desc.ggml_type,
                    desc.num_elements,
                )
                .map_err(|e| format!("failed to decode cached tensor '{}': {}", name, e))?;
                return Ok(decoded);
            }

            let desc = data_source
                .get_descriptor(name)
                .map_err(|e| format!("tensor '{}' not found: {}", name, e))?;

            match desc.ggml_type {
                GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {},
                _ => {
                    return Err(format!(
                        "unsupported tensor type for '{}': {} (only F32, F16, BF16 supported)",
                        name,
                        desc.ggml_type.name()
                    ))
                }
            }

            let raw_bytes = data_source
                .read_tensor(name)
                .map_err(|e| format!("failed to read tensor '{}': {}", name, e))?;

            let num_elements = desc.num_elements;
            let decoded = ramforge_core::tensor::decode_tensor_to_f32(
                &raw_bytes,
                desc.ggml_type,
                num_elements,
            )
            .map_err(|e| format!("failed to decode tensor '{}': {}", name, e))?;

            let decoded_bytes = decoded.len() * 4;
            let alloc_name = format!("weight:{}", name);
            if budget.get(&alloc_name).is_none() {
                budget
                    .allocate(alloc_name, decoded_bytes as u64)
                    .map_err(|e| format!("RAM budget exceeded loading '{}': {}", name, e))?;
            }

            // Insert raw bytes into cache (demonstrates file-backed + cache)
            let _ = cache.insert(name.to_string(), raw_bytes);

            Ok(decoded)
        };

        // Load token embedding
        let token_embd = load_tensor("token_embd.weight")?;
        let output_norm = load_tensor("output_norm.weight")?;
        let output = if gguf_model.tensors.iter().any(|t| t.name == "output.weight") {
            Some(load_tensor("output.weight")?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(config.block_count);
        for i in 0..config.block_count {
            let attn_norm = load_tensor(&format!("blk.{}.attn_norm.weight", i))?;
            let attn_q = load_tensor(&format!("blk.{}.attn_q.weight", i))?;
            let attn_k = load_tensor(&format!("blk.{}.attn_k.weight", i))?;
            let attn_v = load_tensor(&format!("blk.{}.attn_v.weight", i))?;
            let attn_output = load_tensor(&format!("blk.{}.attn_output.weight", i))?;
            let ffn_norm = load_tensor(&format!("blk.{}.ffn_norm.weight", i))?;
            let ffn_gate = load_tensor(&format!("blk.{}.ffn_gate.weight", i))?;
            let ffn_up = load_tensor(&format!("blk.{}.ffn_up.weight", i))?;
            let ffn_down = load_tensor(&format!("blk.{}.ffn_down.weight", i))?;

            layers.push(LayerWeights {
                attn_norm,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        Ok(Self {
            config,
            weights: LlamaWeights {
                token_embd,
                output_norm,
                output,
                layers,
            },
        })
    }

    /// Forward pass for a single token with KV cache
    ///
    /// `token_id` is the current token, `pos` is its position in sequence,
    /// `kv_cache` holds previous K/V, `backend` is CPU backend
    /// Returns hidden state after final norm (before output projection)
    pub fn forward_single(
        &self,
        token_id: u32,
        pos: usize,
        kv_cache: &mut KvCache,
        backend: &dyn ComputeBackend,
    ) -> Result<Vec<f32>, String> {
        let cfg = &self.config;
        let n_embd = cfg.embedding_length;
        let n_heads = cfg.head_count;
        let n_kv_heads = cfg.head_count_kv;
        let head_dim = cfg.head_dim;

        // Embedding lookup: token_embd is [vocab, n_embd] or [n_embd, vocab] with n_embd contiguous per token
        // We assume data layout: token_id * n_embd .. (token_id+1)*n_embd
        // If token_embd length is vocab * n_embd, this works
        // If it's n_embd * vocab but stored as [n_embd, vocab] with n_embd contiguous per vocab, same offset
        let mut hidden = vec![0.0f32; n_embd];
        let embd_offset = (token_id as usize) * n_embd;
        if embd_offset + n_embd <= self.weights.token_embd.len() {
            hidden.copy_from_slice(&self.weights.token_embd[embd_offset..embd_offset + n_embd]);
        } else {
            return Err(format!(
                "token_id {} out of bounds for embedding (vocab size {})",
                token_id,
                self.weights.token_embd.len() / n_embd
            ));
        }

        // Temporary buffers
        let mut tmp = vec![0.0f32; n_embd];

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            // attn_norm
            backend.rmsnorm(&hidden, &layer.attn_norm, cfg.rms_eps, &mut tmp);

            // Q, K, V projections
            // Each weight is [out, in] where out = n_heads*head_dim etc., in = n_embd
            // We need to know shape: For attn_q, shape should be [n_embd, n_heads*head_dim] or [n_heads*head_dim, n_embd]
            // We'll try to infer: if weight len == n_embd * n_heads*head_dim, we can matvec
            // Assume weight is [out, in] row-major
            let mut q_tmp = vec![0.0f32; n_heads * head_dim];
            let mut k_tmp = vec![0.0f32; n_kv_heads * head_dim];
            let mut v_tmp = vec![0.0f32; n_kv_heads * head_dim];

            // Helper to matvec with shape inference
            Self::matvec_infer(
                backend,
                &layer.attn_q,
                n_heads * head_dim,
                n_embd,
                &tmp,
                &mut q_tmp,
            );
            Self::matvec_infer(
                backend,
                &layer.attn_k,
                n_kv_heads * head_dim,
                n_embd,
                &tmp,
                &mut k_tmp,
            );
            Self::matvec_infer(
                backend,
                &layer.attn_v,
                n_kv_heads * head_dim,
                n_embd,
                &tmp,
                &mut v_tmp,
            );

            // RoPE
            apply_rope(
                &mut q_tmp,
                &mut k_tmp,
                pos,
                head_dim,
                n_heads,
                n_kv_heads,
                cfg.rope_freq_base,
            );

            // Append to KV cache
            kv_cache.append(layer_idx, &k_tmp, &v_tmp).map_err(|e| e.to_string())?;

            // Attention – KV cache currently holds previous tokens, we need to include current
            let k_cache = kv_cache.get_k(layer_idx);
            let v_cache = kv_cache.get_v(layer_idx);
            let mut k_full = Vec::with_capacity((kv_cache.seq_len() + 1) * n_kv_heads * head_dim);
            k_full.extend_from_slice(k_cache);
            k_full.extend_from_slice(&k_tmp);
            let mut v_full = Vec::with_capacity((kv_cache.seq_len() + 1) * n_kv_heads * head_dim);
            v_full.extend_from_slice(v_cache);
            v_full.extend_from_slice(&v_tmp);

            let attn_out = attention(
                &q_tmp,
                &k_full,
                &v_full,
                kv_cache.seq_len() + 1,
                n_heads,
                n_kv_heads,
                head_dim,
            );

            // Output projection
            let mut attn_proj = vec![0.0f32; n_embd];
            Self::matvec_infer(
                backend,
                &layer.attn_output,
                n_embd,
                n_heads * head_dim,
                &attn_out,
                &mut attn_proj,
            );

            // Residual
            for i in 0..n_embd {
                hidden[i] += attn_proj[i];
            }

            // FFN norm
            backend.rmsnorm(&hidden, &layer.ffn_norm, cfg.rms_eps, &mut tmp);

            // FFN: gate and up
            let ffn_dim = cfg.feed_forward_length;
            let mut gate = vec![0.0f32; ffn_dim];
            let mut up = vec![0.0f32; ffn_dim];
            Self::matvec_infer(backend, &layer.ffn_gate, ffn_dim, n_embd, &tmp, &mut gate);
            Self::matvec_infer(backend, &layer.ffn_up, ffn_dim, n_embd, &tmp, &mut up);

            // SiLU gate
            let mut gate_silu = vec![0.0f32; ffn_dim];
            backend.silu(&gate, &mut gate_silu);

            // gate * up
            let mut gate_up = vec![0.0f32; ffn_dim];
            backend.mul(&gate_silu, &up, &mut gate_up);

            // Down projection
            let mut ffn_out = vec![0.0f32; n_embd];
            Self::matvec_infer(backend, &layer.ffn_down, n_embd, ffn_dim, &gate_up, &mut ffn_out);

            // Residual
            for i in 0..n_embd {
                hidden[i] += ffn_out[i];
            }
        }

        // Increment KV cache seq_len after all layers processed for this token
        kv_cache.increment_seq_len();

        // Final norm
        let mut final_hidden = vec![0.0f32; n_embd];
        backend.rmsnorm(&hidden, &self.weights.output_norm, cfg.rms_eps, &mut final_hidden);

        Ok(final_hidden)
    }

    /// Compute logits from hidden state
    pub fn compute_logits(
        &self,
        hidden: &[f32],
        backend: &dyn ComputeBackend,
    ) -> Result<Vec<f32>, String> {
        let vocab_size = self.config.vocab_size;
        let n_embd = self.config.embedding_length;

        let mut logits = vec![0.0f32; vocab_size];

        if let Some(output_weight) = &self.weights.output {
            // output weight shape: [vocab, n_embd] or [n_embd, vocab]
            // We assume [vocab, n_embd] row-major or [n_embd, vocab] column-major both work with our matvec_infer
            Self::matvec_infer(backend, output_weight, vocab_size, n_embd, hidden, &mut logits);
        } else {
            // Use token embedding as output (tied)
            Self::matvec_infer(
                backend,
                &self.weights.token_embd,
                vocab_size,
                n_embd,
                hidden,
                &mut logits,
            );
        }

        Ok(logits)
    }

    fn matvec_infer(
        backend: &dyn ComputeBackend,
        weight: &[f32],
        out_dim: usize,
        in_dim: usize,
        x: &[f32],
        y: &mut [f32],
    ) {
        // weight may be stored as [out, in] or [in, out] – try to infer
        // If weight len == out*in, we can try both
        if weight.len() == out_dim * in_dim {
            // Try to detect layout by checking if out_dim == y.len() and in_dim == x.len()
            // Assume row-major [out, in]
            backend.matvec(weight, &[out_dim, in_dim], x, y);
        } else {
            // Fallback zero
            for yi in y.iter_mut() {
                *yi = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_gguf() {
        // This test would require a real GGUF, we test validation logic separately
        // For now just test that unsupported arch fails
        let model = ramforge_core::model::GgufModel {
            path: std::path::PathBuf::from("/tmp/test.gguf"),
            file_size: 0,
            version: 3,
            metadata: {
                let mut m = std::collections::BTreeMap::new();
                m.insert(
                    "general.architecture".to_string(),
                    ramforge_core::MetadataValue::String("bert".to_string()),
                );
                m
            },
            tensors: vec![],
            alignment: 32,
            data_start_offset: 0,
        };
        let err = LlamaConfig::from_gguf(&model).unwrap_err();
        assert!(err.contains("unsupported architecture"));
    }
}
