//! LLaMA/Qwen2 architecture configuration and tensor validation.
//!
//! Supported architecture families:
//! - "llama" (dense transformer)
//! - "qwen2" (same tensor layout as llama, with qwen2.* metadata keys)
//!
//! Supported GGUF metadata values:
//! - general.architecture = "llama" or "qwen2"
//! - {arch}.vocab_size, context_length, embedding_length, block_count,
//!   feed_forward_length, attention.head_count, attention.head_count_kv,
//!   attention.layer_norm_rms_epsilon, rope.freq_base
//!
//! The actual weights are handled by `streaming_model.rs` (out-of-core,
//! compact quantized residency). The former fully-resident F32 model
//! loader was removed in Milestone 6: it violated budget integrity,
//! guessed matrix orientation, and duplicated the KV prefix per token.

use ramforge_core::model::GgufModel;

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
        let arch = info
            .architecture
            .as_deref()
            .unwrap_or("unknown")
            .to_string();

        // Supported architectures: llama and qwen2 (both use same dense transformer layout)
        let supported = ["llama", "qwen2"];
        if !supported.contains(&arch.as_str()) {
            return Err(format!(
                "unsupported architecture '{}': only {:?} are supported (found general.architecture = '{}')",
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
            .or_else(|| {
                model
                    .get_metadata("tokenizer.ggml.tokens")
                    .and_then(|v| v.as_array())
                    .map(|a| a.values.len() as u64)
            })
            .ok_or_else(|| format!("missing vocab_size (tried {}.vocab_size)", arch))?
            as usize;

        let context_length = get_u64_arch("context_length")
            .ok_or_else(|| format!("missing {}.context_length", arch))?
            as usize;

        let embedding_length = get_u64_arch("embedding_length")
            .ok_or_else(|| format!("missing {}.embedding_length", arch))?
            as usize;

        let block_count = get_u64_arch("block_count")
            .ok_or_else(|| format!("missing {}.block_count", arch))?
            as usize;

        let feed_forward_length = get_u64_arch("feed_forward_length")
            .or_else(|| get_u64_arch("intermediate_size"))
            .or_else(|| {
                // Some models use feed_forward_length as intermediate size
                model
                    .get_metadata("llama.feed_forward_length")
                    .and_then(|v| v.as_u64())
            })
            .ok_or_else(|| format!("missing {}.feed_forward_length", arch))?
            as usize;

        let head_count = get_u64_arch("attention.head_count")
            .ok_or_else(|| format!("missing {}.attention.head_count", arch))?
            as usize;

        let head_count_kv =
            get_u64_arch("attention.head_count_kv").unwrap_or(head_count as u64) as usize;

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

/// Validate that all tensors required by the llama/qwen2 dense transformer
/// architecture are present in the model (used by the streaming loader).
///
/// Returns a descriptive error listing the first missing tensors.
pub fn validate_required_tensors(model: &GgufModel, config: &LlamaConfig) -> Result<(), String> {
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
