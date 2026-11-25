//! Context configuration shared by planning, the CLI, and the forgeCORE adapter.
//!
//! This module is deliberately independent of forgeCORE types. It owns the
//! string forms, validation rules, and KV-cache byte accounting that RAMforge
//! reports; [`crate::forgecore`] maps these values onto the native options.
//!
//! A requested value is not an applied value. The adapter reports native
//! readback for batch, micro-batch, sequence count, and threads. KV data types
//! have no native getter at the pinned revision, so their status stays
//! "forwarded, not natively readable" and capacity is estimated from metadata.

use serde::{Deserialize, Serialize};

/// KV cache element type accepted by the native context.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheType {
    F32,
    #[default]
    F16,
    Q8_0,
    Q4_0,
}

impl KvCacheType {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f32" => Ok(Self::F32),
            "f16" | "fp16" | "half" => Ok(Self::F16),
            "q8" | "q8_0" => Ok(Self::Q8_0),
            "q4" | "q4_0" => Ok(Self::Q4_0),
            other => Err(format!(
                "unknown KV cache type '{other}'; expected f32, f16, q8_0, or q4_0"
            )),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }

    pub fn is_quantized(self) -> bool {
        matches!(self, Self::Q8_0 | Self::Q4_0)
    }

    /// Quantized KV blocks are 32 elements wide at the pinned ggml revision.
    fn block_width(self) -> Option<u64> {
        match self {
            Self::F32 | Self::F16 => None,
            Self::Q8_0 | Self::Q4_0 => Some(32),
        }
    }

    fn block_bytes(self) -> Option<u64> {
        match self {
            Self::F32 | Self::F16 => None,
            Self::Q8_0 => Some(34),
            Self::Q4_0 => Some(18),
        }
    }

    /// Bytes for one K or V tensor with `elements` scalar entries.
    fn tensor_bytes(self, elements: u64) -> Option<u64> {
        match (self.block_width(), self.block_bytes()) {
            (Some(width), Some(bytes)) => elements
                .checked_add(width - 1)?
                .checked_div(width)?
                .checked_mul(bytes),
            _ => {
                let width = match self {
                    Self::F32 => 4,
                    Self::F16 => 2,
                    Self::Q8_0 | Self::Q4_0 => return None,
                };
                elements.checked_mul(width)
            }
        }
    }

    /// Estimated bytes for one K or V cache side.
    pub fn cache_side_bytes(
        self,
        layers: u64,
        kv_heads: u64,
        head_dim: u64,
        context_tokens: u64,
    ) -> Option<u64> {
        if layers == 0 || kv_heads == 0 || head_dim == 0 || context_tokens == 0 {
            return None;
        }
        self.tensor_bytes(head_dim)?
            .checked_mul(context_tokens)?
            .checked_mul(kv_heads)?
            .checked_mul(layers)
    }

    /// Estimated KV bytes for one context, counting K and V separately.
    ///
    /// `head_dim` is quantized per head, so block padding is computed per
    /// `(layer, kv_head, position)` rather than over the flattened tensor.
    pub fn cache_bytes(
        self,
        layers: u64,
        kv_heads: u64,
        head_dim: u64,
        context_tokens: u64,
    ) -> Option<u64> {
        self.cache_side_bytes(layers, kv_heads, head_dim, context_tokens)?
            .checked_mul(2)
    }

    /// Bytes per generated token for one context, counting K and V.
    pub fn bytes_per_token(self, layers: u64, kv_heads: u64, head_dim: u64) -> Option<u64> {
        self.cache_bytes(layers, kv_heads, head_dim, 1)
    }
}

/// Flash-attention selection forwarded to the native context.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlashAttentionMode {
    #[default]
    Auto,
    Enabled,
    Disabled,
}

impl FlashAttentionMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" | "default" => Ok(Self::Auto),
            "on" | "enabled" | "enable" | "true" => Ok(Self::Enabled),
            "off" | "disabled" | "disable" | "false" => Ok(Self::Disabled),
            other => Err(format!(
                "unknown flash-attention mode '{other}'; expected auto, on, or off"
            )),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Enabled => "enabled",
            Self::Disabled => "disabled",
        }
    }
}

/// Requested context settings. `None` preserves the native default for that
/// field; concrete values are resolved into the runtime plan before execution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContextConfig {
    pub kv_type_k: Option<KvCacheType>,
    pub kv_type_v: Option<KvCacheType>,
    pub flash_attn: Option<FlashAttentionMode>,
    pub batch_tokens: Option<u32>,
    pub ubatch_tokens: Option<u32>,
    pub sequence_max: Option<u32>,
    pub threads_batch: Option<u32>,
    pub offload_kqv: Option<bool>,
    pub op_offload: Option<bool>,
}

impl ContextConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.batch_tokens == Some(0) {
            return Err("batch tokens must be greater than zero".to_string());
        }
        if self.ubatch_tokens == Some(0) {
            return Err("micro-batch tokens must be greater than zero".to_string());
        }
        if let (Some(batch), Some(ubatch)) = (self.batch_tokens, self.ubatch_tokens) {
            if ubatch > batch {
                return Err(format!(
                    "micro-batch {ubatch} exceeds batch {batch}; the native runtime would clamp or fail"
                ));
            }
        }
        if let Some(sequence_max) = self.sequence_max {
            if sequence_max == 0 || sequence_max > 256 {
                return Err("sequence count must be between 1 and 256".to_string());
            }
        }
        if self.threads_batch == Some(0) {
            return Err("batch thread count must be greater than zero".to_string());
        }
        if self.kv_type_v.is_some_and(KvCacheType::is_quantized)
            && self.flash_attn == Some(FlashAttentionMode::Disabled)
        {
            return Err(
                "a quantized V cache requires flash attention; remove --flash-attn off".to_string(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_kv_types_and_reject_unknown() {
        assert_eq!(KvCacheType::parse("Q8_0").unwrap(), KvCacheType::Q8_0);
        assert_eq!(KvCacheType::parse("f16").unwrap(), KvCacheType::F16);
        assert_eq!(KvCacheType::parse("q4").unwrap(), KvCacheType::Q4_0);
        assert!(KvCacheType::parse("q5_1").is_err());
    }

    #[test]
    fn qwen_7b_kv_slope_matches_measured_values() {
        // 28 layers, 4 KV heads, 128 head_dim: the measured Qwen2.5-7B shape.
        let (layers, heads, dim) = (28, 4, 128);
        assert_eq!(
            KvCacheType::F16.bytes_per_token(layers, heads, dim),
            Some(57_344)
        );
        assert_eq!(
            KvCacheType::Q8_0.bytes_per_token(layers, heads, dim),
            Some(30_464)
        );
        assert_eq!(
            KvCacheType::Q4_0.bytes_per_token(layers, heads, dim),
            Some(16_128)
        );
    }

    #[test]
    fn cache_bytes_scale_linearly_and_count_kv() {
        let one = KvCacheType::F16.cache_bytes(2, 2, 32, 1).expect("bytes");
        assert_eq!(one, 2 * 2 * 32 * 2 * 2);
        let many = KvCacheType::F16.cache_bytes(2, 2, 32, 512).expect("bytes");
        assert_eq!(many, one * 512);
    }

    #[test]
    fn quantized_block_padding_is_per_head() {
        // 34-byte Q8_0 blocks over 33 elements need two blocks per K and V.
        assert_eq!(KvCacheType::Q8_0.cache_bytes(1, 1, 33, 1), Some(136));
    }

    #[test]
    fn invalid_context_combinations_are_rejected() {
        let config = ContextConfig {
            batch_tokens: Some(4),
            ubatch_tokens: Some(8),
            ..ContextConfig::default()
        };
        assert!(config.validate().is_err());

        let config = ContextConfig {
            kv_type_v: Some(KvCacheType::Q8_0),
            flash_attn: Some(FlashAttentionMode::Disabled),
            ..ContextConfig::default()
        };
        assert!(config.validate().is_err());

        let config = ContextConfig {
            sequence_max: Some(257),
            ..ContextConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn defaults_preserve_native_behavior() {
        let config = ContextConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(KvCacheType::default(), KvCacheType::F16);
        assert_eq!(FlashAttentionMode::default(), FlashAttentionMode::Auto);
    }
}
