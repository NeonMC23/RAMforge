use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::types::{GgmlType, MetadataValue};

/// Describes a single tensor in a GGUF file without loading its data.
///
/// This is file-backed: it records where the tensor data lives inside the
/// original file so future out-of-core access can mmap or read only the needed
/// bytes.
#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    /// Tensor name (e.g. "blk.0.attn_q.weight")
    pub name: String,
    /// Dimensions (as stored, typically reversed order vs ggml)
    pub dimensions: Vec<u64>,
    /// GGML element type
    pub ggml_type: GgmlType,
    /// Byte offset relative to the start of the tensor data section
    pub offset: u64,
    /// Absolute file offset where tensor data begins
    pub file_offset: u64,
    /// Byte length of tensor data when determinable (based on type info and shape)
    pub byte_length: Option<u64>,
    /// Number of elements (product of dimensions)
    pub num_elements: u64,
}

impl TensorDescriptor {
    pub fn shape_string(&self) -> String {
        if self.dimensions.is_empty() {
            "scalar".to_string()
        } else {
            self.dimensions
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x")
        }
    }
}

/// File-backed representation of a GGUF model
///
/// The parser avoids copying tensor payloads into memory. It only reads the
/// header, metadata, and tensor descriptors, and records file offsets for
/// later out-of-core access.
#[derive(Debug, Clone)]
pub struct GgufModel {
    /// Path to source file
    pub path: PathBuf,
    /// File size in bytes
    pub file_size: u64,
    /// GGUF version
    pub version: u32,
    /// Metadata key/value pairs
    pub metadata: BTreeMap<String, MetadataValue>,
    /// Tensor descriptors
    pub tensors: Vec<TensorDescriptor>,
    /// Alignment used for tensor data (default 32, overridable via general.alignment)
    pub alignment: u64,
    /// Absolute file offset where tensor data section starts
    pub data_start_offset: u64,
}

impl GgufModel {
    /// Get metadata value by key
    pub fn get_metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    /// Try to extract a u64 from metadata, handling multiple numeric types
    fn get_u64(&self, key: &str) -> Option<u64> {
        self.get_metadata(key)?.as_u64()
    }

    fn get_string(&self, key: &str) -> Option<String> {
        self.get_metadata(key)?.as_string().map(|s| s.to_string())
    }

    /// Normalized model information
    pub fn info(&self) -> ModelInfo {
        // Architecture is stored in general.architecture
        let architecture = self
            .get_string("general.architecture")
            .or_else(|| self.get_metadata("general.architecture").map(|v| format!("{}", v)));

        // Helper to try architecture-specific keys, e.g. "llama.context_length"
        let arch = architecture.clone().unwrap_or_else(|| "unknown".to_string());

        let try_arch_key = |suffix: &str| -> Option<u64> {
            let key = format!("{}.{}", arch, suffix);
            self.get_u64(&key)
        };

        // Context length, embedding, block count etc.
        let context_length = try_arch_key("context_length")
            .or_else(|| self.get_u64("general.context_length"))
            .or_else(|| self.get_u64("llama.context_length"));

        let embedding_length = try_arch_key("embedding_length")
            .or_else(|| self.get_u64("general.embedding_length"))
            .or_else(|| self.get_u64("llama.embedding_length"));

        let block_count = try_arch_key("block_count")
            .or_else(|| self.get_u64("general.block_count"))
            .or_else(|| self.get_u64("llama.block_count"));

        let head_count = try_arch_key("attention.head_count")
            .or_else(|| self.get_u64("llama.attention.head_count"))
            .or_else(|| self.get_u64("general.attention.head_count"));

        let head_count_kv = try_arch_key("attention.head_count_kv")
            .or_else(|| self.get_u64("llama.attention.head_count_kv"));

        let expert_count = try_arch_key("expert_count")
            .or_else(|| self.get_u64("llama.expert_count"))
            .or_else(|| try_arch_key("experts_count"));

        let expert_used_count = try_arch_key("expert_used_count")
            .or_else(|| self.get_u64("llama.expert_used_count"));

        let file_type = self
            .get_u64("general.file_type")
            .map(|v| v as u32);

        let tokenizer_model = self
            .get_string("tokenizer.ggml.model")
            .or_else(|| self.get_string("general.tokenizer_model"));

        let vocab_size = self
            .get_metadata("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|arr| arr.values.len());

        let name = self.get_string("general.name");
        let description = self.get_string("general.description");

        ModelInfo {
            architecture,
            context_length,
            embedding_length,
            block_count,
            head_count,
            head_count_kv,
            expert_count,
            expert_used_count,
            file_type,
            tokenizer_model,
            vocab_size,
            name,
            description,
        }
    }

    /// Summarize tensor types: map from type name to count
    pub fn type_summary(&self) -> BTreeMap<String, usize> {
        let mut map = BTreeMap::new();
        for t in &self.tensors {
            let name = t.ggml_type.name();
            *map.entry(name).or_insert(0) += 1;
        }
        map
    }

    /// Total tensor data size if all determinable, else None
    pub fn total_tensor_bytes(&self) -> Option<u64> {
        let mut total = 0u64;
        for t in &self.tensors {
            let len = t.byte_length?;
            total = total.checked_add(len)?;
        }
        Some(total)
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub architecture: Option<String>,
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    pub block_count: Option<u64>,
    pub head_count: Option<u64>,
    pub head_count_kv: Option<u64>,
    pub expert_count: Option<u64>,
    pub expert_used_count: Option<u64>,
    pub file_type: Option<u32>,
    pub tokenizer_model: Option<String>,
    pub vocab_size: Option<usize>,
    pub name: Option<String>,
    pub description: Option<String>,
}

/// Helper to align offset to alignment boundary
pub fn align_offset(offset: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return offset;
    }
    let remainder = offset % alignment;
    if remainder == 0 {
        offset
    } else {
        offset + (alignment - remainder)
    }
}
