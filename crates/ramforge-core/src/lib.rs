//! RAMforge Core – GGUF parsing and model inspection
//!
//! This crate provides a file-backed GGUF parser that avoids loading tensor
//! payloads into memory. It is designed as the foundational layer for
//! RAMforge's hierarchical memory system (RAM, VRAM, storage).
//!
//! # Design
//!
//! - Only header, metadata KV pairs, and tensor descriptors are read.
//! - Tensor data is NOT loaded; only file offsets and optional byte lengths
//!   are recorded.
//! - The parser validates magic, version, and basic structure, and returns
//!   clear errors for invalid or truncated files.
//! - Metadata is exposed as structured types, not raw maps, via helpers.
//!
//! # Milestone 2 additions
//!
//! - `memory`: `MemoryBudget` and human size parsing (`parse_memory_size`)
//! - `cache`: Strict bounded LRU cache with byte-exact accounting
//! - `datasource`: File-backed tensor access without loading entire model
//!
//! RAMforge-managed memory is defined as memory explicitly tracked via
//! `MemoryBudget`. It does NOT include total process RSS or OS page cache.

pub mod cache;
pub mod datasource;
pub mod error;
pub mod gguf;
pub mod memory;
pub mod model;
pub mod quant;
pub mod tensor;
pub mod tokenizer;
pub mod types;

pub use cache::{BoundedCache, CacheStats};
pub use datasource::GgufDataSource;
pub use error::{CacheError, DataSourceError, GgufError, MemoryError, ParseSizeError, Result};
pub use gguf::parse_gguf_file;
pub use memory::{parse_memory_size, MemoryBudget};
pub use model::{GgufModel, ModelInfo, TensorDescriptor};
pub use quant::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ5K, BlockQ6K, BlockQ8K, BlockQ8_0,
    BLOCK_SIZE_Q2_K, BLOCK_SIZE_Q3_K, BLOCK_SIZE_Q4_0, BLOCK_SIZE_Q4_K, BLOCK_SIZE_Q5_K,
    BLOCK_SIZE_Q6_K, BLOCK_SIZE_Q8_0, BLOCK_SIZE_Q8_K, QK4_0, QK8_0, QK_K,
};
pub use tensor::{decode_tensor_to_f32, QuantizedTensor, TensorData};
pub use tokenizer::Tokenizer;
pub use types::{GgmlType, MetadataValue};

/// Convenience function to inspect a GGUF file from a path
pub fn inspect<P: AsRef<std::path::Path>>(path: P) -> Result<GgufModel> {
    parse_gguf_file(path)
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_file_backed_no_payload_load() {
        // Create a file with 1 tensor, but claim large dimensions; we don't write the full payload.
        // Parser should succeed without needing the payload.
        let mut buf = Vec::new();
        // magic
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        buf.extend_from_slice(&0u64.to_le_bytes()); // kv count
                                                    // tensor
        let name = "large.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&1_000_000u64.to_le_bytes()); // 1M elements
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset
                                                    // align
        let pos = buf.len() as u64;
        let aligned = crate::model::align_offset(pos, 32);
        buf.extend(vec![0u8; (aligned - pos) as usize]);
        // Don't write 4MB of data; write only 0 bytes. Parser should still succeed because it doesn't read payload.
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        let model = parse_gguf_file(tmp.path()).unwrap();
        assert_eq!(model.tensors[0].num_elements, 1_000_000);
        assert_eq!(model.tensors[0].byte_length, Some(4_000_000));
        // file_offset is recorded, but data not loaded
        assert!(model.tensors[0].file_offset >= aligned);
    }
}
