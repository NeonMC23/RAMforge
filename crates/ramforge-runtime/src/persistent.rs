//! Persistent weight streaming – token_embd, output_norm, output
//!
//! Policy: small persistent tensor (<25% of the total budget) is kept
//! resident, a large one is streamed on demand.
//!
//! M6 memory contract:
//! - streamed embedding lookup reads exactly one row and is budget-charged;
//! - streamed matvec/logits run in row chunks sized by the *available*
//!   budget (bounded temp, no full-tensor reads, no full F32 expansions);
//! - logits are written into a caller-provided buffer (single allocation,
//!   owned by the engine).

#![allow(clippy::needless_range_loop)]

use ramforge_core::{
    datasource::GgufDataSource,
    memory::MemoryBudget,
    model::TensorDescriptor,
    tensor::TensorData,
    types::GgmlType,
};

use crate::residency::ResidencyStats;

/// Upper bound for a single streamed chunk read (also requires budget room).
const MAX_STREAM_CHUNK_BYTES: u64 = 16 * 1024 * 1024;
/// Never leave less than this fraction of the budget outside chunk temps.
const STREAM_CHUNK_BUDGET_SHARE: u64 = 4;

#[derive(Debug, Clone)]
pub enum PersistentWeight {
    Resident(TensorData),
    Streamed(TensorDescriptor),
}

impl PersistentWeight {
    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::Resident(td) => td.resident_bytes(),
            Self::Streamed(_) => 0,
        }
    }

    pub fn is_resident(&self) -> bool {
        matches!(self, Self::Resident(_))
    }

    pub fn is_streamed(&self) -> bool {
        matches!(self, Self::Streamed(_))
    }

    /// Embedding lookup for `token_id`; returns `n_embd` f32 values.
    ///
    /// Resident tensors answer from memory. Streamed tensors read exactly one
    /// quantized row from disk and dequantize only it; the row's raw bytes and
    /// its F32 form are charged to the budget for the duration of the call.
    pub fn get_embedding(
        &self,
        token_id: usize,
        n_embd: usize,
        data_source: &GgufDataSource,
        budget: &mut MemoryBudget,
        _stats: &mut ResidencyStats,
    ) -> Result<Vec<f32>, String> {
        match self {
            Self::Resident(td) => budget.with_temp("tmp:embd_row", (n_embd * 4) as u64, |_b| {
                td.get_embedding(token_id, n_embd)
                    .map_err(|e| format!("embedding lookup failed: {}", e))
            }),
            Self::Streamed(desc) => {
                let row_bytes = row_bytes_for(desc, n_embd)? as u64;
                let temp_bytes = row_bytes + (n_embd * 4) as u64;
                budget.with_temp("tmp:embd_row", temp_bytes, |_b| {
                    let raw = data_source
                        .read_tensor_range(&desc.name, (token_id as u64) * row_bytes, row_bytes)
                        .map_err(|e| format!("failed to read embedding row: {}", e))?;
                    let td = TensorData::from_bytes(
                        desc.ggml_type,
                        vec![n_embd as u64],
                        n_embd as u64,
                        raw,
                    )
                    .map_err(|e| format!("failed to create TensorData for embedding row: {}", e))?;
                    td.to_f32_vec()
                        .map_err(|e| format!("failed to dequantize embedding row: {}", e))
                })
            }
        }
    }

    /// `y = W * x` where W is this persistent weight (ggml layout
    /// `[in, out]`; x.len() == in, y.len() == out).
    ///
    /// - Resident: compact matvec in memory (block-wise for quantized).
    /// - Streamed: row-chunked pass over the file; the chunk buffer is
    ///   budget-charged and bounded by `min(16 MiB, available/share)`.
    pub fn matvec_into(
        &self,
        x: &[f32],
        y: &mut [f32],
        data_source: &GgufDataSource,
        budget: &mut MemoryBudget,
    ) -> Result<(), String> {
        match self {
            Self::Resident(td) => td.matvec(x, y).map_err(|e| e.to_string()),
            Self::Streamed(desc) => streamed_matvec_into(desc, x, y, data_source, budget),
        }
    }

    /// Output projection: `logits_out = W * hidden` (ggml layout
    /// `[n_embd, vocab]`; also used with tied `token_embd` when the model has
    /// no `output.weight`). Writes into the caller-provided buffer — no full
    /// vocabulary duplication.
    pub fn compute_logits_into(
        &self,
        hidden: &[f32],
        data_source: &GgufDataSource,
        budget: &mut MemoryBudget,
        logits_out: &mut [f32],
    ) -> Result<(), String> {
        self.matvec_into(hidden, logits_out, data_source, budget)
    }

    /// Whole-tensor F32 materialization. Intended for small 1D tensors such
    /// as `output_norm.weight` (n_embd elements); charged by the caller.
    pub fn to_f32_vec(&self, data_source: &GgufDataSource) -> Result<Vec<f32>, String> {
        match self {
            Self::Resident(td) => td.to_f32_vec().map_err(|e| e.to_string()),
            Self::Streamed(desc) => {
                let raw = data_source
                    .read_tensor(&desc.name)
                    .map_err(|e| format!("failed to read persistent tensor '{}': {}", desc.name, e))?;
                let td = TensorData::from_bytes(
                    desc.ggml_type,
                    desc.dimensions.clone(),
                    desc.num_elements,
                    raw,
                )
                .map_err(|e| format!("failed to create TensorData: {}", e))?;
                td.to_f32_vec().map_err(|e| e.to_string())
            }
        }
    }
}

/// Row bytes for a contiguous ggml row of `in_dim` elements of `desc`'s type.
pub fn row_bytes_for(desc: &TensorDescriptor, in_dim: usize) -> Result<usize, String> {
    let elems = in_dim;
    let bytes = match desc.ggml_type {
        GgmlType::F32 => elems * 4,
        GgmlType::F16 | GgmlType::BF16 => elems * 2,
        GgmlType::Q4_0 => (elems / 32) * 18,
        GgmlType::Q8_0 => (elems / 32) * 34,
        GgmlType::Q4_K => (elems / 256) * 144,
        GgmlType::Q5_K => (elems / 256) * 176,
        GgmlType::Q6_K => (elems / 256) * 210,
        GgmlType::Q2_K => (elems / 256) * 84,
        GgmlType::Q3_K => (elems / 256) * 110,
        GgmlType::Q8_K => (elems / 256) * 292,
        _ => {
            return Err(format!(
                "unsupported persistent weight type for streaming: {}",
                desc.ggml_type.name()
            ))
        }
    };
    Ok(bytes)
}

/// Chunked, budget-aware streamed matvec for a non-resident 2D tensor.
///
/// Layout: ggml `[in, out]`; the file stores `out` contiguous rows of `in`
/// elements. Each iteration reads one chunk of complete rows, dequantizes
/// one row at a time into a small reused buffer, and dots it with `x`.
/// The chunk + row buffer are charged to the budget via `with_temp`.
fn streamed_matvec_into(
    desc: &TensorDescriptor,
    x: &[f32],
    y: &mut [f32],
    data_source: &GgufDataSource,
    budget: &mut MemoryBudget,
) -> Result<(), String> {
    if desc.dimensions.len() != 2 {
        return Err(format!(
            "streamed matvec expects a 2D tensor, '{}' has shape {:?}",
            desc.name, desc.dimensions
        ));
    }
    let in_dim = desc.dimensions[0] as usize;
    let out_dim = desc.dimensions[1] as usize;
    if x.len() != in_dim || y.len() != out_dim {
        return Err(format!(
            "streamed matvec arity mismatch (ggml layout [in, out]): '{}' {:?} implies in={}, out={}, but got x.len()={}, y.len()={}",
            desc.name,
            desc.dimensions,
            in_dim,
            out_dim,
            x.len(),
            y.len()
        ));
    }

    let row_bytes = row_bytes_for(desc, in_dim)? as u64;
    let row_f32_bytes = (in_dim * 4) as u64;

    // Chunk size: bounded by budget share and an absolute cap, at least one
    // row, rounded down to whole rows. If one row does not fit, fail clearly.
    let avail_share = budget.available_bytes() / STREAM_CHUNK_BUDGET_SHARE;
    let chunk_target = avail_share.min(MAX_STREAM_CHUNK_BYTES).max(row_bytes);
    let rows_per_chunk = (chunk_target / row_bytes).max(1);
    let chunk_bytes = rows_per_chunk * row_bytes;
    let temp_bytes = chunk_bytes + row_f32_bytes;
    if !budget.can_allocate(temp_bytes) {
        return Err(format!(
            "RAM budget too small to stream tensor '{}': need {} bytes of temporary working set ({} rows x {} row bytes + row buffer), available {}",
            desc.name,
            temp_bytes,
            rows_per_chunk,
            row_bytes,
            budget.available_bytes()
        ));
    }

    budget.with_temp("tmp:streamed_matvec", temp_bytes, |_b| {
        let mut row_buf = vec![0.0f32; in_dim];
        let mut row_start = 0usize;
        while row_start < out_dim {
            let rows_this = rows_per_chunk.min((out_dim - row_start) as u64) as usize;
            let offset = (row_start as u64) * row_bytes;
            let len = (rows_this as u64) * row_bytes;
            let chunk = data_source
                .read_tensor_range(&desc.name, offset, len)
                .map_err(|e| format!("failed to read chunk of '{}': {}", desc.name, e))?;
            for r in 0..rows_this {
                let row_slice =
                    &chunk[(r as u64 * row_bytes) as usize..((r as u64 + 1) * row_bytes) as usize];
                ramforge_core::tensor::decode_row_to_f32(desc.ggml_type, row_slice, in_dim, &mut row_buf)
                    .map_err(|e| format!("failed to decode row of '{}': {}", desc.name, e))?;
                let mut sum = 0.0f32;
                for i in 0..in_dim {
                    sum += row_buf[i] * x[i];
                }
                y[row_start + r] = sum;
            }
            row_start += rows_this;
        }
        Ok(())
    })
}

pub fn should_keep_resident(tensor_bytes: u64, budget_total: u64) -> bool {
    tensor_bytes * 4 <= budget_total
}

#[cfg(test)]
mod tests {
    use super::*;
    use ramforge_core::model::align_offset;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_string<W: Write>(w: &mut W, s: &str) {
        w.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
        w.write_all(s.as_bytes()).unwrap();
    }
    fn write_u32<W: Write>(w: &mut W, v: u32) { w.write_all(&v.to_le_bytes()).unwrap(); }
    fn write_u64<W: Write>(w: &mut W, v: u64) { w.write_all(&v.to_le_bytes()).unwrap(); }

    #[test]
    fn test_persistent_policy() {
        assert!(should_keep_resident(100, 1000));
        assert!(!should_keep_resident(300, 1000));
    }

    #[test]
    fn test_persistent_streaming_embedding() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        write_string(&mut buf, "token_embd.weight");
        write_u32(&mut buf, 2);
        write_u64(&mut buf, 8);
        write_u64(&mut buf, 4);
        write_u32(&mut buf, 0);
        write_u64(&mut buf, 0);
        let pos = buf.len() as u64;
        let aligned = align_offset(pos, 32);
        buf.extend(vec![0u8; (aligned - pos) as usize]);
        for token_id in 0..4 {
            for _ in 0..8 {
                buf.extend_from_slice(&(token_id as f32).to_le_bytes());
            }
        }
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        tmp.flush().unwrap();

        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let desc = ds.get_descriptor("token_embd.weight").unwrap().clone();
        let streamed = PersistentWeight::Streamed(desc);

        let mut budget = ramforge_core::memory::MemoryBudget::new(1024).unwrap();
        let mut stats = crate::residency::ResidencyStats::new(0);

        let embd = streamed
            .get_embedding(2, 8, &ds, &mut budget, &mut stats)
            .unwrap();
        assert_eq!(embd.len(), 8);
        assert!((embd[0] - 2.0).abs() < 1e-5);
        assert_eq!(budget.used_bytes(), 0, "temp must be released after lookup");
    }

    /// Streamed output projection over a non-square [in, out] F32 tensor,
    /// produced in several small chunks. Verifies correctness against a
    /// resident reference and that no budget charge leaks.
    #[test]
    fn test_persistent_streamed_output_chunked() {
        // ggml layout [in=4, out=6], rows: r0=[1,0,0,0], r1=[0,1,0,0], r2=[0,0,1,0],
        // r3=[0,0,0,1], r4=[1,1,0,0], r5=[0,0,1,1]
        let rows: [[f32; 4]; 6] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ];
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        write_string(&mut buf, "output.weight");
        write_u32(&mut buf, 2);
        write_u64(&mut buf, 4);
        write_u64(&mut buf, 6);
        write_u32(&mut buf, 0);
        write_u64(&mut buf, 0);
        let pos = buf.len() as u64;
        let aligned = align_offset(pos, 32);
        buf.extend(vec![0u8; (aligned - pos) as usize]);
        for r in rows.iter() {
            for v in r.iter() {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        tmp.flush().unwrap();

        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let desc = ds.get_descriptor("output.weight").unwrap().clone();

        // Rows are 16 bytes; budget forces chunks of ~2-3 rows and must hold
        // chunk+row temp only (tiny). Reserve most of a small budget so only
        // a thin share is available for chunks.
        let mut budget = ramforge_core::memory::MemoryBudget::new(1024).unwrap();
        budget.allocate("pinned", 800).unwrap(); // available = 224 -> share 56B -> 3 rows/chunk
        let streamed = PersistentWeight::Streamed(desc.clone());
        let hidden = [1.0f32, 2.0, 3.0, 4.0];
        let mut logits = [0.0f32; 6];
        streamed
            .compute_logits_into(&hidden, &ds, &mut budget, &mut logits)
            .unwrap();
        let expected = [1.0, 2.0, 3.0, 4.0, 3.0, 7.0];
        for i in 0..6 {
            assert!((logits[i] - expected[i]).abs() < 1e-5, "logit {} = {}", i, logits[i]);
        }
        assert_eq!(budget.used_bytes(), 800, "no temp charge may leak");
        let _ = budget.release("pinned");

        // Resident path must agree with streamed path.
        let raw = ds.read_tensor("output.weight").unwrap();
        let td = TensorData::from_bytes(GgmlType::F32, desc.dimensions.clone(), desc.num_elements, raw).unwrap();
        let resident = PersistentWeight::Resident(td);
        let mut logits_res = [0.0f32; 6];
        resident
            .compute_logits_into(&hidden, &ds, &mut budget, &mut logits_res)
            .unwrap();
        for i in 0..6 {
            assert!((logits_res[i] - expected[i]).abs() < 1e-5);
        }
    }

    /// If even one row plus its buffer cannot fit the budget, streaming must
    /// fail with a clear error instead of overflowing memory.
    #[test]
    fn test_persistent_streamed_output_budget_too_small() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        write_string(&mut buf, "output.weight");
        write_u32(&mut buf, 2);
        write_u64(&mut buf, 256); // in=256
        write_u64(&mut buf, 4); // out=4
        write_u32(&mut buf, 0); // F32
        write_u64(&mut buf, 0);
        let pos = buf.len() as u64;
        let aligned = align_offset(pos, 32);
        buf.extend(vec![0u8; (aligned - pos) as usize]);
        buf.extend(vec![0u8; 256 * 4 * 4]);
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        tmp.flush().unwrap();

        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let desc = ds.get_descriptor("output.weight").unwrap().clone();
        let streamed = PersistentWeight::Streamed(desc);
        // One row = 1024 B raw + 1024 B f32 buffer; budget far below that.
        let mut budget = ramforge_core::memory::MemoryBudget::new(256).unwrap();
        let hidden = vec![1.0f32; 256];
        let mut logits = vec![0.0f32; 4];
        let err = streamed
            .compute_logits_into(&hidden, &ds, &mut budget, &mut logits)
            .unwrap_err();
        assert!(err.contains("budget too small") || err.contains("insufficient"), "got: {}", err);
    }
}
