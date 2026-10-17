//! Persistent weight streaming – token_embd, output_norm, output
//!
//! Policy: a persistent tensor whose actual resident representation consumes
//! at most 25% of the total budget is kept resident; a larger one is streamed
//! on demand.
//!
//! Memory contract (M7.2):
//! - streamed embedding lookup reads exactly one row and is budget-charged;
//!   F32 goes directly into final storage, while converted/quantized formats
//!   charge their raw row workspace separately from the caller-owned decoded
//!   output buffer;
//! - streamed matvec/logits run in row chunks sized by the *available*
//!   budget (bounded temp, no full-tensor reads, no full F32 expansions);
//! - logits are written into a caller-provided buffer (single allocation,
//!   owned by the engine).

#![allow(clippy::needless_range_loop)]

use ramforge_core::{
    datasource::GgufDataSource, memory::MemoryBudget, model::TensorDescriptor, tensor::TensorData,
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

    /// Compatibility helper returning an owned embedding vector.
    ///
    /// The returned vector belongs to the caller and outlives the temporary
    /// `MemoryBudget` scope used to construct it. Consequently this helper does
    /// not claim that the returned vector is budget-covered. Managed execution
    /// must use [`Self::get_embedding_into`], whose caller-owned output buffer
    /// remains inside its surrounding charge.
    pub fn get_embedding(
        &self,
        token_id: usize,
        n_embd: usize,
        data_source: &GgufDataSource,
        budget: &mut MemoryBudget,
        _stats: &mut ResidencyStats,
    ) -> Result<Vec<f32>, String> {
        let mut output = Vec::new();
        output
            .try_reserve_exact(n_embd)
            .map_err(|error| format!("failed to reserve embedding output: {error}"))?;
        output.resize(n_embd, 0.0);
        self.get_embedding_into(token_id, n_embd, data_source, budget, &mut output)?;
        Ok(output)
    }

    /// Fill a caller-owned embedding buffer without returning an allocation
    /// whose charge could expire before the buffer is dropped. The caller must
    /// keep its charge alive for `output`'s lifetime.
    pub fn get_embedding_into(
        &self,
        token_id: usize,
        n_embd: usize,
        data_source: &GgufDataSource,
        budget: &mut MemoryBudget,
        output: &mut [f32],
    ) -> Result<(), String> {
        if n_embd == 0 || output.len() != n_embd {
            return Err(format!(
                "embedding output length mismatch: expected {}, got {}",
                n_embd,
                output.len()
            ));
        }
        let n_embd_u64 = u64::try_from(n_embd)
            .map_err(|_| "embedding dimension exceeds u64".to_string())?;
        let decoded_bytes = n_embd_u64
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| "embedding output byte size overflow".to_string())?;

        match self {
            Self::Resident(td) => budget.with_temp("tmp:embd_row", decoded_bytes, |_budget| {
                let row = td
                    .get_embedding(token_id, n_embd)
                    .map_err(|e| format!("embedding lookup failed: {}", e))?;
                output.copy_from_slice(&row);
                Ok::<(), String>(())
            }),
            Self::Streamed(desc) => {
                if desc.dimensions.len() != 2 {
                    return Err(format!(
                        "streamed embedding tensor '{}' must be 2D, got {:?}",
                        desc.name, desc.dimensions
                    ));
                }
                let row_width = usize::try_from(desc.dimensions[0])
                    .map_err(|_| "streamed embedding row width exceeds platform usize".to_string())?;
                let row_count = usize::try_from(desc.dimensions[1])
                    .map_err(|_| "streamed embedding row count exceeds platform usize".to_string())?;
                if row_width != n_embd || token_id >= row_count {
                    return Err(format!(
                        "streamed embedding shape mismatch: tensor '{}' is [{}, {}], requested row {} with width {}",
                        desc.name, row_width, row_count, token_id, n_embd
                    ));
                }
                let token_id_u64 = u64::try_from(token_id)
                    .map_err(|_| "embedding token index exceeds u64".to_string())?;
                let element_offset = token_id_u64
                    .checked_mul(n_embd_u64)
                    .ok_or_else(|| "embedding row element offset overflow".to_string())?;
                if desc.ggml_type == GgmlType::F32 {
                    budget.with_temp("tmp:embd_row", decoded_bytes, |_budget| {
                        let row = data_source
                            .read_f32_tensor_range_by_descriptor(
                                desc,
                                element_offset,
                                n_embd_u64,
                            )
                            .map_err(|e| format!("failed to read embedding row: {}", e))?;
                        output.copy_from_slice(&row);
                        Ok::<(), String>(())
                    })
                } else {
                    let row_bytes = u64::try_from(row_bytes_for(desc, n_embd)?)
                        .map_err(|_| "embedding row byte size exceeds u64".to_string())?;
                    let byte_offset = token_id_u64
                        .checked_mul(row_bytes)
                        .ok_or_else(|| "embedding row byte offset overflow".to_string())?;
                    // `output` is caller-owned and must be covered by its
                    // surrounding lifetime charge. This path decodes directly
                    // into it, so the temporary charge covers only the raw
                    // streamed row; adding `decoded_bytes` here would count
                    // the same output buffer twice.
                    let temp_bytes = row_bytes;
                    budget.with_temp("tmp:embd_row", temp_bytes, |_budget| {
                        let raw = data_source
                            .read_tensor_range_by_descriptor(desc, byte_offset, row_bytes)
                            .map_err(|e| format!("failed to read embedding row: {}", e))?;
                        ramforge_core::tensor::decode_row_to_f32(
                            desc.ggml_type,
                            &raw,
                            n_embd,
                            output,
                        )
                        .map_err(|e| format!("failed to dequantize embedding row: {}", e))
                    })
                }
            }
        }
    }

    /// `y = W * x` where W is this persistent weight (ggml layout
    /// `[in, out]`; x.len() == in, y.len() == out).
    ///
    /// - Resident: compact matvec in memory (the scalar quantized utility
    ///   decodes one output row at a time; the streaming model's dispatch
    ///   retains the fused block-wise path).
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
    /// as `output_norm.weight` (n_embd elements); the caller must keep a
    /// matching charge live for the returned `Vec`'s full lifetime. Managed
    /// forward execution uses this only inside its coarse forward charge.
    pub fn to_f32_vec(&self, data_source: &GgufDataSource) -> Result<Vec<f32>, String> {
        match self {
            Self::Resident(td) => td.to_f32_vec().map_err(|e| e.to_string()),
            Self::Streamed(desc) => {
                if desc.ggml_type == GgmlType::F32 {
                    data_source
                        .read_f32_tensor_by_descriptor(desc)
                        .map_err(|e| {
                            format!("failed to read persistent tensor '{}': {}", desc.name, e)
                        })
                } else {
                    let raw = data_source.read_tensor_by_descriptor(desc).map_err(|e| {
                        format!("failed to read persistent tensor '{}': {}", desc.name, e)
                    })?;
                    let td = TensorData::from_bytes(
                        desc.ggml_type,
                        desc.dimensions.clone(),
                        desc.num_elements,
                        raw,
                    )
                    .map_err(|e| format!("failed to create TensorData: {}", e))?;
                    td.into_f32_vec().map_err(|e| e.to_string())
                }
            }
        }
    }
}

/// Row bytes for a contiguous ggml row of `in_dim` elements of `desc`'s type.
pub fn row_bytes_for(desc: &TensorDescriptor, in_dim: usize) -> Result<usize, String> {
    let (block_elements, block_bytes) = match desc.ggml_type {
        GgmlType::F32 => {
            return in_dim
                .checked_mul(4)
                .ok_or_else(|| "F32 row byte size overflow".to_string())
        },
        GgmlType::F16 | GgmlType::BF16 => {
            return in_dim
                .checked_mul(2)
                .ok_or_else(|| "16-bit row byte size overflow".to_string())
        },
        GgmlType::Q4_0 => (32, 18),
        GgmlType::Q8_0 => (32, 34),
        GgmlType::Q4_K => (256, 144),
        GgmlType::Q5_K => (256, 176),
        GgmlType::Q6_K => (256, 210),
        GgmlType::Q2_K => (256, 84),
        GgmlType::Q3_K => (256, 110),
        GgmlType::Q8_K => (256, 292),
        _ => {
            return Err(format!(
                "unsupported persistent weight type for streaming: {}",
                desc.ggml_type.name()
            ))
        }
    };
    if !in_dim.is_multiple_of(block_elements) {
        return Err(format!(
            "row dimension {} is not divisible by block size {} for {}",
            in_dim,
            block_elements,
            desc.ggml_type.name()
        ));
    }
    (in_dim / block_elements)
        .checked_mul(block_bytes)
        .ok_or_else(|| format!("{} row byte size overflow", desc.ggml_type.name()))
}

/// Chunked, budget-aware streamed matvec for a non-resident 2D tensor.
///
/// Layout: ggml `[in, out]`; the file stores `out` contiguous rows of `in`
/// elements. F32 chunks are read directly into final F32 storage and dotted
/// in place. Other formats retain a raw chunk plus one reused decoded row.
/// The exact owned workspace is charged via `with_temp`.
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
    let in_dim = usize::try_from(desc.dimensions[0])
        .map_err(|_| format!("tensor '{}' input dimension exceeds platform usize", desc.name))?;
    let out_dim = usize::try_from(desc.dimensions[1])
        .map_err(|_| format!("tensor '{}' output dimension exceeds platform usize", desc.name))?;
    if in_dim == 0 || out_dim == 0 {
        return Err(format!(
            "streamed matvec requires non-zero dimensions, '{}' has [{}, {}]",
            desc.name, in_dim, out_dim
        ));
    }
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

    let in_dim_u64 = u64::try_from(in_dim)
        .map_err(|_| "streamed matvec input dimension exceeds u64".to_string())?;
    let row_bytes = u64::try_from(row_bytes_for(desc, in_dim)?)
        .map_err(|_| "streamed matvec row byte size exceeds u64".to_string())?;
    let row_f32_bytes = in_dim_u64
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .ok_or_else(|| "streamed matvec F32 row byte size overflow".to_string())?;

    // Chunk size: bounded by budget share and an absolute cap, at least one
    // row, rounded down to whole rows. If one row does not fit, fail clearly.
    let avail_share = budget.available_bytes() / STREAM_CHUNK_BUDGET_SHARE;
    let chunk_target = avail_share.min(MAX_STREAM_CHUNK_BYTES).max(row_bytes);
    let rows_per_chunk = (chunk_target / row_bytes).max(1);
    let chunk_bytes = rows_per_chunk
        .checked_mul(row_bytes)
        .ok_or_else(|| "streamed matvec chunk byte size overflow".to_string())?;
    let direct_f32 = desc.ggml_type == GgmlType::F32;
    let temp_bytes = if direct_f32 {
        chunk_bytes
    } else {
        chunk_bytes
            .checked_add(row_f32_bytes)
            .ok_or_else(|| "streamed matvec temporary byte size overflow".to_string())?
    };
    if !budget.can_allocate(temp_bytes) {
        let detail = if direct_f32 {
            "direct F32 chunk"
        } else {
            "raw chunk + decoded row buffer"
        };
        return Err(format!(
            "RAM budget too small to stream tensor '{}': need {} bytes of temporary working set ({}), available {}",
            desc.name,
            temp_bytes,
            detail,
            budget.available_bytes()
        ));
    }

    if direct_f32 {
        budget.with_temp("tmp:streamed_matvec", temp_bytes, |_b| {
            let mut row_start = 0usize;
            while row_start < out_dim {
                let remaining_rows = u64::try_from(out_dim - row_start)
                    .map_err(|_| "streamed F32 remaining row count exceeds u64".to_string())?;
                let rows_this = usize::try_from(rows_per_chunk.min(remaining_rows))
                    .map_err(|_| "streamed F32 chunk row count exceeds usize".to_string())?;
                let row_start_u64 = u64::try_from(row_start)
                    .map_err(|_| "streamed F32 row index exceeds u64".to_string())?;
                let rows_this_u64 = u64::try_from(rows_this)
                    .map_err(|_| "streamed F32 chunk row count exceeds u64".to_string())?;
                let element_offset = row_start_u64
                    .checked_mul(in_dim_u64)
                    .ok_or_else(|| "streamed F32 element offset overflow".to_string())?;
                let element_count = rows_this_u64
                    .checked_mul(in_dim_u64)
                    .ok_or_else(|| "streamed F32 element count overflow".to_string())?;
                let chunk = data_source
                    .read_f32_tensor_range_by_descriptor(desc, element_offset, element_count)
                    .map_err(|e| {
                        format!("failed to read direct F32 chunk of '{}': {}", desc.name, e)
                    })?;
                for r in 0..rows_this {
                    let chunk_row_start = r
                        .checked_mul(in_dim)
                        .ok_or_else(|| "streamed F32 row offset overflow".to_string())?;
                    let row_end = (r + 1)
                        .checked_mul(in_dim)
                        .ok_or_else(|| "streamed F32 row end overflow".to_string())?;
                    let row = chunk.get(chunk_row_start..row_end).ok_or_else(|| {
                        "streamed F32 row is outside chunk".to_string()
                    })?;
                    let mut sum = 0.0f32;
                    for i in 0..in_dim {
                        sum += row[i] * x[i];
                    }
                    y[row_start + r] = sum;
                }
                row_start += rows_this;
            }
            Ok(())
        })
    } else {
        budget.with_temp("tmp:streamed_matvec", temp_bytes, |_b| {
            let mut row_buf = Vec::new();
            row_buf
                .try_reserve_exact(in_dim)
                .map_err(|error| format!("failed to reserve streamed decoded row: {error}"))?;
            row_buf.resize(in_dim, 0.0);
            let mut row_start = 0usize;
            while row_start < out_dim {
                let remaining_rows = u64::try_from(out_dim - row_start)
                    .map_err(|_| "streamed matvec remaining row count exceeds u64".to_string())?;
                let rows_this = usize::try_from(rows_per_chunk.min(remaining_rows))
                    .map_err(|_| "streamed matvec chunk row count exceeds usize".to_string())?;
                let row_start_u64 = u64::try_from(row_start)
                    .map_err(|_| "streamed matvec row index exceeds u64".to_string())?;
                let rows_this_u64 = u64::try_from(rows_this)
                    .map_err(|_| "streamed matvec chunk row count exceeds u64".to_string())?;
                let offset = row_start_u64
                    .checked_mul(row_bytes)
                    .ok_or_else(|| "streamed matvec byte offset overflow".to_string())?;
                let len = rows_this_u64
                    .checked_mul(row_bytes)
                    .ok_or_else(|| "streamed matvec chunk length overflow".to_string())?;
                let chunk = data_source
                    .read_tensor_range_by_descriptor(desc, offset, len)
                    .map_err(|e| format!("failed to read chunk of '{}': {}", desc.name, e))?;
                for r in 0..rows_this {
                    let row_u64 = u64::try_from(r)
                        .map_err(|_| "streamed matvec row index exceeds u64".to_string())?;
                    let row_offset = row_u64
                        .checked_mul(row_bytes)
                        .ok_or_else(|| "streamed matvec row offset overflow".to_string())?;
                    let row_end = row_offset
                        .checked_add(row_bytes)
                        .ok_or_else(|| "streamed matvec row end overflow".to_string())?;
                    let chunk_row_start = usize::try_from(row_offset)
                        .map_err(|_| "streamed matvec row offset exceeds usize".to_string())?;
                    let chunk_row_end = usize::try_from(row_end)
                        .map_err(|_| "streamed matvec row end exceeds usize".to_string())?;
                    let row_slice = chunk.get(chunk_row_start..chunk_row_end).ok_or_else(|| {
                        "streamed matvec decoded row is outside chunk".to_string()
                    })?;
                    ramforge_core::tensor::decode_row_to_f32(
                        desc.ggml_type,
                        row_slice,
                        in_dim,
                        &mut row_buf,
                    )
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
}

/// Keep a persistent tensor resident only when its actual in-memory
/// representation is at most one quarter of the configured budget.
/// Division avoids overflow while preserving the existing 25% boundary.
pub fn should_keep_resident(resident_bytes: u64, budget_total: u64) -> bool {
    resident_bytes <= budget_total / 4
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
    fn write_u32<W: Write>(w: &mut W, v: u32) {
        w.write_all(&v.to_le_bytes()).unwrap();
    }
    fn write_u64<W: Write>(w: &mut W, v: u64) {
        w.write_all(&v.to_le_bytes()).unwrap();
    }

    #[test]
    fn test_persistent_policy_uses_resident_bytes_and_is_overflow_safe() {
        assert!(should_keep_resident(250, 1000));
        assert!(!should_keep_resident(251, 1000));
        assert!(!should_keep_resident(u64::MAX, u64::MAX));
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

        let mut output = [0.0f32; 8];
        budget
            .with_temp("tmp:embedding_output", 8 * 4, |budget| {
                streamed.get_embedding_into(1, 8, &ds, budget, &mut output)?;
                assert_eq!(budget.get("tmp:embedding_output"), Some(32));
                assert_eq!(output[0], 1.0);
                Ok::<(), String>(())
            })
            .unwrap();
        assert_eq!(budget.used_bytes(), 0);
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

        // Rows are 16 bytes; budget forces chunks of ~2-3 rows. The direct
        // F32 path owns only the final chunk (no raw chunk + decoded row).
        // Reserve most of a small budget so only a thin share is available.
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
            assert!(
                (logits[i] - expected[i]).abs() < 1e-5,
                "logit {} = {}",
                i,
                logits[i]
            );
        }
        assert_eq!(budget.used_bytes(), 800, "no temp charge may leak");
        budget.release("pinned").unwrap();

        // Resident path must agree with streamed path.
        let raw = ds.read_tensor("output.weight").unwrap();
        let td = TensorData::from_bytes(
            GgmlType::F32,
            desc.dimensions.clone(),
            desc.num_elements,
            raw,
        )
        .unwrap();
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
        // One direct F32 row = 1024 B; budget is far below that.
        let mut budget = ramforge_core::memory::MemoryBudget::new(256).unwrap();
        let hidden = vec![1.0f32; 256];
        let mut logits = vec![0.0f32; 4];
        let err = streamed
            .compute_logits_into(&hidden, &ds, &mut budget, &mut logits)
            .unwrap_err();
        assert!(
            err.contains("budget too small") || err.contains("insufficient"),
            "got: {}",
            err
        );
    }
}
