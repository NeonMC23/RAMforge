//! Runtime adapter between materialized TensorData and the pure compute core.
//!
//! This module owns only representation dispatch, profiling, and work
//! partitioning. It does not define transformer mathematics; scalar/reference
//! semantics live in `ramforge_core::compute` and quantized block semantics
//! live in `ramforge_core::quant`.

use std::borrow::Cow;

use rayon::prelude::*;

use ramforge_core::{
    quant::{
        matvec_q2_k, matvec_q3_k, matvec_q4_0_row_range, matvec_q4_k, matvec_q5_k,
        matvec_q6_k_row_range, matvec_q8_0, matvec_q8_k, BLOCK_SIZE_Q4_0, BLOCK_SIZE_Q6_K,
        QK4_0, QK_K,
    },
    tensor::TensorData,
    types::GgmlType,
};

use crate::backend::ComputeBackend;
use crate::profile::{ProfileEvent, Profiler};

pub(crate) fn tensor_f32_view(tensor: &TensorData) -> Result<Cow<'_, [f32]>, String> {
    if let Some((data, _)) = tensor.as_f32_slice() {
        Ok(Cow::Borrowed(data))
    } else {
        tensor
            .to_f32_vec()
            .map(Cow::Owned)
            .map_err(|error| error.to_string())
    }
}

/// Matvec dispatch under the single explicit ggml layout (`shape = [in, out]`):
/// resident F32 data goes through the SIMD/threaded compute backend; all other
/// types use the compact block-wise kernels (no full F32 expansion, no
/// orientation guessing).
pub(crate) fn matvec_backend<B: ComputeBackend>(
    backend: &B,
    profiler: &Profiler,
    td: &TensorData,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    if let Some((data, shape)) = td.as_f32_slice() {
        let started = profiler.start();
        let result = backend.matvec(data, shape, x, y);
        profiler.record_since(ProfileEvent::FloatMatvec, started);
        result
    } else {
        let started = profiler.start();
        let result = quantized_matvec_dispatch(backend, td, x, y, profiler);
        profiler.record_since(ProfileEvent::QuantizedMatvec, started);
        result
    }
}

/// Minimum output rows (out_dim) before we hand Q4_0 / Q6_K work to the
/// backend thread pool. Below this threshold we keep the fused scalar
/// kernel and avoid Rayon pool/spawn overhead.
///
/// Derived from the `quantized_matvec_bench` microbenchmark in this file:
///   * 32×1536 and 64×1536 (in_dim=1536): no measurable speedup at 2/4/8/9
///     threads; best-case times match the scalar path to noise.
///   * 128×1536 and larger: parallel dispatch produces measurable speedup
///     (≈1.3–2.0× at 2 threads in the sandbox; higher core counts scale
///     further on real hardware).
/// Real Qwen2.5-1.5B layer shapes are 1536 / 8960 / 151936 rows — all well
/// above this threshold and parallelize cleanly.
const QUANT_PARALLEL_ROW_THRESHOLD: usize = 128;

/// Split `out_dim` rows into `n_chunks` contiguous (start, count) ranges
/// with at most one row of imbalance across chunks.
fn make_row_chunks(out_dim: usize, n_chunks: usize) -> Vec<(usize, usize)> {
    let n = n_chunks.min(out_dim).max(1);
    let base = out_dim / n;
    let extra = out_dim % n;
    let mut out = Vec::with_capacity(n);
    let mut cursor = 0usize;
    for i in 0..n {
        let count = base + if i < extra { 1 } else { 0 };
        out.push((cursor, count));
        cursor += count;
    }
    out
}

pub(crate) fn quantized_matvec_dispatch<B: ComputeBackend>(
    backend: &B,
    td: &TensorData,
    x: &[f32],
    y: &mut [f32],
    profiler: &Profiler,
) -> Result<(), String> {
    /// Returns (raw_data, kernel_shape=[out, in]) for a quantized tensor if
    /// it matches one of the row-parallel kernels, otherwise None to fall
    /// back to the compact format-specific dispatcher below.
    fn as_quantized_view<'a>(td: &'a TensorData) -> Option<(GgmlType, &'a [u8], [usize; 2])> {
        let (qt, ty) = match td {
            TensorData::Q4_0(qt) => (qt, GgmlType::Q4_0),
            TensorData::Q6_K(qt) => (qt, GgmlType::Q6_K),
            _ => return None,
        };
        if qt.shape.len() != 2 {
            return None;
        }
        // ggml [in, out] → kernel layout [out, in]
        Some((ty, &qt.raw_data, [qt.shape[1], qt.shape[0]]))
    }

    match as_quantized_view(td) {
        Some((GgmlType::Q4_0, raw, [out_dim, in_dim])) => {
            // Counters follow the same [in, out] semantics as before.
            let blocks = (out_dim as u64).saturating_mul((in_dim / QK4_0) as u64);
            let weight_bytes = blocks.saturating_mul(BLOCK_SIZE_Q4_0 as u64);
            profiler.record_q4_0_matvec(out_dim as u64, in_dim as u64, blocks, weight_bytes);

            let w_shape = [out_dim, in_dim];
            let n_threads = backend.num_threads();
            let parallelize = n_threads > 1 && out_dim >= QUANT_PARALLEL_ROW_THRESHOLD;
            if parallelize {
                // Split the output rows into n_threads contiguous ranges and
                // run each range through the fused scalar kernel on the
                // backend's bounded Rayon pool. No per-row heap allocation;
                // x and raw_data are shared read-only.
                //
                // SAFETY: Row ranges are non-overlapping by construction
                // (`make_row_chunks` partitions [0, out_dim)), so each worker
                // writes to a disjoint y subslice. We hand out raw pointers
                // because Rayon's `into_par_iter().try_for_each` requires a
                // `Fn` (not `FnMut`) closure and would otherwise forbid
                // capturing an outer `&mut [f32]`.
                let blocks_per_row = in_dim / QK4_0;
                let row_bytes = blocks_per_row * BLOCK_SIZE_Q4_0;
                let ranges = make_row_chunks(out_dim, n_threads);
                // SAFETY: row ranges are non-overlapping partitions of
                // [0, out_dim), so each worker writes to a disjoint y subslice
                // and reads a disjoint w subslice. We transmit the base
                // addresses as usizes because raw pointers are neither Send
                // nor Sync; the addresses remain valid for the whole call
                // since both `y` and `raw` outlive the worker pool scope.
                let y_addr = y.as_mut_ptr() as usize;
                let raw_addr = raw.as_ptr() as usize;
                backend.in_worker_pool(move || {
                    ranges.into_par_iter().try_for_each(
                        |(row_start, row_count)| -> Result<(), String> {
                            let w_offset = row_start * row_bytes;
                            let w_sub = unsafe {
                                std::slice::from_raw_parts(
                                    (raw_addr + w_offset) as *const u8,
                                    row_count * row_bytes,
                                )
                            };
                            let y_sub = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (y_addr + row_start * std::mem::size_of::<f32>()) as *mut f32,
                                    row_count,
                                )
                            };
                            let sub_shape = [row_count, in_dim];
                            matvec_q4_0_row_range(w_sub, &sub_shape, x, y_sub, 0, None)
                                .map_err(|e| e.to_string())
                        },
                    )
                })?;
                Ok(())
            } else {
                matvec_q4_0_row_range(raw, &w_shape, x, y, 0, None).map_err(|e| e.to_string())
            }
        }
        Some((GgmlType::Q6_K, raw, [out_dim, in_dim])) => {
            let w_shape = [out_dim, in_dim];
            let n_threads = backend.num_threads();
            let parallelize = n_threads > 1 && out_dim >= QUANT_PARALLEL_ROW_THRESHOLD;
            if parallelize {
                let blocks_per_row = in_dim / QK_K;
                let row_bytes = blocks_per_row * BLOCK_SIZE_Q6_K;
                let ranges = make_row_chunks(out_dim, n_threads);
                // SAFETY: same disjoint-range reasoning as the Q4_0 branch.
                let y_addr = y.as_mut_ptr() as usize;
                let raw_addr = raw.as_ptr() as usize;
                backend.in_worker_pool(move || {
                    ranges.into_par_iter().try_for_each(
                        |(row_start, row_count)| -> Result<(), String> {
                            let w_offset = row_start * row_bytes;
                            let w_sub = unsafe {
                                std::slice::from_raw_parts(
                                    (raw_addr + w_offset) as *const u8,
                                    row_count * row_bytes,
                                )
                            };
                            let y_sub = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (y_addr + row_start * std::mem::size_of::<f32>()) as *mut f32,
                                    row_count,
                                )
                            };
                            let sub_shape = [row_count, in_dim];
                            matvec_q6_k_row_range(w_sub, &sub_shape, x, y_sub, 0, None)
                                .map_err(|e| e.to_string())
                        },
                    )
                })?;
                Ok(())
            } else {
                matvec_q6_k_row_range(raw, &w_shape, x, y, 0, None).map_err(|e| e.to_string())
            }
        }
        _ => optimized_quantized_matvec(td, x, y),
    }
}

/// Keep the compact, block-wise optimized path for formats without a row
/// parallel dispatcher. The pure reference remains the authority for tests
/// and the storage-independent API; this adapter avoids introducing an
/// uncharged heap allocation into the bounded-memory runtime.
fn optimized_quantized_matvec(td: &TensorData, x: &[f32], y: &mut [f32]) -> Result<(), String> {
    let (ggml_type, raw, shape) = match td {
        TensorData::Q2_K(tensor) => (GgmlType::Q2_K, &tensor.raw_data, &tensor.shape),
        TensorData::Q3_K(tensor) => (GgmlType::Q3_K, &tensor.raw_data, &tensor.shape),
        TensorData::Q4_K(tensor) => (GgmlType::Q4_K, &tensor.raw_data, &tensor.shape),
        TensorData::Q5_K(tensor) => (GgmlType::Q5_K, &tensor.raw_data, &tensor.shape),
        TensorData::Q8_0(tensor) => (GgmlType::Q8_0, &tensor.raw_data, &tensor.shape),
        TensorData::Q8_K(tensor) => (GgmlType::Q8_K, &tensor.raw_data, &tensor.shape),
        other => {
            return Err(format!(
                "optimized quantized dispatch received unsupported tensor variant {:?}",
                other.ggml_type()
            ))
        }
    };
    if shape.len() != 2 {
        return Err(format!(
            "{} matvec expects 2D tensor shape, got {:?}",
            ggml_type.name(),
            shape
        ));
    }
    let w_shape = [shape[1], shape[0]];
    let result = match ggml_type {
        GgmlType::Q2_K => matvec_q2_k(raw, &w_shape, x, y),
        GgmlType::Q3_K => matvec_q3_k(raw, &w_shape, x, y),
        GgmlType::Q4_K => matvec_q4_k(raw, &w_shape, x, y),
        GgmlType::Q5_K => matvec_q5_k(raw, &w_shape, x, y),
        GgmlType::Q8_0 => matvec_q8_0(raw, &w_shape, x, y),
        GgmlType::Q8_K => matvec_q8_k(raw, &w_shape, x, y),
        _ => unreachable!("optimized dispatch match is exhaustive"),
    };
    result.map_err(|error| error.to_string())
}

