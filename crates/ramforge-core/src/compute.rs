//! Pure numerical compute core for RAMforge.
//!
//! This module deliberately has no dependency on GGUF storage, streaming,
//! `MemoryBudget`, layer caches, planners, or generation. It accepts
//! materialized numerical buffers and returns numerical results.
//!
//! ## Matrix convention
//!
//! `MatrixShape` is the explicit GGML/GGUF convention:
//!
//! * `input` is `ne[0]` and is the length of the input vector;
//! * `output` is `ne[1]` and is the number of output rows;
//! * raw/F32 weights are row-major `[output][input]`;
//! * `y[o] = sum_i weights[o * input + i] * x[i]`.
//!
//! Quantized reference matvecs decode one complete output row through the
//! authoritative block decoders in [`crate::quant`] and then perform the same
//! scalar F32 accumulation as `matvec_f32_reference`. Optimized kernels must
//! be compared with these functions; they must not redefine the layout.

use std::fmt;

use crate::quant;
use crate::types::GgmlType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeError(pub String);

impl fmt::Display for ComputeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ComputeError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixShape {
    pub input: usize,
    pub output: usize,
}

impl MatrixShape {
    pub const fn new(input: usize, output: usize) -> Self {
        Self { input, output }
    }

    pub fn validate(&self, weights_len: usize, x_len: usize, y_len: usize) -> Result<(), ComputeError> {
        let expected_weights = self
            .input
            .checked_mul(self.output)
            .ok_or_else(|| ComputeError("matrix weight size overflow".to_string()))?;
        if weights_len != expected_weights || x_len != self.input || y_len != self.output {
            return Err(ComputeError(format!(
                "matvec arity/shape mismatch: shape [input={}, output={}] requires weights exactly {}, x {}, y {}; got weights {}, x {}, y {}",
                self.input,
                self.output,
                expected_weights,
                self.input,
                self.output,
                weights_len,
                x_len,
                y_len
            )));
        }
        Ok(())
    }
}

/// Explicit dimensions and history for one-token causal attention.
///
/// K/V tensors use `[history_position][kv_head][head_dim]` layout, while Q and
/// the current K/V use `[head][head_dim]` layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionConfig {
    pub history_len: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl AttentionConfig {
    pub const fn new(
        history_len: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            history_len,
            query_heads,
            kv_heads,
            head_dim,
        }
    }
}

/// Scalar F32 matrix-vector multiplication. Accumulation is deliberately F32
/// and proceeds in increasing input-index order.
pub fn matvec_f32_reference(
    weights: &[f32],
    shape: MatrixShape,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), ComputeError> {
    shape.validate(weights.len(), x.len(), y.len())?;
    for output_index in 0..shape.output {
        let row = &weights[output_index * shape.input..(output_index + 1) * shape.input];
        let mut sum = 0.0f32;
        for input_index in 0..shape.input {
            sum += row[input_index] * x[input_index];
        }
        y[output_index] = sum;
    }
    Ok(())
}

/// Scalar reference matvec over a contiguous output-row range. `y` is local
/// to the supplied range: `y[0]` corresponds to `row_start`, never to the
/// absolute tensor row `0`.
pub fn matvec_f32_reference_row_range(
    weights: &[f32],
    shape: MatrixShape,
    x: &[f32],
    y: &mut [f32],
    row_start: usize,
    row_count: usize,
) -> Result<(), ComputeError> {
    if row_start
        .checked_add(row_count)
        .is_none_or(|end| end > shape.output)
    {
        return Err(ComputeError(format!(
            "row range {}..{} exceeds output dimension {}",
            row_start,
            row_start.saturating_add(row_count),
            shape.output
        )));
    }
    if y.len() != row_count {
        return Err(ComputeError(format!(
            "row-range output length mismatch: expected {}, got {}",
            row_count,
            y.len()
        )));
    }
    if x.len() != shape.input {
        return Err(ComputeError(format!(
            "row-range input length mismatch: expected {}, got {}",
            shape.input,
            x.len()
        )));
    }
    let expected_weights = shape
        .input
        .checked_mul(shape.output)
        .ok_or_else(|| ComputeError("matrix weight size overflow".to_string()))?;
    if weights.len() != expected_weights {
        return Err(ComputeError(format!(
            "matrix weight buffer length mismatch: expected {}, got {}",
            expected_weights,
            weights.len()
        )));
    }
    for (local_row, output) in y.iter_mut().enumerate() {
        let output_index = row_start + local_row;
        let row = &weights[output_index * shape.input..(output_index + 1) * shape.input];
        let mut sum = 0.0f32;
        for (&weight, &input) in row.iter().zip(x.iter()) {
            sum += weight * input;
        }
        *output = sum;
    }
    Ok(())
}

/// Scalar reference matvec for the quantized formats currently materialized by
/// RAMforge. The row decoder in `quant` is the only block interpretation used
/// here; no fused quantized interpretation is duplicated in this module.
pub fn quantized_matvec_reference(
    ggml_type: GgmlType,
    raw: &[u8],
    shape: MatrixShape,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), ComputeError> {
    if x.len() != shape.input || y.len() != shape.output {
        return Err(ComputeError(format!(
            "quantized matvec shape mismatch: shape {:?}, x {}, y {}",
            shape,
            x.len(),
            y.len()
        )));
    }
    let row_bytes = quantized_row_bytes(ggml_type, shape.input)?;
    let expected_bytes = row_bytes
        .checked_mul(shape.output)
        .ok_or_else(|| ComputeError("quantized weight size overflow".to_string()))?;
    if raw.len() < expected_bytes {
        return Err(ComputeError(format!(
            "quantized weight buffer truncated: expected {}, got {}",
            expected_bytes,
            raw.len()
        )));
    }

    for output_index in 0..shape.output {
        let row_bytes_slice = &raw[output_index * row_bytes..(output_index + 1) * row_bytes];
        let mut decoded = vec![0.0f32; shape.input];
        decode_quantized_row(ggml_type, row_bytes_slice, shape.input, &mut decoded)?;
        let mut sum = 0.0f32;
        for input_index in 0..shape.input {
            sum += decoded[input_index] * x[input_index];
        }
        y[output_index] = sum;
    }
    Ok(())
}

pub fn quantized_row_bytes(ggml_type: GgmlType, input: usize) -> Result<usize, ComputeError> {
    let (qk, block_bytes) = match ggml_type {
        GgmlType::Q4_0 => (quant::QK4_0, quant::BLOCK_SIZE_Q4_0),
        GgmlType::Q8_0 => (quant::QK8_0, quant::BLOCK_SIZE_Q8_0),
        GgmlType::Q2_K => (quant::QK_K, quant::BLOCK_SIZE_Q2_K),
        GgmlType::Q3_K => (quant::QK_K, quant::BLOCK_SIZE_Q3_K),
        GgmlType::Q4_K => (quant::QK_K, quant::BLOCK_SIZE_Q4_K),
        GgmlType::Q5_K => (quant::QK_K, quant::BLOCK_SIZE_Q5_K),
        GgmlType::Q6_K => (quant::QK_K, quant::BLOCK_SIZE_Q6_K),
        GgmlType::Q8_K => (quant::QK_K, quant::BLOCK_SIZE_Q8_K),
        unsupported => {
            return Err(ComputeError(format!(
                "no reference row layout registered for {}",
                unsupported.name()
            )))
        }
    };
    if !input.is_multiple_of(qk) {
        return Err(ComputeError(format!(
            "{} input dimension {} is not divisible by block width {}",
            ggml_type.name(),
            input,
            qk
        )));
    }
    (input / qk)
        .checked_mul(block_bytes)
        .ok_or_else(|| ComputeError("quantized row size overflow".to_string()))
}

fn decode_quantized_row(
    ggml_type: GgmlType,
    bytes: &[u8],
    input: usize,
    out: &mut [f32],
) -> Result<(), ComputeError> {
    let result = match ggml_type {
        GgmlType::Q4_0 => quant::dequantize_row_q4_0(bytes, input, out),
        GgmlType::Q8_0 => quant::dequantize_row_q8_0(bytes, input, out),
        GgmlType::Q2_K => quant::dequantize_row_q2_k(bytes, input, out),
        GgmlType::Q3_K => quant::dequantize_row_q3_k(bytes, input, out),
        GgmlType::Q4_K => quant::dequantize_row_q4_k(bytes, input, out),
        GgmlType::Q5_K => quant::dequantize_row_q5_k(bytes, input, out),
        GgmlType::Q6_K => quant::dequantize_row_q6_k(bytes, input, out),
        GgmlType::Q8_K => quant::dequantize_row_q8_k(bytes, input, out),
        unsupported => {
            return Err(ComputeError(format!(
                "no reference decoder registered for {}",
                unsupported.name()
            )))
        }
    };
    result.map_err(|error| ComputeError(error.to_string()))
}

/// RMSNorm reference semantics used by the CPU model path: sum of squares,
/// mean, epsilon addition, square root, normalization, then elementwise weight;
/// reduction and output arithmetic are F32 in input order.
pub fn rms_norm_reference(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> Result<(), ComputeError> {
    if input.is_empty() || input.len() != weight.len() || input.len() != output.len() {
        return Err(ComputeError(format!(
            "RMSNorm shape mismatch: input {}, weight {}, output {}",
            input.len(),
            weight.len(),
            output.len()
        )));
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(ComputeError("RMSNorm epsilon must be finite and non-negative".to_string()));
    }
    let mut sum_squares = 0.0f32;
    for &value in input {
        sum_squares += value * value;
    }
    let rms = (sum_squares / input.len() as f32 + epsilon).sqrt();
    for index in 0..input.len() {
        output[index] = input[index] / rms * weight[index];
    }
    Ok(())
}

pub fn add_reference(a: &[f32], b: &[f32], output: &mut [f32]) -> Result<(), ComputeError> {
    validate_elementwise(a, b, output)?;
    for index in 0..a.len() {
        output[index] = a[index] + b[index];
    }
    Ok(())
}

pub fn mul_reference(a: &[f32], b: &[f32], output: &mut [f32]) -> Result<(), ComputeError> {
    validate_elementwise(a, b, output)?;
    for index in 0..a.len() {
        output[index] = a[index] * b[index];
    }
    Ok(())
}

pub fn silu_reference(input: &[f32], output: &mut [f32]) -> Result<(), ComputeError> {
    if input.len() != output.len() {
        return Err(ComputeError(format!(
            "SiLU shape mismatch: input {}, output {}",
            input.len(),
            output.len()
        )));
    }
    for index in 0..input.len() {
        let value = input[index];
        output[index] = value / (1.0 + (-value).exp());
    }
    Ok(())
}

/// SwiGLU reference: `SiLU(gate) * up`, with both inputs and output
/// materialized in the caller's buffers.
pub fn swiglu_reference(
    gate: &[f32],
    up: &[f32],
    output: &mut [f32],
) -> Result<(), ComputeError> {
    validate_elementwise(gate, up, output)?;
    for index in 0..gate.len() {
        let value = gate[index];
        output[index] = value / (1.0 + (-value).exp()) * up[index];
    }
    Ok(())
}

pub fn softmax_reference(values: &mut [f32]) -> Result<(), ComputeError> {
    if values.is_empty() {
        return Err(ComputeError("softmax requires a non-empty vector".to_string()));
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if !sum.is_finite() || sum == 0.0 {
        return Err(ComputeError("softmax normalization is not finite".to_string()));
    }
    for value in values.iter_mut() {
        *value /= sum;
    }
    Ok(())
}

/// Half-split llama/Qwen RoPE. The first `head_dim / 2` coordinates pair with
/// the second half of the same head. Q/K are independently laid out as
/// `[heads, head_dim]`.
pub fn rope_reference(
    q: &mut [f32],
    k: &mut [f32],
    position: usize,
    head_dim: usize,
    query_heads: usize,
    kv_heads: usize,
    theta: f32,
) -> Result<(), ComputeError> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(ComputeError("RoPE head dimension must be non-zero and even".to_string()));
    }
    if query_heads == 0 || kv_heads == 0 {
        return Err(ComputeError("RoPE requires non-zero query and KV head counts".to_string()));
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(ComputeError("RoPE frequency base must be finite and positive".to_string()));
    }
    let q_width = query_heads
        .checked_mul(head_dim)
        .ok_or_else(|| ComputeError("RoPE query width overflow".to_string()))?;
    let k_width = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| ComputeError("RoPE KV width overflow".to_string()))?;
    if q.len() != q_width || k.len() != k_width {
        return Err(ComputeError(format!(
            "RoPE shape mismatch: q {}, k {}, heads {}:{}, head_dim {}",
            q.len(),
            k.len(),
            query_heads,
            kv_heads,
            head_dim
        )));
    }
    for head in 0..query_heads {
        rope_head(&mut q[head * head_dim..(head + 1) * head_dim], position, theta);
    }
    for head in 0..kv_heads {
        rope_head(&mut k[head * head_dim..(head + 1) * head_dim], position, theta);
    }
    Ok(())
}

fn rope_head(values: &mut [f32], position: usize, theta: f32) {
    let half = values.len() / 2;
    for index in 0..half {
        let angle = theta.powf(-2.0 * index as f32 / values.len() as f32) * position as f32;
        let (cosine, sine) = (angle.cos(), angle.sin());
        let first = values[index];
        let second = values[index + half];
        values[index] = first * cosine - second * sine;
        values[index + half] = first * sine + second * cosine;
    }
}

/// Single-token causal attention reference with explicit GQA mapping. The
/// cache slices contain exactly `history_len` positions; the current K/V is
/// supplied separately and is read at position `history_len`.
pub fn attention_reference(
    q: &[f32],
    k_history: &[f32],
    v_history: &[f32],
    k_current: &[f32],
    v_current: &[f32],
    config: AttentionConfig,
) -> Result<Vec<f32>, ComputeError> {
    let AttentionConfig {
        history_len,
        query_heads,
        kv_heads,
        head_dim,
    } = config;
    if query_heads == 0 || kv_heads == 0 || head_dim == 0 {
        return Err(ComputeError(
            "attention requires non-zero query heads, KV heads, and head dimension".to_string(),
        ));
    }
    if !query_heads.is_multiple_of(kv_heads) {
        return Err(ComputeError(format!(
            "GQA requires query_heads {} to be divisible by kv_heads {}",
            query_heads, kv_heads
        )));
    }
    let kv_width = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| ComputeError("attention KV width overflow".to_string()))?;
    let q_width = query_heads
        .checked_mul(head_dim)
        .ok_or_else(|| ComputeError("attention query width overflow".to_string()))?;
    let history_width = history_len
        .checked_mul(kv_width)
        .ok_or_else(|| ComputeError("attention history size overflow".to_string()))?;
    if q.len() != q_width
        || k_history.len() != history_width
        || v_history.len() != history_width
        || k_current.len() != kv_width
        || v_current.len() != kv_width
    {
        return Err(ComputeError("attention tensor shape mismatch".to_string()));
    }
    let mut output = vec![0.0f32; q_width];
    let total_positions = history_len
        .checked_add(1)
        .ok_or_else(|| ComputeError("attention position count overflow".to_string()))?;
    for query_head in 0..query_heads {
        let kv_head = query_head * kv_heads / query_heads;
        let q_head = &q[query_head * head_dim..(query_head + 1) * head_dim];
        let mut scores = vec![0.0f32; total_positions];
        for position in 0..total_positions {
            let k_all = if position < history_len {
                &k_history[position * kv_width..(position + 1) * kv_width]
            } else {
                k_current
            };
            let k_head = &k_all[kv_head * head_dim..(kv_head + 1) * head_dim];
            let mut dot = 0.0f32;
            for index in 0..head_dim {
                dot += q_head[index] * k_head[index];
            }
            scores[position] = dot / (head_dim as f32).sqrt();
        }
        softmax_reference(&mut scores)?;
        let output_head = &mut output[query_head * head_dim..(query_head + 1) * head_dim];
        for position in 0..total_positions {
            let v_all = if position < history_len {
                &v_history[position * kv_width..(position + 1) * kv_width]
            } else {
                v_current
            };
            let v_head = &v_all[kv_head * head_dim..(kv_head + 1) * head_dim];
            for index in 0..head_dim {
                output_head[index] += scores[position] * v_head[index];
            }
        }
    }
    Ok(output)
}

fn validate_elementwise(a: &[f32], b: &[f32], output: &[f32]) -> Result<(), ComputeError> {
    if a.len() != b.len() || a.len() != output.len() {
        return Err(ComputeError(format!(
            "elementwise shape mismatch: a {}, b {}, output {}",
            a.len(),
            b.len(),
            output.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant;

    #[test]
    fn f32_matvec_uses_explicit_input_output_shape() {
        let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = [0.0; 2];
        matvec_f32_reference(&weights, MatrixShape::new(3, 2), &[1.0, 1.0, 1.0], &mut output)
            .unwrap();
        assert_eq!(output, [6.0, 15.0]);
    }

    #[test]
    fn row_range_is_relative_to_local_output_slice() {
        let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = [0.0; 1];
        matvec_f32_reference_row_range(
            &weights,
            MatrixShape::new(2, 3),
            &[1.0, 1.0],
            &mut output,
            2,
            1,
        )
        .unwrap();
        assert_eq!(output, [11.0]);
    }

    #[test]
    fn rms_norm_reference_has_explicit_f32_reduction() {
        let mut output = [0.0; 2];
        rms_norm_reference(&[3.0, 4.0], &[1.0, 1.0], 0.0, &mut output).unwrap();
        assert!((output[0] - 0.84852814).abs() < 1e-6);
        assert!((output[1] - 1.1313709).abs() < 1e-6);
    }

    #[test]
    fn swiglu_reference_matches_silu_times_up() {
        let mut output = [0.0f32; 3];
        swiglu_reference(&[-1.0, 0.0, 2.0], &[2.0, 3.0, 4.0], &mut output).unwrap();
        assert!((output[0] - (-1.0 / (1.0 + 1.0f32.exp()) * 2.0)).abs() < 1e-6);
        assert_eq!(output[1], 0.0);
        assert!((output[2] - (2.0 / (1.0 + (-2.0f32).exp()) * 4.0)).abs() < 1e-6);
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let mut q = [1.0, 2.0, 3.0, 4.0];
        let mut k = q;
        rope_reference(&mut q, &mut k, 0, 4, 1, 1, 10_000.0).unwrap();
        assert_eq!(q, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(k, q);
    }

    #[test]
    fn rope_half_split_nonzero_position_has_expected_rotation() {
        let mut q = [1.0, 0.0];
        let mut k = [1.0, 0.0];
        rope_reference(&mut q, &mut k, 1, 2, 1, 1, 10_000.0).unwrap();
        assert!((q[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((q[1] - 1.0f32.sin()).abs() < 1e-6);
        assert_eq!(q, k);
    }

    #[test]
    fn gqa_attention_maps_query_heads_to_kv_heads() {
        let output = attention_reference(
            &[1.0, 0.0, 0.0, 1.0],
            &[],
            &[],
            &[1.0, 0.0],
            &[2.0, 3.0],
            AttentionConfig::new(0, 2, 1, 2),
        )
        .unwrap();
        assert_eq!(output.len(), 4);
        assert_eq!(output, [2.0, 3.0, 2.0, 3.0]);
    }

    #[test]
    fn quantized_reference_matches_q4_0_and_q6_k_fused_rows() {
        let q4_shape = MatrixShape::new(32, 3);
        let mut q4_raw = vec![0u8; 3 * quant::BLOCK_SIZE_Q4_0];
        for row in 0..3 {
            let offset = row * quant::BLOCK_SIZE_Q4_0;
            q4_raw[offset..offset + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
            for byte in &mut q4_raw[offset + 2..offset + quant::BLOCK_SIZE_Q4_0] {
                *byte = 0x11u8.wrapping_add(row as u8);
            }
        }
        let q4_x = (0..32).map(|index| index as f32 * 0.03125).collect::<Vec<_>>();
        let mut q4_reference = vec![0.0f32; 3];
        let mut q4_optimized = vec![0.0f32; 3];
        quantized_matvec_reference(
            GgmlType::Q4_0,
            &q4_raw,
            q4_shape,
            &q4_x,
            &mut q4_reference,
        )
        .unwrap();
        quant::matvec_q4_0(&q4_raw, &[3, 32], &q4_x, &mut q4_optimized).unwrap();
        for (reference, optimized) in q4_reference.iter().zip(q4_optimized.iter()) {
            assert!((reference - optimized).abs() < 1e-3);
        }

        let q6_shape = MatrixShape::new(256, 2);
        let mut q6_raw = vec![0u8; 2 * quant::BLOCK_SIZE_Q6_K];
        for row in 0..2 {
            let offset = row * quant::BLOCK_SIZE_Q6_K;
            for index in 0..128 {
                q6_raw[offset + index] = (index as u8).wrapping_add(row as u8);
            }
            for index in 0..64 {
                q6_raw[offset + 128 + index] = (index as u8).wrapping_mul(3);
            }
            for index in 0..16 {
                q6_raw[offset + 192 + index] = 1;
            }
            q6_raw[offset + 208..offset + 210].copy_from_slice(&0x3c00u16.to_le_bytes());
        }
        let q6_x = (0..256)
            .map(|index| (index as f32 * 0.0078125).sin())
            .collect::<Vec<_>>();
        let mut q6_reference = vec![0.0f32; 2];
        let mut q6_optimized = vec![0.0f32; 2];
        quantized_matvec_reference(
            GgmlType::Q6_K,
            &q6_raw,
            q6_shape,
            &q6_x,
            &mut q6_reference,
        )
        .unwrap();
        quant::matvec_q6_k(&q6_raw, &[2, 256], &q6_x, &mut q6_optimized).unwrap();
        for (reference, optimized) in q6_reference.iter().zip(q6_optimized.iter()) {
            assert!((reference - optimized).abs() < 1e-3);
        }
    }

    #[test]
    fn quantized_reference_covers_every_materialized_format() {
        type Kernel = fn(
            &[u8],
            &[usize],
            &[f32],
            &mut [f32],
        ) -> Result<(), crate::error::DataSourceError>;

        fn compare(
            ggml_type: GgmlType,
            input: usize,
            block: Vec<u8>,
            kernel: Kernel,
        ) {
            let output = 2usize;
            let raw = block.repeat(output);
            let x = (0..input)
                .map(|index| (index as f32 * 0.013).cos())
                .collect::<Vec<_>>();
            let mut reference = vec![0.0f32; output];
            let mut optimized = vec![0.0f32; output];
            quantized_matvec_reference(
                ggml_type,
                &raw,
                MatrixShape::new(input, output),
                &x,
                &mut reference,
            )
            .unwrap();
            kernel(&raw, &[output, input], &x, &mut optimized).unwrap();
            for (expected, actual) in reference.iter().zip(optimized.iter()) {
                let tolerance = 1.0e-2 + 1.0e-5 * expected.abs().max(actual.abs());
                assert!(
                    (expected - actual).abs() <= tolerance,
                    "{} reference {} != optimized {} (tol {})",
                    ggml_type.name(),
                    expected,
                    actual,
                    tolerance
                );
            }
        }

        let mut q4_0 = vec![0x11u8; quant::BLOCK_SIZE_Q4_0];
        q4_0[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        compare(GgmlType::Q4_0, quant::QK4_0, q4_0, quant::matvec_q4_0);

        let mut q8_0 = vec![0x11u8; quant::BLOCK_SIZE_Q8_0];
        q8_0[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        compare(GgmlType::Q8_0, quant::QK8_0, q8_0, quant::matvec_q8_0);

        let mut q2_k = vec![0x11u8; quant::BLOCK_SIZE_Q2_K];
        q2_k[80..82].copy_from_slice(&0x3c00u16.to_le_bytes());
        q2_k[82..84].copy_from_slice(&0u16.to_le_bytes());
        compare(GgmlType::Q2_K, quant::QK_K, q2_k, quant::matvec_q2_k);

        let mut q3_k = vec![0x11u8; quant::BLOCK_SIZE_Q3_K];
        q3_k[108..110].copy_from_slice(&0x3c00u16.to_le_bytes());
        compare(GgmlType::Q3_K, quant::QK_K, q3_k, quant::matvec_q3_k);

        let mut q4_k = vec![0x11u8; quant::BLOCK_SIZE_Q4_K];
        q4_k[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        q4_k[2..4].copy_from_slice(&0u16.to_le_bytes());
        compare(GgmlType::Q4_K, quant::QK_K, q4_k, quant::matvec_q4_k);

        let mut q5_k = vec![0x11u8; quant::BLOCK_SIZE_Q5_K];
        q5_k[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        q5_k[2..4].copy_from_slice(&0u16.to_le_bytes());
        compare(GgmlType::Q5_K, quant::QK_K, q5_k, quant::matvec_q5_k);

        let mut q6_k = vec![0x11u8; quant::BLOCK_SIZE_Q6_K];
        q6_k[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());
        compare(GgmlType::Q6_K, quant::QK_K, q6_k, quant::matvec_q6_k);

        let mut q8_k = vec![0x11u8; quant::BLOCK_SIZE_Q8_K];
        q8_k[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        compare(GgmlType::Q8_K, quant::QK_K, q8_k, quant::matvec_q8_k);
    }
}
