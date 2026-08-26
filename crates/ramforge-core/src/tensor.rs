//! Tensor representation – supports F32/F16/BF16 and quantized
//! Q4_0/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K.
//!
//! Quantized tensors remain quantized while resident; dequantization happens
//! block-wise inside the matvec kernels (working set bounded to one block).
//!
//! ## Matrix layout convention (GGML/GGUF)
//!
//! GGUF stores tensor dimensions in ggml `ne[]` order. For the 2D weight
//! matrices used by llama/qwen2 models this means:
//!
//! - `shape = [in_features, out_features]` (i.e. `ne[0] = in`, `ne[1] = out`)
//! - the raw buffer holds `out` contiguous rows of `in` elements
//!   (row-major `[out][in]`)
//! - matvec computes `y[o] = sum_i W[o * in + i] * x[i]`
//!
//! All `matvec`/`dequantize_row`/`get_embedding` implementations in this
//! module follow this single convention and return hard errors on arity
//! mismatches instead of guessing orientation.

#![allow(clippy::needless_range_loop, clippy::manual_is_multiple_of)]

use crate::error::DataSourceError;
use crate::types::GgmlType;
use crate::quant;

#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub ggml_type: GgmlType,
    pub shape: Vec<usize>,
    pub num_elements: usize,
    pub raw_data: Vec<u8>,
}

impl QuantizedTensor {
    pub fn new(
        ggml_type: GgmlType,
        shape: Vec<usize>,
        num_elements: usize,
        raw_data: Vec<u8>,
    ) -> Result<Self, DataSourceError> {
        // Validate size
        let expected_bytes = match ggml_type {
            GgmlType::Q4_0 => {
                if num_elements % quant::QK4_0 != 0 {
                    return Err(DataSourceError::General(format!(
                        "Q4_0 elements {} not divisible by {}",
                        num_elements,
                        quant::QK4_0
                    )));
                }
                (num_elements / quant::QK4_0) * quant::BLOCK_SIZE_Q4_0
            }
            GgmlType::Q8_0 => {
                if num_elements % quant::QK8_0 != 0 {
                    return Err(DataSourceError::General(format!(
                        "Q8_0 elements {} not divisible by {}",
                        num_elements,
                        quant::QK8_0
                    )));
                }
                (num_elements / quant::QK8_0) * quant::BLOCK_SIZE_Q8_0
            }
            GgmlType::Q4_K => {
                if num_elements % quant::QK_K != 0 {
                    return Err(DataSourceError::General(format!(
                        "Q4_K elements {} not divisible by {}",
                        num_elements,
                        quant::QK_K
                    )));
                }
                (num_elements / quant::QK_K) * quant::BLOCK_SIZE_Q4_K
            }
            GgmlType::Q5_K => {
                if num_elements % quant::QK_K != 0 {
                    return Err(DataSourceError::General(format!(
                        "Q5_K elements {} not divisible by {}",
                        num_elements,
                        quant::QK_K
                    )));
                }
                (num_elements / quant::QK_K) * quant::BLOCK_SIZE_Q5_K
            }
            GgmlType::Q6_K => {
                if num_elements % quant::QK_K != 0 {
                    return Err(DataSourceError::General(format!(
                        "Q6_K elements {} not divisible by {}",
                        num_elements,
                        quant::QK_K
                    )));
                }
                (num_elements / quant::QK_K) * quant::BLOCK_SIZE_Q6_K
            }
            GgmlType::Q2_K => {
                if num_elements % quant::QK_K != 0 {
                    return Err(DataSourceError::General(format!(
                        "Q2_K elements {} not divisible by {}",
                        num_elements,
                        quant::QK_K
                    )));
                }
                (num_elements / quant::QK_K) * quant::BLOCK_SIZE_Q2_K
            }
            GgmlType::Q3_K => {
                if num_elements % quant::QK_K != 0 {
                    return Err(DataSourceError::General(format!(
                        "Q3_K elements {} not divisible by {}",
                        num_elements,
                        quant::QK_K
                    )));
                }
                (num_elements / quant::QK_K) * quant::BLOCK_SIZE_Q3_K
            }
            GgmlType::Q8_K => {
                if num_elements % quant::QK_K != 0 {
                    return Err(DataSourceError::General(format!(
                        "Q8_K elements {} not divisible by {}",
                        num_elements,
                        quant::QK_K
                    )));
                }
                (num_elements / quant::QK_K) * quant::BLOCK_SIZE_Q8_K
            }
            _ => {
                return Err(DataSourceError::General(format!(
                    "not a quantized type: {}",
                    ggml_type.name()
                )))
            }
        };

        if raw_data.len() < expected_bytes {
            return Err(DataSourceError::General(format!(
                "quantized tensor truncated: expected {} bytes, got {} for type {}",
                expected_bytes,
                raw_data.len(),
                ggml_type.name()
            )));
        }

        Ok(Self {
            ggml_type,
            shape,
            num_elements,
            raw_data,
        })
    }

    pub fn resident_bytes(&self) -> usize {
        self.raw_data.len()
    }

    /// Dequantize entire tensor to F32 (reference, not for permanent residency)
    pub fn dequantize_to_f32(&self) -> Result<Vec<f32>, DataSourceError> {
        let mut out = vec![0.0f32; self.num_elements];
        match self.ggml_type {
            GgmlType::Q4_0 => quant::dequantize_row_q4_0(&self.raw_data, self.num_elements, &mut out)?,
            GgmlType::Q8_0 => quant::dequantize_row_q8_0(&self.raw_data, self.num_elements, &mut out)?,
            GgmlType::Q4_K => quant::dequantize_row_q4_k(&self.raw_data, self.num_elements, &mut out)?,
            GgmlType::Q5_K => quant::dequantize_row_q5_k(&self.raw_data, self.num_elements, &mut out)?,
            GgmlType::Q6_K => quant::dequantize_row_q6_k(&self.raw_data, self.num_elements, &mut out)?,
            GgmlType::Q2_K => quant::dequantize_row_q2_k(&self.raw_data, self.num_elements, &mut out)?,
            GgmlType::Q3_K => quant::dequantize_row_q3_k(&self.raw_data, self.num_elements, &mut out)?,
            GgmlType::Q8_K => quant::dequantize_row_q8_k(&self.raw_data, self.num_elements, &mut out)?,
            _ => {
                return Err(DataSourceError::General(format!(
                    "unsupported quantized type for dequant: {}",
                    self.ggml_type.name()
                )))
            }
        }
        Ok(out)
    }

    /// Dequantize a single row (for embedding lookup or row-wise matvec).
    ///
    /// Explicit GGML/GGUF convention (see `TensorData` docs): a 2D tensor has
    /// `shape = [in_features, out_features]` and the raw buffer stores `out`
    /// contiguous rows of `in` quantized elements each. Row `r` therefore
    /// occupies bytes `[r * row_bytes, (r + 1) * row_bytes)` where
    /// `row_bytes = (shape[0] / QK) * BLOCK_SIZE`.
    pub fn dequantize_row(&self, row_idx: usize) -> Result<Vec<f32>, DataSourceError> {
        if self.shape.is_empty() {
            return Err(DataSourceError::General("cannot get row from scalar".to_string()));
        }

        // For 1D tensor, row_idx must be 0
        if self.shape.len() == 1 {
            if row_idx != 0 {
                return Err(DataSourceError::General(format!(
                    "row_idx {} out of bounds for 1D tensor",
                    row_idx
                )));
            }
            return self.dequantize_to_f32();
        }

        if self.shape.len() != 2 {
            return Err(DataSourceError::General(format!(
                "dequantize_row supports 1D/2D tensors only, got shape {:?}",
                self.shape
            )));
        }

        // Explicit layout: row length = shape[0] (in), row count = shape[1] (out)
        let row_elements = self.shape[0];
        let num_rows = self.shape[1];
        if row_idx >= num_rows {
            return Err(DataSourceError::General(format!(
                "row_idx {} out of bounds: shape {:?} has {} rows (ggml layout [in={}, out={}])",
                row_idx, self.shape, num_rows, row_elements, num_rows
            )));
        }

        let (qk, block_size) = match self.ggml_type {
            GgmlType::Q4_0 => (quant::QK4_0, quant::BLOCK_SIZE_Q4_0),
            GgmlType::Q8_0 => (quant::QK8_0, quant::BLOCK_SIZE_Q8_0),
            GgmlType::Q4_K => (quant::QK_K, quant::BLOCK_SIZE_Q4_K),
            GgmlType::Q5_K => (quant::QK_K, quant::BLOCK_SIZE_Q5_K),
            GgmlType::Q6_K => (quant::QK_K, quant::BLOCK_SIZE_Q6_K),
            GgmlType::Q2_K => (quant::QK_K, quant::BLOCK_SIZE_Q2_K),
            GgmlType::Q3_K => (quant::QK_K, quant::BLOCK_SIZE_Q3_K),
            GgmlType::Q8_K => (quant::QK_K, quant::BLOCK_SIZE_Q8_K),
            _ => {
                return Err(DataSourceError::General(format!(
                    "dequantize_row not supported for type {}",
                    self.ggml_type.name()
                )))
            }
        };

        if row_elements % qk != 0 {
            return Err(DataSourceError::General(format!(
                "row_elements {} not divisible by block size {} for type {}",
                row_elements, qk, self.ggml_type.name()
            )));
        }

        let row_bytes = (row_elements / qk) * block_size;
        let row_start = row_idx * row_bytes;

        if row_start + row_bytes > self.raw_data.len() {
            return Err(DataSourceError::General(format!(
                "row {} out of bounds for quantized tensor: row_start {} + row_bytes {} > raw_len {}",
                row_idx,
                row_start,
                row_bytes,
                self.raw_data.len()
            )));
        }

        let row_slice = &self.raw_data[row_start..row_start + row_bytes];
        let mut out = vec![0.0f32; row_elements];

        match self.ggml_type {
            GgmlType::Q4_0 => quant::dequantize_row_q4_0(row_slice, row_elements, &mut out)?,
            GgmlType::Q8_0 => quant::dequantize_row_q8_0(row_slice, row_elements, &mut out)?,
            GgmlType::Q4_K => quant::dequantize_row_q4_k(row_slice, row_elements, &mut out)?,
            GgmlType::Q5_K => quant::dequantize_row_q5_k(row_slice, row_elements, &mut out)?,
            GgmlType::Q6_K => quant::dequantize_row_q6_k(row_slice, row_elements, &mut out)?,
            GgmlType::Q2_K => quant::dequantize_row_q2_k(row_slice, row_elements, &mut out)?,
            GgmlType::Q3_K => quant::dequantize_row_q3_k(row_slice, row_elements, &mut out)?,
            GgmlType::Q8_K => quant::dequantize_row_q8_k(row_slice, row_elements, &mut out)?,
            _ => unreachable!(),
        }

        Ok(out)
    }

    /// Quantized matvec: `y = W * x` with W kept in compact quantized form.
    ///
    /// Explicit GGML/GGUF convention: `shape = [in_features, out_features]` and
    /// `raw_data` stores `out_features` contiguous rows of `in_features`
    /// quantized elements. The kernels dequantize block-by-block; the working
    /// set never exceeds one block — the entire tensor is never expanded to
    /// F32. Any arity/layout mismatch is a hard error (no silent fallbacks).
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) -> Result<(), DataSourceError> {
        if self.shape.len() != 2 {
            return Err(DataSourceError::General(format!(
                "quantized matvec expects 2D weight, got shape {:?}",
                self.shape
            )));
        }

        let in_dim = self.shape[0];
        let out_dim = self.shape[1];
        if x.len() != in_dim || y.len() != out_dim {
            return Err(DataSourceError::General(format!(
                "quantized matvec arity mismatch (ggml layout [in, out]): W {:?} implies in={}, out={}, but got x.len()={}, y.len()={}",
                self.shape,
                in_dim,
                out_dim,
                x.len(),
                y.len()
            )));
        }

        match self.ggml_type {
            GgmlType::Q4_0 => quant::matvec_q4_0(&self.raw_data, &[out_dim, in_dim], x, y),
            GgmlType::Q8_0 => quant::matvec_q8_0(&self.raw_data, &[out_dim, in_dim], x, y),
            GgmlType::Q4_K => quant::matvec_q4_k(&self.raw_data, &[out_dim, in_dim], x, y),
            GgmlType::Q5_K => quant::matvec_q5_k(&self.raw_data, &[out_dim, in_dim], x, y),
            GgmlType::Q6_K => quant::matvec_q6_k(&self.raw_data, &[out_dim, in_dim], x, y),
            GgmlType::Q2_K => quant::matvec_q2_k(&self.raw_data, &[out_dim, in_dim], x, y),
            GgmlType::Q3_K => quant::matvec_q3_k(&self.raw_data, &[out_dim, in_dim], x, y),
            GgmlType::Q8_K => quant::matvec_q8_k(&self.raw_data, &[out_dim, in_dim], x, y),
            _ => Err(DataSourceError::General(format!(
                "unsupported quantized matvec type {}",
                self.ggml_type.name()
            ))),
        }
    }
}

#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub enum TensorData {
    F32 {
        data: Vec<f32>,
        shape: Vec<usize>,
    },
    F16 {
        data: Vec<f32>,
        shape: Vec<usize>,
    },
    BF16 {
        data: Vec<f32>,
        shape: Vec<usize>,
    },
    Q4_0(QuantizedTensor),
    Q8_0(QuantizedTensor),
    Q4_K(QuantizedTensor),
    Q5_K(QuantizedTensor),
    Q6_K(QuantizedTensor),
    Q2_K(QuantizedTensor),
    Q3_K(QuantizedTensor),
    Q8_K(QuantizedTensor),
}

impl TensorData {
    /// Resident bytes for the representation produced by `from_bytes`.
    ///
    /// Float tensors are retained decoded as `Vec<f32>`, including F16/BF16.
    /// Supported quantized tensors retain the exact compact file buffer. This
    /// descriptor-level calculation lets callers make residency decisions and
    /// establish a load charge before reading the tensor; a successful decode
    /// must agree with `resident_bytes()`.
    pub fn resident_bytes_for(
        ggml_type: GgmlType,
        num_elements: u64,
        file_bytes: u64,
    ) -> Result<u64, DataSourceError> {
        match ggml_type {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => num_elements
                .checked_mul(std::mem::size_of::<f32>() as u64)
                .ok_or_else(|| {
                    DataSourceError::General(format!(
                        "resident size overflow for {} tensor with {} elements",
                        ggml_type.name(),
                        num_elements
                    ))
                }),
            GgmlType::Q4_0
            | GgmlType::Q8_0
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q8_K => Ok(file_bytes),
            _ => Err(DataSourceError::General(format!(
                "unsupported tensor type for inference: {} (supported: F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, Q8_K)",
                ggml_type.name()
            ))),
        }
    }

    /// Construct F32 tensor storage from an already initialized final buffer.
    /// This takes ownership without decoding or copying and is the endpoint of
    /// the direct F32 datasource path.
    pub fn from_f32_vec(
        shape: Vec<u64>,
        num_elements: u64,
        data: Vec<f32>,
    ) -> Result<Self, DataSourceError> {
        let expected_len = usize::try_from(num_elements).map_err(|_| {
            DataSourceError::General(format!(
                "F32 element count {} does not fit this platform",
                num_elements
            ))
        })?;
        if data.len() != expected_len {
            return Err(DataSourceError::General(format!(
                "F32 data length mismatch: expected {} elements, got {}",
                expected_len,
                data.len()
            )));
        }
        let shape = shape
            .into_iter()
            .map(|dimension| {
                usize::try_from(dimension).map_err(|_| {
                    DataSourceError::General(format!(
                        "F32 tensor dimension {} does not fit this platform",
                        dimension
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::F32 { data, shape })
    }

    pub fn from_bytes(
        ggml_type: GgmlType,
        shape: Vec<u64>,
        num_elements: u64,
        raw_bytes: Vec<u8>,
    ) -> Result<Self, DataSourceError> {
        let shape_usize: Vec<usize> = shape.iter().map(|&v| v as usize).collect();

        match ggml_type {
            GgmlType::F32 => {
                let data = decode_f32(&raw_bytes, num_elements)?;
                Ok(Self::F32 {
                    data,
                    shape: shape_usize,
                })
            }
            GgmlType::F16 => {
                let data = decode_f16(&raw_bytes, num_elements)?;
                Ok(Self::F16 {
                    data,
                    shape: shape_usize,
                })
            }
            GgmlType::BF16 => {
                let data = decode_bf16(&raw_bytes, num_elements)?;
                Ok(Self::BF16 {
                    data,
                    shape: shape_usize,
                })
            }
            GgmlType::Q4_0 => {
                let qt = QuantizedTensor::new(ggml_type, shape_usize, num_elements as usize, raw_bytes)?;
                Ok(Self::Q4_0(qt))
            }
            GgmlType::Q8_0 => {
                let qt = QuantizedTensor::new(ggml_type, shape_usize, num_elements as usize, raw_bytes)?;
                Ok(Self::Q8_0(qt))
            }
            GgmlType::Q4_K => {
                let qt = QuantizedTensor::new(ggml_type, shape_usize, num_elements as usize, raw_bytes)?;
                Ok(Self::Q4_K(qt))
            }
            GgmlType::Q5_K => {
                let qt = QuantizedTensor::new(ggml_type, shape_usize, num_elements as usize, raw_bytes)?;
                Ok(Self::Q5_K(qt))
            }
            GgmlType::Q6_K => {
                let qt = QuantizedTensor::new(ggml_type, shape_usize, num_elements as usize, raw_bytes)?;
                Ok(Self::Q6_K(qt))
            }
            GgmlType::Q2_K => {
                let qt = QuantizedTensor::new(ggml_type, shape_usize, num_elements as usize, raw_bytes)?;
                Ok(Self::Q2_K(qt))
            }
            GgmlType::Q3_K => {
                let qt = QuantizedTensor::new(ggml_type, shape_usize, num_elements as usize, raw_bytes)?;
                Ok(Self::Q3_K(qt))
            }
            GgmlType::Q8_K => {
                let qt = QuantizedTensor::new(ggml_type, shape_usize, num_elements as usize, raw_bytes)?;
                Ok(Self::Q8_K(qt))
            }
            _ => Err(DataSourceError::General(format!(
                "unsupported tensor type for inference: {} (supported: F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, Q8_K)",
                ggml_type.name()
            ))),
        }
    }

    pub fn resident_bytes(&self) -> usize {
        match self {
            // Float variants are held *decoded* as Vec<f32> in RAM, so the
            // resident size is the f32 slice size regardless of the on-disk
            // representation (F16/BF16 files are half that size).
            Self::F32 { data, .. } => std::mem::size_of_val(data.as_slice()),
            Self::F16 { data, .. } => std::mem::size_of_val(data.as_slice()),
            Self::BF16 { data, .. } => std::mem::size_of_val(data.as_slice()),
            Self::Q4_0(qt) => qt.resident_bytes(),
            Self::Q8_0(qt) => qt.resident_bytes(),
            Self::Q4_K(qt) => qt.resident_bytes(),
            Self::Q5_K(qt) => qt.resident_bytes(),
            Self::Q6_K(qt) => qt.resident_bytes(),
            Self::Q2_K(qt) => qt.resident_bytes(),
            Self::Q3_K(qt) => qt.resident_bytes(),
            Self::Q8_K(qt) => qt.resident_bytes(),
        }
    }

    pub fn num_elements(&self) -> usize {
        match self {
            Self::F32 { data, .. } => data.len(),
            Self::F16 { data, .. } => data.len(),
            Self::BF16 { data, .. } => data.len(),
            Self::Q4_0(qt) => qt.num_elements,
            Self::Q8_0(qt) => qt.num_elements,
            Self::Q4_K(qt) => qt.num_elements,
            Self::Q5_K(qt) => qt.num_elements,
            Self::Q6_K(qt) => qt.num_elements,
            Self::Q2_K(qt) => qt.num_elements,
            Self::Q3_K(qt) => qt.num_elements,
            Self::Q8_K(qt) => qt.num_elements,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32 { shape, .. } => shape,
            Self::F16 { shape, .. } => shape,
            Self::BF16 { shape, .. } => shape,
            Self::Q4_0(qt) => &qt.shape,
            Self::Q8_0(qt) => &qt.shape,
            Self::Q4_K(qt) => &qt.shape,
            Self::Q5_K(qt) => &qt.shape,
            Self::Q6_K(qt) => &qt.shape,
            Self::Q2_K(qt) => &qt.shape,
            Self::Q3_K(qt) => &qt.shape,
            Self::Q8_K(qt) => &qt.shape,
        }
    }

    pub fn ggml_type(&self) -> GgmlType {
        match self {
            Self::F32 { .. } => GgmlType::F32,
            Self::F16 { .. } => GgmlType::F16,
            Self::BF16 { .. } => GgmlType::BF16,
            Self::Q4_0(qt) => qt.ggml_type,
            Self::Q8_0(qt) => qt.ggml_type,
            Self::Q4_K(qt) => qt.ggml_type,
            Self::Q5_K(qt) => qt.ggml_type,
            Self::Q6_K(qt) => qt.ggml_type,
            Self::Q2_K(qt) => qt.ggml_type,
            Self::Q3_K(qt) => qt.ggml_type,
            Self::Q8_K(qt) => qt.ggml_type,
        }
    }

    /// For float variants: the decoded element buffer plus the explicit ggml
    /// shape `[in, out]`. Lets the runtime route float matvecs through a
    /// `ComputeBackend` (SIMD/threads) without copying. Quantized variants
    /// return `None` (they keep their own compact block-wise path).
    pub fn as_f32_slice(&self) -> Option<(&[f32], &[usize])> {
        match self {
            Self::F32 { data, shape, .. } => Some((data, shape)),
            Self::F16 { data, shape, .. } => Some((data, shape)),
            Self::BF16 { data, shape, .. } => Some((data, shape)),
            _ => None,
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(
            self,
            Self::Q4_0(_)
                | Self::Q8_0(_)
                | Self::Q4_K(_)
                | Self::Q5_K(_)
                | Self::Q6_K(_)
                | Self::Q2_K(_)
                | Self::Q3_K(_)
                | Self::Q8_K(_)
        )
    }

    /// Matvec: `y = W * x`.
    ///
    /// Explicit GGML/GGUF convention for all variants: `shape = [in, out]`
    /// (ggml `ne[0], ne[1]`) and the element buffer is row-major `[out][in]`
    /// (`in` contiguous). Arity mismatches are hard errors — no orientation
    /// guessing, no silent transposed fallbacks.
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) -> Result<(), DataSourceError> {
        match self {
            Self::F32 { data, shape, .. } => matvec_f32_ggml(data, shape, x, y, "F32"),
            Self::F16 { data, shape, .. } | Self::BF16 { data, shape, .. } => {
                matvec_f32_ggml(data, shape, x, y, "F16/BF16")
            }
            Self::Q4_0(qt)
            | Self::Q8_0(qt)
            | Self::Q4_K(qt)
            | Self::Q5_K(qt)
            | Self::Q6_K(qt)
            | Self::Q2_K(qt)
            | Self::Q3_K(qt)
            | Self::Q8_K(qt) => qt.matvec(x, y),
        }
    }

    /// Get embedding for `token_id` (for `token_embd.weight`).
    /// Returns Vec<f32> of length `n_embd`.
    ///
    /// Explicit GGML convention: `token_embd` has `shape = [n_embd, vocab]`
    /// and embedding row `token_id` is contiguous (row length == `shape[0]`).
    pub fn get_embedding(&self, token_id: usize, n_embd: usize) -> Result<Vec<f32>, DataSourceError> {
        // Convention check for 2D tensors: the embedding row length must be
        // the contiguous dimension `shape[0]`.
        let shape = self.shape();
        if shape.len() == 2 && shape[0] != n_embd {
            return Err(DataSourceError::General(format!(
                "get_embedding expects shape {:?} with n_embd == shape[0] (ggml layout [n_embd, vocab]), got n_embd={}",
                shape, n_embd
            )));
        }
        match self {
            Self::F32 { data, .. } => {
                let offset = token_id * n_embd;
                if offset + n_embd <= data.len() {
                    Ok(data[offset..offset + n_embd].to_vec())
                } else {
                    Err(DataSourceError::General(format!(
                        "token_id {} out of bounds for F32 embedding len {} n_embd {}",
                        token_id,
                        data.len(),
                        n_embd
                    )))
                }
            }
            Self::F16 { data, .. } | Self::BF16 { data, .. } => {
                let offset = token_id * n_embd;
                if offset + n_embd <= data.len() {
                    Ok(data[offset..offset + n_embd].to_vec())
                } else {
                    Err(DataSourceError::General(format!(
                        "token_id {} out of bounds for F16 embedding",
                        token_id
                    )))
                }
            }
            Self::Q4_0(qt)
            | Self::Q8_0(qt)
            | Self::Q4_K(qt)
            | Self::Q5_K(qt)
            | Self::Q6_K(qt)
            | Self::Q2_K(qt)
            | Self::Q3_K(qt)
            | Self::Q8_K(qt) => qt.dequantize_row(token_id),
        }
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>, DataSourceError> {
        match self {
            Self::F32 { data, .. } => Ok(data.clone()),
            Self::F16 { data, .. } => Ok(data.clone()),
            Self::BF16 { data, .. } => Ok(data.clone()),
            Self::Q4_0(qt)
            | Self::Q8_0(qt)
            | Self::Q4_K(qt)
            | Self::Q5_K(qt)
            | Self::Q6_K(qt)
            | Self::Q2_K(qt)
            | Self::Q3_K(qt)
            | Self::Q8_K(qt) => qt.dequantize_to_f32(),
        }
    }

    /// Consume a temporary tensor and return its F32 representation without
    /// cloning already-decoded F32/F16/BF16 storage. Quantized tensors still
    /// dequantize once while their compact bytes remain live.
    pub fn into_f32_vec(self) -> Result<Vec<f32>, DataSourceError> {
        match self {
            Self::F32 { data, .. }
            | Self::F16 { data, .. }
            | Self::BF16 { data, .. } => Ok(data),
            Self::Q4_0(qt)
            | Self::Q8_0(qt)
            | Self::Q4_K(qt)
            | Self::Q5_K(qt)
            | Self::Q6_K(qt)
            | Self::Q2_K(qt)
            | Self::Q3_K(qt)
            | Self::Q8_K(qt) => qt.dequantize_to_f32(),
        }
    }
}

/// F32 matvec under the explicit GGML/GGUF layout:
/// `shape = [in, out]`, `data` is row-major `[out][in]` (`in` contiguous).
/// Strict arity: `x.len() == shape[0]`, `y.len() == shape[1]`.
fn matvec_f32_ggml(
    data: &[f32],
    shape: &[usize],
    x: &[f32],
    y: &mut [f32],
    label: &str,
) -> Result<(), DataSourceError> {
    if shape.len() != 2 {
        return Err(DataSourceError::General(format!(
            "{} matvec expects 2D weight, got shape {:?}",
            label, shape
        )));
    }
    let in_dim = shape[0];
    let out_dim = shape[1];
    if x.len() != in_dim || y.len() != out_dim {
        return Err(DataSourceError::General(format!(
            "{} matvec arity mismatch (ggml layout [in, out]): W {:?} implies in={}, out={}, but got x.len()={}, y.len()={}",
            label,
            shape,
            in_dim,
            out_dim,
            x.len(),
            y.len()
        )));
    }
    let needed = out_dim
        .checked_mul(in_dim)
        .ok_or_else(|| DataSourceError::General("matvec shape overflow".to_string()))?;
    if data.len() < needed {
        return Err(DataSourceError::General(format!(
            "{} tensor truncated: need {} f32 elements for {:?}, have {}",
            label,
            needed,
            shape,
            data.len()
        )));
    }
    for (o, yi) in y.iter_mut().enumerate() {
        let row = &data[o * in_dim..o * in_dim + in_dim];
        let mut sum = 0.0;
        for (i, &wi) in row.iter().enumerate() {
            sum += wi * x[i];
        }
        *yi = sum;
    }
    Ok(())
}

/// Decode one contiguous row of `num_elements` values of any supported type
/// into `out` without allocating. Used by budget-charged streamed reads
/// (embeddings, chunked output projection).
pub fn decode_row_to_f32(
    ggml_type: GgmlType,
    bytes: &[u8],
    num_elements: usize,
    out: &mut [f32],
) -> Result<(), DataSourceError> {
    if out.len() != num_elements {
        return Err(DataSourceError::General(format!(
            "decode_row_to_f32: out buffer {} != num_elements {}",
            out.len(),
            num_elements
        )));
    }
    match ggml_type {
        GgmlType::F32 => {
            let need = num_elements * 4;
            if bytes.len() < need {
                return Err(DataSourceError::General(format!(
                    "F32 row truncated: expected {} bytes, got {}",
                    need,
                    bytes.len()
                )));
            }
            for (i, chunk) in bytes[..need].as_chunks::<4>().0.iter().enumerate() {
                out[i] = f32::from_le_bytes(*chunk);
            }
            Ok(())
        }
        GgmlType::F16 | GgmlType::BF16 => {
            let need = num_elements * 2;
            if bytes.len() < need {
                return Err(DataSourceError::General(format!(
                    "F16/BF16 row truncated: expected {} bytes, got {}",
                    need,
                    bytes.len()
                )));
            }
            for (i, chunk) in bytes[..need].as_chunks::<2>().0.iter().enumerate() {
                let bits = u16::from_le_bytes(*chunk);
                out[i] = if ggml_type == GgmlType::F16 {
                    f16_to_f32(bits)
                } else {
                    bf16_to_f32(bits)
                };
            }
            Ok(())
        }
        GgmlType::Q4_0 => quant::dequantize_row_q4_0(bytes, num_elements, out),
        GgmlType::Q8_0 => quant::dequantize_row_q8_0(bytes, num_elements, out),
        GgmlType::Q4_K => quant::dequantize_row_q4_k(bytes, num_elements, out),
        GgmlType::Q5_K => quant::dequantize_row_q5_k(bytes, num_elements, out),
        GgmlType::Q6_K => quant::dequantize_row_q6_k(bytes, num_elements, out),
        GgmlType::Q2_K => quant::dequantize_row_q2_k(bytes, num_elements, out),
        GgmlType::Q3_K => quant::dequantize_row_q3_k(bytes, num_elements, out),
        GgmlType::Q8_K => quant::dequantize_row_q8_k(bytes, num_elements, out),
        _ => Err(DataSourceError::General(format!(
            "decode_row_to_f32: unsupported tensor type {}",
            ggml_type.name()
        ))),
    }
}

// ---------- Legacy helpers kept for backward compatibility ----------
pub fn decode_tensor_to_f32(
    bytes: &[u8],
    ggml_type: GgmlType,
    num_elements: u64,
) -> Result<Vec<f32>, DataSourceError> {
    // For F32/F16/BF16, decode directly
    // For quantized, we now support via quant module but still return full F32 for legacy path
    // However milestone 5 says do NOT force quantized through this path if it would require full expansion
    // We keep it for backward compatibility but it will fully dequantize quantized tensors (not ideal for memory, but allowed for tests)
    match ggml_type {
        GgmlType::F32 => decode_f32(bytes, num_elements),
        GgmlType::F16 => decode_f16(bytes, num_elements),
        GgmlType::BF16 => decode_bf16(bytes, num_elements),
        GgmlType::Q4_0 => {
            let mut out = vec![0.0f32; num_elements as usize];
            quant::dequantize_row_q4_0(bytes, num_elements as usize, &mut out)?;
            Ok(out)
        }
        GgmlType::Q8_0 => {
            let mut out = vec![0.0f32; num_elements as usize];
            quant::dequantize_row_q8_0(bytes, num_elements as usize, &mut out)?;
            Ok(out)
        }
        GgmlType::Q4_K => {
            let mut out = vec![0.0f32; num_elements as usize];
            quant::dequantize_row_q4_k(bytes, num_elements as usize, &mut out)?;
            Ok(out)
        }
        GgmlType::Q5_K => {
            let mut out = vec![0.0f32; num_elements as usize];
            quant::dequantize_row_q5_k(bytes, num_elements as usize, &mut out)?;
            Ok(out)
        }
        GgmlType::Q6_K => {
            let mut out = vec![0.0f32; num_elements as usize];
            quant::dequantize_row_q6_k(bytes, num_elements as usize, &mut out)?;
            Ok(out)
        }
        GgmlType::Q2_K => {
            let mut out = vec![0.0f32; num_elements as usize];
            quant::dequantize_row_q2_k(bytes, num_elements as usize, &mut out)?;
            Ok(out)
        }
        GgmlType::Q3_K => {
            let mut out = vec![0.0f32; num_elements as usize];
            quant::dequantize_row_q3_k(bytes, num_elements as usize, &mut out)?;
            Ok(out)
        }
        GgmlType::Q8_K => {
            let mut out = vec![0.0f32; num_elements as usize];
            quant::dequantize_row_q8_k(bytes, num_elements as usize, &mut out)?;
            Ok(out)
        }
        _ => Err(DataSourceError::General(format!(
            "unsupported tensor type for inference: {} (supported: F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, Q8_K)",
            ggml_type.name()
        ))),
    }
}

/// Legacy/reference decoder for callers that already own a raw byte buffer.
/// Inference tensor loading bypasses this loop via the datasource direct-F32
/// path; keeping it provides backward compatibility and a parity oracle.
fn decode_f32(bytes: &[u8], num_elements: u64) -> Result<Vec<f32>, DataSourceError> {
    let expected = num_elements as usize * 4;
    if bytes.len() < expected {
        return Err(DataSourceError::General(format!(
            "F32 tensor truncated: expected {} bytes, got {}",
            expected,
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(num_elements as usize);
    for chunk in bytes[..expected].chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().unwrap();
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

fn decode_f16(bytes: &[u8], num_elements: u64) -> Result<Vec<f32>, DataSourceError> {
    let expected = num_elements as usize * 2;
    if bytes.len() < expected {
        return Err(DataSourceError::General(format!(
            "F16 tensor truncated: expected {} bytes, got {}",
            expected,
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(num_elements as usize);
    for chunk in bytes[..expected].chunks_exact(2) {
        let arr: [u8; 2] = chunk.try_into().unwrap();
        let bits = u16::from_le_bytes(arr);
        out.push(f16_to_f32(bits));
    }
    Ok(out)
}

fn decode_bf16(bytes: &[u8], num_elements: u64) -> Result<Vec<f32>, DataSourceError> {
    let expected = num_elements as usize * 2;
    if bytes.len() < expected {
        return Err(DataSourceError::General(format!(
            "BF16 tensor truncated: expected {} bytes, got {}",
            expected,
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(num_elements as usize);
    for chunk in bytes[..expected].chunks_exact(2) {
        let arr: [u8; 2] = chunk.try_into().unwrap();
        let bits = u16::from_le_bytes(arr);
        out.push(bf16_to_f32(bits));
    }
    Ok(out)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut e = 0;
            let mut f = frac;
            while (f & 0x400) == 0 {
                f <<= 1;
                e += 1;
            }
            f &= 0x3FF;
            let exp = (127 - 15 - e) as u32;
            (sign << 31) | (exp << 23) | (f << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | (0xFF << 23) | (frac << 13)
    } else {
        let exp = exp + (127 - 15);
        (sign << 31) | (exp << 23) | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

fn bf16_to_f32(bits: u16) -> f32 {
    let bits_u32 = (bits as u32) << 16;
    f32::from_bits(bits_u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_decode() {
        let vals = [1.0f32, 2.0, 3.5];
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let decoded = decode_f32(&bytes, 3).unwrap();
        assert_eq!(decoded, vals);
    }

    #[test]
    fn test_f16_decode() {
        let bytes = [0x00, 0x3C, 0x00, 0x38];
        let decoded = decode_f16(&bytes, 2).unwrap();
        assert!((decoded[0] - 1.0).abs() < 1e-3);
        assert!((decoded[1] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_tensor_data_f32() {
        let vals = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut raw = Vec::new();
        for v in &vals {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let td = TensorData::from_bytes(
            crate::types::GgmlType::F32,
            vec![2, 2],
            4,
            raw,
        )
        .unwrap();
        assert_eq!(td.resident_bytes(), 16);
        assert!(!td.is_quantized());
    }

    #[test]
    fn test_tensor_data_q4_0() {
        // Q4_0: 32 elements, 18 bytes
        let d_fp16: u16 = 0x3C00;
        let mut raw = Vec::new();
        raw.extend_from_slice(&d_fp16.to_le_bytes());
        raw.extend_from_slice(&[0x88; 16]);
        let td = TensorData::from_bytes(
            crate::types::GgmlType::Q4_0,
            vec![32],
            32,
            raw,
        )
        .unwrap();
        assert!(td.is_quantized());
        assert_eq!(td.resident_bytes(), 18);
        // F32 equivalent would be 128 bytes, so quantized is smaller
        assert!(td.resident_bytes() < 128);
    }

    #[test]
    fn test_quantized_vs_f32_size() {
        // 256 elements F32 = 1024 bytes, Q4_K = 144 bytes
        let d_fp16: u16 = 0x3C00;
        let dmin_fp16: u16 = 0x0000;
        let mut raw = Vec::new();
        raw.extend_from_slice(&d_fp16.to_le_bytes());
        raw.extend_from_slice(&dmin_fp16.to_le_bytes());
        raw.extend_from_slice(&[1u8; 12]);
        raw.extend_from_slice(&[0x11; 128]);
        let td = TensorData::from_bytes(
            crate::types::GgmlType::Q4_K,
            vec![256],
            256,
            raw,
        )
        .unwrap();
        assert_eq!(td.resident_bytes(), 144);
        assert!(td.resident_bytes() < 1024, "quantized should be smaller than F32");
    }

    // ---------- M6: explicit GGML layout, non-square correctness ----------

    /// Independent reference dequant for one Q4_0 block (18 bytes → 32 f32).
    fn ref_dequant_q4_0_block(block: &[u8]) -> [f32; 32] {
        let d = {
            let bits = u16::from_le_bytes([block[0], block[1]]);
            // tiny fp16->f32 for the simple values used here (exp!=0)
            let sign = ((bits >> 15) & 1) as u32;
            let exp = ((bits >> 10) & 0x1F) as u32;
            let frac = (bits & 0x3FF) as u32;
            let f32_bits = ((sign) << 31) | ((exp + (127 - 15)) << 23) | (frac << 13);
            f32::from_bits(f32_bits)
        };
        let mut out = [0f32; 32];
        for j in 0..16 {
            let byte = block[2 + j];
            out[j] = d * ((byte & 0x0F) as i32 - 8) as f32;
            out[16 + j] = d * ((byte >> 4) as i32 - 8) as f32;
        }
        out
    }

    fn q4_0_block(d_fp16: u16, fill: u8) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&d_fp16.to_le_bytes());
        v.extend_from_slice(&[fill; 16]);
        v
    }

    #[test]
    fn test_matvec_f32_nonsquare_ggml_layout() {
        // ggml convention: shape [in, out]; buffer row-major [out][in].
        // in=2, out=3, rows: [1,2], [3,4], [5,6]
        let mut raw = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let td = TensorData::from_bytes(crate::types::GgmlType::F32, vec![2, 3], 6, raw).unwrap();
        let x = [10.0f32, 20.0];
        let mut y = [0.0f32; 3];
        td.matvec(&x, &mut y).unwrap();
        assert_eq!(y, [50.0, 110.0, 170.0]);
    }

    #[test]
    fn test_matvec_f32_nonsquare_reversed_dims() {
        // in=3, out=2, rows: [1,2,3], [4,5,6]
        let mut raw = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let td = TensorData::from_bytes(crate::types::GgmlType::F32, vec![3, 2], 6, raw).unwrap();
        let x = [1.0f32, 0.0, -1.0];
        let mut y = [0.0f32; 2];
        td.matvec(&x, &mut y).unwrap();
        assert_eq!(y, [1.0 - 3.0, 4.0 - 6.0]);
    }

    #[test]
    fn test_matvec_f16_nonsquare_ggml_layout() {
        // in=2, out=2 values 1..4 as f16
        let bits: [u16; 4] = [0x3C00, 0x4000, 0x4200, 0x4400]; // 1,2,3,4
        let mut raw = Vec::new();
        for b in bits {
            raw.extend_from_slice(&b.to_le_bytes());
        }
        let td = TensorData::from_bytes(crate::types::GgmlType::F16, vec![2, 2], 4, raw).unwrap();
        let x = [1.0f32, 1.0];
        let mut y = [0.0f32; 2];
        td.matvec(&x, &mut y).unwrap();
        assert!((y[0] - 3.0).abs() < 1e-3);
        assert!((y[1] - 7.0).abs() < 1e-3);
    }

    #[test]
    fn test_matvec_f32_arity_error_not_silently_guessed() {
        // in=2, out=3. Passing x of len 3 / y of len 2 must be a hard error,
        // never silently reinterpreted as the transposed orientation.
        let mut raw = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let td = TensorData::from_bytes(crate::types::GgmlType::F32, vec![2, 3], 6, raw).unwrap();
        let x = [1.0f32, 1.0, 1.0];
        let mut y = [0.0f32; 2];
        let err = td.matvec(&x, &mut y).unwrap_err();
        assert!(err.to_string().contains("arity mismatch"), "got: {}", err);
    }

    #[test]
    fn test_quant_matvec_q4_0_nonsquare_matches_reference() {
        // ggml dims [in=64, out=3]: 3 rows, each 2 Q4_0 blocks.
        // Row0 fill 0x11 (all -7), Row1 fill 0x23 (first16 -5, last16 -6),
        // Row2 fill 0xF8 (first16 +7, last16 0).
        let d_fp16: u16 = 0x3C00; // 1.0
        let mut raw = Vec::new();
        for _ in 0..2 {
            raw.extend_from_slice(&q4_0_block(d_fp16, 0x11));
        }
        for _ in 0..2 {
            raw.extend_from_slice(&q4_0_block(d_fp16, 0x23));
        }
        for _ in 0..2 {
            raw.extend_from_slice(&q4_0_block(d_fp16, 0xF8));
        }
        let td = TensorData::from_bytes(crate::types::GgmlType::Q4_0, vec![64, 3], 192, raw).unwrap();

        // x: first block region =1.0, second block region =2.0
        let mut x = [2.0f32; 64];
        for v in x.iter_mut().take(32) {
            *v = 1.0;
        }
        let mut y = [0.0f32; 3];
        td.matvec(&x, &mut y).unwrap();

        // Independent reference: dequant rows with the local helper, dot with x.
        let raw = match &td {
            TensorData::Q4_0(qt) => &qt.raw_data,
            _ => panic!("expected Q4_0"),
        };
        for r in 0..3 {
            let mut ref_row = [0.0f32; 64];
            for b in 0..2 {
                let blk = &raw[r * 36 + b * 18..r * 36 + b * 18 + 18];
                ref_row[b * 32..b * 32 + 32].copy_from_slice(&ref_dequant_q4_0_block(blk));
            }
            let expected: f32 = ref_row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            assert!(
                (y[r] - expected).abs() < 1e-3,
                "row {} kernel {} vs reference {}",
                r,
                y[r],
                expected
            );
        }
        // Absolute anchors: row0 = -7*32*1 + -7*32*2 = -672; row1 = -176 -352 = -528; row2 = 112+224 = 336
        assert!((y[0] - (-672.0)).abs() < 1e-3);
        assert!((y[1] - (-528.0)).abs() < 1e-3);
        assert!((y[2] - 336.0).abs() < 1e-3);
    }

    #[test]
    fn test_quant_matvec_q4_k_nonsquare_matches_reference() {
        // ggml dims [in=256, out=2]: Row0 qs=0x11 (all 1.0), Row1 qs=0x22 (all 2.0)
        let d_fp16: u16 = 0x3C00;
        let dmin_fp16: u16 = 0x0000;
        let mut raw = Vec::new();
        for fill in [0x11u8, 0x22u8] {
            raw.extend_from_slice(&d_fp16.to_le_bytes());
            raw.extend_from_slice(&dmin_fp16.to_le_bytes());
            raw.extend_from_slice(&[1u8; 12]); // scales
            raw.extend_from_slice(&[fill; 128]); // qs
        }
        let td = TensorData::from_bytes(crate::types::GgmlType::Q4_K, vec![256, 2], 512, raw).unwrap();
        let x = [1.0f32; 256];
        let mut y = [0.0f32; 2];
        td.matvec(&x, &mut y).unwrap();

        // Reference: dequantize each row via the (independently tested) row
        // dequant, then dot manually.
        if let TensorData::Q4_K(qt) = &td {
            for r in 0..2 {
                let row = qt.dequantize_row(r).unwrap();
                let expected: f32 = row.iter().sum();
                assert!((y[r] - expected).abs() < 1e-3);
            }
        } else {
            panic!("expected Q4_K");
        }
        assert!((y[0] - 256.0).abs() < 1e-3);
        assert!((y[1] - 512.0).abs() < 1e-3);
    }

    #[test]
    fn test_quant_matvec_arity_error() {
        // [in=32, out=2] but wrong vector sizes must fail explicitly.
        let d_fp16: u16 = 0x3C00;
        let mut raw = Vec::new();
        for _ in 0..2 {
            raw.extend_from_slice(&q4_0_block(d_fp16, 0x11));
        }
        let td = TensorData::from_bytes(crate::types::GgmlType::Q4_0, vec![32, 2], 64, raw).unwrap();
        let x = [1.0f32; 2];
        let mut y = [0.0f32; 32];
        assert!(td.matvec(&x, &mut y).is_err());
    }

    #[test]
    fn test_dequantize_row_explicit_layout() {
        // [in=32, out=3]: rows 0/1/2 with distinct fills 0x11/0x23/0xF8.
        let d_fp16: u16 = 0x3C00;
        let mut raw = Vec::new();
        raw.extend_from_slice(&q4_0_block(d_fp16, 0x11));
        raw.extend_from_slice(&q4_0_block(d_fp16, 0x23));
        raw.extend_from_slice(&q4_0_block(d_fp16, 0xF8));
        let td = TensorData::from_bytes(crate::types::GgmlType::Q4_0, vec![32, 3], 96, raw).unwrap();
        if let TensorData::Q4_0(qt) = &td {
            let row0 = qt.dequantize_row(0).unwrap();
            let row1 = qt.dequantize_row(1).unwrap();
            let row2 = qt.dequantize_row(2).unwrap();
            assert_eq!(row0.len(), 32);
            assert!(row0.iter().all(|&v| (v + 7.0).abs() < 1e-4));
            assert!(row1[..16].iter().all(|&v| (v + 5.0).abs() < 1e-4));
            assert!(row1[16..].iter().all(|&v| (v + 6.0).abs() < 1e-4));
            assert!(row2[..16].iter().all(|&v| v.abs() < 1e-4));
            assert!(row2[16..].iter().all(|&v| (v - 7.0).abs() < 1e-4));
            assert!(qt.dequantize_row(3).is_err(), "out-of-range row must error");
        } else {
            panic!("expected Q4_0");
        }
    }

    #[test]
    fn test_descriptor_resident_bytes_match_owned_representations() {
        assert_eq!(TensorData::resident_bytes_for(GgmlType::F32, 32, 128).unwrap(), 128);
        assert_eq!(TensorData::resident_bytes_for(GgmlType::F16, 32, 64).unwrap(), 128);
        assert_eq!(TensorData::resident_bytes_for(GgmlType::BF16, 32, 64).unwrap(), 128);
        assert_eq!(TensorData::resident_bytes_for(GgmlType::Q4_0, 32, 18).unwrap(), 18);
        assert!(TensorData::resident_bytes_for(GgmlType::Q4_1, 32, 20).is_err());
    }

    #[test]
    fn test_resident_bytes_f16_reflects_decoded_f32_storage() {
        // F16 tensors are held decoded as Vec<f32>: residency must be
        // 4 B/elem even though the file form is 2 B/elem (M6.1 BUG-3 fix).
        let raw: Vec<u8> = (0..32).flat_map(|_| 0x3C00u16.to_le_bytes()).collect();
        let td = TensorData::from_bytes(GgmlType::F16, vec![32], 32, raw).unwrap();
        assert_eq!(td.num_elements(), 32);
        assert_eq!(td.resident_bytes(), 32 * 4);
        assert_eq!(
            TensorData::resident_bytes_for(GgmlType::F16, 32, 32 * 2).unwrap(),
            td.resident_bytes() as u64
        );
    }

    #[test]
    fn test_resident_bytes_bf16_reflects_decoded_f32_storage() {
        let one: u16 = (1.0f32.to_bits() >> 16) as u16;
        let raw: Vec<u8> = (0..16).flat_map(|_| one.to_le_bytes()).collect();
        let td = TensorData::from_bytes(GgmlType::BF16, vec![4, 4], 16, raw).unwrap();
        assert_eq!(td.num_elements(), 16);
        assert_eq!(td.resident_bytes(), 16 * 4);
        // And the decoded values are what we put in.
        let (data, shape) = td.as_f32_slice().unwrap();
        assert_eq!(shape, &[4, 4]);
        assert!(data.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_get_embedding_quantized_rows() {
        // token_embd-style [n_embd=32, vocab=2] with distinct rows.
        let d_fp16: u16 = 0x3C00;
        let mut raw = Vec::new();
        raw.extend_from_slice(&q4_0_block(d_fp16, 0x11)); // token 0: all -7
        raw.extend_from_slice(&q4_0_block(d_fp16, 0x99)); // token 1: all +1
        let td = TensorData::from_bytes(crate::types::GgmlType::Q4_0, vec![32, 2], 64, raw).unwrap();
        let e0 = td.get_embedding(0, 32).unwrap();
        let e1 = td.get_embedding(1, 32).unwrap();
        assert!(e0.iter().all(|&v| (v + 7.0).abs() < 1e-4));
        assert!(e1.iter().all(|&v| (v - 1.0).abs() < 1e-4));
    }
}
