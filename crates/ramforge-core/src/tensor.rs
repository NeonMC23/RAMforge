//! Tensor decoding for supported GGML types
//!
//! Supported formats for Milestone 3:
//! - F32
//! - F16 (converted to F32)
//!
//! Other types produce a clear error.

use crate::error::DataSourceError;
use crate::types::GgmlType;

/// Decode raw tensor bytes into Vec<f32>
///
/// `bytes` must be exactly the tensor's byte length.
/// Supports F32 and F16.
pub fn decode_tensor_to_f32(
    bytes: &[u8],
    ggml_type: GgmlType,
    num_elements: u64,
) -> Result<Vec<f32>, DataSourceError> {
    match ggml_type {
        GgmlType::F32 => decode_f32(bytes, num_elements),
        GgmlType::F16 => decode_f16(bytes, num_elements),
        GgmlType::BF16 => decode_bf16(bytes, num_elements),
        _ => Err(DataSourceError::General(format!(
            "unsupported tensor type for inference: {} (only F32 and F16 are supported in milestone 3)",
            ggml_type.name()
        ))),
    }
}

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
    #[allow(clippy::chunks_exact_to_as_chunks)]
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
    #[allow(clippy::chunks_exact_to_as_chunks)]
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
    #[allow(clippy::chunks_exact_to_as_chunks)]
    for chunk in bytes[..expected].chunks_exact(2) {
        let arr: [u8; 2] = chunk.try_into().unwrap();
        let bits = u16::from_le_bytes(arr);
        out.push(bf16_to_f32(bits));
    }
    Ok(out)
}

/// Convert IEEE 754 binary16 (half) to f32
fn f16_to_f32(bits: u16) -> f32 {
    // Using half crate logic manually to avoid dependency, but we have half crate available
    // We'll implement manual conversion for clarity
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // subnormal
            // Normalize
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
        // Inf/NaN
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
        // 1.0 in f16 is 0x3C00
        let bytes = [0x00, 0x3C, 0x00, 0x38]; // 1.0, 0.5
        let decoded = decode_f16(&bytes, 2).unwrap();
        assert!((decoded[0] - 1.0).abs() < 1e-3);
        assert!((decoded[1] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_unsupported_type() {
        let bytes = vec![0u8; 18]; // Q4_0 size
        let err = decode_tensor_to_f32(&bytes, crate::types::GgmlType::Q4_0, 32).unwrap_err();
        assert!(err.to_string().contains("unsupported"));
    }
}
