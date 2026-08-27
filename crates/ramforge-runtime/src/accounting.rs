//! Shared runtime memory-accounting formulas.
//!
//! Planning and execution must use the same load-transient rules. These
//! helpers calculate charges only; they never allocate or read tensor data.

use ramforge_core::types::GgmlType;

/// Peak managed charge while constructing one tensor from `file_bytes`.
///
/// - Direct F32 and compact quantized tensors own one representation: 1x.
/// - F16/BF16 temporarily own raw 2-byte values plus decoded F32: 3x.
pub(crate) fn tensor_load_charge_bytes(
    ggml_type: GgmlType,
    file_bytes: u64,
) -> Result<u64, String> {
    let factor = match ggml_type {
        GgmlType::F16 | GgmlType::BF16 => 3,
        _ => 1,
    };
    file_bytes.checked_mul(factor).ok_or_else(|| {
        format!(
            "tensor size overflow computing load charge ({} bytes x {} for {})",
            file_bytes,
            factor,
            ggml_type.name()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_load_charge_factors_match_runtime_representations() {
        assert_eq!(tensor_load_charge_bytes(GgmlType::F32, 100).unwrap(), 100);
        assert_eq!(tensor_load_charge_bytes(GgmlType::Q4_K, 100).unwrap(), 100);
        assert_eq!(tensor_load_charge_bytes(GgmlType::F16, 100).unwrap(), 300);
        assert_eq!(tensor_load_charge_bytes(GgmlType::BF16, 100).unwrap(), 300);
    }

    #[test]
    fn test_tensor_load_charge_overflow_is_rejected() {
        let error = tensor_load_charge_bytes(GgmlType::F16, u64::MAX).unwrap_err();
        assert!(error.contains("overflow"));
    }
}
