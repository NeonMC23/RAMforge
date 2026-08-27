//! Shared runtime memory-accounting formulas.
//!
//! Planning and execution must use the same load-transient rules. These
//! helpers calculate charges only; they never allocate or read tensor data.

use ramforge_core::model::TensorDescriptor;
use ramforge_core::tensor::TensorData;
use ramforge_core::types::GgmlType;

use crate::layer_read::LayerReadPlan;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct LayerMemoryEstimate {
    pub resident_bytes: u64,
    pub load_peak_bytes: u64,
}

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

pub(crate) fn estimate_layer_memory(
    tensors: &[TensorDescriptor],
) -> Result<LayerMemoryEstimate, String> {
    let mut settled = 0u64;
    let mut load_peak = 0u64;
    for descriptor in tensors {
        let file_bytes = descriptor.byte_length.ok_or_else(|| {
            format!("tensor '{}' byte length is unknown", descriptor.name)
        })?;
        let load_charge = tensor_load_charge_bytes(descriptor.ggml_type, file_bytes)?.max(1);
        load_peak = load_peak.max(settled.checked_add(load_charge).ok_or_else(|| {
            format!("layer load peak overflow at tensor '{}'", descriptor.name)
        })?);
        let resident = TensorData::resident_bytes_for(
            descriptor.ggml_type,
            descriptor.num_elements,
            file_bytes,
        )
        .map_err(|error| format!("tensor '{}': {}", descriptor.name, error))?;
        settled = settled.checked_add(resident.max(1)).ok_or_else(|| {
            format!("layer resident size overflow at tensor '{}'", descriptor.name)
        })?;
    }
    Ok(LayerMemoryEstimate {
        resident_bytes: settled,
        load_peak_bytes: load_peak,
    })
}

pub(crate) fn estimate_grouped_layer_memory(
    tensors: &[TensorDescriptor],
    plan: &LayerReadPlan,
) -> Result<LayerMemoryEstimate, String> {
    let mut settled = 0u64;
    let mut load_peak = 0u64;
    for range in &plan.ranges {
        if range.tensors.len() == 1 {
            let descriptor = &tensors[range.tensors[0].descriptor_index];
            let file_bytes = descriptor.byte_length.ok_or_else(|| {
                format!("tensor '{}' byte length is unknown", descriptor.name)
            })?;
            let charge = tensor_load_charge_bytes(descriptor.ggml_type, file_bytes)?.max(1);
            load_peak = load_peak.max(settled.checked_add(charge).ok_or_else(|| {
                format!("layer load peak overflow at tensor '{}'", descriptor.name)
            })?);
            let resident = TensorData::resident_bytes_for(
                descriptor.ggml_type,
                descriptor.num_elements,
                file_bytes,
            )
            .map_err(|error| format!("tensor '{}': {}", descriptor.name, error))?;
            settled = settled.checked_add(resident.max(1)).ok_or_else(|| {
                format!("layer resident size overflow at tensor '{}'", descriptor.name)
            })?;
            continue;
        }

        let range_resident = range.tensors.iter().try_fold(0u64, |total, tensor| {
            let descriptor = &tensors[tensor.descriptor_index];
            let file_bytes = descriptor.byte_length.ok_or_else(|| {
                format!("tensor '{}' byte length is unknown", descriptor.name)
            })?;
            let resident = TensorData::resident_bytes_for(
                descriptor.ggml_type,
                descriptor.num_elements,
                file_bytes,
            )
            .map_err(|error| format!("tensor '{}': {}", descriptor.name, error))?;
            total
                .checked_add(resident.max(1))
                .ok_or_else(|| "coalesced range resident size overflow".to_string())
        })?;
        let grouped_peak = settled
            .checked_add(range.byte_length)
            .and_then(|value| value.checked_add(range_resident))
            .ok_or_else(|| "coalesced layer load peak overflow".to_string())?;
        load_peak = load_peak.max(grouped_peak);
        settled = settled
            .checked_add(range_resident)
            .ok_or_else(|| "layer resident size overflow".to_string())?;
    }
    Ok(LayerMemoryEstimate {
        resident_bytes: settled,
        load_peak_bytes: load_peak,
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
