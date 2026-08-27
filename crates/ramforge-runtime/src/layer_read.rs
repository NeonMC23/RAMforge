//! Descriptor-only bounded read planning for streamed layers.

use ramforge_core::model::TensorDescriptor;

/// Alignment and metadata gaps up to this size may be read as bounded overhead.
pub(crate) const MAX_COALESCED_GAP_BYTES: u64 = 4 * 1024;
/// A grouped temporary buffer may never exceed this span.
pub(crate) const MAX_COALESCED_SPAN_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlannedTensor {
    pub descriptor_index: usize,
    pub offset_in_range: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlannedReadRange {
    pub file_offset: u64,
    pub byte_length: u64,
    pub logical_bytes: u64,
    pub gap_bytes: u64,
    pub tensors: Vec<PlannedTensor>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct LayerReadPlan {
    pub logical_tensor_count: usize,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
    pub gap_bytes: u64,
    pub ranges: Vec<PlannedReadRange>,
}

pub(crate) fn build_layer_read_plan(
    descriptors: &[TensorDescriptor],
) -> Result<LayerReadPlan, String> {
    let mut ordered = Vec::with_capacity(descriptors.len());
    let mut logical_bytes = 0u64;
    for (index, descriptor) in descriptors.iter().enumerate() {
        let byte_length = descriptor.byte_length.ok_or_else(|| {
            format!("tensor '{}' byte length is unknown", descriptor.name)
        })?;
        let end = descriptor.file_offset.checked_add(byte_length).ok_or_else(|| {
            format!("tensor '{}' file range overflows", descriptor.name)
        })?;
        logical_bytes = logical_bytes.checked_add(byte_length).ok_or_else(|| {
            "layer logical tensor byte count overflow".to_string()
        })?;
        ordered.push((descriptor.file_offset, end, byte_length, index));
    }
    ordered.sort_by_key(|(start, _, _, index)| (*start, *index));

    let mut ranges: Vec<PlannedReadRange> = Vec::new();
    for (start, end, byte_length, descriptor_index) in ordered {
        if let Some(current) = ranges.last_mut() {
            let current_end = current
                .file_offset
                .checked_add(current.byte_length)
                .ok_or_else(|| "planned range end overflow".to_string())?;
            if start < current_end {
                return Err(format!(
                    "tensor ranges overlap at descriptor index {}",
                    descriptor_index
                ));
            }
            let gap = start - current_end;
            let candidate_span = end
                .checked_sub(current.file_offset)
                .ok_or_else(|| "planned range span underflow".to_string())?;
            if gap <= MAX_COALESCED_GAP_BYTES
                && candidate_span <= MAX_COALESCED_SPAN_BYTES
            {
                current.tensors.push(PlannedTensor {
                    descriptor_index,
                    offset_in_range: start - current.file_offset,
                });
                current.byte_length = candidate_span;
                current.logical_bytes = current
                    .logical_bytes
                    .checked_add(byte_length)
                    .ok_or_else(|| "planned range logical byte overflow".to_string())?;
                current.gap_bytes = current
                    .gap_bytes
                    .checked_add(gap)
                    .ok_or_else(|| "planned range gap byte overflow".to_string())?;
                continue;
            }
        }

        ranges.push(PlannedReadRange {
            file_offset: start,
            byte_length,
            logical_bytes: byte_length,
            gap_bytes: 0,
            tensors: vec![PlannedTensor {
                descriptor_index,
                offset_in_range: 0,
            }],
        });
    }

    let physical_bytes = ranges.iter().try_fold(0u64, |total, range| {
        total.checked_add(range.byte_length)
    }).ok_or_else(|| "layer physical read byte count overflow".to_string())?;
    let gap_bytes = ranges.iter().try_fold(0u64, |total, range| {
        total.checked_add(range.gap_bytes)
    }).ok_or_else(|| "layer coalesced gap byte count overflow".to_string())?;

    Ok(LayerReadPlan {
        logical_tensor_count: descriptors.len(),
        logical_bytes,
        physical_bytes,
        gap_bytes,
        ranges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ramforge_core::types::GgmlType;

    fn descriptor(name: &str, offset: u64, length: u64) -> TensorDescriptor {
        TensorDescriptor {
            name: name.to_string(),
            dimensions: vec![length / 4],
            ggml_type: GgmlType::F32,
            offset,
            file_offset: offset,
            byte_length: Some(length),
            num_elements: length / 4,
        }
    }

    #[test]
    fn test_adjacent_and_arbitrarily_ordered_tensors_coalesce() {
        let descriptors = vec![
            descriptor("c", 108, 4),
            descriptor("a", 100, 4),
            descriptor("b", 104, 4),
        ];
        let plan = build_layer_read_plan(&descriptors).unwrap();
        assert_eq!(plan.logical_tensor_count, 3);
        assert_eq!(plan.ranges.len(), 1);
        assert_eq!(plan.ranges[0].file_offset, 100);
        assert_eq!(plan.ranges[0].byte_length, 12);
        assert_eq!(plan.ranges[0].gap_bytes, 0);
        let indices: Vec<usize> = plan.ranges[0]
            .tensors
            .iter()
            .map(|tensor| tensor.descriptor_index)
            .collect();
        assert_eq!(indices, vec![1, 2, 0]);
    }

    #[test]
    fn test_small_gap_is_bounded_overhead_not_logical_data() {
        let descriptors = vec![descriptor("a", 100, 8), descriptor("b", 116, 8)];
        let plan = build_layer_read_plan(&descriptors).unwrap();
        assert_eq!(plan.ranges.len(), 1);
        assert_eq!(plan.logical_bytes, 16);
        assert_eq!(plan.physical_bytes, 24);
        assert_eq!(plan.gap_bytes, 8);
        assert_eq!(plan.ranges[0].tensors[1].offset_in_range, 16);
    }

    #[test]
    fn test_large_gap_and_max_span_remain_separate() {
        let gap = MAX_COALESCED_GAP_BYTES + 1;
        let descriptors = vec![descriptor("a", 0, 8), descriptor("b", 8 + gap, 8)];
        assert_eq!(build_layer_read_plan(&descriptors).unwrap().ranges.len(), 2);

        let descriptors = vec![
            descriptor("a", 0, MAX_COALESCED_SPAN_BYTES),
            descriptor("b", MAX_COALESCED_SPAN_BYTES, 4),
        ];
        assert_eq!(build_layer_read_plan(&descriptors).unwrap().ranges.len(), 2);
    }
}
