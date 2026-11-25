//! Model capacity accounting that separates routed experts from weights that
//! are always in the execution path.
//!
//! These are metadata-derived estimates. They do not prove residency, and an
//! "active expert" fraction is not a guarantee that only those experts are
//! read from storage during decode.

use serde::{Deserialize, Serialize};

use ramforge_core::GgufModel;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertAccounting {
    /// Sum of every tensor payload in the file.
    pub total_tensor_bytes: u64,
    /// Routed-expert tensors (`*_exps` and names containing `.expert`).
    pub routed_expert_bytes: u64,
    /// Shared-expert tensors (`*_shexp`), which are not router-selected.
    pub shared_expert_bytes: u64,
    /// Attention, embeddings, output, norms, and other non-expert tensors.
    pub other_tensor_bytes: u64,
    pub expert_count: Option<u64>,
    pub expert_used_count: Option<u64>,
    pub routed_expert_fraction_bps: Option<u32>,
    /// `routed_expert_bytes * used / count`, rounded up.
    pub estimated_active_expert_bytes: Option<u64>,
    /// Other tensors plus shared experts plus the estimated active routed set.
    /// This is an arithmetic lower bound for a decode step, not a residency
    /// observation and not a complete peak-memory figure.
    pub estimated_active_weight_bytes: Option<u64>,
    pub scope: String,
}

pub fn expert_accounting(model: &GgufModel) -> ExpertAccounting {
    let info = model.info();
    let mut total = 0u64;
    let mut routed = 0u64;
    let mut shared = 0u64;
    let mut routed_tensors = 0usize;
    for tensor in &model.tensors {
        let bytes = tensor.byte_length.unwrap_or(0);
        total = total.saturating_add(bytes);
        let name = tensor.name.as_str();
        if name.contains("_shexp") || name.contains("shared_expert") {
            shared = shared.saturating_add(bytes);
        } else if name.contains("_exps") || name.contains(".expert") {
            routed = routed.saturating_add(bytes);
            routed_tensors += 1;
        }
    }
    let other = total.saturating_sub(routed).saturating_sub(shared);
    let fraction = info
        .expert_count
        .filter(|count| *count > 0)
        .zip(info.expert_used_count)
        .map(|(count, used)| {
            u32::try_from(used.saturating_mul(10_000) / count).unwrap_or(u32::MAX)
        });
    let active_expert = if routed > 0 {
        info.expert_count
            .filter(|count| *count > 1)
            .zip(info.expert_used_count)
            .map(|(count, used)| routed.saturating_mul(used).saturating_add(count - 1) / count)
    } else {
        None
    };
    let active_weight =
        active_expert.map(|active| other.saturating_add(shared).saturating_add(active));
    ExpertAccounting {
        total_tensor_bytes: total,
        routed_expert_bytes: routed,
        shared_expert_bytes: shared,
        other_tensor_bytes: other,
        expert_count: info.expert_count,
        expert_used_count: info.expert_used_count,
        routed_expert_fraction_bps: fraction,
        estimated_active_expert_bytes: active_expert,
        estimated_active_weight_bytes: active_weight,
        scope: format!(
            "metadata-derived estimate from {routed_tensors} routed-expert tensors; not a native allocation, residency proof, or guarantee of routing locality"
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ramforge_core::{GgmlType, MetadataValue, TensorDescriptor};

    use super::*;

    fn model(names: &[(&str, u64)]) -> GgufModel {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            MetadataValue::String("deepseek2".to_string()),
        );
        metadata.insert(
            "deepseek2.expert_count".to_string(),
            MetadataValue::UInt64(64),
        );
        metadata.insert(
            "deepseek2.expert_used_count".to_string(),
            MetadataValue::UInt64(6),
        );
        GgufModel {
            path: std::path::PathBuf::from("/m"),
            file_size: names.iter().map(|(_, bytes)| *bytes).sum(),
            version: 3,
            metadata,
            tensors: names
                .iter()
                .map(|(name, bytes)| TensorDescriptor {
                    name: (*name).to_string(),
                    dimensions: vec![1],
                    ggml_type: GgmlType::Q4_K,
                    offset: 0,
                    file_offset: 0,
                    byte_length: Some(*bytes),
                    num_elements: 1,
                })
                .collect(),
            alignment: 32,
            data_start_offset: 0,
        }
    }

    #[test]
    fn separates_routed_shared_and_other_tensors() {
        let model = model(&[
            ("blk.0.attn_q.weight", 100),
            ("blk.1.ffn_gate_exps.weight", 640),
            ("blk.1.ffn_down_exps.weight", 640),
            ("blk.1.ffn_up_shexp.weight", 100),
            ("output.weight", 60),
        ]);
        let accounting = expert_accounting(&model);
        assert_eq!(accounting.total_tensor_bytes, 1540);
        assert_eq!(accounting.routed_expert_bytes, 1280);
        assert_eq!(accounting.shared_expert_bytes, 100);
        assert_eq!(accounting.other_tensor_bytes, 160);
        assert_eq!(accounting.routed_expert_fraction_bps, Some(937));
        // 1280 * 6 / 64 = 120, rounded up.
        assert_eq!(accounting.estimated_active_expert_bytes, Some(120));
        assert_eq!(
            accounting.estimated_active_weight_bytes,
            Some(160 + 100 + 120)
        );
    }

    #[test]
    fn dense_models_have_no_active_expert_estimate() {
        let model = model(&[("blk.0.attn_q.weight", 100)]);
        let accounting = expert_accounting(&model);
        assert_eq!(accounting.routed_expert_bytes, 0);
        assert_eq!(accounting.estimated_active_expert_bytes, None);
        assert_eq!(accounting.estimated_active_weight_bytes, None);
    }
}
