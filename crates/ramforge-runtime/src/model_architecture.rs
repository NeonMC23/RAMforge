//! GGUF-derived architecture metadata for planning and reporting.
//!
//! Fields come from GGUF metadata only. Missing metadata stays unknown; active
//! parameter fractions are explicitly estimates and do not prove execution.

use serde::{Deserialize, Serialize};

use ramforge_core::GgufModel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchitectureKind {
    Dense,
    MixtureOfExperts,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArchitecture {
    pub kind: ArchitectureKind,
    pub architecture: Option<String>,
    pub layer_count: Option<u64>,
    pub head_count: Option<u64>,
    pub head_count_kv: Option<u64>,
    pub embedding_length: Option<u64>,
    pub context_max: Option<u64>,
    pub expert_count: Option<u64>,
    pub expert_used_count: Option<u64>,
    /// Routed-expert fraction in basis points (2500 = 25%). Shared weights
    /// are not separated by GGUF metadata, so this is not a complete active
    /// parameter ratio.
    pub routed_expert_fraction_bps: Option<u32>,
    pub provenance: String,
    pub notes: Vec<String>,
}

pub fn inspect_architecture(model: &GgufModel) -> ModelArchitecture {
    let info = model.info();
    let expert_count = info.expert_count;
    let expert_used = info.expert_used_count;
    let kind = match (expert_count, expert_used) {
        (Some(count), Some(used)) if count > 1 && used > 0 => ArchitectureKind::MixtureOfExperts,
        (Some(_), _) => ArchitectureKind::Unknown,
        _ => ArchitectureKind::Dense,
    };
    let routed_fraction = expert_count
        .filter(|count| *count > 0)
        .zip(expert_used)
        .map(|(count, used)| {
            u32::try_from(used.saturating_mul(10_000) / count).unwrap_or(u32::MAX)
        });
    let mut notes = Vec::new();
    if kind == ArchitectureKind::MixtureOfExperts {
        notes.push(
            "routed expert fraction is not a complete active-parameter ratio; shared weights are unknown"
                .to_string(),
        );
    }
    if kind == ArchitectureKind::Unknown {
        notes.push("expert metadata is incomplete".to_string());
    }
    ModelArchitecture {
        kind,
        architecture: info.architecture,
        layer_count: info.block_count,
        head_count: info.head_count,
        head_count_kv: info.head_count_kv,
        embedding_length: info.embedding_length,
        context_max: info.context_length,
        expert_count,
        expert_used_count: expert_used,
        routed_expert_fraction_bps: routed_fraction,
        provenance: "GGUF metadata".to_string(),
        notes,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ramforge_core::{GgmlType, MetadataValue, TensorDescriptor};

    use super::*;

    fn model(metadata: BTreeMap<String, MetadataValue>) -> GgufModel {
        GgufModel {
            path: std::path::PathBuf::from("/m"),
            file_size: 1,
            version: 3,
            metadata,
            tensors: vec![TensorDescriptor {
                name: "x".to_string(),
                dimensions: vec![1],
                ggml_type: GgmlType::F32,
                offset: 0,
                file_offset: 0,
                byte_length: Some(4),
                num_elements: 1,
            }],
            alignment: 32,
            data_start_offset: 0,
        }
    }

    fn base() -> BTreeMap<String, MetadataValue> {
        let mut values = BTreeMap::new();
        values.insert(
            "general.architecture".to_string(),
            MetadataValue::String("llama".to_string()),
        );
        values
    }

    #[test]
    fn dense_architecture_is_detected_without_expert_metadata() {
        let architecture = inspect_architecture(&model(base()));
        assert_eq!(architecture.kind, ArchitectureKind::Dense);
        assert_eq!(architecture.architecture.as_deref(), Some("llama"));
    }

    #[test]
    fn moe_metadata_is_explicit_and_fraction_is_estimated() {
        let mut metadata = base();
        metadata.insert("llama.expert_count".to_string(), MetadataValue::UInt64(8));
        metadata.insert(
            "llama.expert_used_count".to_string(),
            MetadataValue::UInt64(2),
        );
        let architecture = inspect_architecture(&model(metadata));
        assert_eq!(architecture.kind, ArchitectureKind::MixtureOfExperts);
        assert_eq!(architecture.routed_expert_fraction_bps, Some(2500));
        assert!(architecture
            .notes
            .iter()
            .any(|note| note.contains("shared")));
    }

    #[test]
    fn incomplete_expert_metadata_is_unknown() {
        let mut metadata = base();
        metadata.insert("llama.expert_count".to_string(), MetadataValue::UInt64(8));
        let architecture = inspect_architecture(&model(metadata));
        assert_eq!(architecture.kind, ArchitectureKind::Unknown);
    }
}
