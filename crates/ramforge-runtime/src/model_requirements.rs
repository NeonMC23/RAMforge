//! Conservative model resource requirements derived from GGUF metadata.
//!
//! File size, tensor payload size, mapped size, resident memory, and runtime
//! allocation are distinct facts. Estimates stay labeled as estimates and
//! missing metadata stays unknown.

use serde::{Deserialize, Serialize};

use ramforge_core::GgufModel;

use crate::context_config::KvCacheType;

pub const MODEL_REQUIREMENTS_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequirementCertainty {
    Exact,
    Estimated,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequirementBytes {
    pub bytes: Option<u64>,
    pub certainty: RequirementCertainty,
    pub provenance: String,
    pub detail: Option<String>,
}

impl RequirementBytes {
    pub fn exact(bytes: u64, provenance: impl Into<String>) -> Self {
        Self {
            bytes: Some(bytes),
            certainty: RequirementCertainty::Exact,
            provenance: provenance.into(),
            detail: None,
        }
    }

    pub fn estimated(bytes: u64, detail: impl Into<String>) -> Self {
        Self {
            bytes: Some(bytes),
            certainty: RequirementCertainty::Estimated,
            provenance: "derived estimate".to_string(),
            detail: Some(detail.into()),
        }
    }

    pub fn unknown(detail: impl Into<String>) -> Self {
        Self {
            bytes: None,
            certainty: RequirementCertainty::Unknown,
            provenance: "unavailable".to_string(),
            detail: Some(detail.into()),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        match (self.certainty, self.bytes) {
            (RequirementCertainty::Unknown, None) => Ok(()),
            (RequirementCertainty::Unknown, Some(_)) => {
                Err("unknown requirement carries a byte value".to_string())
            }
            (_, Some(_)) => Ok(()),
            (_, None) => Err("exact/estimated requirement has no byte value".to_string()),
        }
    }

    pub fn checked_add(&self, other: &Self) -> Self {
        let Some(left) = self.bytes else {
            return Self::unknown(
                self.detail
                    .clone()
                    .unwrap_or_else(|| "left unknown".to_string()),
            );
        };
        let Some(right) = other.bytes else {
            return Self::unknown(
                other
                    .detail
                    .clone()
                    .unwrap_or_else(|| "right unknown".to_string()),
            );
        };
        let Some(sum) = left.checked_add(right) else {
            return Self::unknown("requirement byte sum overflow");
        };
        let (certainty, provenance, detail) = if self.certainty == RequirementCertainty::Exact
            && other.certainty == RequirementCertainty::Exact
        {
            (RequirementCertainty::Exact, "exact sum", None)
        } else {
            (
                RequirementCertainty::Estimated,
                "estimated sum",
                Some("one or more contributions is estimated".to_string()),
            )
        };
        Self {
            bytes: Some(sum),
            certainty,
            provenance: provenance.to_string(),
            detail,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelRequirementConfig {
    pub context_tokens: u32,
    pub batch_tokens: u32,
    /// Bytes per KV element. GGUF does not always expose this; f16 (2) is a
    /// common assumption and remains marked as an estimate.
    pub kv_bytes_per_element: u8,
}

impl Default for ModelRequirementConfig {
    fn default() -> Self {
        Self {
            context_tokens: 2048,
            batch_tokens: 1,
            kv_bytes_per_element: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRequirements {
    pub schema_version: u32,
    pub model_fingerprint: Option<String>,
    pub model_name: Option<String>,
    pub architecture: Option<String>,
    pub architecture_info: Option<crate::model_architecture::ModelArchitecture>,
    pub quantization: Option<String>,
    pub tensor_count: usize,
    pub file_bytes: RequirementBytes,
    pub tensor_payload_bytes: RequirementBytes,
    pub fully_resident_weight_bytes: RequirementBytes,
    pub kv_cache_bytes: RequirementBytes,
    pub context_tokens: u32,
    pub batch_tokens: u32,
    pub model_max_context_tokens: Option<u64>,
    pub resident_weights_plus_kv_bytes: RequirementBytes,
    pub mapped_weights_plus_kv_bytes: RequirementBytes,
    pub unknown_requirements: Vec<String>,
    pub assumptions: Vec<String>,
}

impl ModelRequirements {
    /// Recompute the KV estimate for a selected cache dtype.
    ///
    /// The base estimate in [`model_requirements`] keeps the historical
    /// F16-per-element assumption. This method replaces it with exact
    /// per-type block accounting when architecture metadata is available and
    /// refreshes the dependent sums. It never claims a native allocation.
    pub fn apply_kv_cache_type(&mut self, kv_type: KvCacheType) -> Result<(), String> {
        self.apply_kv_cache_types(kv_type, kv_type)
    }

    /// Recompute the KV estimate with independent K and V cache dtypes.
    pub fn apply_kv_cache_types(
        &mut self,
        kv_type_k: KvCacheType,
        kv_type_v: KvCacheType,
    ) -> Result<(), String> {
        let architecture = self
            .architecture_info
            .as_ref()
            .ok_or_else(|| "architecture metadata is unavailable; cannot size KV".to_string())?;
        let layers = architecture
            .layer_count
            .ok_or_else(|| "layer count is unavailable; cannot size KV".to_string())?;
        let kv_heads = architecture
            .head_count_kv
            .ok_or_else(|| "KV head count is unavailable; cannot size KV".to_string())?;
        let heads = architecture
            .head_count
            .ok_or_else(|| "head count is unavailable; cannot derive head dimension".to_string())?;
        let embedding = architecture
            .embedding_length
            .ok_or_else(|| "embedding length is unavailable; cannot derive head dimension".to_string())?;
        if heads == 0 || embedding % heads != 0 {
            return Err("embedding length is not divisible by head count".to_string());
        }
        let head_dim = embedding / heads;
        let context = u64::from(self.context_tokens);
        let k_bytes = kv_type_k
            .cache_side_bytes(layers, kv_heads, head_dim, context)
            .ok_or_else(|| "K cache byte accounting overflowed".to_string())?;
        let v_bytes = kv_type_v
            .cache_side_bytes(layers, kv_heads, head_dim, context)
            .ok_or_else(|| "V cache byte accounting overflowed".to_string())?;
        let bytes = k_bytes
            .checked_add(v_bytes)
            .ok_or_else(|| "KV byte accounting overflowed".to_string())?;
        let assumption = format!(
            "KV estimate uses {} K / {} V block accounting: {} layers, {} KV heads, head dimension {}, {} context tokens; batch traffic is not included",
            kv_type_k.label(),
            kv_type_v.label(),
            layers,
            kv_heads,
            head_dim,
            self.context_tokens
        );
        self.assumptions
            .retain(|value| {
                !value.starts_with("KV estimate ") && !value.starts_with("MLA architectures")
            });
        self.assumptions.push(assumption);
        if self
            .architecture
            .as_deref()
            .is_some_and(|architecture| architecture.contains("deepseek"))
        {
            self.assumptions.push(
                "MLA architectures may allocate additional compressed/rope KV state beyond the head-count formula; treat this estimate as a lower bound and prefer native context readback"
                    .to_string(),
            );
        }
        self.kv_cache_bytes =
            RequirementBytes::estimated(bytes, "architecture-derived KV cache estimate");
        self.resident_weights_plus_kv_bytes = self
            .fully_resident_weight_bytes
            .checked_add(&self.kv_cache_bytes);
        self.mapped_weights_plus_kv_bytes = RequirementBytes::estimated(
            bytes,
            "mapped weights are not assumed resident; only the KV estimate is included",
        );
        Ok(())
    }

    pub fn validate(&self) -> Result<(), String> {
        for requirement in [
            &self.file_bytes,
            &self.tensor_payload_bytes,
            &self.fully_resident_weight_bytes,
            &self.kv_cache_bytes,
            &self.resident_weights_plus_kv_bytes,
            &self.mapped_weights_plus_kv_bytes,
        ] {
            requirement.validate()?;
        }
        if self.context_tokens == 0 {
            return Err("context_tokens must be greater than zero".to_string());
        }
        if self.batch_tokens == 0 {
            return Err("batch_tokens must be greater than zero".to_string());
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(|error| error.to_string())
    }
}

/// Stable-enough, non-cryptographic file fingerprint for local comparison.
/// Reads a bounded header sample and the file size; it is not a security hash.
pub fn model_file_fingerprint(model: &GgufModel) -> Option<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(&model.path).ok()?;
    let mut header = vec![0u8; 64 * 1024];
    let read = file.read(&mut header).ok()?;
    header.truncate(read);
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in &header {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash ^= model.file_size;
    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    Some(format!("fnv1a64:{hash:016x}:size={}", model.file_size))
}

/// Derive requirements from GGUF metadata without reading tensor payloads.
pub fn model_requirements(model: &GgufModel, config: ModelRequirementConfig) -> ModelRequirements {
    let info = model.info();
    let mut unknown_requirements = Vec::new();
    let mut assumptions = Vec::new();

    let tensor_payload = match sum_tensor_bytes(model) {
        Some(bytes) => RequirementBytes::exact(bytes, "sum of GGUF tensor byte lengths"),
        None => {
            unknown_requirements.push("some tensor byte lengths are unknown".to_string());
            RequirementBytes::unknown("one or more tensor byte lengths are unavailable")
        }
    };
    let file = RequirementBytes::exact(model.file_size, "GGUF file size");
    let fully_resident = if tensor_payload.bytes.is_some() {
        RequirementBytes::estimated(
            tensor_payload.bytes.unwrap_or(0),
            "assumes GGUF tensor payloads are resident without alignment or backend overhead",
        )
    } else {
        RequirementBytes::unknown("tensor payload size is incomplete")
    };

    let kv = kv_cache_requirement(&info, config, &mut unknown_requirements, &mut assumptions);
    let resident_sum = fully_resident.checked_add(&kv);
    // A file-backed mapping does not imply that all weights are resident. The
    // mapped-plus-KV figure is intentionally only a lower-bound-shaped estimate
    // for planning discussion, not a fit guarantee.
    let mapped_sum = if let Some(kv_bytes) = kv.bytes {
        RequirementBytes::estimated(
            kv_bytes,
            "mapped weights are not assumed resident; only the KV estimate is included",
        )
    } else {
        RequirementBytes::unknown("KV estimate unavailable; mapped residency cannot be bounded")
    };

    ModelRequirements {
        schema_version: MODEL_REQUIREMENTS_SCHEMA_VERSION,
        model_fingerprint: model_file_fingerprint(model),
        model_name: info.name.clone(),
        architecture: info.architecture.clone(),
        architecture_info: Some(crate::model_architecture::inspect_architecture(model)),
        quantization: info.file_type.map(quantization_label),
        tensor_count: model.tensors.len(),
        file_bytes: file,
        tensor_payload_bytes: tensor_payload,
        fully_resident_weight_bytes: fully_resident,
        kv_cache_bytes: kv,
        context_tokens: config.context_tokens,
        batch_tokens: config.batch_tokens,
        model_max_context_tokens: info.context_length,
        resident_weights_plus_kv_bytes: resident_sum,
        mapped_weights_plus_kv_bytes: mapped_sum,
        unknown_requirements,
        assumptions,
    }
}

fn sum_tensor_bytes(model: &GgufModel) -> Option<u64> {
    let mut total = 0u64;
    for tensor in &model.tensors {
        let bytes = tensor.byte_length?;
        total = total.checked_add(bytes)?;
    }
    Some(total)
}

fn kv_cache_requirement(
    info: &ramforge_core::model::ModelInfo,
    config: ModelRequirementConfig,
    unknown: &mut Vec<String>,
    assumptions: &mut Vec<String>,
) -> RequirementBytes {
    if config.kv_bytes_per_element == 0 {
        unknown.push("KV element size is zero".to_string());
        return RequirementBytes::unknown("KV element size is zero");
    }
    let Some(layers) = info.block_count else {
        unknown.push("block_count unavailable; KV cache cannot be estimated".to_string());
        return RequirementBytes::unknown("block_count unavailable");
    };
    let Some(kv_heads) = info.head_count_kv else {
        unknown
            .push("attention.head_count_kv unavailable; KV cache cannot be estimated".to_string());
        return RequirementBytes::unknown("attention.head_count_kv unavailable");
    };
    let Some(embedding) = info.embedding_length else {
        unknown.push("embedding_length unavailable; KV cache cannot be estimated".to_string());
        return RequirementBytes::unknown("embedding_length unavailable");
    };
    let Some(head_count) = info.head_count else {
        unknown
            .push("attention.head_count unavailable; head dimension cannot be derived".to_string());
        return RequirementBytes::unknown("attention.head_count unavailable");
    };
    if head_count == 0 || embedding % head_count != 0 {
        unknown.push("embedding_length is not divisible by attention.head_count".to_string());
        return RequirementBytes::unknown("head dimension cannot be derived");
    }
    let head_dim = embedding / head_count;
    let Some(elements) = layers
        .checked_mul(2)
        .and_then(|value| value.checked_mul(kv_heads))
        .and_then(|value| value.checked_mul(head_dim))
        .and_then(|value| value.checked_mul(u64::from(config.context_tokens)))
    else {
        unknown.push("KV cache element count overflowed".to_string());
        return RequirementBytes::unknown("KV cache element count overflow");
    };
    let Some(bytes) = elements.checked_mul(u64::from(config.kv_bytes_per_element)) else {
        unknown.push("KV cache byte count overflowed".to_string());
        return RequirementBytes::unknown("KV cache byte count overflow");
    };
    assumptions.push(format!(
        "KV estimate assumes {} bytes per element, K and V, {} layers, {} KV heads, head dimension {}, and {} context tokens; batch {} is not included in the KV total",
        config.kv_bytes_per_element,
        layers,
        kv_heads,
        head_dim,
        config.context_tokens,
        config.batch_tokens,
    ));
    RequirementBytes::estimated(bytes, "architecture-derived KV cache estimate")
}

fn quantization_label(file_type: u32) -> String {
    match file_type {
        0 => "all-f32".to_string(),
        1 => "mostly-f16".to_string(),
        2 => "mostly-q4_0".to_string(),
        3 => "mostly-q4_1".to_string(),
        7 => "mostly-q8_0".to_string(),
        8 => "mostly-q5_0".to_string(),
        9 => "mostly-q5_1".to_string(),
        10 => "mostly-q2_k".to_string(),
        11 => "mostly-q3_k".to_string(),
        12 => "mostly-q4_k".to_string(),
        13 => "mostly-q5_k".to_string(),
        14 => "mostly-q6_k".to_string(),
        15 => "mostly-q8_k".to_string(),
        other => format!("unknown-file-type-{other}"),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ramforge_core::{GgmlType, MetadataValue, TensorDescriptor};

    use super::*;

    fn tensor(name: &str, length: Option<u64>) -> TensorDescriptor {
        TensorDescriptor {
            name: name.to_string(),
            dimensions: vec![16, 16],
            ggml_type: GgmlType::F16,
            offset: 0,
            file_offset: 0,
            byte_length: length,
            num_elements: 256,
        }
    }

    fn model(
        tensors: Vec<TensorDescriptor>,
        metadata: BTreeMap<String, MetadataValue>,
    ) -> GgufModel {
        GgufModel {
            path: std::path::PathBuf::from("/models/test.gguf"),
            file_size: 4096,
            version: 3,
            metadata,
            tensors,
            alignment: 32,
            data_start_offset: 0,
        }
    }

    fn metadata() -> BTreeMap<String, MetadataValue> {
        let mut values = BTreeMap::new();
        values.insert(
            "general.architecture".to_string(),
            MetadataValue::String("llama".to_string()),
        );
        values.insert("llama.block_count".to_string(), MetadataValue::UInt64(2));
        values.insert(
            "llama.attention.head_count".to_string(),
            MetadataValue::UInt64(4),
        );
        values.insert(
            "llama.attention.head_count_kv".to_string(),
            MetadataValue::UInt64(2),
        );
        values.insert(
            "llama.embedding_length".to_string(),
            MetadataValue::UInt64(128),
        );
        values.insert("general.file_type".to_string(), MetadataValue::UInt32(12));
        values
    }

    #[test]
    fn kv_cache_type_recomputes_exact_block_accounting() {
        let model = model(
            vec![tensor("a", Some(1000)), tensor("b", Some(24))],
            metadata(),
        );
        let mut requirements = model_requirements(
            &model,
            ModelRequirementConfig {
                context_tokens: 512,
                batch_tokens: 1,
                kv_bytes_per_element: 2,
            },
        );
        requirements
            .apply_kv_cache_type(KvCacheType::Q8_0)
            .expect("q8 accounting");
        // 2 layers * 2 K/V * 2 kv heads * 1 block per 32-wide head * 34 B * 512 tokens
        assert_eq!(requirements.kv_cache_bytes.bytes, Some(139_264));
        assert!(requirements
            .assumptions
            .iter()
            .any(|value| value.contains("q8_0 K / q8_0 V block accounting")));

        requirements
            .apply_kv_cache_type(KvCacheType::Q4_0)
            .expect("q4 accounting");
        // 2 layers * 2 K/V * 2 kv heads * 1 block * 18 B * 512 tokens
        assert_eq!(requirements.kv_cache_bytes.bytes, Some(73_728));
        assert!(!requirements
            .assumptions
            .iter()
            .any(|value| value.contains("q8_0 K / q8_0 V block accounting")));
    }

    #[test]
    fn exact_and_estimated_requirements_stay_distinct() {
        let model = model(
            vec![tensor("a", Some(1000)), tensor("b", Some(24))],
            metadata(),
        );
        let requirements = model_requirements(
            &model,
            ModelRequirementConfig {
                context_tokens: 512,
                batch_tokens: 1,
                kv_bytes_per_element: 2,
            },
        );
        requirements.validate().expect("requirements");
        assert_eq!(
            requirements.file_bytes.certainty,
            RequirementCertainty::Exact
        );
        assert_eq!(requirements.tensor_payload_bytes.bytes, Some(1024));
        assert_eq!(
            requirements.tensor_payload_bytes.certainty,
            RequirementCertainty::Exact
        );
        assert_eq!(
            requirements.kv_cache_bytes.certainty,
            RequirementCertainty::Estimated
        );
        // 2 layers * 2 (K/V) * 2 kv heads * 32 head_dim * 512 tokens * 2 bytes
        assert_eq!(requirements.kv_cache_bytes.bytes, Some(262_144));
        assert_eq!(
            requirements.resident_weights_plus_kv_bytes.certainty,
            RequirementCertainty::Estimated
        );
    }

    #[test]
    fn missing_metadata_is_unknown_not_zero() {
        let model = model(vec![tensor("a", None)], BTreeMap::new());
        let requirements = model_requirements(&model, ModelRequirementConfig::default());
        requirements.validate().expect("requirements");
        assert_eq!(requirements.tensor_payload_bytes.bytes, None);
        assert_eq!(
            requirements.tensor_payload_bytes.certainty,
            RequirementCertainty::Unknown
        );
        assert_eq!(requirements.kv_cache_bytes.bytes, None);
        assert!(!requirements.unknown_requirements.is_empty());
        assert_eq!(requirements.resident_weights_plus_kv_bytes.bytes, None);
    }

    #[test]
    fn overflow_is_reported_as_unknown() {
        let left = RequirementBytes::exact(u64::MAX, "test");
        let right = RequirementBytes::exact(1, "test");
        let sum = left.checked_add(&right);
        assert_eq!(sum.bytes, None);
        assert_eq!(sum.certainty, RequirementCertainty::Unknown);
    }

    #[test]
    fn requirement_serialization_is_deterministic() {
        let model = model(vec![tensor("a", Some(100))], metadata());
        let requirements = model_requirements(&model, ModelRequirementConfig::default());
        let first = serde_json::to_string(&requirements).unwrap();
        let second = serde_json::to_string(&requirements).unwrap();
        assert_eq!(first, second);
        let decoded: ModelRequirements = serde_json::from_str(&first).unwrap();
        assert_eq!(decoded, requirements);
    }
}
