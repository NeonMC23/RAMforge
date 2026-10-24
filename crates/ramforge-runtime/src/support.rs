//! Explicit model and quantization capability registry.
//!
//! GGUF parsing/inspection is architecture-neutral. Execution support is
//! deliberately narrower and must never be inferred from a similar name.

use ramforge_core::{model::GgufModel, types::GgmlType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunSupport {
    Supported,
    ViaLlamaGguf,
    NotYet,
}

impl RunSupport {
    pub fn label(self) -> &'static str {
        match self {
            Self::Supported => "yes",
            Self::ViaLlamaGguf => "via-llama-gguf",
            Self::NotYet => "not-yet",
        }
    }

    fn is_directly_supported(self) -> bool {
        self == Self::Supported
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchitectureCapability {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub inspect: bool,
    pub plan: bool,
    pub tokenizer: &'static str,
    pub run: RunSupport,
    pub notes: &'static str,
}

pub const ARCHITECTURES: &[ArchitectureCapability] = &[
    ArchitectureCapability {
        name: "llama",
        aliases: &[],
        inspect: true,
        plan: true,
        tokenizer: "sentencepiece/bpe",
        run: RunSupport::Supported,
        notes: "dense llama-compatible transformer",
    },
    ArchitectureCapability {
        name: "qwen2",
        aliases: &["qwen2.5"],
        inspect: true,
        plan: true,
        tokenizer: "bpe",
        run: RunSupport::Supported,
        notes: "dense qwen2 transformer (including qwen2.5), including validated Q/K/V biases",
    },
    ArchitectureCapability {
        name: "mistral",
        aliases: &[],
        inspect: true,
        plan: true,
        tokenizer: "sentencepiece",
        run: RunSupport::ViaLlamaGguf,
        notes: "runnable only when the GGUF declares llama-compatible architecture/tensors",
    },
    ArchitectureCapability {
        name: "qwen3",
        aliases: &[],
        inspect: true,
        plan: true,
        tokenizer: "inspect-only",
        run: RunSupport::NotYet,
        notes: "execution backend not implemented",
    },
    ArchitectureCapability {
        name: "qwen35",
        aliases: &["qwen3.5", "qwen3_5"],
        inspect: true,
        plan: true,
        tokenizer: "inspect-only",
        run: RunSupport::NotYet,
        notes: "hybrid attention/SSM architecture; intentionally not aliased to qwen2",
    },
    ArchitectureCapability {
        name: "gemma",
        aliases: &["gemma2", "gemma3"],
        inspect: true,
        plan: true,
        tokenizer: "inspect-only",
        run: RunSupport::NotYet,
        notes: "execution backend not implemented",
    },
    ArchitectureCapability {
        name: "phi",
        aliases: &["phi2", "phi3", "phi4"],
        inspect: true,
        plan: true,
        tokenizer: "inspect-only",
        run: RunSupport::NotYet,
        notes: "execution backend not implemented",
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizationCapability {
    pub name: &'static str,
    pub inference: bool,
}

pub const QUANTIZATIONS: &[QuantizationCapability] = &[
    QuantizationCapability {
        name: "F32",
        inference: true,
    },
    QuantizationCapability {
        name: "F16",
        inference: true,
    },
    QuantizationCapability {
        name: "BF16",
        inference: true,
    },
    QuantizationCapability {
        name: "Q4_0",
        inference: true,
    },
    QuantizationCapability {
        name: "Q5_0",
        inference: true,
    },
    QuantizationCapability {
        name: "Q8_0",
        inference: true,
    },
    QuantizationCapability {
        name: "Q2_K",
        inference: true,
    },
    QuantizationCapability {
        name: "Q3_K",
        inference: true,
    },
    QuantizationCapability {
        name: "Q4_K",
        inference: true,
    },
    QuantizationCapability {
        name: "Q5_K",
        inference: true,
    },
    QuantizationCapability {
        name: "Q6_K",
        inference: true,
    },
    QuantizationCapability {
        name: "Q8_K",
        inference: true,
    },
    QuantizationCapability {
        name: "other GGUF types",
        inference: false,
    },
];

pub fn architecture_capability(name: &str) -> Option<&'static ArchitectureCapability> {
    let normalized = name.to_ascii_lowercase();
    ARCHITECTURES.iter().find(|capability| {
        capability.name == normalized.as_str() || capability.aliases.contains(&normalized.as_str())
    })
}

pub fn execution_architectures() -> Vec<&'static str> {
    ARCHITECTURES
        .iter()
        .filter(|capability| capability.run.is_directly_supported())
        .map(|capability| capability.name)
        .collect()
}

pub fn ensure_execution_supported(architecture: &str) -> Result<(), String> {
    if architecture_capability(architecture)
        .map(|capability| capability.run.is_directly_supported())
        .unwrap_or(false)
    {
        return Ok(());
    }

    let supported = execution_architectures().join(", ");
    let detail = architecture_capability(architecture)
        .map(|capability| {
            format!(
                "inspection={} planning={} execution={} ({})",
                capability.inspect,
                capability.plan,
                capability.run.label(),
                capability.notes
            )
        })
        .unwrap_or_else(|| {
            "inspection=true planning=true execution=not-yet (unknown execution architecture)"
                .to_string()
        });

    Err(format!(
        "unsupported execution architecture '{}': {}; currently supported execution architectures: [{}]. GGUF inspection and planning remain available even when execution is unsupported",
        architecture, detail, supported
    ))
}

pub fn ggml_type_supported_for_inference(ggml_type: GgmlType) -> bool {
    matches!(
        ggml_type,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q4_0
            | GgmlType::Q5_0
            | GgmlType::Q8_0
            | GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_K
    )
}

/// Names of the tensor formats the legacy runtime can execute. This is the
/// single source of truth used by both the registry and model-aware checks.
pub fn supported_inference_format_names() -> Vec<&'static str> {
    QUANTIZATIONS
        .iter()
        .filter(|capability| capability.inference)
        .map(|capability| capability.name)
        .collect()
}

/// Unique tensor formats in `model` that the legacy runtime cannot execute.
/// Ordering follows the tensor order in the GGUF file for stable diagnostics.
pub fn unsupported_inference_tensor_formats(model: &GgufModel) -> Vec<String> {
    let mut unsupported = Vec::new();
    for tensor in &model.tensors {
        if !ggml_type_supported_for_inference(tensor.ggml_type) {
            let name = tensor.ggml_type.name();
            if !unsupported.contains(&name) {
                unsupported.push(name);
            }
        }
    }
    unsupported
}

/// Model-specific execution support. Architecture-level registry answers are
/// not enough: a supported architecture can still contain a tensor format the
/// legacy runtime cannot decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelExecutionSupport {
    pub architecture: Option<String>,
    pub architecture_run: RunSupport,
    pub unsupported_tensor_formats: Vec<String>,
}

impl ModelExecutionSupport {
    pub fn is_supported(&self) -> bool {
        self.architecture_run == RunSupport::Supported && self.unsupported_tensor_formats.is_empty()
    }

    pub fn label(&self) -> &'static str {
        if self.is_supported() {
            "yes"
        } else {
            "no"
        }
    }
}

pub fn model_execution_support(model: &GgufModel) -> ModelExecutionSupport {
    let architecture = model.info().architecture;
    let architecture_run = architecture
        .as_deref()
        .and_then(architecture_capability)
        .map(|capability| capability.run)
        .unwrap_or(RunSupport::NotYet);
    ModelExecutionSupport {
        architecture,
        architecture_run,
        unsupported_tensor_formats: unsupported_inference_tensor_formats(model),
    }
}

/// Preflight used by the legacy execution path. It fails before any tensor
/// payload is read when the architecture or a tensor format is unsupported.
pub fn ensure_model_execution_supported(model: &GgufModel) -> Result<(), String> {
    let architecture = model
        .info()
        .architecture
        .unwrap_or_else(|| "unknown".to_string());
    ensure_execution_supported(&architecture)?;

    let unsupported = unsupported_inference_tensor_formats(model);
    if !unsupported.is_empty() {
        return Err(format!(
            "unsupported tensor type(s) for legacy execution: {} (supported: {})",
            unsupported.join(", "),
            supported_inference_format_names().join(", ")
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use ramforge_core::{MetadataValue, TensorDescriptor};

    use super::*;

    fn model_with_tensor_type(architecture: &str, ggml_type: GgmlType) -> GgufModel {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            MetadataValue::String(architecture.to_string()),
        );
        let byte_length = match ggml_type {
            GgmlType::Q4_1 => Some(80),
            GgmlType::Q5_0 => Some(88),
            _ => Some(128),
        };
        GgufModel {
            path: PathBuf::from("/models/tiny.gguf"),
            file_size: byte_length.unwrap_or(0),
            version: 3,
            metadata,
            tensors: vec![TensorDescriptor {
                name: "blk.0.ffn_gate.weight".to_string(),
                dimensions: vec![8, 16],
                ggml_type,
                offset: 0,
                file_offset: 0,
                byte_length,
                num_elements: 128,
            }],
            alignment: 32,
            data_start_offset: 0,
        }
    }

    #[test]
    fn test_supported_execution_architectures() {
        assert!(ensure_execution_supported("llama").is_ok());
        assert!(ensure_execution_supported("qwen2").is_ok());
        assert_eq!(execution_architectures(), vec!["llama", "qwen2"]);
    }

    #[test]
    fn test_qwen35_is_inspectable_but_not_runnable() {
        let capability = architecture_capability("qwen3.5").unwrap();
        assert!(capability.inspect);
        assert!(capability.plan);
        assert_eq!(capability.run, RunSupport::NotYet);
        let error = ensure_execution_supported("qwen35").unwrap_err();
        assert!(error.contains("hybrid attention/SSM"));
        assert!(error.contains("inspection=true"));
        assert!(error.contains("currently supported execution architectures"));
    }

    #[test]
    fn test_mistral_requires_llama_compatible_gguf_declaration() {
        let capability = architecture_capability("mistral").unwrap();
        assert_eq!(capability.run, RunSupport::ViaLlamaGguf);
        assert!(ensure_execution_supported("mistral").is_err());
    }

    #[test]
    fn test_quantization_registry_matches_runtime_types() {
        assert!(ggml_type_supported_for_inference(GgmlType::Q4_K));
        assert!(ggml_type_supported_for_inference(GgmlType::F32));
        assert!(ggml_type_supported_for_inference(GgmlType::Q5_0));
        assert!(!ggml_type_supported_for_inference(GgmlType::Q4_1));
    }

    #[test]
    fn test_supported_format_names_match_registry() {
        let names = supported_inference_format_names();
        assert!(names.contains(&"Q4_0"));
        assert!(names.contains(&"Q5_0"));
        assert!(names.contains(&"Q5_K"));
        assert!(!names.contains(&"Q4_1"));
        assert!(!names.contains(&"other GGUF types"));
    }

    #[test]
    fn test_model_specific_execution_rejects_unsupported_tensor_format() {
        let model = model_with_tensor_type("llama", GgmlType::Q4_1);
        let support = model_execution_support(&model);
        assert!(!support.is_supported());
        assert_eq!(support.architecture_run, RunSupport::Supported);
        assert_eq!(support.unsupported_tensor_formats, vec!["Q4_1"]);
        let error = ensure_model_execution_supported(&model).unwrap_err();
        assert!(error.contains("Q4_1"));
        assert!(error.contains("supported: F32"));
    }

    #[test]
    fn test_supported_architecture_and_tensor_format_is_executable() {
        let model = model_with_tensor_type("qwen2", GgmlType::Q5_0);
        let support = model_execution_support(&model);
        assert!(support.is_supported());
        assert_eq!(support.label(), "yes");
        assert!(ensure_model_execution_supported(&model).is_ok());
    }
}
