//! Explicit model and quantization capability registry.
//!
//! GGUF parsing/inspection is architecture-neutral. Execution support is
//! deliberately narrower and must never be inferred from a similar name.
//!
//! Two execution backends are tracked separately:
//!
//! * the **legacy** RAMforge executor implements a small, RAMforge-owned
//!   transformer subset (`llama`, `qwen2`) with managed out-of-core residency;
//! * the **forgeCORE/libllama** backend delegates execution to upstream
//!   llama.cpp and therefore covers the architectures that the pinned native
//!   revision genuinely implements.
//!
//! A `forgecore_run = yes` claim requires real-model execution evidence. It is
//! not inferred from an architecture identifier merely being present in
//! upstream dispatch tables.

use ramforge_core::{model::GgufModel, types::GgmlType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunSupport {
    Supported,
    NotYet,
}

impl RunSupport {
    pub fn label(self) -> &'static str {
        match self {
            Self::Supported => "yes",
            Self::NotYet => "not-yet",
        }
    }

    pub fn is_supported(self) -> bool {
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
    /// RAMforge's own managed-memory executor.
    pub legacy_run: RunSupport,
    /// forgeCORE/libllama backend, only after real-model validation.
    pub forgecore_run: RunSupport,
    pub notes: &'static str,
}

pub const ARCHITECTURES: &[ArchitectureCapability] = &[
    ArchitectureCapability {
        name: "llama",
        aliases: &[],
        inspect: true,
        plan: true,
        tokenizer: "sentencepiece/bpe",
        legacy_run: RunSupport::Supported,
        forgecore_run: RunSupport::Supported,
        notes: "dense llama-compatible transformer; validated on both backends",
    },
    ArchitectureCapability {
        name: "qwen2",
        aliases: &["qwen2.5"],
        inspect: true,
        plan: true,
        tokenizer: "bpe",
        legacy_run: RunSupport::Supported,
        forgecore_run: RunSupport::Supported,
        notes: "dense qwen2 transformer (including qwen2.5), including validated Q/K/V biases",
    },
    ArchitectureCapability {
        name: "gptneox",
        aliases: &["gpt_neox"],
        inspect: true,
        plan: true,
        tokenizer: "bpe",
        legacy_run: RunSupport::NotYet,
        forgecore_run: RunSupport::Supported,
        notes: "GPT-NeoX/Pythia parallel-residual transformer; executed through forgeCORE/libllama (use --backend forgecore)",
    },
    ArchitectureCapability {
        name: "mistral",
        aliases: &[],
        inspect: true,
        plan: true,
        tokenizer: "sentencepiece",
        legacy_run: RunSupport::NotYet,
        forgecore_run: RunSupport::NotYet,
        notes: "Mistral GGUFs normally declare general.architecture=llama; a literal 'mistral' tag has no validated execution path",
    },
    ArchitectureCapability {
        name: "qwen3",
        aliases: &[],
        inspect: true,
        plan: true,
        tokenizer: "inspect-only",
        legacy_run: RunSupport::NotYet,
        forgecore_run: RunSupport::NotYet,
        notes: "inspect/plan only; forgeCORE/libllama execution has not been validated",
    },
    ArchitectureCapability {
        name: "qwen35",
        aliases: &["qwen3.5", "qwen3_5"],
        inspect: true,
        plan: true,
        tokenizer: "inspect-only",
        legacy_run: RunSupport::NotYet,
        forgecore_run: RunSupport::NotYet,
        notes: "hybrid attention/SSM architecture; intentionally not aliased to qwen2",
    },
    ArchitectureCapability {
        name: "gemma",
        aliases: &["gemma2", "gemma3"],
        inspect: true,
        plan: true,
        tokenizer: "inspect-only",
        legacy_run: RunSupport::NotYet,
        forgecore_run: RunSupport::NotYet,
        notes: "inspect/plan only; forgeCORE/libllama execution has not been validated",
    },
    ArchitectureCapability {
        name: "phi",
        aliases: &["phi2", "phi3", "phi4"],
        inspect: true,
        plan: true,
        tokenizer: "inspect-only",
        legacy_run: RunSupport::NotYet,
        forgecore_run: RunSupport::NotYet,
        notes: "inspect/plan only; forgeCORE/libllama execution has not been validated",
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

/// Architectures executable by RAMforge's legacy managed-memory executor.
pub fn execution_architectures() -> Vec<&'static str> {
    ARCHITECTURES
        .iter()
        .filter(|capability| capability.legacy_run.is_supported())
        .map(|capability| capability.name)
        .collect()
}

/// Architectures with validated forgeCORE/libllama execution.
pub fn forgecore_execution_architectures() -> Vec<&'static str> {
    ARCHITECTURES
        .iter()
        .filter(|capability| capability.forgecore_run.is_supported())
        .map(|capability| capability.name)
        .collect()
}

/// Preflight used only by the legacy executor.
///
/// When an architecture is valid through the ForgeCore/libllama backend, the
/// error names that backend explicitly instead of implying that no execution
/// path exists. It never switches backends on the caller's behalf.
pub fn ensure_execution_supported(architecture: &str) -> Result<(), String> {
    let Some(capability) = architecture_capability(architecture) else {
        return Err(format!(
            "unsupported execution architecture '{}': inspection=true planning=true legacy-execution=not-yet forgecore-execution=not-yet (unknown execution architecture); currently supported legacy execution architectures: [{}]. GGUF inspection and planning remain available even when execution is unsupported",
            architecture,
            execution_architectures().join(", ")
        ));
    };

    if capability.legacy_run.is_supported() {
        return Ok(());
    }

    let detail = format!(
        "inspection={} planning={} legacy-execution={} forgecore-execution={} ({})",
        capability.inspect,
        capability.plan,
        capability.legacy_run.label(),
        capability.forgecore_run.label(),
        capability.notes
    );

    if capability.forgecore_run.is_supported() {
        return Err(format!(
            "architecture '{}' is not supported by RAMforge's legacy executor, but it is validated for execution through the forgeCORE/libllama backend: {}; re-run with `--backend forgecore`. Legacy execution architectures: [{}]. GGUF inspection and planning remain available",
            architecture,
            detail,
            execution_architectures().join(", ")
        ));
    }

    Err(format!(
        "unsupported execution architecture '{}': {}; currently supported legacy execution architectures: [{}]. GGUF inspection and planning remain available even when execution is unsupported",
        architecture,
        detail,
        execution_architectures().join(", ")
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
/// not enough: a legacy-supported architecture can still contain a tensor
/// format the legacy runtime cannot decode.
///
/// `unsupported_tensor_formats` is a legacy-executor concept. The
/// forgeCORE/libllama backend delegates tensor-type support to the pinned
/// native runtime and must not be labelled unsupported from this list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelExecutionSupport {
    pub architecture: Option<String>,
    pub legacy_run: RunSupport,
    pub forgecore_run: RunSupport,
    pub unsupported_tensor_formats: Vec<String>,
}

impl ModelExecutionSupport {
    /// Legacy backend + every tensor format decodable by RAMforge's kernels.
    pub fn is_legacy_supported(&self) -> bool {
        self.legacy_run == RunSupport::Supported && self.unsupported_tensor_formats.is_empty()
    }

    /// Architecture-level claim for forgeCORE/libllama. Tensor-type support is
    /// the pinned native backend's responsibility, not this registry's.
    pub fn is_forgecore_supported(&self) -> bool {
        self.forgecore_run == RunSupport::Supported
    }

    /// Legacy label retained for existing inspect/plan JSON consumers.
    pub fn label(&self) -> &'static str {
        if self.is_legacy_supported() {
            "yes"
        } else {
            "no"
        }
    }
}

pub fn model_execution_support(model: &GgufModel) -> ModelExecutionSupport {
    let architecture = model.info().architecture;
    let capability = architecture.as_deref().and_then(architecture_capability);
    ModelExecutionSupport {
        architecture,
        legacy_run: capability
            .map(|capability| capability.legacy_run)
            .unwrap_or(RunSupport::NotYet),
        forgecore_run: capability
            .map(|capability| capability.forgecore_run)
            .unwrap_or(RunSupport::NotYet),
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
    fn test_gptneox_registry_separates_legacy_from_forgecore() {
        let capability = architecture_capability("gptneox").unwrap();
        assert!(capability.inspect);
        assert!(capability.plan);
        assert_eq!(capability.legacy_run, RunSupport::NotYet);
        assert_eq!(capability.forgecore_run, RunSupport::Supported);
        assert!(forgecore_execution_architectures().contains(&"gptneox"));

        let model = model_with_tensor_type("gptneox", GgmlType::Q8_0);
        let support = model_execution_support(&model);
        assert!(!support.is_legacy_supported());
        assert!(support.is_forgecore_supported());
        assert_eq!(support.label(), "no");

        let error = ensure_execution_supported("gptneox").unwrap_err();
        assert!(error.contains("forgeCORE/libllama"));
        assert!(error.contains("--backend forgecore"));
        assert!(error.contains("legacy-execution=not-yet"));
        assert!(error.contains("forgecore-execution=yes"));
        // The message must still contain the phrase handled by the smoke
        // harness's unsupported-architecture classifier.
        assert!(error.contains("not supported by RAMforge's legacy executor"));
    }

    #[test]
    fn test_gpt_neox_alias_resolves_to_the_same_capability() {
        assert_eq!(
            architecture_capability("gpt_neox").map(|capability| capability.name),
            Some("gptneox")
        );
    }

    #[test]
    fn test_qwen35_is_inspectable_but_not_runnable() {
        let capability = architecture_capability("qwen3.5").unwrap();
        assert!(capability.inspect);
        assert!(capability.plan);
        assert_eq!(capability.legacy_run, RunSupport::NotYet);
        assert_eq!(capability.forgecore_run, RunSupport::NotYet);
        let error = ensure_execution_supported("qwen35").unwrap_err();
        assert!(error.contains("hybrid attention/SSM"));
        assert!(error.contains("inspection=true"));
        assert!(error.contains("currently supported legacy execution architectures"));
    }

    #[test]
    fn test_mistral_literal_tag_is_not_claimed_runnable() {
        let capability = architecture_capability("mistral").unwrap();
        assert_eq!(capability.legacy_run, RunSupport::NotYet);
        assert_eq!(capability.forgecore_run, RunSupport::NotYet);
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
        assert!(!support.is_legacy_supported());
        assert_eq!(support.legacy_run, RunSupport::Supported);
        assert_eq!(support.unsupported_tensor_formats, vec!["Q4_1"]);
        let error = ensure_model_execution_supported(&model).unwrap_err();
        assert!(error.contains("Q4_1"));
        assert!(error.contains("supported: F32"));
    }

    #[test]
    fn test_supported_architecture_and_tensor_format_is_executable() {
        let model = model_with_tensor_type("qwen2", GgmlType::Q5_0);
        let support = model_execution_support(&model);
        assert!(support.is_legacy_supported());
        assert!(support.is_forgecore_supported());
        assert_eq!(support.label(), "yes");
        assert!(ensure_model_execution_supported(&model).is_ok());
    }
}
