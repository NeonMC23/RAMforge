//! Explicit model and quantization capability registry.
//!
//! GGUF parsing/inspection is architecture-neutral. Execution support is
//! deliberately narrower and must never be inferred from a similar name.

use ramforge_core::types::GgmlType;

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
        aliases: &[],
        inspect: true,
        plan: true,
        tokenizer: "bpe",
        run: RunSupport::Supported,
        notes: "dense qwen2 transformer, including validated Q/K/V biases",
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
    QuantizationCapability { name: "F32", inference: true },
    QuantizationCapability { name: "F16", inference: true },
    QuantizationCapability { name: "BF16", inference: true },
    QuantizationCapability { name: "Q4_0", inference: true },
    QuantizationCapability { name: "Q8_0", inference: true },
    QuantizationCapability { name: "Q2_K", inference: true },
    QuantizationCapability { name: "Q3_K", inference: true },
    QuantizationCapability { name: "Q4_K", inference: true },
    QuantizationCapability { name: "Q5_K", inference: true },
    QuantizationCapability { name: "Q6_K", inference: true },
    QuantizationCapability { name: "Q8_K", inference: true },
    QuantizationCapability { name: "other GGUF types", inference: false },
];

pub fn architecture_capability(name: &str) -> Option<&'static ArchitectureCapability> {
    let normalized = name.to_ascii_lowercase();
    ARCHITECTURES.iter().find(|capability| {
        capability.name == normalized.as_str()
            || capability.aliases.contains(&normalized.as_str())
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
            | GgmlType::Q8_0
            | GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_K
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(!ggml_type_supported_for_inference(GgmlType::Q4_1));
    }
}
