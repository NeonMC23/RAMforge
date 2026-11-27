//! Feature-level model/backend compatibility resolution.
//!
//! A recognized architecture is not an execution guarantee. This module
//! resolves the features a proposed execution strategy actually needs and
//! reports each one as fully supported, limited, unvalidated, unsupported, or
//! insufficiently informed. The result is advisory input for planning; it never
//! starts execution.

use serde::{Deserialize, Serialize};

use crate::model_architecture::ArchitectureKind;
use crate::profiling::{HardwareProfile, ModelProfile, PerformanceRecord, WorkloadProfile};

pub const CAPABILITY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportClass {
    FullySupported,
    SupportedWithLimitations,
    SupportedUnvalidated,
    Unsupported,
    InsufficientInformation,
}

impl SupportClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::FullySupported => "fully-supported",
            Self::SupportedWithLimitations => "supported-with-limitations",
            Self::SupportedUnvalidated => "supported-unvalidated",
            Self::Unsupported => "unsupported",
            Self::InsufficientInformation => "insufficient-information",
        }
    }

    fn severity(self) -> u8 {
        match self {
            Self::FullySupported => 0,
            Self::SupportedWithLimitations => 1,
            Self::SupportedUnvalidated => 2,
            Self::InsufficientInformation => 3,
            Self::Unsupported => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionFeature {
    CpuExecution,
    GpuOffload,
    HybridPlacement,
    StagedResidency,
    SpeculativeGreedy,
    SpeculativeSampling,
    MoeExecution,
    KvCacheQuantization,
    FlashAttention,
    GpuPlacementReadback,
    DraftVocabularyCompatibility,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceSource {
    GgufMetadata,
    SupportRegistry,
    MachineObservation,
    NativeBackend,
    MeasuredProfile,
    HeuristicEstimate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureSupport {
    pub feature: ExecutionFeature,
    pub class: SupportClass,
    pub detail: String,
    pub provenance: Vec<EvidenceSource>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompatibilityRequest {
    pub device_id: String,
    pub gpu_layers: u32,
    pub staged: bool,
    pub speculative: bool,
    pub draft_compatible: Option<bool>,
    pub kv_type_k: String,
    pub kv_type_v: String,
    /// Reserved for future sampling-preserving speculative paths. The current
    /// implementation is greedy-only.
    pub sampling_greedy: bool,
}

impl CompatibilityRequest {
    pub fn from_workload(workload: &WorkloadProfile, draft_compatible: Option<bool>) -> Self {
        Self {
            device_id: workload.device_id.clone(),
            gpu_layers: workload.gpu_layers,
            staged: workload.residency == "staged",
            speculative: workload.draft_tokens.is_some(),
            draft_compatible,
            kv_type_k: workload.kv_type_k.clone(),
            kv_type_v: workload.kv_type_v.clone(),
            sampling_greedy: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompatibilityReport {
    pub schema_version: u32,
    pub overall: SupportClass,
    pub features: Vec<FeatureSupport>,
    pub reasons: Vec<String>,
    pub unknowns: Vec<String>,
}

impl CompatibilityReport {
    pub fn feature(&self, feature: ExecutionFeature) -> Option<&FeatureSupport> {
        self.features.iter().find(|entry| entry.feature == feature)
    }

    pub fn is_supported(&self) -> bool {
        matches!(
            self.overall,
            SupportClass::FullySupported
                | SupportClass::SupportedWithLimitations
                | SupportClass::SupportedUnvalidated
        )
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != CAPABILITY_SCHEMA_VERSION {
            return Err(format!(
                "unsupported capability schema {}",
                self.schema_version
            ));
        }
        if self.features.is_empty() {
            return Err("compatibility report has no features".to_string());
        }
        for feature in &self.features {
            if feature.detail.trim().is_empty() {
                return Err("compatibility feature has an empty detail".to_string());
            }
        }
        Ok(())
    }
}

/// Resolve the features required by `request` for this model/hardware/workload.
pub fn resolve_compatibility(
    model: &ModelProfile,
    hardware: &HardwareProfile,
    workload: &WorkloadProfile,
    request: &CompatibilityRequest,
) -> CompatibilityReport {
    resolve_compatibility_with_evidence(model, hardware, workload, request, None)
}

/// Evidence-aware resolution. A persisted record that matches the model and
/// hardware fingerprint upgrades "unvalidated" claims to "limited" and names
/// the measurement; it never claims the recorded workload matches the current
/// one.
pub fn resolve_compatibility_with_evidence(
    model: &ModelProfile,
    hardware: &HardwareProfile,
    workload: &WorkloadProfile,
    request: &CompatibilityRequest,
    measured: Option<&PerformanceRecord>,
) -> CompatibilityReport {
    let mut features = Vec::new();
    let mut reasons = Vec::new();
    let mut unknowns = Vec::new();

    let forgecore = model.forgecore_execution_supported;
    let legacy = model.legacy_execution_supported;
    let gpu = hardware.gpu_available();
    let gpu_requested = request.gpu_layers > 0 || !request.device_id.eq_ignore_ascii_case("cpu");

    if forgecore {
        features.push(FeatureSupport {
            feature: ExecutionFeature::CpuExecution,
            class: SupportClass::FullySupported,
            detail: "architecture is in the validated forgeCORE/libllama execution registry"
                .to_string(),
            provenance: vec![
                EvidenceSource::GgufMetadata,
                EvidenceSource::SupportRegistry,
            ],
        });
    } else if legacy {
        features.push(FeatureSupport {
            feature: ExecutionFeature::CpuExecution,
            class: SupportClass::SupportedWithLimitations,
            detail: "only RAMforge's legacy managed-memory executor is registered; the benchmark and service path use forgeCORE"
                .to_string(),
            provenance: vec![
                EvidenceSource::GgufMetadata,
                EvidenceSource::SupportRegistry,
            ],
        });
    } else {
        features.push(FeatureSupport {
            feature: ExecutionFeature::CpuExecution,
            class: SupportClass::Unsupported,
            detail: "no validated execution registry entry for this architecture".to_string(),
            provenance: vec![
                EvidenceSource::GgufMetadata,
                EvidenceSource::SupportRegistry,
            ],
        });
    }

    if gpu_requested {
        if !gpu {
            features.push(FeatureSupport {
                feature: ExecutionFeature::GpuOffload,
                class: SupportClass::Unsupported,
                detail: "no selectable GPU device is available".to_string(),
                provenance: vec![EvidenceSource::MachineObservation],
            });
        } else if !forgecore {
            features.push(FeatureSupport {
                feature: ExecutionFeature::GpuOffload,
                class: SupportClass::Unsupported,
                detail: "GPU offload requires a forgeCORE-supported architecture".to_string(),
                provenance: vec![
                    EvidenceSource::MachineObservation,
                    EvidenceSource::SupportRegistry,
                ],
            });
        } else {
            features.push(FeatureSupport {
                feature: ExecutionFeature::GpuOffload,
                class: SupportClass::SupportedUnvalidated,
                detail: "native GPU placement is available, but this model/device combination has no recorded inference validation in this milestone"
                    .to_string(),
                provenance: vec![
                    EvidenceSource::MachineObservation,
                    EvidenceSource::NativeBackend,
                ],
            });
            features.push(FeatureSupport {
                feature: ExecutionFeature::GpuPlacementReadback,
                class: SupportClass::SupportedWithLimitations,
                detail: "requested layers are accepted only when native readback reports the same offloaded layer count"
                    .to_string(),
                provenance: vec![EvidenceSource::NativeBackend],
            });
        }
    }

    if request.staged {
        if !gpu {
            features.push(FeatureSupport {
                feature: ExecutionFeature::StagedResidency,
                class: SupportClass::Unsupported,
                detail: "staged residency requires a selectable GPU device".to_string(),
                provenance: vec![EvidenceSource::MachineObservation],
            });
        } else if !forgecore {
            features.push(FeatureSupport {
                feature: ExecutionFeature::StagedResidency,
                class: SupportClass::Unsupported,
                detail: "staged residency requires the forgeCORE execution path".to_string(),
                provenance: vec![EvidenceSource::SupportRegistry],
            });
        } else if model.kind == ArchitectureKind::Unknown {
            features.push(FeatureSupport {
                feature: ExecutionFeature::StagedResidency,
                class: SupportClass::InsufficientInformation,
                detail:
                    "dense/MoE classification is unknown; staging eligibility cannot be decided"
                        .to_string(),
                provenance: vec![EvidenceSource::GgufMetadata],
            });
            unknowns.push("staged residency needs a known dense/MoE classification".to_string());
        } else {
            features.push(FeatureSupport {
                feature: ExecutionFeature::StagedResidency,
                class: if model.kind == ArchitectureKind::Dense {
                    SupportClass::SupportedWithLimitations
                } else {
                    SupportClass::SupportedUnvalidated
                },
                detail: if model.kind == ArchitectureKind::Dense {
                    "bounded layer staging exists for dense models; benefit depends on measured VRAM and storage throughput"
                        .to_string()
                } else {
                    "staging is generic in the runtime but MoE routing locality has not been measured; treat as unvalidated"
                        .to_string()
                },
                provenance: vec![
                    EvidenceSource::NativeBackend,
                    EvidenceSource::HeuristicEstimate,
                ],
            });
        }
    }

    if model.kind == ArchitectureKind::MixtureOfExperts {
        features.push(FeatureSupport {
            feature: ExecutionFeature::MoeExecution,
            class: SupportClass::SupportedWithLimitations,
            detail: "architecture is registered, but active-expert bytes and routing locality remain metadata estimates"
                .to_string(),
            provenance: vec![
                EvidenceSource::GgufMetadata,
                EvidenceSource::SupportRegistry,
            ],
        });
    }

    let kv_quantized = !request.kv_type_k.eq_ignore_ascii_case("f16")
        && !request.kv_type_k.eq_ignore_ascii_case("f32")
        || !request.kv_type_v.eq_ignore_ascii_case("f16")
            && !request.kv_type_v.eq_ignore_ascii_case("f32");
    if kv_quantized {
        features.push(FeatureSupport {
            feature: ExecutionFeature::KvCacheQuantization,
            class: SupportClass::SupportedWithLimitations,
            detail: format!(
                "KV cache types K={} V={} are native-supported but trade memory for numerical fidelity; no quality equivalence is claimed",
                request.kv_type_k, request.kv_type_v
            ),
            provenance: vec![EvidenceSource::NativeBackend],
        });
    }

    if !request.sampling_greedy {
        features.push(FeatureSupport {
            feature: ExecutionFeature::SpeculativeSampling,
            class: SupportClass::Unsupported,
            detail: "distribution-preserving speculative sampling is not implemented; only greedy verification exists"
                .to_string(),
            provenance: vec![EvidenceSource::SupportRegistry],
        });
    }

    if request.speculative {
        if !request.sampling_greedy {
            features.push(FeatureSupport {
                feature: ExecutionFeature::SpeculativeGreedy,
                class: SupportClass::Unsupported,
                detail: "speculative decoding requires greedy sampling".to_string(),
                provenance: vec![EvidenceSource::SupportRegistry],
            });
        } else if request.draft_compatible == Some(false) {
            features.push(FeatureSupport {
                feature: ExecutionFeature::SpeculativeGreedy,
                class: SupportClass::Unsupported,
                detail: "draft tokenizer/merge compatibility check failed".to_string(),
                provenance: vec![EvidenceSource::GgufMetadata],
            });
        } else if request.draft_compatible.is_none() {
            features.push(FeatureSupport {
                feature: ExecutionFeature::SpeculativeGreedy,
                class: SupportClass::InsufficientInformation,
                detail: "no draft compatibility result was supplied".to_string(),
                provenance: vec![EvidenceSource::GgufMetadata],
            });
            unknowns.push("draft compatibility was not evaluated".to_string());
        } else if request.gpu_layers > 0 {
            features.push(FeatureSupport {
                feature: ExecutionFeature::SpeculativeGreedy,
                class: SupportClass::Unsupported,
                detail: "the current speculative path is CPU-target-only; GPU-targeted verification is not implemented"
                    .to_string(),
                provenance: vec![EvidenceSource::SupportRegistry],
            });
        } else if !forgecore {
            features.push(FeatureSupport {
                feature: ExecutionFeature::SpeculativeGreedy,
                class: SupportClass::Unsupported,
                detail: "speculative decoding requires a forgeCORE-supported target architecture"
                    .to_string(),
                provenance: vec![EvidenceSource::SupportRegistry],
            });
        } else {
            features.push(FeatureSupport {
                feature: ExecutionFeature::SpeculativeGreedy,
                class: SupportClass::SupportedWithLimitations,
                detail: format!(
                    "greedy speculative decoding is supported at K={}; correctness is provided by exact token-id verification, and acceptance/throughput must be measured",
                    workload.draft_tokens.unwrap_or(0)
                ),
                provenance: vec![
                    EvidenceSource::SupportRegistry,
                    EvidenceSource::GgufMetadata,
                ],
            });
        }
        if request.draft_compatible == Some(true) {
            features.push(FeatureSupport {
                feature: ExecutionFeature::DraftVocabularyCompatibility,
                class: SupportClass::FullySupported,
                detail: "draft tokenizer tokens/merges/types are a verified prefix of the target"
                    .to_string(),
                provenance: vec![EvidenceSource::GgufMetadata],
            });
        }
    }

    if let Some(record) = measured {
        let matching = record.identity.model_fingerprint == model.fingerprint
            && record.identity.hardware_fingerprint == hardware.fingerprint;
        if matching {
            for feature in features.iter_mut() {
                if feature.class != SupportClass::SupportedUnvalidated {
                    continue;
                }
                if matches!(
                    feature.feature,
                    ExecutionFeature::GpuOffload
                        | ExecutionFeature::StagedResidency
                        | ExecutionFeature::MoeExecution
                ) {
                    feature.class = SupportClass::SupportedWithLimitations;
                    feature.detail = format!(
                        "{}; a measured record exists for this model and hardware (workload may differ)",
                        feature.detail
                    );
                    feature.provenance.push(EvidenceSource::MeasuredProfile);
                }
            }
        }
    }

    let overall = features
        .iter()
        .map(|feature| feature.class)
        .max_by_key(|class| class.severity())
        .unwrap_or(SupportClass::InsufficientInformation);

    for feature in &features {
        match feature.class {
            SupportClass::Unsupported => reasons.push(format!(
                "{}: {}",
                feature_label(feature.feature),
                feature.detail
            )),
            SupportClass::InsufficientInformation => reasons.push(format!(
                "{}: {}",
                feature_label(feature.feature),
                feature.detail
            )),
            _ => {}
        }
    }

    CompatibilityReport {
        schema_version: CAPABILITY_SCHEMA_VERSION,
        overall,
        features,
        reasons,
        unknowns,
    }
}

pub fn feature_label(feature: ExecutionFeature) -> &'static str {
    match feature {
        ExecutionFeature::CpuExecution => "cpu-execution",
        ExecutionFeature::GpuOffload => "gpu-offload",
        ExecutionFeature::HybridPlacement => "hybrid-placement",
        ExecutionFeature::StagedResidency => "staged-residency",
        ExecutionFeature::SpeculativeGreedy => "speculative-greedy",
        ExecutionFeature::SpeculativeSampling => "speculative-sampling",
        ExecutionFeature::MoeExecution => "moe-execution",
        ExecutionFeature::KvCacheQuantization => "kv-cache-quantization",
        ExecutionFeature::FlashAttention => "flash-attention",
        ExecutionFeature::GpuPlacementReadback => "gpu-placement-readback",
        ExecutionFeature::DraftVocabularyCompatibility => "draft-vocabulary-compatibility",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_architecture::ArchitectureKind;

    fn model(kind: ArchitectureKind, forgecore: bool) -> ModelProfile {
        ModelProfile {
            schema_version: 2,
            fingerprint: Some("model".to_string()),
            path: None,
            architecture: Some("qwen2".to_string()),
            kind,
            parameters: Some(1),
            file_bytes: 1024,
            tensor_payload_bytes: Some(1024),
            tensor_count: 1,
            layer_count: Some(24),
            embedding_length: Some(128),
            head_count: Some(4),
            head_count_kv: Some(4),
            context_max: Some(4096),
            vocab_size: Some(100),
            tokenizer_model: Some("bpe".to_string()),
            expert_count: None,
            expert_used_count: None,
            quantization: Some("mostly-q4_k_m".to_string()),
            quantization_mix: Vec::new(),
            legacy_execution_supported: true,
            forgecore_execution_supported: forgecore,
            unsupported_tensor_formats: Vec::new(),
            routed_expert_bytes: None,
            estimated_active_weight_bytes: None,
            notes: Vec::new(),
        }
    }

    fn hardware(gpu: bool) -> HardwareProfile {
        HardwareProfile {
            schema_version: 2,
            fingerprint: "hw".to_string(),
            cpu_model: Some("cpu".to_string()),
            physical_cores: Some(6),
            logical_cpus: Some(12),
            total_ram_bytes: Some(16 << 30),
            available_ram_bytes: Some(8 << 30),
            os: "linux".to_string(),
            kernel: None,
            gpus: if gpu {
                vec![crate::profiling::GpuProfile {
                    id: "gpu0".to_string(),
                    backend: "cuda".to_string(),
                    name: "gpu".to_string(),
                    total_bytes: Some(4 << 30),
                    free_bytes: Some(4 << 30),
                    selectable: true,
                }]
            } else {
                Vec::new()
            },
            backends: vec!["cpu".to_string()],
            storage_kind: None,
            storage_read_bytes_per_second: None,
            cgroup_limit_bytes: None,
            execution_environment: crate::adaptive::environment::ExecutionEnvironment {
                schema_version: 1,
                os: "linux".to_string(),
                kernel: None,
                ramforge_version: "0.1.0".to_string(),
                native_contract_version: 1,
                native_revision: None,
                backend_build: "cpu".to_string(),
                gpu_driver_version: None,
                cpu_feature_fingerprint: None,
                fingerprint: "0000000000000000".to_string(),
            },
            notes: Vec::new(),
        }
    }

    fn workload(gpu_layers: u32, staged: bool, speculative: bool) -> WorkloadProfile {
        WorkloadProfile {
            context_tokens: 256,
            batch_tokens: 64,
            ubatch_tokens: 64,
            sequence_max: 1,
            threads: 6,
            gpu_layers,
            kv_type_k: "f16".to_string(),
            kv_type_v: "f16".to_string(),
            load_mode: "mmap".to_string(),
            no_repack: false,
            backend: "cpu".to_string(),
            device_id: if gpu_layers > 0 { "gpu0" } else { "cpu" }.to_string(),
            residency: if staged { "staged" } else { "off" }.to_string(),
            draft_model_fingerprint: None,
            draft_tokens: speculative.then_some(8),
            max_tokens: Some(8),
            ram_budget_bytes: Some(10 << 30),
        }
    }

    fn request(workload: &WorkloadProfile, draft: Option<bool>) -> CompatibilityRequest {
        CompatibilityRequest::from_workload(workload, draft)
    }

    #[test]
    fn cpu_baseline_is_supported() {
        let m = model(ArchitectureKind::Dense, true);
        let h = hardware(false);
        let w = workload(0, false, false);
        let report = resolve_compatibility(&m, &h, &w, &request(&w, None));
        assert_eq!(report.overall, SupportClass::FullySupported);
        assert!(report.is_supported());
    }

    #[test]
    fn gpu_without_device_is_unsupported() {
        let m = model(ArchitectureKind::Dense, true);
        let h = hardware(false);
        let w = workload(10, false, false);
        let report = resolve_compatibility(&m, &h, &w, &request(&w, None));
        assert_eq!(report.overall, SupportClass::Unsupported);
    }

    #[test]
    fn gpu_target_makes_speculation_unsupported() {
        let m = model(ArchitectureKind::Dense, true);
        let h = hardware(true);
        let w = workload(10, false, true);
        let report = resolve_compatibility(&m, &h, &w, &request(&w, Some(true)));
        assert_eq!(report.overall, SupportClass::Unsupported);
        assert!(report
            .reasons
            .iter()
            .any(|reason| reason.contains("CPU-target-only")));
    }

    #[test]
    fn speculative_sampling_is_explicitly_unsupported() {
        let m = model(ArchitectureKind::Dense, true);
        let h = hardware(false);
        let w = workload(0, false, false);
        let mut r = request(&w, None);
        r.sampling_greedy = false;
        let report = resolve_compatibility(&m, &h, &w, &r);
        assert_eq!(report.overall, SupportClass::Unsupported);
    }

    #[test]
    fn measured_record_upgrades_unvalidated_gpu_claim() {
        let model = model(ArchitectureKind::Dense, true);
        let hardware = hardware(true);
        let workload = workload(10, false, false);
        let request = request(&workload, None);
        let baseline = resolve_compatibility(&model, &hardware, &workload, &request);
        assert_eq!(baseline.overall, SupportClass::SupportedUnvalidated);

        let record = crate::profiling::PerformanceRecord {
            identity: crate::profiling::PerformanceIdentity {
                model_fingerprint: model.fingerprint.clone(),
                architecture: model.architecture.clone(),
                quantization: model.quantization.clone(),
                hardware_fingerprint: hardware.fingerprint.clone(),
                backend: "gpu".to_string(),
                device_id: "gpu0".to_string(),
                workload: workload.clone(),
                runtime_schema: crate::profiling::PROFILING_SCHEMA_VERSION,
            },
            captured_at_unix_ms: Some(1),
            generation_tokens_per_second: 10.0,
            decode_tokens_per_second: None,
            prompt_tokens_per_second: None,
            time_to_first_token_ns: None,
            total_generation_ns: 1,
            emitted_tokens: 4,
            target_forwards: None,
            accepted_tokens: None,
            draft_tokens_generated: None,
            acceptance_rate_bps: None,
            storage_read_bytes: None,
            pss_bytes: None,
            target_forward_ns: None,
            draft_forward_ns: None,
            cpu_busy_bps: None,
            staged_vram_bytes: None,
            condition: "warm".to_string(),
            limitations: Vec::new(),
        };
        let report = resolve_compatibility_with_evidence(
            &model,
            &hardware,
            &workload,
            &request,
            Some(&record),
        );
        assert_eq!(report.overall, SupportClass::SupportedWithLimitations);
        assert!(report.features.iter().any(|feature| {
            feature.feature == ExecutionFeature::GpuOffload
                && feature
                    .provenance
                    .contains(&EvidenceSource::MeasuredProfile)
        }));
    }

    #[test]
    fn kv_quantization_is_a_reported_limitation() {
        let m = model(ArchitectureKind::Dense, true);
        let h = hardware(false);
        let mut w = workload(0, false, false);
        w.kv_type_k = "q8_0".to_string();
        w.kv_type_v = "q8_0".to_string();
        let report = resolve_compatibility(&m, &h, &w, &request(&w, None));
        assert_eq!(report.overall, SupportClass::SupportedWithLimitations);
    }
}
