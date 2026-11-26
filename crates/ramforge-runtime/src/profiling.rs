//! Model-agnostic profiling, performance identity, and simple adaptive advice.
//!
//! Profiles separate model facts, hardware facts, workload configuration, and
//! measured results. A measurement is reusable only when the model fingerprint,
//! hardware fingerprint, and workload configuration match exactly. Estimates are
//! never presented as measurements.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use ramforge_core::GgufModel;

use crate::benchmark_runner::BenchmarkReport;
use crate::capacity::expert_accounting;
use crate::machine_profile::MachineProfile;
use crate::model_architecture::{inspect_architecture, ArchitectureKind};
use crate::model_requirements::{model_file_fingerprint, quantization_label};
use crate::runtime_plan::RuntimePlan;
use crate::support::model_execution_support;

pub const PROFILING_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationBucket {
    pub name: String,
    pub tensors: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelProfile {
    pub schema_version: u32,
    pub fingerprint: Option<String>,
    pub path: Option<String>,
    pub architecture: Option<String>,
    pub kind: ArchitectureKind,
    pub parameters: Option<u64>,
    pub file_bytes: u64,
    pub tensor_payload_bytes: Option<u64>,
    pub tensor_count: usize,
    pub layer_count: Option<u64>,
    pub embedding_length: Option<u64>,
    pub head_count: Option<u64>,
    pub head_count_kv: Option<u64>,
    pub context_max: Option<u64>,
    pub vocab_size: Option<u64>,
    pub tokenizer_model: Option<String>,
    pub expert_count: Option<u64>,
    pub expert_used_count: Option<u64>,
    pub quantization: Option<String>,
    pub quantization_mix: Vec<QuantizationBucket>,
    pub legacy_execution_supported: bool,
    pub forgecore_execution_supported: bool,
    pub unsupported_tensor_formats: Vec<String>,
    pub routed_expert_bytes: Option<u64>,
    pub estimated_active_weight_bytes: Option<u64>,
    pub notes: Vec<String>,
}

pub fn model_profile(model: &GgufModel) -> ModelProfile {
    let info = model.info();
    let architecture = inspect_architecture(model);
    let support = model_execution_support(model);
    let parameters = model
        .tensors
        .iter()
        .try_fold(0u64, |total, tensor| total.checked_add(tensor.num_elements));
    let mut mix: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for tensor in &model.tensors {
        let entry = mix.entry(tensor.ggml_type.name()).or_default();
        entry.0 += 1;
        entry.1 = entry.1.saturating_add(tensor.byte_length.unwrap_or(0));
    }
    let quantization_mix = mix
        .into_iter()
        .map(|(name, (tensors, bytes))| QuantizationBucket {
            name,
            tensors,
            bytes,
        })
        .collect();
    let experts = expert_accounting(model);
    let mut notes = architecture.notes.clone();
    if info.embedding_length.is_none() || info.head_count.is_none() {
        notes.push("attention shape metadata is incomplete".to_string());
    }
    if info.vocab_size.is_none() {
        notes.push("vocabulary size is unavailable from GGUF metadata".to_string());
    }
    if experts.routed_expert_bytes > 0 {
        notes.push(
            "routed expert bytes are metadata arithmetic; active-expert residency is not guaranteed"
                .to_string(),
        );
    }
    ModelProfile {
        schema_version: PROFILING_SCHEMA_VERSION,
        fingerprint: model_file_fingerprint(model),
        path: Some(model.path.display().to_string()),
        architecture: info.architecture,
        kind: architecture.kind,
        parameters,
        file_bytes: model.file_size,
        tensor_payload_bytes: model.total_tensor_bytes(),
        tensor_count: model.tensors.len(),
        layer_count: info.block_count,
        embedding_length: info.embedding_length,
        head_count: info.head_count,
        head_count_kv: info.head_count_kv,
        context_max: info.context_length,
        vocab_size: info.vocab_size.map(|value| value as u64),
        tokenizer_model: info.tokenizer_model,
        expert_count: info.expert_count,
        expert_used_count: info.expert_used_count,
        quantization: info.file_type.map(quantization_label),
        quantization_mix,
        legacy_execution_supported: support.is_legacy_supported(),
        forgecore_execution_supported: support.is_forgecore_supported(),
        unsupported_tensor_formats: support.unsupported_tensor_formats,
        routed_expert_bytes: (experts.routed_expert_bytes > 0)
            .then_some(experts.routed_expert_bytes),
        estimated_active_weight_bytes: experts.estimated_active_weight_bytes,
        notes,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuProfile {
    pub id: String,
    pub backend: String,
    pub name: String,
    pub total_bytes: Option<u64>,
    pub free_bytes: Option<u64>,
    pub selectable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub schema_version: u32,
    pub fingerprint: String,
    pub cpu_model: Option<String>,
    pub physical_cores: Option<u32>,
    pub logical_cpus: Option<u32>,
    pub total_ram_bytes: Option<u64>,
    pub available_ram_bytes: Option<u64>,
    pub os: String,
    pub kernel: Option<String>,
    pub gpus: Vec<GpuProfile>,
    pub backends: Vec<String>,
    pub storage_kind: Option<String>,
    pub storage_read_bytes_per_second: Option<u64>,
    pub notes: Vec<String>,
}

pub fn hardware_profile(machine: &MachineProfile) -> HardwareProfile {
    let gpus: Vec<GpuProfile> = machine
        .devices
        .iter()
        .filter(|device| device.kind != "cpu")
        .map(|device| GpuProfile {
            id: device.id.clone(),
            backend: device.backend_id.clone(),
            name: device.name.clone(),
            total_bytes: device.memory_total_bytes.as_ref().copied(),
            free_bytes: device.memory_free_bytes.as_ref().copied(),
            selectable: device.selectable,
        })
        .collect();
    let mut backends: Vec<String> = machine
        .backend_inventory
        .registered
        .iter()
        .map(|backend| backend.name.clone())
        .collect();
    backends.extend(machine.backends.iter().map(|backend| backend.id.clone()));
    backends.sort();
    backends.dedup();
    let storage = machine.storage.first();
    let storage_kind = storage
        .and_then(|storage| storage.storage_kind.as_ref())
        .map(|kind| format!("{kind:?}").to_ascii_lowercase());
    let storage_bps = storage
        .and_then(|storage| storage.benchmark.as_ref())
        .map(|benchmark| benchmark.median_bytes_per_second);
    let cpu_model = machine.cpu.model.as_ref().cloned();
    let physical_cores = machine.cpu.hardware_physical_cores.as_ref().copied();
    let logical_cpus = machine.cpu.hardware_logical_cpus.as_ref().copied();
    let total_ram = machine.memory.physical_total_bytes.as_ref().copied();
    let mut identity = String::new();
    for part in [
        cpu_model.clone().unwrap_or_default(),
        physical_cores.map(|v| v.to_string()).unwrap_or_default(),
        logical_cpus.map(|v| v.to_string()).unwrap_or_default(),
        total_ram.map(|v| v.to_string()).unwrap_or_default(),
        machine.os.clone(),
        machine.kernel.clone().unwrap_or_default(),
        machine
            .devices
            .iter()
            .map(|device| format!("{}:{}", device.backend_id, device.name))
            .collect::<Vec<_>>()
            .join(","),
        backends.join(","),
    ] {
        identity.push_str(&part);
        identity.push('\u{1f}');
    }
    HardwareProfile {
        schema_version: PROFILING_SCHEMA_VERSION,
        fingerprint: fnv_hex(identity.as_bytes()),
        cpu_model,
        physical_cores,
        logical_cpus,
        total_ram_bytes: total_ram,
        available_ram_bytes: machine.memory.os_available_bytes.as_ref().copied(),
        os: machine.os.clone(),
        kernel: machine.kernel.clone(),
        gpus,
        backends,
        storage_kind,
        storage_read_bytes_per_second: storage_bps,
        notes: machine.limitations.clone(),
    }
}

impl HardwareProfile {
    pub fn gpu_available(&self) -> bool {
        self.gpus.iter().any(|gpu| gpu.selectable)
    }

    pub fn max_free_vram_bytes(&self) -> Option<u64> {
        self.gpus.iter().filter_map(|gpu| gpu.free_bytes).max()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadProfile {
    pub context_tokens: u32,
    pub batch_tokens: u32,
    pub ubatch_tokens: u32,
    pub sequence_max: u32,
    pub threads: u32,
    pub gpu_layers: u32,
    pub kv_type_k: String,
    pub kv_type_v: String,
    pub load_mode: String,
    pub backend: String,
    pub device_id: String,
    pub residency: String,
    pub draft_model_fingerprint: Option<String>,
    pub draft_tokens: Option<u32>,
    pub max_tokens: Option<u32>,
    pub ram_budget_bytes: Option<u64>,
}

pub fn workload_from_plan(
    plan: &RuntimePlan,
    residency: impl Into<String>,
    draft_tokens: Option<u32>,
    max_tokens: Option<u32>,
    ram_budget_bytes: Option<u64>,
) -> WorkloadProfile {
    WorkloadProfile {
        context_tokens: plan.effective.context_tokens,
        batch_tokens: plan.effective.batch_tokens,
        ubatch_tokens: plan.effective.ubatch_tokens,
        sequence_max: plan.effective.sequence_max,
        threads: plan.effective.threads,
        gpu_layers: plan.effective.gpu_layers,
        kv_type_k: plan.effective.kv_type_k.clone(),
        kv_type_v: plan.effective.kv_type_v.clone(),
        load_mode: plan.effective.load_mode.clone(),
        backend: plan.effective.backend.clone(),
        device_id: plan.effective.device_id.clone(),
        residency: residency.into(),
        draft_model_fingerprint: None,
        draft_tokens,
        max_tokens,
        ram_budget_bytes,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerformanceIdentity {
    pub model_fingerprint: Option<String>,
    pub architecture: Option<String>,
    pub quantization: Option<String>,
    pub hardware_fingerprint: String,
    pub backend: String,
    pub device_id: String,
    pub workload: WorkloadProfile,
    pub runtime_schema: u32,
}

impl PerformanceIdentity {
    pub fn is_reusable(&self) -> bool {
        self.model_fingerprint.is_some() && !self.hardware_fingerprint.is_empty()
    }

    pub fn key(&self) -> String {
        let mut text = format!(
            "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
            self.model_fingerprint.as_deref().unwrap_or(""),
            self.architecture.as_deref().unwrap_or(""),
            self.quantization.as_deref().unwrap_or(""),
            self.hardware_fingerprint,
            self.backend,
            self.device_id,
            self.workload.context_tokens,
            self.workload.batch_tokens,
            self.workload.ubatch_tokens,
            self.workload.sequence_max,
            self.workload.threads,
            self.workload.gpu_layers,
            self.workload.kv_type_k,
            self.workload.kv_type_v,
            self.workload.load_mode,
            self.workload.residency,
            self.workload.draft_tokens.unwrap_or(0),
            self.workload
                .draft_model_fingerprint
                .as_deref()
                .unwrap_or(""),
        );
        text.push_str(&format!("|{}", self.runtime_schema));
        fnv_hex(text.as_bytes())
    }
}

pub fn performance_identity(
    model: &ModelProfile,
    hardware: &HardwareProfile,
    workload: &WorkloadProfile,
) -> PerformanceIdentity {
    PerformanceIdentity {
        model_fingerprint: model.fingerprint.clone(),
        architecture: model.architecture.clone(),
        quantization: model.quantization.clone(),
        hardware_fingerprint: hardware.fingerprint.clone(),
        backend: workload.backend.clone(),
        device_id: workload.device_id.clone(),
        workload: workload.clone(),
        runtime_schema: PROFILING_SCHEMA_VERSION,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceRecord {
    pub identity: PerformanceIdentity,
    pub captured_at_unix_ms: Option<u64>,
    pub generation_tokens_per_second: f64,
    pub decode_tokens_per_second: Option<f64>,
    pub prompt_tokens_per_second: Option<f64>,
    pub time_to_first_token_ns: Option<u64>,
    pub total_generation_ns: u64,
    pub emitted_tokens: u64,
    pub target_forwards: Option<u64>,
    pub accepted_tokens: Option<u64>,
    pub draft_tokens_generated: Option<u64>,
    pub acceptance_rate_bps: Option<u32>,
    pub storage_read_bytes: Option<u64>,
    pub pss_bytes: Option<u64>,
    pub condition: String,
    pub limitations: Vec<String>,
}

impl PerformanceRecord {
    pub fn from_benchmark(identity: PerformanceIdentity, report: &BenchmarkReport) -> Self {
        let last = report.runs_detail.last();
        let speculative = last.and_then(|run| run.speculative.as_ref());
        let total_generation_ns = last.map(|run| run.generation_ns).unwrap_or(0);
        Self {
            identity,
            captured_at_unix_ms: now_unix_ms(),
            generation_tokens_per_second: report.median_generation_tokens_per_second,
            decode_tokens_per_second: report.median_decode_tokens_per_second,
            prompt_tokens_per_second: report.median_prompt_tokens_per_second,
            time_to_first_token_ns: report.median_time_to_first_token_ns,
            total_generation_ns,
            emitted_tokens: last.map(|run| run.generated_tokens).unwrap_or(0),
            target_forwards: speculative.map(|stats| stats.target_forwards),
            accepted_tokens: speculative.map(|stats| stats.accepted_tokens),
            draft_tokens_generated: speculative.map(|stats| stats.draft_tokens_generated),
            acceptance_rate_bps: speculative.map(|stats| stats.acceptance_rate_bps),
            storage_read_bytes: last.and_then(|run| run.storage_read_bytes),
            pss_bytes: last.and_then(|run| run.pss_bytes),
            condition: report.condition.clone(),
            limitations: report.limitations.clone(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.generation_tokens_per_second.is_finite() || self.generation_tokens_per_second < 0.0
        {
            return Err("generation throughput must be finite and non-negative".to_string());
        }
        if self.emitted_tokens == 0 {
            return Err("performance record has no emitted tokens".to_string());
        }
        if self.condition.is_empty() {
            return Err("performance record condition is empty".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceProfileStore {
    pub schema_version: u32,
    pub records: Vec<PerformanceRecord>,
}

impl Default for PerformanceProfileStore {
    fn default() -> Self {
        Self {
            schema_version: PROFILING_SCHEMA_VERSION,
            records: Vec::new(),
        }
    }
}

impl PerformanceProfileStore {
    pub fn load(path: &Path) -> Result<Self, String> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
        let store: Self = serde_json::from_str(&text).map_err(|error| error.to_string())?;
        store.validate()?;
        Ok(store)
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let temp = path.with_extension("tmp");
        let text = serde_json::to_string_pretty(self).map_err(|error| error.to_string())?;
        std::fs::write(&temp, text).map_err(|error| error.to_string())?;
        std::fs::rename(&temp, path).map_err(|error| error.to_string())?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != PROFILING_SCHEMA_VERSION {
            return Err(format!(
                "unsupported performance profile schema {}",
                self.schema_version
            ));
        }
        for record in &self.records {
            record.validate()?;
        }
        Ok(())
    }

    /// Insert or replace the exact identity. Never overwrite a different model,
    /// hardware, or workload configuration.
    pub fn insert(&mut self, record: PerformanceRecord) {
        if let Some(existing) = self
            .records
            .iter_mut()
            .find(|existing| existing.identity == record.identity)
        {
            *existing = record;
        } else {
            self.records.push(record);
        }
    }

    pub fn exact(&self, identity: &PerformanceIdentity) -> Option<&PerformanceRecord> {
        self.records
            .iter()
            .find(|record| record.identity == *identity)
    }

    /// Best measured record for the same model and machine, ignoring workload
    /// differences. Used only for advisory recommendations.
    pub fn best_for_model_hardware(
        &self,
        model_fingerprint: Option<&str>,
        hardware_fingerprint: &str,
    ) -> Option<&PerformanceRecord> {
        let model_fingerprint = model_fingerprint?;
        self.records
            .iter()
            .filter(|record| {
                record.identity.model_fingerprint.as_deref() == Some(model_fingerprint)
                    && record.identity.hardware_fingerprint == hardware_fingerprint
            })
            .max_by(|left, right| {
                left.generation_tokens_per_second
                    .total_cmp(&right.generation_tokens_per_second)
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidencyAdvice {
    pub stage_layers: bool,
    pub budget_bytes: Option<u64>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacementAdvice {
    pub mode: String,
    pub gpu_layers: u32,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculativeAdvice {
    pub draft_compatible: bool,
    pub recommended: bool,
    pub draft_tokens: u32,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasuredSummary {
    pub generation_tokens_per_second: f64,
    pub decode_tokens_per_second: Option<f64>,
    pub acceptance_rate_bps: Option<u32>,
    pub condition: String,
    pub key: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveRecommendation {
    pub model_fingerprint: Option<String>,
    pub hardware_fingerprint: String,
    pub kind: ArchitectureKind,
    pub residency: ResidencyAdvice,
    pub placement: PlacementAdvice,
    pub speculative: SpeculativeAdvice,
    pub measured_best: Option<MeasuredSummary>,
    pub evidence_quality: String,
    pub notes: Vec<String>,
}

/// Produce advisory recommendations from metadata and, optionally, a measured
/// record. Nothing here is applied automatically.
pub fn recommend(
    model: &ModelProfile,
    hardware: &HardwareProfile,
    workload: &WorkloadProfile,
    draft_compatible: Option<bool>,
    measured_best: Option<&PerformanceRecord>,
) -> AdaptiveRecommendation {
    let ram_budget = workload
        .ram_budget_bytes
        .or(hardware.total_ram_bytes)
        .unwrap_or(0);
    let model_fits_ram = ram_budget > 0 && model.file_bytes <= ram_budget;
    let gpu = hardware.gpu_available();
    let free_vram = hardware.max_free_vram_bytes().unwrap_or(0);
    let safety = 256 * 1024 * 1024u64;

    let (placement, residency) = if gpu
        && model.file_bytes.saturating_add(512 * 1024 * 1024) <= free_vram
    {
        (
            PlacementAdvice {
                mode: "gpu".to_string(),
                gpu_layers: model.layer_count.unwrap_or(0).saturating_add(1) as u32,
                reason: "model plus a small reserve fits free VRAM; full offload is viable"
                    .to_string(),
            },
            ResidencyAdvice {
                stage_layers: false,
                budget_bytes: None,
                reason: "full GPU placement does not need host staging".to_string(),
            },
        )
    } else if gpu && !model_fits_ram && free_vram > safety {
        (
            PlacementAdvice {
                mode: "hybrid".to_string(),
                gpu_layers: 0,
                reason: "model exceeds RAM; keep CPU placement and use bounded staging".to_string(),
            },
            ResidencyAdvice {
                stage_layers: true,
                budget_bytes: Some(free_vram - safety),
                reason: "model exceeds the planning RAM budget; stage layers within free VRAM minus 256 MiB"
                    .to_string(),
            },
        )
    } else if model_fits_ram {
        (
            PlacementAdvice {
                mode: "cpu".to_string(),
                gpu_layers: 0,
                reason: "model fits the planning RAM budget".to_string(),
            },
            ResidencyAdvice {
                stage_layers: false,
                budget_bytes: None,
                reason: "no staging needed when the model fits RAM".to_string(),
            },
        )
    } else {
        (
            PlacementAdvice {
                mode: "cpu".to_string(),
                gpu_layers: 0,
                reason: "model exceeds RAM and no usable GPU staging path is available".to_string(),
            },
            ResidencyAdvice {
                stage_layers: false,
                budget_bytes: None,
                reason: "insufficient free VRAM for staged residency".to_string(),
            },
        )
    };

    let measured_k = measured_best
        .and_then(|record| record.identity.workload.draft_tokens)
        .or(workload.draft_tokens)
        .unwrap_or(8);
    let speculative = match draft_compatible {
        Some(true) => SpeculativeAdvice {
            draft_compatible: true,
            recommended: !model_fits_ram || measured_best.is_some(),
            draft_tokens: measured_k,
            reason: if !model_fits_ram {
                "compatible draft and an oversized target justify greedy speculative decoding"
                    .to_string()
            } else if measured_best.is_some() {
                "a measured record exists for this model/hardware; use its best workload as evidence".to_string()
            } else {
                "compatible draft exists, but the model already fits RAM; benchmark before enabling"
                    .to_string()
            },
        },
        _ => SpeculativeAdvice {
            draft_compatible: false,
            recommended: false,
            draft_tokens: 0,
            reason: "no compatible draft was supplied".to_string(),
        },
    };

    let measured_summary = measured_best.map(|record| MeasuredSummary {
        generation_tokens_per_second: record.generation_tokens_per_second,
        decode_tokens_per_second: record.decode_tokens_per_second,
        acceptance_rate_bps: record.acceptance_rate_bps,
        condition: record.condition.clone(),
        key: record.identity.key(),
    });

    AdaptiveRecommendation {
        model_fingerprint: model.fingerprint.clone(),
        hardware_fingerprint: hardware.fingerprint.clone(),
        kind: model.kind,
        residency,
        placement,
        speculative,
        measured_best: measured_summary,
        evidence_quality: if measured_best.is_some() {
            "measured".to_string()
        } else {
            "metadata-estimate".to_string()
        },
        notes: vec![
            "advisory only; RAMforge does not apply this recommendation automatically".to_string(),
            "planning RAM budgets are not enforced OS limits unless cgroup enforcement is enabled"
                .to_string(),
            if model.kind == ArchitectureKind::MixtureOfExperts {
                "MoE active-expert bytes are estimates; routing locality must be measured"
                    .to_string()
            } else {
                "dense models read their full weight working set per forward when they exceed RAM"
                    .to_string()
            },
        ],
    }
}

pub fn now_unix_ms() -> Option<u64> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_millis() as u64)
}

fn fnv_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ramforge_core::{GgmlType, MetadataValue, TensorDescriptor};

    use super::*;

    fn model(
        metadata: BTreeMap<String, MetadataValue>,
        tensors: Vec<TensorDescriptor>,
    ) -> GgufModel {
        GgufModel {
            path: std::path::PathBuf::from("/tmp/generic-model.gguf"),
            file_size: tensors
                .iter()
                .map(|tensor| tensor.byte_length.unwrap_or(0))
                .sum(),
            version: 3,
            metadata,
            tensors,
            alignment: 32,
            data_start_offset: 0,
        }
    }

    fn tensor(name: &str, ggml_type: GgmlType, bytes: u64, elements: u64) -> TensorDescriptor {
        TensorDescriptor {
            name: name.to_string(),
            dimensions: vec![1],
            ggml_type,
            offset: 0,
            file_offset: 0,
            byte_length: Some(bytes),
            num_elements: elements,
        }
    }

    fn dense() -> GgufModel {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            MetadataValue::String("llama".to_string()),
        );
        metadata.insert("general.file_type".to_string(), MetadataValue::UInt32(15));
        metadata.insert("llama.block_count".to_string(), MetadataValue::UInt64(2));
        model(
            metadata,
            vec![
                tensor("blk.0.attn_q.weight", GgmlType::Q4_K, 100, 50),
                tensor("blk.1.ffn_gate.weight", GgmlType::Q8_0, 50, 10),
            ],
        )
    }

    #[test]
    fn dense_profile_has_parameters_and_quant_mix() {
        let profile = model_profile(&dense());
        assert_eq!(profile.kind, ArchitectureKind::Dense);
        assert_eq!(profile.parameters, Some(60));
        assert_eq!(profile.quantization.as_deref(), Some("mostly-q4_k_m"));
        assert_eq!(profile.quantization_mix.len(), 2);
        assert!(profile.legacy_execution_supported);
    }

    #[test]
    fn moe_profile_reports_expert_structure() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            MetadataValue::String("deepseek2".to_string()),
        );
        metadata.insert(
            "deepseek2.expert_count".to_string(),
            MetadataValue::UInt64(8),
        );
        metadata.insert(
            "deepseek2.expert_used_count".to_string(),
            MetadataValue::UInt64(2),
        );
        let model = model(
            metadata,
            vec![
                tensor("blk.1.ffn_gate_exps.weight", GgmlType::Q4_K, 800, 100),
                tensor("blk.1.ffn_up_shexp.weight", GgmlType::Q4_K, 100, 10),
            ],
        );
        let profile = model_profile(&model);
        assert_eq!(profile.kind, ArchitectureKind::MixtureOfExperts);
        assert_eq!(profile.expert_count, Some(8));
        assert_eq!(profile.expert_used_count, Some(2));
        assert_eq!(profile.routed_expert_bytes, Some(800));
        assert!(profile
            .notes
            .iter()
            .any(|note| note.contains("active-expert")));
    }

    fn gpu_hardware() -> HardwareProfile {
        HardwareProfile {
            schema_version: 1,
            fingerprint: "hw-gpu".to_string(),
            cpu_model: Some("cpu".to_string()),
            physical_cores: Some(6),
            logical_cpus: Some(12),
            total_ram_bytes: Some(16 * 1024 * 1024 * 1024),
            available_ram_bytes: Some(8 * 1024 * 1024 * 1024),
            os: "linux".to_string(),
            kernel: None,
            gpus: vec![GpuProfile {
                id: "gpu0".to_string(),
                backend: "cuda".to_string(),
                name: "gpu".to_string(),
                total_bytes: Some(4 * 1024 * 1024 * 1024),
                free_bytes: Some(4 * 1024 * 1024 * 1024),
                selectable: true,
            }],
            backends: vec!["cuda".to_string(), "cpu".to_string()],
            storage_kind: Some("nvme".to_string()),
            storage_read_bytes_per_second: None,
            notes: Vec::new(),
        }
    }

    #[test]
    fn oversized_model_recommends_staging_and_requested_draft_length() {
        let mut model = model_profile(&dense());
        model.file_bytes = 20 * 1024 * 1024 * 1024;
        model.layer_count = Some(64);
        let hardware = gpu_hardware();
        let workload = WorkloadProfile {
            context_tokens: 128,
            batch_tokens: 64,
            ubatch_tokens: 64,
            sequence_max: 1,
            threads: 6,
            gpu_layers: 0,
            kv_type_k: "q8_0".to_string(),
            kv_type_v: "q8_0".to_string(),
            load_mode: "mmap".to_string(),
            backend: "cpu".to_string(),
            device_id: "cpu".to_string(),
            residency: "auto".to_string(),
            draft_model_fingerprint: Some("draft".to_string()),
            draft_tokens: Some(12),
            max_tokens: Some(8),
            ram_budget_bytes: Some(10 * 1024 * 1024 * 1024),
        };
        let advice = recommend(&model, &hardware, &workload, Some(true), None);
        assert!(advice.residency.stage_layers);
        assert_eq!(advice.placement.mode, "hybrid");
        assert!(advice.speculative.recommended);
        assert_eq!(advice.speculative.draft_tokens, 12);
    }

    #[test]
    fn identity_requires_exact_model_and_hardware() {
        let mut profile = model_profile(&dense());
        profile.fingerprint = Some("model".to_string());
        let hardware = HardwareProfile {
            schema_version: 1,
            fingerprint: "hw".to_string(),
            cpu_model: Some("cpu".to_string()),
            physical_cores: Some(4),
            logical_cpus: Some(8),
            total_ram_bytes: Some(1024),
            available_ram_bytes: Some(1024),
            os: "linux".to_string(),
            kernel: None,
            gpus: Vec::new(),
            backends: vec!["cpu".to_string()],
            storage_kind: None,
            storage_read_bytes_per_second: None,
            notes: Vec::new(),
        };
        let workload = WorkloadProfile {
            context_tokens: 256,
            batch_tokens: 64,
            ubatch_tokens: 64,
            sequence_max: 1,
            threads: 4,
            gpu_layers: 0,
            kv_type_k: "f16".to_string(),
            kv_type_v: "f16".to_string(),
            load_mode: "mmap".to_string(),
            backend: "cpu".to_string(),
            device_id: "cpu".to_string(),
            residency: "off".to_string(),
            draft_model_fingerprint: None,
            draft_tokens: None,
            max_tokens: Some(8),
            ram_budget_bytes: Some(1024),
        };
        let identity = performance_identity(&profile, &hardware, &workload);
        assert!(identity.is_reusable());
        let mut store = PerformanceProfileStore::default();
        store.insert(PerformanceRecord {
            identity: identity.clone(),
            captured_at_unix_ms: None,
            generation_tokens_per_second: 1.0,
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
            condition: "warm".to_string(),
            limitations: Vec::new(),
        });
        assert!(store.exact(&identity).is_some());
        let mut other = identity.clone();
        other.model_fingerprint = Some("different".to_string());
        assert!(store.exact(&other).is_none());
    }

    #[test]
    fn recommendation_does_not_stage_when_model_fits() {
        let model = model_profile(&dense());
        let hardware = HardwareProfile {
            schema_version: 1,
            fingerprint: "hw".to_string(),
            cpu_model: Some("cpu".to_string()),
            physical_cores: Some(4),
            logical_cpus: Some(8),
            total_ram_bytes: Some(1_000_000),
            available_ram_bytes: Some(1_000_000),
            os: "linux".to_string(),
            kernel: None,
            gpus: Vec::new(),
            backends: vec!["cpu".to_string()],
            storage_kind: Some("nvme".to_string()),
            storage_read_bytes_per_second: None,
            notes: Vec::new(),
        };
        let workload = WorkloadProfile {
            context_tokens: 256,
            batch_tokens: 64,
            ubatch_tokens: 64,
            sequence_max: 1,
            threads: 4,
            gpu_layers: 0,
            kv_type_k: "f16".to_string(),
            kv_type_v: "f16".to_string(),
            load_mode: "mmap".to_string(),
            backend: "cpu".to_string(),
            device_id: "cpu".to_string(),
            residency: "auto".to_string(),
            draft_model_fingerprint: None,
            draft_tokens: None,
            max_tokens: Some(8),
            ram_budget_bytes: Some(1_000_000),
        };
        let advice = recommend(&model, &hardware, &workload, None, None);
        assert!(!advice.residency.stage_layers);
        assert_eq!(advice.placement.mode, "cpu");
        assert_eq!(advice.evidence_quality, "metadata-estimate");
    }

    #[test]
    fn profile_store_roundtrips_through_json() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("profiles.json");
        let mut store = PerformanceProfileStore::default();
        let record = PerformanceRecord {
            identity: PerformanceIdentity {
                model_fingerprint: Some("model".to_string()),
                architecture: Some("llama".to_string()),
                quantization: Some("mostly-q4_k_m".to_string()),
                hardware_fingerprint: "hw".to_string(),
                backend: "cpu".to_string(),
                device_id: "cpu".to_string(),
                workload: WorkloadProfile {
                    context_tokens: 128,
                    batch_tokens: 64,
                    ubatch_tokens: 64,
                    sequence_max: 1,
                    threads: 6,
                    gpu_layers: 0,
                    kv_type_k: "q8_0".to_string(),
                    kv_type_v: "q8_0".to_string(),
                    load_mode: "mmap".to_string(),
                    backend: "cpu".to_string(),
                    device_id: "cpu".to_string(),
                    residency: "auto".to_string(),
                    draft_model_fingerprint: None,
                    draft_tokens: None,
                    max_tokens: Some(8),
                    ram_budget_bytes: None,
                },
                runtime_schema: PROFILING_SCHEMA_VERSION,
            },
            captured_at_unix_ms: Some(1),
            generation_tokens_per_second: 0.5,
            decode_tokens_per_second: None,
            prompt_tokens_per_second: None,
            time_to_first_token_ns: None,
            total_generation_ns: 1,
            emitted_tokens: 8,
            target_forwards: None,
            accepted_tokens: None,
            draft_tokens_generated: None,
            acceptance_rate_bps: None,
            storage_read_bytes: None,
            pss_bytes: None,
            condition: "warm".to_string(),
            limitations: vec!["short-run".to_string()],
        };
        store.insert(record.clone());
        store.save(&path).unwrap();
        let loaded = PerformanceProfileStore::load(&path).unwrap();
        assert_eq!(loaded.records.len(), 1);
        assert_eq!(loaded.records[0], record);
        assert!(loaded.exact(&record.identity).is_some());
        assert!(!path.with_extension("tmp").exists());
    }
}
