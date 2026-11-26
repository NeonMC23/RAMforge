//! Bounded measured profile comparison.
//!
//! This is the first adaptive optimization foundation: it executes only
//! explicitly selected bounded workloads, ranks profiles by measured generation
//! throughput, and returns a recommendation. It never changes a live session.

use serde::{Deserialize, Serialize};

use crate::benchmark_runner::{run_benchmark, BenchmarkConfig, BenchmarkReport};
use crate::machine_profile::MachineProfile;
use crate::model_requirements::ModelRequirements;
use crate::runtime_plan::{CustomRuntimeRequest, RuntimePlan, RuntimeProfileKind};

pub const OPTIMIZATION_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileMeasurement {
    pub profile: String,
    pub plan: RuntimePlan,
    pub benchmark: BenchmarkReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub profile: String,
    pub median_generation_tokens_per_second: f64,
    pub median_decode_tokens_per_second: Option<f64>,
    pub median_time_to_first_token_ns: Option<u64>,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationReport {
    pub schema_version: u32,
    pub model_path: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub runs_per_profile: u32,
    pub measurements: Vec<ProfileMeasurement>,
    pub recommendation: Option<OptimizationRecommendation>,
    pub failures: Vec<String>,
    pub limitations: Vec<String>,
}

impl OptimizationReport {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != OPTIMIZATION_SCHEMA_VERSION {
            return Err(format!(
                "unsupported optimization schema {}",
                self.schema_version
            ));
        }
        if self.measurements.is_empty() && self.failures.is_empty() {
            return Err("optimization produced neither measurements nor failures".to_string());
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(|error| error.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizationConfig {
    pub prompt: String,
    pub max_tokens: u32,
    pub runs: u32,
    pub warmup_runs: u32,
    pub context_tokens: u32,
    pub threads: Option<u32>,
    pub no_repack: Option<bool>,
    pub load_mode: Option<String>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            prompt: "The capital of France is".to_string(),
            max_tokens: 4,
            runs: 2,
            warmup_runs: 1,
            context_tokens: 256,
            threads: None,
            no_repack: None,
            load_mode: None,
        }
    }
}

impl OptimizationConfig {
    fn validate(&self) -> Result<(), String> {
        if self.prompt.is_empty() {
            return Err("optimization prompt is empty".to_string());
        }
        if !(1..=3).contains(&self.runs) {
            return Err("optimization runs must be in 1..=3".to_string());
        }
        if self.warmup_runs > 2 {
            return Err("optimization warmup runs must be at most 2".to_string());
        }
        if !(1..=512).contains(&self.max_tokens) {
            return Err("optimization max_tokens must be in 1..=512".to_string());
        }
        Ok(())
    }
}

/// Compare maximum-performance, balanced, and background profiles with real
/// bounded generation. Measurement order is deterministic.
pub fn run_profile_optimization(
    machine: &MachineProfile,
    requirements: &ModelRequirements,
    model_path: &str,
    config: OptimizationConfig,
) -> OptimizationReport {
    let failures = match config.validate() {
        Ok(()) => Vec::new(),
        Err(error) => vec![error],
    };
    let mut measurements = Vec::new();
    let mut failures = failures;
    if failures.is_empty() {
        for profile in [
            RuntimeProfileKind::MaximumPerformance,
            RuntimeProfileKind::Balanced,
            RuntimeProfileKind::Background,
        ] {
            let plan = match RuntimePlan::resolve(
                machine,
                requirements,
                model_path,
                profile,
                &CustomRuntimeRequest {
                    context_tokens: Some(config.context_tokens),
                    batch_tokens: Some(1),
                    threads: config.threads,
                    load_mode: config.load_mode.clone(),
                    no_repack: config.no_repack,
                    device: None,
                    gpu_layers: None,
                },
            ) {
                Ok(plan) => plan,
                Err(error) => {
                    failures.push(format!("{profile:?}: {error}"));
                    continue;
                }
            };
            match run_benchmark(
                plan.clone(),
                BenchmarkConfig {
                    runs: config.runs,
                    warmup_runs: config.warmup_runs,
                    prompt: config.prompt.clone(),
                    max_tokens: config.max_tokens,
                    staged_layers: Vec::new(),
                    staged_budget_bytes: None,
                    staged_auto: false,
                    staged_margin_bytes: crate::staging_plan::DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
                    selection_policy: crate::staging_plan::SelectionPolicy::Ascending,
                    host_staging_mode: crate::host_staging::HostStagingMode::Copy,
                    read_batching: crate::host_staging::ReadBatching::Coalesced,
                    pinned_pool_bytes: 0,
                    storage_concurrency: 1,
                    storage_max_outstanding_bytes: 0,
                    draft_model: None,
                    draft_tokens: 4,
                    draft_threads: 0,
                },
            ) {
                Ok(benchmark) => measurements.push(ProfileMeasurement {
                    profile: format!("{profile:?}").to_ascii_lowercase(),
                    plan,
                    benchmark,
                }),
                Err(error) => failures.push(format!("{profile:?}: {error}")),
            }
        }
    }
    let recommendation = measurements
        .iter()
        .max_by(|left, right| {
            left.benchmark
                .median_generation_tokens_per_second
                .total_cmp(&right.benchmark.median_generation_tokens_per_second)
        })
        .map(|measurement| OptimizationRecommendation {
            profile: measurement.profile.clone(),
            median_generation_tokens_per_second: measurement
                .benchmark
                .median_generation_tokens_per_second,
            median_decode_tokens_per_second: measurement.benchmark.median_decode_tokens_per_second,
            median_time_to_first_token_ns: measurement.benchmark.median_time_to_first_token_ns,
            reasons: vec![
                "highest median measured generation tokens/second in this bounded run".to_string(),
                "selection is advisory; no live session configuration was changed".to_string(),
            ],
        });
    OptimizationReport {
        schema_version: OPTIMIZATION_SCHEMA_VERSION,
        model_path: model_path.to_string(),
        prompt: config.prompt,
        max_tokens: config.max_tokens,
        runs_per_profile: config.runs,
        measurements,
        recommendation,
        failures,
        limitations: vec![
            "Single-machine warm observation; no cold-cache claim.".to_string(),
            "Profiles were measured sequentially; system noise is not controlled.".to_string(),
            "A recommendation does not prove a larger model will fit.".to_string(),
            "No automatic profile switching or live reconfiguration was performed.".to_string(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine_profile::{collect_machine_profile, MachineProfileConfig};
    use crate::model_requirements::RequirementBytes;

    fn requirements() -> ModelRequirements {
        ModelRequirements {
            schema_version: crate::model_requirements::MODEL_REQUIREMENTS_SCHEMA_VERSION,
            model_fingerprint: Some("fnv:test".to_string()),
            model_name: Some("test".to_string()),
            architecture: Some("llama".to_string()),
            architecture_info: None,
            quantization: Some("q4".to_string()),
            tensor_count: 1,
            file_bytes: RequirementBytes::exact(100, "test"),
            tensor_payload_bytes: RequirementBytes::exact(100, "test"),
            fully_resident_weight_bytes: RequirementBytes::exact(100, "test"),
            kv_cache_bytes: RequirementBytes::estimated(10, "test"),
            context_tokens: 128,
            batch_tokens: 1,
            model_max_context_tokens: Some(128),
            resident_weights_plus_kv_bytes: RequirementBytes::estimated(110, "test"),
            mapped_weights_plus_kv_bytes: RequirementBytes::estimated(10, "test"),
            unknown_requirements: Vec::new(),
            assumptions: Vec::new(),
        }
    }

    #[test]
    fn invalid_configuration_does_not_invent_measurements() {
        let machine = collect_machine_profile(&MachineProfileConfig::default());
        let report = run_profile_optimization(
            &machine,
            &requirements(),
            "/definitely/not/a/model.gguf",
            OptimizationConfig {
                runs: 0,
                ..OptimizationConfig::default()
            },
        );
        assert!(report.measurements.is_empty());
        assert!(!report.failures.is_empty());
        assert!(report.recommendation.is_none());
    }

    #[test]
    fn report_requires_measurements_or_failures() {
        let report = OptimizationReport {
            schema_version: OPTIMIZATION_SCHEMA_VERSION,
            model_path: "/m".to_string(),
            prompt: "x".to_string(),
            max_tokens: 1,
            runs_per_profile: 1,
            measurements: Vec::new(),
            recommendation: None,
            failures: Vec::new(),
            limitations: Vec::new(),
        };
        assert!(report.validate().is_err());
    }
}
