//! Bounded repeated generation benchmarks over the inference service.
//!
//! Tokens per second are measured from actual native generation. Model load and
//! context creation are reported separately; no synthetic throughput is used.

use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::gpu_observability::GpuObservability;
use crate::inference_service::{
    GenerationOptions, InferenceService, ServiceError, INFERENCE_SERVICE_API_VERSION,
};
use crate::runtime_plan::RuntimePlan;

pub const BENCHMARK_SCHEMA_VERSION: u32 = 6;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkRun {
    pub index: u32,
    pub generated_tokens: u64,
    pub prompt_tokens: u64,
    pub generation_ns: u64,
    pub time_to_first_token_ns: Option<u64>,
    pub decode_ns: Option<u64>,
    pub prompt_tokens_per_second: f64,
    pub decode_tokens_per_second: Option<f64>,
    pub generation_tokens_per_second: f64,
    pub stop_reason: String,
    pub rss_before_bytes: Option<u64>,
    pub rss_after_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub schema_version: u32,
    pub api_version: u32,
    pub profile: String,
    pub model_path: String,
    /// Exact resolved plan, including requested/effective field statuses.
    pub plan: RuntimePlan,
    pub prompt: String,
    pub max_tokens: u32,
    pub runs: u32,
    pub warmup_runs: u32,
    pub model_load_ns: u64,
    pub device_id: String,
    pub requested_gpu_layers: Option<u32>,
    /// Native placement readback after model load.
    pub effective_gpu_layers: Option<u32>,
    pub gpu_transformer_layers: Option<u32>,
    pub cpu_transformer_layers: Option<u32>,
    pub output_device_type: Option<String>,
    /// Native free-memory snapshots for the selected device.
    pub device_memory_free_before_bytes: Option<u64>,
    pub device_memory_free_after_load_bytes: Option<u64>,
    /// Free device memory after warmup contexts were created; includes
    /// backend/compute buffers. Still a point-in-time value, not a peak.
    pub device_memory_free_after_warmup_bytes: Option<u64>,
    pub device_memory_free_after_unload_bytes: Option<u64>,
    /// Native/derived observability snapshots at lifecycle boundaries.
    pub memory_after_warmup: Option<GpuObservability>,
    pub memory_after_generation: Option<GpuObservability>,
    pub memory_after_unload: Option<GpuObservability>,
    pub context_create_ns: Vec<u64>,
    pub runs_detail: Vec<BenchmarkRun>,
    pub median_generation_tokens_per_second: f64,
    pub min_generation_tokens_per_second: f64,
    pub max_generation_tokens_per_second: f64,
    pub p25_generation_tokens_per_second: f64,
    pub p75_generation_tokens_per_second: f64,
    pub p95_generation_tokens_per_second: f64,
    pub iqr_generation_tokens_per_second: f64,
    pub median_prompt_tokens_per_second: Option<f64>,
    pub median_decode_tokens_per_second: Option<f64>,
    pub median_time_to_first_token_ns: Option<u64>,
    pub median_context_create_ns: u64,
    pub condition: String,
    pub limitations: Vec<String>,
}

impl BenchmarkReport {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != BENCHMARK_SCHEMA_VERSION {
            return Err(format!(
                "unsupported benchmark schema {}",
                self.schema_version
            ));
        }
        if self.runs == 0 || self.runs_detail.is_empty() {
            return Err("benchmark has no measured runs".to_string());
        }
        if self.context_create_ns.is_empty() {
            return Err("benchmark has no context-creation measurements".to_string());
        }
        let ordered = self.min_generation_tokens_per_second
            <= self.p25_generation_tokens_per_second
            && self.p25_generation_tokens_per_second <= self.median_generation_tokens_per_second
            && self.median_generation_tokens_per_second <= self.p75_generation_tokens_per_second
            && self.p75_generation_tokens_per_second <= self.p95_generation_tokens_per_second
            && self.p95_generation_tokens_per_second <= self.max_generation_tokens_per_second;
        if !ordered {
            return Err("generation throughput statistics are not ordered".to_string());
        }
        let iqr = self.p75_generation_tokens_per_second - self.p25_generation_tokens_per_second;
        if (iqr - self.iqr_generation_tokens_per_second).abs() > 1e-9 {
            return Err("generation throughput IQR does not match P25/P75".to_string());
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(|error| error.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkConfig {
    pub runs: u32,
    pub warmup_runs: u32,
    pub prompt: String,
    pub max_tokens: u32,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            runs: 3,
            warmup_runs: 1,
            prompt: "The capital of France is".to_string(),
            max_tokens: 4,
        }
    }
}

impl BenchmarkConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=10).contains(&self.runs) {
            return Err("benchmark runs must be in 1..=10".to_string());
        }
        if self.warmup_runs > 3 {
            return Err("benchmark warmup runs must be at most 3".to_string());
        }
        if self.prompt.is_empty() {
            return Err("benchmark prompt is empty".to_string());
        }
        if self.max_tokens > 512 {
            return Err("benchmark max_tokens must be at most 512".to_string());
        }
        Ok(())
    }
}

/// Run bounded repeated generation. Model load is measured once; each measured
/// run creates a fresh context before generating.
pub fn run_benchmark(
    plan: RuntimePlan,
    config: BenchmarkConfig,
) -> Result<BenchmarkReport, ServiceError> {
    config.validate().map_err(ServiceError::Configuration)?;
    let profile = format!("{:?}", plan.profile).to_ascii_lowercase();
    let model_path = plan.model_path.clone();
    let plan_snapshot = plan.clone();
    let device_id = plan_snapshot.effective.device_id.clone();
    let requested_gpu_layers = plan_snapshot.requested_gpu_layers;
    let free_before = device_free_bytes(&device_id);
    let mut service = InferenceService::new(plan)?;
    let load_started = Instant::now();
    service.load()?;
    let model_load_ns = duration_ns(load_started.elapsed());
    let load_status = service.status();
    let free_after_load = device_free_bytes(&device_id);
    let options = GenerationOptions {
        prompt: config.prompt.clone(),
        max_tokens: config.max_tokens,
        temperature: 0.0,
        top_k: None,
        top_p: None,
    };
    let cancel = AtomicBool::new(false);
    for _ in 0..config.warmup_runs {
        service.create_session()?;
        let _ = service.generate(&options, &cancel, |_| Ok(()))?;
    }
    let free_after_warmup = device_free_bytes(&device_id);
    let memory_after_warmup = Some(service.memory_observability()?);
    let mut runs = Vec::with_capacity(config.runs as usize);
    let mut context_ns = Vec::with_capacity(config.runs as usize);
    for index in 0..config.runs {
        let started = Instant::now();
        service.create_session()?;
        context_ns.push(duration_ns(started.elapsed()));
        let report = service.generate(&options, &cancel, |_| Ok(()))?;
        runs.push(BenchmarkRun {
            index,
            generated_tokens: report.stats.generated_tokens,
            prompt_tokens: report.stats.prompt_tokens,
            generation_ns: report.stats.total_generation_ns,
            time_to_first_token_ns: report.stats.time_to_first_token_ns,
            decode_ns: report.stats.decode_ns,
            prompt_tokens_per_second: report.stats.prompt_tokens_per_second,
            decode_tokens_per_second: report.stats.decode_tokens_per_second,
            generation_tokens_per_second: report.stats.generation_tokens_per_second,
            stop_reason: report.stats.stop_reason,
            rss_before_bytes: report.stats.rss_before_bytes,
            rss_after_bytes: report.stats.rss_after_bytes,
        });
    }
    let memory_after_generation = Some(service.memory_observability()?);
    service.unload();
    let free_after_unload = device_free_bytes(&device_id);
    let memory_after_unload = Some(service.memory_observability()?);
    let mut rates: Vec<f64> = runs
        .iter()
        .map(|run| run.generation_tokens_per_second)
        .collect();
    let mut decode_rates: Vec<f64> = runs
        .iter()
        .filter_map(|run| run.decode_tokens_per_second)
        .collect();
    let mut prompt_rates: Vec<f64> = runs
        .iter()
        .map(|run| run.prompt_tokens_per_second)
        .collect();
    let mut ttfts: Vec<u64> = runs
        .iter()
        .filter_map(|run| run.time_to_first_token_ns)
        .collect();
    rates.sort_by(|left, right| left.total_cmp(right));
    decode_rates.sort_by(|left, right| left.total_cmp(right));
    prompt_rates.sort_by(|left, right| left.total_cmp(right));
    ttfts.sort_unstable();
    let median = median_f64(&rates);
    let min = rates.first().copied().unwrap_or(0.0);
    let max = rates.last().copied().unwrap_or(0.0);
    let p25 = percentile_f64(&rates, 1, 4).unwrap_or(median);
    let p75 = percentile_f64(&rates, 3, 4).unwrap_or(median);
    let p95 = percentile_f64(&rates, 95, 100).unwrap_or(max);
    let report = BenchmarkReport {
        schema_version: BENCHMARK_SCHEMA_VERSION,
        api_version: INFERENCE_SERVICE_API_VERSION,
        profile,
        model_path,
        plan: plan_snapshot,
        prompt: config.prompt,
        max_tokens: config.max_tokens,
        runs: config.runs,
        warmup_runs: config.warmup_runs,
        model_load_ns,
        device_id,
        requested_gpu_layers,
        effective_gpu_layers: load_status.effective_gpu_layers,
        gpu_transformer_layers: load_status.gpu_transformer_layers,
        cpu_transformer_layers: load_status.cpu_transformer_layers,
        output_device_type: load_status.output_device_type,
        device_memory_free_before_bytes: free_before,
        device_memory_free_after_load_bytes: free_after_load,
        device_memory_free_after_warmup_bytes: free_after_warmup,
        device_memory_free_after_unload_bytes: free_after_unload,
        memory_after_warmup,
        memory_after_generation,
        memory_after_unload,
        median_context_create_ns: median_u64(&mut context_ns.clone()),
        context_create_ns: context_ns,
        median_decode_tokens_per_second: if decode_rates.is_empty() {
            None
        } else {
            Some(median_f64(&decode_rates))
        },
        median_time_to_first_token_ns: if ttfts.is_empty() {
            None
        } else {
            Some(ttfts[ttfts.len() / 2])
        },
        runs_detail: runs,
        median_generation_tokens_per_second: median,
        min_generation_tokens_per_second: min,
        max_generation_tokens_per_second: max,
        p25_generation_tokens_per_second: p25,
        p75_generation_tokens_per_second: p75,
        p95_generation_tokens_per_second: p95,
        iqr_generation_tokens_per_second: (p75 - p25).max(0.0),
        median_prompt_tokens_per_second: if prompt_rates.is_empty() {
            None
        } else {
            Some(median_f64(&prompt_rates))
        },
        condition: "warm-uncontrolled".to_string(),
        limitations: vec![
            "Single-machine warm observation; no cold-cache claim.".to_string(),
            "Model load is measured once and excluded from generation rate.".to_string(),
            "Context creation is measured separately and included in no generation rate."
                .to_string(),
            "No GPU or synthetic throughput is claimed.".to_string(),
        ],
    };
    report.validate().map_err(ServiceError::Configuration)?;
    Ok(report)
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn median_f64(sorted: &[f64]) -> f64 {
    if sorted.is_empty() {
        0.0
    } else {
        sorted[sorted.len() / 2]
    }
}

fn device_free_bytes(device_id: &str) -> Option<u64> {
    if device_id == "cpu" {
        return None;
    }
    crate::DeviceRegistry::discover()
        .devices()
        .iter()
        .find(|device| device.id.as_str() == device_id)
        .and_then(|device| device.memory_free.bytes())
}

fn percentile_f64(sorted: &[f64], numerator: usize, denominator: usize) -> Option<f64> {
    if sorted.is_empty() || denominator == 0 {
        return None;
    }
    sorted
        .get(sorted.len().saturating_mul(numerator) / denominator)
        .copied()
}

fn median_u64(values: &mut [u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    values[values.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_config_is_bounded() {
        assert!(BenchmarkConfig {
            runs: 0,
            warmup_runs: 0,
            prompt: "x".to_string(),
            max_tokens: 1,
        }
        .validate()
        .is_err());
        assert!(BenchmarkConfig {
            runs: 1,
            warmup_runs: 4,
            prompt: "x".to_string(),
            max_tokens: 1,
        }
        .validate()
        .is_err());
        assert!(BenchmarkConfig::default().validate().is_ok());
    }

    #[test]
    fn median_handles_empty_and_ordered_values() {
        assert_eq!(median_f64(&[]), 0.0);
        assert_eq!(median_f64(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median_u64(&mut []), 0);
        assert_eq!(median_u64(&mut [3, 1, 2]), 2);
    }

    #[test]
    fn nearest_rank_percentiles_are_ordered_and_handle_small_samples() {
        let values = [96.17, 102.43, 102.80, 102.98, 103.02];
        assert_eq!(percentile_f64(&values, 1, 4), Some(102.43));
        assert_eq!(percentile_f64(&values, 3, 4), Some(102.98));
        assert_eq!(percentile_f64(&values, 95, 100), Some(103.02));
        assert_eq!(percentile_f64(&[7.0], 1, 4), Some(7.0));
        assert_eq!(percentile_f64(&[7.0], 3, 4), Some(7.0));
        assert_eq!(percentile_f64(&[], 1, 4), None);
    }
}
