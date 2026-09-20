//! Structured diagnostic result collection, comparison, observation, and JSON export.
//!
//! This module consumes only the existing public profiling/report/runtime APIs.
//! It never fabricates values; missing data is explicitly marked unavailable.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use ramforge_core::datasource::IoProfile;
use ramforge_runtime::inference::{GenerationProfile, TensorReadProfile};
use ramforge_runtime::memory_report::MemoryReport;
use ramforge_runtime::profile::ProfileSnapshot;
use ramforge_runtime::runtime_config::RuntimeConfig;

pub const DIAGNOSTIC_SCHEMA_VERSION: u32 = 1;
pub const MAX_RUN_HISTORY: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GenerationPhase {
    #[default]
    Prompt,
    Decode,
    Finished,
    Failed,
}

impl GenerationPhase {
    pub fn label(self) -> &'static str {
        match self {
            Self::Prompt => "Prompt / prefill",
            Self::Decode => "Decode",
            Self::Finished => "Finished",
            Self::Failed => "Failed",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LiveStatus {
    pub started_at: Option<std::time::Instant>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub max_tokens: usize,
    pub phase: GenerationPhase,
}

impl LiveStatus {
    pub fn elapsed(&self) -> Duration {
        self.started_at
            .map(|t| t.elapsed())
            .unwrap_or(Duration::ZERO)
    }
}

/// Snapshot of the runtime+profiler state captured after a generation completes
/// (or fails). All fields are plain, owned, serde-serializable values.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DiagnosticResult {
    pub schema_version: u32,
    pub generation: GenerationDiagnostics,
    pub cpu_compute: CpuComputeDiagnostics,
    pub memory: MemoryDiagnostics,
    pub cache: CacheDiagnostics,
    pub io: IoDiagnostics,
    pub layers: LayerDiagnostics,
    pub quantization: QuantizationDiagnostics,
    pub calibration: CalibrationDiagnostics,
    pub observations: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GenerationDiagnostics {
    pub total_elapsed_seconds: f64,
    pub wall_elapsed_seconds: f64,
    pub prompt_elapsed_seconds: f64,
    pub decode_elapsed_seconds: Option<f64>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub max_tokens: usize,
    pub tokens_per_second_overall: Option<f64>,
    pub tokens_per_second_generated: Option<f64>,
    pub prompt_forwards: u64,
    pub decode_forwards: u64,
    pub terminal_forwards_skipped: u64,
    pub sampling_seconds: f64,
    pub output_seconds: f64,
    pub max_token_latency_seconds: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CpuComputeDiagnostics {
    pub cpu_threads: usize,
    pub f32_matvec_seconds: f64,
    pub quantized_matvec_seconds: f64,
    pub dequantization_seconds: f64,
    pub tensor_construction_seconds: f64,
    pub layer_compute_seconds: f64,
    pub logits_seconds: f64,
    pub allocation_seconds: f64,
    pub grouped_quantized_copy_count: u64,
    pub grouped_quantized_copy_seconds: f64,
    pub grouped_quantized_copy_bytes: u64,
    pub q4_0_matvec_calls: u64,
    pub q4_0_matvec_rows: u64,
    pub q4_0_matvec_input_elements: u64,
    pub q4_0_matvec_blocks: u64,
    pub q4_0_matvec_weight_bytes: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MemoryDiagnostics {
    pub configured_budget_bytes: u64,
    pub managed_current_bytes: u64,
    pub managed_peak_bytes: u64,
    pub process_rss_bytes: Option<u64>,
    pub system_total_bytes: Option<u64>,
    pub system_available_bytes: Option<u64>,
    pub layer_cache_capacity_bytes: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CacheDiagnostics {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub current_cached_layers: u64,
    pub peak_cached_layers: u64,
    pub current_cache_bytes: u64,
    pub peak_cache_bytes: u64,
    pub capacity_bytes: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct IoDiagnostics {
    pub logical_tensor_reads: u64,
    pub logical_tensor_bytes: u64,
    pub physical_reads: u64,
    pub physical_bytes: u64,
    pub read_failures: u64,
    pub seek_operations: u64,
    pub seeks_avoided: u64,
    pub coalesced_ranges: u64,
    pub coalesced_gap_bytes: u64,
    pub read_buffer_reuses: u64,
    pub read_buffer_growths: u64,
    pub read_elapsed_seconds: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LayerDiagnostics {
    pub loads: u64,
    pub releases: u64,
    pub load_seconds: f64,
    pub compute_seconds: f64,
    pub release_seconds: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct QuantizationDiagnostics {
    pub tensor_type_counts: BTreeMap<String, u64>,
    pub tensor_type_bytes: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CalibrationDiagnostics {
    pub level: String,
    pub calibration_identifier: Option<String>,
    pub applied_observation_identifiers: Vec<String>,
    pub consumed_by_planning: bool,
}

/// A single entry in the bounded, session-local run history.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RunRecord {
    pub run_id: u64,
    pub timestamp_unix_seconds: u64,
    pub model_name: Option<String>,
    pub model_architecture: Option<String>,
    pub model_descriptor_fingerprint: Option<u64>,
    pub machine_fingerprint: Option<u64>,
    pub storage_fingerprint: Option<u64>,
    pub model_path: String,
    pub user_config: UserConfigSnapshot,
    pub runtime_config: RuntimeConfigSnapshot,
    pub generation_params: GenerationParamsSnapshot,
    pub diagnostics: DiagnosticResult,
    pub success: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct UserConfigSnapshot {
    pub ram_budget_bytes: u64,
    pub ram_input: String,
    pub mode_label: String,
    pub preset_label: Option<String>,
    pub calibration_level: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RuntimeConfigSnapshot {
    pub cpu_thread_count: usize,
    pub ram_budget_bytes: u64,
    pub layer_cache_enabled: bool,
    pub layer_cache_capacity_bytes: u64,
    pub read_coalescing_enabled: bool,
    pub grouped_read_buffer_reuse_enabled: bool,
    pub execution_device: String,
}

impl From<&RuntimeConfig> for RuntimeConfigSnapshot {
    fn from(config: &RuntimeConfig) -> Self {
        Self {
            cpu_thread_count: config.cpu_thread_count,
            ram_budget_bytes: config.ram_budget_bytes,
            layer_cache_enabled: config.layer_cache_enabled,
            layer_cache_capacity_bytes: config.layer_cache_capacity_bytes,
            read_coalescing_enabled: config.read_coalescing_enabled,
            grouped_read_buffer_reuse_enabled: config.grouped_read_buffer_reuse_enabled,
            execution_device: format!("{:?}", config.execution_device),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GenerationParamsSnapshot {
    pub prompt_token_count: usize,
    pub max_tokens: usize,
    pub sampler: String,
}

#[derive(Debug, Clone)]
pub struct DiagnosticInputs<'a> {
    pub runtime_config: &'a RuntimeConfig,
    pub profile: &'a GenerationProfile,
    pub memory_report: &'a MemoryReport,
    pub tensor_type_counts: BTreeMap<String, (u64, u64)>,
    pub wall_elapsed: Duration,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub max_tokens: usize,
    pub calibration_level_label: String,
    pub calibration_identifier: Option<String>,
    pub calibration_applied_ids: Vec<String>,
    pub calibration_consumed: bool,
}

pub fn build_diagnostic_result(inputs: DiagnosticInputs<'_>) -> DiagnosticResult {
    let rt = inputs.runtime_config;
    let p = inputs.profile;
    let run = &p.runtime;
    let io = &p.io;
    let mem = inputs.memory_report;

    let wall = inputs.wall_elapsed.as_secs_f64();
    let profiled_total = run.total.as_secs_f64();
    let total_elapsed = if wall > 0.0 { wall } else { profiled_total };

    let generated = inputs.generated_tokens as f64;
    let total_tokens = (inputs.prompt_tokens + inputs.generated_tokens) as f64;

    let tokens_per_second_overall = (total_elapsed > 0.0).then(|| total_tokens / total_elapsed);
    let tokens_per_second_generated = if run.decode_forwards > 0 && generated > 0.0 {
        let decode = run.layer_compute.as_secs_f64() + run.dequantization.as_secs_f64()
            - run.prompt.as_secs_f64();
        if decode > 0.0 {
            Some(generated / decode)
        } else if total_elapsed > run.prompt.as_secs_f64() {
            let decode_wall = total_elapsed - run.prompt.as_secs_f64();
            if decode_wall > 0.0 {
                Some(generated / decode_wall)
            } else {
                None
            }
        } else {
            None
        }
    } else if generated > 0.0 && total_elapsed > 0.0 {
        // Fallback: wall-clock rate excluding prompt share.
        Some(generated / total_elapsed)
    } else {
        None
    };

    let decode_elapsed_seconds = (run.decode_forwards > 0).then(|| {
        let t = total_elapsed - run.prompt.as_secs_f64();
        if t > 0.0 {
            t
        } else {
            0.0
        }
    });

    let mut tensor_type_counts = BTreeMap::new();
    let mut tensor_type_bytes = BTreeMap::new();
    for (name, (count, bytes)) in &inputs.tensor_type_counts {
        tensor_type_counts.insert(name.clone(), *count);
        tensor_type_bytes.insert(name.clone(), *bytes);
    }

    let generation = GenerationDiagnostics {
        total_elapsed_seconds: total_elapsed,
        wall_elapsed_seconds: wall,
        prompt_elapsed_seconds: run.prompt.as_secs_f64(),
        decode_elapsed_seconds,
        prompt_tokens: inputs.prompt_tokens,
        generated_tokens: inputs.generated_tokens,
        max_tokens: inputs.max_tokens,
        tokens_per_second_overall,
        tokens_per_second_generated,
        prompt_forwards: run.prompt_forwards,
        decode_forwards: run.decode_forwards,
        terminal_forwards_skipped: run.terminal_forwards_skipped,
        sampling_seconds: run.sampling.as_secs_f64(),
        output_seconds: run.output.as_secs_f64(),
        max_token_latency_seconds: run.max_token_latency.as_secs_f64(),
    };

    let cpu_compute = CpuComputeDiagnostics {
        cpu_threads: rt.cpu_thread_count,
        f32_matvec_seconds: run.float_matvec.as_secs_f64(),
        quantized_matvec_seconds: run.quantized_matvec.as_secs_f64(),
        dequantization_seconds: run.dequantization.as_secs_f64(),
        tensor_construction_seconds: run.tensor_construction.as_secs_f64(),
        layer_compute_seconds: run.layer_compute.as_secs_f64(),
        logits_seconds: run.logits.as_secs_f64(),
        allocation_seconds: run.allocation.as_secs_f64(),
        grouped_quantized_copy_count: run.grouped_quantized_copy_count,
        grouped_quantized_copy_seconds: run.grouped_quantized_copy_time.as_secs_f64(),
        grouped_quantized_copy_bytes: run.grouped_quantized_copy_bytes,
        q4_0_matvec_calls: run.q4_0_matvec_calls,
        q4_0_matvec_rows: run.q4_0_matvec_rows,
        q4_0_matvec_input_elements: run.q4_0_matvec_input_elements,
        q4_0_matvec_blocks: run.q4_0_matvec_blocks,
        q4_0_matvec_weight_bytes: run.q4_0_matvec_weight_bytes,
    };

    let memory = MemoryDiagnostics {
        configured_budget_bytes: rt.ram_budget_bytes,
        managed_current_bytes: mem.ramforge_current_bytes,
        managed_peak_bytes: mem.ramforge_peak_bytes,
        process_rss_bytes: mem.process_rss_bytes,
        system_total_bytes: mem.system_total_bytes,
        system_available_bytes: mem.system_available_bytes,
        layer_cache_capacity_bytes: p.layer_cache_capacity_bytes,
    };

    let cache = CacheDiagnostics {
        hits: run.cache_hits,
        misses: run.cache_misses,
        evictions: run.cache_evictions,
        current_cached_layers: run.cached_layer_count,
        peak_cached_layers: run.peak_cached_layer_count,
        current_cache_bytes: run.cache_bytes,
        peak_cache_bytes: run.peak_cache_bytes,
        capacity_bytes: p.layer_cache_capacity_bytes,
    };

    let io_diag = IoDiagnostics {
        logical_tensor_reads: io.logical_tensor_reads,
        logical_tensor_bytes: io.logical_tensor_bytes,
        physical_reads: io.read_operations,
        physical_bytes: io.bytes_read,
        read_failures: io.read_failures,
        seek_operations: io.seek_operations,
        seeks_avoided: io.seeks_avoided,
        coalesced_ranges: io.coalesced_ranges,
        coalesced_gap_bytes: io.coalesced_gap_bytes,
        read_buffer_reuses: io.read_buffer_reuses,
        read_buffer_growths: io.read_buffer_growths,
        read_elapsed_seconds: io.elapsed.as_secs_f64(),
    };

    let layers = LayerDiagnostics {
        loads: run.layer_loads,
        releases: run.layer_releases,
        load_seconds: run.layer_load.as_secs_f64(),
        compute_seconds: run.layer_compute.as_secs_f64(),
        release_seconds: run.layer_release.as_secs_f64(),
    };

    let quantization = QuantizationDiagnostics {
        tensor_type_counts,
        tensor_type_bytes,
    };

    let calibration = CalibrationDiagnostics {
        level: inputs.calibration_level_label,
        calibration_identifier: inputs.calibration_identifier,
        applied_observation_identifiers: inputs.calibration_applied_ids,
        consumed_by_planning: inputs.calibration_consumed,
    };

    let observations = derive_observations(
        &generation,
        &cpu_compute,
        &memory,
        &cache,
        &io_diag,
        &layers,
    );

    DiagnosticResult {
        schema_version: DIAGNOSTIC_SCHEMA_VERSION,
        generation,
        cpu_compute,
        memory,
        cache,
        io: io_diag,
        layers,
        quantization,
        calibration,
        observations,
    }
}

fn derive_observations(
    _g: &GenerationDiagnostics,
    c: &CpuComputeDiagnostics,
    m: &MemoryDiagnostics,
    cache: &CacheDiagnostics,
    io: &IoDiagnostics,
    layers: &LayerDiagnostics,
) -> Vec<String> {
    let mut out = Vec::new();

    let compute_total = c.f32_matvec_seconds
        + c.quantized_matvec_seconds
        + c.dequantization_seconds
        + c.layer_compute_seconds
        + c.logits_seconds
        + c.tensor_construction_seconds;
    if compute_total > 0.0 {
        let q_share = c.quantized_matvec_seconds / compute_total * 100.0;
        if q_share > 1.0 {
            out.push(format!(
                "Quantized matvec accounts for {:.1}% of measured compute time.",
                q_share
            ));
        }
        let f_share = c.f32_matvec_seconds / compute_total * 100.0;
        if f_share > 1.0 {
            out.push(format!(
                "F32 matvec accounts for {:.1}% of measured compute time.",
                f_share
            ));
        }
    }

    if io.logical_tensor_reads > 0 && io.physical_reads > 0 {
        if io.physical_reads > io.logical_tensor_reads.saturating_mul(2) {
            out.push("Physical GGUF reads substantially exceed logical tensor reads.".to_string());
        } else if io.physical_reads + io.coalesced_ranges < io.logical_tensor_reads {
            out.push(
                "Read coalescing reduced physical reads below logical tensor reads.".to_string(),
            );
        }
    }

    if cache.hits == 0 && (cache.misses > 0 || layers.loads > 0) {
        out.push("Cache hit count was zero during this run.".to_string());
    }
    if cache.evictions > 0 {
        out.push(format!(
            "Layer cache recorded {} eviction(s); cache pressure was present.",
            cache.evictions
        ));
    }

    if m.configured_budget_bytes > 0 {
        let ratio = m.managed_peak_bytes as f64 / m.configured_budget_bytes as f64;
        if ratio < 0.5 {
            out.push("Managed memory remained far below the configured budget.".to_string());
        } else if ratio > 0.9 {
            out.push("Managed memory peaked within 10% of the configured budget.".to_string());
        }
    }

    if layers.compute_seconds > 0.0 && io.read_elapsed_seconds > 0.0 {
        let layer_total = layers.load_seconds + layers.compute_seconds;
        if layer_total > 0.0 {
            let read_share =
                io.read_elapsed_seconds / (layer_total + io.read_elapsed_seconds) * 100.0;
            if read_share > 50.0 {
                out.push("Read time dominates the measured layer execution time.".to_string());
            }
        }
    }

    if io.coalesced_ranges == 0 && io.logical_tensor_reads > 0 {
        out.push("No coalesced read ranges were recorded; each logical read mapped to its own physical request.".to_string());
    }
    if io.read_buffer_reuses > 0 {
        out.push(format!(
            "Grouped read buffers were reused {} time(s) ({} growth(s)).",
            io.read_buffer_reuses, io.read_buffer_growths
        ));
    }

    if c.grouped_quantized_copy_count > 0 {
        out.push(format!(
            "Grouped quantized copies: {} calls, {:.3} s.",
            c.grouped_quantized_copy_count, c.grouped_quantized_copy_seconds
        ));
    }

    out
}

/// Export a diagnostic run record to pretty JSON. Returns the path written.
/// Refuses to overwrite an existing destination.
pub fn export_run_json(record: &RunRecord, path: &Path) -> std::io::Result<()> {
    if path.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("refusing to overwrite existing file: {}", path.display()),
        ));
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let json = serde_json::to_string_pretty(record)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)
}

pub fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn next_run_id(history: &[RunRecord]) -> u64 {
    history.iter().map(|r| r.run_id).max().unwrap_or(0) + 1
}

/// Format a duration in seconds compactly.
pub fn format_seconds(seconds: f64) -> String {
    if seconds.is_nan() || seconds <= 0.0 {
        "0.000 s".to_string()
    } else if seconds < 0.001 {
        format!("{:.3} ms", seconds * 1000.0)
    } else if seconds < 1.0 {
        format!("{:.3} ms", seconds * 1000.0)
    } else if seconds < 120.0 {
        format!("{:.3} s", seconds)
    } else {
        format!("{:.1} s", seconds)
    }
}

pub fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.2} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.2} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

pub fn unavailable() -> &'static str {
    "Unavailable"
}

pub fn not_applicable() -> &'static str {
    "Not Applicable"
}

/// Difference view between two run records for the comparison screen.
#[derive(Debug, Clone, Default)]
pub struct RunComparison {
    pub left_id: u64,
    pub right_id: u64,
    pub fields: Vec<ComparisonField>,
}

#[derive(Debug, Clone)]
pub struct ComparisonField {
    pub label: &'static str,
    pub left: String,
    pub right: String,
}

pub fn compare_runs(left: &RunRecord, right: &RunRecord) -> RunComparison {
    let mut fields = Vec::new();
    let uc_l = &left.user_config;
    let uc_r = &right.user_config;
    let rc_l = &left.runtime_config;
    let rc_r = &right.runtime_config;
    let gp_l = &left.generation_params;
    let gp_r = &right.generation_params;
    let dg_l = &left.diagnostics.generation;
    let dg_r = &right.diagnostics.generation;
    let c_l = &left.diagnostics.cpu_compute;
    let c_r = &right.diagnostics.cpu_compute;
    let m_l = &left.diagnostics.memory;
    let m_r = &right.diagnostics.memory;
    let ca_l = &left.diagnostics.cache;
    let ca_r = &right.diagnostics.cache;
    let io_l = &left.diagnostics.io;
    let io_r = &right.diagnostics.io;

    fields.push(ComparisonField {
        label: "RAM budget",
        left: format_bytes(uc_l.ram_budget_bytes),
        right: format_bytes(uc_r.ram_budget_bytes),
    });
    fields.push(ComparisonField {
        label: "Mode",
        left: uc_l.mode_label.clone(),
        right: uc_r.mode_label.clone(),
    });
    fields.push(ComparisonField {
        label: "CPU threads",
        left: rc_l.cpu_thread_count.to_string(),
        right: rc_r.cpu_thread_count.to_string(),
    });
    fields.push(ComparisonField {
        label: "Layer cache",
        left: format!(
            "{} ({})",
            if rc_l.layer_cache_enabled {
                "on"
            } else {
                "off"
            },
            format_bytes(rc_l.layer_cache_capacity_bytes)
        ),
        right: format!(
            "{} ({})",
            if rc_r.layer_cache_enabled {
                "on"
            } else {
                "off"
            },
            format_bytes(rc_r.layer_cache_capacity_bytes)
        ),
    });
    fields.push(ComparisonField {
        label: "Read coalescing",
        left: if rc_l.read_coalescing_enabled {
            "on"
        } else {
            "off"
        }
        .to_string(),
        right: if rc_r.read_coalescing_enabled {
            "on"
        } else {
            "off"
        }
        .to_string(),
    });
    fields.push(ComparisonField {
        label: "Max tokens",
        left: gp_l.max_tokens.to_string(),
        right: gp_r.max_tokens.to_string(),
    });
    fields.push(ComparisonField {
        label: "Elapsed",
        left: format_seconds(dg_l.total_elapsed_seconds),
        right: format_seconds(dg_r.total_elapsed_seconds),
    });
    fields.push(ComparisonField {
        label: "Generated tokens",
        left: dg_l.generated_tokens.to_string(),
        right: dg_r.generated_tokens.to_string(),
    });
    fields.push(ComparisonField {
        label: "Tokens/sec (gen)",
        left: dg_l
            .tokens_per_second_generated
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| unavailable().to_string()),
        right: dg_r
            .tokens_per_second_generated
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| unavailable().to_string()),
    });
    fields.push(ComparisonField {
        label: "Prompt time",
        left: format_seconds(dg_l.prompt_elapsed_seconds),
        right: format_seconds(dg_r.prompt_elapsed_seconds),
    });
    fields.push(ComparisonField {
        label: "Quantized matvec",
        left: format_seconds(c_l.quantized_matvec_seconds),
        right: format_seconds(c_r.quantized_matvec_seconds),
    });
    fields.push(ComparisonField {
        label: "F32 matvec",
        left: format_seconds(c_l.f32_matvec_seconds),
        right: format_seconds(c_r.f32_matvec_seconds),
    });
    fields.push(ComparisonField {
        label: "Dequantization",
        left: format_seconds(c_l.dequantization_seconds),
        right: format_seconds(c_r.dequantization_seconds),
    });
    fields.push(ComparisonField {
        label: "Layer compute",
        left: format_seconds(c_l.layer_compute_seconds),
        right: format_seconds(c_r.layer_compute_seconds),
    });
    fields.push(ComparisonField {
        label: "Read time",
        left: format_seconds(io_l.read_elapsed_seconds),
        right: format_seconds(io_r.read_elapsed_seconds),
    });
    fields.push(ComparisonField {
        label: "I/O physical reads",
        left: io_l.physical_reads.to_string(),
        right: io_r.physical_reads.to_string(),
    });
    fields.push(ComparisonField {
        label: "I/O physical bytes",
        left: format_bytes(io_l.physical_bytes),
        right: format_bytes(io_r.physical_bytes),
    });
    fields.push(ComparisonField {
        label: "Cache hits",
        left: ca_l.hits.to_string(),
        right: ca_r.hits.to_string(),
    });
    fields.push(ComparisonField {
        label: "Cache misses",
        left: ca_l.misses.to_string(),
        right: ca_r.misses.to_string(),
    });
    fields.push(ComparisonField {
        label: "Cache evictions",
        left: ca_l.evictions.to_string(),
        right: ca_r.evictions.to_string(),
    });
    fields.push(ComparisonField {
        label: "Managed peak",
        left: format_bytes(m_l.managed_peak_bytes),
        right: format_bytes(m_r.managed_peak_bytes),
    });

    RunComparison {
        left_id: left.run_id,
        right_id: right.run_id,
        fields,
    }
}

/// Collect per-tensor-type counts/bytes from a list of TensorReadProfile's
/// tensor name → type mapping is not embedded in TensorReadProfile, so we
/// accept caller-provided type-name -> (count,bytes) pairs.
pub fn tensor_type_counts_from_profiles(
    _tensor_reads: &[TensorReadProfile],
    provided: BTreeMap<String, (u64, u64)>,
) -> BTreeMap<String, (u64, u64)> {
    provided
}

// Helper accessors for profiles
pub fn io_profile_ref(profile: &GenerationProfile) -> &IoProfile {
    &profile.io
}

pub fn runtime_profile_ref(profile: &GenerationProfile) -> &ProfileSnapshot {
    &profile.runtime
}
