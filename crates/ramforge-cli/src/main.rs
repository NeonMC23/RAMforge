#![recursion_limit = "256"]

use clap::{Parser, Subcommand, ValueEnum};
use ramforge_core::{
    discover_model_set, parse_memory_size, ChatMessage, ChatTemplate, GgufModel, GgufModelSet,
    Tokenizer,
};
use ramforge_runtime::execution_profile::{
    inspect_profile, load_profile, ExecutionProfile, ProfileIdentity,
};
use ramforge_runtime::graph_manifest::{GraphManifest, LastUseStatus};
use ramforge_runtime::memory_ledger::MemoryLedger;
use ramforge_runtime::memory_report::{process_rss, ForgeCoreMemoryReport};
use ramforge_runtime::page_residency::{
    correlate_prefetch_residency, evict_file_cache, model_mapping_residency,
    observe_page_residency, PageObservationConfig, PageRange, PageResidencySnapshot,
    PrefetchCorrelationReport,
};
use ramforge_runtime::prediction::{
    load_prediction_history, predict_from_profile, PredictionConfig, PredictionHistory,
    PredictionMetrics, PredictionSet,
};
use ramforge_runtime::prediction_planner::{
    evaluate_predictions, execute_staging, plan_predictions, PredictionPlan, StagingExecution,
};
use ramforge_runtime::residency_engine::process_faults;
use ramforge_runtime::residency_engine::{
    AdmissionDecision, AdmissionOutcome, DecisionAction, ExecutionTiming, ManagerError,
    ModelGeneration, ModelId, OperationContext, OperationTiming, ResidencyCapabilities,
    ResidencyManager, ResidencyPlan, ResidencyPolicy, ResidencyPolicyMode, ResidencyReport,
    ResidencySnapshot, ResourceInventory, StagingPlan, StagingResult,
};
use ramforge_runtime::{
    DeviceRegistry, DeviceSelection, ExecutionConfig, ForgeCoreContextOptions, ForgeCoreLogits,
    ForgeCoreModel, ForgeCoreModelOptions, ForgeCoreSession,
};
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(
    name = "ramforge",
    version,
    about = "RAMforge – usable, profiled out-of-core GGUF inference"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum InferenceBackend {
    /// Existing RAMforge custom CPU engine; retained as the default reference path.
    Legacy,
    /// ForgeCore/libllama model/context/batch/decode/logits path (CPU in the current RAMforge adapter).
    #[value(name = "forgecore", alias = "forge-core")]
    ForgeCore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ResidencyPolicyArg {
    Off,
    Observe,
    Conservative,
    MemoryBound,
    ReuseAware,
    Active,
}

impl ResidencyPolicyArg {
    fn mode(self) -> ResidencyPolicyMode {
        match self {
            Self::Off => ResidencyPolicyMode::Off,
            Self::Observe => ResidencyPolicyMode::Observe,
            Self::Conservative => ResidencyPolicyMode::Conservative,
            Self::MemoryBound => ResidencyPolicyMode::MemoryBound,
            Self::ReuseAware => ResidencyPolicyMode::ReuseAware,
            Self::Active => ResidencyPolicyMode::Active,
        }
    }

    fn applies(self) -> bool {
        matches!(
            self,
            Self::Conservative | Self::MemoryBound | Self::ReuseAware | Self::Active
        )
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum Commands {
    /// Inspect a GGUF model file and report metadata
    Inspect {
        /// Path to the GGUF model file
        model: PathBuf,

        /// Output in JSON format
        #[arg(long)]
        json: bool,

        /// Show detailed tensor list (first N tensors)
        #[arg(long, default_value_t = 20)]
        max_tensors: usize,
    },
    /// Plan execution with a real RAM budget and file-backed access
    Plan {
        /// Path to the GGUF model file
        model: PathBuf,

        /// RAM budget (e.g. 8G, 8GiB, 8192M, 512MiB)
        #[arg(long)]
        ram: String,

        /// Output in JSON format
        #[arg(long)]
        json: bool,
    },
    /// Show explicit architecture, tokenizer, and quantization capabilities
    Support,
    /// Inspect/validate a persisted execution profile
    Profile {
        /// Profile JSON path
        path: PathBuf,

        /// Model path used to validate model identity
        #[arg(long, value_name = "MODEL")]
        model: Option<PathBuf>,

        /// Inventory generation expected by the profile
        #[arg(long, default_value_t = 1)]
        inventory_generation: u64,

        /// Output in JSON format
        #[arg(long)]
        json: bool,
    },
    /// Run a deterministic prediction/residency simulation
    Simulate {
        /// Simulated memory budget
        #[arg(long, default_value = "12K")]
        budget: String,

        /// Simulated staging window
        #[arg(long, default_value = "8K")]
        window: String,

        /// Use a deliberately mismatched test sequence
        #[arg(long)]
        wrong_history: bool,

        /// Output in JSON format
        #[arg(long)]
        json: bool,
    },
    /// Run real inference; select legacy or explicit ForgeCore execution
    Run {
        /// Path to the GGUF model file
        model: PathBuf,

        /// RAM budget (e.g. 8G, 8GiB, 8192M, 512MiB)
        #[arg(long)]
        ram: String,

        /// Prompt text
        #[arg(long)]
        prompt: String,

        /// Apply the model's tokenizer.chat_template to a single user message
        #[arg(long)]
        chat: bool,

        /// Optional system message used with --chat
        #[arg(long, requires = "chat")]
        system: Option<String>,

        /// Explicit inference backend; legacy remains the default comparison path
        #[arg(long, value_enum, default_value_t = InferenceBackend::Legacy)]
        backend: InferenceBackend,

        /// Static ForgeCore device: cpu, id:<device-id>, or backend:<backend-id>:<ForgeCore-registry-index>
        #[arg(long, default_value = "cpu")]
        device: String,

        /// Context length override in tokens (ForgeCore only; default: model training context)
        #[arg(long)]
        ctx: Option<u32>,

        /// Disable llama.cpp CPU weight repacking; keeps quantized weights in the file-backed mapping
        #[arg(long)]
        no_repack: bool,

        /// Skip whole-file prefaulting of the model mapping (forgeCORE patched native mode)
        #[arg(long)]
        no_prefetch: bool,

        /// Release resident model-mapping pages after load (forgeCORE patched native mode)
        #[arg(long)]
        release_mapping_pages: bool,

        /// Print tensor-to-mapping ranges for layer N (blk.N.* tensor naming)
        #[arg(long, value_name = "N", conflicts_with_all = ["release_layer", "release_mapping_pages"])]
        inspect_layer: Option<u32>,

        /// Release eligible mapping pages for layer N (blk.N.* tensor naming)
        #[arg(long, value_name = "N", conflicts_with_all = ["inspect_layer", "release_mapping_pages"])]
        release_layer: Option<u32>,

        /// Residency policy: off, observe, conservative, memory-bound, reuse-aware, or active (compatibility)
        #[arg(long, value_enum, default_value_t = ResidencyPolicyArg::Off)]
        residency_policy: ResidencyPolicyArg,

        /// Soft residency target (default: --ram). Not a hard limit.
        #[arg(long, value_name = "SIZE")]
        residency_target: Option<String>,

        /// Safety reserve subtracted from the soft target
        #[arg(long, value_name = "SIZE")]
        residency_reserve: Option<String>,

        /// Print the structured residency inventory/plan/application report
        #[arg(long)]
        residency_report: bool,

        /// Write a versioned JSON residency report to PATH
        #[arg(long, value_name = "PATH")]
        residency_report_json: Option<PathBuf>,

        /// Load a validated execution profile from PATH (reuse-aware/staging input)
        #[arg(long, value_name = "PATH")]
        residency_profile_load: Option<PathBuf>,

        /// Save the current complete trace profile atomically to PATH
        #[arg(long, value_name = "PATH", requires = "graph_trace")]
        residency_profile_save: Option<PathBuf>,

        /// Observe bounded per-resource page residency with Linux mincore
        #[arg(long)]
        residency_observe: bool,

        /// Maximum pages inspected per residency snapshot
        #[arg(
            long,
            value_name = "N",
            default_value_t = 4096,
            requires = "residency_observe"
        )]
        residency_observe_pages: usize,

        /// Best-effort bounded page prefetch budget (MADV_WILLNEED hint)
        #[arg(long, value_name = "SIZE", conflicts_with = "prediction")]
        residency_prefetch: Option<String>,

        /// Enable deterministic predictive staging from a loaded profile
        #[arg(long)]
        prediction: bool,

        /// Minimum prediction confidence (percent, 0-100)
        #[arg(long, value_name = "PERCENT", requires = "prediction")]
        prediction_min_confidence: Option<u8>,

        /// Bound on predicted page bytes considered for staging
        #[arg(long, value_name = "SIZE", requires = "prediction")]
        prediction_window: Option<String>,

        /// Portion of the staging window reserved for low-priority speculative hints
        #[arg(long, value_name = "SIZE", requires = "prediction")]
        prediction_speculative: Option<String>,

        /// Maximum configured in-flight staging operations (synchronous implementation executes one native call)
        #[arg(long, value_name = "N", default_value_t = 1, requires = "prediction")]
        prediction_max_inflight: u16,

        /// Load a validated prediction feedback history from PATH
        #[arg(long, value_name = "PATH", requires = "prediction")]
        prediction_history_load: Option<PathBuf>,

        /// Save prediction feedback history atomically to PATH (requires graph trace)
        #[arg(long, value_name = "PATH", requires_all = ["prediction", "graph_trace"])]
        prediction_history_save: Option<PathBuf>,

        /// Enable bounded scheduler graph tracing (adds per-node synchronization overhead)
        #[arg(long)]
        graph_trace: bool,

        /// Maximum retained graph trace events
        #[arg(long, default_value_t = 4096, requires = "graph_trace")]
        graph_trace_events: usize,

        /// Print individual graph trace events
        #[arg(long, requires = "graph_trace")]
        graph_trace_detail: bool,

        /// Enforce --ram as a hard cgroup v2 memory.max limit for the run
        #[arg(long)]
        enforce_ram: bool,

        /// Wall-clock timeout for an enforced run, in seconds
        #[arg(long, default_value_t = 900, requires = "enforce_ram")]
        enforce_timeout: u64,

        /// Maximum tokens to generate
        #[arg(long, default_value_t = 32)]
        max_tokens: usize,

        /// Temperature (0 = greedy)
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,

        /// Top-k sampling (optional)
        #[arg(long)]
        top_k: Option<usize>,

        /// Top-p sampling (optional)
        #[arg(long)]
        top_p: Option<f32>,

        /// Verbose diagnostics including residency stats
        #[arg(long)]
        verbose: bool,

        /// Measure generation I/O, layer, compute, allocation, and token timings
        #[arg(long)]
        profile: bool,

        /// Report RAMforge-managed, process RSS, and system memory separately
        #[arg(long)]
        memory_report: bool,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Inspect {
            model,
            json,
            max_tensors,
        } => {
            if let Err(e) = run_inspect(model, json, max_tensors) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Plan { model, ram, json } => {
            if let Err(e) = run_plan(model, ram, json) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Support => output_support(),
        Commands::Profile {
            path,
            model,
            inventory_generation,
            json,
        } => {
            if let Err(error) =
                run_profile_inspect(&path, model.as_deref(), inventory_generation, json)
            {
                eprintln!("Error: {error}");
                std::process::exit(1);
            }
        }
        Commands::Simulate {
            budget,
            window,
            wrong_history,
            json,
        } => {
            if let Err(error) = run_simulation(&budget, &window, wrong_history, json) {
                eprintln!("Error: {error}");
                std::process::exit(1);
            }
        }
        Commands::Run {
            model,
            ram,
            prompt,
            chat,
            system,
            backend,
            device,
            ctx,
            no_repack,
            no_prefetch,
            release_mapping_pages,
            inspect_layer,
            release_layer,
            residency_policy,
            residency_target,
            residency_reserve,
            residency_report,
            residency_report_json,
            residency_profile_load,
            residency_profile_save,
            residency_prefetch,
            residency_observe,
            residency_observe_pages,
            prediction,
            prediction_min_confidence,
            prediction_window,
            prediction_speculative,
            prediction_max_inflight,
            prediction_history_load,
            prediction_history_save,
            graph_trace,
            graph_trace_events,
            graph_trace_detail,
            enforce_ram,
            enforce_timeout,
            max_tokens,
            temperature,
            top_k,
            top_p,
            verbose,
            profile,
            memory_report,
        } => {
            // The outer process creates the kernel limit and re-execs the same
            // command. The child sees RAMFORGE_SUPERVISED and runs normally.
            if enforce_ram && !ramforge_cli::supervisor::is_supervised() {
                let ram_bytes = match parse_memory_size(&ram) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        eprintln!("Error: invalid --ram '{}': {}", ram, error);
                        std::process::exit(1);
                    }
                };
                if enforce_timeout == 0 {
                    eprintln!("Error: --enforce-timeout must be greater than zero");
                    std::process::exit(2);
                }
                match ramforge_cli::supervisor::run_supervised(
                    ram_bytes,
                    Duration::from_secs(enforce_timeout),
                ) {
                    Ok(outcome) => {
                        ramforge_cli::supervisor::print_report(&outcome, ram_bytes);
                        std::process::exit(outcome.exit_code);
                    }
                    Err(error) => {
                        eprintln!("Error: {error}");
                        std::process::exit(1);
                    }
                }
            }
            if let Err(e) = run_inference(
                model,
                ram,
                prompt,
                chat,
                system,
                backend,
                device,
                ctx,
                no_repack,
                no_prefetch,
                release_mapping_pages,
                inspect_layer,
                release_layer,
                residency_policy,
                residency_target,
                residency_reserve,
                residency_report,
                residency_report_json,
                residency_profile_load,
                residency_profile_save,
                residency_prefetch,
                residency_observe,
                residency_observe_pages,
                prediction,
                prediction_min_confidence,
                prediction_window,
                prediction_speculative,
                prediction_max_inflight,
                prediction_history_load,
                prediction_history_save,
                graph_trace,
                graph_trace_events,
                graph_trace_detail,
                max_tokens,
                temperature,
                top_k,
                top_p,
                verbose,
                profile,
                memory_report,
            ) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
    }
}

fn run_inspect(model_path: PathBuf, json_output: bool, max_tensors: usize) -> anyhow::Result<()> {
    let model_set = discover_model_set(&model_path).map_err(|e| anyhow::anyhow!(e))?;
    let gguf = model_set.merged_model();

    if json_output {
        output_json(&gguf, max_tensors, Some(&model_set))?;
    } else {
        output_human(&gguf, max_tensors, Some(&model_set));
    }

    Ok(())
}

fn run_plan(model_path: PathBuf, ram_str: String, json_output: bool) -> anyhow::Result<()> {
    let ram_bytes = parse_memory_size(&ram_str)
        .map_err(|e| anyhow::anyhow!("invalid --ram '{}': {}", ram_str, e))?;

    let model_set = discover_model_set(&model_path).map_err(|e| anyhow::anyhow!(e))?;
    let gguf = model_set.merged_model();

    let plan =
        ramforge_runtime::plan::plan_model(&gguf, ram_bytes).map_err(|e| anyhow::anyhow!(e))?;

    if json_output {
        output_plan_json(&plan, &gguf)?;
    } else {
        output_plan_human(&plan, &gguf, &ram_str);
    }

    Ok(())
}

fn prepare_prompt(
    model: &GgufModel,
    prompt: String,
    chat: bool,
    system: Option<String>,
) -> anyhow::Result<String> {
    if !chat {
        if system.is_some() {
            return Err(anyhow::anyhow!("--system requires --chat"));
        }
        return Ok(prompt);
    }

    let template = ChatTemplate::from_gguf(model)
        .map_err(|error| anyhow::anyhow!("--chat requested but {error}"))?;
    let mut messages = Vec::new();
    if let Some(system) = system {
        messages.push(ChatMessage::system(system));
    }
    messages.push(ChatMessage::user(prompt));
    let rendered = template
        .render(&messages, true)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    if rendered.is_empty() {
        return Err(anyhow::anyhow!(
            "tokenizer.chat_template rendered an empty prompt"
        ));
    }
    Ok(rendered)
}

#[allow(clippy::too_many_arguments)]
fn run_inference(
    model_path: PathBuf,
    ram_str: String,
    prompt: String,
    chat: bool,
    system: Option<String>,
    backend: InferenceBackend,
    device: String,
    ctx: Option<u32>,
    no_repack: bool,
    no_prefetch: bool,
    release_mapping_pages: bool,
    inspect_layer: Option<u32>,
    release_layer: Option<u32>,
    residency_policy: ResidencyPolicyArg,
    residency_target: Option<String>,
    residency_reserve: Option<String>,
    residency_report: bool,
    residency_report_json: Option<PathBuf>,
    residency_profile_load: Option<PathBuf>,
    residency_profile_save: Option<PathBuf>,
    residency_prefetch: Option<String>,
    residency_observe: bool,
    residency_observe_pages: usize,
    prediction: bool,
    prediction_min_confidence: Option<u8>,
    prediction_window: Option<String>,
    prediction_speculative: Option<String>,
    prediction_max_inflight: u16,
    prediction_history_load: Option<PathBuf>,
    prediction_history_save: Option<PathBuf>,
    graph_trace: bool,
    graph_trace_events: usize,
    graph_trace_detail: bool,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    verbose: bool,
    profile: bool,
    memory_report: bool,
) -> anyhow::Result<()> {
    let ram_bytes = parse_memory_size(&ram_str)
        .map_err(|e| anyhow::anyhow!("invalid --ram '{}': {}", ram_str, e))?;
    let residency_target_bytes = residency_target
        .as_deref()
        .map(parse_memory_size)
        .transpose()
        .map_err(|error| anyhow::anyhow!("invalid --residency-target: {error}"))?;
    let residency_reserve_bytes = residency_reserve
        .as_deref()
        .map(parse_memory_size)
        .transpose()
        .map_err(|error| anyhow::anyhow!("invalid --residency-reserve: {error}"))?
        .unwrap_or(0);
    let residency_prefetch_bytes = residency_prefetch
        .as_deref()
        .map(parse_memory_size)
        .transpose()
        .map_err(|error| anyhow::anyhow!("invalid --residency-prefetch: {error}"))?;
    if residency_prefetch_bytes == Some(0) {
        return Err(anyhow::anyhow!(
            "--residency-prefetch must be greater than zero"
        ));
    }
    if residency_target_bytes == Some(0) {
        return Err(anyhow::anyhow!(
            "--residency-target must be greater than zero"
        ));
    }
    let default_prediction = PredictionConfig::default();
    let prediction_min_bps = prediction_min_confidence
        .map(|percent| u16::from(percent).saturating_mul(100))
        .unwrap_or(default_prediction.min_confidence_bps);
    let prediction_window_bytes = prediction_window
        .as_deref()
        .map(parse_memory_size)
        .transpose()
        .map_err(|error| anyhow::anyhow!("invalid --prediction-window: {error}"))?
        .unwrap_or(default_prediction.staging_window_bytes);
    let prediction_speculative_bytes = prediction_speculative
        .as_deref()
        .map(parse_memory_size)
        .transpose()
        .map_err(|error| anyhow::anyhow!("invalid --prediction-speculative: {error}"))?
        .unwrap_or(default_prediction.speculative_budget_bytes);
    let prediction_config = PredictionConfig {
        enabled: prediction,
        min_confidence_bps: prediction_min_bps,
        staging_window_bytes: prediction_window_bytes,
        speculative_budget_bytes: prediction_speculative_bytes,
        max_in_flight: prediction_max_inflight,
        lookahead: default_prediction.lookahead,
    };
    prediction_config
        .validate()
        .map_err(|error| anyhow::anyhow!(error))?;
    if prediction && residency_profile_load.is_none() {
        return Err(anyhow::anyhow!(
            "--prediction requires --residency-profile-load"
        ));
    }
    if prediction && residency_prefetch_bytes.is_some() {
        return Err(anyhow::anyhow!(
            "--prediction and --residency-prefetch are mutually exclusive"
        ));
    }
    if let Some(target) = residency_target_bytes {
        if residency_reserve_bytes > target {
            return Err(anyhow::anyhow!(
                "--residency-reserve {} exceeds --residency-target {}",
                format_bytes(residency_reserve_bytes),
                format_bytes(target)
            ));
        }
    }
    if residency_policy == ResidencyPolicyArg::Off
        && !residency_report
        && residency_report_json.is_none()
        && (residency_target_bytes.is_some() || residency_reserve_bytes != 0)
    {
        return Err(anyhow::anyhow!(
            "--residency-target/--residency-reserve require --residency-policy observe|conservative|memory-bound|reuse-aware|active or --residency-report/--residency-report-json"
        ));
    }
    if graph_trace && graph_trace_events == 0 {
        return Err(anyhow::anyhow!(
            "--graph-trace-events must be greater than zero"
        ));
    }
    let device_selection = device
        .parse::<DeviceSelection>()
        .map_err(|error| anyhow::anyhow!(error))?;

    let residency_requested = residency_policy != ResidencyPolicyArg::Off
        || residency_target_bytes.is_some()
        || residency_reserve_bytes != 0
        || residency_report
        || residency_report_json.is_some()
        || residency_profile_load.is_some()
        || residency_profile_save.is_some()
        || residency_prefetch_bytes.is_some()
        || residency_observe
        || prediction
        || prediction_history_load.is_some()
        || prediction_history_save.is_some();
    if residency_requested
        && (release_mapping_pages || inspect_layer.is_some() || release_layer.is_some())
    {
        return Err(anyhow::anyhow!(
            "manual mapping release/inspection cannot be combined with residency policy controls"
        ));
    }
    if graph_trace && !matches!(backend, InferenceBackend::ForgeCore) {
        return Err(anyhow::anyhow!(
            "--graph-trace is only supported with --backend forgecore"
        ));
    }
    if matches!(backend, InferenceBackend::ForgeCore) {
        return run_forgecore_inference(
            model_path,
            ram_str,
            ram_bytes,
            prompt,
            chat,
            system,
            device_selection.clone(),
            ctx,
            no_repack,
            no_prefetch,
            release_mapping_pages,
            inspect_layer,
            release_layer,
            residency_policy,
            residency_target_bytes,
            residency_reserve_bytes,
            residency_report,
            residency_report_json.clone(),
            residency_profile_load.clone(),
            residency_profile_save.clone(),
            residency_prefetch_bytes,
            residency_observe,
            residency_observe_pages,
            prediction_config,
            prediction,
            prediction_history_load.clone(),
            prediction_history_save.clone(),
            graph_trace,
            graph_trace_events,
            graph_trace_detail,
            max_tokens,
            temperature,
            top_k,
            top_p,
            verbose,
            profile,
            memory_report,
        );
    }
    if !matches!(device_selection, DeviceSelection::Cpu) {
        return Err(anyhow::anyhow!(
            "--device '{}' is only supported with --backend forgecore; legacy execution remains CPU-only",
            device
        ));
    }
    if ctx.is_some() {
        return Err(anyhow::anyhow!(
            "--ctx is only supported with --backend forgecore; the legacy executor uses the model's stored context length"
        ));
    }
    if no_repack {
        return Err(anyhow::anyhow!(
            "--no-repack is only supported with --backend forgecore"
        ));
    }
    if no_prefetch {
        return Err(anyhow::anyhow!(
            "--no-prefetch is only supported with --backend forgecore"
        ));
    }
    if release_mapping_pages {
        return Err(anyhow::anyhow!(
            "--release-mapping-pages is only supported with --backend forgecore"
        ));
    }
    if inspect_layer.is_some() || release_layer.is_some() {
        return Err(anyhow::anyhow!(
            "--inspect-layer and --release-layer are only supported with --backend forgecore"
        ));
    }
    let residency_requested = residency_policy != ResidencyPolicyArg::Off
        || residency_target_bytes.is_some()
        || residency_reserve_bytes != 0
        || residency_report
        || residency_report_json.is_some()
        || residency_profile_load.is_some()
        || residency_profile_save.is_some()
        || residency_prefetch_bytes.is_some()
        || residency_observe
        || prediction
        || prediction_history_load.is_some()
        || prediction_history_save.is_some();
    if residency_requested {
        return Err(anyhow::anyhow!(
            "residency policy, target, reserve, prefetch, and report controls are only supported with --backend forgecore"
        ));
    }
    if residency_requested
        && (release_mapping_pages || inspect_layer.is_some() || release_layer.is_some())
    {
        return Err(anyhow::anyhow!(
            "manual mapping release/inspection cannot be combined with residency policy controls"
        ));
    }

    // Diagnostics to stderr
    eprintln!("RAMforge – Usable Out-of-Core Inference");
    eprintln!("========================================================");
    eprintln!("Model: {}", model_path.display());
    if let Some(limit) = ramforge_cli::supervisor::enforced_budget_bytes() {
        eprintln!(
            "RAM budget: {} ({} bytes; ENFORCED cgroup v2 memory.max={} bytes)",
            ram_str, ram_bytes, limit
        );
    } else {
        eprintln!(
            "RAM budget: {} ({} bytes; managed budget; pass --enforce-ram for a hard OS cap)",
            ram_str, ram_bytes
        );
    }
    eprintln!("Max tokens: {}", max_tokens);
    eprintln!("Temperature: {}", temperature);
    if let Some(k) = top_k {
        eprintln!("Top-k: {}", k);
    }
    if let Some(p) = top_p {
        eprintln!("Top-p: {}", p);
    }
    eprintln!("Execution model: Layer-wise streaming (load one layer → compute → release)");
    eprintln!("Memory model: every weight/KV/temp allocation charged to the RAM budget");
    eprintln!();

    // Create inference engine – file-backed GgufDataSource + MemoryBudget
    eprintln!("Loading model (file-backed, persistent weights only)...");
    let model_load_started = Instant::now();
    let mut engine = ramforge_runtime::inference::InferenceEngine::new(
        model_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("invalid model path"))?,
        ram_bytes,
    )
    .map_err(|e| anyhow::anyhow!(e))?;
    let model_load_elapsed = model_load_started.elapsed();
    engine.set_profiling(profile);
    let raw_prompt = prompt;
    let prompt = prepare_prompt(engine.data_source.model(), raw_prompt.clone(), chat, system)?;
    eprintln!("Prompt: {:?}", raw_prompt);
    if chat {
        eprintln!("Chat template: applied (rendered {} bytes)", prompt.len());
        eprintln!("Chat prompt: {:?}", prompt);
    }

    let detected_architecture = engine
        .data_source
        .model()
        .info()
        .architecture
        .unwrap_or_else(|| "unknown".to_string());
    eprintln!("Model config: arch={} vocab={}, context={}, embedding={}, layers={}, heads={}, kv_heads={}, ffn={}, head_dim={}",
        detected_architecture,
        engine.config().vocab_size,
        engine.config().context_length,
        engine.config().embedding_length,
        engine.config().block_count,
        engine.config().head_count,
        engine.config().head_count_kv,
        engine.config().feed_forward_length,
        engine.config().head_dim
    );
    eprintln!(
        "Tokenizer: model={}, vocab_size={}, bos={:?}, eos={:?}",
        engine.tokenizer.model,
        engine.tokenizer.vocab_size(),
        engine.tokenizer.bos_id,
        engine.tokenizer.eos_id
    );
    eprintln!(
        "Execution backend: CPU ({} mode)",
        ramforge_runtime::backend::ComputeBackend::name(&engine.backend)
    );
    eprintln!(
        "Memory budget: total={} used={} (resident weights) available={}",
        engine.budget.total_bytes(),
        engine.budget.used_bytes(),
        engine.budget.available_bytes()
    );
    eprintln!(
        "Total model weight bytes: {} (from descriptors)",
        engine.model.total_weight_bytes
    );
    eprintln!();

    let sampler = ramforge_runtime::sampling::Sampler::new(temperature, top_k, top_p);

    eprintln!("Generating with layer streaming...");
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    write!(stdout, "Assistant: ")?;
    stdout.flush()?;
    let generation = engine.generate_with_callback(&prompt, max_tokens, &sampler, |piece| {
        write!(stdout, "{}", piece).map_err(|error| error.to_string())?;
        stdout.flush().map_err(|error| error.to_string())
    });
    writeln!(stdout)?;
    stdout.flush()?;
    let outcome = generation.map_err(|e| anyhow::anyhow!(e))?;

    let eos_label = match outcome.eos_token_id {
        Some(id) => id.to_string(),
        None => "<unavailable>".to_string(),
    };
    eprintln!("Generated {} tokens", outcome.generated_token_ids.len());
    eprintln!(
        "Stop reason: {} (eos_id={}, eos_encountered={}, configured_max_tokens={}, context_length={})",
        outcome.stop_reason.label(),
        eos_label,
        outcome.eos_encountered,
        outcome.configured_max_tokens,
        outcome.context_length,
    );
    eprintln!(
        "State reset: kv_cache_reset={} token_history_reset={} sampler_stateless={}",
        outcome.kv_cache_reset, outcome.token_history_reset, outcome.sampler_is_stateless,
    );

    if verbose {
        eprintln!(
            "Budget after run: used={} available={} charges={:?}",
            engine.budget.used_bytes(),
            engine.budget.available_bytes(),
            engine.budget.allocations().keys().collect::<Vec<_>>()
        );
        let stats = &engine.residency_stats;
        eprintln!();
        eprintln!("Residency stats (verbose):");
        eprintln!(
            "  Total model weight bytes: {} ({:.2} MiB)",
            stats.total_model_weight_bytes,
            stats.total_model_weight_bytes as f64 / (1024.0 * 1024.0)
        );
        eprintln!(
            "  Current resident layer bytes: {}",
            stats.current_resident_layer_bytes
        );
        eprintln!(
            "  Peak resident layer bytes: {} ({:.2} MiB)",
            stats.peak_resident_layer_bytes,
            stats.peak_resident_layer_bytes as f64 / (1024.0 * 1024.0)
        );
        eprintln!(
            "  Peak managed bytes: {} ({:.2} MiB) / budget {} ({:.2} MiB)",
            stats.peak_managed_bytes,
            stats.peak_managed_bytes as f64 / (1024.0 * 1024.0),
            ram_bytes,
            ram_bytes as f64 / (1024.0 * 1024.0)
        );
        eprintln!("  Layer loads: {}", stats.num_layer_loads);
        eprintln!("  Layer releases: {}", stats.num_layer_releases);
        eprintln!(
            "  Cache placements (cumulative transitions): {}",
            stats.num_layer_cached
        );
        eprintln!(
            "  Current cached layers: {} ({} bytes)",
            stats.current_cached_layer_count, stats.current_cached_bytes
        );
        eprintln!(
            "  Peak cached layers: {} ({} bytes, {:.2} MiB)",
            stats.peak_cached_layer_count,
            stats.peak_cached_bytes,
            stats.peak_cached_bytes as f64 / (1024.0 * 1024.0)
        );
        eprintln!(
            "  Cache hits: {} misses: {} evictions: {}",
            stats.cache_hits, stats.cache_misses, stats.cache_evictions
        );
        eprintln!(
            "  Fits check: total {} > budget {} ? {}",
            stats.total_model_weight_bytes,
            ram_bytes,
            stats.total_model_weight_bytes > ram_bytes
        );
        eprintln!(
            "  Peak layer < total ? {}",
            stats.peak_resident_layer_bytes < stats.total_model_weight_bytes
        );
        eprintln!(
            "  Peak managed <= budget ? {}",
            stats.peak_managed_bytes <= ram_bytes
        );
    }

    if profile {
        output_generation_profile(&engine.generation_profile(), model_load_elapsed);
    }
    if memory_report || profile {
        output_memory_report(&engine.memory_report());
    }

    eprintln!();

    Ok(())
}

fn decode_with_residency(
    session: &mut ForgeCoreSession,
    tokens: &[u32],
    position: u32,
    mut residency: Option<&mut ResidencyManager>,
) -> anyhow::Result<ForgeCoreLogits> {
    if let Some(manager) = residency.as_mut() {
        manager
            .begin_decode()
            .map_err(|error| anyhow::anyhow!(error))?;
    }
    let decoded = session
        .decode_tokens(tokens, position)
        .map_err(|error| anyhow::anyhow!(error));
    let logits = match decoded {
        Ok(logits) => {
            // decode_tokens reads logits through a synchronizing native getter;
            // make the lifecycle boundary explicit before any residency mutation.
            session.synchronize();
            if let Some(manager) = residency.as_mut() {
                manager
                    .finish_decode()
                    .map_err(|error| anyhow::anyhow!(error))?;
            }
            logits
        }
        Err(error) => {
            if let Some(manager) = residency.as_mut() {
                manager.fail();
            }
            return Err(error);
        }
    };
    Ok(logits)
}

#[allow(clippy::too_many_arguments)]
fn run_forgecore_inference(
    model_path: PathBuf,
    ram_str: String,
    ram_bytes: u64,
    prompt: String,
    chat: bool,
    system: Option<String>,
    device: DeviceSelection,
    ctx: Option<u32>,
    no_repack: bool,
    no_prefetch: bool,
    release_mapping_pages: bool,
    inspect_layer: Option<u32>,
    release_layer: Option<u32>,
    residency_policy: ResidencyPolicyArg,
    residency_target: Option<u64>,
    residency_reserve: u64,
    residency_report: bool,
    residency_report_json: Option<PathBuf>,
    residency_profile_load: Option<PathBuf>,
    residency_profile_save: Option<PathBuf>,
    residency_prefetch_bytes: Option<u64>,
    residency_observe: bool,
    residency_observe_pages: usize,
    prediction_config: PredictionConfig,
    prediction: bool,
    prediction_history_load: Option<PathBuf>,
    prediction_history_save: Option<PathBuf>,
    graph_trace: bool,
    graph_trace_events: usize,
    graph_trace_detail: bool,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    verbose: bool,
    profile: bool,
    memory_report: bool,
) -> anyhow::Result<()> {
    let run_started = Instant::now();
    let model_load_started = Instant::now();
    let rss_before_load = process_rss();
    let (minor_faults_before_load, major_faults_before_load) = process_faults();

    // The explicit ForgeCore backend is the only owner of model execution in
    // this branch. RAMforge still owns device selection, prompt tokenization,
    // and sampling; ForgeCore owns native model execution. Resolve the exact
    // device before parsing/loading so an unavailable request fails directly;
    // RAMforge does not implicitly switch devices or backends.
    let registry = DeviceRegistry::discover();
    let execution =
        ExecutionConfig::new(device, 1, None, false).map_err(|error| anyhow::anyhow!(error))?;
    let model_set = discover_model_set(&model_path).map_err(|error| anyhow::anyhow!(error))?;
    let primary_path = model_set.primary_path().to_path_buf();
    let shard_paths: Vec<PathBuf> = model_set
        .shards
        .iter()
        .map(|shard| shard.path.clone())
        .collect();
    if ramforge_cli::supervisor::enforced_budget_bytes().is_some() {
        let mut evicted = 0usize;
        for shard in &model_set.shards {
            match evict_file_cache(&shard.path) {
                Ok(()) => evicted += 1,
                Err(error) => eprintln!(
                    "Warning: could not drop page cache for {}: {}",
                    shard.path.display(),
                    error
                ),
            }
        }
        eprintln!(
            "Cold-cache priming: requested cache drop for {} of {} shard(s)",
            evicted,
            model_set.shards.len()
        );
    }
    let mut model_options = ForgeCoreModelOptions {
        check_tensors: execution.check_tensors,
        use_extra_bufts: !no_repack,
        ..ForgeCoreModelOptions::default()
    };
    if no_prefetch {
        model_options = model_options.with_no_prefetch();
    }
    let native_model_load_started = Instant::now();
    let mut model = ForgeCoreModel::load_for_execution_with_options(
        &primary_path,
        execution,
        &registry,
        model_options,
    )
    .map_err(|error| anyhow::anyhow!(error))?;
    let native_model_load_ns = native_model_load_started.elapsed().as_nanos();
    let rss_after_load = process_rss();
    if release_mapping_pages {
        let before = model_mapping_residency(&primary_path);
        model
            .release_mapping_pages()
            .map_err(|error| anyhow::anyhow!(error))?;
        let after = model_mapping_residency(&primary_path);
        eprintln!(
            "Mapping page release: resident_before={} resident_after={}",
            before
                .as_ref()
                .map(|residency| format_bytes(residency.rss_bytes))
                .unwrap_or_else(|| "unavailable".to_string()),
            after
                .as_ref()
                .map(|residency| format_bytes(residency.rss_bytes))
                .unwrap_or_else(|| "unavailable".to_string())
        );
    }
    if let Some(layer) = inspect_layer.or(release_layer) {
        let mappings = model
            .tensor_mappings()
            .map_err(|error| anyhow::anyhow!(error))?;
        let selected = layer_tensor_indices(&mappings, layer);
        if selected.is_empty() {
            return Err(anyhow::anyhow!(
                "layer {layer} has no tensors named blk.{layer}.*; layer grouping uses the model's tensor naming"
            ));
        }
        let file_backed = selected
            .iter()
            .filter(|index| mappings[**index].file_backed)
            .count();
        let before = model_mapping_residency(&primary_path);
        eprintln!(
            "Layer {layer} mapping metadata: tensors={} file_backed={} anonymous={}",
            selected.len(),
            file_backed,
            selected.len() - file_backed
        );
        for index in &selected {
            let mapping = &mappings[*index];
            if mapping.file_backed {
                eprintln!(
                    "  [{index}] {}: mapping={} offset={} size={} pages=[{}, {})",
                    mapping.name,
                    mapping.mapping_index.unwrap_or(u32::MAX),
                    format_bytes(mapping.offset),
                    format_bytes(mapping.size),
                    format_bytes(mapping.page_start),
                    format_bytes(mapping.page_end)
                );
            } else {
                eprintln!(
                    "  [{index}] {}: anonymous (not file-backed; {} bytes)",
                    mapping.name,
                    format_bytes(mapping.size)
                );
            }
        }
        if release_layer.is_some() {
            let indices: Vec<i32> = selected
                .iter()
                .map(|index| {
                    i32::try_from(*index).map_err(|_| anyhow::anyhow!("tensor index exceeds i32"))
                })
                .collect::<anyhow::Result<_>>()?;
            let result = model
                .release_tensor_pages(&indices)
                .map_err(|error| anyhow::anyhow!(error))?;
            let after = model_mapping_residency(&primary_path);
            eprintln!(
                "Layer {layer} release: selected={} file_backed={} anonymous={} released={} skipped_shared={}",
                selected.len(),
                result.selected_tensors,
                result.non_file_backed_tensors,
                format_bytes(result.released_bytes),
                format_bytes(result.skipped_shared_bytes)
            );
            eprintln!(
                "Layer {layer} mapping residency: before={} after={}",
                before
                    .as_ref()
                    .map(|residency| format_bytes(residency.rss_bytes))
                    .unwrap_or_else(|| "unavailable".to_string()),
                after
                    .as_ref()
                    .map(|residency| format_bytes(residency.rss_bytes))
                    .unwrap_or_else(|| "unavailable".to_string())
            );
        }
    }
    let profile_identity = if graph_trace
        || residency_profile_load.is_some()
        || residency_profile_save.is_some()
        || prediction_history_load.is_some()
        || prediction_history_save.is_some()
    {
        let model_file_bytes = model_set.total_file_bytes();
        Some(
            ProfileIdentity::from_model_path(
                model.path(),
                model_file_bytes,
                1,
                ForgeCoreModel::native_contract_version(),
            )
            .map_err(|error| anyhow::anyhow!(error))?,
        )
    } else {
        None
    };
    let mut loaded_profile: Option<ExecutionProfile> = None;
    let mut profile_load_ns: Option<u128> = None;
    let history_load_ns: Option<u128> = None;
    if let (Some(path), Some(identity)) =
        (residency_profile_load.as_ref(), profile_identity.as_ref())
    {
        let started = Instant::now();
        let profile = load_profile(path, identity).map_err(|error| anyhow::anyhow!(error))?;
        profile_load_ns = Some(started.elapsed().as_nanos());
        eprintln!(
            "Residency profile loaded: path={} entries={} complete_observations={}",
            path.display(),
            profile.access.len(),
            profile.complete_observations
        );
        loaded_profile = Some(profile);
    }
    let mut prediction_history: Option<PredictionHistory> = None;
    if let (Some(path), Some(identity)) =
        (prediction_history_load.as_ref(), profile_identity.as_ref())
    {
        let history =
            load_prediction_history(path, identity).map_err(|error| anyhow::anyhow!(error))?;
        eprintln!(
            "Prediction history loaded: path={} records={} hits={} misses={}",
            path.display(),
            history.records.len(),
            history.hits,
            history.misses
        );
        prediction_history = Some(history);
    }
    let mut prediction_set: Option<PredictionSet> = None;
    let mut prediction_plan: Option<PredictionPlan> = None;
    let mut prediction_staging: Option<StagingExecution> = None;
    let mut prediction_metrics: Option<PredictionMetrics> = None;
    let mut page_before: Option<PageResidencySnapshot> = None;
    let mut page_after_staging: Option<PageResidencySnapshot> = None;
    let mut page_after: Option<PageResidencySnapshot> = None;
    let mut prefetch_correlation: Option<PrefetchCorrelationReport> = None;
    let mut observation_ranges: Option<Vec<PageRange>> = None;
    let mut prefetch_names: Vec<String> = Vec::new();
    let mut prediction_ns: Option<u128> = None;
    let mut planning_ns: Option<u128> = None;
    let mut staging_ns: Option<u128> = None;
    let residency_started = Instant::now();
    let mut residency_manager = if residency_policy != ResidencyPolicyArg::Off
        || residency_report
        || residency_report_json.is_some()
        || residency_profile_load.is_some()
        || residency_profile_save.is_some()
        || prediction
        || prediction_history_load.is_some()
        || prediction_history_save.is_some()
        || residency_target.is_some()
        || residency_reserve != 0
        || residency_prefetch_bytes.is_some()
    {
        let model_id = ModelId(
            std::fs::canonicalize(model.path())
                .unwrap_or_else(|_| model.path().to_path_buf())
                .display()
                .to_string(),
        );
        let mappings = model
            .tensor_mappings()
            .map_err(|error| anyhow::anyhow!(error))?;
        let inventory = ResourceInventory::from_tensor_mappings(
            model_id.clone(),
            ModelGeneration(1),
            &mappings,
        )
        .map_err(|error| anyhow::anyhow!(error))?;
        let soft_target = residency_target
            .or_else(|| (residency_policy != ResidencyPolicyArg::Off).then_some(ram_bytes));
        let policy = ResidencyPolicy {
            mode: residency_policy.mode(),
            target_bytes: soft_target,
            safety_reserve_bytes: residency_reserve,
        }
        .validate()
        .map_err(|error| anyhow::anyhow!(error))?;
        let mut manager = ResidencyManager::new(
            model_id,
            inventory,
            policy,
            ramforge_cli::supervisor::enforced_budget_bytes(),
        )
        .map_err(|error| anyhow::anyhow!(error))?;
        manager
            .bind_native_model(
                model.instance_id(),
                ForgeCoreModel::native_contract_version(),
            )
            .map_err(|error| anyhow::anyhow!(error))?;
        let lifetime = model
            .lifetime_state()
            .map_err(|error| anyhow::anyhow!(error))?;
        manager.observe_native_lifetime(
            lifetime.quiescent,
            lifetime.active_contexts,
            lifetime.inflight_contexts,
            None,
        );
        Some(manager)
    } else {
        None
    };
    let mut residency_admission: Option<AdmissionDecision> = None;
    let mut pre_operation_plan: Option<ResidencyPlan> = None;
    let mut residency_staging: Option<StagingPlan> = None;
    let mut residency_staging_result: Option<StagingResult> = None;
    if let Some(manager) = residency_manager.as_mut() {
        let observation_config = PageObservationConfig {
            max_pages: residency_observe_pages,
            max_resources: 512,
        };
        observation_ranges = if residency_observe {
            Some(manager.inventory().page_ranges(&shard_paths))
        } else {
            None
        };
        page_before = observation_ranges
            .as_ref()
            .map(|ranges| observe_page_residency(ranges, observation_config));
        let current = process_rss()
            .or_else(|| ResidencySnapshot::capture_paths(&shard_paths).mapping_rss_bytes)
            .or(Some(manager.inventory().summary.page_bytes));
        let admission = manager.admission(current);
        residency_admission = Some(admission.clone());

        if prediction {
            let prediction_started = Instant::now();
            let predictions = predict_from_profile(
                &prediction_config,
                loaded_profile.as_ref(),
                manager.inventory(),
                prediction_history.as_ref(),
            );
            if prediction_history.is_none() {
                if let Some(identity) = profile_identity.clone() {
                    prediction_history = Some(PredictionHistory::new(identity));
                }
            }
            if let Some(history) = prediction_history.as_mut() {
                history.record_predictions(&predictions);
            }
            let plan = plan_predictions(
                &prediction_config,
                &predictions,
                manager.inventory(),
                current,
                manager.hard_limit_bytes(),
                manager.policy().safety_reserve_bytes,
                admission.outcome == AdmissionOutcome::Rejected,
            );
            prediction_ns = Some(prediction_started.elapsed().as_nanos());
            let before = ResidencySnapshot::capture_paths(&shard_paths);
            let mut execution = execute_staging(&prediction_config, &plan, |indices| {
                model
                    .prefetch_tensor_pages(indices)
                    .map_err(|error| error.to_string())
            });
            let after = ResidencySnapshot::capture_paths(&shard_paths);
            execution.mapping_before_bytes = before.mapping_rss_bytes;
            execution.mapping_after_bytes = after.mapping_rss_bytes;
            eprintln!(
                "Prediction plan: predictions={} high={} medium={} low={} stage={} prefetch={} requested={} accepted={} deferred={} rejected={} history_valid={}",
                predictions.predictions.len(),
                predictions.high_confidence,
                predictions.medium_confidence,
                predictions.low_confidence,
                plan.stage_keys.len(),
                plan.prefetch_keys.len(),
                format_bytes(plan.requested_bytes()),
                format_bytes(execution.native_accepted_bytes),
                plan.deferred,
                plan.rejected,
                predictions.history_valid
            );
            for note in &plan.notes {
                eprintln!("  prediction note: {note}");
            }
            if let Some(error) = &execution.error {
                eprintln!("  prediction staging error: {error}");
            }
            prefetch_names = plan
                .decisions
                .iter()
                .filter(|decision| {
                    matches!(
                        decision.action,
                        ramforge_runtime::prediction_planner::PredictiveAction::StageNow
                            | ramforge_runtime::prediction_planner::PredictiveAction::Prefetch
                    )
                })
                .map(|decision| decision.name.clone())
                .collect();
            page_after_staging = observation_ranges
                .as_ref()
                .map(|ranges| observe_page_residency(ranges, observation_config));
            prediction_set = Some(predictions);
            prediction_plan = Some(plan);
            staging_ns = Some(execution.elapsed_ns);
            prediction_staging = Some(execution);
        } else if let Some(limit) = residency_prefetch_bytes {
            let plan = if loaded_profile.is_some() {
                manager.staging_plan_with_profile(limit, loaded_profile.as_ref())
            } else {
                manager.staging_plan(limit)
            };
            residency_staging = Some(plan.clone());
            eprintln!(
                "Residency staging plan: selected={} requested_pages={} skipped={}",
                plan.selected.len(),
                format_bytes(plan.selected_bytes),
                format_bytes(plan.skipped_bytes)
            );
            for note in &plan.notes {
                eprintln!("  staging note: {note}");
            }
            if !plan.is_empty() {
                let indices: Vec<i32> = plan.selected.iter().map(|key| key.tensor_index).collect();
                let before = ResidencySnapshot::capture_paths(&shard_paths);
                let staging_started = Instant::now();
                let requested = model
                    .prefetch_tensor_pages(&indices)
                    .map_err(|error| anyhow::anyhow!(error))?;
                staging_ns = Some(staging_started.elapsed().as_nanos());
                let after = ResidencySnapshot::capture_paths(&shard_paths);
                eprintln!(
                    "Residency staging result: native_requested={} mapping_before={} mapping_after={} delta={}",
                    format_bytes(requested),
                    optional_bytes(before.mapping_rss_bytes),
                    optional_bytes(after.mapping_rss_bytes),
                    before
                        .mapping_delta(&after)
                        .map(|delta| format!("{delta:+} bytes"))
                        .unwrap_or_else(|| "unavailable".to_string())
                );
                prefetch_names = plan
                    .selected
                    .iter()
                    .filter_map(|key| {
                        manager
                            .inventory()
                            .get(*key)
                            .map(|resource| resource.name.clone())
                    })
                    .collect();
                page_after_staging = observation_ranges
                    .as_ref()
                    .map(|ranges| observe_page_residency(ranges, observation_config));
                residency_staging_result = Some(StagingResult {
                    requested_keys: plan.selected.clone(),
                    native_requested_bytes: requested,
                    mapping_before_bytes: before.mapping_rss_bytes,
                    mapping_after_bytes: after.mapping_rss_bytes,
                    error: None,
                });
            }
        }

        if page_after_staging.is_none() {
            page_after_staging = observation_ranges
                .as_ref()
                .map(|ranges| observe_page_residency(ranges, observation_config));
        }
        if !residency_report && residency_policy != ResidencyPolicyArg::Off {
            eprintln!(
                "Residency admission: {:?}; current={} soft_target={} hard_limit={} reserve={}",
                admission.outcome,
                optional_bytes(admission.current_bytes),
                optional_bytes(admission.soft_target_bytes),
                optional_bytes(admission.hard_limit_bytes),
                format_bytes(admission.safety_reserve_bytes)
            );
            for reason in &admission.reasons {
                eprintln!("  admission: {reason}");
            }
        }
        let context = OperationContext {
            required: manager.inventory().file_backed_keys(),
            pinned: Default::default(),
        };
        let planning_started = Instant::now();
        let plan = manager
            .plan_with_profile(current, &context, loaded_profile.as_ref())
            .map_err(|error| anyhow::anyhow!(error))?;
        planning_ns = Some(planning_started.elapsed().as_nanos());
        if !residency_report && residency_policy != ResidencyPolicyArg::Off {
            print_residency_plan("pre-operation", &plan);
        }
        pre_operation_plan = Some(plan);
        manager
            .begin_operation()
            .map_err(|error| anyhow::anyhow!(error))?;
    }

    let metadata = model.metadata().clone();

    let gguf = model_set.merged_model();
    let architecture = gguf
        .info()
        .architecture
        .unwrap_or_else(|| "unknown".to_string());
    let architecture_support = ramforge_runtime::support::model_execution_support(&gguf);
    let raw_prompt = prompt;
    let prompt = prepare_prompt(&gguf, raw_prompt.clone(), chat, system)?;
    let tokenizer = Tokenizer::from_gguf(&gguf).map_err(|error| anyhow::anyhow!(error))?;
    let mut prompt_tokens = tokenizer.encode(&prompt, true);
    if prompt_tokens.is_empty() {
        if let Some(bos) = tokenizer.bos_id {
            prompt_tokens.push(bos);
        } else {
            return Err(anyhow::anyhow!(
                "prompt produced no tokens and the GGUF has no BOS token"
            ));
        }
    }
    if tokenizer.vocab_size() != metadata.vocab_size as usize {
        return Err(anyhow::anyhow!(
            "RAMforge tokenizer vocabulary {} does not match ForgeCore vocabulary {}",
            tokenizer.vocab_size(),
            metadata.vocab_size
        ));
    }

    let mut session = model
        .open_context(ForgeCoreContextOptions {
            n_ctx: ctx,
            n_threads: 1,
            trace_max_events: graph_trace.then_some(graph_trace_events),
        })
        .map_err(|error| anyhow::anyhow!(error))?;
    let context_length = session.context_length() as usize;
    if prompt_tokens.len() > context_length {
        return Err(anyhow::anyhow!(
            "prompt has {} tokens but ForgeCore context length is {}",
            prompt_tokens.len(),
            context_length
        ));
    }

    let load_elapsed = model_load_started.elapsed();
    let prompt_started = Instant::now();
    let mut logits =
        decode_with_residency(&mut session, &prompt_tokens, 0, residency_manager.as_mut())?;
    let prompt_elapsed = prompt_started.elapsed();
    let sampler = ramforge_runtime::sampling::Sampler::new(temperature, top_k, top_p);
    let eos_id = tokenizer.eos_id;
    let mut decoder = tokenizer.decoder();
    let mut generated_token_ids = Vec::new();
    let mut first_token_elapsed = None;
    let mut decode_started = None;
    let mut stop_reason = if max_tokens == 0 {
        "max_tokens"
    } else {
        "context_length"
    };

    eprintln!("RAMforge – ForgeCore static device inference");
    eprintln!("========================================================");
    eprintln!("Model: {}", model_path.display());
    if model_set.is_sharded() {
        eprintln!(
            "Model set: {} shards, primary={}, total_file_bytes={}",
            model_set.shards.len(),
            primary_path.display(),
            model_set.total_file_bytes()
        );
    }
    if let Some(limit) = ramforge_cli::supervisor::enforced_budget_bytes() {
        eprintln!(
            "RAM budget: {} ({} bytes; ENFORCED cgroup v2 memory.max={} bytes)",
            ram_str, ram_bytes, limit
        );
    } else {
        eprintln!(
            "RAM budget: {} ({} bytes; planning/diagnostics only for ForgeCore; pass --enforce-ram for a hard OS limit)",
            ram_str, ram_bytes
        );
    }
    eprintln!("Prompt: {:?}", raw_prompt);
    eprintln!(
        "Architecture: {} (legacy-run={}, forgecore-run={})",
        architecture,
        architecture_support.legacy_run.label(),
        architecture_support.forgecore_run.label()
    );
    if !architecture_support.is_forgecore_supported() {
        eprintln!(
            "Warning: architecture '{}' has no RAMforge-validated forgeCORE execution evidence; the request is delegated to the pinned llama.cpp runtime and may fail",
            architecture
        );
    }
    if chat {
        eprintln!("Chat template: applied (rendered {} bytes)", prompt.len());
        eprintln!("Chat prompt: {:?}", prompt);
    }
    eprintln!("Prompt tokens: {}", prompt_tokens.len());
    eprintln!("Max tokens: {}", max_tokens);
    eprintln!("Temperature: {}", temperature);
    if let Some(k) = top_k {
        eprintln!("Top-k: {}", k);
    }
    if let Some(p) = top_p {
        eprintln!("Top-p: {}", p);
    }
    eprintln!(
        "Execution backend: ForgeCore/libllama ({:?})",
        model.device().kind
    );
    eprintln!(
        "Model metadata: params={}, vocab={}, context={}, embedding={}, layers={}, heads={}, kv_heads={}",
        metadata.n_params,
        metadata.vocab_size,
        metadata.context_length,
        metadata.embedding_length,
        metadata.layer_count,
        metadata.head_count,
        metadata.kv_head_count
    );
    eprintln!("Model description: {}", metadata.description);
    eprintln!(
        "Model load mode: {} (weights file-backed when the platform supports mmap)",
        metadata.load_mode_label()
    );
    eprintln!(
        "CPU weight repacking: {}",
        metadata.weight_repacking_label()
    );
    eprintln!(
        "Context length: {} (requested {})",
        context_length,
        ctx.map(|value| value.to_string())
            .unwrap_or_else(|| "model default".to_string())
    );
    eprintln!("Selection request: {}", model.execution_config().device);
    eprintln!(
        "Selected backend/device: {} / {} ({:?}, registry_index={:?})",
        model.device().backend_id,
        model.device().id,
        model.device().kind,
        model.device().registry_index
    );
    eprintln!("Device description: {}", model.device().description);
    eprintln!(
        "Device accounting domain: {} (selection is not proof of residency)",
        model.device().accounting_domain()
    );
    eprintln!(
        "Device capabilities: model_execution={}, static_offload={}, memory_reporting={}",
        model.device().capabilities.model_execution,
        model.device().capabilities.static_model_offload,
        model.device().capabilities.memory_reporting
    );
    eprintln!(
        "Device memory: total={}, free={}",
        format_measurement(&model.device().memory_total),
        format_measurement(&model.device().memory_free)
    );
    eprintln!("ForgeCore model load: {}", format_duration(load_elapsed));
    eprintln!(
        "Actual initialized device: {} ({})",
        model.device().id,
        model.device().backend_id
    );
    eprintln!(
        "Accounting note: ForgeCore native allocations, KV state, and backend memory are not measured by RAMforge's Phase-3 resource accounting unless a safe measurement is provided"
    );
    eprintln!();

    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    write!(stdout, "Assistant: ")?;
    stdout.flush()?;

    for _generation_index in 0..max_tokens {
        let current_length = prompt_tokens
            .len()
            .checked_add(generated_token_ids.len())
            .ok_or_else(|| anyhow::anyhow!("generation token count overflow"))?;
        if current_length >= context_length {
            stop_reason = "context_length";
            break;
        }

        let next_token = sampler.sample(logits.values());
        if Some(next_token) == eos_id {
            stop_reason = "eos";
            break;
        }
        generated_token_ids.push(next_token);
        if first_token_elapsed.is_none() {
            first_token_elapsed = Some(prompt_started.elapsed());
            decode_started = Some(Instant::now());
        }
        let piece = decoder.push(next_token);
        if !piece.is_empty() {
            write!(stdout, "{piece}")?;
            stdout.flush()?;
        }

        if generated_token_ids.len() >= max_tokens {
            stop_reason = "max_tokens";
            break;
        }
        let next_position = prompt_tokens
            .len()
            .checked_add(generated_token_ids.len())
            .and_then(|length| length.checked_sub(1))
            .ok_or_else(|| anyhow::anyhow!("next token position overflow"))?;
        let next_position = u32::try_from(next_position)
            .map_err(|_| anyhow::anyhow!("next token position exceeds ForgeCore's u32 limit"))?;
        logits = decode_with_residency(
            &mut session,
            &[next_token],
            next_position,
            residency_manager.as_mut(),
        )?;
    }

    let rss_after_decode = process_rss();
    let trailing = decoder.finish();
    if !trailing.is_empty() {
        write!(stdout, "{trailing}")?;
    }
    writeln!(stdout)?;
    stdout.flush()?;

    eprintln!("Generated {} tokens", generated_token_ids.len());
    eprintln!(
        "Stop reason: {} (eos_id={}, context_length={})",
        stop_reason,
        eos_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "<unavailable>".to_string()),
        context_length
    );
    if verbose {
        eprintln!(
            "ForgeCore session: context_length={} vocab={} CPU threads={}",
            session.context_length(),
            session.vocab_size(),
            session.context_options().n_threads
        );
        eprintln!(
            "RAMforge retained: GGUF inspection, tokenizer, sampler, diagnostics, and planning/accounting infrastructure"
        );
    }
    let decode_elapsed = decode_started.map(|started| started.elapsed());
    let decode_tokens = generated_token_ids.len().saturating_sub(1);
    let total_generation_elapsed = prompt_started.elapsed();
    if profile {
        eprintln!(
            "Profile (RAMforge-observed wall clock; native kernel/I/O counters unavailable):"
        );
        eprintln!(
            "  Prompt processing: {} tokens in {} ({})",
            prompt_tokens.len(),
            format_duration(prompt_elapsed),
            format_rate(prompt_tokens.len(), prompt_elapsed)
        );
        eprintln!(
            "  Time to first token: {}",
            first_token_elapsed
                .map(format_duration)
                .unwrap_or_else(|| "unavailable".to_string())
        );
        match (decode_elapsed, decode_tokens) {
            (Some(elapsed), tokens) => eprintln!(
                "  Decode: {} tokens in {} ({})",
                tokens,
                format_duration(elapsed),
                format_rate(tokens, elapsed)
            ),
            (None, _) => eprintln!("  Decode: unavailable"),
        }
        eprintln!(
            "  Total after context open: {}",
            format_duration(total_generation_elapsed)
        );
        eprintln!(
            "  Model load plus tokenization: {}",
            format_duration(load_elapsed)
        );
    }
    let mut execution_profile = None;
    let mut native_context_id: Option<u64> = None;
    let mut native_execution_generation: Option<u64> = None;
    if graph_trace {
        let summary = session.trace_summary();
        let events = session.trace_events();
        let manifest = GraphManifest::from_trace(&events);
        eprintln!("Graph trace summary: {:?}", summary);
        eprintln!("Graph manifest: {:?}", manifest.summary());
        if summary
            .as_ref()
            .is_some_and(|summary| summary.executions > 1)
        {
            eprintln!(
                "Graph manifest note: capture spans multiple graph executions; merged edges are intentionally conservative and the profile analyzes each execution separately"
            );
        }
        if let Some(manager) = residency_manager.as_ref() {
            let mut profile =
                ExecutionProfile::new(manager.model_id().clone(), manager.inventory().generation);
            let trace_complete = summary
                .as_ref()
                .is_some_and(|summary| summary.dropped_events == 0);
            profile.update_from_trace(&events, trace_complete);
            eprintln!(
                "Execution profile: version={} complete={} observations={} incomplete={} tensors={} valid={}",
                profile.version,
                profile.is_complete(),
                profile.complete_observations,
                profile.incomplete_observations,
                profile.access.len(),
                profile.is_valid_for(&manager.model_id().clone(), manager.inventory().generation)
            );
            if let Some(path) = residency_profile_save.as_ref() {
                let identity = profile_identity
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("profile save requires a model identity"))?;
                profile
                    .save_json_file(identity, path)
                    .map_err(|error| anyhow::anyhow!(error))?;
                eprintln!(
                    "Residency profile saved: path={} entries={}",
                    path.display(),
                    profile.access.len()
                );
            }
            if prediction {
                if profile.is_valid_for(&manager.model_id().clone(), manager.inventory().generation)
                {
                    if let (Some(set), Some(plan)) =
                        (prediction_set.as_ref(), prediction_plan.as_ref())
                    {
                        let metrics = evaluate_predictions(
                            set,
                            &profile,
                            manager.inventory(),
                            plan,
                            prediction_staging.as_ref(),
                            prediction_config.lookahead,
                        );
                        if let Some(history) = prediction_history.as_mut() {
                            history.record_feedback(set, &profile, &plan.staged_bytes_by_name);
                            history.record_evaluation(&metrics);
                            let summary = history.summary();
                            eprintln!(
                                "Prediction evaluation: matched={} within_horizon={} precision_bps={:?} recall_bps={:?} fpr_bps={:?} accepted_resources={:?} followed_by_access={:?} not_followed={:?} observed_residency={:?}",
                                metrics.matched_observed,
                                metrics.matched_within_horizon,
                                metrics.precision_bps,
                                metrics.recall_bps,
                                metrics.false_positive_rate_bps,
                                metrics.prefetch_accepted_resources,
                                metrics.prefetch_followed_by_observed_access,
                                metrics.prefetch_not_followed_by_observed_access,
                                metrics.prefetch_observed_residency_resources
                            );
                            eprintln!(
                                "Prediction history: hits={} misses={} within_horizon={} staged_requests={} accepted_bytes={} followed_by_access={} wasted_bytes={} accuracy_bps={:?}",
                                summary.hits,
                                summary.misses,
                                summary.matched_within_horizon,
                                summary.staged_requests,
                                summary.prefetch_accepted_bytes,
                                summary.prefetch_followed_by_access,
                                summary.wasted_bytes,
                                summary.accuracy_bps
                            );
                        }
                        prediction_metrics = Some(metrics);
                    }
                }
                if let Some(path) = prediction_history_save.as_ref() {
                    let history = prediction_history.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("prediction history save requested without a history")
                    })?;
                    history.save(path).map_err(|error| anyhow::anyhow!(error))?;
                    eprintln!(
                        "Prediction history saved: path={} records={}",
                        path.display(),
                        history.records.len()
                    );
                }
            }
            execution_profile = Some(profile);
        }
        let unknown = manifest
            .last_use
            .values()
            .filter(|last| last.status == LastUseStatus::Unknown)
            .count();
        eprintln!(
            "Graph last-use: graph_order={} unknown={} intra_decode_safe={}",
            manifest.last_use.len().saturating_sub(unknown),
            unknown,
            manifest.intra_decode_safe
        );
        if graph_trace_detail {
            for event in &events {
                eprintln!(
                    "  trace exec={} seq={} phase={:?} completion={:?} name={:?} op={:?} bytes={} deps={:?}",
                    event.execution_id,
                    event.sequence,
                    event.phase,
                    event.completion,
                    event.node_name,
                    event.op_name,
                    event.bytes,
                    event.dependencies
                );
            }
        }
    }
    if let Ok(lifetime) = session.lifetime_state() {
        native_context_id = Some(lifetime.context_id);
        native_execution_generation = Some(lifetime.execution_generation);
    }
    let synchronization_started = Instant::now();
    session.synchronize();
    let synchronization_ns = Some(synchronization_started.elapsed().as_nanos());
    let end_to_end_ns = run_started.elapsed().as_nanos();
    drop(session);

    let mut post_operation_plan = None;
    let mut residency_application = None;
    let mut residency_before = None;
    let mut residency_after = None;
    if let Some(manager) = residency_manager.as_mut() {
        let native_lifetime = model
            .lifetime_state()
            .map_err(|error| anyhow::anyhow!(error))?;
        manager.observe_native_lifetime(
            native_lifetime.quiescent,
            native_lifetime.active_contexts,
            native_lifetime.inflight_contexts,
            native_execution_generation,
        );
        if !native_lifetime.quiescent || native_lifetime.active_contexts != 0 {
            eprintln!(
                "Residency native guard: quiescent={} active_contexts={} inflight_contexts={}",
                native_lifetime.quiescent,
                native_lifetime.active_contexts,
                native_lifetime.inflight_contexts
            );
        }
        let before = ResidencySnapshot::capture_paths(&shard_paths);
        let current = before
            .process_rss_bytes
            .or(before.mapping_rss_bytes)
            .or(Some(manager.inventory().summary.page_bytes));
        let plan = manager
            .plan_with_profile(
                current,
                &OperationContext::default(),
                execution_profile
                    .as_ref()
                    .filter(|profile| {
                        profile.is_valid_for(
                            &manager.model_id().clone(),
                            manager.inventory().generation,
                        )
                    })
                    .or(loaded_profile.as_ref()),
            )
            .map_err(|error| anyhow::anyhow!(error))?;
        if !residency_report && residency_policy != ResidencyPolicyArg::Off {
            print_residency_plan("post-operation", &plan);
        }
        post_operation_plan = Some(plan);
        residency_before = Some(before);
        if residency_policy.applies() && manager.last_plan().is_some_and(|plan| !plan.is_noop()) {
            match manager.apply_last_plan(|indices| {
                model
                    .release_tensor_pages(indices)
                    .map_err(|error| error.to_string())
            }) {
                Ok(application) => residency_application = Some(application),
                Err(ManagerError::Release(message)) => {
                    residency_application = manager.application().cloned();
                    eprintln!("Residency application incomplete: {message}");
                }
                Err(error) => return Err(anyhow::anyhow!(error)),
            }
        }
        residency_after = Some(ResidencySnapshot::capture_paths(&shard_paths));
    }

    if let Some(ranges) = observation_ranges.as_ref() {
        page_after = Some(observe_page_residency(
            ranges,
            PageObservationConfig {
                max_pages: residency_observe_pages,
                max_resources: 512,
            },
        ));
        if let (Some(before), Some(after_staging)) =
            (page_before.as_ref(), page_after_staging.as_ref())
        {
            let observed_names = execution_profile.as_ref().map(|profile| {
                profile
                    .access
                    .keys()
                    .cloned()
                    .collect::<std::collections::BTreeSet<_>>()
            });
            prefetch_correlation = Some(correlate_prefetch_residency(
                &prefetch_names,
                before,
                after_staging,
                observed_names.as_ref(),
            ));
        }
    }

    if residency_report || residency_report_json.is_some() {
        if let (Some(manager), Some(admission), Some(before), Some(after)) = (
            residency_manager.as_ref(),
            residency_admission.clone(),
            residency_before.clone(),
            residency_after.clone(),
        ) {
            let report = ResidencyReport {
                model_id: manager.model_id().clone(),
                generation: manager.inventory().generation,
                inventory: manager.inventory().summary,
                admission,
                pre_plan: pre_operation_plan.clone(),
                post_plan: post_operation_plan.clone(),
                application: residency_application.clone(),
                staging: residency_staging.clone(),
                staging_result: residency_staging_result.clone(),
                profile: execution_profile
                    .clone()
                    .filter(|profile| {
                        profile.is_valid_for(
                            &manager.model_id().clone(),
                            manager.inventory().generation,
                        )
                    })
                    .or_else(|| loaded_profile.clone()),
                profile_source: if execution_profile.as_ref().is_some_and(|profile| {
                    profile
                        .is_valid_for(&manager.model_id().clone(), manager.inventory().generation)
                }) {
                    "current_trace".to_string()
                } else if loaded_profile.is_some() {
                    "loaded_file".to_string()
                } else if execution_profile.is_some() {
                    "invalid_trace".to_string()
                } else {
                    "none".to_string()
                },
                profile_identity: profile_identity.clone(),
                native_model_instance: manager.native_model_instance(),
                native_contract_version: manager.native_contract_version(),
                native_quiescent: manager.native_quiescent(),
                native_context_id,
                native_execution_generation,
                minor_faults_before_load,
                major_faults_before_load,
                lifetime_counts: manager.lifetime_counts(),
                prediction_set: prediction_set.clone(),
                prediction_plan: prediction_plan.clone(),
                prediction_staging: prediction_staging.clone(),
                prediction_metrics: prediction_metrics.clone(),
                prediction_history: prediction_history.as_ref().map(|history| history.summary()),
                page_before: page_before.clone(),
                page_after_staging: page_after_staging.clone(),
                page_after: page_after.clone(),
                prefetch_correlation: prefetch_correlation.clone(),
                execution_timing: ExecutionTiming {
                    model_load_ns: Some(native_model_load_ns),
                    profile_load_ns,
                    history_load_ns,
                    prediction_ns,
                    planning_ns,
                    staging_ns,
                    observation_ns: [
                        page_before.as_ref().map(|value| value.elapsed_ns),
                        page_after_staging.as_ref().map(|value| value.elapsed_ns),
                        page_after.as_ref().map(|value| value.elapsed_ns),
                    ]
                    .into_iter()
                    .flatten()
                    .reduce(|left, right| left.saturating_add(right)),
                    inference_ns: Some(total_generation_elapsed.as_nanos()),
                    synchronization_ns,
                    end_to_end_ns,
                },
                ledger: MemoryLedger::from_inventory(
                    manager.model_id().clone(),
                    manager.inventory().generation,
                    &manager.inventory().summary,
                    &before,
                    residency_staging.as_ref(),
                    residency_staging_result.as_ref(),
                ),
                before,
                after,
                timing: OperationTiming {
                    prompt_tokens: prompt_tokens.len(),
                    generated_tokens: generated_token_ids.len(),
                    first_token_elapsed,
                    prompt_elapsed: Some(prompt_elapsed),
                    decode_elapsed,
                    total_elapsed: Some(total_generation_elapsed),
                    stop_reason: stop_reason.to_string(),
                    output_fingerprint: token_fingerprint(&generated_token_ids),
                },
                elapsed: residency_started.elapsed(),
            };
            if residency_report {
                print_residency_report(&report);
            }
            if let Some(path) = residency_report_json.as_ref() {
                let value = residency_report_json_value(&report);
                let encoded = serde_json::to_vec_pretty(&value)
                    .map_err(|error| anyhow::anyhow!("serialize residency report: {error}"))?;
                std::fs::write(path, encoded).map_err(|error| {
                    anyhow::anyhow!("write residency JSON report '{}': {error}", path.display())
                })?;
                eprintln!("Residency JSON report: {}", path.display());
            }
        }
    } else if residency_policy != ResidencyPolicyArg::Off {
        let pre_selected = pre_operation_plan
            .as_ref()
            .map(|plan| plan.selected.len())
            .unwrap_or(0);
        let post_selected = post_operation_plan
            .as_ref()
            .map(|plan| plan.selected.len())
            .unwrap_or(0);
        let released = residency_application
            .as_ref()
            .and_then(|application| application.native_result.as_ref())
            .map(|result| result.released_bytes)
            .unwrap_or(0);
        eprintln!(
            "Residency: policy={:?} pre_selected={} post_selected={} applied={} released={}",
            residency_policy,
            pre_selected,
            post_selected,
            residency_application.is_some(),
            format_bytes(released)
        );
    }
    if memory_report {
        let mut report = ForgeCoreMemoryReport::new(
            ram_bytes,
            gguf.total_tensor_bytes(),
            metadata.size_bytes,
            rss_before_load,
            rss_after_load,
            rss_after_decode,
            model.device(),
        );
        if let Some(stats) = ramforge_cli::supervisor::child_report_stats() {
            report = report.with_cgroup(stats);
        }
        report = report.with_load_mode(metadata.load_mode_label());
        report = report.with_weight_repacking(metadata.weight_repacking_label());
        if let Some(residency) = model_mapping_residency(&primary_path) {
            report = report.with_model_residency(residency);
        }
        output_forgecore_memory_report(&report);
    }

    Ok(())
}

fn run_profile_inspect(
    path: &std::path::Path,
    model: Option<&std::path::Path>,
    inventory_generation: u64,
    json: bool,
) -> anyhow::Result<()> {
    let inspection = inspect_profile(path).map_err(|error| anyhow::anyhow!(error))?;
    let valid = true;
    if let Some(model_path) = model {
        let model_set = discover_model_set(model_path)
            .map_err(|error| anyhow::anyhow!("discover model set: {error}"))?;
        let model_bytes = model_set.total_file_bytes();
        let identity = ProfileIdentity::from_model_path(
            model_path,
            model_bytes,
            inventory_generation,
            ForgeCoreModel::native_contract_version(),
        )
        .map_err(|error| anyhow::anyhow!(error))?;
        if let Err(error) = load_profile(path, &identity) {
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "valid": false,
                        "error": error.to_string(),
                        "inspection": format!("{inspection:?}"),
                    })
                );
                return Ok(());
            }
            return Err(anyhow::anyhow!(error));
        }
    }
    if json {
        println!(
            "{}",
            serde_json::json!({
                "valid": valid,
                "schema_version": inspection.schema_version,
                "kind": inspection.kind,
                "profile_version": inspection.profile_version,
                "entries": inspection.entries,
                "complete_observations": inspection.complete_observations,
                "incomplete_observations": inspection.incomplete_observations,
                "invalidated": inspection.invalidated,
                "identity": inspection.identity,
                "model_identity_validated": model.is_some(),
            })
        );
    } else {
        println!("Profile: {}", path.display());
        println!(
            "  schema={} kind={} profile_version={} valid={}",
            inspection.schema_version, inspection.kind, inspection.profile_version, valid
        );
        println!(
            "  entries={} complete_observations={} incomplete_observations={} invalidated={}",
            inspection.entries,
            inspection.complete_observations,
            inspection.incomplete_observations,
            inspection.invalidated
        );
        println!("  model={}", inspection.identity.model_path);
        println!(
            "  model_size_bytes={} model_modified_unix_ns={:?}",
            inspection.identity.model_size_bytes, inspection.identity.model_modified_unix_ns
        );
        println!(
            "  inventory_generation={} native_contract={}",
            inspection.identity.inventory_generation, inspection.identity.native_contract_version
        );
        println!("  model_identity_validated={}", model.is_some());
    }
    Ok(())
}

fn run_simulation(
    budget: &str,
    window: &str,
    wrong_history: bool,
    json: bool,
) -> anyhow::Result<()> {
    use ramforge_runtime::prediction_sim::{simulate, SimResource, SimScenario};
    let budget =
        parse_memory_size(budget).map_err(|error| anyhow::anyhow!("invalid --budget: {error}"))?;
    let window =
        parse_memory_size(window).map_err(|error| anyhow::anyhow!("invalid --window: {error}"))?;
    let names = ["a", "b", "c", "d", "e"];
    let scenario = SimScenario {
        name: "deterministic-five-resource".to_string(),
        budget_bytes: budget,
        staging_window_bytes: window.min(budget),
        speculative_budget_bytes: window.saturating_div(2).min(budget),
        min_confidence_bps: 5_000,
        prediction_horizon: 4,
        resources: names
            .iter()
            .map(|name| SimResource {
                name: (*name).to_string(),
                bytes: 4096,
                file_backed: true,
                known_size: true,
            })
            .collect(),
        training_sequence: names.iter().map(|name| (*name).to_string()).collect(),
        test_sequence: if wrong_history {
            vec!["e".to_string()]
        } else {
            names.iter().map(|name| (*name).to_string()).collect()
        },
    };
    let result = simulate(&scenario).map_err(|error| anyhow::anyhow!(error))?;
    if json {
        println!(
            "{}",
            serde_json::json!({
                "baseline": {
                    "strategy": result.baseline.strategy,
                    "hits": result.baseline.hits,
                    "misses": result.baseline.misses,
                    "reread_bytes": result.baseline.reread_bytes,
                    "peak_resident_bytes": result.baseline.peak_resident_bytes,
                    "budget_violations": result.baseline.budget_violations,
                },
                "predictive": {
                    "strategy": result.predictive.strategy,
                    "hits": result.predictive.hits,
                    "misses": result.predictive.misses,
                    "reread_bytes": result.predictive.reread_bytes,
                    "staged_bytes": result.predictive.staged_bytes,
                    "wasted_staged_bytes": result.predictive.wasted_staged_bytes,
                    "predicted_resources": result.predictive.predicted_resources,
                    "matched_resources": result.predictive.matched_resources,
                    "within_horizon_matches": result.predictive.within_horizon_matches,
                    "prediction_precision_bps": result.predictive.prediction_precision_bps,
                    "prediction_recall_bps": result.predictive.prediction_recall_bps,
                    "peak_resident_bytes": result.predictive.peak_resident_bytes,
                    "budget_violations": result.predictive.budget_violations,
                    "release_authorized": result.predictive.release_authorized,
                },
                "miss_reduction": result.miss_reduction(),
                "reread_byte_reduction": result.reread_byte_reduction(),
                "simulation_only": true,
            })
        );
    } else {
        println!("Simulation: {}", scenario.name);
        println!(
            "  baseline: hits={} misses={} reread_bytes={} peak={} violations={}",
            result.baseline.hits,
            result.baseline.misses,
            result.baseline.reread_bytes,
            result.baseline.peak_resident_bytes,
            result.baseline.budget_violations
        );
        println!(
            "  predictive: hits={} misses={} reread_bytes={} staged_bytes={} wasted={} peak={} violations={} release_authorized={}",
            result.predictive.hits,
            result.predictive.misses,
            result.predictive.reread_bytes,
            result.predictive.staged_bytes,
            result.predictive.wasted_staged_bytes,
            result.predictive.peak_resident_bytes,
            result.predictive.budget_violations,
            result.predictive.release_authorized
        );
        println!(
            "  miss_reduction={} reread_byte_reduction={} (simulation only, not hardware evidence)",
            result.miss_reduction(),
            result.reread_byte_reduction()
        );
    }
    Ok(())
}

fn output_support() {
    use ramforge_runtime::support::{
        execution_architectures, forgecore_execution_architectures, ARCHITECTURES, QUANTIZATIONS,
    };

    println!("RAMforge capability registry");
    println!("Architecture support:");
    println!(
        "  {:<10} {:<9} {:<9} {:<12} {:<24} Tokenizer",
        "Architecture", "Inspect", "Plan", "Legacy run", "ForgeCore/libllama run"
    );
    for capability in ARCHITECTURES {
        println!(
            "  {:<10} {:<9} {:<9} {:<12} {:<24} {}",
            capability.name,
            yes_no(capability.inspect),
            yes_no(capability.plan),
            capability.legacy_run.label(),
            capability.forgecore_run.label(),
            capability.tokenizer
        );
        println!("    {}", capability.notes);
    }
    println!();
    println!(
        "  Legacy execution architectures: [{}]",
        execution_architectures().join(", ")
    );
    println!(
        "  Validated forgeCORE/libllama architectures: [{}]",
        forgecore_execution_architectures().join(", ")
    );
    println!();
    println!("Quantization support (RAMforge legacy executor):");
    for capability in QUANTIZATIONS {
        println!(
            "  {:<18} inference={}",
            capability.name,
            yes_no(capability.inference)
        );
    }
    println!();
    println!("Inspection/planning are generic GGUF operations and do not imply execution support.");
    println!("Legacy run=yes is architecture-level; tensor-format support is model-specific. Use `inspect` or `plan` for the final legacy-execution verdict.");
    println!("ForgeCore/libllama run=yes means real-model execution has been validated for at least one fixture of that architecture. The backend delegates tensor-type support to the pinned native runtime; it does not use RAMforge's legacy quantization list.");
    println!();
    let capabilities = ResidencyCapabilities::current();
    println!("Residency capabilities (CPU baseline):");
    println!(
        "  tensor metadata: {:?}",
        capabilities.tensor_mapping_metadata
    );
    println!(
        "  file-backed page release: {:?}",
        capabilities.file_backed_release
    );
    println!("  page prefetch: {:?}", capabilities.page_prefetch);
    println!("  policy profiles: off, observe, conservative, memory-bound, reuse-aware, active (compatibility)");
    println!(
        "  memory ledger/reconciliation: supported; KV/scratch/runtime remain explicitly unknown"
    );
    println!("  operation reports: human-readable and versioned JSON schema 1");
    println!(
        "  execution profiles: bounded/versioned; optional validated persistence via --residency-profile-load/save"
    );
    println!("  graph trace: {:?}", capabilities.graph_trace);
    println!(
        "  graph dependency analysis: {:?}",
        capabilities.graph_dependency_analysis
    );
    println!(
        "  proven last-use completion: {:?}",
        capabilities.proven_last_use_completion
    );
    println!(
        "  native lifetime contract: {:?}",
        capabilities.native_lifetime_contract
    );
    println!(
        "  predictive staging: {:?}",
        capabilities.predictive_staging
    );
    println!(
        "  prediction history: {:?}",
        capabilities.prediction_history
    );
    println!(
        "  context-owned resource release: {:?}",
        capabilities.context_owned_release
    );
    println!(
        "  between-operation residency: {:?}",
        capabilities.between_operation_residency
    );
    println!(
        "  intra-decode residency: {:?}",
        capabilities.intra_decode_residency
    );
    println!(
        "  hard cgroup enforcement: {:?}",
        capabilities.hard_cgroup_enforcement
    );
    println!("  CPU execution: {:?}", capabilities.cpu_execution);
    println!("  GPU execution: {:?}", capabilities.gpu_execution);
    println!("  Intra-decode residency is unsupported by the pinned native callback contract; between-operation streaming remains the safe mode.");
}

fn output_generation_profile(
    profile: &ramforge_runtime::inference::GenerationProfile,
    model_load_elapsed: Duration,
) {
    let runtime = &profile.runtime;
    eprintln!();
    eprintln!("Generation profile:");
    eprintln!("  Tokens: {}", runtime.tokens);
    eprintln!("  Model startup: {}", format_duration(model_load_elapsed));
    eprintln!("  Generation total: {}", format_duration(runtime.total));
    if let Some(average) = profile.average_token_latency() {
        eprintln!(
            "  Average per generated token: {}",
            format_duration(average)
        );
    }
    eprintln!(
        "  Maximum token loop: {}",
        format_duration(runtime.max_token_latency)
    );
    eprintln!(
        "  Logical tensor reads: {}",
        profile.io.logical_tensor_reads
    );
    eprintln!(
        "  Logical tensor bytes: {}",
        format_bytes(profile.io.logical_tensor_bytes)
    );
    eprintln!("  Physical GGUF reads: {}", profile.io.read_operations);
    eprintln!(
        "  Physical GGUF bytes: {}",
        format_bytes(profile.io.bytes_read)
    );
    eprintln!("  Physical seeks: {}", profile.io.seek_operations);
    eprintln!("  Seeks safely avoided: {}", profile.io.seeks_avoided);
    eprintln!("  GGUF read failures: {}", profile.io.read_failures);
    eprintln!(
        "  Coalesced physical ranges: {}",
        profile.io.coalesced_ranges
    );
    eprintln!(
        "  Coalesced gap overhead: {}",
        format_bytes(profile.io.coalesced_gap_bytes)
    );
    eprintln!("  Read-buffer reuses: {}", profile.io.read_buffer_reuses);
    eprintln!("  Read-buffer growths: {}", profile.io.read_buffer_growths);
    eprintln!(
        "  Grouped quantized copies: {}",
        runtime.grouped_quantized_copy_count
    );
    eprintln!(
        "  Grouped quantized copy bytes: {}",
        format_bytes(runtime.grouped_quantized_copy_bytes)
    );
    eprintln!("  Prompt token forwards: {}", runtime.prompt_forwards);
    eprintln!("  Decode token forwards: {}", runtime.decode_forwards);
    eprintln!(
        "  Terminal forwards skipped: {}",
        runtime.terminal_forwards_skipped
    );
    eprintln!("  Layer loads: {}", runtime.layer_loads);
    eprintln!("  Layer releases: {}", runtime.layer_releases);
    eprintln!("  Layer-cache hits: {}", runtime.cache_hits);
    eprintln!(
        "  Layer-cache misses (disk loads): {}",
        runtime.cache_misses
    );
    eprintln!("  Layer-cache evictions: {}", runtime.cache_evictions);
    eprintln!(
        "  Cached layers: current={} peak={}",
        runtime.cached_layer_count, runtime.peak_cached_layer_count
    );
    eprintln!(
        "  Layer-cache bytes: current={} peak={} capacity={}",
        format_bytes(runtime.cache_bytes),
        format_bytes(runtime.peak_cache_bytes),
        format_bytes(profile.layer_cache_capacity_bytes)
    );
    eprintln!(
        "  Peak RAMforge-managed memory: {}",
        format_bytes(profile.ramforge_peak_bytes)
    );
    if !profile.tensor_reads.is_empty() {
        eprintln!("  Top tensor reads by volume:");
        for tensor in profile.tensor_reads.iter().take(12) {
            eprintln!(
                "    {}: {} reads, {}{}",
                tensor.name,
                tensor.read_operations,
                format_bytes(tensor.bytes_read),
                if tensor.read_failures > 0 {
                    " (includes failures)"
                } else {
                    ""
                }
            );
        }
    }
    eprintln!();
    eprintln!("Measured timings (some detailed categories are subsets of layer time):");
    eprintln!(
        "  GGUF read path:           {}",
        format_duration(profile.io.elapsed)
    );
    eprintln!(
        "  Prompt/prefill:           {}",
        format_duration(runtime.prompt)
    );
    eprintln!(
        "  Layer loading:            {}",
        format_duration(runtime.layer_load)
    );
    eprintln!(
        "  Layer compute:            {}",
        format_duration(runtime.layer_compute)
    );
    eprintln!(
        "  Layer release:            {}",
        format_duration(runtime.layer_release)
    );
    eprintln!(
        "  Tensor construction:      {}",
        format_duration(runtime.tensor_construction)
    );
    eprintln!(
        "  Grouped quantized copy:   {}",
        format_duration(runtime.grouped_quantized_copy_time)
    );
    eprintln!(
        "  Explicit dequant/copies:  {}",
        format_duration(runtime.dequantization)
    );
    eprintln!(
        "  F32 matvec (subset):      {}",
        format_duration(runtime.float_matvec)
    );
    eprintln!(
        "  Quant matvec/dequant:     {}",
        format_duration(runtime.quantized_matvec)
    );
    if runtime.q4_0_matvec_calls > 0 {
        eprintln!(
            "  Q4_0 matvec workload:     {} calls, {} rows, {} in elements, {} blocks, {} weight bytes",
            runtime.q4_0_matvec_calls,
            runtime.q4_0_matvec_rows,
            runtime.q4_0_matvec_input_elements,
            runtime.q4_0_matvec_blocks,
            runtime.q4_0_matvec_weight_bytes,
        );
    }
    eprintln!(
        "  Activation allocation:    {}",
        format_duration(runtime.allocation)
    );
    eprintln!(
        "  Output projection:        {}",
        format_duration(runtime.logits)
    );
    eprintln!(
        "  Sampling:                 {}",
        format_duration(runtime.sampling)
    );
    eprintln!(
        "  Stdout callback:          {}",
        format_duration(runtime.output)
    );
}

fn output_memory_report(report: &ramforge_runtime::memory_report::MemoryReport) {
    eprintln!();
    eprintln!("Memory report:");
    eprintln!(
        "  RAMforge managed: current={} peak={} budget={}",
        format_bytes(report.ramforge_current_bytes),
        format_bytes(report.ramforge_peak_bytes),
        format_bytes(report.ramforge_budget_bytes)
    );
    eprintln!(
        "  Process RSS: {}",
        report
            .process_rss_bytes
            .map(format_bytes)
            .unwrap_or_else(|| "unavailable".to_string())
    );
    eprintln!(
        "  System memory: total={} available={}",
        report
            .system_total_bytes
            .map(format_bytes)
            .unwrap_or_else(|| "unavailable".to_string()),
        report
            .system_available_bytes
            .map(format_bytes)
            .unwrap_or_else(|| "unavailable".to_string())
    );
    eprintln!("  Note: RAMforge MemoryBudget does not control Linux page cache or total RSS.");
}

fn print_residency_plan(label: &str, plan: &ResidencyPlan) {
    eprintln!(
        "Residency {label} plan: mode={:?} target={} reserve={} current={} ({:?}) selected={} profile_used={} estimated_reclaim={} estimated_after={} estimated_budget_met={}",
        plan.mode,
        optional_bytes(plan.target_bytes),
        format_bytes(plan.safety_reserve_bytes),
        optional_bytes(plan.observed_bytes),
        plan.measurement_source,
        plan.selected.len(),
        plan.profile_used,
        format_bytes(plan.estimated_reclaim_bytes),
        optional_bytes(plan.estimated_after_bytes),
        plan.estimated_budget_met
    );
    let mut reason_counts: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for decision in &plan.decisions {
        *reason_counts
            .entry(format!("{:?}", decision.reason))
            .or_default() += 1;
    }
    eprintln!("  decisions: {reason_counts:?}");
    for (shown, decision) in plan
        .decisions
        .iter()
        .filter(|decision| decision.action == DecisionAction::Release)
        .enumerate()
    {
        if shown >= 20 {
            eprintln!("  ... additional release decisions omitted");
            break;
        }
        eprintln!(
            "  release {} (layer={:?}) pages={} shared={} estimated_reclaim={}",
            decision.name,
            decision.layer,
            format_bytes(decision.page_bytes),
            format_bytes(decision.shared_page_bytes),
            format_bytes(decision.estimated_reclaim_bytes)
        );
    }
    for note in &plan.notes {
        eprintln!("  note: {note}");
    }
}

fn token_fingerprint(tokens: &[u32]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for token in tokens {
        for byte in token.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    format!("{hash:016x}")
}

const RESIDENCY_REPORT_SCHEMA_VERSION: u32 = 1;
const JSON_SELECTED_LIMIT: usize = 1000;

fn residency_report_json_value(report: &ResidencyReport) -> serde_json::Value {
    let reconciliation = report.ledger.reconciliation(report.inventory.page_bytes);
    let decision_counts = {
        let mut counts = std::collections::BTreeMap::new();
        if let Some(plan) = report.post_plan.as_ref().or(report.pre_plan.as_ref()) {
            for decision in &plan.decisions {
                *counts
                    .entry(format!("{:?}", decision.reason))
                    .or_insert(0usize) += 1;
            }
        }
        counts
    };
    let release_decisions = report
        .post_plan
        .as_ref()
        .or(report.pre_plan.as_ref())
        .map(|plan| {
            plan.decisions
                .iter()
                .filter(|decision| decision.action == DecisionAction::Release)
                .take(200)
                .map(|decision| {
                    serde_json::json!({
                        "tensor_index": decision.key.tensor_index,
                        "generation": decision.key.generation,
                        "name": decision.name,
                        "layer": format!("{:?}", decision.layer),
                        "reason": format!("{:?}", decision.reason),
                        "page_bytes": decision.page_bytes,
                        "shared_page_bytes": decision.shared_page_bytes,
                        "estimated_reclaim_bytes": decision.estimated_reclaim_bytes,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    serde_json::json!({
        "schema_version": RESIDENCY_REPORT_SCHEMA_VERSION,
        "kind": "ramforge.residency.operation_report",
        "model": {
            "id": report.model_id.0,
            "generation": report.generation.0,
        },
        "inventory": {
            "tensors": report.inventory.tensors,
            "file_backed_tensors": report.inventory.file_backed_tensors,
            "anonymous_tensors": report.inventory.anonymous_tensors,
            "mappings": report.inventory.mappings,
            "logical_file_backed_bytes": report.inventory.logical_file_backed_bytes,
            "logical_anonymous_bytes": report.inventory.logical_anonymous_bytes,
            "page_bytes": report.inventory.page_bytes,
            "shared_page_bytes": report.inventory.shared_page_bytes,
            "unknown_tensors": report.inventory.unknown_tensors,
        },
        "admission": {
            "outcome": format!("{:?}", report.admission.outcome),
            "native_model_instance": report.admission.native_model_instance,
            "native_contract_version": report.admission.native_contract_version,
            "native_quiescent": report.admission.native_quiescent,
            "eligible_reclaimable_bytes": report.admission.eligible_reclaimable_bytes,
            "soft_target_bytes": report.admission.soft_target_bytes,
            "hard_limit_bytes": report.admission.hard_limit_bytes,
            "safety_reserve_bytes": report.admission.safety_reserve_bytes,
            "current_bytes": report.admission.current_bytes,
            "known_tensor_bytes": report.admission.known_tensor_bytes,
            "reasons": report.admission.reasons,
            "unknown_components": report.admission.unknown_components,
        },
        "ledger": {
            "entries": report.ledger.entries.iter().map(|entry| serde_json::json!({
                "category": format!("{:?}", entry.category),
                "bytes": entry.bytes,
                "kind": format!("{:?}", entry.kind),
                "source": entry.source,
                "note": entry.note,
            })).collect::<Vec<_>>(),
            "unknown_components": report.ledger.unknown_components,
            "warnings": report.ledger.warnings,
        },
        "reconciliation": {
            "mapped_virtual_bytes": reconciliation.mapped_virtual_bytes,
            "mapped_resident_bytes": reconciliation.mapped_resident_bytes,
            "process_rss_bytes": reconciliation.process_rss_bytes,
            "cgroup_current_bytes": reconciliation.cgroup_current_bytes,
            "cgroup_peak_bytes": reconciliation.cgroup_peak_bytes,
            "rss_minus_mapping": reconciliation.rss_minus_mapping,
            "cgroup_minus_rss": reconciliation.cgroup_minus_rss,
            "unknown_components": reconciliation.unknown_components,
            "notes": reconciliation.notes,
        },
        "pre_plan": report.pre_plan.as_ref().map(plan_json_value),
        "post_plan": report.post_plan.as_ref().map(plan_json_value),
        "plan_decision_counts": decision_counts,
        "release_decisions": release_decisions,
        "staging_plan": report.staging.as_ref().map(|staging| serde_json::json!({
            "selected_count": staging.selected.len(),
            "selected": staging.selected.iter().take(JSON_SELECTED_LIMIT).map(|key| serde_json::json!({
                "tensor_index": key.tensor_index,
                "generation": key.generation,
            })).collect::<Vec<_>>(),
            "selected_bytes": staging.selected_bytes,
            "skipped_bytes": staging.skipped_bytes,
            "overlap_bytes_avoided": staging.overlap_bytes_avoided,
            "notes": staging.notes,
        })),
        "staging_result": report.staging_result.as_ref().map(|result| serde_json::json!({
            "requested": result.requested_keys.len(),
            "native_requested_bytes": result.native_requested_bytes,
            "mapping_before_bytes": result.mapping_before_bytes,
            "mapping_after_bytes": result.mapping_after_bytes,
            "mapping_delta_bytes": match (result.mapping_before_bytes, result.mapping_after_bytes) {
                (Some(before), Some(after)) => Some(after as i128 - before as i128),
                _ => None,
            },
            "error": result.error,
        })),
        "application": report.application.as_ref().map(|application| {
            let native = application.native_result.as_ref();
            serde_json::json!({
                "attempted": application.attempted_keys.len(),
                "native_indices": application.native_indices.len(),
                "elapsed_ns": application.elapsed.as_nanos(),
                "native_released_bytes_submitted": native.map(|result| result.released_bytes),
                "native_skipped_shared_bytes": native.map(|result| result.skipped_shared_bytes),
                "native_non_file_backed_tensors": native.map(|result| result.non_file_backed_tensors),
                "native_selected_tensors": native.map(|result| result.selected_tensors),
                "error": application.error,
            })
        }),
        "resource_lifetime_states": report.lifetime_counts,
        "residency_observation": {
            "before": report.page_before.as_ref().map(page_snapshot_json),
            "after_staging": report.page_after_staging.as_ref().map(page_snapshot_json),
            "after": report.page_after.as_ref().map(page_snapshot_json),
            "prefetch_correlation": report.prefetch_correlation.as_ref().map(prefetch_correlation_json),
        },
        "prediction": report.prediction_set.as_ref().map(|set| {
            let plan = report.prediction_plan.as_ref();
            serde_json::json!({
                "enabled": true,
                "source": format!("{:?}", set.source),
                "profile_version": set.profile_version,
                "complete_observations": set.complete_observations,
                "prediction_count": set.predictions.len(),
                "high_confidence": set.high_confidence,
                "medium_confidence": set.medium_confidence,
                "low_confidence": set.low_confidence,
                "counts": {
                    "predictions": set.predictions.len(),
                    "high": set.high_confidence,
                    "medium": set.medium_confidence,
                    "low": set.low_confidence,
                },
                "history_valid": set.history_valid,
                "rejected": set.rejected,
                "warnings": set.warnings,
                "predictions": set.predictions.iter().map(|prediction| serde_json::json!({
                    "tensor": prediction.name,
                    "tensor_index": prediction.key.tensor_index,
                    "predicted_order": prediction.predicted_order,
                    "expected_distance": prediction.expected_distance,
                    "confidence_bps": prediction.confidence_bps,
                    "confidence": format!("{:?}", prediction.confidence),
                    "size_bytes": prediction.size_bytes,
                    "known_size": prediction.known_size,
                    "reason": prediction.reason,
                })).collect::<Vec<_>>(),
                "metrics": report.prediction_metrics.as_ref().map(|metrics| serde_json::json!({
                    "horizon": metrics.horizon,
                    "predictions": metrics.predictions_generated,
                    "matched_observed": metrics.matched_observed,
                    "matched_within_horizon": metrics.matched_within_horizon,
                    "missed_predictions": metrics.missed_observed,
                    "observed_distinct": metrics.observed_distinct,
                    "precision_bps": metrics.precision_bps,
                    "recall_bps": metrics.recall_bps,
                    "false_positive_rate_bps": metrics.false_positive_rate_bps,
                    "prefetch_requested_resources": metrics.prefetch_requested_resources,
                    "prefetch_requested_bytes": metrics.prefetch_requested_bytes,
                    "prefetch_accepted_bytes": metrics.prefetch_accepted_bytes,
                    "prefetch_accepted_resources": metrics.prefetch_accepted_resources,
                    "prefetch_followed_by_observed_access": metrics.prefetch_followed_by_observed_access,
                    "prefetch_not_followed_by_observed_access": metrics.prefetch_not_followed_by_observed_access,
                    "prefetch_observed_residency_resources": metrics.prefetch_observed_residency_resources,
                    "prefetch_benefit_unknown_resources": metrics.prefetch_benefit_unknown_resources,
                    "duplicate_or_overlap_avoided": metrics.duplicate_or_overlap_avoided,
                    "rejected": metrics.rejected,
                    "deferred": metrics.deferred,
                    "observed_mapping_delta_bytes": metrics.observed_mapping_delta_bytes,
                    "bucket_high": {"predictions": metrics.bucket_high.predictions, "hits": metrics.bucket_high.hits, "within_horizon": metrics.bucket_high.within_horizon, "precision_bps": metrics.bucket_high.precision_bps()},
                    "bucket_medium": {"predictions": metrics.bucket_medium.predictions, "hits": metrics.bucket_medium.hits, "within_horizon": metrics.bucket_medium.within_horizon, "precision_bps": metrics.bucket_medium.precision_bps()},
                    "bucket_low": {"predictions": metrics.bucket_low.predictions, "hits": metrics.bucket_low.hits, "within_horizon": metrics.bucket_low.within_horizon, "precision_bps": metrics.bucket_low.precision_bps()},
                })),
                "plan": plan.map(|plan| serde_json::json!({
                    "stage_count": plan.stage_keys.len(),
                    "prefetch_count": plan.prefetch_keys.len(),
                    "requested_bytes": plan.requested_bytes(),
                    "deferred": plan.deferred,
                    "rejected": plan.rejected,
                    "release_authorized": plan.release_authorized,
                    "notes": plan.notes,
                    "decisions": plan.decisions.iter().map(|decision| serde_json::json!({
                        "tensor": decision.name,
                        "action": format!("{:?}", decision.action),
                        "reason": format!("{:?}", decision.reason),
                        "confidence_bps": decision.confidence_bps,
                        "predicted_order": decision.predicted_order,
                        "requested_bytes": decision.requested_bytes,
                        "detail": decision.detail,
                    })).collect::<Vec<_>>(),
                })),
                "staging": report.prediction_staging.as_ref().map(|staging| serde_json::json!({
                    "requested_bytes": staging.requested_bytes,
                    "native_accepted_bytes": staging.native_accepted_bytes,
                    "elapsed_ns": staging.elapsed_ns,
                    "observed_mapping_delta_bytes": staging.observed_mapping_delta(),
                    "operations": staging.operations,
                    "configured_max_in_flight": staging.configured_max_in_flight,
                    "error": staging.error,
                    "cancelled": staging.cancelled,
                })),
                "history": report.prediction_history.as_ref().map(|history| serde_json::json!({
                    "records": history.records,
                    "total_predictions": history.total_predictions,
                    "hits": history.hits,
                    "misses": history.misses,
                    "matched_within_horizon": history.matched_within_horizon,
                    "prefetch_requested": history.prefetch_requested,
                    "prefetch_accepted_bytes": history.prefetch_accepted_bytes,
                    "prefetch_accepted_resources": history.prefetch_accepted_resources,
                    "prefetch_followed_by_access": history.prefetch_followed_by_access,
                    "prefetch_not_followed_by_access": history.prefetch_not_followed_by_access,
                    "prefetch_benefit_unknown": history.prefetch_benefit_unknown,
                    "duplicate_or_overlap_avoided": history.duplicate_or_overlap_avoided,
                    "rejected": history.rejected,
                    "deferred": history.deferred,
                    "bucket_high_precision_bps": history.bucket_high_precision_bps,
                    "bucket_medium_precision_bps": history.bucket_medium_precision_bps,
                    "bucket_low_precision_bps": history.bucket_low_precision_bps,
                    "staged_requests": history.staged_requests,
                    "staged_bytes": history.staged_bytes,
                    "wasted_bytes": history.wasted_bytes,
                    "accuracy_bps": history.accuracy_bps,
                })),
            })
        }),
        "native_lifetime": {
            "contract_version": report.native_contract_version,
            "model_instance_id": report.native_model_instance,
            "quiescent": report.native_quiescent,
            "context_id": report.native_context_id,
            "execution_generation": report.native_execution_generation,
        },
        "profile_source": report.profile_source,
        "profile_identity": report.profile_identity,
        "profile": report.profile.as_ref().map(|profile| serde_json::json!({
            "version": profile.version,
            "complete_observations": profile.complete_observations,
            "incomplete_observations": profile.incomplete_observations,
            "last_node_count": profile.last_node_count,
            "last_tensor_count": profile.last_tensor_count,
            "invalidated_reason": profile.invalidated_reason,
            "access": profile.access.iter().take(200).map(|(name, pattern)| serde_json::json!({
                "tensor": name,
                "first_order": pattern.first_order,
                "last_order": pattern.last_order,
                "observed_consumers": pattern.observed_consumers,
            })).collect::<Vec<_>>(),
        })),
        "execution_timing": {
            "model_load_ns": report.execution_timing.model_load_ns,
            "profile_load_ns": report.execution_timing.profile_load_ns,
            "history_load_ns": report.execution_timing.history_load_ns,
            "prediction_ns": report.execution_timing.prediction_ns,
            "planning_ns": report.execution_timing.planning_ns,
            "staging_ns": report.execution_timing.staging_ns,
            "observation_ns": report.execution_timing.observation_ns,
            "inference_ns": report.execution_timing.inference_ns,
            "synchronization_ns": report.execution_timing.synchronization_ns,
            "end_to_end_ns": report.execution_timing.end_to_end_ns,
        },
        "timing": {
            "prompt_tokens": report.timing.prompt_tokens,
            "generated_tokens": report.timing.generated_tokens,
            "ttft_ns": report.timing.first_token_elapsed.map(|value| value.as_nanos()),
            "prompt_ns": report.timing.prompt_elapsed.map(|value| value.as_nanos()),
            "decode_ns": report.timing.decode_elapsed.map(|value| value.as_nanos()),
            "total_ns": report.timing.total_elapsed.map(|value| value.as_nanos()),
            "stop_reason": report.timing.stop_reason,
            "output_fingerprint": report.timing.output_fingerprint,
        },
        "process_faults": {
            "before_load_minor": report.minor_faults_before_load,
            "before_load_major": report.major_faults_before_load,
            "after_minor": report.after.process_minor_faults,
            "after_major": report.after.process_major_faults,
        },
        "measurements": {
            "before": snapshot_json_value(&report.before),
            "after": snapshot_json_value(&report.after),
        },
        "estimated_vs_observed": {
            "estimated_reclaim_bytes": report.post_plan.as_ref().or(report.pre_plan.as_ref()).map(|plan| plan.estimated_reclaim_bytes),
            "observed_mapping_delta_bytes": report.before.mapping_delta(&report.after),
        },
        "elapsed_ns": report.elapsed.as_nanos(),
        "warnings": [
            "released_bytes is page bytes submitted to madvise, not confirmed physical reclamation",
            "mapped bytes, resident bytes, process RSS, and cgroup usage are distinct measurements",
            "prefetch is a hint, not a residency guarantee"
        ],
    })
}

fn plan_json_value(plan: &ResidencyPlan) -> serde_json::Value {
    serde_json::json!({
        "mode": format!("{:?}", plan.mode),
        "target_bytes": plan.target_bytes,
        "safety_reserve_bytes": plan.safety_reserve_bytes,
        "observed_bytes": plan.observed_bytes,
        "measurement_source": format!("{:?}", plan.measurement_source),
        "selected_count": plan.selected.len(),
        "selected": plan.selected.iter().take(JSON_SELECTED_LIMIT).map(|key| serde_json::json!({
            "tensor_index": key.tensor_index,
            "generation": key.generation,
        })).collect::<Vec<_>>(),
        "profile_used": plan.profile_used,
        "estimated_reclaim_bytes": plan.estimated_reclaim_bytes,
        "estimated_after_bytes": plan.estimated_after_bytes,
        "estimated_budget_met": plan.estimated_budget_met,
        "notes": plan.notes,
    })
}

fn page_snapshot_json(snapshot: &PageResidencySnapshot) -> serde_json::Value {
    serde_json::json!({
        "supported": snapshot.supported,
        "source": snapshot.source,
        "page_size": snapshot.page_size,
        "inspected_pages": snapshot.inspected_pages,
        "total_pages": snapshot.total_pages,
        "resident_pages": snapshot.resident_pages,
        "nonresident_pages": snapshot.nonresident_pages,
        "unknown_pages": snapshot.unknown_pages,
        "unique_resident_pages": snapshot.unique_resident_pages,
        "unique_nonresident_pages": snapshot.unique_nonresident_pages,
        "unique_unknown_pages": snapshot.unique_unknown_pages,
        "sampled": snapshot.sampled,
        "elapsed_ns": snapshot.elapsed_ns,
        "unsupported_reason": snapshot.unsupported_reason,
        "resources": snapshot.resources.iter().take(512).map(|resource| serde_json::json!({
            "tensor": resource.name,
            "tensor_index": resource.tensor_index,
            "generation": resource.generation,
            "mapping_index": resource.mapping_index,
            "total_pages": resource.total_pages,
            "inspected_pages": resource.inspected_pages,
            "resident_pages": resource.resident_pages,
            "nonresident_pages": resource.nonresident_pages,
            "unknown_pages": resource.unknown_pages,
            "sampled": resource.sampled,
            "error": resource.error,
        })).collect::<Vec<_>>(),
    })
}

fn prefetch_correlation_json(correlation: &PrefetchCorrelationReport) -> serde_json::Value {
    serde_json::json!({
        "already_resident": correlation.already_resident,
        "became_resident": correlation.became_resident,
        "remained_nonresident": correlation.remained_nonresident,
        "unknown": correlation.unknown,
        "access_observed": correlation.access_observed,
        "access_not_observed": correlation.access_not_observed,
        "access_unknown": correlation.access_unknown,
        "clearly_useful": correlation.clearly_useful,
        "potentially_useful": correlation.potentially_useful,
        "redundant": correlation.redundant,
        "wasted": correlation.wasted,
        "benefit_unknown": correlation.benefit_unknown,
        "notes": correlation.notes,
        "resources": correlation.resources.iter().map(|resource| serde_json::json!({
            "tensor": resource.name,
            "before_resident_pages": resource.before_resident_pages,
            "before_total_pages": resource.before_total_pages,
            "before_unknown_pages": resource.before_unknown_pages,
            "after_resident_pages": resource.after_resident_pages,
            "after_total_pages": resource.after_total_pages,
            "after_unknown_pages": resource.after_unknown_pages,
            "class": format!("{:?}", resource.class),
            "benefit": format!("{:?}", resource.benefit),
            "access_observed": resource.access_observed,
        })).collect::<Vec<_>>(),
    })
}

fn snapshot_json_value(snapshot: &ResidencySnapshot) -> serde_json::Value {
    serde_json::json!({
        "process_rss_bytes": snapshot.process_rss_bytes,
        "process_rss_peak_bytes": snapshot.process_rss_peak_bytes,
        "mapping_rss_bytes": snapshot.mapping_rss_bytes,
        "mapping_pss_bytes": snapshot.mapping_pss_bytes,
        "cgroup_current_bytes": snapshot.cgroup_current_bytes,
        "cgroup_peak_bytes": snapshot.cgroup_peak_bytes,
        "process_minor_faults": snapshot.process_minor_faults,
        "process_major_faults": snapshot.process_major_faults,
        "cgroup_events": snapshot.cgroup_events.as_ref().map(|events| serde_json::json!({
            "low": events.low,
            "high": events.high,
            "max": events.max,
            "oom": events.oom,
            "oom_kill": events.oom_kill,
            "oom_group_kill": events.oom_group_kill,
        })),
    })
}

fn print_residency_report(report: &ResidencyReport) {
    eprintln!();
    eprintln!("Residency report:");
    eprintln!(
        "  Model: {} generation={}",
        report.model_id.0, report.generation.0
    );
    eprintln!(
        "  Inventory: tensors={} file_backed={} anonymous={} mappings={} file_bytes={} anonymous_bytes={} page_bytes={} shared_page_bytes={}",
        report.inventory.tensors,
        report.inventory.file_backed_tensors,
        report.inventory.anonymous_tensors,
        report.inventory.mappings,
        format_bytes(report.inventory.logical_file_backed_bytes),
        format_bytes(report.inventory.logical_anonymous_bytes),
        format_bytes(report.inventory.page_bytes),
        format_bytes(report.inventory.shared_page_bytes)
    );
    eprintln!("  Ledger (categories are not interchangeable totals):");
    for entry in &report.ledger.entries {
        eprintln!(
            "    {:?}: {} [{:?}] source={} note={}",
            entry.category,
            optional_bytes(entry.bytes),
            entry.kind,
            entry.source,
            entry.note
        );
    }
    let reconciliation = report.ledger.reconciliation(report.inventory.page_bytes);
    eprintln!(
        "  Reconciliation: mapped_virtual={} mapped_resident={} process_rss={} cgroup_current={} cgroup_peak={} rss_minus_mapping={} cgroup_minus_rss={}",
        format_bytes(reconciliation.mapped_virtual_bytes),
        optional_bytes(reconciliation.mapped_resident_bytes),
        optional_bytes(reconciliation.process_rss_bytes),
        optional_bytes(reconciliation.cgroup_current_bytes),
        optional_bytes(reconciliation.cgroup_peak_bytes),
        optional_signed_bytes(reconciliation.rss_minus_mapping),
        optional_signed_bytes(reconciliation.cgroup_minus_rss)
    );
    for unknown in &reconciliation.unknown_components {
        eprintln!("    unknown component: {unknown}");
    }
    if let Some(staging) = &report.staging {
        eprintln!(
            "  Staging plan: selected={} requested_pages={} skipped={} overlap_avoided={}",
            staging.selected.len(),
            format_bytes(staging.selected_bytes),
            format_bytes(staging.skipped_bytes),
            format_bytes(staging.overlap_bytes_avoided)
        );
        for note in &staging.notes {
            eprintln!("    staging note: {note}");
        }
    }
    if let Some(result) = &report.staging_result {
        eprintln!(
            "  Staging result: requested={} native_accepted={} mapping_before={} mapping_after={} delta={}",
            result.requested_keys.len(),
            format_bytes(result.native_requested_bytes),
            optional_bytes(result.mapping_before_bytes),
            optional_bytes(result.mapping_after_bytes),
            match (result.mapping_before_bytes, result.mapping_after_bytes) {
                (Some(before), Some(after)) => optional_signed_bytes(Some(after as i128 - before as i128)),
                _ => "unavailable".to_string(),
            }
        );
    }
    if let Some(before) = &report.page_before {
        eprintln!(
            "  Page residency: supported={} source={} sampled={} inspected={} total={} resident={} nonresident={} unknown={} observation_ms={:.3}",
            before.supported,
            before.source,
            before.sampled,
            before.inspected_pages,
            before.total_pages,
            before.resident_pages,
            before.nonresident_pages,
            before.unknown_pages,
            before.elapsed_ns as f64 / 1_000_000.0
        );
        if let Some(after) = &report.page_after_staging {
            eprintln!(
                "    after_staging: resident={} nonresident={} unknown={} inspected={} sampled={} observation_ms={:.3}",
                after.resident_pages,
                after.nonresident_pages,
                after.unknown_pages,
                after.inspected_pages,
                after.sampled,
                after.elapsed_ns as f64 / 1_000_000.0
            );
        }
        if let Some(correlation) = &report.prefetch_correlation {
            eprintln!(
                "    prefetch correlation: already_resident={} became_resident={} remained_nonresident={} unknown={} access_observed={} access_not_observed={} access_unknown={} clearly_useful={} potentially_useful={} redundant={} wasted={} benefit_unknown={}",
                correlation.already_resident,
                correlation.became_resident,
                correlation.remained_nonresident,
                correlation.unknown,
                correlation.access_observed,
                correlation.access_not_observed,
                correlation.access_unknown,
                correlation.clearly_useful,
                correlation.potentially_useful,
                correlation.redundant,
                correlation.wasted,
                correlation.benefit_unknown
            );
            for note in &correlation.notes {
                eprintln!("      correlation note: {note}");
            }
        }
    }
    eprintln!("  Resource lifetime states: {:?}", report.lifetime_counts);
    eprintln!(
        "  Native lifetime: contract={:?} instance={:?} quiescent={} context_id={:?} execution_generation={:?}",
        report.native_contract_version,
        report.native_model_instance,
        report.native_quiescent,
        report.native_context_id,
        report.native_execution_generation
    );
    if let Some(profile) = &report.profile {
        eprintln!(
            "  Execution profile: version={} complete_observations={} incomplete_observations={} tensors={} nodes={} invalidated={:?}",
            profile.version,
            profile.complete_observations,
            profile.incomplete_observations,
            profile.access.len(),
            profile.last_node_count,
            profile.invalidated_reason
        );
    }
    eprintln!(
        "  Execution profile source: {} identity={:?}",
        report.profile_source, report.profile_identity
    );
    eprintln!(
        "  Admission: {:?}; current={} soft_target={} hard_limit={} reserve={} known_tensor_bytes={} eligible_reclaimable={} native_quiescent={} contract={:?}",
        report.admission.outcome,
        optional_bytes(report.admission.current_bytes),
        optional_bytes(report.admission.soft_target_bytes),
        optional_bytes(report.admission.hard_limit_bytes),
        format_bytes(report.admission.safety_reserve_bytes),
        format_bytes(report.admission.known_tensor_bytes),
        format_bytes(report.admission.eligible_reclaimable_bytes),
        report.admission.native_quiescent,
        report.admission.native_contract_version
    );
    for reason in &report.admission.reasons {
        eprintln!("    reason: {reason}");
    }
    eprintln!(
        "  Admission unknown components: {}",
        report.admission.unknown_components.join(", ")
    );
    if let Some(plan) = &report.pre_plan {
        print_residency_plan("pre-operation", plan);
    }
    if let Some(plan) = &report.post_plan {
        print_residency_plan("post-operation", plan);
    }
    if let Some(application) = &report.application {
        eprintln!(
            "  Application: attempted={} native_indices={} elapsed={}",
            application.attempted_keys.len(),
            application.native_indices.len(),
            format_duration(application.elapsed)
        );
        if let Some(result) = &application.native_result {
            eprintln!(
                "    native: released={} skipped_shared={} non_file_backed={} selected={}",
                format_bytes(result.released_bytes),
                format_bytes(result.skipped_shared_bytes),
                result.non_file_backed_tensors,
                result.selected_tensors
            );
        }
        if let Some(error) = &application.error {
            eprintln!("    error: {error}");
        }
    } else {
        eprintln!("  Application: none");
    }
    eprintln!(
        "  Mapping residency: before={} after={} delta={}",
        optional_bytes(report.before.mapping_rss_bytes),
        optional_bytes(report.after.mapping_rss_bytes),
        report
            .before
            .mapping_delta(&report.after)
            .map(|delta| format!("{delta:+} bytes"))
            .unwrap_or_else(|| "unavailable".to_string())
    );
    eprintln!(
        "  Process RSS: before={} after={} peak={}",
        optional_bytes(report.before.process_rss_bytes),
        optional_bytes(report.after.process_rss_bytes),
        optional_bytes(report.after.process_rss_peak_bytes)
    );
    eprintln!(
        "  Page faults (process-wide, not attributable to prefetch): before_load_minor={:?} after_minor={:?} before_load_major={:?} after_major={:?}",
        report.minor_faults_before_load,
        report.after.process_minor_faults,
        report.major_faults_before_load,
        report.after.process_major_faults
    );
    eprintln!(
        "  Cgroup: before_current={} after_current={} peak={} events={:?}",
        optional_bytes(report.before.cgroup_current_bytes),
        optional_bytes(report.after.cgroup_current_bytes),
        optional_bytes(report.after.cgroup_peak_bytes),
        report.after.cgroup_events
    );
    eprintln!(
        "  Execution timing: end_to_end={} model_load={:?} profile_load={:?} history_load={:?} prediction={:?} planning={:?} staging={:?} observation={:?} inference={:?} synchronization={:?}",
        format_duration(std::time::Duration::from_nanos(report.execution_timing.end_to_end_ns as u64)),
        report.execution_timing.model_load_ns,
        report.execution_timing.profile_load_ns,
        report.execution_timing.history_load_ns,
        report.execution_timing.prediction_ns,
        report.execution_timing.planning_ns,
        report.execution_timing.staging_ns,
        report.execution_timing.observation_ns,
        report.execution_timing.inference_ns,
        report.execution_timing.synchronization_ns
    );
    eprintln!(
        "  Timing: prompt_tokens={} generated_tokens={} ttft={} decode={} total={} stop={} output_fingerprint={}",
        report.timing.prompt_tokens,
        report.timing.generated_tokens,
        report
            .timing
            .first_token_elapsed
            .map(format_duration)
            .unwrap_or_else(|| "unavailable".to_string()),
        report
            .timing
            .decode_elapsed
            .map(format_duration)
            .unwrap_or_else(|| "unavailable".to_string()),
        report
            .timing
            .total_elapsed
            .map(format_duration)
            .unwrap_or_else(|| "unavailable".to_string()),
        report.timing.stop_reason,
        report.timing.output_fingerprint
    );
    eprintln!("  Residency elapsed: {}", format_duration(report.elapsed));
    eprintln!(
        "  Note: released bytes are page bytes submitted to madvise, not proof of RSS reduction; layer grouping uses tensor-name convention."
    );
}

fn output_forgecore_memory_report(report: &ForgeCoreMemoryReport) {
    eprintln!();
    eprintln!("ForgeCore memory report (measured/native-reported facts):");
    eprintln!(
        "  Requested RAM budget: {} ({})",
        format_bytes(report.requested_budget_bytes),
        match report.enforcement {
            ramforge_runtime::memory_report::BudgetEnforcement::CgroupV2 =>
                "enforced by cgroup v2 memory.max",
            ramforge_runtime::memory_report::BudgetEnforcement::DiagnosticOnly =>
                "diagnostic-only; not enforced",
        }
    );
    eprintln!(
        "  Model load mode: {}",
        report.load_mode.as_deref().unwrap_or("unavailable")
    );
    eprintln!(
        "  CPU weight repacking: {}",
        report.weight_repacking.as_deref().unwrap_or("unavailable")
    );
    eprintln!(
        "  Logical GGUF tensor bytes: {}",
        report
            .logical_model_bytes
            .map(format_bytes)
            .unwrap_or_else(|| "unknown".to_string())
    );
    eprintln!(
        "  Native model bytes (libllama): {}",
        format_bytes(report.native_model_bytes)
    );
    eprintln!(
        "  Logical model exceeds requested budget: {}",
        report
            .logical_model_exceeds_budget()
            .map(yes_no)
            .unwrap_or("unknown")
    );
    eprintln!(
        "  Process RSS: before_load={} after_load={} after_decode={} peak={}",
        optional_bytes(report.rss_before_load_bytes),
        optional_bytes(report.rss_after_load_bytes),
        optional_bytes(report.rss_after_decode_bytes),
        optional_bytes(report.rss_peak_bytes)
    );
    eprintln!(
        "  Measured RSS delta: load={} decode={}",
        optional_signed_bytes(report.rss_delta_after_load_bytes()),
        optional_signed_bytes(report.rss_delta_after_decode_bytes())
    );
    if let Some(residency) = &report.model_residency {
        eprintln!(
            "  Model mapping residency: mapped={} resident={} pss={} mappings={} ({:.1}% of mapping)",
            format_bytes(residency.mapped_bytes),
            format_bytes(residency.rss_bytes),
            format_bytes(residency.pss_bytes),
            residency.mapping_count,
            residency
                .resident_fraction()
                .map(|fraction| fraction * 100.0)
                .unwrap_or(0.0)
        );
        eprintln!(
            "    clean_shared={} clean_private={} dirty_private={} anonymous={}",
            format_bytes(residency.shared_clean_bytes),
            format_bytes(residency.private_clean_bytes),
            format_bytes(residency.private_dirty_bytes),
            format_bytes(residency.anonymous_bytes)
        );
    } else {
        eprintln!("  Model mapping residency: unavailable (no matching file mapping)");
    }
    if let Some(cgroup) = &report.cgroup {
        eprintln!(
            "  Cgroup: limit={} peak={} current={} swap_peak={}",
            optional_bytes(cgroup.limit_bytes),
            optional_bytes(cgroup.peak_bytes),
            optional_bytes(cgroup.current_bytes),
            optional_bytes(cgroup.swap_peak_bytes)
        );
        if let Some(events) = cgroup.events {
            eprintln!(
                "  Cgroup events: max={} high={} oom={} oom_kill={} oom_group_kill={}",
                events.max, events.high, events.oom, events.oom_kill, events.oom_group_kill
            );
        }
        eprintln!(
            "  Cgroup stat: anon={} file={} shmem={} file_mapped={} kernel={} pgfault={} pgmajfault={}",
            optional_bytes(cgroup.anon_bytes),
            optional_bytes(cgroup.file_bytes),
            optional_bytes(cgroup.shmem_bytes),
            optional_bytes(cgroup.file_mapped_bytes),
            optional_bytes(cgroup.kernel_bytes),
            cgroup
                .pgfault
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unavailable".to_string()),
            cgroup
                .pgmajfault
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unavailable".to_string())
        );
    }
    eprintln!(
        "  System memory: total={} available={}",
        optional_bytes(report.system_total_bytes),
        optional_bytes(report.system_available_bytes)
    );
    eprintln!(
        "  Device memory: total={} free={}",
        format_measurement(&report.device_memory_total),
        format_measurement(&report.device_memory_free)
    );
    eprintln!(
        "  Not exposed by the pinned native API: {}",
        report.unexposed_categories().join(", ")
    );
    eprintln!(
        "  Note: RSS includes the loader, page-cache effects, Rust heap, and all other process memory. A file-backed model mapping is reclaimable, but the pinned native API exposes no per-layer eviction hook; cgroup v2 supplies the hard cap when --enforce-ram is set."
    );
}

fn optional_bytes(bytes: Option<u64>) -> String {
    bytes
        .map(format_bytes)
        .unwrap_or_else(|| "unavailable".to_string())
}

fn optional_signed_bytes(delta: Option<i128>) -> String {
    match delta {
        Some(value) if value >= 0 => {
            format!("+{}", format_bytes(value.unsigned_abs() as u64))
        }
        Some(value) => format!("-{}", format_bytes(value.unsigned_abs() as u64)),
        None => "unavailable".to_string(),
    }
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

fn layer_tensor_indices(
    mappings: &[ramforge_runtime::TensorMappingInfo],
    layer: u32,
) -> Vec<usize> {
    let prefix = format!("blk.{layer}.");
    mappings
        .iter()
        .enumerate()
        .filter_map(|(index, mapping)| mapping.name.starts_with(&prefix).then_some(index))
        .collect()
}

fn format_duration(duration: Duration) -> String {
    if duration.as_secs() >= 1 {
        format!("{:.3} s", duration.as_secs_f64())
    } else {
        format!("{:.3} ms", duration.as_secs_f64() * 1000.0)
    }
}

fn format_rate(tokens: usize, duration: Duration) -> String {
    if tokens == 0 || duration.is_zero() {
        return "n/a".to_string();
    }
    format!("{:.2} tok/s", tokens as f64 / duration.as_secs_f64())
}

fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let bytes_f = bytes as f64;
    if bytes_f >= GIB {
        format!("{:.2} GiB ({} bytes)", bytes_f / GIB, bytes)
    } else if bytes_f >= MIB {
        format!("{:.2} MiB ({} bytes)", bytes_f / MIB, bytes)
    } else if bytes_f >= KIB {
        format!("{:.2} KiB ({} bytes)", bytes_f / KIB, bytes)
    } else {
        format!("{} bytes", bytes)
    }
}

fn format_measurement(measurement: &ramforge_core::ByteMeasurement) -> String {
    match measurement.bytes() {
        Some(bytes) => format!(
            "{} ({:?}; {})",
            format_bytes(bytes),
            measurement.certainty(),
            measurement.provenance().description()
        ),
        None => format!("unknown ({})", measurement.provenance().description()),
    }
}

fn output_human(model: &GgufModel, max_tensors: usize, model_set: Option<&GgufModelSet>) {
    let info = model.info();

    println!("RAMforge – GGUF Model Inspection");
    println!("================================");
    println!();
    println!("Model: {}", model.path.display());
    if let Some(model_set) = model_set {
        let sharded = model_set.is_sharded();
        println!(
            "Model set: {} shard(s), {} total file bytes{}",
            model_set.shards.len(),
            model_set.total_file_bytes(),
            if sharded { " (split GGUF)" } else { "" }
        );
        if sharded {
            for (index, shard) in model_set.shards.iter().enumerate() {
                let split_no = shard
                    .get_metadata(ramforge_core::model_set::SPLIT_NO_KEY)
                    .and_then(|value| value.as_u64())
                    .unwrap_or(index as u64);
                println!(
                    "  shard {}: {} ({} bytes, {} tensors)",
                    split_no,
                    shard.path.display(),
                    shard.file_size,
                    shard.tensors.len()
                );
            }
        }
    }
    println!(
        "File size: {} bytes ({:.2} MB)",
        model.file_size,
        model.file_size as f64 / (1024.0 * 1024.0)
    );
    println!("GGUF version: {}", model.version);
    println!("Alignment: {} bytes", model.alignment);
    println!(
        "Data start offset: {} bytes{}",
        model.data_start_offset,
        if model_set.is_some_and(GgufModelSet::is_sharded) {
            " (virtual concatenated shard data)"
        } else {
            ""
        }
    );
    println!();

    println!("Metadata:");
    let architecture = info.architecture.as_deref().unwrap_or("unknown");
    println!("  Architecture: {}", architecture);
    if let Some(capability) = ramforge_runtime::support::architecture_capability(architecture) {
        println!(
            "  Architecture support: inspect={} plan={} legacy-run={} forgecore-run={} ({})",
            yes_no(capability.inspect),
            yes_no(capability.plan),
            capability.legacy_run.label(),
            capability.forgecore_run.label(),
            capability.notes
        );
    } else {
        println!(
            "  Architecture support: inspect=yes plan=yes legacy-run=not-yet forgecore-run=not-yet"
        );
    }
    let model_support = ramforge_runtime::support::model_execution_support(model);
    println!("  Legacy execution support: {}", model_support.label());
    println!(
        "  ForgeCore/libllama execution support: {}",
        yes_no(model_support.is_forgecore_supported())
    );
    if !model_support.unsupported_tensor_formats.is_empty() {
        println!(
            "  Unsupported tensor formats for legacy execution: {}",
            model_support.unsupported_tensor_formats.join(", ")
        );
    }
    if let Some(name) = &info.name {
        println!("  Name: {}", name);
    }
    if let Some(desc) = &info.description {
        let short = if desc.len() > 120 {
            format!("{}...", &desc[..120])
        } else {
            desc.clone()
        };
        println!("  Description: {}", short);
    }
    println!("  Tensor count: {}", model.tensors.len());
    println!("  Metadata KV count: {}", model.metadata.len());
    println!();

    println!("Known model parameters:");
    println!(
        "  Context length: {}",
        info.context_length
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "  Embedding size: {}",
        info.embedding_length
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "  Layer count (block_count): {}",
        info.block_count
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "  Attention head count: {}",
        info.head_count
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "  Attention head count KV: {}",
        info.head_count_kv
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "  Expert count: {}",
        info.expert_count
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "  Experts used per token: {}",
        info.expert_used_count
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    );
    if let Some(ft) = info.file_type {
        println!("  File type: {}", ft);
    } else {
        println!("  File type: unknown");
    }
    println!();

    println!("Tokenizer:");
    println!(
        "  Model: {}",
        info.tokenizer_model.as_deref().unwrap_or("unknown")
    );
    println!(
        "  Vocab size: {}",
        info.vocab_size
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    );
    println!();

    println!("Quantization / tensor type summary:");
    let summary = model.type_summary();
    if summary.is_empty() {
        println!("  (no tensors)");
    } else {
        for (ty, count) in summary {
            println!("  {}: {} tensors", ty, count);
        }
    }
    println!();

    if let Some(total) = model.total_tensor_bytes() {
        println!(
            "Total tensor data (estimated from shapes & types): {} bytes ({:.2} MB)",
            total,
            total as f64 / (1024.0 * 1024.0)
        );
    } else {
        println!("Total tensor data: unknown (contains types with undetermined size)");
    }
    println!();

    println!(
        "Tensors (first {} of {}):",
        max_tensors.min(model.tensors.len()),
        model.tensors.len()
    );
    println!(
        "  {:<50} {:<12} {:<20} {:<12} Bytes",
        "Name", "Type", "Shape", "FileOff"
    );
    println!("  {}", "-".repeat(110));
    for tensor in model.tensors.iter().take(max_tensors) {
        let shape = tensor.shape_string();
        let bytes_str = tensor
            .byte_length
            .map(|b| b.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let display_name = if tensor.name.len() > 48 {
            format!("...{}", &tensor.name[tensor.name.len() - 45..])
        } else {
            tensor.name.clone()
        };
        println!(
            "  {:<50} {:<12} {:<20} {:<12} {}",
            display_name,
            tensor.ggml_type.name(),
            shape,
            tensor.file_offset,
            bytes_str
        );
    }
    if model.tensors.len() > max_tensors {
        println!(
            "  ... and {} more tensors",
            model.tensors.len() - max_tensors
        );
    }
    println!();

    println!(
        "Note: Tensor payloads were NOT loaded into RAM. Only metadata and descriptors were read."
    );
    println!("This design enables future out-of-core access for models larger than RAM.");
}

fn output_plan_human(plan: &ramforge_runtime::plan::PlanResult, model: &GgufModel, ram_str: &str) {
    let info = model.info();

    println!("RAMforge – Out-of-Core Execution Plan");
    println!("=======================================");
    println!();
    println!("Model:");
    println!("  Path: {}", model.path.display());
    println!(
        "  File size: {} bytes ({:.2} MB, {:.2} GiB)",
        plan.file_size,
        plan.file_size as f64 / (1024.0 * 1024.0),
        plan.file_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    let architecture = info.architecture.as_deref().unwrap_or("unknown");
    println!("  Architecture: {}", architecture);
    let model_support = ramforge_runtime::support::model_execution_support(model);
    println!(
        "  Legacy execution support: {} (inspection/planning remain available)",
        model_support.label()
    );
    println!(
        "  ForgeCore/libllama execution support: {} (RAM budget is diagnostics-only on this backend)",
        yes_no(model_support.is_forgecore_supported())
    );
    if !model_support.unsupported_tensor_formats.is_empty() {
        println!(
            "  Unsupported tensor formats for legacy execution: {}",
            model_support.unsupported_tensor_formats.join(", ")
        );
    }
    println!("  Tensor count: {}", plan.tensor_count);
    if let Some(total) = plan.total_tensor_bytes {
        println!(
            "  Total tensor bytes (estimated): {} bytes ({:.2} MiB)",
            total,
            total as f64 / (1024.0 * 1024.0)
        );
    }
    println!();

    println!("RAM Budget:");
    println!("  Requested: {} ({} bytes)", ram_str, plan.ram_requested);
    println!("  Total budget: {} bytes", plan.budget.total_bytes());
    println!("  Pre-reserved allocations: none (runtime charges weights, one streamed layer, KV cache, and scoped temps on demand)");
    println!("  Used: {} bytes", plan.budget.used_bytes());
    println!(
        "  Available: {} bytes ({:.2} MiB)",
        plan.available,
        plan.available as f64 / (1024.0 * 1024.0)
    );
    println!();

    println!("Model Residency:");
    if plan.fits_in_ram {
        println!("  Fits entirely in RAM budget: yes");
        println!("  File-backed needed: 0 bytes");
    } else {
        println!("  Fits entirely in RAM budget: no");
        println!(
            "  File-backed needed: {} bytes ({:.2} MiB, {:.2} GiB)",
            plan.file_backed_needed,
            plan.file_backed_needed as f64 / (1024.0 * 1024.0),
            plan.file_backed_needed as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!("  Strategy: Model larger than budget – tensor data will be accessed file-backed on demand");
    }
    println!();

    println!("Legacy execution memory preflight:");
    if let Some(execution) = &plan.execution_memory {
        println!(
            "  Policy-resident persistent weights: {} across {} tensors",
            format_bytes(execution.persistent_resident_bytes),
            execution.resident_persistent_count
        );
        println!(
            "  Streamed persistent tensors: {}",
            execution.streamed_persistent_count
        );
        println!(
            "  Persistent startup peak: {}",
            format_bytes(execution.persistent_startup_peak_bytes)
        );
        println!(
            "  Largest layer: {} ({} tensors, resident {}, load peak {})",
            execution.largest_layer_index,
            execution.largest_layer_tensor_count,
            format_bytes(execution.largest_layer_resident_bytes),
            format_bytes(execution.largest_layer_load_peak_bytes)
        );
        println!(
            "  Managed layer-streaming lower bound: {}",
            format_bytes(execution.managed_lower_bound_bytes)
        );
        println!(
            "  Lower bound fits requested budget: {}",
            yes_no(execution.layer_streaming_lower_bound_fits)
        );
        println!(
            "  Layer-cache byte capacity: {}",
            format_bytes(execution.layer_cache_capacity_bytes)
        );
        println!(
            "  Estimated maximum complete cached layers: {} (smallest layer {})",
            execution.max_complete_cached_layers,
            format_bytes(execution.min_layer_resident_bytes)
        );
        println!("  Cache capacity does not guarantee hits; execution order and mandatory workspaces can force eviction");
        println!(
            "  Layer reads per full forward: {} logical tensors -> {} estimated physical ranges",
            execution.logical_tensor_reads_per_forward,
            execution.estimated_physical_reads_per_forward
        );
        println!(
            "  Estimated read bytes per full forward: logical {} physical {} (gap overhead {})",
            format_bytes(execution.logical_tensor_bytes_per_forward),
            format_bytes(execution.estimated_physical_bytes_per_forward),
            format_bytes(execution.estimated_gap_bytes_per_forward)
        );
        println!("  Read estimate is descriptor-only; runtime falls back to individual reads when grouped-buffer headroom is unavailable");
        println!("  Scope: necessary but not sufficient; excludes forward activations, KV cache, logits, and streamed-persistent workspaces");
    } else {
        println!(
            "  Unavailable: {}",
            plan.execution_preflight_error
                .as_deref()
                .unwrap_or("execution preflight unavailable")
        );
    }
    println!();

    println!("Generic byte-cache bound (informational):");
    println!(
        "  Capacity bound (informational): {} bytes ({:.2} MiB, {:.2} GiB)",
        plan.cache_capacity,
        plan.cache_capacity as f64 / (1024.0 * 1024.0),
        plan.cache_capacity as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "  Policy: LRU (least recently used eviction), contents charged to the budget per entry"
    );
    println!("  Static overhead pre-reservation: none (scoped temp guards charge exact lifetimes)");
    println!("  Accounting: RAMforge-managed memory = memory tracked via MemoryBudget. Does NOT include total process RSS or OS page cache.");
    println!();

    println!("File-backed access:");
    println!("  Data source: {}", model.path.display());
    println!(
        "  Tensor data location: file-backed via offsets (data_start {} + tensor.offset)",
        model.data_start_offset
    );
    println!("  Access method: explicit read_range with validation (no full model load)");
    println!();

    println!("Note: the RAMforge budget covers tracked weights, KV cache, and temporary runtime allocations. For inference, use 'ramforge run'.");
}

fn output_json(
    model: &GgufModel,
    max_tensors: usize,
    model_set: Option<&GgufModelSet>,
) -> anyhow::Result<()> {
    use serde_json::json;

    let info = model.info();

    let mut metadata_json = serde_json::Map::new();
    for (k, v) in &model.metadata {
        let json_val = metadata_value_to_json(v);
        metadata_json.insert(k.clone(), json_val);
    }

    let tensors_json: Vec<_> = model
        .tensors
        .iter()
        .take(max_tensors)
        .map(|t| {
            json!({
                "name": t.name,
                "dimensions": t.dimensions,
                "ggml_type": t.ggml_type.name(),
                "ggml_type_id": t.ggml_type.as_u32(),
                "offset": t.offset,
                "file_offset": t.file_offset,
                "byte_length": t.byte_length,
                "num_elements": t.num_elements,
            })
        })
        .collect();

    let summary = model.type_summary();
    let model_support = ramforge_runtime::support::model_execution_support(model);
    let execution_supported = model_support.is_legacy_supported();

    let shards_json: Option<Vec<_>> = model_set.map(|set| {
        set.shards
            .iter()
            .enumerate()
            .map(|(index, shard)| {
                let split_no = shard
                    .get_metadata(ramforge_core::model_set::SPLIT_NO_KEY)
                    .and_then(|value| value.as_u64())
                    .unwrap_or(index as u64);
                json!({
                    "split_no": split_no,
                    "path": shard.path,
                    "file_size": shard.file_size,
                    "data_start_offset": shard.data_start_offset,
                    "tensor_count": shard.tensors.len(),
                })
            })
            .collect()
    });

    let output = json!({
        "path": model.path,
        "file_size": model.file_size,
        "sharded": model_set.is_some_and(GgufModelSet::is_sharded),
        "shard_count": model_set.map_or(1, |set| set.shards.len()),
        "total_file_size": model_set.map_or(model.file_size, GgufModelSet::total_file_bytes),
        "shards": shards_json,
        "version": model.version,
        "alignment": model.alignment,
        "data_start_offset": model.data_start_offset,
        "architecture": info.architecture,
        "name": info.name,
        "description": info.description,
        "tensor_count": model.tensors.len(),
        "metadata_kv_count": model.metadata.len(),
        "context_length": info.context_length,
        "embedding_length": info.embedding_length,
        "block_count": info.block_count,
        "head_count": info.head_count,
        "head_count_kv": info.head_count_kv,
        "expert_count": info.expert_count,
        "expert_used_count": info.expert_used_count,
        "file_type": info.file_type,
        "tokenizer_model": info.tokenizer_model,
        "vocab_size": info.vocab_size,
        "type_summary": summary,
        "execution_supported": execution_supported,
        "execution_supported_scope": "legacy-ramforge-executor",
        "forgecore_execution_supported": model_support.is_forgecore_supported(),
        "unsupported_tensor_formats": model_support.unsupported_tensor_formats,
        "total_tensor_bytes": model.total_tensor_bytes(),
        "tensors": tensors_json,
        "tensors_truncated": model.tensors.len() > max_tensors,
        "total_tensors": model.tensors.len(),
        "metadata": metadata_json,
    });

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn output_plan_json(
    plan: &ramforge_runtime::plan::PlanResult,
    model: &GgufModel,
) -> anyhow::Result<()> {
    use serde_json::json;

    let allocations: std::collections::BTreeMap<String, u64> =
        plan.budget.allocations().clone().into_iter().collect();
    let execution_preflight = if let Some(execution) = &plan.execution_memory {
        json!({
            "available": true,
            "block_count": execution.block_count,
            "resident_persistent_count": execution.resident_persistent_count,
            "streamed_persistent_count": execution.streamed_persistent_count,
            "persistent_resident_bytes": execution.persistent_resident_bytes,
            "persistent_startup_peak_bytes": execution.persistent_startup_peak_bytes,
            "largest_layer_index": execution.largest_layer_index,
            "largest_layer_tensor_count": execution.largest_layer_tensor_count,
            "largest_layer_resident_bytes": execution.largest_layer_resident_bytes,
            "largest_layer_load_peak_bytes": execution.largest_layer_load_peak_bytes,
            "managed_lower_bound_bytes": execution.managed_lower_bound_bytes,
            "layer_streaming_lower_bound_fits": execution.layer_streaming_lower_bound_fits,
            "layer_cache_capacity_bytes": execution.layer_cache_capacity_bytes,
            "max_complete_cached_layers": execution.max_complete_cached_layers,
            "min_layer_resident_bytes": execution.min_layer_resident_bytes,
            "logical_tensor_reads_per_forward": execution.logical_tensor_reads_per_forward,
            "estimated_physical_reads_per_forward": execution.estimated_physical_reads_per_forward,
            "logical_tensor_bytes_per_forward": execution.logical_tensor_bytes_per_forward,
            "estimated_physical_bytes_per_forward": execution.estimated_physical_bytes_per_forward,
            "estimated_gap_bytes_per_forward": execution.estimated_gap_bytes_per_forward,
            "read_estimate_scope": "descriptor-only; runtime may fall back to individual reads when grouped-buffer headroom is unavailable",
            "cache_scope": "capacity estimate only; execution order and mandatory workspaces determine actual retention and hits",
            "scope": "necessary but not sufficient; excludes activations, KV cache, logits, and streamed-persistent workspaces",
        })
    } else {
        json!({
            "available": false,
            "reason": plan.execution_preflight_error.as_deref(),
        })
    };

    let model_support = ramforge_runtime::support::model_execution_support(model);
    let execution_supported = model_support.is_legacy_supported();
    let output = json!({
        "model": {
            "path": model.path,
            "file_size": plan.file_size,
            "architecture": plan.architecture,
            "tensor_count": plan.tensor_count,
            "total_tensor_bytes": plan.total_tensor_bytes,
            "execution_supported": execution_supported,
            "execution_supported_scope": "legacy-ramforge-executor",
            "forgecore_execution_supported": model_support.is_forgecore_supported(),
            "unsupported_tensor_formats": model_support.unsupported_tensor_formats,
        },
        "ram_budget": {
            "requested_str": format!("{} bytes", plan.ram_requested),
            "requested_bytes": plan.ram_requested,
            "total": plan.budget.total_bytes(),
            "used": plan.budget.used_bytes(),
            "available": plan.available,
            "allocations": allocations,
        },
        "residency": {
            "fits_in_ram": plan.fits_in_ram,
            "file_backed_needed": plan.file_backed_needed,
        },
        "execution_preflight": execution_preflight,
        "cache": {
            "capacity": plan.cache_capacity,
            "policy": "LRU",
            "overhead_reserved": plan.overhead_reserved,
        }
    });

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn metadata_value_to_json(v: &ramforge_core::MetadataValue) -> serde_json::Value {
    use ramforge_core::MetadataValue;
    match v {
        MetadataValue::UInt8(x) => serde_json::json!(*x),
        MetadataValue::Int8(x) => serde_json::json!(*x),
        MetadataValue::UInt16(x) => serde_json::json!(*x),
        MetadataValue::Int16(x) => serde_json::json!(*x),
        MetadataValue::UInt32(x) => serde_json::json!(*x),
        MetadataValue::Int32(x) => serde_json::json!(*x),
        MetadataValue::Float32(x) => serde_json::json!(*x),
        MetadataValue::Bool(x) => serde_json::json!(*x),
        MetadataValue::String(s) => serde_json::json!(s),
        MetadataValue::UInt64(x) => serde_json::json!(*x),
        MetadataValue::Int64(x) => serde_json::json!(*x),
        MetadataValue::Float64(x) => serde_json::json!(*x),
        MetadataValue::Array(arr) => {
            let vals: Vec<_> = arr.values.iter().map(metadata_value_to_json).collect();
            serde_json::json!(vals)
        }
    }
}
