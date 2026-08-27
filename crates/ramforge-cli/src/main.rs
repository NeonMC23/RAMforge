use clap::{Parser, Subcommand};
use ramforge_core::{parse_gguf_file, parse_memory_size, GgufModel};
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(name = "ramforge", version, about = "RAMforge – usable, profiled out-of-core GGUF inference")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

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
    /// Run real inference (CPU, supported dense architectures, out-of-core streaming)
    Run {
        /// Path to the GGUF model file
        model: PathBuf,

        /// RAM budget (e.g. 8G, 8GiB, 8192M, 512MiB)
        #[arg(long)]
        ram: String,

        /// Prompt text
        #[arg(long)]
        prompt: String,

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
        Commands::Run {
            model,
            ram,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            verbose,
            profile,
            memory_report,
        } => {
            if let Err(e) = run_inference(
                model,
                ram,
                prompt,
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
    let gguf = parse_gguf_file(&model_path).map_err(|e| anyhow::anyhow!(e))?;

    if json_output {
        output_json(&gguf, max_tensors)?;
    } else {
        output_human(&gguf, max_tensors);
    }

    Ok(())
}

fn run_plan(model_path: PathBuf, ram_str: String, json_output: bool) -> anyhow::Result<()> {
    let ram_bytes = parse_memory_size(&ram_str).map_err(|e| anyhow::anyhow!("invalid --ram '{}': {}", ram_str, e))?;

    let gguf = parse_gguf_file(&model_path).map_err(|e| anyhow::anyhow!(e))?;

    let plan = ramforge_runtime::plan::plan_model(&gguf, ram_bytes).map_err(|e| anyhow::anyhow!(e))?;

    if json_output {
        output_plan_json(&plan, &gguf)?;
    } else {
        output_plan_human(&plan, &gguf, &ram_str);
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_inference(
    model_path: PathBuf,
    ram_str: String,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    verbose: bool,
    profile: bool,
    memory_report: bool,
) -> anyhow::Result<()> {
    let ram_bytes = parse_memory_size(&ram_str).map_err(|e| anyhow::anyhow!("invalid --ram '{}': {}", ram_str, e))?;

    // Diagnostics to stderr
    eprintln!("RAMforge – Usable Out-of-Core Inference");
    eprintln!("========================================================");
    eprintln!("Model: {}", model_path.display());
    eprintln!("RAM budget: {} ({} bytes)", ram_str, ram_bytes);
    eprintln!("Prompt: {:?}", prompt);
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
        model_path.to_str().ok_or_else(|| anyhow::anyhow!("invalid model path"))?,
        ram_bytes,
    )
    .map_err(|e| anyhow::anyhow!(e))?;
    let model_load_elapsed = model_load_started.elapsed();
    engine.set_profiling(profile);

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
    eprintln!("Tokenizer: model={}, vocab_size={}, bos={:?}, eos={:?}",
        engine.tokenizer.model,
        engine.tokenizer.vocab_size(),
        engine.tokenizer.bos_id,
        engine.tokenizer.eos_id
    );
    eprintln!(
        "Execution backend: CPU ({} mode)",
        ramforge_runtime::backend::ComputeBackend::name(&engine.backend)
    );
    eprintln!("Memory budget: total={} used={} (resident weights) available={}",
        engine.budget.total_bytes(),
        engine.budget.used_bytes(),
        engine.budget.available_bytes()
    );
    eprintln!("Total model weight bytes: {} (from descriptors)", engine.model.total_weight_bytes);
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
    let (gen_tokens, _gen_text) = generation.map_err(|e| anyhow::anyhow!(e))?;

    eprintln!("Generated {} tokens", gen_tokens.len());

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
        eprintln!("  Total model weight bytes: {} ({:.2} MiB)", stats.total_model_weight_bytes, stats.total_model_weight_bytes as f64 / (1024.0*1024.0));
        eprintln!("  Current resident layer bytes: {}", stats.current_resident_layer_bytes);
        eprintln!("  Peak resident layer bytes: {} ({:.2} MiB)", stats.peak_resident_layer_bytes, stats.peak_resident_layer_bytes as f64 / (1024.0*1024.0));
        eprintln!("  Peak managed bytes: {} ({:.2} MiB) / budget {} ({:.2} MiB)", stats.peak_managed_bytes, stats.peak_managed_bytes as f64 / (1024.0*1024.0), ram_bytes, ram_bytes as f64 / (1024.0*1024.0));
        eprintln!("  Layer loads: {}", stats.num_layer_loads);
        eprintln!("  Layer releases: {}", stats.num_layer_releases);
        eprintln!("  Layers retained in cache: {}", stats.num_layer_cached);
        eprintln!("  Fits check: total {} > budget {} ? {}", stats.total_model_weight_bytes, ram_bytes, stats.total_model_weight_bytes > ram_bytes);
        eprintln!("  Peak layer < total ? {}", stats.peak_resident_layer_bytes < stats.total_model_weight_bytes);
        eprintln!("  Peak managed <= budget ? {}", stats.peak_managed_bytes <= ram_bytes);
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

fn output_support() {
    use ramforge_runtime::support::{ARCHITECTURES, QUANTIZATIONS};

    println!("RAMforge capability registry");
    println!("Architecture support:");
    println!("  {:<10} {:<9} {:<9} {:<18} Tokenizer", "Architecture", "Inspect", "Plan", "Run");
    for capability in ARCHITECTURES {
        println!(
            "  {:<10} {:<9} {:<9} {:<18} {}",
            capability.name,
            yes_no(capability.inspect),
            yes_no(capability.plan),
            capability.run.label(),
            capability.tokenizer
        );
        println!("    {}", capability.notes);
    }
    println!();
    println!("Quantization support:");
    for capability in QUANTIZATIONS {
        println!(
            "  {:<18} inference={}",
            capability.name,
            yes_no(capability.inference)
        );
    }
    println!();
    println!("Inspection/planning are generic GGUF operations and do not imply execution support.");
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
        eprintln!("  Average per generated token: {}", format_duration(average));
    }
    eprintln!("  Maximum token loop: {}", format_duration(runtime.max_token_latency));
    eprintln!("  Logical tensor reads: {}", profile.io.logical_tensor_reads);
    eprintln!(
        "  Logical tensor bytes: {}",
        format_bytes(profile.io.logical_tensor_bytes)
    );
    eprintln!("  Physical GGUF reads: {}", profile.io.read_operations);
    eprintln!("  Physical GGUF bytes: {}", format_bytes(profile.io.bytes_read));
    eprintln!("  Physical seeks: {}", profile.io.seek_operations);
    eprintln!("  Seeks safely avoided: {}", profile.io.seeks_avoided);
    eprintln!("  GGUF read failures: {}", profile.io.read_failures);
    eprintln!("  Coalesced physical ranges: {}", profile.io.coalesced_ranges);
    eprintln!(
        "  Coalesced gap overhead: {}",
        format_bytes(profile.io.coalesced_gap_bytes)
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
    eprintln!("  Layer-cache misses (disk loads): {}", runtime.cache_misses);
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
    eprintln!("  Peak RAMforge-managed memory: {}", format_bytes(profile.ramforge_peak_bytes));
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
    eprintln!("  GGUF read path:           {}", format_duration(profile.io.elapsed));
    eprintln!("  Prompt/prefill:           {}", format_duration(runtime.prompt));
    eprintln!("  Layer loading:            {}", format_duration(runtime.layer_load));
    eprintln!("  Layer compute:            {}", format_duration(runtime.layer_compute));
    eprintln!("  Layer release:            {}", format_duration(runtime.layer_release));
    eprintln!("  Tensor construction:      {}", format_duration(runtime.tensor_construction));
    eprintln!("  Explicit dequant/copies:  {}", format_duration(runtime.dequantization));
    eprintln!("  F32 matvec (subset):      {}", format_duration(runtime.float_matvec));
    eprintln!("  Quant matvec/dequant:     {}", format_duration(runtime.quantized_matvec));
    eprintln!("  Activation allocation:    {}", format_duration(runtime.allocation));
    eprintln!("  Output projection:        {}", format_duration(runtime.logits));
    eprintln!("  Sampling:                 {}", format_duration(runtime.sampling));
    eprintln!("  Stdout callback:          {}", format_duration(runtime.output));
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

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn format_duration(duration: Duration) -> String {
    if duration.as_secs() >= 1 {
        format!("{:.3} s", duration.as_secs_f64())
    } else {
        format!("{:.3} ms", duration.as_secs_f64() * 1000.0)
    }
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

fn output_human(model: &GgufModel, max_tensors: usize) {
    let info = model.info();

    println!("RAMforge – GGUF Model Inspection");
    println!("================================");
    println!();
    println!("Model: {}", model.path.display());
    println!("File size: {} bytes ({:.2} MB)", model.file_size, model.file_size as f64 / (1024.0 * 1024.0));
    println!("GGUF version: {}", model.version);
    println!("Alignment: {} bytes", model.alignment);
    println!("Data start offset: {} bytes", model.data_start_offset);
    println!();

    println!("Metadata:");
    let architecture = info.architecture.as_deref().unwrap_or("unknown");
    println!("  Architecture: {}", architecture);
    if let Some(capability) = ramforge_runtime::support::architecture_capability(architecture) {
        println!(
            "  Architecture support: inspect={} plan={} run={} ({})",
            yes_no(capability.inspect),
            yes_no(capability.plan),
            capability.run.label(),
            capability.notes
        );
    } else {
        println!("  Architecture support: inspect=yes plan=yes run=not-yet");
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
    println!("  Context length: {}", info.context_length.map(|v| v.to_string()).unwrap_or_else(|| "unknown".to_string()));
    println!("  Embedding size: {}", info.embedding_length.map(|v| v.to_string()).unwrap_or_else(|| "unknown".to_string()));
    println!("  Layer count (block_count): {}", info.block_count.map(|v| v.to_string()).unwrap_or_else(|| "unknown".to_string()));
    println!("  Attention head count: {}", info.head_count.map(|v| v.to_string()).unwrap_or_else(|| "unknown".to_string()));
    println!("  Attention head count KV: {}", info.head_count_kv.map(|v| v.to_string()).unwrap_or_else(|| "unknown".to_string()));
    println!("  Expert count: {}", info.expert_count.map(|v| v.to_string()).unwrap_or_else(|| "unknown".to_string()));
    println!("  Experts used per token: {}", info.expert_used_count.map(|v| v.to_string()).unwrap_or_else(|| "unknown".to_string()));
    if let Some(ft) = info.file_type {
        println!("  File type: {}", ft);
    } else {
        println!("  File type: unknown");
    }
    println!();

    println!("Tokenizer:");
    println!("  Model: {}", info.tokenizer_model.as_deref().unwrap_or("unknown"));
    println!("  Vocab size: {}", info.vocab_size.map(|v| v.to_string()).unwrap_or_else(|| "unknown".to_string()));
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
        println!("Total tensor data (estimated from shapes & types): {} bytes ({:.2} MB)", total, total as f64 / (1024.0*1024.0));
    } else {
        println!("Total tensor data: unknown (contains types with undetermined size)");
    }
    println!();

    println!("Tensors (first {} of {}):", max_tensors.min(model.tensors.len()), model.tensors.len());
    println!("  {:<50} {:<12} {:<20} {:<12} Bytes", "Name", "Type", "Shape", "FileOff");
    println!("  {}", "-".repeat(110));
    for tensor in model.tensors.iter().take(max_tensors) {
        let shape = tensor.shape_string();
        let bytes_str = tensor.byte_length.map(|b| b.to_string()).unwrap_or_else(|| "unknown".to_string());
        let display_name = if tensor.name.len() > 48 {
            format!("...{}", &tensor.name[tensor.name.len()-45..])
        } else {
            tensor.name.clone()
        };
        println!("  {:<50} {:<12} {:<20} {:<12} {}", display_name, tensor.ggml_type.name(), shape, tensor.file_offset, bytes_str);
    }
    if model.tensors.len() > max_tensors {
        println!("  ... and {} more tensors", model.tensors.len() - max_tensors);
    }
    println!();

    println!("Note: Tensor payloads were NOT loaded into RAM. Only metadata and descriptors were read.");
    println!("This design enables future out-of-core access for models larger than RAM.");
}

fn output_plan_human(plan: &ramforge_runtime::plan::PlanResult, model: &GgufModel, ram_str: &str) {
    let info = model.info();

    println!("RAMforge – Out-of-Core Execution Plan");
    println!("=======================================");
    println!();
    println!("Model:");
    println!("  Path: {}", model.path.display());
    println!("  File size: {} bytes ({:.2} MB, {:.2} GiB)", plan.file_size, plan.file_size as f64 / (1024.0*1024.0), plan.file_size as f64 / (1024.0*1024.0*1024.0));
    let architecture = info.architecture.as_deref().unwrap_or("unknown");
    println!("  Architecture: {}", architecture);
    let run_support = ramforge_runtime::support::architecture_capability(architecture)
        .map(|capability| capability.run.label())
        .unwrap_or("not-yet");
    println!("  Execution support: {} (inspection/planning remain available)", run_support);
    println!("  Tensor count: {}", plan.tensor_count);
    if let Some(total) = plan.total_tensor_bytes {
        println!("  Total tensor bytes (estimated): {} bytes ({:.2} MiB)", total, total as f64 / (1024.0*1024.0));
    }
    println!();

    println!("RAM Budget:");
    println!("  Requested: {} ({} bytes)", ram_str, plan.ram_requested);
    println!("  Total budget: {} bytes", plan.budget.total_bytes());
    println!("  Pre-reserved allocations: none (runtime charges weights, one streamed layer, KV cache, and scoped temps on demand)");
    println!("  Used: {} bytes", plan.budget.used_bytes());
    println!("  Available: {} bytes ({:.2} MiB)", plan.available, plan.available as f64 / (1024.0*1024.0));
    println!();

    println!("Model Residency:");
    if plan.fits_in_ram {
        println!("  Fits entirely in RAM budget: yes");
        println!("  File-backed needed: 0 bytes");
    } else {
        println!("  Fits entirely in RAM budget: no");
        println!("  File-backed needed: {} bytes ({:.2} MiB, {:.2} GiB)", plan.file_backed_needed, plan.file_backed_needed as f64 / (1024.0*1024.0), plan.file_backed_needed as f64 / (1024.0*1024.0*1024.0));
        println!("  Strategy: Model larger than budget – tensor data will be accessed file-backed on demand");
    }
    println!();

    println!("Execution memory preflight:");
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
    println!("  Capacity bound (informational): {} bytes ({:.2} MiB, {:.2} GiB)", plan.cache_capacity, plan.cache_capacity as f64 / (1024.0*1024.0), plan.cache_capacity as f64 / (1024.0*1024.0*1024.0));
    println!("  Policy: LRU (least recently used eviction), contents charged to the budget per entry");
    println!("  Static overhead pre-reservation: none (scoped temp guards charge exact lifetimes)");
    println!("  Accounting: RAMforge-managed memory = memory tracked via MemoryBudget. Does NOT include total process RSS or OS page cache.");
    println!();

    println!("File-backed access:");
    println!("  Data source: {}", model.path.display());
    println!("  Tensor data location: file-backed via offsets (data_start {} + tensor.offset)", model.data_start_offset);
    println!("  Access method: explicit read_range with validation (no full model load)");
    println!();

    println!("Note: the RAMforge budget covers tracked weights, KV cache, and temporary runtime allocations. For inference, use 'ramforge run'.");
}

fn output_json(model: &GgufModel, max_tensors: usize) -> anyhow::Result<()> {
    use serde_json::json;

    let info = model.info();

    let mut metadata_json = serde_json::Map::new();
    for (k, v) in &model.metadata {
        let json_val = metadata_value_to_json(v);
        metadata_json.insert(k.clone(), json_val);
    }

    let tensors_json: Vec<_> = model.tensors.iter().take(max_tensors).map(|t| {
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
    }).collect();

    let summary = model.type_summary();

    let output = json!({
        "path": model.path,
        "file_size": model.file_size,
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
        "total_tensor_bytes": model.total_tensor_bytes(),
        "tensors": tensors_json,
        "tensors_truncated": model.tensors.len() > max_tensors,
        "total_tensors": model.tensors.len(),
        "metadata": metadata_json,
    });

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn output_plan_json(plan: &ramforge_runtime::plan::PlanResult, model: &GgufModel) -> anyhow::Result<()> {
    use serde_json::json;

    let allocations: std::collections::BTreeMap<String, u64> = plan.budget.allocations().clone().into_iter().collect();
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

    let output = json!({
        "model": {
            "path": model.path,
            "file_size": plan.file_size,
            "architecture": plan.architecture,
            "tensor_count": plan.tensor_count,
            "total_tensor_bytes": plan.total_tensor_bytes,
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
