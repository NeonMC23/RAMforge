use clap::{Parser, Subcommand};
use ramforge_core::{parse_gguf_file, GgufModel};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "ramforge", version, about = "RAMforge – hierarchical memory inference runtime (milestone 1: inspection only)")]
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
    }
}

fn run_inspect(model_path: PathBuf, json_output: bool, max_tensors: usize) -> anyhow::Result<()> {
    // Parse file – file-backed, no tensor payload loading
    let gguf = parse_gguf_file(&model_path).map_err(|e| anyhow::anyhow!(e))?;

    if json_output {
        output_json(&gguf, max_tensors)?;
    } else {
        output_human(&gguf, max_tensors);
    }

    Ok(())
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
    println!("  Architecture: {}", info.architecture.as_deref().unwrap_or("unknown"));
    if let Some(name) = &info.name {
        println!("  Name: {}", name);
    }
    if let Some(desc) = &info.description {
        // Truncate long descriptions
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
        // Truncate long names
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

fn output_json(model: &GgufModel, max_tensors: usize) -> anyhow::Result<()> {
    use serde_json::json;

    let info = model.info();

    // Convert metadata to JSON-friendly map
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
