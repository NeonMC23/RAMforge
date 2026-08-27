//! Planning logic for `ramforge plan` command
//!
//! The plan distinguishes *file size vs budget* (out-of-core viability) from
//! what the runtime actually charges. For directly executable architectures it
//! also computes a runtime-aligned lower bound for resident persistent weights
//! plus the largest streamed layer load. This lower bound deliberately excludes
//! prompt-dependent KV, activations, logits, and streamed-persistent workspaces.

use ramforge_core::{memory::MemoryBudget, tensor::TensorData, GgufModel};

use crate::accounting::{estimate_layer_memory, tensor_load_charge_bytes};
use crate::layer::{group_layers, PersistentDescriptors};
use crate::model::{validate_required_tensors, LlamaConfig};
use crate::persistent::should_keep_resident;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionMemoryPlan {
    pub block_count: usize,
    pub resident_persistent_count: usize,
    pub streamed_persistent_count: usize,
    pub persistent_resident_bytes: u64,
    pub persistent_startup_peak_bytes: u64,
    pub largest_layer_index: usize,
    pub largest_layer_tensor_count: usize,
    pub largest_layer_resident_bytes: u64,
    pub largest_layer_load_peak_bytes: u64,
    /// Necessary managed-memory lower bound for startup or loading the largest
    /// layer alongside policy-resident persistent weights.
    pub managed_lower_bound_bytes: u64,
    /// Remaining budget after the necessary layer-streaming lower bound. The
    /// runtime uses the same value as its hard layer-cache byte capacity.
    pub layer_cache_capacity_bytes: u64,
    pub max_complete_cached_layers: usize,
    pub min_layer_resident_bytes: u64,
    /// Necessary but not sufficient: forward activations, KV, logits, and
    /// streamed persistent workspaces are intentionally not included.
    pub layer_streaming_lower_bound_fits: bool,
}

#[derive(Debug, Clone)]
pub struct PlanResult {
    pub file_size: u64,
    pub architecture: Option<String>,
    pub tensor_count: usize,
    pub ram_requested: u64,
    /// Informational cache bound (charged per entry at runtime, not pre-reserved)
    pub cache_capacity: u64,
    /// Kept for CLI compatibility; the runtime reserves scoped temps on
    /// demand instead of a static overhead allocation, so this is 0.
    pub overhead_reserved: u64,
    pub available: u64,
    pub fits_in_ram: bool,
    pub file_backed_needed: u64,
    pub total_tensor_bytes: Option<u64>,
    pub execution_memory: Option<ExecutionMemoryPlan>,
    pub execution_preflight_error: Option<String>,
    pub budget: MemoryBudget,
}

pub fn plan_model(model: &GgufModel, ram_budget_bytes: u64) -> Result<PlanResult, String> {
    let budget = MemoryBudget::new(ram_budget_bytes).map_err(|e| e.to_string())?;

    // Informational cache bound: 80% of budget (50% for tiny budgets).
    // Contents are charged per entry at runtime via insert_budgeted.
    let cache_capacity = (ram_budget_bytes as f64 * 0.8) as u64;
    let cache_capacity = if ram_budget_bytes < 2 * 1024 * 1024 {
        ram_budget_bytes / 2
    } else {
        cache_capacity
            .max(1024 * 1024)
            .min(ram_budget_bytes.saturating_sub(1024 * 1024))
    };

    // No static overhead pre-reservation: scoped `tmp:*` guards charge exact
    // temps at runtime for exactly their lifetimes.
    let overhead_reserved = 0;

    let file_size = model.file_size;
    let fits_in_ram = file_size <= ram_budget_bytes;
    let file_backed_needed = if fits_in_ram {
        0
    } else {
        file_size - ram_budget_bytes
    };

    let architecture = model.info().architecture.clone();
    let tensor_count = model.tensors.len();
    let total_tensor_bytes = model.total_tensor_bytes();
    let available = budget.available_bytes();
    let (execution_memory, execution_preflight_error) =
        match plan_execution_memory(model, ram_budget_bytes) {
            Ok(memory) => (Some(memory), None),
            Err(error) => (None, Some(error)),
        };

    Ok(PlanResult {
        file_size,
        architecture,
        tensor_count,
        ram_requested: ram_budget_bytes,
        cache_capacity,
        overhead_reserved,
        available,
        fits_in_ram,
        file_backed_needed,
        total_tensor_bytes,
        execution_memory,
        execution_preflight_error,
        budget,
    })
}

fn plan_execution_memory(
    model: &GgufModel,
    ram_budget_bytes: u64,
) -> Result<ExecutionMemoryPlan, String> {
    let config = LlamaConfig::from_gguf(model)?;
    validate_required_tensors(model, &config)?;

    let persistent = PersistentDescriptors::from_model(model);
    let mut persistent_resident_bytes = 0u64;
    let mut persistent_startup_peak_bytes = 0u64;
    let mut resident_persistent_count = 0usize;
    let mut streamed_persistent_count = 0usize;

    for descriptor in [&persistent.token_embd, &persistent.output_norm, &persistent.output]
        .into_iter()
        .flatten()
    {
        let file_bytes = descriptor.byte_length.ok_or_else(|| {
            format!(
                "cannot preflight persistent tensor '{}': byte length is unknown",
                descriptor.name
            )
        })?;
        let resident_bytes = TensorData::resident_bytes_for(
            descriptor.ggml_type,
            descriptor.num_elements,
            file_bytes,
        )
        .map_err(|error| {
            format!(
                "cannot preflight persistent tensor '{}': {}",
                descriptor.name, error
            )
        })?;
        if should_keep_resident(resident_bytes, ram_budget_bytes) {
            resident_persistent_count += 1;
            let load_charge = tensor_load_charge_bytes(descriptor.ggml_type, file_bytes)?.max(1);
            let startup_peak = checked_add(
                persistent_resident_bytes,
                load_charge,
                "persistent startup peak",
            )?;
            persistent_startup_peak_bytes = persistent_startup_peak_bytes.max(startup_peak);
            persistent_resident_bytes = checked_add(
                persistent_resident_bytes,
                resident_bytes.max(1),
                "persistent resident bytes",
            )?;
        } else {
            streamed_persistent_count += 1;
        }
    }

    let layers = group_layers(model, config.block_count);
    let mut largest_layer_index = 0usize;
    let mut largest_layer_tensor_count = 0usize;
    let mut largest_layer_resident_bytes = 0u64;
    let mut largest_layer_load_peak_bytes = 0u64;
    let mut layer_resident_sizes = Vec::with_capacity(layers.len());

    for layer in &layers {
        let estimate = estimate_layer_memory(&layer.tensors).map_err(|error| {
            format!("cannot preflight layer {}: {}", layer.layer_idx, error)
        })?;
        layer_resident_sizes.push(estimate.resident_bytes);
        if estimate.load_peak_bytes > largest_layer_load_peak_bytes {
            largest_layer_index = layer.layer_idx;
            largest_layer_tensor_count = layer.tensors.len();
            largest_layer_resident_bytes = estimate.resident_bytes;
            largest_layer_load_peak_bytes = estimate.load_peak_bytes;
        }
    }

    let layer_with_persistents = checked_add(
        persistent_resident_bytes,
        largest_layer_load_peak_bytes,
        "persistent plus largest layer lower bound",
    )?;
    let managed_lower_bound_bytes = persistent_startup_peak_bytes.max(layer_with_persistents);
    let layer_cache_capacity_bytes = ram_budget_bytes.saturating_sub(managed_lower_bound_bytes);
    layer_resident_sizes.sort_unstable();
    let min_layer_resident_bytes = layer_resident_sizes.first().copied().unwrap_or(0);
    let max_complete_cached_layers =
        max_complete_layers_for_capacity(&layer_resident_sizes, layer_cache_capacity_bytes);

    Ok(ExecutionMemoryPlan {
        block_count: config.block_count,
        resident_persistent_count,
        streamed_persistent_count,
        persistent_resident_bytes,
        persistent_startup_peak_bytes,
        largest_layer_index,
        largest_layer_tensor_count,
        largest_layer_resident_bytes,
        largest_layer_load_peak_bytes,
        managed_lower_bound_bytes,
        layer_cache_capacity_bytes,
        max_complete_cached_layers,
        min_layer_resident_bytes,
        layer_streaming_lower_bound_fits: managed_lower_bound_bytes <= ram_budget_bytes,
    })
}

fn max_complete_layers_for_capacity(sorted_layer_bytes: &[u64], capacity: u64) -> usize {
    let mut used = 0u64;
    let mut count = 0usize;
    for &layer_bytes in sorted_layer_bytes {
        let Some(next) = used.checked_add(layer_bytes) else {
            break;
        };
        if next > capacity {
            break;
        }
        used = next;
        count += 1;
    }
    count
}

fn checked_add(left: u64, right: u64, label: &str) -> Result<u64, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("{} overflow: {} + {}", label, left, right))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ramforge_core::model::{GgufModel, TensorDescriptor};
    use ramforge_core::{GgmlType, MetadataValue};
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn dummy_model(file_size: u64) -> GgufModel {
        GgufModel {
            path: PathBuf::from("/tmp/dummy.gguf"),
            file_size,
            version: 3,
            metadata: BTreeMap::new(),
            tensors: vec![],
            alignment: 32,
            data_start_offset: 0,
        }
    }

    fn supported_tiny_model() -> GgufModel {
        let mut metadata = BTreeMap::new();
        for (key, value) in [
            ("llama.vocab_size", 16),
            ("llama.context_length", 64),
            ("llama.embedding_length", 8),
            ("llama.block_count", 1),
            ("llama.feed_forward_length", 16),
            ("llama.attention.head_count", 2),
            ("llama.attention.head_count_kv", 2),
        ] {
            metadata.insert(key.to_string(), MetadataValue::UInt32(value));
        }
        metadata.insert(
            "general.architecture".to_string(),
            MetadataValue::String("llama".to_string()),
        );

        let definitions = [
            ("token_embd.weight", vec![8, 16]),
            ("output_norm.weight", vec![8]),
            ("blk.0.attn_norm.weight", vec![8]),
            ("blk.0.attn_q.weight", vec![8, 8]),
            ("blk.0.attn_k.weight", vec![8, 8]),
            ("blk.0.attn_v.weight", vec![8, 8]),
            ("blk.0.attn_output.weight", vec![8, 8]),
            ("blk.0.ffn_norm.weight", vec![8]),
            ("blk.0.ffn_gate.weight", vec![8, 16]),
            ("blk.0.ffn_up.weight", vec![8, 16]),
            ("blk.0.ffn_down.weight", vec![16, 8]),
        ];
        let tensors = definitions
            .into_iter()
            .scan(0u64, |offset, (name, dimensions)| {
                let num_elements = dimensions.iter().product::<u64>();
                let byte_length = num_elements * 4;
                let descriptor = TensorDescriptor {
                    name: name.to_string(),
                    dimensions,
                    ggml_type: GgmlType::F32,
                    offset: *offset,
                    file_offset: *offset,
                    byte_length: Some(byte_length),
                    num_elements,
                };
                *offset += byte_length;
                Some(descriptor)
            })
            .collect::<Vec<_>>();
        let file_size = tensors
            .iter()
            .filter_map(|tensor| tensor.byte_length)
            .sum();

        GgufModel {
            path: PathBuf::from("/tmp/supported-tiny.gguf"),
            file_size,
            version: 3,
            metadata,
            tensors,
            alignment: 32,
            data_start_offset: 0,
        }
    }

    #[test]
    fn test_plan_fits() {
        let model = dummy_model(1_000_000);
        let plan = plan_model(&model, 8 * 1024 * 1024 * 1024).unwrap();
        assert!(plan.fits_in_ram);
        assert_eq!(plan.file_backed_needed, 0);
        assert!(plan.cache_capacity > 0);
        assert!(plan.cache_capacity <= plan.ram_requested);
        assert!(plan.execution_memory.is_none());
        assert!(plan.execution_preflight_error.is_some());
    }

    #[test]
    fn test_plan_exceeds() {
        let model = dummy_model(10 * 1024 * 1024 * 1024);
        let plan = plan_model(&model, 8 * 1024 * 1024 * 1024).unwrap();
        assert!(!plan.fits_in_ram);
        assert_eq!(plan.file_backed_needed, 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_execution_preflight_matches_runtime_layer_accounting() {
        let model = supported_tiny_model();
        let plan = plan_model(&model, 10_000).unwrap();
        let execution = plan.execution_memory.unwrap();
        assert_eq!(execution.block_count, 1);
        assert_eq!(execution.resident_persistent_count, 2);
        assert_eq!(execution.streamed_persistent_count, 0);
        assert_eq!(execution.persistent_resident_bytes, 544);
        assert_eq!(execution.persistent_startup_peak_bytes, 544);
        assert_eq!(execution.largest_layer_index, 0);
        assert_eq!(execution.largest_layer_tensor_count, 9);
        assert_eq!(execution.largest_layer_resident_bytes, 2624);
        assert_eq!(execution.largest_layer_load_peak_bytes, 2624);
        assert_eq!(execution.managed_lower_bound_bytes, 3168);
        assert_eq!(execution.layer_cache_capacity_bytes, 6832);
        assert_eq!(execution.max_complete_cached_layers, 1);
        assert_eq!(execution.min_layer_resident_bytes, 2624);
        assert!(execution.layer_streaming_lower_bound_fits);
    }

    #[test]
    fn test_planner_reports_maximum_complete_layer_capacity() {
        assert_eq!(max_complete_layers_for_capacity(&[100, 200, 300], 99), 0);
        assert_eq!(max_complete_layers_for_capacity(&[100, 200, 300], 300), 2);
        assert_eq!(max_complete_layers_for_capacity(&[100, 200, 300], 600), 3);
    }

    #[test]
    fn test_execution_preflight_flags_too_small_layer_lower_bound() {
        let model = supported_tiny_model();
        let plan = plan_model(&model, 3_000).unwrap();
        let execution = plan.execution_memory.unwrap();
        assert_eq!(execution.managed_lower_bound_bytes, 3168);
        assert_eq!(execution.layer_cache_capacity_bytes, 0);
        assert_eq!(execution.max_complete_cached_layers, 0);
        assert!(!execution.layer_streaming_lower_bound_fits);
    }
}
