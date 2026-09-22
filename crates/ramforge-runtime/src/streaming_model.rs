//! Streaming LLaMA model – out-of-core layer streaming with quantized support
//!
//! Only persistent weights (token_embd, output_norm, output) are loaded initially.
//! Transformer layers are loaded on demand, one at a time, and released after use.
//! Quantized tensors remain quantized while resident; dequantization happens block-wise during matvec.

use std::borrow::Cow;
use std::sync::Mutex;

use ramforge_core::{
    datasource::GgufDataSource,
    memory::MemoryBudget,
    quant::{
        matvec_q4_0_row_range, matvec_q6_k_row_range, BLOCK_SIZE_Q4_0, BLOCK_SIZE_Q6_K, QK4_0, QK_K,
    },
    tensor::{decode_tensor_to_f32, QuantizedTensor, TensorData},
    types::GgmlType,
};

use rayon::prelude::*;

use crate::accounting::{
    estimate_grouped_layer_memory, estimate_layer_memory, tensor_load_charge_bytes,
    LayerMemoryEstimate,
};
use crate::backend::ComputeBackend;
use crate::kv_cache::KvCache;
use crate::layer::{group_layers, LayerDescriptor, PersistentDescriptors};
use crate::layer_cache::{InsertOutcome, LayerCache};
use crate::layer_read::{build_layer_read_plan, LayerReadPlan, PlannedReadRange};
use crate::model::{validate_required_tensors, LlamaConfig};
use crate::persistent::{row_bytes_for, should_keep_resident, PersistentWeight};
use crate::profile::{ProfileEvent, Profiler};
use crate::residency::ResidencyStats;
use crate::runtime_config::RuntimeConfig;

#[derive(Debug, Clone)]
pub struct StreamingLayerWeights {
    pub attn_norm: TensorData,
    pub attn_q: TensorData,
    pub attn_k: TensorData,
    pub attn_v: TensorData,
    pub attn_output: TensorData,
    pub ffn_norm: TensorData,
    pub ffn_gate: TensorData,
    pub ffn_up: TensorData,
    pub ffn_down: TensorData,
    /// Optional qwen2-style Q/K/V biases (applied after the matvec, before
    /// RoPE insertion into the KV cache). Either all three are present or
    /// none – partial sets are rejected at load time.
    pub attn_q_bias: Option<TensorData>,
    pub attn_k_bias: Option<TensorData>,
    pub attn_v_bias: Option<TensorData>,
}

impl StreamingLayerWeights {
    pub fn total_resident_bytes(&self) -> u64 {
        let bias_bytes = [&self.attn_q_bias, &self.attn_k_bias, &self.attn_v_bias]
            .iter()
            .filter_map(|b| b.as_ref())
            .map(|b| b.resident_bytes() as u64)
            .sum::<u64>();
        self.attn_norm.resident_bytes() as u64
            + self.attn_q.resident_bytes() as u64
            + self.attn_k.resident_bytes() as u64
            + self.attn_v.resident_bytes() as u64
            + self.attn_output.resident_bytes() as u64
            + self.ffn_norm.resident_bytes() as u64
            + self.ffn_gate.resident_bytes() as u64
            + self.ffn_up.resident_bytes() as u64
            + self.ffn_down.resident_bytes() as u64
            + bias_bytes
    }
}

#[derive(Debug)]
pub struct StreamingLlamaModel {
    pub config: LlamaConfig,
    pub token_embd: PersistentWeight,
    pub output_norm: PersistentWeight,
    pub output: Option<PersistentWeight>,
    pub layer_descriptors: Vec<LayerDescriptor>,
    pub persistent_descriptors: PersistentDescriptors,
    pub total_weight_bytes: u64,
    pub quantized_weight_bytes: u64,
    /// True when the model carries qwen2-style Q/K/V bias tensors
    /// (`blk.{i}.attn_{q,k,v}.bias`). Used to size the forward workspace.
    pub attn_bias_present: bool,
    layer_memory_estimates: Vec<LayerMemoryEstimate>,
    grouped_layer_memory_estimates: Vec<LayerMemoryEstimate>,
    layer_read_plans: Vec<LayerReadPlan>,
    layer_cache: Mutex<LayerCache<StreamingLayerWeights>>,
    read_coalescing_enabled: bool,
    grouped_read_buffer_reuse_enabled: bool,
    pub(crate) profiler: Profiler,
    #[cfg(test)]
    test_layer_hidden_hook: Mutex<Option<fn(usize, &[f32])>>,
}

impl StreamingLlamaModel {
    /// Load with the existing runtime behavior used by legacy/manual callers.
    pub fn load(data_source: &GgufDataSource, budget: &mut MemoryBudget) -> Result<Self, String> {
        Self::load_internal(data_source, budget, None)
    }

    /// Load with concrete decisions that have already crossed the planning
    /// boundary. Runtime feasibility checks remain authoritative at the point
    /// where allocations are attempted.
    pub fn load_with_runtime_config(
        data_source: &GgufDataSource,
        budget: &mut MemoryBudget,
        runtime_config: &RuntimeConfig,
    ) -> Result<Self, String> {
        runtime_config
            .validate()
            .map_err(|error| format!("invalid runtime configuration: {error}"))?;
        if budget.total_bytes() != runtime_config.ram_budget_bytes {
            return Err(format!(
                "runtime configuration RAM budget {} does not match MemoryBudget {}",
                runtime_config.ram_budget_bytes,
                budget.total_bytes()
            ));
        }
        Self::load_internal(data_source, budget, Some(runtime_config))
    }

    /// Load persistent weights only if they fit comfortably, otherwise stream.
    fn load_internal(
        data_source: &GgufDataSource,
        budget: &mut MemoryBudget,
        runtime_config: Option<&RuntimeConfig>,
    ) -> Result<Self, String> {
        let gguf_model = data_source.model();
        let config = LlamaConfig::from_gguf(gguf_model)?;
        let profiler = Profiler::default();

        validate_required_tensors(gguf_model, &config)?;

        let total_weight_bytes = gguf_model
            .tensors
            .iter()
            .filter_map(|t| t.byte_length)
            .sum();

        let quantized_weight_bytes = gguf_model
            .tensors
            .iter()
            .filter(|t| {
                matches!(
                    t.ggml_type,
                    GgmlType::Q4_0
                        | GgmlType::Q8_0
                        | GgmlType::Q4_K
                        | GgmlType::Q5_K
                        | GgmlType::Q6_K
                        | GgmlType::Q2_K
                        | GgmlType::Q3_K
                        | GgmlType::Q8_K
                )
            })
            .filter_map(|t| t.byte_length)
            .sum();
        let layer_descriptors = group_layers(gguf_model, config.block_count);
        let layer_read_plans = layer_descriptors
            .iter()
            .map(|layer| {
                build_layer_read_plan(&layer.tensors).map_err(|error| {
                    format!("failed to plan layer {} reads: {}", layer.layer_idx, error)
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let layer_memory_estimates = layer_descriptors
            .iter()
            .map(|layer| {
                estimate_layer_memory(&layer.tensors).map_err(|error| {
                    format!(
                        "failed to estimate layer {} memory: {}",
                        layer.layer_idx, error
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let grouped_layer_memory_estimates = layer_descriptors
            .iter()
            .zip(&layer_read_plans)
            .map(|(layer, read_plan)| {
                estimate_grouped_layer_memory(&layer.tensors, read_plan).map_err(|error| {
                    format!(
                        "failed to estimate grouped layer {} memory: {}",
                        layer.layer_idx, error
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let max_layer_load_peak = layer_memory_estimates
            .iter()
            .map(|estimate| estimate.load_peak_bytes)
            .max()
            .unwrap_or(0);
        let persistent_descriptors = PersistentDescriptors::from_model(gguf_model);

        // Persistent startup is transactional. Each resident tensor first
        // establishes its conservative load charge, then reads/decodes and
        // atomically settles to exact residency. If a later persistent fails,
        // every earlier weight charge created by this load is rolled back.
        let mut persistent_allocations = Vec::new();
        let persistent_result = (|| -> Result<
            (PersistentWeight, PersistentWeight, Option<PersistentWeight>),
            String,
        > {
            let (token_embd, token_charge) =
                load_persistent_weight(data_source, "token_embd.weight", budget, &profiler)?;
            if let Some(name) = token_charge {
                persistent_allocations.push(name);
            }

            let (output_norm, norm_charge) =
                load_persistent_weight(data_source, "output_norm.weight", budget, &profiler)?;
            if let Some(name) = norm_charge {
                persistent_allocations.push(name);
            }

            let output = if gguf_model.tensors.iter().any(|t| t.name == "output.weight") {
                let (weight, output_charge) =
                    load_persistent_weight(data_source, "output.weight", budget, &profiler)?;
                if let Some(name) = output_charge {
                    persistent_allocations.push(name);
                }
                Some(weight)
            } else {
                None
            };

            Ok((token_embd, output_norm, output))
        })();

        let (token_embd, output_norm, output) = match persistent_result {
            Ok(weights) => weights,
            Err(error) => {
                for name in persistent_allocations.iter().rev() {
                    let _ = budget.release(name);
                }
                return Err(error);
            }
        };

        // Match the planner's necessary lower bound: retain neither startup
        // peak nor largest-layer headroom as cache capacity. Additional
        // mandatory workspaces dynamically evict cache entries.
        let layer_lower_bound = budget
            .used_bytes()
            .checked_add(max_layer_load_peak)
            .ok_or_else(|| "layer cache lower-bound overflow".to_string())?;
        let managed_lower_bound = budget.peak_used_bytes().max(layer_lower_bound);
        let maximum_layer_cache_capacity = budget.total_bytes().saturating_sub(managed_lower_bound);
        let layer_cache_capacity = runtime_config
            .map(|config| {
                if config.layer_cache_enabled {
                    config.layer_cache_capacity_bytes
                } else {
                    0
                }
            })
            .unwrap_or(maximum_layer_cache_capacity);
        if layer_cache_capacity > maximum_layer_cache_capacity {
            drop(output);
            drop(output_norm);
            drop(token_embd);
            for name in persistent_allocations.iter().rev() {
                let _ = budget.release(name);
            }
            return Err(format!(
                "configured layer cache capacity {} exceeds runtime maximum {}",
                layer_cache_capacity, maximum_layer_cache_capacity
            ));
        }
        let read_coalescing_enabled = runtime_config
            .map(|config| config.read_coalescing_enabled)
            .unwrap_or(RuntimeConfig::CURRENT_READ_COALESCING_ENABLED);
        let grouped_read_buffer_reuse_enabled = runtime_config
            .map(|config| config.grouped_read_buffer_reuse_enabled)
            .unwrap_or(RuntimeConfig::CURRENT_GROUPED_READ_BUFFER_REUSE_ENABLED);
        let attn_bias_present = gguf_model
            .tensors
            .iter()
            .any(|t| t.name.ends_with(".attn_q.bias"));

        Ok(Self {
            config,
            token_embd,
            output_norm,
            output,
            layer_descriptors,
            persistent_descriptors,
            total_weight_bytes,
            quantized_weight_bytes,
            attn_bias_present,
            layer_memory_estimates,
            grouped_layer_memory_estimates,
            layer_read_plans,
            layer_cache: Mutex::new(LayerCache::new(layer_cache_capacity)),
            read_coalescing_enabled,
            grouped_read_buffer_reuse_enabled,
            profiler,
            #[cfg(test)]
            test_layer_hidden_hook: Mutex::new(None),
        })
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_single_streaming_with_layer_hook<B: ComputeBackend>(
        &self,
        token_id: u32,
        pos: usize,
        kv_cache: &mut KvCache,
        backend: &B,
        data_source: &GgufDataSource,
        budget: &mut MemoryBudget,
        stats: &mut ResidencyStats,
        final_hidden: &mut [f32],
        layer_hook: fn(usize, &[f32]),
    ) -> Result<(), String> {
        *self
            .test_layer_hidden_hook
            .lock()
            .map_err(|_| "test layer hidden hook lock poisoned".to_string())? = Some(layer_hook);
        let result = self.forward_single_streaming(
            token_id,
            pos,
            kv_cache,
            backend,
            data_source,
            budget,
            stats,
            final_hidden,
        );
        if let Ok(mut hook) = self.test_layer_hidden_hook.lock() {
            *hook = None;
        }
        result
    }

    #[cfg(test)]
    fn emit_test_layer_hidden(&self, layer_idx: usize, hidden: &[f32]) {
        let hook = self
            .test_layer_hidden_hook
            .lock()
            .ok()
            .and_then(|hook| *hook);
        if let Some(hook) = hook {
            hook(layer_idx, hidden);
        }
    }

    pub(crate) fn set_profiling(&self, enabled: bool) {
        self.profiler.set_enabled(enabled);
    }

    pub(crate) fn reset_profile(&self) {
        self.profiler.reset();
    }

    pub fn layer_cache_capacity_bytes(&self) -> u64 {
        self.layer_cache
            .lock()
            .map(|cache| cache.capacity_bytes())
            .unwrap_or(0)
    }

    pub fn read_coalescing_enabled(&self) -> bool {
        self.read_coalescing_enabled
    }

    pub fn grouped_read_buffer_reuse_enabled(&self) -> bool {
        self.grouped_read_buffer_reuse_enabled
    }

    pub fn clear_layer_cache(&self, budget: &mut MemoryBudget) -> Result<(), String> {
        let mut cache = self
            .layer_cache
            .lock()
            .map_err(|_| "layer cache lock poisoned".to_string())?;
        cache.clear(budget)?;
        self.profiler
            .record_cache_state(cache.entry_count(), cache.used_bytes());
        Ok(())
    }

    pub(crate) fn ensure_layer_cache_headroom(
        &self,
        budget: &mut MemoryBudget,
        required_bytes: u64,
    ) -> Result<(), String> {
        let mut cache = self
            .layer_cache
            .lock()
            .map_err(|_| "layer cache lock poisoned".to_string())?;
        let evictions = cache.evict_until_available(budget, required_bytes)?;
        self.profiler.record_cache_evictions(evictions);
        self.profiler
            .record_cache_state(cache.entry_count(), cache.used_bytes());
        Ok(())
    }

    /// Load a single layer on demand, accounting actual quantized resident size
    ///
    /// Memory integrity (M7.2): each tensor is charged BEFORE reading. F32
    /// reads directly into final storage (1x file bytes); F16/BF16 retain their
    /// raw+decoded 3x transient; quantized data remains compact at 1x. After
    /// construction the charge is atomically settled to exact residency. On
    /// any failure, all charges made for this layer are released.
    pub fn load_layer(
        &self,
        layer_idx: usize,
        data_source: &GgufDataSource,
        budget: &mut MemoryBudget,
        stats: &mut ResidencyStats,
    ) -> Result<StreamingLayerWeights, String> {
        if layer_idx >= self.layer_descriptors.len() {
            return Err(format!("layer {} out of bounds", layer_idx));
        }

        let layer_desc = &self.layer_descriptors[layer_idx];
        let mut loaded: Vec<(String, TensorData)> = Vec::new();
        let mut total_layer_bytes = 0u64;

        let read_plan = &self.layer_read_plans[layer_idx];
        let use_grouped_reads = self.read_coalescing_enabled
            && read_plan.ranges.iter().any(|range| range.tensors.len() > 1)
            && budget.can_allocate(self.grouped_layer_memory_estimates[layer_idx].load_peak_bytes);
        let reusable_buffer_bytes = read_plan.reusable_group_buffer_bytes();
        let reusable_peak = reusable_buffer_bytes.and_then(|scratch_bytes| {
            scratch_bytes.checked_add(self.grouped_layer_memory_estimates[layer_idx].resident_bytes)
        });
        let use_reusable_buffer = self.grouped_read_buffer_reuse_enabled
            && use_grouped_reads
            && reusable_peak.is_some_and(|required| budget.can_allocate(required));
        let result = (|budget: &mut MemoryBudget| -> Result<(), String> {
            if !use_grouped_reads {
                for tensor_desc in &layer_desc.tensors {
                    let name = &tensor_desc.name;
                    let file_bytes = tensor_desc.byte_length.unwrap_or(0);
                    let charge =
                        tensor_load_charge_bytes(tensor_desc.ggml_type, file_bytes)?.max(1);
                    let alloc_name = format!("layer:{}:{}", layer_idx, name);
                    budget
                        .allocate(alloc_name.clone(), charge)
                        .map_err(|error| {
                            format!(
                                "RAM budget too small for layer {} tensor '{}': {}",
                                layer_idx, name, error
                            )
                        })?;
                    let tensor_data = load_tensor_data(data_source, tensor_desc, &self.profiler)?;
                    let resident = tensor_data.resident_bytes() as u64;
                    total_layer_bytes = total_layer_bytes
                        .checked_add(resident)
                        .ok_or_else(|| "layer resident byte total overflow".to_string())?;
                    if resident.max(1) != charge {
                        budget
                            .resize(&alloc_name, resident.max(1))
                            .map_err(|error| {
                                format!(
                                    "RAM budget error settling layer {} tensor '{}': {}",
                                    layer_idx, name, error
                                )
                            })?;
                    }
                    loaded.push((name.clone(), tensor_data));
                }
                return Ok(());
            }

            if use_reusable_buffer {
                for range in &read_plan.ranges {
                    for tensor in &range.tensors {
                        let descriptor = &layer_desc.tensors[tensor.descriptor_index];
                        let file_bytes = descriptor.byte_length.ok_or_else(|| {
                            format!("tensor '{}' byte length is unknown", descriptor.name)
                        })?;
                        let resident = TensorData::resident_bytes_for(
                            descriptor.ggml_type,
                            descriptor.num_elements,
                            file_bytes,
                        )
                        .map_err(|error| {
                            format!("failed to size tensor '{}': {}", descriptor.name, error)
                        })?
                        .max(1);
                        budget
                            .allocate(
                                format!("layer:{}:{}", layer_idx, descriptor.name),
                                resident,
                            )
                            .map_err(|error| {
                                format!(
                                    "RAM budget too small for reusable grouped layer {} tensor '{}': {}",
                                    layer_idx, descriptor.name, error
                                )
                            })?;
                    }
                }

                let scratch_bytes = reusable_buffer_bytes.unwrap_or(0);
                let scratch_capacity = usize::try_from(scratch_bytes)
                    .map_err(|_| "reusable grouped read buffer is too large".to_string())?;
                let temp_name = format!("tmp:layer_read:{}:reused", layer_idx);
                budget.with_temp(&temp_name, scratch_bytes, |budget| {
                    let mut buffer = Vec::with_capacity(scratch_capacity);
                    for (range_index, range) in read_plan.ranges.iter().enumerate() {
                        read_and_decode_grouped_range(
                            data_source,
                            layer_desc,
                            range,
                            layer_idx,
                            range_index,
                            budget,
                            &self.profiler,
                            &mut buffer,
                            &mut loaded,
                            &mut total_layer_bytes,
                        )?;
                    }
                    Ok::<(), String>(())
                })?;
                return Ok(());
            }

            for (range_index, range) in read_plan.ranges.iter().enumerate() {
                if range.tensors.len() > 1 {
                    for tensor in &range.tensors {
                        let descriptor = &layer_desc.tensors[tensor.descriptor_index];
                        let file_bytes = descriptor.byte_length.ok_or_else(|| {
                            format!("tensor '{}' byte length is unknown", descriptor.name)
                        })?;
                        let resident = TensorData::resident_bytes_for(
                            descriptor.ggml_type,
                            descriptor.num_elements,
                            file_bytes,
                        )
                        .map_err(|error| {
                            format!("failed to size tensor '{}': {}", descriptor.name, error)
                        })?
                        .max(1);
                        budget
                            .allocate(format!("layer:{}:{}", layer_idx, descriptor.name), resident)
                            .map_err(|error| {
                                format!(
                                    "RAM budget too small for grouped layer {} tensor '{}': {}",
                                    layer_idx, descriptor.name, error
                                )
                            })?;
                    }

                    let temp_name = format!("tmp:layer_read:{}:{}", layer_idx, range_index);
                    budget.with_temp(&temp_name, range.byte_length, |budget| {
                        let mut buffer = Vec::new();
                        read_and_decode_grouped_range(
                            data_source,
                            layer_desc,
                            range,
                            layer_idx,
                            range_index,
                            budget,
                            &self.profiler,
                            &mut buffer,
                            &mut loaded,
                            &mut total_layer_bytes,
                        )
                    })?;
                    continue;
                }

                for tensor in &range.tensors {
                    let tensor_desc = &layer_desc.tensors[tensor.descriptor_index];
                    let name = &tensor_desc.name;
                    let file_bytes = tensor_desc.byte_length.unwrap_or(0);
                    // Peak owned representation during an individual load.
                    let charge =
                        tensor_load_charge_bytes(tensor_desc.ggml_type, file_bytes)?.max(1);
                    let alloc_name = format!("layer:{}:{}", layer_idx, name);
                    budget.allocate(alloc_name.clone(), charge).map_err(|e| {
                        format!(
                            "RAM budget too small for layer {} tensor '{}': {}",
                            layer_idx, name, e
                        )
                    })?;

                    let tensor_data = load_tensor_data(data_source, tensor_desc, &self.profiler)?;
                    let resident = tensor_data.resident_bytes() as u64;
                    total_layer_bytes = total_layer_bytes
                        .checked_add(resident)
                        .ok_or_else(|| "layer resident byte total overflow".to_string())?;

                    // Atomically settle the charge to exact resident size.
                    if resident.max(1) != charge {
                        budget.resize(&alloc_name, resident.max(1)).map_err(|e| {
                            format!(
                                "RAM budget error settling layer {} tensor '{}': {}",
                                layer_idx, name, e
                            )
                        })?;
                    }
                    loaded.push((name.clone(), tensor_data));
                }
            }
            Ok(())
        })(budget);

        if let Err(e) = result {
            // Budget-integrity: never leave charges behind for a layer that
            // failed to load.
            self.release_layer(layer_idx, budget, stats);
            return Err(e);
        }

        stats.on_layer_load(total_layer_bytes, budget.used_bytes());

        let mut map = std::collections::HashMap::new();
        for (name, data) in loaded {
            map.insert(name, data);
        }

        // Extraction (incl. Q/K/V bias validation): on ANY error here the
        // layer's budget charges must be released, not leaked.
        let extract_result = (|map: &mut std::collections::HashMap<String, TensorData>| {
            let mut get = |suffix: &str| -> Result<TensorData, String> {
                let full = format!("blk.{}.{}", layer_idx, suffix);
                map.remove(&full).ok_or_else(|| {
                    format!("missing tensor '{}' in loaded layer {}", full, layer_idx)
                })
            };

            let weights = StreamingLayerWeights {
                attn_norm: get("attn_norm.weight")?,
                attn_q: get("attn_q.weight")?,
                attn_k: get("attn_k.weight")?,
                attn_v: get("attn_v.weight")?,
                attn_output: get("attn_output.weight")?,
                ffn_norm: get("ffn_norm.weight")?,
                ffn_gate: get("ffn_gate.weight")?,
                ffn_up: get("ffn_up.weight")?,
                ffn_down: get("ffn_down.weight")?,
                attn_q_bias: map.remove(&format!("blk.{}.attn_q.bias", layer_idx)),
                attn_k_bias: map.remove(&format!("blk.{}.attn_k.bias", layer_idx)),
                attn_v_bias: map.remove(&format!("blk.{}.attn_v.bias", layer_idx)),
            };
            validate_qkv_bias(&weights, layer_idx, &self.config)?;
            Ok(weights)
        })(&mut map);

        match extract_result {
            Ok(w) => Ok(w),
            Err(e) => {
                self.release_layer(layer_idx, budget, stats);
                Err(e)
            }
        }
    }

    pub fn release_layer(
        &self,
        layer_idx: usize,
        budget: &mut MemoryBudget,
        stats: &mut ResidencyStats,
    ) {
        let prefix = format!("layer:{}:", layer_idx);
        let names: Vec<String> = budget
            .allocations()
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .cloned()
            .collect();
        for name in names {
            let _ = budget.release(&name);
        }
        stats.on_layer_release(budget.used_bytes());
    }

    fn embedding_workspace_bytes(&self, n_embd: usize) -> Result<u64, String> {
        let decoded_bytes = (n_embd as u64)
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| "embedding workspace size overflow".to_string())?;
        match &self.token_embd {
            PersistentWeight::Resident(_) => Ok(decoded_bytes),
            PersistentWeight::Streamed(descriptor) if descriptor.ggml_type == GgmlType::F32 => {
                Ok(decoded_bytes)
            }
            PersistentWeight::Streamed(descriptor) => {
                let raw_bytes = row_bytes_for(descriptor, n_embd)? as u64;
                raw_bytes
                    .checked_add(decoded_bytes)
                    .ok_or_else(|| "embedding workspace size overflow".to_string())
            }
        }
    }

    /// Forward pass for one token with out-of-core layer streaming.
    ///
    /// M7.1 memory contract:
    /// - one scoped `tmp:forward` reservation covers every transient
    ///   activation buffer allocated inside this call (worst-case size);
    /// - the caller owns `final_hidden` and must keep its separate charge live
    ///   for the full lifetime of that buffer (the inference engine uses
    ///   `tmp:hidden` around the complete generation loop);
    /// - each layer's weights carry their own `layer:{i}:*` charges inside
    ///   the same scope and are released before the next layer loads;
    /// - attention reads the KV history in place – no prefix copies;
    /// - per-token K/V is appended once and read back through the cache.
    ///
    /// Matrix layout: all weights use the explicit ggml convention
    /// `shape = [in, out]`; F32 matvecs go through the SIMD/threaded
    /// backend, quantized matvecs run block-wise on compact bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_single_streaming<B: ComputeBackend>(
        &self,
        token_id: u32,
        pos: usize,
        kv_cache: &mut KvCache,
        backend: &B,
        data_source: &GgufDataSource,
        budget: &mut MemoryBudget,
        stats: &mut ResidencyStats,
        final_hidden: &mut [f32],
    ) -> Result<(), String> {
        let cfg = &self.config;
        let n_embd = cfg.embedding_length;
        if final_hidden.len() != n_embd {
            return Err(format!(
                "final hidden buffer size mismatch: expected {}, got {}",
                n_embd,
                final_hidden.len()
            ));
        }
        let n_heads = cfg.head_count;
        let n_kv_heads = cfg.head_count_kv;
        let head_dim = cfg.head_dim;
        let ffn_dim = cfg.feed_forward_length;
        let kv_dim = n_kv_heads * head_dim;
        let q_dim = n_heads * head_dim;
        let seq = pos + 1; // history + current token (attention work)

        // Worst-case transient floats for this forward pass:
        //   hidden + tmp + 2 norm decode/copy fallbacks: 4 * n_embd
        //   q_tmp + attention output:                   2 * q_dim
        //   k_tmp + v_tmp:                              2 * kv_dim
        //   attn_proj + ffn_out + output norm fallback: 3 * n_embd
        //   gate + up + gate_silu + gate_up:            4 * ffn_dim
        //   attention scores (per head):                n_heads * seq
        //   decoded qwen2 Q/K/V bias fallbacks (per layer, if present)
        let bias_floats = if self.attn_bias_present {
            q_dim + 2 * kv_dim
        } else {
            0
        };
        let act_floats =
            7 * n_embd + 2 * q_dim + 2 * kv_dim + 4 * ffn_dim + n_heads * seq + bias_floats;
        let act_bytes = (act_floats * 4) as u64;

        self.ensure_layer_cache_headroom(budget, act_bytes)?;
        budget.with_temp("tmp:forward", act_bytes, |budget| {
            // Embedding lookup (streams one row if non-resident; charges its
            // own `tmp:embd_row` inside this scope).
            self.ensure_layer_cache_headroom(budget, self.embedding_workspace_bytes(n_embd)?)?;
            let mut hidden = self.token_embd.get_embedding(
                token_id as usize,
                n_embd,
                data_source,
                budget,
                stats,
            )?;

            let allocation_started = self.profiler.start();
            let mut tmp = vec![0.0f32; n_embd];
            let mut q_tmp = vec![0.0f32; q_dim];
            let mut k_tmp = vec![0.0f32; kv_dim];
            let mut v_tmp = vec![0.0f32; kv_dim];
            let mut attn_proj = vec![0.0f32; n_embd];
            let mut gate = vec![0.0f32; ffn_dim];
            let mut up = vec![0.0f32; ffn_dim];
            let mut gate_silu = vec![0.0f32; ffn_dim];
            let mut gate_up = vec![0.0f32; ffn_dim];
            let mut ffn_out = vec![0.0f32; n_embd];
            // Stage K/V for every layer so they can be appended to the KV
            // cache only AFTER every layer has consumed the prior seq_len
            // history. Writing a layer's own K/V into the cache while later
            // layers still need to run would leak future tokens across
            // layers: later layers' attention would "see" an earlier layer's
            // freshly-written K/V as history even though it belongs to the
            // SAME token position.
            let mut staged_k: Vec<Vec<f32>> = Vec::with_capacity(cfg.block_count);
            let mut staged_v: Vec<Vec<f32>> = Vec::with_capacity(cfg.block_count);
            self.profiler
                .record_since(ProfileEvent::Allocation, allocation_started);

            for layer_idx in 0..cfg.block_count {
                let compute_started = self.profiler.start();
                let cached_result = {
                    let mut cache = self
                        .layer_cache
                        .lock()
                        .map_err(|_| "layer cache lock poisoned".to_string())?;
                    let result = cache.with_entry(layer_idx, |layer| {
                        Self::forward_layer(
                            layer,
                            layer_idx,
                            pos,
                            kv_cache,
                            backend,
                            cfg,
                            &self.profiler,
                            &mut hidden,
                            &mut tmp,
                            &mut q_tmp,
                            &mut k_tmp,
                            &mut v_tmp,
                            &mut attn_proj,
                            &mut gate,
                            &mut up,
                            &mut gate_silu,
                            &mut gate_up,
                            &mut ffn_out,
                        )
                    });
                    if result.is_some() {
                        self.profiler
                            .record_cache_state(cache.entry_count(), cache.used_bytes());
                    }
                    result
                };
                if let Some(result) = cached_result {
                    self.profiler.record_cache_hit();
                    self.profiler
                        .record_since(ProfileEvent::LayerCompute, compute_started);
                    result?;
                    // Exact test-only boundary: forward_layer has completed
                    // the layer residual/FFN output, and the next layer has
                    // not yet consumed `hidden`.
                    #[cfg(test)]
                    self.emit_test_layer_hidden(layer_idx, &hidden);
                    staged_k.push(k_tmp.clone());
                    staged_v.push(v_tmp.clone());
                    continue;
                }

                self.profiler.record_cache_miss();
                let required_load_bytes = self.layer_memory_estimates[layer_idx].load_peak_bytes;
                self.ensure_layer_cache_headroom(budget, required_load_bytes)?;
                let load_started = self.profiler.start();
                let layer_result = self.load_layer(layer_idx, data_source, budget, stats);
                self.profiler
                    .record_since(ProfileEvent::LayerLoad, load_started);
                let layer = layer_result?;
                self.profiler.record_layer_load();

                let compute_started = self.profiler.start();
                let layer_result = Self::forward_layer(
                    &layer,
                    layer_idx,
                    pos,
                    kv_cache,
                    backend,
                    cfg,
                    &self.profiler,
                    &mut hidden,
                    &mut tmp,
                    &mut q_tmp,
                    &mut k_tmp,
                    &mut v_tmp,
                    &mut attn_proj,
                    &mut gate,
                    &mut up,
                    &mut gate_silu,
                    &mut gate_up,
                    &mut ffn_out,
                );
                self.profiler
                    .record_since(ProfileEvent::LayerCompute, compute_started);
                if let Err(error) = layer_result {
                    let release_started = self.profiler.start();
                    self.release_layer(layer_idx, budget, stats);
                    self.profiler
                        .record_since(ProfileEvent::LayerRelease, release_started);
                    self.profiler.record_layer_release();
                    return Err(error);
                }
                // Exact test-only boundary: forward_layer has completed the
                // layer residual/FFN output before the next layer starts.
                #[cfg(test)]
                self.emit_test_layer_hidden(layer_idx, &hidden);
                staged_k.push(k_tmp.clone());
                staged_v.push(v_tmp.clone());

                let layer_bytes = layer.total_resident_bytes();
                let insert_result = {
                    let mut cache = match self.layer_cache.lock() {
                        Ok(cache) => cache,
                        Err(_) => {
                            self.release_layer(layer_idx, budget, stats);
                            return Err("layer cache lock poisoned".to_string());
                        }
                    };
                    let result = cache.insert_loaded(layer_idx, layer, layer_bytes, budget);
                    match &result {
                        Ok(InsertOutcome::Cached { evictions }) => {
                            self.profiler.record_cache_evictions(*evictions)
                        }
                        Ok(InsertOutcome::Skipped { evictions, .. }) => {
                            self.profiler.record_cache_evictions(*evictions)
                        }
                        Err(_) => {}
                    }
                    self.profiler
                        .record_cache_state(cache.entry_count(), cache.used_bytes());
                    result
                };
                match insert_result {
                    Ok(InsertOutcome::Cached { .. }) => {
                        stats.on_layer_cached(budget.used_bytes());
                    }
                    Ok(InsertOutcome::Skipped { value: layer, .. }) => {
                        let release_started = self.profiler.start();
                        self.release_layer(layer_idx, budget, stats);
                        drop(layer);
                        self.profiler
                            .record_since(ProfileEvent::LayerRelease, release_started);
                        self.profiler.record_layer_release();
                    }
                    Err(error) => {
                        self.release_layer(layer_idx, budget, stats);
                        return Err(error);
                    }
                }
            }

            // Commit all per-layer K/V for the current token atomically:
            // this avoids leaking a shallow layer's K/V into a deeper
            // layer's attention on the same forward pass.
            for (layer_idx, (k, v)) in staged_k.iter().zip(staged_v.iter()).enumerate() {
                kv_cache.append(layer_idx, k, v)?;
            }

            kv_cache.increment_seq_len();

            let dequant_started = self.profiler.start();
            let output_norm_f32 = persistent_f32_view(&self.output_norm, data_source)?;
            self.profiler
                .record_since(ProfileEvent::Dequantization, dequant_started);
            backend.rmsnorm(&hidden, output_norm_f32.as_ref(), cfg.rms_eps, final_hidden);

            Ok(())
        })
    }

    /// One transformer block over pre-allocated scratch buffers.
    /// All matvecs honor the explicit ggml layout of `layer` tensors.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer<B: ComputeBackend>(
        layer: &StreamingLayerWeights,
        layer_idx: usize,
        pos: usize,
        kv_cache: &mut KvCache,
        backend: &B,
        cfg: &LlamaConfig,
        profiler: &Profiler,
        hidden: &mut [f32],
        tmp: &mut [f32],
        q_tmp: &mut [f32],
        k_tmp: &mut [f32],
        v_tmp: &mut [f32],
        attn_proj: &mut [f32],
        gate: &mut [f32],
        up: &mut [f32],
        gate_silu: &mut [f32],
        gate_up: &mut [f32],
        ffn_out: &mut [f32],
    ) -> Result<(), String> {
        let n_embd = cfg.embedding_length;
        let n_heads = cfg.head_count;
        let n_kv_heads = cfg.head_count_kv;
        let head_dim = cfg.head_dim;

        // attn_norm
        let dequant_started = profiler.start();
        let attn_norm_f32 = tensor_f32_view(&layer.attn_norm)
            .map_err(|e| format!("failed to decode attn_norm of layer {}: {}", layer_idx, e))?;
        profiler.record_since(ProfileEvent::Dequantization, dequant_started);
        backend.rmsnorm(hidden, attn_norm_f32.as_ref(), cfg.rms_eps, tmp);

        matvec_backend(backend, profiler, &layer.attn_q, tmp, q_tmp)?;
        matvec_backend(backend, profiler, &layer.attn_k, tmp, k_tmp)?;
        matvec_backend(backend, profiler, &layer.attn_v, tmp, v_tmp)?;

        // qwen2-style Q/K/V biases: added to the fresh projections BEFORE
        // RoPE and KV-cache insertion (matches llama.cpp/HF qwen2 ordering).
        // Partial sets are rejected at layer load; this match is defensive.
        match (&layer.attn_q_bias, &layer.attn_k_bias, &layer.attn_v_bias) {
            (None, None, None) => {}
            (Some(bq), Some(bk), Some(bv)) => {
                let dequant_started = profiler.start();
                let bq = tensor_f32_view(bq).map_err(|e| {
                    format!("failed to decode attn_q.bias of layer {}: {}", layer_idx, e)
                })?;
                let bk = tensor_f32_view(bk).map_err(|e| {
                    format!("failed to decode attn_k.bias of layer {}: {}", layer_idx, e)
                })?;
                let bv = tensor_f32_view(bv).map_err(|e| {
                    format!("failed to decode attn_v.bias of layer {}: {}", layer_idx, e)
                })?;
                profiler.record_since(ProfileEvent::Dequantization, dequant_started);
                for (x, b) in q_tmp.iter_mut().zip(bq.iter()) {
                    *x += *b;
                }
                for (x, b) in k_tmp.iter_mut().zip(bk.iter()) {
                    *x += *b;
                }
                for (x, b) in v_tmp.iter_mut().zip(bv.iter()) {
                    *x += *b;
                }
            }
            _ => {
                return Err(format!(
                    "internal error: incomplete Q/K/V bias set survived load of layer {}",
                    layer_idx
                ))
            }
        }

        crate::ops::apply_rope(
            q_tmp,
            k_tmp,
            pos,
            head_dim,
            n_heads,
            n_kv_heads,
            cfg.rope_freq_base,
        );

        // History holds `seq_len` previous tokens (BEFORE this token's K/V
        // are appended). The current token's K/V is passed separately to
        // attention and MUST be appended to the cache for ALL layers
        // simultaneously after the whole per-token block loop finishes —
        // appending mid-loop would leak early layers' new-K/V into later
        // layers' attention, which is how information bleeds across the
        // wrong depth.
        let hist_len = kv_cache.seq_len();
        let attn_out = crate::ops::attention(
            q_tmp,
            kv_cache.get_k(layer_idx),
            kv_cache.get_v(layer_idx),
            k_tmp,
            v_tmp,
            hist_len,
            n_heads,
            n_kv_heads,
            head_dim,
        );

        matvec_backend(backend, profiler, &layer.attn_output, &attn_out, attn_proj)?;
        for i in 0..n_embd {
            hidden[i] += attn_proj[i];
        }

        // ffn_norm
        let dequant_started = profiler.start();
        let ffn_norm_f32 = tensor_f32_view(&layer.ffn_norm)
            .map_err(|e| format!("failed to decode ffn_norm of layer {}: {}", layer_idx, e))?;
        profiler.record_since(ProfileEvent::Dequantization, dequant_started);
        backend.rmsnorm(hidden, ffn_norm_f32.as_ref(), cfg.rms_eps, tmp);

        matvec_backend(backend, profiler, &layer.ffn_gate, tmp, gate)?;
        matvec_backend(backend, profiler, &layer.ffn_up, tmp, up)?;

        backend.silu(gate, gate_silu);
        backend.mul(gate_silu, up, gate_up);

        matvec_backend(backend, profiler, &layer.ffn_down, gate_up, ffn_out)?;
        for i in 0..n_embd {
            hidden[i] += ffn_out[i];
        }

        Ok(())
    }

    fn logits_workspace_min_bytes(&self) -> Result<u64, String> {
        let weight = self.output.as_ref().unwrap_or(&self.token_embd);
        let PersistentWeight::Streamed(descriptor) = weight else {
            return Ok(0);
        };
        let row_bytes = row_bytes_for(descriptor, self.config.embedding_length)? as u64;
        if descriptor.ggml_type == GgmlType::F32 {
            Ok(row_bytes)
        } else {
            row_bytes
                .checked_add((self.config.embedding_length * 4) as u64)
                .ok_or_else(|| "logits workspace size overflow".to_string())
        }
    }

    /// Output projection into the caller-provided logits buffer (single
    /// allocation owned by the engine – no duplicate vocab-sized copies).
    ///
    /// Uses `output.weight` when present, otherwise the tied embedding
    /// matrix. Resident F32 goes through the SIMD/threaded backend;
    /// quantized stays compact (block-wise matvec); streamed weights run
    /// the budget-charged chunked row pass.
    pub fn compute_logits<B: ComputeBackend>(
        &self,
        hidden: &[f32],
        backend: &B,
        data_source: &GgufDataSource,
        budget: &mut MemoryBudget,
        logits_out: &mut [f32],
    ) -> Result<(), String> {
        let vocab_size = self.config.vocab_size;
        if logits_out.len() != vocab_size {
            return Err(format!(
                "logits buffer size mismatch: expected {}, got {}",
                vocab_size,
                logits_out.len()
            ));
        }

        self.ensure_layer_cache_headroom(budget, self.logits_workspace_min_bytes()?)?;
        let weight = self.output.as_ref().unwrap_or(&self.token_embd);
        match weight {
            PersistentWeight::Resident(td) => {
                matvec_backend(backend, &self.profiler, td, hidden, logits_out)
            }
            PersistentWeight::Streamed(desc) => {
                let started = self.profiler.start();
                let result = weight.compute_logits_into(hidden, data_source, budget, logits_out);
                let event = if desc.ggml_type == GgmlType::F32 {
                    ProfileEvent::FloatMatvec
                } else {
                    ProfileEvent::QuantizedMatvec
                };
                self.profiler.record_since(event, started);
                result
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn read_and_decode_grouped_range(
    data_source: &GgufDataSource,
    layer: &LayerDescriptor,
    range: &PlannedReadRange,
    layer_index: usize,
    range_index: usize,
    budget: &MemoryBudget,
    profiler: &Profiler,
    buffer: &mut Vec<u8>,
    loaded: &mut Vec<(String, TensorData)>,
    total_layer_bytes: &mut u64,
) -> Result<(), String> {
    let descriptors: Vec<&ramforge_core::model::TensorDescriptor> = range
        .tensors
        .iter()
        .map(|tensor| &layer.tensors[tensor.descriptor_index])
        .collect();
    data_source
        .read_coalesced_tensor_range_into(
            &descriptors,
            range.file_offset,
            range.byte_length,
            buffer,
        )
        .map_err(|error| {
            let tensor_names = descriptors
                .iter()
                .map(|descriptor| descriptor.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "failed grouped read for layer {} range {} tensors [{}]: {}",
                layer_index, range_index, tensor_names, error
            )
        })?;

    for tensor in &range.tensors {
        let descriptor = &layer.tensors[tensor.descriptor_index];
        let tensor_bytes = descriptor.byte_length.unwrap_or(0);
        let start = usize::try_from(tensor.offset_in_range)
            .map_err(|_| format!("tensor '{}' grouped offset is too large", descriptor.name))?;
        let length = usize::try_from(tensor_bytes)
            .map_err(|_| format!("tensor '{}' byte length is too large", descriptor.name))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| format!("tensor '{}' grouped slice overflows", descriptor.name))?;
        let slice = buffer.get(start..end).ok_or_else(|| {
            format!(
                "tensor '{}' is outside grouped read buffer",
                descriptor.name
            )
        })?;
        let tensor_data = load_tensor_data_from_borrowed_bytes(descriptor, slice, profiler)?;
        let resident = tensor_data.resident_bytes() as u64;
        let charge_name = format!("layer:{}:{}", layer_index, descriptor.name);
        if budget.get(&charge_name) != Some(resident.max(1)) {
            return Err(format!(
                "grouped tensor '{}' resident charge mismatch",
                descriptor.name
            ));
        }
        *total_layer_bytes = (*total_layer_bytes)
            .checked_add(resident)
            .ok_or_else(|| "layer resident byte total overflow".to_string())?;
        loaded.push((descriptor.name.clone(), tensor_data));
    }
    Ok(())
}

/// Load one tensor into its resident representation.
///
/// F32 takes the direct datasource path into final `Vec<f32>` storage. F16,
/// BF16, and quantized formats retain their M7.1 raw-byte construction paths;
/// quantized bytes are moved into compact TensorData without expansion.
fn load_tensor_data(
    data_source: &GgufDataSource,
    desc: &ramforge_core::model::TensorDescriptor,
    profiler: &Profiler,
) -> Result<TensorData, String> {
    if desc.ggml_type == GgmlType::F32 {
        let data = data_source
            .read_f32_tensor_by_descriptor(desc)
            .map_err(|e| format!("failed to read tensor '{}': {}", desc.name, e))?;
        let started = profiler.start();
        let result = TensorData::from_f32_vec(desc.dimensions.clone(), desc.num_elements, data)
            .map_err(|e| format!("failed to create TensorData for '{}': {}", desc.name, e));
        profiler.record_since(ProfileEvent::TensorConstruction, started);
        result
    } else {
        let raw_bytes = data_source
            .read_tensor_by_descriptor(desc)
            .map_err(|e| format!("failed to read tensor '{}': {}", desc.name, e))?;
        let started = profiler.start();
        let result = TensorData::from_bytes(
            desc.ggml_type,
            desc.dimensions.clone(),
            desc.num_elements,
            raw_bytes,
        )
        .map_err(|e| format!("failed to create TensorData for '{}': {}", desc.name, e));
        let elapsed = started.map(|instant| instant.elapsed());
        if let Some(elapsed) = elapsed {
            profiler.record(ProfileEvent::TensorConstruction, elapsed);
            if matches!(desc.ggml_type, GgmlType::F16 | GgmlType::BF16) {
                profiler.record(ProfileEvent::Dequantization, elapsed);
            }
        }
        result
    }
}

fn load_tensor_data_from_borrowed_bytes(
    desc: &ramforge_core::model::TensorDescriptor,
    bytes: &[u8],
    profiler: &Profiler,
) -> Result<TensorData, String> {
    let started = profiler.start();
    let result = if matches!(
        desc.ggml_type,
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16
    ) {
        let decoded = decode_tensor_to_f32(bytes, desc.ggml_type, desc.num_elements)
            .map_err(|error| format!("failed to decode tensor '{}': {}", desc.name, error))?;
        TensorData::from_decoded_float_vec(
            desc.ggml_type,
            desc.dimensions.clone(),
            desc.num_elements,
            decoded,
        )
        .map_err(|error| format!("failed to create TensorData for '{}': {}", desc.name, error))
    } else {
        let owned_bytes = if started.is_some() {
            // Keep the timed interval limited to the authoritative payload copy.
            let copy_started = std::time::Instant::now();
            let owned_bytes = bytes.to_vec();
            let copy_elapsed = copy_started.elapsed();
            profiler.record_grouped_quantized_copy(bytes.len() as u64, copy_elapsed);
            owned_bytes
        } else {
            bytes.to_vec()
        };
        TensorData::from_bytes(
            desc.ggml_type,
            desc.dimensions.clone(),
            desc.num_elements,
            owned_bytes,
        )
        .map_err(|error| format!("failed to create TensorData for '{}': {}", desc.name, error))
    };
    let elapsed = started.map(|instant| instant.elapsed());
    if let Some(elapsed) = elapsed {
        profiler.record(ProfileEvent::TensorConstruction, elapsed);
        if matches!(
            desc.ggml_type,
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16
        ) {
            profiler.record(ProfileEvent::Dequantization, elapsed);
        }
    }
    result
}

/// Apply the persistent residency policy and, when resident, load one tensor
/// under a charge that covers its complete I/O/construction lifetime.
///
/// Returns the allocation name only for a newly resident tensor so the model
/// loader can roll it back transactionally if a later persistent fails.
fn load_persistent_weight(
    data_source: &GgufDataSource,
    name: &str,
    budget: &mut MemoryBudget,
    profiler: &Profiler,
) -> Result<(PersistentWeight, Option<String>), String> {
    let desc = data_source
        .get_descriptor(name)
        .map_err(|e| format!("tensor '{}' not found: {}", name, e))?
        .clone();
    let file_bytes = desc.byte_length.ok_or_else(|| {
        format!(
            "cannot load persistent tensor '{}': byte length is unknown for {}",
            name,
            desc.ggml_type.name()
        )
    })?;
    let expected_resident =
        TensorData::resident_bytes_for(desc.ggml_type, desc.num_elements, file_bytes)
            .map_err(|e| format!("failed to determine resident size for '{}': {}", name, e))?;

    if !should_keep_resident(expected_resident, budget.total_bytes()) {
        return Ok((PersistentWeight::Streamed(desc), None));
    }

    let transient_charge = tensor_load_charge_bytes(desc.ggml_type, file_bytes)?;
    if transient_charge < expected_resident {
        return Err(format!(
            "persistent tensor '{}' load charge {} is smaller than predicted resident size {}",
            name, transient_charge, expected_resident
        ));
    }
    let alloc_name = format!("weight:{}", name);
    budget
        .allocate(alloc_name.clone(), transient_charge)
        .map_err(|e| {
            format!(
                "RAM budget exceeded establishing {}-byte load charge for '{}': {}",
                transient_charge, name, e
            )
        })?;

    let load_result = (|| -> Result<TensorData, String> {
        // The load charge is already live before I/O. F32 is read directly
        // into final storage; other formats retain the raw+decode path covered
        // by their larger transient charges.
        let tensor_data = load_tensor_data(data_source, &desc, profiler)?;

        let actual_resident = tensor_data.resident_bytes() as u64;
        if actual_resident != expected_resident {
            return Err(format!(
                "persistent tensor '{}' resident-size mismatch: descriptor predicted {} bytes, decoded representation owns {} bytes",
                name, expected_resident, actual_resident
            ));
        }

        // F32 already occupies only final storage. For converted floats the
        // raw buffer is now dropped; quantized bytes moved into TensorData.
        // Atomically settle any conservative transient to exact residency.
        budget
            .resize(&alloc_name, actual_resident)
            .map_err(|e| format!("failed to settle resident charge for '{}': {}", name, e))?;
        Ok(tensor_data)
    })();

    match load_result {
        Ok(tensor_data) => Ok((PersistentWeight::Resident(tensor_data), Some(alloc_name))),
        Err(error) => {
            // Covers read, decode, residency-consistency, and settlement
            // failures. `resize` is atomic, so this always removes whichever
            // transient/settled charge is currently present.
            let _ = budget.release(&alloc_name);
            Err(error)
        }
    }
}

/// Validate optional qwen2 Q/K/V bias tensors in a loaded layer:
/// - all-or-none: a partial bias set is a corrupt/unsupported model;
/// - exact 1D shape: q.bias = [n_heads*head_dim], k/v.bias =
///   [n_kv_heads*head_dim].
///
/// Biases ride the normal layer-load budget charges (already accounted by
/// `load_layer` before extraction) and are released together with the layer.
fn validate_qkv_bias(
    weights: &StreamingLayerWeights,
    layer_idx: usize,
    cfg: &LlamaConfig,
) -> Result<(), String> {
    let present = [
        weights.attn_q_bias.is_some(),
        weights.attn_k_bias.is_some(),
        weights.attn_v_bias.is_some(),
    ];
    let count = present.iter().filter(|p| **p).count();
    if count == 0 {
        return Ok(());
    }
    if count != 3 {
        return Err(format!(
            "layer {} has an incomplete Q/K/V bias set (q.bias: {}, k.bias: {}, v.bias: {}); refusing to run inference with partial biases",
            layer_idx, present[0], present[1], present[2]
        ));
    }
    let q_dim = cfg.head_count * cfg.head_dim;
    let kv_dim = cfg.head_count_kv * cfg.head_dim;
    for (name, bias, expected) in [
        ("attn_q.bias", &weights.attn_q_bias, q_dim),
        ("attn_k.bias", &weights.attn_k_bias, kv_dim),
        ("attn_v.bias", &weights.attn_v_bias, kv_dim),
    ] {
        let b = bias.as_ref().expect("all-or-none checked above");
        let shape = b.shape();
        if shape.len() != 1 || shape[0] != expected {
            return Err(format!(
                "invalid blk.{}.{} shape: expected 1D [{}], got {:?}",
                layer_idx, name, expected, shape
            ));
        }
    }
    Ok(())
}

/// Borrow already-decoded float storage and allocate only when a compact
/// quantized tensor must be expanded for a non-matvec operation.
fn tensor_f32_view(tensor: &TensorData) -> Result<Cow<'_, [f32]>, String> {
    if let Some((data, _)) = tensor.as_f32_slice() {
        Ok(Cow::Borrowed(data))
    } else {
        tensor
            .to_f32_vec()
            .map(Cow::Owned)
            .map_err(|error| error.to_string())
    }
}

/// Persistent streamed weights necessarily produce owned decoded storage;
/// resident decoded floats can be borrowed for the duration of the operation.
fn persistent_f32_view<'a>(
    weight: &'a PersistentWeight,
    data_source: &GgufDataSource,
) -> Result<Cow<'a, [f32]>, String> {
    match weight {
        PersistentWeight::Resident(tensor) => tensor_f32_view(tensor),
        PersistentWeight::Streamed(_) => weight.to_f32_vec(data_source).map(Cow::Owned),
    }
}

/// Matvec dispatch under the single explicit ggml layout (`shape = [in, out]`):
/// resident F32 data goes through the SIMD/threaded compute backend; all other
/// types use the compact block-wise kernels (no full F32 expansion, no
/// orientation guessing).
fn matvec_backend<B: ComputeBackend>(
    backend: &B,
    profiler: &Profiler,
    td: &TensorData,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    if let Some((data, shape)) = td.as_f32_slice() {
        let started = profiler.start();
        let result = backend.matvec(data, shape, x, y);
        profiler.record_since(ProfileEvent::FloatMatvec, started);
        result
    } else {
        let started = profiler.start();
        let result = quantized_matvec_dispatch(backend, td, x, y, profiler);
        profiler.record_since(ProfileEvent::QuantizedMatvec, started);
        result
    }
}

/// Minimum output rows (out_dim) before we hand Q4_0 / Q6_K work to the
/// backend thread pool. Below this threshold we keep the fused scalar
/// kernel and avoid Rayon pool/spawn overhead.
///
/// Derived from the `quantized_matvec_bench` microbenchmark in this file:
///   * 32×1536 and 64×1536 (in_dim=1536): no measurable speedup at 2/4/8/9
///     threads; best-case times match the scalar path to noise.
///   * 128×1536 and larger: parallel dispatch produces measurable speedup
///     (≈1.3–2.0× at 2 threads in the sandbox; higher core counts scale
///     further on real hardware).
/// Real Qwen2.5-1.5B layer shapes are 1536 / 8960 / 151936 rows — all well
/// above this threshold and parallelize cleanly.
const QUANT_PARALLEL_ROW_THRESHOLD: usize = 128;

/// Split `out_dim` rows into `n_chunks` contiguous (start, count) ranges
/// with at most one row of imbalance across chunks.
fn make_row_chunks(out_dim: usize, n_chunks: usize) -> Vec<(usize, usize)> {
    let n = n_chunks.min(out_dim).max(1);
    let base = out_dim / n;
    let extra = out_dim % n;
    let mut out = Vec::with_capacity(n);
    let mut cursor = 0usize;
    for i in 0..n {
        let count = base + if i < extra { 1 } else { 0 };
        out.push((cursor, count));
        cursor += count;
    }
    out
}

fn quantized_matvec_dispatch<B: ComputeBackend>(
    backend: &B,
    td: &TensorData,
    x: &[f32],
    y: &mut [f32],
    profiler: &Profiler,
) -> Result<(), String> {
    /// Returns (raw_data, kernel_shape=[out, in]) for a quantized tensor if
    /// it matches one of the row-parallel kernels, otherwise None to fall
    /// back to the generic `td.matvec` path.
    fn as_quantized_view<'a>(td: &'a TensorData) -> Option<(GgmlType, &'a [u8], [usize; 2])> {
        let (qt, ty) = match td {
            TensorData::Q4_0(qt) => (qt, GgmlType::Q4_0),
            TensorData::Q6_K(qt) => (qt, GgmlType::Q6_K),
            _ => return None,
        };
        if qt.shape.len() != 2 {
            return None;
        }
        // ggml [in, out] → kernel layout [out, in]
        Some((ty, &qt.raw_data, [qt.shape[1], qt.shape[0]]))
    }

    match as_quantized_view(td) {
        Some((GgmlType::Q4_0, raw, [out_dim, in_dim])) => {
            // Counters follow the same [in, out] semantics as before.
            let blocks = (out_dim as u64).saturating_mul((in_dim / QK4_0) as u64);
            let weight_bytes = blocks.saturating_mul(BLOCK_SIZE_Q4_0 as u64);
            profiler.record_q4_0_matvec(out_dim as u64, in_dim as u64, blocks, weight_bytes);

            let w_shape = [out_dim, in_dim];
            let n_threads = backend.num_threads();
            let parallelize = n_threads > 1 && out_dim >= QUANT_PARALLEL_ROW_THRESHOLD;
            if parallelize {
                // Split the output rows into n_threads contiguous ranges and
                // run each range through the fused scalar kernel on the
                // backend's bounded Rayon pool. No per-row heap allocation;
                // x and raw_data are shared read-only.
                //
                // SAFETY: Row ranges are non-overlapping by construction
                // (`make_row_chunks` partitions [0, out_dim)), so each worker
                // writes to a disjoint y subslice. We hand out raw pointers
                // because Rayon's `into_par_iter().try_for_each` requires a
                // `Fn` (not `FnMut`) closure and would otherwise forbid
                // capturing an outer `&mut [f32]`.
                let blocks_per_row = in_dim / QK4_0;
                let row_bytes = blocks_per_row * BLOCK_SIZE_Q4_0;
                let ranges = make_row_chunks(out_dim, n_threads);
                // SAFETY: row ranges are non-overlapping partitions of
                // [0, out_dim), so each worker writes to a disjoint y subslice
                // and reads a disjoint w subslice. We transmit the base
                // addresses as usizes because raw pointers are neither Send
                // nor Sync; the addresses remain valid for the whole call
                // since both `y` and `raw` outlive the worker pool scope.
                let y_addr = y.as_mut_ptr() as usize;
                let raw_addr = raw.as_ptr() as usize;
                backend.in_worker_pool(move || {
                    ranges.into_par_iter().try_for_each(
                        |(row_start, row_count)| -> Result<(), String> {
                            let w_offset = row_start * row_bytes;
                            let w_sub = unsafe {
                                std::slice::from_raw_parts(
                                    (raw_addr + w_offset) as *const u8,
                                    row_count * row_bytes,
                                )
                            };
                            let y_sub = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (y_addr + row_start * std::mem::size_of::<f32>()) as *mut f32,
                                    row_count,
                                )
                            };
                            let sub_shape = [row_count, in_dim];
                            matvec_q4_0_row_range(w_sub, &sub_shape, x, y_sub, 0, None)
                                .map_err(|e| e.to_string())
                        },
                    )
                })?;
                Ok(())
            } else {
                matvec_q4_0_row_range(raw, &w_shape, x, y, 0, None).map_err(|e| e.to_string())
            }
        }
        Some((GgmlType::Q6_K, raw, [out_dim, in_dim])) => {
            let w_shape = [out_dim, in_dim];
            let n_threads = backend.num_threads();
            let parallelize = n_threads > 1 && out_dim >= QUANT_PARALLEL_ROW_THRESHOLD;
            if parallelize {
                let blocks_per_row = in_dim / QK_K;
                let row_bytes = blocks_per_row * BLOCK_SIZE_Q6_K;
                let ranges = make_row_chunks(out_dim, n_threads);
                // SAFETY: same disjoint-range reasoning as the Q4_0 branch.
                let y_addr = y.as_mut_ptr() as usize;
                let raw_addr = raw.as_ptr() as usize;
                backend.in_worker_pool(move || {
                    ranges.into_par_iter().try_for_each(
                        |(row_start, row_count)| -> Result<(), String> {
                            let w_offset = row_start * row_bytes;
                            let w_sub = unsafe {
                                std::slice::from_raw_parts(
                                    (raw_addr + w_offset) as *const u8,
                                    row_count * row_bytes,
                                )
                            };
                            let y_sub = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (y_addr + row_start * std::mem::size_of::<f32>()) as *mut f32,
                                    row_count,
                                )
                            };
                            let sub_shape = [row_count, in_dim];
                            matvec_q6_k_row_range(w_sub, &sub_shape, x, y_sub, 0, None)
                                .map_err(|e| e.to_string())
                        },
                    )
                })?;
                Ok(())
            } else {
                matvec_q6_k_row_range(raw, &w_shape, x, y, 0, None).map_err(|e| e.to_string())
            }
        }
        _ => {
            // Other quantized types: original scalar path unchanged.
            td.matvec(x, y).map_err(|e| e.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ramforge_core::model::{align_offset, TensorDescriptor};
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Boxed value-writer closure used by the GGUF test fixtures.
    type WriteValFn<'a> = Box<dyn FnMut(&mut Vec<u8>) + 'a>;

    fn write_string<W: Write>(w: &mut W, s: &str) {
        w.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
        w.write_all(s.as_bytes()).unwrap();
    }
    fn write_u32<W: Write>(w: &mut W, v: u32) {
        w.write_all(&v.to_le_bytes()).unwrap();
    }
    fn write_u64<W: Write>(w: &mut W, v: u64) {
        w.write_all(&v.to_le_bytes()).unwrap();
    }
    fn write_f32<W: Write>(w: &mut W, v: f32) {
        w.write_all(&v.to_le_bytes()).unwrap();
    }

    /// Minimal one-tensor GGUF used to exercise persistent loading directly.
    fn create_single_tensor_gguf(
        name: &str,
        ggml_type: GgmlType,
        dims: &[u64],
        raw_data: &[u8],
    ) -> NamedTempFile {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        buf.extend_from_slice(&0u64.to_le_bytes()); // metadata count

        write_string(&mut buf, name);
        write_u32(&mut buf, dims.len() as u32);
        for &dim in dims {
            write_u64(&mut buf, dim);
        }
        write_u32(&mut buf, ggml_type.as_u32());
        write_u64(&mut buf, 0); // relative tensor-data offset

        let pos = buf.len() as u64;
        let aligned = align_offset(pos, 32);
        buf.extend(vec![0u8; (aligned - pos) as usize]);
        buf.extend_from_slice(raw_data);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        tmp.flush().unwrap();
        tmp
    }

    fn test_descriptor(
        name: &str,
        ggml_type: GgmlType,
        dimensions: &[u64],
        byte_length: u64,
    ) -> TensorDescriptor {
        TensorDescriptor {
            name: name.to_string(),
            dimensions: dimensions.to_vec(),
            ggml_type,
            offset: 0,
            file_offset: 0,
            byte_length: Some(byte_length),
            num_elements: dimensions.iter().product(),
        }
    }

    fn assert_grouped_float_copy_not_profiled(ggml_type: GgmlType, bytes: Vec<u8>) {
        let descriptor = test_descriptor("blk.0.test.weight", ggml_type, &[2], bytes.len() as u64);
        let profiler = Profiler::default();
        profiler.set_enabled(true);

        let tensor = load_tensor_data_from_borrowed_bytes(&descriptor, &bytes, &profiler).unwrap();
        assert!(
            !tensor.is_quantized(),
            "{} must decode as float",
            ggml_type.name()
        );
        let snapshot = profiler.snapshot();
        assert_eq!(snapshot.grouped_quantized_copy_count, 0);
        assert_eq!(snapshot.grouped_quantized_copy_bytes, 0);
        assert_eq!(
            snapshot.grouped_quantized_copy_time,
            std::time::Duration::ZERO
        );
    }

    fn create_model_with_n_layers(n_layers: usize, n_embd: usize, ffn: usize) -> NamedTempFile {
        create_model_with_optional_gap(n_layers, n_embd, ffn, None)
    }

    fn create_model_with_optional_gap(
        n_layers: usize,
        n_embd: usize,
        ffn: usize,
        gap_before_definition: Option<(usize, u64)>,
    ) -> NamedTempFile {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        let tensor_count = 2 + n_layers * 9;
        buf.extend_from_slice(&(tensor_count as u64).to_le_bytes());
        buf.extend_from_slice(&11u64.to_le_bytes());

        let mut add_kv = |key: &str, val_type: u32, mut write_val: WriteValFn<'_>| {
            write_string(&mut buf, key);
            write_u32(&mut buf, val_type);
            write_val(&mut buf);
        };
        add_kv(
            "general.architecture",
            8,
            Box::new(|b| write_string(b, "llama")),
        );
        add_kv("llama.vocab_size", 4, Box::new(|b| write_u32(b, 16)));
        add_kv("llama.context_length", 4, Box::new(|b| write_u32(b, 64)));
        add_kv(
            "llama.embedding_length",
            4,
            Box::new(|b| write_u32(b, n_embd as u32)),
        );
        add_kv(
            "llama.block_count",
            4,
            Box::new(|b| write_u32(b, n_layers as u32)),
        );
        add_kv(
            "llama.feed_forward_length",
            4,
            Box::new(|b| write_u32(b, ffn as u32)),
        );
        add_kv(
            "llama.attention.head_count",
            4,
            Box::new(|b| write_u32(b, 2)),
        );
        add_kv(
            "llama.attention.head_count_kv",
            4,
            Box::new(|b| write_u32(b, 2)),
        );
        add_kv(
            "llama.attention.layer_norm_rms_epsilon",
            6,
            Box::new(|b| write_f32(b, 1e-5)),
        );
        add_kv(
            "llama.rope.freq_base",
            6,
            Box::new(|b| write_f32(b, 10000.0)),
        );
        add_kv(
            "tokenizer.ggml.model",
            8,
            Box::new(|b| write_string(b, "llama")),
        );

        let mut offset = 0u64;
        let mut defs: Vec<(String, Vec<u64>, u32)> = Vec::new();
        defs.push(("token_embd.weight".to_string(), vec![n_embd as u64, 16], 0));
        defs.push(("output_norm.weight".to_string(), vec![n_embd as u64], 0));
        for i in 0..n_layers {
            defs.push((
                format!("blk.{}.attn_norm.weight", i),
                vec![n_embd as u64],
                0,
            ));
            defs.push((
                format!("blk.{}.attn_q.weight", i),
                vec![n_embd as u64, n_embd as u64],
                0,
            ));
            defs.push((
                format!("blk.{}.attn_k.weight", i),
                vec![n_embd as u64, n_embd as u64],
                0,
            ));
            defs.push((
                format!("blk.{}.attn_v.weight", i),
                vec![n_embd as u64, n_embd as u64],
                0,
            ));
            defs.push((
                format!("blk.{}.attn_output.weight", i),
                vec![n_embd as u64, n_embd as u64],
                0,
            ));
            defs.push((format!("blk.{}.ffn_norm.weight", i), vec![n_embd as u64], 0));
            // ggml layout [in, out]: gate/up map n_embd -> ffn, down maps ffn -> n_embd
            defs.push((
                format!("blk.{}.ffn_gate.weight", i),
                vec![n_embd as u64, ffn as u64],
                0,
            ));
            defs.push((
                format!("blk.{}.ffn_up.weight", i),
                vec![n_embd as u64, ffn as u64],
                0,
            ));
            defs.push((
                format!("blk.{}.ffn_down.weight", i),
                vec![ffn as u64, n_embd as u64],
                0,
            ));
        }

        for (definition_index, (name, dims, ty)) in defs.iter().enumerate() {
            if let Some((gap_index, gap_bytes)) = gap_before_definition {
                if definition_index == gap_index {
                    offset += gap_bytes;
                }
            }
            write_string(&mut buf, name);
            write_u32(&mut buf, dims.len() as u32);
            for d in dims {
                write_u64(&mut buf, *d);
            }
            write_u32(&mut buf, *ty);
            write_u64(&mut buf, offset);
            let elems: u64 = dims.iter().product();
            offset += elems * 4;
        }

        let pos = buf.len() as u64;
        let aligned = align_offset(pos, 32);
        buf.extend(vec![0u8; (aligned - pos) as usize]);
        buf.extend(vec![0u8; offset as usize]);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        tmp.flush().unwrap();
        tmp
    }

    /// F16/BF16 accounting fixture: token_embd + attn_q are F16, attn_k is
    /// BF16, everything else F32. 1 layer, n_embd 8, ffn 16, vocab 16.
    fn create_f16_bf16_model() -> NamedTempFile {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        let tensor_count = 2 + 9;
        buf.extend_from_slice(&(tensor_count as u64).to_le_bytes());
        buf.extend_from_slice(&11u64.to_le_bytes());

        let mut add_kv = |key: &str, val_type: u32, mut write_val: WriteValFn<'_>| {
            write_string(&mut buf, key);
            write_u32(&mut buf, val_type);
            write_val(&mut buf);
        };
        add_kv(
            "general.architecture",
            8,
            Box::new(|b| write_string(b, "llama")),
        );
        add_kv("llama.vocab_size", 4, Box::new(|b| write_u32(b, 16)));
        add_kv("llama.context_length", 4, Box::new(|b| write_u32(b, 64)));
        add_kv("llama.embedding_length", 4, Box::new(|b| write_u32(b, 8)));
        add_kv("llama.block_count", 4, Box::new(|b| write_u32(b, 1)));
        add_kv(
            "llama.feed_forward_length",
            4,
            Box::new(|b| write_u32(b, 16)),
        );
        add_kv(
            "llama.attention.head_count",
            4,
            Box::new(|b| write_u32(b, 2)),
        );
        add_kv(
            "llama.attention.head_count_kv",
            4,
            Box::new(|b| write_u32(b, 2)),
        );
        add_kv(
            "llama.attention.layer_norm_rms_epsilon",
            6,
            Box::new(|b| write_f32(b, 1e-5)),
        );
        add_kv(
            "llama.rope.freq_base",
            6,
            Box::new(|b| write_f32(b, 10000.0)),
        );
        add_kv(
            "tokenizer.ggml.model",
            8,
            Box::new(|b| write_string(b, "llama")),
        );

        // (name, dims, ggml type id, bytes per element)
        let n_embd: u64 = 8;
        let ffn: u64 = 16;
        let mut defs: Vec<(String, Vec<u64>, u32, u64)> = vec![
            ("token_embd.weight".into(), vec![n_embd, 16], 1, 2), // F16
            ("output_norm.weight".into(), vec![n_embd], 0, 4),
        ];
        defs.push(("blk.0.attn_norm.weight".into(), vec![n_embd], 0, 4));
        defs.push(("blk.0.attn_q.weight".into(), vec![n_embd, n_embd], 1, 2)); // F16
        defs.push(("blk.0.attn_k.weight".into(), vec![n_embd, n_embd], 30, 2)); // BF16
        defs.push(("blk.0.attn_v.weight".into(), vec![n_embd, n_embd], 0, 4));
        defs.push((
            "blk.0.attn_output.weight".into(),
            vec![n_embd, n_embd],
            0,
            4,
        ));
        defs.push(("blk.0.ffn_norm.weight".into(), vec![n_embd], 0, 4));
        defs.push(("blk.0.ffn_gate.weight".into(), vec![n_embd, ffn], 0, 4));
        defs.push(("blk.0.ffn_up.weight".into(), vec![n_embd, ffn], 0, 4));
        defs.push(("blk.0.ffn_down.weight".into(), vec![ffn, n_embd], 0, 4));

        let mut offset = 0u64;
        for (name, dims, ty, bpe) in &defs {
            write_string(&mut buf, name);
            write_u32(&mut buf, dims.len() as u32);
            for d in dims {
                write_u64(&mut buf, *d);
            }
            write_u32(&mut buf, *ty);
            write_u64(&mut buf, offset);
            let elems: u64 = dims.iter().product();
            offset += elems * bpe;
        }
        let pos = buf.len() as u64;
        let aligned = align_offset(pos, 32);
        buf.extend(vec![0u8; (aligned - pos) as usize]);

        // Data in the same order as defs.
        let f16_one: u16 = 0x3C00;
        let bf16_one: u16 = (1.0f32.to_bits() >> 16) as u16;
        for (name, dims, ty, bpe) in &defs {
            let elems: usize = dims.iter().product::<u64>() as usize;
            let _ = (name, ty);
            match bpe {
                4 => {
                    for _ in 0..elems {
                        buf.extend_from_slice(&0.5f32.to_le_bytes());
                    }
                }
                2 => {
                    let bits = if *ty == 30 { bf16_one } else { f16_one };
                    for _ in 0..elems {
                        buf.extend_from_slice(&bits.to_le_bytes());
                    }
                }
                _ => unreachable!(),
            }
        }

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        tmp.flush().unwrap();
        tmp
    }

    /// qwen2 arch fixture with optional Q/K/V bias tensors (mode-controlled).
    /// 1 layer, n_embd 8 (heads 2 x head_dim 4 => q_dim 8, kv_dim 8), ffn 16.
    #[derive(PartialEq)]
    enum BiasMode {
        Full,
        OnlyQ,
        WrongDim,
    }

    fn create_model_with_qkv_bias(mode: BiasMode) -> NamedTempFile {
        let n_embd: u64 = 8;
        let ffn: u64 = 16;

        let mut defs: Vec<(String, Vec<u64>)> = vec![
            ("token_embd.weight".into(), vec![n_embd, 16]),
            ("output_norm.weight".into(), vec![n_embd]),
            ("blk.0.attn_norm.weight".into(), vec![n_embd]),
            ("blk.0.attn_q.weight".into(), vec![n_embd, n_embd]),
            ("blk.0.attn_k.weight".into(), vec![n_embd, n_embd]),
            ("blk.0.attn_v.weight".into(), vec![n_embd, n_embd]),
            ("blk.0.attn_output.weight".into(), vec![n_embd, n_embd]),
            ("blk.0.ffn_norm.weight".into(), vec![n_embd]),
            ("blk.0.ffn_gate.weight".into(), vec![n_embd, ffn]),
            ("blk.0.ffn_up.weight".into(), vec![n_embd, ffn]),
            ("blk.0.ffn_down.weight".into(), vec![ffn, n_embd]),
        ];
        match mode {
            BiasMode::Full => {
                defs.push(("blk.0.attn_q.bias".into(), vec![n_embd]));
                defs.push(("blk.0.attn_k.bias".into(), vec![n_embd]));
                defs.push(("blk.0.attn_v.bias".into(), vec![n_embd]));
            }
            BiasMode::OnlyQ => {
                defs.push(("blk.0.attn_q.bias".into(), vec![n_embd]));
            }
            BiasMode::WrongDim => {
                defs.push(("blk.0.attn_q.bias".into(), vec![4])); // wrong: q_dim is 8
                defs.push(("blk.0.attn_k.bias".into(), vec![n_embd]));
                defs.push(("blk.0.attn_v.bias".into(), vec![n_embd]));
            }
        }

        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(defs.len() as u64).to_le_bytes());
        buf.extend_from_slice(&11u64.to_le_bytes());

        let mut add_kv = |key: &str, val_type: u32, mut write_val: WriteValFn<'_>| {
            write_string(&mut buf, key);
            write_u32(&mut buf, val_type);
            write_val(&mut buf);
        };
        // qwen2 arch + qwen2.* metadata keys (same tensor naming as llama)
        add_kv(
            "general.architecture",
            8,
            Box::new(|b| write_string(b, "qwen2")),
        );
        add_kv("qwen2.vocab_size", 4, Box::new(|b| write_u32(b, 16)));
        add_kv("qwen2.context_length", 4, Box::new(|b| write_u32(b, 64)));
        add_kv("qwen2.embedding_length", 4, Box::new(|b| write_u32(b, 8)));
        add_kv("qwen2.block_count", 4, Box::new(|b| write_u32(b, 1)));
        add_kv(
            "qwen2.feed_forward_length",
            4,
            Box::new(|b| write_u32(b, 16)),
        );
        add_kv(
            "qwen2.attention.head_count",
            4,
            Box::new(|b| write_u32(b, 2)),
        );
        add_kv(
            "qwen2.attention.head_count_kv",
            4,
            Box::new(|b| write_u32(b, 2)),
        );
        add_kv(
            "qwen2.attention.layer_norm_rms_epsilon",
            6,
            Box::new(|b| write_f32(b, 1e-5)),
        );
        add_kv(
            "qwen2.rope.freq_base",
            6,
            Box::new(|b| write_f32(b, 10000.0)),
        );
        add_kv(
            "tokenizer.ggml.model",
            8,
            Box::new(|b| write_string(b, "llama")),
        );

        let mut offset = 0u64;
        for (name, dims) in &defs {
            write_string(&mut buf, name);
            write_u32(&mut buf, dims.len() as u32);
            for d in dims {
                write_u64(&mut buf, *d);
            }
            write_u32(&mut buf, 0); // F32
            write_u64(&mut buf, offset);
            let elems: u64 = dims.iter().product();
            offset += elems * 4;
        }
        let pos = buf.len() as u64;
        let aligned = align_offset(pos, 32);
        buf.extend(vec![0u8; (aligned - pos) as usize]);
        buf.extend(vec![0u8; offset as usize]);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        tmp.flush().unwrap();
        tmp
    }

    #[test]
    fn test_grouped_quantized_copy_profiles_one_exact_payload() {
        let descriptor = test_descriptor(
            "blk.0.attn_q.weight",
            GgmlType::Q4_K,
            &[ramforge_core::quant::QK_K as u64],
            ramforge_core::quant::BLOCK_SIZE_Q4_K as u64,
        );
        let bytes = vec![0x5a; ramforge_core::quant::BLOCK_SIZE_Q4_K];
        let profiler = Profiler::default();
        profiler.set_enabled(true);

        let tensor = load_tensor_data_from_borrowed_bytes(&descriptor, &bytes, &profiler).unwrap();
        match tensor {
            TensorData::Q4_K(quantized) => assert_eq!(quantized.raw_data, bytes),
            other => panic!("expected Q4_K tensor, got {:?}", other),
        }

        let snapshot = profiler.snapshot();
        assert_eq!(snapshot.grouped_quantized_copy_count, 1);
        assert_eq!(
            snapshot.grouped_quantized_copy_bytes,
            descriptor.byte_length.unwrap()
        );
        assert!(snapshot.grouped_quantized_copy_time <= snapshot.tensor_construction);
    }

    #[test]
    fn test_grouped_quantized_copy_profiles_multiple_payloads_exactly() {
        let q4_k = test_descriptor(
            "blk.0.attn_q.weight",
            GgmlType::Q4_K,
            &[ramforge_core::quant::QK_K as u64],
            ramforge_core::quant::BLOCK_SIZE_Q4_K as u64,
        );
        let q4_0 = test_descriptor(
            "blk.0.attn_k.weight",
            GgmlType::Q4_0,
            &[ramforge_core::quant::QK4_0 as u64],
            ramforge_core::quant::BLOCK_SIZE_Q4_0 as u64,
        );
        let profiler = Profiler::default();
        profiler.set_enabled(true);

        load_tensor_data_from_borrowed_bytes(
            &q4_k,
            &[0; ramforge_core::quant::BLOCK_SIZE_Q4_K],
            &profiler,
        )
        .unwrap();
        load_tensor_data_from_borrowed_bytes(
            &q4_0,
            &[0; ramforge_core::quant::BLOCK_SIZE_Q4_0],
            &profiler,
        )
        .unwrap();

        let snapshot = profiler.snapshot();
        assert_eq!(snapshot.grouped_quantized_copy_count, 2);
        assert_eq!(
            snapshot.grouped_quantized_copy_bytes,
            q4_k.byte_length.unwrap() + q4_0.byte_length.unwrap()
        );
    }

    #[test]
    fn test_individual_quantized_load_does_not_profile_grouped_copy() {
        let bytes = vec![0; ramforge_core::quant::BLOCK_SIZE_Q4_K];
        let tmp = create_single_tensor_gguf(
            "test.weight",
            GgmlType::Q4_K,
            &[ramforge_core::quant::QK_K as u64],
            &bytes,
        );
        let data_source = GgufDataSource::open(tmp.path()).unwrap();
        let descriptor = data_source.get_descriptor("test.weight").unwrap();
        let profiler = Profiler::default();
        profiler.set_enabled(true);

        let tensor = load_tensor_data(&data_source, descriptor, &profiler).unwrap();
        assert!(tensor.is_quantized());
        let snapshot = profiler.snapshot();
        assert_eq!(snapshot.grouped_quantized_copy_count, 0);
        assert_eq!(snapshot.grouped_quantized_copy_bytes, 0);
        assert_eq!(
            snapshot.grouped_quantized_copy_time,
            std::time::Duration::ZERO
        );
    }

    #[test]
    fn test_grouped_f32_does_not_profile_quantized_copy() {
        assert_grouped_float_copy_not_profiled(GgmlType::F32, 1.0f32.to_le_bytes().repeat(2));
    }

    #[test]
    fn test_grouped_f16_does_not_profile_quantized_copy() {
        assert_grouped_float_copy_not_profiled(GgmlType::F16, 0x3c00u16.to_le_bytes().repeat(2));
    }

    #[test]
    fn test_grouped_bf16_does_not_profile_quantized_copy() {
        let one = ((1.0f32.to_bits() >> 16) as u16).to_le_bytes();
        assert_grouped_float_copy_not_profiled(GgmlType::BF16, one.repeat(2));
    }

    #[test]
    fn test_grouped_quantized_copy_profiling_disabled_is_noop() {
        let descriptor = test_descriptor(
            "blk.0.attn_q.weight",
            GgmlType::Q4_K,
            &[ramforge_core::quant::QK_K as u64],
            ramforge_core::quant::BLOCK_SIZE_Q4_K as u64,
        );
        let bytes = vec![0; ramforge_core::quant::BLOCK_SIZE_Q4_K];
        let profiler = Profiler::default();

        let tensor = load_tensor_data_from_borrowed_bytes(&descriptor, &bytes, &profiler).unwrap();
        assert!(tensor.is_quantized());
        assert_eq!(
            profiler.snapshot(),
            crate::profile::ProfileSnapshot::default()
        );
    }

    #[test]
    fn test_tensor_f32_view_borrows_float_storage_and_owns_quantized_fallback() {
        for ggml_type in [GgmlType::F32, GgmlType::F16, GgmlType::BF16] {
            let tensor =
                TensorData::from_decoded_float_vec(ggml_type, vec![4], 4, vec![1.0, 2.0, 3.0, 4.0])
                    .unwrap();
            let resident = tensor.as_f32_slice().unwrap().0;
            let resident_ptr = resident.as_ptr();
            match tensor_f32_view(&tensor).unwrap() {
                std::borrow::Cow::Borrowed(view) => {
                    assert_eq!(view.as_ptr(), resident_ptr);
                    assert_eq!(view, &[1.0, 2.0, 3.0, 4.0]);
                }
                std::borrow::Cow::Owned(_) => {
                    panic!("{} decoded storage must be borrowed", ggml_type.name())
                }
            }
        }

        let mut raw = Vec::with_capacity(ramforge_core::quant::BLOCK_SIZE_Q4_0);
        raw.extend_from_slice(&0x3c00u16.to_le_bytes());
        raw.extend_from_slice(&[0x88; 16]);
        let tensor = TensorData::from_bytes(GgmlType::Q4_0, vec![32], 32, raw).unwrap();
        match tensor_f32_view(&tensor).unwrap() {
            std::borrow::Cow::Owned(view) => {
                assert_eq!(view.len(), 32);
                assert!(view.iter().all(|value| *value == 0.0));
            }
            std::borrow::Cow::Borrowed(_) => panic!("quantized storage must be decoded"),
        }
    }

    #[test]
    fn test_persistent_f32_view_borrows_resident_and_owns_streamed_storage() {
        let mut raw = Vec::with_capacity(2 * std::mem::size_of::<f32>());
        for value in [1.0f32, 2.0] {
            raw.extend_from_slice(&value.to_le_bytes());
        }
        let tmp = create_single_tensor_gguf("test.weight", GgmlType::F32, &[2], &raw);
        let data_source = GgufDataSource::open(tmp.path()).unwrap();

        let resident_tensor = TensorData::from_f32_vec(vec![2], 2, vec![1.0, 2.0]).unwrap();
        let resident_ptr = resident_tensor.as_f32_slice().unwrap().0.as_ptr();
        let resident_weight = PersistentWeight::Resident(resident_tensor);
        match persistent_f32_view(&resident_weight, &data_source).unwrap() {
            std::borrow::Cow::Borrowed(view) => {
                assert_eq!(view.as_ptr(), resident_ptr);
                assert_eq!(view, &[1.0, 2.0]);
            }
            std::borrow::Cow::Owned(_) => panic!("resident F32 storage must be borrowed"),
        }

        let descriptor = data_source.get_descriptor("test.weight").unwrap().clone();
        let streamed_weight = PersistentWeight::Streamed(descriptor);
        match persistent_f32_view(&streamed_weight, &data_source).unwrap() {
            std::borrow::Cow::Owned(view) => assert_eq!(view.as_slice(), &[1.0, 2.0]),
            std::borrow::Cow::Borrowed(_) => {
                panic!("streamed storage must remain owned for the operation")
            }
        }
    }

    #[test]
    fn test_configured_cache_above_runtime_maximum_rolls_back_startup_charges() {
        let tmp = create_model_with_n_layers(1, 8, 16);
        let data_source = GgufDataSource::open(tmp.path()).unwrap();
        let mut budget = MemoryBudget::new(4096).unwrap();
        budget.allocate("existing", 17).unwrap();
        let used_before = budget.used_bytes();
        let config = RuntimeConfig::new(1, 4096, true, 4096, true, true).unwrap();

        let error =
            StreamingLlamaModel::load_with_runtime_config(&data_source, &mut budget, &config)
                .unwrap_err();
        assert!(error.contains("configured layer cache capacity"));
        assert_eq!(budget.used_bytes(), used_before);
        assert_eq!(budget.get("existing"), Some(17));
        assert!(!budget
            .allocations()
            .keys()
            .any(|name| name.starts_with("weight:")));
    }

    #[test]
    fn test_persistent_f32_direct_load_uses_one_owned_representation() {
        let raw = vec![0u8; 8 * 4];
        let tmp = create_single_tensor_gguf("test.weight", GgmlType::F32, &[8], &raw);
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let desc = ds.get_descriptor("test.weight").unwrap();
        assert_eq!(desc.byte_length, Some(32));
        assert_eq!(
            TensorData::resident_bytes_for(GgmlType::F32, 8, 32).unwrap(),
            32
        );
        assert_eq!(tensor_load_charge_bytes(GgmlType::F32, 32).unwrap(), 32);

        // Exactly 32 bytes remain. The direct loader succeeds because its only
        // owned tensor representation is the final 32-byte Vec<f32>; the old
        // raw+decoded 64-byte path would have failed this budget.
        let mut budget = MemoryBudget::new(128).unwrap();
        budget.allocate("existing", 96).unwrap();
        let (weight, allocation) =
            load_persistent_weight(&ds, "test.weight", &mut budget, &Profiler::default()).unwrap();
        assert!(weight.is_resident());
        assert_eq!(weight.resident_bytes(), 32);
        assert_eq!(allocation.as_deref(), Some("weight:test.weight"));
        assert_eq!(budget.get("weight:test.weight"), Some(32));
        assert_eq!(budget.used_bytes(), 128);
        drop(weight);
        budget.release("weight:test.weight").unwrap();
        assert_eq!(budget.used_bytes(), 96);

        // One byte less than the final representation must fail before I/O and
        // leave the previous budget state untouched.
        let mut tight = MemoryBudget::new(128).unwrap();
        tight.allocate("existing", 97).unwrap();
        let before = tight.used_bytes();
        let error = load_persistent_weight(&ds, "test.weight", &mut tight, &Profiler::default())
            .unwrap_err();
        assert!(error.contains("load charge"), "unexpected error: {}", error);
        assert_eq!(tight.used_bytes(), before);
        assert!(tight.get("weight:test.weight").is_none());
        assert!(!tight
            .allocations()
            .keys()
            .any(|name| name.starts_with("tmp:")));
    }

    #[test]
    fn test_persistent_f16_transient_and_settled_accounting() {
        let raw: Vec<u8> = (0..8).flat_map(|_| 0x3C00u16.to_le_bytes()).collect();
        let tmp = create_single_tensor_gguf("test.weight", GgmlType::F16, &[8], &raw);
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let file_bytes = ds
            .get_descriptor("test.weight")
            .unwrap()
            .byte_length
            .unwrap();
        assert_eq!(file_bytes, 8 * 2);
        assert_eq!(
            tensor_load_charge_bytes(GgmlType::F16, file_bytes).unwrap(),
            3 * file_bytes
        );
        assert_eq!(
            TensorData::resident_bytes_for(GgmlType::F16, 8, file_bytes).unwrap(),
            8 * 4
        );

        let mut budget = MemoryBudget::new(128).unwrap();
        let (weight, allocation) =
            load_persistent_weight(&ds, "test.weight", &mut budget, &Profiler::default()).unwrap();
        assert!(weight.is_resident());
        assert_eq!(weight.resident_bytes(), 8 * 4);
        assert_eq!(allocation.as_deref(), Some("weight:test.weight"));
        assert_eq!(budget.get("weight:test.weight"), Some(8 * 4));
        assert_eq!(budget.used_bytes(), 8 * 4);

        drop(weight);
        budget.release("weight:test.weight").unwrap();
        assert_eq!(budget.used_bytes(), 0);
    }

    #[test]
    fn test_persistent_bf16_transient_and_settled_accounting() {
        let one = (1.0f32.to_bits() >> 16) as u16;
        let raw: Vec<u8> = (0..8).flat_map(|_| one.to_le_bytes()).collect();
        let tmp = create_single_tensor_gguf("test.weight", GgmlType::BF16, &[8], &raw);
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let file_bytes = ds
            .get_descriptor("test.weight")
            .unwrap()
            .byte_length
            .unwrap();
        assert_eq!(file_bytes, 8 * 2);
        assert_eq!(
            tensor_load_charge_bytes(GgmlType::BF16, file_bytes).unwrap(),
            3 * file_bytes
        );
        assert_eq!(
            TensorData::resident_bytes_for(GgmlType::BF16, 8, file_bytes).unwrap(),
            8 * 4
        );

        let mut budget = MemoryBudget::new(128).unwrap();
        let (weight, allocation) =
            load_persistent_weight(&ds, "test.weight", &mut budget, &Profiler::default()).unwrap();
        assert!(weight.is_resident());
        assert_eq!(weight.resident_bytes(), 8 * 4);
        assert_eq!(allocation.as_deref(), Some("weight:test.weight"));
        assert_eq!(budget.get("weight:test.weight"), Some(8 * 4));
        assert_eq!(budget.used_bytes(), 8 * 4);

        drop(weight);
        budget.release("weight:test.weight").unwrap();
        assert_eq!(budget.used_bytes(), 0);
    }

    #[test]
    fn test_persistent_quantized_stays_compact() {
        let mut raw = Vec::with_capacity(18);
        raw.extend_from_slice(&0x3C00u16.to_le_bytes());
        raw.extend_from_slice(&[0x88; 16]);
        let tmp = create_single_tensor_gguf("test.weight", GgmlType::Q4_0, &[32], &raw);
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let file_bytes = ds
            .get_descriptor("test.weight")
            .unwrap()
            .byte_length
            .unwrap();
        assert_eq!(file_bytes, 18);
        assert_eq!(
            tensor_load_charge_bytes(GgmlType::Q4_0, file_bytes).unwrap(),
            file_bytes
        );
        assert_eq!(
            TensorData::resident_bytes_for(GgmlType::Q4_0, 32, file_bytes).unwrap(),
            file_bytes
        );

        let mut budget = MemoryBudget::new(72).unwrap();
        let (weight, allocation) =
            load_persistent_weight(&ds, "test.weight", &mut budget, &Profiler::default()).unwrap();
        assert!(weight.is_resident());
        assert_eq!(weight.resident_bytes(), 18);
        match &weight {
            PersistentWeight::Resident(tensor) => assert!(tensor.is_quantized()),
            PersistentWeight::Streamed(_) => {
                panic!("Q4_0 tensor should fit at the policy boundary")
            }
        }
        assert_eq!(allocation.as_deref(), Some("weight:test.weight"));
        assert_eq!(budget.get("weight:test.weight"), Some(18));

        drop(weight);
        budget.release("weight:test.weight").unwrap();
        assert_eq!(budget.used_bytes(), 0);
    }

    #[test]
    fn test_persistent_policy_uses_decoded_resident_size() {
        // F32: 32 resident bytes fit exactly at the 25% boundary.
        let f32_raw = vec![0u8; 8 * 4];
        let f32_tmp = create_single_tensor_gguf("test.weight", GgmlType::F32, &[8], &f32_raw);
        let f32_ds = ramforge_core::datasource::GgufDataSource::open(f32_tmp.path()).unwrap();
        let mut f32_budget = MemoryBudget::new(128).unwrap();
        let (f32_weight, f32_allocation) = load_persistent_weight(
            &f32_ds,
            "test.weight",
            &mut f32_budget,
            &Profiler::default(),
        )
        .unwrap();
        assert!(f32_weight.is_resident());
        assert!(f32_allocation.is_some());
        drop(f32_weight);
        f32_budget.release("weight:test.weight").unwrap();

        // F16/BF16: each file is only 16 bytes, which the old file-based
        // policy would retain under a 96-byte budget (16*4 <= 96). Their true
        // decoded residency is 32 bytes, over the 24-byte threshold, so both
        // must remain streamed and create no resident charge.
        for ggml_type in [GgmlType::F16, GgmlType::BF16] {
            let raw = vec![0u8; 8 * 2];
            let tmp = create_single_tensor_gguf("test.weight", ggml_type, &[8], &raw);
            let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
            let mut budget = MemoryBudget::new(96).unwrap();
            let (weight, allocation) =
                load_persistent_weight(&ds, "test.weight", &mut budget, &Profiler::default())
                    .unwrap();
            assert!(
                weight.is_streamed(),
                "{} should be streamed",
                ggml_type.name()
            );
            assert!(allocation.is_none());
            assert_eq!(budget.used_bytes(), 0);
            assert!(budget.get("weight:test.weight").is_none());
        }
    }

    #[test]
    fn test_persistent_load_failure_rolls_back_all_startup_charges() {
        // token_embd loads successfully, then output_norm reads past a
        // deliberately truncated file. Both the current transient and the
        // already-settled earlier persistent charge must be rolled back.
        let tmp = create_model_with_n_layers(0, 8, 16);
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let token = ds.get_descriptor("token_embd.weight").unwrap();
        let token_end = token.file_offset + token.byte_length.unwrap();
        tmp.as_file().set_len(token_end).unwrap();

        let mut budget = MemoryBudget::new(4096).unwrap();
        budget.allocate("existing", 17).unwrap();
        let before = budget.used_bytes();
        let error = StreamingLlamaModel::load(&ds, &mut budget).unwrap_err();
        assert!(
            error.contains("output_norm.weight") && error.contains("read"),
            "unexpected error: {}",
            error
        );
        assert_eq!(budget.used_bytes(), before);
        assert_eq!(budget.get("existing"), Some(17));
        assert!(!budget
            .allocations()
            .keys()
            .any(|name| name.starts_with("weight:")));
        assert!(!budget
            .allocations()
            .keys()
            .any(|name| name.starts_with("tmp:")));
    }

    #[test]
    fn test_final_hidden_is_caller_owned_for_full_charged_lifetime() {
        let tmp = create_model_with_n_layers(1, 8, 16);
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let mut budget = MemoryBudget::new(1024 * 1024).unwrap();
        let model = StreamingLlamaModel::load(&ds, &mut budget).unwrap();
        let persistent_used = budget.used_bytes();
        let mut kv_cache = KvCache::new(
            model.config.block_count,
            model.config.head_count_kv,
            model.config.head_dim,
            1,
        )
        .unwrap();
        let backend = crate::backend::CpuBackend::scalar();
        let mut stats = ResidencyStats::new(model.total_weight_bytes);
        let hidden_bytes = (model.config.embedding_length * 4) as u64;

        budget
            .with_temp("tmp:hidden", hidden_bytes, |budget| {
                // Allocate only after the charge is live. The forward writes
                // into this caller-owned buffer and returns while both buffer
                // and charge remain alive for subsequent consumers.
                assert_eq!(budget.get("tmp:hidden"), Some(hidden_bytes));
                let mut final_hidden = vec![0.0f32; model.config.embedding_length];
                model.forward_single_streaming(
                    0,
                    0,
                    &mut kv_cache,
                    &backend,
                    &ds,
                    budget,
                    &mut stats,
                    &mut final_hidden,
                )?;
                assert_eq!(budget.get("tmp:hidden"), Some(hidden_bytes));
                assert_eq!(final_hidden.len(), model.config.embedding_length);
                Ok::<(), String>(())
            })
            .unwrap();

        model.clear_layer_cache(&mut budget).unwrap();
        assert_eq!(budget.used_bytes(), persistent_used);
        assert!(budget.get("tmp:hidden").is_none());
        assert!(!budget
            .allocations()
            .keys()
            .any(|name| name.starts_with("tmp:")));
    }

    #[test]
    fn test_layer_load_reuses_one_buffer_across_grouped_ranges() {
        let gap = crate::layer_read::MAX_COALESCED_GAP_BYTES + 1;
        let tmp = create_model_with_optional_gap(1, 8, 16, Some((6, gap)));
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let mut budget = MemoryBudget::new(1024 * 1024).unwrap();
        let model = StreamingLlamaModel::load(&ds, &mut budget).unwrap();
        assert_eq!(model.layer_read_plans[0].ranges.len(), 2);
        assert!(model.layer_read_plans[0]
            .reusable_group_buffer_bytes()
            .is_some());

        ds.set_profiling(true);
        ds.reset_io_profile();
        let before = budget.used_bytes();
        let mut stats = ResidencyStats::new(model.total_weight_bytes);
        let layer = model.load_layer(0, &ds, &mut budget, &mut stats).unwrap();
        let profile = ds.io_profile();
        assert_eq!(profile.logical_tensor_reads, 9);
        assert_eq!(profile.read_operations, 2);
        assert_eq!(profile.read_buffer_reuses, 2);
        assert_eq!(profile.read_buffer_growths, 0);
        assert!(!budget
            .allocations()
            .keys()
            .any(|name| name.starts_with("tmp:layer_read:")));
        model.release_layer(0, &mut budget, &mut stats);
        drop(layer);
        assert_eq!(budget.used_bytes(), before);
    }

    #[test]
    fn test_runtime_config_can_disable_grouped_buffer_reuse_without_disabling_coalescing() {
        let gap = crate::layer_read::MAX_COALESCED_GAP_BYTES + 1;
        let tmp = create_model_with_optional_gap(1, 8, 16, Some((6, gap)));
        let data_source = GgufDataSource::open(tmp.path()).unwrap();
        let mut budget = MemoryBudget::new(1024 * 1024).unwrap();
        let config = RuntimeConfig::new(1, 1024 * 1024, false, 0, true, false).unwrap();
        let model =
            StreamingLlamaModel::load_with_runtime_config(&data_source, &mut budget, &config)
                .unwrap();

        data_source.set_profiling(true);
        data_source.reset_io_profile();
        let before = budget.used_bytes();
        let mut stats = ResidencyStats::new(model.total_weight_bytes);
        let layer = model
            .load_layer(0, &data_source, &mut budget, &mut stats)
            .unwrap();
        let profile = data_source.io_profile();
        assert_eq!(profile.read_operations, 2);
        assert_eq!(profile.coalesced_ranges, 2);
        assert_eq!(profile.read_buffer_reuses, 0);
        assert_eq!(profile.read_buffer_growths, 2);
        model.release_layer(0, &mut budget, &mut stats);
        drop(layer);
        assert_eq!(budget.used_bytes(), before);
    }

    #[test]
    fn test_reusable_buffer_peak_falls_back_to_per_range_buffers() {
        let gap = crate::layer_read::MAX_COALESCED_GAP_BYTES + 1;
        let tmp = create_model_with_optional_gap(1, 8, 16, Some((9, gap)));
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let mut budget = MemoryBudget::new(4_500).unwrap();
        let model = StreamingLlamaModel::load(&ds, &mut budget).unwrap();
        assert!(model.layer_read_plans[0]
            .reusable_group_buffer_bytes()
            .is_some());

        ds.set_profiling(true);
        ds.reset_io_profile();
        let before = budget.used_bytes();
        let mut stats = ResidencyStats::new(model.total_weight_bytes);
        let layer = model.load_layer(0, &ds, &mut budget, &mut stats).unwrap();
        let profile = ds.io_profile();
        assert_eq!(profile.read_operations, 2);
        assert_eq!(profile.coalesced_ranges, 2);
        assert_eq!(profile.read_buffer_reuses, 0);
        assert_eq!(profile.read_buffer_growths, 2);
        model.release_layer(0, &mut budget, &mut stats);
        drop(layer);
        assert_eq!(budget.used_bytes(), before);
    }

    #[test]
    fn test_layer_grouping_and_streaming() {
        let tmp = create_model_with_n_layers(4, 8, 16);
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let mut budget = ramforge_core::memory::MemoryBudget::new(1024 * 1024).unwrap();

        let model = StreamingLlamaModel::load(&ds, &mut budget).unwrap();
        assert_eq!(model.layer_descriptors.len(), 4);
        assert!(model.total_weight_bytes > 0);

        let mut stats = crate::residency::ResidencyStats::new(model.total_weight_bytes);
        let layer0 = model.load_layer(0, &ds, &mut budget, &mut stats).unwrap();
        assert!(stats.current_resident_layer_bytes > 0);
        assert_eq!(stats.num_layer_loads, 1);
        model.release_layer(0, &mut budget, &mut stats);
        assert_eq!(stats.current_resident_layer_bytes, 0);
        assert_eq!(stats.num_layer_releases, 1);
        drop(layer0);
    }

    #[test]
    fn test_direct_f32_layer_short_read_releases_all_charges() {
        let tmp = create_model_with_n_layers(1, 8, 16);
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let q_desc = ds.get_descriptor("blk.0.attn_q.weight").unwrap();
        tmp.as_file()
            .set_len(q_desc.file_offset + q_desc.byte_length.unwrap() - 1)
            .unwrap();

        // Persistent tensors precede the truncated layer payload and still
        // load successfully. The layer's first tensor settles, then the direct
        // F32 Q read fails short; every layer charge must roll back.
        let mut budget = MemoryBudget::new(1024 * 1024).unwrap();
        let model = StreamingLlamaModel::load(&ds, &mut budget).unwrap();
        let before = budget.used_bytes();
        let mut stats = ResidencyStats::new(model.total_weight_bytes);
        let error = model
            .load_layer(0, &ds, &mut budget, &mut stats)
            .map(|_| ())
            .unwrap_err();
        assert!(
            error.contains("blk.0.attn_q.weight") && error.contains("read"),
            "unexpected error: {}",
            error
        );
        assert_eq!(budget.used_bytes(), before);
        assert!(!budget
            .allocations()
            .keys()
            .any(|name| name.starts_with("layer:0:")));
        assert!(!budget
            .allocations()
            .keys()
            .any(|name| name.starts_with("tmp:")));
    }

    #[test]
    fn test_out_of_core_model_larger_than_budget() {
        let tmp = create_model_with_n_layers(8, 32, 64);
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let total_bytes: u64 = ds
            .model()
            .tensors
            .iter()
            .filter_map(|t| t.byte_length)
            .sum();

        // Per layer ~41 KiB; direct F32 loading charges one final
        // representation, plus persistents (~2.1 KiB). 96 KiB fits one layer
        // comfortably; the whole model (~324 KiB) never has to fit at once.
        let ram_budget = 96 * 1024;

        let mut budget = ramforge_core::memory::MemoryBudget::new(ram_budget).unwrap();

        let model = StreamingLlamaModel::load(&ds, &mut budget).unwrap();

        assert!(total_bytes > ram_budget);

        let mut stats = crate::residency::ResidencyStats::new(total_bytes);
        for i in 0..model.config.block_count {
            let _layer = model.load_layer(i, &ds, &mut budget, &mut stats).unwrap();
            assert!(stats.current_resident_layer_bytes < total_bytes);
            assert!(budget.used_bytes() <= ram_budget);
            model.release_layer(i, &mut budget, &mut stats);
        }

        assert!(stats.peak_resident_layer_bytes < total_bytes);
        assert!(stats.peak_managed_bytes <= ram_budget);
        assert_eq!(stats.num_layer_loads, 8);
        assert_eq!(stats.num_layer_releases, 8);
    }

    #[test]
    fn test_quantized_layer_loading() {
        // Create a model with Q4_0 tensors
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        let n_layers = 1;
        let n_embd = 32;
        let ffn = 64;
        let tensor_count = 2 + n_layers * 9;
        buf.extend_from_slice(&(tensor_count as u64).to_le_bytes());
        buf.extend_from_slice(&11u64.to_le_bytes());

        let mut add_kv = |key: &str, val_type: u32, mut write_val: WriteValFn<'_>| {
            write_string(&mut buf, key);
            write_u32(&mut buf, val_type);
            write_val(&mut buf);
        };
        add_kv(
            "general.architecture",
            8,
            Box::new(|b| write_string(b, "llama")),
        );
        add_kv("llama.vocab_size", 4, Box::new(|b| write_u32(b, 16)));
        add_kv("llama.context_length", 4, Box::new(|b| write_u32(b, 64)));
        add_kv(
            "llama.embedding_length",
            4,
            Box::new(|b| write_u32(b, n_embd as u32)),
        );
        add_kv(
            "llama.block_count",
            4,
            Box::new(|b| write_u32(b, n_layers as u32)),
        );
        add_kv(
            "llama.feed_forward_length",
            4,
            Box::new(|b| write_u32(b, ffn as u32)),
        );
        add_kv(
            "llama.attention.head_count",
            4,
            Box::new(|b| write_u32(b, 2)),
        );
        add_kv(
            "llama.attention.head_count_kv",
            4,
            Box::new(|b| write_u32(b, 2)),
        );
        add_kv(
            "llama.attention.layer_norm_rms_epsilon",
            6,
            Box::new(|b| write_f32(b, 1e-5)),
        );
        add_kv(
            "llama.rope.freq_base",
            6,
            Box::new(|b| write_f32(b, 10000.0)),
        );
        add_kv(
            "tokenizer.ggml.model",
            8,
            Box::new(|b| write_string(b, "llama")),
        );

        let mut offset = 0u64;
        let mut defs: Vec<(String, Vec<u64>, u32)> = Vec::new();
        // Use Q4_0 for some tensors: type 2
        defs.push(("token_embd.weight".to_string(), vec![n_embd as u64, 16], 0)); // F32 for simplicity
        defs.push(("output_norm.weight".to_string(), vec![n_embd as u64], 0));
        for i in 0..n_layers {
            defs.push((
                format!("blk.{}.attn_norm.weight", i),
                vec![n_embd as u64],
                0,
            ));
            defs.push((
                format!("blk.{}.attn_q.weight", i),
                vec![n_embd as u64, n_embd as u64],
                2,
            )); // Q4_0
            defs.push((
                format!("blk.{}.attn_k.weight", i),
                vec![n_embd as u64, n_embd as u64],
                2,
            ));
            defs.push((
                format!("blk.{}.attn_v.weight", i),
                vec![n_embd as u64, n_embd as u64],
                2,
            ));
            defs.push((
                format!("blk.{}.attn_output.weight", i),
                vec![n_embd as u64, n_embd as u64],
                2,
            ));
            defs.push((format!("blk.{}.ffn_norm.weight", i), vec![n_embd as u64], 0));
            // ggml layout [in, out]: gate/up map n_embd -> ffn, down maps ffn -> n_embd
            defs.push((
                format!("blk.{}.ffn_gate.weight", i),
                vec![n_embd as u64, ffn as u64],
                2,
            ));
            defs.push((
                format!("blk.{}.ffn_up.weight", i),
                vec![n_embd as u64, ffn as u64],
                2,
            ));
            defs.push((
                format!("blk.{}.ffn_down.weight", i),
                vec![ffn as u64, n_embd as u64],
                2,
            ));
        }

        for (name, dims, ty) in &defs {
            write_string(&mut buf, name);
            write_u32(&mut buf, dims.len() as u32);
            for d in dims {
                write_u64(&mut buf, *d);
            }
            write_u32(&mut buf, *ty);
            write_u64(&mut buf, offset);
            let elems: u64 = dims.iter().product();
            let bytes = match *ty {
                2 => (elems / 32) * 18, // Q4_0
                _ => elems * 4,
            };
            offset += bytes;
        }

        let pos = buf.len() as u64;
        let aligned = align_offset(pos, 32);
        buf.extend(vec![0u8; (aligned - pos) as usize]);

        // Write dummy data: for F32 tensors 1.0, for Q4_0: d=1.0, qs=0x88 (0)
        // token_embd F32
        for _ in 0..16 * n_embd {
            buf.extend_from_slice(&1.0f32.to_le_bytes());
        }
        // output_norm
        for _ in 0..n_embd {
            buf.extend_from_slice(&1.0f32.to_le_bytes());
        }
        for _ in 0..n_layers {
            // attn_norm F32
            for _ in 0..n_embd {
                buf.extend_from_slice(&1.0f32.to_le_bytes());
            }
            // Q4_0 tensors: each 32 elements => 18 bytes per block
            // n_embd 32 => 1 block per row? For [32,32] => 32 rows * 18 =576 bytes per tensor
            // We'll write zeros for simplicity: d=1.0, qs=0x88 (dequant 0)
            for _ in 0..4 {
                // 4 tensors * 576?
                for _ in 0..n_embd {
                    let d_fp16: u16 = 0x3C00;
                    buf.extend_from_slice(&d_fp16.to_le_bytes());
                    buf.extend_from_slice(&[0x88; 16]);
                }
            }
            // ffn_norm
            for _ in 0..n_embd {
                buf.extend_from_slice(&1.0f32.to_le_bytes());
            }
            // ffn_gate, up, down Q4_0
            // ffn 64, n_embd 32: [64,32] => 64 rows, each row 32 elements => 1 block per row => 64*18=1152 per tensor
            for _ in 0..2 {
                for _ in 0..ffn {
                    let d_fp16: u16 = 0x3C00;
                    buf.extend_from_slice(&d_fp16.to_le_bytes());
                    buf.extend_from_slice(&[0x88; 16]);
                }
            }
            // ffn_down [32,64]: 32 rows, each 64 elements => 2 blocks per row => 2*18=36 per row, 32*36=1152
            for _ in 0..n_embd {
                for _ in 0..2 {
                    let d_fp16: u16 = 0x3C00;
                    buf.extend_from_slice(&d_fp16.to_le_bytes());
                    buf.extend_from_slice(&[0x88; 16]);
                }
            }
        }

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        tmp.flush().unwrap();

        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let mut budget = ramforge_core::memory::MemoryBudget::new(1024 * 1024).unwrap();

        let model = StreamingLlamaModel::load(&ds, &mut budget).unwrap();
        // Check that quantized tensors are detected
        assert!(model.total_weight_bytes > 0);
        // Load layer should succeed even with quantized
        let mut stats = crate::residency::ResidencyStats::new(model.total_weight_bytes);
        let layer = model.load_layer(0, &ds, &mut budget, &mut stats).unwrap();
        assert!(layer.attn_q.is_quantized());
        // Matvec should work (with zeros, output zeros)
        let x = vec![1.0f32; n_embd];
        let mut y = vec![0.0f32; n_embd];
        layer.attn_q.matvec(&x, &mut y).unwrap();
        // Since dequantized zeros, y should be zeros
        for &v in &y {
            assert!(v.abs() < 1e-5);
        }
    }

    #[test]
    fn test_quantized_cache_hit_does_not_repeat_grouped_copy() {
        let descriptor = test_descriptor(
            "blk.0.attn_q.weight",
            GgmlType::Q4_K,
            &[ramforge_core::quant::QK_K as u64],
            ramforge_core::quant::BLOCK_SIZE_Q4_K as u64,
        );
        let bytes = vec![0; ramforge_core::quant::BLOCK_SIZE_Q4_K];
        let profiler = Profiler::default();
        profiler.set_enabled(true);
        let tensor = load_tensor_data_from_borrowed_bytes(&descriptor, &bytes, &profiler).unwrap();
        let after_load = profiler.snapshot();
        assert_eq!(after_load.grouped_quantized_copy_count, 1);

        let resident_bytes = tensor.resident_bytes() as u64;
        let mut budget = MemoryBudget::new(resident_bytes).unwrap();
        budget
            .allocate("layer:0:attn_q.weight", resident_bytes)
            .unwrap();
        let mut cache = crate::layer_cache::LayerCache::new(resident_bytes);
        assert!(matches!(
            cache
                .insert_loaded(0, tensor, resident_bytes, &mut budget)
                .unwrap(),
            InsertOutcome::Cached { evictions: 0 }
        ));
        assert_eq!(cache.with_entry(0, TensorData::is_quantized), Some(true));

        let after_hit = profiler.snapshot();
        assert_eq!(
            after_hit.grouped_quantized_copy_count,
            after_load.grouped_quantized_copy_count
        );
        assert_eq!(
            after_hit.grouped_quantized_copy_bytes,
            after_load.grouped_quantized_copy_bytes
        );
        assert_eq!(
            after_hit.grouped_quantized_copy_time,
            after_load.grouped_quantized_copy_time
        );
        assert_eq!(cache.clear(&mut budget).unwrap(), 1);
        assert_eq!(budget.used_bytes(), 0);
        assert!(budget.allocations().is_empty());
    }

    #[test]
    fn test_f16_bf16_persistent_and_layer_accounting() {
        let tmp = create_f16_bf16_model();
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let mut budget = ramforge_core::memory::MemoryBudget::new(1024 * 1024).unwrap();

        let model = StreamingLlamaModel::load(&ds, &mut budget).unwrap();

        // F16 persistent embedding is booked at DECODED residency
        // (128 elems * 4 B, not the 2 B/elem file size).
        assert_eq!(budget.get("weight:token_embd.weight"), Some(8 * 16 * 4));
        assert_eq!(budget.get("weight:output_norm.weight"), Some(8 * 4));
        let persistents_used = budget.used_bytes();
        assert_eq!(persistents_used, 512 + 32);

        let mut stats = crate::residency::ResidencyStats::new(model.total_weight_bytes);
        let layer = model.load_layer(0, &ds, &mut budget, &mut stats).unwrap();

        // F16 attn_q and BF16 attn_k settle to decoded F32 size (4 B/elem).
        assert_eq!(budget.get("layer:0:blk.0.attn_q.weight"), Some(8 * 8 * 4));
        assert_eq!(budget.get("layer:0:blk.0.attn_k.weight"), Some(8 * 8 * 4));

        // Exact layer total: norms 2*32 + q,k,v,o 4*256 + ffn 3*512 = 2624.
        assert_eq!(budget.used_bytes(), persistents_used + 2624);
        assert_eq!(layer.total_resident_bytes(), 2624);

        // Release restores exactly the pre-load state.
        model.release_layer(0, &mut budget, &mut stats);
        assert_eq!(budget.used_bytes(), persistents_used);
        drop(layer);
        assert!(!budget
            .allocations()
            .keys()
            .any(|k| k.starts_with("layer:0:")));
    }

    #[test]
    fn test_failed_layer_load_preserves_existing_cache_and_accounting() {
        let tmp = create_model_with_n_layers(2, 8, 16);
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let mut budget = MemoryBudget::new(1024 * 1024).unwrap();
        let model = StreamingLlamaModel::load(&ds, &mut budget).unwrap();
        let mut stats = ResidencyStats::new(model.total_weight_bytes);

        let layer0 = model.load_layer(0, &ds, &mut budget, &mut stats).unwrap();
        let layer0_bytes = layer0.total_resident_bytes();
        {
            let mut cache = model.layer_cache.lock().unwrap();
            assert!(matches!(
                cache
                    .insert_loaded(0, layer0, layer0_bytes, &mut budget)
                    .unwrap(),
                InsertOutcome::Cached { .. }
            ));
        }
        let before = budget.used_bytes();
        let layer1_q = ds.get_descriptor("blk.1.attn_q.weight").unwrap();
        tmp.as_file()
            .set_len(layer1_q.file_offset + layer1_q.byte_length.unwrap() - 1)
            .unwrap();

        let error = model
            .load_layer(1, &ds, &mut budget, &mut stats)
            .map(|_| ())
            .unwrap_err();
        assert!(error.contains("blk.1.attn_q.weight"));
        assert_eq!(budget.used_bytes(), before);
        let cache = model.layer_cache.lock().unwrap();
        assert!(cache.contains(0));
        assert_eq!(cache.entry_count(), 1);
        drop(cache);
        model.clear_layer_cache(&mut budget).unwrap();
    }

    #[test]
    fn test_layer_load_failure_cleans_budget() {
        let tmp = create_f16_bf16_model();
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        // Under the corrected resident-size policy, the decoded 512-byte F16
        // embedding exceeds this budget's 25% threshold and stays streamed;
        // only the 32-byte output norm is resident. Layer loading progresses
        // through the F16/BF16 projections, then fails when the next F32 load
        // transient cannot fit. The partial layer must still roll back fully.
        let mut budget = ramforge_core::memory::MemoryBudget::new(1024).unwrap();

        let model = StreamingLlamaModel::load(&ds, &mut budget).unwrap();
        assert!(model.token_embd.is_streamed());
        let before = budget.used_bytes();
        assert_eq!(before, 32);

        let mut stats = crate::residency::ResidencyStats::new(model.total_weight_bytes);
        let err = model
            .load_layer(0, &ds, &mut budget, &mut stats)
            .map(|_| ())
            .unwrap_err();
        assert!(
            err.contains("budget") || err.contains("insufficient"),
            "expected budget error, got: {}",
            err
        );
        // The failed load must leave no partial layer or cache charges behind.
        assert_eq!(budget.used_bytes(), before);
        assert!(!budget
            .allocations()
            .keys()
            .any(|k| k.starts_with("layer:0:")));
        let cache = model.layer_cache.lock().unwrap();
        assert_eq!(cache.entry_count(), 0);
        assert_eq!(cache.used_bytes(), 0);
    }

    #[test]
    fn test_qkv_bias_layer_loading_and_accounting() {
        let tmp = create_model_with_qkv_bias(BiasMode::Full);
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let mut budget = ramforge_core::memory::MemoryBudget::new(1024 * 1024).unwrap();

        let model = StreamingLlamaModel::load(&ds, &mut budget).unwrap();
        assert!(model.attn_bias_present);

        let persistents_used = budget.used_bytes();
        let mut stats = crate::residency::ResidencyStats::new(model.total_weight_bytes);
        let layer = model.load_layer(0, &ds, &mut budget, &mut stats).unwrap();

        assert!(layer.attn_q_bias.is_some());
        assert!(layer.attn_k_bias.is_some());
        assert!(layer.attn_v_bias.is_some());
        assert_eq!(layer.attn_q_bias.as_ref().unwrap().shape(), &[8]);

        // Exact total: 9 weights (2624) + 3 biases (96).
        assert_eq!(budget.used_bytes(), persistents_used + 2720);
        assert_eq!(layer.total_resident_bytes(), 2720);

        model.release_layer(0, &mut budget, &mut stats);
        assert_eq!(budget.used_bytes(), persistents_used);
        drop(layer);
    }

    #[test]
    fn test_qkv_bias_partial_set_rejected() {
        let tmp = create_model_with_qkv_bias(BiasMode::OnlyQ);
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let mut budget = ramforge_core::memory::MemoryBudget::new(1024 * 1024).unwrap();

        let model = StreamingLlamaModel::load(&ds, &mut budget).unwrap();
        assert!(model.attn_bias_present); // presence of q bias detected at load

        let before = budget.used_bytes();
        let mut stats = crate::residency::ResidencyStats::new(model.total_weight_bytes);
        let err = model
            .load_layer(0, &ds, &mut budget, &mut stats)
            .map(|_| ())
            .unwrap_err();
        assert!(
            err.contains("incomplete Q/K/V bias"),
            "expected bias-set rejection, got: {}",
            err
        );
        // Rejection must not leak the layer's charges.
        assert_eq!(budget.used_bytes(), before);
        assert!(!budget
            .allocations()
            .keys()
            .any(|k| k.starts_with("layer:0:")));
    }

    #[test]
    fn test_qkv_bias_shape_mismatch_rejected() {
        let tmp = create_model_with_qkv_bias(BiasMode::WrongDim);
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let mut budget = ramforge_core::memory::MemoryBudget::new(1024 * 1024).unwrap();

        let model = StreamingLlamaModel::load(&ds, &mut budget).unwrap();
        let before = budget.used_bytes();
        let mut stats = crate::residency::ResidencyStats::new(model.total_weight_bytes);
        let err = model
            .load_layer(0, &ds, &mut budget, &mut stats)
            .map(|_| ())
            .unwrap_err();
        assert!(
            err.contains("attn_q.bias") && err.contains("shape"),
            "expected bias shape rejection, got: {}",
            err
        );
        assert_eq!(budget.used_bytes(), before);
    }

    #[test]
    fn test_q4_0_parallel_matvec_matches_scalar_at_qwen2_5_q_shape() {
        // Reproduce the Qwen2.5-1.5B Q-projection shape (in=1536, out=1536,
        // which exceeds QUANT_PARALLEL_ROW_THRESHOLD=128 so the row-parallel
        // dispatcher is exercised). Compare against a reference that
        // dequantizes each row and dots with x in scalar order. Any
        // partition or offset bug would manifest as a discrepancy here.
        use ramforge_core::quant::{BLOCK_SIZE_Q4_0, QK4_0};
        let in_dim = 1536usize;
        let out_dim = 1536usize;
        assert_eq!(in_dim % QK4_0, 0);
        let blocks_per_row = in_dim / QK4_0;
        let row_bytes = blocks_per_row * BLOCK_SIZE_Q4_0;
        let mut seed: u32 = 0xC0FFEE;
        let mut raw = vec![0u8; out_dim * row_bytes];
        for r in 0..out_dim {
            let mut s = seed;
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let off = r * row_bytes;
            for b in 0..blocks_per_row {
                // d: pseudorandom scale in [0.05, 0.5)
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let d: f32 = 0.05 + ((s >> 8) as f32 / (1u32 << 24) as f32) * 0.45;
                let d_bits = half::f16::from_f32(d).to_bits().to_le_bytes();
                raw[off + b * BLOCK_SIZE_Q4_0] = d_bits[0];
                raw[off + b * BLOCK_SIZE_Q4_0 + 1] = d_bits[1];
                for j in 0..16 {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    let lo = (s & 0x0F) as u8;
                    let hi = ((s >> 4) & 0x0F) as u8;
                    raw[off + b * BLOCK_SIZE_Q4_0 + 2 + j] = lo | (hi << 4);
                }
            }
        }
        let w = TensorData::from_bytes(
            GgmlType::Q4_0,
            vec![in_dim as u64, out_dim as u64],
            (in_dim * out_dim) as u64,
            raw.clone(),
        )
        .unwrap();

        // Deterministic x vector
        let mut x = vec![0.0f32; in_dim];
        for i in 0..in_dim {
            x[i] = ((i as f32) * 0.0137 - 0.5).sin() * 0.7;
        }

        // Reference: dequant each row via BlockQ4_0, then scalar dot
        use ramforge_core::quant::BlockQ4_0;
        let mut ref_y = vec![0.0f32; out_dim];
        for r in 0..out_dim {
            let mut dot = 0.0f32;
            for b in 0..blocks_per_row {
                let block_start = r * row_bytes + b * BLOCK_SIZE_Q4_0;
                let blk = BlockQ4_0::from_bytes(&raw[block_start..block_start + BLOCK_SIZE_Q4_0])
                    .unwrap();
                let mut dq = [0f32; 32];
                blk.dequantize(&mut dq);
                for j in 0..32 {
                    dot += dq[j] * x[b * 32 + j];
                }
            }
            ref_y[r] = dot;
        }

        let profiler = Profiler::default();
        // Use 4 threads to force parallel partitioning.
        let backend = crate::backend::CpuBackend::with_threads(4);
        let mut y = vec![0.0f32; out_dim];
        matvec_backend(&backend, &profiler, &w, &x, &mut y).unwrap();

        // Compare with generous tolerance to absorb FMADD reordering
        let mut max_abs = 0.0f32;
        for r in 0..out_dim {
            let diff = (y[r] - ref_y[r]).abs();
            if diff > max_abs {
                max_abs = diff;
            }
        }
        assert!(
            max_abs < 1e-2,
            "parallel Q4_0 matvec diverges from scalar reference (max_abs={}, ref0={}, got0={})",
            max_abs, ref_y[0], y[0]
        );
    }

    #[test]
    fn test_q4_0_matvec_counters_are_recorded_without_affecting_result() {
        use ramforge_core::quant::{BLOCK_SIZE_Q4_0, QK4_0};
        use std::time::Duration;

        // Build a small Q4_0 weight matrix under the ggml [in, out] layout:
        // in_dim=32 columns, out_dim=2 rows. Each block encodes zero
        // magnitudes (qs[i] = 0x88 dequants to 0) so the numerical output is
        // deterministic zeros regardless of input.
        let in_dim = 32u64;
        let out_dim = 2u64;
        assert_eq!(
            in_dim as usize % QK4_0,
            0,
            "in_dim must be a multiple of QK4_0"
        );
        let blocks_per_row = in_dim as usize / QK4_0;
        let weight_bytes = (out_dim as usize) * blocks_per_row * BLOCK_SIZE_Q4_0;
        let mut raw = Vec::with_capacity(weight_bytes);
        for _ in 0..(out_dim as usize) * blocks_per_row {
            // d = 1.0 in fp16 (0x3C00), qs all 0x88 => dequants to zeros
            raw.extend_from_slice(&0x3C00u16.to_le_bytes());
            raw.extend_from_slice(&[0x88; 16]);
        }
        assert_eq!(raw.len(), weight_bytes);
        let w =
            TensorData::from_bytes(GgmlType::Q4_0, vec![in_dim, out_dim], in_dim * out_dim, raw)
                .unwrap();

        let profiler = Profiler::default();
        profiler.set_enabled(true);
        let backend = crate::backend::CpuBackend::scalar();
        let x = vec![0.5f32; in_dim as usize];
        let mut y = vec![-999.0f32; out_dim as usize];
        matvec_backend(&backend, &profiler, &w, &x, &mut y).unwrap();

        // Numerical result must be unchanged (zeros).
        for value in y {
            assert!(value.abs() < 1e-6, "expected zero output, got {}", value);
        }

        let blocks_total = out_dim * blocks_per_row as u64;
        let weight_total = weight_bytes as u64;
        let snapshot = profiler.snapshot();
        assert_eq!(snapshot.q4_0_matvec_calls, 1);
        assert_eq!(snapshot.q4_0_matvec_rows, out_dim);
        assert_eq!(snapshot.q4_0_matvec_input_elements, in_dim);
        assert_eq!(snapshot.q4_0_matvec_blocks, blocks_total);
        assert_eq!(snapshot.q4_0_matvec_weight_bytes, weight_total);
        // Timing must still be recorded under QuantizedMatvec.
        assert!(snapshot.quantized_matvec >= Duration::ZERO);

        // A second call accumulates additively.
        let mut y2 = vec![-999.0f32; out_dim as usize];
        matvec_backend(&backend, &profiler, &w, &x, &mut y2).unwrap();
        let snapshot2 = profiler.snapshot();
        assert_eq!(snapshot2.q4_0_matvec_calls, 2);
        assert_eq!(snapshot2.q4_0_matvec_rows, 2 * out_dim);
        assert_eq!(snapshot2.q4_0_matvec_input_elements, 2 * in_dim);
        assert_eq!(snapshot2.q4_0_matvec_blocks, 2 * blocks_total);
        assert_eq!(snapshot2.q4_0_matvec_weight_bytes, 2 * weight_total);

        // Disabled profiler must not count (counters remain at zero after reset).
        profiler.reset();
        profiler.set_enabled(false);
        let mut y3 = vec![-999.0f32; out_dim as usize];
        matvec_backend(&backend, &profiler, &w, &x, &mut y3).unwrap();
        let snapshot3 = profiler.snapshot();
        assert_eq!(snapshot3.q4_0_matvec_calls, 0);
        assert_eq!(snapshot3.q4_0_matvec_rows, 0);
        assert_eq!(snapshot3.q4_0_matvec_blocks, 0);
        assert_eq!(snapshot3.q4_0_matvec_weight_bytes, 0);
    }
}

// ---------- Quantized matvec microbenchmarks (opt-in) ----------
//
// Run with:  cargo test -p ramforge-runtime --release -- --ignored --nocapture quantized_matvec_bench
//
// These construct synthetic Q4_0 / Q6_K weight matrices at representative
// Qwen2.5-1.5B shapes, warm the scalar path for reference, then time scalar
// vs parallel dispatch at 1/2/4/8/9 threads and report elapsed/throughput/
// speedup plus maximum absolute numerical error. They do not allocate in
// the hot path beyond the one-time weight bytes and the caller-provided
// x/y vectors.
//
// Helpers below are only referenced from the `#[ignore]`-gated benchmark
// and intentionally left dead-code when running the normal test binary.

#[allow(dead_code)]
fn fill_q4_0_row(row_bytes: &mut [u8], blocks_per_row: usize, seed: &mut u32) {
    // Cheap deterministic LCG so we get varied-but-reproducible values.
    let mut next = || -> u32 {
        *seed = (*seed).wrapping_mul(1664525).wrapping_add(1013904223);
        *seed
    };
    let mut off = 0;
    for _ in 0..blocks_per_row {
        // d in fp16: uniform in [0.05, 0.5)
        let d: f32 = 0.05 + ((next() >> 8) as f32 / (1u64 << 24) as f32) * 0.45;
        let d_bits = half::f16::from_f32(d).to_bits().to_le_bytes();
        row_bytes[off] = d_bits[0];
        row_bytes[off + 1] = d_bits[1];
        for j in 0..16 {
            // low nibble and high nibble: q values 0..15
            let a = (next() & 0x0F) as u8;
            let b = (next() & 0x0F) as u8;
            row_bytes[off + 2 + j] = a | (b << 4);
        }
        off += BLOCK_SIZE_Q4_0;
    }
}

#[allow(dead_code)]
fn fill_q6_k_row(row_bytes: &mut [u8], blocks_per_row: usize, seed: &mut u32) {
    let mut next = || -> u32 {
        *seed = (*seed).wrapping_mul(1664525).wrapping_add(1013904223);
        *seed
    };
    let mut off = 0;
    for _ in 0..blocks_per_row {
        // Layout matches BlockQ6K::from_bytes: ql[128] qh[64] scales[16] d(f16, 2)
        for j in 0..128 {
            row_bytes[off + j] = (next() & 0xFF) as u8;
        }
        for j in 0..64 {
            row_bytes[off + 128 + j] = (next() & 0xFF) as u8;
        }
        for j in 0..16 {
            // Scales are i8 centered around zero; use -4..=4
            row_bytes[off + 192 + j] = ((next() % 9) as i8 - 4) as u8;
        }
        let d: f32 = 0.05 + ((next() >> 8) as f32 / (1u64 << 24) as f32) * 0.45;
        let d_bits = half::f16::from_f32(d).to_bits().to_le_bytes();
        row_bytes[off + 208] = d_bits[0];
        row_bytes[off + 209] = d_bits[1];
        off += BLOCK_SIZE_Q6_K;
    }
}

/// Returns (raw_q4_0, shape=[out, in]) for a synthetic Q4_0 matrix.
#[allow(dead_code)]
fn make_q4_0_matrix(out_dim: usize, in_dim: usize, seed: u32) -> (Vec<u8>, [usize; 2]) {
    assert_eq!(in_dim % QK4_0, 0, "in_dim must be a multiple of QK4_0=32");
    let blocks_per_row = in_dim / QK4_0;
    let row_bytes_len = blocks_per_row * BLOCK_SIZE_Q4_0;
    let mut bytes = vec![0u8; out_dim * row_bytes_len];
    let mut s = seed;
    for r in 0..out_dim {
        fill_q4_0_row(
            &mut bytes[r * row_bytes_len..(r + 1) * row_bytes_len],
            blocks_per_row,
            &mut s,
        );
    }
    (bytes, [out_dim, in_dim])
}

#[allow(dead_code)]
fn make_q6_k_matrix(out_dim: usize, in_dim: usize, seed: u32) -> (Vec<u8>, [usize; 2]) {
    assert_eq!(in_dim % QK_K, 0, "in_dim must be a multiple of QK_K=256");
    let blocks_per_row = in_dim / QK_K;
    let row_bytes_len = blocks_per_row * BLOCK_SIZE_Q6_K;
    let mut bytes = vec![0u8; out_dim * row_bytes_len];
    let mut s = seed;
    for r in 0..out_dim {
        fill_q6_k_row(
            &mut bytes[r * row_bytes_len..(r + 1) * row_bytes_len],
            blocks_per_row,
            &mut s,
        );
    }
    (bytes, [out_dim, in_dim])
}

#[allow(dead_code)]
fn random_x(n: usize, seed: u64) -> Vec<f32> {
    // Simple xorshift64 for deterministic, varied inputs.
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        // Map to [-1, 1]
        let u = (s >> 33) as f32 / (1u64 << 30) as f32 - 1.0;
        out.push(u);
    }
    out
}

#[allow(dead_code)]
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[allow(dead_code)]
fn make_q4_0_td(raw: Vec<u8>, out_dim: usize, in_dim: usize) -> TensorData {
    TensorData::Q4_0(QuantizedTensor {
        ggml_type: GgmlType::Q4_0,
        shape: vec![in_dim, out_dim], // ggml [in, out] layout
        num_elements: in_dim * out_dim,
        raw_data: raw,
    })
}

#[allow(dead_code)]
fn make_q6_k_td(raw: Vec<u8>, out_dim: usize, in_dim: usize) -> TensorData {
    TensorData::Q6_K(QuantizedTensor {
        ggml_type: GgmlType::Q6_K,
        shape: vec![in_dim, out_dim],
        num_elements: in_dim * out_dim,
        raw_data: raw,
    })
}

#[allow(dead_code)]
fn bench_q4_0_for_threads(
    out_dim: usize,
    in_dim: usize,
    threads: usize,
    iters: usize,
    raw: &[u8],
    _w_shape: &[usize; 2],
    x: &[f32],
    reference_y: &[f32],
    label: &str,
) {
    use std::time::Instant;
    let backend = crate::backend::CpuBackend::with_threads(threads);
    let profiler = Profiler::default();

    // Warm up + correctness check.
    let mut y = vec![0.0f32; out_dim];
    let td = make_q4_0_td(raw.to_vec(), out_dim, in_dim);
    quantized_matvec_dispatch(&backend, &td, x, &mut y, &profiler).expect("warmup failed");
    let err = max_abs_diff(&y, reference_y);

    // Reuse the same raw bytes across iterations; per-iter we must hand the
    // dispatch a fresh TensorData because QuantizedTensor owns its bytes. We
    // do not measure this clone cost (it's a single memcpy of the weight
    // buffer and is not representative of inference, where weights stay
    // resident). In real inference the TensorData is loaded once per layer.
    let mut total = std::time::Duration::ZERO;
    let mut best = std::time::Duration::from_secs(u64::MAX);
    for _ in 0..iters {
        y.fill(0.0);
        let td = make_q4_0_td(raw.to_vec(), out_dim, in_dim);
        let t0 = Instant::now();
        quantized_matvec_dispatch(&backend, &td, x, &mut y, &profiler).expect("bench iter failed");
        let elapsed = t0.elapsed();
        total += elapsed;
        if elapsed < best {
            best = elapsed;
        }
    }
    let avg = total / iters as u32;
    let flops = (out_dim as f64) * (in_dim as f64) * 2.0; // multiplies + adds per output
    let avg_secs = avg.as_secs_f64();
    let gflops = if avg_secs > 0.0 {
        flops / avg_secs / 1e9
    } else {
        0.0
    };
    let weight_mb = raw.len() as f64 / (1024.0 * 1024.0);
    println!(
        "  [Q4_0 {:>18}] threads={:<2} iters={:<3} avg={:>10.3?} best={:>10.3?} throughput={:>7.2} GFLOP/s weight={:>6.2} MiB max|err|={:.2e}",
        label, threads, iters, avg, best, gflops, weight_mb, err
    );
}

#[test]
#[ignore]
fn quantized_matvec_bench() {
    use std::time::Instant;

    // Representative Qwen2.5-1.5B shapes (ggml [in, out] layout; out x in):
    //   * attn_q/k/v/o, ffn_down  : in=1536, out=1536    (small per-layer matvec)
    //   * ffn_gate/up             : in=1536, out=8960    (largest Q4_0 in model)
    //   * output                  : in=1536, out=151936  (vocab projection)
    // Plus small shapes below/around the parallel threshold to verify the
    // scalar fallback is neutral on workloads too small for threading.
    let shapes: &[(&str, usize, usize)] = &[
        ("small(32x1536)", 32, 1536),
        ("small(64x1536)", 64, 1536),
        ("med(128x1536)", 128, 1536),
        ("attn(1536x1536)", 1536, 1536),
        ("ffn(8960x1536)", 8960, 1536),
        ("output(151936x1536)", 151936, 1536),
    ];

    println!("\n=== Q4_0 quantized matvec microbenchmark (scalar baseline + thread scaling) ===");
    for (label, out_dim, in_dim) in shapes {
        let (out_dim, in_dim) = (*out_dim, *in_dim);
        let (raw, _w_shape) = make_q4_0_matrix(
            out_dim,
            in_dim,
            0x12345678 ^ (out_dim as u32).wrapping_mul(31) ^ (in_dim as u32),
        );
        let backend = crate::backend::CpuBackend::scalar();
        let profiler = Profiler::default();
        let x = random_x(in_dim, 0xC0FFEE);
        let mut y_ref = vec![0.0f32; out_dim];
        let ref_td = make_q4_0_td(raw.clone(), out_dim, in_dim);
        quantized_matvec_dispatch(&backend, &ref_td, &x, &mut y_ref, &profiler)
            .expect("reference failed");

        let iters = if out_dim >= 100_000 {
            5
        } else if out_dim >= 4000 {
            20
        } else if out_dim >= 128 {
            50
        } else {
            200
        };
        bench_q4_0_for_threads(
            out_dim,
            in_dim,
            1,
            iters,
            &raw,
            &[out_dim, in_dim],
            &x,
            &y_ref,
            label,
        );
        for t in [2, 4, 8, 9] {
            bench_q4_0_for_threads(
                out_dim,
                in_dim,
                t,
                iters,
                &raw,
                &[out_dim, in_dim],
                &x,
                &y_ref,
                label,
            );
        }
    }

    println!("\n=== Q6_K quantized matvec microbenchmark (scalar + parallel) ===");
    for (label, out_dim, in_dim) in shapes {
        let out_dim = *out_dim;
        let in_dim = if *in_dim % QK_K == 0 {
            *in_dim
        } else {
            (*in_dim + QK_K - 1) / QK_K * QK_K
        };
        let (raw, _w_shape) = make_q6_k_matrix(
            out_dim,
            in_dim,
            0xA5A5A5 ^ (out_dim as u32).wrapping_mul(131) ^ (in_dim as u32),
        );
        let backend = crate::backend::CpuBackend::scalar();
        let profiler = Profiler::default();
        let x = random_x(in_dim, 0xDEADBEEF);
        let mut y_ref = vec![0.0f32; out_dim];
        let ref_td = make_q6_k_td(raw.clone(), out_dim, in_dim);
        quantized_matvec_dispatch(&backend, &ref_td, &x, &mut y_ref, &profiler)
            .expect("q6_k reference failed");

        let iters = if out_dim >= 100_000 {
            5
        } else if out_dim >= 4000 {
            20
        } else if out_dim >= 128 {
            50
        } else {
            200
        };
        for threads in [1, 2, 4, 8, 9] {
            let be = crate::backend::CpuBackend::with_threads(threads);
            let mut y = vec![0.0f32; out_dim];
            let td = make_q6_k_td(raw.clone(), out_dim, in_dim);
            quantized_matvec_dispatch(&be, &td, &x, &mut y, &profiler).expect("q6_k warmup failed");
            let err = max_abs_diff(&y, &y_ref);
            let mut total = std::time::Duration::ZERO;
            let mut best = std::time::Duration::from_secs(u64::MAX);
            for _ in 0..iters {
                y.fill(0.0);
                let td = make_q6_k_td(raw.clone(), out_dim, in_dim);
                let t0 = Instant::now();
                quantized_matvec_dispatch(&be, &td, &x, &mut y, &profiler)
                    .expect("q6_k iter failed");
                let elapsed = t0.elapsed();
                total += elapsed;
                if elapsed < best {
                    best = elapsed;
                }
            }
            let avg = total / iters as u32;
            let flops = (out_dim as f64) * (in_dim as f64) * 2.0;
            let avg_secs = avg.as_secs_f64();
            let gflops = if avg_secs > 0.0 {
                flops / avg_secs / 1e9
            } else {
                0.0
            };
            let weight_mb = raw.len() as f64 / (1024.0 * 1024.0);
            println!(
                "  [Q6_K {:>18}] threads={:<2} iters={:<3} avg={:>10.3?} best={:>10.3?} throughput={:>7.2} GFLOP/s weight={:>6.2} MiB max|err|={:.2e}",
                label, threads, iters, avg, best, gflops, weight_mb, err
            );
        }
    }
    println!();
}
