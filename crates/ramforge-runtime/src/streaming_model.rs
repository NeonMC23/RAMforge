//! Streaming LLaMA model – out-of-core layer streaming with quantized support
//!
//! Only persistent weights (token_embd, output_norm, output) are loaded initially.
//! Transformer layers are loaded on demand, one at a time, and released after use.
//! Quantized tensors remain quantized while resident; dequantization happens block-wise during matvec.

use std::sync::Mutex;

use ramforge_core::{
    datasource::GgufDataSource,
    memory::MemoryBudget,
    tensor::TensorData,
    types::GgmlType,
};

use crate::accounting::{estimate_layer_memory, tensor_load_charge_bytes, LayerMemoryEstimate};
use crate::backend::ComputeBackend;
use crate::kv_cache::KvCache;
use crate::layer_cache::{InsertOutcome, LayerCache};
use crate::layer::{group_layers, LayerDescriptor, PersistentDescriptors};
use crate::model::{validate_required_tensors, LlamaConfig};
use crate::persistent::{row_bytes_for, PersistentWeight, should_keep_resident};
use crate::profile::{ProfileEvent, Profiler};
use crate::residency::ResidencyStats;

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
    layer_cache: Mutex<LayerCache<StreamingLayerWeights>>,
    pub(crate) profiler: Profiler,
}

impl StreamingLlamaModel {
    /// Load persistent weights only if they fit comfortably, otherwise stream
    pub fn load(
        data_source: &GgufDataSource,
        budget: &mut MemoryBudget,
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
            .filter(|t| matches!(t.ggml_type, GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q2_K | GgmlType::Q3_K | GgmlType::Q8_K))
            .filter_map(|t| t.byte_length)
            .sum();
        let layer_descriptors = group_layers(gguf_model, config.block_count);
        let layer_memory_estimates = layer_descriptors
            .iter()
            .map(|layer| {
                estimate_layer_memory(&layer.tensors)
                    .map_err(|error| format!("failed to estimate layer {} memory: {}", layer.layer_idx, error))
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
        let layer_cache_capacity = budget.total_bytes().saturating_sub(managed_lower_bound);
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
            layer_cache: Mutex::new(LayerCache::new(layer_cache_capacity)),
            profiler,
        })
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

        let result = (|budget: &mut MemoryBudget| -> Result<(), String> {
            for tensor_desc in &layer_desc.tensors {
                let name = &tensor_desc.name;
                let file_bytes = tensor_desc.byte_length.unwrap_or(0);
                // Peak owned representation during load, charged up front.
                let charge = tensor_load_charge_bytes(tensor_desc.ggml_type, file_bytes)?.max(1);
                let alloc_name = format!("layer:{}:{}", layer_idx, name);
                budget
                    .allocate(alloc_name.clone(), charge)
                    .map_err(|e| format!("RAM budget too small for layer {} tensor '{}': {}", layer_idx, name, e))?;

                let tensor_data = load_tensor_data(data_source, tensor_desc, &self.profiler)?;

                let resident = tensor_data.resident_bytes() as u64;
                total_layer_bytes += resident;

                // Atomically settle the charge to the exact resident size;
                // the allocation name remains live while TensorData exists.
                if resident.max(1) != charge {
                    budget
                        .resize(&alloc_name, resident.max(1))
                        .map_err(|e| format!("RAM budget error settling layer {} tensor '{}': {}", layer_idx, name, e))?;
                }

                loaded.push((name.clone(), tensor_data));
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
                map.remove(&full)
                    .ok_or_else(|| format!("missing tensor '{}' in loaded layer {}", full, layer_idx))
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
    pub fn forward_single_streaming(
        &self,
        token_id: u32,
        pos: usize,
        kv_cache: &mut KvCache,
        backend: &dyn ComputeBackend,
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
        //   hidden + tmp + 2 norm-weight copies: 4 * n_embd
        //   q_tmp + attention output:            2 * q_dim
        //   k_tmp + v_tmp:                       2 * kv_dim
        //   attn_proj + ffn_out + output_norm copy: 3 * n_embd
        //   gate + up + gate_silu + gate_up:     4 * ffn_dim
        //   attention scores (per head):         n_heads * seq
        //   decoded qwen2 Q/K/V bias vectors (per layer, if present)
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
            self.ensure_layer_cache_headroom(
                budget,
                self.embedding_workspace_bytes(n_embd)?,
            )?;
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

            kv_cache.increment_seq_len();

            let dequant_started = self.profiler.start();
            let output_norm_f32 = self
                .output_norm
                .to_f32_vec(data_source)
                .map_err(|e| e.to_string())?;
            self.profiler
                .record_since(ProfileEvent::Dequantization, dequant_started);
            backend.rmsnorm(&hidden, &output_norm_f32, cfg.rms_eps, final_hidden);

            Ok(())
        })
    }

    /// One transformer block over pre-allocated scratch buffers.
    /// All matvecs honor the explicit ggml layout of `layer` tensors.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer(
        layer: &StreamingLayerWeights,
        layer_idx: usize,
        pos: usize,
        kv_cache: &mut KvCache,
        backend: &dyn ComputeBackend,
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
        let attn_norm_f32 = layer
            .attn_norm
            .to_f32_vec()
            .map_err(|e| format!("failed to decode attn_norm of layer {}: {}", layer_idx, e))?;
        profiler.record_since(ProfileEvent::Dequantization, dequant_started);
        backend.rmsnorm(hidden, &attn_norm_f32, cfg.rms_eps, tmp);

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
                let bq = bq
                    .to_f32_vec()
                    .map_err(|e| format!("failed to decode attn_q.bias of layer {}: {}", layer_idx, e))?;
                let bk = bk
                    .to_f32_vec()
                    .map_err(|e| format!("failed to decode attn_k.bias of layer {}: {}", layer_idx, e))?;
                let bv = bv
                    .to_f32_vec()
                    .map_err(|e| format!("failed to decode attn_v.bias of layer {}: {}", layer_idx, e))?;
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

        kv_cache.append(layer_idx, k_tmp, v_tmp)?;

        // History holds `seq_len` previous tokens; the freshly appended
        // current-token K/V is passed separately so attention can read the
        // cache prefix in place (no concatenated copy).
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
        let ffn_norm_f32 = layer
            .ffn_norm
            .to_f32_vec()
            .map_err(|e| format!("failed to decode ffn_norm of layer {}: {}", layer_idx, e))?;
        profiler.record_since(ProfileEvent::Dequantization, dequant_started);
        backend.rmsnorm(hidden, &ffn_norm_f32, cfg.rms_eps, tmp);

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
    pub fn compute_logits(
        &self,
        hidden: &[f32],
        backend: &dyn ComputeBackend,
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
    let expected_resident = TensorData::resident_bytes_for(
        desc.ggml_type,
        desc.num_elements,
        file_bytes,
    )
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

/// Matvec dispatch under the single explicit ggml layout (`shape = [in, out]`):
/// resident F32 data goes through the SIMD/threaded compute backend; all other
/// types use the compact block-wise kernels (no full F32 expansion, no
/// orientation guessing).
fn matvec_backend(
    backend: &dyn ComputeBackend,
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
        let result = td.matvec(x, y).map_err(|e| e.to_string());
        profiler.record_since(ProfileEvent::QuantizedMatvec, started);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ramforge_core::model::align_offset;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Boxed value-writer closure used by the GGUF test fixtures.
    type WriteValFn<'a> = Box<dyn FnMut(&mut Vec<u8>) + 'a>;

    fn write_string<W: Write>(w: &mut W, s: &str) {
        w.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
        w.write_all(s.as_bytes()).unwrap();
    }
    fn write_u32<W: Write>(w: &mut W, v: u32) { w.write_all(&v.to_le_bytes()).unwrap(); }
    fn write_u64<W: Write>(w: &mut W, v: u64) { w.write_all(&v.to_le_bytes()).unwrap(); }
    fn write_f32<W: Write>(w: &mut W, v: f32) { w.write_all(&v.to_le_bytes()).unwrap(); }

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

    fn create_model_with_n_layers(n_layers: usize, n_embd: usize, ffn: usize) -> NamedTempFile {
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
        add_kv("general.architecture", 8, Box::new(|b| write_string(b, "llama")));
        add_kv("llama.vocab_size", 4, Box::new(|b| write_u32(b, 16)));
        add_kv("llama.context_length", 4, Box::new(|b| write_u32(b, 64)));
        add_kv("llama.embedding_length", 4, Box::new(|b| write_u32(b, n_embd as u32)));
        add_kv("llama.block_count", 4, Box::new(|b| write_u32(b, n_layers as u32)));
        add_kv("llama.feed_forward_length", 4, Box::new(|b| write_u32(b, ffn as u32)));
        add_kv("llama.attention.head_count", 4, Box::new(|b| write_u32(b, 2)));
        add_kv("llama.attention.head_count_kv", 4, Box::new(|b| write_u32(b, 2)));
        add_kv("llama.attention.layer_norm_rms_epsilon", 6, Box::new(|b| write_f32(b, 1e-5)));
        add_kv("llama.rope.freq_base", 6, Box::new(|b| write_f32(b, 10000.0)));
        add_kv("tokenizer.ggml.model", 8, Box::new(|b| write_string(b, "llama")));

        let mut offset = 0u64;
        let mut defs: Vec<(String, Vec<u64>, u32)> = Vec::new();
        defs.push(("token_embd.weight".to_string(), vec![n_embd as u64, 16], 0));
        defs.push(("output_norm.weight".to_string(), vec![n_embd as u64], 0));
        for i in 0..n_layers {
            defs.push((format!("blk.{}.attn_norm.weight", i), vec![n_embd as u64], 0));
            defs.push((format!("blk.{}.attn_q.weight", i), vec![n_embd as u64, n_embd as u64], 0));
            defs.push((format!("blk.{}.attn_k.weight", i), vec![n_embd as u64, n_embd as u64], 0));
            defs.push((format!("blk.{}.attn_v.weight", i), vec![n_embd as u64, n_embd as u64], 0));
            defs.push((format!("blk.{}.attn_output.weight", i), vec![n_embd as u64, n_embd as u64], 0));
            defs.push((format!("blk.{}.ffn_norm.weight", i), vec![n_embd as u64], 0));
            // ggml layout [in, out]: gate/up map n_embd -> ffn, down maps ffn -> n_embd
            defs.push((format!("blk.{}.ffn_gate.weight", i), vec![n_embd as u64, ffn as u64], 0));
            defs.push((format!("blk.{}.ffn_up.weight", i), vec![n_embd as u64, ffn as u64], 0));
            defs.push((format!("blk.{}.ffn_down.weight", i), vec![ffn as u64, n_embd as u64], 0));
        }

        for (name, dims, ty) in &defs {
            write_string(&mut buf, name);
            write_u32(&mut buf, dims.len() as u32);
            for d in dims { write_u64(&mut buf, *d); }
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
        add_kv("general.architecture", 8, Box::new(|b| write_string(b, "llama")));
        add_kv("llama.vocab_size", 4, Box::new(|b| write_u32(b, 16)));
        add_kv("llama.context_length", 4, Box::new(|b| write_u32(b, 64)));
        add_kv("llama.embedding_length", 4, Box::new(|b| write_u32(b, 8)));
        add_kv("llama.block_count", 4, Box::new(|b| write_u32(b, 1)));
        add_kv("llama.feed_forward_length", 4, Box::new(|b| write_u32(b, 16)));
        add_kv("llama.attention.head_count", 4, Box::new(|b| write_u32(b, 2)));
        add_kv("llama.attention.head_count_kv", 4, Box::new(|b| write_u32(b, 2)));
        add_kv("llama.attention.layer_norm_rms_epsilon", 6, Box::new(|b| write_f32(b, 1e-5)));
        add_kv("llama.rope.freq_base", 6, Box::new(|b| write_f32(b, 10000.0)));
        add_kv("tokenizer.ggml.model", 8, Box::new(|b| write_string(b, "llama")));

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
        defs.push(("blk.0.attn_output.weight".into(), vec![n_embd, n_embd], 0, 4));
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
        add_kv("general.architecture", 8, Box::new(|b| write_string(b, "qwen2")));
        add_kv("qwen2.vocab_size", 4, Box::new(|b| write_u32(b, 16)));
        add_kv("qwen2.context_length", 4, Box::new(|b| write_u32(b, 64)));
        add_kv("qwen2.embedding_length", 4, Box::new(|b| write_u32(b, 8)));
        add_kv("qwen2.block_count", 4, Box::new(|b| write_u32(b, 1)));
        add_kv("qwen2.feed_forward_length", 4, Box::new(|b| write_u32(b, 16)));
        add_kv("qwen2.attention.head_count", 4, Box::new(|b| write_u32(b, 2)));
        add_kv("qwen2.attention.head_count_kv", 4, Box::new(|b| write_u32(b, 2)));
        add_kv("qwen2.attention.layer_norm_rms_epsilon", 6, Box::new(|b| write_f32(b, 1e-5)));
        add_kv("qwen2.rope.freq_base", 6, Box::new(|b| write_f32(b, 10000.0)));
        add_kv("tokenizer.ggml.model", 8, Box::new(|b| write_string(b, "llama")));

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
    fn test_persistent_f32_direct_load_uses_one_owned_representation() {
        let raw = vec![0u8; 8 * 4];
        let tmp = create_single_tensor_gguf("test.weight", GgmlType::F32, &[8], &raw);
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let desc = ds.get_descriptor("test.weight").unwrap();
        assert_eq!(desc.byte_length, Some(32));
        assert_eq!(TensorData::resident_bytes_for(GgmlType::F32, 8, 32).unwrap(), 32);
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
        let error = load_persistent_weight(&ds, "test.weight", &mut tight, &Profiler::default()).unwrap_err();
        assert!(error.contains("load charge"), "unexpected error: {}", error);
        assert_eq!(tight.used_bytes(), before);
        assert!(tight.get("weight:test.weight").is_none());
        assert!(!tight.allocations().keys().any(|name| name.starts_with("tmp:")));
    }

    #[test]
    fn test_persistent_f16_transient_and_settled_accounting() {
        let raw: Vec<u8> = (0..8).flat_map(|_| 0x3C00u16.to_le_bytes()).collect();
        let tmp = create_single_tensor_gguf("test.weight", GgmlType::F16, &[8], &raw);
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let file_bytes = ds.get_descriptor("test.weight").unwrap().byte_length.unwrap();
        assert_eq!(file_bytes, 8 * 2);
        assert_eq!(tensor_load_charge_bytes(GgmlType::F16, file_bytes).unwrap(), 3 * file_bytes);
        assert_eq!(TensorData::resident_bytes_for(GgmlType::F16, 8, file_bytes).unwrap(), 8 * 4);

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
        let file_bytes = ds.get_descriptor("test.weight").unwrap().byte_length.unwrap();
        assert_eq!(file_bytes, 8 * 2);
        assert_eq!(tensor_load_charge_bytes(GgmlType::BF16, file_bytes).unwrap(), 3 * file_bytes);
        assert_eq!(TensorData::resident_bytes_for(GgmlType::BF16, 8, file_bytes).unwrap(), 8 * 4);

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
        let file_bytes = ds.get_descriptor("test.weight").unwrap().byte_length.unwrap();
        assert_eq!(file_bytes, 18);
        assert_eq!(tensor_load_charge_bytes(GgmlType::Q4_0, file_bytes).unwrap(), file_bytes);
        assert_eq!(TensorData::resident_bytes_for(GgmlType::Q4_0, 32, file_bytes).unwrap(), file_bytes);

        let mut budget = MemoryBudget::new(72).unwrap();
        let (weight, allocation) =
            load_persistent_weight(&ds, "test.weight", &mut budget, &Profiler::default()).unwrap();
        assert!(weight.is_resident());
        assert_eq!(weight.resident_bytes(), 18);
        match &weight {
            PersistentWeight::Resident(tensor) => assert!(tensor.is_quantized()),
            PersistentWeight::Streamed(_) => panic!("Q4_0 tensor should fit at the policy boundary"),
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
        let f32_tmp =
            create_single_tensor_gguf("test.weight", GgmlType::F32, &[8], &f32_raw);
        let f32_ds =
            ramforge_core::datasource::GgufDataSource::open(f32_tmp.path()).unwrap();
        let mut f32_budget = MemoryBudget::new(128).unwrap();
        let (f32_weight, f32_allocation) =
            load_persistent_weight(&f32_ds, "test.weight", &mut f32_budget, &Profiler::default()).unwrap();
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
                load_persistent_weight(&ds, "test.weight", &mut budget, &Profiler::default()).unwrap();
            assert!(weight.is_streamed(), "{} should be streamed", ggml_type.name());
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
        assert!(!budget.allocations().keys().any(|name| name.starts_with("weight:")));
        assert!(!budget.allocations().keys().any(|name| name.starts_with("tmp:")));
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
        assert!(!budget.allocations().keys().any(|name| name.starts_with("tmp:")));
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
        assert!(!budget.allocations().keys().any(|name| name.starts_with("tmp:")));
    }

    #[test]
    fn test_out_of_core_model_larger_than_budget() {
        let tmp = create_model_with_n_layers(8, 32, 64);
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let total_bytes: u64 = ds.model().tensors.iter().filter_map(|t| t.byte_length).sum();

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
        add_kv("general.architecture", 8, Box::new(|b| write_string(b, "llama")));
        add_kv("llama.vocab_size", 4, Box::new(|b| write_u32(b, 16)));
        add_kv("llama.context_length", 4, Box::new(|b| write_u32(b, 64)));
        add_kv("llama.embedding_length", 4, Box::new(|b| write_u32(b, n_embd as u32)));
        add_kv("llama.block_count", 4, Box::new(|b| write_u32(b, n_layers as u32)));
        add_kv("llama.feed_forward_length", 4, Box::new(|b| write_u32(b, ffn as u32)));
        add_kv("llama.attention.head_count", 4, Box::new(|b| write_u32(b, 2)));
        add_kv("llama.attention.head_count_kv", 4, Box::new(|b| write_u32(b, 2)));
        add_kv("llama.attention.layer_norm_rms_epsilon", 6, Box::new(|b| write_f32(b, 1e-5)));
        add_kv("llama.rope.freq_base", 6, Box::new(|b| write_f32(b, 10000.0)));
        add_kv("tokenizer.ggml.model", 8, Box::new(|b| write_string(b, "llama")));

        let mut offset = 0u64;
        let mut defs: Vec<(String, Vec<u64>, u32)> = Vec::new();
        // Use Q4_0 for some tensors: type 2
        defs.push(("token_embd.weight".to_string(), vec![n_embd as u64, 16], 0)); // F32 for simplicity
        defs.push(("output_norm.weight".to_string(), vec![n_embd as u64], 0));
        for i in 0..n_layers {
            defs.push((format!("blk.{}.attn_norm.weight", i), vec![n_embd as u64], 0));
            defs.push((format!("blk.{}.attn_q.weight", i), vec![n_embd as u64, n_embd as u64], 2)); // Q4_0
            defs.push((format!("blk.{}.attn_k.weight", i), vec![n_embd as u64, n_embd as u64], 2));
            defs.push((format!("blk.{}.attn_v.weight", i), vec![n_embd as u64, n_embd as u64], 2));
            defs.push((format!("blk.{}.attn_output.weight", i), vec![n_embd as u64, n_embd as u64], 2));
            defs.push((format!("blk.{}.ffn_norm.weight", i), vec![n_embd as u64], 0));
            // ggml layout [in, out]: gate/up map n_embd -> ffn, down maps ffn -> n_embd
            defs.push((format!("blk.{}.ffn_gate.weight", i), vec![n_embd as u64, ffn as u64], 2));
            defs.push((format!("blk.{}.ffn_up.weight", i), vec![n_embd as u64, ffn as u64], 2));
            defs.push((format!("blk.{}.ffn_down.weight", i), vec![ffn as u64, n_embd as u64], 2));
        }

        for (name, dims, ty) in &defs {
            write_string(&mut buf, name);
            write_u32(&mut buf, dims.len() as u32);
            for d in dims { write_u64(&mut buf, *d); }
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
        for _ in 0..16*n_embd { buf.extend_from_slice(&1.0f32.to_le_bytes()); }
        // output_norm
        for _ in 0..n_embd { buf.extend_from_slice(&1.0f32.to_le_bytes()); }
        for _ in 0..n_layers {
            // attn_norm F32
            for _ in 0..n_embd { buf.extend_from_slice(&1.0f32.to_le_bytes()); }
            // Q4_0 tensors: each 32 elements => 18 bytes per block
            // n_embd 32 => 1 block per row? For [32,32] => 32 rows * 18 =576 bytes per tensor
            // We'll write zeros for simplicity: d=1.0, qs=0x88 (dequant 0)
            for _ in 0..4 { // 4 tensors * 576?
                for _ in 0..n_embd {
                    let d_fp16: u16 = 0x3C00;
                    buf.extend_from_slice(&d_fp16.to_le_bytes());
                    buf.extend_from_slice(&[0x88; 16]);
                }
            }
            // ffn_norm
            for _ in 0..n_embd { buf.extend_from_slice(&1.0f32.to_le_bytes()); }
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
        assert!(!budget.allocations().keys().any(|k| k.starts_with("layer:0:")));
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
        assert!(!budget.allocations().keys().any(|k| k.starts_with("layer:0:")));
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
        assert!(!budget.allocations().keys().any(|k| k.starts_with("layer:0:")));
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
}
