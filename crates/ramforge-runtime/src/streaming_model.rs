//! Streaming LLaMA model – out-of-core layer streaming with quantized support
//!
//! Only persistent weights (token_embd, output_norm, output) are loaded initially.
//! Transformer layers are loaded on demand, one at a time, and released after use.
//! Quantized tensors remain quantized while resident; dequantization happens block-wise during matvec.

use ramforge_core::{
    datasource::GgufDataSource,
    memory::MemoryBudget,
    tensor::TensorData,
    types::GgmlType,
};

use crate::backend::ComputeBackend;
use crate::kv_cache::KvCache;
use crate::layer::{group_layers, LayerDescriptor, PersistentDescriptors};
use crate::model::{validate_required_tensors, LlamaConfig};
use crate::persistent::{PersistentWeight, should_keep_resident};
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
}

impl StreamingLayerWeights {
    pub fn total_resident_bytes(&self) -> u64 {
        self.attn_norm.resident_bytes() as u64
            + self.attn_q.resident_bytes() as u64
            + self.attn_k.resident_bytes() as u64
            + self.attn_v.resident_bytes() as u64
            + self.attn_output.resident_bytes() as u64
            + self.ffn_norm.resident_bytes() as u64
            + self.ffn_gate.resident_bytes() as u64
            + self.ffn_up.resident_bytes() as u64
            + self.ffn_down.resident_bytes() as u64
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
}

impl StreamingLlamaModel {
    /// Load persistent weights only if they fit comfortably, otherwise stream
    pub fn load(
        data_source: &GgufDataSource,
        budget: &mut MemoryBudget,
    ) -> Result<Self, String> {
        let gguf_model = data_source.model();
        let config = LlamaConfig::from_gguf(gguf_model)?;

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

        // Helper to load persistent tensor with streaming policy
        let mut load_persistent = |name: &str| -> Result<PersistentWeight, String> {
            let desc = data_source
                .get_descriptor(name)
                .map_err(|e| format!("tensor '{}' not found: {}", name, e))?
                .clone();

            let tensor_bytes = desc.byte_length.unwrap_or(0);

            if should_keep_resident(tensor_bytes, budget.total_bytes()) {
                // Keep resident: direct file read (no cache duplication).
                let raw_bytes = data_source
                    .read_tensor(name)
                    .map_err(|e| format!("failed to read tensor '{}': {}", name, e))?;

                let shape_u64 = desc.dimensions.clone();
                let tensor_data = TensorData::from_bytes(
                    desc.ggml_type,
                    shape_u64,
                    desc.num_elements,
                    raw_bytes,
                )
                .map_err(|e| format!("failed to create TensorData for '{}': {}", name, e))?;

                let resident = tensor_data.resident_bytes() as u64;
                let alloc_name = format!("weight:{}", name);
                if budget.get(&alloc_name).is_none() {
                    budget
                        .allocate(alloc_name, resident)
                        .map_err(|e| format!("RAM budget exceeded loading '{}': {}", name, e))?;
                }

                Ok(PersistentWeight::Resident(tensor_data))
            } else {
                // Stream on demand – no resident allocation
                Ok(PersistentWeight::Streamed(desc))
            }
        };

        let token_embd = load_persistent("token_embd.weight")?;
        let output_norm = load_persistent("output_norm.weight")?;
        let output = if gguf_model.tensors.iter().any(|t| t.name == "output.weight") {
            Some(load_persistent("output.weight")?)
        } else {
            None
        };

        let layer_descriptors = group_layers(gguf_model, config.block_count);
        let persistent_descriptors = PersistentDescriptors::from_model(gguf_model);

        Ok(Self {
            config,
            token_embd,
            output_norm,
            output,
            layer_descriptors,
            persistent_descriptors,
            total_weight_bytes,
            quantized_weight_bytes,
        })
    }

    /// Load a single layer on demand, accounting actual quantized resident size
    ///
    /// Memory integrity (M6): each tensor's file bytes are charged to the
    /// budget BEFORE reading (covering the read buffer; float types charge an
    /// extra byte_length to cover the raw+f32 decode spike). After
    /// construction the charge is settled to the exact resident size. On any
    /// failure, all charges made for this layer are released.
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
                // Peak during from_bytes: quantized types move the raw buffer
                // (1x), float types decode raw->f32 (2x). Charge the peak.
                let is_float = matches!(
                    tensor_desc.ggml_type,
                    GgmlType::F32 | GgmlType::F16 | GgmlType::BF16
                );
                let charge = if is_float { file_bytes * 2 } else { file_bytes };
                let alloc_name = format!("layer:{}:{}", layer_idx, name);
                budget
                    .allocate(alloc_name.clone(), charge.max(1))
                    .map_err(|e| format!("RAM budget too small for layer {} tensor '{}': {}", layer_idx, name, e))?;

                let raw_bytes = data_source
                    .read_tensor(name)
                    .map_err(|e| format!("failed to read tensor '{}': {}", name, e))?;

                let shape_u64 = tensor_desc.dimensions.clone();
                let tensor_data = TensorData::from_bytes(
                    tensor_desc.ggml_type,
                    shape_u64,
                    tensor_desc.num_elements,
                    raw_bytes,
                )
                .map_err(|e| format!("failed to create TensorData for '{}': {}", name, e))?;

                let resident = tensor_data.resident_bytes() as u64;
                total_layer_bytes += resident;

                // Settle the charge to the exact resident size.
                if resident != charge {
                    let _ = budget.release(&alloc_name);
                    budget
                        .allocate(alloc_name, resident.max(1))
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

        let mut get = |suffix: &str| -> Result<TensorData, String> {
            let full = format!("blk.{}.{}", layer_idx, suffix);
            map.remove(&full)
                .ok_or_else(|| format!("missing tensor '{}' in loaded layer {}", full, layer_idx))
        };

        Ok(StreamingLayerWeights {
            attn_norm: get("attn_norm.weight")?,
            attn_q: get("attn_q.weight")?,
            attn_k: get("attn_k.weight")?,
            attn_v: get("attn_v.weight")?,
            attn_output: get("attn_output.weight")?,
            ffn_norm: get("ffn_norm.weight")?,
            ffn_gate: get("ffn_gate.weight")?,
            ffn_up: get("ffn_up.weight")?,
            ffn_down: get("ffn_down.weight")?,
        })
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

    /// Forward pass for one token with out-of-core layer streaming.
    ///
    /// M6 memory contract:
    /// - one scoped `tmp:forward` reservation covers every transient
    ///   activation buffer (allocated once per forward call, worst-case size);
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
    ) -> Result<Vec<f32>, String> {
        let cfg = &self.config;
        let n_embd = cfg.embedding_length;
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
        //   attn_proj + ffn_out + final_hidden + output_norm copy: 4 * n_embd
        //   gate + up + gate_silu + gate_up:     4 * ffn_dim
        //   attention scores (per head):         n_heads * seq
        let act_floats = 8 * n_embd + 2 * q_dim + 2 * kv_dim + 4 * ffn_dim + n_heads * seq;
        let act_bytes = (act_floats * 4) as u64;

        budget.with_temp("tmp:forward", act_bytes, |budget| {
            // Embedding lookup (streams one row if non-resident; charges its
            // own `tmp:embd_row` inside this scope).
            let mut hidden = self.token_embd.get_embedding(
                token_id as usize,
                n_embd,
                data_source,
                budget,
                stats,
            )?;

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

            for layer_idx in 0..cfg.block_count {
                let layer = self.load_layer(layer_idx, data_source, budget, stats)?;

                // The layer always gets released – also on the error path.
                let layer_result = Self::forward_layer(
                    &layer,
                    layer_idx,
                    pos,
                    kv_cache,
                    backend,
                    cfg,
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
                self.release_layer(layer_idx, budget, stats);
                layer_result?;
            }

            kv_cache.increment_seq_len();

            let output_norm_f32 = self
                .output_norm
                .to_f32_vec(data_source)
                .map_err(|e| e.to_string())?;
            let mut final_hidden = vec![0.0f32; n_embd];
            backend.rmsnorm(&hidden, &output_norm_f32, cfg.rms_eps, &mut final_hidden);

            Ok(final_hidden)
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
        let attn_norm_f32 = layer
            .attn_norm
            .to_f32_vec()
            .map_err(|e| format!("failed to decode attn_norm of layer {}: {}", layer_idx, e))?;
        backend.rmsnorm(hidden, &attn_norm_f32, cfg.rms_eps, tmp);

        matvec_backend(backend, &layer.attn_q, tmp, q_tmp)?;
        matvec_backend(backend, &layer.attn_k, tmp, k_tmp)?;
        matvec_backend(backend, &layer.attn_v, tmp, v_tmp)?;

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

        matvec_backend(backend, &layer.attn_output, &attn_out, attn_proj)?;
        for i in 0..n_embd {
            hidden[i] += attn_proj[i];
        }

        // ffn_norm
        let ffn_norm_f32 = layer
            .ffn_norm
            .to_f32_vec()
            .map_err(|e| format!("failed to decode ffn_norm of layer {}: {}", layer_idx, e))?;
        backend.rmsnorm(hidden, &ffn_norm_f32, cfg.rms_eps, tmp);

        matvec_backend(backend, &layer.ffn_gate, tmp, gate)?;
        matvec_backend(backend, &layer.ffn_up, tmp, up)?;

        backend.silu(gate, gate_silu);
        backend.mul(gate_silu, up, gate_up);

        matvec_backend(backend, &layer.ffn_down, gate_up, ffn_out)?;
        for i in 0..n_embd {
            hidden[i] += ffn_out[i];
        }

        Ok(())
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

        let weight = self.output.as_ref().unwrap_or(&self.token_embd);
        match weight {
            PersistentWeight::Resident(td) => matvec_backend(backend, td, hidden, logits_out),
            PersistentWeight::Streamed(_) => {
                weight.compute_logits_into(hidden, data_source, budget, logits_out)
            }
        }
    }
}

/// Matvec dispatch under the single explicit ggml layout (`shape = [in, out]`):
/// resident F32 data goes through the SIMD/threaded compute backend; all other
/// types use the compact block-wise kernels (no full F32 expansion, no
/// orientation guessing).
fn matvec_backend(
    backend: &dyn ComputeBackend,
    td: &TensorData,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    if let Some((data, shape)) = td.as_f32_slice() {
        backend.matvec(data, shape, x, y)
    } else {
        td.matvec(x, y).map_err(|e| e.to_string())
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
    fn test_out_of_core_model_larger_than_budget() {
        let tmp = create_model_with_n_layers(8, 32, 64);
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let total_bytes: u64 = ds.model().tensors.iter().filter_map(|t| t.byte_length).sum();

        // Per layer ~41 KiB; charge-before-read peaks at ~2x for F32 tensors,
        // plus persistents (~2.1 KiB). 96 KiB fits one layer comfortably; the
        // whole model (~324 KiB) never has to fit at once.
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
}
