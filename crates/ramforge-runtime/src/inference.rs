//! Inference engine for RAMforge (CPU, llama/qwen2, out-of-core).
//!
//! Milestone 6 memory contract for `generate()`:
//! - every allocation that lives across a step is budget-charged for its
//!   lifetime: persistent weights (`weight:*`), streamed layers
//!   (`layer:{i}:*`), the KV cache (`kv_cache`), the single logits buffer
//!   plus the hidden state (`tmp:logits`, one allocation for the whole call);
//! - short-lived working sets use scoped guards: `tmp:forward` (per-token
//!   activations), `tmp:embd_row` (streamed embedding row),
//!   `tmp:streamed_matvec` (chunked streamed projection), `tmp:sampling`
//!   (sampler scratch for non-greedy sampling);
//! - the KV cache starts at the prompt length and grows chunk-wise
//!   (256-token chunks, capped at prompt+max_tokens) with budget checks
//!   and deterministic rollback on failure.

use ramforge_core::{
    datasource::GgufDataSource,
    memory::MemoryBudget,
    tokenizer::Tokenizer,
};

use crate::backend::CpuBackend;
use crate::kv_cache::KvCache;
use crate::model::LlamaConfig;
use crate::sampling::Sampler;
use crate::streaming_model::StreamingLlamaModel;
use crate::residency::ResidencyStats;

#[derive(Debug)]
pub struct InferenceEngine {
    pub data_source: GgufDataSource,
    pub tokenizer: Tokenizer,
    pub model: StreamingLlamaModel,
    pub kv_cache: Option<KvCache>,
    pub backend: CpuBackend,
    pub budget: MemoryBudget,
    pub ram_budget_bytes: u64,
    pub residency_stats: ResidencyStats,
}

impl InferenceEngine {
    pub fn new(
        model_path: &str,
        ram_budget_bytes: u64,
    ) -> Result<Self, String> {
        // Parse model file-backed (does NOT load tensor payloads)
        let data_source = GgufDataSource::open(model_path)
            .map_err(|e| format!("failed to open GGUF data source: {}", e))?;

        let gguf_model = data_source.model();
        // Validate architecture via config (will error if unsupported)
        let _ = LlamaConfig::from_gguf(gguf_model)?;

        // Tokenizer
        let tokenizer = Tokenizer::from_gguf(gguf_model)
            .map_err(|e| format!("failed to load tokenizer: {}", e))?;

        // Budget – RAMforge-managed memory
        let mut budget = MemoryBudget::new(ram_budget_bytes)
            .map_err(|e| format!("invalid RAM budget: {}", e))?;

        // Load persistent weights; transformer layers are streamed per
        // forward call. Resident persistents are charged to the budget
        // (weight:*), anything that does not fit is streamed on demand with
        // charged, bounded temps.
        let model = StreamingLlamaModel::load(&data_source, &mut budget)
            .map_err(|e| format!("failed to load model weights: {}", e))?;

        let residency_stats = ResidencyStats::new(model.total_weight_bytes);

        Ok(Self {
            data_source,
            tokenizer,
            model,
            kv_cache: None,
            backend: CpuBackend::new(),
            budget,
            ram_budget_bytes,
            residency_stats,
        })
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampler: &Sampler,
    ) -> Result<(Vec<u32>, String), String> {
        // Tokenize prompt
        let prompt_tokens = self.tokenizer.encode(prompt, true);
        if prompt_tokens.is_empty() {
            return Err("prompt tokenization produced no tokens".to_string());
        }

        // Check context length
        if prompt_tokens.len() + max_tokens > self.model.config.context_length {
            return Err(format!(
                "prompt + max_tokens exceeds context length: {} + {} > {}",
                prompt_tokens.len(),
                max_tokens,
                self.model.config.context_length
            ));
        }
        let needed_len = prompt_tokens.len() + max_tokens;

        // KV cache: start at the prompt length; grow chunk-wise as tokens
        // are generated (capped at needed_len rather than context_length).
        let mut kv_cache = KvCache::new(
            self.model.config.block_count,
            self.model.config.head_count_kv,
            self.model.config.head_dim,
            prompt_tokens.len(),
        )
        .map_err(|e| format!("failed to create KV cache: {}", e))?;

        let kv_bytes = kv_cache.total_bytes() as u64;
        self.budget
            .allocate("kv_cache", kv_bytes)
            .map_err(|e| {
                format!(
                    "RAM budget too small for initial KV cache: need {} bytes ({} layers, {} kv_heads, head_dim {}, {} prompt tokens): {}",
                    kv_bytes,
                    self.model.config.block_count,
                    self.model.config.head_count_kv,
                    self.model.config.head_dim,
                    prompt_tokens.len(),
                    e
                )
            })?;

        // Initialize residency stats with total model weight bytes
        let mut residency_stats = ResidencyStats::new(self.model.total_weight_bytes);
        residency_stats.update_managed(self.budget.used_bytes());

        // Disjoin field borrows so the scoped temp closures can use them.
        let model = &self.model;
        let backend = &self.backend;
        let data_source = &self.data_source;
        let eos_id = self.tokenizer.eos_id;
        let context_length = self.model.config.context_length;

        let vocab = model.config.vocab_size;
        let n_embd = model.config.embedding_length;
        // Single logits buffer + the hidden state that lives across steps.
        let io_bytes = ((vocab + n_embd) * 4) as u64;
        // Sampler scratch: scaled logits + kept-index table (non-greedy only).
        let sampler_scratch = if sampler.temperature > 0.0 {
            (vocab * 4 * 5) as u64
        } else {
            0
        };

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut hidden: Option<Vec<f32>> = None;
        let mut current_pos = prompt_tokens.len();

        let budget = &mut self.budget;
        let run_result: Result<(), String> = budget.with_temp("tmp:logits", io_bytes, |budget| {
            let mut logits = vec![0.0f32; vocab];

            // Prompt pass – fills the KV cache one token at a time.
            for (pos, &token_id) in prompt_tokens.iter().enumerate() {
                let h = model.forward_single_streaming(
                    token_id,
                    pos,
                    &mut kv_cache,
                    backend,
                    data_source,
                    budget,
                    &mut residency_stats,
                )?;
                hidden = Some(h);
            }

            // Generation loop
            for _ in 0..max_tokens {
                let hidden_state = hidden.as_ref().ok_or("no hidden state")?;
                model.compute_logits(
                    hidden_state,
                    backend,
                    data_source,
                    budget,
                    &mut logits,
                )?;

                let next_token = budget.with_temp("tmp:sampling", sampler_scratch, |_b| {
                    Ok::<u32, String>(sampler.sample(&logits))
                })?;

                if let Some(eos) = eos_id {
                    if next_token == eos {
                        break;
                    }
                }

                generated_tokens.push(next_token);

                // Chunk-wise KV growth with deterministic rollback: release
                // the old charge, try the new one; if it fails, restore the
                // old charge so the budget stays consistent.
                if current_pos + 1 > kv_cache.capacity_tokens() {
                    let target = kv_cache
                        .chunk_aligned_capacity(current_pos + 1)
                        .min(needed_len);
                    let old_bytes = kv_cache.total_bytes() as u64;
                    let new_bytes = kv_cache.bytes_for_tokens(target) as u64;
                    let _ = budget.release("kv_cache");
                    if let Err(e) = budget.allocate("kv_cache", new_bytes) {
                        let _ = budget.allocate("kv_cache", old_bytes);
                        return Err(format!(
                            "RAM budget too small to grow KV cache from {} to {} tokens ({} -> {} bytes): {}",
                            kv_cache.capacity_tokens(),
                            target,
                            old_bytes,
                            new_bytes,
                            e
                        ));
                    }
                    kv_cache
                        .grow_to(target)
                        .map_err(|e| format!("failed to grow KV cache: {}", e))?;
                }

                let h = model.forward_single_streaming(
                    next_token,
                    current_pos,
                    &mut kv_cache,
                    backend,
                    data_source,
                    budget,
                    &mut residency_stats,
                )?;
                hidden = Some(h);
                current_pos += 1;

                if current_pos >= context_length {
                    break;
                }
            }
            Ok(())
        });
        run_result?;

        self.kv_cache = Some(kv_cache);
        self.residency_stats = residency_stats;

        // Budget integrity: after the scoped temps are gone, only the
        // persistent charges (weights, KV cache) may remain.
        debug_assert_eq!(
            self.budget.allocations().keys().filter(|k| k.starts_with("tmp:")).count(),
            0,
            "temp charges must be released after generate()"
        );

        let text = self.tokenizer.decode(&generated_tokens);
        Ok((generated_tokens, text))
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.model.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Boxed value-writer closure used by the GGUF test fixtures.
    type WriteValFn<'a> = Box<dyn FnMut(&mut Vec<u8>) + 'a>;

    // Create a tiny deterministic LLaMA model for testing
    // Config: vocab 16, n_embd 8, n_layer 1, n_head 2, head_dim 4, ffn 16
    fn create_tiny_llama_gguf() -> NamedTempFile {
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

        // Create correctly
        let mut buf2 = Vec::new();
        buf2.extend_from_slice(b"GGUF");
        buf2.extend_from_slice(&3u32.to_le_bytes());
        buf2.extend_from_slice(&11u64.to_le_bytes());
        buf2.extend_from_slice(&16u64.to_le_bytes());
        // re-add kvs
        let mut add_kv = |key: &str, val_type: u32, mut write_val: WriteValFn<'_>| {
            write_string(&mut buf2, key);
            write_u32(&mut buf2, val_type);
            write_val(&mut buf2);
        };
        add_kv("general.architecture", 8, Box::new(|b| write_string(b, "llama")));
        add_kv("llama.vocab_size", 4, Box::new(|b| write_u32(b, 16)));
        add_kv("llama.context_length", 4, Box::new(|b| write_u32(b, 32)));
        add_kv("llama.embedding_length", 4, Box::new(|b| write_u32(b, 8)));
        add_kv("llama.block_count", 4, Box::new(|b| write_u32(b, 1)));
        add_kv("llama.feed_forward_length", 4, Box::new(|b| write_u32(b, 16)));
        add_kv("llama.attention.head_count", 4, Box::new(|b| write_u32(b, 2)));
        add_kv("llama.attention.head_count_kv", 4, Box::new(|b| write_u32(b, 2)));
        add_kv("llama.attention.layer_norm_rms_epsilon", 6, Box::new(|b| write_f32(b, 1e-5)));
        add_kv("llama.rope.freq_base", 6, Box::new(|b| write_f32(b, 10000.0)));
        add_kv("tokenizer.ggml.model", 8, Box::new(|b| write_string(b, "llama")));
        add_kv("tokenizer.ggml.tokens", 9, Box::new(|b| {
            write_u32(b, 8);
            write_u64(b, 16);
            for tok in ["<unk>", "<s>", "</s>", "▁hello", "▁world", "hello", "world", "!", "▁", "a", "b", "c", "d", "e", "f", "g"] {
                write_string(b, tok);
            }
        }));
        add_kv("tokenizer.ggml.scores", 9, Box::new(|b| {
            write_u32(b, 6);
            write_u64(b, 16);
            for _ in 0..16 { write_f32(b, 0.0); }
        }));
        add_kv("tokenizer.ggml.token_type", 9, Box::new(|b| {
            write_u32(b, 5);
            write_u64(b, 16);
            for t in [2,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1] { write_u32(b, t); }
        }));
        add_kv("tokenizer.ggml.bos_token_id", 4, Box::new(|b| write_u32(b, 1)));
        add_kv("tokenizer.ggml.eos_token_id", 4, Box::new(|b| write_u32(b, 2)));

        // Now tensors
        let tensor_defs = vec![
            ("token_embd.weight", vec![8u64, 16u64], 0u32), // F32
            ("output_norm.weight", vec![8], 0),
            ("blk.0.attn_norm.weight", vec![8], 0),
            ("blk.0.attn_q.weight", vec![8, 8], 0),
            ("blk.0.attn_k.weight", vec![8, 8], 0),
            ("blk.0.attn_v.weight", vec![8, 8], 0),
            ("blk.0.attn_output.weight", vec![8, 8], 0),
            ("blk.0.ffn_norm.weight", vec![8], 0),
            // ggml layout [in, out]: gate/up map 8 -> 16, down maps 16 -> 8
            ("blk.0.ffn_gate.weight", vec![8, 16], 0),
            ("blk.0.ffn_up.weight", vec![8, 16], 0),
            ("blk.0.ffn_down.weight", vec![16, 8], 0),
        ];

        let mut offset = 0u64;
        for (name, dims, ty) in &tensor_defs {
            write_string(&mut buf2, name);
            write_u32(&mut buf2, dims.len() as u32);
            for d in dims {
                write_u64(&mut buf2, *d);
            }
            write_u32(&mut buf2, *ty);
            write_u64(&mut buf2, offset);
            // compute byte length
            let elems: u64 = dims.iter().product();
            let bytes = elems * 4;
            offset += bytes;
        }

        let pos = buf2.len() as u64;
        let aligned = ramforge_core::model::align_offset(pos, 32);
        buf2.extend(vec![0u8; (aligned - pos) as usize]);

        // Now write tensor data
        // token_embd: each token embedding is token_id as f32 repeated
        for token_id in 0..16 {
            for _ in 0..8 {
                buf2.extend_from_slice(&(token_id as f32).to_le_bytes());
            }
        }
        // output_norm: ones
        for _ in 0..8 {
            buf2.extend_from_slice(&1.0f32.to_le_bytes());
        }
        // attn_norm: ones
        for _ in 0..8 {
            buf2.extend_from_slice(&1.0f32.to_le_bytes());
        }
        // attn_q: identity
        for i in 0..8 {
            for j in 0..8 {
                let v: f32 = if i == j { 1.0 } else { 0.0 };
                buf2.extend_from_slice(&v.to_le_bytes());
            }
        }
        // attn_k: identity
        for i in 0..8 {
            for j in 0..8 {
                let v: f32 = if i == j { 1.0 } else { 0.0 };
                buf2.extend_from_slice(&v.to_le_bytes());
            }
        }
        // attn_v: identity
        for i in 0..8 {
            for j in 0..8 {
                let v: f32 = if i == j { 1.0 } else { 0.0 };
                buf2.extend_from_slice(&v.to_le_bytes());
            }
        }
        // attn_output: identity
        for i in 0..8 {
            for j in 0..8 {
                let v: f32 = if i == j { 1.0 } else { 0.0 };
                buf2.extend_from_slice(&v.to_le_bytes());
            }
        }
        // ffn_norm: ones
        for _ in 0..8 {
            buf2.extend_from_slice(&1.0f32.to_le_bytes());
        }
        // ffn_gate: small random but deterministic: 0.1
        for _ in 0..16*8 {
            buf2.extend_from_slice(&0.1f32.to_le_bytes());
        }
        // ffn_up: 0.1
        for _ in 0..16*8 {
            buf2.extend_from_slice(&0.1f32.to_le_bytes());
        }
        // ffn_down: 0.1
        for _ in 0..8*16 {
            buf2.extend_from_slice(&0.1f32.to_le_bytes());
        }

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf2).unwrap();
        tmp.flush().unwrap();
        tmp
    }

    #[test]
    fn test_end_to_end_inference() {
        let tmp = create_tiny_llama_gguf();
        let mut engine = InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        let sampler = crate::sampling::Sampler::greedy();
        let (tokens, text) = engine.generate("hello", 5, &sampler).unwrap();
        assert_eq!(tokens.len(), 5);
        let mut engine2 = InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        let (tokens2, text2) = engine2.generate("hello", 5, &sampler).unwrap();
        assert_eq!(tokens, tokens2);
        assert_eq!(text, text2);
    }

    fn create_out_of_core_gguf() -> NamedTempFile {
        // Create model where total weights > budget but layers fit
        // Use same logic as streaming_model test but with tokenizer
        fn write_string<W: Write>(w: &mut W, s: &str) {
            w.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
            w.write_all(s.as_bytes()).unwrap();
        }
        fn write_u32<W: Write>(w: &mut W, v: u32) { w.write_all(&v.to_le_bytes()).unwrap(); }
        fn write_u64<W: Write>(w: &mut W, v: u64) { w.write_all(&v.to_le_bytes()).unwrap(); }
        fn write_f32<W: Write>(w: &mut W, v: f32) { w.write_all(&v.to_le_bytes()).unwrap(); }

        let n_layers = 8;
        let n_embd = 16;
        let ffn = 32;

        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        let tensor_count = 2 + n_layers * 9;
        buf.extend_from_slice(&(tensor_count as u64).to_le_bytes());
        buf.extend_from_slice(&17u64.to_le_bytes());

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
        add_kv("tokenizer.ggml.tokens", 9, Box::new(|b| {
            write_u32(b, 8);
            write_u64(b, 16);
            for tok in ["<unk>", "<s>", "</s>", "▁hello", "▁world", "hello", "world", "!", "▁", "a", "b", "c", "d", "e", "f", "g"] {
                write_string(b, tok);
            }
        }));
        add_kv("tokenizer.ggml.scores", 9, Box::new(|b| {
            write_u32(b, 6);
            write_u64(b, 16);
            for _ in 0..16 { write_f32(b, 0.0); }
        }));
        add_kv("tokenizer.ggml.token_type", 9, Box::new(|b| {
            write_u32(b, 5);
            write_u64(b, 16);
            for t in [2,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1] { write_u32(b, t); }
        }));
        add_kv("tokenizer.ggml.bos_token_id", 4, Box::new(|b| write_u32(b, 1)));
        add_kv("tokenizer.ggml.eos_token_id", 4, Box::new(|b| write_u32(b, 2)));
        // add one more to make 16
        add_kv("general.name", 8, Box::new(|b| write_string(b, "tiny-out-of-core")));

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
        let aligned = ramforge_core::model::align_offset(pos, 32);
        buf.extend(vec![0u8; (aligned - pos) as usize]);
        // Write data: token_embd as token_id, others as 0.1 or identity
        for token_id in 0..16 {
            for _ in 0..n_embd {
                buf.extend_from_slice(&(token_id as f32).to_le_bytes());
            }
        }
        for _ in 0..n_embd { buf.extend_from_slice(&1.0f32.to_le_bytes()); }
        for _layer in 0..n_layers {
            for _ in 0..n_embd { buf.extend_from_slice(&1.0f32.to_le_bytes()); } // attn_norm
            for i in 0..n_embd { for j in 0..n_embd { let v: f32 = if i==j {1.0} else {0.0}; buf.extend_from_slice(&v.to_le_bytes()); } } // q
            for i in 0..n_embd { for j in 0..n_embd { let v: f32 = if i==j {1.0} else {0.0}; buf.extend_from_slice(&v.to_le_bytes()); } } // k
            for i in 0..n_embd { for j in 0..n_embd { let v: f32 = if i==j {1.0} else {0.0}; buf.extend_from_slice(&v.to_le_bytes()); } } // v
            for i in 0..n_embd { for j in 0..n_embd { let v: f32 = if i==j {1.0} else {0.0}; buf.extend_from_slice(&v.to_le_bytes()); } } // output
            for _ in 0..n_embd { buf.extend_from_slice(&1.0f32.to_le_bytes()); } // ffn_norm
            for _ in 0..ffn*n_embd { buf.extend_from_slice(&0.1f32.to_le_bytes()); } // gate
            for _ in 0..ffn*n_embd { buf.extend_from_slice(&0.1f32.to_le_bytes()); } // up
            for _ in 0..n_embd*ffn { buf.extend_from_slice(&0.1f32.to_le_bytes()); } // down
        }

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        tmp.flush().unwrap();
        tmp
    }

    #[test]
    fn test_out_of_core_inference() {
        // Total model size > budget, but per-layer fits
        let tmp = create_out_of_core_gguf();
        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let total_bytes: u64 = ds.model().tensors.iter().filter_map(|t| t.byte_length).sum();

        // n_embd 16 / ffn 32: per layer 10368 B (F32), persistents 1088 B.
        // 8 layers => total 84032 B (~82 KiB) > 32 KiB budget, while one
        // layer (with charge-before-read transient up to ~2x) plus KV cache
        // (~5 tokens) and forward temps comfortably fits.
        let ram_budget = 32 * 1024;

        // The engine should still succeed because it streams layers
        let mut engine = InferenceEngine::new(tmp.path().to_str().unwrap(), ram_budget).unwrap();

        // Check total > budget
        assert!(total_bytes > ram_budget, "total {} should be > budget {}", total_bytes, ram_budget);

        let sampler = crate::sampling::Sampler::greedy();
        let (tokens, _text) = engine.generate("hello", 3, &sampler).unwrap();
        assert_eq!(tokens.len(), 3);

        // Check residency stats: peak layer < total, peak managed <= budget
        let stats = &engine.residency_stats;
        assert!(stats.total_model_weight_bytes > ram_budget);
        assert!(stats.peak_resident_layer_bytes < stats.total_model_weight_bytes);
        assert!(stats.peak_managed_bytes <= ram_budget, "peak managed {} should be <= budget {}", stats.peak_managed_bytes, ram_budget);
        assert!(stats.num_layer_loads > 0);
        assert!(stats.num_layer_releases > 0);
    }

    #[test]
    fn test_budget_too_small_failure() {
        let tmp = create_tiny_llama_gguf();

        // Zero budget is rejected at engine creation.
        let result = InferenceEngine::new(tmp.path().to_str().unwrap(), 0);
        assert!(result.is_err());

        // A near-zero budget may construct the engine (persistents become
        // streamed), but generate() must fail clearly at the first
        // budget-checked allocation (KV cache).
        let mut engine = InferenceEngine::new(tmp.path().to_str().unwrap(), 64).unwrap();
        let sampler = crate::sampling::Sampler::greedy();
        let err = engine.generate("hello", 2, &sampler).unwrap_err();
        assert!(
            err.contains("budget") || err.contains("Budget"),
            "expected budget-related error, got: {}",
            err
        );
        // Budget must not be corrupted by the failed attempt.
        assert!(engine.budget.used_bytes() <= engine.budget.total_bytes());
    }
}
