//! Inference engine for RAMforge (CPU, llama/qwen2, out-of-core).
//!
//! Memory contract for `generate()` (hardened in Milestone 7.1):
//! - every allocation that lives across a step is budget-charged for its
//!   lifetime: persistent weights (`weight:*`), streamed layers
//!   (`layer:{i}:*`), the KV cache (`kv_cache`), the single logits buffer
//!   (`tmp:logits`), and the caller-owned hidden state (`tmp:hidden`);
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

    /// Drop any live KV cache together with its budget charge (idempotent).
    ///
    /// Guarantees the invariant "engine has an active KV cache" <=>
    /// "a `kv_cache` charge exists in the budget". After this call the
    /// engine can start (or retry) a generation from a clean state.
    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
        if self.budget.get("kv_cache").is_some() {
            let _ = self.budget.release("kv_cache");
        }
    }

    /// Generate tokens for `prompt`. Supports multiple sequential calls on
    /// the same engine: any previous KV cache is explicitly reset first,
    /// and a failed generation never leaves a stale `"kv_cache"` charge
    /// behind (proper cleanup on every exit path).
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampler: &Sampler,
    ) -> Result<(Vec<u32>, String), String> {
        // Explicit reset of any previous run's KV state (charge + object).
        self.clear_kv_cache();
        match self.generate_impl(prompt, max_tokens, sampler) {
            Ok(out) => Ok(out),
            Err(e) => {
                // Failed run: no stale KV budget state, engine stays reusable.
                self.clear_kv_cache();
                Err(e)
            }
        }
    }

    fn generate_impl(
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
        // Caller-owned output buffers have independent charges whose scopes
        // exactly match their lifetimes. In particular, `tmp:hidden` remains
        // live while every per-token forward writes and later consumers read
        // the hidden vector returned through the caller-owned slice.
        let logits_bytes = (vocab * 4) as u64;
        let hidden_bytes = (n_embd * 4) as u64;
        // Sampler scratch: scaled logits + kept-index table (non-greedy only).
        let sampler_scratch = if sampler.temperature > 0.0 {
            (vocab * 4 * 5) as u64
        } else {
            0
        };

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut current_pos = prompt_tokens.len();

        let budget = &mut self.budget;
        let run_result: Result<(), String> =
            budget.with_temp("tmp:hidden", hidden_bytes, |budget| {
                let mut hidden = vec![0.0f32; n_embd];
                budget.with_temp("tmp:logits", logits_bytes, |budget| {
                    let mut logits = vec![0.0f32; vocab];

                    // Prompt pass – fills the KV cache one token at a time.
                    for (pos, &token_id) in prompt_tokens.iter().enumerate() {
                        model.forward_single_streaming(
                            token_id,
                            pos,
                            &mut kv_cache,
                            backend,
                            data_source,
                            budget,
                            &mut residency_stats,
                            &mut hidden,
                        )?;
                    }

                    // Generation loop
                    for _ in 0..max_tokens {
                        model.compute_logits(
                            &hidden,
                            backend,
                            data_source,
                            budget,
                            &mut logits,
                        )?;

                        let next_token =
                            budget.with_temp("tmp:sampling", sampler_scratch, |_b| {
                                Ok::<u32, String>(sampler.sample(&logits))
                            })?;

                        if let Some(eos) = eos_id {
                            if next_token == eos {
                                break;
                            }
                        }

                        generated_tokens.push(next_token);

                        // Chunk-wise KV growth with deterministic rollback:
                        // release the old charge, try the new one; if it fails,
                        // restore the old charge so the budget stays consistent.
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

                        model.forward_single_streaming(
                            next_token,
                            current_pos,
                            &mut kv_cache,
                            backend,
                            data_source,
                            budget,
                            &mut residency_stats,
                            &mut hidden,
                        )?;
                        current_pos += 1;

                        if current_pos >= context_length {
                            break;
                        }
                    }
                    Ok(())
                })
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
        // The failed attempt must not leave a stale KV charge behind
        // (the engine constructor loaded only persistent weights).
        assert!(engine.budget.get("kv_cache").is_none());
        assert!(engine.kv_cache.is_none());
    }

    #[test]
    fn test_generate_twice_same_engine() {
        // Regression test for M6.1 BUG-1: sequential generate() calls.
        let tmp = create_tiny_llama_gguf();
        let mut engine = InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        let sampler = crate::sampling::Sampler::greedy();

        let (tokens1, text1) = engine.generate("hello", 5, &sampler).unwrap();
        assert_eq!(tokens1.len(), 5);
        // After a successful run the KV charge exists and matches the cache.
        let kv_charge_after_first = engine.budget.get("kv_cache").unwrap();
        assert_eq!(
            kv_charge_after_first as usize,
            engine.kv_cache.as_ref().unwrap().total_bytes()
        );

        // Second call on the same engine must succeed (M6 BUG-1 regression).
        let (tokens2, text2) = engine.generate("hello", 3, &sampler).unwrap();
        assert_eq!(tokens2.len(), 3);
        // Deterministic: same prompt prefix → same token prefix.
        assert_eq!(&tokens1[..3], &tokens2[..]);
        assert!(text1.starts_with(&text2) || text2.starts_with(&text1));

        // Exactly one KV charge, never duplicates, no temp leftovers.
        let keys: Vec<&String> = engine.budget.allocations().keys().collect();
        assert_eq!(keys.iter().filter(|k| k.as_str() == "kv_cache").count(), 1);
        assert!(!keys.iter().any(|k| k.starts_with("tmp:")));
        let kv_charge_after_second = engine.budget.get("kv_cache").unwrap();
        assert_eq!(
            kv_charge_after_second as usize,
            engine.kv_cache.as_ref().unwrap().total_bytes()
        );
    }

    #[test]
    fn test_failed_generate_releases_kv_charge() {
        // Budget tuned to fail *inside* the first forward pass (after the
        // KV charge exists): the failure must clean up the KV charge.
        let tmp = create_tiny_llama_gguf();

        // n_embd=8: output_norm resident 32 B; KV initial 2 tokens*64 B=128;
        // caller-owned logits + hidden charges total 96 B. Those startup
        // charges fit, but the subsequent forward/layer working set does not.
        let budget_bytes = 900;
        let mut engine = InferenceEngine::new(tmp.path().to_str().unwrap(), budget_bytes).unwrap();
        let used_before = engine.budget.used_bytes();

        let sampler = crate::sampling::Sampler::greedy();
        let err = engine.generate("hello", 5, &sampler).unwrap_err();
        assert!(
            err.contains("budget") || err.contains("Budget") || err.contains("insufficient"),
            "expected a budget-related error, got: {}",
            err
        );

        // No stale KV state: object dropped and charge released.
        assert!(engine.kv_cache.is_none());
        assert!(
            engine.budget.get("kv_cache").is_none(),
            "stale kv_cache charge after failed generate (allocations: {:?})",
            engine.budget.allocations()
        );
        assert_eq!(engine.budget.used_bytes(), used_before);
        assert!(engine.budget.used_bytes() <= engine.budget.total_bytes());
    }

    #[test]
    fn test_engine_reusable_after_failed_generate() {
        let tmp = create_tiny_llama_gguf();

        // Case A: failure inside generate (same tight budget as above) ->
        // a second attempt must fail identically, NOT with a stale-charge
        // "already exists" error, and the engine remains reusable.
        let mut tight = InferenceEngine::new(tmp.path().to_str().unwrap(), 900).unwrap();
        let sampler = crate::sampling::Sampler::greedy();
        let err1 = tight.generate("hello", 5, &sampler).unwrap_err();
        let err2 = tight.generate("hello", 5, &sampler).unwrap_err();
        assert!(!err1.contains("already exists"), "err1: {}", err1);
        assert!(!err2.contains("already exists"), "err2: {}", err2);
        // Same failure class both times (deterministic, clean state).
        assert!(
            (err1.contains("budget") || err1.contains("insufficient"))
                && (err2.contains("budget") || err2.contains("insufficient")),
            "err1 = {} ; err2 = {}",
            err1,
            err2
        );

        // Case B: failure before any allocation (context-length check) must
        // leave the engine fully reusable for a subsequent valid call.
        let mut healthy = InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        let err_ctx = healthy.generate("hello", 100, &sampler).unwrap_err();
        assert!(err_ctx.contains("context length"), "got: {}", err_ctx);
        assert!(healthy.budget.get("kv_cache").is_none());
        let (tokens, _text) = healthy.generate("hello", 3, &sampler).unwrap();
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn test_clear_kv_cache_releases_charge() {
        let tmp = create_tiny_llama_gguf();
        let mut engine = InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        let sampler = crate::sampling::Sampler::greedy();

        let (tokens, _) = engine.generate("hello", 3, &sampler).unwrap();
        assert_eq!(tokens.len(), 3);
        let used_with_kv = engine.budget.used_bytes();
        let kv_bytes = engine.budget.get("kv_cache").unwrap();

        engine.clear_kv_cache();
        assert!(engine.kv_cache.is_none());
        assert!(engine.budget.get("kv_cache").is_none());
        assert_eq!(engine.budget.used_bytes(), used_with_kv - kv_bytes);

        // Idempotent: clearing again is a no-op and errors nothing.
        engine.clear_kv_cache();
        assert_eq!(engine.budget.used_bytes(), used_with_kv - kv_bytes);

        // Engine stays fully functional afterwards.
        let (tokens2, _) = engine.generate("hello", 2, &sampler).unwrap();
        assert_eq!(tokens2.len(), 2);
    }

    // ------------------------------------------------------------------
    // qwen2 with Q/K/V biases: deterministic fixture + independent
    // reference implementation (proves bias placement, half-split RoPE,
    // and ggml [in,out] layout together, without a real model download).
    // ------------------------------------------------------------------

    /// Deterministic weight tables for the biased qwen2 fixture.
    /// n_embd=8, heads=2, head_dim=4, kv_heads=2, ffn=16, vocab=16.
    struct Qwen2FixtureWeights {
        embd: Vec<f32>,    // [16][8] rows = token embeddings
        output_norm: Vec<f32>,
        attn_norm: Vec<f32>,
        attn_q: Vec<f32>,  // ggml [8,8]: row o of in 8
        attn_k: Vec<f32>,
        attn_v: Vec<f32>,
        attn_output: Vec<f32>,
        ffn_norm: Vec<f32>,
        ffn_gate: Vec<f32>, // ggml [8,16]
        ffn_up: Vec<f32>,
        ffn_down: Vec<f32>, // ggml [16,8]
        bias_q: Vec<f32>,
        bias_k: Vec<f32>,
        bias_v: Vec<f32>,
    }

    fn qwen2_weights() -> Qwen2FixtureWeights {
        let mut embd = vec![0.0f32; 16 * 8];
        for t in 0..16 {
            for i in 0..8 {
                embd[t * 8 + i] = 0.02 * (t + 1) as f32 + 0.003 * i as f32;
            }
        }
        let w = |n: usize, f: &dyn Fn(usize) -> f32| -> Vec<f32> { (0..n).map(f).collect() };
        Qwen2FixtureWeights {
            embd,
            output_norm: vec![1.0; 8],
            attn_norm: vec![1.0; 8],
            attn_q: w(64, &|i| 0.03 + 0.01 * (((i / 8) + 2 * (i % 8)) % 5) as f32),
            attn_k: w(64, &|i| 0.02 + 0.01 * (((i / 8) + 3 * (i % 8)) % 7) as f32),
            attn_v: w(64, &|i| 0.01 + 0.02 * ((2 * (i / 8) + (i % 8)) % 5) as f32),
            attn_output: w(64, &|i| 0.03 + 0.01 * (((i / 8) + (i % 8)) % 4) as f32),
            ffn_norm: vec![1.0; 8],
            ffn_gate: w(128, &|i| 0.02 + 0.01 * (((i / 8) + (i % 8)) % 6) as f32),
            ffn_up: w(128, &|i| 0.01 + 0.015 * (((i / 8) + 2 * (i % 8)) % 5) as f32),
            ffn_down: w(128, &|i| 0.02 + 0.01 * (((i / 8) + 3 * (i % 8)) % 7) as f32),
            bias_q: w(8, &|i| 0.05 + 0.01 * i as f32),
            bias_k: w(8, &|i| 0.04 + 0.02 * i as f32),
            bias_v: w(8, &|i| 0.03 + 0.03 * i as f32),
        }
    }

    fn create_qwen2_biased_gguf() -> NamedTempFile {
        fn write_string<W: Write>(w: &mut W, s: &str) {
            w.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
            w.write_all(s.as_bytes()).unwrap();
        }
        fn write_u32<W: Write>(w: &mut W, v: u32) { w.write_all(&v.to_le_bytes()).unwrap(); }
        fn write_u64<W: Write>(w: &mut W, v: u64) { w.write_all(&v.to_le_bytes()).unwrap(); }
        fn write_f32<W: Write>(w: &mut W, v: f32) { w.write_all(&v.to_le_bytes()).unwrap(); }

        let w = qwen2_weights();
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&14u64.to_le_bytes()); // 2 persistent + 9 weights + 3 biases
        buf.extend_from_slice(&16u64.to_le_bytes()); // metadata kv count

        let mut add_kv = |key: &str, val_type: u32, mut write_val: WriteValFn<'_>| {
            write_string(&mut buf, key);
            write_u32(&mut buf, val_type);
            write_val(&mut buf);
        };
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
            for t in [2i32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] { write_u32(b, t as u32); }
        }));
        add_kv("tokenizer.ggml.bos_token_id", 4, Box::new(|b| write_u32(b, 1)));
        add_kv("tokenizer.ggml.eos_token_id", 4, Box::new(|b| write_u32(b, 2)));

        let mut offset = 0u64;
        let mut defs: Vec<(&str, Vec<u64>)> = vec![
            ("token_embd.weight", vec![8, 16]),
            ("output_norm.weight", vec![8]),
            ("blk.0.attn_norm.weight", vec![8]),
            ("blk.0.attn_q.weight", vec![8, 8]),
            ("blk.0.attn_q.bias", vec![8]),
            ("blk.0.attn_k.weight", vec![8, 8]),
            ("blk.0.attn_k.bias", vec![8]),
            ("blk.0.attn_v.weight", vec![8, 8]),
            ("blk.0.attn_v.bias", vec![8]),
            ("blk.0.attn_output.weight", vec![8, 8]),
            ("blk.0.ffn_norm.weight", vec![8]),
            ("blk.0.ffn_gate.weight", vec![8, 16]),
            ("blk.0.ffn_up.weight", vec![8, 16]),
            ("blk.0.ffn_down.weight", vec![16, 8]),
        ];
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
        let aligned = ramforge_core::model::align_offset(pos, 32);
        buf.extend(vec![0u8; (aligned - pos) as usize]);

        // Write data in exactly the descriptor order.
        let mut push = |data: &[f32]| {
            for v in data {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        };
        defs.clear();
        push(&w.embd);
        push(&w.output_norm);
        push(&w.attn_norm);
        push(&w.attn_q);
        push(&w.bias_q);
        push(&w.attn_k);
        push(&w.bias_k);
        push(&w.attn_v);
        push(&w.bias_v);
        push(&w.attn_output);
        push(&w.ffn_norm);
        push(&w.ffn_gate);
        push(&w.ffn_up);
        push(&w.ffn_down);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        tmp.flush().unwrap();
        tmp
    }

    /// Independent reference forward for the biased qwen2 fixture.
    /// Mirrors the intended semantics: ggml [in,out] matvec, bias AFTER
    /// projection, half-split RoPE, causal attention over KV history,
    /// SwiGLU FFN, RMSNorm. Returns the post-output-norm hidden state.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn reference_forward_biased(
        w: &Qwen2FixtureWeights,
        token: usize,
        pos: usize,
        k_hist: &mut Vec<f32>,
        v_hist: &mut Vec<f32>,
        eps: f32,
    ) -> Vec<f32> {
        let rmsnorm = |x: &[f32], wt: &[f32]| -> Vec<f32> {
            let mean: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let rms = (mean + eps).sqrt();
            (0..x.len()).map(|i| x[i] / rms * wt[i]).collect()
        };
        let matvec = |wt: &[f32], in_dim: usize, out_dim: usize, x: &[f32]| -> Vec<f32> {
            (0..out_dim)
                .map(|o| (0..in_dim).map(|i| wt[o * in_dim + i] * x[i]).sum())
                .collect()
        };
        let rope_ref = |x: &mut [f32], pos: usize| {
            let dim = 4;
            for head in 0..2 {
                let off = head * dim;
                for j in 0..dim / 2 {
                    let theta = 10000.0f32.powf(-2.0 * j as f32 / dim as f32) * pos as f32;
                    let (c, s) = (theta.cos(), theta.sin());
                    let (a, b) = (x[off + j], x[off + j + dim / 2]);
                    x[off + j] = a * c - b * s;
                    x[off + j + dim / 2] = a * s + b * c;
                }
            }
        };

        let mut hidden: Vec<f32> = (0..8).map(|i| w.embd[token * 8 + i]).collect();
        let tmp = rmsnorm(&hidden, &w.attn_norm);

        // Projections + biases (bias added AFTER matvec, before RoPE).
        let mut q = matvec(&w.attn_q, 8, 8, &tmp);
        let mut k = matvec(&w.attn_k, 8, 8, &tmp);
        let mut v = matvec(&w.attn_v, 8, 8, &tmp);
        for i in 0..8 {
            q[i] += w.bias_q[i];
            k[i] += w.bias_k[i];
            v[i] += w.bias_v[i];
        }
        rope_ref(&mut q, pos);
        rope_ref(&mut k, pos);

        // Attention over history + current token (2 heads, head_dim 4).
        let total = pos + 1;
        let k_at = |p: usize| -> &[f32] {
            if p < pos {
                &k_hist[p * 8..p * 8 + 8]
            } else {
                &k
            }
        };
        let v_at = |p: usize| -> &[f32] {
            if p < pos {
                &v_hist[p * 8..p * 8 + 8]
            } else {
                &v
            }
        };
        let mut attn_out = vec![0.0f32; 8];
        for h in 0..2 {
            let qh = &q[h * 4..h * 4 + 4];
            let mut scores = vec![0.0f32; total];
            for p in 0..total {
                let kp = k_at(p);
                let kh = &kp[h * 4..h * 4 + 4];
                scores[p] = (0..4).map(|i| qh[i] * kh[i]).sum::<f32>() / 2.0; // sqrt(4)
            }
            let max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }
            for p in 0..total {
                let vp = v_at(p);
                let vh = &vp[h * 4..h * 4 + 4];
                for i in 0..4 {
                    attn_out[h * 4 + i] += scores[p] * vh[i];
                }
            }
        }

        let attn_proj = matvec(&w.attn_output, 8, 8, &attn_out);
        for i in 0..8 {
            hidden[i] += attn_proj[i];
        }

        // KV history append happens after use (matches model ordering:
        // the current token was passed as k_new/v_new, not read from hist).
        k_hist.extend_from_slice(&k);
        v_hist.extend_from_slice(&v);

        let tmp = rmsnorm(&hidden, &w.ffn_norm);
        let gate = matvec(&w.ffn_gate, 8, 16, &tmp);
        let up = matvec(&w.ffn_up, 8, 16, &tmp);
        let mut gu = vec![0.0f32; 16];
        for i in 0..16 {
            let gv = gate[i];
            gu[i] = (gv / (1.0 + (-gv).exp())) * up[i];
        }
        let ffn_out = matvec(&w.ffn_down, 16, 8, &gu);
        for i in 0..8 {
            hidden[i] += ffn_out[i];
        }

        rmsnorm(&hidden, &w.output_norm)
    }

    #[test]
    fn test_qwen2_biased_forward_matches_reference() {
        let tmp = create_qwen2_biased_gguf();
        let mut engine = InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        assert!(engine.model.attn_bias_present);
        assert_eq!(engine.config().head_count, 2);
        assert_eq!(engine.config().head_count_kv, 2);

        let w = qwen2_weights();
        let eps = engine.config().rms_eps;

        let mut ref_k: Vec<f32> = Vec::new();
        let mut ref_v: Vec<f32> = Vec::new();

        let mut kv = crate::kv_cache::KvCache::new(
            engine.config().block_count,
            engine.config().head_count_kv,
            engine.config().head_dim,
            4,
        )
        .unwrap();
        let budget: &mut MemoryBudget = &mut engine.budget;
        let mut stats = crate::residency::ResidencyStats::new(engine.model.total_weight_bytes);
        let model = &engine.model;
        let backend = &engine.backend;
        let ds = &engine.data_source;

        let mut logits = vec![0.0f32; 16];
        let eps_ln = 2e-4f32;

        // Two sequential tokens exercise: no KV history (pos 0) and
        // history+biases+RoPE at a non-zero position (pos 1).
        for (token, pos) in [(1usize, 0usize), (5usize, 1usize)] {
            budget
                .with_temp("tmp:hidden", 8 * 4, |budget| {
                    let mut hidden = vec![0.0f32; 8];
                    model.forward_single_streaming(
                        token as u32,
                        pos,
                        &mut kv,
                        backend,
                        ds,
                        budget,
                        &mut stats,
                        &mut hidden,
                    )?;
                    model.compute_logits(&hidden, backend, ds, budget, &mut logits)?;

                    let ref_hidden =
                        reference_forward_biased(&w, token, pos, &mut ref_k, &mut ref_v, eps);
                    for i in 0..8 {
                        assert!(
                            (hidden[i] - ref_hidden[i]).abs() < eps_ln,
                            "pos {} hidden[{}]: model {} vs reference {}",
                            pos,
                            i,
                            hidden[i],
                            ref_hidden[i]
                        );
                    }
                    for (o, &logit) in logits.iter().enumerate() {
                        let ref_logit: f32 =
                            (0..8).map(|i| w.embd[o * 8 + i] * ref_hidden[i]).sum();
                        assert!(
                            (logit - ref_logit).abs() < eps_ln,
                            "pos {} logit[{}]: model {} vs reference {}",
                            pos,
                            o,
                            logit,
                            ref_logit
                        );
                    }
                    Ok::<(), String>(())
                })
                .unwrap();
        }

        // A biased qwen2 engine also runs end-to-end generation cleanly and
        // deterministically.
        let mut engine2 = InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        let sampler = crate::sampling::Sampler::greedy();
        let (t1, _) = engine2.generate("hello", 4, &sampler).unwrap();
        let (t2, _) = engine2.generate("hello", 4, &sampler).unwrap();
        assert_eq!(t1.len(), 4);
        assert_eq!(t1, t2);
    }
}
