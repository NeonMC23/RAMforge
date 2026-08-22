//! Inference loop for LLaMA

use ramforge_core::{
    datasource::GgufDataSource,
    memory::MemoryBudget,
    cache::BoundedCache,
    tokenizer::Tokenizer,
};

use crate::backend::CpuBackend;
use crate::kv_cache::KvCache;
use crate::model::{LlamaConfig, LlamaModel};
use crate::sampling::Sampler;

pub struct InferenceEngine {
    pub data_source: GgufDataSource,
    pub tokenizer: Tokenizer,
    pub model: LlamaModel,
    pub kv_cache: Option<KvCache>,
    pub backend: CpuBackend,
    pub budget: MemoryBudget,
    pub cache: BoundedCache,
    pub ram_budget_bytes: u64,
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

        // Cache capacity: 80% of budget for weights (simple deterministic strategy)
        let cache_capacity = (ram_budget_bytes as f64 * 0.8) as u64;
        let cache_capacity = cache_capacity
            .max(1024 * 1024)
            .min(ram_budget_bytes.saturating_sub(1024 * 1024));
        let mut cache = BoundedCache::new(cache_capacity)
            .map_err(|e| format!("failed to create cache: {}", e))?;

        // For milestone 2 compatibility, we previously allocated cache capacity from budget.
        // For milestone 3, we allocate weights individually from budget, and cache capacity is a separate limit.
        // To avoid double counting, we do NOT pre-allocate cache capacity, but we keep the cache as bounded.
        // Instead, we will allocate weights as they are loaded.

        // Load model weights through GgufDataSource and BoundedCache with budget accounting
        // This demonstrates file-backed access: each tensor is read via data_source.read_tensor()
        let model = LlamaModel::load(&data_source, &mut cache, &mut budget)
            .map_err(|e| format!("failed to load model weights: {}", e))?;

        // KV cache is created lazily in generate() based on actual needed length to save memory
        // For now, keep None

        Ok(Self {
            data_source,
            tokenizer,
            model,
            kv_cache: None,
            backend: CpuBackend::new(),
            budget,
            cache,
            ram_budget_bytes,
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

        // Create KV cache based on actual needed length (prompt + max_tokens) to save memory
        // This is more efficient than allocating for full context_length (which can be 32k)
        let needed_len = prompt_tokens.len() + max_tokens;
        let mut kv_cache = KvCache::new(
            self.model.config.block_count,
            self.model.config.head_count_kv,
            self.model.config.head_dim,
            needed_len,
        )
        .map_err(|e| format!("failed to create KV cache: {}", e))?;

        let kv_bytes = kv_cache.total_bytes() as u64;
        if !self.budget.can_allocate(kv_bytes) {
            return Err(format!(
                "RAM budget too small for KV cache: need {} bytes for KV cache ({} layers, {} kv_heads, head_dim {}, needed_len {}), but only {} bytes available (total {}, used {})",
                kv_bytes,
                self.model.config.block_count,
                self.model.config.head_count_kv,
                self.model.config.head_dim,
                needed_len,
                self.budget.available_bytes(),
                self.budget.total_bytes(),
                self.budget.used_bytes()
            ));
        }
        // Allocate KV cache from budget (if not already allocated)
        if self.budget.get("kv_cache").is_none() {
            self.budget
                .allocate("kv_cache", kv_bytes)
                .map_err(|e| format!("failed to allocate KV cache: {}", e))?;
        }

        // Process prompt tokens one by one to fill KV cache
        let mut all_tokens = prompt_tokens.clone();
        let mut hidden = None;

        for (pos, &token_id) in prompt_tokens.iter().enumerate() {
            let h = self
                .model
                .forward_single(token_id, pos, &mut kv_cache, &self.backend)?;
            hidden = Some(h);
        }

        // Generate
        let mut generated_tokens = Vec::new();
        let mut current_pos = prompt_tokens.len();

        for _ in 0..max_tokens {
            let hidden_state = hidden.as_ref().ok_or("no hidden state")?;
            let logits = self.model.compute_logits(hidden_state, &self.backend)?;

            let next_token = sampler.sample(&logits);

            // Check EOS
            if let Some(eos_id) = self.tokenizer.eos_id {
                if next_token == eos_id {
                    break;
                }
            }

            generated_tokens.push(next_token);
            all_tokens.push(next_token);

            // Forward next token
            let h = self.model.forward_single(
                next_token,
                current_pos,
                &mut kv_cache,
                &self.backend,
            )?;
            hidden = Some(h);
            current_pos += 1;

            if current_pos >= self.model.config.context_length {
                break;
            }
        }

        // Store kv_cache back for stats
        self.kv_cache = Some(kv_cache);

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
        let mut add_kv = |key: &str, val_type: u32, mut write_val: Box<dyn FnMut(&mut Vec<u8>)>| {
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
            ("blk.0.ffn_gate.weight", vec![16, 8], 0),
            ("blk.0.ffn_up.weight", vec![16, 8], 0),
            ("blk.0.ffn_down.weight", vec![8, 16], 0),
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
        // With deterministic tiny model, output should be deterministic
        // We don't assert exact text, but check that we generated 5 tokens and text is not empty and deterministic
        assert_eq!(tokens.len(), 5);
        // Run again and check determinism
        let mut engine2 = InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        let (tokens2, text2) = engine2.generate("hello", 5, &sampler).unwrap();
        assert_eq!(tokens, tokens2);
        assert_eq!(text, text2);
    }
}
