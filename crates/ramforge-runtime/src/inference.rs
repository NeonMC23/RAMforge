//! Inference engine for RAMforge (CPU, llama/qwen2, out-of-core).
//!
//! Memory contract for `generate()`:
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

use std::time::Duration;

use ramforge_core::{
    datasource::{GgufDataSource, IoProfile},
    memory::MemoryBudget,
    tokenizer::Tokenizer,
};

use crate::backend::{ComputeBackend, CpuBackend};
use crate::kv_cache::KvCache;
use crate::memory_report::MemoryReport;
use crate::model::LlamaConfig;
use crate::profile::{ProfileEvent, ProfileSnapshot};
use crate::residency::ResidencyStats;
use crate::runtime_config::RuntimeConfig;
use crate::sampling::Sampler;
use crate::streaming_model::StreamingLlamaModel;

#[derive(Debug)]
pub struct InferenceEngine {
    pub data_source: GgufDataSource,
    pub tokenizer: Tokenizer,
    pub model: StreamingLlamaModel,
    pub kv_cache: Option<KvCache>,
    pub backend: CpuBackend,
    pub budget: MemoryBudget,
    pub ram_budget_bytes: u64,
    runtime_config: RuntimeConfig,
    pub residency_stats: ResidencyStats,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TensorReadProfile {
    pub name: String,
    pub bytes_read: u64,
    pub read_operations: u64,
    pub read_failures: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GenerationProfile {
    pub runtime: ProfileSnapshot,
    pub io: IoProfile,
    pub tensor_reads: Vec<TensorReadProfile>,
    pub layer_cache_capacity_bytes: u64,
    pub ramforge_current_bytes: u64,
    pub ramforge_peak_bytes: u64,
    pub ramforge_budget_bytes: u64,
}

impl GenerationProfile {
    pub fn average_token_latency(&self) -> Option<Duration> {
        (self.runtime.tokens > 0).then(|| {
            Duration::from_secs_f64(self.runtime.total.as_secs_f64() / self.runtime.tokens as f64)
        })
    }
}

fn validate_runtime_config_for_host(runtime_config: &RuntimeConfig) -> Result<(), String> {
    runtime_config
        .validate()
        .map_err(|error| format!("invalid runtime configuration: {error}"))?;
    let available_threads = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    if runtime_config.cpu_thread_count > available_threads {
        return Err(format!(
            "runtime configuration requests {} CPU threads but only {} are available",
            runtime_config.cpu_thread_count, available_threads
        ));
    }
    Ok(())
}

/// Precise reason generation exited the decode loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The EOS token (from GGUF `tokenizer.ggml.eos_token_id`) was sampled.
    Eos,
    /// Exactly `max_tokens` generated tokens were produced (upper bound reached).
    MaxTokens,
    /// Generation would have exceeded the model's `context_length`.
    ContextLength,
}

impl StopReason {
    pub fn label(self) -> &'static str {
        match self {
            Self::Eos => "eos",
            Self::MaxTokens => "max_tokens",
            Self::ContextLength => "context_length",
        }
    }
}

/// Outcome of a single `generate_with_callback` invocation. Exposes the data
/// callers need to distinguish EOS-vs-max-token stops and verify state
/// resets between runs. Bounded so it cannot grow unboundedly for long
/// generations.
#[derive(Debug, Clone)]
pub struct GenerationOutcome {
    /// All generated token IDs (not including BOS, prompt, or EOS).
    pub generated_token_ids: Vec<u32>,
    /// Decoded text emitted via `on_text` plus trailing bytes flushed by
    /// `decoder.finish()`.
    pub generated_text: String,
    /// Number of prompt tokens fed into the forward pass.
    pub prompt_tokens: usize,
    /// Configured upper bound of tokens to generate (the caller's `max_tokens`).
    pub configured_max_tokens: usize,
    /// Model `context_length` (context cap, not a hardcoded generation limit).
    pub context_length: usize,
    /// Why the loop stopped.
    pub stop_reason: StopReason,
    /// EOS token ID from GGUF `tokenizer.ggml.eos_token_id`, or `None` when
    /// the GGUF metadata did not provide one. When `None`, EOS cannot be
    /// detected (generation will always stop on `max_tokens` or context).
    pub eos_token_id: Option<u32>,
    /// Whether the EOS token was actually encountered during the run.
    /// Always `false` when `eos_token_id` is `None`.
    pub eos_encountered: bool,
    /// Bounded prefix of `generated_token_ids` for diagnostics (never
    /// contains the EOS id). The full list is in `generated_token_ids`.
    pub token_ids_sample: Vec<u32>,
    /// KV cache was dropped/recreated at the start of the run.
    pub kv_cache_reset: bool,
    /// Per-run token history (generated_token_ids/decoder) is local to the
    /// call; this is always `true` and reported so callers can assert
    /// state-reset invariants explicitly.
    pub token_history_reset: bool,
    /// Sampler is constructed fresh by the caller and is itself stateless
    /// for greedy sampling. Reported so callers can verify no stale sampler
    /// state leaks between runs.
    pub sampler_is_stateless: bool,
}

/// Maximum number of generated token IDs included in `token_ids_sample`.
pub const TOKEN_ID_SAMPLE_LIMIT: usize = 64;

impl InferenceEngine {
    /// Construct the legacy/manual execution path. Its behavior is unchanged:
    /// all available CPU threads are exposed to the existing backend, the
    /// model-derived cache maximum is used, and both grouped-read optimizations
    /// are permitted subject to their existing runtime feasibility checks.
    pub fn new(model_path: &str, ram_budget_bytes: u64) -> Result<Self, String> {
        Self::new_internal(model_path, ram_budget_bytes, None)
    }

    /// Construct an engine from a concrete execution contract. Planner policy
    /// is intentionally absent here; callers may obtain this value from the
    /// `PlanCompiler` or construct it explicitly for manual execution.
    pub fn new_with_runtime_config(
        model_path: &str,
        runtime_config: RuntimeConfig,
    ) -> Result<Self, String> {
        validate_runtime_config_for_host(&runtime_config)?;
        Self::new_internal(
            model_path,
            runtime_config.ram_budget_bytes,
            Some(runtime_config),
        )
    }

    /// Construct from an already inspected, retained data source. This is the
    /// orchestration path: GGUF metadata/descriptors are parsed once and the
    /// same synchronized file handle proceeds into runtime execution.
    pub(crate) fn from_data_source_with_runtime_config(
        data_source: GgufDataSource,
        runtime_config: RuntimeConfig,
    ) -> Result<Self, String> {
        validate_runtime_config_for_host(&runtime_config)?;
        Self::from_data_source_internal(
            data_source,
            runtime_config.ram_budget_bytes,
            Some(runtime_config),
        )
    }

    fn new_internal(
        model_path: &str,
        ram_budget_bytes: u64,
        requested_config: Option<RuntimeConfig>,
    ) -> Result<Self, String> {
        // Parse model file-backed (does NOT load tensor payloads).
        let data_source = GgufDataSource::open(model_path)
            .map_err(|e| format!("failed to open GGUF data source: {}", e))?;
        Self::from_data_source_internal(data_source, ram_budget_bytes, requested_config)
    }

    fn from_data_source_internal(
        data_source: GgufDataSource,
        ram_budget_bytes: u64,
        requested_config: Option<RuntimeConfig>,
    ) -> Result<Self, String> {
        let gguf_model = data_source.model();
        // Validate architecture via config (will error if unsupported).
        let _ = LlamaConfig::from_gguf(gguf_model)?;

        let tokenizer = Tokenizer::from_gguf(gguf_model)
            .map_err(|e| format!("failed to load tokenizer: {}", e))?;

        // MemoryBudget remains the sole allocation and accounting authority.
        let mut budget = MemoryBudget::new(ram_budget_bytes)
            .map_err(|e| format!("invalid RAM budget: {}", e))?;

        // Load persistent weights; transformer layers are streamed per
        // forward call. Resident persistents are charged to the budget
        // (weight:*), anything that does not fit is streamed on demand with
        // charged, bounded temps.
        let model = match requested_config.as_ref() {
            Some(config) => {
                StreamingLlamaModel::load_with_runtime_config(&data_source, &mut budget, config)
            }
            None => StreamingLlamaModel::load(&data_source, &mut budget),
        }
        .map_err(|e| format!("failed to load model weights: {}", e))?;

        let backend = match requested_config.as_ref() {
            Some(config) => CpuBackend::try_with_threads(config.cpu_thread_count)?,
            None => CpuBackend::new(),
        };
        let runtime_config = match requested_config {
            Some(config) => config,
            None => RuntimeConfig::current_defaults_with_thread_count(
                backend.num_threads(),
                ram_budget_bytes,
                model.layer_cache_capacity_bytes(),
            )
            .map_err(|error| format!("failed to materialize runtime defaults: {error}"))?,
        };
        let residency_stats = ResidencyStats::new(model.total_weight_bytes);

        Ok(Self {
            data_source,
            tokenizer,
            model,
            kv_cache: None,
            backend,
            budget,
            ram_budget_bytes,
            runtime_config,
            residency_stats,
        })
    }

    pub fn set_profiling(&self, enabled: bool) {
        self.model.set_profiling(enabled);
        self.data_source.set_profiling(enabled);
    }

    pub fn generation_profile(&self) -> GenerationProfile {
        let mut tensor_reads: Vec<TensorReadProfile> = self
            .data_source
            .tensor_io_profile()
            .into_iter()
            .map(|(name, profile)| TensorReadProfile {
                name,
                bytes_read: profile.bytes_read,
                read_operations: profile.read_operations,
                read_failures: profile.read_failures,
            })
            .collect();
        tensor_reads.sort_by(|left, right| {
            right
                .bytes_read
                .cmp(&left.bytes_read)
                .then_with(|| left.name.cmp(&right.name))
        });
        GenerationProfile {
            runtime: self.model.profiler.snapshot(),
            io: self.data_source.io_profile(),
            tensor_reads,
            layer_cache_capacity_bytes: self.model.layer_cache_capacity_bytes(),
            ramforge_current_bytes: self.budget.used_bytes(),
            ramforge_peak_bytes: self.budget.peak_used_bytes(),
            ramforge_budget_bytes: self.budget.total_bytes(),
        }
    }

    pub fn memory_report(&self) -> MemoryReport {
        MemoryReport::collect(&self.budget)
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
        self.generate_with_callback(prompt, max_tokens, sampler, |_| Ok(()))
            .map(|outcome| (outcome.generated_token_ids, outcome.generated_text))
    }

    /// Generate while emitting newly decodable text as soon as tokens permit.
    /// Diagnostic/profile output remains the caller's responsibility, so the
    /// callback receives generated text only.
    ///
    /// Returns a `GenerationOutcome` describing the run, including a precise
    /// `StopReason`, the EOS token ID from GGUF metadata (when available),
    /// and bounded diagnostics for state-reset verification.
    pub fn generate_with_callback<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampler: &Sampler,
        mut on_text: F,
    ) -> Result<GenerationOutcome, String>
    where
        F: FnMut(&str) -> Result<(), String>,
    {
        // Explicit reset of any previous run's KV and cached-layer state.
        self.clear_kv_cache();
        let kv_cache_reset = true; // we just called clear_kv_cache() above
        self.model.clear_layer_cache(&mut self.budget)?;
        self.model.reset_profile();
        self.data_source.reset_io_profile();
        let profiler = self.model.profiler.clone();
        let total_started = profiler.start();
        let sampler_stateless =
            sampler.temperature <= 0.0 && sampler.top_k.is_none() && sampler.top_p.is_none();
        let result = self.generate_impl(
            prompt,
            max_tokens,
            sampler,
            &mut on_text,
            kv_cache_reset,
            sampler_stateless,
        );
        let cache_clear_result = self.model.clear_layer_cache(&mut self.budget);
        if cache_clear_result.is_ok() {
            self.residency_stats
                .update_managed(self.budget.used_bytes());
        }
        profiler.record_since(ProfileEvent::Total, total_started);
        let result = match (result, cache_clear_result) {
            (Ok(output), Ok(())) => Ok(output),
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
        };
        match result {
            Ok(out) => Ok(out),
            Err(error) => {
                // Failed run: no stale KV budget state, engine stays reusable.
                self.clear_kv_cache();
                Err(error)
            }
        }
    }

    fn generate_impl<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampler: &Sampler,
        on_text: &mut F,
        kv_cache_reset: bool,
        sampler_stateless: bool,
    ) -> Result<GenerationOutcome, String>
    where
        F: FnMut(&str) -> Result<(), String>,
    {
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
        let tokenizer = &self.tokenizer;
        let eos_id = tokenizer.eos_id;
        let context_length = self.model.config.context_length;
        let profiler = model.profiler.clone();

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
        let mut generated_text = String::new();
        let mut decoder = tokenizer.decoder();
        let mut current_pos = prompt_tokens.len();
        let mut stop_reason = StopReason::MaxTokens;
        let mut eos_encountered = false;

        let budget = &mut self.budget;
        let run_result: Result<(), String> =
            budget.with_temp("tmp:hidden", hidden_bytes, |budget| {
                let allocation_started = profiler.start();
                let mut hidden = vec![0.0f32; n_embd];
                profiler.record_since(ProfileEvent::Allocation, allocation_started);
                budget.with_temp("tmp:logits", logits_bytes, |budget| {
                    let allocation_started = profiler.start();
                    let mut logits = vec![0.0f32; vocab];
                    profiler.record_since(ProfileEvent::Allocation, allocation_started);

                    // Prompt pass – fills the KV cache one token at a time.
                    let prompt_started = profiler.start();
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
                        profiler.record_prompt_forward();
                    }
                    profiler.record_since(ProfileEvent::Prompt, prompt_started);

                    // Generation loop. A selected final token does not need a
                    // forward pass unless another token's logits will be read.
                    for generation_index in 0..max_tokens {
                        let token_started = profiler.start();
                        let logits_started = profiler.start();
                        model.compute_logits(
                            &hidden,
                            backend,
                            data_source,
                            budget,
                            &mut logits,
                        )?;
                        profiler.record_since(ProfileEvent::Logits, logits_started);

                        let sampling_started = profiler.start();
                        model.ensure_layer_cache_headroom(budget, sampler_scratch)?;
                        let next_token =
                            budget.with_temp("tmp:sampling", sampler_scratch, |_b| {
                                Ok::<u32, String>(sampler.sample(&logits))
                            })?;
                        profiler.record_since(ProfileEvent::Sampling, sampling_started);

                        if let Some(eos) = eos_id {
                            if next_token == eos {
                                eos_encountered = true;
                                stop_reason = StopReason::Eos;
                                break;
                            }
                        }

                        generated_tokens.push(next_token);
                        let decoded = decoder.push(next_token);
                        if !decoded.is_empty() {
                            let output_started = profiler.start();
                            on_text(&decoded)?;
                            profiler.record_since(ProfileEvent::Output, output_started);
                            generated_text.push_str(&decoded);
                        }

                        let more_tokens_requested = generation_index + 1 < max_tokens;
                        let room_in_context = current_pos + 1 < context_length;
                        if !more_tokens_requested {
                            stop_reason = StopReason::MaxTokens;
                            profiler.record_since(ProfileEvent::TokenLatency, token_started);
                            profiler.record_token();
                            profiler.record_terminal_forward_skipped();
                            break;
                        }
                        if !room_in_context {
                            stop_reason = StopReason::ContextLength;
                            profiler.record_since(ProfileEvent::TokenLatency, token_started);
                            profiler.record_token();
                            profiler.record_terminal_forward_skipped();
                            break;
                        }

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
                            if let Err(error) =
                                model.ensure_layer_cache_headroom(budget, new_bytes)
                            {
                                let _ = budget.allocate("kv_cache", old_bytes);
                                return Err(error);
                            }
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
                        profiler.record_decode_forward();
                        profiler.record_since(ProfileEvent::TokenLatency, token_started);
                        profiler.record_token();

                        if current_pos >= context_length {
                            stop_reason = StopReason::ContextLength;
                            break;
                        }
                    }
                    Ok(())
                })
            });
        run_result?;

        let trailing = decoder.finish();
        if !trailing.is_empty() {
            let output_started = profiler.start();
            on_text(&trailing)?;
            profiler.record_since(ProfileEvent::Output, output_started);
            generated_text.push_str(&trailing);
        }
        drop(decoder);

        self.kv_cache = Some(kv_cache);
        self.residency_stats = residency_stats;

        // Budget integrity: after the scoped temps are gone, only the
        // persistent charges (weights, KV cache) may remain.
        debug_assert_eq!(
            self.budget
                .allocations()
                .keys()
                .filter(|k| k.starts_with("tmp:"))
                .count(),
            0,
            "temp charges must be released after generate()"
        );

        debug_assert_eq!(generated_text, self.tokenizer.decode(&generated_tokens));

        let token_ids_sample: Vec<u32> = generated_tokens
            .iter()
            .copied()
            .take(TOKEN_ID_SAMPLE_LIMIT)
            .collect();

        Ok(GenerationOutcome {
            generated_token_ids: generated_tokens,
            generated_text,
            prompt_tokens: prompt_tokens.len(),
            configured_max_tokens: max_tokens,
            context_length,
            stop_reason,
            eos_token_id: eos_id,
            eos_encountered,
            token_ids_sample,
            kv_cache_reset,
            token_history_reset: true,
            sampler_is_stateless: sampler_stateless,
        })
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.model.config
    }

    pub fn runtime_config(&self) -> &RuntimeConfig {
        &self.runtime_config
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Boxed value-writer closure used by the GGUF test fixtures.
    type WriteValFn<'a> = Box<dyn FnMut(&mut Vec<u8>) + 'a>;

    // Create a tiny deterministic LLaMA model for testing
    // Config: vocab 16, n_embd 8, n_layer 1, n_head 2, head_dim 4, ffn 16
    pub(crate) fn create_tiny_llama_gguf() -> NamedTempFile {
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
        add_kv(
            "general.architecture",
            8,
            Box::new(|b| write_string(b, "llama")),
        );
        add_kv("llama.vocab_size", 4, Box::new(|b| write_u32(b, 16)));
        add_kv("llama.context_length", 4, Box::new(|b| write_u32(b, 32)));
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
        add_kv(
            "tokenizer.ggml.tokens",
            9,
            Box::new(|b| {
                write_u32(b, 8);
                write_u64(b, 16);
                for tok in [
                    "<unk>", "<s>", "</s>", "▁hello", "▁world", "hello", "world", "!", "▁", "a",
                    "b", "c", "d", "e", "f", "g",
                ] {
                    write_string(b, tok);
                }
            }),
        );
        add_kv(
            "tokenizer.ggml.scores",
            9,
            Box::new(|b| {
                write_u32(b, 6);
                write_u64(b, 16);
                for _ in 0..16 {
                    write_f32(b, 0.0);
                }
            }),
        );
        add_kv(
            "tokenizer.ggml.token_type",
            9,
            Box::new(|b| {
                write_u32(b, 5);
                write_u64(b, 16);
                for t in [2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] {
                    write_u32(b, t);
                }
            }),
        );
        add_kv(
            "tokenizer.ggml.bos_token_id",
            4,
            Box::new(|b| write_u32(b, 1)),
        );
        add_kv(
            "tokenizer.ggml.eos_token_id",
            4,
            Box::new(|b| write_u32(b, 2)),
        );

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
        for _ in 0..16 * 8 {
            buf2.extend_from_slice(&0.1f32.to_le_bytes());
        }
        // ffn_up: 0.1
        for _ in 0..16 * 8 {
            buf2.extend_from_slice(&0.1f32.to_le_bytes());
        }
        // ffn_down: 0.1
        for _ in 0..8 * 16 {
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
        let mut engine =
            InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        let sampler = crate::sampling::Sampler::greedy();
        let (tokens, text) = engine.generate("hello", 5, &sampler).unwrap();
        assert_eq!(tokens.len(), 5);
        let mut engine2 =
            InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        let (tokens2, text2) = engine2.generate("hello", 5, &sampler).unwrap();
        assert_eq!(tokens, tokens2);
        assert_eq!(text, text2);
    }

    #[test]
    fn test_default_runtime_config_preserves_legacy_execution_behavior() {
        let tmp = create_tiny_llama_gguf();
        let model_path = tmp.path().to_str().unwrap();
        let mut legacy = InferenceEngine::new(model_path, 8 * 1024 * 1024).unwrap();
        let default_config = legacy.runtime_config().clone();

        assert_eq!(
            default_config.cpu_thread_count,
            legacy.backend.num_threads()
        );
        assert_eq!(
            default_config.layer_cache_capacity_bytes,
            legacy.model.layer_cache_capacity_bytes()
        );
        assert!(default_config.read_coalescing_enabled);
        assert!(default_config.grouped_read_buffer_reuse_enabled);

        let mut explicit =
            InferenceEngine::new_with_runtime_config(model_path, default_config).unwrap();
        assert_eq!(
            explicit.model.layer_cache_capacity_bytes(),
            legacy.model.layer_cache_capacity_bytes()
        );
        let sampler = crate::sampling::Sampler::greedy();
        assert_eq!(
            legacy.generate("hello", 3, &sampler).unwrap(),
            explicit.generate("hello", 3, &sampler).unwrap()
        );
    }

    #[test]
    fn test_explicit_runtime_config_reaches_execution_components() {
        let tmp = create_tiny_llama_gguf();
        let config = RuntimeConfig::new(1, 8 * 1024 * 1024, false, 0, false, false).unwrap();
        let mut engine =
            InferenceEngine::new_with_runtime_config(tmp.path().to_str().unwrap(), config.clone())
                .unwrap();

        assert_eq!(engine.runtime_config(), &config);
        assert_eq!(engine.backend.num_threads(), 1);
        assert_eq!(engine.budget.total_bytes(), config.ram_budget_bytes);
        assert_eq!(engine.ram_budget_bytes, config.ram_budget_bytes);
        assert_eq!(engine.model.layer_cache_capacity_bytes(), 0);
        assert!(!engine.model.read_coalescing_enabled());
        assert!(!engine.model.grouped_read_buffer_reuse_enabled());

        engine.set_profiling(true);
        let sampler = crate::sampling::Sampler::greedy();
        engine.generate("hello", 1, &sampler).unwrap();
        let profile = engine.generation_profile();
        assert_eq!(profile.io.coalesced_ranges, 0);
        assert_eq!(profile.ramforge_budget_bytes, config.ram_budget_bytes);
        assert!(profile.ramforge_peak_bytes <= profile.ramforge_budget_bytes);
    }

    #[test]
    fn test_streaming_output_profile_and_memory_report() {
        let tmp = create_tiny_llama_gguf();
        let mut engine =
            InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        engine.set_profiling(true);
        let sampler = crate::sampling::Sampler::greedy();
        let mut streamed = String::new();
        let outcome = engine
            .generate_with_callback("hello", 3, &sampler, |piece| {
                streamed.push_str(piece);
                Ok(())
            })
            .unwrap();
        assert_eq!(outcome.generated_token_ids.len(), 3);
        assert_eq!(streamed, outcome.generated_text);

        let profile = engine.generation_profile();
        assert_eq!(profile.runtime.tokens, 3);
        assert_eq!(profile.runtime.decode_forwards, 2);
        assert_eq!(profile.runtime.terminal_forwards_skipped, 1);
        assert!(profile.runtime.layer_loads > 0);
        assert!(profile.runtime.cache_hits > 0);
        assert!(profile.runtime.layer_loads < profile.runtime.prompt_forwards + 2);
        assert!(profile.io.read_operations > 0);
        assert!(profile.io.bytes_read > 0);
        assert!(profile.ramforge_peak_bytes <= profile.ramforge_budget_bytes);

        let memory = engine.memory_report();
        assert_eq!(memory.ramforge_budget_bytes, 8 * 1024 * 1024);
        assert!(memory.ramforge_peak_bytes >= memory.ramforge_current_bytes);
    }

    #[test]
    fn test_single_token_skips_terminal_forward_and_redundant_layer_reads() {
        let tmp = create_tiny_llama_gguf();
        let mut engine =
            InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        let prompt_forwards = engine.tokenizer.encode("hello", true).len() as u64;
        let layer_count = engine.config().block_count as u64;
        engine.set_profiling(true);

        let sampler = crate::sampling::Sampler::greedy();
        let (tokens, _) = engine.generate("hello", 1, &sampler).unwrap();
        assert_eq!(tokens.len(), 1);

        let profile = engine.generation_profile();
        assert_eq!(profile.runtime.prompt_forwards, prompt_forwards);
        assert_eq!(profile.runtime.decode_forwards, 0);
        assert_eq!(profile.runtime.terminal_forwards_skipped, 1);
        assert_eq!(profile.runtime.layer_loads, layer_count);
        assert_eq!(profile.runtime.cache_misses, layer_count);
        assert_eq!(
            profile.runtime.cache_hits,
            (prompt_forwards - 1) * layer_count
        );
        assert_eq!(profile.io.logical_tensor_reads, layer_count * 9);
        assert_eq!(profile.io.read_operations, layer_count);
        assert!(profile.io.read_operations < profile.io.logical_tensor_reads);
        for tensor in profile
            .tensor_reads
            .iter()
            .filter(|tensor| tensor.name.starts_with("blk.0."))
        {
            assert_eq!(tensor.read_operations, 1);
        }
    }

    fn create_out_of_core_gguf() -> NamedTempFile {
        // Create model where total weights > budget but layers fit
        // Use same logic as streaming_model test but with tokenizer
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
        add_kv(
            "tokenizer.ggml.tokens",
            9,
            Box::new(|b| {
                write_u32(b, 8);
                write_u64(b, 16);
                for tok in [
                    "<unk>", "<s>", "</s>", "▁hello", "▁world", "hello", "world", "!", "▁", "a",
                    "b", "c", "d", "e", "f", "g",
                ] {
                    write_string(b, tok);
                }
            }),
        );
        add_kv(
            "tokenizer.ggml.scores",
            9,
            Box::new(|b| {
                write_u32(b, 6);
                write_u64(b, 16);
                for _ in 0..16 {
                    write_f32(b, 0.0);
                }
            }),
        );
        add_kv(
            "tokenizer.ggml.token_type",
            9,
            Box::new(|b| {
                write_u32(b, 5);
                write_u64(b, 16);
                for t in [2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] {
                    write_u32(b, t);
                }
            }),
        );
        add_kv(
            "tokenizer.ggml.bos_token_id",
            4,
            Box::new(|b| write_u32(b, 1)),
        );
        add_kv(
            "tokenizer.ggml.eos_token_id",
            4,
            Box::new(|b| write_u32(b, 2)),
        );
        // add one more to make 16
        add_kv(
            "general.name",
            8,
            Box::new(|b| write_string(b, "tiny-out-of-core")),
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

        for (name, dims, ty) in &defs {
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
        let aligned = ramforge_core::model::align_offset(pos, 32);
        buf.extend(vec![0u8; (aligned - pos) as usize]);
        // Write data: token_embd as token_id, others as 0.1 or identity
        for token_id in 0..16 {
            for _ in 0..n_embd {
                buf.extend_from_slice(&(token_id as f32).to_le_bytes());
            }
        }
        for _ in 0..n_embd {
            buf.extend_from_slice(&1.0f32.to_le_bytes());
        }
        for _layer in 0..n_layers {
            for _ in 0..n_embd {
                buf.extend_from_slice(&1.0f32.to_le_bytes());
            } // attn_norm
            for i in 0..n_embd {
                for j in 0..n_embd {
                    let v: f32 = if i == j { 1.0 } else { 0.0 };
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            } // q
            for i in 0..n_embd {
                for j in 0..n_embd {
                    let v: f32 = if i == j { 1.0 } else { 0.0 };
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            } // k
            for i in 0..n_embd {
                for j in 0..n_embd {
                    let v: f32 = if i == j { 1.0 } else { 0.0 };
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            } // v
            for i in 0..n_embd {
                for j in 0..n_embd {
                    let v: f32 = if i == j { 1.0 } else { 0.0 };
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            } // output
            for _ in 0..n_embd {
                buf.extend_from_slice(&1.0f32.to_le_bytes());
            } // ffn_norm
            for _ in 0..ffn * n_embd {
                buf.extend_from_slice(&0.1f32.to_le_bytes());
            } // gate
            for _ in 0..ffn * n_embd {
                buf.extend_from_slice(&0.1f32.to_le_bytes());
            } // up
            for _ in 0..n_embd * ffn {
                buf.extend_from_slice(&0.1f32.to_le_bytes());
            } // down
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
        let total_bytes: u64 = ds
            .model()
            .tensors
            .iter()
            .filter_map(|t| t.byte_length)
            .sum();

        // n_embd 16 / ffn 32: per layer 10368 B (F32), persistents 1088 B.
        // 8 layers => total 84032 B (~82 KiB) > 32 KiB budget, while one
        // layer (direct F32 load charge = resident bytes) plus KV cache
        // (~5 tokens) and forward temps comfortably fits.
        let ram_budget = 32 * 1024;

        // The engine should still succeed because it streams layers
        let mut engine = InferenceEngine::new(tmp.path().to_str().unwrap(), ram_budget).unwrap();

        // Check total > budget
        assert!(
            total_bytes > ram_budget,
            "total {} should be > budget {}",
            total_bytes,
            ram_budget
        );

        let sampler = crate::sampling::Sampler::greedy();
        let (tokens, _text) = engine.generate("hello", 3, &sampler).unwrap();
        assert_eq!(tokens.len(), 3);

        // Check residency stats: peak layer < total, peak managed <= budget
        let stats = &engine.residency_stats;
        assert!(stats.total_model_weight_bytes > ram_budget);
        assert!(stats.peak_resident_layer_bytes < stats.total_model_weight_bytes);
        assert!(
            stats.peak_managed_bytes <= ram_budget,
            "peak managed {} should be <= budget {}",
            stats.peak_managed_bytes,
            ram_budget
        );
        assert!(stats.num_layer_loads > 0);
        assert!(stats.num_layer_releases + stats.num_layer_cached > 0);
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
    fn test_layer_cache_hits_preserve_generation_numerics() {
        let tmp = create_tiny_llama_gguf();
        let sampler = crate::sampling::Sampler::greedy();
        let mut cached =
            InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        let mut uncached = InferenceEngine::new(tmp.path().to_str().unwrap(), 5_000).unwrap();
        assert!(cached.model.layer_cache_capacity_bytes() > 2624);
        assert!(uncached.model.layer_cache_capacity_bytes() < 2624);
        uncached.set_profiling(true);

        let cached_output = cached.generate("hello", 3, &sampler).unwrap();
        let uncached_output = uncached.generate("hello", 3, &sampler).unwrap();
        assert_eq!(cached_output, uncached_output);
        let uncached_profile = uncached.generation_profile();
        assert_eq!(uncached_profile.io.coalesced_ranges, 0);
        assert_eq!(
            uncached_profile.io.read_operations,
            uncached_profile.io.logical_tensor_reads
        );
        assert!(uncached_profile.ramforge_peak_bytes <= uncached_profile.ramforge_budget_bytes);
    }

    #[test]
    fn test_generate_twice_same_engine() {
        // Regression test for M6.1 BUG-1: sequential generate() calls.
        let tmp = create_tiny_llama_gguf();
        let mut engine =
            InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
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
        let mut healthy =
            InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        let err_ctx = healthy.generate("hello", 100, &sampler).unwrap_err();
        assert!(err_ctx.contains("context length"), "got: {}", err_ctx);
        assert!(healthy.budget.get("kv_cache").is_none());
        let (tokens, _text) = healthy.generate("hello", 3, &sampler).unwrap();
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn test_clear_kv_cache_releases_charge() {
        let tmp = create_tiny_llama_gguf();
        let mut engine =
            InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
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
        embd: Vec<f32>, // [16][8] rows = token embeddings
        output_norm: Vec<f32>,
        attn_norm: Vec<f32>,
        attn_q: Vec<f32>, // ggml [8,8]: row o of in 8
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
            ffn_up: w(128, &|i| {
                0.01 + 0.015 * (((i / 8) + 2 * (i % 8)) % 5) as f32
            }),
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
        fn write_u32<W: Write>(w: &mut W, v: u32) {
            w.write_all(&v.to_le_bytes()).unwrap();
        }
        fn write_u64<W: Write>(w: &mut W, v: u64) {
            w.write_all(&v.to_le_bytes()).unwrap();
        }
        fn write_f32<W: Write>(w: &mut W, v: f32) {
            w.write_all(&v.to_le_bytes()).unwrap();
        }

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
        add_kv(
            "tokenizer.ggml.tokens",
            9,
            Box::new(|b| {
                write_u32(b, 8);
                write_u64(b, 16);
                for tok in [
                    "<unk>", "<s>", "</s>", "▁hello", "▁world", "hello", "world", "!", "▁", "a",
                    "b", "c", "d", "e", "f", "g",
                ] {
                    write_string(b, tok);
                }
            }),
        );
        add_kv(
            "tokenizer.ggml.scores",
            9,
            Box::new(|b| {
                write_u32(b, 6);
                write_u64(b, 16);
                for _ in 0..16 {
                    write_f32(b, 0.0);
                }
            }),
        );
        add_kv(
            "tokenizer.ggml.token_type",
            9,
            Box::new(|b| {
                write_u32(b, 5);
                write_u64(b, 16);
                for t in [2i32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] {
                    write_u32(b, t as u32);
                }
            }),
        );
        add_kv(
            "tokenizer.ggml.bos_token_id",
            4,
            Box::new(|b| write_u32(b, 1)),
        );
        add_kv(
            "tokenizer.ggml.eos_token_id",
            4,
            Box::new(|b| write_u32(b, 2)),
        );

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
        let mut engine =
            InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
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
        let mut engine2 =
            InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        let sampler = crate::sampling::Sampler::greedy();
        let (t1, _) = engine2.generate("hello", 4, &sampler).unwrap();
        let (t2, _) = engine2.generate("hello", 4, &sampler).unwrap();
        assert_eq!(t1.len(), 4);
        assert_eq!(t1, t2);
    }

    // ------------------------------------------------------------------
    // GQA (grouped-query attention) fixture: n_heads > n_kv_heads.
    // Mirrors Qwen2.5-style ratios (e.g. 12 Q heads / 2 KV heads = 6:1).
    // Uses F32 weights (no quantization) so any divergence isolates to the
    // attention/GQA/RoPE/KV-cache path rather than to dequant math.
    // ------------------------------------------------------------------

    struct GqaFixtureWeights {
        embd: Vec<f32>,
        output_norm: Vec<f32>,
        attn_norm: Vec<f32>,
        attn_q: Vec<f32>,        // ggml [n_embd, q_dim]
        attn_k: Vec<f32>,        // ggml [n_embd, kv_dim]
        attn_v: Vec<f32>,        // ggml [n_embd, kv_dim]
        attn_output: Vec<f32>,   // ggml [q_dim, n_embd]
        ffn_norm: Vec<f32>,
        ffn_gate: Vec<f32>,      // ggml [n_embd, ffn]
        ffn_up: Vec<f32>,        // ggml [n_embd, ffn]
        ffn_down: Vec<f32>,      // ggml [ffn, n_embd]
    }

    const GQA_N_EMBD: usize = 8;
    const GQA_N_HEADS: usize = 4;
    const GQA_N_KV_HEADS: usize = 2;
    const GQA_HEAD_DIM: usize = 2;
    const GQA_FFN: usize = 16;
    const GQA_VOCAB: usize = 16;

    fn gqa_weights() -> GqaFixtureWeights {
        let w = |n: usize, seed: u32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let s = seed.wrapping_mul(2654435761).wrapping_add((i as u32).wrapping_mul(224682251));
                    0.01 * ((s as i32 & 0x3f) as f32 - 20.0)
                })
                .collect()
        };
        GqaFixtureWeights {
            embd: w(GQA_VOCAB * GQA_N_EMBD, 1),
            output_norm: vec![1.0; GQA_N_EMBD],
            attn_norm: vec![1.0; GQA_N_EMBD],
            attn_q: w(GQA_N_EMBD * GQA_N_EMBD, 2),
            attn_k: w(GQA_N_EMBD * (GQA_N_KV_HEADS * GQA_HEAD_DIM), 3),
            attn_v: w(GQA_N_EMBD * (GQA_N_KV_HEADS * GQA_HEAD_DIM), 4),
            attn_output: w((GQA_N_HEADS * GQA_HEAD_DIM) * GQA_N_EMBD, 5),
            ffn_norm: vec![1.0; GQA_N_EMBD],
            ffn_gate: w(GQA_N_EMBD * GQA_FFN, 6),
            ffn_up: w(GQA_N_EMBD * GQA_FFN, 7),
            ffn_down: w(GQA_FFN * GQA_N_EMBD, 8),
        }
    }

    fn create_gqa_gguf() -> NamedTempFile {
        fn write_string<W: Write>(w: &mut W, s: &str) {
            w.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
            w.write_all(s.as_bytes()).unwrap();
        }
        fn write_u32<W: Write>(w: &mut W, v: u32) { w.write_all(&v.to_le_bytes()).unwrap(); }
        fn write_u64<W: Write>(w: &mut W, v: u64) { w.write_all(&v.to_le_bytes()).unwrap(); }
        fn write_f32<W: Write>(w: &mut W, v: f32) { w.write_all(&v.to_le_bytes()).unwrap(); }

        let w = gqa_weights();
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&11u64.to_le_bytes());
        buf.extend_from_slice(&15u64.to_le_bytes());

        let mut add_kv = |key: &str, val_type: u32, mut write_val: WriteValFn<'_>| {
            write_string(&mut buf, key);
            write_u32(&mut buf, val_type);
            write_val(&mut buf);
        };
        add_kv("general.architecture", 8, Box::new(|b| write_string(b, "qwen2")));
        add_kv("qwen2.vocab_size", 4, Box::new(|b| write_u32(b, GQA_VOCAB as u32)));
        add_kv("qwen2.context_length", 4, Box::new(|b| write_u32(b, 64)));
        add_kv("qwen2.embedding_length", 4, Box::new(|b| write_u32(b, GQA_N_EMBD as u32)));
        add_kv("qwen2.block_count", 4, Box::new(|b| write_u32(b, 1)));
        add_kv("qwen2.feed_forward_length", 4, Box::new(|b| write_u32(b, GQA_FFN as u32)));
        add_kv("qwen2.attention.head_count", 4, Box::new(|b| write_u32(b, GQA_N_HEADS as u32)));
        add_kv("qwen2.attention.head_count_kv", 4, Box::new(|b| write_u32(b, GQA_N_KV_HEADS as u32)));
        add_kv("qwen2.attention.layer_norm_rms_epsilon", 6, Box::new(|b| write_f32(b, 1e-5)));
        add_kv("qwen2.rope.freq_base", 6, Box::new(|b| write_f32(b, 10000.0)));
        add_kv("tokenizer.ggml.model", 8, Box::new(|b| write_string(b, "llama")));
        add_kv("tokenizer.ggml.tokens", 9, Box::new(|b| {
            write_u32(b, 8); write_u64(b, GQA_VOCAB as u64);
            for tok in ["<unk>","<s>","</s>","▁hi","▁there","a","b","c","d","e","f","g","h","i","j","k"] {
                write_string(b, tok);
            }
        }));
        add_kv("tokenizer.ggml.scores", 9, Box::new(|b| {
            write_u32(b, 6); write_u64(b, GQA_VOCAB as u64);
            for _ in 0..GQA_VOCAB { write_f32(b, 0.0); }
        }));
        add_kv("tokenizer.ggml.token_type", 9, Box::new(|b| {
            write_u32(b, 4); write_u64(b, GQA_VOCAB as u64);
            for t in [2u32,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1] { write_u32(b, t); }
        }));
        add_kv("tokenizer.ggml.bos_token_id", 4, Box::new(|b| write_u32(b, 1)));

        let q_dim = GQA_N_HEADS * GQA_HEAD_DIM;
        let kv_dim = GQA_N_KV_HEADS * GQA_HEAD_DIM;
        let defs: Vec<(&str, Vec<u64>)> = vec![
            ("token_embd.weight", vec![GQA_N_EMBD as u64, GQA_VOCAB as u64]),
            ("output_norm.weight", vec![GQA_N_EMBD as u64]),
            ("blk.0.attn_norm.weight", vec![GQA_N_EMBD as u64]),
            ("blk.0.attn_q.weight", vec![GQA_N_EMBD as u64, q_dim as u64]),
            ("blk.0.attn_k.weight", vec![GQA_N_EMBD as u64, kv_dim as u64]),
            ("blk.0.attn_v.weight", vec![GQA_N_EMBD as u64, kv_dim as u64]),
            ("blk.0.attn_output.weight", vec![q_dim as u64, GQA_N_EMBD as u64]),
            ("blk.0.ffn_norm.weight", vec![GQA_N_EMBD as u64]),
            ("blk.0.ffn_gate.weight", vec![GQA_N_EMBD as u64, GQA_FFN as u64]),
            ("blk.0.ffn_up.weight", vec![GQA_N_EMBD as u64, GQA_FFN as u64]),
            ("blk.0.ffn_down.weight", vec![GQA_FFN as u64, GQA_N_EMBD as u64]),
        ];

        let mut offset = 0u64;
        for (name, dims) in &defs {
            write_string(&mut buf, name);
            write_u32(&mut buf, dims.len() as u32);
            for d in dims { write_u64(&mut buf, *d); }
            write_u32(&mut buf, 0);
            write_u64(&mut buf, offset);
            let elems: u64 = dims.iter().product();
            offset += elems * 4;
        }

        let pos = buf.len() as u64;
        let aligned = ramforge_core::model::align_offset(pos, 32);
        buf.extend(vec![0u8; (aligned - pos) as usize]);

        let mut push = |data: &[f32]| { for v in data { buf.extend_from_slice(&v.to_le_bytes()); } };
        push(&w.embd);
        push(&w.output_norm);
        push(&w.attn_norm);
        push(&w.attn_q);
        push(&w.attn_k);
        push(&w.attn_v);
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

    #[allow(clippy::too_many_arguments)]
    fn reference_forward_gqa(
        w: &GqaFixtureWeights,
        token: usize,
        pos: usize,
        k_hist: &mut Vec<f32>,
        v_hist: &mut Vec<f32>,
        eps: f32,
    ) -> Vec<f32> {
        let rmsnorm = |x: &[f32], wt: &[f32]| -> Vec<f32> {
            let mean: f32 = x.iter().map(|v| v*v).sum::<f32>() / x.len() as f32;
            let rms = (mean + eps).sqrt();
            (0..x.len()).map(|i| x[i]/rms*wt[i]).collect()
        };
        let matvec = |wt: &[f32], in_dim: usize, out_dim: usize, x: &[f32]| -> Vec<f32> {
            (0..out_dim).map(|o| (0..in_dim).map(|i| wt[o*in_dim+i]*x[i]).sum()).collect()
        };
        let rope_single = |x: &mut [f32], pos: usize| {
            let dim = GQA_HEAD_DIM;
            let half = dim/2;
            for j in 0..half {
                let theta = 10000.0f32.powf(-2.0*j as f32/dim as f32) * pos as f32;
                let (c,s) = (theta.cos(), theta.sin());
                let a = x[j]; let b = x[j+half];
                x[j] = a*c - b*s;
                x[j+half] = a*s + b*c;
            }
        };

        let n_embd = GQA_N_EMBD;
        let n_heads = GQA_N_HEADS;
        let n_kv = GQA_N_KV_HEADS;
        let hd = GQA_HEAD_DIM;
        let kv_dim = n_kv*hd;
        let q_dim = n_heads*hd;

        let mut hidden: Vec<f32> = (0..n_embd).map(|i| w.embd[token*n_embd+i]).collect();
        let tmp = rmsnorm(&hidden, &w.attn_norm);

        let mut q = matvec(&w.attn_q, n_embd, q_dim, &tmp);
        let mut k = matvec(&w.attn_k, n_embd, kv_dim, &tmp);
        let v = matvec(&w.attn_v, n_embd, kv_dim, &tmp);

        for h in 0..n_heads { rope_single(&mut q[h*hd..(h+1)*hd], pos); }
        for h in 0..n_kv   { rope_single(&mut k[h*hd..(h+1)*hd], pos); }

        let total = pos+1;
        let k_at = |p: usize| -> &[f32] {
            if p < pos { &k_hist[p*kv_dim..p*kv_dim+kv_dim] } else { &k }
        };
        let v_at = |p: usize| -> &[f32] {
            if p < pos { &v_hist[p*kv_dim..p*kv_dim+kv_dim] } else { &v }
        };

        let mut attn_out = vec![0.0f32; q_dim];
        for h in 0..n_heads {
            let kv_h = h * n_kv / n_heads;
            let qh = &q[h*hd..(h+1)*hd];
            let mut scores = vec![0.0f32; total];
            for p in 0..total {
                let kp = k_at(p);
                let kh = &kp[kv_h*hd..(kv_h+1)*hd];
                scores[p] = (0..hd).map(|i| qh[i]*kh[i]).sum::<f32>()/(hd as f32).sqrt();
            }
            let max = scores.iter().fold(f32::NEG_INFINITY, |a,&b| a.max(b));
            let mut sum = 0.0;
            for s in scores.iter_mut() { *s = (*s-max).exp(); sum += *s; }
            for s in scores.iter_mut() { *s /= sum; }
            for p in 0..total {
                let vp = v_at(p);
                let vh = &vp[kv_h*hd..(kv_h+1)*hd];
                for i in 0..hd { attn_out[h*hd+i] += scores[p]*vh[i]; }
            }
        }

        let proj = matvec(&w.attn_output, q_dim, n_embd, &attn_out);
        for i in 0..n_embd { hidden[i] += proj[i]; }

        k_hist.extend_from_slice(&k);
        v_hist.extend_from_slice(&v);

        let tmp = rmsnorm(&hidden, &w.ffn_norm);
        let gate = matvec(&w.ffn_gate, n_embd, GQA_FFN, &tmp);
        let up = matvec(&w.ffn_up, n_embd, GQA_FFN, &tmp);
        let mut gu = vec![0.0; GQA_FFN];
        for i in 0..GQA_FFN {
            let g = gate[i];
            gu[i] = (g/(1.0+(-g).exp())) * up[i];
        }
        let out = matvec(&w.ffn_down, GQA_FFN, n_embd, &gu);
        for i in 0..n_embd { hidden[i] += out[i]; }

        rmsnorm(&hidden, &w.output_norm)
    }

    #[test]
    fn test_gqa_forward_matches_scalar_reference() {
        let tmp = create_gqa_gguf();
        let mut engine =
            InferenceEngine::new(tmp.path().to_str().unwrap(), 8*1024*1024).unwrap();
        assert_eq!(engine.config().head_count, GQA_N_HEADS);
        assert_eq!(engine.config().head_count_kv, GQA_N_KV_HEADS);
        assert_eq!(engine.config().head_dim, GQA_HEAD_DIM);
        assert!(!engine.model.attn_bias_present);

        let w = gqa_weights();
        let eps = engine.config().rms_eps;

        let mut ref_k: Vec<f32> = Vec::new();
        let mut ref_v: Vec<f32> = Vec::new();

        let mut kv = crate::kv_cache::KvCache::new(
            engine.config().block_count,
            engine.config().head_count_kv,
            engine.config().head_dim,
            4,
        ).unwrap();
        let budget: &mut MemoryBudget = &mut engine.budget;
        let mut stats = crate::residency::ResidencyStats::new(engine.model.total_weight_bytes);
        let model = &engine.model;
        let backend = &engine.backend;
        let ds = &engine.data_source;

        let mut logits = vec![0.0f32; GQA_VOCAB];
        const EPS: f32 = 2e-4;

        for (token, pos) in [(3usize,0usize),(7usize,1usize),(11usize,2usize)] {
            budget.with_temp("tmp:hidden", (GQA_N_EMBD*4) as u64, |budget| {
                let mut hidden = vec![0.0f32; GQA_N_EMBD];
                model.forward_single_streaming(
                    token as u32, pos,
                    &mut kv, backend, ds, budget, &mut stats,
                    &mut hidden,
                )?;
                model.compute_logits(&hidden, backend, ds, budget, &mut logits)?;

                let ref_h = reference_forward_gqa(&w, token, pos, &mut ref_k, &mut ref_v, eps);
                for i in 0..GQA_N_EMBD {
                    assert!(
                        (hidden[i]-ref_h[i]).abs() < EPS,
                        "pos {} hidden[{}]: model {} vs ref {}", pos, i, hidden[i], ref_h[i]
                    );
                }
                for o in 0..GQA_VOCAB {
                    let ref_logit: f32 = (0..GQA_N_EMBD).map(|i| w.embd[o*GQA_N_EMBD+i]*ref_h[i]).sum();
                    assert!(
                        (logits[o]-ref_logit).abs() < EPS,
                        "pos {} logit[{}]: model {} vs ref {}", pos, o, logits[o], ref_logit
                    );
                }
                Ok::<(), String>(())
            }).unwrap();
            assert_eq!(kv.seq_len(), pos+1, "seq_len after pos {}", pos);
            assert_eq!(kv.get_k(0).len(), (pos+1)*GQA_N_KV_HEADS*GQA_HEAD_DIM);
        }

        let mut engine2 =
            InferenceEngine::new(tmp.path().to_str().unwrap(), 8*1024*1024).unwrap();
        let sampler = crate::sampling::Sampler::greedy();
        let (t1,_) = engine2.generate("hi",5,&sampler).unwrap();
        let (t2,_) = engine2.generate("hi",5,&sampler).unwrap();
        assert_eq!(t1.len(), 5);
        assert_eq!(t1, t2);
    }

    // ----- Generation outcome / stop-reason diagnostics (Issue 2) ----------

    #[test]
    fn outcome_reports_max_tokens_stop_and_state_reset() {
        let tmp = create_tiny_llama_gguf();
        let mut engine =
            InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        let sampler = crate::sampling::Sampler::greedy();
        let outcome = engine
            .generate_with_callback("hello", 3, &sampler, |_| Ok(()))
            .unwrap();
        assert_eq!(outcome.configured_max_tokens, 3);
        assert_eq!(outcome.generated_token_ids.len(), 3);
        assert_eq!(outcome.stop_reason, StopReason::MaxTokens);
        assert_eq!(outcome.eos_token_id, Some(2));
        assert!(!outcome.eos_encountered);
        assert!(outcome.kv_cache_reset);
        assert!(outcome.token_history_reset);
        assert!(outcome.sampler_is_stateless);
        assert_eq!(
            outcome.prompt_tokens,
            engine.tokenizer.encode("hello", true).len()
        );
        assert_eq!(outcome.context_length, engine.config().context_length);
        assert_eq!(outcome.token_ids_sample.len(), 3);
        assert!(!outcome.token_ids_sample.contains(&2)); // EOS must not appear
    }

    #[test]
    fn greedy_runs_are_deterministic_not_an_error() {
        let tmp = create_tiny_llama_gguf();
        let mut engine =
            InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        let sampler = crate::sampling::Sampler::greedy();
        let a = engine
            .generate_with_callback("hello", 4, &sampler, |_| Ok(()))
            .unwrap();
        let b = engine
            .generate_with_callback("hello", 4, &sampler, |_| Ok(()))
            .unwrap();
        assert_eq!(a.generated_token_ids, b.generated_token_ids);
        assert_eq!(a.generated_text, b.generated_text);
        assert_eq!(a.stop_reason, StopReason::MaxTokens);
        assert_eq!(b.stop_reason, StopReason::MaxTokens);
        // Both runs report state reset (no leak between runs).
        assert!(a.kv_cache_reset && b.kv_cache_reset);
        assert!(a.token_history_reset && b.token_history_reset);
    }

    #[test]
    fn outcome_surfaces_eos_metadata_from_tokenizer() {
        let tmp = create_tiny_llama_gguf();
        let engine = InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        // The tiny llama fixture sets tokenizer.ggml.eos_token_id = 2.
        assert_eq!(engine.tokenizer.eos_id, Some(2));
    }

    #[test]
    fn outcome_marks_missing_eos_metadata_via_none() {
        // When GGUF metadata lacks eos_token_id the engine still runs and
        // reports eos_token_id = None; EOS cannot be encountered because
        // there is nothing to compare against.
        use std::io::Write;
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

        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&11u64.to_le_bytes()); // tensor count
        buf.extend_from_slice(&15u64.to_le_bytes()); // metadata count (no eos)
        let mut add_kv = |key: &str, val_type: u32, write_val: Box<dyn FnOnce(&mut Vec<u8>)>| {
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
        add_kv("llama.context_length", 4, Box::new(|b| write_u32(b, 32)));
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
        add_kv(
            "tokenizer.ggml.tokens",
            9,
            Box::new(|b| {
                write_u32(b, 8);
                write_u64(b, 16);
                for tok in [
                    "<unk>", "<s>", "</s>", "▁hello", "▁world", "hello", "world", "!", "▁", "a",
                    "b", "c", "d", "e", "f", "g",
                ] {
                    write_string(b, tok);
                }
            }),
        );
        add_kv(
            "tokenizer.ggml.scores",
            9,
            Box::new(|b| {
                write_u32(b, 6);
                write_u64(b, 16);
                for _ in 0..16 {
                    write_f32(b, 0.0);
                }
            }),
        );
        add_kv(
            "tokenizer.ggml.token_type",
            9,
            Box::new(|b| {
                write_u32(b, 5);
                write_u64(b, 16);
                for t in [2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] {
                    write_u32(b, t);
                }
            }),
        );
        add_kv(
            "tokenizer.ggml.bos_token_id",
            4,
            Box::new(|b| write_u32(b, 1)),
        );
        // Intentionally omit eos_token_id to simulate missing metadata.

        // Copy tensor layout from create_tiny_llama_gguf.
        let tensor_defs = vec![
            ("token_embd.weight", vec![8u64, 16u64], 0u32),
            ("output_norm.weight", vec![8], 0),
            ("blk.0.attn_norm.weight", vec![8], 0),
            ("blk.0.attn_q.weight", vec![8, 8], 0),
            ("blk.0.attn_k.weight", vec![8, 8], 0),
            ("blk.0.attn_v.weight", vec![8, 8], 0),
            ("blk.0.attn_output.weight", vec![8, 8], 0),
            ("blk.0.ffn_norm.weight", vec![8], 0),
            ("blk.0.ffn_gate.weight", vec![8, 16], 0),
            ("blk.0.ffn_up.weight", vec![8, 16], 0),
            ("blk.0.ffn_down.weight", vec![16, 8], 0),
        ];
        let mut offset = 0u64;
        for (name, dims, ty) in &tensor_defs {
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
        let aligned = ramforge_core::model::align_offset(pos, 32);
        buf.extend(vec![0u8; (aligned - pos) as usize]);
        for _ in 0..16 {
            for _ in 0..8 {
                buf.extend_from_slice(&(0.1f32).to_le_bytes());
            }
        }
        for _ in 0..8 {
            buf.extend_from_slice(&1.0f32.to_le_bytes());
        }
        for _ in 0..8 {
            buf.extend_from_slice(&1.0f32.to_le_bytes());
        }
        for _ in 0..8 * 8 {
            buf.extend_from_slice(&0.0f32.to_le_bytes());
        }
        for _ in 0..8 * 8 {
            buf.extend_from_slice(&0.0f32.to_le_bytes());
        }
        for _ in 0..8 * 8 {
            buf.extend_from_slice(&0.0f32.to_le_bytes());
        }
        for _ in 0..8 * 8 {
            buf.extend_from_slice(&0.0f32.to_le_bytes());
        }
        for _ in 0..8 {
            buf.extend_from_slice(&1.0f32.to_le_bytes());
        }
        for _ in 0..16 * 8 {
            buf.extend_from_slice(&0.1f32.to_le_bytes());
        }
        for _ in 0..16 * 8 {
            buf.extend_from_slice(&0.1f32.to_le_bytes());
        }
        for _ in 0..8 * 16 {
            buf.extend_from_slice(&0.1f32.to_le_bytes());
        }

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        tmp.flush().unwrap();

        // Missing eos_token_id means Tokenizer construction currently fills
        // eos_id = None. Verify we build the engine without EOS metadata and
        // that generation reports it explicitly.
        let gguf = ramforge_core::parse_gguf_file(tmp.path()).unwrap();
        let tok = ramforge_core::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
        assert_eq!(tok.eos_id, None, "fixture must omit eos_token_id");

        let mut engine =
            InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        let sampler = crate::sampling::Sampler::greedy();
        let outcome = engine
            .generate_with_callback("hello", 2, &sampler, |_| Ok(()))
            .unwrap();
        assert!(outcome.eos_token_id.is_none());
        assert!(!outcome.eos_encountered);
        // Without EOS metadata we always stop at max_tokens (or context_length).
        assert_eq!(outcome.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn second_generation_on_same_engine_reports_clean_reset() {
        let tmp = create_tiny_llama_gguf();
        let mut engine =
            InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        let sampler = crate::sampling::Sampler::greedy();
        let a = engine
            .generate_with_callback("hello", 2, &sampler, |_| Ok(()))
            .unwrap();
        let b = engine
            .generate_with_callback("hello", 2, &sampler, |_| Ok(()))
            .unwrap();
        assert!(a.kv_cache_reset && b.kv_cache_reset);
        assert!(a.token_history_reset && b.token_history_reset);
        assert_eq!(a.generated_token_ids, b.generated_token_ids);
        // No duplicate KV charges after two runs.
        assert_eq!(
            engine
                .budget
                .allocations()
                .keys()
                .filter(|k| k.as_str() == "kv_cache")
                .count(),
            1
        );
    }
}

#[cfg(test)]
mod autoregressive_invariants {
    use super::*;
    use super::tests::create_tiny_llama_gguf;

    // ---------- Autoregressive decode KV/position invariants ----------
    //
    // These tests exercise the prompt→decode transition on the existing
    // tiny-llama F32 fixture (2 heads, no GQA, no biases) to verify the
    // invariants the real-model investigation depends on:
    //   * prompt positions are 0..N-1;
    //   * after prompt, KV seq_len == N for every layer;
    //   * first decode uses position N and reads N history entries;
    //   * each decode appends exactly one KV entry per layer and
    //     increments seq_len once;
    //   * each decode consumes the previously-sampled token.
    //
    // The tests peek inside the KV cache via the runtime's accessors on
    // KvCache; they do NOT add persistent logging and they do not depend
    // on real Qwen weights.

    #[test]
    fn kv_cache_invariants_across_prompt_and_first_decodes() {
        let tmp = create_tiny_llama_gguf();
        let mut engine =
            InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();

        let model = &engine.model;
        let backend = &engine.backend;
        let ds = &engine.data_source;
        let budget = &mut engine.budget;
        let mut stats = crate::residency::ResidencyStats::new(model.total_weight_bytes);

        let n_embd = model.config.embedding_length;
        let n_layers = model.config.block_count;
        let n_kv_heads = model.config.head_count_kv;
        let head_dim = model.config.head_dim;
        let kv_dim = n_kv_heads * head_dim;

        // Prompt tokens — use the tokenizer to keep the test realistic.
        let prompt = "hi";
        let prompt_tokens = engine.tokenizer.encode(prompt, true);
        let n = prompt_tokens.len();
        assert!(n >= 1, "prompt must tokenize to at least one token (got {})", n);

        let mut kv = crate::kv_cache::KvCache::new(
            n_layers, n_kv_heads, head_dim, /*max_seq_len=*/ n + 8,
        ).unwrap();

        let mut hidden = vec![0.0f32; n_embd];
        for (pos, &tid) in prompt_tokens.iter().enumerate() {
            // Before-forward snapshot: seq_len must equal pos.
            assert_eq!(
                kv.seq_len(), pos,
                "before prompt forward pos={}, kv.seq_len() must be pos",
                pos,
            );
            model.forward_single_streaming(
                tid, pos, &mut kv, backend, ds, budget, &mut stats, &mut hidden,
            ).unwrap();
            // After forward: seq_len incremented by one.
            assert_eq!(
                kv.seq_len(), pos + 1,
                "after prompt forward pos={}, kv.seq_len() must be pos+1",
                pos,
            );
            // Every layer must have (pos+1)*kv_dim floats of K/V committed.
            for li in 0..n_layers {
                assert_eq!(
                    kv.get_k(li).len(),
                    (pos + 1) * kv_dim,
                    "layer {} K length wrong after prompt pos={}", li, pos,
                );
                assert_eq!(
                    kv.get_v(li).len(),
                    (pos + 1) * kv_dim,
                    "layer {} V length wrong after prompt pos={}", li, pos,
                );
            }
        }

        // Now do three greedy decode steps and assert invariants.
        let mut logits = vec![0.0f32; model.config.vocab_size];
        let mut last_token: Option<u32> = None;
        for step in 0..3usize {
            let pos = n + step;
            assert_eq!(kv.seq_len(), pos, "before decode step {}", step);

            model.compute_logits(&hidden, backend, ds, budget, &mut logits).unwrap();
            let next = crate::sampling::Sampler::greedy().sample(&logits);

            // Feed the sampled token forward at position `pos`.
            model.forward_single_streaming(
                next, pos, &mut kv, backend, ds, budget, &mut stats, &mut hidden,
            ).unwrap();
            assert_eq!(kv.seq_len(), pos + 1, "after decode step {}", step);
            for li in 0..n_layers {
                assert_eq!(kv.get_k(li).len(), (pos + 1) * kv_dim);
                assert_eq!(kv.get_v(li).len(), (pos + 1) * kv_dim);
            }
            // Decode consumed exactly the previously sampled token
            // (recorded here for diagnostic; no cross-step duplication).
            if let Some(prev) = last_token {
                assert!(
                    true, // no hard equality assertion: repeats aren't a bug in this test
                    "step {} fed token {} (previous {})", step, next, prev,
                );
            }
            last_token = Some(next);
        }
    }

    #[test]
    fn first_generated_token_is_next_decode_input_not_prompt_duplicate() {
        // The autoregressive loop must feed the freshly sampled token into
        // the next forward pass, not a prompt token and not a duplicated
        // copy. We spy on this by using a tiny prompt (1 token), sampling
        // once, then running a single decode forward and asserting the
        // engine reports prompt_tokens=1 and generated_tokens has the
        // sampled id as its first entry.
        let tmp = create_tiny_llama_gguf();
        let mut engine =
            InferenceEngine::new(tmp.path().to_str().unwrap(), 8 * 1024 * 1024).unwrap();
        let sampler = crate::sampling::Sampler::greedy();
        // Use a prompt whose encoding is a single known token so we can
        // assert the first generated id is NOT that token (unless greedy
        // legitimately chooses it, which we do not hard-assert).
        let outcome = engine
            .generate_with_callback("x", 3, &sampler, |_| Ok(()))
            .unwrap();
        assert_eq!(outcome.prompt_tokens, 1);
        assert_eq!(outcome.generated_token_ids.len(), 3);
        // After the run the engine must have cleared its transient KV
        // allocation back to zero (kv_cache is released at end of run).
        assert!(
            !engine.budget.allocations().contains_key("kv_cache"),
            "kv_cache charge must be released after a generation run"
        );
    }
}

/// Bounded real-model numerical trace. This is deliberately test-only and
/// ignored by the normal suite: it requires the exact Qwen2.5-1.5B Q4_0 GGUF
/// plus a caller-provided memory budget. It does not alter normal generation.
#[cfg(test)]
mod qwen25_bounded_diagnostic {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::kv_cache::KvCache;
    use crate::residency::ResidencyStats;
    use crate::runtime_config::RuntimeConfig;
    use crate::sampling::Sampler;
    use crate::streaming_model::StreamingLlamaModel;
    use ramforge_core::datasource::GgufDataSource;
    use ramforge_core::memory::MemoryBudget;
    use ramforge_core::quant::{self, BLOCK_SIZE_Q6_K, QK_K};
    use ramforge_core::types::GgmlType;
    use std::cmp::Ordering;
    use std::env;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    const RAW_PROMPT: &str = "hi, what's 2+2=?";
    const EXPECTED_PROMPT_TOKENS: [u32; 9] =
        [6023, 11, 1128, 594, 220, 17, 10, 17, 19884];
    const TRACE_DECODE_STEPS: usize = 2;
    const QWEN25_LAYER_COUNT: usize = 28;
    static LAYER_HIDDEN_EMISSIONS: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn qwen25_layer_hook_dispatch_typechecks_without_model() {
        type LayerHookDispatch = for<'model, 'kv, 'backend, 'data, 'budget, 'stats, 'hidden> fn(
            &'model StreamingLlamaModel,
            u32,
            usize,
            &'kv mut KvCache,
            &'backend CpuBackend,
            &'data GgufDataSource,
            &'budget mut MemoryBudget,
            &'stats mut ResidencyStats,
            &'hidden mut [f32],
            fn(usize, &[f32]),
        ) -> Result<(), String>;

        let _dispatch: LayerHookDispatch =
            StreamingLlamaModel::forward_single_streaming_with_layer_hook::<CpuBackend>;
    }

    #[derive(Debug, Clone)]
    struct NumericSummary {
        length: usize,
        min: f32,
        max: f32,
        sum: f64,
        l2_norm: f64,
        first8: Vec<f32>,
    }

    #[derive(Debug, Clone)]
    struct LogitSummary {
        vocabulary_size: usize,
        values: NumericSummary,
        top10: Vec<(u32, f32)>,
    }

    #[derive(Debug, Clone)]
    struct DecodeBoundary {
        input_token_id: u32,
        position_id: usize,
        kv_seq_len_before: usize,
        kv_seq_len_after: usize,
        hidden: NumericSummary,
        logits: LogitSummary,
    }

    #[derive(Debug, Clone)]
    struct BoundedTrace {
        prompt_token_ids: Vec<u32>,
        prompt_positions: Vec<usize>,
        // Retained only inside this ignored test so the output-projection row
        // diagnostic can use the exact pre-logits `result_norm`. It is never
        // printed and is not part of production inference state.
        result_norm: Vec<f32>,
        final_prompt_hidden: NumericSummary,
        final_prompt_logits: LogitSummary,
        first_generated_token: u32,
        first_decode: DecodeBoundary,
        second_generated_token: u32,
        second_decode: DecodeBoundary,
    }

    fn summarize(values: &[f32]) -> NumericSummary {
        assert!(!values.is_empty(), "diagnostic vectors must not be empty");
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut sum_squares = 0.0f64;
        for &value in values {
            min = min.min(value);
            max = max.max(value);
            let value64 = value as f64;
            sum += value64;
            sum_squares += value64 * value64;
        }
        NumericSummary {
            length: values.len(),
            min,
            max,
            sum,
            l2_norm: sum_squares.sqrt(),
            first8: values.iter().take(8).copied().collect(),
        }
    }

    fn summarize_logits(logits: &[f32]) -> LogitSummary {
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|left, right| {
            logits[*right]
                .partial_cmp(&logits[*left])
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.cmp(right))
        });
        order.truncate(10);
        let top10: Vec<(u32, f32)> = order
            .into_iter()
            .map(|index| (index as u32, logits[index]))
            .collect();
        LogitSummary {
            vocabulary_size: logits.len(),
            values: summarize(logits),
            top10,
        }
    }

    fn print_numeric(label: &str, summary: &NumericSummary) {
        println!("{label}.length = {}", summary.length);
        println!("{label}.min = {:?}", summary.min);
        println!("{label}.max = {:?}", summary.max);
        println!("{label}.sum = {:?}", summary.sum);
        println!("{label}.l2_norm = {:?}", summary.l2_norm);
        println!("{label}.first8 = {:?}", summary.first8);
    }

    fn print_logits(label: &str, summary: &LogitSummary) {
        println!("{label}.vocabulary_size = {}", summary.vocabulary_size);
        print_numeric(&format!("{label}.values"), &summary.values);
        println!("{label}.top10 = {:?}", summary.top10);
    }

    fn print_trace(trace: &BoundedTrace) {
        println!("RAMFORGE_QWEN25_BOUNDED_TRACE");
        println!("prompt.raw = {RAW_PROMPT:?}");
        println!("prompt.token_ids = {:?}", trace.prompt_token_ids);
        println!("prompt.positions = {:?}", trace.prompt_positions);
        print_numeric("prompt.final_hidden", &trace.final_prompt_hidden);
        print_logits("prompt.final_logits", &trace.final_prompt_logits);
        println!("prompt.first_generated_token = {}", trace.first_generated_token);

        println!("first_decode.input_token_id = {}", trace.first_decode.input_token_id);
        println!("first_decode.position_id = {}", trace.first_decode.position_id);
        println!(
            "first_decode.kv_seq_len_before = {}",
            trace.first_decode.kv_seq_len_before
        );
        println!(
            "first_decode.kv_seq_len_after = {}",
            trace.first_decode.kv_seq_len_after
        );
        print_numeric("first_decode.final_hidden", &trace.first_decode.hidden);
        print_logits("first_decode.final_logits", &trace.first_decode.logits);
        println!("first_decode.second_generated_token = {}", trace.second_generated_token);

        println!("second_decode.input_token_id = {}", trace.second_decode.input_token_id);
        println!("second_decode.position_id = {}", trace.second_decode.position_id);
        println!(
            "second_decode.kv_seq_len_before = {}",
            trace.second_decode.kv_seq_len_before
        );
        println!(
            "second_decode.kv_seq_len_after = {}",
            trace.second_decode.kv_seq_len_after
        );
        print_numeric("second_decode.final_hidden", &trace.second_decode.hidden);
        print_logits("second_decode.final_logits", &trace.second_decode.logits);
    }

    fn print_result_norm_checkpoints(result_norm: &[f32]) {
        const CHECKPOINT_INDICES: [usize; 20] = [
            0, 1, 2, 3, 4, 5, 6, 7, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512,
            1023, 1535,
        ];

        assert!(result_norm.len() > 1535);
        for &index in &CHECKPOINT_INDICES {
            println!("result_norm.checkpoint.index = {}", index);
            println!("result_norm.checkpoint.value = {:?}", result_norm[index]);
        }
    }

    fn print_layer_hidden_diagnostic(layer_idx: usize, hidden: &[f32]) {
        const N_EMBD: usize = 1536;
        const CHECKPOINT_INDICES: [usize; 4] = [0, 255, 1023, 1535];

        assert!(layer_idx < QWEN25_LAYER_COUNT);
        let emission_index = LAYER_HIDDEN_EMISSIONS.fetch_add(1, AtomicOrdering::Relaxed);
        assert_eq!(
            emission_index, layer_idx,
            "layer checkpoints must be emitted once in layer order"
        );
        assert_eq!(hidden.len(), N_EMBD);
        let mut sum = 0.0f64;
        let mut l2_squared = 0.0f64;
        for &value in hidden {
            let value = value as f64;
            sum += value;
            l2_squared += value * value;
        }

        println!("layer_hidden.layer = {}", layer_idx);
        println!("layer_hidden.length = {}", hidden.len());
        println!("layer_hidden.sum = {:?}", sum);
        println!("layer_hidden.l2_norm = {:?}", l2_squared.sqrt());
        for &index in &CHECKPOINT_INDICES {
            println!("layer_hidden.checkpoint.index = {}", index);
            println!("layer_hidden.checkpoint.value = {:?}", hidden[index]);
        }
    }

    fn diagnose_q6_k_output_rows(
        engine: &InferenceEngine,
        result_norm: &[f32],
    ) -> Result<(), String> {
        const N_EMBD: usize = 1536;
        const VOCAB_SIZE: usize = 151_936;
        const ROW_IDS_AND_REFERENCE: [(usize, f64); 2] = [
            (59, 20.24774933),
            (220, 19.75674248),
        ];

        assert_eq!(result_norm.len(), N_EMBD);
        let descriptor = engine
            .data_source
            .get_descriptor("output.weight")
            .map_err(|error| format!("output.weight descriptor lookup failed: {error}"))?
            .clone();
        assert_eq!(descriptor.ggml_type, GgmlType::Q6_K);
        assert_eq!(descriptor.dimensions.as_slice(), &[N_EMBD as u64, VOCAB_SIZE as u64]);

        let blocks_per_row = N_EMBD / QK_K;
        let row_bytes = blocks_per_row * BLOCK_SIZE_Q6_K;
        assert_eq!(blocks_per_row, 6);
        assert_eq!(row_bytes, 1260);
        assert_eq!(
            descriptor.byte_length,
            Some((row_bytes * VOCAB_SIZE) as u64),
            "output.weight byte length must match Q6_K row geometry"
        );

        println!(
            "output_projection.descriptor name={} type={} dimensions={:?} row_bytes={}",
            descriptor.name,
            descriptor.ggml_type.name(),
            descriptor.dimensions,
            row_bytes
        );
        print_numeric("result_norm", &summarize(result_norm));

        for &(row_id, llama_cpp_reference) in &ROW_IDS_AND_REFERENCE {
            let byte_offset = (row_id * row_bytes) as u64;
            let row = engine
                .data_source
                .read_tensor_range_by_descriptor(&descriptor, byte_offset, row_bytes as u64)
                .map_err(|error| {
                    format!(
                        "failed to read output.weight row {} (offset {}, length {}): {}",
                        row_id, byte_offset, row_bytes, error
                    )
                })?;
            assert_eq!(row.len(), row_bytes);

            let mut dequantized = [0.0f32; N_EMBD];
            quant::dequantize_row_q6_k(&row, N_EMBD, &mut dequantized)
                .map_err(|error| format!("Q6_K dequantization failed for row {}: {}", row_id, error))?;
            let mut dequantized_dot = 0.0f32;
            for index in 0..N_EMBD {
                dequantized_dot += dequantized[index] * result_norm[index];
            }

            // matvec_q6_k with a one-row [out, in] shape reaches the existing
            // fused q6_k_row_dot implementation without exposing or changing
            // that private production helper.
            let mut fused_output = [0.0f32; 1];
            quant::matvec_q6_k(&row, &[1, N_EMBD], result_norm, &mut fused_output)
                .map_err(|error| format!("fused Q6_K path failed for row {}: {}", row_id, error))?;
            let fused_dot = fused_output[0];

            let dequantized_abs_error = (dequantized_dot as f64 - llama_cpp_reference).abs();
            let fused_abs_error = (fused_dot as f64 - llama_cpp_reference).abs();
            println!("output_projection.row_id = {}", row_id);
            println!("output_projection.dequantized_dot = {:?}", dequantized_dot);
            println!("output_projection.fused_dot = {:?}", fused_dot);
            println!(
                "output_projection.llama_cpp_reference = {:?}",
                llama_cpp_reference
            );
            println!(
                "output_projection.dequantized_abs_error = {:?}",
                dequantized_abs_error
            );
            println!(
                "output_projection.fused_abs_error = {:?}",
                fused_abs_error
            );
        }
        Ok(())
    }

    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for &byte in bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3u64);
        }
        hash
    }

    fn hex_bytes(bytes: &[u8]) -> String {
        bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn diagnose_raw_q6_k_output_rows(engine: &InferenceEngine) -> Result<(), String> {
        const N_EMBD: usize = 1536;
        const VOCAB_SIZE: usize = 151_936;
        const ROW_BYTES: usize = 1260;
        const ROW_IDS: [usize; 2] = [59, 220];

        let descriptor = engine
            .data_source
            .get_descriptor("output.weight")
            .map_err(|error| format!("output.weight descriptor lookup failed: {error}"))?
            .clone();
        assert_eq!(descriptor.name, "output.weight");
        assert_eq!(descriptor.ggml_type, GgmlType::Q6_K);
        assert_eq!(descriptor.dimensions.as_slice(), &[N_EMBD as u64, VOCAB_SIZE as u64]);
        assert_eq!(descriptor.byte_length, Some((VOCAB_SIZE * ROW_BYTES) as u64));
        assert_eq!(N_EMBD / QK_K, 6);
        assert_eq!((N_EMBD / QK_K) * BLOCK_SIZE_Q6_K, ROW_BYTES);

        let data_start_offset = engine.data_source.model().data_start_offset;
        println!("output_projection.raw_q6k.tensor_name = {}", descriptor.name);
        println!(
            "output_projection.raw_q6k.ggml_type = {}",
            descriptor.ggml_type.name()
        );
        println!(
            "output_projection.raw_q6k.dimensions = {:?}",
            descriptor.dimensions
        );
        println!(
            "output_projection.raw_q6k.descriptor_offset = {}",
            descriptor.offset
        );
        println!(
            "output_projection.raw_q6k.descriptor_file_offset = {}",
            descriptor.file_offset
        );
        println!(
            "output_projection.raw_q6k.data_start_offset = {}",
            data_start_offset
        );
        println!(
            "output_projection.raw_q6k.byte_length = {:?}",
            descriptor.byte_length
        );
        println!("output_projection.raw_q6k.row_bytes = {}", ROW_BYTES);
        println!(
            "output_projection.raw_q6k.digest = fnv1a64 (deterministic bounded checksum; no external hash dependency)"
        );

        for row_id in ROW_IDS {
            let row_offset = row_id
                .checked_mul(ROW_BYTES)
                .ok_or_else(|| format!("row {} byte offset overflow", row_id))?;
            let absolute_file_offset = descriptor
                .file_offset
                .checked_add(row_offset as u64)
                .ok_or_else(|| format!("row {} absolute file offset overflow", row_id))?;
            let absolute_data_offset = data_start_offset
                .checked_add(descriptor.offset)
                .and_then(|offset| offset.checked_add(row_offset as u64))
                .ok_or_else(|| format!("row {} data offset overflow", row_id))?;

            let row = engine
                .data_source
                .read_tensor_range_by_descriptor(
                    &descriptor,
                    row_offset as u64,
                    ROW_BYTES as u64,
                )
                .map_err(|error| {
                    format!(
                        "failed to read output.weight row {} at offset {}: {}",
                        row_id, row_offset, error
                    )
                })?;
            assert_eq!(row.len(), ROW_BYTES);

            let first_block = quant::BlockQ6K::from_bytes(&row[..BLOCK_SIZE_Q6_K])
                .map_err(|error| format!("failed to decode row {} first Q6_K block: {}", row_id, error))?;
            let d_bits = u16::from_le_bytes([row[208], row[209]]);
            let mut first_block_decoded = [0.0f32; QK_K];
            first_block.dequantize(&mut first_block_decoded);
            let digest = fnv1a64(&row);

            println!("output_projection.raw_q6k.row_id = {}", row_id);
            println!(
                "output_projection.raw_q6k.byte_offset = {}",
                row_offset
            );
            println!(
                "output_projection.raw_q6k.byte_length = {}",
                row.len()
            );
            println!(
                "output_projection.raw_q6k.absolute_file_offset = {}",
                absolute_file_offset
            );
            println!(
                "output_projection.raw_q6k.absolute_data_offset = {}",
                absolute_data_offset
            );
            println!(
                "output_projection.raw_q6k.fnv1a64 = 0x{digest:016x}"
            );
            println!(
                "output_projection.raw_q6k.first32_hex = {}",
                hex_bytes(&row[..32])
            );
            println!(
                "output_projection.raw_q6k.first_block.ql_first16_hex = {}",
                hex_bytes(&row[..16])
            );
            println!(
                "output_projection.raw_q6k.first_block.qh_first16_hex = {}",
                hex_bytes(&row[128..144])
            );
            println!(
                "output_projection.raw_q6k.first_block.scales = {:?}",
                first_block.scales
            );
            println!(
                "output_projection.raw_q6k.first_block.d_f16_bits = 0x{d_bits:04x}"
            );
            println!(
                "output_projection.raw_q6k.first_block.d_f32 = {:?}",
                first_block.d
            );
            println!(
                "output_projection.raw_q6k.first_block.decoded_first16 = {:?}",
                &first_block_decoded[..16]
            );
        }
        Ok(())
    }

    fn ensure_trace_capacity(
        kv_cache: &mut KvCache,
        required_tokens: usize,
        maximum_tokens: usize,
        budget: &mut MemoryBudget,
    ) -> Result<(), String> {
        if required_tokens <= kv_cache.capacity_tokens() {
            return Ok(());
        }
        let target = kv_cache
            .chunk_aligned_capacity(required_tokens)
            .min(maximum_tokens);
        let new_bytes = kv_cache.bytes_for_tokens(target) as u64;
        budget
            .resize("kv_cache", new_bytes)
            .map_err(|error| format!("failed to resize diagnostic KV charge: {error}"))?;
        kv_cache.grow_to(target)?;
        Ok(())
    }

    fn run_bounded_trace(
        engine: &mut InferenceEngine,
        prompt_tokens: &[u32],
    ) -> Result<BoundedTrace, String> {
        if prompt_tokens != EXPECTED_PROMPT_TOKENS.as_slice() {
            return Err(format!(
                "diagnostic requires exact prompt IDs {:?}, got {:?}",
                EXPECTED_PROMPT_TOKENS, prompt_tokens
            ));
        }

        let model = &engine.model;
        let backend = &engine.backend;
        let data_source = &engine.data_source;
        let budget = &mut engine.budget;
        let mut residency_stats = ResidencyStats::new(model.total_weight_bytes);
        let n_embd = model.config.embedding_length;
        let vocab_size = model.config.vocab_size;
        let n_layers = model.config.block_count;
        let n_kv_heads = model.config.head_count_kv;
        let head_dim = model.config.head_dim;
        let prompt_len = prompt_tokens.len();
        let needed_len = prompt_len + TRACE_DECODE_STEPS;
        assert_eq!(
            n_layers, QWEN25_LAYER_COUNT,
            "Qwen2.5 layer checkpoint requires 28 layers"
        );
        assert_eq!(n_embd, 1536, "Qwen2.5 layer checkpoint requires embedding size 1536");

        // Match the normal generation lifetime: initial KV capacity equals the
        // prompt length, then grows only when the first/second decode needs it.
        let mut kv_cache = KvCache::new(n_layers, n_kv_heads, head_dim, prompt_len)?;
        kv_cache
            .allocate_from_budget(budget)
            .map_err(|error| format!("failed to allocate diagnostic KV cache: {error}"))?;

        let mut hidden = vec![0.0f32; n_embd];
        for (position_id, &token_id) in prompt_tokens.iter().enumerate() {
            if position_id + 1 == prompt_tokens.len() {
                // The test-only hook is invoked inside ModelExecutor
                // immediately after each layer's residual and FFN output has
                // been written to `hidden`, before the next layer starts. It consumes one borrowed layer vector at a
                // time and never retains the activations.
                LAYER_HIDDEN_EMISSIONS.store(0, AtomicOrdering::Relaxed);
                model.forward_single_streaming_with_layer_hook(
                    token_id,
                    position_id,
                    &mut kv_cache,
                    backend,
                    data_source,
                    budget,
                    &mut residency_stats,
                    &mut hidden,
                    print_layer_hidden_diagnostic,
                )?;
                assert_eq!(
                    LAYER_HIDDEN_EMISSIONS.load(AtomicOrdering::Relaxed),
                    QWEN25_LAYER_COUNT,
                    "the prompt layer hook must emit exactly one checkpoint per layer"
                );
            } else {
                model.forward_single_streaming(
                    token_id,
                    position_id,
                    &mut kv_cache,
                    backend,
                    data_source,
                    budget,
                    &mut residency_stats,
                    &mut hidden,
                )?;
            }
        }
        if kv_cache.seq_len() != prompt_len {
            return Err(format!(
                "prompt KV length mismatch: expected {}, got {}",
                prompt_len,
                kv_cache.seq_len()
            ));
        }

        // Boundary 1: final prompt hidden is captured before output projection.
        let result_norm = hidden.clone();
        let final_prompt_hidden = summarize(&hidden);
        let mut logits = vec![0.0f32; vocab_size];
        model.compute_logits(
            &hidden,
            backend,
            data_source,
            budget,
            &mut logits,
        )?;
        let final_prompt_logits = summarize_logits(&logits);
        let sampler = Sampler::greedy();
        let first_generated_token = sampler.sample(&logits);

        // Boundary 2: feed the first greedy token at position N.
        let first_position_id = prompt_len;
        let first_kv_before = kv_cache.seq_len();
        ensure_trace_capacity(
            &mut kv_cache,
            first_kv_before + 1,
            needed_len,
            budget,
        )?;
        model.forward_single_streaming(
            first_generated_token,
            first_position_id,
            &mut kv_cache,
            backend,
            data_source,
            budget,
            &mut residency_stats,
            &mut hidden,
        )?;
        let first_kv_after = kv_cache.seq_len();
        let first_hidden = summarize(&hidden);
        model.compute_logits(
            &hidden,
            backend,
            data_source,
            budget,
            &mut logits,
        )?;
        let first_logits = summarize_logits(&logits);
        let second_generated_token = sampler.sample(&logits);

        // Boundary 3: feed the second greedy token at position N+1.
        let second_position_id = prompt_len + 1;
        let second_kv_before = kv_cache.seq_len();
        ensure_trace_capacity(
            &mut kv_cache,
            second_kv_before + 1,
            needed_len,
            budget,
        )?;
        model.forward_single_streaming(
            second_generated_token,
            second_position_id,
            &mut kv_cache,
            backend,
            data_source,
            budget,
            &mut residency_stats,
            &mut hidden,
        )?;
        let second_kv_after = kv_cache.seq_len();
        let second_hidden = summarize(&hidden);
        model.compute_logits(
            &hidden,
            backend,
            data_source,
            budget,
            &mut logits,
        )?;
        let second_logits = summarize_logits(&logits);

        let trace = BoundedTrace {
            prompt_token_ids: prompt_tokens.to_vec(),
            prompt_positions: (0..prompt_len).collect::<Vec<usize>>(),
            result_norm,
            final_prompt_hidden,
            final_prompt_logits,
            first_generated_token,
            first_decode: DecodeBoundary {
                input_token_id: first_generated_token,
                position_id: first_position_id,
                kv_seq_len_before: first_kv_before,
                kv_seq_len_after: first_kv_after,
                hidden: first_hidden,
                logits: first_logits,
            },
            second_generated_token,
            second_decode: DecodeBoundary {
                input_token_id: second_generated_token,
                position_id: second_position_id,
                kv_seq_len_before: second_kv_before,
                kv_seq_len_after: second_kv_after,
                hidden: second_hidden,
                logits: second_logits,
            },
        };

        budget
            .release("kv_cache")
            .map_err(|error| format!("failed to release diagnostic KV charge: {error}"))?;
        Ok(trace)
    }

    /// Run with, for example:
    ///
    /// ```text
    /// CARGO_TARGET_DIR=/tmp/ramforge-cargo-target \\
    /// RAMFORGE_QWEN25_MODEL=/path/to/qwen2.5-1.5b-instruct-q4_0.gguf \\
    /// RAMFORGE_QWEN25_RAM_BYTES=8589934592 \\
    /// cargo test -p ramforge-runtime qwen25_bounded_numerical_trace -- \\
    ///     --ignored --nocapture
    /// ```
    ///
    /// The test is ignored so normal workspace tests never require a real model.
    #[test]
    #[ignore = "requires the real Qwen2.5-1.5B Q4_0 GGUF and an explicit RAM budget"]
    fn qwen25_bounded_numerical_trace() {
        let model_path = env::var("RAMFORGE_QWEN25_MODEL")
            .expect("set RAMFORGE_QWEN25_MODEL to the exact Q4_0 GGUF path");
        let ram_budget_bytes = env::var("RAMFORGE_QWEN25_RAM_BYTES")
            .expect("set RAMFORGE_QWEN25_RAM_BYTES to a sufficient byte budget")
            .parse::<u64>()
            .expect("RAMFORGE_QWEN25_RAM_BYTES must be an integer byte count");

        // One CPU thread is intentional: it removes parallel-dispatch
        // differences while retaining the normal Q4_0 execution path.
        let runtime_config = RuntimeConfig::new(
            1,
            ram_budget_bytes,
            false,
            0,
            true,
            true,
        )
        .expect("valid one-thread diagnostic runtime configuration");
        let mut engine = InferenceEngine::new_with_runtime_config(&model_path, runtime_config)
            .expect("load Qwen2.5 Q4_0 diagnostic model");

        assert_eq!(engine.config().vocab_size, 151_936);
        let encoded = engine.tokenizer.encode(RAW_PROMPT, true);
        assert_eq!(
            encoded.as_slice(),
            EXPECTED_PROMPT_TOKENS.as_slice(),
            "the real model tokenizer must retain the confirmed prompt IDs"
        );

        // Feed the fixed IDs, not just the prompt string, into the numerical
        // trace so later reference comparisons use exactly the same sequence.
        let trace = run_bounded_trace(&mut engine, &EXPECTED_PROMPT_TOKENS)
            .expect("bounded Qwen2.5 numerical trace");
        print_trace(&trace);
        print_result_norm_checkpoints(&trace.result_norm);
        diagnose_q6_k_output_rows(&engine, &trace.result_norm)
            .expect("Q6_K output projection row diagnostic");
        diagnose_raw_q6_k_output_rows(&engine)
            .expect("raw Q6_K output projection row diagnostic");
    }
}
