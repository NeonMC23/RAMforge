# RAMforge — Milestone 6 Final Report: “True Out-of-Care Integrity”

Date: 2026-08-23. Scope: M6 priorities (memory accounting, cache charging, explicit GGML layout, budget-aware output projection, KV/attention temps, backend wiring) + full gates.

## 1. Initial workspace state (verified this session)
- Baseline: Milestone 5.6.1 codebase, 93 tests (62 core / 31 runtime), all passing; `llama`/`qwen2` CPU inference with out-of-core layer streaming and compact quantized residency already present.
- Rust toolchain installed *outside* the project at `/home/user/.cargo/bin` (`--no-modify-path`); every cargo invocation used `export PATH="/home/user/.cargo/bin:$PATH"`.

## 2. `.git` removed confirmation
- `.git/` was deleted by order and never recreated. No git command was run at any point (no init/clone/fetch/push/commit), GitHub was never consulted, no commit hashes referenced. Filesystem state is the only source of truth; the user manages Git externally.

## 3. `.cargo` absent confirmation
- No `.cargo/` exists inside the project (verified: `ls .cargo` → No such file or directory). Also confirmed absent: `.profile`, `models/`, generated GGUF fixtures on disk, editor caches. Only `target/` from normal builds (already in `.gitignore`) and test in-memory `tempfile` fixtures (existing pattern).

## 4. Fresh technical audit (performed on current source before edits)
- **Budget-tracked:** resident persistents (`weight:*`), per-layer tensors (`layer:{i}:*`), KV cache, embedding row temp.
- **Budget-bypassed (violations):** cache contents (up to 0.8×budget untracked); logits double-allocation (`compute_logits` returned a Vec then copied into another); sampler clones/sorts (up to 2 vocab tables + HashSet); per-token `k_full`/`v_full` KV prefix copies (~2×KV per layer per token); per-layer activation Vecs; streamed-embedding read; streamed `PersistentWeight::matvec` full read+dequant; transpose-branch full `dequantize_to_f32` (up to ~512 MiB on 7B tied embeddings) whose indexing also mismatched the ggml layout (wrong math on real files).
- **Orientation heuristic:** `QuantizedTensor::matvec`/`dequantize_row`/`TensorData::matvec`/`backend.matvec` compared buffer dims to x/y lengths to guess orientation — because old fixtures declared ffn dims in the wrong convention.
- **Dead/unused power:** existing `simd.rs` (AVX2+FMA, runtime-detected) and rayon `backend.rs` were unreachable from `ramforge run` (`forward_single_streaming` never called `backend.matvec`); legacy resident-F32 `LlamaModel` unused by the engine.

## 5. Confirmed M6 problems (as fixed)
1. Temporary allocations bypassed the budget → fixed via RAII guard + hoisted workspace.
2. Cache contents unaccounted / potential double buffering → budget-charged per entry.
3. Orientation guessing + hidden full-F32 fallbacks (correctness *and* memory) → strict ggml layout, hard errors.
4. Output projection double-allocated vocab logits; streamed path read/dequantized the whole matrix → single caller buffer + budget-bounded row chunks.
5. KV prefix copies per token per layer; capacity preallocated at prompt+max_tokens → zero-copy attention + chunk-wise growth capped at actual need.
6. SIMD/rayon present but unused in the hot path → wired for resident-F32 matvec.

## 6. Files modified (19 source files + README + this report)
- core: `error.rs`, `memory.rs`, `cache.rs`, `tensor.rs`, `quant.rs` (test lint only), `gguf.rs` (test lint only)
- runtime: `ops.rs`, `kv_cache.rs`, `persistent.rs`, `streaming_model.rs`, `inference.rs`, `model.rs`, `backend.rs`, `sampling.rs`, `lib.rs`, `plan.rs`, `simd.rs` (test lint only)
- cli: `main.rs`; docs: `README.md`

## 7. Memory accounting changes
- New `MemoryBudget::with_temp(name, bytes, f)`: reserves then runs the closure (receiving `&mut MemoryBudget` for nested charges), releases on **both** Ok and Err; zero-byte calls are no-ops. `From<MemoryError> for String` added for error composition.
- `forward_single_streaming` runs the whole token pass inside one `tmp:forward` reservation sized to the worst-case activation floats (`8·n_embd + 2·q_dim + 2·kv_dim + 4·ffn + n_heads·seq`); all per-layer buffers are hoisted and allocated once per forward call.
- `tmp:logits` (vocab+n_embd floats) wraps `generate()` for its whole duration; `tmp:sampling` (0 for greedy, else 5×vocab floats) guards sampler scratch; streamed embedding/streamed matvec charge `tmp:embd_row`/`tmp:streamed_matvec` inside `persistent.rs`.
- Layer loading now charges *before* reading (file bytes for quant, 2× file bytes for float to cover the raw+f32 decode spike), settles each charge to the exact `resident_bytes()` after construction, and releases all layer charges on any failure.

## 8. Cache changes
- New `insert_budgeted`/`remove_budgeted`/`clear_budgeted`: every cached entry is charged `cache:{key}`; LRU evictions release charges; if the budget has no room even after full eviction, the insert returns `Ok(false)` (entry simply not cached) instead of failing; `TooLarge` still rejects over-capacity entries.
- `InferenceEngine` no longer owns a `BoundedCache` (layer streaming reads directly; no large-buffer duplication). `Runtime::get_tensor` now inserts via `insert_budgeted`. Removed the fake `tensor_cache` + `runtime_overhead` pre-reservations (double counting) from `Runtime::new` and `plan_model` (capacity stays an informational bound).

## 9. Matrix layout changes
- Single explicit GGML/GGUF convention, documented in `tensor.rs`: `shape = [in, out]` (ne[0]=in contiguous, ne[1]=out), buffer row-major `[out][in]`, `y[o] = Σ_i W[o·in+i]·x[i]`.
- All orientation heuristics and the transpose/full-dequantize fallbacks deleted from `QuantizedTensor::matvec`, `dequantize_row`, `TensorData::matvec` (float variants now use `matvec_f32_ggml`) and `ComputeBackend::matvec` (now returns `Result`, strict arity errors).
- Real shapes verified: `token_embd=[n_embd,vocab]`, `ffn_gate/up=[n_embd,n_ff]`, `ffn_down=[n_ff,n_embd]`, attn square. Test fixtures updated to the correct convention (correction, not weakening).
- Legacy resident-F32 `LlamaModel` (model.rs, ~340 lines) deleted: it embodied every banned pattern and was unused; `LlamaConfig` + new free fn `validate_required_tensors` retained.

## 10. Quantized matvec changes
- Compact-residency pipeline enforced end to end: GGUF bytes → `QuantizedTensor` (raw) → block-wise quantized matvec (bounded per-block temp) → output. No full-F32 expansion of 2D weights anywhere.
- New allocation-free `decode_row_to_f32(ggml_type, bytes, n, out)` covering all 8 supported quants + F32/F16/BF16 (used by streamed reads); it hard-fails (explicit error) on truncated input instead of guessing.
- Layout anchors added: non-square Q4_0 `[64,3]` matched against hand-computed reference (`y=[-672,-528,336]`), Q4_K `[256,2]` anchors `[256,512]`, F32/F16 non-square, plus arity-error tests. All 8 quant formats retain block-size/decode/matvec kernel tests.

## 11. Output projection changes
- `StreamingLlamaModel::compute_logits(hidden, backend, data_source, budget, logits_out)` writes into the single engine-owned logits buffer (allocated once per `generate()`, charged via `tmp:logits`). Tied-embedding fallback via `token_embd`.
- Resident F32 → backend (SIMD/rayon); resident quantized → compact block-wise matvec; streamed → `compute_logits_into` → `streamed_matvec_into`: row chunks sized `min(16 MiB, available/4)`, ≥1 whole row, per-row `decode_row_to_f32` into one reused buffer; clear error if a single row + buffer can't fit the budget.

## 12. KV + attention changes
- `ops::attention(q, k_hist, v_hist, k_new, v_new, hist_len, …)` reads the cache prefix in place via slicing accessors — the `k_full`/`v_full` concatenations (~2×KV per token per layer) are gone; tested against a naive concatenated reference.
- `KvCache` gains `GROW_CHUNK_TOKENS=256`, `capacity_tokens()`, `bytes_for_tokens()`, `chunk_aligned_capacity()`, `grow_to()` (exact target, data-preserving, no shrink). Engine starts at prompt length and grows chunk-wise **capped at prompt+max_tokens** with deterministic rollback (release old charge → try new → restore old + clear error if it fails) before each growth.
- KV stays F32, no KV quantization (as scoped).

## 13. SIMD + threading changes
- `matvec_backend` helper wires `ComputeBackend` into the inference path: any resident **F32** weight (`as_f32_slice`) runs `backend.matvec` (AVX2+FMA `dot_f32_avx2`/`matvec_f32_avx2`, rayon row-parallel when threads>1 and out≥4, scalar fallback); quantized/others run the compact kernels. No full-F32 dequantization was introduced for this.
- `backend.matvec` is strict-ggml and infallible-shape-checked; ambiguity is an error, never a guess. CLI prints the active backend mode (`CPU-SIMD`/`CPU-scalar`). Known pre-existing quirk retained: rayon global pool ignores the `num_threads` count.

## 14. Tests added (24 new; 93 → 117)
- core `memory.rs` +4: with_temp success/error/zero-byte/nested-allocation integrity.
- core `cache.rs` +4: budgeted insert charge, eviction releases charges, skip-when-budget-full keeps serving, remove/clear release charges.
- core `tensor.rs` +9: non-square Q4_0/Q4_K/F32/F16 anchors, explicit-layout dequantize_row, quantized get_embedding rows, arity-error rejections.
- runtime `ops.rs` +1: no-copy attention vs naive reference.
- runtime `kv_cache.rs` +2: chunk growth preserves data/allows more appends; exact bytes_for_tokens.
- runtime `persistent.rs` +2: chunked streamed output projection (resident/streamed parity, no charge leak, forced multi-chunk) + budget-too-small clear error.
- runtime `backend.rs` +2: non-square ggml matvec + orientation-flip arity error. (existing SIMD/thread parity tests adapted)
- Fixture corrections: ffn dims flipped to ggml convention in all GGUF fixtures; out-of-core budgets rescaled (`streaming_model` 96 KiB / `inference` 32 KiB with 8 layers) for the stricter charge-before-read accounting; `test_budget_too_small_failure` now proves too-small engine fails clearly at the first budget-checked allocation without corrupting the budget.

## 15. `cargo test --workspace` result
- **117 passed, 0 failed** — ramforge-core 79, ramforge-runtime 38, CLI 0 (no tests), 3 empty doc/other suites. All pre-existing tests preserved or corrected (fixture dims/budgets), none weakened.

## 16. `cargo clippy --workspace -- -D warnings` result
- **Clean** (no warnings). Also clean under `--all-targets` (test code fixed alongside: type-alias for fixture closures, useless-vec, approx-constant, unused variable).

## 17. `cargo build --workspace` result
- **Clean** (dev profile). CLI smoke-tested: `ramforge --help` prints the Milestone 6 banner and `inspect|plan|run`.

## 18. Remaining intentionally unmanaged memory (documented, out of budget scope by design)
- Tokenizer vocabulary/merges tables (loaded once at engine start, O(vocab)).
- OS thread stacks, rayon pool internals, allocator fragmentation.
- `ResidencyStats` counters and CLI/strings (`Vec<u8>` GGUF header/metadata kept by `GgufDataSource` — part of the data source object, not transient).
- Process RSS / OS page cache (RAMforge-managed memory is explicitly defined as budget-tracked only).

## 19. Known limitations (M6)
- CPU-only; quantized matvec scalar/block-wise (AVX2/rayon on F32 path only); quantized formats limited to Q4_0/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K (Q4_1/Q5_*/Q8_1/IQ* → clear errors).
- Minimum budget = persistents (or their streamed row temps) + one layer in charge-before-read transient (≤2× file bytes for float tensors) + `tmp:forward` workspace + growing KV + logits; a single streamed output row (raw + F32) must fit.
- KV cache F32, no quantization/eviction; growth granularity 256 tokens, capped at prompt+max_tokens.
- Unsupported layouts fail loudly rather than guessing (by design).
- Not implemented (unchanged scope): GPU, HTTP server, model downloading, MoE, speculative decoding, new quantization formats.
- rayon `num_threads` count not enforced on the global pool (pre-existing).

## 20. Final workspace tree + status
```
RAMforge/
  .gitignore (target/ ignored)   Cargo.toml   Cargo.lock   LICENSE.md   README.md   MILESTONE6_REPORT.md
  crates/
    ramforge-cli/       Cargo.toml, src/main.rs                                  (1 .rs)
    ramforge-core/      Cargo.toml, src/{cache,datasource,error,gguf,lib,memory,
                        model,quant,tensor,tokenizer,types}.rs                   (11 .rs)
    ramforge-runtime/   Cargo.toml, src/{backend,inference,kv_cache,layer,lib,
                        model,ops,persistent,plan,residency,sampling,simd,
                        streaming_model}.rs                                       (13 .rs)
  target/  (build artifact, allowed)
```
Status: **no `.git/`, no `.cargo/`, no `.profile/`, no model files, no on-disk fixtures**; 25 `.rs` files; gates green: `cargo test --workspace` 117/117, `cargo clippy --workspace -- -D warnings` clean, `cargo build --workspace` clean. Ready for external Git management.
