# RAMforge — Post-Milestone-6 Audit Report

Date: 2026-08-23. Scope: audit only — **no source code was modified** during this audit.
Sources of truth: the current filesystem and a fresh verification run. The M6 report was used as context only; every number below was re-derived.

---

## A. Verification executed during this audit (all on current workspace)

| Gate | Result |
|---|---|
| `cargo test --workspace` | ✅ **117 passed / 0 failed** (core 79, runtime 38, CLI 0, 3 empty suites) |
| `cargo clippy --workspace -- -D warnings` | ✅ clean |
| `cargo clippy --workspace --all-targets -- -D warnings` | ✅ clean |
| `cargo build --workspace` | ✅ clean |
| `ramforge --help` | ✅ M6 banner, `inspect`/`plan`/`run` |
| End-to-end CLI probe (synthetic 1-layer GGUF in /tmp, deleted after) | ✅ `run` emits 5 deterministic tokens incl. `--verbose` residency stats and post-run budget charge listing; `plan` prints M6 accounting text |
| Regression probe (scratch harness linked against built rlib, /tmp, deleted after) | ❌ confirmed BUG #1 below |
| Workspace cleanliness scan | ✅ no `.git/`, no `.cargo/`, no `.profile`, no `.github`, no model/fixture files; only `target/` (ignored) and the M6 report/README |

Verdict on the M6 report's own claims: **test count, clippy, build, and CLI claims re-verified exactly (117, both clippy modes clean)**. Two claims need qualification (§20).

## B. The 20 audit points

1. **Project structure** — VERIFIED. 3 crates (`ramforge-core` 11 files, `ramforge-runtime` 13, `ramforge-cli` 1), 25 `.rs` files, workspaces deps only (thiserror/serde/serde_json/clap + cargo deps rand/half/rayon/tempfile-dev), edition 2021, resolver 2.
2. **Source implementation** — VERIFIED by reading: streaming (`streaming_model.rs`), engine (`inference.rs`), backend/simd/ops/kv/persistent, core gguf/datasource/memory/cache/tensor/quant/tokenizer. Removed M6 code (legacy resident `LlamaModel`, orientation fallbacks) is really gone; no `k_full`/`v_full`/`matvec_infer`/transpose references remain.
3. **Supported tensor formats** — VERIFIED. Inference-accepted: F32, F16, BF16, Q4_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K; everything else (Q4_1/Q5_*/Q8_1/IQ*) → explicit `unsupported tensor type for inference` error (`tensor.rs:410`, `decode_row_to_f32:709`). GGUF *parsing* knows more types than inference accepts — correct fail-loud behavior on load.
  ⚠️ Caveat: F16/BF16 are **decoded to `Vec<f32>` at load time** (resident RAM = 4 B/elem, not 2) — see BUG #3.
4. **Inference execution path** — VERIFIED (code + CLI probe): `CLI run → InferenceEngine::new (datasource, tokenizer, budget, StreamingLlamaModel::load persistents) → generate { tokenize → KvCache(prompt_len) → tmp:logits scope { per token: forward_single_streaming { tmp:forward scope { embd row → per layer: load_layer → rmsnorm/qkv/rope/append/attention/swigluf → release_layer } → final norm } → compute_logits → tmp:sampling → KV chunk growth } }`.
5. **MemoryBudget ownership** — VERIFIED. `inference.rs` docs the charge lifetime map; post-run CLI output shows exactly `kv_cache` + `weight:*` charges remaining (temporary scopes released). `MemoryBudget` gives exact accounting *of what it is asked to book* (capacity-reservation style), not byte-exact per-allocation tracking.
6. **Temporary allocation accounting** — VERIFIED. `with_temp` sites: `tmp:logits` (engine), `tmp:sampling` (engine), `tmp:forward` (forward), `tmp:embd_row` (persistent), `tmp:streamed_matvec` (persistent). Release-on-success-and-error proven by 4 unit tests; streamed-matvec no-leak proven by test asserting `used_bytes` unchanged after.
7. **Cache behavior** — VERIFIED. `insert_budgeted/remove_budgeted/clear_budgeted` charge `cache:{key}`; LRU evict releases; budget-full → `Ok(false)` (skip caching); 4 dedicated tests. Engine no longer uses a cache; `BoundedCache` survives only in core + `Runtime::get_tensor` (legacy API, off the run path — see §C-8).
8. **Tensor layout assumptions** — VERIFIED. Single documented ggml convention `shape=[in,out]`, buffer `[out][in]`, `y[o]=Σ_i W[o·in+i]·x[i]` across `tensor.rs`, `backend.rs`, `persistent.rs` streamed matvec; arity mismatches error (tests: `test_matvec_f32_arity_error_not_silently_guessed`, `test_quant_matvec_arity_error`, `test_matvec_arity_mismatch_is_error`).
9. **Quantized matvec behavior** — VERIFIED. Compact residency (`QuantizedTensor` raw bytes; `resident_bytes`=raw len; Q4_K 144 B/256 = 7.1×); block-wise kernels for all 8 formats with block-size/decode/matvec tests (25 in quant.rs); **no eager full-F32 expansion of 2D weights anywhere in the inference path**. Note: anchors with hand-checked reference values exist for Q4_0 `[64,3]` and Q4_K `[256,2]`; the other 6 formats are validated roundtrip-vs-dequant rather than against an external reference (PARTIALLY VERIFIED at reference level).
10. **Output projection** — VERIFIED. Single caller-owned logits buffer (`tmp:logits`, (vocab+n_embd)·4 B); resident F32 → backend; resident quant → compact matvec; streamed → `streamed_matvec_into` (chunks `min(available/4, 16 MiB)`, whole rows, per-row `decode_row_to_f32`, reused buffer; clear error if one row can't fit). Tied-embedding fallback present and exercised (CLI probe model had no `output.weight`).
11. **KV cache behavior** — VERIFIED. Zero prefix copies (`attention(k_hist…, k_new…, hist_len)` slice accessors, tested vs naive reference); F32; `GROW_CHUNK_TOKENS=256`, exact `grow_to` (data-preserving), caller-aligned, growth budget-checked with rollback (release → try → restore → clean error), start at prompt len. **But continued-use across two `generate()` calls is broken — BUG #1.**
12. **Backend/SIMD/rayon integration** — VERIFIED. `matvec_backend` dispatches resident-decoded F32/F16/BF16 → `backend.matvec` (AVX2+FMA via `simd.rs`, runtime-detected; rayon row-parallel when threads>1 ∧ out≥4; scalar fallback); quantized → compact kernels. CLI printed `CPU (CPU-SIMD mode)`. Known caveat stands: `num_threads` not enforced on rayon's global pool.
13. **Tokenizer** — VERIFIED (8 unit tests; CLI probe tokenized/decoded). SentencePiece unigram (Viterbi) + BPE (merges) + byte fallback; tables owned once per engine, **unbudgeted by design** (documented). Not proven byte-exact vs a real SentencePiece/BPE reference (UNVERIFIED externally).
14. **CLI behavior** — VERIFIED. `inspect`/`plan`/`run` + `--json`, `--ram` parser, `--verbose` residency dump; `--help` works; `run` prints M6 accounting (backend mode, budget charges after run); error paths produce clear messages (observed during a malformed-fixture probe: `tensor '…' file_offset … > file_size …`).
15. **Test count and coverage** — VERIFIED. 117 total: quant 25, tensor 14, cache 10, memory 9, backend 8, tokenizer 8, gguf 7, kv_cache 5, datasource 5, sampling 4, persistent 4, streaming_model 3, ops 3, inference 3, simd 2, plan 2, layer 2, residency 1, runtime model 1, core lib 1. Gaps: no end-to-end **quantized** out-of-core generation test (see §C-4), no failure-injection test for failed `generate()` budget state (would have caught BUG #1), no RSS-vs-budget measurement test.
16. **Remaining correctness risks** — see BUG #2, DESIGN LIMITATION #1-#3 (RoPE convention, qwen2 biases, sampler/RMS minor tolerances) — all only observable against real models, which the rules forbid downloading → flagged as UNVERIFIED externally, high-confidence by inspection.
17. **Remaining performance bottlenecks** — see §C-6/§C-7 (per-token full re-stream + per-read `File::open`; scalar quantized matvec; no batched prefill).
18. **Remaining memory-accounting blind spots** — see §C-8 (F16/BF16 underbooking BUG #3; persistent charge-after-alloc skew; `Runtime::get_tensor` clone; unmanaged tokenizer/header/stats/rayon/RSS — the last group is intentionally unmanaged and documented).
19. **Regressions introduced by M6** — ONE found and reproduced: BUG #1. Everything else from M5 behavior (117 suite incl. all adapted fixtures, CLI shapes, plan output semantics minus the removed fake pre-reservations) still passes.
20. **README / MILESTONE6_REPORT.md vs code** — mostly match; previous report's numbers all re-verified. Two qualifications: (a) M6 report item 7 "settles … to the exact resident size" is **not exact for F16/BF16** (settles to file bytes while RAM holds decoded F32 — BUG #3); (b) README's M5-era "synthetic Q4K 800 KB / peak 429 KB" scenario is historical and not part of the current automated suite — under M6's stricter charge-before-read accounting it was **not re-run** (PARTIALLY VERIFIED claim).

## C. Findings (classified)

### BUG-1 (M6 regression, CONFIRMED empirically): `generate()` corrupts budget on repeat/failure
`inference.rs` allocates `"kv_cache"` unconditionally at `generate()` start and never releases it (only swapped on growth). Reproduced: `generate #1: Ok(3)`, `budget after #1: charges=["kv_cache","weight:*"]`, `generate #2: ERR allocation 'kv_cache' already exists`. Consequences: (i) an engine cannot run `generate()` twice; (ii) a *failed* `generate()` leaks the KV charge so the next call also fails (leak verified by reading — same path). The M5 code had an `is_none()` guard. Fix direction (implementation only after approval): clear/release previous KV charge at `generate()` start, and release on error (RAII-style), keeping `self.kv_cache` consistent.

### BUG-2 (correctness, high-confidence by inspection, externally UNVERIFIED): RoPE uses the wrong pairing convention for llama/qwen2
`ops.rs::rope_single` rotates **adjacent pairs** `(x[i], x[i+1])` with `θ = base^(-2·(i/2)/d)·pos` — the GPT-J/interleaved convention. Reference llama/qwen2 (HF `rotate_half`; llama.cpp rope NORMAL) rotates **half-split pairs** `(x[j], x[j+d/2])` with `θ_j = base^(-2j/d)`. Deterministic but numerically different rotations ⇒ all position-dependent math wrong vs real weights. Pre-existing (since M3), invisible to the 117-test suite because fixtures use pos≈0/identity-ish weights. Cannot be 100% proven end-to-end without a real model (forbidden); mathematically unambiguous.

### BUG-3 (accounting defect, confirmed by inspection): F16/BF16 resident RAM underbooked 2×
`TensorData::F16/BF16` stores **decoded `Vec<f32>`** (4 B/elem) but `resident_bytes()` returns `raw_bytes_len` (2 B/elem). Consequences: (i) resident persistents in F16 are booked at half their true occupancy forever; (ii) layer settle "exact resident" underbooks 2×; (iii) load transient books `2×file` while physical peak is `3×file` (raw + decoded). F32 is exact (2× booked = raw+f32). Not codified by tests. Fix direction: book decoded size (4 B/elem), or keep compact F16/BF16 residency with decode-on-use like quants.

### BUG-4 (behavioral defect, externally UNVERIFIED): qwen2 q/k/v bias tensors silently ignored
`group_layers` sweeps all `blk.{i}.*` tensors; real qwen2 files add `attn_q/k/v.bias`. They are loaded + charged but never applied in the forward pass ⇒ real qwen2 math wrong (llama unaffected). Fix direction: either apply biases (correctness) or hard-reject models containing them (fail loudly).

### PERFORMANCE-BOTTLENECK-1 (structural, known since M4): full re-streaming per token
Every forward re-reads **all layer bytes from disk** ⇒ decode I/O ≈ `tokens × model_size`, plus per-tensor `File::open` each call (9 opens/layer/token), alloc churn for fresh per-token tensor buffers, and no prefetch/double-buffering (explicitly out of scope). OS page cache warms content, but syscall+memcpy+parse cost remains the dominant term. This is the single biggest cost and sets the throughput floor for big models.

### PERFORMANCE-BOTTLENECK-2 (known, documented): compute kernels
Quantized matvec is scalar block-wise (no AVX2 for quants); prompt prefill is strictly sequential token-by-token (each with full layer streaming); attention scores allocate per head (inside the reserved workspace — memory-bounded, but malloc churn).

### DESIGN-LIMITATION-1: budget semantics are capacity reservations
`tmp:forward`-style charges book worst-case workspace capacity, not exact per-allocation bytes (attention output/scores inside `ops.rs` are covered by the reservation, not individually booked). Intentional and documented; just implies budget ≈ upper bound, not byte-exact trace.

### DESIGN-LIMITATION-2: unmanaged-but-documented memory
Tokenizer tables (~O(vocab), one-time), GGUF header/metadata (owned by `GgufDataSource`), `ResidencyStats`, allocator/rayon stacks, OS RSS/page cache. Outside the "RAMforge-managed" definition — consistent with docs, but the budget number is not RSS.

### DESIGN-LIMITATION-3: smaller gaps
- Persistent resident loads book the charge **after** the actual allocation (transient skew ≤ 25 % of budget by the resident policy).
- `Runtime::get_tensor` clones data (`read` → cached copy charged, returned copy uncharged) — legacy API off the run path.
- `backend.rmsnorm` silently defaults missing weight elements to 1.0 (`unwrap_or(1.0)`); sampler falls back to token `len-1` on exhausted cumsum.
- rayon `num_threads` not enforced on the global pool (pre-existing, documented).
- KV remains F32, unquantized, no eviction; context growth capped at prompt+max_tokens by design.

## D. Summary for milestone planning

- The M6 milestone claims are **standing** — all gates pass on the current tree, and the six M6 fix areas are present and tested in the current source.
- Two findings deserve priority before any M7 feature work: **BUG-1** (small, local: engine robustness) and a decision on **BUG-2** (RoPE convention — blocks any claim of real-model correctness), with **BUG-3** next (integrity of the central M6 promise for F16/BF16 files), and **BUG-4** if real qwen2 support matters.
- No new dependencies were needed for anything audited. No files were created or modified inside the project during this audit (probes lived in `/tmp` and were deleted).

Awaiting approval before any implementation.
