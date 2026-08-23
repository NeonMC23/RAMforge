# Milestone 6.1 Report — Correctness & Accounting Fixes

Date: 2026-08-23 · Toolchain: rustc/cargo 1.98.0 · Scope: fix the four audit bugs (BUG-1..BUG-4) from
`POST_M6_AUDIT.md` as one coherent correctness/accounting pass. No new dependencies, no git
operations, no real model downloads, no performance work, no existing tests weakened.

## 1. Bugs fixed

| Bug | Symptom | Fix |
|-----|---------|-----|
| **BUG-1** | Second `generate()` on one `InferenceEngine` failed with `allocation 'kv_cache' already exists`; failed runs leaked their KV charge | `generate()` now explicitly resets any previous KV state via `clear_kv_cache()` (drops the cache object *and* releases the budget charge, idempotent) before running, and runs the cleanup again on every error path. Failed generations leave zero KV/budget residue and the engine stays reusable. |
| **BUG-2** | RoPE used GPT-J/interleaved adjacent pairs `(x[2j], x[2j+1])` — wrong for llama/qwen2 weights | `rope_single` rewritten to the llama/qwen2 **half-split** convention: pairs `(x[j], x[j + head_dim/2])` with `theta_j = pos * base^(-2j/head_dim)`, per head, position 0 = identity. |
| **BUG-3** | F16/BF16 tensors are stored decoded as `Vec<f32>` (4 B/elem) but `resident_bytes()` reported the 2 B/elem file size — a 2× underbooking; the layer-load transient only charged 2× file bytes instead of 3 | `resident_bytes()` now returns `data.len() * 4` for F32/F16/BF16 (the `raw_bytes_len` field was deleted). New `load_charge_bytes()` charges the transient at: quantized 1×, F32 2×, F16/BF16 3× file bytes, with `checked_mul` overflow error. Invariant restored: **charge ≥ actual RAM representation intentionally owned by RAMforge**. |
| **BUG-4** | qwen2 `blk.{i}.attn_q/k/v.bias` were loaded and budget-charged but never applied (silently ignored) | **Option A — biases supported.** Detected per model (`attn_q.bias` presence); validated all-or-none and exact 1D shape `[q_dim]`/`[kv_dim]`; applied after the Q/K/V projection matvec, before RoPE and KV insertion; charged with the layer and released with it. Partial sets and shape mismatches are hard errors that fully clean the budget. |

A pre-existing related leak was fixed along the way: a tensor-extraction error mid-`load_layer`
(e.g. missing tensor in file) previously kept the already-allocated layer charges; the extraction
closure now releases every layer charge on any error.

## 2. Files modified

- `crates/ramforge-runtime/src/ops.rs` — half-split `rope_single` + doc; 4 new convention tests (old `test_rope` removed).
- `crates/ramforge-core/src/tensor.rs` — `resident_bytes()` truthfulness for F32/F16/BF16; `raw_bytes_len` field and its uses removed; 2 new tests.
- `crates/ramforge-runtime/src/streaming_model.rs` — `load_charge_bytes()`; bias fields on `StreamingLayerWeights` (+ `total_resident_bytes`); `attn_bias_present` detection; error-clean tensor extraction; `validate_qkv_bias` (all-or-none + shape); bias application in `forward_layer`; `act_floats` budget term for biases; 5 new tests + F16/BF16 and biased-qwen2 fixtures.
- `crates/ramforge-runtime/src/inference.rs` — `clear_kv_cache()`; `generate()` wrapper (reset before, cleanup on error) with the old body preserved verbatim as `generate_impl`; 5 new tests incl. an independent reference-forward fixture for biased qwen2.
- `README.md` — M6.1 status header, M6.1 section, F16/BF16 accounting bullet, test totals (132).
- `MILESTONE6_REPORT.md` — M6.1 addendum correcting stale claims.

## 3. Tests added (net +15: 117 → 132)

- `ops::tests` (**+4**): `test_rope_position_zero_is_identity` (dims 4/8/16), `test_rope_half_split_convention_nonzero_position` (dim 8, pos 3, asserts exact half-split math *and* > 0.05 divergence from the adjacent-pair convention), `test_rope_pairing_discriminator_sparse_vector` (sparse `x[1]=1, x[5]=2`), `test_rope_multi_head_uses_per_head_block`.
- `tensor::tests` (**+2**): `test_resident_bytes_f16_reflects_decoded_f32_storage`, `test_resident_bytes_bf16_reflects_decoded_f32_storage`.
- `streaming_model::tests` (**+5**): `test_f16_bf16_persistent_and_layer_accounting`, `test_layer_load_failure_cleans_budget`, `test_qkv_bias_layer_loading_and_accounting`, `test_qkv_bias_partial_set_rejected`, `test_qkv_bias_shape_mismatch_rejected`.
- `inference::tests` (**+5**): `test_generate_twice_same_engine`, `test_failed_generate_releases_kv_charge`, `test_engine_reusable_after_failed_generate`, `test_clear_kv_cache_releases_charge`, `test_qwen2_biased_forward_matches_reference` (two tokens at positions 0 and 1; hidden state *and* logits match an independent reference implementation within 2e-4).

## 4. Existing tests preserved

All 117 M6 tests still pass. Exactly one test was *replaced, not weakened*: the old `test_rope`
encoded the incorrect adjacent-pair convention; it was superseded by the four stricter convention
tests above (non-zero positions, non-symmetric vectors, explicit cross-convention divergence check).
`MILESTONE6_REPORT.md` items 5/7/19 were updated where they documented the old behavior.

## 5. Exact verification results (this machine, 2026-08-23)

| Gate | Result |
|------|--------|
| `cargo test --workspace` | **132 passed, 0 failed** (81 ramforge-core + 51 ramforge-runtime; CLI floor suites 0) |
| `cargo clippy --workspace -- -D warnings` | exit 0, 0 errors/warnings (forced fresh rebuild) |
| `cargo clippy --workspace --all-targets -- -D warnings` | exit 0, 0 errors/warnings (forced fresh rebuild) |
| `cargo build --workspace` | exit 0 |
| `./target/debug/ramforge --help` | prints usage (inspect / plan / run), exit 0 |

Empirical /tmp harness (probe binary linked against the compiled rlibs + real CLI runs on synthetic
GGUFs — all artifacts deleted afterwards); **all checks passed**:

- Repeated `generate()` ×3 on one engine: success, deterministic, no budget drift for same-shape
  calls, no stale `tmp:*`/`layer:*`/`kv_cache` allocations, never an `already exists` error.
- Failed → retry: context-overflow failure and a genuine budget failure
  (`insufficient memory: requested 648 bytes for 'tmp:forward', but only 344 bytes available
  (total 600, used 256)`) both leave `used_bytes` byte-identical to the pre-attempt value with zero
  KV residue; the engine then generates successfully (or fails again in the *same class* on the
  tight budget — never `already exists`).
- F16/BF16 model: `weight:token_embd.weight` booked at **512 B** decoded (file size 256 B; the old
  bug said 256), settled residency = 544 B; mixed F16/BF16/F32 generation succeeds end-to-end.
- RoPE: direct `apply_rope` at pos 3/dim 8 matches half-split math exactly (≤ 1e-6) and diverges
  from the adjacent-pair convention by 1.354; pos 0 is an exact identity; per-head blocks correct.
- qwen2 biased model via CLI (`ramforge run`, two separate processes): identical greedy output
  (`gggg`); post-run charges exactly `[kv_cache, weight:output_norm.weight, weight:token_embd.weight]`.
- Partial-bias qwen2 model: rejected with `layer 0 has an incomplete Q/K/V bias set (q.bias: true,
  k.bias: false, v.bias: false); refusing to run inference with partial biases`, budget fully clean.

## 6. qwen2 biases: supported or rejected?

**Supported (option A).** Loaded, budgeted, shape-validated (all-or-none; exact `[q_dim]`/`[kv_dim]`
1D), applied after the Q/K/V projection matvec and before RoPE/KV insertion, released with the
layer. Nothing is silently ignored: incomplete sets and wrong shapes are hard errors.

## 7. Exact F16/BF16 accounting behavior

- **Resident charge (persistent + settled layer):** decoded F32 size, `elems × 4 B` — the true RAM
  representation RAMforge owns. The earlier "exact resident size" claims are now literally true;
  no reservation semantics remain for float tensors.
- **Load transient per tensor:** 3× file bytes (raw 2 B/elem + decoded 4 B/elem), charged *before*
  reading via `checked_mul` (overflow → clear error: `tensor size overflow computing load charge …`).
  Quantized tensors keep 1× (compact residency), F32 keeps 2×.
- Failed layer loads release 100% of that layer's charges; a failed `generate()` additionally
  releases the KV cache.

## 8. Exact RoPE convention

llama/qwen2 **half-split rotary** (HF Transformers / llama.cpp RoPE-NORMAL, GPT-NeoX style):
for each head and each `j < head_dim/2`, the pair `(x[j], x[j + head_dim/2])` is rotated by
`theta_j = pos * base^(-2j/head_dim)`; `x'[j] = x_j·cos θ − x_{j+h}·sin θ`,
`x'[j+h] = x_j·sin θ + x_{j+h}·cos θ`. Position 0 is the identity. The ggml matrix convention
(`shape = [in, out]`, buffer `[out][in]`, `y[o] = Σ_i W[o·in+i]·x[i]`) remains the single
authoritative layout — no orientation heuristics anywhere.

## 9. Remaining limitations (unchanged by this milestone)

- **Unverified against real model files** — no model downloads were permitted; all evidence is from
  synthetic GGUFs (unit tests + end-to-end probe + CLI). Real llama/qwen2 files remain untested.
- Per-token **full layer re-stream** (every generated token re-reads all layer weights from disk) —
  the known M6 performance bottleneck, out of scope here.
- Quantized matvec kernels are scalar; AVX2/rayon only accelerate the F32 weight path.
- rayon `num_threads` config is not enforced on the global pool.
- Tokenizer/header/process RSS are outside `MemoryBudget` ("RAMforge-managed memory" =
  budget-tracked allocations only).
- KV cache is F32-only (no KV quantization).
- Legacy `Runtime::get_tensor` clone-per-read path retained for API compatibility.
- `rmsnorm` tolerates a missing weight by substituting 1.0 (`unwrap_or(1.0)`).

## 10. Workspace cleanliness

- No git: no `.git/`, no git commands were run; filesystem is the only source of truth.
- No `.cargo/`, `.profile`, `.github/`, model files (`*.gguf/*.bin/…`), or editor artifacts in the
  project tree (only `target/`, which is gitignored, and only 25 source `.rs` files:
  11 core + 13 runtime + 1 CLI).
- All /tmp probes, fixtures, logs, and binaries deleted after use.
- No new dependencies; `Cargo.toml` files untouched.
