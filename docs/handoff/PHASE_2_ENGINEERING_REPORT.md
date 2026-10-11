# RAMforge Phase 2 ForgeCore engineering report

> **Historical record.** This report documents the original ForgeCore `v2.0`
> integration at the revision shown below. The current RAMforge source has
> since been refreshed to verified ForgeCore `v3.1.2` at
> `d3203099a315db134193404e79ba2581d9fc53ab`; the historical pin and API
> statements below are retained as evidence, not as the current dependency
> state. See the current [README](../../README.md) and
> [PROJECT_STATE](PROJECT_STATE.md).

**Date:** 2026-10-02  
**Scope:** CPU-only RAMforge → ForgeCore → llama.cpp/ggml integration  
**ForgeCore revision:** `944bc3ca1ed7e48e2ad897107d9dcda5dd5eedc9`  
**ForgeCore package:** `forge-core` from `https://github.com/NeonMC23/forgeCORE.git`

## Executive summary

Phase 2 adds the smallest explicit RAMforge adapter around the current ForgeCore
v2.0 safe API. The new opt-in execution path is:

```text
RAMforge tokenizer/token IDs
    → ForgeCore Model
    → ForgeCore Context
    → ForgeCore BatchBuilder / Batch
    → Context::decode()
    → Context::logits()
    → owned ForgeCore logits copied to RAMforge
    → existing RAMforge sampler/output layer
```

The existing custom `InferenceEngine` and its numerical modules remain intact
and are still the CLI default. The ForgeCore backend is selected explicitly;
there is no implicit fallback from a ForgeCore model, context, batch, decode, or
logits error to the legacy engine.

This phase does not claim GPU, VRAM, multi-GPU, mmap residency, SSD streaming,
eviction, prefetch, async transfer, out-of-core execution, Phase-3 accounting,
or migration of RAMforge's legacy transformer implementation.

## Exact dependency and native boundary

The root workspace adds the following normal Cargo dependency:

```toml
forge-core = { git = "https://github.com/NeonMC23/forgeCORE.git", rev = "944bc3ca1ed7e48e2ad897107d9dcda5dd5eedc9" }
```

`ramforge-runtime` consumes `forge-core` through the workspace dependency. It
does not declare `forge-sys`, and the Phase-2 adapter adds no `llama_*` or
`ggml_*` native declarations, `extern "C"`, or raw native handles. Existing
RAMforge metadata names such as `ggml_type` and legacy numerical code are not
an FFI boundary. `forge-sys` remains a transitive implementation detail of
ForgeCore.

The pinned ForgeCore API inspected for this integration is:

- `Model::load_with_options(&Path, &ModelOptions)`;
- `Model` metadata getters such as `vocab_size`, `n_ctx_train`, `n_embd`,
  `n_layer`, `n_head`, `n_head_kv`, `description`, and `size_bytes`;
- `Context::open(&Model, &ContextOptions)`;
- `BatchBuilder::new(vocab_size)`, `push(token, position, want_logits)`, and
  `build()`;
- `Context::decode(&Batch)`;
- `Context::logits(token_index)` returning an owned `Logits` value;
- `Logits::values()` and `Logits::n_vocab()`.

ForgeCore's `ModelOptions` and `ContextOptions` are `#[non_exhaustive]`; the
adapter correctly starts with `Default` values and assigns the CPU fields
rather than constructing those structs with external struct literals.

## Implemented files and API

### `crates/ramforge-runtime/src/forgecore.rs`

Adds the public adapter types:

- `ForgeCoreModelOptions` — CPU model loading options, currently only
  `check_tensors`;
- `ForgeCoreModel` — owns the safe ForgeCore `Model`, path, and copied native
  metadata; provides `load`, `load_with_options`, `metadata`, and
  `open_context`;
- `ForgeCoreModelMetadata` — copied model facts used for diagnostics and
  tokenizer vocabulary compatibility checks;
- `ForgeCoreContextOptions` — explicit CPU context length/thread options;
- `ForgeCoreSession` — owns the ForgeCore context and shared model handle;
- `ForgeCoreSession::decode_tokens` — builds a ForgeCore batch with consecutive
  positions, marks only the final token for logits, calls `decode`, calls
  `logits`, and copies the values into RAMforge-owned `ForgeCoreLogits`;
- `ForgeCoreSession::logits` — explicit safe logits-copy access with translated
  invalid/unflagged-index errors;
- `ForgeCoreSession::decode_token` — single-token decode convenience method;
- `ForgeCoreAdapterError` — operation-domain translation for configuration,
  model, context, batch, decode, and logits failures;
- `ForgeCoreLogits` — an owned `Vec<f32>` result for the RAMforge sampling
  boundary.

No transformer, attention, normalization, RoPE, quantized kernel, matmul, graph,
KV policy, residency policy, or allocation accounting was added to this module.

### `crates/ramforge-runtime/src/lib.rs`

Registers and re-exports the adapter without changing the legacy public
`InferenceEngine` module.

### `crates/ramforge-cli/src/main.rs`

Adds the smallest explicit selection surface to the existing `run` command:

```text
--backend legacy       # default; existing InferenceEngine
--backend forgecore    # explicit CPU ForgeCore path
--backend forge-core   # alias
```

The ForgeCore branch retains RAMforge GGUF inspection and tokenizer parsing,
passes prompt token IDs through the adapter, and sends returned logits to the
existing RAMforge `Sampler`. `--ram` remains accepted for CLI compatibility and
is reported as planning/diagnostic input in ForgeCore mode; it is not used to
invent a ForgeCore memory budget. `--profile` and `--memory-report` explicitly
report that Phase-2 native counters are unavailable rather than fabricating
RAMforge-managed values.

The TUI and orchestration APIs remain on their existing `InferenceEngine` path;
no broad TUI migration was performed.

### `crates/ramforge-runtime/tests/forgecore_adapter.rs`

Adds focused integration coverage using `FORGE_TEST_MODEL` and ForgeCore's
existing deterministic tiny GGUF fixture mechanism. The fixture binary is not
stored in RAMforge. Tests cover:

- ForgeCore model load and copied metadata;
- context open and effective context/vocabulary values;
- batch construction, CPU decode, final-position logits, finite/non-trivial
  owned values;
- empty-batch and out-of-vocabulary batch errors;
- native decode error translation with a one-token context;
- missing-model error translation;
- source/dependency separation (`forge-sys` and direct C FFI are absent from
  RAMforge's adapter/manifest).

Without `FORGE_TEST_MODEL`, fixture-dependent tests report `SKIP` and return,
matching ForgeCore's own `tests/cpu_decode.rs` convention.

## Preserved code and infrastructure

The following remain intentionally present and disconnected from the new
ForgeCore execution path:

- `InferenceEngine` and its generation/callback/profiling APIs;
- `model.rs`, `model_executor.rs`, `ops.rs`, `backend.rs`,
  `compute_dispatch.rs`, `kv_cache.rs`, `streaming_model.rs`, and the existing
  numerical/quantized modules;
- GGUF inspection, `GgufDataSource`, tokenizer metadata/decoder, planning,
  `MemoryBudget`, bounded caches, residency reporting, diagnostics, profiles,
  orchestration, CLI structure, and TUI structure;
- future-facing backend/planner/runtime configuration types.

No old engine was deleted, silently disconnected, replaced, or used as an
implicit ForgeCore fallback. The retained code is the comparison/reference and
future policy substrate for later migration phases.

## Validation record

Validation was run with Rust/Cargo `1.99.0`, ForgeCore's CPU native libraries
under `/var/tmp/forge-native`, and all generated Cargo state under
`/var/tmp/forge-toolchain` (outside the RAMforge workspace):

```text
cargo fmt --all -- --check                                      PASS
cargo check --workspace --all-targets                           PASS
cargo test --workspace                                          PASS
cargo clippy --workspace --all-targets --all-features -- -D warnings  PASS
cargo build --workspace --release                              PASS
FORGE_TEST_MODEL=/var/tmp/forge-native/models/tiny-llama.gguf \
  cargo test -p ramforge-runtime --test forgecore_adapter -- --nocapture  PASS
```

The final workspace test run reported 44 `ramforge-cli`, 112
`ramforge-core`, 198 `ramforge-runtime` unit, and 4 Phase-2 adapter
integration tests passed: **358 passed, 0 failed, 2 ignored**. The two ignored
tests are pre-existing legacy diagnostics/benchmarks, not Phase-2 failures.
The focused adapter integration test passed all four tests, including native
model metadata, context, batch/decode/logits, translated errors, and FFI
separation. A release binary also completed an explicit tiny-fixture
`ramforge run --backend forgecore` smoke run and returned sampled output.

The deterministic ForgeCore tiny fixture was generated at
`/var/tmp/forge-native/models/tiny-llama.gguf` (6,880 bytes) using ForgeCore's
existing `scripts/make-tiny-gguf.py`; it was supplied through `FORGE_TEST_MODEL`
and was not copied into this workspace. ForgeCore's pinned upstream CPU native
checkout and libraries also remain outside `/home/user/`.

No model file, native library, `target/`, generated fixture, or second
llama.cpp/ggml source tree was added to the RAMforge workspace.
