# Project state

**As-of date:** 2026-10-02; local verification refreshed 2026-10-24.  
**Repository:** RAMforge workspace at the handoff source checkout.  
**Status labels:** see [README.md](README.md).

## Local verification (2026-10-24)

- The workspace builds and tests against the pinned `forge-core` revision
  `d320309` (v3.1.2) with Rust/Cargo 1.98.1 and the workspace native tree at
  `/home/neon/Documents/CODE LOCAL/forge/native` (llama.cpp/ggml v0.5.0).
- `scripts/test.sh --workspace --locked` sets `FORGE_LLAMA_DIR` and the
  process-local `LD_LIBRARY_PATH`, auto-detects
  `$FORGE_LLAMA_DIR/models/tiny-llama.gguf` for `FORGE_TEST_MODEL`, then runs
  `cargo test`: **423 passed, 0 failed, 2 ignored** (the real-model Qwen2.5
  diagnostic and the quantized microbenchmark).
- The ForgeCore adapter integration suite checks independent native context
  state across drops, owned logits surviving session drop, and model reuse
  after context drops.
- A fixture-backed memory test exercises logical GGUF bytes, native
  `size_bytes()`, measured RSS, and explicit budget non-enforcement against the
  real native model.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: passed.
- `cargo fmt --all -- --check`: **not clean**; the repository has pre-existing
  formatting differences in several files. They are not mass-reformatted here.
- The ForgeCore CLI `--memory-report` path now reports separate logical GGUF,
  native `size_bytes()`, measured process-RSS, and device facts, and states
  that the requested RAM budget is **not enforced** by that backend.
- The `forge-core` pin is intentionally retained at v3.1.2. The newer local
  v3.1.3 adds GGUF/quant inspection APIs that RAMforge's adapter does not
  consume. A temporary path-override check showed RAMforge compiles against
  v3.1.3, so the pin is a stability choice, not a compatibility requirement.
  Revisit only when RAMforge needs a newer API.
- See [../../scripts/test.sh](../../scripts/test.sh) for the reproducible test
  invocation.

### Real-model smoke evidence (2026-10-24)

`scripts/smoke-real-models.sh --no-build --matrix --include-7b --max-tokens 4
--prompt Hello --timeout 900` ran each case in a separate process, sequentially,
with a 4 GiB `MemAvailable` start threshold. Evidence was written to
`/tmp/ramforge-real-model-smoke/20261024-193321`; TSV, JSON, per-case commands,
stdout/stderr, hashes, and system-memory samples are retained there. The model
paths were directories from the new layout, so the split-GGUF discovery and
merged virtual data source were exercised by the ordinary CLI path.

Exact budgets: `8G` = 8,589,934,592 B, `4G` = 4,294,967,296 B, `1G` =
1,073,741,824 B, `100M` = 104,857,600 B.

| Model / case | Budget | Exit | Wall | Max RSS | Managed peak | Evictions |
|---|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M Q4_K_M legacy | 8G | 0 | 7.36 s | 138,732 KiB | 104,007,876 B | 0 |
| SmolLM2-135M Q4_K_M legacy | 4G | 0 | 7.33 s | 138,596 KiB | 104,007,876 B | 0 |
| SmolLM2-135M Q4_K_M legacy | 1G | 0 | 7.42 s | 138,496 KiB | 104,007,876 B | 0 |
| SmolLM2-135M Q4_K_M legacy | 100M | 0 | 7.66 s | 116,752 KiB | 73,926,852 B | 0 |
| Qwen2.5-1.5B Q4_0 legacy | 8G | 0 | 8.69 s | 1,151,564 KiB | 1,061,215,600 B | 0 |
| Qwen2.5-1.5B Q4_0 legacy | 4G | 0 | 8.70 s | 1,151,644 KiB | 1,061,215,600 B | 0 |
| Qwen2.5-1.5B Q4_0 legacy | 1G | 0 | 8.98 s | 1,127,204 KiB | 1,061,445,120 B | 85 |
| Qwen2.5-1.5B Q4_0 legacy | 100M | 0 | 23.71 s | 169,240 KiB | 80,198,656 B | 110 |
| Qwen2.5-7B Q4_K_M, two shards, legacy | 4G | 0 | 5:18.67 | 4,279,892 KiB | 4,285,007,616 B | 88 |
| SmolLM2-135M Q4_K_M ForgeCore CPU | 4G | 0 | 0.25 s | 374,700 KiB | not exposed | n/a |
| Qwen2.5-1.5B Q4_0 ForgeCore CPU | 4G | 0 | 1.23 s | 2,839,396 KiB | not exposed | n/a |

All cases generated 4 tokens with greedy sampling. Legacy `/usr/bin/time`
reported `Swaps: 0`; system-wide swap usage is sampled separately and is not
attributed to the process. The 7B case used 112 layer loads, retained 24 layers
at peak, and stayed 9,959,680 B below its 4 GiB budget. The GPU remained
unused: the pinned native build is CPU-only, and RAMforge has no GPU execution
backend.

A controlled failure at `--ram 100M` on Qwen2.5-7B exited 1 in 0.41 s with
`managed lower bound 149127168 exceeds RAMforge budget 104857600` and
94,464 KiB max RSS. It did not OOM, hang, or fall back. The 100M Qwen2.5-1.5B
success above is not a claim that every model fits in 100M.

The real 20B `gptneox` GGUF (13,135,032,320 B, 12.23 GiB) demonstrates the
inspect/plan/run boundary. `inspect` and `plan --ram 100M` both exit 0 and
report the architecture as inspectable/plannable but not executable. `run`
exits 1 in 0.02 s with 26,148 KiB max RSS and `Swaps: 0`, error:
`unsupported execution architecture 'gptneox': inspection=true planning=true
execution=not-yet... currently supported execution architectures: [llama,
qwen2]`. No weights were loaded and no fallback occurred.

Chat-template support was validated on the real Qwen2.5-1.5B GGUF. The model's
own `tokenizer.chat_template` rendered a 177-byte ChatML prompt, tokenized the
control markers as their model IDs, generated 2 tokens (then EOS), and output
`4` for `What is 2+2? Answer briefly.` The run used 1G, took 1:07.21, peaked
at 1,064,027,136 B managed memory, loaded 1008 layers, and evicted 981. The
same model with a raw prompt did not produce answer-quality output. SmolLM2's
ChatML template also parsed and rendered, though its small-model answer was not
coherent. A chat-templated Qwen2.5-7B run at 4G was stopped by the 900 s harness
timeout before its first token; that is a performance limit, not a quality
result.

Q5_0 block decoding is additionally checked against a vector produced by the
pinned native `ggml` `dequantize_row_q5_0` implementation, not just by
end-to-end text coherence.

The earlier budget-pressure run used `--ram 768M` on Qwen2.5 legacy, below its
managed footprint. It exited 0, generated 8 tokens in 18.36 s, and reported
peak managed 798,260,928 B (761.28 MiB) under the 805,306,368 B (768 MiB)
budget, peak resident layer 26,341,376 B (25.12 MiB), 224 layer loads, 207
layer-cache evictions, max RSS 869,312 KiB, and `Swaps: 0`. This demonstrates
budget-constrained eviction. It does not demonstrate control of total RSS,
which includes unmanaged loader, allocator, page-cache, and workspace memory.

Read-only model hashes from the matrix inventory:

- SmolLM2: `2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d`
- Qwen2.5-1.5B: `dcd819ff094852c38faba6873d8ff0c9d51eadb2844539e52042ae5d647bbfdb`
- Qwen2.5-7B shard 1: `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db`
- Qwen2.5-7B shard 2: `539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a`

**Current ForgeCore note:** static ForgeCore device discovery, explicit
selection, and execution metadata are implemented. The verified ForgeCore
`v3.1.2` Model/Context surface now includes public static model-load/offload
options, but the RAMforge adapter remains deliberately CPU-only in this
refresh; GPU facts may be discoverable but non-CPU model execution is
explicitly unsupported, with no CPU fallback. See [README.md](../../README.md)
and the historical [PHASE_4_ENGINEERING_REPORT.md](PHASE_4_ENGINEERING_REPORT.md).

## What RAMforge is

RAMforge is a local, CPU-oriented GGUF inspection, planning, and inference runtime with an explicit static ForgeCore device-discovery boundary. Its original purpose is **usable out-of-core inference**: run a model whose complete tensor payload does not fit in the selected RAM budget by treating the model file as backing storage, retaining only selected persistent weights and the active/cacheable transformer layers in memory.

The repository describes RAM, VRAM, and storage as a hierarchical memory system. Phase 4 can snapshot native CPU/GPU/accelerator facts through ForgeCore and associate them with separate accounting domains. Although verified ForgeCore `v3.1.2` exposes static model-load/offload options, the implemented RAMforge ForgeCore adapter remains CPU-only in this refresh and the legacy planner/runtime remains a separate CPU-only path. RAMforge explicitly tracks the memory it owns through `MemoryBudget`; this is not a claim to control process RSS, the operating-system page cache, VRAM, or native backend allocations.

## Workspace and version assumptions

**VERIFIED from the manifests:** the workspace has three crates and uses Cargo resolver 2:

- `crates/ramforge-core` — GGUF parsing, metadata, tensor storage/decoding, tokenizer, quantization, memory budget, bounded cache, and pure reference compute.
- `crates/ramforge-runtime` — discovery, profiles, planner, calibration, persistence, runtime configuration, streaming model, model execution, inference, cache, profiling, and CPU backend.
- `crates/ramforge-cli` — `ramforge` CLI and `ramforge-tui` interactive interface.

All crates currently declare package version `0.1.0`, Rust edition `2021`, and the workspace has no `rust-toolchain.toml` or `rust-version` constraint. The recorded final validation used rustup stable **Rust/Cargo 1.98.1**. Do not interpret that as a repository-pinned toolchain requirement; it is the toolchain used for the current results.

No model file, external llama.cpp checkout, or external reference output is present in this repository.

## Implemented milestones

The following is the repository's milestone history, cross-checked against the current source. These are implementation milestones, not a claim that every numerical path is production-correct for every architecture.

- **GGUF inspection:** `parse_gguf_file` reads GGUF header, metadata, tensor descriptors, offsets, dimensions, and type information without loading tensor payloads. GGUF versions 1–3 are accepted by the parser.
- **Split/model-directory support:** `discover_model_set` resolves a file, a standard `*-00001-of-0000N.gguf` shard set, or a single-model directory; validates split counts/indices/tensor counts, architecture and duplicate tensor names, then exposes a merged model view with virtual data offsets. `GgufDataSource` reads across shard handles without concatenating files on disk.
- **Chat templates:** `ChatTemplate` parses GGUF `tokenizer.chat_template` with a conservative Jinja subset and renders one user message plus an optional system message. `--chat` is opt-in; unsupported constructs fail explicitly. Tokenizer encoding recognizes model control tokens such as role markers.
- **Budgeted file-backed access:** `GgufDataSource` retains a synchronized file handle, performs bounded range reads, tracks logical versus physical I/O, can reuse the file cursor, and supports bounded coalesced reads.
- **Memory and cache accounting:** `MemoryBudget` provides named allocation/resize/release accounting and scoped temporary reservations. `BoundedCache` and the runtime `LayerCache` enforce byte limits and release charges on eviction.
- **CPU dense inference:** the runtime has a Llama/Qwen2-shaped transformer path with tokenizer, RMSNorm, RoPE, attention, GQA, SwiGLU, residuals, KV cache, output projection, sampling, and generation.
- **Out-of-core layer streaming:** `StreamingLlamaModel` loads layers on demand, executes them, and either caches or releases their materialized representation. Persistent weights can remain resident or be streamed according to actual resident size and budget.
- **Quantized tensor support:** compact resident representations and scalar/block-wise dequantization/matvec paths exist for the formats listed below.
- **Accounting hardening:** direct F32 reads, F16/BF16 decoded residency accounting, transactional persistent-weight loading, grouped reads, grouped-buffer reuse, chunked output projection, and chunk-growing KV-cache accounting are implemented and tested.
- **Compute-layer remodel:** `ramforge_core::compute` now contains storage-independent scalar/reference operations and explicit matrix/attention contracts. Runtime scalar operations delegate to those references and optimized paths have focused equivalence tests. This was a structural boundary improvement, not proof of Qwen2.5 numerical parity; see [CURRENT_INFERENCE_ENGINE.md](CURRENT_INFERENCE_ENGINE.md) and [QWEN25_CORRECTNESS_INVESTIGATION.md](QWEN25_CORRECTNESS_INVESTIGATION.md).
- **Planning/orchestration:** inspection, machine/storage discovery, model profiles, capability derivation, calibration, deterministic planner output, plan compilation, plan persistence, and runtime activation are implemented.
- **Static ForgeCore execution boundary:** `DeviceId`, `BackendId`, `DeviceKind`, `DeviceInfo`, capabilities, `DeviceRegistry`, explicit selection, and `ExecutionConfig` are implemented. CPU is selectable; native non-CPU facts are preserved but rejected by the RAMforge adapter's deliberate CPU-only contract. ForgeCore `v3.1.2` has the public static offload surface, but this task does not enable it. Selection is not residency proof.
- **CLI/TUI diagnostics:** CLI `inspect`, `plan`, `support`, and `run` commands exist. The CLI can emit profile and memory reports, and `run --device` now carries an explicit static ForgeCore selection. The TUI has planning, calibration, persisted-plan compatibility, generation, diagnostics, run history, and export flows; no TUI redesign was made for Phase 4.

## GGUF and model support

### Generic GGUF support

`ramforge-core/src/gguf.rs` parses metadata types, arrays, tensor names, dimensions, GGML types, alignment, descriptor offsets, and computed byte lengths. `GgufModel` in `ramforge-core/src/model.rs` exposes generic inspection data and `ModelInfo` derives common architecture, context, embedding, layer, head, tokenizer, and vocabulary metadata.

Generic inspection and planning are intentionally broader than execution. An architecture name appearing in a GGUF does not by itself make it executable.

### Execution architectures

The explicit registry is `ramforge-runtime/src/support.rs`:

| Architecture/alias | Inspect | Plan | Direct run status in registry |
|---|---:|---:|---|
| `llama` | yes | yes | supported |
| `qwen2`, including alias `qwen2.5` | yes | yes | supported |
| `mistral` | yes | yes | `via-llama-gguf` classification; not a direct independent Mistral implementation |
| `qwen3` | yes | yes | not yet |
| `qwen35` / `qwen3.5` / `qwen3_5` | yes | yes | not yet; explicitly not aliased to Qwen2 |
| `gemma` family | yes | yes | not yet |
| `phi` family | yes | yes | not yet |

The current runtime model parser is `LlamaConfig` in `crates/ramforge-runtime/src/model.rs`. It accepts the dense Llama/Qwen2 tensor contract and reads architecture-specific or fallback metadata keys. The registry labels Qwen2.5 as a Qwen2-shaped execution target, but the correctness investigation shows that this path must not be treated as numerically validated for the named Qwen2.5 model.

### Tensor/quantization formats

The source-level inference registry marks these as supported:

- `F32`
- `F16`
- `BF16`
- `Q4_0`
- `Q5_0`
- `Q8_0`
- `Q2_K`
- `Q3_K`
- `Q4_K`
- `Q5_K`
- `Q6_K`
- `Q8_K`

`Q4_1`, `Q5_1`, `Q8_1`, IQ formats, and other unimplemented GGUF types remain unsupported for inference even though the generic type enum can preserve unknown/future tags for inspection.

`Q5_0` was added to the inference registry after validating the real `SmolLM2-135M-Instruct-Q4_K_M.gguf` model through the legacy backend. Its block decode is checked against a vector produced by the pinned native `ggml` `dequantize_row_q5_0` implementation.

## Current subsystem state

### Production-oriented infrastructure

The following infrastructure is implemented with extensive unit/integration coverage and is the part intended to survive a ForgeCore rewrite:

- GGUF descriptor parsing and validation.
- File-backed `GgufDataSource`, bounded reads, cursor reuse, read profiling, coalescing, and grouped buffer reuse.
- Named `MemoryBudget` accounting and scoped temporary reservations.
- Persistent-weight policy, layer grouping, layer read plans, `LayerCache`, and rollback/eviction accounting.
- Machine and storage discovery facts, model/storage/machine fingerprints, capability derivation, planner profiles, and plan compilation.
- Bounded calibration tasks and persisted execution-plan compatibility checks.
- CLI/TUI diagnostic collection, profiling, memory visibility, run history, persistence, and error staging.

“Production-oriented” here means that these are concrete implemented subsystems with tests. It is not a blanket production-readiness or model-quality guarantee.

### Experimental or not fully validated

- Real-model numerical quality and compatibility, especially Qwen2.5-1.5B, are not established.
- The current scalar/reference remodel has synthetic parity tests but does not explain the recorded Qwen2.5 divergence.
- Real-model generated-text quality is not validated by the repository test suite.
- The Qwen2.5 numerical trace is an ignored test requiring an external model and explicit budget.
- Calibration is bounded and records measured facts; it does not prove inference correctness or choose a different numerical implementation.
- The quantized microbenchmark is ignored and is not part of normal workspace validation.

### Incomplete or unsupported

The repository does not implement GPU model execution or GPU offload through RAMforge (the current adapter intentionally leaves ForgeCore's static offload surface unused), asynchronous I/O, prefetch/double buffering, HTTP serving, model downloading, MoE execution, speculative decoding, or the additional quantization formats listed as unsupported. `qwen35` is intentionally inspect/plan-only because its hybrid attention/SSM architecture is not the Qwen2 path.

## Important commands

These are the repository's normal commands. Commands requiring a model are templates and were not run during the handoff unless explicitly marked in [VALIDATION.md](VALIDATION.md).

```text
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo build --workspace --release

cargo run -p ramforge-cli -- inspect MODEL.gguf
cargo run -p ramforge-cli -- plan MODEL.gguf --ram 8G
cargo run -p ramforge-cli -- support
cargo run -p ramforge-cli -- run MODEL.gguf --ram 8G --prompt "Hello" --max-tokens 32 --profile --memory-report
cargo run -p ramforge-cli -- run MODEL.gguf --ram 8G --backend forgecore --device cpu --prompt "Hello"
```

The ignored real-model diagnostic is documented in [VALIDATION.md](VALIDATION.md) and the source doc comment in `crates/ramforge-runtime/src/inference.rs`; do not run it casually.

## Validation snapshots

The following results are historical validation for the pre-Phase-4 source
snapshot, recorded with Rust/Cargo 1.98.1:

- `cargo fmt --all -- --check`: passed.
- `cargo test --workspace`: passed — 44 CLI tests, 112 core tests, 195 runtime tests; **351 passed, 0 failed, 2 ignored**.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: passed.
- `cargo build --workspace --release`: passed.

For the current Phase-4 source, `cargo` and `rustc` are not available on the
current `PATH`, the historical local toolchain/native locations are absent,
and no compatible native ForgeCore build is present. Phase-4 formatting,
compilation, tests, clippy, release build, and native-device execution are
therefore **not run**. See [PHASE_4_ENGINEERING_REPORT.md](PHASE_4_ENGINEERING_REPORT.md)
for the authoritative Phase-4 validation record. The two historical ignored
tests include the expensive real-model Qwen2.5 diagnostic and the quantized
microbenchmark.
