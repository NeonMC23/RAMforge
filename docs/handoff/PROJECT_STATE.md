# Project state

**As-of date:** 2026-09-22.  
**Repository:** RAMforge workspace at the handoff source checkout.  
**Status labels:** see [README.md](README.md).

## What RAMforge is

RAMforge is a local, CPU-oriented GGUF inspection, planning, and inference runtime. Its original purpose is **usable out-of-core inference**: run a model whose complete tensor payload does not fit in the selected RAM budget by treating the model file as backing storage, retaining only selected persistent weights and the active/cacheable transformer layers in memory.

The repository describes RAM, VRAM, and storage as a hierarchical memory system, but the current execution registry has no GPU backend. The implemented execution strategy is CPU layer streaming. RAMforge explicitly tracks the memory it owns through `MemoryBudget`; this is not a claim to control process RSS or the operating system page cache.

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
- **Budgeted file-backed access:** `GgufDataSource` retains a synchronized file handle, performs bounded range reads, tracks logical versus physical I/O, can reuse the file cursor, and supports bounded coalesced reads.
- **Memory and cache accounting:** `MemoryBudget` provides named allocation/resize/release accounting and scoped temporary reservations. `BoundedCache` and the runtime `LayerCache` enforce byte limits and release charges on eviction.
- **CPU dense inference:** the runtime has a Llama/Qwen2-shaped transformer path with tokenizer, RMSNorm, RoPE, attention, GQA, SwiGLU, residuals, KV cache, output projection, sampling, and generation.
- **Out-of-core layer streaming:** `StreamingLlamaModel` loads layers on demand, executes them, and either caches or releases their materialized representation. Persistent weights can remain resident or be streamed according to actual resident size and budget.
- **Quantized tensor support:** compact resident representations and scalar/block-wise dequantization/matvec paths exist for the formats listed below.
- **Accounting hardening:** direct F32 reads, F16/BF16 decoded residency accounting, transactional persistent-weight loading, grouped reads, grouped-buffer reuse, chunked output projection, and chunk-growing KV-cache accounting are implemented and tested.
- **Compute-layer remodel:** `ramforge_core::compute` now contains storage-independent scalar/reference operations and explicit matrix/attention contracts. Runtime scalar operations delegate to those references and optimized paths have focused equivalence tests. This was a structural boundary improvement, not proof of Qwen2.5 numerical parity; see [CURRENT_INFERENCE_ENGINE.md](CURRENT_INFERENCE_ENGINE.md) and [QWEN25_CORRECTNESS_INVESTIGATION.md](QWEN25_CORRECTNESS_INVESTIGATION.md).
- **Planning/orchestration:** inspection, machine/storage discovery, model profiles, capability derivation, calibration, deterministic planner output, plan compilation, plan persistence, and runtime activation are implemented.
- **CLI/TUI diagnostics:** CLI `inspect`, `plan`, `support`, and `run` commands exist. The CLI can emit profile and memory reports. The TUI has planning, calibration, persisted-plan compatibility, generation, diagnostics, run history, and export flows.

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
- `Q8_0`
- `Q2_K`
- `Q3_K`
- `Q4_K`
- `Q5_K`
- `Q6_K`
- `Q8_K`

`Q4_1`, `Q5_0`, `Q5_1`, `Q8_1`, IQ formats, and other unimplemented GGUF types remain unsupported for inference even though the generic type enum can preserve unknown/future tags for inspection.

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

The repository does not implement GPU execution, GPU offload, asynchronous I/O, prefetch/double buffering, HTTP serving, model downloading, MoE execution, speculative decoding, or the additional quantization formats listed as unsupported. `qwen35` is intentionally inspect/plan-only because its hybrid attention/SSM architecture is not the Qwen2 path.

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
```

The ignored real-model diagnostic is documented in [VALIDATION.md](VALIDATION.md) and the source doc comment in `crates/ramforge-runtime/src/inference.rs`; do not run it casually.

## Current repository validation snapshot

**VERIFIED by the final commands run against this checkout with Rust/Cargo 1.98.1:**

- `cargo fmt --all -- --check`: passed.
- `cargo test --workspace`: passed — 44 CLI tests, 112 core tests, 195 runtime tests; **351 passed, 0 failed, 2 ignored**.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: passed.
- `cargo build --workspace --release`: passed.

The current two ignored tests include the expensive real-model Qwen2.5 diagnostic and the quantized microbenchmark. The Qwen2.5 diagnostic was not run for this handoff. See [VALIDATION.md](VALIDATION.md) for the distinction between repository validation and external real-model evidence.
