# Repository transfer

This document is for the engineer or team taking custody of RAMforge after the documentation-only handoff.

## Transfer summary

RAMforge is a Rust workspace for GGUF inspection, budgeted file-backed access, CPU layer-streamed dense Llama/Qwen2 inference, planning, persistence, profiling, CLI, and TUI operation.

The transfer decision is:

- **Preserve:** GGUF parsing, descriptor/data-source I/O, memory accounting, layer grouping/read plans, persistent-weight policy, layer cache, planner/discovery/capability structures, calibration, plan compiler, runtime configuration, persistence, profiling, diagnostics, CLI/TUI, and the existing test/build baseline.
- **Rebuild:** the numerical compute/inference core used for trustworthy Qwen2.5 execution. The proposed replacement name is **ForgeCore**.
- **Do not claim:** llama.cpp compatibility, Qwen2.5 numerical parity, real-model output quality, or real-model performance.
- **Do not change in this handoff:** production code, dependencies, package versions, milestone numbering, current engine behavior, or repository files outside `docs/handoff/`.

ForgeCore is a proposed name and architecture only. There is no ForgeCore implementation in this repository.

## Files in this handoff package

Read in this order:

1. [`README.md`](README.md) — index, reading order, labels, conclusion, and limits.
2. [`PROJECT_STATE.md`](PROJECT_STATE.md) — current scope, subsystem state, support registry, and recorded validation snapshot.
3. [`ARCHITECTURE.md`](ARCHITECTURE.md) — current modules, data flow, ownership, and boundaries.
4. [`CURRENT_INFERENCE_ENGINE.md`](CURRENT_INFERENCE_ENGINE.md) — current tensor/matrix/operator/KV/generation semantics.
5. [`QWEN25_CORRECTNESS_INVESTIGATION.md`](QWEN25_CORRECTNESS_INVESTIGATION.md) — fixed target identity, prompt IDs, trace schema, output geometry, discrepancy evidence, and unknowns.
6. [`REBUILD_PLAN.md`](REBUILD_PLAN.md) — proposed ForgeCore sequence and acceptance gates.
7. [`VALIDATION.md`](VALIDATION.md) — commands that actually ran, test counts, diagnostic status, and historical-report reconciliation.
8. [`REPOSITORY_TRANSFER.md`](REPOSITORY_TRANSFER.md) — this operational transfer checklist.

## Source map

### Core crate: `crates/ramforge-core`

| Path | Role | Transfer note |
|---|---|---|
| `src/gguf.rs` | GGUF header/metadata/tensor descriptor parsing | preserve; add tests only with an explicit task |
| `src/model.rs` | `GgufModel`, `TensorDescriptor`, generic inspection | preserve as source-of-truth descriptor model |
| `src/types.rs` | metadata values and `GgmlType` | preserve broad inspection type handling |
| `src/datasource.rs` | file-backed bounded reads and I/O profiling | preserve; this is a key ForgeCore storage adapter candidate |
| `src/memory.rs` | named/scoped budget accounting | preserve; numerical integration must use it |
| `src/cache.rs` | bounded generic cache | preserve where appropriate |
| `src/tensor.rs` | current tensor representations and matvec-facing contract | numerical rebuild candidate; do not assume semantics are final |
| `src/quant.rs` | current quantized block layouts/decoders/row paths | rebuild/audit candidate; use raw-row evidence before reusing |
| `src/compute.rs` | current scalar/reference compute boundary | structural starting point, but not a Qwen2.5 correctness certificate |
| `src/tokenizer.rs` | GGUF tokenizer loading, Qwen2/GPT2 BPE, byte fallback, streaming decoder | preserve infrastructure candidate; revalidate target metadata and exact IDs |

### Runtime crate: `crates/ramforge-runtime`

| Path | Role | Transfer note |
|---|---|---|
| `src/inference.rs` | `InferenceEngine`, generation lifecycle, ignored Qwen2.5 diagnostic | retain diagnostic/evidence; generation numerical path is rebuild target |
| `src/model.rs` | dense Llama/Qwen2 `LlamaConfig` and required tensors | preserve metadata boundary; validate model-specific assumptions |
| `src/streaming_model.rs` | layer streaming, persistent weights, forward orchestration, output projection | preserve resource seams; replace/adapt numerical calls |
| `src/model_executor.rs` | one transformer layer's current numerical execution | primary ForgeCore integration/replacement candidate |
| `src/backend.rs` | CPU backend and optimized dispatch adapter | keep only after parity against ForgeCore reference |
| `src/compute_dispatch.rs` | representation-based matvec dispatch | keep resource/dispatch role; prove quantized parity |
| `src/ops.rs` | runtime operation adapters | audit/reduce to adapters around one authoritative reference |
| `src/kv_cache.rs` | budget-aware F32 KV cache, chunk growth, layout | preserve resource container; revalidate numerical staging contract |
| `src/layer.rs` | layer/persistent descriptor grouping | preserve |
| `src/layer_read.rs` | descriptor-only/coalesced layer read plans | preserve |
| `src/layer_cache.rs` | budget-charged LRU decoded layer cache | preserve |
| `src/persistent.rs` | resident/streamed persistent weights and chunked projection | preserve memory/I/O behavior; adapt numerical representation |
| `src/accounting.rs` | shared planning/accounting formulas | preserve and keep distinct from runtime sufficiency |
| `src/plan.rs` | static execution-memory plan | preserve |
| `src/planner.rs` | profiles, capabilities, execution plans | preserve |
| `src/plan_compiler.rs` | decision/resource safety boundary | preserve |
| `src/runtime_config.rs` | compiled runtime execution contract | preserve |
| `src/discovery.rs` | machine/storage facts | preserve |
| `src/calibration.rs` | bounded measurement tasks | preserve; not a correctness oracle |
| `src/plan_persistence.rs` | versioned plan persistence/compatibility | preserve |
| `src/profile.rs` / `src/residency.rs` | profiling/residency counters | preserve diagnostic vocabulary |
| `src/memory_report.rs` | managed/RSS/system memory distinction | preserve |
| `src/support.rs` | architecture/format capability registry | preserve, but avoid overstating Qwen2.5 numerical support |

### CLI crate: `crates/ramforge-cli`

- `src/main.rs` — `inspect`, `plan`, `support`, and `run` commands.
- `src/tui/app.rs` — TUI state machine and orchestration flows.
- `src/tui/mod.rs` — TUI command/action sequencing.
- `src/tui/render.rs` — display screens and fields.
- `src/tui/terminal.rs` — terminal input/commands.
- `src/tui/diagnostics.rs` — run history, diagnostic schema, and JSON export.

Preserve these operator-facing interfaces unless a change is required by a documented ForgeCore contract migration.

## Current engineering rules

1. Treat `MemoryBudget` as the managed-memory authority. It does not control process RSS or the OS page cache.
2. Treat planner lower bounds as necessary estimates, not sufficient runtime guarantees.
3. Keep generic GGUF inspection/planning separate from executable architecture support.
4. Keep the one authoritative `ramforge_core::compute::AttentionConfig`/`attention_reference` API; do not introduce a conflicting attention API during migration without documenting the transition.
5. Keep matrix layout explicit: GGUF shape `[in, out]`, resident row-major `[out][in]`, `y[o] = dot(W[o], x)`.
6. Keep KV history and current K/V staging explicit; do not infer sequence length from allocation capacity.
7. Keep tokenizer IDs and generated token IDs distinct from decoded text.
8. Keep prefill and decode traces distinct; a sampled final token does not automatically require a final forward.
9. Keep external reference results in artifacts with model identity and command provenance.
10. Mark anything not observed as UNKNOWN rather than filling it with a likely value.

## First actions for the receiving engineer

### Before changing code

- Read all eight handoff files.
- Read `README.md` and `REMODEL_ENGINEERING_REPORT.md`; treat the latter's tool-availability statement as historical.
- Run `git status`/equivalent source-control inspection and confirm only intended documentation files are new or modified.
- Inspect Cargo manifests and preserve package versions/milestone naming.
- Check whether the exact Qwen2.5 model and any reference artifacts are available outside the repository; do not copy them into the source tree without an explicit data-handling decision.

### Establish the numerical evidence workspace

Create a separate, reproducible evidence directory outside production source containing:

- model filename, size, cryptographic hash, and provenance;
- GGUF metadata and tensor descriptor dump;
- tokenizer metadata dump;
- prompt/token ID record;
- raw Q6_K rows 59/220 and checksums;
- independent reference outputs;
- RAMforge/ForgeCore trace outputs;
- command lines, toolchain versions, thread count, and environment values.

The exact Qwen2.5 target is **Qwen2.5-1.5B-Instruct, Q4_0 GGUF**. Do not substitute a similarly named file.

### Run only the cheap baseline first

```text
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo build --workspace --release
```

The recorded baseline is 351 passed, 0 failed, 2 ignored. If a new checkout differs, record the difference before modifying numerical code.

### Reproduce the target trace only with artifacts

```text
CARGO_TARGET_DIR=/tmp/ramforge-cargo-target \
RAMFORGE_QWEN25_MODEL=/path/to/qwen2.5-1.5b-instruct-q4_0.gguf \
RAMFORGE_QWEN25_RAM_BYTES=8589934592 \
cargo test -p ramforge-runtime qwen25_bounded_numerical_trace -- \
    --ignored --nocapture
```

This command was **not run for this handoff**. It is expensive and requires the unavailable model. Compare the trace in boundary order; do not begin with generated text.

## Recommended ownership split

- **Infrastructure owner:** GGUF/data source, memory budget, layer reads/cache, planner/compiler/persistence, profile/diagnostic schema, CLI/TUI.
- **Numerical owner:** ForgeCore tensor contract, quantized decode, operators, transformer, KV semantics, output projection, and reference traces.
- **Validation owner:** independent oracle, model artifact/provenance, byte-level and boundary-level comparison, acceptance records.

No ownership assignment is implied by current Rust module ownership; the split is proposed for the rebuild.

## Transfer risks

| Risk | Why it matters | Required response |
|---|---|---|
| Missing model/reference artifacts | prevents rechecking the discrepancy | preserve UNKNOWN; obtain exact artifacts before claiming resolution |
| Reusing current quantized kernels too early | may carry a byte/layout defect into ForgeCore | pass raw-row and scalar parity gates first |
| Mixing streaming and mathematics | makes failures impossible to localize | keep storage/budget/cache adapters outside numerical reference code |
| Trusting synthetic tests alone | synthetic fixtures do not cover target metadata/quantization | require target boundary trace |
| Treating support registry as proof | registry says what is wired, not what is numerically validated | separate capability from validation status |
| Running a different Qwen2.5 variant | changes tokenizer, metadata, weights, and expected values | record exact hash/name and reject substitutions |
| Optimizing before reference parity | hides the first divergent operation | scalar/reference first, optimized parity second |
| Accidental version/milestone edits | violates transfer scope and confuses history | keep numbering unchanged |

## Definition of a successful transfer

The receiving team has successfully taken custody when it can answer, from files and artifacts rather than memory:

- what RAMforge currently does;
- which infrastructure is safe to preserve and why;
- which numerical assumptions are not trusted;
- how to reproduce the fixed prompt IDs;
- where the Qwen2.5 diagnostic lives and why it is ignored;
- what the exact layer-27 trace contains or why it remains UNKNOWN;
- how output.weight Q6_K rows are addressed and compared;
- which validation commands actually passed;
- which claims remain prohibited;
- what ForgeCore must prove before integration.

Until the external model/reference artifacts and layer-27 values are recovered, the handoff is complete as documentation but incomplete as numerical validation.
