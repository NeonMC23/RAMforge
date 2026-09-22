# Validation and evidence

**As-of date:** 2026-09-22.  
**Purpose:** distinguish commands that actually ran from source-only claims and unavailable real-model evidence.

## Evidence labels

- **VERIFIED** — directly established by repository source or a command/result that actually ran against this checkout.
- **OBSERVED** — a recorded behavior/result or source observation, but not necessarily a complete acceptance proof.
- **UNKNOWN** — not available in the repository/workspace or not run; no value is inferred.
- **HYPOTHESIS** — a possible explanation awaiting an experiment.
- **PROPOSED** — future design, test, or ForgeCore boundary; not current behavior.

## Recorded repository validation

The following commands were run against the current workspace with Rust/Cargo `1.98.1`. These are **VERIFIED recorded results** for the source snapshot used by this handoff; this documentation pass did not rerun them.

| Command | Result |
|---|---|
| `cargo fmt --all -- --check` | passed |
| `cargo test --workspace` | passed: **351 passed, 0 failed, 2 ignored** |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | passed |
| `cargo build --workspace --release` | passed |

The workspace test count was recorded as:

| Crate | Tests passed |
|---|---:|
| `ramforge-cli` | 44 |
| `ramforge-core` | 112 |
| `ramforge-runtime` | 195 |
| **Total** | **351** |

The two ignored tests are the real-model Qwen2.5 bounded diagnostic and the quantized row-decode microbenchmark. An ignored test is not a pass, and source code for a diagnostic is not a real-model result.

## What the workspace tests establish

**VERIFIED by the passing workspace suite:**

- GGUF parsing and metadata/type handling on test fixtures.
- Descriptor offsets, byte lengths, alignment, and bounded range validation.
- `GgufDataSource` logical/physical I/O accounting, cursor reuse, coalescing, and grouped-buffer behavior on fixtures.
- `MemoryBudget` allocation/resize/release and scoped temporary release on success/error.
- Cache charge/eviction/rollback behavior.
- F32/F16/BF16 representation and accounting rules on fixtures.
- Quantized block decoder and row/matvec behavior covered by the source tests.
- Explicit matrix layout and arity rejection.
- Scalar/reference compute functions and focused optimized/reference equivalence tests.
- RMSNorm, elementwise operations, SiLU/SwiGLU, softmax, half-split RoPE, GQA mapping, and synthetic attention fixtures.
- Transformer execution order and Qwen2 bias handling on deterministic tiny GGUF fixtures.
- KV-cache layout, chunk growth, preservation, sequence-length invariants, and streaming layer behavior on fixtures.
- Generation reset/failure cleanup and generated-loop boundary behavior on synthetic fixtures.
- Planner, capability registry, calibration boundaries, plan compilation, persistence, CLI/TUI diagnostic schemas, and release/build integration.

These results establish source-level and synthetic behavior. They do **not** establish that the current numerical path reproduces the external Qwen2.5-1.5B target.

## What the workspace tests do not establish

**UNKNOWN / not proven by the suite:**

- exact Qwen2.5-1.5B-Instruct Q4_0 model output;
- compatibility with llama.cpp or any other external implementation;
- exact prompt logits, layer-27 values, `result_norm` values, or generated token sequence for the target model;
- correctness of the current Q6_K output projection against an external target for rows 59 and 220;
- generated-text quality for the real model;
- real-model performance, RAM peak, physical I/O, cache hit rate, or token latency;
- correctness of model-specific metadata for a target GGUF not present in the workspace;
- GPU behavior (no GPU execution implementation exists).

## Qwen2.5 diagnostic status

Source path:

```text
crates/ramforge-runtime/src/inference.rs
```

Test:

```text
qwen25_bounded_diagnostic::qwen25_bounded_numerical_trace
```

Required inputs:

```text
RAMFORGE_QWEN25_MODEL=/path/to/qwen2.5-1.5b-instruct-q4_0.gguf
RAMFORGE_QWEN25_RAM_BYTES=8589934592
```

Command:

```text
CARGO_TARGET_DIR=/tmp/ramforge-cargo-target \
RAMFORGE_QWEN25_MODEL=/path/to/qwen2.5-1.5b-instruct-q4_0.gguf \
RAMFORGE_QWEN25_RAM_BYTES=8589934592 \
cargo test -p ramforge-runtime qwen25_bounded_numerical_trace -- \
    --ignored --nocapture
```

**Status for this handoff: NOT RUN.** The exact model payload and external llama.cpp checkout are unavailable, and the task explicitly forbids expensive real-model generation. The command is preserved for a future operator with the required artifacts.

The diagnostic's fixed inputs are:

```text
model: Qwen2.5-1.5B-Instruct, Q4_0 GGUF
prompt: hi, what's 2+2=?
prompt IDs: [6023, 11, 1128, 594, 220, 17, 10, 17, 19884]
```

It expects 28 layers, hidden size 1536, vocabulary size 151936, and checks `output.weight` as Q6_K `[1536, 151936]` with 1260 bytes per row. It compares rows 59 and 220 to embedded reference literals `20.24774933` and `19.75674248`. The completed RAMforge values and full layer-27 trace are **UNKNOWN** in this workspace.

## External reference status

No external llama.cpp checkout, model file, generated trace, or reference-logit artifact is present. Therefore:

- llama.cpp is documented only as a future **external validation oracle**;
- no compatibility claim is made;
- no final-logit statistics are reconstructed;
- no result norm, layer checkpoint, or generated text is fabricated;
- the row-reference literals present in source are preserved as source evidence, not represented as a completed comparison result.

The current conclusion that Qwen2.5 numerical inference is not trustworthy is based on the recorded investigation evidence and the unresolved numerical discrepancy, not on a new real-model run in this documentation pass.

## Historical report reconciliation

`REMODEL_ENGINEERING_REPORT.md` is dated 2026-09-22 and contains historical wording that Rust tools and external Qwen evidence were unavailable when that report was authored. Its statement that validation commands were not run is **historical context**, not the latest validation record.

The later recorded workspace results in this document are the authoritative validation snapshot for the current source. The historical report remains authoritative for its description of the compute-boundary remodel and its explicit non-compatibility limitation; it must not be silently treated as the latest test report.

## Validation commands for a future checkout

Fast, non-model validation:

```text
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo build --workspace --release
```

Useful non-generation model inspection, when a model is intentionally supplied:

```text
cargo run -p ramforge-cli -- inspect MODEL.gguf
cargo run -p ramforge-cli -- plan MODEL.gguf --ram 8G
cargo run -p ramforge-cli -- support
```

A real run such as the following is **not** part of this handoff validation and should not be launched without an explicit model/budget decision:

```text
cargo run -p ramforge-cli -- run MODEL.gguf --ram 8G --prompt "Hello" --max-tokens 32 --profile --memory-report
```

## Proposed acceptance record for ForgeCore

Before declaring the replacement numerical core validated, record:

1. source revision and Rust/Cargo/toolchain versions;
2. exact model path, file size, hash, GGUF version, and relevant metadata;
3. tokenizer metadata and fixed prompt IDs;
4. descriptor geometry/offsets for all comparison tensors;
5. raw Q6_K row checksums and block decode values;
6. layer 0 through layer 27 checkpoints;
7. final `result_norm`, prompt logits, first token, and decode boundaries;
8. scalar/reference/optimized parity results;
9. out-of-core memory/I/O/profile report;
10. external reference command, version, and output artifact.

The acceptance record must mark every missing value UNKNOWN. A passing build or passing synthetic test cannot fill an absent real-model artifact.
