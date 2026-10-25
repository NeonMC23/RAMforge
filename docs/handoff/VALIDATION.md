# Validation and evidence

> **Historical snapshot.** The results below predate the 2026-10-24 local
> verification recorded in [PROJECT_STATE.md](PROJECT_STATE.md). They remain
> useful evidence for the cited source state, but they are not the current test
> count or command recipe. Use [`../../scripts/test.sh`](../../scripts/test.sh)
> for the current repeatable invocation.

**As-of date:** 2026-09-22 for the historical repository results below; Phase-4
status is recorded in the final section dated 2026-10-02.  
**Purpose:** distinguish commands that actually ran from source-only claims and unavailable real-model evidence.

## Evidence labels

- **VERIFIED** — directly established by repository source or a command/result that actually ran against this checkout.
- **OBSERVED** — a recorded behavior/result or source observation, but not necessarily a complete acceptance proof.
- **UNKNOWN** — not available in the repository/workspace or not run; no value is inferred.
- **HYPOTHESIS** — a possible explanation awaiting an experiment.
- **PROPOSED** — future design, test, or ForgeCore boundary; not current behavior.

## Recorded repository validation

The following commands were run against the pre-Phase-4 workspace source with
Rust/Cargo `1.98.1`. These are **VERIFIED historical results** for that source
snapshot; the Phase-4 source was not revalidated because the toolchain is not
available in the current environment.

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

## What the historical workspace tests establish

**VERIFIED by the historical passing workspace suite:**

- GGUF parsing and metadata/type handling on test fixtures.
- Descriptor offsets, byte lengths, alignment, and bounded range validation.
- `GgufDataSource` logical/physical I/O accounting, cursor reuse, coalescing, and grouped-buffer behavior on fixtures.
- `MemoryBudget` allocation/resize/release and scoped temporary release on success/error.
- Cache charge/eviction/rollback behavior.
- F32/F16/BF16 representation and accounting rules on fixtures.
- Quantized block decoder and row/matvec behavior covered by the source tests.
- Explicit matrix layout and arity rejection.
- Scalar/reference compute functions and focused optimized/reference equivalence tests.
- RMSNorm, elementwise operations, SiLU/SwiGLU, add/mul, softmax, half-split RoPE, GQA mapping, and synthetic attention fixtures.
- Transformer execution order and Qwen2 bias handling on deterministic tiny GGUF fixtures.
- KV-cache layout, chunk growth, preservation, sequence-length invariants, and streaming layer behavior on fixtures.
- Generation reset/failure cleanup and generated-loop boundary behavior on synthetic fixtures.
- Planner, capability registry, calibration boundaries, plan compilation, persistence, CLI/TUI diagnostic schemas, and release/build integration.

These results establish source-level and synthetic behavior for the historical
snapshot. They do not establish that the current Phase-4 source compiles or
that the numerical path reproduces the external Qwen2.5-1.5B target.

## What the workspace tests do not establish

**UNKNOWN / not proven by the suite:**

- exact Qwen2.5-1.5B-Instruct Q4_0 model output;
- compatibility with llama.cpp or any other external implementation;
- exact prompt logits, layer-27 values, `result_norm` values, or generated token sequence for the target model;
- correctness of the current Q6_K output projection against an external target for rows 59 and 220;
- generated-text quality for the real model;
- real-model performance, RAM peak, physical I/O, cache hit rate, or token latency;
- correctness of model-specific metadata for a target GGUF not present in the workspace;
- GPU model execution or GPU offload.

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

`REMODEL_ENGINEERING_REPORT.md` is dated 2026-09-22 and contains historical wording that Rust tools and external Qwen evidence were unavailable when that report was authored. Its statement that validation commands were not run is **historical context**, not the latest pre-Phase-4 validation record.

The later recorded workspace results above are the authoritative historical validation snapshot for the pre-Phase-4 source. `PHASE_4_ENGINEERING_REPORT.md` is authoritative for the current Phase-4 source status and must not be silently treated as compiled/tested until the listed commands are rerun.

## Phase-4 validation status — 2026-10-02

**UNKNOWN / NOT RUN:** each attempted `cargo` command returned status 127
(`cargo: not found`); `rustc`, the historical `/var/tmp/forge-toolchain`
location, and the historical `/var/tmp/forge-native` location are also absent
from the current environment. No compatible local native build for verified
ForgeCore `v3.1.2` was found. Consequently, these commands could not run
against the Phase-4 source
or this dependency refresh:

```text
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo build --workspace --release
```

Also not run:

- `RAMFORGE_RUN_NATIVE_DEVICE_TESTS=1` optional device registry coverage;
- ForgeCore tiny-fixture CPU execution after the Phase-4 adapter integration;
- real GPU discovery or GPU execution;
- release binary execution or model run.

Source-level Phase-4 facts are documented in
[PHASE_4_ENGINEERING_REPORT.md](PHASE_4_ENGINEERING_REPORT.md):
`DeviceRegistry::discover` uses ForgeCore's safe `enumerate_devices`; CPU is
selectable; non-CPU model selection is rejected by RAMforge's deliberate
CPU-only adapter contract; current ForgeCore `v3.1.2` does expose public static
offload options, but this task does not enable them; unknown memory is
preserved; and selection is not residency proof. These are implementation
claims awaiting compiler/test confirmation, not substituted validation
results.

## ForgeCore revision-refresh validation status — 2026-10-11

The dependency/API refresh was applied to the final source tree, but the
requested validation could not execute on this host. Every `cargo` command
listed below returned status **127** with `/bin/bash: cargo: command not found`.
No result is marked passed. The only available Rustup binary reports no
installed toolchains and no active toolchain; `FORGE_LLAMA_DIR` is unset, and
no local native build or model fixture is present.

Commands attempted against the final tree:

```text
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
cargo test --workspace --release
cargo build --workspace --release
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo test --workspace --doc
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo test -p ramforge-runtime --test forgecore_adapter -- --nocapture
cargo test -p ramforge-runtime --test execution_devices -- --nocapture
RAMFORGE_RUN_NATIVE_DEVICE_TESTS=1 cargo test -p ramforge-runtime --test execution_devices -- --nocapture
```

The ForgeCore fixture-gated tests and optional native-device test therefore
remain **UNKNOWN / NOT RUN** in terms of Rust execution. Source inspection
verified that the adapter uses current non-exhaustive defaults, explicitly
selects `GpuLayers::Cpu`, observes ForgeCore's effective padded context length,
and preserves selection/error propagation without fallback.

## Validation commands for a future checkout

Fast, non-model validation:

```text
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo build --workspace --release
```

Optional Phase-4 native-registry coverage:

```text
RAMFORGE_RUN_NATIVE_DEVICE_TESTS=1 cargo test -p ramforge-runtime --test execution_devices -- --nocapture
```

Useful non-generation model inspection, when a model is intentionally supplied:

```text
cargo run -p ramforge-cli -- inspect MODEL.gguf
cargo run -p ramforge-cli -- plan MODEL.gguf --ram 8G
cargo run -p ramforge-cli -- support
```

A real run such as the following is **not** part of this handoff validation and should not be launched without an explicit model/budget decision:

```text
cargo run -p ramforge-cli -- run MODEL.gguf --ram 8G --backend forgecore --device cpu --prompt "Hello" --max-tokens 32 --profile --memory-report
```

## Proposed acceptance record for future native GPU support

Before claiming static non-CPU execution after a future safe ForgeCore API
extension, record:

1. source revision and Rust/Cargo/toolchain versions;
2. exact model path, file size, hash, GGUF version, and relevant metadata;
3. tokenizer metadata and fixed prompt IDs;
4. discovered ForgeCore device snapshots, backend/device selection syntax, and capability states;
5. the safe ForgeCore model/context binding and native offload configuration actually used;
6. host/device accounting-domain reports with unknown allocations left unknown;
7. CPU versus selected-device no-fallback behavior for unavailable and invalid selections;
8. decode/logits success and native execution error behavior;
9. optional real-device output/performance evidence, clearly separated from source tests;
10. external reference command, version, and output artifact if numerical compatibility is claimed.

Every missing value must remain UNKNOWN. A passing build or a discovered GPU
cannot fill an absent native execution or residency artifact.

## Current residency validation — 2026-10-25

Environment:

```text
FORGE_LLAMA_DIR=/home/neon/Documents/CODE LOCAL/forge/native
LD_LIBRARY_PATH=$FORGE_LLAMA_DIR/build/bin
FORGE_TEST_MODEL=/home/neon/Documents/CODE LOCAL/forge/models/SmolLM2-135M-Instruct-Q4_K_M/SmolLM2-135M-Instruct-Q4_K_M.gguf
```

Passed:

```text
cargo check --workspace
cargo test --workspace                  # 481 passed, 0 failed, 16 ignored
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p ramforge-cli --test enforce_ram \
  json_report_is_versioned_and_keeps_measurements_distinct -- --ignored --exact
cargo test -p ramforge-cli --test enforce_ram \
  reuse_aware_reports_profile_without_intra_decode_claims -- --ignored --exact
sh -n scripts/residency-bench.sh
```

The real-model CLI checks required `--enforce-ram`; the host supplied cgroup v2
delegation. The JSON report was parsed back as schema version 1 and checked for
ledger/reconciliation/timing fields. The reuse-aware check observed a complete
profile, `intra_decode_safe=false`, and no intra-decode release claim.

The benchmark harness ran all six policies with prefetch off/on, two repeats,
SmolLM2-135M Q4_K_M, context 256, target 128 MiB, four greedy tokens, without
cgroup enforcement: 24/24 runs exited 0. Results and caveats are recorded in
[BENCHMARKS.md](BENCHMARKS.md). The reuse-aware result required a 16,384-event
trace; at 4,096 events the four-token capture was truncated and the profile was
correctly rejected as incomplete.

Not run or not claimed:

- intra-decode eviction (blocked by the native lifetime contract);
- GPU execution or residency;
- reproducible cross-machine performance;
- exact KV-cache or graph-scratch byte accounting.
