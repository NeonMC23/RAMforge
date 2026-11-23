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
cargo test --workspace                  # 493 passed, 0 failed, 17 ignored
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

## Predictive residency validation — 2026-10-30

- `cargo test --workspace`: 632 passed, 0 failed, 45 ignored.
- `cargo clippy --workspace --all-targets -- -D warnings`: passed.
- `ramforge simulate` deterministic output and budget safety: passed.
- Real prediction run with saved profile: 64 predictions, 16,650,240 B
  requested/accepted, mapping delta 0 B; feedback saved atomically.
- Warm-history run: 64 high-confidence predictions, 17,375,232 B
  requested/accepted, mapping delta 0 B.
- Prediction JSON report and profile/history identity validation: passed.
- Prediction never sets `release_authorized=true`; `FinalUseProven` remains
  unreachable for model weights.
- Cache-preparation helper tests: `mincore` probe counts, missing-source failure,
  copy/preparation failure invalidating cold classification, warm versus
  controlled distinction, source-file immutability, and profile-identity
  rewriting for an identical copy.
- `scripts/prediction-bench.py` five-arm/four-workload matrix:
  - 100/100 no-pressure runs succeeded, 5 repetitions;
  - 100/100 256 MiB cgroup runs succeeded, 5 repetitions;
  - cache state uncontrolled (`cache_controlled=false`);
  - mapping delta 0 B in all arms; no RSS reduction;
  - 256 MiB predictive medians: major faults 4 vs baseline 5 (alternate),
    2 vs 3 (short workloads), and TTFT lower on alternate/short workloads;
  - long-workload prediction was neutral/slightly worse. This is an
    association, not a proven causal speedup.
- Metric calibration: 64 predicted, 64 matched observed, 63 within horizon,
  precision 100%, recall 23.5%, 57 prefetch-followed, 57 benefit-unknown,
  7 duplicate/overlap avoided. Correct prediction is not useful prefetch.
- `mincore` observation tests cover maps parsing, page alignment, zero ranges,
  targeted-first ordering, remaining-budget general sampling, overlap
  deduplication, partial selected resources, missing selected ranges, zero
  budget, unreadable/missing mappings, unknown pages, and shared/overlap
  correlation.
- Earlier warm observation: 25,581 pages, all resident before staging; 57
  selected resources all redundant; observation 53.8 ms/snapshot (158.5 ms
  total). With a 2,048-page targeted budget: 2,048/4,065 selected pages
  sampled, 33 selected resources fully observed, 4 partial, 19 not observed.
  With 64 targeted pages: 2 fully observed, 3 partial, 51 not observed. These
  partial results are not evidence for unobserved resources.
- Targeted observation uses interval arithmetic for the available population
  instead of allocating a key per page. Three 64-page snapshots cost about
  1.6–3.8 ms for SmolLM2 and 3.4–4.0 ms for Qwen2.5-1.5B in the final runs.
  It remains diagnostic-only and disabled by default.
- End-to-end timing includes setup/prediction/staging/inference; TTFT excludes
  staging. Five-repetition pressure matrix: baseline vs predictive end-to-end
  short-short 285.09 vs 294.98 ms, short-long 447.93 vs 469.54 ms,
  long-short 911.15 vs 921.13 ms. No predictive speedup is claimed.
- Deterministic simulator: 16 scenarios x 5 strategies. `useful` baseline
  10,217 ns vs predictive 2,675 ns, net +7,542 ns; observation-overhead net
  -992,458 ns; already-resident is redundant and unknown outcomes set
  `benefit_known=false`. Simulator results now expose targeted/general page
  coverage, partial selected resources, before-access availability, and cache
  condition labels. All simulated budget violations are zero.
- Controlled file-cache benchmark (SmolLM2, five repetitions, 64 targeted
  pages, 100/100 controlled-nonresident and 100/100 warm runs succeeded).
  Controlled predictive end-to-end medians were 343.3, 475.7, 922.5, and
  378.9 ms versus baseline 323.9, 491.4, 907.7, and 363.5 ms for
  short-short/short-long/long-short/alternate: one workload improved, three
  worsened. Conservative was 318.1, 470.1, 933.8, and 377.4 ms; aggressive
  was 319.7, 492.9, 926.4, and 370.1 ms. Predictive inference and TTFT were
  lower on all four, but end-to-end cost was not consistently better. Warm
  predictive end-to-end was worse on all four. No speedup is claimed.
  Mapping delta was 0 B; controlled major faults were 3/3/5/5 baseline versus
  3/3/5/5 predictive and 2/2/4/4 aggressive.
- Qwen2.5-1.5B Q4_0: 100/100 controlled-nonresident runs succeeded, five
  repetitions per arm/workload. Predictive end-to-end medians were 1349.6,
  2323.4, 6951.4, and 1988.1 ms versus baseline 1371.3, 2213.1, 6987.3, and
  1996.2 ms for short-short/short-long/long-short/alternate. It was faster on
  three workloads and slower on short-long, with wide variability; this is not
  a general speedup. Mapping delta was 0 B.')

## Operational runtime validation — 2026-11-21

- `cargo test --workspace --locked`: **632 passed, 0 failed, 45 ignored**.
- Real inference-service tests: load, context, stream, throughput, cancellation
  and unload passed.
- CLI benchmark: median generation 136.87 tok/s (112.58–155.41), decode 171.70,
  TTFT 11.40 ms, prompt 423.31, context 5.52 ms, load 234.91 ms (debug,
  warm/uncontrolled).
- CLI serve: status/token/done JSON-lines events passed.
- CLI optimize: measured 121.98/137.87/103.34 tok/s for maximum/balanced/
  background; advisory balanced recommendation.
- CLI storage: 105,454,432-byte verified copy with source preserved.
- CUDA build validated: pin unchanged, `native/build-cuda`, SM 86 cubins,
  `ggml_backend_cuda_reg` present; CUDA and CPU backends registered.
- CUDA0 enumerated as RTX 3050 Laptop GPU, compute capability 8.6, 4 GiB VRAM.
- Physical GPU/hybrid inference validated with SmolLM2-135M Q4_K_M: effective
  layers 8 (7 transformer + output) and 31 (30 transformer + output), output
  device GPU, real tokens generated. Cancellation, repeated sessions, unload,
  and VRAM release validated.
- Release comparison: first session CPU 240.70 tok/s, hybrid 289.13, full GPU
  484.65; reverse order CPU 249.62, hybrid 309.53, GPU 488.89; instrumented
  session CPU 252.53, hybrid 317.56, GPU 488.09.
- GPU observability schema 3 validated: native CUDA0/CPU_Mapped/CPU buffer
  model/KV/compute totals, device used, sampling time/sequence; unsupported
  counters remain explicit.
- Allocation/deallocation counters, transfer accounting, peak VRAM, process
  attribution, and synchronization duration were not available and are marked
  unsupported.
- Repeated GPU load/unload cycles released buffers and returned to baseline.
- Larger-model hybrid: Qwen2.5-7B Q4_K_M, 12 effective layers, 9.30 tok/s,
  coherent output, 2.90 s load.
- Read-only tensor storage introspection validated physically: 273 tensors,
  2 model buffers, layer 0 = 2,596,608 bytes; file-backed and CUDA0 metadata
  agree with placement and breakdown.
- Staging primitives validated physically: bounded buffers, private-stream
  H2D/D2H, event completion, counters, release/double-release/busy checks,
  allocation failure, and cancellation refusal during active copy.
- Single-layer prototype: 2,596,608 bytes copied and read back byte-identically;
  model remained usable. Inference did not consume the staged copy.
- Graph-integrated staged inference validated: SmolLM2 layer 0, 9 tensors,
  2,596,608 B, 9 substitutions per graph, equal argmax, cosine 0.9933,
  matching generated text; corruption control changed logits. Qwen2.5-7B
  layer 0: 12 tensors, 149,112,832 B, 24 substitutions, 5.84 tok/s.
- Multi-layer staged inference validated: SmolLM2 layers 0+1, 5,197,928 B,
  18 substitutions per layer, matching text; explicit 1M budget rejected before
  staging; auto 10M accepted layers 0-3. Qwen2.5-7B auto 1G staged 7 layers
  (989,857,792 B, 4.40 tok/s).
- Native lifecycle validated: eviction during/after execution is refused until
  `llama_context_invalidate_graph()` quiesces backends, resets cached graph
  results and clears scheduler mappings. Duplicate invalidation is idempotent;
  duplicate eviction is `not_staged`.
- Dynamic same-context residency validated: SmolLM2 replaced layers 0+1 with
  1+2 for five cycles (5 invalidations, 5 releases, 7 allocations, 0 reuse),
  median total 3.188 ms (P25 3.063, P75 3.272, P95 3.491). Corrupting the
  replacement layer changed logits by 10.25 with 9 substitutions.
- Qwen2.5-7B replaced layers 0-6 (989,857,792 B) with 7-13 (953,903,104 B) and
  back in one context (~2.0 s each) with coherent output.
- Injected stage failure restored the old selection; injected rollback failure
  moved the service to `failed` and rejected generation.
- Bounded allocation reuse validated: pooled evicted buffers are reused; native
  `reuse_count` advances and allocation count stabilizes (SmolLM2 pipeline 4
  allocations across five cycles; replacement 3 allocations with reuse 1,1,2,3).
- Pending double buffering validated: active inference executes while a complete
  pending selection is allocated and filled; commit requires invalidation.
- Async batch transfers validated with CUDA start/completion events and a
  low-priority staging stream.
- Measured overlap: SmolLM2 transfer 0.94–1.31 ms hidden in 70–78 ms compute;
  Qwen 7B transfer 85–109 ms hidden in 400–535 ms compute.
- Corruption control on a committed pending layer changed logits by 10.25 with
  9 substitutions.
- Hardware calibration validated physically: pinned 6.3–7.1 GB/s, pageable
  4.7–5.7 GB/s, 64 KiB latency 8–21 µs, chunking medians within 2.5% (no
  material benefit), identity-bound and versioned JSON.
- Adaptive policy tests: missing/stale calibration fallback, profile constraints,
  interleaved exploration, measured sync/async selection, hysteresis and user
  override.
- Physical adaptive runs: Qwen Balanced chose synchronous (async within 5%),
  Maximum Performance chose synchronous after measurement, identity mismatch
  and Background profile forced synchronous.
- Smallest-first residency: 8 layers/1,049,083,904 B vs 7/989,857,792 B;
  6.418 vs 6.161 tok/s median over five runs; TTFT 212.5 vs 221.5 ms.
- Host staging validated physically: direct-to-pinned read, coalesced GGUF
  ranges, bounded pinned pool, mapping cache and async worker reads.
- Qwen host preparation: 276.1 ms copy → 178.4 ms direct coalesced; read
  137.3 → 69.7 ms; 48 tensors → 2 groups.
- End-to-end Qwen five-cycle order-controlled medians: 728.4/758.2 ms sync vs
  691.4/687.2 ms async host-read (~7% lower).
- Direct-read bytes matched model reads for two split GGUF shard files; layers
  20–23 generated coherent text.
- Bounded parallel storage reads validated: byte-exact output and outstanding-
  byte admission limit tested; coalesced c=1 remained best (59.9 ms vs 61.0/83.5
  at c=2/4); per-tensor c=4 improved to 59.9 ms but did not beat coalesced.
- Predictive prefetch accounting validated: three SmolLM2 cycles committed
  15,040,512 B with zero waste; abort + commit produced 500 permille hit rate and
  non-zero wasted bytes.
- io_uring rejected with evidence: runtime libs/sysctl present, headers absent,
  coalesced pread already saturates warm-cache workload.
- Partial residency validated physically: shared active allocation device
  pointer survives delta commit; only new layer is staged; removed layer is
  released; inference consumes the new layer.
- Qwen overlapping transition: read/transfer 280.2/298.2 MB versus 560.5/578.5
  full, preserved 280.2 MB, transition 42.1–44.2 ms versus 82.5–87.6 ms,
  allocations 6 versus 8.
- Removal-only delta commits zero transfer bytes. Injected pending failure and
  abort preserve the active set; concurrent prepare is rejected.
- Decision: PARTIAL RESIDENCY VALIDATED — SHARED LAYERS PRESERVED, ONLY DELTAS
  READ AND TRANSFERRED, TRANSITION LATENCY ROUGHLY HALVED.
- Native effective placement readback is wired through forgeCORE and RAMforge;
  zero-effective GPU loads are rejected. No silent CPU fallback.
- `RuntimePlan` schema is 2; benchmark schema is 14; service API 6; staging
  plan schema 7; calibration schema 1; GPU observability schema 3.
- Release-build throughput (latest session): balanced 8 threads 223.04 tok/s
  (P25–P75 218.94–224.91), maximum-performance 4 threads repacked 205.25,
  balanced 4 threads 203.92, background 81.44. Earlier sessions measured
  283.81/267.65 and 269.05/252.02 for balanced 8/4; a final spot check measured
  219.91/229.50. Absolute rates vary by roughly 25% and the 4/8 ranking
  reverses, so no reliable winner between 4 and 8 threads is claimed.
- Out-of-core re-verification (2026-11-21): GPT-NeoX-20B Q4_K_M (13.13 GB)
  under an 8 GiB cgroup with `--no-repack --no-prefetch` produced one greedy
  token (` Paris`), exit 0, 10.877 s wall, 184 ms model load, peak charged
  exactly 8 GiB, 9661 `max` events, 0 OOM kills, and 216,464 page faults (257
  major). This demonstrates fit and correctness under reclaim, not throughput.
- Page-scoped release/prefetch/staging is implemented and separately validated;
  it is OS paging over existing mappings, not tensor/layer streaming.
- CPU, CUDA, and hybrid execution were exercised on one physical machine.
  Cold-cache, transfer, utilization, MoE, and tensor/layer streaming claims
  remain unavailable.

## Runtime calibration validation — 2026-11-20

- `cargo test --workspace --locked`: **632 passed, 0 failed, 45 ignored**.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: passed.
- Ignored calibration CLI test: passed; nine lifecycle stages, mapping
  residency, comparisons, attribution, model fingerprint, and plan evidence.
- `ramforge calibrate` on SmolLM2-135M Q4_K_M: model-load RSS +126,545,920;
  context +6,586,368; prompt +8,364,032; decode 0; unload -105,639,936.
- `ramforge machine --calibration` applies evidence only for matching
  model/context/batch/backend.
- Lifecycle benchmark: 9 snapshots 2.3–3.4 ms; calibration build 0.02 ms;
  serialization round-trip 1.1–1.5 ms.
- Compatibility tests cover exact, constrained, incompatible, incomparable,
  stale, missing, and insufficient evidence; model/machine/backend/device/
  load-mode/repacking/context/batch/micro-batch/sequence/thread mismatches;
  incompatible evidence is verified not to change plan estimates.
- Only one physical CPU-only machine was exercised. KV, compute-buffer,
  graph/activation, backend, and staging allocations remain unobservable.

## Machine profile and candidate plan validation — 2026-11-20

- `cargo test --workspace`: **632 passed, 0 failed, 45 ignored**.
- `cargo clippy --workspace --all-targets -- -D warnings`: passed.
- `ramforge machine --model ... --ram 1G --json`: passed; includes CPU,
  memory, storage, device, model requirements, four plans, and comparison.
- `ramforge machine --model ... --ram 1M --json`: resident performance plan is
  infeasible; file-backed plans are not reported as definitely feasible.
- Ignored CLI machine tests: 2 passed.
- `scripts/resource-profile-bench.sh` with 5 runs: collection median 1.48–2.39
  ms, serialization round-trip 0.32–0.50 ms, requirements 0.02–0.03 ms,
  plans/comparison 0.06–0.07 ms.
- Synthetic tests cover unknown-vs-zero, observed invariants, missing metadata,
  overflow, missing GPU, context limits, budget failures, redaction, and
  deterministic serialization.
- Only one physical machine was exercised. Multi-machine validation remains
  outstanding.

## CPU node-dispatch validation — 2026-10-31

- forgeCORE `cargo test --workspace`: **423 passed, 0 failed, 21 ignored**.
  New `node_dispatch` tests cover disabled-by-default behavior, installation
  and removal, event drain, overflow, multiple contexts, four-worker
  exactly-once publication, identity fields, and bounded options.
- RAMforge `cargo test --workspace`: **632 passed, 0 failed, 45 ignored**.
  The ignored real-model CLI test verifies `--node-dispatch-observe` JSON with
  attributed and unattributed events, zero identity mismatches, CPU backend,
  and bounded serialization.
- Native CPU hook: the callback is invoked by worker 0 immediately before
  `ggml_compute_forward` for each non-empty, compute-enabled node. An
  observer-only barrier follows publication while active; the disabled path
  adds neither callback nor barrier. Four-worker test observed one event per
  active node and strictly increasing event sequence.
- Observer overhead smoke (SmolLM2, five measured decodes per setting):
  one thread off 15.98 ms vs on 17.54–18.26 ms; four threads off 6.42 ms vs
  on 6.83–6.97 ms. The spread is noisy; no speedup is claimed. Events were
  bounded, with no drops in the measured settings.
- Pre-dispatch sampling smoke: one thread 17.22 ms disabled vs 17.61 ms at
  64 pages/range/4,096-page execution budget; four threads 6.54 vs 6.66 ms.
  A second run measured 18.53/17.67 ms and 6.45/8.14 ms. Strategy ranges across
  four runs: systematic 17.6–19.4 ms, targeted 17.6–18.1 ms with 252 skips,
  adaptive 17.7–19.7 ms with 252 reuses. Differences are dominated by noise.
  Sampling is diagnostic-only and disabled by default.
- Mapping metadata tests check a nonzero model-instance generation, mapping id,
  file path/size, exact page bounds, and bounded/partial native sampling.
- Pre-dispatch sampling is implemented for validated file-backed tensor
  ranges and remains opt-in (`--node-dispatch-sample-pages`, default 0).
  Source-name correlation is exact-unique only; unknown names remain
  unattributed. Dispatch and residency observations never authorize release
  or eviction.

## Native lifetime validation — 2026-10-30

ForgeCORE pin: `457c4f75193b004ed01c1cc0c8e620a4a7683196`.

Native tests executed with the pinned tree and `FORGE_TEST_MODEL` pointing at
SmolLM2-135M Q4_K_M:

- `release_is_busy_until_native_synchronization_then_safe`
- `two_contexts_must_both_be_synchronized_before_release`
- `context_drop_synchronizes_and_releases_native_count`
- `model_instance_identity_is_process_local_and_distinct`

They prove the current contract: release is rejected with `ResidencyError::Busy`
until all contexts synchronize or drop; a context drop synchronizes; model
instance IDs are distinct; page release after synchronization leaves inference
valid.

RAMforge checks:

```text
cargo check --workspace                         passed
cargo test --workspace                          632 passed, 0 failed, 45 ignored
cargo clippy --workspace --all-targets -D warnings  passed
```

Real-model checks passed individually:

- versioned JSON report with native lifetime fields;
- profile save then `ramforge profile --model` identity validation;
- reuse-aware inference with a loaded persisted profile;
- memory-bound post-operation release with native quiescence and state-machine
  reconciliation.

Not validated/claimed:

- intra-decode release;
- context-owned KV/scratch page release;
- GPU execution/residency;
- cross-machine performance or cgroup attribution.
