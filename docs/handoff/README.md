# RAMforge engineering handoff

This directory is the transfer package for a new developer or agent arriving without the prior conversation. It documents the repository as it exists, separates verified facts from investigation records and proposals, and identifies the planned replacement numerical core as **ForgeCore**.

> **Latest audit (2026-11-25):** [AUDIT_2026-11-25.md](AUDIT_2026-11-25.md) is the current evidence-driven assessment of capacity, throughput, overflow behavior, forgeCORE/native capability gaps, and the prioritized engineering roadmap. Where older documents disagree with it, prefer the audit's measured findings and verified commit pins.

> **Execution-core milestone (2026-11-25):** [INFERENCE_CORE_OPTIMIZATION_2026-11-25.md](INFERENCE_CORE_OPTIMIZATION_2026-11-25.md) records the implemented DeepSeek2 support, KV dtype/batch/flash forwarding, CLI threading fix, combined static-offload plus dynamic-staging mode, benchmark session reuse, and per-run storage/fault instrumentation, with before/after measurements.

> **MoE routing milestone (2026-11-25):** [MOE_EXPERT_RESIDENCY_2026-11-25.md](MOE_EXPERT_RESIDENCY_2026-11-25.md) documents verified read-only CPU expert-routing observation, measured DeepSeek2 expert locality, per-layer expert accounting, cache/prefetch simulations, the exact expert-granularity blocker, and the native capability proposal.

> **Dense overflow milestone (2026-11-25):** [DENSE_OVERFLOW_2026-11-25.md](DENSE_OVERFLOW_2026-11-25.md) measures the 32B dense storage-bound baseline, adds per-token storage/fault and PSS/mapping instrumentation, introduces the `--residency auto` bounded staging policy, and reports +24.8% decode / 13-32% fewer reads along with the negative experiments.

> **Night performance research (2026-11-27):** [NIGHT_PERFORMANCE_2026-11-27.md](NIGHT_PERFORMANCE_2026-11-27.md) executes the outstanding 32B plan, raises the best measured 32B generation throughput from 0.554 to **0.7215 tok/s** (7B draft, K20, no-repack/no-prefetch) and 0.711 tok/s with a cache-stable 1.5B draft, validates GPT-NeoX and DeepSeek2 inference, adds CPU/per-position/resource observability, and documents the measured storage/cache ceiling that keeps the 2 tok/s target out of reach.

> **Adaptive inference and resource management (2026-11-26):** [ADAPTIVE_2026-11-26.md](ADAPTIVE_2026-11-26.md) integrates feature-level compatibility, conservative resource admission, bounded multi-objective calibration, environment-aware performance invalidation, and confirmed `adaptive-plan` / `adaptive-apply` execution plans with native readback and applied/verified field states.

> **Model-agnostic profiling foundation (2026-11-26):** [PROFILING_2026-11-26.md](PROFILING_2026-11-26.md) adds structured model/hardware/workload profiles, exact performance-profile identity and persistence, and an advisory `ramforge recommend` path validated across Qwen2, Llama, GPT-NeoX, and DeepSeek2 metadata.

> **Overnight 32B campaign (2026-11-26):** [OVERNIGHT_32B_CAMPAIGN_2026-11-26.md](OVERNIGHT_32B_CAMPAIGN_2026-11-26.md) integrates speculative decoding into the inference service and benchmark path, combines it with `--residency auto`, and measures 16-token generation improving from 0.2145 to **0.554 tok/s (2.58x)** with identical token IDs and storage reads falling from 132.6 GB to 53.3 GB.

> **Speculative decoding milestone (2026-11-25):** [SPECULATIVE_DECODING_2026-11-25.md](SPECULATIVE_DECODING_2026-11-25.md) proves multi-token target verification, validates 1.5B/7B Qwen draft vocabulary compatibility, implements greedy speculative decoding with exact KV rollback, and reports 1.69-1.97x on 8 tokens and 2.94x on a controlled 16-token run, with byte-identical output.

## Reading order

1. [PROJECT_STATE.md](PROJECT_STATE.md) — what exists, what is supported, and the current conclusion.
2. [ARCHITECTURE.md](ARCHITECTURE.md) — the actual source-tree architecture and its boundaries.
3. [CURRENT_INFERENCE_ENGINE.md](CURRENT_INFERENCE_ENGINE.md) — the current numerical and streaming implementation.
4. [PREDICTION.md](PREDICTION.md) — predictive staging, confidence, feedback, simulation, and limits.
5. [MEMORY_ACCOUNTING.md](MEMORY_ACCOUNTING.md) — ledger categories, provenance, reconciliation, policy profiles, and report semantics.
6. [GRAPH_SAFETY.md](GRAPH_SAFETY.md) — trace/manifest/profile guarantees and the intra-decode safety blocker.
7. [BENCHMARKS.md](BENCHMARKS.md) — reproducible residency benchmark commands, columns, baseline, and caveats.
8. [QWEN25_CORRECTNESS_INVESTIGATION.md](QWEN25_CORRECTNESS_INVESTIGATION.md) — the numerical-divergence evidence and its limits.
9. [REBUILD_PLAN.md](REBUILD_PLAN.md) — the proposed ForgeCore replacement plan.
10. [VALIDATION.md](VALIDATION.md) — commands, results, diagnostic entry points, and unavailable evidence.
11. [PHASE_2_ENGINEERING_REPORT.md](PHASE_2_ENGINEERING_REPORT.md) — the implemented pinned ForgeCore adapter and historical validation record.
12. [PHASE_4_ENGINEERING_REPORT.md](PHASE_4_ENGINEERING_REPORT.md) — static device discovery, explicit selection, adapter integration, and current validation status.
13. [REPOSITORY_TRANSFER.md](REPOSITORY_TRANSFER.md) — how to start a new repository/session without losing the useful infrastructure or historical evidence.

## Evidence labels

- **VERIFIED** — directly supported by current source or a command actually run against this checkout.
- **OBSERVED** — an investigation result recorded in the handoff or task history, but not reproducible from files currently present in the sandbox.
- **UNKNOWN** — not present in the repository or available recorded evidence; no value is inferred.
- **HYPOTHESIS** — a possible explanation that has not been established.
- **PROPOSED** — future architecture or work, not an implemented RAMforge component.

## Current conclusion

The legacy RAMforge numerical engine is **not considered numerically trustworthy for Qwen2.5-1.5B** based on the collected evidence. The infrastructure around it remains valuable. The explicit `--backend forgecore` path delegates numerical execution to the pinned llama.cpp/ggml implementation through forgeCORE; it is the current validated execution path, while the legacy path remains available for comparison. GPU execution and intra-decode page eviction remain unimplemented.

Phase 2 has a real RAMforge adapter in `crates/ramforge-runtime/src/forgecore.rs`; the current ForgeCore pin is `457c4f75193b004ed01c1cc0c8e620a4a7683196`. Phase 4 adds safe ForgeCore device discovery, explicit static selection, capability/accounting metadata, and `run --device`; CPU remains selectable while this RAMforge adapter deliberately rejects non-CPU model binding without fallback. The pinned native llama.cpp/ggml source remains `7fe450e19305b828c199d602c23a8337aaa1f03b`, with residency behavior supplied by the tracked patch rather than a native upgrade. The adapter owns the safe `Model → Context → Batch → decode → Logits` boundary; the legacy `InferenceEngine` and its numerical modules remain available as the default comparison path.

The current milestone adds a native execution-quiescence contract, model/context lifetime identities, a non-blocking release gate, an explicit resource-lifetime state machine, validated persisted execution profiles, and typed `Busy`/`Unsupported` release errors. Page release is still restricted to the post-operation safe boundary; intra-decode eviction remains unsupported because synchronized completion does not prove future reuse. See [MEMORY_ACCOUNTING.md](MEMORY_ACCOUNTING.md) and [GRAPH_SAFETY.md](GRAPH_SAFETY.md).

The real-model Qwen2.5 diagnostic is intentionally ignored and was not run during the current documentation handoff. Exact external llama.cpp statistics that are not present in the repository are marked **UNKNOWN** rather than reconstructed or fabricated.
