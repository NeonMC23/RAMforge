# RAMforge engineering handoff

This directory is the transfer package for a new developer or agent arriving without the prior conversation. It documents the repository as it exists, separates verified facts from investigation records and proposals, and identifies the planned replacement numerical core as **ForgeCore**.

## Reading order

1. [PROJECT_STATE.md](PROJECT_STATE.md) — what exists, what is supported, and the current conclusion.
2. [ARCHITECTURE.md](ARCHITECTURE.md) — the actual source-tree architecture and its boundaries.
3. [CURRENT_INFERENCE_ENGINE.md](CURRENT_INFERENCE_ENGINE.md) — the current numerical and streaming implementation.
4. [MEMORY_ACCOUNTING.md](MEMORY_ACCOUNTING.md) — ledger categories, provenance, reconciliation, policy profiles, and report semantics.
5. [GRAPH_SAFETY.md](GRAPH_SAFETY.md) — trace/manifest/profile guarantees and the intra-decode safety blocker.
6. [BENCHMARKS.md](BENCHMARKS.md) — reproducible residency benchmark commands, columns, baseline, and caveats.
7. [QWEN25_CORRECTNESS_INVESTIGATION.md](QWEN25_CORRECTNESS_INVESTIGATION.md) — the numerical-divergence evidence and its limits.
8. [REBUILD_PLAN.md](REBUILD_PLAN.md) — the proposed ForgeCore replacement plan.
9. [VALIDATION.md](VALIDATION.md) — commands, results, diagnostic entry points, and unavailable evidence.
10. [PHASE_2_ENGINEERING_REPORT.md](PHASE_2_ENGINEERING_REPORT.md) — the implemented pinned ForgeCore adapter and historical validation record.
11. [PHASE_4_ENGINEERING_REPORT.md](PHASE_4_ENGINEERING_REPORT.md) — static device discovery, explicit selection, adapter integration, and current validation status.
12. [REPOSITORY_TRANSFER.md](REPOSITORY_TRANSFER.md) — how to start a new repository/session without losing the useful infrastructure or historical evidence.

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
