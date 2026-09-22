# RAMforge engineering handoff

This directory is the transfer package for a new developer or agent arriving without the prior conversation. It documents the repository as it exists, separates verified facts from investigation records and proposals, and identifies the planned replacement numerical core as **ForgeCore**.

## Reading order

1. [PROJECT_STATE.md](PROJECT_STATE.md) — what exists, what is supported, and the current conclusion.
2. [ARCHITECTURE.md](ARCHITECTURE.md) — the actual source-tree architecture and its boundaries.
3. [CURRENT_INFERENCE_ENGINE.md](CURRENT_INFERENCE_ENGINE.md) — the current numerical and streaming implementation.
4. [QWEN25_CORRECTNESS_INVESTIGATION.md](QWEN25_CORRECTNESS_INVESTIGATION.md) — the numerical-divergence evidence and its limits.
5. [REBUILD_PLAN.md](REBUILD_PLAN.md) — the proposed ForgeCore replacement plan.
6. [VALIDATION.md](VALIDATION.md) — commands, results, diagnostic entry points, and unavailable evidence.
7. [REPOSITORY_TRANSFER.md](REPOSITORY_TRANSFER.md) — how to start a new repository/session without losing the useful infrastructure or historical evidence.

## Evidence labels

- **VERIFIED** — directly supported by current source or a command actually run against this checkout.
- **OBSERVED** — an investigation result recorded in the handoff or task history, but not reproducible from files currently present in the sandbox.
- **UNKNOWN** — not present in the repository or available recorded evidence; no value is inferred.
- **HYPOTHESIS** — a possible explanation that has not been established.
- **PROPOSED** — future architecture or work, not an implemented RAMforge component.

## Current conclusion

The current RAMforge inference engine is **not considered numerically trustworthy for Qwen2.5-1.5B** based on the collected evidence. The infrastructure around it remains valuable. The next project should rebuild the numerical core as **ForgeCore** while preserving the validated infrastructure where appropriate.

ForgeCore is a proposed name and architecture. There is no `ForgeCore` implementation in this repository.

The real-model Qwen2.5 diagnostic is intentionally ignored and was not run during the current documentation handoff. Exact external llama.cpp statistics that are not present in the repository are marked **UNKNOWN** rather than reconstructed or fabricated.
