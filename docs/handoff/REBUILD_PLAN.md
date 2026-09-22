# ForgeCore rebuild plan

**Status:** PROPOSED.  
**Scope:** numerical compute/inference core only, unless an interface change is required at a documented seam.  
**Current repository:** RAMforge remains unchanged by this handoff.

ForgeCore is the proposed name for the replacement numerical core. No ForgeCore implementation exists in the repository. This plan does not authorize production-code edits, dependency additions, version changes, or removal of current RAMforge code.

## Objective

Build a small, independently testable numerical engine that can consume RAMforge's validated model/storage abstractions and produce an auditable Qwen2.5 result before it is optimized or integrated into the out-of-core runtime.

The objective is **not** to redesign RAMforge's planner, cache, persistence, UI, or storage system. The objective is to replace the untrusted numerical path and establish an evidence chain from raw GGUF bytes to prompt logits, decode boundaries, and generated text.

## Non-goals and guardrails

- Do not claim llama.cpp compatibility until an independently reproducible comparison passes.
- Do not use generated-text similarity as the first correctness gate.
- Do not optimize before scalar/reference semantics pass.
- Do not change the current RAMforge engine as part of a documentation-only phase.
- Do not add a dependency merely to hide an unresolved layout or quantization assumption.
- Do not make `qwen35` an alias of Qwen2; its hybrid attention/SSM architecture remains a separate future target.
- Do not change package versions or milestone numbering to represent ForgeCore progress.
- Do not let planner estimates stand in for runtime numerical validation.
- Keep every production allocation and read visible to `MemoryBudget` when the new core is integrated.
- Preserve all evidence artifacts: model identity/hash, metadata dump, descriptor table, prompt IDs, raw row checksums, layer checkpoints, toolchain, command line, and reference outputs.

## Design principles

1. **Storage-independent mathematics:** operator implementations accept explicit spans/slices and shape contracts, not file handles or implicit tensor orientation.
2. **One authoritative semantic definition:** each operator has one scalar/reference implementation; optimized variants prove equivalence against it.
3. **Explicit layouts:** every tensor, matrix, head, KV, quantized block, and row-range convention is named and validated.
4. **Boundary-first validation:** compare small artifacts at each boundary before running the full model.
5. **Deterministic execution:** fixed token IDs, positions, thread count, accumulation policy, and sampler for reference traces.
6. **Memory-aware integration:** streaming, caching, and budget charging remain infrastructure around the core rather than being mixed into operator code.
7. **Failure is data:** a mismatch produces a minimized artifact and stops the next layer of validation; it is not silently tolerated.

## Proposed layers

```text
ForgeCore::model_contract
    explicit Qwen2/Llama tensor names, shapes, dtypes, metadata

ForgeCore::storage_adapter
    TensorReader / row reader / bounded block reader
    implemented using RAMforge GgufDataSource

ForgeCore::quant_reference
    byte-exact Q4_0/Q6_K/... block parse and row dot

ForgeCore::ops_reference
    matvec, dequantized matvec, RMSNorm, RoPE, softmax,
    GQA attention, SwiGLU, residuals, output projection

ForgeCore::transformer
    one layer, prefill, decode, KV staging/commit

ForgeCore::trace
    prompt IDs, hidden checkpoints, logits, row comparisons,
    reproducible artifacts

RAMforge runtime adapter
    MemoryBudget, LayerReadPlan, LayerCache, streaming policy,
    planner, profiling, CLI/TUI, persistence
```

The names above are proposed conceptual boundaries, not APIs that currently exist.

## Phase 0 — Freeze and inventory

**Deliverable:** a source-and-evidence baseline before numerical changes.

- Record the current RAMforge commit/source snapshot if version control is available.
- Preserve the current `docs/handoff/` package and historical `REMODEL_ENGINEERING_REPORT.md` as context.
- Record the exact target model filename, file size, cryptographic hash, GGUF version, metadata, tensor descriptor table, and file offsets.
- Record tokenizer metadata and verify the fixed prompt IDs:

  ```text
  prompt: hi, what's 2+2=?
  ids:    [6023, 11, 1128, 594, 220, 17, 10, 17, 19884]
  ```

- Record Qwen2.5 dimensions: vocabulary `151936`, hidden `1536`, layers `28`, query heads `12`, KV heads `2`, head dimension `128`, FFN `8960`, and all model-specific norm/RoPE/context metadata from the actual GGUF.
- Recover or explicitly mark missing the exact layer-27 trace values.
- Capture `output.weight` descriptor and raw Q6_K rows 59 and 220.

No numerical implementation change should start until this inventory is reproducible.

## Phase 1 — Contract tests on synthetic tensors

**Deliverable:** small deterministic tests that fail on shape/layout ambiguity.

Define and test, independently of GGUF:

- matrix shape `[input, output]`, materialized `[output][input]`, and `y[o]` formula;
- row ranges where output index zero is local to the supplied output slice;
- F32 accumulation order and finite-value behavior;
- F16/BF16 conversion policy;
- RMSNorm epsilon and reduction order;
- half-split RoPE for position zero and nonzero positions;
- head dimension, query/KV head divisibility, GQA mapping;
- attention history/current-K/current-V shapes and causal positions;
- SwiGLU order `SiLU(gate) * up` and residual order;
- KV layout `[position][kv_head][head_dim]`, staging, commit, and sequence length.

Use tiny nonsquare matrices and hand-computed values. Every invalid shape must fail explicitly rather than trigger a transpose or truncation guess.

## Phase 2 — Quantized byte and row authority

**Deliverable:** a byte-level Qwen2.5 Q6_K row test and independent scalar row-dot result.

Start with the two target rows used by the existing diagnostic:

- row 59;
- row 220;
- 1536 input values;
- 6 Q6_K blocks per row;
- 210 bytes/block;
- 1260 bytes/row.

For each row:

1. read exactly the descriptor-bounded 1260-byte slice;
2. record descriptor offset, data-start offset, row offset, absolute file offset, and checksum;
3. parse one block with an independently reviewed field map;
4. compare raw first bytes, scales, `d` bits/value, and first decoded values;
5. dequantize the complete row into a test-owned buffer;
6. compute a scalar dot against a fixed known input vector;
7. compare with the external oracle's row result, preserving full precision.

Only after the row bytes and scalar result agree should the fused Q6_K path be admitted. Then compare fused versus scalar across all six blocks and uneven row ranges. If the two RAMforge paths disagree, stop and repair the numerical contract before touching the transformer.

## Phase 3 — Single-operator reference comparison

**Deliverable:** per-operator trace artifacts.

Compare ForgeCore against an independent reference or a minimized script for:

1. token embedding row;
2. attention RMSNorm output;
3. Q/K/V projections before bias;
4. Q/K/V bias application;
5. RoPE Q and K at known positions;
6. attention scores and probabilities for one head;
7. attention output before output projection;
8. attention residual;
9. FFN RMSNorm;
10. gate/up projections;
11. SiLU and elementwise gate multiplication;
12. down projection and FFN residual;
13. final hidden after each layer;
14. final output RMSNorm (`result_norm`);
15. selected output projection rows and complete prompt logits.

Use exact tensor names and explicit dimensions. Record min/max/sum/L2/first-eight plus selected indexed values, and preserve higher precision where the comparison needs it.

## Phase 4 — One-layer and full prefill

**Deliverable:** a one-layer fixture, then a 28-layer prompt trace.

- Build a tiny Qwen2-shaped synthetic layer with known weights and compare the complete block to a separate reference implementation.
- Run the actual target's first layer with the fixed prompt's final token and compare all intermediate boundaries.
- Extend through layers 0–27. Compare the exact layer-27 values when the archived evidence is recovered; do not substitute a newly computed value without marking it as a new artifact.
- Compare final prompt `result_norm`, selected logits, full top-k ordering, and first greedy token.
- Ensure prefill reads prompt positions `0..8` and produces the first token from final prompt logits.

This phase should not involve sampling randomness, long generation, or cache eviction.

## Phase 5 — Decode and KV validation

**Deliverable:** a two-step decode trace matching the diagnostic boundary.

Use the exact first and second generated token IDs from the reference trace once available. Verify:

- first decode input is the first sampled token at position 9;
- KV length before/after each forward is correct;
- current K/V are not read as history before commit;
- all layers consume the same prior history;
- second decode input is the second sampled token at position 10;
- logits are computed after the forward that needs them;
- the normal loop skips a needless final forward when no later logits are required.

Test prefill and decode separately. A passing prefill does not prove KV/generation correctness.

## Phase 6 — Representation and optimized-path parity

**Deliverable:** optimized kernels proven against ForgeCore reference.

- Keep scalar F32 and dequantized reference paths as the authority.
- Compare AVX2/Rayon F32 matvec against the scalar path for small, nonsquare, and real Qwen shapes.
- Compare Q4_0/Q6_K fused row paths against scalar dequantized rows, including row starts, row ends, uneven partitions, and one-row output.
- Compare F16/BF16 loading/conversion with known bit patterns.
- Test non-finite/error behavior and shape rejection.
- Do not permit a fast path to allocate an uncharged full matrix or silently switch layout.

## Phase 7 — Integrate with RAMforge infrastructure

**Deliverable:** ForgeCore-backed streaming execution without planner/storage regression.

Preserve, subject to contract tests:

- `GgufDataSource` and bounded range reads;
- descriptor validation and read coalescing;
- `MemoryBudget` named charges and scoped release;
- `PersistentWeight`, `LayerDescriptor`, `LayerReadPlan`, and `LayerCache`;
- planner profiles, capabilities, `ExecutionPlan`, `PlanCompiler`, `RuntimeConfig`;
- calibration, persistence/compatibility, CLI/TUI, profiling, diagnostics, memory reports, and run history.

Replace or adapt only the numerical-facing interfaces:

- tensor-to-core representation conversion;
- quantized row access/decode;
- `ModelExecutor` layer call;
- KV/cache staging boundary;
- final norm/output projection;
- generation adapter.

Run synthetic out-of-core fixtures first. Then run the real target only after the full prefill/decode trace passes.

## Phase 8 — Acceptance and release decision

**Deliverable:** an evidence-backed decision, not a version bump.

Required evidence:

- exact model hash and GGUF descriptor/metadata dump;
- exact tokenizer metadata and prompt IDs;
- raw Q6_K row artifacts and independent row-dot comparison;
- full per-layer trace including layer 27;
- `result_norm`, prompt logits, first/second decode traces;
- optimized/reference parity report;
- workspace format/test/Clippy/release results;
- bounded out-of-core run with memory/I/O/profile report;
- explicit list of any remaining unsupported model/quantization cases.

Only after these pass may the runtime describe the target as validated. “Compatible with llama.cpp” remains a separate claim requiring an independently repeatable oracle procedure and should not be inferred from a single matching output.

## Suggested file/module ownership

| Concern | Preserve or proposed owner |
|---|---|
| GGUF parsing/descriptors | preserve `ramforge-core/src/gguf.rs`, `model.rs` |
| file-backed reads/profiling | preserve `ramforge-core/src/datasource.rs` |
| memory accounting | preserve `ramforge-core/src/memory.rs` |
| planner/discovery/persistence | preserve `ramforge-runtime` planner/discovery/plan/persistence modules |
| layer grouping/cache | preserve `layer.rs`, `layer_read.rs`, `layer_cache.rs`, `persistent.rs` with contract tests |
| scalar numerical semantics | proposed ForgeCore reference module |
| quantized byte decode | proposed ForgeCore quant module, initially independently verified against source data |
| transformer and KV semantics | proposed ForgeCore transformer module |
| fast kernels | proposed adapters proven against ForgeCore reference |
| generation/CLI adapter | preserve outer API where possible; replace inner engine implementation |

The table is a migration hypothesis, not a promise that names or APIs will remain unchanged.
