# MoE Expert-Aware Residency — Routing Observability and Feasibility

**Date:** 2026-11-25
**Scope:** Verify the MoE routing path, implement safe read-only routing observation,
measure real expert locality on DeepSeek2, quantify residency/prefetch feasibility,
and prepare the native and dense-overflow handoffs.
**Native checkout:** pinned `7fe450e19305b828c199d602c23a8337aaa1f03b`, unmodified.

## 1. Executive summary

Per-token expert routing **is observable today without modifying native
llama.cpp**. The pinned graph already names the selected-expert tensor
`ffn_moe_topk-<layer>`, and the existing scheduler eval callback exposes the
completed tensor. forgeCORE now copies the first token row (six i32 ids for
DeepSeek2) under an opt-in, bounded, CPU-only observer. RAMforge exposes it as
`ramforge run ... --routing-observe --routing-report PATH`.

This is a real observation, not an inference from tensor names or expert counts.
On DeepSeek-Coder-V2-Lite Q4_K_M, a 64-token CPU run produced:

- 1,639 complete single-token decode observations and 25 partial prefill rows.
- 26 MoE layers, 6 selected experts per layer per token.
- 1,457 of 1,664 layer-expert pairs selected at least once (49-64 unique experts
  per layer).
- Consecutive-token overlap **25.79%**; mean within-layer reuse distance
  **6.16 token observations**.
- Theoretical LRU hit rates: 27.7% at 256 expert slots, 53.9% at 512, 76.7% at
  1,024. These are simulations, not implemented residency.

The critical correction to earlier reasoning: **the ~1.73 GB active estimate is
per token, not a workload working set.** Over 64 tokens, almost every expert is
touched. A useful policy is a bounded cache with measured locality, not a static
"active expert" set.

Expert-level **movement** remains blocked. The native API exposes whole-tensor
page prefetch/release and whole-layer CUDA staging. It does not expose expert-row
prefetch, expert-granular staging, or expert-level buffer substitution. The file
ranges of individual experts are computable from GGUF dimensions and mapping
metadata; the absence is in movement control, not identification.

Prefetch is also not justified today: the previous token's active set is about
**893 MB** across 26 layers, while measured consecutive overlap is only 25.79%.
A one-token-ahead prefetch would move ~893 MB to use ~230 MB (74.2% waste), and
the top-k tensor is produced immediately before `MUL_MAT_ID`, leaving no
within-layer lead time. No speculative prefetch was implemented.

## 2. Native routing path (verified)

In `native/src/src/llama-graph.cpp`, `llm_graph_context::build_moe_ffn`:

1. Computes router logits (`ffn_moe_logits`), optional bias/group selection.
2. Selects experts with `ggml_argsort_top_k(..., n_expert_used)` and names the
   result through `cb(selected_experts, "ffn_moe_topk", il)`.
3. Uses that index tensor in `MUL_MAT_ID` for the gate/up/down expert tensors.

The eval callback (`llama_context_params.cb_eval`) is invoked for each scheduler
node with `ask = true` before and `ask = false` after host-visible completion. In
the trace, `ffn_moe_topk-<layer>` appears twice; the completed event carries the
data. forgeCORE reads it with `ggml_backend_tensor_get` from the existing
callback.

Observed topology for DeepSeek2:

- `ffn_moe_argsort-<layer>`: full 64-expert order.
- `ffn_moe_topk-<layer>`: selected `n_expert_used = 6` view.
- `ffn_moe_weights-<layer>`: six F32 weights; byte count infers graph tokens.
- Layers 1-26 are MoE; layer 0 is dense.

For a single-token decode, the top-k tensor is 24 bytes (6 i32) and the copied row
is the complete selection. For prompt batches, the tensor is a strided view; the
prototype copies only the first token row and marks those observations partial.
`tokens_in_graph` is inferred from `ffn_moe_weights` bytes divided by
`expert_used * 4`.

## 3. Implemented observability

### forgeCORE

`ContextOptions.routing: Option<ExpertRoutingOptions>` is new. It carries:

- `max_observations` (bounded 1..=65,536)
- `expert_count`
- `expert_used_count`

When enabled, the existing `TraceState` also records:

- execution id and observation sequence;
- layer parsed from the node suffix (no guessing when absent);
- selected expert ids from the first token row;
- invalid-index count;
- native top-k byte span;
- inferred graph token count;
- whether the observation is complete for the copied token.

New accessors: `Context::routing_observations`,
`routing_summary`, `take_routing_observations`, `clear_routing_observations`.
Routing is rejected when the model has any GPU-offloaded layers, because the
prototype only reads host-visible CPU tensors safely. It is disabled by default.

Validation: enabling observation produced **byte-identical greedy output** to
the same run without it. No measurable slowdown was detected in warm CPU runs
(off 7.38/5.36 s vs on 6.16/4.57 s over 64 tokens; order/cache effects dominate,
so no speed claim is made).

### RAMforge

`ramforge run --backend forgecore --routing-observe` enables observation and
prints a bounded locality summary. `--routing-report PATH` writes:

- summary counts and invalid indices;
- per-layer expert layout;
- per-layer frequency, top experts, overlap, reuse distance;
- LRU and offline-frequency cache simulations;
- prefetch assessment;
- one sample set of exact expert byte ranges;
- raw observation records (expert ids only, no prompt text).

`ramforge plan --json` now includes per-layer expert layout in addition to
aggregate expert accounting.

## 4. Expert layout and accounting

DeepSeek-Coder-V2-Lite Q4_K_M, full file:

| Component | Bytes |
|---|---:|
| Routed expert tensors (`*_exps`) | 9,527,361,536 |
| Shared experts (`*_shexp`) | 270,876,672 |
| Other tensors (attention, norms, dense layer, embeddings, output) | 562,182,144 |
| Total tensor payload | 10,360,420,352 |

Per-layer routed bytes vary because `ffn_down_exps` is Q8_0 while gate/up are
Q4_K: layer 1 is 403,701,760 B (6,307,840 B per expert); later layers are smaller
(about 5.23-6.31 MB per expert). The layout is derived from actual GGUF tensor
names, dimensions, quantization, and byte lengths, not parameter-count formulas.

The per-token active-set estimate remains arithmetic: 9,527,361,536 × 6/64 =
893,190,144 B routed, plus 270,876,672 shared and 562,182,144 other =
**1,726,248,960 B (~1.73 GB)** before KV/compute. It is not a guaranteed
residency set, and Section 5 shows why it is not a workload working set.

## 5. Measured routing locality

### 64-token chat, CPU, 6 threads, ctx 512, Q8_0 KV

| Metric | Value |
|---|---:|
| Complete decode observations | 1,639 |
| Partial prefill observations | 25 |
| Executions | 65 |
| Layers | 26 |
| Unique layer-expert pairs | 1,457 / 1,664 |
| Unique experts per layer | 49-64 |
| Consecutive overlap | 25.79% (1.55 of 6) |
| Mean reuse distance | 6.16 observations |
| Invalid indices | 0 |

The short 16-token run looked more local (41.68% overlap, 2.23 reuse distance),
but that result is workload-length dependent. Over 64 tokens, nearly every expert
in every layer is selected at least once. This is the central residency finding:
**expert popularity is broad; a small per-layer hot set does not cover the
workload.**

### Theoretical cache simulations

Global `(layer, expert)` slots, ordered by decode execution:

| Capacity | LRU hit rate | Offline frequency (upper bound) |
|---:|---:|---:|
| 64 | 0.0% | 15.2% |
| 128 | 0.0% | 25.3% |
| 256 | 27.7% | 41.4% |
| 512 | 53.9% | 64.2% |
| 1,024 | 76.7% | 91.0% |

These are simulations over observed indices. They do not move data, do not model
allocation granularity, and are not evidence of achievable throughput. A 1,024-slot
cache would hold roughly 6 GB of expert bytes on this model, which competes with
the model mapping and KV; the hit rate alone is not a feasibility proof.

## 6. Residency and prefetch feasibility

### What is possible without native changes

- Read-only routing observation (implemented).
- Exact per-expert file byte ranges from GGUF + mapping metadata (implemented as
  metadata only).
- Per-layer expert layout and active-set arithmetic (implemented).
- Routing trace aggregation, overlap/reuse measurement, LRU and offline-frequency
  simulation (implemented).
- Whole-layer staging, standby, delta transitions, page prefetch/release at
  tensor granularity (existing).

### What is not possible

- **Expert-granular staging or eviction.** `stage_layer_cuda` allocates all
  `blk.N` tensors into one CUDA buffer and substitutes whole tensors.
- **Expert-row prefetch.** `llama_model_prefetch_tensor_pages` takes tensor
  indices and prefetches whole tensors. `llama_model_sample_mapping_pages`
  samples ranges but does not prefetch them.
- **Expert-level buffer substitution.** The graph's `MUL_MAT_ID` reads the full
  expert tensor and uses the index tensor to select rows. Rebinding a subset
  would require a native change to the graph or a remapped expert buffer.
- **Safe partial eviction.** Releasing pages of an expert tensor would evict
  other experts in the same mapping and has no completion/ownership proof for
  the current graph.

### Why one-token-ahead prefetch was rejected

- Previous-token active set: **893.2 MB** across 26 layers.
- Measured reuse: **25.79%**, so only **230.4 MB** would be used.
- Wasted transfer: **662.8 MB (74.2%)** per token, before transfer/compute
  contention.
- The top-k node completes immediately before `MUL_MAT_ID`; there is no
  within-layer lead time. Cross-token prefetch would have to run between decodes,
  competing with the next token's compute and I/O.
- The pinned API cannot prefetch an expert row range anyway; only whole tensors.

No speculative transfer was implemented.

## 7. Native capability proposal (not applied)

The observation hook did not require a native change. The next capability does.

### 7.1 Minimal read-only residency API

Add to `include/llama.h`:

```c
struct llama_expert_page_info {
    uint64_t tensor_id;      // stable within the loaded model
    int32_t  layer;
    int32_t  expert;
    uint64_t file_offset;    // absolute in the model file
    uint64_t bytes;          // contiguous expert slice
    uint64_t mapping_id;
    uint64_t mapping_generation;
};

LLAMA_API int32_t llama_model_expert_page_info(
        const struct llama_model * model,
        const char * tensor_name,
        int32_t expert,
        struct llama_expert_page_info * out);

LLAMA_API int32_t llama_model_prefetch_expert_pages(
        struct llama_model * model,
        const struct llama_expert_page_info * ranges,
        size_t count,
        uint64_t * out_requested_bytes);

LLAMA_API int32_t llama_model_sample_expert_pages(
        const struct llama_model * model,
        const struct llama_expert_page_info * ranges,
        size_t count,
        uint64_t max_pages,
        struct llama_page_sample_result * out);
```

`prefetch_expert_pages` should only issue OS readahead (`madvise`/`posix_fadvise`
or platform equivalent) on validated file-backed ranges, never allocate or
rebind graph memory. `sample_expert_pages` gives read-only residency evidence so a
policy can be evaluated before any transfer is authorized.

### 7.2 Expert-granular staging (later)

A real staged-expert cache needs more than a prefetch hint:

- a compact device/host buffer containing selected experts;
- an index remap from global expert id to cache slot;
- graph integration so `MUL_MAT_ID` reads the compact buffer;
- per-execution ownership and quiescence proof;
- capacity/eviction policy and a miss path that reads the exact expert slice;
- CPU and CUDA implementations, with CUDA graph invalidation on cache change.

This is a substantial native graph/backend change and should be a separate
authorized milestone, not a patch to the protected checkout.

### 7.3 Graph and synchronization constraints

- CPU: top-k data is host-visible after the completed callback; current capture is
  safe and cheap.
- CUDA: the top-k tensor may be device-resident; the prototype rejects GPU model
  execution rather than issuing an unvalidated device copy inside the callback.
- CUDA graphs: `cb_eval` changes scheduler observation; graph replay may not call
  per-node callbacks. Routing observation should remain a diagnostic mode and not
  be enabled in graph-replay throughput paths until validated.
- Any expert buffer change must occur only at a quiescent boundary, like the
  existing staged-layer contract.

## 8. Dense overflow handoff

Do not treat this work as a solution to dense overflow.

Shared foundations that carry over:

- exact tensor byte ranges and per-layer layout;
- storage read/fault accounting already added in the previous milestone;
- page-residency sampling and mapping metadata;
- transfer/staging latency measurement;
- benchmark replay from recorded access traces.

Dense-specific work remains:

- reducing repeated reads when the dense working set exceeds page cache;
- ordered prefetch/read-ahead aligned to layer execution;
- bounded asynchronous I/O and overlap without kernel slowdown;
- eviction/reuse policy for dense layers;
- honest cold/warm separation without destructive global cache drops.

The next milestone should be **Dense Overflow Streaming & Adaptive Memory
Residency**, followed by **Hardware Performance Calibration**, and only then a
native-authorized **MoE expert buffer** implementation if the measured policy
justifies it.

## 9. Validation

Executed:

- forgeCORE: 141 lib tests, 31 batch, 9 CPU decode, 24 CPU exec proof, 9 ggml
  smoke, 1 GGUF, 2 graph trace, 44 KV state, plus the opt-in DeepSeek routing
  test. All passed; fixture-gated tests ignored.
- RAMforge: 426 runtime lib tests, 1 device test, 7 adapter tests, 61 CLI tests;
  all passed. `moe_routing` unit tests cover layout, byte ranges, overlap,
  LRU, invalid ids, and incomplete observations.
- Real-model: DeepSeek 64-token routing run; 16/8-token runs; routing on/off
  output byte-identical; GPU routing request rejected explicitly; opt-in
  RAMforge adapter test passed.

Not validated:

- CUDA routing observation (rejected by design).
- Graph-replay routing observation.
- Actual expert cache hit rate/throughput (no expert buffer exists).
- KV/quality trade-offs.
- Cold-cache avoidance (no destructive cache drop).

## 10. Honest conclusion

This milestone **implements verified routing observability and a validated
measurement/simulation foundation**. It does **not** implement expert-granular
residency or improve actual MoE throughput.

What is proven:

- Actual per-token expert ids are observable on CPU through the existing native
  callback, with no native modification and no output change.
- DeepSeek2 expert layout and byte ranges are measurable.
- Over 64 tokens, expert reuse exists but is broad: 25.79% consecutive overlap,
  6.16 reuse distance, 1,457/1,664 layer-expert pairs touched.
- A static ~1.73 GB active set is not a workload working set.
- One-token-ahead prefetch of the previous active set would waste ~74% of ~893 MB
  per token and has no within-layer lead time.

What remains blocked:

- Moving or retaining individual experts safely.
- Device-side routing observation without a native device-copy contract.
- Any claim that the simulated LRU hit rates are achievable in RAMforge today.

No 180B/500B feasibility claim follows from this work.
