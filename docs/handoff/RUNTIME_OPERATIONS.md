# Operational runtime, service, and throughput

This milestone connects execution plans to real forgeCORE inference, adds a
transport-independent inference service, measures real generation throughput,
and provides bounded profile comparison. It does not add eviction, automatic
reclamation, automatic live switching, or GPU offload.

## Execution-plan contract

`RuntimePlan` (schema version 2) resolves a profile request against the machine
and model:

- profile: maximum-performance, balanced, background, custom
- model path and model fingerprint
- effective CPU context, batch, micro-batch, sequence count, threads
- load mode (`mmap` or `mmap-no-prefetch`)
- weight repacking
- KV type and KV type source
- per-field status: applied, constrained, advisory, unsupported, unknown
- warnings for unavailable topology, advisory RAM budget, or non-selectable
  devices

Requested values are distinguished from effective values. The service reads
back native effective context, batch, micro-batch, sequence count, and thread
count after context creation.

Currently applied to forgeCORE:

- CPU backend/device
- context length (native may pad it; effective value is read back)
- generation/batch thread count
- mmap vs mmap-no-prefetch
- CPU weight repacking
- single sequence

Advisory/unsupported:

- RAM budget is advisory unless cgroup enforcement is active
- GPU offload/placement is unsupported by the current adapter
- KV effective types are not exposed by the pinned native getter

## Runtime lifecycle

`InferenceService` states:

`empty → model_loaded → ready → generating → ready → terminated`

- Model load and context creation are explicit.
- A service owns one model and one context at a time.
- `generate` requires `ready`; no concurrent generation is allowed.
- A fresh context is created before each measured benchmark run.
- `unload` drops context, model, and tokenizer.
- Errors are typed: invalid state, configuration, model, tokenizer, context,
  execution, cancellation, or stream sink.
- Cancellation is cooperative and checked between token decodes.
- Streaming decoded text is delivered through a caller callback.

No service operation authorizes eviction or release.

## Local service transport

`ramforge serve --stdio` exposes the service over JSON-lines on stdin/stdout.
This is the smallest real transport; the service contract is independent of it
and can be used by a TUI or GUI adapter.

`--threads 0` (the default) lets the selected profile choose the CPU thread
count; an explicit value overrides the profile and is still constrained and
read back from the native context.

The native context rounds `n_ctx` up with `GGML_PAD(n_ctx, 256)` in
`llama-context.cpp`. A requested context of 128 therefore reports an effective
context of 256 in `status`. `capabilities.max_context_tokens` is the planned
request; `status.effective_context_tokens` is the native readback.

`architecture` returns the GGUF-derived architecture metadata held by the
resolved plan. `hierarchy` returns a fresh memory-hierarchy observation (schema 2, including
machine collection time and sequence), the plan's placement model, and an
f16-assumption feasibility evaluation. It is not
a snapshot captured at model load and does not retain the original `--ram`
budget; the response states both limitations.

Commands:

```json
{"id":1,"command":"capabilities"}
{"id":2,"command":"status"}
{"id":3,"command":"generate","prompt":"hello","max_tokens":8}
{"id":4,"command":"session"}
{"id":5,"command":"unload"}
{"id":6,"command":"architecture"}
{"id":7,"command":"hierarchy"}
{"id":8,"command":"memory"}
{"command":"cancel"}
```

`memory` returns GPU observability schema 3:

- device total/free/used bytes with timestamps and native provenance;
- per-buffer model, KV/context, and compute bytes;
- per-layer/device/buffer tensor storage totals and file-backed bytes;
- overlapping category flags;
- session buffer totals;
- explicit unsupported allocation/deallocation counters and transfer metrics.

It works before a context exists (model-only buffers), after a session is
created (KV and compute), after cancellation, and after unload (device-only,
empty buffers). CPU-only plans explicitly pin the CPU device, so a CUDA build
does not allocate CUDA compute buffers for a CPU plan.

Generation events:

```json
{"id":3,"event":"token","text":"..."}
{"id":3,"event":"done","text":"...","stats":{...}}
{"id":3,"event":"cancelled"}
{"id":3,"event":"error","kind":"...","error":"..."}
```

A cancel line sets a shared atomic flag; the generation loop observes it
between token decodes. The transport is local-process only. It does not expose
authentication or network access.

`status` includes `execution_mode`, `device_id`, requested and effective GPU
layers, GPU/CPU transformer-layer counts, output-device type, and
`staged_layers` (tensor count, device bytes, graph substitutions). All effective
values come from native placement readback after load. `capabilities` includes
`execution_modes` and `gpu_execution`; with the CUDA build these report
`["cpu","gpu","hybrid"]` and `true`. Without CUDA they report `["cpu"]` and
`false`.

### GPU requests

`benchmark` and `serve` accept:

- `--device cpu|auto|<device-id>` (default `cpu`)
- `--gpu-layers N` (default `0`)

Plans represent CPU, GPU, and hybrid modes. With the CUDA build,
`--device auto --gpu-layers N` selects CUDA0 and loads with native readback:
the service refuses to report success if the effective layer count is zero.
`n_gpu_layers` includes the output layer; for SmolLM2-135M (30 transformer
layers), requesting 8 yields 7 GPU transformer layers plus the GPU output
layer, and requesting 31 offloads all 30 transformer layers plus output.

If no GPU backend/device is registered, a non-CPU request fails before model
load with an explicit error and **no CPU fallback**. GPU load or placement
failure is returned as an error; the service never silently retries on CPU.
See [GPU_ENVIRONMENT.md](GPU_ENVIRONMENT.md) for validated results and
limitations.

With CUDA, `status` reports `execution_mode=gpu` for full offload or `hybrid`
for partial offload. Benchmark schema 6 includes VRAM snapshots before load,
after load, after warmup, after generation, and after unload, plus native buffer
breakdowns. Allocation event counters, transfer bytes/duration, synchronization
duration, and peak VRAM are explicitly unsupported.

### Controlled staged layers

`benchmark` and `serve` accept one or more explicit:

```text
--staged-layer N
```

Additional flags:

- `--staged-budget SIZE`: hard limit on RAMforge-owned staged CUDA bytes;
- `--staged-auto`: select layers ascending within the budget;
- `--staged-margin SIZE`: safety margin subtracted from observed free VRAM
  (default 256M).

This is a separate execution mode, not a profile and not normal GPU offload:

- the model loads CPU-resident with both CPU and CUDA devices exposed;
- layer `N` is allocated in an independent CUDA buffer by the native context;
- forge-staging transfers exact tensor bytes into that allocation;
- after graph construction, node weight sources are replaced by the staged
  tensors before scheduler allocation;
- `status.staged_layers` reports tensor count, device bytes, and a monotonic
  graph-substitution count. Zero substitutions would mean the graph did not use
  the staged weights.
- `inference_consumed_staged_data` is not asserted; substitution counts and
  logits/outputs are the evidence.

Planning is deterministic: exact native per-layer byte totals are checked before
allocation. Explicit layer sets are all-or-nothing; auto selection may be
partial. `status.staging_plan` reports configured/effective budget, observed
free VRAM, planned bytes, accepted/rejected layers, and
`exact`/`partial`/`rejected`/`unknown` state.

The plan reports `eviction_supported=true`,
`allocation_reuse_supported=true`, and
`transfer_compute_overlap_supported=false`. An evicted layer's CUDA buffer is
freed; later staging allocates a new buffer rather than reusing capacity.

### Same-context staged replacement

`ramforge staging-cycle MODEL` validates and exercises dynamic replacement in
one live context:

```text
ramforge staging-cycle MODEL \
  --staged-layer 0 --staged-layer 1 \
  --replace-layer 1 --replace-layer 2 \
  --staged-budget 20M --cycles 2 --max-tokens 8 --json
```

Flags:

- `--staged-layer N` / `--staged-auto`: initial selection;
- `--replace-layer N` (repeatable): replacement selection;
- `--cycles N`: alternate replacement and initial selections in the same
  context;
- `--staged-budget`, `--staged-margin`: budget policy shared by every cycle;
- `--staged-pool SIZE`: bounded pool for retaining/reusing evicted CUDA buffers;
- `--pipeline`: prepare the next selection in the pending slot and transfer it
  asynchronously while the active selection generates;
- `--pipeline-sync`: use the same pending slot and batch path but commit before
  generation (controlled no-overlap comparison);
- `--json`: full plans, statuses, timings and native lifecycle counters.

Replacement sequence:

1. build an exact replacement plan from current residency and effective budget;
2. reject an unfitting explicit set before mutation;
3. invalidate the graph and scheduler (quiesces outstanding backend work);
4. evict removed layers;
5. allocate, fill and wait for each added layer;
6. leave the graph invalidated; the next generation rebuilds and substitutes
   the new selection.

If staging fails, RAMforge evicts partial new layers and re-stages the previous
selection. A successful rollback raises a `rolled_back` replacement error and
keeps the context usable. If rollback also fails, the service enters `failed`
and rejects generation.

`status.staging_lifecycle` reports the native graph phase, whether no cached
graph can consume the staged sources, resident layers/bytes, allocation,
release, reuse, invalidation, rebuild and scheduler-reset counts, quiescence and
graph-build durations, and the supported capability flags. Counters are native
facts; budget planning remains RAMforge policy.

### Pipelined replacement

Pipeline mode uses two slots:

1. the active selection remains graph-visible and is used for generation;
2. the complete next selection is allocated in the native pending slot and
   filled by one asynchronous H2D batch on a private stream;
3. `wait_started()` confirms the DMA has begun, then generation runs;
4. after generation, RAMforge queries/wait for transfer completion, invalidates
   the graph and commits the pending selection atomically;
5. the next decode rebuilds the graph against the new selection.

Pending tensors are never referenced by the active graph, so staging is safe
while it executes. Commit requires successful invalidation and no in-flight
execution. Abort waits for the transfer and discards pending allocations while
preserving the active selection. Unload, session reset and service drop abort a
pending pipeline safely.

Budget: pipeline planning requires
`active + pending + pool <= effective budget`. The pool is bounded by
`--staged-pool` and never exceeds the effective budget. A configuration that
does not fit both slots is rejected before allocation.

`status.pipeline_pending`, `staging_lifecycle.pending_layers/pending_bytes`,
`pool_bytes/pool_limit_bytes/pool_buffers`, `commit_count` and `reuse_count`
expose the pipeline state. `reuse_count` is a native fact; allocation reuse only
happens after invalidation.

Overlap is measured with CUDA transfer events and a synchronized compute
interval. The transfer must start before compute; the prepare→compute gap is
subtracted from the overlap bound. On this hardware the transfer is fully hidden
but concurrent DMA reduces compute throughput, so pipeline mode is experimental
and not the default.

Staged weights are loaded once per layer and remain static between replacements.
There is no per-token streaming. A failure is explicit; the service does not
fall back to CPU or normal GPU placement.

## Throughput benchmark

`ramforge benchmark MODEL --profile balanced --runs 3 --warmup 1` measures:

- model load duration
- context creation duration per run
- prompt tokens and prompt processing throughput
- time to first token
- generation tokens/s
- decode tokens/s
- generated token count and stop reason
- RSS before/after generation
- effective native configuration
- requested/effective GPU layers, GPU/CPU transformer-layer counts, output device
- device free-memory snapshots before load, after load, after warmup, after
  generation, and after unload
- native buffer model/KV/compute breakdowns and versioned GPU observability
  snapshots
- staged-layer substitution counts when `--staged-layer` is used (benchmark
  schema 7)

Generation throughput and prompt throughput are separate. A single run is not
presented as a reliable comparison.

Measured baseline on the development machine, SmolLM2-135M Q4_K_M, 256 context,
4 threads, no repacking, debug build, warm/uncontrolled:

| Metric | Median | Range |
|---|---:|---:|
| Generation tokens/s | 136.87 | 112.58–155.41 |
| Decode tokens/s | 171.70 | — |
| TTFT | 11.40 ms | — |
| Prompt throughput | 423.31 tokens/s | — |
| Context creation | 5.52 ms | — |
| Model load | 234.91 ms | — |

These debug-build figures are single-machine warm observations, not a
performance guarantee. Release-build results, which are the authoritative
throughput evidence, are in [BENCHMARKS.md](BENCHMARKS.md): in the latest
session balanced 8 threads measured 223.04 tok/s (P25–P75 218.94–224.91),
maximum-performance 4 threads repacked 205.25, balanced 4 threads 203.92, and
background 81.44. An earlier session measured 283.81 and 267.65 for balanced 8
and 4; an order-reversed session measured 269.05 and 252.02; a final spot check
measured 219.91 and 229.50. Absolute rates vary by roughly 25% between sessions
and the 4/8 ranking reverses. No reliable winner between 4 and 8 threads is
claimed. Only within-session comparisons are treated as evidence.

## Machine execution readiness

`ramforge machine --json` includes:

- `profile.backend_inventory`: registered backend names and device counts,
  `gpu_offload_available`, and explicit notes.
- `execution_readiness`: CPU/GPU execution flags, selected device, registered
  backends, enumerated non-CPU devices (with memory observations), and the
  same backend notes.
- `memory_hierarchy`: host, cgroup, storage, device total/free, KV, mapping,
  backend-buffer, and staging categories without double counting.

A detected device is not automatically a selectable device. With the validated
CUDA build, CUDA0 is selectable; with a CPU-only build it is not registered at
all. `execution_readiness.gpu_execution` reflects current selectability.

## Staging transfer prototype

`ramforge staging-probe MODEL --layer N [--json]` runs the native staging
prototype:

- selects one `blk.N` layer and reads its exact tensor bytes while the model is
  quiescent;
- allocates pinned host, CUDA device, and pinned readback buffers;
- transfers H2D and D2H through the native engine;
- verifies FNV-1a integrity and reports byte counts, event durations,
  submission/wait overhead, allocation/transfer counters, and high-water bytes;
- releases buffers after completion.

**This is not inference staging.** The report states
`inference_consumed_staged_data=false`; llama.cpp does not consume the staged
copy. The graph has no safe layer-boundary rebind/consume contract yet. See
[STAGING_ARCHITECTURE.md](STAGING_ARCHITECTURE.md).

## Profile optimization loop

`ramforge optimize MODEL --runs 2 --max-tokens 4` runs maximum-performance,
balanced, and background candidates with the same bounded workload, ranks them
by measured median generation tokens/s, and returns an advisory recommendation.

One-run smoke:

| Profile | Generation tokens/s |
|---|---:|
| maximum-performance | 121.98 |
| balanced | 137.87 |
| background | 103.34 |

Recommendation: balanced. The result is noisy and advisory. No live session is
reconfigured, no model/backend/device is switched, and no profile is executed
automatically.

## Storage assessment and copy

`ramforge storage MODEL --dest DIR` checks capacity, read-only state, and
filesystem identity. It never copies or deletes without explicit action.

`ramforge storage MODEL --copy --destination PATH --yes` performs a verified
copy:

- copies to a temporary destination;
- fsyncs and verifies byte count and a full-file FNV-1a fingerprint;
- renames into place;
- preserves the source;
- does not delete or replace the source automatically.

A faster storage device is not claimed to increase token throughput without
measurement.

## Profiles

- **Maximum performance:** all process-visible CPUs, larger context, repacking
  enabled.
- **Balanced:** one fewer CPU than available, moderate context, repacking
  disabled.
- **Background:** one CPU, smaller context, mmap-no-prefetch, repacking
  disabled.
- **Custom:** explicit context/batch/thread/load/repacking values, with clamps
  and field statuses.

All profiles use the same planning and execution machinery. They are resource
policies, not separate inference engines.

## Architecture and memory hierarchy

`ModelArchitecture` distinguishes dense and MoE models from GGUF metadata,
including layer/head/KV-head/context fields and routed-expert fractions. Missing
metadata stays unknown; routed fraction is not presented as a complete active
parameter ratio.

`MemoryHierarchy` and `PlacementModel` separate host RAM, cgroup budget, device
memory, storage, file-backed mappings, anonymous/process memory, KV estimate,
compute/graph, backend buffers, and staging. Categories overlap and are never
summed. File-backed mmap is confirmed only when repacking is disabled; compute,
graph, backend, and staging placement remain unknown. See
[LARGE_MODEL_EXECUTION.md](LARGE_MODEL_EXECUTION.md).

## Large-model position

The pinned native backend performs CPU inference and supports file-backed mmap.
Measured out-of-core runs load and generate under a cgroup cap (for example a
13.13 GB model under 8 GiB), but throughput becomes storage- and reclaim-bound.

RAMforge also exposes model-scoped page release (`MADV_DONTNEED`), bounded
page-residency sampling (`mincore`), tensor-page release with shared-page
skipping, bounded prefetch hints (`MADV_WILLNEED`), and staging plans with
quiescence gates. All of these are explicit and advisory: prefetch is a hint,
not a synchronous read, and release does not unmap memory or invalidate tensor
pointers.

RAMforge does **not** provide tensor/layer streaming, caller-owned weight
staging consumed by the graph, transfer/compute overlap, anonymous or backend
buffer eviction, GPU placement, or MoE routing-aware residency. The 500B
parameter target remains a product ambition, not a claimed capability.

## Exclusions

Automatic action is excluded: RAMforge does not automatically evict, reclaim,
prefetch, switch plans, move models, or change backend/device. Page operations
require an explicit flag/policy and a quiescent model. Predictive staging is
advisory. No live tensor migration or unvalidated GPU allocation behavior is
implemented.
