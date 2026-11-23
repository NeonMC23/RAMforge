# Controlled GPU staging architecture (investigation)

Status: **allocation reuse, double buffering, async batch transfers and measured
transfer/compute overlap validated; the overlap produced no end-to-end
throughput gain on the RTX 3050.**

This document traces the pinned llama.cpp/ggml execution path, identifies safe
native extension points, measures the relevant transfer/storage limits, and
defines the minimum primitives required before controlled staging can be
implemented.

## Baseline

- Native pin: `7fe450e19305b828c199d602c23a8337aaa1f03b`.
- 12 preserved native modified files plus read-only bridge extensions.
- CPU build: `native/build`.
- CUDA build: `native/build-cuda`, SM 86, CUDA 13.2.51, driver 620.70.06.
- GPU: RTX 3050 Laptop, 4 GiB VRAM, about 3.69–3.87 GB free in tests.
- CPU: i5-11400H, 12 threads; host RAM 15 GiB.
- Existing inference: CPU, conventional hybrid, and full GPU, with native
  placement and memory observability.

## Execution path trace

| Step | Pinned source | Behavior |
|---|---|---|
| GGUF/tensor discovery | `src/llama-model-loader.cpp` | Loader reads metadata and maps/loads tensors. |
| Model tensor allocation | `src/llama-model.cpp` around 2140–2250 | `ctxs_bufs` owns one context plus backend buffers per buffer type/file. Tensors receive `tensor->buffer` and `tensor->data`. |
| Model buffer ownership | `src/llama-model.h` `ctxs_bufs`; `src/llama-model.cpp:2245` | Buffers live for the model lifetime; release is bulk via RAII, not per layer/tensor. |
| Placement | `src/llama-model.cpp` layer assignment; `dev_layer()`/`dev_output()` | Layer-to-device assignment is fixed at load. |
| Context/KV | `src/llama-context.cpp`, `llama_memory_*` | KV objects allocate context buffers; `llama_context::memory_breakdown()` groups model/context/compute. |
| Graph build | `src/llama-context.cpp:process_ubatch` (1436+) | Builds/reuses one full-model cgraph. Graph reuse depends on topology parameters. |
| Graph allocation | same, `ggml_backend_sched_reset` / `alloc_graph` | Scheduler allocates temporary compute buffers for the graph. |
| Graph execution | `src/llama-context.cpp:graph_compute` (2614) | Calls `ggml_backend_sched_graph_compute_async`; execution gate held until synchronization. |
| Scheduler split | `ggml/src/ggml-backend.cpp:1066` | Splits the graph by backend and inserts input copies. |
| Split execution | `ggml/src/ggml-backend.cpp:1646` | Copies split inputs, synchronizes/uses events when needed, submits backend graph async. |
| CUDA kernels | `ggml/src/ggml-cuda/ggml-cuda.cu` | Kernels execute on the CUDA backend context streams. |
| Host/device tensor copy | `ggml-cuda.cu:788–837` | `set/get/copy` use `cudaMemcpyAsync` on `cudaStreamPerThread` and then **synchronize immediately**. |
| Synchronization | `llama_context::graph_compute` / scheduler events | The execution gate covers submission through native synchronization. |
| Logits/sampling | `src/llama-context.cpp`, `forgeCORE`/RAMforge | Host copies of logits and Rust sampling. |
| Context/model free | `llama_context` / `llama_model` RAII | Buffers released in bulk; contexts cannot outlive the model. |

## Existing hooks and safety

| Hook/API | Safe for staging? | Reason |
|---|---|---|
| `ggml_backend_buffer_get_base/size/type/name` | read-only yes | Describes existing allocation; does not release it. |
| `ggml_backend_buffer_is_host` | read-only yes | Host vs device placement evidence. |
| `ggml_backend_buft_get_device` | read-only yes | Device association. |
| `ggml_backend_tensor_set/get` | no for live inference | Mutates/copies tensor storage synchronously; graph may reference it. |
| `ggml_backend_tensor_set/get_async` | **not asynchronous in CUDA backend** | The implementation synchronizes immediately (lines 788–797). |
| `ggml_backend_tensor_copy` | only at a fully synchronized boundary | Device copy, then immediate synchronization. |
| `ggml_backend_event_*` | yes for timing/ordering | Public events can order work; they do not create staging buffers or rebind tensors. |
| `ggml_backend_sched_reset/alloc_graph` | not per-layer | Rebuilds scheduler allocations; no independent layer residency. |
| `cb_eval` graph callback | observation only | Forces per-node/per-view synchronization when returning need; not a production overlap mechanism. |
| `llama_graph_reserve` | not a staging hook | Reserves compute graph; does not manage weight residency. |
| Model tensor buffers | not independently releasable | Owned in `ctxs_bufs` for model lifetime. |

## Language trade-off

| Approach | Strengths | Limits | Verdict |
|---|---|---|---|
| A. Rust over current forgeCORE APIs | Simple, safe, no native churn | No tensor/buffer rebind, no layer pause, no async copy control | Insufficient |
| B. C++ bridge in forgeCORE | Can access ggml buffers/tensors, graph/scheduler internals with narrow C ABI, existing patch workflow | Still cannot create new graph boundaries without llama changes | Necessary for ownership/scheduling primitives |
| C. C++ + CUDA hooks | Needed for true async staging, streams, events, pinned buffers, overlap | Highest maintenance; must respect ggml scheduler and model lifetime | Necessary only after B establishes layer boundaries and buffer ownership |

Rust should keep policy, budgets, profiles, cancellation decisions, and
observability. C++ should own native buffer descriptors and graph-level
operations. Hot paths must be coarse-grained; per-tensor FFI chatter is not
acceptable.

## Layer and tensor staging feasibility

### Tensor level

- Individual model tensors have unique buffer ownership and byte ranges.
- Existing introspection is sufficient to identify layer tensors, sizes,
  quantization, host/file-backed state, and shared buffers.
- There is **no public API** to rebind a loaded tensor to a different buffer or
  to free one model tensor while preserving graph validity.
- `ggml_backend_tensor_alloc` can assign a tensor to a buffer, but the model
  loader and graph references are already established. Using it on a live model
  would risk invalid pointers and stale graph metadata.

### Layer level

- Layer tensors are grouped by `blk.N.` and currently share two large model
  buffers (CUDA0 model buffer and CPU-mapped buffer) for the validated 7B and
  135M runs.
- A layer can only be staged independently if its tensors are allocated in a
  dedicated, releasable buffer and the graph/scheduler is told when that buffer
  becomes valid.
- The current graph is a single full-model cgraph; the scheduler splits by
  backend, not by an application-visible layer boundary.
- `process_ubatch` builds and reuses this graph; there is no callback at
  "layer completed, next layer may be staged".

**Layer staging is not feasible with the current public API.**

## Double/triple buffering and overlap

- The public CUDA tensor copy path synchronizes immediately. It does not expose
  a stream/event contract to RAMforge.
- Public backend events exist, but using them around synchronous copies would
  not create overlap.
- Real overlap requires caller-owned pinned host buffers, device staging
  buffers, a private CUDA stream, events, and a graph execution hook that
  consumes the staged buffer after the event.
- A single staging buffer serializes load and compute. Double buffering can
  overlap one weight transfer with the previous layer's compute only if the
  compute backend uses a different stream and the scheduler can wait per layer.
  Triple buffering adds latency tolerance but more VRAM; on a 4 GiB GPU it is
  unlikely to be affordable for 7B layers (~156–183 MB each).
- No overlap is claimed or measured today.

## Storage-to-GPU path

Measured on this machine (read-only, isolated):

- PCIe pinned H2D: **6.30 GB/s**; pageable H2D: **5.10 GB/s**.
- PCIe pinned D2H: **6.37 GB/s**; pageable D2H: **4.64 GB/s**.
- Direct NVMe read (256 MiB, `O_DIRECT`, 3 runs): **1.3–1.8 GB/s**.

Path for a cold weight:

`storage → page cache/host buffer → pinned host staging → CUDA device`

Direct storage-to-GPU is not exposed by the current backend. The pageable vs
pinned difference is material but not transformative; storage read dominates.

## Performance model

For Qwen2.5-7B Q4_K_M:

- Total weights: 4.68 GB; 28 transformer layers; ~156–183 MB/layer.
- Current 12-layer hybrid: 9.30 tok/s → 107.5 ms/token.
- Cold layer from storage: 183 MB / 1.6 GB/s ≈ 114 ms; plus H2D
  183 MB / 6.3 GB/s ≈ 29 ms; plus synchronization. A single cold layer already
  exceeds the token budget; 28 layers are not viable per token.
- If the layer is already in host RAM and only H2D is required: ~29 ms/layer;
  28 layers ≈ 812 ms/token, far slower than keeping CPU layers resident.
- For 135M layers (~2.3–2.6 MB), H2D is ~0.4 ms/layer; 31 layers ~12 ms versus
  a ~2 ms token at 488 tok/s. Dense per-token staging is also not viable.

For a 500B Q4 model (~281 GB), a full cold read at measured storage bandwidth:

- 281 GB / 1.6 GB/s ≈ 176 s/token (~0.0057 tok/s).
- 281 GB / 3 GB/s ≈ 94 s/token (~0.011 tok/s).

These are lower-bound-like storage limits, excluding compute and synchronization.
Dense staging cannot overcome them. MoE can help only if active experts are
much smaller and reused; that requires routing metadata and selective residency.

## Failure-state model

| Failure | Responsible layer | Safe behavior |
|---|---|---|
| Host staging allocation failure | Native staging allocator | Fail closed; no inference restart; release partial buffers. |
| Pinned allocation failure | CUDA staging allocator | Fall back to pageable only if explicitly configured; otherwise error. |
| CUDA allocation failure | CUDA staging allocator | Destroy staging objects; keep model/context unchanged; error. |
| Partial layer allocation | Native staging manager | Release all new buffers; do not execute the layer. |
| Transfer/event failure | CUDA staging layer | Synchronize/destroy events and buffers; discard context if graph state is uncertain. |
| Device loss/reset | CUDA runtime | Discard CUDA context; do not automatically retry on CPU in the same session. |
| Cancellation during transfer | Staging layer | Finish or safely cancel the transfer, then release; never reuse before completion. |
| Cancellation during graph | llama/RAMforge | Existing cooperative cancellation between decodes; partial output is not a successful result. |
| Unload with outstanding work | llama context | Synchronize before destruction; never free staging under an active graph. |
| Buffer reuse before event completion | Staging layer | Forbidden; wait on the event. |

Fail closed: no silent placement switch, no partial response presented as
success, no reclamation while tensors may be referenced.

## Required instrumentation for staging decisions

Necessary first:

- transferred bytes and duration per stage operation;
- effective H2D/D2H bandwidth;
- staging buffer occupancy/high-water;
- allocation/reuse/failure counts;
- event wait and synchronization duration;
- per-stage/queue wait;
- cancellation latency.

Later:

- per-layer graph execution duration;
- storage read duration per layer;
- pinned-copy duration;
- peak VRAM and host staging memory.

Current observability provides device and per-buffer snapshots, but transfer
and allocation-event counters are unsupported. Transfer instrumentation must be
built with the staging primitive, not inferred from wall-clock timings.

## Implemented native foundation

Read-only tensor storage introspection was added:

- `llama_model_tensor_storage_count/_entry`;
- tensor name, buffer type, device, quantization type, `blk.N` layer, stable
  buffer index, host/file-backed flags, tensor bytes/offset, buffer bytes;
- forgeCORE `Model::tensor_storage()`;
- RAMforge observability schema 3 aggregates per layer/device/buffer.

Physical validation on SmolLM2-135M, full GPU:

- 273 tensors, 2 model buffers;
- layer-0 bytes 2,596,608 across 9 tensors;
- token-embedding/non-layer CPU-mapped 30,081,024 B;
- all 30 transformer layers and output assigned to CUDA0.

This is read-only. It does not allocate, move, rebind, or release tensor data.

## Recommended first controlled-staging architecture

Minimum native primitives, in order:

1. **Per-layer allocation ownership**: allocate layer weight tensors in
   dedicated buffers with explicit retain/release and stable identities.
2. **Layer-boundary execution hook**: a pre/post-layer callback in
   `llama_context`/scheduler that can pause safely, accept a staged buffer, and
   synchronize only at the boundary.
3. **Tensor rebind or per-layer graph construction**: a safe way to point graph
   nodes at the active layer buffer without invalidating graph/tensor
   references, or rebuild a small graph per layer.
4. **Bounded staging primitive**: host-pinned + device buffers, private CUDA
   stream, CUDA events, occupancy/backpressure, cancellation, and rollback.
5. **Transfer/allocation instrumentation** at the staging primitive.
6. **Failure and recovery tests** covering allocation, transfer, event, cancel,
   unload, and device error.

Rust remains responsible for budgets, planning, policy, profiles, and service
contracts. C++ owns native buffers and graph hooks. CUDA code owns streams,
events, pinned memory, and transfers.

## Implemented primitives (this milestone)

`forge-staging` (forgeCORE) provides a handle-based C ABI with runtime CUDA
driver loading. CPU-only builds compile without CUDA headers; engine creation
returns `Unsupported` when `libcuda` is unavailable.

### Buffer ownership

- Memory types: pageable host, CUDA-pinned host, CUDA device.
- Requested capacity is explicit; allocated size is reported.
- Handles are never reused; released handles remain queryable as released.
- Allocation is bounded by the request; no fallback conversion.
- State: empty, allocated, in-flight, released, failed.
- Release is refused while in flight or while an engine still owns buffers.
- Double release is reported as `ALREADY_RELEASED`.
- Engine destruction is refused while buffers or transfers remain.

### Transfer engine

- Private CUDA context, non-blocking stream, start/end events.
- Single in-flight operation in the prototype.
- H2D/D2H only with correct buffer types.
- Byte range validation and buffer-in-flight exclusion.
- Submission duration, wait duration, event duration, byte counts, counters,
  high-water bytes.
- Cancellation during an active copy is explicitly refused; the caller must
  wait. No device-wide synchronize is used.
- On a CUDA submission/event failure the engine is marked unusable and rejects
  further work.

### Tensor/layer integration

- Read-only tensor storage introspection.
- `Model::layer_staging_plan(layer)` groups `blk.N` tensors and bytes.
- `Model::read_tensor_bytes` copies a bounded tensor range while quiescent.
- No tensor pointer/buffer rebinding; no inference consumption.

### Measured transfer matrix

Pinned and pageable transfers, 5 measured iterations after warmup, RTX 3050,
private staging context:

| Size | Type | H2D median | H2D p25–p75 | D2H median | Effective H2D | Effective D2H |
|---:|---|---:|---:|---:|---:|---:|
| 2.60 MB | pinned | 534 µs | 484–564 µs | 569 µs | 4.86 GB/s | 4.56 GB/s |
| 2.60 MB | pageable | 459 µs | 446–497 µs | 374 µs | 5.66 GB/s | 6.94 GB/s |
| 16 MiB | pinned | 3.53 ms | 3.51–4.40 ms | 3.46 ms | 4.76 GB/s | 4.86 GB/s |
| 16 MiB | pageable | 4.12 ms | 3.50–4.84 ms | 3.55 ms | 4.07 GB/s | 4.73 GB/s |
| 64 MiB | pinned | 12.49 ms | 12.23–13.29 ms | 13.44 ms | 5.37 GB/s | 4.99 GB/s |
| 64 MiB | pageable | 19.40 ms | 16.81–20.16 ms | 14.37 ms | 3.46 GB/s | 4.67 GB/s |
| 183 MiB | pinned | 40.51 ms | 40.14–41.06 ms | 41.95 ms | 4.74 GB/s | 4.57 GB/s |

These are lower than the earlier ad-hoc 6.3/6.37 GB/s maxima. The matrix uses a
private context, single stream, and includes event/stream overhead. Pageable
submission is expensive (0.26–19 ms) because the driver stages it internally;
pinned submission is 4–32 µs. Small-size results are noisy and are not used for
bandwidth claims.

Engine stats snapshot overhead: 1,000 snapshots in 78.7 µs (~79 ns each).

### Single-layer prototype

SmolLM2-135M layer 0: 9 tensors, 2,596,608 bytes.

- Source bytes read from the model into a pinned buffer.
- H2D to a device buffer, event wait.
- D2H to a second pinned buffer, event wait.
- FNV-1a source and readback hashes matched.
- H2D 6.53 GB/s, D2H 5.41 GB/s in this run.
- Buffers released; counters show zero current bytes.
- Model remained usable; a context opened successfully afterward.

**The graph did not consume the staged bytes.** This is a transfer/ownership
prototype, not inference staging.

## Reproduction

```sh
export FORGE_LLAMA_DIR=/path/to/native
export FORGE_LLAMA_BUILD_DIR=build-cuda
export LD_LIBRARY_PATH="$FORGE_LLAMA_DIR/build-cuda/bin"

cd forgeCORE
cargo test -p forge-staging
cargo test -p forge-staging -- --ignored transfer_benchmark_matrix --nocapture
FORGE_TEST_MODEL=/path/to/SmolLM2.gguf \
  cargo test -p forge-core --test staging_prototype -- --ignored --nocapture

cd ../RAMforge
cargo build --release -p ramforge-cli
./target/release/ramforge staging-probe /path/to/SmolLM2.gguf --layer 0 --json
```

The CLI probe intentionally reports `inference_consumed_staged_data=false`.

## Graph-integrated staged inference (implemented)

Selected architecture: **hybrid layer residency with native graph rewrite**.

1. The model is loaded with CPU weights (`gpu_layers=0`) but with both CPU and
   CUDA devices exposed to the context. No model layer is offloaded by the
   normal loader.
2. `llama_context_stage_layer_cuda(layer)` allocates all `blk.N` tensors in an
   independent CUDA buffer through the context's own CUDA backend. Tensor
   objects are cloned with the same names, types, shapes, and quantization;
   no live tensor/buffer pointer is rebound.
3. Each staged tensor's device pointer and byte size are exposed through a
   read-only C descriptor.
4. forge-staging uses the CUDA primary context and writes source bytes directly
   into those device pointers through its private non-blocking stream/event
   path. No intermediate llama copy is used.
5. After a new graph is built and before scheduler allocation,
   `llama_context::apply_staged_layers` replaces graph node weight sources by
   exact tensor name. The scheduler then sees CUDA-backed weight tensors.
6. Substitution counts are reported by layer. This proves the graph referenced
   the staged allocations, not just that bytes were copied.
7. Staged buffers are owned by the context and released when the context is
   destroyed. There is no per-token eviction or buffer reuse yet.

This preserves llama.cpp/ggml graph construction, scheduling, quantized ops,
and kernels. It does not implement a parallel inference engine.

### SmolLM2-135M staged inference evidence

- Layer 0: 9 tensors, 2,596,608 bytes.
- Graph substitutions: 9 per graph (18 over warmup + measured graph).
- Reference/staged argmax: equal (198).
- Logit cosine similarity: 0.9933.
- Max logit difference: 8.996 (CUDA vs CPU quantized-kernel accumulation;
  ranges differed, but argmax was stable).
- Generated service text matched the CPU baseline:
  ` Paris. Paris is a city that is`.
- Corrupting `blk.0.attn_q.weight` in the staged copy changed logits by 2.79,
  demonstrating that the staged bytes were consumed rather than ignored.
- Staged allocation was 2,598,964 bytes of CUDA storage.

### Qwen2.5-7B staged inference

- Layer 0: 12 tensors, 149,112,832 bytes.
- Graph substitutions: 24 over the measured graph.
- 4 generated tokens, 5.84 tok/s, TTFT 326.4 ms.
- Model load 2.18 s; remaining 27 layers stayed CPU-resident.
- This demonstrates one staged layer on a larger model, not 7B staging
  scalability.

### Performance comparison

Release, same 32-token workload:

| Mode | Median tok/s | P25–P75 | P95 |
|---|---:|---:|---:|
| CPU | 265.72 | 262.01–273.77 | 277.77 |
| Staged layer 0 | 269.04 | 268.67–270.20 | 273.22 |
| Full GPU (31 layers) | 492.19 | 462.39–496.30 | 500.83 |

Staging one layer on this small model is not a throughput improvement; it is
within CPU noise. The value is architectural: a bounded, explicitly managed
CUDA layer allocation was consumed by the real graph.

### Current limits

- At the single-layer milestone, staged layers were static for the context
  lifetime. Dynamic invalidation, eviction and replacement are implemented
  later in this document.
- No double buffering or transfer/compute overlap (still unsupported).
- CPU KV and CPU residual tensors cause scheduler copies around staged layers.
- Multi-layer and budgeted staging are validated in the next section.
- Staged tensors are allocated through the context CUDA backend, not the
  staging engine; the engine transfers into their exposed device pointers.
- Graph lifecycle phases and counters are now exposed; cancellation and unload
  still destroy staged buffers safely with the context.

## Multi-layer budgeted residency (implemented)

Rust planning + native allocation/substitution now support multiple layers.

- Explicit `--staged-layer N` requests are sorted/deduplicated and treated as an
  all-or-nothing set.
- `--staged-auto` selects layers in ascending order within the effective budget.
- `--staged-budget` is a hard limit on RAMforge-owned staged CUDA allocations.
- `--staged-margin` (default 256 MiB) is subtracted from observed free VRAM.
- Effective budget = minimum of configured budget and observed-free-minus-margin
  when both are known.
- Exact per-layer bytes are obtained from native CUDA buffer allocation-size
  accounting before any allocation.
- Explicit requests that do not fit are rejected as a set with a serialized
  plan; no partial staging and no fallback.
- Auto requests may be partial; accepted/rejected layers and bytes are reported.

Plan states:

- requested layers;
- configured/effective budget;
- observed free VRAM;
- planned bytes;
- accepted and rejected layers;
- exact/estimated/partial/rejected/unknown state;
- `eviction_supported=true`;
- `allocation_reuse_supported=true`;
- `transfer_compute_overlap_supported=false`.

### Physical multi-layer evidence

SmolLM2-135M, layers 0 and 1:

- 2 layers, 9 tensors each, 2,598,964 bytes each;
- planned/allocated 5,197,928 bytes;
- 18 graph substitutions per layer over warmup + measured graph;
- generated text identical to CPU:
  ` Paris. Paris is a city that is`.

Auto selection with a 10M budget accepted layers 0–3 (10,126,160 bytes) and
rejected the rest as `partial`. A 1M explicit request was rejected as a set
before staging.

Qwen2.5-7B Q4_K_M, auto budget 1G:

- accepted layers 0–6, 7 layers, 989,857,792 bytes;
- per-layer CUDA bytes 131–149 MB;
- 24 graph substitutions per layer;
- 21 layers rejected by budget (`partial` plan);
- 4.40 tok/s, TTFT 329.2 ms, load 2.73 s;
- generated ` Paris. Which of` for a 4-token request.

## Graph lifecycle and invalidation contract (implemented)

### Lifecycle audit

- `llama_context::process_ubatch` either reuses the previous `llm_graph_result`
  or resets it and calls `model.build_graph`.
- `llm_graph_result::reset()` destroys and reinitializes its `ggml_context` and
  graph. Every graph node and every substituted `src` pointer lives in that
  context, so resetting it removes all cached graph references to staged
  tensors.
- `gf_res_prev_active` is the reusable graph marker. It is set only after a
  graph is allocated and cleared by reset paths.
- `ggml_backend_sched_reset()` clears the scheduler hash table, backend-id and
  copy tables, and allocation state. `ggml_backend_sched_split_graph()` rebuilds
  the scheduler-owned split context on the next allocation; split inputs are not
  read while `n_splits == 0`.
- `synchronize()` waits on every scheduler backend and releases the execution
  gate. It is the narrowest existing quiescence operation; no device-wide
  synchronization is added.
- The reserve graph (`gf_res_reserve`) is built without staged substitution and
  therefore never references staged buffers.

### Native contract

`llama_context_invalidate_graph()` performs, in order:

1. mark the graph `invalidating`;
2. quiesce with `synchronize()` (backend synchronization plus execution gate);
3. clear `gf_res_prev_active` and reset every cached graph result;
4. `ggml_backend_sched_reset()` to drop scheduler mappings/copies;
5. mark the graph `invalidated` and record that no cached graph can consume the
   current staged sources.

`llama_context_evict_layer_cuda(layer)` refuses with a typed `busy` result if
any graph is active, execution is in flight, or invalidation has not completed.
After success it erases the owning `ggml_context` and CUDA buffer. Duplicate
eviction returns `not_staged`.

`llama_context_stage_layer_cuda()` likewise refuses when a cached graph may
reference staged sources.

The state is exposed through `llama_context_staging_state()` and resident
layers through `llama_context_staged_layer_count/at()`.

### Lifecycle states

Graph phases: `idle`, `building`, `prepared`, `executing`,
`completion_pending`, `quiescing`, `invalidating`, `invalidated`, `rebuilding`,
`failed`, `destroying`.

Layer states are represented by allocation/eviction operations and native
counters: unallocated, allocated/filling (RAMforge transfer), resident,
referenced by graph, execution in flight, quiescent, evicting, released, failed.
Only invalidation plus quiescence permits the evicting/released transition while
the context remains alive.

### Dynamic replacement and rollback

RAMforge replacement is transactional:

1. build a plan from current resident layers, exact native per-layer bytes, the
   effective budget and observed free memory;
2. reject an unfitting explicit set before touching the context;
3. invalidate the graph and scheduler;
4. evict layers not in the accepted set;
5. stage newly accepted layers by allocating, copying exact tensor bytes and
   waiting for each transfer;
6. leave the context invalidated so the next decode rebuilds the graph and
   substitutes the new sources.

On a staging failure, newly staged layers are evicted and every missing old
layer is re-staged. If rollback succeeds, the old selection and context remain
usable. If rollback also fails, the service enters `failed` and rejects further
generation. A rejected plan never mutates residency.

### Allocation reuse

Not implemented. Eviction frees the CUDA buffer. Reusing capacity would require
proving identical tensor inventory, type, shape and alignment for the incoming
layer and remapping every staged tensor view; this milestone does not fake it.
`reuse_count` is always 0 and plans report
`allocation_reuse_supported=true`.

### Dynamic evidence

SmolLM2-135M, one live context, layers 0+1 replaced by 1+2:

- evicted layer 0, kept layer 1, staged layer 2;
- plan `exact`, 5,197,928 B planned, 2,326,964 B reclaimed, 2,329,268 B newly
  allocated;
- replacement: invalidation 0.019 ms, eviction 0.137 ms, staging 3.028 ms,
  total 3.188 ms (medians over five cycles);
- generated text coherent and stable across cycles;
- five alternating cycles: invalidation count 5, release count 5, allocation
  count 7, reuse count 0, resident layers stable;
- corruption of `blk.1.attn_q.weight` after replacement changed logits by
  10.25 with 9 substitutions, proving the replacement tensors are consumed.

Qwen2.5-7B Q4_K_M, one live context:

- initial layers 0–6: 989,857,792 B;
- replaced with layers 7–13: 953,903,104 B;
- 7 evictions, 7 allocations;
- invalidation 0.03 ms, eviction 7.84 ms, staging 1,901.56 ms, total
  1,909.43 ms;
- replaced back to layers 0–6 in about 2.0 s;
- coherent output throughout.

### Transfer/compute overlap blocker

Staging still completes before whole-graph execution. There is no native
per-layer execution boundary or double-buffered scheduler consume point, and
the public CUDA copies are synchronous. Dynamic replacement happens between
executions, not during them. Asynchronous submission alone is not overlap.

Overlap therefore remains unsupported and unmeasured.

## Execution pipeline, double buffering and overlap (implemented)

### Execution-boundary finding

The pinned scheduler executes a whole graph:
`ggml_backend_sched_alloc_graph()` allocates the complete graph and
`ggml_backend_sched_graph_compute_async()` submits it. `cb_eval` observes nodes
but cannot pause or split a graph safely. Graph nodes are built by llama.cpp and
depend on the full topology; manually executing transformer-layer fragments
would require a new scheduler contract and risks violating backend/dependency
semantics.

The selected design is therefore an **inter-execution pipeline**, not an
intra-graph layer split:

- the active selection executes normally;
- a complete next selection is allocated in a separate pending slot;
- its weights are copied into that slot by one asynchronous CUDA batch on a
  private stream while the active graph computes;
- after execution completion the graph is invalidated and the pending slot is
  committed atomically;
- the next decode rebuilds the graph against the new selection.

Per-layer intra-graph execution remains unsupported.

### Allocation pool

`llama_context_set_staging_pool_limit()` bounds retained evicted buffers.
Eviction moves a CUDA buffer into a best-fit pool instead of freeing it. When a
new layer requires no more capacity, the matching buffer is reused and the
tensor metadata context is recreated over it. Pool bytes, buffer count and
reuse count are reported natively. Buffers are only pooled after invalidation
proved that no graph or scheduler mapping references them. The pool never
exceeds its configured limit; reduction frees excess buffers immediately.

SmolLM2 evidence: allocation count stayed at 4 across five pipeline cycles while
`reuse_count` advanced 0→2→4→6→8. Non-pipelined replacement with a 20M pool also
stabilized allocation count at 3 with reuse counts 1,1,2,3.

### Double-buffered pending slot

`llama_context_stage_layer_pending_cuda()` allocates layers in a separate
pending vector. Pending tensors are not substituted into the active graph, so
pending staging is allowed while the active selection executes. Commit requires
`staged_graph_invalidated == true`, no in-flight execution and no active graph;
it recycles the old active buffers and moves the pending vector into the active
selection. Discard releases pending allocations without changing the active
selection.

Corrupting `blk.1.attn_q.weight` in the pending slot before commit changed logits
by 10.25 with 9 substitutions after commit, proving the committed pending
weights are consumed by inference.

### Asynchronous transfers and events

`fstage_transfer_submit_batch_to_ptr()` enqueues all layer tensor copies on one
non-blocking staging stream, records CUDA start/end events and returns after
submission. `fstage_transfer_wait_started()` waits only for the start event, so
compute can begin once the DMA has begun. `query()` reports completion without
blocking; `wait()` completes with device-event duration and host wait time.
The staging stream is created at the lowest CUDA stream priority so compute
kernels are preferred where the driver honours priorities. Cancellation never
releases an in-flight source; shutdown waits or fails closed.

### Overlap evidence and method

Transfer duration is measured by CUDA events on the staging stream. Compute
duration is measured on the host around a generation that explicitly
synchronizes. The transfer is confirmed to have started before compute via
`wait_started`, and the prepare→compute host gap is measured and subtracted from
the overlap bound. This is a conservative device-event/host-synchronization
bound, not an asynchronous-API claim.

SmolLM2, 16 generated tokens, 5 cycles:

- transfer 0.94–1.31 ms, compute 70–78 ms, commit wait 0;
- transfer 100% hidden, overlap estimate equal to the full transfer duration;
- allocation count constant, reuse advanced every cycle.

Qwen2.5-7B Q4_K_M, four layers per slot, 2 generated tokens, 3 cycles:

- median transfer 81.3 ms hidden inside median compute 443.7 ms, commit wait 0;
- overlap estimate 100% of transfer;
- resident slot 560–578 MB, pool 560–578 MB, reuse 4 per cycle after the first.

### Performance result

The overlap is real but does not improve end-to-end latency on this hardware.
In a controlled comparison using the same batch staging path:

- overlapped median (prepare + compute + commit): 790.2 ms;
- commit-before-compute median: 728.1 ms.

Concurrent DMA reduced compute throughput by roughly the hidden transfer time,
consistent with shared memory bandwidth on the laptop GPU. The pipeline is
therefore an experimental capability, not a throughput win, and is not the
default. The required transfer/compute concurrency is demonstrably present; the
end-to-end benefit is not.

### Remaining overlap blockers

- No per-layer graph execution boundary; overlap is between executions.
- Transfer and compute share device memory bandwidth; no measured net speedup.
- No bandwidth isolation or copy-engine prioritization beyond stream priority.
- Pipelined execution requires both slots to fit the effective budget.

## Adaptive scheduling, calibration and residency policy

### Resource measurements

The physical RTX 3050 was calibrated with pinned and pageable transfers at
64 KiB–128 MiB. Large pinned transfers measured roughly 6.3–7.1 GB/s; pageable
transfers roughly 4.7–5.7 GB/s. Event overhead at 64 KiB was about 8–21 µs.
Allocation/release cost and chunking were measured separately.

Chunking a 64 MiB payload produced medians of 9.75 ms for 4 MiB × 16 batches,
9.98 ms for 16 MiB × 4 and 10.0 ms for one 64 MiB batch. The difference is
within run-to-run noise; the coarse batch path is retained.

### Contention diagnosis

The overlapped Qwen path hid 80–100 ms of transfer inside 350–500 ms of compute,
but compute slowed by a similar amount. This is consistent with shared device
memory bandwidth and copy/compute contention. It is not proven by hardware
counters because no external profiler is installed; the evidence is:

- calibration shows the transfer rate is bandwidth-bound at the measured sizes;
- overlapped compute is consistently slower than commit-before-compute;
- low-priority staging streams did not remove the slowdown;
- disabling overlap restores compute duration.

No causal claim beyond shared-bandwidth contention is made.

### Adaptive policy

`AdaptiveController` combines calibration with in-session measurements. It
interleaves exploration, requires two samples per strategy, applies profile
hysteresis, and falls back to synchronous deterministically. Calibration
identity mismatch, missing compute estimates and Background profiles all select
synchronous replacement. Explicit overrides are reported as advisory.

Profiles:

- Maximum Performance: 2% hysteresis, may select measured async if faster;
- Balanced: 5% hysteresis, prefers synchronous unless clearly faster;
- Background: never concurrent prefetch;
- Custom: explicit override or synchronous fallback.

### Residency policy

Auto-selection can be ordered ascending, by smallest exact layer bytes, or
resident-first. On Qwen2.5-7B with a 1 GiB staged budget and warm page cache,
smallest-first selected 8 layers (1,049,083,904 B) versus 7 ascending
(989,857,792 B). Five measured runs: 6.418 vs 6.161 tok/s median (P25 6.227 vs
6.107, P75 6.423 vs 6.173), and TTFT 212.5 vs 221.5 ms. The gain is modest and
model/workload-specific; ascending remains the default and the policy is opt-in.

### Calibration state

Records are `missing`, `identity_mismatch` or `valid`; schema/tool version
mismatches fail loading. Calibration is explicit, bounded and never runs at
startup. It is tied to device, driver, backend, model path/size/mtime and tool
version.

## Host-side staging optimization

The previous pipeline read each tensor into a pageable `Vec`, then copied it
into pinned host memory, then submitted H2D. On Qwen2.5-7B, four layers
(560,023,552 B) cost 209–276 ms host preparation, while H2D was ~80 ms.

Implemented:

- positional `pread` directly into a pinned staging buffer
  (`fstage_buffer_read_fd`), with EINTR, short-read and error handling;
- exact tensor-range coalescing using native GGUF mappings, including split
  files; non-adjacent ranges are never over-read;
- a bounded best-fit pinned host pool with explicit idle limit, reuse counters
  and peak accounting;
- cached tensor mappings (the mapping enumeration itself cost ~100 ms per
  prepare);
- asynchronous host reads on a worker thread that owns the pinned buffer until
  join, overlapping file-to-pinned work with generation.

Measured on Qwen2.5-7B, four layers per slot, warm page cache:

- copy read/prepare: 137 ms read, 276 ms preparation;
- direct per-tensor: 80.5 ms read, 190 ms preparation;
- direct coalesced: 69.7 ms read, 179 ms preparation;
- coalesced groups: 2 (48 tensors, 560 MB);
- pinned pool with two retained slot sizes: 2 allocations, 3 reuses, ~709 MiB
  idle, no unbounded growth;
- async host read: prepare returns in ~0.3 ms with a warm pool; the read is
  fully hidden and commit reports zero read wait.

End-to-end, order-controlled five-cycle Qwen runs with 4 generated tokens:
synchronous median 728.4/758.2 ms; async host-read median 691.4/687.2 ms,
about 7% lower end-to-end. Compute variance remains and is reported; the
improvement is the hidden 70 ms host read and reduced preparation stalls.

Split GGUF verification: layers 20–23 (560,496,640 B, 3 coalesced groups)
staged with direct reads and generated coherent text; direct-read bytes were
compared against model tensor reads for two distinct shard files.

This does not change graph consumption: the prepared bytes are copied into the
same pending CUDA tensors and committed through the existing invalidation and
corruption-controlled path.

## Bounded storage pipeline and prediction

### Storage backend decision

The target has NVMe storage and `io_uring` runtime libraries, but no liburing
development headers. A raw-syscall io_uring backend was not added: the validated
coalesced `pread` path already reads Qwen layer data at roughly 8 GB/s from a
warm page cache, and no evidence showed syscall overhead as the limiter. The
fallback remains explicit and portable.

### Bounded multi-request reads

`read_file_groups` supports bounded parallel workers with:

- configurable concurrency;
- a maximum outstanding-byte limit, enforced with a condition variable;
- deterministic queue order and error propagation;
- a single-group admission exception when no other request is active;
- file descriptors cached per worker;
- exact disjoint host offsets.

Measured Qwen four-layer reads (warm cache, one cycle each):

| Batching | Concurrency 1 | 2 | 4 |
|---|---|---|---|
| Coalesced | 59.9 ms | 61.0 ms | 83.5 ms |
| Per-tensor | 67.9 ms | 67.1 ms | 59.9 ms |

Coalesced reads do not benefit from more concurrency; 2–4 workers add
contention. Per-tensor concurrency 4 improves that path, but remains no better
than coalesced concurrency 1. The default stays coalesced with concurrency 1;
advanced users may raise concurrency. A unit test verifies byte-exact output
and that a 2048-byte outstanding limit prevents two 2048-byte groups from being
in flight simultaneously.

### Predictive prefetch

Prediction is application-level and explicit: pipeline mode prepares the known
next selection in the pending slot while the current selection generates. No
fabricated per-layer execution-order signal is used. The prefetch window is one
complete pending selection, bounded by the existing active/pending VRAM budget.

`status.prefetch` reports prepared/committed/aborted counts, prefetched,
committed and wasted bytes, and hit-rate permille. A three-cycle SmolLM2 run
prefetched and committed 15,040,512 bytes with no waste. Aborting a prepared
selection records wasted bytes and lowers the hit rate; the integration test
asserts one abort plus one commit yields 500 permille.

### Resource bounds

- Storage concurrency and outstanding bytes are explicit.
- Pinned memory is bounded by the pinned-pool limit.
- Active and pending VRAM remain bounded by the effective staged budget.
- Backpressure is provided by worker count, outstanding-byte waiting and pinned
  acquisition; no unbounded queue exists.
- Cancellation waits for in-flight `pread` workers; buffers are recycled only
  after join.

## Partial residency and delta transactions

### Native contract

`llama_context_commit_pending_delta_cuda(remove_layers, n_remove)` validates the
entire delta before mutation:

- the graph must be invalidated and quiescent;
- every removal must exist and be unique;
- pending layers must not already be active unless also removed;
- removals are applied in descending index order, recycling buffers into the
  bounded pool;
- pending layers are appended to the surviving active selection.

The active graph is never mutated. Shared active `llama_staged_layer` objects,
tensor metadata, and device pointers survive unchanged. Failure before commit
leaves the active selection usable.

### RAMforge delta planning

The existing plan already exposes `kept_layers`, `newly_staged_layers` and
`evicted_layers`. Partial mode uses those directly:

- preserve `A ∩ D`;
- stage only `D \ A`;
- remove `A \ D`;
- keep the active set untouched until transfer/host completion and invalidation;
- commit the delta atomically.

Pipeline budget uses `active + new + pool`, not `active + complete next
selection + pool`. Removal-only transitions allocate no host buffer, read no
storage, and transfer no bytes.

### Measured Qwen transition

Selections `[0,1,2,3] → [2,3,4,5]`, four layers per slot:

| Mode | Read bytes | New bytes | Preserved bytes | Transition median | Allocations |
|---|---|---|---|---|---|
| Full selection | 560.5/578.5 MB | 560.5/578.5 MB | 0 | 82.5–87.6 ms | 8 |
| Partial delta | 280.2/298.2 MB | 280.2/298.2 MB | 280.2 MB | 42.1–44.2 ms | 6 |

- Storage read and H2D bytes approximately halved.
- Shared two-layer allocations and graph metadata preserved.
- Transition median reduced by roughly 48–52% after warmup.
- End-to-end five-cycle medians with 4 generated tokens varied with compute:
  717.8–880.7 ms full versus 640.3–687.6 ms partial (roughly 4–11% lower).
- Pool reuse counters advanced; total allocations fell from 8 to 6.
- Removed layers are recycled only after invalidation; preserved layers are
  never re-read or re-transferred.

### Failure behavior

- Failed pending allocation/storage leaves active residency unchanged; the
  integration test injects a pending-layer failure and verifies the active set.
- Abort discards pending allocations and preserves the active set.
- Concurrent prepare is rejected.
- A removal-only commit is accepted with zero transfer bytes.
- Native test verifies the preserved layer's device pointer is identical before
  and after delta commit and that inference still consumes the new layer.

### Identity and limits

Delta interchangeability is valid only within one live context/model instance.
Layer indices are not treated as cross-model identities. Model identity is
fixed by the service/model instance; no cross-model reuse is implemented.

The prefetch window remains one pending selection. Partial mode removes duplicate
reads for shared layers but does not implement arbitrary lookahead or
reprioritization of obsolete in-flight reads beyond abort.

## Decision

**PARTIAL RESIDENCY VALIDATED — SHARED LAYERS PRESERVED, ONLY DELTAS READ AND
TRANSFERRED, TRANSITION LATENCY ROUGHLY HALVED — END-TO-END GAIN REMAINS
MODEL/COMPUTE-DEPENDENT.**

- Direct-to-pinned reads eliminate the pageable intermediate copy.
- Coalesced GGUF ranges reduce read calls and read time.
- A bounded pinned pool prevents repeated large pinned allocations.
- Cached tensor mappings remove a recurring ~100 ms preparation cost.
- Async host reads hide file-to-pinned work under generation.
- Split GGUF direct reads and real graph consumption are verified.
- Partial residency halves transition bytes/latency on overlapping Qwen
  selections, but end-to-end generation improvement is smaller and noisy.
- Parallel per-tensor reads help only marginally; coalesced sequential remains
  the default.
- Transfer/compute overlap remains slower than synchronous on this hardware.

- Calibration measures pinned/pageable bandwidth, latency, chunking and
  allocation overhead with explicit identity and invalidation.
- Adaptive scheduling explores and measures both paths, then deterministically
  falls back to synchronous when no clear win exists.
- Profile hysteresis changes selection behavior; Background never prefetches.
- Smallest-first residency selected one more Qwen layer and measured ~4% higher
  generation throughput (P25/P75 non-overlapping in the five-run sample).
- Chunking showed no material benefit; the coarse batch remains.
- Adaptive overlap scheduling did not demonstrate a reliable end-to-end win and
  remains opt-in.

Per-layer intra-graph execution, profiler-backed contention attribution and a
universal adaptive default remain unsupported.
