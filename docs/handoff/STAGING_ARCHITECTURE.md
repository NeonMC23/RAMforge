# Controlled GPU staging architecture (investigation)

Status: **architecture and feasibility; no staging engine implemented.**

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
- RAMforge observability schema 2 aggregates per layer/device/buffer.

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

## Decision

**READY AFTER SPECIFIC NATIVE PRIMITIVES.**

The architecture is understood and the read-only ownership foundation exists,
but controlled staging cannot be implemented safely on the current public API.
The next milestone should implement primitives 1–4 with narrow C ABI tests and
must not claim overlap or throughput before measuring real transfers.
