# Large-model execution architecture and feasibility

This document records the native capability audit and the technical path toward
models whose weights exceed RAM/VRAM. It separates measured capability from
estimates and proposed extensions.

## Native backend capability matrix

Audited pin: `7fe450e19305b828c199d602c23a8337aaa1f03b`.

Build flags from the actual CMake cache:

| Capability | Build state | Evidence |
|---|---|---|
| CPU backend | ON | `GGML_CPU:BOOL=ON` |
| CPU repacking | ON | `GGML_CPU_REPACK:BOOL=ON` |
| CUDA | OFF | `GGML_CUDA:BOOL=OFF` |
| Vulkan | OFF | `GGML_VULKAN:BOOL=OFF` |
| SYCL | OFF | `GGML_SYCL:BOOL=OFF` |
| OpenCL | OFF | `GGML_OPENCL:BOOL=OFF` |
| Metal | OFF | `GGML_METAL:BOOL=OFF` |
| RPC backend | OFF | `GGML_RPC:BOOL=OFF` |
| Dynamic backend loading | OFF | `GGML_BACKEND_DL:BOOL=OFF` |

Available native API surface at the pin (line numbers refer to the pinned
checkout; the page APIs are local patches):

| Capability | State | Evidence |
|---|---|---|
| Backend/device registry | available | `ggml/include/ggml-backend.h:241–244`, `ggml_backend_reg_*` |
| Registered backend names/counts | available | `ggml/include/ggml-backend.h:236–238`; forgeCORE `registered_backends()` |
| Device properties and memory | available | `ggml/src/ggml-backend.cpp:589–599` |
| Dynamic backend loading code | present, build OFF | `ggml/src/ggml-backend-reg.cpp:574` |
| mmap / no-prefetch load modes | available | `src/llama-model-loader.cpp:559,830`; `src/llama-mmap.cpp:472` |
| GPU offload parameters | API present, backend not built | `include/llama.h:322–332` |
| Effective layer/output placement readback | available (patch) | `llama_model_effective_gpu_layers`, `llama_model_layer_device_type`, `llama_model_output_device_type` |
| Tensor split / multi-device | API present, backend not built | `src/llama-model.cpp:1208` |
| File-backed tensor mapping metadata | available (patch) | `include/llama.h:665–666`; `src/llama-model.cpp:3285–3292` |
| Page residency sampling | available on Linux | `include/llama.h:691`; `src/llama-model.cpp:3259` |
| Model/tensor page release | available on Linux | `include/llama.h:552,675`; `src/llama-model.cpp:3278,3300` |
| Bounded page prefetch hint | available; hint only | `include/llama.h:684`; `src/llama-model.cpp:3308` |
| Backend tensor set/get/copy | low-level only | `ggml/include/ggml-backend.h:71,85–95` |
| Subset tensor load / per-layer execution | unsupported | no public loader or scheduler API; all tensors are mapped or allocated at load |
| Anonymous/backend buffer eviction | unsupported | no public API |
| Context padding | observed | `src/llama-context.cpp:291` (`GGML_PAD(n_ctx, 256)`) |

Runtime device registry on the development machine contains one CPU device.
No GPU device is selectable in the current process. The physical machine does
contain an NVIDIA RTX 3050 Laptop GPU, but CUDA is blocked: the loaded kernel
module is 595.112.07 while userspace is 620.70.06, and no CUDA compiler/toolkit
is installed. See [GPU_ENVIRONMENT.md](GPU_ENVIRONMENT.md).

### forgeCORE surface

Available and safe:

- `enumerate_devices()` from the ggml backend registry.
- Device name/description/type and native memory total/free.
- `ModelOptions` supports `GpuLayers`, `SplitMode`, `main_gpu`, `tensor_split`,
  and explicit device lists.
- Model/context loading, mmap/no-prefetch modes, CPU repacking, model mapping
  metadata, tensor mapping ranges, page sampling, and effective context/batch/
  thread readback.

Current RAMforge operational adapter:

- uses CPU-only model execution;
- records GPU offload as unsupported for the verified build;
- never infers effective GPU placement from a requested layer count;
- keeps CPU execution fully functional without a GPU.

GPU offload exists in the safe forgeCORE model options but is **not physically
validated** in this build. No CUDA/Vulkan/SYCL/Metal test result is claimed.

## GPU execution status

- Hardware detected: RTX 3050 Laptop GPU (GA107M).
- Native GPU backend built: **no** (all GPU backend flags OFF). A guarded
  `forgeCORE/scripts/setup-native-cuda.sh` is prepared for a separate
  `build-cuda` tree; it was not run because `nvidia-smi` and `nvcc` are not
  usable.
- CUDA toolkit available: **no** (`nvcc` and headers absent).
- Driver usable by CUDA: **no** in this session (kernel/userspace mismatch).
- Physical GPU inference or offload: **not performed**.
- Claimed effective GPU layers: **none**; CPU loads report zero through native
  readback.

RAMforge now records registered backends, `gpu_offload_available`, and device
memory in the machine profile. GPU execution requests (`--device auto` or
`--gpu-layers N`) fail with an explicit error when no GPU backend/device is
registered; they are never converted into a CPU plan. Plans can represent
`cpu`, `gpu`, and `hybrid` modes and mark placement fields advisory, but the
service rejects non-CPU plans because no physical GPU load path has been
validated. No silent CPU fallback occurs.

## Memory hierarchy

`MemoryHierarchy` and `PlacementModel` represent:

- host RAM capacity/availability;
- cgroup limit and configured budget;
- per-device total/free memory;
- storage capacity/availability;
- file-backed model mappings;
- anonymous/process memory;
- KV cache estimate;
- compute/graph allocation (unknown);
- backend buffers (unknown);
- staging (unknown);
- storage-backed model data.

Categories are explicitly overlapping and are never summed into a fabricated
total. Each hierarchy and category carries the machine-profile sampling time;
each device snapshot carries its own sampling time. Quality is observed,
estimated, unknown, or unsupported; provenance is recorded separately. Placement states are:

`gpu_resident`, `host_resident`, `file_backed_mapped`, `storage_backed`,
`staged`, `unknown`. The local service exposes `architecture` and `hierarchy`
commands in addition to `capabilities` and `status`, so application clients can
inspect the same plan-derived architecture, placement, and feasibility data as
the CLI.

Only confirmed states are evidence-backed. CPU KV placement is host; file-backed
mmap is confirmed only when repacking is disabled; compute/graph/backend/staging
remain unknown.

## mmap and page-cache behavior

The native runtime maps model files and executes CPU kernels. The virtual mapping
size, resident mapping bytes, RSS, and physically used memory are distinct
values. `--no-prefetch` selects `MmapNoPrefetch`, which skips the native
whole-file `MAP_POPULATE`; default prefaulted mmap faults the mapping during
load.

Measured out-of-core behavior from the existing validated runs:

- GPT-NeoX-20B Q4_K_M, 13.13 GB tensor bytes, 8 GiB cgroup, `--no-repack`,
  `--no-prefetch`: the process loaded and generated the same output, with a
  7.385 s TTFT, 7.64 GiB of the model mapping resident after decode (62.4%),
  an 8 GiB peak charge, and 25,383,576 512-byte input blocks (about 13 GB).
  The prefetch variant took 16.94 s wall and 50,728,408 blocks (about 26 GB).
- Qwen2.5-1.5B Q4_0 (1.07 GB), 8 greedy tokens: 11.84 tok/s unbounded, 3.29
  tok/s at a 1 GiB cap with `--no-prefetch`, 1.59 tok/s at 512 MiB, and 1.53
  tok/s at 256 MiB. Input-block counters grew from about 2.7 GB to 9.4 GB as
  the cap fell.
- Re-verified on 2026-11-21 with the current binary: the 13.13 GB GPT-NeoX-20B
  Q4_K_M model under an 8 GiB cgroup, `--no-repack --no-prefetch`, one greedy
  token: exit 0, output ` Paris`, model load 184 ms, 10.877 s wall, peak
  charged exactly 8 GiB, 9661 `max` events, 0 OOM kills, and 216,464 page
  faults (257 major). This is a correctness and fit demonstration, not a
  throughput result.

So a model larger than the enforced budget can load and generate, but the work
becomes storage- and reclaim-bound. This is OS demand paging under a cgroup
cap, not controlled streaming. It does not make 500B-scale throughput plausible.

## Page-level staging, release, and prefetch

The pinned revision is patched to expose model-scoped page controls. These are
safe because they operate on existing file-backed mappings and never unmap or
change tensor pointers:

- `llama_model_tensor_mapping_count` / `llama_model_tensor_mapping`
  (`include/llama.h:665`, `src/llama-model.cpp:3285`): exact tensor mapping
  metadata, including file-backed versus anonymous tensors and page bounds.
- `llama_model_sample_tensor_pages` (`include/llama.h:691`): bounded Linux
  `mincore()` residency sampling.
- `llama_model_release_pages` (`include/llama.h:552`): model-scoped
  `madvise(MADV_DONTNEED)`; tensor pointers stay valid and later access
  refaults pages.
- `llama_model_release_tensor_pages` (`include/llama.h:675`): selected-tensor
  page release that conservatively skips pages shared with unselected tensors.
- `llama_model_prefetch_tensor_pages` (`include/llama.h:684`): bounded
  `MADV_WILLNEED` hint. It is not a synchronous read or a residency guarantee.

Measured page release (Qwen2.5-1.5B, 1 GiB, `--no-repack`, 8 tokens,
`--release-mapping-pages`): mapping residency fell from 936.70 MiB to 0 B,
TTFT was 498.0 ms, decode was 3.52 tok/s, and wall was 3.36 s. For the 20B
model, `--no-prefetch` reduced load-time RSS from 7.95 GiB to 35.36 MiB, but
decode then paid the refaults.

RAMforge uses these primitives through bounded staging plans, quiescence gates,
admission decisions, and observe-only reporting. Release and prefetch are
advisory at the OS level; they do not guarantee a working set or a read.

What is still not available:

- loading only selected tensors or layers for a live graph;
- a caller-owned bounded weight buffer consumed by the compute graph;
- per-layer compute/transfer overlap or a layer-boundary hook;
- eviction of anonymous weights or backend buffers;
- synchronous completion and error reporting for a staged transfer.

`ggml_backend_tensor_set/get/copy` exist (`ggml/include/ggml-backend.h:71,85–95`),
but they mutate live buffers and provide no ownership, eviction, or scheduling
contract. Using them per token would not be controlled weight streaming.

A real streaming extension still needs:

1. stable tensor/layer ownership and residency state;
2. a bounded staging buffer owned by the caller/forgeCORE;
3. a scheduler or graph hook that consumes staged data at a synchronization
   boundary;
4. explicit invalidation and lifetime rules;
5. failure, cancellation, and rollback cleanup.

No such tensor-streaming prototype was implemented. The page-level controls are
the safe prototype that exists, and they are OS paging, not inference streaming.

## Dense versus MoE

GGUF metadata can expose expert count and used-expert count. RAMforge records
this as `ModelArchitecture`:

- dense vs mixture-of-experts;
- layer/head/KV-head/embedding/context fields;
- expert count and used count;
- routed-expert fraction in basis points, explicitly a partial estimate because
  shared weights are not separated by metadata.

The native runtime loads model tensors according to the architecture. The
current adapter does not control expert residency or routing. Inactive experts
cannot be assumed absent from storage or process mapping. No MoE model was
available for physical validation; the measured 20B model is dense GPT-NeoX.
MoE-aware placement requires native routing/residency metadata and is not
implemented.

## 500B feasibility arithmetic

Illustrative assumptions, not measurements:

- 500B parameters.
- Q4_K ≈ 4.5 bits/weight → ~281 GB weights.
- Q2_K ≈ 2.625 bits/weight → ~164 GB weights.
- Host DDR bandwidth example: 50 GB/s.
- NVMe example: 3 GB/s.

If every token required streaming the full weight set:

- 281 GB / 50 GB/s ≈ 5.6 s/token → ~0.18 tok/s upper-bound-like estimate.
- 281 GB / 3 GB/s ≈ 94 s/token → ~0.011 tok/s.

A measured-scale anchor: the 20B Q4_K_M model is 13.13 GB and took a 7.385 s
TTFT with about 13 GB of input blocks under an 8 GiB cap. Scaling those bytes
linearly to 281 GB gives about 158 s for a comparable cold token, or roughly
0.006 tok/s. This is an order-of-magnitude consistency check, not a prediction:
it ignores compute, reuse, and the fact that a 281 GB working set cannot be
kept resident.

These figures show why naive full-streaming is not an acceptable solution.
MoE can reduce active weight movement if routing metadata and residency control
exist. A 30B-active subset at Q4 is still ~17 GB, above many consumer VRAM
working sets once KV and compute are included. Without native routing-aware
residency, this is not an implemented strategy.

## What works today

- Real CPU inference for models that fit the CPU/memory working set.
- File-backed mmap execution with OS paging, including measured out-of-core
  execution of a 13.13 GB model under an 8 GiB cgroup cap.
- Bounded page release, page-residency sampling, and prefetch hints with
  quiescence gates and explicit unknown states.
- Registered-backend inventory and explicit GPU-unavailable reporting.
- Native effective layer/output placement readback (zero GPU layers on CPU).
- Effective CPU configuration and throughput measurement.
- Bounded profile comparison, architecture metadata, memory hierarchy, and
  placement reporting.
- Storage assessment and verified copy.
- Honest unsupported/unknown reporting for GPU execution and tensor streaming.

## Required next native work

- Build and validate a GPU backend on compatible hardware.
- Expose effective GPU layer/device placement and allocation failures.
- Add a native layer/tensor staging API with bounded buffers, transfer/compute
  overlap, synchronization, and eviction rules.
- Add MoE routing/expert residency metadata and control.
- Expose anonymous/backend buffer accounting and safe reclamation.
- Measure bandwidth, working-set, and page-fault behavior under additional
  controlled (non-destructive) conditions.

Until those exist, RAMforge must not claim efficient execution of 500B models
or arbitrary models exceeding available RAM/VRAM. The page controls improve
working-set management for mapped models; they are not a 500B streaming engine.
