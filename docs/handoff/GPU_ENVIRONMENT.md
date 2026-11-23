# GPU environment and CUDA validation (2026-11-21)

This document records the resolved NVIDIA environment and the physical CUDA
validation. The previous blocker (module 595 vs userspace 620, no toolkit) was
resolved by the user with a reboot and package installation; this repository did
not perform those system changes.

## Current hardware and software

| Item | Observed value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (GA107M), PCI `0000:01:00.0` |
| Compute capability | 8.6 |
| VRAM | 4096 MiB total; about 3690–3869 MiB free during tests |
| Driver / NVML | 620.70.06 |
| Kernel module | 620.70.06 |
| CUDA toolkit | 13.2.51 |
| CUDA host compiler | `/usr/bin/g++-15` |
| CUDA UMD | 13.5 |
| CPU | Intel Core i5-11400H, 12 logical CPUs |
| Host RAM | 15 GiB |

`nvidia-smi` succeeds. The GUI used about 25 MiB VRAM at audit time.

## Native CUDA build

- Pin: `7fe450e19305b828c199d602c23a8337aaa1f03b` (unchanged).
- Build tree: `native/build-cuda`.
- Artifact: `native/build-cuda/bin/libggml-cuda.so.0.25.1`.
- `nm -D` reports `ggml_backend_cuda_reg`.
- `cuobjdump --list-elf` reports `sm_86` cubins.
- `GGML_CUDA=ON`, `GGML_CPU=ON`, `GGML_CPU_REPACK=ON`.
- `GGML_BACKEND_DL=OFF`; llama links `libggml-cuda.so` directly.
- RUNPATH points at `native/build-cuda/bin`.
- The CPU build in `native/build` is untouched.

Build command:

```sh
cd forgeCORE
FORGE_LLAMA_DIR=/path/to/native sh scripts/setup-native-cuda.sh
```

## Runtime discovery

With `FORGE_LLAMA_BUILD_DIR=build-cuda`:

- `registered_backends()`: `CUDA` (1 device), `CPU` (1 device).
- `enumerate_devices()`: CUDA0 = NVIDIA GeForce RTX 3050 Laptop GPU, 3,974,103,040
  bytes total, about 3,868,590,080 bytes free.
- `llama_supports_gpu_offload()`: true.
- The device is selectable after the adapter GPU-path integration.

Runtime discovery is not inferred from the shared library alone; these values
come from ggml initialization in the running process.

## Physical inference evidence

Model: `SmolLM2-135M-Instruct-Q4_K_M.gguf` (30 transformer layers, 31 including
output).

| Requested layers | Effective | GPU transformer layers | CPU transformer layers | Output device |
|---:|---:|---:|---:|---|
| 1 | 1 | 0 | 30 | GPU |
| 8 | 8 | 7 | 23 | GPU |
| 31 | 31 | 30 | 0 | GPU |

Upstream counts the output layer in `n_gpu_layers`; requesting `N` offloads
`N-1` transformer layers plus the output layer for this model.

Real generation succeeded for CPU, hybrid, and full GPU execution:

- CPU prompt: ` Paris. Paris is a city that is`
- GPU prompt: ` Paris. Paris is the political, cultural`

The first token matches; later greedy tokens can diverge because CPU and CUDA
kernels accumulate quantized operations in different orders. Both outputs are
coherent. GPU cancellation, repeated sessions, unload, and VRAM release were
also exercised.

## Release benchmark

5 measured runs + 1 warmup, 32 generated tokens, 256 context, prompt
`The capital of France is`, CPU threads 4, no repacking, release build.

| Mode | Effective layers | Median tok/s | P25–P75 | P95 | Decode tok/s | Prompt tok/s | TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU | 0 | 252.53 | 238.30–257.28 | 261.10 | 259.74 | 669.6 | 7.51 ms |
| Hybrid | 8 | 317.56 | 315.53–324.41 | 325.21 | 328.60 | 805.0 | 6.26 ms |
| GPU | 31 | 488.09 | 486.21–493.36 | 496.39 | 498.69 | 1449.4 | 3.49 ms |

The first validated session measured CPU 240.70, hybrid 289.13, and GPU 484.65;
a reverse-order session measured CPU 249.62, hybrid 309.53, and GPU 488.89.
The instrumented session measured 252.53 / 317.56 / 488.09. Instrumentation
overhead is below the session-to-session noise. Full offload is about 1.9–2.0x
CPU and hybrid about 1.2–1.3x for this small model. These are single-machine,
warm/uncontrolled results and are not a universal default.

VRAM snapshot (free bytes):

| Mode | Before load | After load | After warmup | After unload |
|---|---:|---:|---:|---:|
| Hybrid 8 | 3,868,590,080 | 3,820,355,584 | 3,755,343,872 | 3,860,201,472 |
| GPU 31 | 3,868,590,080 | 3,763,732,480 | 3,690,332,160 | 3,856,007,168 |

The GPU model used about 178 MiB after warmup contexts for full offload and
about 113 MiB for hybrid. Unload returned near baseline; about 12.6 MiB
remained in CUDA/driver caches. These are native free-memory snapshots, not
peak measurements and not exclusive reservations.

## Memory observability (schema 3)

The native context already maintains model/context/compute byte totals per
backend buffer type. A minimal read-only C bridge exposes them; RAMforge wraps
the values in a versioned observability schema.

Point-in-time schema-6 benchmark snapshot, after warmup:

| Mode | Buffer | Model bytes | KV/context bytes | Compute bytes |
|---|---|---:|---:|---:|
| CPU | CPU_Mapped | 103,668,480 | 0 | 0 |
| CPU | CPU | 0 | 5,898,240 | 51,512,320 |
| Hybrid 8 | CUDA0 | 46,928,640 | 1,376,256 | 50,921,472 |
| Hybrid 8 | CUDA_Host | 0 | 0 | 1,317,888 |
| Hybrid 8 | CPU_Mapped | 103,666,176 | 0 | 0 |
| Hybrid 8 | CPU | 0 | 4,521,984 | 0 |
| GPU 31 | CUDA0 | 103,739,904 | 5,898,240 | 50,921,472 |
| GPU 31 | CUDA_Host | 0 | 0 | 1,317,888 |
| GPU 31 | CPU_Mapped | 30,081,024 | 0 | 0 |

Device used (total minus native free) after warmup was about 105.5 MB for CPU,
218.8 MB for hybrid, and 283.8 MB for full offload. Native CUDA buffer totals
were about 100.5 MB and 161.9 MB respectively. The remaining ~118–122 MB is
driver, desktop, and other-process usage; the schema reports it as device-level
used, not as RAMforge-attributed allocation.

Late in the milestone CPU plans were changed to pass an explicit CPU device
list. Before that change, a CPU plan in the CUDA build allocated a CUDA compute
buffer even with zero offloaded layers. The current CPU path registers no CUDA
buffer and no CUDA compute allocation.

Unsupported metrics are explicit, not zero:

- allocation/deallocation event counters;
- host-to-device and device-to-host counts, bytes, and duration;
- per-synchronization duration;
- peak VRAM;
- process-attributed device memory;
- separable driver overhead.

Schema 2 adds per-layer/device/buffer tensor storage totals from the read-only
tensor introspection. Full GPU SmolLM2-135M exposes 273 tensors in 2 model
buffers; layer 0 is 2,596,608 bytes across 9 tensors. See
[STAGING_ARCHITECTURE.md](STAGING_ARCHITECTURE.md).

`memory` service snapshots and benchmark reports capture before-load,
after-load, after-warmup, after-generation, and after-unload points. Repeated
GPU load/unload cycles released buffers each time; the final free value returned
to the 3,868,590,080-byte baseline. One transient 76 MB drift was observed while
the desktop compositor changed usage, so the regression bound is 256 MB.

Observability sampling overhead: 100 snapshots after GPU load took 5.29 ms
total, about 53 µs per snapshot.

## Larger-model hybrid experiment

Qwen2.5-7B-Instruct Q4_K_M (4,683,073,632 bytes, 28 transformer layers) was run
with 12 requested layers, 256 context, and 4 CPU threads.

| Metric | Value |
|---|---|
| Effective layers | 12 (11 transformer + output) |
| GPU transformer / CPU transformer | 11 / 17 |
| CUDA0 model buffer | 2,015,414,272 B |
| CUDA0 KV | 5,767,168 B |
| CUDA0 compute | 159,383,552 B |
| CUDA_Host compute | 7,478,272 B |
| CPU-mapped model | 2,661,705,728 B |
| CPU KV | 8,912,896 B |
| Load time | 2.90 s |
| Generation | 9.30 tok/s |
| Prompt throughput | 18.4 tok/s |
| TTFT | 271.6 ms |
| Device used | 2,299,133,952 B |
| Output | ` Paris. Paris is a city in the` |

This was a conservative fit: roughly 2.3 GB device used of 4 GB total, with
17 transformer layers and 2.66 GB of weights on CPU. It is a correctness and
fit result, not a throughput recommendation. No VRAM exhaustion test was run.

## Staging transfer primitives

A bounded native staging engine (pageable/pinned host, CUDA device) and a
private-stream transfer engine are compiled into forgeCORE. Measured transfer
matrix and the single-layer integrity prototype are in
[STAGING_ARCHITECTURE.md](STAGING_ARCHITECTURE.md). The prototype transferred
SmolLM2 layer 0 (2,596,608 bytes), verified byte-identical readback, and
released all buffers. It did **not** feed inference; no graph integration exists.

`ramforge staging-probe MODEL --layer 0 --json` reproduces the transfer-only
prototype.

## Graph-integrated staged inference

The real graph consumed independent staged CUDA layer tensors:

- SmolLM2 layer 0: 9 tensors, 2,596,608 bytes, 9 substitutions per graph,
  equal argmax, logit cosine 0.9933, generated text matching the CPU baseline.
  Corrupting one staged weight changed logits, proving consumption.
- Qwen2.5-7B layer 0: 12 tensors, 149,112,832 bytes, 24 graph substitutions,
  5.84 tok/s with the other 27 layers CPU-resident.
- Release comparison for SmolLM2 (32 tokens): CPU 265.72 tok/s, staged layer 0
  269.04, full GPU 492.19. Single-layer staging is architectural evidence, not a
  throughput win.

At the single-layer milestone, staged layers were static for the context
lifetime. Dynamic eviction and replacement are covered below; capacity reuse and
overlap remain unsupported. See
[STAGING_ARCHITECTURE.md](STAGING_ARCHITECTURE.md).

## Multi-layer budgeted residency

- SmolLM2 layers 0+1: 2 × 2,598,964 B, 18 substitutions per layer over warmup +
  measured graph, generated text matching CPU.
- Auto budget 10M: layers 0–3 accepted (10,126,160 B), rest rejected (`partial`).
- Explicit 1M request: rejected as a set before allocation.
- Qwen2.5-7B auto budget 1G: layers 0–6, 989,857,792 B, 21 layers rejected,
  4.40 tok/s, TTFT 329.2 ms.
- Same-session SmolLM2 comparison from the static-residency milestone: CPU
  225.34 tok/s, 2 staged layers 204.30, full GPU 483.15. Two staged layers
  introduced scheduler-copy overhead on this small model; this is not a
  throughput win.

## Dynamic same-context residency

- SmolLM2 layers 0+1 were replaced by 1+2 in one live context: layer 0 evicted,
  layer 1 kept, layer 2 allocated and transferred.
- Five measured replacement cycles: invalidation median 0.019 ms, eviction
  0.137 ms, staging 3.028 ms, total 3.188 ms (P25 3.063, P75 3.272, P95
  3.491 ms).
- Resident selection alternated cleanly across five cycles; native counters
  ended at 5 invalidations, 5 releases, 7 allocations, 0 reuse.
- Corrupting `blk.1.attn_q.weight` after replacement changed logits by 10.25
  with 9 substitutions.
- Qwen2.5-7B replaced layers 0–6 (989,857,792 B) with 7–13 (953,903,104 B) in
  one context: invalidation 0.03 ms, eviction 7.84 ms, staging 1,901.56 ms,
  total 1,909.43 ms. Replacing back to 0–6 took about 2.0 s. Output remained
  coherent throughout.
- Allocation capacity reuse and transfer/compute overlap remain unsupported.

## Allocation reuse, double buffering and overlap

- Bounded pool reuse is validated. SmolLM2 pipeline cycles kept allocation count
  at 4 while reuse advanced 0→2→4→6→8. Non-pipelined replacement with a 20M pool
  stabilized allocation count at 3 with reuse 1,1,2,3.
- Double buffering is validated. The active slot generated while a complete
  pending slot was filled and committed after invalidation.
- Async batch transfers are validated. One batch enqueued all tensor copies on a
  private low-priority stream, with start/completion events and explicit waits.
- Corruption of a pending layer before commit changed logits by 10.25 with 9
  substitutions after commit, proving inference consumed the committed slot.
- SmolLM2 overlap: transfer 0.94–1.31 ms hidden inside 70–78 ms compute
  (100% hidden), commit wait 0.
- Qwen2.5-7B overlap: four layers per slot, median transfer 81.3 ms hidden
  inside median 443.7 ms compute, 100% hidden, 4 reuse per cycle after the first.
- No end-to-end win: overlapped median 790.2 ms (P25 771.2, P75 813.7, P95
  842.9) vs synchronized-batch 728.1 ms (P25 701.7, P75 729.6, P95 732.0) over
  five cycles. Concurrent DMA slowed compute by roughly the hidden transfer
  time.

Controlled SmolLM2 throughput (5 runs, 32 tokens, context 256, 4 threads):
CPU median 204.68 tok/s (P25 184.18, P75 236.80, P95 237.07); two staged
layers 217.03 (204.43/224.42/224.70); full GPU 492.35 (467.35/493.52/494.69).
The staged result is within run-to-run variance and is not claimed as a
throughput improvement.

A previous session measured the same modes under heavy external CPU load (load
average about 12) and reported much lower absolute rates. Those numbers must not
be compared across sessions. Within the current controlled matrix all modes were
measured under the same conditions.

## Hardware calibration and adaptive policy

Bounded `ramforge calibrate-hw` measurements on the RTX 3050:

- pinned H2D: 6.3–7.1 GB/s for 16–128 MiB payloads; 6.93–7.05 GB/s at 1 MiB;
- pageable H2D: 4.7–5.7 GB/s for 1–64 MiB;
- 64 KiB event/latency floor: 8–21 µs;
- 64 MiB payload chunking medians: 9.75 ms (4 MiB × 16), 9.98 ms (16 MiB × 4),
  10.0 ms (one batch) — no material benefit, coarse batch retained;
- allocation/release overhead is recorded separately in the calibration JSON.

Adaptive policy evidence:

- Balanced Qwen run: async median 654.8 ms vs sync 682.7 ms, within 5%
  hysteresis → synchronous selected.
- Maximum Performance Qwen run: sync median 834.0 ms vs async 855.8 ms → sync.
- Calibration identity mismatch on a different model → synchronous fallback.
- Background profile → synchronous with explicit advisory reason.
- Interleaved exploration plus page-cache warming removes strategy-order bias.

Residency ordering on Qwen2.5-7B, 1 GiB staged budget, warm cache, five runs:

| Policy | Layers | Staged bytes | Median tok/s | P25 | P75 | P95 | TTFT |
|---|---|---|---|---|---|---|---|
| ascending | 7 | 989,857,792 | 6.161 | 6.107 | 6.173 | 6.241 | 221.5 ms |
| smallest-first | 8 | 1,049,083,904 | 6.418 | 6.227 | 6.423 | 6.429 | 212.5 ms |

Smallest-first is opt-in and model/workload-specific; it was 4.2% faster and
resident about 6% more model bytes in this controlled sample.

## Host-side staging pipeline

Qwen2.5-7B, four layers (560,023,552 B), warm page cache:

| Path | Read | Host preparation | H2D |
|---|---|---|---|
| pageable copy | 137.3 ms | 276.1 ms | ~80 ms |
| direct per-tensor | 80.5 ms | 190.1 ms | ~80 ms |
| direct coalesced | 69.7 ms | 178.4 ms | ~80 ms |
| async direct coalesced, warm pool | hidden | ~0.3 ms prepare | ~80 ms |

- 48 tensors collapsed to 2 coalesced read groups.
- Bounded pinned pool (1.5 GiB) retained both slot sizes: 2 allocations,
  3 reuses, ~709 MiB idle, no unbounded growth.
- Tensor-mapping caching removed a recurring ~100 ms enumeration cost.
- End-to-end Qwen, 5 cycles, 4 tokens, order-controlled: synchronous median
  728.4/758.2 ms; async host-read median 691.4/687.2 ms (~7% lower).
- Split GGUF: layers 20–23 read directly (3 groups, 560,496,640 B) and
  generated coherent text; direct bytes matched model reads for two shard files.

## Bounded storage concurrency and prefetch

Qwen four-layer read time, warm cache, one cycle per configuration:

| Batching | c=1 | c=2 | c=4 |
|---|---|---|---|
| Coalesced | 59.9 ms | 61.0 ms | 83.5 ms |
| Per-tensor | 67.9 ms | 67.1 ms | 59.9 ms |

Concurrency does not help coalesced reads and can hurt. Per-tensor c=4 improves
that path but is still no better than coalesced c=1. Default: coalesced,
concurrency 1. Outstanding-byte limits are enforced; the unit test verifies a
2048-byte limit prevents two 2048-byte groups from being in flight together.

Predictive prefetch is application-level (known next selection). SmolLM2:
three prefetch/commit cycles, 15,040,512 bytes prefetched and committed, zero
wasted, 1000 permille hit rate. Abort test: one abort + one commit yields
500 permille with non-zero wasted bytes.

`io_uring` was rejected: runtime libraries exist and the sysctl is enabled, but
no development headers are installed and the coalesced pread path already
saturates the warm-cache workload. No dependency was added.

## Partial residency

Qwen selections `[0,1,2,3] → [2,3,4,5]`, warm cache, four layers per slot:

- full: 560.5/578.5 MB read and transferred, 0 preserved, transition median
  82.5–87.6 ms, 8 allocations;
- partial: 280.2/298.2 MB read/transferred, 280.2 MB preserved, transition median
  42.1–44.2 ms, 6 allocations.

Both produce coherent output. End-to-end five-cycle medians with 4 tokens were
717.8–880.7 ms full versus 640.3–687.6 ms partial, but compute variance is large;
transition latency is the deterministic improvement. Removal-only delta commits
transfer zero bytes.

Failure/abort tests verify the active set is unchanged when pending preparation
fails or is aborted.

## Reproduction

```sh
cd forgeCORE
export FORGE_LLAMA_DIR=/path/to/native
export FORGE_LLAMA_BUILD_DIR=build-cuda
export LD_LIBRARY_PATH="$FORGE_LLAMA_DIR/build-cuda/bin"

cd ../RAMforge
cargo build --release -p ramforge-cli
MODEL=/path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf

# CPU
./target/release/ramforge benchmark "$MODEL" --device cpu \
  --runs 5 --warmup 1 --max-tokens 32 --context 256 --threads 4 --no-repack --json

# Hybrid
./target/release/ramforge benchmark "$MODEL" --device auto --gpu-layers 8 \
  --runs 5 --warmup 1 --max-tokens 32 --context 256 --threads 4 --no-repack --json

# Full offload
./target/release/ramforge benchmark "$MODEL" --device auto --gpu-layers 31 \
  --runs 5 --warmup 1 --max-tokens 32 --context 256 --threads 4 --no-repack --json
```

## Non-claims and limitations

- VRAM transfer volume, transfer time, GPU utilization, and overlap were not
  instrumented.
- The GPU has 4 GiB VRAM; larger models and larger contexts will constrain
  feasible offload.
- No multi-GPU, MoE, tensor streaming, or 500B claim follows from these results.
- CPU/GPU greedy output can diverge after shared prefix tokens.
- Absolute throughput varies between sessions; the reverse-order run is the
  ordering check, not a guarantee.
