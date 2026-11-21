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

## Memory observability (schema 1)

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
