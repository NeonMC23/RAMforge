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
| CPU | 0 | 240.70 | 234.26–245.45 | 255.32 | 249.12 | 704.4 | 7.15 ms |
| Hybrid | 8 | 289.13 | 286.92–303.01 | 310.10 | 310.51 | 805.7 | 6.25 ms |
| GPU | 31 | 484.65 | 460.70–488.73 | 491.16 | 495.54 | 1448.5 | 3.50 ms |

A reverse-order session measured GPU 488.89, hybrid 309.53, CPU 249.62 tok/s,
reproducing the ordering. Full offload was about 2.0x CPU and hybrid about 1.2x
for this small model on this laptop GPU. These are single-machine,
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
