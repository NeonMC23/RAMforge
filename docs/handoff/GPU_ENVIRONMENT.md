# GPU environment audit (2026-11-21)

This document records what the physical machine and native build actually
expose. It does not claim CUDA execution.

## Hardware

| Item | Observed value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (GA107M), PCI `0000:01:00.0`, `[10de:25a2]` |
| GPU kernel driver | `nvidia` loaded; `nvidia_uvm`, `nvidia_drm`, `nvidia_modeset` loaded |
| Integrated GPU | Intel TigerLake-H UHD Graphics, `i915` |
| CPU | 11th Gen Intel Core i5-11400H, 12 logical CPUs |
| Host RAM | 15 GiB total, about 7 GiB available during the audit |
| Free disk on `/home` | about 195 GiB |

VRAM capacity and free VRAM could not be queried because NVML initialization
fails (below). Do not infer VRAM from the model or from vendor marketing.

## Driver state

| Item | Value |
|---|---|
| Running kernel module | `595.112.07` (Open Kernel Module) |
| NVIDIA GPU firmware | `595.112.07` |
| Userspace driver / `libcuda` | `620.70.06` |
| Installed DKMS / kernel package | `620.70.06` |
| `nvidia-smi` | fails: `Failed to initialize NVML: Driver/library version mismatch` |
| `NVML library version` reported | `620.70` |

The kernel module currently loaded does not match the 620.70.06 userspace
libraries. The 620.70.06 modules are installed for the running and adjacent
kernels. This is consistent with a reboot-pending driver update. CUDA runtime
initialization cannot succeed in this state. No reboot, module reload, or
driver change was performed.

## CUDA toolkit state

| Item | State |
|---|---|
| `nvcc` | not present anywhere on `PATH` or in `/usr/local`/`/opt` |
| CUDA headers (`cuda.h`, `cuda_runtime.h`) | absent |
| CUDA runtime library | `libcudart.so.13.2.51` present |
| CUDA driver API library | `libcuda.so.620.70.06` present |
| CUDA packages | `cuda`, `cuda-cudart`, `cuda-libs`, `cuda-nvrtc`, `cuda-nvml-devel` |
| Toolkit compiler/headers | **not installed** |

The installed `cuda` package contains documentation only; there is no CUDA
compiler or development toolkit. A CUDA-enabled llama.cpp/ggml build therefore
cannot be produced without installing additional packages.

## Native build state

- Pin: `7fe450e19305b828c199d602c23a8337aaa1f03b` (unchanged).
- CMake cache: `GGML_CPU=ON`, `GGML_CPU_REPACK=ON`.
- `GGML_CUDA`, `GGML_VULKAN`, `GGML_SYCL`, `GGML_OPENCL`, `GGML_METAL`,
  `GGML_RPC`, and `GGML_BACKEND_DL` are all `OFF`.
- The registered backend list in this process is `CPU` with one device.
- `llama_supports_gpu_offload()` is false.
- No GPU device is selectable.

## Why this milestone could not validate CUDA

Two independent blockers exist:

1. **No CUDA compiler/toolkit.** `nvcc` and headers are absent, so a CUDA
   backend cannot be built.
2. **Driver/library mismatch.** Even with a CUDA-enabled binary, the running
   595 kernel module cannot service the 620 userspace CUDA driver.

Neither blocker was bypassed. No CUDA-enabled build, fake device, or synthetic
GPU result is reported as physical validation.

## Safe follow-up procedure (requires user authorization)

The audit was repeated on 2026-11-21 after six days of uptime. The driver
module was still 595.112.07 with 620.70.06 userspace, `nvidia-smi` still
failed, and `nvcc` was still absent. Nothing below was executed.

### Exact minimal package proposal

The Nobara repository `nobara-nvidia-production` provides CUDA 13.2 packages.
There is no `cuda-toolkit` package in this repository. The supported
development meta-package is `cuda-devel`; `cuda-nvcc` is its compiler
dependency.

Proposed command, after explicit authorization:

```sh
sudo dnf install cuda-devel cuda-gcc gcc15-c++
```

- Source: `nobara-nvidia-production`.
- Versions currently offered: `cuda-devel`/`cuda-nvcc` 13.2.51, `cuda-cccl-devel`
  13.2.27, `cuda-gcc` 15, `gcc15-c++` 15.3.1.
- `cuda-devel` pulls the CUDA headers, compiler, cuBLAS/cuFFT/cuRAND/cuSOLVER/
  cuSPARSE/NPP/NVJITLink/NVJPEG development packages; it is larger than the
  runtime-only packages already installed. Exact transaction size was not
  queried because that requires a package-manager transaction.
- The system GCC is 16.2.1. `cuda-gcc` + `gcc15-c++` provide GCC 15, which is
  the safer CUDA 13 host compiler. The setup script auto-selects `g++-15` when
  present.
- CUDA 13.2 supports the RTX 3050's compute capability 8.6. The installed
  620.70.06 driver is compatible with CUDA 13.x; the running 595 module is the
  blocker.

### Driver recovery

The 620.70.06 NVIDIA driver and kernel packages are already installed. A
reboot is required so the loaded kernel module matches userspace. No package
change is needed for the driver itself.

1. Reboot (user action).
2. Verify:

   ```sh
   cat /proc/driver/nvidia/version   # expect 620.70.06
   nvidia-smi                        # expect RTX 3050 and VRAM
   ```

3. Install the toolkit packages above (user authorization).
4. Verify:

   ```sh
   nvcc --version
   ls /usr/include/cuda.h /usr/include/cuda_runtime.h
   ```

### CUDA build and validation

The repository now contains a guarded script that performs preflight,
configure, and build without modifying the pin or the CPU build:

```sh
cd forgeCORE
FORGE_LLAMA_DIR=/path/to/native sh scripts/setup-native-cuda.sh --check
FORGE_LLAMA_DIR=/path/to/native sh scripts/setup-native-cuda.sh
export FORGE_LLAMA_DIR=/path/to/native
export FORGE_LLAMA_BUILD_DIR=build-cuda
```

The script keeps `GGML_CPU=ON`, `GGML_CPU_REPACK=ON`, enables `GGML_CUDA`,
disables NCCL for the single GPU, targets the reported compute capability, and
verifies that `libggml-cuda.so` exports `ggml_backend_cuda_reg`.

Then validate in order:

1. `registered_backends()` includes CUDA.
2. `enumerate_devices()` reports the RTX 3050 with VRAM.
3. CPU-only inference still works with the CUDA build.
4. Small offload: effective GPU layer count and per-layer placement readback.
5. Real generation, streaming, cancellation, unload.
6. Increasing offload counts up to the model's feasible maximum.
7. Repeated release benchmarks with reversed order.

A registered CUDA device is not yet proof of inference. Only real generated
tokens and effective placement readback close the milestone.

## Non-claims

- The GPU is detected by PCI and the kernel driver, but is not proven usable
  by CUDA in this session.
- The presence of `libcudart` is not evidence of a CUDA build or execution.
- No VRAM total or free value is reported because NVML failed.
- No CPU/GPU hybrid or offload benchmark was run.
