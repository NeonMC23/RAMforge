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

1. Reboot so the loaded NVIDIA kernel module matches the installed 620.70.06
   userspace package; verify `nvidia-smi` succeeds and reports the RTX 3050.
2. Install a CUDA toolkit/compiler matching the driver, for example the
   distribution's `cuda-toolkit` package. This is a system change and was not
   performed.
3. Configure a CUDA build in a separate build directory, preserving the pin:

   ```sh
   cmake -S native/src -B native/build-cuda \
     -DGGML_CUDA=ON -DGGML_CPU=ON -DGGML_CPU_REPACK=ON \
     -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
   cmake --build native/build-cuda -j"$(nproc)"
   ```

4. Point `FORGE_LLAMA_DIR` at a tree whose `build/bin` contains the CUDA
   libraries and run the forgeCORE/RAMforge suites.
5. Validate in this order: backend registration, device enumeration, CPU-only
   load, small GPU-offload load with native effective-placement readback,
   real generation, then repeated release benchmarks.

## Non-claims

- The GPU is detected by PCI and the kernel driver, but is not proven usable
  by CUDA in this session.
- The presence of `libcudart` is not evidence of a CUDA build or execution.
- No VRAM total or free value is reported because NVML failed.
- No CPU/GPU hybrid or offload benchmark was run.
