# RAMforge Phase 4 Engineering Report

**Date:** 2026-10-02  
**Status:** Implemented in source; compilation and native execution validation unavailable in this environment  
**Boundary:** Static CPU/device discovery and explicit execution selection only

## 1. Executive summary

Phase 4 adds RAMforge-owned static device/backend abstractions on top of the
existing safe ForgeCore adapter. RAMforge now owns the selection request,
capability interpretation, execution configuration, accounting-domain
association, and diagnostics. ForgeCore remains the only boundary to the
pinned llama.cpp/ggml implementation.

The pinned ForgeCore revision exposes safe device enumeration, but its safe
`Model` and `Context` APIs do not expose a model/backend binding or an
`n_gpu_layers`/offload option. Its model loader sets native GPU layers to zero.
Consequently, the implementation does **not** claim GPU execution. Native
non-CPU devices are preserved as discovery facts with their safely reported
memory values, but they are explicitly non-selectable for model execution.
Selecting one cannot silently fall back to CPU.

CPU remains a first-class selectable device. Existing `ForgeCoreModel::load`,
`ForgeCoreModel::load_with_options`, `open_context`, the legacy inference
engine, the legacy default CLI path, and the `--backend forgecore` /
`--backend forge-core` aliases remain available.

No Phase-5 policy was added. There is no dynamic movement, residency policy,
eviction, prefetch, transfer queue, double buffering, placement/admission/
replacement/cost policy, out-of-core execution policy, streaming policy,
multi-device scheduling, multi-GPU balancing, distributed execution, custom
kernel, direct FFI, or `forge-sys` dependency in RAMforge.

## 2. Pinned native boundary findings

The implementation is based on the safe API at ForgeCore revision
`944bc3ca1ed7e48e2ad897107d9dcda5dd5eedc9`:

- `forge_core::enumerate_devices()` returns owned `DeviceInfo` snapshots with
  an index, name, description, `DeviceType`, and free/total memory values.
- `DeviceType` distinguishes CPU, GPU, integrated GPU, accelerator, meta, and
  unknown native kinds.
- `Backend::open_cpu` and `Backend::open_device` exist as safe backend-handle
  APIs, but the backend handle is not accepted by the pinned `Model` or
  `Context` constructors.
- `ModelOptions` exposes tensor checking only; model loading sets native
  `n_gpu_layers` to zero.
- `ContextOptions` exposes context length and CPU thread count only, and
  `Context::open` accepts a `Model`, not a backend handle.

This is a concrete Phase-4 blocker for static non-CPU model execution. No
ForgeCore source, pinned revision, native source tree, or native build was
modified to work around it. The adapter reports the limitation rather than
using private APIs or fabricating offload.

## 3. RAMforge-owned execution model

The new module is:

```text
crates/ramforge-runtime/src/execution.rs
```

It is registered and re-exported by
`crates/ramforge-runtime/src/lib.rs`.

### 3.1 Identity and kind

| Abstraction | Responsibility | Deliberately does not mean |
| --- | --- | --- |
| `DeviceId` | Stable RAMforge/Phase-3 identity used in diagnostics and device accounting domains | A native allocation handle or proof of residency |
| `BackendId` | RAMforge identity for the discovered backend/device grouping | A private ForgeCore/ggml backend pointer |
| `DeviceKind` | CPU, GPU, integrated GPU, accelerator, meta, or unknown | A claim that the kind is executable by the model API |
| `DeviceInfo` | Owned snapshot of native facts plus RAMforge capability/selectability facts | A live native handle |
| `DeviceCapabilities` | Supported/unsupported/unknown states for initialization, model execution, static offload, memory reporting, and CPU threads | An allocation or residency report |
| `MemoryDomain` | Host RAM for CPU and `Device(DeviceId)` for GPU-like facts | A statement that model bytes are resident in that domain |

ForgeCore does not expose a vendor or persistent hardware UUID through the
safe snapshot. The implementation therefore derives deterministic IDs from
native kind, registry index, and a normalized native name, and derives a
backend ID from the native kind and normalized name. The native registry index
is retained separately for explicit backend/index lookup. These IDs are stable
for the same native registry facts and are not advertised as cross-machine
hardware UUIDs.

### 3.2 Discovery and capability semantics

`DeviceRegistry::discover()` calls only ForgeCore's safe
`enumerate_devices()` API. It does not call native FFI, load model weights,
open a raw backend handle, or inspect operating-system GPU state. A CPU-only
native build is valid.

For each snapshot:

- CPU is available and selectable for the current model/context API.
- Non-CPU facts remain available in the registry when ForgeCore reports them.
- Non-CPU model execution and static offload are `Unsupported` for the pinned
  safe API, so those entries are not selectable.
- Zero native memory values are represented as unknown, not as zero-byte VRAM.
- Positive native values are exact measurements with
  `MeasurementProvenance::ForgeCoreSafeApi`.
- Meta and unknown device kinds use `MemoryDomain::Unknown` rather than being
  reclassified as host or device memory.

`DeviceRegistry::cpu_only()` is used only to preserve the old CPU default
adapter constructor without requiring that constructor to change its native
startup behavior. The CLI's explicit ForgeCore path uses
`DeviceRegistry::discover()`.

### 3.3 Explicit selection

`DeviceSelection` accepts exactly these forms:

```text
cpu
id:<device-id>
backend:<backend-id>:<index>
```

`DeviceRegistry::select` resolves one snapshot and returns `SelectedDevice` or
an explicit error. It never tries another device. Errors distinguish invalid
syntax, missing CPU, missing device ID, missing backend/index, and a discovered
but non-selectable device. An invalid backend index is not clamped and an
unavailable GPU is not changed into CPU execution.

### 3.4 Execution configuration

`ExecutionConfig` contains only current adapter inputs:

- explicit `DeviceSelection`;
- CPU thread count;
- optional context length; and
- ForgeCore tensor-checking choice.

It validates nonzero thread/context values and resolves through a supplied
registry. It intentionally has no placement, budget, movement, eviction,
transfer, queue, residency, or multi-device fields.

## 4. ForgeCore adapter integration

`ForgeCoreModel` now retains the selected `DeviceInfo` and
`ExecutionConfig` alongside the existing native model, path, and metadata.
Existing constructors remain backward compatible and use an explicit CPU
configuration.

The new `ForgeCoreModel::load_for_execution` path:

1. validates and resolves the requested selection in the supplied registry;
2. rejects every selected non-CPU device with
   `ForgeCoreAdapterError::UnsupportedDevice`;
3. does not load the model and then silently execute it on CPU;
4. forwards the safe execution inputs into the existing model/context path;
5. retains the selected device/accounting metadata for diagnostics.

`open_selected_context` forwards the configured context length and CPU thread
count into ForgeCore's safe `ContextOptions`. The existing explicit
`open_context` remains available for callers that need the previous API.

The adapter error enum now distinguishes configuration, device selection,
unsupported device, model load, context initialization, batch construction,
native decode, and logits extraction failures. No native error is converted to
a successful fallback.

A selected device is metadata and orchestration state only. It is not a claim
that weights, KV state, activations, or any other resource has been allocated
or is resident there.

## 5. CLI behavior

`ramforge run` has a minimal new option:

```text
--device cpu
--device id:<device-id>
--device backend:<backend-id>:<index>
```

The default is `cpu`. The option is parsed for every run so malformed device
syntax is not silently ignored. The legacy backend remains the default. A
non-CPU device request with the legacy backend fails clearly instead of being
ignored or switching to ForgeCore. The explicit ForgeCore branch discovers
native devices, resolves the exact selection, and reports the selected backend,
device ID, kind, registry index, accounting domain, capabilities, and native
memory measurements.

If a non-CPU device is discovered on this pinned ForgeCore revision, the
explicit command reports that the device is unsupported by the safe
Model/Context binding and exits without CPU fallback. CPU ForgeCore execution
continues through ForgeCore's model/context/batch/decode/logits API.

Examples:

```text
# Existing default behavior; legacy remains the default.
cargo run -p ramforge-cli -- run MODEL.gguf --ram 8G --prompt "Hello"

# Explicit ForgeCore CPU execution.
cargo run -p ramforge-cli -- run MODEL.gguf --ram 8G \
    --backend forgecore --device cpu --prompt "Hello"

# Explicit selection syntax; unavailable/non-selectable selections fail.
cargo run -p ramforge-cli -- run MODEL.gguf --ram 8G \
    --backend forge-core --device id:forgecore/gpu/1/cuda0 --prompt "Hello"
```

The RAM budget remains a planning/diagnostic input for ForgeCore execution.
It is not presented as a measurement of ForgeCore native allocations, VRAM,
KV state, page cache, or total RSS.

## 6. Accounting and residency invariants

Phase-3 resource semantics are unchanged:

- GGUF file-backed bytes remain storage-domain/source facts, not resident RAM.
- Logical tensor bytes remain separate from requested, actual allocation, and
  resident bytes.
- Native free/total device memory is retained only when ForgeCore safely
  reports a nonzero value; otherwise it is unknown.
- Host and device accounting domains remain separate.
- Device selection does not prove model residency.
- The implementation does not derive VRAM or native allocation usage from
  model size, logical tensor size, RAM budget, RSS, OS GPU discovery, or CPU
  estimates.

## 7. Tests added

Focused unit coverage in `execution.rs` covers:

- CPU-first-class selection and host accounting domain;
- deterministic identity mapping from native facts;
- exact versus unknown native memory values;
- explicit missing-ID and non-selectable-device errors;
- selection syntax and invalid shorthand;
- execution-config validation; and
- unknown native device kinds.

`tests/execution_devices.rs` adds an opt-in native registry test controlled by
`RAMFORGE_RUN_NATIVE_DEVICE_TESTS=1`. It accepts CPU-only builds, requires a
reported CPU only when explicitly enabled, and does not assert that a GPU
exists or attempt unsupported GPU model execution.

The existing ForgeCore adapter tests remain unchanged in their public API use.
They continue to cover model metadata, context behavior, batch/decode/logits
translation, and the no-direct-FFI boundary.

## 8. Validation record

The requested validation commands were attempted but were not executable in
the current workspace environment: each `cargo` command returned status 127
(`cargo: not found`). `cargo` and `rustc` are not on `PATH`, the historical
`/var/tmp/forge-toolchain` and `/var/tmp/forge-native` locations are absent,
and no local pinned ForgeCore/native build was found. Therefore the following
status is **UNKNOWN / NOT RUN**, not a claimed pass:

```text
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo build --workspace --release
```

The following are also **NOT RUN**:

- focused Phase-4 device tests;
- optional `RAMFORGE_RUN_NATIVE_DEVICE_TESTS=1` registry coverage;
- real ForgeCore CPU fixture execution;
- real GPU discovery/execution; and
- release build validation.

No `target/`, model, native library, build tree, vendored dependency, or
large generated artifact was added. Compilation should be the first follow-up
when a compatible local Rust toolchain and `FORGE_LLAMA_DIR` native build are
available. Any compiler/API mismatch must be corrected against the pinned safe
ForgeCore API rather than by adding direct FFI or a fallback.

## 9. Deliberate non-claims and next boundary

Phase 4 does not claim static GPU execution because the pinned safe ForgeCore
API does not support it. Enabling that capability would require a separately
reviewed safe ForgeCore API change that accepts a backend/device or exposes a
safe offload option, while preserving the native revision constraints. That is
not implemented here.

No Phase 5 work was started.
