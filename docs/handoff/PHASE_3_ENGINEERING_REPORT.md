# RAMforge Phase 3 Engineering Report

> **Historical record.** This report predates the current RAMforge ForgeCore
> refresh and records the v2-era boundary it audited. Current RAMforge uses
> verified ForgeCore `v3.1.2` at
> `d3203099a315db134193404e79ba2581d9fc53ab`; current upstream capability and
> scope are summarized in the repository README. No historical result below is
> being rewritten as a current API claim.

**Date:** 2026-10-02  
**Status:** Implemented and validated  
**Boundary:** Representation and accounting only

## 1. Executive summary

Phase 3 adds a single, domain-aware representation/accounting foundation in
`ramforge-core::resource`. It records what RAMforge knows about a resource
without pretending that a GGUF file range is resident memory or that a safe
ForgeCore API has reported a backend allocation.

The foundation is intentionally read-only from an accounting-policy
perspective: it validates identities, stores explicit measurements, and
produces snapshots. It does **not** move, place, evict, prefetch, stream,
admit, replace, double-buffer, or asynchronously transfer resources. It does
not add GPU execution, multi-GPU execution, out-of-core execution policy, or a
new inference engine.

Existing ForgeCore CPU execution, the legacy inference path, layer cache,
residency, planner, diagnostics, CLI/TUI structure, and the pinned ForgeCore
revision remain intact.

## 2. Implemented representation

The new public module is:

- `crates/ramforge-core/src/resource.rs`

It is registered and re-exported by `ramforge-core/src/lib.rs`. The model and
runtime integrations are deliberately small:

- `ResourceAccounting::from_data_source(&GgufDataSource, ...)` reuses the
  existing parsed file-backed model without reading payloads or implying
  residency.
- `MemoryBudget::resource_budgets()` adapts the existing single host budget to
  the new separate-budget representation without changing allocation behavior.
- `PlanResult::resource_accounting_snapshot(&GgufModel)` builds a read-only
  model representation snapshot using the plan's host budget.
- `MemoryReport::resource_budgets()` exposes the existing managed host budget
  as Phase-3 budget metadata while keeping RSS, system memory, and page-cache
  concepts separate.

No required field was added to the existing public `PlanResult` or
`MemoryReport` structs, so existing struct-literal/API compatibility is
preserved.

### 2.1 Source and logical tensor layers

`ResourceOrigin::GgufTensor` contains three distinct records:

| Record | Contents | Meaning |
| --- | --- | --- |
| `GgufTensorMetadata` | name, dimensions, GGML type, element count, relative and absolute offsets | Facts copied from `TensorDescriptor` |
| `FileBackedRange` | model ID, path, relative offset, absolute file offset, byte length measurement | Addressable source range; not RAM residency |
| `LogicalTensorProperties` | stable tensor ID, shape, dtype, element count, logical byte measurement | Logical representation facts derived from shape/type |

`ResourceRecord` adds model/layer/tensor ownership, resource kind, source
domain, and an optional `AllocationRecord`. A tensor can consequently have a
known file range and logical size while having no runtime allocation record
because it has not been allocated. If an allocation exists but the safe API
cannot report its actual size, the caller records
`Some(AllocationRecord::unmeasured(...))`; it is not converted to zero.

### 2.2 Resource kinds and ownership

The resource kind set includes:

- weights with embedding, layer, output, or other persistent scope;
- KV cache;
- activations;
- scratch/workspace;
- staging;
- pinned host memory;
- mapped model data;
- model metadata and an explicit other category.

The canonical `blk.<index>.*` naming family is classified into
`WeightScope::Layer(LayerId)`, matching the existing runtime layer grouping.
`token_embd.weight`, `output.weight`, and `output_norm.weight` retain their
persistent distinctions. Classification is metadata only; it is not a
residency or placement decision.

`ResourceOwner` stores optional `ModelId`, `LayerId`, and `TensorId` links.
Tensor/resource/allocation IDs are deterministic strings derived from stable
caller/model/tensor names; no insertion order or process-local counter is
used.

## 3. Measurement and sizing semantics

### 3.1 Explicit certainty and provenance

Every byte value is a `ByteMeasurement` with:

- `MeasurementCertainty::Exact`;
- `MeasurementCertainty::Estimated`; or
- `MeasurementCertainty::Unknown`.

Unknown values contain no numeric byte value and carry an unavailable reason.
Every measurement also has `MeasurementProvenance`, including GGUF
 descriptors, derived GGUF type calculations, `TensorData` representation,
`MemoryBudget`, runtime allocator, ForgeCore safe API, operating-system data,
planner estimates, caller-provided values, derived values, and unavailable
reasons.

`ByteTotal` preserves three separate accumulators:

- `exact_bytes`;
- `estimated_bytes`; and
- `unknown_contributions` plus reasons.

Thus an unknown contribution is visible in a snapshot and is never silently
added as zero. A total with estimated or unknown contributions is not reported
as exact/complete.

### 3.2 Tensor sizing

`measure_logical_tensor_bytes` uses the existing `GgmlType::type_info()`
table and checked arithmetic:

- scalar/float/integer types use their known type size;
- block-quantized types require a complete block and use block count times
  block storage size;
- zero-element tensors are represented as exact zero logical bytes, including
  for a future/unknown dtype;
- an incomplete quantized block, unknown dtype, or overflow is explicitly
  unknown;
- the existing best-effort extension entries (for example `Q8_1`, IQ/TQ,
  and `MXFP4` entries marked approximate in the existing type table) are
  marked estimated rather than exact.

This is logical GGUF representation sizing. It is not a claim about a decoded
`TensorData` buffer, backend allocation, VRAM, page cache, or resident working
set. Those values require a separate allocation measurement.

### 3.3 Requested, actual, padding, and resident bytes

`AllocationRecord` keeps these measurements independent:

- `requested_bytes`: what a caller/budget asked for;
- `actual_bytes`: what the runtime/backend allocator reported, possibly
  unknown;
- `alignment_bytes`: optional alignment metadata;
- `alignment_padding_bytes`: derived as actual minus requested only when that
  subtraction is valid; and
- `resident_bytes`: physical residency, independently measured or unknown.

Padding is included in its own snapshot map and is **not** added to actual
allocation totals a second time. A domain is carried on the allocation, not
inferred from the tensor's storage source.

## 4. Domains, budgets, and snapshots

`MemoryDomain` has separate keys for:

- `Storage`;
- `HostRam`;
- `Device(DeviceId)` for device memory/VRAM;
- `MappedVirtual`; and
- `Unknown`.

`ResourceBudgets` holds an explicit host-RAM measurement and a map of
per-device VRAM measurements. The default host budget is unknown, not zero.
`ResourceBudgets::from_memory_budget` and `MemoryBudget::resource_budgets`
preserve compatibility with the existing exact single-domain budget.

`ResourceAccounting` owns records and validates:

- duplicate resource IDs;
- duplicate physical allocation IDs; and
- empty identities.

It does not enforce budgets and has no movement/residency policy. Duplicate
allocation IDs are rejected so two resource aliases cannot silently charge the
same physical allocation twice. A tensor's logical/file ledgers and its actual
allocation ledger are also separate by construction.

`ResourceAccounting::snapshot()` produces:

- logical model bytes;
- file-backed bytes and their storage-domain map;
- requested allocation bytes by domain;
- actual allocation bytes by domain;
- alignment padding by domain;
- independently measured resident bytes by domain; and
- actual bytes by `(domain, ResourceCategory)`.

There is no all-domains “total memory” field. Storage/file-backed values,
host allocations, device allocations, mapped virtual bytes, and unknown-domain
bytes cannot be mistaken for one another. A known allocation with an unknown
domain is kept under `MemoryDomain::Unknown`; an unknown actual size remains an
unknown contribution under its recorded domain.

## 5. ForgeCore and Phase-2 boundary

The safe ForgeCore adapter currently exposes model/context/decode/logits facts,
but not exact backend allocation, VRAM, KV-cache, graph-workspace, page-cache,
or working-set measurements. Phase 3 therefore provides the explicit
`AllocationRecord::unmeasured` path and does not derive those values from
model-file size, GGUF logical tensor bytes, CPU estimates, or RSS.

No ForgeCore source, ForgeCore revision, native library, FFI boundary, or
`forge-sys` dependency declaration was modified. The normal RAMforge
dependency remains pinned to:

```text
944bc3ca1ed7e48e2ad897107d9dcda5dd5eedc9
```

The existing `InferenceEngine` and numerical modules were not replaced or
disconnected.

## 6. Focused test coverage

The resource module includes focused tests for:

- float, quantized, zero-sized, incomplete-block, estimated, and unknown
  tensor sizing;
- GGUF relative/absolute file offsets and unknown file lengths;
- embedding, output, persistent, layer, and other weight classification;
- model/layer/tensor ownership;
- storage, host, device, mapped-virtual, and unknown-domain separation;
- requested versus actual allocation uncertainty;
- alignment/padding and independent residency uncertainty;
- exact, estimated, unknown, and provenance-preserving snapshots;
- host/device budgets and the existing `MemoryBudget` adapter;
- deterministic stable IDs;
- duplicate physical allocation rejection and logical/file/actual
  double-counting prevention; and
- runtime plan/memory-report adapters.

## 7. Validation record

Validation was run without adding build output to the workspace. The local
shell did not have `cargo` on its default `PATH`, so the already-present local
toolchain and caches were used explicitly:

```text
Toolchain:    /var/tmp/forge-toolchain/rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin
CARGO_HOME:   /var/tmp/forge-toolchain/cargo
CARGO_TARGET_DIR: /var/tmp/forge-toolchain/target
CARGO_NET_OFFLINE: true
FORGE_LLAMA_DIR: /var/tmp/forge-native
LD_LIBRARY_PATH: /var/tmp/forge-native/build/bin
```

Commands and results:

| Command | Result |
| --- | --- |
| `cargo fmt --all -- --check` | Passed |
| `cargo check --workspace` | Passed |
| `cargo test -p ramforge-core resource::tests` | 10 passed |
| `cargo test -p ramforge-runtime plan::tests::test_plan_fits` and the memory-report adapter test | Passed |
| `cargo test --workspace` with the external native-library environment above | 368 passed, 0 failed, 2 ignored; this includes 44 CLI tests, 122 core tests, 198 runtime unit tests, and 4 ForgeCore adapter integration tests |
| `cargo clippy --workspace --all-targets -- -D warnings` | Passed |
| `cargo build --workspace --release` | Passed |

An initial workspace-test attempt without `LD_LIBRARY_PATH` demonstrated the
expected environment limitation: the ForgeCore integration test binary could
not load `libggml.so.0`. No source change was made for that failure; rerunning
with the already-present pinned native checkout passed all four integration
tests.

Filesystem inspection after validation confirmed that the RAMforge workspace
contains no `target/`, model files, native libraries, vendored dependencies,
`.cargo` directory, or generated prohibited artifacts. The workspace is not a
Git checkout in this environment, so Git status/diff could not be used; file
inventory was used instead. The pre-existing `.cargo` directory was absent and
was not recreated.

## 8. Explicit non-goals retained for later phases

This Phase-3 change does not implement or imply:

- SSD/RAM/VRAM movement, streaming, eviction, prefetch, double buffering, or
  asynchronous transfers;
- placement, residency, replacement, admission, or cost policies;
- GPU/multi-GPU execution or an out-of-core execution engine;
- backend allocation probing beyond facts already exposed by safe APIs;
- fake KV, activation, graph scratch, VRAM, page-cache, RSS, or working-set
  measurements;
- a new inference engine or changes to the default CLI backend; or
- project, phase, milestone, dependency, or schema-version increments.
