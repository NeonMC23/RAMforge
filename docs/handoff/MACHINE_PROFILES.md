# Machine-profile and scheduling foundations

This document audits what the current RAMforge/forgeCORE stack can and cannot
provide to a future machine-specific execution planner. It intentionally does
not introduce a scheduler.

## Available today

### forgeCORE

- Safe device enumeration through `enumerate_devices()`.
- Device identity, backend id, device kind, name/description, availability.
- Optional device memory total/free as `DeviceInfo` values.
- Capability states for backend initialization, model execution, static model
  offload, memory reporting, and CPU thread configuration.
- Model metadata: parameter count, vocabulary, file size, context length,
  embedding length, layer/head counts, load mode, and repacking mode.
- Tensor mapping metadata: mapping id/generation, mapping size, tensor offset
  and size, page bounds, file path/size.
- Context/execution identity and residency observation primitives.
- Per-dispatch page observations and per-source attribution.

### RAMforge

- Explicit RAM budgets and charged allocations.
- Process RSS, system total/available memory, and optional cgroup statistics.
- Device selection and a static device registry snapshot.
- Model discovery, sharding, tensor inventory, and quantization descriptors.
- Execution profiles, prediction histories, residency plans, staging plans,
  and graph traces.
- Native memory/residency reports and the scheduler observation contract.

## Important distinctions

- Physical RAM, available RAM, cgroup limits, process RSS, RAMforge-managed
  bytes, and OS page cache are different facts and must not be collapsed.
- Device `memory_total`/`memory_free` are optional native observations. They
  are not a guarantee that VRAM is available to RAMforge.
- Mapping presence is not residency; residency is not physical access; neither
  proves I/O or final use.
- Storage throughput is not mapping throughput and is not page-cache
  throughput.
- An unavailable capability is not the same as a zero value or an unsupported
  backend.

## Gaps for future machine-specific planning

1. **Storage characterization.** No safe, portable observation of storage
   type, sequential/random throughput, latency, or filesystem behavior.
2. **CPU capability profile.** No structured SIMD, cache, NUMA, frequency, or
   heterogeneous-core report.
3. **GPU/VRAM placement.** Devices are discoverable, but RAMforge's current
   adapter does not enable non-CPU model placement. VRAM availability and
   per-backend allocation limits are not modeled.
4. **Transfer/staging cost model.** Staging exists, but transfer bandwidth and
   overlap capability are not measured.
5. **Per-backend buffer semantics.** Host-mapped, device, and repacked buffers
   are distinguishable in diagnostics, but not yet represented as a unified
   placement model.
6. **Cgroup/container memory semantics.** Statistics exist, but hard/soft
   limits, reclaim behavior, and OOM policy are not part of a planning model.
7. **Thermal/power constraints.** Not observed.
8. **Page-cache competition with other processes.** Not attributable.

## Proposed smallest future interfaces

These are architectural targets, not implemented scheduler behavior:

```text
ResourceObservationV1
  resource: Cpu | HostRam | Gpu | Vram | Storage | Backend | Transfer
  identity: owned stable id
  provenance: native | os | measured | configured | unknown | unsupported
  freshness: observation sequence or timestamp
  value: typed quantity or explicit unavailable state
  limits: optional hard/soft limits
  validity: scope and invalidation conditions
```

A machine profile would aggregate these records. An execution plan would
reference profile records by identity and record the assumptions it used.

Planner-facing requirements:

- Never compare an unavailable value as zero.
- Never treat free VRAM as fully available without backend/allocator evidence.
- Never equate storage throughput with page-cache throughput.
- Never infer that a model fits because file size is below RAM.
- Keep plan decisions reproducible from the recorded profile and constraints.

## Recommended next step

Complete the observation contract as the stable evidence layer, then add
storage and CPU resource observations with explicit unknown/unsupported states.
Do not begin eviction or scheduler policy until those evidence contracts are
validated on more than one machine.
