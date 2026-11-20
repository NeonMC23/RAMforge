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

## Profile schema and persistence

- `MACHINE_PROFILE_SCHEMA_VERSION = 1`.
- `MachineProfile::validate()` rejects unsupported versions and broken
  observed-value invariants.
- Unknown fields are additive-compatible within version 1; removing or
  changing the meaning of a field requires a version increment.
- `to_json` / `from_json` round-trip and validate.
- `collection_sequence` identifies a collection pass; `Observed.sequence`
  ties each value to it.
- `Observed.freshness` marks stable hardware facts versus dynamic observations.
- `MachineProfile::redacted()` removes local storage paths and the cgroup path.
  `ramforge machine --json --export` uses the redacted form.
- Local runtime profiles may retain paths; exported profiles should use the
  redacted form.

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

## Implemented in this milestone

### `machine_profile`

- `Observed<T>` with `Known`, `Unavailable`, `Unsupported`, and `Failed`
  states, provenance, freshness (`Stable`/`Dynamic`), collection sequence, and
  detail text.
- Versioned `MachineProfile` (`schema_version = 1`) with CPU, memory, storage,
  device, and backend sections.
- Local JSON round-trip and `redacted()` export that removes private paths.
- `ramforge machine` CLI inspection.

### CPU

Linux collection from `/proc/cpuinfo`, `/proc/self/status`, `/sys`, and cgroup
v2: architecture, vendor/model, hardware logical CPUs, unique physical cores,
affinity, process-visible threads, feature flags, SIMD flags, NUMA nodes,
cache topology, frequency, and cgroup CPU quota.

### Memory

Separate observations for physical total, OS available, swap, process RSS /
peak / virtual size, configured budget, cgroup limit/current/peak/anon/file/
mapped/shmem/swap, and PSI pressure. Unavailable cgroup fields stay explicit.

### Storage

Per-path filesystem, filesystem magic, mount point, block device identity,
rotational/SSD/NVMe/network/memory-backed/virtual classification when
reliably supported, read-only state, capacity, and available space. A bounded
file-scoped read benchmark records cache condition and never claims physical
device throughput.

### Devices and backends

Existing forgeCORE device discovery is mapped into explicit profiles with
memory total/free observations, capability states, selectability, and backend
aggregates. Missing GPUs are explicit and do not break CPU plans.

### Model requirements

`ModelRequirements` separates exact file/tensor bytes from estimated KV cache
and unknown requirements. GGUF file size is never treated as the complete RAM
requirement, and mapped weights are not assumed resident.

### Candidate plans

`ExecutionPlanCandidate`, `evaluate_plan()`, and `compare_plans()` expose four
initial profiles, conservative feasibility, explicit reasons/unknowns, and
multi-dimensional comparison without a single hidden score. See
[EXECUTION_PLANS.md](EXECUTION_PLANS.md).

## Remaining gaps

1. **Random/read-latency storage characterization.** Only a bounded
   sequential read benchmark and advisory cache-drop request are implemented.
2. **GPU/VRAM placement.** Devices are discoverable, but RAMforge's adapter
   still does not enable non-CPU model placement, and GPU KV/staging placement
   is not modeled.
3. **Transfer/staging cost model.** Transfer bandwidth and overlap capability
   are not measured.
4. **Per-backend buffer semantics.** Host-mapped, device, and repacked buffers
   are not yet a unified placement model.
5. **Cgroup hard/soft limit and reclaim semantics.** Values exist, but a
   planning model does not yet infer reclaimability or OOM behavior.
6. **Thermal/power constraints.** Not observed.
7. **Page-cache competition with other processes.** Not attributable.
8. **Multi-machine validation.** Only one physical machine and deterministic
   synthetic fixtures have been exercised.

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

## Current validation

Validated on:

- Linux 7.3.4, x86_64.
- Intel i5-11400H, 12 logical threads, 6 physical cores.
- 15.39 GiB physical RAM.
- btrfs root over an NVMe-backed device-mapper target; `/tmp` is tmpfs.
- CPU: AVX2/FMA/AVX-512 feature flags, cache topology, NUMA node 0,
  frequency range, no cgroup CPU quota.
- Memory: physical/available, process RSS/virtual, cgroup files present but
  `memory.max` unlimited.
- Storage: btrfs, block device `dm-0` (253:0), rotational=0, solid-state.
- Devices: CPU only; no GPU present.
- Model: SmolLM2-135M Q4_K_M; exact tensor payload 103,668,480 bytes;
  estimated 2,048-token KV cache 47,185,920 bytes.

Only one physical machine has been exercised. Synthetic tests cover missing
fields, invalid values, missing GPU devices, budget failures, context limits,
and serialization. They do not constitute multi-machine validation.

## Validation procedure for future machines

1. Run `ramforge machine --json` with no model to inspect CPU/RAM/storage.
2. Run it with a representative model and a user budget.
3. Inspect `plan_comparison` for infeasible/unknown reasons.
4. Run `scripts/resource-profile-bench.sh` and record the machine, kernel,
   filesystem, and cache condition.
5. Repeat on a materially different CPU, container/cgroup, or storage setup.

## Recommended next step

Add storage latency/random-read observations and CPU/container fixtures, then
compare candidate plans against measured execution on a second machine. Do not
begin eviction or automatic plan switching until the feasibility assumptions
are validated beyond one machine.
