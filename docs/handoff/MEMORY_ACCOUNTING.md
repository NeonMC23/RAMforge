# Memory accounting and residency policy

This document describes the RAMforge memory ledger and the policy profiles that
consume it. It is intentionally narrower than “process memory”: RAMforge can
measure and reason about some components, cannot measure others, and must not
pretend that they are interchangeable.

## Ownership boundaries

- **RAMforge** owns policy, inventory, ledger categories, budgets, planning,
  staging plans, admission, reports, and CLI controls.
- **forgeCORE** owns safe Rust wrappers, native FFI, resource lifetimes, and the
  release/prefetch calls.
- **llama.cpp / ggml** owns model mappings, inference graphs, kernels, KV
  allocation, and backend execution.

RAMforge does not call llama.cpp/ggml FFI and does not depend on `forge-sys`.

## Ledger categories

`ramforge-runtime/src/memory_ledger.rs` defines model-scoped ledger entries.
Each entry has a category, optional byte value, provenance kind, source, and
note.

| Category | Meaning | Typical provenance |
|---|---|---|
| `FileBackedLogical` | Sum of file-backed tensor payload bytes from native metadata | Native logical |
| `FileBackedResident` | Resident process pages in model file mappings | `/proc/self/smaps` measured |
| `AnonymousWeights` | Anonymous/repacked tensor payloads | Native logical |
| `MappingPss` | Proportional share of model mappings | `/proc/self/smaps` measured |
| `ProcessRss` | Whole-process RSS | `/proc/self/status` measured |
| `ProcessRssPeak` | Process RSS high-water mark | `VmHWM` measured |
| `CgroupCurrent` | Charged cgroup memory | cgroup v2 `memory.current` |
| `CgroupPeak` | Charged cgroup high-water mark | cgroup v2 `memory.peak` |
| `StagingRequested` | Page bytes in a prefetch plan | RAMforge requested |
| `StagingAccepted` | Native-accepted prefetch bytes | Native result |
| `KvCache` | KV-cache allocation | **Unknown** unless a measurement exists |
| `GraphScratch` | Graph/backend scratch | **Unknown** |
| `Runtime` | Allocator/runtime/loader overhead | **Unknown** |

Do not add these categories into one “total memory”. In particular:

- logical model bytes are not resident bytes;
- mapping RSS is not process RSS;
- process RSS is not cgroup usage;
- cgroup usage includes charged page cache and kernel accounting and may not
  change when a process drops resident mapping pages;
- a prefetch hint does not prove that pages became resident.

## Provenance and certainty

`MeasurementKind` distinguishes:

- `ExactMeasured` — read from a kernel or native counter at a point in time;
- `NativeLogical` — a native logical size, not a residency measurement;
- `Estimated` — derived conservatively from mapping ranges;
- `Unknown` — no reliable value is available;
- `Requested` — bytes submitted to a hint or release request;
- `NativeAccepted` — bytes the native layer reports accepting.

The ledger does not silently turn unknown components into zero. KV cache,
graph scratch, and runtime overhead always produce explicit unknown entries
with a reason. Where a measurement is absent, its value is `Option::None`.

## Checked arithmetic

`MemoryLedger::checked_total` sums only explicitly requested categories with
`u64::checked_add`. Overflow returns `LedgerError::Overflow`; it does not
saturate into a plausible-looking total.

Admission adds the safety reserve with checked arithmetic. If the addition
overflows, admission rejects the request instead of continuing with a wrapped
value. Planner and inventory totals still use conservative saturating
arithmetic in a few existing paths; those calculations are labelled as
estimates and must not be presented as measured totals.

## Reconciliation

`MemoryLedger::reconciliation` reports, when available:

- mapped virtual bytes from inventory page ranges;
- mapped resident bytes;
- process RSS;
- cgroup current and peak;
- `RSS - mapping resident`;
- `cgroup current - process RSS`.

The differences are signed, not “overhead”. Examples:

- RSS includes KV cache, graph scratch, allocator metadata, stacks, runtime,
  tokenizer data, and other process pages not present in model mappings.
- cgroup usage may stay flat while process RSS drops because some model pages
  are not charged to this cgroup, because the cgroup includes other work, or
  because the kernel has not reconciled charged file pages.
- Mapping PSS differs from RSS when pages are shared.

A successful release reports page bytes submitted to `madvise`. It does not
report confirmed physical reclamation. A measured mapping-RSS drop is evidence
for those mappings at that sample; it is not proof about cgroup memory.

## Resource inventory and deduplication

The inventory uses native tensor metadata:

- stable RAMforge resource key: inventory generation plus tensor index;
- tensor name (for diagnostics and profile matching);
- file-backed flag;
- mapping index, mapping size, offset, logical size, page size, and aligned
  page range;
- eligibility (`required`, anonymous/repacked, unknown, empty range, etc.);
- conservative shared-page bytes within a mapping;
- metadata confidence.

Shared-page bytes are computed as the union of page overlaps with other
file-backed tensors in the same mapping. A tensor with partial sharing has a
positive shared-page cost and a smaller unique reclaimable estimate. The native
release call protects shared pages; the plan still records the shared cost so
the estimate is not overstated.

Duplicate tensor names remain distinct by native tensor index. A generation
refresh replaces the inventory and invalidates old attempts/plans.

## Staging accounting

The bounded staging controller plans per tensor/mapping page range:

- candidates are file-backed, positive-page, non-stale inventory entries;
- page ranges are deduplicated per mapping using interval union;
- `selected_bytes` is the union of selected ranges, not the sum of tensor sizes;
- `overlap_bytes_avoided` records bytes already represented by an earlier
  selection;
- `skipped_bytes` records candidate bytes not selected because of the budget or
  duplicates;
- the requested budget is a strict maximum;
- native accepted bytes and observed mapping changes are reported separately.

Prefetch remains a hint. `MADV_WILLNEED` can return success without loading all
pages, and a zero mapping delta after a successful hint is a valid observation.

## Policy profiles

The existing `off` / `observe` / `active` CLI values remain compatible. New
profiles are explicit:

| Profile | Plans? | Mutates? | Selection rule |
|---|---:|---:|---|
| `off` | No | No | No policy action |
| `observe` | Yes | No | Deterministic memory-bound plan for inspection |
| `conservative` | Yes | Yes, after safe boundary | Only tensors with **zero** shared page bytes |
| `memory-bound` | Yes | Yes, after safe boundary | Cold/large candidates until target or eligibility exhausted |
| `reuse-aware` | Yes | Yes, after safe boundary | Valid profile orders cold resources first; profile must match model generation |
| `active` | Yes | Yes, after safe boundary | Compatibility alias for memory-bound behavior |

A target is required for `memory-bound` and `active`. The soft target is reduced
by the safety reserve. `conservative` intentionally may release nothing on
models whose tensor payloads share aligned pages; that is safer than pretending
the shared bytes are uniquely reclaimable.

`reuse-aware` uses an in-memory execution profile derived from a complete
bounded trace. Profiles contain stable tensor names and graph-order access
information only; transient native node IDs are never persisted. A profile
from another model or inventory generation is ignored. If no valid profile is
available, the profile is not used and the plan reports
`profile_used=false`.

## Admission

Admission is separate from planning:

- a hard cgroup limit, when enforced, is authoritative;
- the soft target only produces an `AdmittedWithWarning` outcome;
- unknown memory components are listed explicitly;
- if the excess is likely anonymous, KV, scratch, or runtime memory, the report
  says page release may not be able to address it;
- a request that cannot be measured is `Unknown`, not silently admitted.

The cgroup remains the hard enforcement mechanism. No policy claims that a
release request lowers cgroup usage by the requested amount.

## Operation report

`--residency-report` prints a bounded human-readable report.
`--residency-report-json PATH` writes a versioned JSON document
(`schema_version: 1`, `kind: ramforge.residency.operation_report`). It includes:

- model identity and inventory generation;
- policy/admission/plan summaries;
- ledger entries and reconciliation;
- staging plan/result;
- application request/result;
- profile completeness and bounded access names;
- bounded selected-resource lists plus explicit selected counts;
- before/after RSS, mapping RSS/PSS, cgroup current/peak/events;
- estimated versus observed mapping delta;
- prompt/decode timing and an output token fingerprint;
- explicit warnings about non-equivalent measurements.

The JSON writer fails the command if it cannot create the file; it does not
silently discard the report.

## Known limits

- KV-cache and graph-scratch sizes are unavailable from the pinned C API.
- Native `madvise` acceptance is not physical reclamation.
- cgroup current may not move after page advice even when mapping/process RSS
  falls.
- Prefetch may produce no measurable residency change.
- Intra-decode release is unsupported; the safe boundary is after decode and
  context teardown.
- GPU residency and execution are not validated.
