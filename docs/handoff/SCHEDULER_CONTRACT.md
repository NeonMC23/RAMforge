# Scheduler-facing observation contract

The versioned contract is implemented in
`crates/ramforge-runtime/src/observation_contract.rs` and emitted as
`scheduler_observation` inside the existing `node_dispatch` JSON report.

## Schema

- `schema_version`: currently `1`.
- `observation_only`: always `true`.
- `model_instance_id`, `context_id`: optional identities from retained events.
- `strategy`: `disabled`, `systematic`, `targeted`, `adaptive`, or `unknown`.
- `sampling_enabled`, `budget_total`, `budget_remaining`.
- `executions`, `events_retained`, `events_dropped`.
- `global`: deduplicated `PageAccount`.
- `provenance`: explicit evidence categories present.
- `dispatches`: bounded (512) dispatch evidence with source attribution.
- `dispatches_truncated`: number of retained dispatches omitted from the
  embedded evidence list.
- `limitations`: fixed, machine-readable statements describing what the
  observation cannot prove.

`PageAccount` carries requested, sampled, fresh, reused, skipped,
budget-excluded, unsampled, resident, nonresident, unknown, and observation
error fields. `SourceAttribution` adds identity, mapping range, exclusive and
shared pages, and the same per-source state categories.

## Provenance

`ObservationProvenance` distinguishes:

- `fresh_mincore`
- `reused_cache`
- `skipped_fresh_cache`
- `budget_excluded`
- `unsampled`
- `unknown`
- `failed`
- `disabled`

A cache skip is not a residency measurement. A reused observation is not a
fresh `mincore()` result. Unknown and budget-excluded pages are not
nonresident pages.

## Invariants

`SchedulerObservation::validate()` checks:

- `global.requested = sampled + skipped + budget_excluded + unsampled`.
- `sampled = resident + nonresident + unknown`.
- `sampled = fresh + reused`.
- Every source has `requested = exclusive + shared`.
- Every source has the same accounting and state invariants as the global
  account.
- Disabled sampling reports no sampled/fresh/reused/skipped/budget-excluded
  pages.
- `budget_remaining <= budget_total`.
- `observation_only` is true.
- The schema version is supported.

The same invariants are enforced after JSON deserialization.

## Consumer safety

The contract has no eviction, release, prefetch, or final-use authorization
method. Consumers must not infer:

- physical tensor access from residency;
- physical I/O or page faults from residency;
- that a resident page is permanently available;
- that a nonresident page is unused;
- that an unobserved page is unnecessary;
- that a mapping is safe to reclaim.

## Compatibility policy

- Adding fields is backward-compatible within schema version 1.
- Removing fields, renaming fields, or changing the meaning of an existing
  field requires a schema-version increment.
- Consumers should call `validate()` after deserialization.
- Unknown enum values should be treated conservatively; they are not evidence
  of a supported observation.
- The embedded contract duplicates the bounded dispatch list intentionally so
  a future consumer can use one self-contained object. Existing
  `node_dispatch` fields remain available.

## Future extension points

The contract is page-residency-specific. Future machine-resource observations
(CPU, RAM, GPU, VRAM, storage, backend placement, transfer/staging) should be
introduced as separate versioned resource-observation records rather than
being overloaded into page accounting.

The smallest future-facing abstraction is a resource record with explicit
identity, provenance, units, availability state, and validation constraints.
It must represent unavailable information as explicit unknown/unsupported
states, not as zero.
