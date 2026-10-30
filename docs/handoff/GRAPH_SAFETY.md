# Graph trace and safety boundaries

This document defines what the pinned graph trace proves, what the manifest and
execution profile infer, and why intra-decode page release remains unsupported.

## Native trace contract

`--graph-trace` installs the pinned public scheduler eval callback
(`ggml_backend_sched_eval_callback`). For observed node groups:

1. the scheduler calls the callback with `ask=true`;
2. RAMforge/forgeCORE requests observation;
3. the scheduler computes the observed group;
4. the scheduler calls the callback with `ask=false` after synchronizing that
   node group’s backend work;
5. forgeCORE records a `CompletedSynchronized` event.

The callback is bounded and copies only:

- execution id;
- sequence;
- transient native node id;
- optional node name and op name;
- node byte size;
- dependency names and transient source ids;
- phase and completion status.

It retains no native pointers. Trace buffer overflow increments dropped events
and does not silently present a truncated capture as complete.

## What a completed node-group event proves

It proves that the observed node group’s backend work was synchronized at the
time of the callback. It does **not** prove:

- that the tensor will not be reused by a later node in the same graph;
- that the tensor will not be reused by a reused graph;
- that the next token’s graph will not access the same mapping range;
- that a tensor’s memory can be released while decode continues.

The trace callback is diagnostic. It is not a native lifetime-token API.

## Manifest analysis

`graph_manifest.rs` builds a manifest from one or more trace events:

- requested events become nodes;
- transient native node ids are used only to resolve dependencies within the
  captured execution(s);
- dependency edges are marked known only when the mapping is unambiguous;
- unresolved external inputs are recorded by name;
- dependency cycles are detected and counted; a cyclic manifest is not usable
  as a complete profile;
- tensor-to-consumer relationships come from source/dependency names;
- graph-order last use is the maximum consumer node index for a tensor;
- completion is proven only for the consumer’s synchronized node id.

A manifest whose capture contains multiple executions or duplicate native ids
can have `edges_known=false`. This is deliberate: merging transient ids across
graphs is invalid. `intra_decode_safe` is always `false`.

`GraphManifestSummary` reports node/tensor counts, edge/completion/cycle
confidence, graph-order and unknown last-use counts, and the safety flag. A
multi-execution capture is summarized conservatively as one merged manifest for
diagnostics, while `ExecutionProfile` analyzes each execution separately; the
CLI prints an explicit note when this distinction applies.

## Execution profiles

`execution_profile.rs` summarizes repeated complete observations without
retaining native pointers:

- model id and inventory generation;
- profile schema version;
- complete and incomplete observation counts;
- last node/tensor counts;
- stable tensor name → first/last graph-order access and consumer count;
- invalidation reason.

`ExecutionProfile::update_from_trace` splits a capture by execution id and
analyzes each execution separately. This prevents native node ids from
different graphs from being merged.

A profile is valid only when:

- the model id and inventory generation match;
- at least one complete observation exists;
- the profile has not been invalidated.

Profiles may be persisted with `--residency-profile-save` and loaded with
`--residency-profile-load`. Files are schema/profile versioned, bounded to
16 MiB/100,000 entries, atomically written, and contain only stable tensor
names and graph-order access. Native node IDs are never serialized. Loading
validates canonical model path, total model file size, primary-file mtime,
inventory generation, native contract version, completeness, and invalidation.
For sharded models, a same-size edit to a non-primary shard would not change
the primary mtime; rotate/invalidate the profile explicitly in that case. `ramforge profile` inspects a
file and can validate it against a model. A loaded profile is a heuristic for a
later operation, not a promise that the next prompt accesses the same
tensors.

## Reuse-aware planning

`reuse-aware` consumes a valid profile only at the post-operation safe boundary.
It orders the least recently used observed tensors first, then applies the same
bounded target/reserve/eligibility rules. The plan reports `profile_used=true`
only when a valid profile was actually used. If no profile is available, the
planner falls back deterministically and records that fact; it does not claim
reuse awareness.

The profile is evidence about one observed execution. It is a heuristic for a
later operation, not a promise that the next prompt or token will access the
same tensors.

## Operation boundary and native gate

The native patch adds a model-scoped shared/exclusive execution gate. A context
holds a shared lease from asynchronous graph submission until `synchronize`
(or destruction) observes completion. Page release attempts the exclusive side
without blocking and returns `EBUSY` if any context is in flight. This makes
the between-operation boundary enforceable and prevents a release call from
racing an active graph.

The gate is deliberately **not** an intra-decode lifetime proof. It proves
that no observed asynchronous execution is in flight at the release instant;
it does not prove that a future token/graph will not reuse a weight. RAMforge
therefore marks resources `ReleaseEligible` only after the native model reports
zero contexts/in-flight work following session drop, and keeps
`FinalUseProven` unreachable for model weights.

## Why intra-decode release is unsupported

The pinned implementation provides synchronized node-group completion, not a
resource-lifetime guarantee. A safe intra-decode release would need a native
contract that proves all of the following for the specific mapping range:

1. every graph that can reference the range has been identified;
2. a reused graph cannot retain a borrowed reference to it;
3. the current token cannot issue a later read after the observed node;
4. the next token’s graph cannot reuse the range before a new mapping/refault;
5. the release operation cannot unmap, relocate, or free native model mappings
   that native code still assumes are valid.

The current callback does not supply (1)–(4), and RAMforge must not infer them
from graph-order last use. The safe mutation point remains after decode
completion and context teardown.

A minimal future native contract would expose explicit resource-lifetime
tokens or ownership counters tied to the scheduler graph and a guaranteed
release boundary where no graph, cached graph, or backend retains the range.
Until such a contract exists, intra-decode streaming/eviction is documented as
unsupported.

## Prediction and release safety

Predictions may order staging and may inform which profile is used at a
post-operation boundary, but `PredictionPlan::release_authorized` is always
false. Confidence, predicted reuse, and graph-order evidence are not lifetime
proof and cannot promote a resource to `ReleaseEligible`. Native quiescence,
native exclusive gate acquisition, model/context identity, and the manager's
safe-boundary lifecycle remain mandatory.

## Operational checks

- `ramforge support` reports graph dependency/last-use analysis as partial and
  intra-decode residency as unsupported.
- `--graph-trace-detail` is diagnostic and should not be enabled for throughput
  measurements.
- A truncated trace is reported with dropped-event counts; its manifest is not
  treated as complete.
- No policy profile bypasses the safe boundary.
