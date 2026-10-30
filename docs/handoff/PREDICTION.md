# Predictive residency engine

RAMforge's prediction layer is deterministic, bounded, and advisory. It never
authorizes release and never bypasses admission or the native execution gate.

## Inputs

Predictions consume:

- a validated execution profile (`--residency-profile-load`);
- optional validated prediction feedback history
  (`--prediction-history-load`);
- the current model-scoped resource inventory;
- observed/soft/hard memory limits and safety reserve;
- the current admission decision.

A profile is usable only when its model identity, inventory generation,
completeness, and native contract match. The predictor rejects unknown or
ambiguous tensor names and never uses transient graph node IDs.

## Confidence heuristic

Confidence is expressed in basis points and is explainable:

```text
base = 5500 + 1000 * min(complete_observations, 4)
confidence = base                              (no history samples)
confidence = (base * 2 + historical_accuracy) / 3   (history samples)
```

`historical_accuracy` is `hits / (hits + misses) * 10000` from the validated
prediction history. Classes are:

- high: `>= 7500`
- medium: `>= 5000`
- low: `< 5000`

The CLI threshold (`--prediction-min-confidence PERCENT`) is applied on top of
this classification. The predictor is deterministic for identical inputs.

Predicted order comes from the profile's stable tensor `last_order`. The
expected distance is `last_order - first_order` where observed. Predictions are
ranked by confidence, predicted order, size, and stable tensor index.

## Planner

`prediction_planner.rs` converts predictions into explicit decisions:

- `StageNow` — high confidence and within the staging window;
- `Prefetch` — medium confidence and within the speculative budget;
- `Defer` — budget/window exhausted;
- `AvoidSpeculative` — below confidence or prediction disabled;
- `Reject` — not file-backed, unknown size, or admission rejected.

The planner deduplicates page ranges per mapping and uses unique page bytes for
budget accounting. It never claims that `MADV_WILLNEED` made pages resident.

`release_authorized` is always false. Predictions do not prove final use and are
not used as release eligibility. Release remains managed by the existing
residency manager and native lifetime gate at a post-operation boundary.

## Staging coordinator

The coordinator is synchronous:

- one bounded native prefetch call over the selected indices;
- native page-range merging handles overlapping/shared pages;
- configured `max_in_flight` is reported, but the implementation performs one
  synchronous native call, so it cannot create uncontrolled concurrent I/O;
- requested bytes, native-accepted bytes, errors, and observed mapping delta
  are separate fields;
- no background thread, timer, or native handle can outlive the model.

If staging fails, the error is reported and the operation continues. A failed
staging request is not presented as accepted or resident.

## Prediction feedback

After a complete current trace, predictions are compared with observed stable
tensor accesses:

- hit/miss counts;
- staged request count and bytes;
- staged bytes whose tensor was not observed (wasted speculative staging);
- rolling accuracy in basis points.

The feedback history is separate from the execution profile, versioned, bounded
to 16 MiB and 100,000 records, atomically written, and identity-validated.
It never contains transient native IDs. `PredictionHistory::reset()` clears it.

Feedback is used only to blend future confidence. It does not rewrite the
profile, admission, or release decisions.

## CLI

```sh
ramforge run MODEL --backend forgecore --ram 192M --ctx 256 \
  --graph-trace --graph-trace-events 16384 \
  --residency-policy observe --residency-target 128M \
  --residency-profile-load profile.json \
  --prediction --prediction-min-confidence 40 \
  --prediction-window 64M --prediction-speculative 16M \
  --prediction-history-load history.json \
  --prediction-history-save history-next.json \
  --residency-report --residency-report-json report.json
```

`--prediction` requires `--residency-profile-load` and conflicts with the
manual `--residency-prefetch` flag. `--prediction-history-save` requires graph
tracing so feedback has observed accesses.

The report's `prediction` section contains the source, confidence counts,
predictions, planner decisions, requested/accepted staging bytes, observed
mapping delta, and feedback counters. JSON and human output use the same data.

## Deterministic simulation

```sh
ramforge simulate --budget 12K --window 8K --json
ramforge simulate --budget 12K --window 8K --wrong-history
```

The simulator compares a baseline against predictive staging with a synthetic
five-resource model. It reports hits/misses, reread bytes, staged/wasted bytes,
peak resident bytes, budget violations, and release authorization. It is
simulation evidence only, not hardware performance evidence.

Tests cover correct/wrong history, zero budget, unknown size, bounded staging,
admission rejection, stale identity, and the invariant that prediction never
authorizes release.

## Limitations

- Prediction is heuristic, not a model of guaranteed future access.
- `MADV_WILLNEED` may produce zero observed mapping change.
- Staging may be wasted when the next prompt differs from the profile.
- KV/scratch/runtime memory remains unknown and cannot be predicted here.
- Intra-decode eviction remains unsupported.
- Hardware throughput improvement is not claimed; only requested/accepted and
  observed measurements are reported.
