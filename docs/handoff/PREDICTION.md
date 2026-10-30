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
confidence = base                              (no usable history)
confidence = (base * 2 + historical_accuracy) / 3   (usable history)
```

`historical_accuracy` uses at least 3 samples for a tensor; if unavailable, it
falls back to global history only after at least 10 samples. A single hit does
not turn a cold profile into 100% confidence. Classes are:

- high: `>= 7500`
- medium: `>= 5000`
- low: `< 5000`

The CLI threshold (`--prediction-min-confidence PERCENT`) is applied on top of
this classification. The predictor is deterministic for identical inputs.

Predicted order comes from the profile's stable tensor `last_order`. The
expected distance is `last_order - first_order` where observed. Predictions are
ranked by confidence, predicted order, size, and stable tensor index.

## Metric definitions

The report separates the following; they are not interchangeable:

- **Prediction generated** — an advisory resource prediction.
- **Prediction matched observed** — the stable tensor appeared in the later
  observed execution. This is prediction accuracy, not prefetch usefulness.
- **Prediction matched within horizon** — the tensor appeared within the first
  `horizon` observed model resources. The current evaluation horizon is the
  configured lookahead (64).
- **Prefetch requested** — resources selected by the planner.
- **Prefetch accepted** — native prefetch returned success. Per-resource
  acceptance is only reported when the aggregate accepted bytes equal the
  requested bytes; partial acceptance is marked unknown.
- **Prefetch followed by observed access** — a selected resource was later
  observed. This is a usefulness proxy, not proof of fault avoidance.
- **Prefetch with observed residency** — currently unknown per resource; the
  pinned API returns aggregate accepted bytes, not per-resource residency.
- **Prefetch not followed within evaluation** — selected resources not observed.
- **Prefetch benefit unknown** — accepted resources for which no page-fault or
  I/O attribution is available.
- **Duplicate/overlap avoided** — planner decisions that deduplicated a request.
- **Rejected/deferred** — explicit planner outcomes.
- **Staging bytes requested/accepted/observed** — three separate quantities.
- **Physical residency benefit** — not inferred from `madvise` success or RSS.
- **Performance benefit** — only from repeated measured timing; not implied by
  prediction accuracy.

Precision = matched predictions / generated predictions. Recall uses only
observed **inventory resources**, not all transient graph tensor names. False
positive rate = unmatched predictions / generated predictions. All counters are
deterministic for fixed inputs and avoid overlapping-page double counting.

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
inventory resources. The history separates prediction accuracy, within-horizon
matches, prefetch accepted/followed/not-followed, unknown benefit, duplicate
avoidance, and calibration buckets. Legacy v1 histories migrate by filling
new fields with zero; v2 is written on save.

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

## Residency observation and cost accounting

`--residency-observe` uses Linux `mincore()` over bounded tensor page ranges
located through `/proc/self/maps`. It never faults pages in, evicts pages, or
exposes virtual/physical addresses. Missing mappings, unsupported kernels, and
pages beyond the inspection budget are **unknown**, not nonresident.

### Targeted sampling

When prediction is enabled, prediction and planning run first because they do
not depend on page residency. The selected prefetch names are then passed to
`observe_page_residency_targeted`. Observation order is deterministic:

1. selected prefetch resources in inventory order;
2. non-selected resources with the remaining page budget.

Unique physical pages are keyed by native mapping index and page index, so
overlapping tensor ranges in one mapping consume the page budget once. Pages in
missing/unreadable mappings remain unknown. A selected resource larger than
the remaining budget is reported as partially observed, not as resident; its
unobserved pages stay unknown. A general sample never inherits targeted
coverage claims.

Snapshots expose separate populations: configured page budget, selected
resources fully/partially/not observed, targeted pages available and sampled,
general pages available and sampled, unique inspected pages, resident,
nonresident, and unknown pages. `targeted_coverage_bps` and
`general_coverage_bps` refer to those separate populations and must not be
combined into one headline percentage. The resource list includes `targeted`,
`total_pages`, `inspected_pages`, `unknown_pages`, and `sampled` so partial
coverage is visible per resource.

On SmolLM2 at a 2,048-page budget, targeting changed the earlier arbitrary
sample into 2,048/4,065 selected pages sampled (33 selected resources fully
observed, 4 partial, 19 not observed); remaining general budget was 0. A
64-page budget sampled 64 targeted pages (2 fully observed, 3 partial, 51 not
observed). These partial results are not evidence for the unobserved resources.

### Observation points and attribution

Available snapshots are initial (after static prediction/planning, before
staging), post-staging, and final. A safe observation immediately before each
native resource access is **not implemented**: llama.cpp/ggml does not expose
an access callback at the required lifetime boundary, and adding one would
require a native change. The missing interval is explicit;
`before_access_observation_available=false`. Consequently eviction before
access and avoided reactive-load causality remain unknown. Do not infer them
from the final snapshot or from ordering alone.

Prefetch correlation compares selected resource page snapshots before and
after staging. Classifications are:

- `AlreadyResident` -> `Redundant`;
- `BecameResident` + observed access -> `PotentiallyUseful`;
- `BecameResident` + no observed access -> `Wasted`;
- `RemainedNonResident` -> `Ineffective`;
- insufficient/partial pages -> `Unknown`.

`ClearlyUseful` and `EvictedBeforeUse` are not assigned without a before-access
observation and per-resource benefit evidence. An accepted staging request is
never treated as proof of residency.

### Measured overhead

Earlier full observation on the SmolLM2 mapping covered 25,581 pages at roughly
54 ms/snapshot. The targeted implementation uses interval arithmetic for the
available population instead of allocating one key per page, so the 64-page
sampling budget stays small: in the final runs, all three snapshots together
cost about 1.6–3.8 ms for SmolLM2 and 3.4–4.0 ms for Qwen2.5-1.5B. This is
diagnostic overhead, not normal execution overhead. Observation remains
disabled by default.

End-to-end latency includes model load, profile/history loading, prediction,
planning, staging, observation, inference, and explicit synchronization. TTFT
excludes pre-execution staging; inference latency is not a proxy for total run
cost. Report serialization is outside `end_to_end_ns`; process wall time
includes it.

### Controlled cache conditions

`scripts/prediction-bench.py --cache-condition isolated-nonresident` copies the
model to an isolated file, calls `fsync` then `posix_fadvise(POSIX_FADV_DONTNEED)`
only on that copy, and verifies initial page-cache state with `mincore` before
starting RAMforge. No global cache drop, privileged command, or modification
of the source model is used. The method is labeled controlled only when every
probed page is nonresident. Partial, warm, unknown, and preparation-failure
runs remain distinct; `cache_controlled=false` is preserved. The profile copy
is rewritten only for model path/size/mtime identity; access records are
unchanged. This controls the file-page-cache state at process start, not the
model's mapped residency after loading and not physical reclamation.

## Calibration findings

On the SmolLM2 test profile, 64 predicted model resources all matched the next
execution (precision 100%) but only 63 fell within the 64-resource horizon and
only 23.5% of the 272 observed inventory resources were covered by the
lookahead. All 57 accepted prefetch resources were later observed, but their
per-resource residency and fault benefit are unknown. This is the key
distinction: a 100% prediction-match rate is **not** evidence of useful
physical prefetch or performance gain.

Cold-start predictions were medium confidence; a warm feedback history raised
them to high confidence. The real-model benchmark did not find a universal
speedup. Under a 256 MiB cgroup, predictive arms showed fewer major faults and
lower TTFT on several workloads, but long-workload results were mixed and this
was not a controlled cold-cache experiment. Synchronous native prefetch latency
was about 3.9–5.4 ms in the constrained smoke matrix; TTFT excludes that
pre-operation time, while process wall time includes it.

## Deterministic simulation

`ramforge simulate` exposes 16 required deterministic scenarios via
`--scenario NAME` or `--scenario all`.

Scenario names:

`useful`, `already-resident`, `no-residency-effect`, `predicted-never-accessed`,
`incorrect-prediction`, `incorrect-already-resident`, `late-prefetch`,
`staging-rejected`, `deferred-then-accessed`, `evicted-before-access`,
`unknown-initial`, `unknown-effect`, `overlapping-duplicate`,
`constrained-competition`, `observation-overhead`, `prediction-cost-worse`.

Each scenario runs five equivalent strategies: baseline, reactive, predictive,
conservative, aggressive. Every result reports initial resident/nonresident/
unknown bytes, predictions and precision/recall, staged/newly resident/
redundant/ineffective/wasted/deferred/rejected bytes, eviction-before-access,
reread bytes, peak residency, budget violations, prediction/planning/
observation/staging/reread/execution/synchronization costs, total end-to-end
cost, and net benefit. Simulated observation also reports targeted/general
page coverage, partial selected resources, before-access availability, and a
cache-condition label. These are synthetic model parameters; they are not
hardware measurements and are never fed into empirical claims.

Accounting:

```text
total = setup + prediction + planning + observation + staging
      + reread + execution + synchronization
net_benefit = baseline_total - strategy_total
improved = benefit_known AND net_benefit > 0
```

`benefit_known` is false for unknown initial residency or unknown staging effect.
Unknown pages are never counted as reclaimable or useful. Prediction precision
is reported independently from net benefit.

Standard `useful` 12K/8K scenario: baseline 10,217 ns, predictive 2,675 ns,
net benefit **7,542 ns**; 4,096 staged/newly resident bytes, 4,096 reread bytes
avoided. `observation-overhead` scenario: predictive total 1,002,675 ns,
net benefit **-992,458 ns**; a correct prediction is slower because observation
cost dominates. `already-resident` stages 4,096 redundant bytes and does not
improve. `unknown-effect` and `unknown-initial` mark benefit unknown and cannot
claim improvement. Standard scenarios are deterministic, budget-safe, and
tested for all strategies.

These are simulated nanosecond units, not hardware measurements.
## Limitations

- Prediction is heuristic, not a model of guaranteed future access.
- `MADV_WILLNEED` may produce zero observed mapping change.
- Staging may be wasted when the next prompt differs from the profile.
- KV/scratch/runtime memory remains unknown and cannot be predicted here.
- Intra-decode eviction remains unsupported.
- Hardware throughput improvement is not claimed; only requested/accepted and
  observed measurements are reported.
