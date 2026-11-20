# Runtime lifecycle measurement and calibration

This milestone adds bounded lifecycle measurement, attribution, and versioned
calibration records. It does not add eviction, release, prefetch, automatic
plan selection, or automatic tensor placement.

## Lifecycle stages

`LifecycleRecorder` captures explicit stages:

1. `baseline`
2. `model_inspected`
3. `model_loaded`
4. `context_created`
5. `prompt_processed`
6. `decode_steady`
7. `context_destroyed`
8. `model_unloaded`
9. `final`

Each snapshot has a strictly increasing sequence, timestamp, model/context
identity where available, process memory, system/cgroup memory, model-mapping
residency, and backend/device observations. Recording is bounded to 32
snapshots; excess oldest snapshots are counted as dropped.

Snapshots are taken only at these boundaries, not during every decode.

## Process memory

From `/proc/self/status`, `smaps_rollup`, and `/proc/self/maps`:

- RSS, peak RSS, virtual size;
- anonymous, file-backed PSS, shared, private;
- swap, locked;
- thread count, mapping count, total mapping bytes.

Model mapping residency is measured by a bounded `/proc/self/smaps` scan
matching the model file name/path. Mapping RSS is resident file-backed pages,
not tensor use or total allocation.

## System and cgroup memory

Physical total, OS availability, cgroup limit/current/peak, cgroup anon/file/
mapped/swap, and PSI pressure remain separate from process memory. Unknown and
unlimited values are explicit. Cgroup values are not added to process values.

## Attribution

`RuntimeAttribution` lists observed and estimated components:

- process RSS, anonymous resident, file-backed resident PSS;
- model mapping resident;
- estimated KV cache;
- compute buffers, graph/activations, backend buffers, staging.

Every component is marked overlapping. Components are **not summed**. No
fabricated unattributed total is produced. `unattributed_bytes` remains `None`
with a reason when exact separation is unavailable.

## Calibration record

`RUNTIME_CALIBRATION_SCHEMA_VERSION = 1`.

`RuntimeCalibration` contains:

- machine identity (schema, CPU architecture/model, RAM total, device count);
- model identity (name, file size, tensor count, architecture, quantization,
  bounded non-cryptographic header fingerprint; no local path);
- runtime configuration (backend/device, load mode, repacking, requested and
  effective context/batch/ubatch/sequences/threads, KV type source, state size);
- predicted `ModelRequirements`;
- all lifecycle snapshots and deltas;
- estimate comparisons;
- attribution and limitations;
- `observation_only = true`.

`state_serialization_size_bytes` is a native serialization size, not allocated/resident
memory. KV effective types are not exposed by the pinned native getter; the
record explicitly says the adapter-default source.

### Validation

`RuntimeCalibration::validate()` checks the schema, observation-only flag,
snapshot sequence, observed-value invariants, predicted requirements, and
attribution. Unknown values remain unknown. `from_json` validates after
deserialization.

### Redaction

The calibration record contains no local paths. `redacted()` is provided for
export consistency; `ramforge calibrate --export` enables it.

## Estimate-versus-observation

`EstimateComparison` reports predicted/observed values, absolute difference,
relative difference only when the prediction is non-zero, validity, and a
reason.

Comparisons are deliberately conservative:

- tensor payload vs loaded mapping RSS — valid only as a resident-page
  comparison, not tensor use;
- GGUF file size vs mapping size — file-backed virtual range, not residency;
- KV estimate vs context-creation RSS delta — **invalid** because compute and
  graph allocations are included;
- resident-weights-plus-KV vs model-load RSS delta — **invalid** because RSS
  includes other categories and page-cache effects.

## Plan integration

`CalibrationEvidence` is produced from a calibration record. The plan evaluator
applies its observed runtime overhead only when all match:

- model fingerprint;
- backend;
- context size;
- requested batch size;
- calibration validity.

Otherwise it emits `calibration_incomparable` and preserves the existing
conservative behavior. `ramforge machine --calibration FILE` supplies evidence
to plan comparison; no plan is selected or executed automatically.

## CLI

```sh
ramforge calibrate MODEL.gguf --ram 1G --context 256 --threads 4 \
  --max-tokens 4 --no-repack --json --output calibration.json

ramforge machine --model MODEL.gguf --ram 200M --context 256 \
  --calibration calibration.json --json
```

## Measured example

SmolLM2-135M Q4_K_M, 256 context, 4 threads, no repacking, warm/uncontrolled:

- model-load RSS delta: 126,545,920 bytes;
- context-create RSS delta: 6,586,368 bytes;
- prompt RSS delta: 8,364,032 bytes;
- decode delta: 0 bytes for the measured steps;
- model-unload delta: -105,639,936 bytes;
- tensor payload prediction: 103,668,480 bytes;
- loaded mapping resident: 103,669,760 bytes;
- KV estimate: 5,898,240 bytes;
- context RSS delta includes compute buffers, so KV comparison is marked
  invalid;
- estimated runtime overhead after subtracting the KV estimate: ca. 9 MB for
  plan calibration.

These are single-run warm observations, not guarantees.

## Lifecycle benchmark

```sh
CALIBRATION_BENCH_RUNS=10 sh scripts/runtime-calibration-bench.sh
```

Five-run smoke:

- 9 lifecycle snapshots: median ca. 2.4–2.6 ms per run;
- calibration build: ca. 0.02 ms;
- JSON round-trip: ca. 1.1 ms.

Snapshot cost is dominated by `/proc` and device discovery. No per-decode
polling is performed.

## Limitations

- No exact native KV, compute-buffer, graph, activation, or staging counters.
- `/proc` RSS includes page cache and allocator effects.
- Model mapping scan is bounded to lifecycle points.
- Only one physical machine and CPU backend were available.
- Calibration is only comparable for matching model/context/batch/backend
  configuration.
- No observation authorizes eviction or release.
