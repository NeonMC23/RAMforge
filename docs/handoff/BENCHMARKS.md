# Residency benchmark and diagnostics guide

The benchmark harness is opt-in. It uses a real small GGUF model supplied by
the user and never downloads a model. It records a versioned JSON report per
run plus a TSV summary so a result can be reproduced and audited.

## Build

```sh
export FORGE_LLAMA_DIR="$PWD/../native"
export LD_LIBRARY_PATH="$FORGE_LLAMA_DIR/build/bin:${LD_LIBRARY_PATH:-}"
cargo build --release -p ramforge-cli
```

## Run

```sh
FORGE_TEST_MODEL=/models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
RAMFORGE_BIN=target/release/ramforge \
TARGET=128M CTX=256 MAX_TOKENS=4 REPEATS=3 \
POLICIES="off observe conservative memory-bound reuse-aware active" \
PREFETCH_MODES="off on" \
sh scripts/residency-bench.sh
```

Useful environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `FORGE_TEST_MODEL` / `RAMFORGE_MODEL` | required | small real GGUF |
| `RAMFORGE_BIN` | `target/release/ramforge` | binary to run |
| `TARGET` | `192M` | soft residency target and `--ram` value |
| `RESERVE` | empty | optional `--residency-reserve` |
| `CTX` | `256` | context length |
| `MAX_TOKENS` | `4` | generated tokens |
| `PROMPT` | `The capital of France is` | prompt |
| `PREFETCH` | `64M` | bounded staging budget when prefetch is on |
| `PREFETCH_MODES` | `off on` | staging comparison |
| `POLICIES` | all six | policy comparison |
| `REPEATS` | `1` | runs per cell |
| `ENFORCE` | `0` | set to `1` for cgroup enforcement |
| `ENFORCE_TIMEOUT` | `120` | seconds |
| `GRAPH_TRACE_EVENTS` | `16384` | trace cap for reuse-aware |
| `RAMFORGE_BENCH_OUT` | `bench-out` | output directory |

The harness passes `--residency-policy reuse-aware` with `--graph-trace` and
`GRAPH_TRACE_EVENTS` (default 16384). A 4096-event cap on a multi-token capture
is expected to be incomplete; the profile is then rejected and the plan reports
`profile_used=false`.

For prediction runs, `PREFETCH_MODES` defaults to `off` because
`--prediction` and `--residency-prefetch` are mutually exclusive. Graph tracing
is enabled automatically so feedback and profile saves are valid.

The prediction harness in `scripts/prediction-bench.py` additionally supports:

```sh
python3 scripts/prediction-bench.py \
  --model /models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --profile profile.json --observe --observe-pages 64 \
  --cache-condition isolated-nonresident --repeats 5
```

`isolated-nonresident` copies the model to a separate file, flushes the copy,
applies `POSIX_FADV_DONTNEED` only to that copy, and verifies the probed pages
with `mmap` + `mincore` before each run. The source file is not modified and no
global cache command is used. A run is marked controlled only when every probed
page is nonresident. `warm` uses the same isolated copy and verification without
the cache hint; partial, unknown, and preparation-failure states are recorded
separately. These labels control file-page-cache state at process start, not
mapped residency after loading or physical reclamation.

The script writes:

- `results.tsv` — one row per run;
- `<timestamp>-<policy>-<prefetch>-r<N>.json` — versioned operation report;
- `<timestamp>-<policy>-<prefetch>-r<N>.stderr` — complete diagnostic log.

## Interpreting the TSV

Important columns:

- `selected` — planned tensor count;
- `estimated_reclaim_bytes` — planner estimate, not an achieved result;
- `observed_mapping_delta_bytes` — measured mapping RSS change (negative means
  fewer resident mapping pages);
- `rss_before` / `rss_after` — whole-process RSS;
- `cgroup_before` / `cgroup_after` — cgroup v2 current, which includes the
  whole cgroup and may be much larger than this process;
- `staging_requested_bytes` — deduplicated page-range budget from RAMforge;
- `staging_native_requested_bytes` — bytes accepted by the native prefetch;
- `profile_source` — `current_trace`, `loaded_file`, or `none`;
- `prediction_count`, `prediction_staged`, `prediction_requested_bytes`,
  `prediction_accepted_bytes`, `prediction_hits`, `prediction_misses`;
- `native_contract`, `native_quiescent`, `lifetime_states` — native contract
  version, point-in-time quiescence, and resource state counts;
- `native_released_bytes` — bytes submitted to `madvise`, not confirmed
  reclamation;
- `ttft_ns`, `decode_ns`, `total_ns` — observed wall-clock from the CLI;
- `output_fingerprint` — FNV-1a fingerprint of generated token ids;
- `stderr_sha256` — hash of the diagnostic log (includes timings/noise).

Always inspect the paired JSON when a number is surprising. The JSON separates
ledger categories, provenance, reconciliation, plan estimates, native results,
and measurements.

## Baseline measured on this checkout

Model: SmolLM2-135M-Instruct Q4_K_M (`98.87 MiB` tensor payload, 272
file-backed tensors, one mapping). Context 256, target 128 MiB, 4 greedy
tokens, CPU, no cgroup enforcement, two repeats for the matrix rows. Medians:

| Policy | Prefetch | Selected | Mapping delta | Process RSS before/after | Cgroup delta | TTFT |
|---|---|---:|---:|---|---:|---:|
| off | off | 0 | 0 B | 164.30 / 164.30 MiB | ~0 B | 28.2 ms |
| observe | off | 12–13 | 0 B | 164.25 / 164.25 MiB | ~0 B | 28.4 ms |
| conservative | off | 0 | 0 B | 164.21 / 164.21 MiB | ~0 B | 25.5 ms |
| memory-bound | off | 12–13 | -38,387,712 B | 164.23 / 127.63 MiB | ~0 B | 27.0 ms |
| active | off | 13 | -38,748,160 B | 164.30 / 127.37 MiB | ~0 B | 27.5 ms |
| reuse-aware (valid profile) | off | 134 | -46,923,776 B | 172.23 / 127.48 MiB | ~0 B | 26.7 ms |

The harness enables graph tracing for reuse-aware runs and defaults to a
16,384 event cap. With the CLI's 4,096 default, a four-token capture was
truncated, the profile was correctly invalid, and `profile_used=false`. That
distinction is visible in the JSON.

Bounded staging on the same model with a 64 MiB budget:

- plan: 227 selected tensors, 63.98 MiB union of selected page ranges;
- native accepted: 63.98 MiB;
- mapping RSS before/after: 98.87 MiB / 98.87 MiB (delta 0).

The zero mapping delta after a successful hint is the expected honest outcome
on an already-resident mapping; it is not a failure to report.

`conservative` selected nothing because every tensor range had non-zero shared
page bytes (aligned neighboring tensors). It intentionally refuses to claim
those pages as unique. `memory-bound` and `active` released tensors while the
native callback protected shared pages, and the measured mapping RSS fell by
the submitted byte count for the selected ranges.

## Native lifetime and profile scenarios

For a profile run, save a complete profile in one process and validate it before
loading it in another:

```sh
ramforge run MODEL --backend forgecore --ram 192M --ctx 256 \
  --graph-trace --graph-trace-events 16384 --residency-policy reuse-aware \
  --residency-target 128M --residency-profile-save /tmp/model.profile.json \
  --prompt "The capital of France is" --max-tokens 1
ramforge profile /tmp/model.profile.json --model MODEL --json
ramforge run MODEL --backend forgecore --ram 192M --ctx 256 \
  --residency-policy reuse-aware --residency-target 128M \
  --residency-profile-load /tmp/model.profile.json --residency-report \
  --prompt "The capital of France is" --max-tokens 1
```

Record `native_lifetime.contract_version`, `model_instance_id`, `quiescent`,
`context_id`, `execution_generation`, `resource_lifetime_states`, and
`profile_source`. A `Busy` release is a failure/partial result, not a
successful reclamation.

## Predictive measurements

The dedicated harness is `scripts/prediction-bench.py`. It rotates five arms
across four workloads, runs five repetitions per arm/workload by default, and
writes per-run JSON, `results.tsv`, and `summary.json`. Cache state was not
controlled: **no run is labeled cold**. The script records
`cache_controlled=false`.

```sh
python3 scripts/prediction-bench.py \
  --model MODEL.gguf --profile profile.json --history history.json \
  --out /tmp/pred-bench --repeats 5 --target 192M
# constrained:
python3 scripts/prediction-bench.py \
  --model MODEL.gguf --profile profile.json --history history.json \
  --out /tmp/pred-press --repeats 5 --target 256M --enforce

# bounded residency observation (diagnostic; adds measurable overhead)
python3 scripts/prediction-bench.py \
  --model MODEL.gguf --profile profile.json --history history.json \
  --out /tmp/pred-observe --repeats 2 --target 192M \
  --observe --observe-pages 32768
```

Arms: baseline (no prediction/no prefetch), reactive (existing bounded
prefetch), predictive, predictive conservative, predictive aggressive. Run
order is rotated by workload/repetition. Cache state was not controlled or
verified, so no result is labeled cold; the summary records
`cache_controlled=false`.

SmolLM2-135M Q4_K_M, five repetitions, no cgroup enforcement, 192 MiB target:

| Workload | Baseline TTFT | Predictive TTFT | Mapping delta | Major-fault delta |
|---|---:|---:|---:|---:|
| alternate | 44.45 ms | 44.84 ms | 0 B | 0 / 0 |
| short-long | 30.55 ms | 30.38 ms | 0 B | 0 / 0 |
| short-short | 30.81 ms | 30.91 ms | 0 B | 0 / 0 |
| long-short | 659.14 ms | 706.49 ms | 0 B | 0 / 0 |

Under a 256 MiB cgroup, five repetitions:

| Workload | Baseline TTFT | Predictive TTFT | Conservative TTFT | Major faults baseline/predictive |
|---|---:|---:|---:|---:|
| alternate | 86.42 ms | 81.13 ms | 77.93 ms | 5 / 4 |
| short-long | 70.29 ms | 65.11 ms | 66.63 ms | 3 / 2 |
| short-short | 73.74 ms | 65.74 ms | 64.65 ms | 3 / 2 |
| long-short | 689.54 ms | 695.68 ms | 714.56 ms | 5 / 4 |

Page residency is observed with `mincore()` over bounded ranges located via
`/proc/self/maps`. Full observation (25,581 pages) costs about 54 ms per
snapshot; three snapshots cost about 158 ms. A 2,048-page sample costs about
3.7 ms per snapshot but leaves 23,533 pages unknown. Full observation found all
25,581 pages resident before prefetch; all 57 selected resources were
`already_resident`/`redundant`, 0 became resident, 0 remained nonresident, and
0 were clearly useful. Sampled observation classified 57 as unknown. This is
why observation is off by default.

Three-run observation overhead comparison (prediction enabled, same workload,
uncontrolled cache):

| Observation | End-to-end median | Inference median | Observation time |
|---|---:|---:|---:|
| none | 231.44 ms | 18.10 ms | 0 ms |
| sampled 2,048 pages | 248.78 ms | 19.47 ms | 11.49 ms |
| full 32,768-page budget (25,581 actual) | 357.92 ms | 19.25 ms | 158.72 ms |

The full observation added ~126 ms over no observation, over 5× the measured
inference time for this short workload. Detailed observation is therefore a
diagnostic mode, not a default benchmark configuration.

Page faults are process-wide `/proc/self/stat` counters sampled before model
load and after decode. They include loader, tokenizer, trace, and harness
activity; the harness reports deltas but cannot attribute them exclusively to
prefetch. Process RSS and cgroup usage are also process/group measurements, not
per-resource proof.

Predictive staging latency was measured separately: median native prefetch call
latency was 3.9–5.4 ms across workloads in a two-repeat constrained smoke run.
TTFT excludes that pre-operation staging time; process wall time includes it.
`end_to_end_ns` also excludes report serialization; the harness wall-clock column
includes it.

End-to-end latency (model load + profile/history + prediction + planning +
staging + inference + explicit synchronization), five repetitions:

| Workload / condition | Baseline end-to-end | Predictive end-to-end | Baseline inference | Predictive inference |
|---|---:|---:|---:|---:|
| short-short / no pressure | 236.62 ms | 252.52 ms | 30.68 ms | 31.42 ms |
| short-long / no pressure | 397.00 ms | 400.05 ms | 136.01 ms | 131.00 ms |
| alternate / no pressure | 321.75 ms | 330.15 ms | 89.96 ms | 92.05 ms |
| long-short / no pressure | 875.17 ms | 876.69 ms | 662.69 ms | 658.24 ms |
| short-short / 256 MiB | 285.09 ms | 294.98 ms | 68.54 ms | 66.04 ms |
| short-long / 256 MiB | 447.93 ms | 469.54 ms | 173.18 ms | 172.29 ms |
| alternate / 256 MiB | 375.03 ms | 379.08 ms | 131.77 ms | 121.83 ms |
| long-short / 256 MiB | 911.15 ms | 921.13 ms | 696.69 ms | 682.56 ms |

Under the cgroup, inference latency sometimes improved while end-to-end latency
worsened because profile/prediction/staging/setup cost is included. No
predictive speedup is claimed.

RSS remained around 168–184 MiB; mapping delta was **0 B** in all arms, so no
physical mapping residency benefit was observed. Prediction metrics for the
warm history were stable across workloads: 64 generated, 64 matched overall,
63 within the 64-resource horizon, precision 100% (prediction accuracy), recall
23.5% of 272 observed inventory resources, 57 native-accepted resources
followed by observed access, 57 with unknown physical benefit, 0 not-followed,
and 7 duplicate/overlap requests avoided. The 100% precision reflects that all
predicted model tensors are touched by a transformer execution; it does not
mean 64 useful prefetches.

### Qwen2.5-1.5B Q4_0, five repetitions, controlled-nonresident

100/100 runs succeeded. The isolated copy was verified nonresident before every
run (`cache_controlled=true`); model, profile identity, 1 GiB target, and
four workloads were otherwise unchanged. Each observation sampled 64 selected
pages; one selected resource was fully observed, one to two were partial, and
36–43 were not observed.

| Workload | Baseline e2e | Reactive e2e | Predictive e2e | Conservative e2e | Aggressive e2e | Baseline/Predictive inference |
|---|---:|---:|---:|---:|---:|---:|
| short-short | 1371.3 ms | 1336.9 ms | 1349.6 ms | 1366.2 ms | 1363.0 ms | 697.5 / 681.8 ms |
| short-long | 2213.1 ms | 2235.8 ms | 2323.4 ms | 2388.9 ms | 2242.9 ms | 1481.7 / 1554.9 ms |
| long-short | 6987.3 ms | 6957.9 ms | 6951.4 ms | 7136.1 ms | 6904.9 ms | 6302.9 / 6281.8 ms |
| alternate | 1996.2 ms | 1947.6 ms | 1988.1 ms | 1984.8 ms | 2044.5 ms | 1241.9 / 1227.8 ms |

Predictive was faster than baseline on three of four workloads and slower on
short-long; conservative was slower on all four, and aggressive was mixed. This
is a five-repetition CPU-only controlled file-cache result, not a general
speedup claim. Mapping delta was 0 B. Major-fault medians were 60/61/61/62 for
baseline versus 59/60/60/61 for predictive and 57/58/58/59 for aggressive.
Only 64 targeted pages were sampled in each run, so most selected resources
remain unknown.

## Controlled file-cache experiments (2026-10-30)

Both matrices used SmolLM2-135M Q4_K_M, a loaded execution profile, 192 MiB
`--ram`, 256 context, five repetitions, rotated arm order, and
`--observe --residency-observe-pages 64`. All 200 runs succeeded. The model was
an isolated copy; the source model was never modified. Warm and
isolated-nonresident matrices are kept separate.

Cache preparation:

- `warm`: isolated copy, populated read, `mmap` + `mincore` verification, no
  cache hint; all 100 probes reported warm pages; `cache_controlled=false`.
- `isolated-nonresident`: isolated copy, `fsync`, `posix_fadvise(DONTNEED)` on
  that copy only, then `mmap` + `mincore` verification; all 100 probes reported
  every probed page nonresident; `cache_controlled=true`.
- No global cache drop, privileged command, or source-file eviction was used.
  This controls file-page-cache state at process start, not mapped residency
  after model load or physical reclamation.

### Controlled-nonresident medians

| Workload | Arm | End-to-end | Inference | TTFT | Observation | Major faults | Residency transition |
|---|---|---:|---:|---:|---:|---:|---|
| short-short | baseline | 323.9 ms | 78.9 ms | 78.8 ms | 1.6 ms | 3 | general sample |
| short-short | reactive | 338.0 ms | 75.2 ms | 75.1 ms | 7.2 ms | 3 | 64 targeted pages; 2 potential, 0 unknown claim |
| short-short | predictive | 343.3 ms | 73.4 ms | 73.4 ms | 3.8 ms | 3 | 64 targeted pages; 2 potential, 56 unknown |
| short-short | conservative | 318.1 ms | 79.4 ms | 79.3 ms | 1.8 ms | 3 | no staging |
| short-short | aggressive | 319.7 ms | 72.4 ms | 72.4 ms | 3.1 ms | 2 | 64 targeted pages; 2 potential, 57 unknown |
| short-long | baseline | 491.4 ms | 192.6 ms | 78.8 ms | 1.7 ms | 3 | general sample |
| short-long | reactive | 467.2 ms | 181.8 ms | 67.3 ms | 5.8 ms | 3 | targeted sample |
| short-long | predictive | 475.7 ms | 182.8 ms | 65.9 ms | 3.7 ms | 3 | 2 potential, 56 unknown |
| short-long | conservative | 470.1 ms | 189.9 ms | 79.2 ms | 1.7 ms | 3 | no staging |
| short-long | aggressive | 492.9 ms | 194.2 ms | 69.9 ms | 3.0 ms | 2 | 2 potential, 57 unknown |
| long-short | baseline | 907.7 ms | 693.0 ms | 692.9 ms | 1.6 ms | 5 | general sample |
| long-short | reactive | 931.1 ms | 693.8 ms | 693.7 ms | 5.7 ms | 5 | targeted sample |
| long-short | predictive | 922.5 ms | 689.0 ms | 688.9 ms | 3.0 ms | 5 | 2 potential, 56 unknown |
| long-short | conservative | 933.8 ms | 696.4 ms | 696.3 ms | 1.8 ms | 5 | no staging |
| long-short | aggressive | 926.4 ms | 688.5 ms | 688.4 ms | 3.2 ms | 4 | 2 potential, 57 unknown |
| alternate | baseline | 363.5 ms | 128.2 ms | 84.7 ms | 1.7 ms | 5 | general sample |
| alternate | reactive | 374.0 ms | 126.2 ms | 81.2 ms | 5.8 ms | 5 | targeted sample |
| alternate | predictive | 378.9 ms | 123.0 ms | 77.0 ms | 2.9 ms | 5 | 2 potential, 56 unknown |
| alternate | conservative | 377.4 ms | 131.6 ms | 86.1 ms | 1.7 ms | 5 | no staging |
| alternate | aggressive | 370.1 ms | 122.0 ms | 77.5 ms | 3.1 ms | 4 | 2 potential, 57 unknown |

Predictive inference and TTFT were lower than baseline on all four workloads,
but end-to-end latency improved only on short-long and worsened on the other
three. Conservative and aggressive were also mixed. There is no repeatable net
benefit. The baseline and predictive initial-residency observations cover
different populations: baseline sampled general inventory pages (64 resident),
while predictive sampled 64 selected pages (0 resident before staging, 64 after
staging). Only one selected resource was fully observed in the 64-page sample
(two in the SmolLM2 profile reported here); the rest were partial or unknown.
Those transitions are `potentially_useful` with access unknown; no
`clearly_useful` claim and no avoided-load attribution are made.

### Warm medians

| Workload | Arm | End-to-end | Inference | TTFT | Observation | Resident before | Redundant selected |
|---|---|---:|---:|---:|---:|---:|---:|
| short-short | baseline | 236.4 ms | 31.6 ms | 31.5 ms | 1.6 ms | 64 general | — |
| short-short | reactive | 243.4 ms | 31.0 ms | 30.9 ms | 6.0 ms | 64 targeted | 2 |
| short-short | predictive | 247.9 ms | 31.7 ms | 31.7 ms | 3.0 ms | 64 targeted | 2 |
| short-short | conservative | 256.3 ms | 32.3 ms | 32.3 ms | 1.7 ms | 64 general | 0 |
| short-short | aggressive | 250.1 ms | 31.5 ms | 31.5 ms | 3.1 ms | 64 targeted | 2 |
| short-long | baseline | 402.5 ms | 136.6 ms | 31.6 ms | 1.6 ms | 64 general | — |
| short-long | predictive | 410.4 ms | 133.9 ms | 31.6 ms | 2.9 ms | 64 targeted | 2 |
| long-short | baseline | 857.9 ms | 648.3 ms | 648.2 ms | 1.6 ms | 64 general | — |
| long-short | predictive | 896.3 ms | 666.4 ms | 666.4 ms | 2.9 ms | 64 targeted | 2 |
| alternate | baseline | 333.2 ms | 93.2 ms | 48.3 ms | 1.6 ms | 64 general | — |
| alternate | predictive | 341.8 ms | 94.1 ms | 48.0 ms | 2.9 ms | 64 targeted | 2 |

In the warm condition, selected pages were already resident; the fully observed
selected resources were redundant, while partial/unobserved resources remain
unknown. Predictive end-to-end was worse than baseline on all four workloads
(by 6.9–38.4 ms), because prediction/observation/staging overhead was added with
no residency work to do. Mapping-RSS delta was 0 B and major-fault delta was 0
in every warm arm.

Interpretation:

- 100% prediction match is not 100% useful prefetch; all model weights are
  observed in a run, and per-resource residency/fault benefit is unknown.
- Under the 256 MiB cgroup, medians show lower TTFT and fewer major faults for
  three workloads, but the long-short workload was neutral/slightly worse.
  This is an association under uncontrolled cache/order variance, not a proven
  causal speedup.
- No-pressure runs were neutral or slightly slower with prediction. No RSS or
  mapping reduction was observed.
- In the controlled SmolLM2 matrix, predictive inference/TTFT improved on all
  four workloads, but end-to-end latency improved on one and worsened on three.
  Conservative and aggressive were also mixed. The Qwen controlled matrix was
  faster on three of four workloads and slower on short-long, with wide
  variability. No statistically supported general speedup is claimed.
- Baseline and predictive observation samples cover different populations.
  Never compare their resident-page counts as if they were the same sample.
- The isolated-copy method controls file-page-cache state at process start; it
  does not prove model-mapping nonresidency after loading or physical
  reclamation. Before-access residency and per-resource avoided-load causality
  remain unknown.
- The results do not justify claiming a general performance improvement.

## Caveats

- These are wall-clock observations on one machine, not a throughput guarantee.
- The native gate proves no in-flight asynchronous work at release time. It
  does not prove future-token reuse safety.
- Profile files are bounded and validated; a loaded profile is a heuristic,
  not a promise that the next prompt accesses the same tensors.
- cgroup current was dominated by other activity in the shared cgroup and did
  not move. That does not contradict the mapping/process RSS drops; the
  categories are different.
- Native `madvise` acceptance is not proof of physical reclamation.
- The benchmark does not implement intra-decode streaming or eviction.
- GPU execution/residency is not validated.
- Do not report a speedup from one run; use `REPEATS` and inspect variability.
- Keep large models opt-in; the harness never downloads anything.
