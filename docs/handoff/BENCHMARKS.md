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

Page faults are process-wide `/proc/self/stat` counters sampled before model
load and after decode. They include loader, tokenizer, trace, and harness
activity; the harness reports deltas but cannot attribute them exclusively to
prefetch. Process RSS and cgroup usage are also process/group measurements, not
per-resource proof.

Predictive staging latency was measured separately: median native prefetch call
latency was 3.9–5.4 ms across workloads in a two-repeat constrained smoke run.
TTFT excludes that pre-operation staging time; process wall time includes it.

RSS remained around 168–184 MiB; mapping delta was **0 B** in all arms, so no
physical mapping residency benefit was observed. Prediction metrics for the
warm history were stable across workloads: 64 generated, 64 matched overall,
63 within the 64-resource horizon, precision 100% (prediction accuracy), recall
23.5% of 272 observed inventory resources, 57 native-accepted resources
followed by observed access, 57 with unknown physical benefit, 0 not-followed,
and 7 duplicate/overlap requests avoided. The 100% precision reflects that all
predicted model tensors are touched by a transformer execution; it does not
mean 64 useful prefetches.

Interpretation:

- 100% prediction match is not 100% useful prefetch; all model weights are
  observed in a run, and per-resource residency/fault benefit is unknown.
- Under the 256 MiB cgroup, medians show lower TTFT and fewer major faults for
  three workloads, but the long-short workload was neutral/slightly worse.
  This is an association under uncontrolled cache/order variance, not a proven
  causal speedup.
- No-pressure runs were neutral or slightly slower with prediction. No RSS or
  mapping reduction was observed.
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
