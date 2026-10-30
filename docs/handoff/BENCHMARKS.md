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

SmolLM2-135M Q4_K_M, one token, loaded profile, `PREDICTION=1`:

- 64 medium-confidence predictions;
- 56 selected for speculative prefetch; 8 deferred by page-range deduplication/window;
- requested **16,650,240 B**, native-accepted **16,650,240 B**;
- observed mapping RSS delta **0 B**;
- feedback: 64 hits, 0 misses, 0 wasted staged bytes for that trace.

With a warm feedback history, confidence classified all 64 as high:
**17,375,232 B** requested/accepted, mapping delta still **0 B**. These are
hint/accounting measurements, not evidence of throughput or physical residency
improvement.

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
