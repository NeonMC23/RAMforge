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
| off | off | 0 | 0 B | 163.73 / 163.73 MiB | ~0 B | 24.7 ms |
| observe | off | 12 | 0 B | 163.64 / 163.64 MiB | ~0 B | 24.8 ms |
| conservative | off | 0 | 0 B | 163.61 / 163.61 MiB | ~0 B | 24.6 ms |
| memory-bound | off | 12 | -38,027,264 B | 163.76 / 127.50 MiB | ~0 B | 25.1 ms |
| active | off | 12 | -38,027,264 B | 163.78 / 127.52 MiB | ~0 B | 24.7 ms |
| reuse-aware (valid profile) | off | 133 | -46,702,592 B | 171.80 / 127.26 MiB | ~0 B | 26.6 ms |

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

## Caveats

- These are wall-clock observations on one machine, not a throughput guarantee.
- cgroup current was dominated by other activity in the shared cgroup and did
  not move. That does not contradict the mapping/process RSS drops; the
  categories are different.
- Native `madvise` acceptance is not proof of physical reclamation.
- The benchmark does not implement intra-decode streaming or eviction.
- GPU execution/residency is not validated.
- Do not report a speedup from one run; use `REPEATS` and inspect variability.
- Keep large models opt-in; the harness never downloads anything.
