#!/usr/bin/env sh
# Reproducible residency benchmark/diagnostic harness.
#
# This is opt-in: it requires a real small GGUF model and a working native
# build. It never downloads a model and never claims a speedup from one sample.
#
# Example:
#   FORGE_TEST_MODEL=/models/tiny.gguf TARGET=192M REPEATS=3 \
#     sh scripts/residency-bench.sh
#
# Results are written to RAMFORGE_BENCH_OUT (default: ./bench-out) as
# results.tsv plus one bounded JSON report and stderr log per run.

set -u

MODEL=${FORGE_TEST_MODEL:-${RAMFORGE_MODEL:-}}
if [ -z "$MODEL" ]; then
  echo "error: set FORGE_TEST_MODEL or RAMFORGE_MODEL to a small GGUF model" >&2
  exit 2
fi
if [ ! -f "$MODEL" ]; then
  echo "error: model not found: $MODEL" >&2
  exit 2
fi

BIN=${RAMFORGE_BIN:-target/release/ramforge}
if [ ! -x "$BIN" ]; then
  echo "error: ramforge binary not found: $BIN (build with: cargo build --release -p ramforge-cli)" >&2
  exit 2
fi

TARGET=${TARGET:-192M}
RESERVE=${RESERVE:-}
CTX=${CTX:-256}
MAX_TOKENS=${MAX_TOKENS:-4}
PROMPT=${PROMPT:-The capital of France is}
PREFETCH=${PREFETCH:-64M}
REPEATS=${REPEATS:-1}
ENFORCE=${ENFORCE:-0}
ENFORCE_TIMEOUT=${ENFORCE_TIMEOUT:-120}
GRAPH_TRACE_EVENTS=${GRAPH_TRACE_EVENTS:-16384}
OUT=${RAMFORGE_BENCH_OUT:-bench-out}
BACKEND=${BACKEND:-forgecore}
POLICIES=${POLICIES:-"off observe conservative memory-bound reuse-aware active"}
PREFETCH_MODES=${PREFETCH_MODES:-"off on"}
mkdir -p "$OUT"
RESULTS="$OUT/results.tsv"

if [ ! -f "$RESULTS" ]; then
  printf 'timestamp\tpolicy\tprefetch\trepeat\tstatus\twall_ns\tmodel\tgeneration\tselected\tprofile_used\testimated_reclaim_bytes\tobserved_mapping_delta_bytes\trss_before\trss_after\tcgroup_before\tcgroup_after\tstaging_requested_bytes\tstaging_native_requested_bytes\tnative_released_bytes\ttrace_complete\tttft_ns\tdecode_ns\ttotal_ns\toutput_fingerprint\tstderr_sha256\n' > "$RESULTS"
fi

for policy in $POLICIES; do
  for prefetch in $PREFETCH_MODES; do
    repeat=1
    while [ "$repeat" -le "$REPEATS" ]; do
      stamp=$(date -u +%Y%m%dT%H%M%SZ)
      base="$OUT/${stamp}-${policy}-${prefetch}-r${repeat}"
      json="$base.json"
      log="$base.stderr"
      set -- run "$MODEL" --ram "$TARGET" --ctx "$CTX" --no-repack \
        --residency-policy "$policy" --residency-target "$TARGET" \
        --residency-report-json "$json" \
        --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" --temperature 0 \
        --backend "$BACKEND"
      if [ -n "$RESERVE" ] && [ "$RESERVE" != "0" ]; then
        set -- "$@" --residency-reserve "$RESERVE"
      fi
      if [ "$prefetch" = "on" ]; then
        set -- "$@" --residency-prefetch "$PREFETCH"
      else
        set -- "$@" --no-prefetch
      fi
      if [ "$policy" = "reuse-aware" ]; then
        set -- "$@" --graph-trace --graph-trace-events "$GRAPH_TRACE_EVENTS"
      fi
      if [ "$ENFORCE" = "1" ]; then
        set -- "$@" --enforce-ram --enforce-timeout "$ENFORCE_TIMEOUT"
      fi
      start=$(date +%s%N)
      "$BIN" "$@" >/dev/null 2>"$log"
      status=$?
      end=$(date +%s%N)
      wall=$((end - start))
      sha=$(sha256sum "$log" 2>/dev/null | awk '{print $1}')
      if [ "$status" -eq 0 ] && [ -f "$json" ]; then
        python3 - "$json" "$status" "$wall" "$sha" "$stamp" "$policy" "$prefetch" "$repeat" <<'PYROW' >> "$RESULTS"
import json, sys
path, status, wall, sha, stamp, policy, prefetch, repeat = sys.argv[1:]
report = json.load(open(path, encoding="utf-8"))
plan = report.get("post_plan") or report.get("pre_plan") or {}
staging = report.get("staging_result") or {}
app = report.get("application") or {}
profile = report.get("profile") or {}
timing = report.get("timing") or {}
before = (report.get("measurements") or {}).get("before") or {}
after = (report.get("measurements") or {}).get("after") or {}
row = [
    stamp, policy, prefetch, repeat, status, wall,
    report["model"]["id"], report["model"]["generation"],
    plan.get("selected_count", len(plan.get("selected") or [])), plan.get("profile_used"),
    (report.get("estimated_vs_observed") or {}).get("estimated_reclaim_bytes"),
    (report.get("estimated_vs_observed") or {}).get("observed_mapping_delta_bytes"),
    before.get("process_rss_bytes"), after.get("process_rss_bytes"),
    before.get("cgroup_current_bytes"), after.get("cgroup_current_bytes"),
    (report.get("staging_plan") or {}).get("selected_bytes"),
    staging.get("native_requested_bytes"),
    app.get("native_released_bytes_submitted"),
    profile.get("complete_observations", 0),
    timing.get("ttft_ns"), timing.get("decode_ns"), timing.get("total_ns"),
    timing.get("output_fingerprint"), sha,
]
print("\t".join("" if value is None else str(value) for value in row))
PYROW
      else
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t%s\n' \
          "$stamp" "$policy" "$prefetch" "$repeat" "$status" "$wall" "$(basename "$log")" "$sha" "$sha" >> "$RESULTS"
        echo "failed: $policy prefetch=$prefetch repeat=$repeat (see $log)" >&2
      fi
      repeat=$((repeat + 1))
    done
  done
done

echo "wrote $RESULTS"
