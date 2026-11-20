#!/usr/bin/env sh
# Statistical CPU-node-dispatch sampling benchmark.
#
# Runs the forgeCORE ignored benchmark that compares disabled, systematic,
# targeted, and adaptive sampling across independent context runs. The output
# is one JSON object per configuration with median, min, max, Q1, Q3, page
# accounting, and cache counters.
#
# This is warm-cache, uncontrolled validation only. It does not claim physical
# cold-cache behavior, physical I/O, or a speedup.
#
# Example:
#   FORGE_TEST_MODEL=/models/tiny.gguf BENCH_RUNS=7 BENCH_ITERATIONS=5 \
#     sh scripts/residency-sampling-bench.sh
set -eu

MODEL=${FORGE_TEST_MODEL:-${RAMFORGE_MODEL:-}}
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
  echo "error: set FORGE_TEST_MODEL to a small GGUF model" >&2
  exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
COREDIR=${FORGECORE_DIR:-$ROOT/forgeCORE}
NATIVE=${FORGE_LLAMA_DIR:-$ROOT/native}
OUT=${RAMFORGE_BENCH_OUT:-bench-out}
BENCH_RUNS=${BENCH_RUNS:-7}
BENCH_ITERATIONS=${BENCH_ITERATIONS:-5}
BENCH_THREADS=${BENCH_THREADS:-1}

if [ ! -f "$COREDIR/Cargo.toml" ]; then
  echo "error: forgeCORE not found at $COREDIR" >&2
  exit 2
fi
mkdir -p "$OUT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULT="$OUT/sampling-bench-$STAMP.jsonl"

FORGE_TEST_MODEL="$MODEL" \
FORGE_LLAMA_DIR="$NATIVE" \
LD_LIBRARY_PATH="$NATIVE/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
FORGE_BENCH_RUNS="$BENCH_RUNS" \
FORGE_BENCH_ITERATIONS="$BENCH_ITERATIONS" \
FORGE_BENCH_THREADS="$BENCH_THREADS" \
cargo test --manifest-path "$COREDIR/Cargo.toml" -p forge-core \
  --test node_dispatch_overhead statistical_sampling_benchmark \
  -- --ignored --nocapture 2>&1 | grep '^{' > "$RESULT"

if [ ! -s "$RESULT" ]; then
  echo "error: benchmark produced no JSON rows; check the build and model" >&2
  exit 1
fi
echo "wrote $RESULT"
