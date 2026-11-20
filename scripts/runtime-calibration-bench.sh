#!/usr/bin/env sh
# Bounded lifecycle-snapshot and calibration benchmark.
#
# Warm, local, synthetic required; not a multi-machine or physical-memory claim.
# Example:
#   CALIBRATION_BENCH_RUNS=10 sh scripts/runtime-calibration-bench.sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
NATIVE=${FORGE_LLAMA_DIR:-$ROOT/../native}
OUT=${RAMFORGE_BENCH_OUT:-bench-out}
RUNS=${CALIBRATION_BENCH_RUNS:-10}

mkdir -p "$OUT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULT="$OUT/runtime-calibration-$STAMP.jsonl"

FORGE_LLAMA_DIR="$NATIVE" \
LD_LIBRARY_PATH="$NATIVE/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
CALIBRATION_BENCH_RUNS="$RUNS" \
cargo test --manifest-path "$ROOT/Cargo.toml" -p ramforge-runtime \
  --test resource_profile_bench lifecycle_calibration_benchmark \
  -- --ignored --nocapture 2>&1 | grep '^{' > "$RESULT"

if [ ! -s "$RESULT" ]; then
  echo "error: benchmark produced no JSON; check the native build" >&2
  exit 1
fi
echo "wrote $RESULT"
