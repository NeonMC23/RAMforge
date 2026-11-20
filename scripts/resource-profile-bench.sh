#!/usr/bin/env sh
# Bounded resource-profile and candidate-plan benchmark.
#
# Warm, local, single-machine observation. Not a multi-machine or physical
# storage-performance claim.
#
# Example:
#   FORGE_TEST_MODEL=/models/tiny.gguf PROFILE_BENCH_RUNS=20 \
#     sh scripts/resource-profile-bench.sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
NATIVE=${FORGE_LLAMA_DIR:-$ROOT/../native}
OUT=${RAMFORGE_BENCH_OUT:-bench-out}
RUNS=${PROFILE_BENCH_RUNS:-20}

mkdir -p "$OUT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULT="$OUT/resource-profile-$STAMP.jsonl"

FORGE_LLAMA_DIR="$NATIVE" \
LD_LIBRARY_PATH="$NATIVE/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
PROFILE_BENCH_RUNS="$RUNS" \
cargo test --manifest-path "$ROOT/Cargo.toml" -p ramforge-runtime \
  --test resource_profile_bench -- --ignored --nocapture 2>&1 \
  | grep '^{' > "$RESULT"

if [ ! -s "$RESULT" ]; then
  echo "error: benchmark produced no JSON; check the native build" >&2
  exit 1
fi
echo "wrote $RESULT"
