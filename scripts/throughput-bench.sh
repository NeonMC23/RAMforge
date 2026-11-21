#!/usr/bin/env sh
# Release-build generation throughput matrix.
#
# Warm, uncontrolled, single-machine. Not a cold-cache or large-model claim.
# Each configuration writes one complete JSON report; the CLI emits
# pretty-printed JSON and native logs on stderr, so reports are not concatenated.
#
# Example:
#   FORGE_TEST_MODEL=/models/tiny.gguf sh scripts/throughput-bench.sh
set -eu

MODEL=${FORGE_TEST_MODEL:-${RAMFORGE_MODEL:-}}
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
  echo "error: set FORGE_TEST_MODEL to a small GGUF model" >&2
  exit 2
fi

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BIN=${RAMFORGE_BIN:-$ROOT/target/release/ramforge}
if [ ! -x "$BIN" ]; then
  echo "error: release binary not found: $BIN" >&2
  echo "run: cargo build --release -p ramforge-cli" >&2
  exit 2
fi

OUT=${RAMFORGE_BENCH_OUT:-bench-out}
RUNS=${RUNS:-3}
MAX_TOKENS=${MAX_TOKENS:-32}
CONTEXT=${CONTEXT:-256}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULT_DIR="$OUT/throughput-$STAMP"
mkdir -p "$RESULT_DIR"

run() {
  name=$1
  profile=$2
  threads=$3
  shift 3
  echo "measuring $name (profile=$profile threads=${threads:-profile})" >&2
  "$BIN" benchmark "$MODEL" --profile "$profile" --ram 1G \
    --runs "$RUNS" --warmup 1 --max-tokens "$MAX_TOKENS" --context "$CONTEXT" \
    --threads "$threads" "$@" --json > "$RESULT_DIR/$name.json"
}

run balanced-4 balanced 4 --no-repack
run balanced-8 balanced 8 --no-repack
run maximum-performance-4 maximum-performance 4
run background background 0 --no-repack

echo "wrote $RESULT_DIR"
