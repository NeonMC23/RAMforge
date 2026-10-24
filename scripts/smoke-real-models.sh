#!/usr/bin/env bash
# Run the real GGUF smoke matrix sequentially and capture reproducible evidence.
#
# The model files are read-only inputs. This script never writes, converts,
# renames, or deletes them. Each run is a separate process and the script waits
# for it to exit before checking memory and starting the next model/backend.
#
# Defaults target the two models in the shared workspace:
#   ../models/SmolLM2-135M-Instruct-Q4_K_M.gguf
#   ../models/qwen2.5-1.5b-instruct-q4_0.gguf
#
# Usage:
#   scripts/smoke-real-models.sh
#   scripts/smoke-real-models.sh --no-build
#   RAMFORGE_BIN=/path/to/ramforge scripts/smoke-real-models.sh --no-build
#
# Outputs are written outside the repository (default:
# ${TMPDIR:-/tmp}/ramforge-real-model-smoke/<timestamp>/). Set
# RAMFORGE_SMOKE_OUT_DIR to change the parent directory.
set -euo pipefail

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
workspace_dir=$(CDPATH= cd -- "$repo_dir/.." && pwd)

native_dir=${FORGE_LLAMA_DIR:-}
if [ -z "$native_dir" ]; then
    if [ -d "$workspace_dir/native/build/bin" ]; then
        native_dir="$workspace_dir/native"
    else
        native_dir="/var/tmp/forge-native"
    fi
fi

model_dir=${RAMFORGE_SMOKE_MODEL_DIR:-"$workspace_dir/models"}
out_parent=${RAMFORGE_SMOKE_OUT_DIR:-"${TMPDIR:-/tmp}/ramforge-real-model-smoke"}
run_dir="$out_parent/$(date +%Y%m%d-%H%M%S)"
min_available_kib=${RAMFORGE_SMOKE_MIN_AVAILABLE_KIB:-4194304}
max_tokens=${RAMFORGE_SMOKE_MAX_TOKENS:-8}
prompt=${RAMFORGE_SMOKE_PROMPT:-Hello}
ram=${RAMFORGE_SMOKE_RAM:-4G}
build=1

usage() {
    sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --no-build)
            build=0
            shift
            ;;
        --model-dir)
            model_dir=$2
            shift 2
            ;;
        --out-dir)
            out_parent=$2
            shift 2
            ;;
        --max-tokens)
            max_tokens=$2
            shift 2
            ;;
        --prompt)
            prompt=$2
            shift 2
            ;;
        --ram)
            ram=$2
            shift 2
            ;;
        --min-available-kib)
            min_available_kib=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

smol_model="$model_dir/SmolLM2-135M-Instruct-Q4_K_M.gguf"
qwen_model="$model_dir/qwen2.5-1.5b-instruct-q4_0.gguf"

for model in "$smol_model" "$qwen_model"; do
    if [ ! -r "$model" ]; then
        echo "model not readable: $model" >&2
        exit 2
    fi
done

lib_dir="$native_dir/build/bin"
if [ ! -d "$lib_dir" ]; then
    echo "native library directory not found: $lib_dir" >&2
    echo "Set FORGE_LLAMA_DIR to a shared-library llama.cpp/ggml build." >&2
    exit 2
fi

export FORGE_LLAMA_DIR="$native_dir"
export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ "$build" -eq 1 ]; then
    cargo build \
        --manifest-path "$repo_dir/Cargo.toml" \
        -p ramforge-cli \
        --locked
fi

bin=${RAMFORGE_BIN:-"${CARGO_TARGET_DIR:-$repo_dir/target}/debug/ramforge"}
if [ ! -x "$bin" ]; then
    echo "RAMforge binary not executable: $bin" >&2
    echo "Build it or set RAMFORGE_BIN explicitly." >&2
    exit 2
fi

if [ ! -x /usr/bin/time ]; then
    echo "/usr/bin/time is required for RSS and wall-time capture" >&2
    exit 2
fi

mkdir -p "$run_dir"

{
    echo "date=$(date -Is)"
    echo "host=$(uname -srm)"
    echo "repo=$repo_dir"
    echo "bin=$bin"
    echo "native=$native_dir"
    echo "model_dir=$model_dir"
    echo "ram=$ram"
    echo "max_tokens=$max_tokens"
    echo "prompt=$prompt"
    echo "min_available_kib=$min_available_kib"
    echo "ld_library_path=$LD_LIBRARY_PATH"
    git -C "$repo_dir" rev-parse HEAD 2>/dev/null | sed 's/^/ramforge_rev=/'
    git -C "$workspace_dir/forgeCORE" rev-parse HEAD 2>/dev/null | sed 's/^/forgecore_rev=/'
    git -C "$workspace_dir" rev-parse HEAD 2>/dev/null | sed 's/^/root_rev=/'
} > "$run_dir/environment.txt"

free -h > "$run_dir/memory-before.txt"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader \
    > "$run_dir/gpu-before.txt" 2>/dev/null || true

for model in "$smol_model" "$qwen_model"; do
    stat --printf='%n\t%s bytes\n' "$model"
done > "$run_dir/models-size.txt"
sha256sum "$smol_model" "$qwen_model" > "$run_dir/models-sha256.txt"

check_memory() {
    local available
    available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    if [ -z "$available" ] || [ "$available" -lt "$min_available_kib" ]; then
        echo "refusing to start: MemAvailable=${available:-unknown} KiB < ${min_available_kib} KiB" >&2
        return 1
    fi
    echo "MemAvailable=${available} KiB (threshold=${min_available_kib} KiB)"
}

run_case() {
    local name=$1
    local model=$2
    local backend=$3
    local case_dir="$run_dir/$name"
    local status

    mkdir -p "$case_dir"
    check_memory > "$case_dir/memory-check.txt"

    free -h > "$case_dir/memory-before.txt"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader \
        > "$case_dir/gpu-before.txt" 2>/dev/null || true

    {
        printf 'env LD_LIBRARY_PATH=%q %q run %q --ram %q --prompt %q --backend %q --max-tokens %q --temperature 0 --memory-report --verbose\n' \
            "$LD_LIBRARY_PATH" "$bin" "$model" "$ram" "$prompt" "$backend" "$max_tokens"
    } > "$case_dir/command.txt"

    echo "running $name" >&2
    set +e
    /usr/bin/time -v -o "$case_dir/time.txt" \
        env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        "$bin" run "$model" \
        --ram "$ram" \
        --prompt "$prompt" \
        --backend "$backend" \
        --max-tokens "$max_tokens" \
        --temperature 0 \
        --memory-report \
        --verbose \
        > "$case_dir/stdout.txt" 2> "$case_dir/stderr.txt"
    status=$?
    set -e
    echo "$status" > "$case_dir/status.txt"

    free -h > "$case_dir/memory-after.txt"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader \
        > "$case_dir/gpu-after.txt" 2>/dev/null || true

    {
        grep -E '^(Model config|Tokenizer|Execution backend|Prompt tokens|Stop reason|Budget after run):' \
            "$case_dir/stderr.txt" || true
        grep -E '^Generated [0-9]+ tokens' "$case_dir/stderr.txt" || true
        grep -E 'Peak managed bytes:|Layer loads:|Layer releases:|Layers retained in cache:' \
            "$case_dir/stderr.txt" || true
        grep -E '^  (Requested RAM budget|Logical GGUF tensor bytes|Native model bytes|Process RSS|Measured RSS delta|System memory|Device memory|Not exposed|Note):' \
            "$case_dir/stderr.txt" || true
        grep -E 'CPU compute buffer size' "$case_dir/stderr.txt" || true
        grep -E 'Maximum resident set size|Elapsed \(wall clock\) time|^[[:space:]]*Swaps:|^[[:space:]]*Exit status:' \
            "$case_dir/time.txt" || true
        echo "captured_exit_status=$status"
    } > "$case_dir/summary.txt"

    if [ "$status" -ne 0 ]; then
        echo "FAILED $name (exit $status); see $case_dir/stderr.txt" >&2
        return "$status"
    fi
    echo "ok $name" >&2
}

failures=0
for spec in \
    "smollm2-legacy|$smol_model|legacy" \
    "smollm2-forgecore|$smol_model|forgecore" \
    "qwen2.5-legacy|$qwen_model|legacy" \
    "qwen2.5-forgecore|$qwen_model|forgecore"
do
    IFS='|' read -r name model backend <<EOF
$spec
EOF
    if ! run_case "$name" "$model" "$backend"; then
        failures=$((failures + 1))
    fi
done

{
    printf 'case\texit\tmax_rss_kib\tmanaged_peak\tgenerated\tstop_reason\n'
    for case_dir in "$run_dir"/*/; do
        name=$(basename "$case_dir")
        status=$(cat "$case_dir/status.txt" 2>/dev/null || echo missing)
        rss=$(awk -F': ' '/Maximum resident set size/ {gsub(/,/, "", $2); print $2}' "$case_dir/time.txt" 2>/dev/null || true)
        managed=$(sed -n 's/^  Peak managed bytes: \([0-9][0-9]*\).*/\1/p' "$case_dir/stderr.txt" 2>/dev/null | tail -1)
        generated=$(sed -n 's/^Generated \([0-9][0-9]*\) tokens.*/\1/p' "$case_dir/stderr.txt" 2>/dev/null | tail -1)
        stop=$(sed -n 's/^Stop reason: //p' "$case_dir/stderr.txt" 2>/dev/null | tail -1)
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$name" "$status" "${rss:-n/a}" "${managed:-n/a}" "${generated:-n/a}" "${stop:-n/a}"
    done
} > "$run_dir/results.tsv"

free -h > "$run_dir/memory-after.txt"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader \
    > "$run_dir/gpu-after.txt" 2>/dev/null || true

echo
echo "evidence directory: $run_dir"
cat "$run_dir/results.tsv"

if [ "$failures" -ne 0 ]; then
    echo "$failures smoke run(s) failed" >&2
    exit 1
fi
