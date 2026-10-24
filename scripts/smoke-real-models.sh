#!/usr/bin/env bash
# Run real GGUF smoke/stress cases sequentially and capture reproducible evidence.
#
# Model paths may be a single GGUF file, a split GGUF shard, or a model
# directory containing one model's shard set. Model files are read-only
# inputs: this script never writes, converts, renames, or deletes them. Each
# case is a separate process and the script waits for it to exit before
# checking memory and starting the next one.
#
# Default model layout:
#   ../models/SmolLM2-135M-Instruct-Q4_K_M/
#   ../models/qwen2.5-1.5b-instruct-q4_0/
#   ../models/qwen2.5-7b-instruct-q4_k_m/   (opt-in with --include-7b)
#
# Usage:
#   scripts/smoke-real-models.sh
#   scripts/smoke-real-models.sh --no-build
#   scripts/smoke-real-models.sh --matrix --include-7b --timeout 900
#   RAMFORGE_GPTNEOX_MODEL=/path/to/pythia.gguf \
#     scripts/smoke-real-models.sh --include-gptneox
#   RAMFORGE_BIN=/path/to/ramforge scripts/smoke-real-models.sh --no-build
#
# The legacy budget matrix uses 8G, 4G, 1G, and 100M by default. ForgeCore
# cases remain in the matrix only at 4G because that backend currently reports
# its requested budget as diagnostics-only.
#
# Outputs are written outside the repository (default:
# ${TMPDIR:-/tmp}/ramforge-real-model-smoke/<timestamp>/). Set
# RAMFORGE_SMOKE_OUT_DIR to change the parent directory. A non-zero case exit
# is treated as unexpected in smoke mode. In matrix mode, a controlled budget
# rejection is recorded without failing the whole suite; timeouts and signals
# still fail the suite.
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
default_ram=${RAMFORGE_SMOKE_RAM:-4G}
matrix_budgets=${RAMFORGE_SMOKE_BUDGETS:-"8G 4G 1G 100M"}
case_timeout=${RAMFORGE_SMOKE_TIMEOUT_SECS:-900}
build=1
matrix=0
include_7b=0
include_gptneox=0

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --no-build)
            build=0
            shift
            ;;
        --matrix)
            matrix=1
            shift
            ;;
        --include-7b)
            include_7b=1
            shift
            ;;
        --include-gptneox)
            include_gptneox=1
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
            default_ram=$2
            shift 2
            ;;
        --budgets)
            matrix_budgets=$2
            shift 2
            ;;
        --timeout)
            case_timeout=$2
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

resolve_model() {
    local base=$1
    if [ -d "$model_dir/$base" ]; then
        printf '%s\n' "$model_dir/$base"
    elif [ -f "$model_dir/$base.gguf" ]; then
        printf '%s\n' "$model_dir/$base.gguf"
    else
        return 1
    fi
}

smol_model=$(resolve_model "SmolLM2-135M-Instruct-Q4_K_M") || {
    echo "SmolLM2 model not found under $model_dir" >&2
    exit 2
}
qwen15_model=$(resolve_model "qwen2.5-1.5b-instruct-q4_0") || {
    echo "Qwen2.5-1.5B model not found under $model_dir" >&2
    exit 2
}
qwen7b_model=
if [ "$include_7b" -eq 1 ]; then
    qwen7b_model=$(resolve_model "qwen2.5-7b-instruct-q4_k_m") || {
        echo "Qwen2.5-7B model not found under $model_dir" >&2
        exit 2
    }
fi
gptneox_model=
if [ "$include_gptneox" -eq 1 ]; then
    if [ -n "${RAMFORGE_GPTNEOX_MODEL:-}" ]; then
        gptneox_model=$RAMFORGE_GPTNEOX_MODEL
    else
        gptneox_model=$(resolve_model "pythia-70m-deduped") || {
            echo "GPT-NeoX model not found: set RAMFORGE_GPTNEOX_MODEL or add pythia-70m-deduped under $model_dir" >&2
            exit 2
        }
    fi
fi

for model in "$smol_model" "$qwen15_model" "$qwen7b_model" "$gptneox_model"; do
    [ -n "$model" ] || continue
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
if ! command -v timeout >/dev/null 2>&1; then
    echo "timeout is required for bounded real-model cases" >&2
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
    echo "default_ram=$default_ram"
    echo "matrix_budgets=$matrix_budgets"
    echo "max_tokens=$max_tokens"
    echo "prompt=$prompt"
    echo "case_timeout_secs=$case_timeout"
    echo "min_available_kib=$min_available_kib"
    echo "matrix=$matrix"
    echo "include_7b=$include_7b"
    echo "include_gptneox=$include_gptneox"
    echo "ld_library_path=$LD_LIBRARY_PATH"
    git -C "$repo_dir" rev-parse HEAD 2>/dev/null | sed 's/^/ramforge_rev=/'
    git -C "$workspace_dir/forgeCORE" rev-parse HEAD 2>/dev/null | sed 's/^/forgecore_rev=/'
    git -C "$workspace_dir" rev-parse HEAD 2>/dev/null | sed 's/^/root_rev=/'
} > "$run_dir/environment.txt"

free -h > "$run_dir/memory-before.txt"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader \
    > "$run_dir/gpu-before.txt" 2>/dev/null || true

model_files() {
    local model=$1
    if [ -d "$model" ]; then
        find "$model" -maxdepth 1 -type f -name '*.gguf' -print0 | sort -z
    else
        printf '%s\0' "$model"
    fi
}

inventory_model() {
    local label=$1
    local model=$2
    local file size digest
    while IFS= read -r -d '' file; do
        size=$(stat --printf='%s' "$file")
        digest=$(sha256sum "$file" | cut -d' ' -f1)
        printf '%s\t%s\t%s\t%s\n' "$label" "$file" "$size" "$digest"
    done < <(model_files "$model")
}

{
    printf 'label\tpath\tbytes\tsha256\n'
    inventory_model "smollm2-135m" "$smol_model"
    inventory_model "qwen2.5-1.5b" "$qwen15_model"
    if [ -n "$qwen7b_model" ]; then
        inventory_model "qwen2.5-7b" "$qwen7b_model"
    fi
    if [ -n "$gptneox_model" ]; then
        inventory_model "gptneox" "$gptneox_model"
    fi
} > "$run_dir/model-inventory.tsv"

{
    find "$model_dir" -mindepth 1 -maxdepth 2 -name '*.gguf' -printf '%p\t%s\n' 2>/dev/null || true
} | sort > "$run_dir/model-directories.tsv"

check_memory() {
    local available
    available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    if [ -z "$available" ] || [ "$available" -lt "$min_available_kib" ]; then
        echo "refusing to start: MemAvailable=${available:-unknown} KiB < ${min_available_kib} KiB" >&2
        return 1
    fi
    echo "MemAvailable=${available} KiB (threshold=${min_available_kib} KiB)"
}

memory_snapshot() {
    awk '
        /^MemAvailable:/ { available=$2 }
        /^SwapTotal:/ { swap_total=$2 }
        /^SwapFree:/ { swap_free=$2 }
        END {
            printf "mem_available_kib=%d\nswap_total_kib=%d\nswap_used_kib=%d\n", available, swap_total, swap_total-swap_free
        }
    ' /proc/meminfo
}

monitor_memory() {
    local process=$1
    local output=$2
    printf 'timestamp\tmem_available_kib\tswap_used_kib\n' > "$output"
    while kill -0 "$process" 2>/dev/null; do
        awk '
            /^MemAvailable:/ { available=$2 }
            /^SwapTotal:/ { swap_total=$2 }
            /^SwapFree:/ { swap_free=$2 }
            END {
                printf "%s\t%d\t%d\n", strftime("%Y-%m-%dT%H:%M:%S"), available, swap_total-swap_free
            }
        ' /proc/meminfo >> "$output"
        sleep 1
    done
}

extract() {
    local file=$1
    local pattern=$2
    sed -n "$pattern" "$file" 2>/dev/null | tail -1 || true
}

classify_exit() {
    local status=$1
    local stderr_file=$2
    if [ "$status" -eq 0 ]; then
        printf 'success\n'
    elif [ "$status" -eq 124 ]; then
        printf 'timeout\n'
    elif [ "$status" -ge 128 ] && [ "$status" -le 159 ]; then
        printf 'signal\n'
    elif grep -qiE 'unsupported execution architecture|not supported by RAMforge.s legacy executor|not implemented by RAMforge.s legacy executor' "$stderr_file"; then
        printf 'unsupported-architecture\n'
    elif grep -qiE 'insufficient memory|too small|does not fit|exceeds|out of memory|allocation .* failed|ram budget' "$stderr_file"; then
        printf 'controlled-budget-failure\n'
    else
        printf 'unexpected-failure\n'
    fi
}

run_case() {
    local name=$1
    local model=$2
    local backend=$3
    local budget=$4
    local expected=${5:-success}
    local case_dir="$run_dir/$name"
    local status

    mkdir -p "$case_dir"
    check_memory > "$case_dir/memory-check.txt"

    free -h > "$case_dir/memory-before.txt"
    memory_snapshot > "$case_dir/memory-before-kv.txt"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader \
        > "$case_dir/gpu-before.txt" 2>/dev/null || true

    {
        printf 'model=%s\n' "$model"
        printf 'backend=%s\n' "$backend"
        printf 'budget=%s\n' "$budget"
        printf 'max_tokens=%s\n' "$max_tokens"
        printf 'prompt=%s\n' "$prompt"
        printf 'timeout_secs=%s\n' "$case_timeout"
        printf 'expected_outcome=%s\n' "$expected"
    } > "$case_dir/case.txt"
    {
        printf 'env LD_LIBRARY_PATH=%q timeout %q %q run %q --ram %q --prompt %q --backend %q --max-tokens %q --temperature 0 --memory-report --verbose\n' \
            "$LD_LIBRARY_PATH" "$case_timeout" "$bin" "$model" "$budget" "$prompt" "$backend" "$max_tokens"
    } > "$case_dir/command.txt"
    inventory_model "$(basename "$name")" "$model" > "$case_dir/model-files.tsv"

    echo "running $name" >&2
    set +e
    /usr/bin/time -v -o "$case_dir/time.txt" \
        timeout --signal=TERM --kill-after=30s "$case_timeout" \
        env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        "$bin" run "$model" \
        --ram "$budget" \
        --prompt "$prompt" \
        --backend "$backend" \
        --max-tokens "$max_tokens" \
        --temperature 0 \
        --memory-report \
        --verbose \
        > "$case_dir/stdout.txt" 2> "$case_dir/stderr.txt" &
    run_pid=$!
    monitor_memory "$run_pid" "$case_dir/memory-monitor.tsv" &
    monitor_pid=$!
    wait "$run_pid"
    status=$?
    set -e
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    echo "$status" > "$case_dir/status.txt"
    classify_exit "$status" "$case_dir/stderr.txt" > "$case_dir/outcome.txt"

    free -h > "$case_dir/memory-after.txt"
    memory_snapshot > "$case_dir/memory-after-kv.txt"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader \
        > "$case_dir/gpu-after.txt" 2>/dev/null || true

    {
        grep -E '^(Model config|Tokenizer|Architecture:|Execution backend|Prompt tokens|Generated [0-9]+ tokens|Stop reason|State reset|Budget after run):' \
            "$case_dir/stderr.txt" || true
        grep -E '^  (Peak resident layer bytes|Peak managed bytes|Layer loads|Layer releases|Cache placements|Current cached layers|Peak cached layers|Cache hits):' \
            "$case_dir/stderr.txt" || true
        grep -E '^  (Requested RAM budget|Logical GGUF tensor bytes|Native model bytes|Process RSS|Measured RSS delta|System memory|Device memory|Not exposed|Note):' \
            "$case_dir/stderr.txt" || true
        grep -E 'CPU compute buffer size' "$case_dir/stderr.txt" || true
        grep -E 'Maximum resident set size|Elapsed \(wall clock\) time|^[[:space:]]*Swaps:|^[[:space:]]*Exit status:' \
            "$case_dir/time.txt" || true
        echo "captured_exit_status=$status"
        echo "expected_outcome=$expected"
        echo "outcome=$(cat "$case_dir/outcome.txt")"
    } > "$case_dir/summary.txt"

    local outcome
    outcome=$(cat "$case_dir/outcome.txt")
    if [ "$outcome" = "$expected" ]; then
        echo "ok $name" >&2
    elif [ "$matrix" -eq 1 ] && [ "$outcome" = "controlled-budget-failure" ] && [ "$expected" = "success" ]; then
        echo "controlled failure $name (budget $budget); see $case_dir/stderr.txt" >&2
    else
        echo "FAILED $name (exit $status, outcome $outcome, expected $expected); see $case_dir/stderr.txt" >&2
        return 1
    fi
}

specs=()
if [ "$matrix" -eq 1 ]; then
    for budget in $matrix_budgets; do
        specs+=("smollm2-legacy-$budget|$smol_model|legacy|$budget|success")
        specs+=("qwen2.5-1.5b-legacy-$budget|$qwen15_model|legacy|$budget|success")
    done
    specs+=("smollm2-forgecore-$default_ram|$smol_model|forgecore|$default_ram|success")
    specs+=("qwen2.5-1.5b-forgecore-$default_ram|$qwen15_model|forgecore|$default_ram|success")
    if [ -n "$qwen7b_model" ]; then
        specs+=("qwen2.5-7b-legacy-4G|$qwen7b_model|legacy|4G|success")
    fi
else
    specs+=("smollm2-legacy|$smol_model|legacy|$default_ram|success")
    specs+=("smollm2-forgecore|$smol_model|forgecore|$default_ram|success")
    specs+=("qwen2.5-1.5b-legacy|$qwen15_model|legacy|$default_ram|success")
    specs+=("qwen2.5-1.5b-forgecore|$qwen15_model|forgecore|$default_ram|success")
    if [ -n "$qwen7b_model" ]; then
        specs+=("qwen2.5-7b-legacy|$qwen7b_model|legacy|$default_ram|success")
    fi
fi
if [ -n "$gptneox_model" ]; then
    # GPT-NeoX is implemented by llama.cpp, not by RAMforge's legacy
    # executor. Both behaviours are asserted: forgeCORE generation succeeds,
    # and the legacy preflight rejects the architecture without loading
    # weights or silently switching backends. On the forgeCORE backend the
    # requested RAM budget is diagnostic-only, so every budget is exercised
    # to make that distinction explicit rather than implied.
    if [ "$matrix" -eq 1 ]; then
        for budget in $matrix_budgets; do
            specs+=("gptneox-forgecore-$budget|$gptneox_model|forgecore|$budget|success")
        done
    else
        specs+=("gptneox-forgecore-1G|$gptneox_model|forgecore|1G|success")
    fi
    specs+=("gptneox-legacy-rejected|$gptneox_model|legacy|1G|unsupported-architecture")
fi

failures=0
for spec in "${specs[@]}"; do
    IFS='|' read -r name model backend budget expected <<EOF
$spec
EOF
    if ! run_case "$name" "$model" "$backend" "$budget" "$expected"; then
        failures=$((failures + 1))
    fi
done

{
    printf 'case\tbackend\tbudget\texit\toutcome\texpected\tmax_rss_kib\tmanaged_peak_bytes\tpeak_cached_layers\tcache_evictions\tgenerated\tstop_reason\n'
    for case_dir in "$run_dir"/*/; do
        [ -d "$case_dir" ] || continue
        name=$(basename "$case_dir")
        backend=$(sed -n 's/^backend=//p' "$case_dir/case.txt" 2>/dev/null | head -1)
        budget=$(sed -n 's/^budget=//p' "$case_dir/case.txt" 2>/dev/null | head -1)
        status=$(cat "$case_dir/status.txt" 2>/dev/null || echo missing)
        outcome=$(cat "$case_dir/outcome.txt" 2>/dev/null || echo missing)
        expected=$(sed -n 's/^expected_outcome=//p' "$case_dir/case.txt" 2>/dev/null | head -1)
        rss=$(awk -F': ' '/Maximum resident set size/ {gsub(/,/, "", $2); print $2}' "$case_dir/time.txt" 2>/dev/null | tail -1 || true)
        managed=$(sed -n 's/^  Peak managed bytes: \([0-9][0-9]*\).*/\1/p' "$case_dir/stderr.txt" 2>/dev/null | tail -1)
        peak_cached=$(sed -n 's/^  Peak cached layers: \([0-9][0-9]*\).*/\1/p' "$case_dir/stderr.txt" 2>/dev/null | tail -1)
        evictions=$(sed -n 's/^  Cache hits: .* evictions: \([0-9][0-9]*\).*/\1/p' "$case_dir/stderr.txt" 2>/dev/null | tail -1)
        generated=$(sed -n 's/^Generated \([0-9][0-9]*\) tokens.*/\1/p' "$case_dir/stderr.txt" 2>/dev/null | tail -1)
        stop=$(sed -n 's/^Stop reason: //p' "$case_dir/stderr.txt" 2>/dev/null | tail -1)
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$name" "${backend:-n/a}" "${budget:-n/a}" "$status" "$outcome" "${expected:-n/a}" \
            "${rss:-n/a}" "${managed:-n/a}" "${peak_cached:-n/a}" "${evictions:-n/a}" \
            "${generated:-n/a}" "${stop:-n/a}"
    done
} > "$run_dir/results.tsv"

if command -v python3 >/dev/null 2>&1; then
    python3 - "$run_dir" <<'PY'
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
cases = []
for case_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
    def read(name):
        path = case_dir / name
        return path.read_text(errors="replace").strip() if path.exists() else ""

    def extract(pattern):
        match = re.search(pattern, read("stderr.txt"), re.MULTILINE)
        return match.group(1).strip() if match else None

    def read_kv(name):
        values = {}
        for line in read(name).splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = int(value) if value.isdigit() else value
        return values

    elapsed = None
    elapsed_seconds = None
    time_text = read("time.txt")
    match = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)", time_text
    )
    if match:
        elapsed = match.group(1).strip()
        parts = elapsed.split(":")
        try:
            elapsed_seconds = sum(float(part) * (60**index) for index, part in enumerate(reversed(parts)))
        except ValueError:
            elapsed_seconds = None
    rss = None
    match = re.search(r"Maximum resident set size \(kbytes\):\s*([0-9]+)", time_text)
    if match:
        rss = int(match.group(1))
    swaps = None
    match = re.search(r"^\s*Swaps:\s*([0-9]+)", time_text, re.MULTILINE)
    if match:
        swaps = int(match.group(1))

    monitor = case_dir / "memory-monitor.tsv"
    min_available = max_swap = None
    if monitor.exists():
        rows = [line.split("\t") for line in monitor.read_text().splitlines()[1:] if line]
        if rows:
            min_available = min(int(row[1]) for row in rows)
            max_swap = max(int(row[2]) for row in rows)

    memory_before = read_kv("memory-before-kv.txt")
    memory_after = read_kv("memory-after-kv.txt")
    swap_delta = None
    if "swap_used_kib" in memory_before and isinstance(max_swap, int):
        swap_delta = max_swap - memory_before["swap_used_kib"]

    case = {
        "name": case_dir.name,
        "status": int(read("status.txt")) if read("status.txt").isdigit() else None,
        "outcome": read("outcome.txt") or None,
        "expected_outcome": next(
            (
                line.split("=", 1)[1]
                for line in read("case.txt").splitlines()
                if line.startswith("expected_outcome=")
            ),
            None,
        ),
        "model": read("case.txt").splitlines()[0].removeprefix("model=") if read("case.txt") else None,
        "backend": read("case.txt").splitlines()[1].removeprefix("backend=") if read("case.txt") else None,
        "budget": read("case.txt").splitlines()[2].removeprefix("budget=") if read("case.txt") else None,
        "elapsed": elapsed,
        "elapsed_seconds": elapsed_seconds,
        "max_rss_kib": rss,
        "swaps": swaps,
        "monitor_min_mem_available_kib": min_available,
        "monitor_max_system_swap_used_kib": max_swap,
        "monitor_system_swap_used_delta_kib": swap_delta,
        "system_memory_before": memory_before,
        "system_memory_after": memory_after,
        "managed_peak_bytes": int(v) if (v := extract(r"^  Peak managed bytes: ([0-9]+)")) else None,
        "peak_cached_layers": int(v) if (v := extract(r"^  Peak cached layers: ([0-9]+)")) else None,
        "cache_evictions": int(v) if (v := extract(r"^  Cache hits: .* evictions: ([0-9]+)")) else None,
        "generated_tokens": int(v) if (v := extract(r"^Generated ([0-9]+) tokens")) else None,
        "stop_reason": extract(r"^Stop reason: (.+)"),
        "output": read("stdout.txt"),
    }
    cases.append(case)

report = {
    "environment": run_dir.joinpath("environment.txt").read_text(errors="replace").splitlines(),
    "model_inventory": run_dir.joinpath("model-inventory.tsv").read_text(errors="replace").splitlines(),
    "cases": cases,
}
(run_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
PY
fi

free -h > "$run_dir/memory-after.txt"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader \
    > "$run_dir/gpu-after.txt" 2>/dev/null || true

echo
echo "evidence directory: $run_dir"
cat "$run_dir/results.tsv"

if [ "$failures" -ne 0 ]; then
    echo "$failures real-model case(s) failed unexpectedly" >&2
    exit 1
fi
