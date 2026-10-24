#!/usr/bin/env sh
# Run RAMforge tests with the native llama.cpp/ggml loader path configured.
#
# RAMforge links `forge-core`, which links the prebuilt upstream shared
# libraries. `cargo:rustc-link-arg` does not propagate across crates, so the
# final RAMforge test binaries do not carry an rpath. This script supplies the
# loader path for the test process only; it does not modify system-wide
# configuration.
#
# Resolution order for the native tree:
#   1. FORGE_LLAMA_DIR, if already set.
#   2. ../native relative to this repository (the current workspace layout).
#   3. /var/tmp/forge-native (the forgeCORE default).
#
# If the standard forgeCORE fixture path exists and FORGE_TEST_MODEL is not
# already set, export it so the real ForgeCore adapter tests run instead of
# reporting SKIP. An explicit caller value always wins.
#
# Usage:
#   scripts/test.sh [cargo test arguments...]
#   FORGE_LLAMA_DIR=/path/to/native scripts/test.sh --workspace
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
workspace_dir=$(CDPATH= cd -- "$repo_dir/.." && pwd)

if [ -z "${FORGE_LLAMA_DIR:-}" ]; then
    if [ -d "$workspace_dir/native/build/bin" ]; then
        FORGE_LLAMA_DIR="$workspace_dir/native"
    else
        FORGE_LLAMA_DIR="/var/tmp/forge-native"
    fi
fi
export FORGE_LLAMA_DIR

lib_dir="$FORGE_LLAMA_DIR/build/bin"
if [ ! -d "$lib_dir" ]; then
    echo "RAMforge test: native library directory not found: $lib_dir" >&2
    echo "Set FORGE_LLAMA_DIR to a llama.cpp/ggml v0.5.0 tree built with shared libraries." >&2
    exit 1
fi

fixture="$FORGE_LLAMA_DIR/models/tiny-llama.gguf"
if [ -z "${FORGE_TEST_MODEL:-}" ] && [ -f "$fixture" ]; then
    FORGE_TEST_MODEL="$fixture"
    export FORGE_TEST_MODEL
fi

LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH

exec cargo test "$@"
