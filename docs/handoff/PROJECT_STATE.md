# Project state

**As-of date:** 2026-10-02; local verification refreshed 2026-10-25.<br>
**Repository:** RAMforge workspace at the handoff source checkout.<br>
**Status labels:** see [README.md](README.md).

## Local verification (2026-10-25)

- The workspace builds and tests against the pinned `forge-core` revision
  `cb39080` with Rust/Cargo 1.98.1 and the workspace native tree at
  `/home/neon/Documents/CODE LOCAL/forge/native` (llama.cpp/ggml v0.5.0 plus
  the documented mmap-no-prefetch patch).
- `scripts/test.sh --workspace --locked` sets `FORGE_LLAMA_DIR` and the
  process-local `LD_LIBRARY_PATH`, auto-detects
  `$FORGE_LLAMA_DIR/models/tiny-llama.gguf` for `FORGE_TEST_MODEL`, then runs
  `cargo test`. A direct `cargo test --workspace --locked` without fixture
  environment variables passes **446 tests, 0 failed, 10 ignored**; the ignored
  tests are the real-model Qwen2.5 diagnostic, the quantized microbenchmark,
  the opt-in GPT-NeoX integration test, two real cgroup enforcement tests, and
  five CLI enforcement tests. Fixture-gated tests report SKIP when their
  environment variables are unset.
- The registry now tracks `legacy_run` and `forgecore_run` separately.
  `gptneox`/`gpt_neox` is `legacy_run=not-yet`, `forgecore_run=yes`; the legacy
  preflight error names `--backend forgecore` and never switches implicitly.
- `inspect`, `plan`, `support`, and JSON output distinguish legacy from
  forgeCORE execution. `execution_supported` remains the legacy verdict for
  JSON compatibility; `execution_supported_scope` states that explicitly, and
  `forgecore_execution_supported` is reported separately.
- The tokenizer now matches llama.cpp defaults when
  `tokenizer.ggml.add_bos_token` is absent: BPE/GPT-2-style vocabularies add no
  BOS; SentencePiece vocabularies add BOS.
- The opt-in GPT-NeoX tests are gated by `FORGE_GPTNEOX_MODEL` in both
  forgeCORE (`tests/gptneox_execution.rs`) and RAMforge
  (`tests/forgecore_adapter.rs`). They load a real fixture, tokenize, decode,
  generate greedily twice, and assert deterministic ids and finite logits.
- The ForgeCore adapter integration suite checks independent native context
  state across drops, owned logits surviving session drop, and model reuse
  after context drops.
- A fixture-backed memory test exercises logical GGUF bytes, native
  `size_bytes()`, measured RSS, and explicit budget non-enforcement against the
  real native model.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: passed.
- `cargo fmt --all -- --check`: **not clean**; the repository has pre-existing
  formatting differences in several files. They are not mass-reformatted here.
- The ForgeCore CLI `--memory-report` path reports separate logical GGUF,
  native `size_bytes()`, measured process RSS/peak RSS, model-mapping RSS/PSS,
  device facts, and (when active) cgroup limit/peak/events/page faults. Without
  `--enforce-ram` it labels the requested budget diagnostic-only; with the flag
  it labels the cgroup v2 `memory.max` enforcement.
- The `forge-core` pin moved to `cb39080` to consume
  `ModelOptions::use_extra_bufts`, `ModelLoadMode::MmapNoPrefetch`,
  `Model::tensor_mappings()`, and `Model::release_tensor_pages()`. The native
  llama.cpp upstream HEAD remains v0.5.0; the capabilities are supplied by the
  documented `patches/llama-model-residency.patch`, applied by
  `setup-native.sh`, not a native upgrade. The adapter uses the options to
  implement `--no-repack`, `--no-prefetch`, model-wide release, and
  layer/range release.
- See [../../scripts/test.sh](../../scripts/test.sh) for the reproducible test
  invocation.

### Real-model smoke evidence (2026-10-25)

`scripts/smoke-real-models.sh --no-build --matrix --include-gptneox
--max-tokens 8` ran each case in a separate process, sequentially, with a 4 GiB
`MemAvailable` start threshold. The GPT-NeoX fixture was supplied through
`RAMFORGE_GPTNEOX_MODEL`. Evidence:
`/tmp/ramforge-real-model-smoke/20261025-002055`; TSV, JSON, per-case commands,
stdout/stderr, hashes, and system-memory samples are retained there.

Exact budgets: `8G` = 8,589,934,592 B, `4G` = 4,294,967,296 B, `1G` =
1,073,741,824 B, `100M` = 104,857,600 B. Prompt `Hello`, 8 greedy tokens,
backend fixed within each comparison.

| Model / case | Backend | Budget | Exit | Wall | Max RSS | Managed peak | Evictions |
|---|---|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M Q4_K_M | legacy | 8G | 0 | 1.22 s | 135,768 KiB | 104,007,876 B | 0 |
| SmolLM2-135M Q4_K_M | legacy | 4G | 0 | 1.16 s | 135,572 KiB | 104,007,876 B | 0 |
| SmolLM2-135M Q4_K_M | legacy | 1G | 0 | 1.26 s | 135,728 KiB | 104,007,876 B | 0 |
| SmolLM2-135M Q4_K_M | legacy | 100M | 0 | 1.19 s | 113,924 KiB | 73,926,852 B | 0 |
| Qwen2.5-1.5B Q4_0 | legacy | 8G | 0 | 3.24 s | 1,148,752 KiB | 1,061,215,600 B | 0 |
| Qwen2.5-1.5B Q4_0 | legacy | 4G | 0 | 3.26 s | 1,148,748 KiB | 1,061,215,600 B | 0 |
| Qwen2.5-1.5B Q4_0 | legacy | 1G | 0 | 3.82 s | 1,125,360 KiB | 1,061,674,688 B | 197 |
| Qwen2.5-1.5B Q4_0 | legacy | 100M | 0 | 6.23 s | 166,444 KiB | 80,428,224 B | 222 |
| SmolLM2-135M Q4_K_M | forgecore | 4G | 0 | 0.20 s | 370,388 KiB | not exposed | n/a |
| Qwen2.5-1.5B Q4_0 | forgecore | 4G | 0 | 1.31 s | 2,839,744 KiB | not exposed | n/a |
| Pythia-70M-deduped Q8_0 (GPT-NeoX) | forgecore | 8G | 0 | 0.14 s | 167,616 KiB | not exposed | n/a |
| Pythia-70M-deduped Q8_0 (GPT-NeoX) | forgecore | 4G | 0 | 0.13 s | 167,576 KiB | not exposed | n/a |
| Pythia-70M-deduped Q8_0 (GPT-NeoX) | forgecore | 1G | 0 | 0.13 s | 167,584 KiB | not exposed | n/a |
| Pythia-70M-deduped Q8_0 (GPT-NeoX) | forgecore | 100M | 0 | 0.13 s | 167,512 KiB | not exposed | n/a |
| Pythia-70M-deduped Q8_0 (GPT-NeoX) | legacy | 1G | 1 | 0.01 s | 23,664 KiB | not exposed | n/a |

- Every legacy comparison generated byte-identical stdout at all four budgets
  (SmolLM2 `da6b6d7d…`, Qwen2.5 `209447e5…`). The 100M Qwen run used 222
  layer-cache evictions and a 76.70 MiB managed peak under the 100.00 MiB
  budget; the 1G run used 197 evictions; 4G/8G used none.
- The earlier GPT-NeoX forgecore runs generated byte-identical stdout
  (`c69c103d…`) at all four budgets **without** `--enforce-ram`. That was
  budget-invariance, not out-of-core evidence. The new enforced matrix below
  uses `--enforce-ram` and shows real reclaim.
- The GPT-NeoX legacy case is the intended controlled rejection: exit 1 in
  0.01 s, `unsupported-architecture` classification, no weights loaded, no
  OOM, no fallback.
- The 20B GPT-NeoX GGUF (13,135,032,320 B file, 13,133,230,080 tensor bytes)
  was executed at 8G under the enforced cgroup cap; see below.
- Previous evidence, not re-run in this matrix: Qwen2.5-7B Q4_K_M two-shard
  legacy at 4G, exit 0, 5:18.67 wall, 4,279,892 KiB max RSS, 4,285,007,616 B
  managed peak, 24 cached layers, 88 evictions. This exercises split
  discovery/cross-shard reads, not forgeCORE streaming.
- ForgeCore `Swaps` was 0. Its RSS includes loader, page cache, Rust heap, and
  native allocations; it is not model residency. The native build is CPU-only,
  so none of these runs is GPU validation.

#### Enforced cgroup v2 evidence (2026-10-25)

`--enforce-ram` creates a child cgroup v2 with `memory.max=<budget>` and
`memory.swap.max=0`, re-execs the run, and reports `memory.peak`,
`memory.events`, selected `memory.stat` counters, and exit status. The cap is
total charged memory for the supervised process group, not RAMforge's logical
ledger. The model remains file-backed (`load_mode=mmap`); the kernel reclaims
clean weight pages under the cap. `--no-repack` disables llama.cpp's anonymous
CPU repack buffers. Cold-cache priming (`posix_fadvise(DONTNEED)`) runs before
load when enforcement is active.

Prompt `The capital of France is`, 1 greedy token, `--ctx 256`,
`--no-repack`, `--backend forgecore`. Exact budgets: 8G = 8,589,934,592;
4G = 4,294,967,296; 1G = 1,073,741,824; 100M = 104,857,600.

| Model | Budget | Exit | Peak charged | `max` events | OOM kills | Model resident | Output |
|---|---:|---:|---:|---:|---:|---|---|
| Qwen2.5-1.5B Q4_0 | 8G | 0 | 1,243,525,120 | 0 | 0 | 1,011.16 MiB (100%) | ` Paris` |
| Qwen2.5-1.5B Q4_0 | 4G | 0 | 1,244,364,800 | 0 | 0 | 1,011.16 MiB (100%) | ` Paris` |
| Qwen2.5-1.5B Q4_0 | 1G | 0 | 1,073,741,824 | 1,457 | 0 | 857.93 MiB (84.8%) | ` Paris` |
| Qwen2.5-1.5B Q4_0 | 100M | 137 | 104,857,600 | 7,814 | 2 | n/a (OOM) | none |
| Pythia-70M Q8_0 | 8G | 0 | 137,224,192 | 0 | 0 | 71.48 MiB (100%) | ` the` |
| Pythia-70M Q8_0 | 4G | 0 | 137,224,192 | 0 | 0 | 71.48 MiB (100%) | ` the` |
| Pythia-70M Q8_0 | 1G | 0 | 137,224,192 | 0 | 0 | 71.48 MiB (100%) | ` the` |
| Pythia-70M Q8_0 | 100M | 0 | 104,857,600 | 373 | 0 | 33.24 MiB (46.5%) | ` the` |
| GPT-NeoX-20B Q4_K_M (prefetch) | 8G | 0 | 8,589,934,592 | 54,721 | 0 | 7.65 GiB (62.5%) | ` Paris` |
| GPT-NeoX-20B Q4_K_M (`--no-prefetch`) | 8G | 0 | 8,589,934,592 | 7,979 | 0 | 7.64 GiB (62.4%) | ` Paris` |

"Model resident" is the end-of-run `/proc/self/smaps` sample, not a peak;
"Peak charged" is the cgroup `memory.peak`.

Qwen at 100M is a controlled failure: its minimum anonymous working set
(155.74 MiB plus kernel) exceeds the cap, so the cgroup OOM-kills the process
group. Pythia at 100M succeeds because its anonymous set is smaller and the
kernel reclaims about half of its weight mapping. Qwen at 1G, 512M, and 256M
all succeeded with 84.8%, 33.1%, and 8.3% of the mapping resident respectively;
the 8G/4G runs were fully resident.

Performance, Qwen2.5-1.5B, 8 greedy tokens, same prompt/context/backend:
unbounded no-cgroup reference 230 ms TTFT, 11.84 tok/s decode, 1.50 s wall,
1.16 GiB max RSS; enforced 1G with repacking 430.9 ms, 3.43 tok/s, 3.86 s;
enforced 1G `--no-repack` 345.1 ms, 3.63 tok/s, 3.07 s, 1.006 GiB RSS;
enforced 512M `--no-repack` 656 ms, 1.59 tok/s, 5.87 s; enforced 256M
`--no-repack` 790 ms, 1.53 tok/s, 6.17 s. The enforced runs are cold-cache
primed; warm and cold no-enforce runs were nearly identical on this SSD, so
the slowdown is dominated by cgroup reclaim. `/usr/bin/time -v` file-system
inputs were 5,211,776 512-byte blocks at 1G, 17,160,536 at 512M, and
18,262,968 at 256M against a 1.07 GB model file; the 20B run reported
50,728,408 blocks for a 13.1 GB file. These counters show the refault I/O cost
of bounded residency.

### No-prefetch load mode (2026-10-25)

forgeCORE now exposes `ModelLoadMode::MmapNoPrefetch`, backed by the documented
`patches/llama-model-residency.patch` and `LLAMA_LOAD_MODE_MMAP_NO_PREFETCH`.
RAMforge wires it as opt-in `--no-prefetch`; the default remains prefaulted
`Mmap`. The mode avoids `MAP_POPULATE`; it does not implement per-layer
residency or replace the cgroup cap.

Qwen2.5-1.5B, 1G, `--no-repack`, 8 greedy tokens:

| Mode | Load | RSS after load | TTFT | Decode | Wall | Peak cgroup | Major faults |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefetch | 706.7 ms | 1,017.01 MiB | 348.3 ms | 3.71 tok/s | 3.06 s | 1 GiB | 552 |
| `--no-prefetch` | 342.3 ms | 80.35 MiB | 709.3 ms | 3.29 tok/s | 3.32 s | 1 GiB | 834 |

Output was identical in both runs. The no-prefetch run moved the initial
weight-fault cost from load to prompt processing.

GPT-NeoX-20B: 13,135,032,320 B file, 13,133,230,080 tensor bytes, SHA-256
`1a35c13535c26bb925a2379d9f021de67a0390124ffe2051aa94412002652d6d`, 8G cap,
`--enforce-timeout 300 --ctx 128 --no-repack`, 1 token.

| Mode | Wall | Setup | TTFT | RSS after load | Max events | Input blocks | Output |
|---|---:|---:|---:|---:|---:|---:|---|
| prefetch | 16.94 s | 6.64 s | 10.03 s | 7.95 GiB | 54,721 | 50,728,408 | ` Paris` |
| `--no-prefetch` | 7.72 s | 0.126 s | 7.385 s | 35.36 MiB | 7,979 | 25,383,576 | ` Paris` |

Both peaked at exactly 8 GiB with no OOM kill. No-prefetch mapping residency
after decode was 7.64 GiB (62.4%); anon 329.79 MiB; file 7.64 GiB; kernel
33.86 MiB; 159 major faults; cgroup swap peak 0; max RSS 7.97 GiB. These are
one-token fit/mechanism results, not throughput or quality evidence.

Model-scoped release (`--release-mapping-pages`) uses the native
`llama_model_release_pages()` over exactly that model's mappings:

- Qwen2.5-1.5B, 1G, prefetch + release: mapping `936.70 MiB -> 0 B`; TTFT
  498.0 ms; decode 3.52 tok/s; wall 3.36 s; cgroup peak 1 GiB.
- Qwen2.5-1.5B, 1G, no-prefetch + release: mapping `0 B -> 0 B`; TTFT
  668.2 ms; decode 3.31 tok/s; wall 3.22 s; cgroup peak 1 GiB.
- Pythia-70M, 192M: mapping `71.48 MiB -> 0 B`; cgroup peak 137,744,384 B;
  output ` the`.
- GPT-NeoX-20B, 8G, prefetch + release: mapping `7.93 GiB -> 0 B`; TTFT
  6.823 s; wall 12.58 s; cgroup peak exactly 8 GiB; 51,732 `max` events;
  44,184,776 input blocks; no OOM; output ` Paris`.
- Fixture scope test: two models, `A 8192->0, B 8192->8192`, followed by a
  successful decode on A. This confirms only the target model's mapping is
  released and pointers remain valid.

Tensor/layer-scoped release (`--inspect-layer`, `--release-layer`) uses the
native tensor metadata and skips pages shared with unselected file-backed
tensors:

- Qwen2.5-1.5B layer 0, `--no-repack`: 12 tensors, 12 file-backed. Release
  `25.12 MiB`, skipped shared `8.00 KiB`; mapping `936.84 -> 911.72 MiB`;
  TTFT 351.6 ms; decode 3.55 tok/s; wall 3.16 s; cgroup peak 1 GiB; output
  unchanged.
- Qwen2.5-1.5B layer 0 with repacking: 12 tensors, 5 file-backed and 7
  anonymous repacked tensors. Anonymous tensors are reported, not released.
- Pythia-70M layer 0 at 192M: released `3.21 MiB`, skipped `8.00 KiB`;
  mapping `71.48 -> 68.27 MiB`; wall 0.209 s; output ` the`.
- GPT-NeoX-20B layer 0 at 8G: 12 tensors, all file-backed; `293.93 MiB`
  submitted, `0` shared pages; mapping `7.92 GiB -> 7.92 GiB`, so no
  measurable RSS drop because those pages were already nonresident; TTFT
  9.207 s; wall 15.18 s; cgroup peak exactly 8 GiB; output ` Paris`.
- `mapping_ranges` fixture test: tiny model selected-tensor release reports
  `released=0, skipped_shared=4096` (conservative shared-page behavior);
  Pythia largest tensor reports `released=27,365,376, skipped_shared=4,096`
  and decode succeeds after repeated release.

`released_bytes` is page bytes submitted to `madvise`, not proof of an RSS
drop. Layer grouping uses the tested `blk.N.*` tensor naming; it is RAMforge
policy, not a native layer index. The operation is called between decode
operations, not during decode.

The cgroup cap does not include shared file pages charged to another cgroup.
The default loader still prefaults with `MAP_POPULATE`; no-prefetch is opt-in
and requires the patched native tree. `memory.swap.max=0` is written when the
swap controller exposes it; `Peak swap` is reported. The enforced timeout path is tested:
Qwen2.5-1.5B at 512M with `--enforce-timeout 1` exits 124, reports
`Result: timeout; cgroup killed`, and leaves no cgroup behind. The 20B model
was not run at 4G, 1G, or 100M. Qwen2.5-7B, DeepSeek-Coder-V2-Lite, and
Qwen2.5-Coder-32B were not run; DeepSeek's `deepseek2` architecture is
inspect/plan-only in the validated registry. See the README for commands and
caveats.

#### GPT-NeoX/Pythia execution evidence

Fixture: `mradermacher/pythia-70m-deduped-GGUF`,
`pythia-70m-deduped.Q8_0.gguf`, 76,722,400 B, SHA-256
`94788a58bb81431b7058837ec2371c303edb20384b615f0afabfa81539294e0d`,
GGUF v3, 6 layers, 512 embedding, 8 heads, 2048 FFN, 16 rotary dimensions,
parallel residual enabled.

```sh
ramforge run /path/to/pythia-70m.Q8_0.gguf --ram 1G \
  --prompt "The capital of France is" --max-tokens 8 \
  --temperature 0 --backend forgecore --memory-report
```

Result: exit 0, **5 prompt tokens** `[510, 5347, 273, 6181, 310]`, output
` the capital of the French Empire.`, 8 tokens, 0.14 s wall, 167,668 KiB max
RSS, 1 GiB requested (this run did not use `--enforce-ram`), logical tensor
bytes 74,948,608 B, `Swaps: 0`. The native llama.cpp probe encodes the same five
ids and produces the same greedy prefix. A second greedy run produces identical
ids.

The tokenizer fix: without an explicit `tokenizer.ggml.add_bos_token`, BPE
vocabularies now default to no BOS, matching llama.cpp; SentencePiece
vocabularies still default to BOS. The previous unconditional `true` inserted
token `0` into Pythia prompts.

Independent numerical check: a NumPy GPT-NeoX forward pass built from the GGUF
metadata and dequantized Q8_0 weights reproduces the native final-token
logits for the same five ids: top-1 id **253** in both, top-10 overlap
**9/10**, cosine similarity **1.000000046**, and max absolute difference
**0.929** on logits of magnitude ~1,579 (about 0.06% relative), consistent
with ggml's f16 GELU lookup and floating-point reduction order. This is a
graph/architecture check, not bit-exact parity and not broad quality
validation.

The 20B GPT-NeoX GGUF has 13,135,032,320 file bytes and 13,133,230,080 tensor
bytes. It is inspectable and plannable at 8G/4G/1G/100M. It was executed at 8G
under `--enforce-ram --no-repack --no-prefetch`: 7.64 GiB of the 12.23 GiB
mapping resident, 7,979 limit events, no OOM kill, one token ` Paris` in
7.72 s (16.94 s with prefetch). This relies on OS demand paging of the
file-backed mapping, not a RAMforge per-layer staging engine. The default
loader still prefaults the whole file; 4G/1G/100M 20B runs were not attempted.

Chat-template support was validated on the real Qwen2.5-1.5B GGUF. The model's
own `tokenizer.chat_template` rendered a 177-byte ChatML prompt, tokenized the
control markers as their model IDs, generated 2 tokens (then EOS), and output
`4` for `What is 2+2? Answer briefly.` The run used 1G, took 1:07.21, peaked
at 1,064,027,136 B managed memory, loaded 1008 layers, and evicted 981. The
same model with a raw prompt did not produce answer-quality output. SmolLM2's
ChatML template also parsed and rendered, though its small-model answer was not
coherent. A chat-templated Qwen2.5-7B run at 4G was stopped by the 900 s harness
timeout before its first token; that is a performance limit, not a quality
result.

Q5_0 block decoding is additionally checked against a vector produced by the
pinned native `ggml` `dequantize_row_q5_0` implementation, not just by
end-to-end text coherence.

The earlier budget-pressure run used `--ram 768M` on Qwen2.5 legacy, below its
managed footprint. It exited 0, generated 8 tokens in 18.36 s, and reported
peak managed 798,260,928 B (761.28 MiB) under the 805,306,368 B (768 MiB)
budget, peak resident layer 26,341,376 B (25.12 MiB), 224 layer loads, 207
layer-cache evictions, max RSS 869,312 KiB, and `Swaps: 0`. This demonstrates
budget-constrained eviction. It does not demonstrate control of total RSS,
which includes unmanaged loader, allocator, page-cache, and workspace memory.

Read-only model hashes from the matrix inventory:

- SmolLM2: `2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d`
- Qwen2.5-1.5B: `dcd819ff094852c38faba6873d8ff0c9d51eadb2844539e52042ae5d647bbfdb`
- Qwen2.5-7B shard 1: `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db`
- Qwen2.5-7B shard 2: `539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a`

**Current ForgeCore note:** static ForgeCore device discovery, explicit
selection, and execution metadata are implemented. The pinned ForgeCore
revision `cb39080` exposes public static model-load/offload options, CPU
weight-repacking control, and `MmapNoPrefetch`, but the RAMforge adapter remains
deliberately CPU-only in this refresh; GPU facts may be discoverable but
non-CPU model execution is explicitly unsupported, with no CPU fallback. The
separate `--enforce-ram` cgroup supervisor is the enforced host-memory path;
`--no-prefetch` only controls the initial mmap fault behavior. See
[README.md](../../README.md) and the historical
[PHASE_4_ENGINEERING_REPORT.md](PHASE_4_ENGINEERING_REPORT.md).

## What RAMforge is

RAMforge is a local, CPU-oriented GGUF inspection, planning, and inference runtime with an explicit static ForgeCore device-discovery boundary. Its original purpose is **usable out-of-core inference**: run a model whose complete tensor payload does not fit in the selected RAM budget by treating the model file as backing storage, retaining only selected persistent weights and the active/cacheable transformer layers in memory.

The repository describes RAM, VRAM, and storage as a hierarchical memory system. Phase 4 can snapshot native CPU/GPU/accelerator facts through ForgeCore and associate them with separate accounting domains. Although the pinned ForgeCore revision exposes static model-load/offload options, the implemented RAMforge ForgeCore adapter remains CPU-only in this refresh and the legacy planner/runtime remains a separate CPU-only path. RAMforge explicitly tracks the memory it owns through `MemoryBudget`; this is not a claim to control process RSS, the operating-system page cache, VRAM, or native backend allocations. `--enforce-ram` adds a separate cgroup v2 hard cap for the supervised process group; it does not turn `MemoryBudget` into an OS cap.

## Workspace and version assumptions

**VERIFIED from the manifests:** the workspace has three crates and uses Cargo resolver 2:

- `crates/ramforge-core` — GGUF parsing, metadata, tensor storage/decoding, tokenizer, quantization, memory budget, bounded cache, and pure reference compute.
- `crates/ramforge-runtime` — discovery, profiles, planner, calibration, persistence, runtime configuration, streaming model, model execution, inference, cache, profiling, and CPU backend.
- `crates/ramforge-cli` — `ramforge` CLI and `ramforge-tui` interactive interface.

All crates currently declare package version `0.1.0`, Rust edition `2021`, and the workspace has no `rust-toolchain.toml` or `rust-version` constraint. The recorded final validation used rustup stable **Rust/Cargo 1.98.1**. Do not interpret that as a repository-pinned toolchain requirement; it is the toolchain used for the current results.

No model file, external llama.cpp checkout, or external reference output is present in this repository.

## Implemented milestones

The following is the repository's milestone history, cross-checked against the current source. These are implementation milestones, not a claim that every numerical path is production-correct for every architecture.

- **GGUF inspection:** `parse_gguf_file` reads GGUF header, metadata, tensor descriptors, offsets, dimensions, and type information without loading tensor payloads. GGUF versions 1–3 are accepted by the parser.
- **Split/model-directory support:** `discover_model_set` resolves a file, a standard `*-00001-of-0000N.gguf` shard set, or a single-model directory; validates split counts/indices/tensor counts, architecture and duplicate tensor names, then exposes a merged model view with virtual data offsets. `GgufDataSource` reads across shard handles without concatenating files on disk.
- **Chat templates:** `ChatTemplate` parses GGUF `tokenizer.chat_template` with a conservative Jinja subset and renders one user message plus an optional system message. `--chat` is opt-in; unsupported constructs fail explicitly. Tokenizer encoding recognizes model control tokens such as role markers.
- **Budgeted file-backed access:** `GgufDataSource` retains a synchronized file handle, performs bounded range reads, tracks logical versus physical I/O, can reuse the file cursor, and supports bounded coalesced reads.
- **Memory and cache accounting:** `MemoryBudget` provides named allocation/resize/release accounting and scoped temporary reservations. `BoundedCache` and the runtime `LayerCache` enforce byte limits and release charges on eviction.
- **CPU dense inference:** the runtime has a Llama/Qwen2-shaped transformer path with tokenizer, RMSNorm, RoPE, attention, GQA, SwiGLU, residuals, KV cache, output projection, sampling, and generation.
- **Out-of-core layer streaming:** `StreamingLlamaModel` loads layers on demand, executes them, and either caches or releases their materialized representation. Persistent weights can remain resident or be streamed according to actual resident size and budget.
- **Quantized tensor support:** compact resident representations and scalar/block-wise dequantization/matvec paths exist for the formats listed below.
- **Accounting hardening:** direct F32 reads, F16/BF16 decoded residency accounting, transactional persistent-weight loading, grouped reads, grouped-buffer reuse, chunked output projection, and chunk-growing KV-cache accounting are implemented and tested.
- **Compute-layer remodel:** `ramforge_core::compute` now contains storage-independent scalar/reference operations and explicit matrix/attention contracts. Runtime scalar operations delegate to those references and optimized paths have focused equivalence tests. This was a structural boundary improvement, not proof of Qwen2.5 numerical parity; see [CURRENT_INFERENCE_ENGINE.md](CURRENT_INFERENCE_ENGINE.md) and [QWEN25_CORRECTNESS_INVESTIGATION.md](QWEN25_CORRECTNESS_INVESTIGATION.md).
- **Planning/orchestration:** inspection, machine/storage discovery, model profiles, capability derivation, calibration, deterministic planner output, plan compilation, plan persistence, and runtime activation are implemented.
- **Static ForgeCore execution boundary:** `DeviceId`, `BackendId`, `DeviceKind`, `DeviceInfo`, capabilities, `DeviceRegistry`, explicit selection, and `ExecutionConfig` are implemented. CPU is selectable; native non-CPU facts are preserved but rejected by the RAMforge adapter's deliberate CPU-only contract. The pinned ForgeCore revision has the public static offload surface, but this task does not enable it. Selection is not residency proof.
- **CLI/TUI diagnostics:** CLI `inspect`, `plan`, `support`, and `run` commands exist. The CLI can emit profile and memory reports, and `run --device` now carries an explicit static ForgeCore selection. The TUI has planning, calibration, persisted-plan compatibility, generation, diagnostics, run history, and export flows; no TUI redesign was made for Phase 4.

## GGUF and model support

### Generic GGUF support

`ramforge-core/src/gguf.rs` parses metadata types, arrays, tensor names, dimensions, GGML types, alignment, descriptor offsets, and computed byte lengths. `GgufModel` in `ramforge-core/src/model.rs` exposes generic inspection data and `ModelInfo` derives common architecture, context, embedding, layer, head, tokenizer, and vocabulary metadata.

Generic inspection and planning are intentionally broader than execution. An architecture name appearing in a GGUF does not by itself make it executable.

### Execution architectures

The explicit registry is `ramforge-runtime/src/support.rs`:

| Architecture/alias | Inspect | Plan | Legacy executor | forgeCORE/libllama | Evidence |
|---|---:|---:|---|---|---|
| `llama` | yes | yes | yes | yes | SmolLM2-135M Q4_K_M, real |
| `qwen2`, including alias `qwen2.5` | yes | yes | yes | yes | Qwen2.5-1.5B Q4_0, real |
| `gptneox` / `gpt_neox` | yes | yes | **not-yet** | **yes** | Pythia-70M-deduped Q8_0, real |
| `mistral` | yes | yes | not-yet | not-yet | literal `mistral` tag has no validated execution path; Mistral GGUFs normally declare `llama` |
| `qwen3` | yes | yes | not-yet | not-yet | not validated |
| `qwen35` / `qwen3.5` / `qwen3_5` | yes | yes | not-yet | not-yet | hybrid attention/SSM; explicitly not aliased to Qwen2 |
| `gemma` family | yes | yes | not-yet | not-yet | not validated |
| `phi` family | yes | yes | not-yet | not-yet | not validated |

The legacy runtime model parser is `LlamaConfig` in
`crates/ramforge-runtime/src/model.rs`. It accepts the dense Llama/Qwen2 tensor
contract and reads architecture-specific or fallback metadata keys. A
`gptneox` model is rejected there before tensor payloads are read; the error
names `--backend forgecore` and the registry reports `legacy-execution=not-yet
forgecore-execution=yes`. The legacy Qwen2.5 path still must not be treated as
broadly numerically validated for every prompt.

`gptneox` is not approximated in RAMforge. The forgeCORE/libllama backend
delegates execution to the pinned native `src/models/gptneox.cpp`
implementation, which covers fused QKV, GPT-NeoX-style partial RoPE, parallel
residual, and GELU MLP. RAMforge still owns orchestration, tokenization,
sampling, CLI output, and the explicit backend choice.

### Tensor/quantization formats

The legacy executor's source-level inference registry marks these as supported:

- `F32`
- `F16`
- `BF16`
- `Q4_0`
- `Q5_0`
- `Q8_0`
- `Q2_K`
- `Q3_K`
- `Q4_K`
- `Q5_K`
- `Q6_K`
- `Q8_K`

`Q4_1`, `Q5_1`, `Q8_1`, IQ formats, and other unimplemented GGUF types remain
unsupported by the **legacy executor** even though the generic type enum can
preserve unknown/future tags for inspection. This list is not the forgeCORE
type-support list: once a model reaches `--backend forgecore`, tensor decoding
belongs to the pinned llama.cpp runtime. RAMforge claims end-to-end real-model
evidence for `Q4_0` (Qwen2.5-1.5B), `Q4_K_M` (SmolLM2-135M), and `Q8_0`
(Pythia-70M), not a complete native type enumeration.

`Q5_0` was added to the legacy inference registry after validating the real
`SmolLM2-135M-Instruct-Q4_K_M.gguf` model. Its block decode is checked against a
vector produced by the pinned native `ggml` `dequantize_row_q5_0`
implementation.

## Current subsystem state

### Production-oriented infrastructure

The following infrastructure is implemented with extensive unit/integration coverage and is the part intended to survive a ForgeCore rewrite:

- GGUF descriptor parsing and validation.
- File-backed `GgufDataSource`, bounded reads, cursor reuse, read profiling, coalescing, and grouped buffer reuse.
- Named `MemoryBudget` accounting and scoped temporary reservations.
- Persistent-weight policy, layer grouping, layer read plans, `LayerCache`, and rollback/eviction accounting.
- Machine and storage discovery facts, model/storage/machine fingerprints, capability derivation, planner profiles, and plan compilation.
- Bounded calibration tasks and persisted execution-plan compatibility checks.
- CLI/TUI diagnostic collection, profiling, memory visibility, run history, persistence, and error staging.
- cgroup v2 hard-budget supervision (`--enforce-ram`) with `memory.peak`, limit/OOM events, page-fault counters, timeout kill, and explicit failure reporting; covered by process-level tests.
- Explicit file-backed mmap loading, `--no-repack` CPU weight-repacking control, opt-in `--no-prefetch` mapping mode, opt-in model-wide and tensor/layer range page release, model-mapping RSS/PSS measurement, and cold-cache priming for enforced runs.
- ForgeCore-observed TTFT, prompt-processing, decode-token-rate, and wall-clock profiling.

“Production-oriented” here means that these are concrete implemented subsystems with tests. It is not a blanket production-readiness or model-quality guarantee.

### Experimental or not fully validated

- GPT-NeoX/Pythia execution is validated through forgeCORE on one 70M Q8_0
  fixture, including native tokenizer/greedy parity and an independent NumPy
  logits check. The 20B GPT-NeoX model executed once at 8G under the cgroup cap
  (one greedy token); that is a fit/mechanism result, not broad GPT-NeoX
  quality validation.
- Real-model numerical quality and compatibility, especially Qwen2.5-1.5B, are not established.
- The current scalar/reference remodel has synthetic parity tests but does not explain the recorded Qwen2.5 divergence.
- Real-model generated-text quality is not validated by the repository test suite.
- The Qwen2.5 numerical trace is an ignored test requiring an external model and explicit budget.
- Calibration is bounded and records measured facts; it does not prove inference correctness or choose a different numerical implementation.
- The quantized microbenchmark is ignored and is not part of normal workspace validation.

### Incomplete or unsupported

The repository does not implement GPU model execution or GPU offload through RAMforge (the current adapter intentionally leaves ForgeCore's static offload surface unused), asynchronous I/O, prefetch/double buffering, HTTP serving, model downloading, MoE execution, speculative decoding, or the additional quantization formats listed as unsupported. `qwen35` is intentionally inspect/plan-only because its hybrid attention/SSM architecture is not the Qwen2 path.

## Important commands

These are the repository's normal commands. Commands requiring a model are templates and were not run during the handoff unless explicitly marked in [VALIDATION.md](VALIDATION.md).

```text
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo build --workspace --release

cargo run -p ramforge-cli -- inspect MODEL.gguf
cargo run -p ramforge-cli -- plan MODEL.gguf --ram 8G
cargo run -p ramforge-cli -- support
cargo run -p ramforge-cli -- run MODEL.gguf --ram 8G --prompt "Hello" --max-tokens 32 --profile --memory-report
cargo run -p ramforge-cli -- run MODEL.gguf --ram 8G --backend forgecore --device cpu --prompt "Hello"
```

The ignored real-model diagnostic is documented in [VALIDATION.md](VALIDATION.md) and the source doc comment in `crates/ramforge-runtime/src/inference.rs`; do not run it casually.

## Validation snapshots

The following results are historical validation for the pre-Phase-4 source
snapshot, recorded with Rust/Cargo 1.98.1:

- `cargo fmt --all -- --check`: passed.
- `cargo test --workspace`: passed — 44 CLI tests, 112 core tests, 195 runtime tests; **351 passed, 0 failed, 2 ignored**.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: passed.
- `cargo build --workspace --release`: passed.

For the current Phase-4 source, `cargo` and `rustc` are not available on the
current `PATH`, the historical local toolchain/native locations are absent,
and no compatible native ForgeCore build is present. Phase-4 formatting,
compilation, tests, clippy, release build, and native-device execution are
therefore **not run**. See [PHASE_4_ENGINEERING_REPORT.md](PHASE_4_ENGINEERING_REPORT.md)
for the authoritative Phase-4 validation record. The two historical ignored
tests include the expensive real-model Qwen2.5 diagnostic and the quantized
microbenchmark.
