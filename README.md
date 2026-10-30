# RAMforge

RAMforge is a local inference runtime designed to run AI models that may be significantly larger than the available RAM or VRAM by treating RAM, VRAM, and storage as a hierarchical memory system.

> **Predictive residency (2026-10-30):** RAMforge now has a deterministic predictive staging layer over validated profiles: bounded confidence-ranked predictions, high-confidence staging and medium-confidence speculative prefetch, explicit budget/admission decisions, feedback history, and a deterministic simulator. Prediction is advisory only; `release_authorized` is always false and release still requires the native quiescence gate. See [PREDICTION.md](docs/handoff/PREDICTION.md).
>
> **Native lifetime contract (2026-10-30):** forgeCORE now exposes the pinned native execution-quiescence/ownership contract: model/context instance identities, execution generations, active/in-flight contexts, a point-in-time quiescence observation, and a non-blocking release gate that returns `EBUSY` while any context has asynchronous compute in flight. RAMforge binds that contract into its planner and lifetime state machine, persists validated execution profiles, and reports the contract facts. Intra-decode eviction remains unsupported because synchronized completion is still not proof of future reuse. See [GRAPH_SAFETY.md](docs/handoff/GRAPH_SAFETY.md).
>
> **Graph-aware residency (2026-10-25):** RAMforge now builds a unified model-scoped memory ledger, execution profiles from bounded graph traces (with validated optional persistence), deduplicated bounded staging plans, and deterministic `conservative`/`memory-bound`/`reuse-aware` policy profiles in addition to `off`/`observe`/`active`. Operation reports include explicit provenance, reconciliation, requested versus native-accepted versus observed bytes, timing, and an output fingerprint; `--residency-report-json PATH` emits schema version 1. Release mutations remain restricted to the post-operation safe boundary. Intra-decode eviction is still unsupported because the pinned callback does not prove resource lifetime. See [MEMORY_ACCOUNTING.md](docs/handoff/MEMORY_ACCOUNTING.md), [GRAPH_SAFETY.md](docs/handoff/GRAPH_SAFETY.md), and [BENCHMARKS.md](docs/handoff/BENCHMARKS.md).
>
> **GPT-NeoX/Pythia execution and enforced budgets (2026-10-25):** `gptneox` GGUFs are executable through the explicit `--backend forgecore` path, which delegates to the pinned llama.cpp/ggml implementation. The legacy RAMforge executor still rejects the architecture before reading weights, with an actionable error naming the required backend; it never falls back or approximates GPT-NeoX math. Real Pythia-70M Q8_0 runs match the native tokenization and greedy continuation, and an independent NumPy forward pass confirms the final-token logits (top-1 exact, cosine similarity 1.000000046). The 20B GPT-NeoX GGUF (13.13 GB tensor bytes) now executes under an 8 GiB hard cgroup v2 cap with `--enforce-ram --no-repack --no-prefetch`: 7.64 GiB of the model mapping resident at the end-of-run sample, 7,979 limit events, no OOM kill, and one greedy token ` Paris` in 7.72 s. The same 8 GiB run with prefetching took 16.94 s. This is out-of-core execution under a kernel-enforced cap, not a throughput or quality guarantee. See [Enforced host-memory budgets](#enforced-host-memory-budgets-cgroup-v2).
>
> **Usable Out-of-Core Inference:** `ramforge run` now streams decoded assistant text to stdout with UTF-8 byte-token accumulation. Optional `--profile` reports measured GGUF I/O, layer load/compute/release, quantized/F32 matvec, allocation, sampling, token latency, bytes read, and peak managed memory. `--memory-report` separates RAMforge-managed memory, process RSS, and system memory. `ramforge support` exposes the architecture/quantization registry; qwen35 remains inspectable/plannable but intentionally not executable.
>
> **Milestone 7.2 Status (Fast Float Load):** F32 tensor payloads now read directly into their final `Vec<f32>` allocation without a raw-byte intermediate, destination zero-fill, or little-endian per-element decode loop on little-endian hosts. Full tensors and streamed F32 chunks/rows use the direct path; the truthful load charge is now 1× file bytes. F16/BF16 remain decoded F32 with their 3× load transient, and quantized representations are unchanged.
>
> **Milestone 7.1 Baseline (Accounting Hardening):** Resident persistent tensors establish their full load-transient charge **before** file I/O, settle atomically to the actual owned representation, and roll back all startup charges on failure. At M7.1 the load factors were quantized 1×, F32 2×, and F16/BF16 3×; M7.2 supersedes only F32 with its direct 1× representation. The 25% retention policy uses decoded/compact resident bytes rather than file bytes. Final hidden state is caller-owned under a lifetime-matched `tmp:hidden` charge.
>
> **Milestone 6.1 Baseline (Correctness & Accounting Fixes):** `generate()` is cleanly repeatable on one engine (explicit KV reset + failure-proof budget); RoPE uses the correct llama/qwen2 **half-split** pair convention; F16/BF16 resident RAM is booked at its true decoded size (4 B/elem, with a 3×-file-byte load transient); qwen2 Q/K/V biases are loaded, budgeted, validated (all-or-none + exact shape), and applied after the projections. The historical M6.1 baseline was verified by 132 synthetic tests; current validation is summarized below.
>
> **Milestone 6 Status (Managed Out-of-Core Path):** The legacy inference path charges its named, RAMforge-managed allocations via scoped accounting helpers; the cache is budget-charged per entry; matrix layout is the explicit GGML/GGUF convention (no orientation guessing, no full-F32 fallbacks); logits use a single budget-charged buffer with a budget-aware chunked streamed projection; the KV cache grows chunk-wise without prefix copies; the F32 matvec hot path uses AVX2/rayon. CPU-only, llama/qwen2 dense models. GPU, MoE, HTTP not implemented. This ledger is not a process-RSS or OS-memory cap; allocator overhead, native/dependency allocations, page cache, and some legacy returned buffers remain outside it.

## Purpose

- GGUF parsing without loading payloads
- File-backed tensor access via `GgufDataSource`
- Checked RAMforge-managed allocation accounting via `MemoryBudget` (not an RSS or OS-memory cap)
- Bounded LRU cache via `BoundedCache`
- Real CPU inference with the retained legacy layer-streaming path
- Native quantized inference without full F32 expansion

### Memory-contract boundary

`MemoryBudget` is a RAMforge-managed logical ledger, not a process RSS cap or a hard operating-system memory limit. The first safe accounting tranche covers named persistent weights, decoded layer ownership, the generation KV cache (including old/new growth overlap), caller-owned generation buffers, staged K/V workspace, and bounded streamed read/decode buffers. Reservations are checked before the managed buffers are allocated, and charges remain live until those owned buffers are dropped.

The ledger does not claim complete process-memory coverage. Allocator metadata and fragmentation, Rust runtime/container overhead not included in a named charge, native/dependency allocations, file-system page cache, tokenizer/output growth, and legacy APIs that return newly owned `Vec` values (including the raw-cache `get_tensor`/`remove_budgeted` and persistent-weight `get_embedding`/`to_f32_vec` compatibility paths) remain outside or only coarsely represented. A standalone `StreamingLlamaModel` receives a borrowed budget, so callers must invoke its terminal `release_persistent` method before dropping it; the owning `InferenceEngine` performs that ordering automatically. Standalone core compute/reference helpers and calibration/profiling containers are likewise not automatically tied to the inference ledger. RSS and OS limits are reported separately where available; they are not enforced by `MemoryBudget`.

## Phase 2 ForgeCore backend

RAMforge now also exposes an explicit CPU-only ForgeCore path:

```text
ramforge run MODEL.gguf --ram 8G --prompt "Hello" --backend forgecore
    → forge_core::Model
    → forge_core::Context
    → forge_core::Batch
    → Context::decode()
    → forge_core::Logits
    → RAMforge tokenizer/sampling/output
```

Use `ramforge run ... --backend forgecore` to select it. The default
`--backend legacy` path remains available as the comparison/reference
`InferenceEngine`; ForgeCore failures never fall back to it implicitly. The
adapter is pinned to verified ForgeCore commit
`8d67cb466dba1ee013efc13ee152391602752473` and keeps direct
`forge-sys` and llama.cpp/ggml FFI out of RAMforge. That ForgeCore revision
exposes `ModelOptions::use_extra_bufts`, which RAMforge uses to disable CPU
weight repacking when requested.

Phase 2 does not claim GPU, ForgeCore-native per-allocation accounting, or
migration of the legacy numerical modules. RAMforge GGUF inspection, planning,
diagnostics, cache, residency, and accounting infrastructure remains retained;
ForgeCore native model, context, KV, and backend allocations remain outside
RAMforge's `MemoryBudget`, while safe device facts are reported only where the
adapter can observe them. The separate `--enforce-ram` supervisor described
below provides an operating-system hard cap; it is not a claim that
`MemoryBudget` controls native allocations. Runtime linking follows ForgeCore's
native runbook; arrange `LD_LIBRARY_PATH=$FORGE_LLAMA_DIR/build/bin` (or an
equivalent final-binary rpath) for the CPU shared libraries.

## Enforced host-memory budgets (cgroup v2)

`--ram` by itself remains a RAMforge-managed/planning value. Add
`--enforce-ram` to turn the same byte value into a hard kernel cap for the
supervised process group:

```sh
ramforge run MODEL.gguf --ram 512M --enforce-ram --ctx 256 --no-repack \
  --prompt "The capital of France is" --max-tokens 8 --temperature 0 \
  --backend forgecore --memory-report --profile
```

The supervisor creates an unprivileged child cgroup v2 below the caller's
delegated user subtree, writes `memory.max=<bytes>` and `memory.swap.max=0`,
sets `memory.oom.group=1`, re-execs the same command, and reports
`memory.peak`, `memory.events`, selected `memory.stat` counters, wall time, and
the child exit status. The cap applies to **charged memory of that process
group**: anonymous allocations, file-backed mapped pages charged to the group,
and kernel memory. It does not cap memory outside the group, and it does not
make RSS equal to the limit.

Weight residency: llama.cpp maps the GGUF file (`load_mode=mmap`) and currently
prefaults it with `MAP_POPULATE`. RAMforge does not copy the weights into its
own buffers, and it does not implement per-layer eviction on the forgeCORE
path. Under the cgroup cap, the kernel evicts clean file-backed weight pages and
refaults them on demand. `--no-repack` disables upstream CPU weight-repacking
buffers, which otherwise create a large anonymous copy of quantized weights.
For Qwen2.5-1.5B Q4_0 this reduced anonymous residency from 858.60 MiB to
155.74 MiB.

Cold-cache priming: when enforcement is active, RAMforge calls
`posix_fadvise(POSIX_FADV_DONTNEED)` for the model shards before load. This
makes the measured run pay for the pages it faults and makes the cgroup
accounting meaningful. It is a hint, not a guarantee; it cannot drop pages
that are still mapped or files on tmpfs. Keep large test models on disk-backed
storage.

Flags:

- `--enforce-ram`: hard cgroup v2 limit for the run.
- `--enforce-timeout SECONDS`: wall-clock cap (default 900). On timeout the
  supervisor writes `cgroup.kill`, reaps the child, and exits 124.
- `--ctx TOKENS`: ForgeCore context override. The legacy executor rejects it.
  Context size controls the KV cache and is essential for small budgets.
- `--no-repack`: disable llama.cpp CPU weight repacking.
- `--no-prefetch`: select `ModelLoadMode::MmapNoPrefetch`, which skips the
  native whole-file `MAP_POPULATE`. This requires the documented forgeCORE
  native patch; default `--backend forgecore` behavior remains prefaulted
  mmap.
- `--release-mapping-pages`: ask the native model to release resident pages
  from exactly its own file-backed mappings after load. On Linux this uses
  `madvise(MADV_DONTNEED)`; it does not unmap memory or invalidate tensor
  pointers. Unsupported platforms fail explicitly. This is independent of
  `--no-prefetch` and is a no-op when the mapping is already nonresident.
- `--inspect-layer N`: print exact native mapping metadata for tensors named
  `blk.N.*`. Shows file-backed versus anonymous tensors, mapping index, offset,
  size, and page bounds. It does not release pages.
- `--release-layer N`: request page release for the file-backed tensors in
  layer `N` after load. Pages shared with unselected file-backed tensors are
  skipped conservatively. This is tensor/range-scoped, not per-layer execution
  scheduling, and cannot guarantee a lower load-time peak. It conflicts with
  `--release-mapping-pages` and `--inspect-layer`.
- `--residency-policy off|observe|conservative|memory-bound|reuse-aware|active`:
  execution-aware residency policy. `off` is the default. `observe` builds
  inventory/plans and reports without mutation. `conservative` releases only
  tensors with no shared page bytes. `memory-bound` targets the configured
  soft budget. `reuse-aware` uses a valid in-memory graph profile to order cold
  resources first. `active` is the compatibility alias for memory-bound
  behavior. All mutation happens at a safe post-operation boundary; none occurs
  during decode.
- `--residency-target SIZE`: soft process residency target. Defaults to
  `--ram` for mutating policies. This is not a hard limit.
- `--residency-reserve SIZE`: safety reserve subtracted from the soft target.
- `--residency-report`: print the structured ledger, reconciliation, admission,
  plans, staging/application results, timing, and before/after observations.
- `--residency-report-json PATH`: write the same operation as a versioned JSON
  document (`schema_version: 1`). A write failure is a command error.
- `--residency-profile-load PATH`: load a validated execution profile for
  reuse-aware planning and staging order. Model identity, model size/mtime,
  inventory generation, schema, and native contract must match.
- `--residency-profile-save PATH`: atomically save the current complete trace
  profile (requires `--graph-trace`). Incomplete captures are rejected.
- `ramforge profile PATH [--model MODEL] [--json]`: inspect a persisted profile
  and, when `--model` is supplied, validate model identity.
- `--prediction`: enable deterministic predictive staging from
  `--residency-profile-load`. It conflicts with `--residency-prefetch`.
- `--prediction-min-confidence PERCENT`: minimum confidence (0–100).
- `--prediction-window SIZE`: strict predicted page-byte staging window.
- `--prediction-speculative SIZE`: portion of the window for medium-confidence
  speculative hints.
- `--prediction-max-inflight N`: configured concurrency bound; the current
  implementation executes one synchronous native prefetch call.
- `--prediction-history-load PATH`: validated feedback history.
- `--prediction-history-save PATH`: atomically save feedback after a complete
  trace (requires `--graph-trace`).
- `--residency-prefetch SIZE`: best-effort bounded tensor-page prefetch
  (`MADV_WILLNEED` hint). The plan deduplicates overlapping page ranges;
  requested, native-accepted, and observed mapping changes are reported
  separately.
- `--graph-trace`: opt-in bounded scheduler eval trace. It forces observed node
  groups to be computed separately and synchronized; expect overhead.
- `--graph-trace-events N`: retained event bound (default 4096).
- `--graph-trace-detail`: print individual trace events.

### Enforced budget matrix

Prompt `The capital of France is`, 1 greedy token, `--ctx 256`,
`--no-repack`, `--backend forgecore`, `--enforce-ram`, cold-cache primed.
Exact bytes: `8G` = 8,589,934,592; `4G` = 4,294,967,296; `1G` =
1,073,741,824; `100M` = 104,857,600.

| Model | Budget | Exact bytes | Exit | Peak charged | `max` events | OOM kills | Model mapping resident | Output |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Qwen2.5-1.5B Q4_0 | 8G | 8,589,934,592 | 0 | 1,243,525,120 | 0 | 0 | 1,011.16 MiB (100%) | ` Paris` |
| Qwen2.5-1.5B Q4_0 | 4G | 4,294,967,296 | 0 | 1,244,364,800 | 0 | 0 | 1,011.16 MiB (100%) | ` Paris` |
| Qwen2.5-1.5B Q4_0 | 1G | 1,073,741,824 | 0 | 1,073,741,824 | 1,457 | 0 | 857.93 MiB (84.8%) | ` Paris` |
| Qwen2.5-1.5B Q4_0 | 100M | 104,857,600 | 137 | 104,857,600 | 7,814 | 2 | n/a (OOM) | none |
| Pythia-70M Q8_0 | 8G | 8,589,934,592 | 0 | 137,224,192 | 0 | 0 | 71.48 MiB (100%) | ` the` |
| Pythia-70M Q8_0 | 4G | 4,294,967,296 | 0 | 137,224,192 | 0 | 0 | 71.48 MiB (100%) | ` the` |
| Pythia-70M Q8_0 | 1G | 1,073,741,824 | 0 | 137,224,192 | 0 | 0 | 71.48 MiB (100%) | ` the` |
| Pythia-70M Q8_0 | 100M | 104,857,600 | 0 | 104,857,600 | 373 | 0 | 33.24 MiB (46.5%) | ` the` |
| GPT-NeoX-20B Q4_K_M (prefetch) | 8G | 8,589,934,592 | 0 | 8,589,934,592 | 54,721 | 0 | 7.65 GiB (62.5%) | ` Paris` |
| GPT-NeoX-20B Q4_K_M (`--no-prefetch`) | 8G | 8,589,934,592 | 0 | 8,589,934,592 | 7,979 | 0 | 7.64 GiB (62.4%) | ` Paris` |

"Model mapping resident" is a point-in-time `/proc/self/smaps` sample taken
after decode, not a peak. "Peak charged" is the cgroup's `memory.peak`.

The Qwen 100M case is a controlled failure: its minimum anonymous working set
(155.74 MiB plus kernel overhead) is larger than 100 MiB, so the kernel OOM-kills
the process group. The supervisor reports exit 137, `oom_kill=2`, and
`Result: cgroup OOM kill`; it does not report success. Pythia-70M at 100M is
different: its smaller anonymous working set fits, and the kernel reclaims
about half of its 71.48 MiB weight mapping while the run still completes.

### Measured cost of bounded residency

Qwen2.5-1.5B Q4_0, same prompt, 8 greedy tokens, `--ctx 256`, cold-cache primed.
The unbounded row is a separate warm/no-cgroup reference run; the enforced rows
are the meaningful comparison. Warm and cold no-enforce runs were nearly
identical on this SSD (TTFT 230 ms, 11.8 tok/s), so the slowdown below is
dominated by cgroup reclaim, not cache temperature.

| Configuration | Budget | Peak charged | Model resident | TTFT | Decode | Wall | Max RSS |
|---|---:|---:|---|---:|---:|---:|---:|
| Unbounded, `--no-repack` | 1G requested (diagnostic) | n/a | 1,011.16 MiB (100%) | 230 ms | 11.84 tok/s | 1.50 s | 1.16 GiB |
| Enforced, repack enabled | 1G | 1 GiB | 147.67 MiB (14.6%) | 430.9 ms | 3.43 tok/s | 3.86 s | 1.005 GiB |
| Enforced, `--no-repack`, prefetch | 1G | 1 GiB | 845.59 MiB (83.6%) | 345.1 ms | 3.63 tok/s | 3.07 s | 1.006 GiB |
| Enforced, `--no-repack`, `--no-prefetch` | 1G | 1 GiB | 843.42 MiB (83.4%) | 709.3 ms | 3.29 tok/s | 3.32 s | 1.01 GiB |
| Enforced, `--no-repack` | 512M | 512 MiB | 334.74 MiB (33.1%) | 656 ms | 1.59 tok/s | 5.87 s | 530 MB |
| Enforced, `--no-repack` | 256M | 256 MiB | 84.36 MiB (8.3%) | 790 ms | 1.53 tok/s | 6.17 s | 270 MB |

Disabling repacking at 1G reduced TTFT from 430.9 ms to 345.1 ms and wall time
from 3.86 s to 3.07 s while cutting anonymous residency by about 5.5x. The
512M and 256M runs keep only 33.1% and 8.3% of the weight mapping resident and
pay for refaults; they are correctness and fit demonstrations, not throughput
targets. All successful Qwen rows produced the same text.

`/usr/bin/time -v` file-system input counts show the refault amplification:
5,211,776 512-byte blocks at 1G (about 2.7 GB), 17,160,536 at 512M (about
8.8 GB), and 18,262,968 at 256M (about 9.4 GB), against a 1.07 GB model file.
The 20B prefetch run reported 50,728,408 blocks (about 26 GB) for a 13.1 GB
file. These are kernel block counters, not model bytes, and confirm that the
bounded runs trade I/O for resident memory.

The `--no-prefetch` mode removes the initial whole-file fault:
Qwen2.5-1.5B at 1G loaded in 342.3 ms instead of 706.7 ms, with process RSS
after load of 80.35 MiB instead of 1,017.01 MiB. TTFT rose from 348 ms to
709 ms because prompt processing then faulted the weights on demand; decode was
3.29 tok/s instead of 3.71 tok/s and wall was 3.32 s instead of 3.07 s. The
cgroup peak stayed exactly 1 GiB, the output was identical, and major faults
rose from 552 to 834. This is controlled demand paging, not per-layer
residency.

For GPT-NeoX-20B at 8G, prefetch took 16.94 s wall, 6.64 s setup, and
50,728,408 input blocks; `--no-prefetch` took 7.72 s wall, 0.126 s setup, and
25,383,576 input blocks. Load-time RSS fell from 7.95 GiB to 35.36 MiB, and
cgroup `max` events fell from 54,721 to 7,979. TTFT was 7.385 s, mapping
residency after decode was 7.64 GiB (62.4%), cgroup peak was exactly 8 GiB,
and the output was the same ` Paris`.

### Model-scoped page release

`--release-mapping-pages` calls the new native
`llama_model_release_pages()` after load, before context creation. The native
operation walks only `model->pimpl->mappings`, releases their mapped fragments
with Linux `madvise(MADV_DONTNEED)`, never unmaps, and returns `ENOTSUP` on
unsupported platforms. It is not pathname matching and cannot touch another
model's mappings.

Qwen2.5-1.5B, 1G, `--no-repack`, 8 greedy tokens:

| Mode | Mapping before | Mapping after | TTFT | Decode | Wall | Peak cgroup | Input blocks |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefetch + release | 936.70 MiB | 0 B | 498.0 ms | 3.52 tok/s | 3.36 s | 1 GiB | 6,158,504 |
| no-prefetch + release | 0 B | 0 B | 668.2 ms | 3.31 tok/s | 3.22 s | 1 GiB | 4,795,560 |

Both produced the same text. Release after a prefetch load removes the
resident mapping before decode, but the later prompt refaults the weights; it
is not a throughput optimization by itself.

Small fixture, Pythia-70M, 192M enforced: mapping `71.48 MiB -> 0 B`, cgroup
peak `137,744,384 B`, output ` the`, exit 0.

GPT-NeoX-20B, 8G, 300 s timeout, `--no-repack --release-mapping-pages`
(default prefetch): mapping `7.93 GiB -> 0 B`; TTFT 6.823 s; wall 12.58 s;
model mapping resident after decode 7.64 GiB (62.5%); cgroup peak exactly
8 GiB; 51,732 `max` events; 44,184,776 input blocks; no OOM; output ` Paris`.
The native test loads two tiny models and reports `A 8192->0, B 8192->8192`,
then decodes successfully against A, confirming model scoping and pointer
validity.

### Layer-scoped mapping release

`--inspect-layer N` and `--release-layer N` use native tensor metadata, not
pathname matching. RAMforge groups tensors by the tested models' `blk.N.*`
naming; this is layer policy, not a native layer index. If no tensors match,
the request fails explicitly.

Qwen2.5-1.5B layer 0:

- `--no-repack`: 12 tensors, 12 file-backed, 0 anonymous.
- Repacking enabled: 12 tensors, 5 file-backed, 7 anonymous. Anonymous
  repacked tensors are reported, not misclassified.
- `--release-layer 0` at 1G: released 25.12 MiB, skipped 8.00 KiB shared;
  mapping residency `936.84 -> 911.72 MiB`; TTFT 351.6 ms; decode 3.55 tok/s;
  wall 3.16 s; cgroup peak 1 GiB; output unchanged. No-release at the same
  settings was TTFT 345.1 ms, 3.63 tok/s, 3.07 s, so layer release was close
  to neutral here.

Pythia-70M layer 0 at 192M: released 3.21 MiB, skipped 8.00 KiB; mapping
`71.48 -> 68.27 MiB`; wall 0.209 s; output ` the`; exit 0.

GPT-NeoX-20B layer 0 at 8G/300 s: 12 tensors, all file-backed; 293.93 MiB
submitted to release, 0 shared pages skipped; mapping residency was
`7.92 GiB -> 7.92 GiB`, so **no measurable RSS drop** was observed because
those pages were already nonresident under the 8G cap. TTFT 9.207 s; wall
15.18 s; cgroup peak exactly 8 GiB; 56,479 `max` events; output ` Paris`.
`released_bytes` reports page bytes submitted, not proof of an RSS drop.

This is range-scoped page advice, not execution scheduling. It does not
release pages during decode, does not implement per-layer eviction, and does
not guarantee a lower load-time peak.

### Execution-aware residency engine

The runtime now has a RAMforge-owned residency engine
(`ramforge-runtime/src/residency_engine.rs`) connected to the real forgeCORE
decode path:

1. Build a model-scoped `ResourceInventory` from native tensor metadata.
2. Classify file-backed, anonymous/repacked, and unknown resources.
3. Infer layer identity only from the tested `blk.N.*` naming convention; an
   unrecognized name remains unknown.
4. Compute page ranges and conservative shared-page bytes.
5. Apply a deterministic policy with soft target, safety reserve,
   required/pinned exclusions, and estimated reclaimable capacity.
6. Enforce lifecycle transitions around decode:
   `ModelReady -> PreparingOperation -> Executing -> Synchronizing -> SafeBoundary -> Planning -> Applying -> SafeBoundary`.
7. Apply releases only after the context is dropped at a safe boundary.
8. Report estimates, native results, and before/after observations separately.

`decode_tokens` copies logits through `llama_get_logits_ith`, which synchronizes
native work; the adapter also calls `synchronize()` explicitly before the
lifecycle enters `SafeBoundary`. No release is attempted while a decode is
active. There is no intra-decode layer callback or layer eviction scheduler.

Qwen2.5-1.5B Q4_0, 1 GiB enforced, `--no-repack`, prompt
`The capital of France is`, 8 greedy tokens, `--ctx 256`:

| Policy | TTFT | Decode | Wall | Mapping before/after | Process RSS before/after | Peak cgroup |
|---|---:|---:|---:|---|---|---:|
| off | 360.3 ms | 3.46 tok/s | 3.21 s | 856.64 MiB / 856.64 MiB | 1.00 GiB / 1.00 GiB | 1 GiB |
| observe, target 512M | 364.0 ms | 3.20 tok/s | 3.41 s | 862.54 / 862.54 MiB | 1.00 / 1.00 GiB | 1 GiB |
| active, target 512M | 353.1 ms | 3.41 tok/s | 3.31 s | 855.68 / 476.57 MiB | 1021.21 / 642.10 MiB | 1 GiB |

Active post-operation result: 30 selected tensors, 514.44 MiB submitted,
84.00 KiB shared pages skipped; mapping delta `-397,520,896` bytes; process RSS
delta `-397,525,760` bytes; cgroup current unchanged at 1022.50 MiB; peak
1 GiB; no OOM. The plan's `estimated_after` was 506.84 MiB and
`estimated_budget_met=true`, but the observed process RSS remained 642.10 MiB
above the 512 MiB target. The estimate is not a guarantee: anon/KV/scratch
memory and unsubmitted/shared pages remain. The cgroup remains the enforcement
mechanism; the soft target is policy only.

Observe mode selected 31 tensors in its post-operation plan and performed no
native release; mapping and process residency were unchanged. Admission for
both target-512M runs was `AdmittedWithWarning`, because current process
memory plus reserve exceeded the soft target.

### Graph trace, manifest, and bounded staging

`--graph-trace` installs the pinned public scheduler eval callback. For each
observed node group, the scheduler computes the group and calls
`ggml_backend_synchronize` before the `ask == false` callback. forgeCORE
records node name, op name, source names/transient IDs, size, phase, and an
explicit `BackendSynchronized` completion. Tracing disabled uses the original
scheduler path.

Qwen2.5-1.5B Q4_0, 1 GiB, one token, `--graph-trace-events 4096`:

- 958 requested and 958 synchronized-completion events; 1,916 retained; 0 dropped.
- Manifest: 958 nodes, 1,050 tensor resources, `edges_known=true`,
  `completion_proven=true`.
- Last-use analysis: 1,050 graph-order relationships, 0 unknown.
- `intra_decode_safe=false`: graph-order last use does **not** prove the next
  graph/token will not reuse the weight.
- Wall 1.108 s; cgroup peak 1 GiB; output ` Paris`.

With `--graph-trace-events 512`, 1,404 events were dropped and only 512
retained; the manifest covered 256 nodes/292 resources. Bounded tracing is
explicitly incomplete at overflow.

`--residency-prefetch 128M --no-prefetch` on Qwen:

- Staging plan selected 95 file-backed tensors, 128.00 MiB;
  884.48 MiB skipped by the staging limit.
- Native `MADV_WILLNEED` accepted 127.72 MiB.
- Mapping RSS before/after: `0 -> 0`; no measurable residency change was
  observed. The request is a hint, not a synchronous read or a guarantee.

`ramforge support` reports residency capabilities: tensor metadata, release,
prefetch, trace, and between-operation residency are supported; graph
dependency/last-use proof is partial; intra-decode residency and GPU are
unsupported. It also lists the policy profiles, the ledger/reconciliation
surface, JSON report schema, the native lifetime contract, and the fact that
execution profiles are bounded diagnostic inputs. Profiles may be persisted
with explicit validation, but they are never execution guarantees.

### Unified ledger, profiles, and policy profiles

The runtime now has a model-scoped memory ledger
(`ramforge-runtime/src/memory_ledger.rs`) with explicit categories and
provenance. It keeps file-backed logical bytes, anonymous weights, mapping
resident bytes, mapping PSS, process RSS/peak, cgroup current/peak, staging
requested/accepted bytes, and unknown KV/scratch/runtime components distinct.
Unknown components remain `None` with a reason; they are not silently added as
zero. Reconciliation reports `RSS - mapping resident` and
`cgroup current - RSS` as signed differences, not as a single “overhead”
number. See [MEMORY_ACCOUNTING.md](docs/handoff/MEMORY_ACCOUNTING.md).

Graph manifests now feed an execution profile
(`ramforge-runtime/src/execution_profile.rs`). A capture containing multiple
graph executions is split by execution id before analysis, so transient native
node ids are never merged across graphs. Profiles store stable tensor names,
graph-order access ranges, observation counts, completeness, and invalidation
state. Incomplete, cyclic, or ambiguous captures are rejected as profile
evidence. Profiles are validated against model identity and inventory
generation before `reuse-aware` planning.

Profiles can now be persisted with `--residency-profile-save` and loaded with
`--residency-profile-load`. The file is versioned, bounded to 16 MiB and
100,000 tensor entries, written atomically, and contains only stable tensor
names/order. It does not contain transient native node IDs. Validation checks
schema, profile version, model path, model size/mtime, inventory generation,
native contract version, completeness, and invalidation. `ramforge profile`
inspects the file; adding `--model` validates model identity. A loaded profile
is used for staging order and post-operation reuse-aware planning, but it is
not proof that the next prompt will access the same tensors. See
[GRAPH_SAFETY.md](docs/handoff/GRAPH_SAFETY.md).

### Native execution-quiescence contract

ForgeCore now exposes the native contract added by the tracked patch:

- a model instance id and residency contract version;
- model active/in-flight context counts and a point-in-time quiescent flag;
- context id, execution generation, in-flight/synchronized state;
- `ResidencyError::{Busy, Unsupported, Invalid, Native}`.

A context holds a native shared lease from asynchronous graph submission until
`llama_synchronize` (or destruction) observes completion. Page release takes
the exclusive side with a non-blocking attempt and returns `EBUSY` if any
context is in flight. RAMforge binds the manager to the model instance,
observes quiescence after session drop, and marks resources
`ReleaseEligible` only at that operation boundary. Graph-node completion alone
cannot reach `ReleaseEligible`; `FinalUseProven` remains unreachable for model
weights. The tests exercise two simultaneous contexts, stale/mismatched
contracts, release rejection, context-drop synchronization, and inference
after page release.

This is a safe **between-operation physical page** contract. It is not
intra-decode eviction: a synchronized context still owns model weights for
future tokens. Context-owned KV/scratch buffers have no independent native
release API and remain unsupported.

SmolLM2-135M Q4_K_M, 272 file-backed tensors, context 256, 4 greedy tokens,
target 128 MiB, no cgroup enforcement, median of two runs unless noted:

| Policy | Selected | Mapping RSS delta | Process RSS before/after | Cgroup current delta | TTFT |
|---|---:|---:|---|---:|---:|
| off | 0 | 0 B | 164.30 / 164.30 MiB | ~0 B | 28.2 ms |
| observe | 12–13 | 0 B | 164.25 / 164.25 MiB | ~0 B | 28.4 ms |
| conservative | 0 | 0 B | 164.21 / 164.21 MiB | ~0 B | 25.5 ms |
| memory-bound | 12–13 | -38,387,712 B | 164.23 / 127.63 MiB | ~0 B | 27.0 ms |
| active | 13 | -38,748,160 B | 164.30 / 127.37 MiB | ~0 B | 27.5 ms |
| reuse-aware (valid profile) | 134 | -46,923,776 B | 172.23 / 127.48 MiB | ~0 B | 26.7 ms |

The reuse-aware result required a complete 16,384-event profile; with the
default 4,096-event cap a four-token capture was truncated, the profile was
rejected as incomplete, and the plan reported `profile_used=false`. That is
intentional. `conservative` selected nothing because this model’s aligned
tensor ranges all share pages; the policy refuses to claim those pages as
uniquely reclaimable. The native release call still protected shared pages in
the other mutating policies. A 64 MiB bounded staging plan selected 227 tensors
and requested 63.98 MiB of deduplicated page ranges; the native accepted
63.98 MiB and mapping RSS stayed at 98.87 MiB, an honest zero-change hint.

The report now includes prompt tokens, generated tokens, TTFT, decode/total
wall time, stop reason, and an output token fingerprint. It remains diagnostic:
the soft target is not a hard limit, release submission is not physical
reclamation, and cgroup usage can stay flat while mapping/process RSS falls.
Persisted profile example:

```sh
ramforge run model.gguf --backend forgecore --ram 192M --ctx 256 \
  --graph-trace --graph-trace-events 16384 --residency-policy reuse-aware \
  --residency-target 128M --residency-profile-save profile.json \
  --prompt "The capital of France is" --max-tokens 1

ramforge profile profile.json --model model.gguf --json
```

A reproducible harness is documented in
[BENCHMARKS.md](docs/handoff/BENCHMARKS.md).

### Predictive staging

`prediction.rs`, `prediction_planner.rs`, and `prediction_sim.rs` implement a
deterministic predictor/planner/coordinator. It consumes a validated execution
profile and optional feedback history, matches only stable tensor names, and
ranks predicted accesses by confidence and observed graph order.

High-confidence predictions may be staged within `--prediction-window`;
medium-confidence predictions may use only `--prediction-speculative`.
Unknown-size, anonymous, admission-rejected, and below-threshold resources are
rejected or deferred with explicit reasons. Page ranges are deduplicated before
the native prefetch call. Requested bytes, native-accepted bytes, and observed
mapping changes remain separate.

Prediction never authorizes release. The report's `prediction.plan.release_authorized`
is always false; release still uses the manager plan plus the native lifetime
gate at a post-operation boundary. `FinalUseProven` remains unreachable for
model weights.

```sh
ramforge run model.gguf --backend forgecore --ram 192M --ctx 256 \
  --no-repack --graph-trace --graph-trace-events 16384 \
  --residency-policy observe --residency-target 128M \
  --residency-profile-load profile.json \
  --prediction --prediction-min-confidence 40 \
  --prediction-window 64M --prediction-speculative 16M \
  --prediction-history-save prediction-history.json \
  --residency-report
```

SmolLM2-135M Q4_K_M, 1 token, one loaded profile, 64 predictions:

- 64 medium-confidence predictions initially; 56 selected for speculative
  prefetch after confidence/budget filtering;
- requested **16,650,240 B**, native-accepted **16,650,240 B**;
- observed mapping RSS delta **0 B** — a successful hint with no observable
  residency change;
- feedback: 64 hits, 0 misses, 0 wasted staged bytes for that trace.

With a warm feedback history, the same profile produced 64 high-confidence
predictions, requested/accepted **17,375,232 B**, and still **0 B** observed
mapping delta. These are hint/accounting measurements, not claims of physical
residency or throughput improvement.

The deterministic simulator:

```sh
ramforge simulate --budget 12K --window 8K --json
```

reports baseline vs predictive hits/misses, reread bytes, staged/wasted bytes,
peak resident bytes, zero budget violations, and `release_authorized=false`.
See [PREDICTION.md](docs/handoff/PREDICTION.md).

### GPT-NeoX-20B under 8 GiB

The 20B GGUF is 13,135,032,320 B on disk and 13,133,230,080 tensor bytes,
SHA-256 `1a35c13535c26bb925a2379d9f021de67a0390124ffe2051aa94412002652d6d`. It
was executed with:

```sh
ramforge run models/gpt-neox-20b.i1-Q4_K_M/gpt-neox-20b.i1-Q4_K_M.gguf \
  --ram 8G --enforce-ram --enforce-timeout 300 --ctx 128 --no-repack \
  --no-prefetch --prompt "The capital of France is" --max-tokens 1 \
  --temperature 0 --backend forgecore --memory-report --profile
```

Remove `--no-prefetch` for the prefetch comparison.

With the default prefetch mapping: exit 0; actual context 256; 5 prompt
tokens; output ` Paris`; wall 16.94 s; model load plus tokenization 6.64 s;
TTFT 10.03 s; model mapping 12.23 GiB with 7.65 GiB resident (62.5%); cgroup
peak exactly 8 GiB; anonymous 319.08 MiB; file-backed 7.65 GiB; kernel
36.45 MiB; 54,721 `max` events; 190 major faults; cgroup swap peak 0; max RSS
7.97 GiB.

With `--no-prefetch`, the same model, prompt, context, budget, and timeout
completed in 7.72 s wall with 0.126 s setup and 7.385 s TTFT. Process RSS after
load was 35.36 MiB; mapping residency after decode was 7.64 GiB (62.4%);
cgroup peak was exactly 8 GiB; anonymous 329.79 MiB; file-backed 7.64 GiB;
7,979 `max` events; 159 major faults; cgroup swap peak 0; max RSS 7.97 GiB;
input blocks 25,383,576. Output was the same ` Paris`.

These are one greedy token from one run each. They are evidence that a model
larger than the cap can execute with bounded charged memory; they are not
throughput, quality, or bit-exact numerical parity evidence. During the tests
the host's overall swap usage moved within roughly 3.5-5.4 GiB (other processes
under page-cache pressure); the supervised cgroup reported zero swap and no
swap configuration was changed.

### Enforcement limits

- The cgroup caps charged memory, not all host page cache. Shared file pages
  charged to another cgroup are outside this cap; mapping RSS/PSS is reported
  separately.
- The default pinned loader prefaults the whole mapping at load. Large models
  therefore pay a long load and reclaim/refault cost even when peak residency
  is bounded. `--no-prefetch` avoids the initial whole-file fault and moves the
  cost to prompt/decode demand faults; it does not reduce total I/O when the
  model is fully exercised.
- There is no per-layer execution scheduler or staging queue on the forgeCORE
  path. `--release-layer` operates on exact tensor ranges and is called
  between operations, not during decode; layer grouping uses the tested
  `blk.N.*` naming convention. The kernel's demand paging remains the per-page
  mechanism.
- `--enforce-ram` requires cgroup v2 memory delegation. If unavailable, the
  command fails with an actionable error and does not silently fall back to
  diagnostic-only mode. cgroup v1 is unsupported.
- `memory.swap.max=0` is written when the swap controller exposes it, and
  `Peak swap` is reported. If that file is absent, `memory.max` still caps
  charged RAM but swap is not additionally excluded.
- The enforced timeout path is tested: Qwen2.5-1.5B at 512M with
  `--enforce-timeout 1` exits 124, reports `Result: timeout; cgroup killed`,
  and leaves no cgroup behind.
- The 20B model was measured only at 8G. It was not run at 4G, 1G, or 100M.
- Not run in this sprint: Qwen2.5-7B, DeepSeek-Coder-V2-Lite, and
  Qwen2.5-Coder-32B. DeepSeek-Coder-V2-Lite is a 10,364,416,768 B
  `deepseek2` model that inspects/plans but is not in the validated forgeCORE
  execution registry. The 32B model was not attempted without a measured plan.
- GPU/VRAM execution remains out of scope.

## Architecture and quantization matrix

RAMforge tracks two execution backends separately. `inspect` and `plan` are
generic GGUF operations; a `yes` in one execution column is not a claim about
the other backend.

| Architecture | Aliases | Inspect | Plan | Legacy executor | forgeCORE/libllama | Real-model evidence |
|---|---|---:|---:|---|---|---|
| `llama` | — | yes | yes | yes | yes | SmolLM2-135M, Q4_K_M |
| `qwen2` | `qwen2.5` | yes | yes | yes | yes | Qwen2.5-1.5B, Q4_0 |
| `gptneox` | `gpt_neox` | yes | yes | **not-yet** | **yes** | Pythia-70M-deduped, Q8_0 |
| `mistral` | — | yes | yes | not-yet | not-yet | literal `mistral` tag is not a validated execution path; Mistral GGUFs normally declare `llama` |
| `qwen3` | — | yes | yes | not-yet | not-yet | not validated |
| `qwen35` | `qwen3.5`, `qwen3_5` | yes | yes | not-yet | not-yet | hybrid attention/SSM; intentionally not aliased to qwen2 |
| `gemma` | `gemma2`, `gemma3` | yes | yes | not-yet | not-yet | not validated |
| `phi` | `phi2`, `phi3`, `phi4` | yes | yes | not-yet | not-yet | not validated |

Quantization is also backend-specific:

| GGUF type | Legacy RAMforge executor | forgeCORE/libllama |
|---|---|---|
| `F32`, `F16`, `BF16` | yes | delegated to the pinned native runtime |
| `Q4_0`, `Q5_0`, `Q8_0` | yes | delegated; real end-to-end evidence for Q4_0 and Q8_0 |
| `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` | yes | delegated; real end-to-end evidence for Q4_K_M |
| `Q4_1`, `Q5_1`, `Q8_1`, `IQ*`, other types | no (explicit preflight error) | delegated to llama.cpp; RAMforge makes no enumeration claim |

The legacy column is backed by the compact `TensorData` variants and the
block-wise kernels in `ramforge-core/src/quant.rs`. The forgeCORE column is
deliberately not an enumeration: once a model reaches the forgeCORE backend,
llama.cpp owns tensor decoding. RAMforge only claims the type/architecture
combinations for which it has real execution evidence.

The registry is visible with:

```sh
ramforge support
ramforge inspect MODEL.gguf
ramforge plan MODEL.gguf --ram 1G
```

`inspect` and `plan` print separate `legacy-run` and `forgecore-run` states.
The legacy preflight for a GPT-NeoX model fails with an actionable message:

```text
architecture 'gptneox' is not supported by RAMforge's legacy executor, but it
is validated for execution through the forgeCORE/libllama backend:
inspection=true planning=true legacy-execution=not-yet forgecore-execution=yes
(...); re-run with `--backend forgecore`.
```

No weights are read on that path, and no backend switch is implicit.

### GPT-NeoX/Pythia evidence (2026-10-25)

Fixture: `mradermacher/pythia-70m-deduped-GGUF`,
`pythia-70m-deduped.Q8_0.gguf`, 76,722,400 B, SHA-256
`94788a58bb81431b7058837ec2371c303edb20384b615f0afabfa81539294e0d`,
GGUF v3, 6 layers, 512 embedding, 8 heads, 2048 FFN, 16 rotary dimensions,
parallel residual enabled.

With the fixture outside the repository:

```sh
ramforge run /path/to/pythia-70m.Q8_0.gguf --ram 1G \
  --prompt "The capital of France is" --max-tokens 8 \
  --temperature 0 --backend forgecore --memory-report
```

Result: exit 0, **5 prompt tokens** `[510, 5347, 273, 6181, 310]`, output
` the capital of the French Empire.`, 8 tokens, 0.14 s wall, max RSS
167,668 KiB, 1 GiB requested (this particular run did not use
`--enforce-ram`; see the enforced matrix above), logical GGUF tensor bytes
74,948,608 B, `Swaps: 0`. The native llama.cpp probe encodes the same prompt as
the same five ids and produces the same greedy prefix. A second greedy run
produces identical ids.

The tokenizer fix that made this possible: when
`tokenizer.ggml.add_bos_token` is absent, BPE/GPT-2 vocabularies now default to
**no BOS**, matching llama.cpp; SentencePiece vocabularies still default to
BOS. The previous unconditional `true` inserted token `0` into Pythia prompts.

Independent numerical check: a NumPy GPT-NeoX forward pass implemented from
the GGUF metadata and dequantized Q8_0 weights, with the same five prompt ids,
reproduces the native final-token logits:

- top-1 token id: **253** in both;
- top-10 overlap: **9/10**;
- cosine similarity: **1.000000046**;
- max absolute logit difference: **0.929** on logits of magnitude ~1,579
  (about 0.06% relative), consistent with ggml's f16 GELU lookup table and
  floating-point reduction order.

This is an independent architecture/graph check, not a claim of bit-exact
parity. It covers one 70M fixture; it is not broad model-quality validation.

The real 20B GPT-NeoX GGUF
(`models/gpt-neox-20b.i1-Q4_K_M/...`, 13,135,032,320 B, 13,133,230,080 tensor
bytes) is inspectable and plannable at 8G, 4G, 1G, and 100M. It was executed
under the 8 GiB cgroup cap described above: exit 0, 7.65 GiB of the 12.23 GiB
mapping resident at the end-of-run sample, 54,721 limit events, no OOM kill,
and one greedy token ` Paris` in 16.94 s. This supersedes the earlier inspect/plan-only status.
It is one controlled run, not a claim of acceptable throughput or quality; the
pinned loader still prefaults the whole file, and there is no per-layer staging
on the forgeCORE path. The 4G, 1G, and 100M 20B cases were not run.

## Phase 4 static device selection

Phase 4 adds a RAMforge-owned static execution boundary without changing the
legacy default. The safe ForgeCore device registry is exposed through
`DeviceRegistry::discover()`, with explicit `DeviceId`, `BackendId`,
`DeviceKind`, capability, selection, and `ExecutionConfig` types. The CLI
accepts:

```text
--backend forgecore --device cpu
--backend forgecore --device id:<device-id>
--backend forgecore --device backend:<backend-id>:<index>
```

CPU is selectable and remains first-class. Native GPU/accelerator facts and
safe free/total memory values are preserved when ForgeCore reports them.
The pinned ForgeCore revision exposes public static model-load/offload options,
but this RAMforge refresh intentionally does not enable non-CPU model
execution, scheduling, or placement policy. Non-CPU selections therefore fail
explicitly with no CPU fallback. Zero/unavailable VRAM and native allocation
values remain unknown; device selection is not residency proof. See
[`docs/handoff/PHASE_4_ENGINEERING_REPORT.md`](docs/handoff/PHASE_4_ENGINEERING_REPORT.md)
for the boundary and validation record.

## Capabilities

### Milestone 1 – GGUF Inspection
- Magic, header, metadata KV, tensor descriptors, file offsets, byte lengths

### Milestone 2 – Memory Budget & File-Backed Access
- `MemoryBudget`, `parse_memory_size()` (`8G`, `8GiB`, `8192M`, `512MiB`, etc.)
- `GgufDataSource` range reads
- `BoundedCache` LRU
- `ramforge plan` with runtime-aligned persistent/largest-layer memory preflight for executable architectures

### Milestone 3 – First Real CPU Inference
- Legacy architectures `llama`, `qwen2`; the explicit forgeCORE/libllama backend additionally executes validated `gptneox` (GPT-NeoX/Pythia)
- F32/F16/BF16, tokenizer, RMSNorm, RoPE, attention, SwiGLU, KV cache, sampling
- `ramforge run`

### Milestone 4 – Out-of-Core Layer Streaming
- Only persistent weights resident initially; layers loaded on demand → compute → release
- `ResidencyStats` proves total > budget while peak resident < total and peak managed ≤ budget

### Milestone 5 – Native Quantized Tensor Support

**Supported formats:**
- `Q4_0`: block 32, 18 bytes (2B half scale `d` + 16B packed 4-bit quants, dequant `d*(q-8)`)
- `Q5_0`: block 32, 22 bytes (2B half scale `d`, 4B fifth-bit mask, 16B packed 4-bit quants; dequant `d*(q-16)`)
- `Q8_0`: block 32, 34 bytes (2B half scale `d` + 32B int8 quants, dequant `d*q`)
- `Q4_K`: block 256, 144 bytes (2B half `d`, 2B half `dmin`, 12B scales (8 scales + 8 mins packed 6-bit), 128B 4-bit quants; unpack via `get_scale_min_k4(j)`, dequant `d*sc*q - dmin*m`)
- Since M5.6.1: `Q2_K`, `Q3_K`, `Q5_K`, `Q6_K`, `Q8_K` block layouts, `dequantize_row_*`, and `matvec_*` are also implemented in `quant.rs` with the same resident-compact representation. `Q5_0` was added after real-model validation against `SmolLM2-135M-Instruct-Q4_K_M.gguf`. `Q4_1`, `Q5_1`, `Q8_1`, `IQ*` remain unsupported.

**Representation:**
- `TensorData` enum: `F32`, `F16`, `BF16`, `Q4_0(QuantizedTensor)`, `Q5_0`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`
- `QuantizedTensor { ggml_type, shape, num_elements, raw_data: Vec<u8> }` keeps quantized compact while resident
- `resident_bytes()` = raw_data.len() (e.g. 144B for 256 values Q4_K vs 1024B F32) – shows memory saving
- Runtime quantized `matvec()` uses compact block-wise kernels and bounds dequantization to block workspaces; the storage-independent reference helper uses one reusable decoded row buffer and remains outside the runtime budget ledger.
- `get_embedding()` for quantized token_embd dequantizes only the requested row.

**Memory accounting:**
- Budget accounts actual resident representation: quantized bytes + temporary block buffers, NOT full F32 expansion.
- F32 payloads read directly into final F32 storage, so both load and settled residency are 1× file bytes; no raw tensor copy coexists.
- F16/BF16 weights: decoded to F32 in RAM at load time; the budget books the true decoded residency (4 B/elem), with a 3× file-byte transient during layer load (hardened in M6.1 – see below).
- Example: Q4_K 256 elements F32 equiv 1024B, quantized resident 144B (7.1× smaller). For model with 4 layers n_embd 256 ffn 512:
  - Quantized total 1.5MB, F32 equiv 10MB
  - Budget 800KB, total quantized 1.5MB > budget, per-layer quantized 370KB fits, peak managed 429KB ≤ budget → inference succeeds with streaming
- Persistent weights (`token_embd`, `output_norm`, `output`) use actual post-decode residency for the 25% retention policy. Resident loads charge before I/O, then settle to decoded F32 bytes or compact quantized bytes; quantized persistents remain compact.

**Layer streaming integration:**
```
Layer descriptor → GgufDataSource::read_tensor() (quantized bytes) → TensorData::Q4_K/Q8_0/Q4_0 resident → quantized matvec (block dequant) → release layer
```
Entire quantized model never resident simultaneously. Layers released after compute.

**Compute backend:**
- `ComputeBackend` trait F32 `matvec` now follows the explicit ggml layout and is wired into the inference hot path (`matvec_backend` in `streaming_model.rs`): resident F32 weights use runtime-detected AVX2 (`simd.rs`) + rayon row-parallelism; quantized weights keep the compact block-wise kernels from `quant.rs`

### Usable Out-of-Core Inference

- **Streaming output:** generation exposes a text callback backed by a stateful tokenizer decoder; split UTF-8 byte tokens are buffered until displayable, EOS remains suppressed, and CLI diagnostics stay on stderr
- **No unused terminal forward:** once the final requested token is selected and emitted, RAMforge does not stream all layers again unless another token's logits are required
- **Real profiling:** `--profile` measures generation wall time, prompt/prefill, prompt/decode forward counts, logical per-tensor reads/bytes, physical GGUF reads/bytes, actual/avoided seeks, coalesced ranges/gap overhead, read-buffer reuse/growth, the GGUF read path, layer loading/compute/release, tensor construction, explicit dequantization/copies, F32 and quantized matvec, allocations, logits, sampling, callback time, token latency, and layer-cache hits/misses/evictions/current/peak residency
- **Grouped quantized copy detail:** `grouped_quantized_copy_count`, `grouped_quantized_copy_bytes`, and `grouped_quantized_copy_time` isolate grouped quantized slice-to-owned payload copies
- **Memory visibility:** `--memory-report` labels RAMforge current/peak/budget separately from Linux process RSS and system memory; MemoryBudget does not claim control over RSS or page cache
- **Capability registry:** `ramforge support` distinguishes generic GGUF inspection/planning from execution, tokenizer, and quantization support, and reports the legacy executor and forgeCORE/libllama columns separately. Legacy execution remains limited to `llama` and `qwen2`; validated forgeCORE/libllama execution covers `llama`, `qwen2`, and `gptneox`. Mistral is runnable only when represented by a validated llama-compatible GGUF; `qwen35` is explicitly inspect/plan-only. Legacy GPT-NeoX rejection names `--backend forgecore` and never switches backends implicitly.
- **Execution memory preflight:** for runnable architectures, `ramforge plan` applies the runtime’s persistent-retention, decoded-residency, and per-tensor load-transient rules to report the largest layer and a necessary managed-memory lower bound. It explicitly excludes prompt-dependent KV and activation workspaces, so it is not presented as a sufficient runtime guarantee.
- **Bounded streamed-layer cache:** recently used decoded layer representations may remain resident under an explicit byte capacity derived from the selected budget and largest-layer headroom. Every cache entry keeps its MemoryBudget charge, LRU eviction releases it exactly, and mandatory KV/activation/output workspaces can force safe eviction. Hits avoid GGUF rereads; planner capacity is an estimate and does not guarantee hits.
- **Bounded streamed-layer read coalescing:** descriptor-sorted tensors that are adjacent or separated by at most 4 KiB may share a physical read only while the full span stays at or below 64 MiB and its grouped raw-plus-resident workspace fits MemoryBudget. Arbitrary ordering and explicit tensor boundaries remain authoritative; large gaps/spans and low-headroom cases use the original individual read path. No mmap or persistent raw payload is involved.
- **Cursor-aware retained handle:** physical reads reuse the synchronized GGUF handle’s known absolute cursor. A seek is skipped only when the preceding successful read ended exactly at the next requested offset; failures invalidate the cursor and force the next seek.
- **Per-layer grouped-buffer reuse:** when a layer has multiple fully coalesced ranges and the complete scratch-plus-resident peak fits MemoryBudget, one buffer sized to the largest range is charged and reused for all ranges. If the stronger lifetime cannot fit, the existing per-range grouped path remains active; no viable model is rejected solely for this optimization.

### Real-model reference validation

A Mistral 7B Q4_K_M GGUF (about 4.37 GB of weights, represented as llama-compatible GGUF) completed a one-token out-of-core run with a 4 GiB RAMforge budget. This is a single environment-specific diagnostic result, not a performance guarantee: generation took about 409 s, read 11.70 GiB over 864 GGUF reads, loaded/released 96 layers, spent about 125 s in the GGUF read path and 282 s in layer compute, and peaked near 306.6 MiB of RAMforge-managed memory (about 189 MiB process RSS at report time). Prompt/prefill took about 272 s. The emitted text was `ikt`; execution and streaming output are validated, but model-output quality is not yet validated. Profile categories include documented overlapping subsets and must not be summed as an exclusive breakdown.

### Milestone 7.2 – Fast Float Load

- **Direct F32 I/O:** full tensors and streamed F32 ranges are read into final `Vec<f32>` storage; little-endian hosts avoid both the raw-byte intermediate and per-element decode loop
- **No destination prefill:** raw range reads append into reserved spare capacity, while direct F32 reads initialize reserved F32 storage with exact I/O; short reads remain errors
- **Endianness:** GGUF little-endian bits are already native on little-endian hosts; big-endian hosts perform an in-place normalization without a second tensor-sized allocation
- **Accounting:** F32 load charge is exactly its one owned final representation (1× file bytes); F16/BF16 stay at 3× transient and 4 B/element settled; quantized loading remains 1× compact bytes
- **Safety boundary:** the only new unsafe operations are the audited conversion of uninitialized, aligned F32 spare capacity to an exact byte destination and `set_len` after successful `read_exact`

### Milestone 7.1 – Accounting Hardening

- **Persistent startup ordering:** resident persistent weights reserve the full load transient before reading; M7.1 used F32 2×, F16/BF16 3×, and compact quantized tensors 1× (M7.2 reduces only F32 to 1×)
- **Atomic settlement:** `MemoryBudget::resize` settles a live transient charge to exact `TensorData::resident_bytes()` without an uncharged gap; read/decode/settlement failures remove the current charge, and model-startup failure rolls back earlier persistent charges
- **Resident policy:** the existing 25% threshold is applied to the representation RAMforge will actually retain (decoded F32 for F32/F16/BF16, raw compact bytes for supported quants), with checked size arithmetic
- **Hidden-state lifetime:** `forward_single_streaming` writes into a caller-owned output slice; `generate()` keeps `tmp:hidden` live for that buffer's complete lifetime instead of returning a vector past `tmp:forward`

### Milestone 6.1 – Correctness & Accounting Fixes

- **KV cache lifecycle:** repeated `generate()` calls on one engine work; failed generations release every charge they made (`clear_kv_cache()`); no stale `"kv_cache"` allocations
- **RoPE:** half-split `(x[j], x[j+head_dim/2])` convention with `theta_j = pos * base^(-2j/head_dim)` — the true llama/qwen2 rotation (was interleaved/GPT-J pairs)
- **F16/BF16 accounting:** decoded-F32 residency (4 B/elem) is what the budget books for persistent weights and settled layer charges
- **Q/K/V biases (qwen2):** loaded + charged, all-or-none validation, bias added after projection matvec (before RoPE), released with the layer; partial sets and shape mismatches are hard errors

### Milestone 6 – True Out-of-Core Integrity

- **Memory accounting:** `MemoryBudget::with_temp(name, bytes, f)` is the scoped accounting helper for transient working sets (`tmp:hidden`, `tmp:forward`, `tmp:embd_row`, `tmp:streamed_matvec`, `tmp:logits`, `tmp:sampling`) – released on success and on error. Layer tensors charge *before* reading (peak = settled prefix + per-tensor transient) and settle to exact resident bytes after construction; a failed layer load releases all its charges.
- **Cache:** `BoundedCache::insert_budgeted` charges each cached entry to the budget (`cache:{key}`), evicting LRU entries to make budget room; if nothing can be evicted, the entry is simply not cached (streaming keeps working) instead of failing or double-counting.
- **Matrix layout:** one explicit GGML/GGUF convention everywhere: `shape = [in, out]`, buffer row-major `[out][in]`, `y[o] = Σ_i W[o·in+i]·x[i]`. No orientation heuristics, no transpose fallbacks, no full-F32 dequantization of 2D weights – arity mismatches are hard errors.
- **Output projection:** single caller-owned logits buffer per `generate()` call; streamed (non-resident) output/embedding matrices are projected in budget-bounded row chunks (`min(16 MiB, available/4)`, ≥ 1 row) with per-row block decode.
- **KV / attention:** attention reads the KV history in place (no per-token prefix copies); the KV cache starts at the prompt length and grows in 256-token chunks capped at prompt+max_tokens, budget-checked with rollback on failure. No KV quantization.
- **Legacy removal:** the pre-M4 fully-resident F32 model loader (`LlamaModel`) was deleted – it violated budget integrity, guessed orientation, and duplicated the KV prefix.

## Build & Test

`scripts/test.sh` resolves `FORGE_LLAMA_DIR` (defaulting to the sibling
`native/` tree) and supplies the process-local `LD_LIBRARY_PATH` needed by the
shared llama.cpp/ggml libraries. When
`$FORGE_LLAMA_DIR/models/tiny-llama.gguf` exists, it also exports
`FORGE_TEST_MODEL` so the real-adapter tests run instead of skipping. It does
not modify system configuration.

```bash
cargo fmt --all -- --check          # known pre-existing diffs; see PROJECT_STATE.md
cargo check --workspace
scripts/test.sh --workspace --locked
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

## Usage

Inspect (shows quantized types):
```bash
cargo run -p ramforge-cli -- inspect model.gguf
# Quantization summary: Q4_K: 169 tensors, F16: 121, etc.
```

Plan:
```bash
cargo run -p ramforge-cli -- plan model.gguf --ram 8G
```

Run with quantized model:
```bash
cargo run -p ramforge-cli -- run model.gguf --ram 8G --prompt "Hello" --max-tokens 32
cargo run -p ramforge-cli -- run model.gguf --ram 1G --prompt "Hello" --max-tokens 16 --verbose
cargo run -p ramforge-cli -- run model.gguf --ram 4G --prompt "Hi" --max-tokens 1 --profile --memory-report
cargo run -p ramforge-cli -- support
# Verbose shows:
# Total model weight bytes: 1500160 (1.43 MiB) quantized
# F32 equiv: 10507264 (10 MiB)
# Peak resident layer bytes: 370688 (0.35 MiB)
# Peak managed bytes: 429056 / budget 819200
# Fits check: total > budget ? true
```

Accepted `--ram` syntax: `8G`, `8GiB`, `8192M`, `512MiB`, `1.5G`, `KB`/`KiB`/`MB`/`MiB`/`GB`/`GiB`.

`MODEL` may be a single GGUF file, any shard of a standard split GGUF set, or a directory containing exactly one model. Standard `*-00001-of-0000N.gguf` shards are grouped and validated; multiple independent models in one directory are rejected instead of guessing. `inspect` reports shard count, per-shard sizes/tensor counts, total file bytes, and merged tensor types.

Chat templates declared by the GGUF are opt-in:

```bash
cargo run -p ramforge-cli -- run models/qwen2.5-1.5b-instruct-q4_0 \
  --ram 1G --prompt "What is 2+2? Answer briefly." --chat --max-tokens 16
cargo run -p ramforge-cli -- run models/qwen2.5-1.5b-instruct-q4_0 \
  --ram 1G --prompt "Explain streaming." --chat --system "Answer in one sentence."
```

Without `--chat`, the prompt is encoded as raw text, preserving the previous behavior.

Diagnostics/profile data go to stderr; `Assistant:` text streams to stdout as tokens become decodable.

**Out-of-core quantized example:**
```bash
# Synthetic Q4_K model: 4 layers, n_embd 256, ffn 512
# Total quantized 1500160 bytes (1.43 MiB), F32 equiv 10507264 (10 MiB), budget 800K
cargo run -p ramforge-cli -- run synthetic_q4k.gguf --ram 800K --prompt "hello" --max-tokens 3 --verbose
# Proves:
# total quantized > budget
# quantized resident < F32 equiv
# peak layer < total
# peak managed <= budget
# inference succeeds
```

## Project Structure

```
crates/
  ramforge-core/
    gguf.rs, model.rs, types.rs
    memory.rs – MemoryBudget, parse_memory_size
    cache.rs – BoundedCache LRU
    datasource.rs – GgufDataSource, exact range reads, split-shard virtual offsets, direct little-endian F32 reads into final storage
    model_set.rs – split-GGUF discovery, validation, merged model view
    chat_template.rs – conservative GGUF chat-template parser/renderer
    tokenizer.rs – Tokenizer from GGUF, control-token-aware encoding
    quant.rs – Q4_0, Q8_0, Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K block layouts, dequant, quantized matvec (scalar, block-wise)
    tensor.rs – TensorData (F32/F16/BF16/Q4_0/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K), direct F32 ownership, QuantizedTensor, resident_bytes, matvec, get_embedding
  ramforge-runtime/
    accounting.rs – shared tensor load-transient formulas used by execution and planning
    backend.rs – ComputeBackend, CpuBackend (rayon threading, optional AVX2 for F32)
    ops.rs – RoPE, attention
    kv_cache.rs – KV cache explicit, budget-accounted
    layer.rs – LayerDescriptor grouping
    layer_cache.rs – bounded, budget-charged LRU cache for decoded streamed layers
    layer_read.rs – descriptor-only bounded coalesced read planning
    residency.rs – ResidencyStats
    profile.rs – optional generation timing/counter collector
    memory_report.rs – managed memory, Linux RSS, and system-memory visibility
    support.rs – architecture/tokenizer/quantization capability registry
    persistent.rs – PersistentWeight: actual resident representation ≤25% of budget, else streamed on demand
    simd.rs – AVX2/FMA F32 dot/matvec kernels, runtime detection + scalar fallback (M5.6.1)
    model.rs – LlamaConfig, validate_required_tensors
    streaming_model.rs – StreamingLlamaModel (charge-before-read + direct F32 loads), load/release layer, caller-owned final hidden + scoped tmp:forward
    inference.rs – InferenceEngine (file-backed + budget + chunk-growing KV + single logits buffer), generate()
    plan.rs – file-size planning plus executable-architecture layer-memory preflight
    sampling.rs – greedy, temperature, top-k/p
  ramforge-cli/ – inspect, plan, support, run --verbose/--profile/--memory-report
```

## Supported / Unsupported

**Legacy executor architectures:** `llama`, `qwen2` (dense). Mistral works only when represented by a validated llama-compatible GGUF architecture/tensor layout. `qwen3`, `qwen35`, `gemma`, and `phi` are inspect/plan-only registry entries.

**Validated forgeCORE/libllama architectures:** `llama`, `qwen2`, and `gptneox` (GPT-NeoX/Pythia). `gptneox` is intentionally not implemented by the legacy executor; use `--backend forgecore`. Other architectures are delegated to llama.cpp only without a RAMforge validation claim.

**Legacy tensor types:** `F32`, `F16`, `BF16`, `Q4_0`, `Q5_0`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` – all usable by the legacy executor

**forgeCORE/libllama tensor types:** delegated to the pinned native runtime; RAMforge has end-to-end real-model evidence for `Q4_0`, `Q4_K_M`, and `Q8_0` but does not claim a complete native type enumeration.

**Unsupported (clear error):**
- Legacy execution of `gptneox` → actionable error naming `--backend forgecore`; no implicit fallback
- Unsupported execution architectures → capability-aware error listing detected architecture, inspect/plan status for both backends, and legacy-supported architectures
- Other quantized types (Q4_1, Q5_1, Q8_1, IQ*, …) for the legacy executor → "unsupported tensor type for inference"
- Missing tensors → "missing tensor 'blk.0.attn_q.weight'"
- Budget too small → "RAM budget too small for layer..."

**CPU-only:** No GPU. The pinned native build is CPU-only; the forgeCORE adapter binds CPU.

## Testing

- Quantization: block size, byte size, scale handling, signed/unsigned, dequant values, truncated rejection, invalid size rejection (in `quant.rs`)
- Matvec: tiny known matrix + known vector vs expected F32 and vs reference dequant + F32 matvec, tolerance 1e-3 (in `quant.rs`)
- Layer grouping, loading, release, memory accounting, peak residency (in `layer.rs`, `streaming_model.rs`)
- Quantized layer loading: Q4_0 model, matvec zeros (in `streaming_model.rs`)
- Out-of-core F32: total > budget while inference succeeds (in `streaming_model.rs`, `inference.rs`)
- Out-of-core quantized: synthetic Q4_K model 1.5MB > 800KB budget, per-layer 370KB fits, peak managed ≤ budget, inference succeeds, quantized resident < F32 equiv (manual run + unit tests)
- Deterministic generation: tiny F32 model greedy 5 tokens deterministic (in `inference.rs`)
- Existing F32/F16/BF16 inference still works
- Direct F32 loading: ordinary-value parity, exact edge bit patterns (signed zero, finite, subnormal, infinities, NaN payloads), non-square GGML layout, exact 1× accounting, and short-read rollback
- Existing Milestone 4 streaming tests still pass
- M6 integrity proofs: scoped temp release on success/error (`memory.rs`), budgeted cache inserts/evictions (`cache.rs`), explicit ggml layout incl. non-square Q4_0/Q4_K/F32/F16 anchors and arity-error rejections (`tensor.rs`, `backend.rs`), chunked streamed output projection + too-small-budget failure (`persistent.rs`), no-copy attention vs naive reference (`ops.rs`), chunk-growing KV preserving data with exact bytes (`kv_cache.rs`), end-to-end out-of-core inference with model > budget (`inference.rs`)

The current workspace run (`cargo test --workspace --locked`) passes **509 tests, 0 failed, 18 ignored**. The ignored tests include the real-model Qwen2.5 diagnostic, the quantized microbenchmark, the opt-in GPT-NeoX integration test, cgroup enforcement, CLI enforcement, graph-trace, bounded prefetch, and the opt-in observe/active/JSON/reuse-aware/profile/prediction residency checks. The cgroup and CLI enforcement tests require explicit `--ignored` execution, cgroup v2 delegation, real models, and libggml; fixture-gated tests report SKIP when their environment variables are unset. Run `scripts/test.sh --workspace --locked` to reproduce the standard invocation.

## Real-model smoke validation

`scripts/smoke-real-models.sh` accepts a single GGUF file, a split GGUF shard, or a model directory containing one model's shard set. Model files remain read-only. Each case runs in a separate process, waits for exit, checks that `MemAvailable` is above a configurable threshold (4 GiB by default), and records the exact command, exit status, `/usr/bin/time -v` facts, per-file model sizes/sha256 hashes, system memory and swap samples, GPU state, and captured output. Results are written as TSV and JSON outside the repository.

```sh
cd RAMforge
scripts/smoke-real-models.sh              # four 4 GiB smoke cases
scripts/smoke-real-models.sh --no-build   # reuse target/debug/ramforge
scripts/smoke-real-models.sh --matrix --include-7b --timeout 900
RAMFORGE_GPTNEOX_MODEL=/path/to/pythia.gguf \
  scripts/smoke-real-models.sh --include-gptneox --matrix
```

The latest opt-in matrix uses exact byte budgets `8G` = 8,589,934,592 B, `4G` = 4,294,967,296 B, `1G` = 1,073,741,824 B, and `100M` = 104,857,600 B. It keeps the prompt (`Hello`), generation length (8 tokens), greedy sampling, and backend fixed within each comparison. A controlled budget rejection is recorded without failing the suite; timeouts and signals still fail. Evidence: `/tmp/ramforge-real-model-smoke/20261025-002055`.

| Model / case | Backend | Budget | Exit | Wall | Max RSS | Managed peak | Evictions |
|---|---|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M Q4_K_M | legacy | 8G | 0 | 1.22 s | 132.6 MiB | 99.19 MiB | 0 |
| SmolLM2-135M Q4_K_M | legacy | 4G | 0 | 1.16 s | 132.4 MiB | 99.19 MiB | 0 |
| SmolLM2-135M Q4_K_M | legacy | 1G | 0 | 1.26 s | 132.5 MiB | 99.19 MiB | 0 |
| SmolLM2-135M Q4_K_M | legacy | 100M | 0 | 1.19 s | 111.3 MiB | 70.50 MiB | 0 |
| Qwen2.5-1.5B Q4_0 | legacy | 8G | 0 | 3.24 s | 1121.8 MiB | 1012.05 MiB | 0 |
| Qwen2.5-1.5B Q4_0 | legacy | 4G | 0 | 3.26 s | 1121.8 MiB | 1012.05 MiB | 0 |
| Qwen2.5-1.5B Q4_0 | legacy | 1G | 0 | 3.82 s | 1099.0 MiB | 1012.49 MiB | 197 |
| Qwen2.5-1.5B Q4_0 | legacy | 100M | 0 | 6.23 s | 162.5 MiB | 76.70 MiB | 222 |
| SmolLM2-135M Q4_K_M | forgecore | 4G | 0 | 0.20 s | 361.7 MiB | not exposed | n/a |
| Qwen2.5-1.5B Q4_0 | forgecore | 4G | 0 | 1.31 s | 2773.2 MiB | not exposed | n/a |
| Pythia-70M-deduped Q8_0 (GPT-NeoX) | forgecore | 8G | 0 | 0.14 s | 163.7 MiB | not exposed | n/a |
| Pythia-70M-deduped Q8_0 (GPT-NeoX) | forgecore | 4G | 0 | 0.13 s | 163.6 MiB | not exposed | n/a |
| Pythia-70M-deduped Q8_0 (GPT-NeoX) | forgecore | 1G | 0 | 0.13 s | 163.7 MiB | not exposed | n/a |
| Pythia-70M-deduped Q8_0 (GPT-NeoX) | forgecore | 100M | 0 | 0.13 s | 163.6 MiB | not exposed | n/a |
| Pythia-70M-deduped Q8_0 (GPT-NeoX) | legacy | 1G | 1 | 0.01 s | 23.1 MiB | not exposed | n/a |

- Every legacy comparison generated byte-identical stdout at all four budgets: SmolLM2 `da6b6d7d…`, Qwen2.5 `209447e5…`. The 100M Qwen run shows the budget pressure mechanism: 222 layer-cache evictions and a 76.70 MiB managed peak under 100.00 MiB. The 1G run used 197 evictions; the 4G/8G runs used none.
- The earlier GPT-NeoX forgecore runs generated byte-identical stdout (`c69c103d…`) at 8G, 4G, 1G, and 100M **without** `--enforce-ram`. That was budget-invariance, not out-of-core execution. The new enforced matrix above uses `--enforce-ram` and shows real reclaim at 1G/512M/256M.
- The GPT-NeoX legacy case is a controlled architecture rejection, classified as `unsupported-architecture` with exit 1 in 0.01 s. No weights were loaded, no OOM occurred, and no fallback happened.
- The 20B `gptneox` GGUF (13,135,032,320 B file, 13,133,230,080 tensor bytes) was executed at 8G under `--enforce-ram --no-repack`: exit 0, 7.65 GiB resident of a 12.23 GiB mapping, 54,721 limit events, no OOM kill, one token ` Paris` in 16.94 s. The 4G/1G/100M 20B cases were not run.
- Previous evidence (not re-run in this matrix): Qwen2.5-7B Q4_K_M as a two-shard set through the legacy backend at 4G, exit 0, 5:18.67 wall, 4179.6 MiB max RSS, 4086.50 MiB managed peak, 24 cached layers, 88 evictions. That result exercises split discovery and cross-shard reads, not GPT-NeoX or forgeCORE streaming.
- ForgeCore `Swaps` was 0 in every case. Its RSS includes the loader, page cache, Rust heap, and native allocations; it is not model residency. The native build is CPU-only, so none of these runs is GPU validation.
- Model inputs are unchanged; per-file sizes and SHA-256 values are recorded in `model-inventory.tsv` and each case's `model-files.tsv`.

### Chat-template evidence

`ramforge run --chat` reads `tokenizer.chat_template` from the GGUF and renders a single user message with an optional `--system` message. The interpreter is deliberately conservative: it supports the `if`/`elif`/`else`, `for`, `set`, member/index access, concatenation, comparisons, `and`/`or`/`not`, `is defined`, `tojson`, and whitespace-trimming constructs needed by the real templates tested here. Unsupported constructs fail explicitly; there is no silent model-specific fallback.

```sh
ramforge run models/qwen2.5-1.5b-instruct-q4_0 --ram 1G \
  --prompt "What is 2+2? Answer briefly." --chat --max-tokens 16 \
  --temperature 0 --verbose --memory-report
```

Result: exit 0, rendered 177-byte Qwen ChatML prompt, generated 2 tokens then EOS, output ` 4` (correct), 1:07.21 wall, max RSS 1,129,396 KiB, peak managed 1,064,027,136 B under 1 GiB, 1008 layer loads, 981 evictions. The same model with the raw non-chat prompt previously produced non-answer text; chat formatting is therefore quality evidence for this prompt. SmolLM2 also rendered its own ChatML template successfully, but its small-model output was not coherent.

A Qwen2.5-7B `--chat` run at 4G/900 s was stopped by the harness before emitting a token; prompt processing of the 177-byte ChatML prompt is too slow in the current scalar quantized path. That timeout is a performance limitation, not quality evidence, and it is not counted as a successful 7B chat validation.

### Budget-pressure eviction evidence

A separate Qwen2.5 legacy run used a budget below the model's managed footprint:

```sh
ramforge run models/qwen2.5-1.5b-instruct-q4_0.gguf \
  --ram 768M --prompt Hello --backend legacy --max-tokens 8 \
  --temperature 0 --memory-report --profile --verbose
```

Result: exit 0, 8 tokens, 18.36 s, max RSS 869,312 KiB, peak managed
798,260,928 B (761.28 MiB) under a 805,306,368 B (768 MiB) budget, peak
resident layer 26,341,376 B (25.12 MiB), 224 layer loads, 207 layer-cache
evictions, and `Swaps: 0`. This demonstrates budget-constrained eviction, not
control of total process RSS: RSS exceeded the budget because it includes
unmanaged loader, allocator, page-cache, and workspace memory.

## Known Limitations

- Current Q4_K_M CPU inference is functional but extremely slow in the measured NAS-backed Mistral run; profiling identifies both GGUF reads and scalar quantized matvec/dequantization as major costs. Performance optimization remains measurement-driven.
- The layer cache is generation-local and strict LRU. If its complete-layer capacity is smaller than a sequential full-model working set, scan-pattern thrashing can produce few or no hits; planner capacity never guarantees hit rate.
- Read coalescing is opportunistic and bounded. Large tensors/gaps, layouts exceeding the 64 MiB span cap, or insufficient grouped-buffer headroom retain one physical read per tensor; fewer reads do not guarantee lower NAS wall time until benchmarked.
- Real-model execution and streaming are validated. Qwen2.5-1.5B produced a correct chat-templated answer (`4`) for a simple arithmetic prompt; broad numerical-quality validation across prompts and models is still incomplete.
- GPT-NeoX/Pythia is validated through forgeCORE on one 70M Q8_0 fixture (`mradermacher/pythia-70m-deduped`), including native/tokenizer parity and an independent NumPy logits check. It is not a broad GPT-NeoX quality claim. The 20B GPT-NeoX GGUF executed once at 8G under the enforced cgroup cap (one greedy token); that is a fit/mechanism result, not broad quality evidence.
- The 20B GPT-NeoX result still relies on OS demand paging, not RAMforge layer staging. The pinned loader prefaults the whole file, so load time and refault cost remain high, and the 20B model was not tested at 4G, 1G, or 100M.
- `--ram` alone is a diagnostic/planning value. `--enforce-ram` adds a cgroup v2 `memory.max` hard cap; without it, identical GPT-NeoX output at 100M and 8G is evidence of non-enforcement, not out-of-core execution. The enforced matrix above shows the difference.
- Chat templates are interpreted by a conservative Jinja subset, not a full Jinja2 engine. Unsupported filters/tests/constructs fail explicitly. `--chat` is opt-in and applies to one user message per invocation.
- Qwen2.5-7B processed real split-shard inference at 4G, but its prompt processing is very slow; a chat-templated 7B run exceeded a 900 s harness timeout before emitting a token.
- Legacy quantized inference is limited to Q4_0, Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K; Q4_1/Q5_1/Q8_1 and IQ* quants are not supported by the legacy executor. The forgeCORE backend delegates tensor decoding to llama.cpp.
- CPU-only: quantized matvec is scalar (block-wise dequant); F32 dot/matvec have runtime-detected AVX2 kernels and rayon row-parallelism; no GPU
- Tokenizer: SentencePiece unigram (score-based Viterbi) and BPE (gpt2/qwen2 merges); other pre-tokenizers/model families untested. Without explicit metadata, BPE defaults to no BOS and SentencePiece to BOS, matching llama.cpp.
- Persistent weights (`token_embd`, `output_norm`, `output`): resident when their actual retained representation is at most 25% of budget, otherwise streamed on demand with budget-charged bounded temps (`persistent.rs`)
- KV cache F32, no quantization, no eviction; grows chunk-wise up to prompt+max_tokens
- Minimum practical: one streamed layer plus its charge-before-read transient (quantized 1× file bytes, direct F32 1×, F16/BF16 3×) and the forward working set must fit the budget; one direct F32 output row, or one raw converted/quantized row plus its F32 decode buffer, must fit as well
- Not budget-tracked by design (documented as out of scope): tokenizer vocabulary table, thread stacks, allocator fragmentation, residency bookkeeping, and optional profiling metadata (including O(tensors) read counters)

## What is NOT Implemented

- GPU, prefetch, double buffering, async I/O
- HTTP server, MoE, speculative decoding, model downloading
- Additional quantization formats beyond Q4_0/Q5_0/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K; no SIMD kernels for quantized matvec (F32 AVX2 only)

## License

MIT OR Apache-2.0
