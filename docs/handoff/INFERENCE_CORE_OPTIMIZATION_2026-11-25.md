# Inference Execution Core Optimization — 2026-11-25

This milestone changed the execution core rather than only documenting it. It
implements DeepSeek2 execution support, truthful context configuration, KV
cache dtype forwarding and accounting, a CLI threading fix, combined static
offload plus dynamic staging, session reuse in benchmarks, and per-generation
storage/fault instrumentation. It also corrects two earlier capacity figures
that were based on a truncated tensor sample.

All numbers below are from this machine: Intel i5-11400H (6 physical / 12
logical), 15 GiB RAM, RTX 3050 Laptop 4 GiB, driver 620.70.06, local NVMe.
The host was a shared desktop with background load; decode results are medians
with run-to-run variance unless stated otherwise.

## What changed

| Area | Before | After |
|---|---|---|
| DeepSeek2 MoE | `forgecore_execution_supported=false` | Registry entry, real 24-token chat execution, opt-in real-model test |
| `run` threads | `n_threads = 1` hard-coded | Physical-core default, explicit `--threads` |
| `run` GPU placement | No `--gpu-layers`; `--device auto` rejected | `--gpu-layers` and `auto`/`gpu`/`cuda` device resolution |
| Context options | ctx/threads/trace only | KV dtype K/V, flash attention, batch, micro-batch, sequences, KV/op offload |
| KV accounting | F16 assumption only | Exact block accounting for F32/F16/Q8_0/Q4_0, independent K/V, native-allocation match |
| Plan truthfulness | `ubatch=context`, KV "f16", no forwarding | Native default batch, real ubatch, requested vs forwarded vs effective statuses |
| Static offload + staging | Rejected unless CPU plan | Combined safely; static GPU layers excluded from staging candidates |
| Benchmark session | New context and re-staging every run | One context reused; KV cleared between runs |
| Instrumentation | RSS only | Per-run storage reads, `rchar`, syscalls, major/minor faults; setup I/O |
| MoE accounting | Not present | Routed/shared/other tensor bytes and active-expert estimate |

## 1. DeepSeek2 execution

The native backend already implemented `deepseek2`; RAMforge's registry did
not. The registry now declares it inspectable, plannable, and forgeCORE-only
(`legacy_run = not-yet`, `forgecore_run = supported`).

Real-model validation used
`DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` (27 layers, 64 experts, 6 used,
15.7B parameters, 10.36 GB).

CLI chat execution, 6 threads, context 256, 24 generated tokens:

```
Assistant:  Certainly! Below is a simple Python function that adds two numbers:
def add_numbers(a
```

Greedy output was identical across F16, Q8_0, and Q4_0 KV caches for the first
four tokens. Service-path benchmark, context 512, batch 256, Q8_0 KV, 6
threads, 32 tokens per run:

| Run | Generated | Decode tok/s | Storage read | RSS |
|---:|---:|---:|---:|---:|
| 0 | 32 | 12.18 | 422.0 MB | 9.43 GB |
| 1 | 32 | 13.74 | 1.8 MB | 9.43 GB |

Median generation 12.27 tok/s, median decode 13.74 tok/s, TTFT 564 ms.

**Correction to the previous audit.** The earlier 2.9 GB peak-RSS and 7.58 GB
expert-byte figures came from a 4-token probe and a 300-tensor inspection
sample. Across the full file, routed-expert tensors are **9.53 GB of 10.36 GB
(92%)**, and a 32-token run touches enough of the mapping to reach **9.43 GB
RSS**. The low short-run RSS was not steady state. This strengthens the case
for expert-aware residency while removing the claim that the current runtime
keeps the model near 3 GB.

## 2. CLI threading and execution-path consistency

`run --backend forgecore` previously forwarded `n_threads = 1` while the
service path forwarded the plan value. `run` now resolves `0` to physical cores
and accepts explicit `--threads`.

DeepSeek chat, 12 generated tokens, same prompt and model, wall clock:

| Configuration | Wall |
|---|---:|
| Pre-milestone binary (hard-coded 1 thread) | 14.41 s |
| New, `--threads 1` (warm cache) | 7.93 s |
| New, `--threads 6` | 2.86 s |

The 6-thread run is 2.77x faster than the new 1-thread run and produced the
same greedy continuation. The pre-milestone number also includes cold page
cache, so only the 1-vs-6 comparison is controlled.

`run` also gained `--gpu-layers` and `auto`/`gpu`/`cuda` device resolution.
Qwen2.5-7B with `--device auto --gpu-layers 8` offloaded 8/29 layers and
produced the same greedy sentence as the CPU run.

When `--ctx` is omitted, `run` now defaults to `min(training context, 4096)`
and prints the choice. Previously it requested the full training context, which
for a 7B model at 131k would allocate 7.52 GB of F16 KV and could fail before
generation. `--ctx` remains the explicit override.

## 3. Context configuration and KV cache

`ForgeCoreContextOptions` now forwards:

- K and V cache dtypes: `f32`, `f16`, `q8_0`, `q4_0`
- Flash attention: `auto`, `on`, `off`
- Logical batch, micro-batch, sequence count
- KV-buffer and operator offload
- Batch thread count
- Existing context length and trace options

Validation rejects zero sizes, micro-batch larger than batch, sequence counts
outside 1..=256, and a quantized V cache with flash attention explicitly off.
The adapter reports native readback for context, batch, micro-batch, sequence,
and thread values. KV dtypes have no native getter at the pin, so they are
reported as forwarded, not independently verified.

### 3.1 Native allocation match (Qwen2.5-7B, 16,384 context)

| KV type | Plan estimate | Native CPU KV | Decode tok/s | RSS |
|---|---:|---:|---:|---:|
| F16 | 939,524,096 B | 939.5 MB | 7.28 | 5.66 GB |
| Q8_0 | 499,122,176 B | 499.1 MB | 7.18 | 5.78 GB |
| Q4_0 | 264,241,152 B | 264.2 MB | 7.58 | 5.55 GB |

The estimate and the live native allocation match exactly for this standard
GQA model. Q8/Q4 decode throughput is within measurement noise of F16 at short
context; the benefit is capacity, not speed.

### 3.2 Long context

Qwen2.5-7B with 8 static GPU layers, KV kept in host RAM, Q4_0 KV, and
131,072-token context opened successfully:

- Native CPU KV: **2,113.9 MB** (plan estimate 2,113,929,216 B)
- RSS: **5.99 GB**
- TTFT: 586 ms for a 5-token prompt
- 8/29 layers offloaded; no KV offload

F16 KV at the same context is 7.52 GB by exact accounting, which does not fit
alongside the model on this host. The F16 case was not run because it would
require destructive memory pressure; the arithmetic is exact for this model.

**MLA caveat.** DeepSeek2 is an MLA architecture. At 4,096 context and Q8_0
KV, the metadata formula gives 481.3 MB while the native context reports
601.6 MB (1.25x). MLA stores additional compressed/rope state. The plan marks
DeepSeek KV as a lower bound and recommends native readback. Qwen2/llama GQA
models do not have this discrepancy.

## 4. Static offload plus dynamic staging

Dynamic staging was previously rejected for any GPU/hybrid plan. The service
now allows it and filters staging candidates to CPU-resident layers. A layer
already offloaded by static placement is rejected explicitly instead of being
duplicated.

Qwen2.5-7B, context 256, 6 threads, 8 generated tokens, 3 runs after 1 warmup:

| Mode | GPU layers | Staged | Generation tok/s | Decode tok/s | TTFT | RSS |
|---|---:|---:|---:|---:|---:|---:|
| CPU + staging | 0 | 8 | 7.05 | 7.83 | 241 ms | 5.30 GB |
| Hybrid, no staging | 8 | 0 | 8.08 | 9.16 | 222 ms | 3.85 GB |
| Hybrid + staging | 8 | 8 | **10.21** | **11.30** | 162 ms | 3.85 GB |

Combined placement improves decode by 44% over CPU+staging and 23% over
hybrid-only in this run. A later run of the same hybrid+staging configuration
measured 14.80 decode tok/s, so the absolute value is condition-dependent; the
ordering (hybrid+staging > hybrid-only > CPU+staging) was stable. RSS also
drops because the statically offloaded layers no longer occupy the CPU mapping.
Sequential runs on a shared desktop should not be treated as precise
treatment effects of less than ~10%.

Transition validation with the same hybrid plan (`staging-cycle`, 3 cycles,
2 tokens per cycle):

| Cycle | Read | Transfer | Promoted | Total |
|---:|---:|---:|---|---:|
| 0 | 280.2 MB | 280.2 MB | — | 816.1 ms |
| 1 | 0.0 MB | 0.0 MB | 0, 1 | 655.6 ms |
| 2 | 0.0 MB | 0.0 MB | 2, 3 | 645.0 ms |

Prefetch hit rate 1000‰, 578 MB served from standby, 0 wasted. Graph
invalidation, delta commit, standby promotion, and generation all succeeded
with static GPU layers present.

## 5. Session reuse and measured data reuse

`run_benchmark` previously called `create_session()` for every warmup and
measured run. That reopened the context and re-staged configured layers every
time. The benchmark now creates the context once and reuses it; `generate`
still clears the KV cache for repeatability.

Qwen2.5-7B, 8 staged layers, context 256, 6 threads, 3 measured runs:

| Implementation | Context setup | Contexts created | Generation tok/s |
|---|---:|---:|---:|
| Pre-milestone | 945.97 / 955.49 / 982.57 ms per run | 3 | 7.55 |
| New | 909.1 ms once | 1 | 6.95 |

Setup cost falls from ~2.87 s to ~0.91 s over three runs (−68%). Decode
throughput is within the observed run-to-run range (per-run 6.43-7.23 vs
7.09-7.98 tok/s); this change is about avoiding repeated setup and data
movement, not about raising steady decode speed.

Each Qwen staged layer is 131,135,488 B. Reopening the context re-stages eight
layers, about **1.05 GB of host reads plus H2D transfers per run**. Reuse
removes two of three such cycles, about **2.1 GB of repeated staging traffic**
over this benchmark. Because the file was page-cache hot, most of those reads
were cache hits rather than new storage I/O; the avoided H2D transfers and
allocation work are real regardless.

The new report exposes per-run storage I/O:

- DeepSeek 32-token run 0: 422.0 MB storage read, run 1: 1.8 MB.
- Qwen KV sweep: 0 MB storage reads because the model was page-cache hot.
- Context setup: `context_setup_read_bytes`, `rchar`, and fault counters.

These counters distinguish OS page-cache reuse from RAMforge-controlled
residency. DeepSeek's second run reading 1.8 MB is page-cache reuse, not proof
that RAMforge retained expert tensors. `rchar` counts read-like syscalls and
does not include pages faulted in through mmap, so `read_bytes` plus fault
counters are the more useful staging evidence; a final hybrid+staging setup
showed 40.8 MB of block reads and 457,983 minor faults with only 561 bytes of
`rchar`, consistent with mapping-based host staging.

## 6. Planning and accounting truthfulness

- Plan schema is now version 3.
- Default batch is `min(2048, context)` and micro-batch is `min(512, batch)`,
  matching native defaults instead of the previous `batch=1`, `ubatch=context`.
- `FieldStatus::Forwarded` distinguishes values sent to native without a
  getter from values with native readback.
- GPU/hybrid plans are no longer labelled advisory-only; effective placement
  is confirmed by native readback.
- `plan` reports KV capacity for the selected dtype/context and expert
  accounting (routed, shared, other, active-routed estimate).
- Service status reports requested KV/flash values and explicitly states that
  native KV readback is unavailable.

DeepSeek full-file expert accounting at context 4,096 with Q8_0 KV:

| Component | Bytes |
|---|---:|
| Total tensor payload | 10,360,420,352 |
| Routed experts | 9,527,361,536 |
| Shared experts | 270,876,672 |
| Other (attention, embeddings, output, norms) | 562,182,144 |
| Estimated active routed set (6/64) | 893,190,144 |
| Estimated active weight total | 1,726,248,960 |

The active total is arithmetic from metadata plus the routed fraction. It is
not a residency observation, does not include KV/compute buffers, and does not
guarantee routing locality.

## 7. Prefill batch

480-token prompt, Qwen2.5-7B CPU, 6 threads, Q8_0 KV, 1 generated token:

| Batch / micro-batch | Prefill tok/s | TTFT |
|---|---:|---:|
| 64 / 64 | 21.42 | 22.41 s |
| 2048 / 512 | 19.39 | 24.75 s |

Batch forwarding is real and read back from native. The 480-token prompt with
batch 64 also exercised the adapter's new chunking path: inputs larger than the
effective native batch are split into position-contiguous chunks, and only the
final chunk supplies logits. On this CPU the smaller micro-batch was slightly
faster; the difference is within one run and should not be generalized. No GPU
prefill claim is made.

MLA models also reject mismatched K/V cache types before execution; the adapter
and planner both return an explicit error instead of relying on a native abort.

## 8. Validation

Executed:

- `cargo test -p ramforge-runtime`: 425 lib tests passed, 2 ignored; 1 device
  test passed; 7 adapter tests passed, 1 ignored; remaining integration tests
  ignored by their fixtures.
- `cargo test -p ramforge-cli`: 61 passed, 33 ignored.
- Opt-in real-model tests run manually and passed:
  - `hybrid_plan_combines_static_offload_with_cpu_layer_staging`
  - `forgecore_deepseek2_generates_with_quantized_kv_and_explicit_batch`
- Real-model CLI/service runs for DeepSeek2, Qwen2.5-7B KV sweep, hybrid+staged
  comparison, hybrid staging transitions, long-context Q4 KV, and thread A/B.

Not validated:

- KV quality/perplexity retention for Q8/Q4.
- Cold-cache behavior (no destructive cache drop).
- Dense 180B/500B feasibility (no such model exists locally).
- Multi-hour stability or concurrent-generation behavior.
- Native KV readback for MLA (not exposed); the plan marks it a lower bound.

## 9. Remaining work and handoff

### Immediate follow-ups

1. Add a persisted, measurable thread/placement calibration profile with
   invalidation keys. The physical-core default is implemented; a full
   calibration loop is not.
2. Measure Q8/Q4 KV quality on a fixed evaluation set before recommending Q4
   for quality-sensitive work.
3. Add a cold-cache-safe benchmark mode that isolates a model copy without
   global cache drops.
4. Extend real-model tests to more MoE and MLA models.

### Handoff: MoE expert-aware residency

This remains the highest-potential capacity track and is **not implemented**.

Evidence this milestone adds:

- Routed experts are 9.53 GB of the 10.36 GB file (92%), not 73%.
- The active routed estimate is 893 MB plus 271 MB shared plus 562 MB other,
  ~1.73 GB for a decode step if routing were perfectly controlled.
- A 32-token run reaches 9.43 GB RSS, showing the OS keeps most of the mapping
  resident when memory allows; short-run RSS is not a capacity result.
- No native API exposes per-token expert selection or expert-granular staging.
  Whole-layer staging moves all 64 experts to use 6.

The next step is a read-only router-observation hook in the native checkout
(separate authorization required), followed by active-set entropy and
consecutive-token overlap measurement. Do not implement expert staging by
guessing routing behavior.

Then:

1. Adaptive memory residency (broader policy work; not this milestone).
2. Hardware performance calibration (persisted, invalidated, explainable).

## 10. Explicit non-goals preserved

No Context Engine, agent orchestration, autonomous indexing, vector database,
multi-model router, new service, competing tensor engine, direct llama FFI in
RAMforge, direct `forge-sys` dependency, native llama.cpp modification, or
destructive cache management was introduced.

**Follow-up:** [MOE_EXPERT_RESIDENCY_2026-11-25.md](MOE_EXPERT_RESIDENCY_2026-11-25.md)
implements read-only CPU expert-routing observation and measures DeepSeek2
expert locality. It confirms that this milestone's ~1.73 GB active estimate is
per-token, not a workload working set.
