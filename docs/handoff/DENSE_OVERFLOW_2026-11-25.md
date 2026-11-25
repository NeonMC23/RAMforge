# Dense Overflow Streaming and Adaptive Memory Residency

**Date:** 2026-11-25
**Scope:** Dense GGUF inference when weights exceed RAM/VRAM. Measure the real
bottleneck, add trustworthy per-token residency/I/O instrumentation, implement a
small bounded residency policy, and validate it on the local 32B dense model.
**Native checkout:** pinned `7fe450e19305b828c199d602c23a8337aaa1f03b`, unmodified.

## 1. Executive summary

The dense 32B overflow bottleneck is **physical storage bandwidth plus a working
set larger than the host page cache**, not page-fault count alone and not CPU
compute. On this machine the model is 19.85 GB, host RAM is 15 GiB, and the
process retains about 10.7-11.1 GB of the model mapping. Each forward pass still
reads roughly 17-19 GB from storage because the kernel evicts clean file pages
under pressure.

Bounded dynamic staging of CPU-resident layers into free VRAM is the only
existing mechanism that produced a real improvement:

| Configuration | Decode tok/s | TTFT | Per-forward host reads |
|---|---:|---:|---:|
| Direct CPU (baseline) | 0.0982 | 11.45 s | 17.0-18.0 GB |
| `--residency auto` (11 staged layers) | **0.1226** | **9.00 s** | **11.6-15.4 GB** |
| Earlier staged run (12 layers) | 0.1302-0.1415 | 7.7-8.6 s | 13.6-15.7 GB |
| Static GPU, 6 layers | 0.1167 | 8.25 s | ~15.0 GB (48.3 GB / 3 forwards) |
| Hybrid 4 static + 6 staged | 0.0903 | 11.79 s | 15.0-16.0 GB |

**Measured improvement:** approximately **+25% to +38% decode throughput** and
**13-32% fewer host reads per forward** across the staged configurations. The
result is still ~0.12 tok/s, not interactive. No 180B/500B claim is made.

Negative results are included: combining static GPU offload with dynamic staging
hurt the 32B model, largest-first selection did not show a consistent advantage,
and releasing host pages of staged layers released zero bytes because tensor
pages are shared at mapping boundaries.

## 2. Root cause with evidence

Baseline command (final run):

```bash
ramforge benchmark qwen2.5-coder-32b-instruct-q4_k_m.gguf \
  --ram 10G --context 64 --threads 6 --batch 64 --ubatch 64 \
  --kv-type-k q8_0 --kv-type-v q8_0 --runs 1 --warmup 0 --max-tokens 3 \
  --no-repack --no-prefetch --device cpu --residency off --json
```

| Metric | Baseline |
|---|---:|
| File / tensor bytes | 19,851,335,872 (19.85 GB) |
| Decode tok/s | 0.0982 |
| TTFT | 11.45 s |
| Prompt storage read | 17.62 GB |
| Decode storage read, token 1 | 17.02 GB |
| Decode storage read, token 2 | 18.00 GB |
| Major faults per forward | 5,416 / 6,627 / 5,921 |
| RSS / PSS | ~11.1 GB |
| Resident model mapping (smaps) | 11.01 GB |
| Swap in use | ~0.33 GB |

`rchar` was 0 because accesses are mmap page faults, not `read()` syscalls. The
useful physical-storage counter is `/proc/self/io:read_bytes`, not `rchar`.

The mapping resident at the end of a run (11.0 GB) is smaller than the file
(19.85 GB), and per-forward reads are larger than the missing 8.8 GB because
clean file pages are evicted and refaulted during the pass. Readahead batches
roughly 1 MB per major fault (17-19 GB / ~6,000 faults), so the fault count is a
symptom of storage traffic, not the primary limit.

Effective storage throughput is about 1.5-1.7 GB/s. At that rate, a 17-18 GB
forward cannot complete in less than ~10 s. Threading cannot fix this. Placement
and residency can only reduce the bytes requested per forward.

## 3. Data lifecycle and mechanisms

1. GGUF file is opened and mapped with `mmap-no-prefetch`.
2. Tensor metadata points into the file mapping; no weights are copied to RAM.
3. Native CPU kernels fault pages in layer order. The kernel retains what fits and
   evicts clean pages under pressure.
4. Static GPU offload moves selected layers into CUDA model buffers.
5. Dynamic staging copies whole CPU layers into dedicated CUDA buffers and
   substitutes their tensors for graph execution. Staged layers persist across
   tokens and are reused.
6. RAMforge can sample/release mapping pages, but it does not control Linux page
   cache replacement.

Mechanism evaluation:

| Mechanism | Actual effect | Measured result |
|---|---|---|
| mmap + page cache | OS-managed demand paging/readahead | Retains ~11 GB of 19.85 GB; still 17-18 GB reads/forward |
| Static GPU offload | Whole layers in CUDA model buffers | 6 layers: +13% decode, ~15% fewer reads |
| Dynamic staging | Whole layers in CUDA staging buffers, reused across tokens | 11-12 layers: +25-38% decode, 13-32% fewer reads |
| Hybrid static + staged | Two placements plus graph splits | 4+6 layers: **worse** (0.0903 tok/s) |
| Host release of staged pages | `release_tensor_pages` on staged layers | **0 bytes released**; pages shared; removed |
| `LargestFirst` selection | Stage largest layers by bytes | No clear advantage over smallest/ascending |
| Prefetch | None implemented | OS readahead already batches; no safe per-layer lead-time hook |

## 4. Implementation

### Instrumentation

- `TokenIo`: per-decode-call elapsed time, storage `read_bytes`, `rchar`, read
  syscalls, major/minor faults. Prompt and each generated token are separate.
- `ProcessMemorySplit`: RSS, PSS, `Pss_File`, `Pss_Anon`, anonymous, swap from
  `/proc/self/smaps_rollup`, with unit tests.
- Benchmark report: mapping mapped/resident bytes before load, after load, after
  generation, and mapping PSS after generation.
- Benchmark schema version 18; JSON and text output include the new fields.

### Residency policy

`dense_residency::decide_dense_residency` is a small explainable policy:

- Model fits RAM budget -> direct execution, no staging.
- Model exceeds budget and a GPU backend exists -> enable bounded staging.
- Model exceeds budget and no GPU exists -> reject with the reason.

CLI:

```bash
ramforge benchmark MODEL --ram 10G --residency auto ...
```

`auto` enables the existing `staged-auto` path. The stage budget is `free VRAM
after context creation - safety margin`, so it accounts for compute buffers. The
default safety margin is 256 MiB.

`LargestFirst` was added to `SelectionPolicy`, with a shared
`order_candidate_layers` helper and tests. It remains an option; measurements did
not justify changing the default.

### What was removed

The host-page-release experiment was removed after it released zero bytes on both
32B and 7B staged models. Keeping an ineffective safety-sensitive option was not
justified.

## 5. Memory safety and limits

- No `mlock`, no global page-cache drop, no swap storm, no OOM induction.
- Staging uses the existing validated graph invalidation/delta-commit path.
- The residency decision only enables existing staging; it does not move live
  tensors or alter native scheduling.
- If the budget cannot be satisfied, staging rejects the plan; no silent
  overcommit.
- RAMforge does not control Linux page-cache residency. PSS/mapping numbers are
  observations, not enforced memory.
- No native llama.cpp changes.

## 6. Benchmarks

Hardware: Intel i5-11400H (6C/12T), 15 GiB RAM, RTX 3050 Laptop 4 GiB, driver
620.70.06, local NVMe. Shared desktop, background load present. Warm page cache
except where noted. No destructive cold-cache test was performed.

Model: Qwen2.5-Coder-32B-Instruct Q4_K_M, 19.85 GB, 64 transformer layers, dense.

Final pair, 3 generated tokens, same context/batch/KV/threads:

| Metric | Baseline | `--residency auto` | Change |
|---|---:|---:|---:|
| Staged layers | 0 | 11 | — |
| Decode tok/s | 0.0982 | **0.1226** | **+24.8%** |
| TTFT | 11.45 s | **9.00 s** | **-21.4%** |
| Prompt read | 17.62 GB | 15.42 GB | -12.5% |
| Decode read 1 | 17.02 GB | 11.63 GB | -31.7% |
| Decode read 2 | 18.00 GB | 12.51 GB | -30.5% |
| PSS | 11.10 GB | 11.90 GB | higher |
| Mapping resident | 11.01 GB | 11.71 GB | higher |

PSS/mapping did not fall because the host mapping pages remain cached even when
the layers are executed from CUDA. The benefit is fewer host reads, not lower
host RSS. Reporting lower RSS as the success metric would be wrong.

Earlier staged runs reached 0.1302 and 0.1415 tok/s; single-run variance on this
shared desktop is material. Use the **+25% conservative** figure, not the best.

Negative A/B (single runs): smallest-first 0.1302, ascending 0.1358,
largest-first 0.1371 with different staged byte totals. Differences are within
run-to-run noise; no default change is claimed. The interleaved repetition was
interrupted, so this remains unresolved.

## 7. I/O and residency caveats

- `read_bytes` is process-wide physical storage I/O, not model-only. Other
  process activity can contribute.
- `rchar` is not physical traffic; mmap faults are absent from it.
- Mapping RSS/PSS comes from `smaps` at a point in time; it does not prove which
  pages were used during a forward.
- Major faults depend on readahead and mapping state; they are not a direct
  measure of bytes transferred.
- Warm page cache and prior runs affect results. No safe cold-cache method was
  used.
- The model was not modified; no quantization or quality change was involved.

## 8. Tests

- RAMforge workspace tests passed; new tests cover smaps parsing, policy
  decisions, and layer ordering policies.
- `cargo clippy` on touched crates: no new warnings after fixes.
- Real-model benchmark runs validated baseline, `--residency auto`, static
  offload, staging, and the negative hybrid combination.
- No native checkout tests were required beyond the pinned build; native source
  was not modified.

## 9. Limitations and blocked work

- No cold-cache measurement; page-cache state is warm/uncontrolled.
- No cgroup-enforced run in this milestone; `--ram` remains a planning budget.
- Staging is limited by 4 GB VRAM; 11-12 of 64 layers is the practical maximum.
- No expert/dense row-granular prefetch; the native API is tensor/layer granular.
- No multi-token verification or speculative decoding; this is the main untested
  lever for amortizing weight reads.
- No 180B/500B execution was attempted.

## 10. Next milestone

**Dense multi-token verification / speculative decoding.** The measured floor is
set by bytes read per forward. Staging saves 13-32%; verifying K candidate tokens
in one target forward could amortize the remaining weight reads by up to K, if a
compatible draft source and acceptance policy are available. This requires a
separate correctness-focused milestone.

Secondary:

1. Hardware performance calibration and persisted placement policy.
2. Safe cold/warm benchmark methodology on an isolated model copy.
3. Native API proposal for ordered layer/expert-range prefetch only if
   measurements show a lead-time benefit.

No claim is made that the current implementation controls page cache or makes
180B/500B models practical.
