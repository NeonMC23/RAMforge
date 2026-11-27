# Night Performance Research and Adaptive Optimization — 2026-11-27

**Scope:** 32B inference performance, host/storage bottleneck analysis, confirmed
plan execution, resource-accuracy measurements, and cross-architecture inference
validation.
**Native checkout:** `7fe450e19305b828c199d602c23a8337aaa1f03b`, unchanged.

## 1. Executive summary

The campaign executed the previously unvalidated 32B hybrid/staged/speculative
plan and improved the best measured 32B generation throughput from the historical
**0.554 tok/s** to **0.7215 tok/s** (three-run median, 7B draft, K20) and
**0.711 tok/s** with a more cache-stable 1.5B draft at K24.

Relative to the fresh ordinary baseline measured on the same machine state:

| Configuration | Generation tok/s | Speedup |
|---|---:|---:|
| Ordinary, 16 tokens, warm baseline | 0.1087 | 1.0x |
| Best measured, 7B draft K20, 32 tokens | **0.7215** | **6.64x** |
| Reproducible alternative, 1.5B draft K24 | 0.7110 | 6.54x |
| Historical best (2026-11-26) | 0.5540 | 5.10x |

The **2 tok/s target was not reached**. Best decode throughput was **1.31 tok/s**
(7B draft K20), about 65% of the target; best generation throughput (including
TTFT) was 0.72 tok/s, about 36% of the target. The limiting factor is measured
storage/cache bandwidth, not an unexplored configuration: each verification
round must read roughly 16.5 GB of non-staged target weights at about 1.8-1.9 GB/s
effective, and the model cannot coexist with its draft in the available RAM/page
cache.

The campaign also produced real, evidence-backed engine improvements:

- Measured memory policy (no-repack + no-prefetch) is applied automatically to
  constrained models and is part of workload identity.
- Whole-machine CPU utilization, per-position speculative acceptance, and
  target/draft forward times are now measured and persisted.
- Plans compare estimated PSS/VRAM against observations.
- Compatibility resolution upgrades “unvalidated” claims only when a persisted
  measurement matches the model and hardware.
- GPT-NeoX and DeepSeek2 inference were executed and validated for the first
  time in this line of work.

## 2. Repository state

| Repository | Branch | Start HEAD | End HEAD | Tree |
|---|---|---|---|---|
| `/forge` | `main` | `d245bf7` | see final session summary | clean |
| `/forge/RAMforge` | `main` | `8f3b6b4` | `203feb6` (code) | clean |
| `/forge/forgeCORE` | `main` | `45bf95f` | unchanged | clean |
| `/forge/native/src` | pinned | `7fe450e…` | unchanged | 12 modified files |

Native verification at the end of the campaign:

- 12 modified files, 2,449 insertions / 2 deletions
- diff SHA-256 `d8708f96fda9b0c77f8af5f8e53008481bc5de012ecdba9822a7e06ec5c71d7d`

The RAMforge code commit is `203feb6`; the documentation update is the commit
that adds this file. No user model was downloaded, moved, renamed, deleted, or
modified.

## 3. Environment audit

| Item | Observed |
|---|---|
| CPU | Intel Core i5-11400H, 6 cores / 12 threads |
| RAM | 15 GiB total; 5-8 GiB available depending on pressure |
| Swap | 24 GiB configured; 11-13 GiB used; zram 8 GiB active |
| GPU | NVIDIA RTX 3050 Laptop, 4,096 MiB total, ~3,451 MiB free |
| Driver / CUDA | 620.70.06 / CUDA 13.2 |
| Model storage | `/home` on LUKS + btrfs (`compress=zstd:1`) over `nvme0n1` |
| Storage read | 2.8 GB/s direct (1 GiB `O_DIRECT`), 1.8 GB/s buffered |
| Model file | 19,851,335,872 bytes (18.49 GiB), extents marked encoded |
| Concurrency | qemu ~85% CPU + 1.5 GB RSS, browsers, Discord, Java; active reclaim |

The machine was under **significant memory pressure** during the campaign.
`/proc/pressure/memory` showed full avg10 up to 4.6% and some avg10 up to 5.5%
during measurements, with qemu and Java holding several GiB in swap. Absolute
throughput is therefore environment-specific and lower than the historical
campaign's conditions.

## 4. Baselines

Fresh 32B ordinary 16-token runs with `--residency auto`, q8 KV, 6 threads:

| Condition | Generation tok/s | TTFT | Reads | Staged | Notes |
|---|---:|---:|---:|---:|---|
| Cold page cache | 0.0782 | 14.38 s | 253.0 GB | 10 | first run after drop/eviction |
| Warm (4.3 GiB model resident) | 0.1087 | 11.11 s | 230.1 GB | 11 | still LRU scan thrash |

The historical 0.2145 tok/s ordinary baseline was measured with a warmer page
cache and less competing workload. It is not directly comparable to the current
environment.

## 5. 32B performance results

All speculative results use greedy decoding, exact token-id verification, 32
generated tokens unless noted, `--no-prefetch --no-repack`, 6 target threads,
q8 KV, context 128, batch/ubatch 64, auto staging (11 layers, 3.358 GB VRAM).

### 5.1 Best measured configuration

**7B draft, K20, three measured runs** (`/tmp/opencode/night-best-7.json`):

| Metric | Median | Min | Max |
|---|---:|---:|---:|
| Generation tok/s | **0.7215** | 0.7118 | 0.7222 |
| Decode tok/s | **1.3103** | — | — |
| TTFT | 21.08 s | — | — |
| CPU busy | 43.6% | — | — |
| Reads / run | ~67.5 GB | 67.4 | 70.1 |
| PSS | ~13.1 GB | — | — |
| Rounds | 3 | 3 | 3 |
| Accepted tokens | 29 | 29 | 29 |
| Acceptance | 65.9% | — | — |
| Target forward time | 26.2 s | 25.2 | 26.5 |
| Draft forward time | 8.9 s | 8.9 | 9.5 |

This configuration is stable at the 1.4% variability level in three consecutive
runs on the same warm machine state.

### 5.2 More cache-stable alternative

**1.5B draft, K24, three measured runs** (`/tmp/opencode/night-best-15.json`):

| Metric | Median | Min | Max |
|---|---:|---:|---:|
| Generation tok/s | **0.7110** | 0.6880 | 0.7254 |
| Decode tok/s | 1.1247 | — | — |
| TTFT | 16.55 s | — | — |
| CPU busy | 47.5% | — | — |
| Reads / run | ~63.9 GB | 52.6 | 67.5 |
| PSS | ~13.0 GB | — | — |
| Rounds | 4 | 4 | 4 |
| Acceptance | 36.8% | — | — |
| Draft forward time | 3.56 s | 3.1 | 3.6 |

The 1.5B draft gives up ~1.5% generation throughput but has a 1.07 GB footprint,
fourfold lower draft time, and much lower sensitivity to target/draft cache
thrash. A separate two-run 7B session during heavier memory pressure collapsed
from 0.626 to 0.434 tok/s when draft time rose from 10 s to 31 s; the 1.5B
configuration did not exhibit that failure mode.

### 5.3 K sweep

1.5B draft (32 tokens, one run each unless noted):

| K | Generation | Decode | Rounds | Acceptance | Target time | Draft time |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.410 | 0.580 | 5 | 45.0% | 61.3 s | 4.6 s |
| 20 | 0.452 | — | 4 | 43.8% | 52.7 s | 5.6 s |
| 24 | 0.462 | 0.718 | 4 | 36.8% | 53.7 s | 4.8 s |
| 28 | 0.445 | — | 4 | 32.9% | 54.0 s | 5.4 s |
| 32 | 0.469 | 0.749 | 4 | 31.5% | 52.3 s | 5.9 s |

7B draft:

| K | Generation | Decode | Rounds | Acceptance | Target time | Draft time |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.374 | 0.562 | 4 | 63.6% | 53.3 s | 18.5 s |
| 18 | 0.487 | 0.702 | 4 | 60.9% | 34.1 s | 21.4 s |
| 20 | **0.703-0.722** | **1.31-1.32** | 3 | 65.9% | 25.2-26.5 s | 8.9-9.5 s |
| 22 | 0.591 | 0.990 | 3 | 60.4% | 28.3 s | 16.0 s |
| 24 | 0.644 | 1.183 | 3 | 55.8% | 28.6 s | 10.4 s |
| 28 | 0.632 | 1.180 | 3 | 50.9% | 29.8 s | 10.6 s |
| 32 | 0.575 | 0.984 | 3 | 48.3% | 29.2 s | 16.4 s |

K20 is the 7B optimum because it is the smallest K that still reaches three
verification rounds for 32 tokens; larger K adds draft cost without reducing the
round count.

### 5.4 Memory-policy A/B (1.5B, K24)

| Configuration | Generation | Notes |
|---|---:|---|
| No-repack only | 0.435 | 6.1 s draft |
| No-prefetch only | 0.394 | 6.2 s draft, 33.5 s TTFT |
| Both | **0.462** | 4.8 s draft |
| Both, 12 threads | 0.388 | draft time doubled to 13.2 s |
| Baseline K16, no flags | 0.322 | compared at K16 |

Both flags are material and complementary. Twelve threads are counterproductive
for the staged target. Repacking and whole-file prefaulting are now disabled
automatically for constrained plans.

### 5.5 Staging and prefetch experiments

- 12 staged layers with a 128 MiB margin: 0.634 tok/s in a noisy run; target
  forward time was unchanged (25.5 s) and draft variability dominated. The safer
  11-layer configuration (3.358 GB of 3.451 GB free VRAM) remains the default.
- `--residency-prefetch 4G` (native `MADV_WILLNEED`) before generation: 0.155 vs
  0.157 tok/s in matched 8-token runs; no reduction in target-forward time.
  This is a **negative result**: a one-shot hint did not keep the subset resident.
- 64-token generation with 7B K20: 0.570 tok/s generation, 0.699 decode, six
  rounds. Longer runs suffer because the 4.7 GB draft and 16.5 GB target working
  set cannot both remain cached; draft time rose from ~9 s to 45.6 s.

### 5.6 Confirmed plan execution

The 32B adaptive plan was generated with a 7B draft, automatic K, staged
residency, no-repack/no-prefetch, and f16 KV, then applied through
`adaptive-apply --confirm`:

- status **verified**
- execution mode CPU with **11 verified staged GPU layers** (hybrid resource use)
- GPU layers 0 (target placement), context/batch/ubatch 128/64/64, threads 6
- speculative K20 verified, acceptance 58.3%
- KV dtype accepted (no native getter)
- measured **0.556 tok/s**, TTFT 19.1 s (16 tokens)

This is the first full execution of the previously unvalidated plan.

## 6. Profiling and bottleneck analysis

### Measured facts

- Each target verification round reads the full non-staged model: 19.85 GB -
  3.358 GB staged ≈ **16.5 GB**.
- Best target forward time: 25.2-26.5 s for 3 rounds ≈ **8.4-8.8 s/round**,
  implying ~1.9-2.0 GB/s effective.
- Direct storage read measured 2.8 GB/s; buffered read 1.8 GB/s; the mmap
  page-fault path is close to the buffered ceiling.
- Whole-machine CPU during a representative run: **43.6% busy**, 15.2% iowait,
  31.8% system, 28.9% user, 24.1% idle. The high system time is consistent with
  kernel page-fault, LUKS, and btrfs decompression work, not with a saturated
  userspace compute path.
- Model extents are btrfs-encoded on a LUKS volume; compression saves no space
  (18.49 GiB disk for 19.85 GB apparent) but still goes through the encoded
  extent path.
- Page cache resident for the model was only ~4.3 GiB after an ordinary run.
  With a 16.5 GB sequential working set, LRU scan behavior yields almost no
  reuse: ordinary warm runs still read 230 GB for 16 tokens.
- Speculative position acceptance for 1.5B K24 collapses after roughly four
  positions; the 7B draft sustains acceptance longer. Position counters are now
  persisted.
- 7B draft weights (4.68 GB) are frequently evicted by the target scan; draft
  time varies from 8.9-9.5 s to 31 s across sessions. The 1.5B draft (1.07 GB)
  remains stable at 3.1-3.6 s.

### Hypotheses consistent with the data

- The workload is **storage/page-fault bound**, with a secondary kernel-CPU cost
  from LUKS/btrfs and page faults.
- The 2 tok/s target requires either ~3.4 GB/s effective read bandwidth or ~17
  accepted tokens per verification round at the current 8.5 s/round. Neither is
  achievable with the present storage path and draft acceptance.
- A host-resident subset of 6-8 GB would materially help, but the machine has no
  free 6-8 GB and the native API exposes only a best-effort WILLNEED hint, not
  pinning. One-shot WILLNEED was measured neutral.

### 2 tok/s gap

| Metric | Best measured | Target | Gap |
|---|---:|---:|---:|
| Generation tok/s | 0.7215 | 2.0 | 2.77x |
| Decode tok/s | 1.3103 | 2.0 | 1.53x |
| Target forward / round | 8.4-8.8 s | ~4.3-5.0 s | ~1.8x |
| Effective read bandwidth | ~1.9-2.0 GB/s | ~3.4 GB/s | ~1.7x |
| Accepted tokens / round | 9.7 | ~17 | ~1.75x |

Both a bandwidth and an acceptance improvement would be required; either alone
is insufficient.

## 7. Implementation report

### 7.1 Measured memory policy in adaptive plans

- `WorkloadProfile` now carries `no_repack`, included in performance identity.
  `PROFILING_SCHEMA_VERSION` is 3; schemas 1 and 2 still load but cannot match a
  current exact identity.
- `AdaptiveStrategy` carries `no_repack`; plans for constrained models use
  `load_mode = mmap-no-prefetch` and `repack = false`.
- Plan decisions include a `memory-policy` entry with measurement provenance.
- Automatic speculative K: 24 for a ≤2 GiB draft, 20 for a larger draft, 8 when
  the target fits comfortably; explicit user values override.
- Calibration sweeps larger K values for staged targets (16/20/24/32 for small
  drafts, 16/20/24/28 for large drafts).

### 7.2 Performance observability

- `CpuUtilization` per `BenchmarkRun` and `median_cpu_busy_bps` per report,
  parsed from `/proc/stat` deltas (whole machine, not per-core, not RAMforge-only).
- `SpeculativeStats.position_attempts` and `position_accepts` expose acceptance
  decay by draft position.
- `PerformanceRecord` and `MeasurementSummary` persist target/draft forward
  times, CPU busy, and staged VRAM bytes.
- Median statistics now average the two middle values for even run counts; the
  previous upper-value choice overstated two-run results.

### 7.3 Resource accuracy

- `AppliedPlanReport.resource_comparison` compares host estimate vs observed PSS
  and staged-VRAM window vs observed resident bytes, with signed error in basis
  points.
- For the 32B plan: host estimate 15.99 GB vs observed PSS 13.0-13.2 GB
  (21-23% conservative); staged window 3.532 GB vs observed 3.358 GB (95% used).
- File size (19.85 GB) is never treated as resident memory.

### 7.4 Capability and compatibility

- `resolve_compatibility_with_evidence` upgrades unvalidated GPU/staged/MoE
  features to limited only when a persisted record matches model and hardware
  fingerprints; workload mismatch is named.
- Staged-hybrid verification now correctly treats “CPU target + verified staged
  GPU layers” as a verified hybrid instead of a partial application.
- Benchmark execution tolerates missing KV-head metadata with a warning instead
  of refusing to run; this enabled GPT-NeoX validation.
- No quality-reducing default was introduced. Speculative decoding remains
  greedy-only; KV quantization remains an explicit tradeoff.

## 8. Cross-architecture validation

| Model | Metadata | Planning | Execution | Performance | Reproduced |
|---|---|---|---|---|---|
| Qwen2.5-Coder-32B Q4_K_M | yes | yes | yes (plan apply) | 0.71-0.72 tok/s best | 3 runs |
| Qwen2.5-7B Q4_K_M | yes | yes | yes (draft/target) | draft path | 3 runs |
| Qwen2.5-1.5B Q4_0 | yes | yes | yes (draft) | draft path | 3 runs |
| SmolLM2-135M Q4_K_M | yes | yes | yes GPU (previous milestone) | 84.83 tok/s | prior session |
| GPT-NeoX-20B Q4_K_M | yes | yes | **yes**, 2 tokens | 0.328 tok/s exploratory | no |
| DeepSeek-Coder-V2-Lite Q4_K_M | yes | yes | **yes**, 2 tokens | 0.631 tok/s exploratory | no |
| DeepSeek MoE planning | yes | yes | staged, 10 layers | not a perf claim | — |

GPT-NeoX ran with unknown KV-head metadata retained as an estimate (warning) and
12 staged layers. DeepSeek2 ran with 10 staged layers and 6.2 GB reads for two
tokens. The 2-token numbers validate execution only.

## 9. Tests and validation

- `cargo test --workspace`: **704 passed, 0 failed, 50 ignored**.
- `cargo test -p ramforge-cli --bin ramforge`: 2 passed.
- `cargo clippy -p ramforge-runtime -p ramforge-cli --all-targets`: no warnings.
- Formatting applied only to touched files; pre-existing main.rs debt preserved.
- Manual end-to-end:
  - 32B adaptive plan generated, inspected, applied, readback verified.
  - 32B K sweep and A/B experiments with persisted measurements.
  - GPT-NeoX and DeepSeek2 short inference.
  - Missing-confirmation and stale-plan rejection remain enforced.
  - No model file was modified.

## 10. Negative and inconclusive results

- One-shot host prefetch (`MADV_WILLNEED`) did not improve target-forward time.
- Twelve target threads were 16% slower than six; draft time doubled.
- Twelve staged layers did not measurably improve a noisy run; 11 remains safer.
- Larger K (22/24/28/32) with the 7B draft did not beat K20.
- 64-token generation regressed to 0.57 tok/s because draft/target cache thrash
  increased draft time fourfold.
- A 7B-draft session under heavy pressure fell from 0.626 to 0.434 tok/s; the
  adaptive engine's caution about single-run winners is justified.

## 11. Remaining opportunities

1. **Faster storage tier.** The 990 PRO (`nvme1n1`) is mounted as NTFS and has not
   been used. A model copy on a faster uncompressed volume could raise effective
   bandwidth toward 2.8-3.5 GB/s (≈1.5-1.8x). Not performed because the campaign
   forbids moving/modifying model assets and a 20 GB copy requires explicit user
   approval.
2. **Host-resident subset with pinning.** `mlock`/`MADV_POPULATE_READ` on a
   bounded 4-6 GB window could reduce reads ~25-35%, but requires native work and
   a safe memory budget; available RAM is currently insufficient for a large
   window.
3. **KV/prefix reuse to cut TTFT.** TTFT is 16-21 s of the 44-46 s wall. A
   reusable prompt prefix would improve generation throughput more than further
   decode tuning, but context/cache redesign is deferred.
4. **GPU draft.** A 1.5B draft on GPU could remove ~3 s/run but consume VRAM now
   used for staging; net benefit is small. A 7B draft cannot fit 4 GB.
5. **Higher-acceptance draft.** Reaching ~17 accepted tokens/round would require
   materially better draft agreement; no new model may be downloaded.
6. **Per-layer prefetch overlap.** A native hook could overlap the next layer's
   reads with current compute, but compute is a small fraction of round time.
7. **GPU utilization sampling.** `nvidia-smi` is available; integrating NVML or
   sampled counters would complete the observability picture.
8. **Free host memory / reduce competing workloads.** The single largest
   environmental lever is user-controlled; freeing 6-8 GB would allow meaningful
   page-cache residency and could roughly double throughput.

## 12. Commands

```bash
# Best measured 32B configuration
ramforge benchmark TARGET.gguf --ram 10G --context 128 --threads 6 \
  --batch 64 --ubatch 64 --kv-type-k q8_0 --kv-type-v q8_0 \
  --runs 3 --warmup 0 --max-tokens 32 --residency auto \
  --draft-model DRAFT_7B.gguf --draft-tokens 20 \
  --no-prefetch --no-repack --json

# More cache-stable alternative
ramforge benchmark TARGET.gguf ... --draft-model DRAFT_1_5B.gguf --draft-tokens 24 \
  --no-prefetch --no-repack --json

# Generate and apply an adaptive plan (auto K and memory policy)
ramforge adaptive-plan TARGET.gguf --ram 10G --context 128 --threads 6 \
  --draft-model DRAFT.gguf --draft-tokens 0 --max-tokens 16 --runs 1 \
  --performance-profile /tmp/profiles.json --output /tmp/plan.json
ramforge adaptive-apply /tmp/plan.json --confirm --json
```

## 13. Final assessment

RAMforge became **measurably faster** on the primary workload under the current
machine state: 0.1087 → 0.7215 tok/s generation (6.64x), and +30% over the
historical best 0.554 tok/s. The improvement is attributable to measured
policy (no-repack/no-prefetch), K selection, draft choice, and the existing
staged/speculative path — not to instrumentation.

It also became **more intelligent and more honest**: CPU/I-O evidence is measured,
plan resource estimates are compared with observations, compatibility can consume
matching measurements, workload identity includes repacking, and the median is
correct for even run counts.

The **2 tok/s target remains blocked by physical storage and memory limits**:
~16.5 GB must be read per verification round at ~1.9 GB/s, and the draft cannot
remain cached alongside the target. Reaching it would require a faster storage
tier, a safely pinned host-resident window, a higher-acceptance draft, or
substantially more free RAM. Those are documented as concrete next steps rather
than claimed as achieved.
