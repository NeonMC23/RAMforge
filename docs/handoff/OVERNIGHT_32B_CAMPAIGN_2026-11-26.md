# Overnight 32B Throughput Campaign — 2026-11-26

**Target:** Qwen2.5-Coder-32B-Instruct Q4_K_M (19.85 GB)
**Hardware:** i5-11400H 6C/12T, 15 GiB RAM, RTX 3050 Laptop 4 GiB, NVMe.
**Objective:** maximize real 32B generation tokens/s, integrate speculative decoding
with adaptive residency, and verify CUDA/residency interactions.
**Native checkout:** `7fe450e…`, unmodified.

## 1. Executive summary

The campaign integrated greedy speculative decoding into `InferenceService` and the
benchmark path, then combined it with `--residency auto` for the first time.

Final controlled 16-token result:

| Configuration | Wall | Generation | TTFT | Reads |
|---|---:|---:|---:|---:|
| Ordinary + `--residency auto` | 74.6 s | 0.2145 tok/s | 7.29 s | 132.6 GB |
| Speculative (7B draft, K=12) + auto | **28.9 s** | **0.5540 tok/s** | 18.68 s | 53.3 GB |
| Speculative + auto, 12 staged layers | 27.9 s | 0.5725 tok/s | 18.27 s | 50.5 GB |

The two configurations emitted **identical token IDs** (16/16). The validated gain
is **2.58x over current ordinary auto-residency**, or **4.78x over the historical
CPU-only 0.116 tok/s baseline**.

The trade-off is TTFT: verification delays the first emitted token from 7.3 s to
18.7 s. The campaign prioritized generation throughput as instructed.

Combining speculative decoding with dynamic staging works. It was not assumed:
both the CPU-only and auto-residency combinations were measured, and the staged
target reduced reads and improved generation by ~15-18% over CPU-only speculation.

## 2. Initial repository state

| Repo | Branch | Initial HEAD | Final HEAD | Tree |
|---|---|---|---|---|
| Root `forge` | `main` | `703344a` | *see final report* | clean |
| `RAMforge` | `main` | `5067f0c` | *see final report* | clean |
| `forgeCORE` | `main` | `45bf95f` | unchanged | clean |
| `native/src` | pin | `7fe450e` | unchanged | 12 modified files |

Native patch SHA `d8708f96…`, 2,449 insertions / 2 deletions, reverse-applies.

Local assets verified: 32B target, 1.5B Q4_0 draft, 7B Q4_K_M draft. No downloads.

## 3. Baseline and verification proof

Historical CPU-only baseline reproduced earlier: 8 tokens 72.86 s, 16 tokens
137.62 s. Under `--residency auto`, current ordinary decoding is faster: 8 tokens
40.5 s (0.1973 tok/s), 16 tokens 74.6 s (0.2145 tok/s).

The previous multi-token verification proof remains valid:

| K | Reads / batch | Reads / evaluated token | Evaluated tok/s |
|---:|---:|---:|---:|
| 1 | 17.4 GB | 17.4 GB | 0.10 |
| 4 | 18.4 GB | 4.60 GB | 0.42 |
| 8 | 18.5 GB | 2.31 GB | 0.78 |
| 16 | 18.6 GB | 1.16 GB | 1.42 |

This establishes the weight-reuse mechanism; it is not generation throughput.

## 4. Service integration

`InferenceService` now supports a draft model end to end:

- `with_speculative_draft(path, draft_tokens, draft_threads)`.
- `load()` validates exact draft/target vocabulary compatibility and loads the draft
  model on CPU.
- `create_session()` opens the target with `n_outputs_max_per_seq = K+1` and opens a
  separate draft context with a small batch/KV.
- `generate_speculative()` runs the existing greedy speculative engine over the live
  target and draft sessions, streams tokens, supports cancellation, and returns
  `GenerationReport.speculative`.
- Target staged layers and standby semantics are preserved.
- `unload()` releases target and draft contexts.

Benchmark integration:

```bash
ramforge benchmark TARGET.gguf --ram 10G --context 128 --threads 6 \
  --batch 64 --ubatch 64 --kv-type-k q8_0 --kv-type-v q8_0 \
  --device cpu --residency auto \
  --draft-model DRAFT.gguf --draft-tokens 12 --json
```

Benchmark schema 21 reports `draft_model`, `draft_tokens`, per-run speculative
stats, and output token IDs. Ordinary decoding remains the default when no draft is
given.

## 5. Results

### 8-token service comparison

| Config | Generation | TTFT | Reads | Acceptance | Target forwards |
|---|---:|---:|---:|---:|---:|
| Ordinary CPU | 0.110 tok/s (historical) | — | — | — | 8 |
| Spec 7B K8 CPU | 0.217 tok/s (historical) | — | — | 75% | 2 |
| Ordinary + auto | 0.1973 | 7.45 s | 78.0 GB | — | 8 |
| Spec 7B K12 + auto | 0.3154 | 16.74 s | 46.9 GB | 75% | 2 |

Output token IDs identical (8/8).

### 16-token service comparison

| Config | Wall | Generation | TTFT | Reads | Correctness |
|---|---:|---:|---:|---:|---|
| Ordinary + auto | 74.6 s | 0.2145 | 7.29 s | 132.6 GB | reference |
| Spec 7B K12 + auto | **28.9 s** | **0.5540** | 18.68 s | 53.3 GB | identical |
| Spec 7B K12 + auto, 12 staged | 27.9 s | 0.5725 | 18.27 s | 50.5 GB | identical |

16-token speculative stats: 2 rounds, 21 drafts, 14 accepted, 7 rejected,
2 fallback, 66.66% acceptance, 2 target forwards, target 14.7 s, draft 4.5 s.

### Draft A/B (8 tokens, K8, auto)

| Draft | Run 1 | Run 2 | Median |
|---|---:|---:|---:|
| 1.5B | 0.2737 | 0.2779 | 0.276 |
| 7B | 0.3034 | 0.3125 | 0.308 |

The 7B draft wins by ~11% and reads ~11% less, at the cost of ~4.7 GB host memory
and higher TTFT.

### K tuning (7B draft, auto, 8 tokens)

| K | Generation | Notes |
|---:|---:|---|
| 4 | 0.2282 | 3 rounds, more target forwards |
| 8 | 0.2958-0.3022 | 2 rounds |
| 10 | 0.3042 | 2 rounds |
| 12 | 0.3025-0.3215 | 2 rounds; best median range |
| 14 | 0.2641 | larger draft cost |
| 16 | ~0.121-0.13 | verification batch too large; rejected |

K=8-12 is the practical range on this hardware. K=12 is marginally better but
within run-to-run noise; K=8 is the safer default.

### Staged layer count

Default `--residency auto` staged 11 layers (3.36 GB) with a 256 MiB margin.
A 128 MiB margin staged 12 layers (3.63 GB) and measured 0.5725 vs 0.5540 tok/s,
with 50.5 vs 53.3 GB reads. The 12-layer configuration leaves very little VRAM
headroom; 11 layers is the safer default.

## 6. CUDA and heterogeneous findings

- Dynamic staging is the productive GPU use for this workload: it removes host
  reads for staged layers and improved ordinary auto-residency to 0.2145 tok/s
  and speculative auto-residency to 0.554 tok/s.
- Static target offload with speculative verification remains a negative result:
  6 static layers + 7B draft K8 measured 127.4 s vs 65.7 s ordinary static-6.
  The CLI rejects `--gpu-layers` with speculative decoding.
- No custom CUDA kernel was justified. The dominant cost is host storage traffic
  and target forward time; draft generation is only 3-4.5 s of a 29 s run. Adding
  a custom kernel would duplicate llama.cpp/ggml without evidence of a bottleneck
  it could remove.
- Draft-on-GPU was not tested: target staging consumes ~3.36 GB of 3.68 GB free
  VRAM. Reducing staging to free ~1 GB for the 1.5B draft is a possible future
  experiment, but draft time is not the current bottleneck.

## 7. Correctness

- Output token IDs identical for ordinary vs speculative at 8 and 16 tokens,
  including with `--residency auto`.
- KV rollback exact-logit probe: one/zero/all accepted match reference.
- Rejection paths: 7 rejected candidates in the 16-token run; output identical.
- EOS path tested on 1.5B; `stop_reason=eos`.
- Draft vocabulary exact prefix (tokens/merges/BOS/EOS); incompatible drafts and
  target-only prompt ids are rejected.
- Non-greedy sampling, GPU target placement, and invalid batch sizes are rejected
  explicitly.
- Ordinary decoding remains the default and is unchanged when no draft is given.

## 8. Tests

- `cargo test --workspace`: 665 passed, 0 failed, 50 ignored.
- New service/benchmark integration compiles and is exercised by real 32B runs.
- Clippy on touched crates: no warnings.
- Real-model validation: 8/16-token ordinary vs speculative with auto residency;
  token-ID equality; 1.5B/7B draft A/B; K sweep; staged 11/12 layers.
- Existing speculative unit tests and opt-in `speculative_real` remain passing.

Not validated:

- Distribution-preserving sampling (temperature/top-k/top-p).
- Draft-on-GPU.
- Static GPU target + speculation (negative, disabled).
- Long-context behavior beyond 256.
- Cold-cache behavior.
- 180B/500B claims.

## 9. Language and technology rationale

- Rust: service integration, policy, benchmark/reporting, safe orchestration.
- forgeCORE/Rust + existing C++/CUDA backend: no native source change needed.
- Python: only for parsing benchmark JSON and computing ratios; no committed
  runtime component.
- No custom C++/CUDA component was added because no measured bottleneck justified
  one.

## 10. Limitations and trade-offs

- Speculative TTFT is 18.7 s vs 7.3 s ordinary: the first output waits for the
  first verification batch. Throughput improved, latency did not.
- 7B draft adds ~4.7 GB host memory and page pressure; 1.5B is lower-risk.
- K=12/12-stage results are close to noise; 11 layers and K8 are safer defaults.
- `--ram` remains advisory unless cgroup enforcement is used.
- The target remains CPU-executed; GPU static placement is slower.
- No 180B/500B feasibility claim.

## 11. Reproduction commands

```bash
# ordinary + adaptive residency
ramforge benchmark TARGET.gguf --ram 10G --context 128 --threads 6 \
  --batch 64 --ubatch 64 --kv-type-k q8_0 --kv-type-v q8_0 \
  --runs 1 --warmup 0 --max-tokens 16 --prompt "The capital of France is" \
  --no-repack --no-prefetch --device cpu --residency auto --json

# speculative + adaptive residency
ramforge benchmark TARGET.gguf --ram 10G --context 128 --threads 6 \
  --batch 64 --ubatch 64 --kv-type-k q8_0 --kv-type-v q8_0 \
  --runs 1 --warmup 0 --max-tokens 16 --prompt "The capital of France is" \
  --no-repack --no-prefetch --device cpu --residency auto \
  --draft-model DRAFT.gguf --draft-tokens 12 --json
```

## 12. Continuation plan

1. Add an optional first-token ordinary decode to restore low TTFT before
   speculative rounds; measure the throughput/TTFT trade-off.
2. Test 1.5B draft on GPU with reduced staging (e.g., 8 layers) now that draft
   time and VRAM headroom are known.
3. Extend exact speculative sampling to temperature/top-k/top-p.
4. Calibrate K and staged-layer count per model/hardware; avoid fixed constants.
5. Measure longer outputs/contexts and cold/warm states with the same protocol.
