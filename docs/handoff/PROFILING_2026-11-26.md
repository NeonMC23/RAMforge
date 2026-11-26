# Model-Agnostic Profiling and Adaptive Performance Foundation

**Date:** 2026-11-26
**Scope:** model/hardware/workload profiling, performance identity and persistence,
and explainable adaptive recommendations across supported GGUF models.
**Native checkout:** `7fe450e…`, unmodified.

## 1. Audit summary

Already generic:

- `MachineProfile` CPU/RAM/GPU/backend/storage observation.
- `inspect_architecture`, `expert_accounting`, `expert_layout`.
- `model_requirements` file/tensor/KV estimates.
- `RuntimePlan` requested/effective fields.
- `dense_residency` bounded staging decision.
- Speculative draft tokenizer/merge prefix validation.
- Benchmark reports with TTFT, throughput, I/O, PSS, mapping residency, staging,
  speculative stats.

Model-specific or incomplete:

- No single model profile combining parameters, quantization mix, tensor types,
  MoE structure, capabilities, and tokenizer facts.
- No hardware fingerprint for performance-profile identity.
- No reusable performance record identity or persistence.
- No advisory configuration selector from model/hardware/workload facts.
- `quantization_label` used an outdated file-type map: file type 15 reported
  `mostly-q8_k` for Q4_K_M models. Fixed to the pinned llama.cpp `llama_ftype`
  values (15 = `mostly-q4_k_m`).
- The 32B campaign's K=12/11-12 staged-layer choices were CLI experiments, not a
  model-agnostic policy.

## 2. Implemented model profile

`profiling::ModelProfile` is built from GGUF metadata and tensor descriptors:

- fingerprint (bounded header + file size), optional path
- architecture and `Dense` / `MixtureOfExperts` / `Unknown`
- parameter count (sum of tensor element counts)
- file bytes and tensor payload bytes
- tensor count, layer/embedding/head/KV-head/context metadata
- vocabulary size and tokenizer model
- expert count/used count
- primary quantization label and per-type quantization mix (tensor count/bytes)
- legacy/forgeCORE execution support and unsupported tensor formats
- routed expert bytes and estimated active weight bytes
- notes for missing metadata and MoE estimate caveats

`ramforge inspect --json` now includes `"profile": { ... }` alongside the existing
metadata and per-tensor details.

## 3. Hardware and workload profiles

`HardwareProfile` contains CPU model, physical/logical cores, total/available RAM,
OS/kernel, GPU list with backend/VRAM/free/selectable, registered backends, storage
kind, and measured storage read bandwidth when available. A stable hardware
fingerprint covers CPU topology, RAM, OS/kernel, GPU identities, and backends;
dynamic free-memory values are deliberately excluded from the identity.

`WorkloadProfile` contains context, batch, micro-batch, sequences, threads, GPU
layers, KV types, load mode, backend, device, residency mode, draft-model
fingerprint, draft length, max output, and RAM budget.

## 4. Performance identity and persistence

`PerformanceIdentity` joins:

- model fingerprint, architecture, quantization
- hardware fingerprint
- backend and device
- exact workload profile
- profiling/runtime schema

`is_reusable()` is false when the model fingerprint or hardware fingerprint is
missing. `key()` is a deterministic FNV display key; it is not a security hash.

`PerformanceRecord` stores median generation/decode/prompt throughput, TTFT, total
generation time, emitted tokens, target forwards, accepted/draft counts,
acceptance rate, storage reads, PSS, condition, and limitations.

`PerformanceProfileStore` is a small JSON store:

- `load` / `save` (atomic temp+rename)
- `insert` replaces only the exact identity
- `exact` requires full identity equality
- `best_for_model_hardware` is advisory: it may propose the best measured config
  for the same model and machine, but never pretends it matches a different
  workload

A benchmark can append a record:

```bash
ramforge benchmark MODEL.gguf ... --performance-profile /tmp/ramforge-profiles.json
```

This is the only persistence layer; no database or service was added.

## 5. Adaptive recommendation

`ramforge recommend` builds model, hardware, and workload profiles and emits an
advisory recommendation:

```bash
ramforge recommend MODEL.gguf --ram 10G --context 128 --threads 6 \
  --draft-model DRAFT.gguf --draft-tokens 12 \
  [--performance-profile /tmp/ramforge-profiles.json] [--json]
```

Rules are intentionally small:

1. Model + 512 MiB fits free VRAM -> recommend GPU placement, no host staging.
2. Model exceeds RAM and VRAM staging is possible -> recommend hybrid placement
   with a stage budget of free VRAM − 256 MiB.
3. Model fits RAM -> recommend CPU, no staging.
4. No usable GPU and model exceeds RAM -> recommend CPU with an explicit
   insufficient-VRAM reason.
5. Compatible draft + oversized target -> recommend speculative decoding with the
   requested/measured K (default 8); a compatible draft with a fitting target is
   not automatically recommended without a measured record.

The output separates `metadata-estimate` from `measured` evidence and includes the
best measured record if one exists. **Nothing is applied automatically.**
Recommendations are not promises of optimality.

## 6. Generalization validation

Real local assets inspected:

| Model | Architecture | Classification | Result |
|---|---|---|---|
| Qwen2.5-Coder-32B Q4_K_M | qwen2 | dense, 32.76B params | profile + recommend |
| Qwen2.5-1.5B Q4_0 | qwen2 | dense, 1.07 GB | profile + recommend |
| Qwen2.5-7B Q4_K_M | qwen2 | dense, 4.68 GB | profile + recommend |
| SmolLM2-135M Q4_K_M | llama | dense | profile + recommend |
| GPT-NeoX-20B Q4_K_M | gptneox | dense, 20.55B params, forgeCORE-only | profile + recommend |
| DeepSeek-Coder-V2-Lite Q4_K_M | deepseek2 | MoE, 64/6 experts, 15.7B params | profile + inspect |

This is **metadata/profile** validation across four architectures. Only Qwen2
models have real inference revalidation in this milestone; GPT-NeoX and DeepSeek2
were not re-benchmarked here.

The GPT-NeoX recommendation path correctly tolerated missing `head_count_kv`
metadata: KV accounting remains unknown, and the recommendation still reports
hybrid staging rather than failing or inventing a KV estimate.

## 7. Correctness of metadata and classification

- Fixed `quantization_label` mapping to the pinned `llama_ftype` enum, including
  the K-quant, IQ-quant, BF16, TQ, MXFP4/NVFP4, and Q1_0/Q2_0 values.
- Dense/MoE classification uses explicit GGUF expert metadata; incomplete expert
  metadata stays `Unknown`.
- Missing attention/vocabulary metadata produces notes, not guessed values.
- Parameters are exact tensor-element sums, not parameter-count formulas.
- Quantization mix is per tensor type with tensor counts and bytes.
- Active MoE bytes remain estimates and carry an explicit caveat.
- Recommendations distinguish exact measured records from metadata estimates.

## 8. Tests

- New unit tests: dense profile, MoE profile, quantization mix, exact identity
  reuse/non-reuse, JSON save/load roundtrip, fitting-model recommendation,
  oversized staging + draft-length recommendation, GPT-NeoX tolerant KV metadata
  path.
- Existing workspace tests pass; Clippy on touched crates reports no warnings.
- Real commands exercised: `inspect --json` profile on Qwen2/llama/gptneox/
  deepseek2; `recommend` on 32B/20B/SmolLM2; benchmark `--performance-profile`
  wrote one measured 32B record and `recommend` read it as `measured` evidence.

## 9. Limitations

- Recommendations are advisory and based on metadata + heuristic rules, not an
  optimizer.
- `best_for_model_hardware` intentionally ignores workload differences for
  advice; exact reuse still requires full identity equality.
- Hardware fingerprint composition was extended in the follow-up adaptive
  milestone: it now includes the execution environment (RAMforge version, native
  contract, registered backends, GPU driver, CPU feature identity). See
  `ADAPTIVE_2026-11-26.md`.
- Performance records are per exact workload; cross-model reuse is never
  automatic.
- Storage bandwidth is included only when `MachineProfile` performed a benchmark.
- MoE active-expert estimates are not measured residency.
- No automatic quality/performance tradeoff was introduced; quantization and
  sampling semantics are unchanged.
- No custom CUDA component was justified; profiling and policy are Rust-only.

## 10. Commands

```bash
# Profile a model
ramforge inspect MODEL.gguf --json

# Advisory recommendation, optionally using measured history
ramforge recommend MODEL.gguf --ram 10G --context 128 --threads 6 \
  --draft-model DRAFT.gguf --draft-tokens 12 \
  --performance-profile /tmp/ramforge-profiles.json --json

# Record a measurement
ramforge benchmark MODEL.gguf --ram 10G --context 128 --threads 6 \
  --batch 64 --ubatch 64 --kv-type-k q8_0 --kv-type-v q8_0 \
  --runs 1 --warmup 0 --max-tokens 8 --residency auto \
  --draft-model DRAFT.gguf --draft-tokens 12 \
  --performance-profile /tmp/ramforge-profiles.json --json
```

## 11. Next highest-value general work

1. **Use recommendations to drive a bounded benchmark sweep.** Measure a small
   set of candidate configs (K 4/8/12, staged vs not, target threads) and persist
   only measured winners. Do not auto-apply untested configs.
2. **Add driver/runtime version and backend build identity** to the hardware
   fingerprint.
3. **Cross-architecture inference validation** for gptneox and MoE after profile
   selection, not just metadata.
4. **Draft-model selection policy** based on compatibility, measured acceptance,
   host memory, and TTFT; no automatic download or model substitution.
5. **Quality guardrails** for any future KV/quantization/sampling tradeoff;
   performance profiles must record the quality condition.
6. **Model-specific calibration invalidation** when weights, driver, backend, or
   runtime schema changes.
