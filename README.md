# RAMforge

RAMforge is a local inference runtime designed to run AI models that may be significantly larger than the available RAM or VRAM by treating RAM, VRAM, and storage as a hierarchical memory system.

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
`d3203099a315db134193404e79ba2581d9fc53ab` (`v3.1.2`) and keeps direct
`forge-sys` and llama.cpp/ggml FFI out of RAMforge.

Phase 2 does not claim GPU, out-of-core execution, controlled residency,
ForgeCore-native memory accounting, or migration of the legacy numerical
modules. RAMforge GGUF inspection, planning, diagnostics, cache, residency,
and accounting infrastructure remains retained; ForgeCore native model,
context, KV, and backend allocations remain outside RAMforge accounting, while
safe device facts are reported only where the adapter can observe them.
Runtime linking follows ForgeCore's native runbook; arrange
`LD_LIBRARY_PATH=$FORGE_LLAMA_DIR/build/bin` (or an equivalent final-binary
rpath) for the CPU shared libraries.

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
Verified ForgeCore `v3.1.2` exposes public static model-load/offload options,
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
- Architectures `llama`, `qwen2`
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
- **Capability registry:** `ramforge support` distinguishes generic GGUF inspection/planning from execution, tokenizer, and quantization support. Direct execution remains limited to `llama` and `qwen2`; Mistral is runnable only when represented by a validated llama-compatible GGUF; `qwen35` is explicitly inspect/plan-only
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

**Supported execution architectures:** `llama`, `qwen2` (dense). Mistral works only when represented by a validated llama-compatible GGUF architecture/tensor layout. `qwen3`, `qwen35`, `gemma`, and `phi` are inspect/plan-only registry entries.

**Supported tensor types:** `F32`, `F16`, `BF16`, `Q4_0`, `Q5_0`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` – all usable for inference

**Unsupported (clear error):**
- Unsupported execution architectures → capability-aware error listing detected architecture, inspect/plan status, and supported execution architectures
- Other quantized types (Q4_1, Q5_1, Q8_1, IQ*, …) → "unsupported tensor type for inference"
- Missing tensors → "missing tensor 'blk.0.attn_q.weight'"
- Budget too small → "RAM budget too small for layer..."

**CPU-only:** No GPU

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

The current full workspace run passes **423 tests, 0 failed, 2 ignored** (the real-model Qwen2.5 diagnostic and the quantized microbenchmark) using the shared native fixture. Run `scripts/test.sh --workspace --locked` to reproduce it.

## Real-model smoke validation

`scripts/smoke-real-models.sh` accepts a single GGUF file, a split GGUF shard, or a model directory containing one model's shard set. Model files remain read-only. Each case runs in a separate process, waits for exit, checks that `MemAvailable` is above a configurable threshold (4 GiB by default), and records the exact command, exit status, `/usr/bin/time -v` facts, per-file model sizes/sha256 hashes, system memory and swap samples, GPU state, and captured output. Results are written as TSV and JSON outside the repository.

```sh
cd RAMforge
scripts/smoke-real-models.sh              # four 4 GiB smoke cases
scripts/smoke-real-models.sh --no-build   # reuse target/debug/ramforge
scripts/smoke-real-models.sh --matrix --include-7b --timeout 900
```

The opt-in matrix uses exact byte budgets `8G` = 8,589,934,592 B, `4G` = 4,294,967,296 B, `1G` = 1,073,741,824 B, and `100M` = 104,857,600 B. It keeps prompt, generation length, sampling, and backend fixed within each comparison. A controlled budget rejection is recorded without failing the suite; timeouts and signals still fail. The latest evidence directory is `/tmp/ramforge-real-model-smoke/20261024-193321`.

| Model / case | Budget | Exit | Wall | Max RSS | Managed peak | Cache evictions |
|---|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M Q4_K_M legacy | 8G | 0 | 7.36 s | 135.5 MiB | 99.19 MiB | 0 |
| SmolLM2-135M Q4_K_M legacy | 4G | 0 | 7.33 s | 135.3 MiB | 99.19 MiB | 0 |
| SmolLM2-135M Q4_K_M legacy | 1G | 0 | 7.42 s | 135.3 MiB | 99.19 MiB | 0 |
| SmolLM2-135M Q4_K_M legacy | 100M | 0 | 7.66 s | 114.0 MiB | 70.50 MiB | 0 |
| Qwen2.5-1.5B Q4_0 legacy | 8G | 0 | 8.69 s | 1124.6 MiB | 1012.05 MiB | 0 |
| Qwen2.5-1.5B Q4_0 legacy | 4G | 0 | 8.70 s | 1124.7 MiB | 1012.05 MiB | 0 |
| Qwen2.5-1.5B Q4_0 legacy | 1G | 0 | 8.98 s | 1100.8 MiB | 1012.30 MiB | 85 |
| Qwen2.5-1.5B Q4_0 legacy | 100M | 0 | 23.71 s | 165.3 MiB | 76.48 MiB | 110 |
| Qwen2.5-7B Q4_K_M, two shards, legacy | 4G | 0 | 5:18.67 | 4179.6 MiB | 4086.50 MiB | 88 |
| SmolLM2-135M Q4_K_M ForgeCore CPU | 4G | 0 | 0.25 s | 365.9 MiB | not exposed | n/a |
| Qwen2.5-1.5B Q4_0 ForgeCore CPU | 4G | 0 | 1.23 s | 2772.9 MiB | not exposed | n/a |

- All cases generated 4 tokens with a fixed `Hello` prompt and greedy sampling. Legacy `Swaps` was `0` in every case; the suite also records system-wide swap samples, which can move for unrelated reasons and are not process swap attribution.
- The 100M Qwen2.5 run demonstrates that a budget below the total model footprint can still stream successfully when one layer and the mandatory workspaces fit. It is not a claim that 100M fits every model.
- A Qwen2.5-7B run at 100M fails before inference with `managed lower bound 149127168 exceeds RAMforge budget 104857600`, exit 1, in 0.41 s and 92.3 MiB max RSS. This is the intended controlled failure, not an OOM.
- The real 20B `gptneox` GGUF (13,135,032,320 B) is inspectable and plannable at 100M, but `run` rejects it before loading weights: `unsupported execution architecture 'gptneox': inspection=true planning=true execution=not-yet`. Exit 1 in 0.02 s, 26,148 KiB max RSS, `Swaps: 0`; no fallback or false success.
- The 7B split model is discovered from its directory and all 339 tensors are read through the merged virtual data source. Its 4G run retains 24 layers at peak with 112 loads and 88 evictions. `Peak managed bytes` stays 9,959,680 B below the 4 GiB budget.
- ForgeCore's requested budget remains diagnostic only and is **not enforced**; its RSS includes loader, page cache, Rust heap, and native allocations. The native build is CPU-only, so these runs are not GPU validation.
- Model inputs are unchanged; per-shard SHA-256 values are recorded in `model-inventory.tsv` and each case's `model-files.tsv`.

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
- Chat templates are interpreted by a conservative Jinja subset, not a full Jinja2 engine. Unsupported filters/tests/constructs fail explicitly. `--chat` is opt-in and applies to one user message per invocation.
- Qwen2.5-7B processed real split-shard inference at 4G, but its prompt processing is very slow; a chat-templated 7B run exceeded a 900 s harness timeout before emitting a token.
- Quantized inference limited to Q4_0, Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K; Q4_1/Q5_1/Q8_1 and IQ* quants not supported
- CPU-only: quantized matvec is scalar (block-wise dequant); F32 dot/matvec have runtime-detected AVX2 kernels and rayon row-parallelism; no GPU
- Tokenizer: SentencePiece unigram (score-based Viterbi) and BPE (gpt2/qwen2 merges); other pre-tokenizers/model families untested
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
