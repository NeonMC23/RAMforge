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
> **Milestone 6 Status (True Out-of-Core Integrity):** Every RAMforge allocation is charged to the RAM budget via RAII-style scoped guards; the cache is budget-charged per entry; matrix layout is the explicit GGML/GGUF convention (no orientation guessing, no full-F32 fallbacks); logits use a single budget-charged buffer with a budget-aware chunked streamed projection; the KV cache grows chunk-wise without prefix copies; the F32 matvec hot path uses AVX2/rayon. CPU-only, llama/qwen2 dense models. GPU, MoE, HTTP not implemented.

## Purpose

- GGUF parsing without loading payloads
- File-backed tensor access via `GgufDataSource`
- Real RAM budget enforcement via `MemoryBudget`
- Bounded LRU cache via `BoundedCache`
- Real CPU inference with layer streaming
- Native quantized inference without full F32 expansion

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
- `Q8_0`: block 32, 34 bytes (2B half scale `d` + 32B int8 quants, dequant `d*q`)
- `Q4_K`: block 256, 144 bytes (2B half `d`, 2B half `dmin`, 12B scales (8 scales + 8 mins packed 6-bit), 128B 4-bit quants; unpack via `get_scale_min_k4(j)`, dequant `d*sc*q - dmin*m`)
- Since M5.6.1: `Q2_K`, `Q3_K`, `Q5_K`, `Q6_K`, `Q8_K` block layouts, `dequantize_row_*`, and `matvec_*` are also implemented in `quant.rs` with the same resident-compact representation. `Q4_1`, `Q5_0`, `Q5_1`, `Q8_1`, `IQ*` remain unsupported.

**Representation:**
- `TensorData` enum: `F32`, `F16`, `BF16`, `Q4_0(QuantizedTensor)`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`
- `QuantizedTensor { ggml_type, shape, num_elements, raw_data: Vec<u8> }` keeps quantized compact while resident
- `resident_bytes()` = raw_data.len() (e.g. 144B for 256 values Q4_K vs 1024B F32) – shows memory saving
- `matvec()` for quantized does block-wise dequant: for each output row, for each quantized block, decode block to temp `[32]` or `[256]` f32, dot with x slice, discard temp. Working set bounded to one block.
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
- **Real profiling:** `--profile` measures generation wall time, prompt/prefill, prompt/decode forward counts, logical per-tensor reads/bytes, physical GGUF reads/bytes, coalesced ranges/gap overhead, the GGUF read path (destination allocation plus seek/read on a reused handle), layer loading/compute/release, tensor construction, explicit dequantization/copies, F32 and quantized matvec, allocations, logits, sampling, callback time, token latency, and layer-cache hits/misses/evictions/current/peak residency
- **Memory visibility:** `--memory-report` labels RAMforge current/peak/budget separately from Linux process RSS and system memory; MemoryBudget does not claim control over RSS or page cache
- **Capability registry:** `ramforge support` distinguishes generic GGUF inspection/planning from execution, tokenizer, and quantization support. Direct execution remains limited to `llama` and `qwen2`; Mistral is runnable only when represented by a validated llama-compatible GGUF; `qwen35` is explicitly inspect/plan-only
- **Execution memory preflight:** for runnable architectures, `ramforge plan` applies the runtime’s persistent-retention, decoded-residency, and per-tensor load-transient rules to report the largest layer and a necessary managed-memory lower bound. It explicitly excludes prompt-dependent KV and activation workspaces, so it is not presented as a sufficient runtime guarantee.
- **Bounded streamed-layer cache:** recently used decoded layer representations may remain resident under an explicit byte capacity derived from the selected budget and largest-layer headroom. Every cache entry keeps its MemoryBudget charge, LRU eviction releases it exactly, and mandatory KV/activation/output workspaces can force safe eviction. Hits avoid GGUF rereads; planner capacity is an estimate and does not guarantee hits.
- **Bounded streamed-layer read coalescing:** descriptor-sorted tensors that are adjacent or separated by at most 4 KiB may share a physical read only while the full span stays at or below 64 MiB and its grouped raw-plus-resident workspace fits MemoryBudget. Arbitrary ordering and explicit tensor boundaries remain authoritative; large gaps/spans and low-headroom cases use the original individual read path. No mmap or persistent raw payload is involved.

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

- **Memory accounting:** `MemoryBudget::with_temp(name, bytes, f)` is the RAII-style scoped guard for transient working sets (`tmp:hidden`, `tmp:forward`, `tmp:embd_row`, `tmp:streamed_matvec`, `tmp:logits`, `tmp:sampling`) – released on success and on error. Layer tensors charge *before* reading (peak = settled prefix + per-tensor transient) and settle to exact resident bytes after construction; a failed layer load releases all its charges.
- **Cache:** `BoundedCache::insert_budgeted` charges each cached entry to the budget (`cache:{key}`), evicting LRU entries to make budget room; if nothing can be evicted, the entry is simply not cached (streaming keeps working) instead of failing or double-counting.
- **Matrix layout:** one explicit GGML/GGUF convention everywhere: `shape = [in, out]`, buffer row-major `[out][in]`, `y[o] = Σ_i W[o·in+i]·x[i]`. No orientation heuristics, no transpose fallbacks, no full-F32 dequantization of 2D weights – arity mismatches are hard errors.
- **Output projection:** single caller-owned logits buffer per `generate()` call; streamed (non-resident) output/embedding matrices are projected in budget-bounded row chunks (`min(16 MiB, available/4)`, ≥ 1 row) with per-row block decode.
- **KV / attention:** attention reads the KV history in place (no per-token prefix copies); the KV cache starts at the prompt length and grows in 256-token chunks capped at prompt+max_tokens, budget-checked with rollback on failure. No KV quantization.
- **Legacy removal:** the pre-M4 fully-resident F32 model loader (`LlamaModel`) was deleted – it violated budget integrity, guessed orientation, and duplicated the KV prefix.

## Build & Test

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
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
    datasource.rs – GgufDataSource, exact range reads, direct little-endian F32 reads into final storage
    tokenizer.rs – Tokenizer from GGUF
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

**Supported tensor types:** `F32`, `F16`, `BF16`, `Q4_0`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` – all usable for inference

**Unsupported (clear error):**
- Unsupported execution architectures → capability-aware error listing detected architecture, inspect/plan status, and supported execution architectures
- Other quantized types (Q4_1, Q5_0, Q5_1, Q8_1, IQ*, …) → "unsupported tensor type for inference"
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
- M6 integrity proofs: RAII temp release on success/error (`memory.rs`), budgeted cache inserts/evictions (`cache.rs`), explicit ggml layout incl. non-square Q4_0/Q4_K/F32/F16 anchors and arity-error rejections (`tensor.rs`, `backend.rs`), chunked streamed output projection + too-small-budget failure (`persistent.rs`), no-copy attention vs naive reference (`ops.rs`), chunk-growing KV preserving data with exact bytes (`kv_cache.rs`), end-to-end out-of-core inference with model > budget (`inference.rs`)

The last executed baseline passed 158 tests with 0 failures and 0 ignored. The current source inventory is 174 tests after adding shared-accounting, execution-preflight, bounded layer-cache, and bounded coalesced-read regressions; rerun the full suite before commit.

## Known Limitations

- Current Q4_K_M CPU inference is functional but extremely slow in the measured NAS-backed Mistral run; profiling identifies both GGUF reads and scalar quantized matvec/dequantization as major costs. Performance optimization remains measurement-driven.
- The layer cache is generation-local and strict LRU. If its complete-layer capacity is smaller than a sequential full-model working set, scan-pattern thrashing can produce few or no hits; planner capacity never guarantees hit rate.
- Read coalescing is opportunistic and bounded. Large tensors/gaps, layouts exceeding the 64 MiB span cap, or insufficient grouped-buffer headroom retain one physical read per tensor; fewer reads do not guarantee lower NAS wall time until benchmarked.
- Real-model execution and streaming are validated, but generated-text quality is not yet validated.
- Quantized inference limited to Q4_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K; Q4_1/Q5_0/Q5_1/Q8_1 and IQ* quants not supported
- CPU-only: quantized matvec is scalar (block-wise dequant); F32 dot/matvec have runtime-detected AVX2 kernels and rayon row-parallelism; no GPU
- Tokenizer: SentencePiece unigram (score-based Viterbi) and BPE (gpt2/qwen2 merges); other pre-tokenizers/model families untested
- Persistent weights (`token_embd`, `output_norm`, `output`): resident when their actual retained representation is at most 25% of budget, otherwise streamed on demand with budget-charged bounded temps (`persistent.rs`)
- KV cache F32, no quantization, no eviction; grows chunk-wise up to prompt+max_tokens
- Minimum practical: one streamed layer plus its charge-before-read transient (quantized 1× file bytes, direct F32 1×, F16/BF16 3×) and the forward working set must fit the budget; one direct F32 output row, or one raw converted/quantized row plus its F32 decode buffer, must fit as well
- Not budget-tracked by design (documented as out of scope): tokenizer vocabulary table, thread stacks, allocator fragmentation, residency bookkeeping, and optional profiling metadata (including O(tensors) read counters)

## What is NOT Implemented

- GPU, prefetch, double buffering, async I/O
- HTTP server, MoE, speculative decoding, model downloading
- Additional quantization formats beyond Q4_0/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K; no SIMD kernels for quantized matvec (F32 AVX2 only)

## License

MIT OR Apache-2.0
