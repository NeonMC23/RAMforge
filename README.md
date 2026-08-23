# RAMforge

RAMforge is a local inference runtime designed to run AI models that may be significantly larger than the available RAM or VRAM by treating RAM, VRAM, and storage as a hierarchical memory system.

> **Milestone 5 Status (HEAD: M5.6.1):** Native GGUF quantized tensor support – Q4_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K. Quantized weights remain quantized while resident; dequantization happens block-wise during matvec. True out-of-core layer streaming preserved. CPU-only, llama/qwen2 dense models. GPU, MoE, HTTP not implemented.

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
- `ramforge plan`

### Milestone 3 – First Real CPU Inference
- Architectures `llama`, `qwen2`
- F32/F16/BF16, tokenizer, RMSNorm, RoPE, attention, SwiGLU, KV cache, sampling
- `ramforge run`

### Milestone 4 – Out-of-Core Layer Streaming
- Only persistent weights resident initially; layers loaded on demand → compute → release
- `ResidencyStats` proves total > budget while peak resident < total and peak managed ≤ budget

### Milestone 5 – Native Quantized Tensor Support (current)

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
- Example: Q4_K 256 elements F32 equiv 1024B, quantized resident 144B (7.1× smaller). For model with 4 layers n_embd 256 ffn 512:
  - Quantized total 1.5MB, F32 equiv 10MB
  - Budget 800KB, total quantized 1.5MB > budget, per-layer quantized 370KB fits, peak managed 429KB ≤ budget → inference succeeds with streaming
- Persistent weights (token_embd, output_norm, output) also accounted via quantized size; if quantized, they remain compact.

**Layer streaming integration:**
```
Layer descriptor → GgufDataSource::read_tensor() (quantized bytes) → TensorData::Q4_K/Q8_0/Q4_0 resident → quantized matvec (block dequant) → release layer
```
Entire quantized model never resident simultaneously. Layers released after compute.

**Compute backend:**
- `ComputeBackend` trait preserved, `CpuBackend` implements scalar quantized matvec via `quant::matvec_*`; F32 dot/matvec optionally use runtime-detected AVX2 (`simd.rs`) and rayon row-parallelism (`CpuBackend::with_threads`, auto-detected by `new()`)
- Future SIMD implementation can replace scalar without rewriting inference engine

## Build & Test

```bash
cargo build
cargo test --workspace
cargo clippy --workspace -- -D warnings
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
# Verbose shows:
# Total model weight bytes: 1500160 (1.43 MiB) quantized
# F32 equiv: 10507264 (10 MiB)
# Peak resident layer bytes: 370688 (0.35 MiB)
# Peak managed bytes: 429056 / budget 819200
# Fits check: total > budget ? true
```

Accepted `--ram` syntax: `8G`, `8GiB`, `8192M`, `512MiB`, `1.5G`, `KB`/`KiB`/`MB`/`MiB`/`GB`/`GiB`.

Diagnostics stderr, generated text stdout.

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
    datasource.rs – GgufDataSource
    tokenizer.rs – Tokenizer from GGUF
    quant.rs – Q4_0, Q8_0, Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K block layouts, dequant, quantized matvec (scalar, block-wise)
    tensor.rs – TensorData (F32/F16/BF16/Q4_0/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K), QuantizedTensor, resident_bytes, matvec, get_embedding
  ramforge-runtime/
    backend.rs – ComputeBackend, CpuBackend (rayon threading, optional AVX2 for F32)
    ops.rs – RoPE, attention
    kv_cache.rs – KV cache explicit, budget-accounted
    layer.rs – LayerDescriptor grouping
    residency.rs – ResidencyStats
    persistent.rs – PersistentWeight: resident if <25% of budget, else streamed on demand (M5.6.1)
    simd.rs – AVX2/FMA F32 dot/matvec kernels, runtime detection + scalar fallback (M5.6.1)
    model.rs – LlamaConfig, LlamaWeights validation
    streaming_model.rs – StreamingLlamaModel (persistent + layer descriptors), load_layer/release_layer, forward_single_streaming with quantized matvec
    inference.rs – InferenceEngine (file-backed + budget + cache + streaming + quantized), generate()
    plan.rs – planning
    sampling.rs – greedy, temperature, top-k/p
  ramforge-cli/ – inspect, plan, run --verbose
```

## Supported / Unsupported

**Supported architectures:** `llama`, `qwen2` (dense, same tensor naming)

**Supported tensor types:** `F32`, `F16`, `BF16`, `Q4_0`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` – all usable for inference

**Unsupported (clear error):**
- Other architectures → "unsupported architecture"
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
- Existing Milestone 4 streaming tests still pass

Total (M5.6.1): 62 core tests + 31 runtime tests = 93 tests.

## Known Limitations (Milestone 5)

- Quantized inference limited to Q4_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K; Q4_1/Q5_0/Q5_1/Q8_1 and IQ* quants not supported
- CPU-only: quantized matvec is scalar (block-wise dequant); F32 dot/matvec have optional runtime-detected AVX2 kernels and rayon row-parallelism (M5.6.1); no GPU
- Tokenizer: SentencePiece unigram (score-based Viterbi) and BPE (gpt2/qwen2 merges); other pre-tokenizers/model families untested
- Persistent weights (token_embd, output_norm, output): resident if under 25% of budget, otherwise streamed on demand (M5.6.1, `persistent.rs`)
- KV cache F32, no quantization, no eviction
- Minimum practical: individual layer working set must fit in budget

## What is NOT Implemented

- GPU, prefetch, double buffering, async I/O
- HTTP server, MoE, speculative decoding, model downloading
- Additional quantization formats beyond Q4_0/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K; no SIMD kernels for quantized matvec (F32 AVX2 only)

## License

MIT OR Apache-2.0
