# RAMforge

RAMforge is a local inference runtime designed to run AI models that may be significantly larger than the available RAM or VRAM by treating RAM, VRAM, and storage as a hierarchical memory system.

> **Milestone 4 Status:** True out-of-core layer streaming implemented. Model weights can exceed RAMforge-managed budget. Layers are loaded on demand, computed, then released. Real CPU inference for LLaMA/Qwen2 (F32/F16) with KV cache and budget enforcement. GPU, MoE, HTTP still not implemented.

## Purpose

- GGUF parsing without loading tensor payloads
- File-backed tensor access via `GgufDataSource`
- Real RAM budget enforcement via `MemoryBudget`
- Bounded LRU cache via `BoundedCache`
- Real CPU inference with layer streaming for models larger than RAM

## Capabilities

### Milestone 1 – GGUF Inspection
- Magic, header, metadata KV, tensor descriptors, file offsets, byte lengths

### Milestone 2 – Memory Budget & File-Backed Access
- `MemoryBudget` with named allocations, exact byte accounting
- `parse_memory_size()` accepts `8G`, `8GiB`, `8192M`, `512MiB`, `1.5G`, etc.
- `GgufDataSource` reads tensors/ranges on demand
- `BoundedCache` LRU with stats
- `ramforge plan`

### Milestone 3 – First Real CPU Inference
- Architectures: `llama`, `qwen2`
- Tensor types: `F32`, `F16`, `BF16`
- Tokenizer from GGUF metadata (naive longest-match, `▁` handling)
- Transformer: RMSNorm, RoPE, causal attention with GQA, SwiGLU FFN, sampling
- KV cache explicit and budget-accounted
- `ramforge run`

### Milestone 4 – True Out-of-Core Layer Streaming (current)

**Defining feature:** Model weights (e.g. 4GB) can be larger than RAMforge budget (e.g. 1GB) and still run by streaming layers.

**Execution model:**
```
GGUF on disk
  ↓ metadata
Layer 0 weights → RAMforge managed memory → CPU compute Layer 0 → Release Layer 0
  ↓
Layer 1 weights → compute → Release
...
Final norm / output → logits → next token
```

**What changed from Milestone 3:**
- Removed assumption that all weights are loaded in `LlamaModel::load()`. Now only persistent weights (`token_embd`, `output_norm`, `output`) are loaded initially.
- Introduced layer-oriented representation: `LayerDescriptor` groups `blk.{i}.*` tensors, `PersistentDescriptors` for non-layer tensors.
- Introduced `StreamingLlamaModel` with `load_layer()` and `release_layer()` – each layer's decoded size allocated from `MemoryBudget` as `layer:{i}:{tensor}`, tracked in `ResidencyStats`, released immediately after compute.
- Forward pass `forward_single_streaming()` loads one layer, computes, releases, next layer – entire stack never resident simultaneously.
- Added `ResidencyStats`: total model weight bytes, current/peak resident layer bytes, num loads/releases, peak managed bytes.
- Added `--verbose` to `run` to expose residency stats.

**Memory accounting:**
- RAMforge-managed memory = memory tracked via `MemoryBudget` (persistent weights, currently resident layer, KV cache). NOT total RSS or OS page cache.
- Every streamed layer allocation goes through `MemoryBudget`; release via `release()`.
- `BoundedCache` still used for raw tensor bytes (may evict), but does not retain every layer – policy is load-compute-release.
- KV cache remains resident across generation, allocated from budget based on needed length (`prompt_len + max_tokens`) to save memory vs full context.

**Verification model:**
Synthetic GGUF generator creates models where total weights > budget but per-layer fits. Example from tests and manual run:

```
Model weights:       42560 bytes (41KB, 4 layers, n_embd 16, ffn 32)
RAM budget:          16384 bytes (16KB)
Cache capacity:      8192 bytes (8KB)
Peak layer residency: 10368 bytes (10KB)
Peak managed memory: 14016 bytes (13KB) <= budget
Layer loads:         20 (4 layers * 5 tokens)
Layer releases:      20
Result:              Inference succeeds with streaming
```

Proves:
- `total_model_weight_bytes > configured_budget`
- `peak_resident_layer_bytes < total_model_weight_bytes`
- `peak_managed_bytes <= configured_budget`

## Build & Test

```bash
cargo build
cargo test --workspace
cargo clippy --workspace -- -D warnings
```

## Usage

Inspect (still works):
```bash
cargo run -p ramforge-cli -- inspect model.gguf
```

Plan (still works):
```bash
cargo run -p ramforge-cli -- plan model.gguf --ram 8G
cargo run -p ramforge-cli -- plan model.gguf --ram 256M --verbose
```

Run with real inference and layer streaming:
```bash
cargo run -p ramforge-cli -- run model.gguf --ram 8G --prompt "Hello" --max-tokens 32
cargo run -p ramforge-cli -- run model.gguf --ram 256M --prompt "Hello" --max-tokens 16 --verbose
cargo run -p ramforge-cli -- run model.gguf --ram 8G --prompt "Explain what a computer is." --max-tokens 32 --temperature 0.7
```

Accepted `--ram` syntax: `8G`, `8GiB`, `8192M`, `512MiB`, `1.5G`, `1024`, `KB`/`KiB`/`MB`/`MiB`/`GB`/`GiB`/`TB`/`TiB`.

Diagnostics to stderr, generated text to stdout.

**Out-of-core example:**
```bash
# Synthetic model 45KB total, 42KB weights, budget 16KB
cargo run -p ramforge-cli -- run synthetic.gguf --ram 16K --prompt "hello" --max-tokens 3 --verbose
# Shows:
# Total model weight bytes: 42560
# Peak resident layer bytes: 10368
# Peak managed bytes: 14016 / budget 16384
# Fits check: total > budget ? true
# Peak layer < total ? true
# Peak managed <= budget ? true
```

## Project Structure

```
crates/
  ramforge-core/      # GGUF parsing, MemoryBudget, BoundedCache, GgufDataSource, Tokenizer, tensor decoding
  ramforge-runtime/   # backend, ops, kv_cache, layer grouping, residency stats, streaming_model, inference (streaming), plan, sampling
  ramforge-cli/       # inspect, plan, run (with --verbose)
```

## Supported / Unsupported

**Supported:**
- Architectures: `llama`, `qwen2` (dense, same tensor naming)
- Tensor types: `F32`, `F16`, `BF16`
- CPU backend only

**Unsupported (clear error):**
- Other architectures → "unsupported architecture"
- Quantized types Q4_0, Q4_K, etc. → "unsupported tensor type"
- Missing tensors → "missing tensor 'blk.0.attn_q.weight'"
- Budget too small for layer or KV cache → clear insufficient memory error

## Known Limitations (Milestone 4)

- Only F32/F16/BF16, no quantization (would be needed for real large models)
- Persistent weights (token_embd, output) kept resident; if they exceed budget, they would need streaming too (documented, not yet implemented)
- No prefetch, double buffering, SIMD, multithreading
- KV cache no eviction
- Tokenizer naive, not fully equivalent to llama.cpp but functional for tests
- Minimum practical requirement: individual layer working set must fit in available managed memory

## What is NOT Implemented

- GPU (CUDA, Metal, Vulkan)
- MoE, speculative decoding
- HTTP server, OpenAI API
- TUI, model downloading
- Quantization, async I/O

## License

MIT OR Apache-2.0
