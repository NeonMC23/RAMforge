# RAMforge

RAMforge is a local inference runtime designed to run AI models that may be significantly larger than the available RAM or VRAM by treating RAM, VRAM, and storage as a hierarchical memory system.

> **Milestone 3 Status:** First real CPU inference is implemented for LLaMA-compatible architecture (F32/F16). `ramforge run` performs actual transformer inference via RAMforge's own Rust code. GPU, MoE, HTTP API, and advanced out-of-core are not implemented yet.

## Purpose

- Provide a correct, reusable foundation for understanding GGUF model files
- Enable file-backed, out-of-core tensor access without loading entire models into RAM
- Enforce a real RAM budget for RAMforge-managed allocations
- Perform real CPU inference for supported architectures

## Current Capabilities

### Milestone 1 (still functional)
- Real GGUF parser (magic, header, metadata KV, tensor descriptors)
- File-backed model representation: tensor payloads NOT loaded during inspection
- Normalized helpers for architecture, context length, etc.

### Milestone 2 (still functional)
- **MemoryBudget**: exact byte accounting, named allocations with enforcement
- **Memory size parsing**: `8G`, `8GiB`, `8192M`, `512MiB`, etc.
- **File-backed tensor access**: `GgufDataSource` reads tensor bytes on demand
- **Bounded LRU cache**: capacity bytes, LRU eviction, stats
- **CLI `ramforge plan`**

### Milestone 3 (new) – Real CPU Inference
- **Supported architectures**: `llama` and `qwen2` (both use same dense transformer layout with `blk.{i}.attn_q.weight` etc.)
  - Documented: only `general.architecture = "llama"` or `"qwen2"` are accepted; others fail clearly
- **Supported tensor formats**: `F32`, `F16`, `BF16` (decoded to F32). Other types (quantized Q4_0, Q4_K, etc.) produce clear error: "unsupported tensor type for inference"
- **Tokenizer**: loads `tokenizer.ggml.model`, `tokens`, `scores`, `token_type`, `merges`, `bos_token_id`, `eos_token_id` from GGUF metadata. Implements naive longest-match encoding for `llama` (handles `▁` → space) and fallback byte handling. Detokenization concatenates tokens and replaces `▁` with space.
- **Inference pipeline**:
  1. tokenizer loading from GGUF
  2. prompt tokenization (with optional BOS)
  3. embedding lookup via `token_embd.weight`
  4. transformer layers (for each `blk.i`):
     - RMSNorm (`attn_norm`)
     - Q/K/V projections (`attn_q`, `attn_k`, `attn_v`)
     - RoPE (`rope.freq_base`, head_dim)
     - KV cache append
     - causal self-attention (with GQA support via `head_count_kv`)
     - output projection (`attn_output`) + residual
     - RMSNorm (`ffn_norm`)
     - SwiGLU FFN: `ffn_gate` (SiLU) * `ffn_up`, then `ffn_down` + residual
  5. final RMSNorm (`output_norm`)
  6. output projection (`output.weight` or tied `token_embd`)
  7. sampling (greedy when temperature=0, temperature, top-k, top-p)
  8. detokenization
  9. autoregressive loop with KV cache
- **KV cache**: explicit struct per layer, stores K/V as `Vec<f32>` sized `[max_seq_len * n_kv_heads * head_dim]`, grows via `append` + `increment_seq_len`, avoids recomputing previous tokens, memory usage accounted via `MemoryBudget` (`kv_cache` allocation), fails clearly if budget too small
- **CPU backend**: `CpuBackend` implements `ComputeBackend` trait with `matvec`, `rmsnorm`, `add`, `mul`, `silu`, `softmax`. Designed to allow future GPU backend (`ComputeBackend` → `CpuBackend` / future GPU)
- **Memory integration**: weights accessed through `GgufDataSource::read_tensor()` (file seek, not `std::fs::read` whole file), decoded via `decode_tensor_to_f32`, allocated from `MemoryBudget` (`weight:{name}`), cached in `BoundedCache` (raw bytes). Entire model never loaded as one giant buffer.
- **CLI `ramforge run`**: performs real inference

### Design for Memory Efficiency

- Parser reads only header/metadata/descriptors
- `GgufDataSource` reads only requested tensor bytes
- `BoundedCache` bounds memory, LRU eviction
- `MemoryBudget` tracks RAMforge-managed memory (tensor cache, KV cache, weights), NOT total RSS
- Inference loads individual tensors via data source, not whole file

## Build

```bash
cargo build
cargo test
cargo clippy --workspace -- -D warnings
```

## Usage

Inspect:
```bash
cargo run -p ramforge-cli -- inspect /path/to/model.gguf
```

Plan:
```bash
cargo run -p ramforge-cli -- plan /path/to/model.gguf --ram 8G
```

Run real inference (CPU, LLaMA):
```bash
cargo run -p ramforge-cli -- run /path/to/model.gguf --ram 8G --prompt "Hello" --max-tokens 32
cargo run -p ramforge-cli -- run /path/to/model.gguf --ram 8G --prompt "Explain what a computer is." --max-tokens 32 --temperature 0.7
```

Supported `run` options:
- `--ram <SIZE>`: RAM budget (e.g. 8G, 512MiB)
- `--prompt <TEXT>`: prompt
- `--max-tokens <N>`: default 32
- `--temperature <FLOAT>`: 0 = greedy, default 0
- `--top-k <K>`: optional
- `--top-p <P>`: optional

Diagnostics go to stderr, generated text to stdout.

Example output:
```
Model: /path/to/model.gguf
RAM budget: 8G (8589934592 bytes)
...
Model config: vocab=32000, context=2048, embedding=4096, layers=32, ...
Tokenizer: model=llama, vocab_size=32000, bos=Some(1), eos=Some(2)
Execution backend: CPU
...

Generated text...
```

## Project Structure

```
crates/
  ramforge-core/      # GGUF parsing, MemoryBudget, BoundedCache, GgufDataSource, Tokenizer, tensor decoding (F32/F16)
  ramforge-runtime/   # Runtime, plan, CPU backend, ops (RoPE, attention), KV cache, sampling, LLaMA model loading & inference
  ramforge-cli/       # inspect, plan, run commands
```

## Supported / Unsupported

**Supported architecture:** `llama`, `qwen2` (dense transformer, same tensor naming)

**Supported tensor types:** `F32`, `F16`, `BF16`

**Unsupported (clear error):**
- Other architectures (e.g. `bert`, `gpt2` architecture) → "unsupported architecture"
- Quantized types (Q4_0, Q4_K, etc.) → "unsupported tensor type for inference"
- Missing tensors → "missing tensor 'blk.0.attn_q.weight'"

**CPU-only:** No CUDA/Metal/Vulkan

## Testing

- Unit tests for tokenizer, F32/F16 decoding, matvec, RMSNorm, SiLU, softmax, RoPE, attention, KV cache, sampling, memory parsing, budget enforcement, cache LRU, file-backed range reads
- End-to-end inference test with tiny deterministic LLaMA GGUF (vocab 16, n_embd 8, 1 layer) – verifies deterministic greedy output for fixed prompt

## Known Limitations

- Only F32/F16/BF16 supported, no quantization
- Only llama/qwen2 dense models, no MoE
- KV cache allocated for needed length (prompt+max_tokens) to save memory, not full context_length
- Simple CPU matvec, no SIMD optimization
- Tokenizer is naive longest-match, not fully equivalent to llama.cpp SentencePiece but functional
- Does not yet implement advanced out-of-core layer-by-layer eviction for huge models – small model must fit in cache, but architecture remains file-backed

## What is NOT Implemented Yet

- GPU support (CUDA, Vulkan, Metal)
- MoE routing, speculative prefetching
- HTTP server, OpenAI API
- TUI, model downloading
- Advanced KV cache eviction

## License

MIT OR Apache-2.0
