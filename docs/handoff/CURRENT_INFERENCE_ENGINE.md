# Current inference engine

This is a technical description of the existing RAMforge inference path. It is not a recommendation that every numerical choice should be retained. The Qwen2.5 evidence in [QWEN25_CORRECTNESS_INVESTIGATION.md](QWEN25_CORRECTNESS_INVESTIGATION.md) is the reason ForgeCore is proposed.

## Entry points and ownership

The main runtime type is `ramforge_runtime::inference::InferenceEngine` in `crates/ramforge-runtime/src/inference.rs`. It owns:

- `GgufDataSource` — parsed descriptors plus file-backed reads;
- `Tokenizer` — loaded from GGUF tokenizer metadata;
- `StreamingLlamaModel` — persistent weights, layer descriptors/cache, load policy, and model execution;
- optional `KvCache`;
- `CpuBackend`;
- `MemoryBudget`;
- `RuntimeConfig` and `ResidencyStats`.

Construction is available through `InferenceEngine::new`, `InferenceEngine::new_with_runtime_config`, and the orchestration-only `from_data_source_with_runtime_config`. The model is inspected and tokenizer metadata loaded before tensor payloads are streamed. The current public generation APIs are:

```text
InferenceEngine::generate(prompt, max_tokens, sampler)
InferenceEngine::generate_with_callback(prompt, max_tokens, sampler, on_text)
```

The callback receives only newly decodable generated text. CLI diagnostics go to stderr while assistant text is written to stdout.

## Tensor and matrix conventions

The repository now states one explicit GGML/GGUF convention:

- A two-dimensional GGUF tensor shape is `[in_features, out_features]`, corresponding to `ne[0]`, `ne[1]`.
- Materialized weights are contiguous row-major `[out][in]`.
- A matvec computes `y[o] = sum_i W[o * in + i] * x[i]`.
- `MatrixShape::new(input, output)` in `ramforge_core::compute` represents this contract.
- Arity mismatches are errors. The current reference/backend paths do not transpose or guess an alternative orientation.

The output projection follows the same convention: `output.weight` is logically `[n_embd, vocab]`, with one contiguous input-width row per vocabulary output. A non-resident output projection is evaluated in budget-bounded row chunks rather than expanded into a full F32 matrix.

The KV cache layout is separate and explicit:

```text
[position][kv_head][head_dim]
```

Each layer has flattened K and V buffers. `seq_len` is the number of committed historical positions. A layer execution reads `[0, history_len)` and returns current K/V separately; the streaming caller stages the current K/V for every layer and commits them after the complete layer sequence has consumed the old history.

For grouped-query attention, `query_heads` must be divisible by `kv_heads`. The current reference mapping is:

```text
kv_head = query_head * kv_heads / query_heads
```

The current `AttentionConfig` in `ramforge-core/src/compute.rs` carries `history_len`, `query_heads`, `kv_heads`, and `head_dim`, and documents Q/current-K/current-V versus history layouts.

## Tensor representations

`ramforge_core::tensor::TensorData` has these inference variants:

```text
F32 { data: Vec<f32>, shape }
F16 { data: Vec<f32>, shape }
BF16 { data: Vec<f32>, shape }
Q4_0(QuantizedTensor)
Q8_0(QuantizedTensor)
Q4_K(QuantizedTensor)
Q5_K(QuantizedTensor)
Q6_K(QuantizedTensor)
Q2_K(QuantizedTensor)
Q3_K(QuantizedTensor)
Q8_K(QuantizedTensor)
```

F32, F16, and BF16 are decoded to F32 vectors for execution. F32 data can use `GgufDataSource::read_f32_tensor_range_by_descriptor`, which reads directly into final F32 storage without a raw-byte intermediate; on little-endian hosts the bytes already have native F32 layout. F16/BF16 use a decoded F32 representation and therefore have a larger load transient than their file representation.

Quantized tensors retain their raw compact byte buffer while resident. `QuantizedTensor::matvec` uses the pure reference path, while runtime dispatch can use format-specific or fused paths without materializing a whole matrix. Embedding lookup reads/dequantizes one requested row when a persistent embedding is streamed.

## Quantized formats and implementation locations

The authoritative block layouts and scalar decoders are in `crates/ramforge-core/src/quant.rs`. The `TensorData` and row-decoding integration is in `crates/ramforge-core/src/tensor.rs`; persistent and streamed row handling is in `crates/ramforge-runtime/src/persistent.rs`; runtime dispatch is in `crates/ramforge-runtime/src/compute_dispatch.rs`.

| Format | Values/block | Encoded block bytes | Current implementation notes |
|---|---:|---:|---|
| `Q4_0` | 32 | 18 | half scale plus packed 4-bit values; scalar/fused row-dot and row-range paths |
| `Q8_0` | 32 | 34 | half scale plus signed 8-bit values; scalar row decode/matvec |
| `Q4_K` | 256 | 144 | `d`, `dmin`, packed scales/mins, 4-bit values |
| `Q5_K` | 256 | 176 | K-quant scales/mins, low 4-bit values, high bits |
| `Q6_K` | 256 | 210 | 16 int8 scales, low/high quant bits, half scale; fused row-dot path |
| `Q2_K` | 256 | 84 | K-quant low-bit scales/mins and 2-bit values |
| `Q3_K` | 256 | 110 | K-quant high mask, low 2-bit values, packed scales |
| `Q8_K` | 256 | 292 | F32 scale, int8 values, block sums |

These byte counts are source-level format contracts. The current Qwen2.5 investigation specifically verified `output.weight` as `Q6_K` with a 1536-wide row and 1260 bytes per row; that is separate from the model filename's `Q4_0` designation and is discussed in detail in the investigation document.

The runtime retains optimized Q4_0/Q6_K row-parallel dispatch for large output-row counts. `ComputeBackend` uses Rayon partitioning and optional AVX2 only for F32 operations; there is no general SIMD quantized matvec implementation in the current source.

## Transformer operations

### RMSNorm

`ramforge_core::compute::rms_norm_reference` validates dimensions and epsilon, computes an F32 sum of squares in input order, forms the RMS denominator, and writes `x / rms * weight`. The runtime backend delegates to this scalar/reference operation. The same operation is applied before attention projections and before the feed-forward projections.

### Q/K/V projections and Qwen2 biases

`ModelExecutor::execute_layer` obtains an F32 view for the attention norm and computes Q/K/V through `matvec_backend`. If qwen2-style `attn_q.bias`, `attn_k.bias`, and `attn_v.bias` are present, all three must be present and shape-valid. Biases are added after the projections and before RoPE/KV-cache insertion. A partial set is rejected.

### RoPE

`ramforge_core::compute::rope_reference` and `ramforge-runtime/src/ops.rs::apply_rope` use half-split Llama/Qwen2 pairs. For a head of dimension `d`, index `j` rotates values at `j` and `j + d/2`; the documented angle is `position * base^(-2j/d)`. The current source explicitly distinguishes this from an interleaved/GPT-J pair convention.

### Attention and GQA

`ramforge-core/src/compute.rs::attention_reference` is the single scalar/reference attention definition. It validates nonzero dimensions, head divisibility, history/current tensor lengths, and computes causal attention over cached history plus current K/V. Scores use the head dimension scale, softmax uses a max-subtracted stable calculation, and GQA maps multiple query heads to one KV head according to `AttentionConfig`.

`ramforge-runtime/src/ops.rs::attention` is a runtime adapter that constructs `AttentionConfig` and delegates to the core reference. It is not a second core implementation. `ModelExecutor` supplies history slices from the `KvCache` and current K/V from its activation scratch.

### Attention residual and SwiGLU

After attention, the output projection is computed and added to the residual hidden state. The feed-forward path then:

1. applies FFN RMSNorm;
2. computes gate and up projections;
3. applies `SiLU` to the gate;
4. multiplies `SiLU(gate)` elementwise by `up`;
5. computes the down projection;
6. adds the result to the residual hidden state.

The pure core functions are `silu_reference`, `swiglu_reference`, `add_reference`, and `mul_reference`. Runtime backend methods delegate to these for scalar semantics.

### Output normalization and projection

`StreamingLlamaModel::compute_logits` obtains `output_norm` as a F32 view, applies RMSNorm to the final hidden state, then calls the persistent output weight's `compute_logits_into`. If `output.weight` is resident, its `TensorData` path is used. If streamed, the projection reads bounded row chunks and decodes one row at a time for non-F32 formats. A tied `token_embd.weight` can be used when there is no separate output weight.

## Streaming and cache interaction

`StreamingLlamaModel::forward_single_streaming` allocates a budget-charged activation workspace sized from the model dimensions and current sequence position. It obtains the token embedding from a resident tensor or a bounded streamed row read, then loops over layers:

1. look for a decoded layer in `LayerCache`;
2. on a hit, execute the cached `StreamingLayerWeights`;
3. on a miss, ensure budget headroom, load the layer with descriptor/read plans, execute it, and attempt a budget-charged cache insert;
4. if cache insertion is skipped or fails, release the active layer charge safely;
5. stage that layer's current K/V;
6. after all layers finish, append all staged K/V and increment the cache sequence once.

The code has tests for cache hits, eviction, grouped reads, grouped-buffer reuse, failure rollback, F32/F16/BF16 accounting, quantized compact residency, and layer/KV invariants. These are infrastructure and synthetic correctness tests; they are not an external numerical oracle for Qwen2.5.

`InferenceEngine::generate_with_callback` resets the previous KV and layer cache before each run, tokenizes the prompt with BOS behavior, performs the prompt pass, samples generated tokens, feeds the selected token to the next decode position only when needed, emits incrementally decodable text, and reports EOS/max-token/context termination. A successful run retains the KV cache and its matching `MemoryBudget` charge until explicit reset or the next generation call.

## Recent compute-core remodel

The structural remodel introduced or consolidated:

- `MatrixShape` and hard arity/layout validation in `ramforge_core::compute`;
- one `AttentionConfig` and one `attention_reference` in the compute crate;
- scalar reference functions for matvec, quantized matvec, RMSNorm, elementwise ops, SiLU/SwiGLU, softmax, RoPE, and attention;
- runtime adapters in `backend.rs`, `compute_dispatch.rs`, and `ops.rs` that delegate scalar semantics to the core;
- focused synthetic tests for nonsquare layouts, GQA mapping, RoPE identity, quantized row ranges, fused/reference parity, and model-executor flow.

### What succeeded structurally

**VERIFIED:** the source now has an explicit compute boundary, explicit matrix conventions, one authoritative attention configuration, shared accounting formulas, and passing synthetic/workspace tests. The final Rust validation also passes formatting, all workspace tests, strict Clippy, and the release build.

### What it did not prove

**OBSERVED:** the remodel did not solve the recorded numerical divergence for the external Qwen2.5-1.5B Q4_0 model. The available evidence shows a mismatch before autoregressive generation and a mismatch in direct Q6_K output-row comparisons. No external reference model or Qwen2.5 payload is present in this repository, and the real diagnostic remains ignored.

Therefore the remodel is a useful structural baseline, not a compatibility or correctness certificate.

## Reuse versus reconsideration for ForgeCore

### Reusable infrastructure candidates

- GGUF parser, metadata values, descriptors, alignment, and bounded validation.
- `GgufDataSource` range I/O, profiling, cursor reuse, coalescing, and grouped-buffer ownership.
- `MemoryBudget` and named/scoped accounting.
- Persistent-weight policy, `LayerDescriptor`, `LayerReadPlan`, `LayerCache`, and rollback/eviction behavior.
- Machine/storage discovery, planner profiles, capability facts, calibration records, plan compiler, runtime configuration, persistence, CLI/TUI, and diagnostic schemas.
- Tokenizer metadata loading and incremental UTF-8 decoding should be retained only after the new engine revalidates the tokenizer contract for each target model family.

### Reconsider/rebuild candidates

- `TensorData` semantic contract and all raw-to-logical tensor layout assumptions.
- Quantized block decode and quantized row/matvec kernels, especially Q6_K.
- F32/F16/BF16 loading and conversion semantics as used by the numerical core.
- RMSNorm, RoPE, Q/K/V projection shapes, attention scaling, GQA mapping, and Qwen biases.
- KV-cache staging/commit semantics and the coupling between cache history and model execution.
- SwiGLU, residual ordering, output normalization, output projection, and sampling input.
- `ModelExecutor` and the current runtime `ops.rs` adapters.
- Generation integration after the prefill path is independently correct.

The reuse decision should be evidence-driven. An infrastructure component may be retained while its numerical inputs/outputs are replaced by ForgeCore contracts.
