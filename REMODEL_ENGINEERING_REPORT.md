# RAMforge inference remodel engineering report

Date: 2026-09-22

## Executive summary

The inference path now has an explicit correctness-first numerical boundary and a
runtime model-execution boundary:

```text
GGUF/data source + MemoryBudget + layer cache + streaming
                    |
                    v
          StreamingLlamaModel
                    |
                    v
             ModelExecutor
                    |
                    v
       runtime representation dispatch
                    |
                    v
       ramforge_core::compute reference contract
```

Existing storage formats, compact residency, layer streaming, cache accounting,
profiling, tokenizer/generation code, and the CPU optimized kernels were kept.
The remodel does not claim llama.cpp compatibility and has not been compared
with an external llama.cpp checkout or reference model in this environment.

## Numerical contract

### Matrix and tensor layout

`MatrixShape::new(input, output)` is the only matrix orientation used by the
pure compute API:

- GGML/GGUF tensor shape is `[input, output]` (`ne[0]`, `ne[1]`).
- Materialized weights are row-major `[output][input]`.
- `y[o] = sum_i weights[o * input + i] * x[i]`.
- F32 reference accumulation is F32 and visits input indices in increasing
  order.
- Arity, row-range, block-width, and buffer-size mismatches are errors; the
  reference path does not transpose, truncate, or guess an orientation.

The runtime F32 scalar path delegates to this contract. SIMD and threaded
optimized paths retain their implementation and are compared against it in
focused tests.

### Quantization

`ramforge_core::quant` remains the authority for block decoders and format
row-dot kernels. `ramforge_core::compute::quantized_matvec_reference` decodes
one complete output row through those decoders and performs the same scalar
F32 input-order dot product. Reference coverage is explicit for:

- Q4_0 and Q8_0, block width 32;
- Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, and Q8_K, block width 256.

Unsupported formats return an explicit error. The bounded runtime dispatch keeps
Q4_0/Q6_K fused row kernels and compact format-specific kernels where using the
pure row-materializing reference would introduce an uncharged heap allocation.
The reference implementation is still the storage-independent authority for
parity tests and `TensorData`'s scalar/reference API.

### Transformer operations

The pure compute module defines and tests:

- RMSNorm: `sqrt(sum(x[i]^2) / n + epsilon)`, then `x / rms * weight`, with
  F32 reduction order and finite, non-negative epsilon validation;
- elementwise add and multiply;
- SiLU: `x / (1 + exp(-x))`;
- SwiGLU: `SiLU(gate) * up` as an explicit reference operation;
- stable softmax (max subtraction, F32 sum, explicit finite normalization);
- half-split Llama/Qwen RoPE for independently shaped Q and K buffers;
- single-token causal attention with explicit history length, current K/V,
  head dimension scaling, and GQA mapping
  `kv_head = query_head * kv_heads / query_heads`.

`ModelExecutor` documents and executes the complete block order, including
Qwen Q/K/V bias application and SwiGLU (`SiLU(gate) * up`). Runtime `ops.rs`
now adapts RoPE and attention to the pure reference implementation instead of
maintaining a second numerical definition. Backend RMSNorm, add, multiply,
SiLU, and softmax likewise delegate to the reference operations.

## Runtime integration

### ModelExecutor boundary

`crates/ramforge-runtime/src/model_executor.rs` owns one transformer layer's
execution over caller-owned activation scratch and materialized
`StreamingLayerWeights`. It does not load GGUF data, own a memory budget,
choose cache residency, perform generation, or manage persistence.

Both the layer-cache hit path and newly-loaded layer path in
`StreamingLlamaModel::forward_single_streaming` call
`ModelExecutor::execute_layer`. The existing K/V staging rule is preserved:
all layers consume the prior history first, then the caller appends every
layer's current K/V and increments the cache sequence length once.

### Runtime representation dispatch

`compute_dispatch.rs` separates representation concerns from mathematics:

- F32 data is sent to `ComputeBackend`;
- Q4_0/Q6_K retain the bounded row-parallel fused dispatch;
- other supported compact formats retain their existing compact format
  kernels;
- the reference contract is used for scalar F32 and the reference API is used
  for parity tests.

`backend.rs` no longer redefines scalar F32 matvec, RMSNorm, elementwise
operations, SiLU, or softmax. Threading only partitions work; it does not alter
the numerical contract.

### KV-cache invariants

`KvCache` now documents and exposes the layout
`[position][kv_head][head_dim]`, committed history `[0, seq_len)`, and explicit
K/V read ranges. The streaming forward boundary rejects a position that does
not equal the cache's historical length and rejects cache/model dimension
mismatches. `ModelExecutor` reads exactly `[0, history_len)` and supplies the
current K/V separately to attention.

## Quantized row-range regression

Q4_0 and Q6_K row-range functions now use a local output-slice contract:
`y[0]` corresponds to `row_start`. Range validation happens before subtraction
or indexing, preventing the prior `row_start`/local-slice out-of-bounds failure
mode. Tests cover nonzero row starts, local output slices, invalid starts, and
Q4_0/Q6_K parallel partition boundaries.

## Tests added or updated

- Pure compute tests for nonsquare layout, local reference row ranges, RMSNorm,
  position-zero RoPE identity, GQA mapping, Q4_0/Q6_K reference-vs-fused
  parity, and every currently materialized quantized format.
- Quantized row-range tests for Q4_0 and Q6_K local indexing and invalid-range
  rejection.
- Runtime Q4_0 parallel-vs-reference regression at a representative large
  projection shape.
- Runtime Q6_K scalar-vs-parallel equivalence at an uneven row-partition
  boundary.
- Existing deterministic tiny GGUF and GQA/streaming tests remain in place and
  now exercise the `ModelExecutor` route because the real streaming model calls
  it.
- Temporary hook-investigation trace markers were removed; the existing
  test-only hook and diagnostic output remain functional.

## Validation status and limitations

Full validation was **not executable in this environment**:

- `cargo` is unavailable;
- `rustc` is unavailable;
- `rustfmt` is unavailable.

Therefore compilation, unit tests, integration tests, formatting, clippy, and
release builds were not run and no pass claim is made. No external llama.cpp
checkout or Qwen2.5 model payload is present here, so no independent reference
comparison or compatibility claim is made. The workspace was checked for
unwanted generated build directories/model payloads; none were added.

The next validation step on a Rust-equipped checkout is:

```text
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo build --workspace --release
```

Those commands should be followed by the real-model, independently sourced
reference comparison required for any future compatibility statement.
