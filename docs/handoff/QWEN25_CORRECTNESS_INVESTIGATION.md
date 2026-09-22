# Qwen2.5 correctness investigation

## Scope and conclusion

This document records the available Qwen2.5 numerical evidence without claiming llama.cpp compatibility. The external llama.cpp checkout and the Qwen2.5 model payload are not available in the current workspace. The values below are therefore separated into repository facts, recorded observations, hypotheses, and unknowns.

**Conclusion:** the current RAMforge numerical inference path is not trustworthy for the target Qwen2.5 model. The failure is observed before ordinary autoregressive quality can be judged: the recorded investigation found a discrepancy in the numerical path and a direct output-projection discrepancy for Q6_K rows. The surrounding GGUF/storage/planning/accounting infrastructure remains useful, but the numerical core should be rebuilt and independently validated as the proposed **ForgeCore**.

This is not a claim that every listed operator is individually wrong, and it is not a claim that RAMforge cannot execute any model. It is a stop-ship conclusion for treating this Qwen2.5 result as compatible with an external reference.

## Target model identity

The target used by the diagnostic is referred to in source and recorded notes as:

> **Qwen2.5-1.5B-Instruct, Q4_0 GGUF**

The exact source command template uses a path such as:

```text
/path/to/qwen2.5-1.5b-instruct-q4_0.gguf
```

The filename/path is an operator-provided input, not a file present in this workspace. Do not silently replace this target with a different Qwen2.5 checkpoint, a different quantization, or an unquantized model.

### Recorded model dimensions

The target's Qwen2.5-1.5B dense Qwen2 configuration is recorded as follows:

| Property | Value | Evidence status |
|---|---:|---|
| vocabulary size | `151936` | **VERIFIED** by constants/assertions in the ignored diagnostic |
| embedding/hidden size | `1536` | **VERIFIED** by constants/assertions in the ignored diagnostic |
| transformer block count | `28` | **VERIFIED** by constants/assertions in the ignored diagnostic |
| attention query heads | `12` | **OBSERVED/recorded model configuration**; not asserted by the trace constants |
| key/value heads | `2` | **OBSERVED/recorded model configuration**; not asserted by the trace constants |
| head dimension | `128` (`1536 / 12`) | **OBSERVED/derived from recorded dimensions** |
| feed-forward/intermediate size | `8960` | **OBSERVED/recorded model configuration** |
| context length | `32768` | **OBSERVED/recorded model configuration; verify from target GGUF before use** |
| RMSNorm epsilon | `1e-6` | **OBSERVED/recorded model configuration; verify from target GGUF before use** |
| RoPE base/theta | `1000000` | **OBSERVED/recorded model configuration; verify from target GGUF before use** |
| architecture registry name | `qwen2` with alias `qwen2.5` | **VERIFIED** in `crates/ramforge-runtime/src/support.rs` |
| execution shape | dense Qwen2/Llama-shaped transformer | **VERIFIED** in runtime source; not a Qwen3/Qwen3.5 implementation |

The ignored test itself hard-checks only the dimensions needed for its checkpoint contract (`vocab_size = 151936`, `embedding_length = 1536`, `block_count = 28`) and checks the output tensor geometry below. The remaining configuration values must be read from the actual target GGUF in any ForgeCore reproduction.

## Tokenizer metadata and fixed prompt

### Metadata contract

The target tokenizer is recorded as the Qwen2 GPT-2-style BPE tokenizer:

| GGUF metadata/key | Recorded value or behavior | Status |
|---|---|---|
| `tokenizer.ggml.model` | `gpt2` | **OBSERVED/recorded target metadata** |
| `tokenizer.ggml.pre` | `qwen2` | **OBSERVED/recorded target metadata** |
| `tokenizer.ggml.tokens` | vocabulary array of length `151936` | **VERIFIED** as the diagnostic vocabulary size; exact token strings are not in this workspace |
| `tokenizer.ggml.merges` | present for BPE | **OBSERVED from the tokenizer contract; exact merge payload unavailable** |
| `tokenizer.ggml.bos_token_id` | `151643` | **OBSERVED/recorded target metadata; re-check the target GGUF** |
| `tokenizer.ggml.eos_token_id` | `151645` | **OBSERVED/recorded target metadata; re-check the target GGUF** |
| `tokenizer.ggml.padding_token_id` | `151643` | **OBSERVED/recorded target metadata; re-check the target GGUF** |
| `tokenizer.ggml.add_bos_token` | `false` in the recorded target metadata; the diagnostic still calls `encode(..., true)` and therefore does not add BOS | **OBSERVED/recorded target metadata; verify from the target GGUF** |
| `tokenizer.ggml.add_eos_token` | `false` in the recorded target metadata; no EOS is part of the fixed nine-token prompt | **OBSERVED/recorded target metadata; verify from the target GGUF** |

The repository implementation dispatches to BPE when `tokenizer.ggml.model == "gpt2"`, when `tokenizer.ggml.pre == "qwen2"`, or when merges are present. It uses the target's merge ranks and byte fallback, then maintains a stateful decoder for streaming output. It does not apply a chat template.

### Raw prompt and token IDs

The fixed prompt is exactly:

```text
hi, what's 2+2=?
```

The diagnostic constant is:

```text
const RAW_PROMPT: &str = "hi, what's 2+2=?";
const EXPECTED_PROMPT_TOKENS: [u32; 9] =
    [6023, 11, 1128, 594, 220, 17, 10, 17, 19884];
```

Thus the expected positions are:

| position | token ID |
|---:|---:|
| 0 | 6023 |
| 1 | 11 |
| 2 | 1128 |
| 3 | 594 |
| 4 | 220 |
| 5 | 17 |
| 6 | 10 |
| 7 | 17 |
| 8 | 19884 |

The ignored diagnostic refuses to proceed if `Tokenizer::encode(RAW_PROMPT, true)` does not produce exactly this sequence. This is a useful tokenizer gate, not evidence that the complete tokenizer implementation is compatible with every Qwen2 tokenizer variant.

## What the diagnostic records

Source path:

```text
crates/ramforge-runtime/src/inference.rs
```

Module/test:

```text
qwen25_bounded_diagnostic::qwen25_bounded_numerical_trace
```

The test is `#[ignore]` because it requires a real model and an explicit RAM budget. It uses one CPU thread to remove parallel-dispatch differences while keeping the normal Q4_0 path. It performs a fixed prompt prefill, then two decode steps, and prints:

- raw prompt and prompt token IDs;
- prompt positions;
- final prompt hidden summary and final prompt logits summary;
- first generated token;
- first decode input token, position, KV sequence length before/after, hidden summary, and logits summary;
- second generated token;
- second decode input token, position, KV sequence length before/after, hidden summary, and logits summary;
- `result_norm` summary plus fixed element checkpoints;
- one hidden summary after every transformer layer for the final prompt token, including layer 27;
- direct and fused Q6_K output projection dot products for rows 59 and 220;
- raw Q6_K row geometry and bounded row-byte diagnostics for rows 59 and 220.

### Numeric summary definition

For a vector, `summarize` prints:

```text
length
min
max
sum              # accumulated as f64 from each f32 value
l2_norm          # sqrt(sum(value_as_f64 * value_as_f64))
first8           # first eight f32 values
```

Logit summaries add vocabulary size and the top ten `(token_id, f32)` pairs, sorted by descending logit with token ID as the tie-break.

### Layer checkpoint definition

For the final prompt token, the test hook emits one record after each layer's residual and FFN output has been written, before the next layer starts. It emits:

```text
layer_hidden.layer = <layer index>
layer_hidden.length = 1536
layer_hidden.sum = <f64>
layer_hidden.l2_norm = <f64>
layer_hidden.checkpoint.index = 0
layer_hidden.checkpoint.value = <f32>
layer_hidden.checkpoint.index = 255
layer_hidden.checkpoint.value = <f32>
layer_hidden.checkpoint.index = 1023
layer_hidden.checkpoint.value = <f32>
layer_hidden.checkpoint.index = 1535
layer_hidden.checkpoint.value = <f32>
```

Layer indices are `0..27`; **layer 27 is the final transformer block**. The exact recorded layer-27 numeric values are not stored in the repository or in an output trace available in this workspace. They must remain **UNKNOWN**, not be recreated from memory or replaced by a new run. The exact checkpoint schema and indices above are preserved so that the supplied/archived layer-27 values can be inserted verbatim when the trace artifact is recovered.

This limitation is material: source code containing `println!` statements is not a numerical result.

### `result_norm` checkpoints

The test retains the final prompt hidden vector under the name `result_norm` for the output-projection diagnostic. It prints summaries and values at these exact indices:

```text
0, 1, 2, 3, 4, 5, 6, 7,
31, 32, 63, 64, 127, 128, 255, 256,
511, 512, 1023, 1535
```

No `result_norm` statistics or checkpoint values from a completed real-model run are available in this workspace. Treat them as **UNKNOWN**.

## Output projection geometry and recorded comparison

The diagnostic verifies the target descriptor:

```text
tensor name:       output.weight
ggml type:         Q6_K
dimensions:        [1536, 151936]
blocks per row:    6        # 1536 / 256
bytes per row:     1260     # 6 * 210
whole tensor bytes: 191439360
```

The exact byte-length assertion in source is:

```text
descriptor.byte_length == Some((1260 * 151936) as u64)
```

For avoidance of ambiguity, the arithmetic result is `191,439,360` bytes. The GGUF dimensions are `[in_features, out_features]`, so each output vocabulary row contains 1536 input values and 6 Q6_K blocks.

The diagnostic compares two RAMforge computations for rows **59** and **220**:

1. dequantize the complete Q6_K row, then perform an F32 dot product with `result_norm`;
2. call the existing fused `quant::matvec_q6_k` one-row path.

The recorded external reference values embedded in the diagnostic are:

| output row | external/reference value embedded in source |
|---:|---:|
| 59 | `20.24774933` |
| 220 | `19.75674248` |

These literals are the only external numerical reference values preserved in the current source. The diagnostic prints both RAMforge values and absolute errors, but no completed run output is available here. Therefore the following are **UNKNOWN** for this handoff:

- the actual RAMforge dequantized dot for row 59;
- the actual RAMforge fused dot for row 59;
- the actual RAMforge dequantized/fused dots for row 220;
- the exact absolute errors;
- whether a discrepancy is caused by raw-row addressing, Q6_K decode, accumulator semantics, `result_norm`, or a combination.

A second raw-row diagnostic reads exactly 1260 bytes for rows 59 and 220 and prints descriptor/data offsets, FNV-1a checksums, first bytes, first-block scales, `d` bits/value, and the first 16 dequantized values. No output from that diagnostic is present in the repository.

## Evidence and interpretation

### VERIFIED source facts

- The runtime accepts `qwen2` and the alias `qwen2.5` as a dense Qwen2 execution classification.
- The ignored diagnostic hard-checks vocabulary `151936`, embedding size `1536`, and `28` layers.
- The fixed prompt and nine token IDs above are exact diagnostic constants.
- The final hidden/layer hook is after each layer's complete residual/FFN output and emits layer 27 as the final block.
- `output.weight` is expected to be Q6_K with dimensions `[1536, 151936]`, six blocks per row, and 1260 bytes per row.
- The source embeds two comparison values: row 59 `20.24774933` and row 220 `19.75674248`.
- The diagnostic is ignored and requires environment variables; its existence does not constitute a run.

### OBSERVED recorded investigation conclusions

- Numerical behavior diverges on the Qwen2.5 target before ordinary generated-text quality is a useful acceptance criterion.
- Direct output-projection row comparisons also show a discrepancy in the recorded investigation.
- The current Q6_K and full-transformer paths should not be treated as independently validated against an external reference.
- The source remodel's synthetic reference/fused parity tests passed in the later workspace validation, but that does not erase the target-model discrepancy.

### HYPOTHESES to test, not fixes

The evidence is insufficient to select one root cause. Candidate failure surfaces include:

1. **Raw GGUF addressing or row geometry:** descriptor offset, data-start offset, row stride, tensor dimension interpretation, or chunk slicing.
2. **Q6_K block decode:** byte grouping, scale/min ordering, high/low quant bits, half-scale conversion, or accumulation order.
3. **F32 input vector:** wrong layer boundary, final RMSNorm placement/epsilon, residual ordering, or hidden-state orientation.
4. **Transformer semantics:** Q/K/V bias placement, GQA head mapping, RoPE pairing/base/position, attention scaling/history, SwiGLU order, or KV staging.
5. **Tokenizer/prompt boundary:** special-token metadata, BOS behavior, model identity, or a prompt/token mismatch—although the fixed nine-token tokenizer gate reduces this particular ambiguity.
6. **Reference comparison setup:** wrong external checkpoint, wrong quantization, wrong row ID, wrong reference convention, or a stale reference artifact.

Do not convert any hypothesis into a code change in this documentation-only handoff.

## Generated-loop distinction

The diagnostic deliberately distinguishes prefill from autoregressive decode:

```text
Prompt prefill:
  for position 0..8:
      forward(prompt_token[position], position)
  logits(final prompt hidden) -> first_generated_token

Decode boundary 1:
  input_token_id = first_generated_token
  position_id = 9
  forward(first_generated_token, 9)
  logits -> second_generated_token

Decode boundary 2:
  input_token_id = second_generated_token
  position_id = 10
  forward(second_generated_token, 10)
  logits -> diagnostic summary
```

The normal `InferenceEngine::generate_with_callback` loop has an additional optimization: after sampling the final requested token, it does **not** run another forward pass when no later logits are needed. It does feed every non-final sampled token into the next decode position. This distinction matters when comparing traces: a “first generated token” is sampled from final prompt logits; it is not itself the output of a forward pass at prompt position 8, and it becomes the next forward input at position 9.

## Reproduction command and prerequisites

The exact source doc comment supplies this command template:

```text
CARGO_TARGET_DIR=/tmp/ramforge-cargo-target \
RAMFORGE_QWEN25_MODEL=/path/to/qwen2.5-1.5b-instruct-q4_0.gguf \
RAMFORGE_QWEN25_RAM_BYTES=8589934592 \
cargo test -p ramforge-runtime qwen25_bounded_numerical_trace -- \
    --ignored --nocapture
```

Prerequisites:

- a Rust toolchain and the current workspace;
- the exact target Qwen2.5-1.5B-Instruct Q4_0 GGUF;
- a sufficient explicit byte budget;
- enough time for a real 28-layer out-of-core trace;
- an independently sourced reference implementation/checkpoint for comparison.

The current handoff intentionally did **not** run this command. It would be an expensive real-model generation/trace, and neither required model nor external llama.cpp checkout is present.

## Acceptance gate for ForgeCore

Do not call the new engine correct based on generated text alone. The first acceptance sequence should be:

1. verify the target GGUF identity, metadata, descriptor offsets, and checksums;
2. reproduce the exact nine tokenizer IDs;
3. compare raw Q6_K row bytes and first-block decoded values;
4. compare a single output row against the embedded reference values;
5. compare embedding, per-layer hidden states including layer 27, `result_norm`, and prompt logits;
6. compare first and second decode boundaries with the exact token IDs and KV lengths;
7. only then compare generated text and longer runs;
8. repeat with at least one independent implementation/reference and record all artifacts.

Every numerical comparison should retain the full precision printed by the tool, the model file identity/hash, command line, toolchain, thread count, and quantization name.
