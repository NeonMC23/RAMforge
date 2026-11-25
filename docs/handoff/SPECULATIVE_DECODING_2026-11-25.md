# Speculative Decoding and Multi-Token Verification

**Date:** 2026-11-25
**Scope:** Prove multi-token target verification on the pinned native API, validate a
compatible draft model, implement greedy speculative decoding with KV rollback, and
measure real end-to-end throughput on the 32B dense model.
**Native checkout:** pinned `7fe450e19305b828c199d602c23a8337aaa1f03b`, unmodified.

## 1. Executive summary

The central hypothesis is **confirmed**: the target model can evaluate a batch of
candidate positions in one forward, and target weight reads are amortized across
those positions.

Isolated verification probe, Qwen2.5-Coder-32B Q4_K_M, CPU, context 256, Q8_0 KV:

| K evaluated positions | Storage read / batch | Read / evaluated token | Batch time | Evaluated tok/s |
|---:|---:|---:|---:|---:|
| 1 | 17.4 GB | 17.4 GB | 9.83 s | 0.10 |
| 2 | 18.3 GB | 9.17 GB | 10.26 s | 0.20 |
| 4 | 18.4 GB | 4.60 GB | 9.63 s | 0.42 |
| 8 | 18.5 GB | 2.31 GB | 10.31 s | 0.78 |
| 16 | 18.6 GB | 1.16 GB | 11.24 s | 1.42 |

The batch reads the same weight working set regardless of K. This is the required
amortization; it is not a claim that supplied tokens were generated.

A complete greedy speculative path was implemented and measured on the same 32B
target:

| Configuration | 8 output tokens | Speed | Correctness |
|---|---:|---:|---|
| Ordinary greedy | 72.86 s | 0.110 tok/s | reference |
| Speculative, 1.5B draft, K=8 | 43.17 s | 0.185 tok/s | **byte-identical** |
| Speculative, 7B draft, K=8 | **36.93 s** | **0.217 tok/s** | **byte-identical** |

A controlled 16-token comparison at low system load measured **46.77 s vs
137.62 s** for ordinary greedy decoding (**2.94x**), 14 accepted / 17 drafted
(82.35%), 3 target forwards, and byte-identical output.

Measured end-to-end gains: **1.69x** with the 1.5B draft and **1.97x** with the 7B
draft, with the same greedy token sequence. A 16-token run with K=8 and the 7B
draft reached 0.314 tok/s; its ordinary 16-token comparison was invalidated by a
concurrent system-load spike and is not used for a ratio.

Negative result: target static GPU offload plus speculative verification was much
slower (127.4 s vs 65.7 s for ordinary static-6-layer decoding). Speculative
target execution is therefore CPU-only in this milestone.

## 2. Native API investigation

Available at the pin:

- `BatchBuilder` accepts multiple tokens with independent positions, sequence ids,
  and per-position logits flags.
- `ContextOptions.n_outputs_max` / `n_outputs_max_per_seq` size the output graph.
  Multi-token verification requires `n_outputs_max_per_seq >= batch length`; the
  default `1` is insufficient.
- `Context::logits(i)` returns an owned copy for batch position `i` when that
  position was flagged.
- `Memory::remove_range(seq, start, end)` removes KV cells in `[start, end)` and
  supports `end = None` for tail removal. This was validated at logit level:
  rollback + fallback produced bit-identical logits to a reference context that
  never decoded the rejected candidates.
- There is no native speculative-decoding helper, no draft-model orchestration, and
  no distribution-level acceptance primitive at this pin.

Exact greedy correctness does not require a new native sampling API: accepted
tokens equal the target argmax, and the first mismatch supplies the target
fallback. Temperature/top-k/top-p speculative sampling is **not** implemented and
is rejected by the CLI.

## 3. Draft compatibility

Local candidates:

| Draft | Quant | Size | Vocab | Verdict |
|---|---|---:|---:|---|
| Qwen2.5-1.5B-Instruct | Q4_0 | 1.07 GB | 151,936 | compatible |
| Qwen2.5-7B-Instruct | Q4_K_M | 4.68 GB | 151,936 | compatible |

Verified against the 32B target (152,064 tokens):

- Draft token list is an **exact prefix** of the target token list over all
  151,936 draft entries.
- Merge list is identical (151,387 entries).
- Both use `tokenizer.ggml.model = gpt2`, `tokenizer.ggml.pre = qwen2`.
- BOS `151643` and EOS `151645` match.
- Target-only tail (128 tokens) can be emitted by target fallback; the draft never
  proposes them.

`draft_vocabulary_compatible` enforces model/pre, BOS/EOS, token list prefix, merge
prefix, and token-type prefix. SmolLM2 is rejected explicitly (different pre).

Memory cost: the 1.5B draft adds ~1.1 GB; the 7B draft adds ~4.7 GB. Both coexist
with the 32B target mapping, but the 7B draft increases host page pressure. The 7B
draft improved measured throughput and acceptance on this machine; that is not a
universal recommendation.

## 4. Implementation

### forgeCORE / adapter

- `ForgeCoreContextOptions` now forwards `n_outputs_max` and
  `n_outputs_max_per_seq`.
- `ForgeCoreSession::decode_tokens_all_logits(tokens, start_position)` decodes one
  verification batch without chunking and returns owned logits for every position.
- `ForgeCoreSession::remove_kv_range(seq, start, end)` exposes safe KV rollback
  through `Memory::remove_range`.

### Runtime

`crates/ramforge-runtime/src/speculative.rs`:

- Greedy draft generation from the draft context.
- One target verification forward per round.
- First mismatch becomes the target fallback.
- The fallback is carried as the first token of the next verification batch, so no
  separate fallback decode is required.
- Rejected candidate KV cells are removed from both target and draft contexts.
- Exact greedy target semantics; EOS, max-token, and context limits handled.
- If the target fallback is an id outside the draft vocabulary (the 32B target has
  128 extra tail tokens), the round switches to ordinary greedy target decoding
  for the remainder instead of feeding an invalid id to the draft.
- Prompts containing target-only ids are rejected before draft prefill.
- Stats: rounds, draft tokens, target positions, accepted/rejected/fallback,
  acceptance rate, target forwards, target/draft time, TTFT, total time, per-round
  detail.

### CLI

```bash
ramforge run TARGET.gguf --backend forgecore --ctx 256 --threads 6 \
  --temperature 0 --draft-model DRAFT.gguf --draft-tokens 8 \
  [--draft-threads N] [--speculative-report PATH]
```

Rejected explicitly:

- `--temperature > 0` or `--top-k/--top-p` (greedy-only).
- `--gpu-layers != 0` (measured slower; CPU target only).
- Incompatible tokenizer/vocabulary.
- `--draft-tokens` > 64 or batch smaller than draft length + 1.
- `--routing-observe` combined with drafting in this path.

## 5. Correctness

- **KV rollback proof:** on Qwen2.5-1.5B, one-accepted, zero-accepted, and
  all-accepted rollback/follow-up logits matched a reference context exactly
  (max absolute difference 0, same top token).
- **Rejection paths:** 32B K=8 run exercised 8 rejected candidates across 3 rounds;
  output was byte-identical to ordinary greedy.
- **All-accepted path:** 1.5B K=4 run accepted 6/6 candidates over 2 rounds; output
  identical.
- **Partial acceptance:** 32B K=4 run accepted 2, rejected 2; output identical.
- **EOS:** 1.5B chat run stopped with `stop_reason=eos`, emitted 2 tokens, 2 accepted,
  0 fallback; output matched ordinary greedy.
- **Continuation after rollback:** the 8- and 16-token comparisons continue through
  multiple rounds and match ordinary greedy token-for-token.
- Opt-in test `speculative_real` compares ordinary target tokens to speculative
  tokens; passes on Qwen2.5-1.5B (target=draft). The 32B/1.5B invocation was
  attempted but the host entered a concurrent-load spike and the run exceeded the
  test timeout; the same pair was validated through the CLI.

Not implemented/validated:

- Temperature/top-k/top-p speculative sampling and residual resampling.
- Grammar/stop-sequence integration.
- Draft model on GPU.
- Combined dynamic staging (`--residency auto`) plus speculative target.
- Cancellation during a speculative round in the CLI path.

## 6. Benchmarks

Hardware: i5-11400H 6C/12T, 15 GiB RAM, RTX 3050 4 GiB, NVMe. Target:
Qwen2.5-Coder-32B Q4_K_M, 19.85 GB. Context 128/256, batch/ubatch 64, Q8_0 KV,
6 target threads, greedy, `--no-repack --no-prefetch`, CPU target.

### 8-token controlled comparison

| Draft | K | Rounds | Accepted / drafted | Target forwards | Target time | Draft time | Wall | Speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | — | — | — | 8 | — | — | 72.86 s | 0.110 tok/s |
| 1.5B | 8 | 3 | 5 / 13 (38.5%) | 3 | 28.4 s | 1.4 s | 43.17 s | 0.185 tok/s |
| 7B | 8 | 2 | 6 / 8 (75.0%) | 2 | 19.6 s | 3.3 s | 36.93 s | 0.217 tok/s |

Output text/token sequence identical in all three cases.

### 4-token comparison

| Draft | K | Wall | Speed | Note |
|---|---:|---:|---:|---|
| none | — | 37.10 s | 0.108 tok/s | reference |
| 1.5B | 4 | 31.86 s | 0.126 tok/s | 50% acceptance, identical output |

### 16-token tuning

- 7B draft, K=8, 16 tokens: **46.77 s**, **0.342 tok/s**, 3 rounds, 14 accepted /
  17 drafted (82.35%), 3 target forwards, target 28.0 s, draft 6.0 s,
  **byte-identical** to the 137.62 s ordinary run (**2.94x**).
- 7B draft, K=16, 16 tokens: 132.69 s, 0.121 tok/s, 2 rounds, 14 accepted / 24
  drafted (58.3%). Target forwards took ~56 s each with a 17-token verification
  batch; larger batches increased compute/memory pressure. K=8 is the better
  measured choice.
- One ordinary 16-token run during an unrelated concurrent build/test load spiked
  to 939 s; it is discarded as an environmental outlier. The controlled 137.62 s
  run was taken at load ~1.8 and is the comparison above.

### Storage

- Verification probe: reads stay 17-19 GB per batch; K=16 lowers bytes per
  evaluated token from 17.4 GB to 1.16 GB.
- Speculative 32B, 8 tokens, 1.5B draft: 75.3 GB process-wide reads.
- Speculative 32B, 8 tokens, 7B draft: 63.1 GB.
- Speculative 32B, 16 tokens, 7B draft: 84.8 GB process-wide reads (~5.3 GB per
  emitted token), versus roughly 17-19 GB per ordinary target forward.
- Ordinary 32B per-token reads measured earlier at 17-18 GB; 8 ordinary tokens
  plus prompt are roughly 150+ GB, so speculative reduces bytes per emitted token
  substantially (process-wide measurement caveat applies).

### Negative result

Target static GPU 6 layers + 7B draft K=8: 127.4 s vs 65.7 s for ordinary static-6
decoding. Target verification with GPU offload was slower; the CLI now rejects
`--gpu-layers` with speculative decoding. This does not evaluate dynamic staging;
that combination remains unimplemented.

## 7. Tests

- Unit tests: argmax tie-breaking, all/partial/pending acceptance, vocabulary
  prefix matching.
- Opt-in real-model `speculative_real`: ordinary vs speculative token equality;
  passes on 1.5B.
- CLI real-model checks: 32B K=4 and K=8, 1.5B and 7B drafts, EOS, rejection
  rounds, incompatible draft, non-greedy rejection, GPU rejection.
- KV rollback probe: exact logits after one/zero/all accepted.
- Workspace tests and Clippy were run after the changes; results are recorded in
  the final milestone report.

## 8. Limitations

- Greedy only. No exact speculative sampling for temperature/top-k/top-p.
- CPU target only. Static GPU target was slower and is rejected.
- No dynamic staging + speculative integration.
- No stop sequences/grammar integration.
- 7B draft increases memory pressure; 1.5B is the lower-risk draft.
- No 180B/500B claim. The 32B target is still ~0.2 tok/s in this configuration.
- Benchmark conditions were shared-desktop and load-sensitive; the 16-token
  ordinary outlier is explicitly excluded.

## 9. Next steps

1. Distribution-preserving speculative sampling for temperature/top-k/top-p.
2. Service/benchmark integration with `--residency auto`, measuring whether staged
   target layers plus verification remain beneficial.
3. Draft policy: choose 1.5B vs 7B from acceptance, memory pressure, and target
   storage impact; test lower-memory quantizations.
4. Tune K per model/hardware and measure acceptance by draft position.
5. Native proposal only if needed: no native change was required for greedy
   verification or KV rollback.
