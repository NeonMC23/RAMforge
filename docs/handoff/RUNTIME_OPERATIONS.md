# Operational runtime, service, and throughput

This milestone connects execution plans to real forgeCORE inference, adds a
transport-independent inference service, measures real generation throughput,
and provides bounded profile comparison. It does not add eviction, automatic
reclamation, automatic live switching, or GPU offload.

## Execution-plan contract

`RuntimePlan` (schema version 1) resolves a profile request against the machine
and model:

- profile: maximum-performance, balanced, background, custom
- model path and model fingerprint
- effective CPU context, batch, micro-batch, sequence count, threads
- load mode (`mmap` or `mmap-no-prefetch`)
- weight repacking
- KV type and KV type source
- per-field status: applied, constrained, advisory, unsupported, unknown
- warnings for unavailable topology, advisory RAM budget, or non-selectable
  devices

Requested values are distinguished from effective values. The service reads
back native effective context, batch, micro-batch, sequence count, and thread
count after context creation.

Currently applied to forgeCORE:

- CPU backend/device
- context length (native may pad it; effective value is read back)
- generation/batch thread count
- mmap vs mmap-no-prefetch
- CPU weight repacking
- single sequence

Advisory/unsupported:

- RAM budget is advisory unless cgroup enforcement is active
- GPU offload/placement is unsupported by the current adapter
- KV effective types are not exposed by the pinned native getter

## Runtime lifecycle

`InferenceService` states:

`empty → model_loaded → ready → generating → ready → terminated`

- Model load and context creation are explicit.
- A service owns one model and one context at a time.
- `generate` requires `ready`; no concurrent generation is allowed.
- A fresh context is created before each measured benchmark run.
- `unload` drops context, model, and tokenizer.
- Errors are typed: invalid state, configuration, model, tokenizer, context,
  execution, cancellation, or stream sink.
- Cancellation is cooperative and checked between token decodes.
- Streaming decoded text is delivered through a caller callback.

No service operation authorizes eviction or release.

## Local service transport

`ramforge serve --stdio` exposes the service over JSON-lines on stdin/stdout.
This is the smallest real transport; the service contract is independent of it
and can be used by a TUI or GUI adapter.

Commands:

```json
{"id":1,"command":"capabilities"}
{"id":2,"command":"status"}
{"id":3,"command":"generate","prompt":"hello","max_tokens":8}
{"id":4,"command":"session"}
{"id":5,"command":"unload"}
{"command":"cancel"}
```

Generation events:

```json
{"id":3,"event":"token","text":"..."}
{"id":3,"event":"done","text":"...","stats":{...}}
{"id":3,"event":"cancelled"}
{"id":3,"event":"error","kind":"...","error":"..."}
```

A cancel line sets a shared atomic flag; the generation loop observes it
between token decodes. The transport is local-process only. It does not expose
authentication or network access.

## Throughput benchmark

`ramforge benchmark MODEL --profile balanced --runs 3 --warmup 1` measures:

- model load duration
- context creation duration per run
- prompt tokens and prompt processing throughput
- time to first token
- generation tokens/s
- decode tokens/s
- generated token count and stop reason
- RSS before/after generation
- effective native configuration

Generation throughput and prompt throughput are separate. A single run is not
presented as a reliable comparison.

Measured baseline on the development machine, SmolLM2-135M Q4_K_M, 256 context,
4 threads, no repacking, debug build, warm/uncontrolled:

| Metric | Median | Range |
|---|---:|---:|
| Generation tokens/s | 136.87 | 112.58–155.41 |
| Decode tokens/s | 171.70 | — |
| TTFT | 11.40 ms | — |
| Prompt throughput | 423.31 tokens/s | — |
| Context creation | 5.52 ms | — |
| Model load | 234.91 ms | — |

These are single-machine warm observations, not a performance guarantee.

## Profile optimization loop

`ramforge optimize MODEL --runs 2 --max-tokens 4` runs maximum-performance,
balanced, and background candidates with the same bounded workload, ranks them
by measured median generation tokens/s, and returns an advisory recommendation.

One-run smoke:

| Profile | Generation tokens/s |
|---|---:|
| maximum-performance | 121.98 |
| balanced | 137.87 |
| background | 103.34 |

Recommendation: balanced. The result is noisy and advisory. No live session is
reconfigured, no model/backend/device is switched, and no profile is executed
automatically.

## Storage assessment and copy

`ramforge storage MODEL --dest DIR` checks capacity, read-only state, and
filesystem identity. It never copies or deletes without explicit action.

`ramforge storage MODEL --copy --destination PATH --yes` performs a verified
copy:

- copies to a temporary destination;
- fsyncs and verifies byte count and bounded fingerprint;
- renames into place;
- preserves the source;
- does not delete or replace the source automatically.

A faster storage device is not claimed to increase token throughput without
measurement.

## Profiles

- **Maximum performance:** all process-visible CPUs, larger context, repacking
  enabled.
- **Balanced:** one fewer CPU than available, moderate context, repacking
  disabled.
- **Background:** one CPU, smaller context, mmap-no-prefetch, repacking
  disabled.
- **Custom:** explicit context/batch/thread/load/repacking values, with clamps
  and field statuses.

All profiles use the same planning and execution machinery. They are resource
policies, not separate inference engines.

## Large-model position

The pinned native backend performs CPU inference and supports file-backed mmap.
RAMforge can therefore run models whose file-backed pages may not all be
resident, but it does **not** provide controlled tensor streaming, layer
swapping, demand paging, asynchronous staging, GPU placement, or reclamation.
The 500B parameter target remains a product ambition, not a claimed capability.

## Exclusions

No eviction, reclamation, prefetch, predictive next-use scheduling, automatic
plan switching, automatic model/backend/device changes, live tensor migration,
or unvalidated GPU allocation behavior is implemented.
