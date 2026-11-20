# Candidate execution plans

This milestone adds inspectable candidate plans and conservative feasibility
analysis. It does not add an automatic scheduler or memory reclamation.

## Representation

`ExecutionPlanCandidate` (`execution_candidate.rs`) records:

- schema version and stable plan id;
- profile (`performance`, `balanced`, `memory_constrained`,
  `maximum_model_size`, `custom`);
- backend (`cpu` or `gpu`) and optional device id;
- CPU thread count;
- context and batch sizes;
- optional RAM and VRAM budgets;
- mapping strategy (`file_backed`, `resident_weights`, `streaming`);
- explicit assumptions and unknown information.

Plans are serializable and inspectable. Policy is typed; there is no opaque
string policy.

## Default candidates

`candidate_plans()` produces four non-executed candidates:

| Profile | Backend | Mapping | Intent |
|---|---|---|---|
| performance | best selectable GPU, else CPU | resident weights | lower latency when memory allows |
| balanced | CPU | file-backed | keep weights mapped and reclaimable |
| memory-constrained | CPU | streaming | minimize resident weights |
| maximum-model-size | CPU | file-backed | prioritize model size over latency |

These are starting points, not optimization results or promises.

## Feasibility

`Feasibility` states:

- `feasible`
- `feasible_with_constraints`
- `conditionally_feasible`
- `infeasible`
- `unknown`

The evaluator checks only supported constraints:

- context above the model maximum is infeasible;
- invalid plans are infeasible;
- resident-weight plans compare the estimated resident weights + KV against
  the RAM budget;
- file-backed/streaming plans are **not** marked definitively feasible merely
  because a file fits; they remain conditional because residency and runtime
  workspace are not bounded;
- missing budgets or requirements remain unknown;
- GPU plans require a matching selectable device and observed free VRAM;
- missing GPUs do not affect CPU plans;
- thread oversubscription is a constraint, not a hard infeasibility.

Unknown information is returned explicitly. No numeric confidence score is
invented.

## Comparison

`compare_plans()` returns dimensions and per-plan evaluations in input order:

- feasibility;
- estimated RAM and VRAM;
- budgets;
- threads, context, batch;
- mapping strategy;
- storage dependency;
- unknown information;
- reasons.

No single aggregate score or hidden ranking is produced.

## Example

```sh
ramforge machine --model MODEL.gguf --ram 1G --context 2048 --json
```

The JSON report contains `profile`, `model_requirements`, `candidate_plans`,
and `plan_comparison`.

For SmolLM2-135M Q4_K_M on the development machine:

- tensor payload: 103,668,480 bytes (exact);
- KV cache at 2,048 tokens: 47,185,920 bytes (estimated, f16 assumption);
- performance plan: 150,854,400 bytes estimated RAM;
- 64 MiB budget: performance infeasible; file-backed plans conditional;
- 1 GiB budget: performance feasible with constraints; file-backed plans
  conditional.

These are local examples, not universal guarantees.

## Limits

- No automatic plan selection or switching.
- No activation, graph-scratch, staging, or backend-buffer measurement.
- GPU KV/staging placement is not modeled.
- File-backed plans cannot be called definitively feasible without knowing
  working-set and reclaim behavior.
