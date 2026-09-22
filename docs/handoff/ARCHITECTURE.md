# Current architecture

This document describes the code that exists now. ForgeCore references at the end are proposed boundaries, not current modules.

## System flow

The implemented high-level flow is:

```text
CLI / TUI
   |
   v
RuntimeOrchestrator::analyze
   |
   +--> StorageDiscovery
   +--> GgufDataSource::open -> parse_gguf_file -> GgufModel / descriptors
   +--> ModelProfile::from_gguf
   +--> plan_model -> static ExecutionMemoryPlan
   +--> MachineDiscovery
   +--> CapabilitySet::derive
   |
   v
PlanningSession
   |
   +--> Planner::plan -> ExecutionPlan
   +--> PlanCompiler::compile -> RuntimeConfig
   +--> PlanningSession::activate_plan
          |
          v
   InferenceEngine
          |
          +--> StreamingLlamaModel
          |      +--> PersistentWeight
          |      +--> LayerCache
          |      +--> LayerReadPlan / grouped reads
          |      +--> ModelExecutor
          |      +--> ComputeBackend / compute_dispatch
          |
          +--> MemoryBudget
          +--> KvCache
          +--> Tokenizer / Sampler
          +--> ProfileSnapshot / MemoryReport / diagnostics
```

The simpler CLI `run` path can construct `InferenceEngine::new` directly. The orchestration path retains the already-open `GgufDataSource`, compiles the plan, and transfers the validated source into the runtime.

## GGUF parsing and data source

### Parser and descriptors

`crates/ramforge-core/src/gguf.rs` contains `parse_gguf_file`. It parses GGUF header/version, metadata key/value pairs, alignment, tensor info, and data offsets. It does not copy tensor payloads into the parsed `GgufModel`.

`crates/ramforge-core/src/model.rs` contains:

- `TensorDescriptor` — tensor name, GGUF dimensions, `GgmlType`, relative and absolute offsets, optional byte length, and element count.
- `GgufModel` — path, file size, version, metadata, tensor descriptors, alignment, and data-start offset.
- `ModelInfo` — generic metadata summary used by inspection and planning.

`crates/ramforge-core/src/types.rs` contains the metadata value representation and the broad `GgmlType` enum. Generic parsing can preserve type tags that the inference runtime does not support.

### File-backed access

`crates/ramforge-core/src/datasource.rs` contains `GgufDataSource`. It owns a parsed model and a mutex-protected retained file handle. Important operations include:

- `open` — parses metadata/descriptors and opens the file without loading tensor payloads.
- `read_tensor_by_descriptor` and `read_tensor_range_by_descriptor` — bounded byte reads.
- `read_f32_tensor_by_descriptor` and range variants — direct final `Vec<f32>` reads for F32 tensors.
- `read_coalesced_tensor_range_into` — a validated physical range containing multiple descriptor requests.
- `IoProfile` and per-tensor profiles — logical/physical reads, bytes, seeks, coalescing, cursor reuse, and buffer reuse/growth.

The data source is infrastructure. It does not define transformer mathematics.

## Profiles, discovery, capabilities, and planning

### ModelProfile

`ModelProfile` is defined in `crates/ramforge-runtime/src/planner.rs`, not in a separate model-profile file. `ModelProfile::from_gguf` derives:

- architecture and execution-compatibility classification;
- context/layer counts, tensor count, total elements, file/tensor bytes;
- per-format tensor counts/elements/bytes and runtime-supported flags;
- whether layer reads can be coalesced or have reusable grouped-buffer candidates;
- a descriptor/metadata identity fingerprint.

It is a planning fact object. It does not load weights.

### Machine and storage discovery

`crates/ramforge-runtime/src/discovery.rs` contains:

- `MachineDiscovery` — OS/architecture, CPU identity, logical/physical CPU facts where available, AVX2 detection, RAM snapshots, and GPU inventory state.
- `StorageDiscovery` — canonical path, regular-file/readability facts, file size, seekability, filesystem/device metadata, and storage kind.

Discovery reports facts. It does not infer performance and does not select a strategy.

### CapabilitySet

`CapabilitySet` is also in `planner.rs`. It combines validated machine, model, storage, and static-plan facts into booleans such as CPU/GPU usability, model execution compatibility, layer-cache possibility, read-coalescing possibility, grouped-buffer-reuse possibility, and supported/unsupported tensor format lists.

The current registry has one runtime strategy marked available: `StrategyId::CpuLayerStreaming`. `GpuLayerOffload` exists as a represented strategy but `GPU_EXECUTION_IMPLEMENTED` is false.

### Static planning

`crates/ramforge-runtime/src/plan.rs` contains `plan_model` and `ExecutionMemoryPlan`. Planning uses descriptors and shared formulas from `accounting.rs`; it does not read tensor payloads or execute the model. It computes, among other values:

- file size versus requested RAM;
- persistent resident/startup estimates;
- largest layer resident/load-peak estimates;
- a necessary managed-memory lower bound;
- an informational layer-cache capacity and maximum complete cached layers;
- logical versus estimated physical per-forward reads and bytes.

The lower bound intentionally excludes prompt-dependent KV, activations, logits, and streamed persistent workspaces. It is necessary, not a sufficient runtime guarantee.

### Calibration

`crates/ramforge-runtime/src/calibration.rs` defines bounded `CalibrationTask`, `CalibrationPlan`, `CalibrationRunner`, task kinds, resource limits, observations, and provenance. Calibration can measure CPU float dot throughput, memory copy, storage reads/latency, selected quantized row decode throughput, and strategy-latency tasks when requirements are available. It records observations; it does not mutate capability facts, alter numerical semantics, or fix a model.

### Planner and ExecutionPlan

`Planner::plan` in `planner.rs` consumes `UserProfile`, `MachineProfile`, `ModelProfile`, `StorageProfile`, `CapabilitySet`, `PlanResult`, and optional `CalibrationResult`. It emits a versioned `ExecutionPlan` containing:

- binding fingerprints;
- mode and strategy;
- RAM budget and CPU thread count;
- layer-cache and I/O decisions;
- GPU decision;
- resource reservations and cost estimates;
- calibration provenance and reason codes.

Planner policy is separate from execution.

### PlanCompiler and RuntimeConfig

`crates/ramforge-runtime/src/plan_compiler.rs` contains `PlanCompiler`. `PlanCompiler::compile` is the safety boundary that validates concrete decisions, bindings, capabilities, static-plan consistency, resource limits, reservations, cache decisions, and I/O decisions. It emits `RuntimeConfig` or a structured `PlanCompilationError`.

`crates/ramforge-runtime/src/runtime_config.rs` contains the execution contract: CPU thread count, RAM budget, layer-cache state/capacity, read-coalescing permission, grouped-buffer-reuse permission, and CPU execution device. It deliberately contains no planner profiles or calibration policy.

## Streaming runtime and memory ownership

### Persistent weights and layers

`crates/ramforge-runtime/src/layer.rs` groups `blk.{i}.*` descriptors into `LayerDescriptor`s and identifies `PersistentDescriptors` for `token_embd.weight`, `output_norm.weight`, and `output.weight`.

`crates/ramforge-runtime/src/persistent.rs` contains `PersistentWeight`, either `Resident(TensorData)` or `Streamed(TensorDescriptor)`. Resident policy uses actual post-decode/compact representation size. Streamed embedding and output projection paths use bounded, budget-charged temporary workspaces.

`crates/ramforge-runtime/src/streaming_model.rs` contains `StreamingLlamaModel`. Loading validates the dense Llama/Qwen2 tensor contract, estimates layer memory and read plans, loads resident persistent weights transactionally, and retains runtime configuration decisions. `load_layer`, `release_layer`, `forward_single_streaming`, and `compute_logits` are the key runtime operations.

### Layer reads and cache

`crates/ramforge-runtime/src/layer_read.rs` contains descriptor-only `LayerReadPlan` construction. Descriptors are sorted by physical offset; ranges with gaps up to 4 KiB can be coalesced while the physical span stays at or below 64 MiB. Descriptor boundaries remain authoritative, and gap bytes are recorded as physical overhead rather than logical tensor bytes.

`crates/ramforge-runtime/src/layer_cache.rs` contains the budget-charged decoded `LayerCache<StreamingLayerWeights>`. It is a strict byte-capacity LRU. Cache hits avoid layer reloads; evictions release each layer charge. If a layer cannot fit, the runtime can continue with a non-cached streamed layer.

### MemoryBudget

`crates/ramforge-core/src/memory.rs` contains `MemoryBudget`. Named allocations are used for persistent weights (`weight:*`), active layers (`layer:*`), KV (`kv_cache`), and temporary workspaces (`tmp:*`). `with_temp` releases scoped charges on both success and error. Runtime code treats budget accounting as the authoritative managed-memory limit.

### ModelExecutor and compute dispatch

`crates/ramforge-runtime/src/model_executor.rs` contains `ModelExecutor::execute_layer`, which receives materialized `StreamingLayerWeights`, caller-owned activation buffers, `KvCache`, `ComputeBackend`, `LlamaConfig`, and `Profiler`. It does not load files, own the budget, choose cache residency, or perform generation.

The current block order is:

1. attention RMSNorm;
2. Q/K/V projections;
3. optional all-or-none Q/K/V bias application;
4. half-split RoPE;
5. cached-history plus current-token GQA attention;
6. attention output projection and residual;
7. FFN RMSNorm;
8. gate/up projections;
9. SiLU(gate) multiplied by up;
10. down projection and residual.

`crates/ramforge-runtime/src/backend.rs` contains `ComputeBackend` and `CpuBackend`. F32 scalar operations delegate to `ramforge_core::compute`; AVX2 and Rayon are dispatch/partitioning optimizations. `compute_dispatch.rs` selects F32 or compact quantized paths, with row-parallel fused Q4_0/Q6_K dispatch where applicable.

`crates/ramforge-core/src/compute.rs` is the current pure compute boundary. It contains `MatrixShape`, `AttentionConfig`, scalar/reference matvec, quantized reference matvec, RMSNorm, elementwise operations, SiLU/SwiGLU, softmax, RoPE, and single-token causal attention. `ramforge-core/src/quant.rs` remains the block-decoder authority.

## KV cache and generation

`crates/ramforge-runtime/src/kv_cache.rs` stores each layer's K and V as flattened `[position][kv_head][head_dim]`, with committed history `[0, seq_len)`. `append` writes at the current sequence position; the caller increments sequence length after all layers have staged/appended their current K/V. Capacity grows in 256-token chunks and is reconciled with the `kv_cache` budget charge.

`crates/ramforge-runtime/src/inference.rs` contains `InferenceEngine`. It owns the file-backed source, tokenizer, streaming model, optional KV cache, CPU backend, budget, runtime config, and residency statistics. `generate_with_callback`:

1. clears prior KV and decoded-layer cache state;
2. tokenizes the raw prompt with `Tokenizer::encode(prompt, true)`;
3. creates a KV cache sized to prompt length;
4. charges hidden/logits/sampling temporaries;
5. forwards every prompt token and builds history;
6. computes logits, samples a token, emits decoded text, and forwards the sampled token only when another iteration needs its logits;
7. grows KV in chunks as needed;
8. reports `StopReason::{Eos, MaxTokens, ContextLength}` and generation-state diagnostics;
9. releases temporary charges and preserves a successful KV cache/charge for explicit reuse/reset.

The tokenizer has no chat-template application. It encodes the prompt text supplied by the caller and applies the GGUF BOS/EOS configuration.

## CLI, TUI, persistence, diagnostics

`crates/ramforge-cli/src/main.rs` exposes:

- `inspect MODEL [--json]`;
- `plan MODEL --ram SIZE [--json]`;
- `support`;
- `run MODEL --ram SIZE --prompt TEXT [--max-tokens N] [--temperature T] [--top-k K] [--top-p P] [--verbose] [--profile] [--memory-report]`.

`crates/ramforge-cli/src/tui/mod.rs` sequences interactive actions through `RuntimeOrchestrator`, calibration, plan creation/validation, runtime activation, generation, plan save/load, and run export. `tui/app.rs` is the state machine; `tui/render.rs` renders screens; `tui/terminal.rs` parses terminal commands; `tui/diagnostics.rs` defines the diagnostic/run-history schema and JSON export.

`crates/ramforge-runtime/src/plan_persistence.rs` serializes bounded planner decisions, fingerprints, reservations, costs, calibration provenance, and reasons. It deliberately excludes model paths as payloads, model weights, runtime allocations, caches, and inference state. Compatibility re-materializes a candidate plan and invokes the normal compiler boundary.

## Boundaries to preserve during ForgeCore work

These are the current seams that should remain stable unless there is a demonstrated reason to change them:

1. **GGUF facts:** `GgufModel`, `TensorDescriptor`, `GgmlType`, metadata, offsets, and byte lengths.
2. **Storage access:** `GgufDataSource` bounded reads, descriptor validation, coalescing, cursor/profile behavior.
3. **Memory ownership:** `MemoryBudget` named charges and scoped release semantics.
4. **Resource infrastructure:** `LayerDescriptor`, `LayerReadPlan`, `LayerCache`, persistent-weight policy, `ResidencyStats`, and profile counters.
5. **Planning contracts:** `ModelProfile`, `MachineProfile`, `StorageProfile`, `CapabilitySet`, `ExecutionPlan`, `PlanCompiler`, `RuntimeConfig`, and persistence schema/compatibility.
6. **Operator-facing diagnostics:** CLI/TUI commands, profile fields, memory-report distinction, run-history records, and the historical Qwen2.5 diagnostic evidence.

The numerical boundary is the part to reconsider: `TensorData` semantics, reference/optimized quantized paths, `ModelExecutor`, `ops.rs`, attention/GQA/RoPE/RMSNorm/SwiGLU conventions, output projection, KV integration, and generation should be treated as ForgeCore candidates until independently revalidated.
