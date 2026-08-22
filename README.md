# RAMforge

RAMforge is a local inference runtime designed to run AI models that may be significantly larger than the available RAM or VRAM by treating RAM, VRAM, and storage as a hierarchical memory system.

> **Milestone 1 Status:** Only model inspection is implemented. Inference (`run`, `serve`) is **not** implemented yet.

## Purpose

- Provide a correct, reusable foundation for understanding GGUF model files
- Enable file-backed, out-of-core tensor access without loading entire models into RAM
- Prepare for future hierarchical memory management

## Current Capabilities (Milestone 1)

- Real GGUF parser (magic, header, metadata KV, tensor descriptors)
- File-backed model representation: tensor payloads are NOT loaded into RAM
- Normalized helpers for architecture, context length, embedding size, layer count, expert config, tokenizer info
- Human-readable and JSON inspection output
- Robust error handling for invalid / truncated files
- Unit tests for parsing primitives, invalid magic, malformed files

### Design for Memory Efficiency

The parser in `ramforge-core` only reads:

1. 24-byte header
2. Metadata key/value pairs
3. Tensor descriptors (name, dimensions, type, offset)

It then computes:

- Absolute file offsets (`data_start + tensor.offset`)
- Byte length when determinable from GGML type info and shape

Tensor data itself is never copied into memory. This is documented in code comments and enables future mmap / streaming access.

## Build

Requires Rust stable.

```bash
cargo build
cargo test
cargo clippy
```

## Usage

Inspect a GGUF model:

```bash
cargo run -p ramforge-cli -- inspect /path/to/model.gguf
```

JSON output:

```bash
cargo run -p ramforge-cli -- inspect /path/to/model.gguf --json
```

Example human-readable output includes:

- Model path and file size
- Architecture, tensor count, KV count
- Known layer count, context length, embedding size, expert config
- Tokenizer model and vocab size
- Quantization / tensor type summary
- First N tensors with name, type, shape, file offset, byte length

If a value is unavailable, `unknown` is displayed rather than invented.

## Project Structure

```
crates/
  ramforge-core/      # GGUF parsing, metadata, tensor descriptors, file-backed locations
  ramforge-runtime/   # Minimal placeholder (future inference & planning)
  ramforge-cli/       # inspect command, human-readable output
```

## Acceptance Criteria for Milestone 1

- `cargo test` passes
- `cargo clippy` passes without avoidable warnings
- `cargo run -p ramforge-cli -- inspect MODEL.gguf` reads real GGUF file and reports real metadata
- Tensor payloads are not fully loaded into RAM
- Invalid files fail with understandable errors
- Code structure ready for file-backed, out-of-core tensor access

## What is NOT Implemented Yet

- Inference (`ramforge run`, `ramforge serve`)
- GPU support
- HTTP APIs
- TUI
- Model downloading
- Any dependency on llama.cpp or external processes

## License

MIT OR Apache-2.0
