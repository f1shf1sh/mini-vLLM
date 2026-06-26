# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`mini-vllm` is a teaching-level reimplementation of the core mechanisms from the vLLM paper *"Efficient Memory Management for Large Language Model Serving with PagedAttention"* (Kwon et al., SOSP 2023).

The project includes a self-trained toy GPT model solely as a controllable testbed. **The model is not the focus — the inference runtime is.** Code lives in `src/lmcore/`; tests in `tests/`; config in `configs/`.

**In scope (paper Section 4):** PagedAttention (4.1), KV Cache Manager (4.2), LLM Serving with paged attention (4.3), Scheduling (4.5).

**Out of scope:** preemption, distributed execution, prefix caching, COW, CUDA/Triton kernels, speculative decoding, HF model weight loading.

## Commands

```bash
uv sync --group dev           # Install dependencies
uv run pytest                 # Run all tests
uv run pytest tests/test_block_manager.py -v
uv run lmcore-train --config configs/base.yaml
uv run lmcore-generate --ckpt outputs/ckpt.pt --prompt "hello"
```

## Architecture

The project has five layers. The **inference layer is the primary focus.**

### 1. Model layer (`model/`) — testbed, not product
- `model/attention.py` — `Head` (single-head) and `MultiHeadAttention` with KV cache support via `layer_past`/`use_cache`. Causal mask: `tril(diagonal=past_len)`.
- `model/block.py` — `FeedForward` (4x GELU) and `Block` (Pre-LN).
- `model/__init__.py` — `GPTMini`: token emb + pos emb → N × Block → LN → lm_head. `forward(idx, targets, past_key_values, use_cache)` returns `(logits, loss, presents_or_None)`.

### 2. Inference layer (`inference/`) — **primary focus**

PagedAttention implementation (paper §4.1–4.5):

| module | role |
|--------|------|
| `block_manager.py` | `BlockTable` (logical→physical mapping per request), `BlockAllocator` (global free-list allocator), `BlockManager` (request-level facade, dependency-injected allocator) |
| `kv_pool.py` | `KVPool` — global physical KV storage pool. Shape `[layers, 2(K/V), blocks, heads, block_size, dim]`. Uses `PoolDim` IntEnum instead of magic 0/1. |
| `scheduler.py` | `Scheduler` — FIFO waiting queue + round-robin running deque. Per-request sampling params, KV cache cleanup. Now requires `block_manager` as a constructor argument. |
| `model_runner.py` | `ModelRunner` — `prefill(input_ids)` / `decode(token, kv_cache)`. Being refactored to use `KVPool` + `BlockManager` instead of returning contiguous KV caches. |
| `generator.py` | `CachedGenerator` (prefill/decode loop via ModelRunner). `NaiveGenerator` reserved. |
| `request.py` | `Request` dataclass with `RequestStatus` (WAITING→PREFILLING→DECODING→FINISHED). |
| `sampler.py` | `Sampler`, `SamplingParams` (temperature, top_k; top_p stub). |
| `types.py` | `KVCache` type alias, `RunnerOutput` dataclass. |

### 3. Config & data layer
- `config.py` — `ProjectConfig`, `ModelConfig`, `TrainingConfig`. Loaded from YAML.
- `tokenizer.py` — `CharTokenizer`, reversible, stored in checkpoints.
- `dataset.py` — `TextDataset`, next-token windows.
- `checkpoint.py` — saves model + optimizer + config + tokenizer vocab + step.

### 4. Runtime entries
- `runtime/train.py` — training loop with AdamW.
- `runtime/generate.py` — checkpoint → model → generate.

### 5. Docs & tests
- `docs/shape_flow.md`, `docs/module_notes.md`, `docs/project-plan.md`, `docs/benchmark.md`
- Tests in `tests/` cover shapes, KV cache, scheduler semantics, block management, KV pool R/W.

## Key design decisions

- **Toy GPT as testbed** — tiny configs (`block_size=8, n_layer=2, n_head=2, n_embd=16`) let tests run in ms. Model is not a product.
- **Paged KV first, continuous batching second** — block management must be correct before scheduling gets complex.
- **Pure PyTorch, no CUDA** — paged storage + contiguous compute (gather before attention). Functionally correct, not performance-competitive.
- **Dependency injection** — `BlockManager` receives an `allocator`, not a `num_blocks` integer. Components testable in isolation.
- **Control plane / data plane separated** — `BlockManager` decides which blocks belong to whom (integers only). `KVPool` stores the actual tensors at those addresses.

## Current status (2026-06)

```text
✅ GPTMini model + training + checkpoint
✅ KV-cache-aware forward
✅ Prefill / decode split (ModelRunner)
✅ Request lifecycle + Scheduler (FIFO + round-robin)
✅ BlockTable, BlockAllocator, BlockManager
✅ KVPool (physical storage with PoolDim enum)
🚧 Paged-aware attention (attention reads/writes pool directly)
📋 Scheduler ↔ BlockManager integration
📋 Benchmark: contiguous vs paged memory efficiency
```

## Test conventions

Tests use `pytest`. Model tests use tiny configs. Block manager tests use `_make_bm()` helper that creates a `BlockManager` with dependency-injected `BlockAllocator`. KVPool tests verify read/write correctness across layers, blocks, and offsets independently. Scheduler tests use `Scheduler(model, block_manager=..., sampler=...)` — `block_manager` is a required argument.
