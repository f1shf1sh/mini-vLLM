# mini-vllm

A teaching-level reimplementation of the core mechanisms from the vLLM paper:

> *Efficient Memory Management for Large Language Model Serving with PagedAttention* (Kwon et al., SOSP 2023).

The project uses a self-trained toy GPT model as a controllable testbed — the model is not the focus. The focus is the **inference runtime**: PagedAttention, KV cache management, and request scheduling.

## Scope

**In scope (paper Section 4):**

- 4.1 PagedAttention — block table, logical-to-physical mapping, paged KV storage
- 4.2 KV Cache Manager — block allocator, lifecycle, allocation / free
- 4.3 LLM Serving — paged-aware attention, ModelRunner with prefill / decode
- 4.5 Scheduling — arrival-order scheduling, round-robin decode

**Out of scope (intentionally):**

- Preemption (swap / recomputation)
- Distributed execution (tensor parallelism)
- Prefix caching, copy-on-write, beam search sharing
- CUDA / Triton kernels (pure PyTorch)
- Speculative decoding
- HuggingFace model weight loading

## Architecture

The project has five layers:

### 1. Model layer (`model/`)

A minimal decoder-only Transformer (`GPTMini`) — used only as a controllable testbed. It produces token logits and exposes the standard `forward(input_ids)` interface, making the inference runtime testable with tiny configs.

### 2. Inference runtime layer (`inference/`)

The core of the project. Implements the vLLM paper mechanisms:

| module | role | paper section |
|--------|------|---------------|
| `block_manager.py` | `BlockTable`, `BlockAllocator`, `BlockManager` | 4.1, 4.2 |
| `kv_pool.py` | `KVPool` — global physical KV storage | 4.1 |
| `scheduler.py` | `Scheduler` — FIFO waiting + round-robin decode | 4.5 |
| `model_runner.py` | `ModelRunner` — prefill / decode entry points | 4.3 |
| `request.py` | `Request` lifecycle (WAITING → PREFILLING → DECODING → FINISHED) | |
| `sampler.py` | Temperature / top-k token sampling | |
| `generator.py` | Single-request cached generation | |

### 3. Training & checkpoint layer

`config.py`, `tokenizer.py` (character-level), `dataset.py`, `checkpoint.py` — enough to train the toy GPT so the inference layer has something to serve.

### 4. CLI layer (`runtime/`)

`train.py` and `generate.py` — thin entry points.

### 5. Docs & tests

`docs/`, `tests/` — shape-flow reference, module notes, and unit tests covering correctness of KV cache, block management, request scheduling, and paged storage.

## Key design decisions

- **toy GPT as testbed** — tiny configs (`n_layer=2, n_head=2, n_embd=16, block_size=8`) let tests run in milliseconds. The model is not a product.
- **paged KV cache first, continuous batching second** — block management must be correct before scheduling gets complex.
- **pure PyTorch** — no CUDA kernels. Paged storage + contiguous compute (gather before attention). Functionally correct, not performance-competitive.
- **dependency injection** — `BlockManager` receives an `allocator`, not a `num_blocks` integer. Components are testable in isolation.

## Data flow (paged path, work in progress)

```text
Prefill:
  ModelRunner.prefill(prompt, request_id)
  → block_manager.allocate_for_request(request_id, prompt_len)
  → model(input_ids, kv_pool, block_table)
  → attention writes K/V directly into pool blocks

Decode:
  ModelRunner.decode(token, request_id)
  → block_manager.append_token(request_id)
  → model(token, kv_pool, block_table, num_past_tokens)
  → attention reads past K/V from pool, writes new token's K/V to pool
```

## Directory

```text
mini-vllm/
├── configs/base.yaml
├── data/input.txt
├── outputs/
├── docs/
│   ├── benchmark.md
│   ├── module_notes.md
│   ├── project-plan.md
│   └── shape_flow.md
├── src/lmcore/
│   ├── config.py
│   ├── checkpoint.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── model/
│   │   ├── __init__.py        # GPTMini
│   │   ├── attention.py       # MultiHeadAttention
│   │   └── block.py           # Block, FeedForward
│   ├── inference/
│   │   ├── block_manager.py   # BlockTable, BlockAllocator, BlockManager
│   │   ├── kv_pool.py         # KVPool
│   │   ├── scheduler.py       # Scheduler
│   │   ├── model_runner.py    # ModelRunner
│   │   ├── generator.py       # CachedGenerator
│   │   ├── request.py         # Request, RequestStatus
│   │   ├── sampler.py         # Sampler, SamplingParams
│   │   └── types.py           # KVCache, RunnerOutput
│   └── runtime/
│       ├── train.py
│       └── generate.py
└── tests/
    ├── test_block_manager.py
    ├── test_kv_pool.py
    ├── test_scheduler.py
    ├── test_model_runner.py
    ├── test_generator.py
    ├── test_model.py
    ├── test_attention.py
    ├── test_block.py
    ├── test_request.py
    ├── test_sampler.py
    ├── test_checkpoint.py
    ├── test_config.py
    └── test_tokenizer.py
```

## Commands

```bash
# Install dependencies
uv sync --group dev

# Run all tests
uv run pytest

# Train the toy GPT
uv run lmcore-train --config configs/base.yaml

# Generate from a checkpoint
uv run lmcore-generate --ckpt outputs/ckpt.pt --prompt "hello"
```

## Current status

```text
✅ GPTMini model + training + checkpoint
✅ KV-cache-aware forward
✅ Prefill / decode split (ModelRunner)
✅ CachedGenerator
✅ Request lifecycle
✅ Scheduler (FIFO + round-robin)
✅ BlockTable, BlockAllocator, BlockManager
✅ KVPool (physical storage)
🚧 Paged-aware attention (attention reads/writes pool directly)
📋 Scheduler ↔ BlockManager integration
📋 Benchmark: contiguous vs paged memory efficiency
```
