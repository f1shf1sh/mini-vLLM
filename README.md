# mini-vLLM

[中文](README.zh-CN.md) | **English**

A PyTorch project for learning LLM inference through request scheduling,
KV cache management, and batch execution. It supports local Qwen2.5-0.5B-Instruct
weights and includes a small GPT with training and generation examples.

## Features

- Paged KV storage and per-request resource management.
- Batch decode with different context lengths.
- Request cancellation, EOS handling, and cleanup on failure.
- Reproducible throughput and latency benchmarks.

Prefill currently runs one request at a time. Attention gathers paged KV into
contiguous tensors; a high-performance paged-attention kernel is not implemented.
Next steps are batch prefill and improved continuous-batching policies.

## Documentation

| Guide | Contents |
| --- | --- |
| [Getting started](docs/en/getting-started.md) | Environment, local weights, and execution |
| [Project structure and architecture](docs/en/architecture.md) | Layout, Runner, scheduler, and cache responsibilities |
| [Benchmarks](docs/en/benchmarking.md) | Workloads, concurrency options, and metrics |

Developed with AI assistance.
