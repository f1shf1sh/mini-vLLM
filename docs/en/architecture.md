# Project structure and architecture

[Project home](../../README.md) | [中文](../zh-CN/architecture.md)

## Project layout and model adapters

```text
src/llm/
  cache/                        blocks.py, pool.py, paged.py
  inference/                    runner.py, scheduler.py, request.py, sampler.py, types.py
  model/
    gpt_mini/                   runner.py
    qwen2_5_0_5b_instruct/       runner.py, cache.py
  runtime/generate.py
  config.py                     Inference configuration

toy/
  model/                        GPTMini, attention, block
  tokenizer.py                  Educational character tokenizer
  generator.py                  Direct generation without Scheduler
  config.py, checkpoint.py, dataset.py, utils.py
  train.py, generate.py
  configs/, tests/, tools/

tests/                          cache/, inference/, models/
configs/                        qwen.yaml
examples/                       qwen_inference.py
benchmarks/                     engine.py, metrics.py, kv_memory.py
docker/                         Dockerfile, dev.sh
```

Adapter directories use full model names normalized to Python identifiers, such
as `gpt_mini` and `qwen2_5_0_5b_instruct`. Add a model by implementing its Runner
and passing the instance to Scheduler. Do not add model-specific branches to the
shared scheduler. There is currently no model registry or model factory.

The import package is `llm`. Distribution packages contain inference code and
metadata; `toy` runs from the checkout. `GPTMiniRunner` accepts an externally
constructed model and does not depend on toy training code. Preserve data,
weights, and artifacts in their respective directories; generated data and
results should stay out of commits. Personal lessons and detailed validation
records belong in the Git-ignored `.local/` directory.

Historical path changes:

| Previous location | Current location |
| --- | --- |
| `llm.inference.block_manager` | `llm.cache.blocks` |
| `llm.inference.kv_pool` | `llm.cache.pool` and `llm.cache.paged` |
| `llm.models.qwen` | `llm.model.qwen2_5_0_5b_instruct` |
| `toy.runner.ModelRunner` | `llm.model.gpt_mini.runner.GPTMiniRunner` |
| `toy.learning.char_tokenizer.CharTokenizer` | `toy.tokenizer.CharTokenizer` |
| Toy Runner integration tests | `tests/models/test_gpt_mini_runner.py` |
| `toy/examples/`, `toy/benchmarks/` | Removed; engine examples and benchmarks live in root `examples/` and `benchmarks/` |

## Runner, scheduler, and resource boundaries

The public Runner contract lives in `llm.inference.runner`:

- `RunnerSpec` exposes context length, device, and dtype.
- Single-request `prefill` and `decode` return `RunnerOutput` with logits shaped `[1, T, V]`.
- The optional `BatchDecodeRunner.decode_batch` accepts `[B, 1]` and returns `[B, 1, V]`.
- Adapters parse model configuration, construct cache resources, and handle model forward and cache format conversion.

`Scheduler(runner)` handles FIFO admission, round-robin decode, sampling,
termination, and cleanup. Through `CacheResources`, it queries capacity and
allocates, grows, and releases resources without inspecting model topology or
KV tensor layouts.

| Component | Responsibility |
| --- | --- |
| `BlockManager` | Owns physical block assignments and BlockTable updates |
| `KVPool` | Holds the actual K/V tensors |
| `PagedKVCache` | Maps logical token positions and performs reads, writes, and gather |
| `PagedCacheResources` | Exposes cache lifecycle operations to the scheduler |
| `GPTMiniRunner` | Connects an externally supplied GPTMini model to the inference engine |
| `QwenModelRunner` | Adapts Transformers forward and distinguishes query heads from KV heads |
| `QwenPagedCache` | Writes model-produced KV into reserved slots; does not schedule requests or allocate/release their resources |

The final sampled token is returned without another forward. For a positive
output limit, the maximum cached length is
`prompt_length + max_new_tokens - 1`. Zero-output requests finish directly
without allocating cache.

Resources are reclaimed on normal completion, EOS, cancellation, or execution
failure. `cancel_request(request_id)` cancels one waiting/running request between
steps and preserves other requests. Unknown or terminal IDs return `False`;
cancellation does not interrupt an in-flight forward. `run()` cleans pending
requests on failure. Callers using `step()` directly can clear the queue with
`close()`.

With `Scheduler(runner, record_timing=True)`, `request.timing` records submission,
admission, each token's CPU-observed readiness, and completion after cleanup.
Timestamps use a monotonic clock in nanoseconds; recording is disabled by
default. Zero-output requests have no admission/token timestamps. Cancelled or
failed requests retain existing events. Benchmark tooling computes statistics.
