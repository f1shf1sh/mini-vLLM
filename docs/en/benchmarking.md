# Benchmarks

[Project home](../../README.md) | [中文](../zh-CN/benchmarking.md)

`benchmarks/engine.py` drives an in-process closed-loop workload and saves raw
records. `benchmarks/metrics.py` validates those records and computes throughput
and latency. The detailed [benchmark guide (Chinese)](../../benchmarks/README.md)
documents measurement rules and file formats.

Run from the repository root with project dependencies and local weights available:

```bash
python -m benchmarks.engine \
  --model models/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype float32 \
  --num-blocks 256 --block-size 16 \
  --prompt-length 128 --max-new-tokens 16 \
  --num-requests 12 --concurrency 4 --max-batch-size 4 \
  --warmup 4 --repeat 3 --seed 0 \
  --eos-policy fixed \
  --output-dir outputs/benchmarks
```

| Option | Meaning |
| --- | --- |
| `--num-requests` | Total requests in each measurement repeat |
| `--concurrency` | Limit on waiting + running requests; completed requests are replaced with later ones |
| `--max-batch-size` | Maximum requests in one decode forward; defaults to 1 |
| `--warmup` | Warm-up requests before each measurement repeat |
| `--repeat` | Number of measurement repeats with the same configuration |
| `--eos-policy fixed` | Ignore EOS and generate a fixed token count for equal-workload comparisons |
| `--eos-policy model` | Use model/tokenizer EOS and report actual output counts |

Inputs are reproducible synthetic token sequences, and sampling is greedy.
The workload does not represent real chat traffic. Model loading, input
encoding, request construction, warm-up, metadata hashing, and result writing
are outside the measurement interval. The full wall-clock interval includes
submission, queuing, execution, draining, cleanup, and final GPU synchronization.

Throughput divides successful request counts or their actual output token
counts by the full measurement duration. TTFT includes queuing. ITL measures
the time between consecutive tokens. Per-request TPOT divides the interval
between the first and last token by `output_count - 1`. E2E includes cleanup.
Token timestamps describe CPU-observed readiness, not individual GPU kernel
duration.

Each invocation creates a new `outputs/benchmarks/<invocation-id>/` directory
containing commands, configuration, input/output IDs, timing events, cache
checks, and summaries. Final cache checks require empty block tables, an empty
allocated set, and a complete set of free block IDs. Results retain key runtime
information and code/model hashes, without exporting environment snapshots.

Replace the placeholder directory below to recompute metrics from raw records:

```bash
python -m benchmarks.metrics 'outputs/benchmarks/<invocation-id>/repeat-000'
```

When comparing batch execution, keep model, device, dtype, input/output workload,
and concurrency fixed. Change `max_batch_size` separately and examine repeated
runs. Paged gather and padding still incur costs; a batch implementation should
not be assumed faster before measurement.
