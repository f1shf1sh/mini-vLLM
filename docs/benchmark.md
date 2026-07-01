# Benchmark: Contiguous vs Paged KV Cache

## 1. Setup

| Parameter | Value |
|-----------|-------|
| Model | GPTMini (n_layer=2, n_head=2, n_embd=8, head_dim=4) |
| Block size | 16 tokens |
| Bytes per token (KV) | 2 × 2 × 2 × 4 × 4 = 256 B (float32) |
| Bytes per block | 256 × 16 = 4096 B |
| Workload | 50 requests, prompt lengths 10–200 tokens |
| Max sequence length (contiguous) | 256 tokens |

## 2. Pure Calculation (no model runs)

```text
                          Contiguous           Paged
        Total memory         3200.0 KB        1344.0 KB
       Wasted memory         1970.3 KB         114.3 KB
         Utilization            38.4%            91.5%
        Memory saved               —             58.0%
```

- Contiguous waste: `(max_len − actual_len) × bytes_per_token` per request.
- Paged waste: at most `(block_size − 1) × bytes_per_token` per request.

## 3. Parameter Sweep (vary request count)

Run via `scripts/benchmark_live.py _sweep()`.

| #Requests | Contiguous blocks | Paged peak blocks | Saved |
|-----------|-------------------|-------------------|-------|
| 10 | 160 | 81 | 49.4% |
| 20 | 320 | 139 | 56.6% |
| 30 | 480 | 212 | 55.8% |
| 40 | 640 | 280 | 56.2% |
| 50 | 800 | 359 | 55.1% |

Savings are consistent (~50–57%) across request counts because the
dominant waste is the per-request `max_len` pre-allocation, which
scales linearly with request count.

## 4. Live Scheduler Block Tracking

50 requests, tracked per scheduler step.  Run via `scripts/benchmark_live.py`.

```text
Requests:           50
Peak blocks used:   359 / 1000
Average blocks used: 227.0
Contiguous needed:  800 blocks (50 req × 256 tok / 16)
Memory saved:       55.1%
```

Block usage over time (text histogram):

```text
step     0: ##############                            134
step    37: ######################################    356
step    74: #######################################   353  ← peak
step   259: #############################             296  ← requests start finishing
step   740:                                            8  ← all done
```

**Shape**: usage rises as requests are added and decoded, peaks when all
requests are active, declines as short requests finish.

## 5. Interpretation

Paged KV cache saves **50–58% memory** compared to contiguous pre-allocation
for this workload configuration.  The saving comes from eliminating the
dominant waste source: reserving `max_seq_len` slots for requests that use
far fewer tokens.

This matches the vLLM paper's finding that **60–80% of KV cache memory**
is wasted in traditional systems (Section 3).

## 6. Running the Benchmarks

```bash
# Pure calculation (instant)
uv run python scripts/benchmark_memory.py

# Live scheduler tracking (~1 minute CPU)
uv run python scripts/benchmark_live.py
```
