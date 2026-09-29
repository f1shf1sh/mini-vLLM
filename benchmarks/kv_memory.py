"""Compare contiguous vs paged KV cache memory usage.

Pure calculation — no model needed, no GPU needed.
"""

import random


# Illustrative standard-KV dimensions; no model or runtime dependency
NUM_LAYERS = 2
NUM_HEADS = 2
HEAD_DIM = 8
DTYPE_BYTES = 4   # float32

BLOCK_SIZE = 16

# Bytes per token's KV cache (one layer, one token)
BYTES_PER_TOKEN = 2 * NUM_LAYERS * NUM_HEADS * HEAD_DIM * DTYPE_BYTES
# = 2 * 2 * 2 * 8 * 4 = 256 bytes per token


def main():
    # Simulate a workload: 50 requests, random prompt lengths
    random.seed(42)
    max_seq_len = 256
    requests = [random.randint(10, 200) for _ in range(50)]

    # — contiguous —
    # Every request reserves max_seq_len upfront
    contiguous_total = len(requests) * max_seq_len * BYTES_PER_TOKEN
    contiguous_waste = sum(
        (max_seq_len - length) * BYTES_PER_TOKEN for length in requests
    )

    # — paged —
    paged_total = 0
    paged_waste = 0
    for length in requests:
        num_blocks = (length + BLOCK_SIZE - 1) // BLOCK_SIZE
        allocated = num_blocks * BLOCK_SIZE  # how many slots allocated
        paged_total += allocated * BYTES_PER_TOKEN
        paged_waste += (allocated - length) * BYTES_PER_TOKEN

    # — results —
    print(f"Workload: {len(requests)} requests, prompt lengths {min(requests)}–{max(requests)}")
    print(f"Bytes per token:  {BYTES_PER_TOKEN}")
    print(f"Max request len:  {max_seq_len}")
    print()
    print(f"{'':>20} {'Contiguous':>15} {'Paged':>15}")
    print(f"{'Total memory':>20} {contiguous_total:>13,} B {paged_total:>13,} B")
    print(f"{'Total memory (KB)':>20} {contiguous_total/1024:>13.1f}   {paged_total/1024:>13.1f}")
    print(f"{'Wasted memory':>20} {contiguous_waste:>13,} B {paged_waste:>13,} B")
    print(f"{'Utilization':>20} {(1-contiguous_waste/contiguous_total)*100:>12.1f}% {(1-paged_waste/paged_total)*100:>12.1f}%")
    print(f"{'Memory saved':>20} {'—':>15} {(1 - paged_total/contiguous_total)*100:>12.1f}%")


if __name__ == "__main__":
    main()
