"""Track block usage while scheduler processes real requests."""
import random
import torch

from lmcore.config import ModelConfig
from lmcore.inference.block_manager import BlockManager, BlockAllocator
from lmcore.inference.kv_pool import KVPool
from lmcore.inference.request import Request
from lmcore.inference.scheduler import Scheduler
from lmcore.inference.sampler import SamplingParams
from lmcore.model import GPTMini


def make_model():
    config = ModelConfig(
        vocab_size=16, block_size=256, n_layer=2, n_head=2, n_embd=8,
        dropout=0.0, bias=False,
    )
    model = GPTMini(config)
    model.eval()
    return model


def make_scheduler(model, num_blocks):
    pool = KVPool(
        num_blocks=num_blocks, num_layers=model.config.n_layer,
        num_heads=model.config.n_head, block_size=16,
        head_dim=model.config.n_embd // model.config.n_head,
    )
    allocator = BlockAllocator(num_blocks=num_blocks)
    bm = BlockManager(allocator=allocator, block_size=16)
    return Scheduler(model, kv_pool=pool, block_manager=bm)


def main():
    torch.manual_seed(42)
    random.seed(42)

    model = make_model()
    num_blocks = 1000

    # Create requests
    requests = [
        Request(f"r{i}", [random.randint(0, 15) for _ in range(random.randint(10, 200))],
                max_new_tokens=random.randint(1, 30),
                sampling_params=SamplingParams(temperature=1.0))
        for i in range(50)
    ]

    # —— PAGED ——
    sched = make_scheduler(model, num_blocks)
    for req in requests:
        sched.add_request(req)

    peak_allocated = 0
    timesteps = []

    while not sched.is_done:
        sched.step()
        used = num_blocks - sched.block_manager.allocator.num_free
        peak_allocated = max(peak_allocated, used)
        timesteps.append(used)

    avg_allocated = sum(timesteps) / len(timesteps)

    # —— CONTINUOUS (calculated) ——
    max_len = 256
    contiguous_blocks = (len(requests) * max_len) // 16  # blocks of 16

    print(f"Requests:           {len(requests)}")
    print(f"Peak blocks used:   {peak_allocated} / {num_blocks}")
    print(f"Average blocks used:{avg_allocated:.1f}")
    print(f"Contiguous needed:  {contiguous_blocks} blocks")
    print(f"Block utilization:  {peak_allocated/num_blocks*100:.1f}%")
    print(f"Vs contiguous:      {peak_allocated/contiguous_blocks*100:.1f}%")
    print(f"Memory saved:       {(1 - peak_allocated/contiguous_blocks)*100:.1f}%")
    print()

    # —— text histogram ——
    print(f"Block usage over {len(timesteps)} steps:")
    buckets = 20
    bucket_size = max(1, len(timesteps) // buckets)
    for i in range(0, len(timesteps), bucket_size):
        chunk = timesteps[i:i + bucket_size]
        mean = sum(chunk) / len(chunk)
        bar = "#" * int(mean / peak_allocated * 40)
        print(f"  step {i:>5d}: {bar:<40s} {mean:.0f}")


def _sweep():
    """Quick parameter sweep: vary request count."""
    print("=" * 55)
    print("  Parameter sweep: request count vs memory")
    print("=" * 55)
    print(f"  {'#req':>6} {'contig_blks':>12} {'peak_paged':>12} {'saved':>8}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*8}")

    for n_req in [10, 20, 30, 40, 50]:
        random.seed(42)
        model = make_model()
        num_blocks = 1000
        sched = make_scheduler(model, num_blocks)
        for i in range(n_req):
            req = Request(
                f"r{i}",
                [random.randint(0, 15) for _ in range(random.randint(10, 200))],
                max_new_tokens=random.randint(1, 30),
                sampling_params=SamplingParams(temperature=1.0),
            )
            sched.add_request(req)

        peak = 0
        while not sched.is_done:
            sched.step()
            used = num_blocks - sched.block_manager.allocator.num_free
            peak = max(peak, used)

        contig = n_req * 256 // 16
        print(f"  {n_req:>6} {contig:>12} {peak:>12} {(1-peak/contig)*100:>7.1f}%")

    print()


if __name__ == "__main__":
    _sweep()
    main()
