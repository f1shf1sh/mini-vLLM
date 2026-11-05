"""Small real-cache runner and ownership checks shared by scheduler tests."""
from collections import Counter

import torch

from llm.cache.blocks import BlockAllocator, BlockManager
from llm.cache.paged import PagedCacheResources
from llm.cache.pool import KVPool
from llm.inference.request import Request, RequestStatus
from llm.inference.runner import RunnerSpec
from llm.inference.sampler import Sampler, SamplingParams
from llm.inference.scheduler import Scheduler
from llm.inference.types import RunnerOutput


class TinyRunner:
    """Write token IDs as K/V so isolation and next-input errors are observable."""

    def __init__(self, num_blocks=4, block_size=2):
        self.spec = RunnerSpec(32, torch.device("cpu"), torch.float32)
        pool = KVPool(num_blocks, 1, 1, block_size, 1)
        self.cache = PagedCacheResources(
            pool, BlockManager(BlockAllocator(num_blocks), block_size),
        )
        self.requests = {}
        self.generations = {}
        self.eos_after = {}
        self.calls = []

    def register(self, request, generation, eos_after=None):
        self.requests[request.request_id] = request
        self.generations[id(request)] = generation
        self.eos_after[id(request)] = eos_after

    def prefill(self, input_ids, request_id):
        return self._forward(input_ids, request_id, "prefill")

    def decode(self, input_ids, request_id):
        return self._forward(input_ids, request_id, "decode")

    def _forward(self, input_ids, request_id, stage):
        request = self.requests[request_id]
        expected_status = (
            RequestStatus.PREFILLING if stage == "prefill" else RequestStatus.DECODING
        )
        assert request.status is expected_status
        assert not request.is_finished
        assert not request.should_stop
        expected_input = (
            request.input_ids if stage == "prefill" else request.generated_ids[-1:]
        )
        assert input_ids.tolist() == [expected_input]
        view = self.cache.view(request_id)
        past = [] if stage == "prefill" else request.input_ids + request.generated_ids[:-1]
        assert view.num_tokens == len(past) + len(expected_input)
        old_k, old_v = view.gather(0, len(past))
        expected_past = torch.tensor(past, dtype=torch.float32).reshape(1, 1, -1, 1)
        torch.testing.assert_close(old_k, expected_past)
        torch.testing.assert_close(old_v, -expected_past)
        values = input_ids.to(torch.float32).reshape(1, 1, -1, 1)
        view.write_range(0, values, -values, start_pos=len(past))
        self.calls.append((stage, self.generations[id(request)]))
        stop_at = self.eos_after[id(request)]
        token = 7 if stop_at == len(request.generated_ids) + 1 else 1
        logits = torch.full((1, input_ids.shape[1], 8), -10.0)
        logits[:, :, token] = 10.0
        return RunnerOutput(logits)


def make_scheduler(num_blocks=4, block_size=2):
    runner = TinyRunner(num_blocks, block_size)
    return Scheduler(runner, Sampler(SamplingParams(do_sample=False)))


def submit(scheduler, accepted, request_id, prompt, limit, eos_after=None):
    request = Request(
        request_id, list(prompt), limit, sampling_params=None,
        stop_token_ids=(7,) if eos_after is not None else (),
    )
    scheduler.add_request(request)
    scheduler.runner.register(request, len(accepted), eos_after)
    accepted.append(request)
    return request


def assert_invariants(scheduler, accepted, terminal_snapshots=None):
    """Check conservation and public outcomes, without simulating scheduling."""
    runner = scheduler.runner
    cache = runner.cache
    manager = cache.block_manager
    allocator = manager.allocator
    waiting = list(scheduler.waiting)
    running = list(scheduler.running)
    active = waiting + running
    finished = scheduler.finished
    finished_objects = {id(r) for r in finished}
    all_objects = [id(r) for r in active + finished]
    assert len(all_objects) == len(set(all_objects))
    assert set(all_objects) == {id(r) for r in accepted}
    assert len({r.request_id for r in active}) == len(active)
    assert set(scheduler._samplers) == {r.request_id for r in active}
    running_ids = {r.request_id for r in running}
    assert set(scheduler._next_inputs) == running_ids
    assert set(manager.block_tables) == running_ids
    assert scheduler.is_done == (not active)

    free = list(allocator.free_blocks)
    owned = [b for t in manager.block_tables.values() for b in t.physical_blocks]
    assert len(free) == len(set(free))
    assert len(owned) == len(set(owned))
    assert set(owned) == allocator.allocated
    assert not set(free) & set(owned)
    assert set(free) | set(owned) == set(range(cache.capacity))
    assert len(free) + len(owned) == cache.capacity

    block_size = cache.pool.block_size
    reserved = 0
    for request in waiting:
        assert request.status is RequestStatus.WAITING
        assert not request.generated_ids
    for request in running:
        assert request.status is RequestStatus.DECODING
        assert not request.is_finished
        assert not request.should_stop
        assert request.generated_ids
        table = manager.block_tables[request.request_id]
        assert table.num_tokens == request.prompt_length + len(request.generated_ids)
        assert len(table.physical_blocks) == (table.num_tokens + block_size - 1) // block_size
        assert scheduler._next_inputs[request.request_id].tolist() == [request.generated_ids[-1:]]
        required_tokens = request.prompt_length + request.max_new_tokens - 1
        reserved += (required_tokens + block_size - 1) // block_size
    assert reserved <= cache.capacity

    call_counts = Counter(generation for _, generation in runner.calls)
    for generation, request in enumerate(accepted):
        assert 0 <= len(request.generated_ids) <= request.max_new_tokens
        assert request.input_ids + request.generated_ids == request.output_ids
        if request.stop_token_ids:
            assert not any(token in request.stop_token_ids for token in request.generated_ids[:-1])
        if request.max_new_tokens == 0:
            assert call_counts[generation] == 0
        if id(request) in finished_objects:
            assert request.status in (
                RequestStatus.FINISHED, RequestStatus.FAILED, RequestStatus.CANCELLED,
            )
            if request.status is RequestStatus.FINISHED:
                assert (
                    len(request.generated_ids) == request.max_new_tokens
                    or request.generated_ids[-1] in request.stop_token_ids
                )
            if terminal_snapshots is not None:
                snapshot = (request.status, tuple(request.generated_ids), call_counts[generation])
                assert terminal_snapshots.setdefault(generation, snapshot) == snapshot
