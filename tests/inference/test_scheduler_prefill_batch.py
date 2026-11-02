"""Prefill scheduling contracts; model padding and attention are tested separately."""
from unittest.mock import patch

import pytest
import torch

from llm.inference.request import Request, RequestStatus
from llm.inference.sampler import Sampler, SamplingParams
from llm.inference.scheduler import Scheduler
from llm.inference.types import RunnerOutput
from _scheduler_support import TinyRunner, assert_invariants, submit


class PrefillTinyRunner(TinyRunner):
    """Reuse the existing single-request test double to observe scheduler inputs.

    This loop is not a batched model implementation and cannot prove one-forward
    numerical equivalence. It adds no mask, position, or paged-writing algorithm.
    """

    def __init__(self, num_blocks=32, block_size=2):
        super().__init__(num_blocks, block_size)
        self.prefill_calls = []

    def prefill_batch(self, input_ids, request_ids):
        assert input_ids and len(input_ids) == len(request_ids)
        # All rows must be prepared before the first row can write any KV.
        for request_id, ids in zip(request_ids, input_ids):
            assert self.requests[request_id].status is RequestStatus.PREFILLING
            assert self.cache.view(request_id).num_tokens == len(ids)
        self.prefill_calls.append((tuple(request_ids), [list(ids) for ids in input_ids]))
        outputs = []
        for request_id, ids in zip(request_ids, input_ids):
            output = self._forward(torch.tensor([ids]), request_id, "prefill")
            outputs.append(output.logits[:, -1:, :])
        return RunnerOutput(torch.cat(outputs, dim=0))


def setup_prefill(num_blocks=32, **kwargs):
    runner = PrefillTinyRunner(num_blocks=num_blocks)
    kwargs.setdefault("max_prefill_batch_size", 2)
    scheduler = Scheduler(runner, Sampler(SamplingParams(do_sample=False)), **kwargs)
    return scheduler, []


def assert_fully_released(scheduler):
    # Observe natural cleanup; never release resources inside this assertion.
    manager = scheduler.cache.block_manager
    assert not manager.block_tables
    assert not manager.allocator.allocated
    assert sorted(manager.allocator.free_blocks) == list(range(scheduler.cache.capacity))


@pytest.mark.parametrize("parameter", ["max_prefill_batch_size", "max_prefill_tokens"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_prefill_limits_require_positive_integers(parameter, value):
    with pytest.raises(ValueError, match="positive integer"):
        Scheduler(PrefillTinyRunner(), **{parameter: value})


def test_prefill_capability_is_optional_and_independent_of_decode():
    Scheduler(TinyRunner(), max_prefill_tokens=3)
    with pytest.raises(ValueError, match="batch prefill runner"):
        Scheduler(TinyRunner(), max_prefill_batch_size=2)
    # This runner has no decode_batch capability; batched prefill still works.
    scheduler, accepted = setup_prefill()
    submit(scheduler, accepted, "a", [2], 2)
    submit(scheduler, accepted, "b", [3, 4], 2)
    scheduler.run()
    assert scheduler.runner.prefill_calls[0][0] == ("a", "b")
    assert_invariants(scheduler, accepted)
    assert_fully_released(scheduler)


def test_default_keeps_single_request_prefill_and_does_not_read_clock():
    def forbidden_clock():
        raise AssertionError("Timing is disabled")

    scheduler, accepted = setup_prefill(max_prefill_batch_size=1, clock=forbidden_clock)
    assert scheduler.max_prefill_tokens == scheduler.runner.spec.context_length
    submit(scheduler, accepted, "a", [2], 2)
    submit(scheduler, accepted, "b", [3, 4], 2)
    scheduler.run()
    assert not scheduler.runner.prefill_calls
    assert_invariants(scheduler, accepted)
    assert_fully_released(scheduler)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_oversize_prompt_rejection_has_no_side_effects_and_zero_output_bypasses_budget(batch_size):
    scheduler, accepted = setup_prefill(
        max_prefill_batch_size=batch_size, max_prefill_tokens=3, record_timing=True,
    )
    rejected = Request("reused", [2, 3, 4, 5], 2)
    with patch.object(scheduler, "_clock", side_effect=AssertionError("No timestamp on rejection")):
        with pytest.raises(ValueError, match="prefill token budget"):
            scheduler.add_request(rejected)
    assert rejected.status is RequestStatus.WAITING
    assert rejected.timing is None
    assert not scheduler.waiting and not scheduler._samplers
    assert_fully_released(scheduler)
    zero = submit(scheduler, accepted, "zero", [2, 3, 4, 5], 0)
    valid = submit(scheduler, accepted, "reused", [2, 3, 4], 1)
    scheduler.run()
    assert zero.timing.admitted_at_ns is None
    assert zero.timing.token_ready_at_ns == []
    assert valid.status is RequestStatus.FINISHED
    assert_invariants(scheduler, accepted)
    assert_fully_released(scheduler)


@pytest.mark.parametrize("prompts,budget", [
    ([[2, 3], [3, 4, 5, 6, 2], [4, 5, 6]], 12),
    ([[2, 3], [3, 4], [4, 5, 6, 2, 3]], 12),
    ([[2, 3], [3, 4, 5, 6, 2], [4]], 10),
])
def test_padded_budget_splits_fifo_batch_and_reserves_only_real_lengths(prompts, budget):
    scheduler, accepted = setup_prefill(max_prefill_batch_size=3, max_prefill_tokens=budget)
    for name, prompt in zip(("a", "b", "c"), prompts):
        submit(scheduler, accepted, name, prompt, 3)
    with patch.object(scheduler.cache, "allocate", wraps=scheduler.cache.allocate) as allocate:
        assert scheduler.step() == []
    assert [call.args for call in allocate.call_args_list] == [
        ("a", len(prompts[0])), ("b", len(prompts[1])),
    ]
    assert scheduler.runner.prefill_calls == [(("a", "b"), prompts[:2])]
    assert list(scheduler.waiting) == [accepted[2]]
    assert [request.input_ids for request in accepted] == prompts
    assert_invariants(scheduler, accepted)
    scheduler.step()
    assert scheduler.runner.prefill_calls[-1] == (("c",), [prompts[2]])
    assert_invariants(scheduler, accepted)
    scheduler.run()
    assert_fully_released(scheduler)


def test_request_count_limit_is_independent_of_token_budget():
    scheduler, accepted = setup_prefill(max_prefill_batch_size=2, max_prefill_tokens=30)
    for name in ("a", "b", "c"):
        submit(scheduler, accepted, name, [2], 1)
    assert scheduler.step() == accepted[:2]
    assert list(scheduler.waiting) == [accepted[2]]
    assert scheduler.step() == accepted[2:]
    assert [call[0] for call in scheduler.runner.prefill_calls] == [("a", "b"), ("c",)]
    assert_invariants(scheduler, accepted)
    assert_fully_released(scheduler)


def test_capacity_counts_running_and_selected_requests_without_skipping_fifo_head():
    scheduler, accepted = setup_prefill(num_blocks=5, max_prefill_batch_size=3)
    old = submit(scheduler, accepted, "old", [2], 3)
    scheduler.step()
    a = submit(scheduler, accepted, "a", [3], 3)
    b = submit(scheduler, accepted, "b", [4], 3)
    c = submit(scheduler, accepted, "c", [5], 1)
    # Each of old/a/b commits two blocks, though their initial prompt needs one.
    scheduler.step()
    assert scheduler.runner.prefill_calls[-1][0] == ("a",)
    assert list(scheduler.running) == [old, a]
    assert list(scheduler.waiting) == [b, c]
    assert_invariants(scheduler, accepted)
    call_count = len(scheduler.runner.prefill_calls)
    scheduler.step()
    assert len(scheduler.runner.prefill_calls) == call_count
    assert old.num_generated_tokens == 2
    assert_invariants(scheduler, accepted)
    scheduler.run()
    assert_invariants(scheduler, accepted)
    assert_fully_released(scheduler)


def test_zero_output_eos_limit_and_continuation_have_independent_lifecycles():
    ticks = iter(range(10, 1000, 10))
    scheduler, accepted = setup_prefill(
        max_prefill_batch_size=3, max_prefill_tokens=6, record_timing=True, clock=lambda: next(ticks),
    )
    empty = submit(scheduler, accepted, "empty", [], 0)
    eos = submit(scheduler, accepted, "eos", [2], 5, eos_after=1)
    zero = submit(scheduler, accepted, "zero", [3] * 7, 0)
    limit = submit(scheduler, accepted, "limit", [3, 4], 1)
    continuing = submit(scheduler, accepted, "continuing", [5], 3)
    with patch.object(scheduler.cache, "append", wraps=scheduler.cache.append) as append:
        completed = scheduler.step()
    assert {request.request_id for request in completed} == {"empty", "zero", "eos", "limit"}
    assert append.call_count == 1 and append.call_args.args == ("continuing",)
    assert scheduler.runner.prefill_calls[0][0] == ("eos", "limit", "continuing")
    assert list(scheduler.running) == [continuing]
    for request in (empty, zero):
        assert request.generated_ids == []
        assert request.timing.admitted_at_ns is None
        assert request.timing.token_ready_at_ns == []
        assert request.timing.finished_at_ns is not None
    ready_time = continuing.timing.token_ready_at_ns[0]
    for request in (eos, limit, continuing):
        assert request.timing.token_ready_at_ns == [ready_time]
        assert request.timing.submitted_at_ns < request.timing.admitted_at_ns < ready_time
    assert eos.timing.finished_at_ns > ready_time
    assert limit.timing.finished_at_ns > ready_time
    assert_invariants(scheduler, accepted)
    scheduler.run()
    assert_fully_released(scheduler)


def test_all_zero_output_requests_never_call_runner_or_allocate():
    scheduler, accepted = setup_prefill()
    for name in ("a", "b", "c"):
        submit(scheduler, accepted, name, [], 0)
    with patch.object(scheduler.cache, "allocate", side_effect=AssertionError("No allocation")):
        assert scheduler.step() == accepted
    assert not scheduler.runner.prefill_calls
    assert_invariants(scheduler, accepted)
    assert_fully_released(scheduler)


def test_each_prefill_row_uses_its_sampler_and_shared_ready_time_follows_token_commit():
    events = []
    ticks = iter(range(10, 1000, 10))
    scheduler, accepted = setup_prefill(record_timing=True, clock=lambda: next(ticks))
    a = submit(scheduler, accepted, "a", [2], 1)
    b = submit(scheduler, accepted, "b", [3, 4], 1)
    scheduler._samplers["b"] = Sampler(SamplingParams(do_sample=False))

    def inspect_clock():
        events.append([len(request.generated_ids) for request in accepted])
        return next(ticks)

    scheduler._clock = inspect_clock
    with patch.object(scheduler._samplers["a"], "sample", return_value=torch.tensor([[4]])) as first:
        with patch.object(scheduler._samplers["b"], "sample", return_value=torch.tensor([[6]])) as second:
            assert scheduler.step() == [a, b]
    assert a.generated_ids == [4] and b.generated_ids == [6]
    assert first.call_args.args[0].shape == second.call_args.args[0].shape == (1, 1, 8)
    assert events[:3] == [[0, 0], [0, 0], [1, 1]]
    assert a.timing.token_ready_at_ns == b.timing.token_ready_at_ns
    assert_invariants(scheduler, accepted)
    assert_fully_released(scheduler)


def test_new_arrival_and_cancelled_id_reuse_preserve_existing_request():
    scheduler, accepted = setup_prefill()
    a = submit(scheduler, accepted, "a", [2], 4)
    b = submit(scheduler, accepted, "b", [3, 4], 4)
    scheduler.step()
    table = scheduler.cache.block_manager.block_tables["b"]
    before = scheduler.cache.pool.pool[:, :, table.physical_blocks].clone()
    assert scheduler.cancel_request("a")
    replacement = submit(scheduler, accepted, "a", [5, 6, 2], 2)
    cancelled = submit(scheduler, accepted, "cancelled", [6], 3)
    assert scheduler.cancel_request("cancelled")
    scheduler.step()
    assert scheduler.runner.prefill_calls[-1] == (("a",), [[5, 6, 2]])
    assert b.num_generated_tokens == 1
    assert replacement.num_generated_tokens == 1
    torch.testing.assert_close(
        scheduler.cache.pool.pool[:, :, table.physical_blocks], before, rtol=0, atol=0,
    )
    assert a.status is cancelled.status is RequestStatus.CANCELLED
    assert_invariants(scheduler, accepted)
    scheduler.run()
    assert_fully_released(scheduler)


@pytest.mark.parametrize("shape", [(1, 1, 8), (2, 2, 8), (2, 8), (2, 1, 0)])
def test_invalid_prefill_output_releases_entire_executing_batch(shape):
    scheduler, accepted = setup_prefill()
    for name in ("a", "b"):
        submit(scheduler, accepted, name, [2], 3)
    with patch.object(scheduler.runner, "prefill_batch", return_value=RunnerOutput(torch.zeros(shape))):
        with pytest.raises(ValueError, match="Batch prefill logits"):
            scheduler.step()
    assert all(request.status is RequestStatus.FAILED for request in accepted)
    assert all(request.generated_ids == [] for request in accepted)
    assert_invariants(scheduler, accepted)
    assert_fully_released(scheduler)


@pytest.mark.parametrize("mode", ["step", "run"])
@pytest.mark.parametrize("stage", ["allocate", "forward", "sample", "append"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, NotImplementedError])
def test_partial_failure_cleans_selected_batch_and_preserves_nonparticipants(mode, stage, error_type):
    scheduler, accepted = setup_prefill(record_timing=True)
    old = submit(scheduler, accepted, "old", [6], 4)
    scheduler.step()
    a = submit(scheduler, accepted, "a", [2, 3], 4)
    b = submit(scheduler, accepted, "b", [4, 5], 4)
    waiting = submit(scheduler, accepted, "waiting", [3], 2)
    old_table = scheduler.cache.block_manager.block_tables["old"]
    old_pages = scheduler.cache.pool.pool[:, :, old_table.physical_blocks].clone()
    scheduler._samplers["b"] = Sampler(SamplingParams(do_sample=False))
    if stage == "allocate":
        target, method, fail_at = scheduler.cache, "allocate", 2
    elif stage == "forward":
        # Fail after row a writes KV, before row b writes: a truly partial forward.
        target, method, fail_at = scheduler.runner, "_forward", 1
    elif stage == "sample":
        target, method, fail_at = scheduler._samplers["b"], "sample", 1
    else:
        target, method, fail_at = scheduler.cache, "append", 2
    original = getattr(target, method)
    calls = 0

    def fail_after_operation(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == fail_at:
            raise error_type("prefill failure")
        return result

    with patch.object(target, method, side_effect=fail_after_operation):
        with pytest.raises(error_type, match="prefill failure"):
            getattr(scheduler, mode)()
    assert a.status is b.status is RequestStatus.FAILED
    if stage == "append":
        assert a.generated_ids == b.generated_ids == [1]
    else:
        assert a.generated_ids == b.generated_ids == []
    if mode == "step":
        assert list(scheduler.running) == [old]
        assert list(scheduler.waiting) == [waiting]
        torch.testing.assert_close(
            scheduler.cache.pool.pool[:, :, old_table.physical_blocks], old_pages, rtol=0, atol=0,
        )
    else:
        assert old.status is waiting.status is RequestStatus.CANCELLED
        assert_fully_released(scheduler)
    for request in accepted:
        assert len(request.generated_ids) == len(request.timing.token_ready_at_ns)
    assert_invariants(scheduler, accepted)
    scheduler.run()
    assert_invariants(scheduler, accepted)
    assert_fully_released(scheduler)


def test_later_growth_failure_does_not_finish_zero_or_completed_rows_twice():
    scheduler, accepted = setup_prefill()
    zero = submit(scheduler, accepted, "zero", [], 0)
    done = submit(scheduler, accepted, "done", [2], 1)
    growing = submit(scheduler, accepted, "growing", [3, 4], 3)
    original = scheduler.cache.append

    def fail_after_growth(request_id):
        original(request_id)
        raise RuntimeError("growth failed")

    with patch.object(scheduler.cache, "append", side_effect=fail_after_growth):
        with pytest.raises(RuntimeError, match="growth failed"):
            scheduler.step()
    assert zero.status is done.status is RequestStatus.FINISHED
    assert growing.status is RequestStatus.FAILED
    assert scheduler.finished == [zero, done, growing]
    assert_invariants(scheduler, accepted)
    assert_fully_released(scheduler)
