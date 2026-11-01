"""Batch scheduling contracts with real paged resources and observable token IDs."""
from unittest.mock import patch

import pytest
import torch

from llm.inference.request import RequestStatus
from llm.inference.sampler import Sampler, SamplingParams
from llm.inference.scheduler import Scheduler
from llm.inference.types import RunnerOutput
from _scheduler_support import TinyRunner, assert_invariants, submit


class BatchTinyRunner(TinyRunner):
    # This is a scheduler test double. Real one-forward batching and numerical
    # equivalence are tested separately with Qwen, not inferred from this loop.
    def __init__(self, num_blocks=32, block_size=2):
        super().__init__(num_blocks, block_size)
        self.batch_calls = []

    def decode(self, input_ids, request_id):
        raise AssertionError("Batch scheduling must call decode_batch")

    def decode_batch(self, input_ids, request_ids):
        assert input_ids.shape == (len(request_ids), 1)
        self.batch_calls.append((tuple(request_ids), input_ids.clone()))
        outputs = [
            self._forward(input_ids[i:i + 1], request_id, "decode").logits
            for i, request_id in enumerate(request_ids)
        ]
        return RunnerOutput(torch.cat(outputs, dim=0))


def setup_batch(max_batch_size=2, num_blocks=32, **kwargs):
    runner = BatchTinyRunner(num_blocks=num_blocks)
    scheduler = Scheduler(
        runner, Sampler(SamplingParams(do_sample=False)),
        max_batch_size=max_batch_size, **kwargs,
    )
    return scheduler, []


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_batch_size_must_be_a_positive_integer(size):
    with pytest.raises(ValueError, match="positive integer"):
        Scheduler(TinyRunner(), max_batch_size=size)


def test_batching_requires_runner_capability_but_default_does_not():
    Scheduler(TinyRunner())
    with pytest.raises(ValueError, match="batch decode runner"):
        Scheduler(TinyRunner(), max_batch_size=2)


def test_decode_batch_limit_order_and_each_requests_next_input():
    scheduler, accepted = setup_batch()
    a = submit(scheduler, accepted, "a", [2], 4)
    b = submit(scheduler, accepted, "b", [3, 4, 5], 4)
    c = submit(scheduler, accepted, "c", [6, 2], 4)
    for _ in accepted:
        scheduler.step()
    assert not scheduler.runner.batch_calls
    assert scheduler.step() == []
    assert scheduler.runner.batch_calls[-1][0] == ("a", "b")
    assert [r.request_id for r in scheduler.running] == ["c", "a", "b"]
    assert [len(r.generated_ids) for r in accepted] == [2, 2, 1]
    assert_invariants(scheduler, accepted)
    scheduler.step()
    assert scheduler.runner.batch_calls[-1][0] == ("c", "a")
    assert [len(r.generated_ids) for r in accepted] == [3, 2, 2]
    scheduler.run()
    assert all(r.status is RequestStatus.FINISHED for r in (a, b, c))
    assert all(len(ids) <= 2 for ids, _ in scheduler.runner.batch_calls)
    assert_invariants(scheduler, accepted)


def test_mixed_eos_length_limit_and_continuation_in_one_batch():
    scheduler, accepted = setup_batch(max_batch_size=3)
    eos = submit(scheduler, accepted, "eos", [2], 5, eos_after=2)
    limit = submit(scheduler, accepted, "limit", [3, 4], 2)
    continuing = submit(scheduler, accepted, "continuing", [5, 6, 2], 4)
    for _ in accepted:
        scheduler.step()
    assert scheduler.step() == [eos, limit]
    assert eos.generated_ids == [1, 7]
    assert list(scheduler.running) == [continuing]
    assert set(scheduler.cache.block_manager.block_tables) == {"continuing"}
    assert_invariants(scheduler, accepted)
    scheduler.run()
    assert scheduler.runner.batch_calls[-1][0] == ("continuing",)
    assert_invariants(scheduler, accepted)


def test_each_batch_row_uses_its_own_sampler():
    scheduler, accepted = setup_batch()
    a = submit(scheduler, accepted, "a", [2], 2)
    b = submit(scheduler, accepted, "b", [3, 4], 2)
    scheduler.step()
    scheduler.step()
    scheduler._samplers["b"] = Sampler(SamplingParams(do_sample=False))
    with patch.object(scheduler._samplers["a"], "sample", return_value=torch.tensor([[4]])) as sa:
        with patch.object(scheduler._samplers["b"], "sample", return_value=torch.tensor([[6]])) as sb:
            assert scheduler.step() == [a, b]
    assert a.generated_ids == [1, 4]
    assert b.generated_ids == [1, 6]
    assert sa.call_args.args[0].shape == (1, 1, 8)
    assert sb.call_args.args[0].shape == (1, 1, 8)
    assert_invariants(scheduler, accepted)


def test_cancelled_row_is_not_executed_and_id_can_be_reused():
    scheduler, accepted = setup_batch(max_batch_size=3)
    a = submit(scheduler, accepted, "a", [2], 4)
    b = submit(scheduler, accepted, "b", [3, 4], 4)
    c = submit(scheduler, accepted, "c", [5, 6, 2], 4)
    for _ in accepted:
        scheduler.step()
    assert scheduler.cancel_request("b")
    scheduler.step()
    assert scheduler.runner.batch_calls[-1][0] == ("a", "c")
    assert b.generated_ids == [1]
    replacement = submit(scheduler, accepted, "b", [6], 2)
    scheduler.step()
    scheduler.run()
    assert replacement.status is RequestStatus.FINISHED
    assert b.status is RequestStatus.CANCELLED
    assert all(r.status is RequestStatus.FINISHED for r in (a, c))
    assert_invariants(scheduler, accepted)


def test_capacity_reservations_survive_batch_selection():
    scheduler, accepted = setup_batch(num_blocks=4)
    for name in ("a", "b", "c"):
        submit(scheduler, accepted, name, [2], 3)
    scheduler.step()
    scheduler.step()
    assert [r.request_id for r in scheduler.waiting] == ["c"]
    scheduler.step()
    assert scheduler.runner.batch_calls[-1][0] == ("a", "b")
    assert [r.request_id for r in scheduler.waiting] == ["c"]
    assert_invariants(scheduler, accepted)
    scheduler.run()
    assert_invariants(scheduler, accepted)


@pytest.mark.parametrize("mode", ["step", "run"])
@pytest.mark.parametrize("stage", ["forward", "sample", "append"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_failure_cleans_executing_batch_with_correct_scope(mode, stage, error_type):
    scheduler, accepted = setup_batch(record_timing=True)
    for name in ("a", "b", "c"):
        submit(scheduler, accepted, name, [2, 3], 4)
    for _ in accepted:
        scheduler.step()
    if stage == "forward":
        target, method = scheduler.runner, "decode_batch"
    elif stage == "sample":
        scheduler._samplers["b"] = Sampler(SamplingParams(do_sample=False))
        target, method = scheduler._samplers["b"], "sample"
    else:
        target, method = scheduler.cache, "append"
    original = getattr(target, method)

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise error_type("batch failure")

    with patch.object(target, method, side_effect=fail):
        with pytest.raises(error_type, match="batch failure"):
            getattr(scheduler, mode)()
    a, b, c = accepted
    assert a.status is b.status is RequestStatus.FAILED
    if mode == "step":
        assert list(scheduler.running) == [c]
        assert c.status is RequestStatus.DECODING
    else:
        assert scheduler.is_done
        assert c.status is RequestStatus.CANCELLED
    assert all(len(r.generated_ids) == len(r.timing.token_ready_at_ns) for r in accepted)
    assert_invariants(scheduler, accepted)
    scheduler.run()
    assert_invariants(scheduler, accepted)


def test_partial_commit_failure_does_not_finish_completed_row_twice():
    scheduler, accepted = setup_batch()
    a = submit(scheduler, accepted, "a", [2], 2)
    b = submit(scheduler, accepted, "b", [3], 4)
    scheduler.step()
    scheduler.step()
    with patch.object(scheduler.cache, "append", side_effect=RuntimeError("append failed")):
        with pytest.raises(RuntimeError, match="append failed"):
            scheduler.step()
    assert a.status is RequestStatus.FINISHED
    assert b.status is RequestStatus.FAILED
    assert scheduler.finished == [a, b]
    assert_invariants(scheduler, accepted)


def test_invalid_batch_output_fails_every_batch_member():
    scheduler, accepted = setup_batch()
    for name in ("a", "b"):
        submit(scheduler, accepted, name, [2], 3)
    scheduler.step()
    scheduler.step()
    with patch.object(scheduler.runner, "decode_batch", return_value=RunnerOutput(torch.zeros(1, 1, 8))):
        with pytest.raises(ValueError, match="Batch decode logits"):
            scheduler.step()
    assert all(r.status is RequestStatus.FAILED for r in accepted)
    assert_invariants(scheduler, accepted)


def test_batch_tokens_share_ready_time_after_all_outputs_are_committed():
    events = []
    ticks = iter(range(10, 1000, 10))
    scheduler, accepted = setup_batch(record_timing=True, clock=lambda: next(ticks))
    for name in ("a", "b"):
        submit(scheduler, accepted, name, [2], 2)
    scheduler.step()
    scheduler.step()

    def inspect_clock():
        events.append([len(r.generated_ids) for r in accepted])
        return next(ticks)

    scheduler._clock = inspect_clock
    scheduler.step()
    a, b = accepted
    assert events[0] == [2, 2]
    assert a.timing.token_ready_at_ns[-1] == b.timing.token_ready_at_ns[-1]
    assert a.timing.finished_at_ns > a.timing.token_ready_at_ns[-1]
    assert b.timing.finished_at_ns > b.timing.token_ready_at_ns[-1]
    assert_invariants(scheduler, accepted)
