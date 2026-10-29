"""Scheduler lifecycle regression tests against the actual paged resource layer."""
from unittest.mock import patch

import pytest

from llm.inference.request import RequestStatus

from _scheduler_support import assert_invariants, make_scheduler, submit


@pytest.mark.parametrize("prompt,limit,eos_after", [
    ([2, 3, 4, 5], 4, 1),
    ([2, 3, 4], 4, 2),
    ([2, 3, 4], 2, None),
])
def test_final_token_at_block_boundary_never_allocates_an_unused_block(prompt, limit, eos_after):
    scheduler = make_scheduler(num_blocks=3, block_size=4)
    accepted = []
    request = submit(scheduler, accepted, "last", prompt, limit, eos_after)
    allocator = scheduler.cache.block_manager.allocator
    with patch.object(allocator, "allocate", wraps=allocator.allocate) as allocate:
        while not scheduler.is_done:
            scheduler.step()
            assert_invariants(scheduler, accepted)
    assert allocate.call_count == 1
    assert request.num_generated_tokens == (eos_after or limit)
    assert request.status is RequestStatus.FINISHED
    if eos_after:
        assert request.generated_ids[-1] == 7
    assert allocator.num_free == scheduler.cache.capacity


def test_zero_generation_does_not_allocate_or_execute():
    scheduler = make_scheduler(num_blocks=1, block_size=1)
    accepted = []
    request = submit(scheduler, accepted, "zero", [], 0)
    with patch.object(scheduler.cache, "allocate", wraps=scheduler.cache.allocate) as allocate:
        assert scheduler.step() == [request]
    allocate.assert_not_called()
    assert not scheduler.runner.calls
    assert_invariants(scheduler, accepted)


def test_cancelling_running_request_unblocks_fifo_admission_with_real_blocks():
    scheduler = make_scheduler(num_blocks=2, block_size=2)
    accepted = []
    first = submit(scheduler, accepted, "first", [2, 3], 3)
    second = submit(scheduler, accepted, "second", [4], 2)
    scheduler.step()
    scheduler.step()
    assert first.generated_ids == [1, 1]
    assert list(scheduler.waiting) == [second]
    assert scheduler.cache.block_manager.allocator.num_free == 0
    assert scheduler.cancel_request("first") is True
    assert first.status is RequestStatus.CANCELLED
    assert scheduler.cache.block_manager.allocator.num_free == 2
    assert_invariants(scheduler, accepted)
    scheduler.step()
    assert list(scheduler.running) == [second]
    scheduler.run()
    assert second.status is RequestStatus.FINISHED
    assert_invariants(scheduler, accepted)


def test_id_reuse_keeps_old_terminal_request_and_cancels_only_new_owner():
    scheduler = make_scheduler()
    accepted = []
    previous = submit(scheduler, accepted, "same", [2], 1)
    scheduler.run()
    current = submit(scheduler, accepted, "same", [3], 3)
    survivor = submit(scheduler, accepted, "other", [4], 3)
    scheduler.step()
    scheduler.step()
    other_blocks = list(scheduler.cache.block_manager.block_tables["other"].physical_blocks)
    assert scheduler.cancel_request("same")
    assert not scheduler.cancel_request("same")
    assert previous.status is RequestStatus.FINISHED
    assert current.status is RequestStatus.CANCELLED
    assert scheduler.cache.block_manager.block_tables["other"].physical_blocks == other_blocks
    scheduler.run()
    assert survivor.status is RequestStatus.FINISHED
    assert_invariants(scheduler, accepted)


def test_resource_release_is_idempotent_but_allocator_double_free_is_an_error():
    scheduler = make_scheduler()
    cache = scheduler.cache
    allocator = cache.block_manager.allocator
    cache.release("missing")
    cache.allocate("request", 3)
    cache.release("request")
    cache.release("request")
    assert allocator.num_free == cache.capacity
    block = allocator.allocate()
    allocator.free(block)
    with pytest.raises(ValueError, match="not allocated"):
        allocator.free(block)
    assert allocator.num_free == cache.capacity


@pytest.mark.parametrize("mode", ["step", "run"])
@pytest.mark.parametrize("stage", ["allocate", "prefill", "decode", "append", "sample"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_failure_cleans_real_resources_and_preserves_step_vs_run_semantics(mode, stage, error_type):
    scheduler = make_scheduler(num_blocks=8, block_size=2)
    accepted = []
    survivor = submit(scheduler, accepted, "survivor", [2], 6)
    scheduler.step()
    victim = submit(scheduler, accepted, "victim", [3], 4)
    if stage == "decode":
        scheduler.step()
        scheduler.step()

    target = scheduler.runner if stage in ("prefill", "decode") else scheduler.cache
    if stage == "sample":
        target = scheduler.default_sampler
    original = getattr(target, stage)
    failures = []

    def fail_once(*args, **kwargs):
        request_id = args[-1] if stage in ("prefill", "decode") else args[0]
        is_victim = stage == "sample" or request_id == "victim"
        if is_victim and not failures:
            failures.append(stage)
            if stage in ("allocate", "append"):
                original(*args, **kwargs)
            raise error_type("injected lifecycle failure")
        return original(*args, **kwargs)

    with patch.object(target, stage, side_effect=fail_once):
        with pytest.raises(error_type, match="injected lifecycle failure"):
            getattr(scheduler, mode)()
    assert failures == [stage]
    assert victim.status is RequestStatus.FAILED
    assert "victim" not in scheduler.cache.block_manager.block_tables
    assert_invariants(scheduler, accepted)
    if mode == "step":
        assert survivor.status is RequestStatus.DECODING
        assert "survivor" in scheduler.cache.block_manager.block_tables
        scheduler.run()
        assert survivor.status is RequestStatus.FINISHED
    else:
        assert survivor.status is RequestStatus.CANCELLED
        assert scheduler.is_done
    scheduler.close()
    scheduler.close()
    assert_invariants(scheduler, accepted)


def test_partial_physical_allocation_failure_rolls_back_without_touching_other_owner():
    scheduler = make_scheduler(num_blocks=4, block_size=2)
    accepted = []
    survivor = submit(scheduler, accepted, "survivor", [2], 2)
    scheduler.step()
    victim = submit(scheduler, accepted, "victim", [3] * 5, 1)
    allocator = scheduler.cache.block_manager.allocator
    original = allocator.allocate
    attempts = []

    def fail_second_block():
        attempts.append(None)
        if len(attempts) == 2:
            raise RuntimeError("physical allocation failure")
        return original()

    with patch.object(allocator, "allocate", side_effect=fail_second_block):
        with pytest.raises(RuntimeError, match="physical allocation failure"):
            scheduler.step()
    assert len(attempts) == 2
    assert victim.status is RequestStatus.FAILED
    assert survivor.status is RequestStatus.DECODING
    assert len(allocator.allocated) == 1
    assert_invariants(scheduler, accepted)
    scheduler.run()
    assert_invariants(scheduler, accepted)
