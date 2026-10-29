"""Lifecycle timestamps use an injected monotonic clock, never sleeps."""
from dataclasses import asdict
from unittest.mock import patch

import pytest

from llm.inference.request import Request, RequestStatus
from llm.inference.sampler import Sampler, SamplingParams
from llm.inference.scheduler import Scheduler
from _scheduler_support import TinyRunner, submit


class FakeClock:
    def __init__(self):
        self.calls = 0
        self.on_tick = None

    def __call__(self):
        self.calls += 1
        value = self.calls * 10
        if self.on_tick:
            self.on_tick(value)
        return value


def setup_timed():
    clock = FakeClock()
    scheduler = Scheduler(
        TinyRunner(num_blocks=8, block_size=2),
        Sampler(SamplingParams(do_sample=False)),
        record_timing=True, clock=clock,
    )
    return scheduler, clock, []


def test_exact_submission_admission_tokens_and_completion_times():
    scheduler, clock, accepted = setup_timed()
    request = submit(scheduler, accepted, "r", [2], 3)
    assert request.timing.submitted_at_ns == 10
    assert request.timing.admitted_at_ns is None
    assert request.timing.token_ready_at_ns == []
    assert request.timing.finished_at_ns is None
    scheduler.run()
    assert asdict(request.timing) == {
        "submitted_at_ns": 10,
        "admitted_at_ns": 20,
        "token_ready_at_ns": [30, 40, 50],
        "finished_at_ns": 60,
    }
    assert clock.calls == 6


def test_default_disabled_timing_never_calls_the_clock():
    def forbidden_clock():
        raise AssertionError("Timing is disabled")

    scheduler = Scheduler(TinyRunner(), clock=forbidden_clock)
    accepted = []
    completed = submit(scheduler, accepted, "done", [2], 2)
    scheduler.run()
    cancelled = submit(scheduler, accepted, "cancel", [3], 3)
    assert scheduler.cancel_request("cancel")
    scheduler.close()
    assert completed.timing is None
    assert cancelled.timing is None


def test_token_timestamp_follows_cpu_read_and_completion_follows_release():
    scheduler, clock, accepted = setup_timed()
    request = submit(scheduler, accepted, "r", [2], 1)
    events = []

    class Token:
        def item(self):
            events.append("cpu_read")
            return 1

    def inspect(value):
        if value == 20:
            assert not scheduler.cache.block_manager.block_tables
        elif value == 30:
            assert events == ["cpu_read"]
            assert request.generated_ids == [1]
        elif value == 40:
            assert request.status is RequestStatus.FINISHED
            assert scheduler.finished == [request]
            assert "r" not in scheduler._samplers
            assert not scheduler.cache.block_manager.block_tables
            assert scheduler.cache.block_manager.allocator.num_free == 8

    clock.on_tick = inspect
    with patch.object(scheduler.default_sampler, "sample", return_value=Token()):
        scheduler.run()
    assert clock.calls == 4


def test_zero_output_has_no_admission_or_token_events():
    scheduler, clock, accepted = setup_timed()
    request = submit(scheduler, accepted, "zero", [], 0)
    assert scheduler.step() == [request]
    assert asdict(request.timing) == {
        "submitted_at_ns": 10, "admitted_at_ns": None,
        "token_ready_at_ns": [], "finished_at_ns": 20,
    }
    assert not scheduler.runner.calls
    assert clock.calls == 2


@pytest.mark.parametrize("started", [False, True])
def test_cancel_retains_events_and_repeated_cancel_does_not_change_time(started):
    scheduler, clock, accepted = setup_timed()
    request = submit(scheduler, accepted, "r", [2], 3)
    if started:
        scheduler.step()
    token_times = list(request.timing.token_ready_at_ns)
    assert scheduler.cancel_request("r")
    assert request.timing.token_ready_at_ns == token_times
    assert request.timing.admitted_at_ns == (20 if started else None)
    assert request.timing.finished_at_ns == (40 if started else 20)
    saved = asdict(request.timing)
    calls = clock.calls
    assert not scheduler.cancel_request("r")
    assert not scheduler.cancel_request("unknown")
    scheduler.close()
    assert asdict(request.timing) == saved
    assert clock.calls == calls


def test_close_records_waiting_and_running_once():
    scheduler, clock, accepted = setup_timed()
    first = submit(scheduler, accepted, "first", [2], 3)
    waiting = submit(scheduler, accepted, "waiting", [3], 3)
    scheduler.step()
    scheduler.close()
    assert first.timing.token_ready_at_ns == [40]
    assert first.timing.finished_at_ns == 60
    assert waiting.timing.admitted_at_ns is None
    assert waiting.timing.token_ready_at_ns == []
    assert waiting.timing.finished_at_ns == 50
    scheduler.close()
    assert clock.calls == 6


@pytest.mark.parametrize("prompt,limit", [([], 1), ([2], -1), ([2] * 33, 1), ([2] * 17, 1)])
def test_rejection_has_no_timing_or_clock_calls(prompt, limit):
    scheduler, clock, _ = setup_timed()
    request = Request("bad", prompt, limit)
    with pytest.raises(ValueError):
        scheduler.add_request(request)
    assert request.timing is None
    assert clock.calls == 0
    assert not scheduler._samplers


def test_duplicate_rejection_does_not_modify_existing_timing():
    scheduler, clock, accepted = setup_timed()
    original = submit(scheduler, accepted, "r", [2], 1)
    duplicate = Request("r", [3], 1)
    with pytest.raises(ValueError):
        scheduler.add_request(duplicate)
    assert duplicate.timing is None
    assert original.timing.submitted_at_ns == 10
    assert clock.calls == 1
    scheduler.run()


def test_reused_id_has_independent_timing_and_token_lists():
    scheduler, _, accepted = setup_timed()
    previous = submit(scheduler, accepted, "r", [2], 1)
    scheduler.run()
    saved = asdict(previous.timing)
    current = submit(scheduler, accepted, "r", [3], 2)
    scheduler.run()
    assert asdict(previous.timing) == saved
    assert current.timing is not previous.timing
    assert current.timing.token_ready_at_ns is not previous.timing.token_ready_at_ns
    assert current.timing.submitted_at_ns > previous.timing.finished_at_ns


@pytest.mark.parametrize("eos_after", [1, 2])
def test_early_eos_timestamps_match_actual_output_count(eos_after):
    scheduler, clock, accepted = setup_timed()
    request = submit(scheduler, accepted, "eos", [2], 5, eos_after=eos_after)
    scheduler.run()
    assert request.generated_ids[-1] == 7
    assert len(request.timing.token_ready_at_ns) == eos_after
    assert request.timing.finished_at_ns == (eos_after + 3) * 10
    assert clock.calls == eos_after + 3


@pytest.mark.parametrize("mode", ["step", "run"])
@pytest.mark.parametrize("stage", ["allocate", "prefill", "decode", "append", "sample"])
def test_failure_records_terminal_time_for_exact_cleanup_scope(mode, stage):
    scheduler, clock, accepted = setup_timed()
    survivor = submit(scheduler, accepted, "survivor", [2], 5)
    scheduler.step()
    victim = submit(scheduler, accepted, "victim", [3], 3)
    if stage == "decode":
        scheduler.step()
        scheduler.step()
    target = scheduler.runner if stage in ("prefill", "decode") else scheduler.cache
    if stage == "sample":
        target = scheduler.default_sampler
    original = getattr(target, stage)
    triggered = []

    def fail_once(*args, **kwargs):
        if not triggered:
            triggered.append(True)
            if stage in ("allocate", "append"):
                original(*args, **kwargs)
            raise RuntimeError("injected timing failure")
        return original(*args, **kwargs)

    with patch.object(target, stage, side_effect=fail_once):
        with pytest.raises(RuntimeError, match="injected timing failure"):
            getattr(scheduler, mode)()
    assert victim.status is RequestStatus.FAILED
    assert victim.timing.finished_at_ns is not None
    assert len(victim.timing.token_ready_at_ns) == len(victim.generated_ids)
    assert victim.timing.finished_at_ns >= victim.timing.admitted_at_ns
    if mode == "step":
        assert survivor.timing.finished_at_ns is None
        scheduler.run()
    else:
        assert survivor.status is RequestStatus.CANCELLED
        assert survivor.timing.finished_at_ns > victim.timing.finished_at_ns
    assert survivor.timing.finished_at_ns is not None
    calls = clock.calls
    scheduler.close()
    assert clock.calls == calls
