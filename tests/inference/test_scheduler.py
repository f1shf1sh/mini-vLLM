import torch
import pytest

from llm.inference.request import Request, RequestStatus
from llm.inference.runner import RunnerSpec
from llm.inference.sampler import Sampler, SamplingParams
from llm.inference.scheduler import Scheduler
from llm.inference.types import RunnerOutput


class FakeResources:
    # No tensors, KV shape, model config, or BlockManager dependency.
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.allocations = {}

    def required_capacity(self, num_tokens):
        return (num_tokens + 3) // 4

    @property
    def num_free(self):
        return self.capacity - sum(self.required_capacity(n) for n in self.allocations.values())

    def allocate(self, request_id, num_tokens):
        assert request_id not in self.allocations
        assert self.required_capacity(num_tokens) <= self.num_free
        self.allocations[request_id] = num_tokens

    def append(self, request_id):
        self.allocations[request_id] += 1
        assert self.num_free >= 0

    def release(self, request_id):
        self.allocations.pop(request_id, None)


class FakeRunner:
    def __init__(self, capacity=50):
        self.spec = RunnerSpec(8, torch.device("cpu"), torch.float32)
        self.cache = FakeResources(capacity)
        self.calls = []

    def prefill(self, input_ids, request_id):
        self.calls.append(("prefill", request_id))
        assert self.cache.allocations[request_id] == input_ids.shape[1]
        return RunnerOutput(torch.arange(16, dtype=torch.float32).view(1, 1, 16))

    def decode(self, input_ids, request_id):
        self.calls.append(("decode", request_id))
        assert input_ids.shape == (1, 1)
        assert request_id in self.cache.allocations
        return RunnerOutput(torch.arange(16, dtype=torch.float32).view(1, 1, 16))


def _make_scheduler(sampler=None):
    return Scheduler(FakeRunner(), sampler=sampler)


# ── is_done ────────────────────────────────────────────────────────

def test_is_done_empty_scheduler():
    assert _make_scheduler().is_done is True


def test_is_done_false_when_waiting():
    s = _make_scheduler()
    s.add_request(Request("r1", [1, 2, 3], max_new_tokens=2))
    assert s.is_done is False


def test_is_done_false_when_running():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("r1", [1, 2, 3], max_new_tokens=3))
    s.step()
    assert s.is_done is False


# ── single-request ─────────────────────────────────────────────────

def test_single_request_completes():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("r1", [1, 2, 3], max_new_tokens=3))
    while not s.is_done:
        s.step()
    req = s.finished[0]
    assert req.status is RequestStatus.FINISHED
    assert req.num_generated_tokens == 3


def test_step_returns_empty_list_when_prefill_only():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("r1", [1, 2, 3], max_new_tokens=3))
    assert s.step() == []


def test_step_returns_request_when_decode_finishes_it():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("r1", [1, 2, 3], max_new_tokens=2))
    s.step()
    finished = s.step()
    assert len(finished) == 1
    assert finished[0].request_id == "r1"
    assert finished[0].is_finished


# ── multiple requests ──────────────────────────────────────────────

def test_multiple_requests_all_finish():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("r1", [1, 2, 3], max_new_tokens=2))
    s.add_request(Request("r2", [4, 5], max_new_tokens=2))
    s.add_request(Request("r3", [6], max_new_tokens=3))
    while not s.is_done:
        s.step()
    assert {r.request_id for r in s.finished} == {"r1", "r2", "r3"}


def test_short_request_finishes_before_long_request():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("short", [1, 2, 3], max_new_tokens=1,
                   sampling_params=SamplingParams(temperature=1.0)))
    s.add_request(Request("long", [4, 5, 6], max_new_tokens=5,
                   sampling_params=SamplingParams(temperature=1.0)))
    order = []
    while not s.is_done:
        for r in s.step():
            order.append(r.request_id)
    assert order[0] == "short"
    assert order[1] == "long"


# ── determinism ────────────────────────────────────────────────────

def test_scheduler_is_deterministic():
    def run():
        torch.manual_seed(99)
        s = _make_scheduler()
        s.add_request(Request("a", [1, 2], max_new_tokens=3))
        s.add_request(Request("b", [3, 4], max_new_tokens=2))
        while not s.is_done:
            s.step()
        return [r.output_ids for r in s.finished]
    assert run() == run()


# ── step semantics ─────────────────────────────────────────────────

def test_step_does_exactly_one_operation():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("r1", [1, 2, 3], max_new_tokens=5))
    s.step()
    assert s.running[0].num_generated_tokens == 1
    s.step()
    assert s.running[0].num_generated_tokens == 2
    s.step()
    assert s.running[0].num_generated_tokens == 3


def test_finished_request_not_processed_again():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("short", [1, 2, 3], max_new_tokens=1))
    s.add_request(Request("long", [4, 5, 6], max_new_tokens=5))
    while not s.is_done:
        s.step()
    short = [r for r in s.finished if r.request_id == "short"][0]
    assert short.num_generated_tokens == 1
    assert len(short.generated_ids) == 1


# ── state transitions ──────────────────────────────────────────────

def test_request_transitions_waiting_to_prefilling_to_decoding_to_finished():
    torch.manual_seed(0)
    s = _make_scheduler()
    req = Request("r1", [1, 2, 3], max_new_tokens=2)
    assert req.status is RequestStatus.WAITING
    s.add_request(req)
    s.step()
    assert req.status is RequestStatus.DECODING
    assert req.num_generated_tokens == 1
    s.step()
    assert req.status is RequestStatus.FINISHED


# ── cleanup ────────────────────────────────────────────────────────

def test_block_manager_cleaned_after_finish():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("r1", [1, 2, 3], max_new_tokens=1))
    while not s.is_done:
        s.step()
    assert "r1" not in s.cache.allocations


# ── FIFO / round-robin ─────────────────────────────────────────────

def test_waiting_is_fifo():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("first", [1, 2], max_new_tokens=1))
    s.add_request(Request("second", [3, 4], max_new_tokens=1))
    order = []
    while not s.is_done:
        for r in s.step():
            order.append(r.request_id)
    assert order == ["first", "second"]


def test_running_decode_interleaves_round_robin():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("a", [1, 2], max_new_tokens=3))
    s.add_request(Request("b", [3, 4], max_new_tokens=3))
    s.step(); s.step()
    assert {r.request_id for r in s.running} == {"a", "b"}
    s.step()
    by_id = {r.request_id: r for r in s.running}
    assert by_id["a"].num_generated_tokens == 2
    assert by_id["b"].num_generated_tokens == 1
    s.step()
    by_id = {r.request_id: r for r in s.running}
    assert by_id["a"].num_generated_tokens == 2
    assert by_id["b"].num_generated_tokens == 2


# ── edge cases ─────────────────────────────────────────────────────

def test_max_new_tokens_zero_generates_nothing():
    torch.manual_seed(0)
    s = _make_scheduler()
    s.add_request(Request("r", [1, 2, 3], max_new_tokens=0))
    finished = s.step()
    assert len(finished) == 1
    assert finished[0].num_generated_tokens == 0
    assert finished[0].output_ids == [1, 2, 3]


def test_duplicate_request_id_rejected():
    import pytest
    s = _make_scheduler()
    s.add_request(Request("dup", [1, 2], max_new_tokens=1))
    with pytest.raises(ValueError, match="Duplicate request_id"):
        s.add_request(Request("dup", [3, 4], max_new_tokens=1))


# ── per-request sampling params ────────────────────────────────────

def test_per_request_sampling_params_used():
    params = SamplingParams(temperature=1.0, top_k=1)
    torch.manual_seed(42)
    s = _make_scheduler(sampler=Sampler(SamplingParams(temperature=2.0)))
    s.add_request(Request("r", [1, 2, 3], max_new_tokens=3,
                   sampling_params=params))
    while not s.is_done:
        s.step()
    assert len(s.finished[0].output_ids) == 6


def test_final_sample_at_block_boundary_does_not_allocate_unused_block():
    runner = FakeRunner(capacity=1)
    manager = runner.cache
    scheduler = Scheduler(runner)
    scheduler.add_request(Request("boundary", [1, 2, 3, 4], 1))
    assert len(scheduler.step()) == 1
    assert manager.num_free == 1
    assert not manager.allocations


def test_waiting_request_is_deferred_until_decode_releases_capacity():
    runner = FakeRunner(capacity=2)
    manager = runner.cache
    scheduler = Scheduler(runner)
    for request_id in ("first", "second"):
        scheduler.add_request(Request(request_id, [1, 2, 3], 4))
    scheduler.step()
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1
    scheduler.step()
    assert scheduler.running[0].num_generated_tokens == 2
    assert len(scheduler.waiting) == 1
    while not scheduler.is_done:
        scheduler.step()
    assert [r.request_id for r in scheduler.finished] == ["first", "second"]
    assert manager.num_free == 2


@pytest.mark.parametrize("input_ids,max_new_tokens,match", [
    ([1], -1, "nonnegative"),
    ([], 1, "prompt token"),
    ([1] * 8, 2, "exceeding"),
])
def test_invalid_request_is_rejected_before_allocation(input_ids, max_new_tokens, match):
    scheduler = _make_scheduler()
    with pytest.raises(ValueError, match=match):
        scheduler.add_request(Request("invalid", input_ids, max_new_tokens))
    assert scheduler.is_done
    assert not scheduler.cache.allocations


@pytest.mark.parametrize("stage", ["allocate", "prefill", "decode", "append", "sample"])
def test_failure_releases_only_affected_request(monkeypatch, stage):
    scheduler = _make_scheduler()
    first = Request("first", [1, 2], 3)
    failing = Request("fail", [3], 3, sampling_params=None)
    scheduler.add_request(first)
    scheduler.add_request(failing)
    scheduler.step()
    if stage == "decode":
        scheduler.step()
        scheduler.step()
    target = scheduler.runner if stage in ("prefill", "decode") else scheduler.cache
    if stage == "sample":
        target = scheduler.default_sampler
    original = getattr(target, stage)

    def fail(*args, **kwargs):
        if stage == "allocate":
            original(*args, **kwargs)  # Even a partial allocation is released.
        raise RuntimeError("injected failure")

    monkeypatch.setattr(target, stage, fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        scheduler.step()
    assert failing.status is RequestStatus.FAILED
    assert set(scheduler.cache.allocations) == {"first"}
    assert "fail" not in scheduler._samplers
    assert "fail" not in scheduler._next_inputs
    monkeypatch.setattr(target, stage, original)
    scheduler.run()
    assert first.status is RequestStatus.FINISHED
    assert not scheduler.cache.allocations


def test_eos_finishes_early_and_releases_resources():
    scheduler = _make_scheduler()
    request = Request("eos", [1], 7, SamplingParams(do_sample=False), stop_token_ids=(15,))
    scheduler.add_request(request)
    assert scheduler.step() == [request]
    assert request.generated_ids == [15]
    assert request.status is RequestStatus.FINISHED
    assert not scheduler.cache.allocations


def test_close_cancels_waiting_and_running_and_is_idempotent():
    scheduler = _make_scheduler()
    requests = [Request(str(i), [1], 3) for i in range(2)]
    for request in requests:
        scheduler.add_request(request)
    scheduler.step()
    scheduler.close()
    scheduler.close()
    assert scheduler.is_done
    assert all(r.status is RequestStatus.CANCELLED for r in requests)
    assert not scheduler.cache.allocations
    assert not scheduler._samplers
    assert not scheduler._next_inputs
    assert len(scheduler.finished) == 2


def test_run_failure_cleans_other_pending_requests(monkeypatch):
    scheduler = _make_scheduler()
    requests = [Request(str(i), [1], 3) for i in range(3)]
    for request in requests:
        scheduler.add_request(request)
    scheduler.step()
    monkeypatch.setattr(scheduler.runner, "decode", lambda *args: (_ for _ in ()).throw(RuntimeError("decode")))
    with pytest.raises(RuntimeError, match="decode"):
        scheduler.run()
    assert scheduler.is_done
    assert not scheduler.cache.allocations
    assert not scheduler._samplers
    assert not scheduler._next_inputs
    assert requests[0].status is RequestStatus.FAILED
    assert all(r.status is RequestStatus.CANCELLED for r in requests[1:])


def test_oversized_capacity_request_is_rejected():
    scheduler = Scheduler(FakeRunner(capacity=1))
    with pytest.raises(ValueError, match="capacity"):
        scheduler.add_request(Request("too-large", [1, 2, 3], 4))
    assert scheduler.is_done
    assert not scheduler.cache.allocations


@pytest.mark.parametrize("max_new_tokens", [0, 3])
def test_cancel_waiting_request_preserves_other_requests(max_new_tokens):
    scheduler = _make_scheduler()
    requests = [
        Request("left", [1], 1),
        Request("target", [2], max_new_tokens),
        Request("right", [3], 1),
    ]
    for request in requests:
        scheduler.add_request(request)
    free_before = scheduler.cache.num_free

    assert scheduler.cancel_request("target") is True

    assert [r.request_id for r in scheduler.waiting] == ["left", "right"]
    assert requests[1].status is RequestStatus.CANCELLED
    assert requests[1].generated_ids == []
    assert scheduler.cache.num_free == free_before
    assert not scheduler.cache.allocations
    assert "target" not in scheduler._samplers
    assert "target" not in scheduler._next_inputs
    assert not scheduler.runner.calls

    scheduler.run()
    assert requests[0].status is RequestStatus.FINISHED
    assert requests[2].status is RequestStatus.FINISHED
    assert all(request_id != "target" for _, request_id in scheduler.runner.calls)
    assert scheduler.cache.num_free == free_before


def test_cancel_running_request_releases_only_its_resources():
    scheduler = _make_scheduler()
    requests = [
        Request(name, [1, 2], 3, SamplingParams(do_sample=False))
        for name in ("left", "target", "right")
    ]
    for request in requests:
        scheduler.add_request(request)
    for _ in requests:
        scheduler.step()
    assert [r.request_id for r in scheduler.running] == ["left", "target", "right"]
    other_allocations = {
        name: length for name, length in scheduler.cache.allocations.items()
        if name != "target"
    }
    generated_before = list(requests[1].generated_ids)
    calls_before = len(scheduler.runner.calls)

    assert scheduler.cancel_request("target") is True

    assert [r.request_id for r in scheduler.running] == ["left", "right"]
    assert scheduler.cache.allocations == other_allocations
    assert requests[1].status is RequestStatus.CANCELLED
    assert requests[1].generated_ids == generated_before
    assert "target" not in scheduler._next_inputs
    assert "target" not in scheduler._samplers
    scheduler.run()
    assert requests[0].status is RequestStatus.FINISHED
    assert requests[2].status is RequestStatus.FINISHED
    assert all(
        request_id != "target"
        for _, request_id in scheduler.runner.calls[calls_before:]
    )
    assert not scheduler.cache.allocations
    assert not scheduler._next_inputs
    assert not scheduler._samplers
    assert scheduler.cache.num_free == scheduler.cache.capacity


def test_cancel_twice_releases_and_records_request_only_once(monkeypatch):
    scheduler = _make_scheduler()
    request = Request("target", [1], 3)
    scheduler.add_request(request)
    scheduler.step()
    releases = []
    original_release = scheduler.cache.release

    def record_release(request_id):
        releases.append(request_id)
        original_release(request_id)

    monkeypatch.setattr(scheduler.cache, "release", record_release)
    assert scheduler.cancel_request("target") is True
    assert scheduler.cancel_request("target") is False
    assert releases == ["target"]
    assert scheduler.finished == [request]
    assert request.status is RequestStatus.CANCELLED
    assert scheduler.is_done
    assert not scheduler.cache.allocations
    scheduler.close()
    assert releases == ["target"]
    assert scheduler.finished == [request]


def test_cancel_unknown_or_completed_request_changes_nothing(monkeypatch):
    scheduler = _make_scheduler()
    request = Request("live", [1], 1)
    scheduler.add_request(request)
    releases = []
    original_release = scheduler.cache.release

    def record_release(request_id):
        releases.append(request_id)
        original_release(request_id)

    monkeypatch.setattr(scheduler.cache, "release", record_release)
    assert scheduler.cancel_request("missing") is False
    assert list(scheduler.waiting) == [request]
    assert request.status is RequestStatus.WAITING
    assert set(scheduler._samplers) == {"live"}
    assert not scheduler.finished
    assert not releases

    scheduler.run()
    assert scheduler.cancel_request("live") is False
    assert releases == ["live"]
    assert scheduler.finished == [request]
    assert request.status is RequestStatus.FINISHED
    assert scheduler.cache.num_free == scheduler.cache.capacity


def test_cancel_uses_active_request_when_id_was_previously_completed():
    scheduler = _make_scheduler()
    previous = Request("reused", [1], 1)
    scheduler.add_request(previous)
    scheduler.run()

    current = Request("reused", [2], 3)
    scheduler.add_request(current)
    assert scheduler.cancel_request("reused") is True
    assert previous.status is RequestStatus.FINISHED
    assert current.status is RequestStatus.CANCELLED
    assert scheduler.finished == [previous, current]
    assert scheduler.is_done
    assert scheduler.cache.num_free == scheduler.cache.capacity
