"""Seeded operation traces; replay_operations accepts the JSON trace on failure."""
import json
import random
from unittest.mock import patch

import pytest

from llm.inference.request import Request, RequestStatus

from _scheduler_support import assert_invariants, make_scheduler, submit


class InjectedFailure(RuntimeError):
    pass


class Scenario:
    def __init__(self, num_blocks, block_size):
        self.scheduler = make_scheduler(num_blocks, block_size)
        self.accepted = []
        self.terminal_snapshots = {}
        self.outcomes = []

    def apply(self, operation):
        scheduler = self.scheduler
        name = operation["op"]
        before = {id(r): tuple(r.generated_ids) for r in self.accepted}
        if name == "add":
            if operation.get("reject", False):
                request = Request(
                    operation["id"], operation["prompt"], operation["limit"],
                    sampling_params=None,
                )
                with pytest.raises(ValueError):
                    scheduler.add_request(request)
                assert request.status is RequestStatus.WAITING
                assert not request.generated_ids
                result = "rejected"
            else:
                request = submit(
                    scheduler, self.accepted, operation["id"], operation["prompt"],
                    operation["limit"], operation.get("eos_after"),
                )
                result = {"generation": len(self.accepted) - 1}
        elif name == "step":
            calls_before = len(scheduler.runner.calls)
            result = [scheduler.runner.generations[id(r)] for r in scheduler.step()]
            assert len(scheduler.runner.calls) - calls_before <= 1
            assert sum(len(r.generated_ids) - len(before[id(r)]) for r in self.accepted) <= 1
        elif name == "cancel":
            active = list(scheduler.waiting) + list(scheduler.running)
            target = next((r for r in active if r.request_id == operation["id"]), None)
            waiting = [r for r in scheduler.waiting if r is not target]
            running = [r for r in scheduler.running if r is not target]
            calls_before = list(scheduler.runner.calls)
            result = scheduler.cancel_request(operation["id"])
            assert result is (target is not None)
            assert list(scheduler.waiting) == waiting
            assert list(scheduler.running) == running
            assert scheduler.runner.calls == calls_before
            assert all(tuple(r.generated_ids) == before[id(r)] for r in self.accepted)
            if target is not None:
                assert target.status is RequestStatus.CANCELLED
        elif name == "close":
            pending = list(scheduler.waiting) + list(scheduler.running)
            calls_before = list(scheduler.runner.calls)
            scheduler.close()
            assert scheduler.runner.calls == calls_before
            assert all(r.status is RequestStatus.CANCELLED for r in pending)
            assert all(tuple(r.generated_ids) == before[id(r)] for r in self.accepted)
            result = None
        elif name == "run":
            pending = list(scheduler.waiting) + list(scheduler.running)
            result = [scheduler.runner.generations[id(r)] for r in scheduler.run()]
            assert all(r.status is RequestStatus.FINISHED for r in pending)
            assert scheduler.is_done
        elif name == "fault":
            result = self._fault(operation)
        else:
            raise AssertionError(f"Unknown operation: {name}")
        for request in self.accepted:
            previous_tokens = before.get(id(request), ())
            assert tuple(request.generated_ids[:len(previous_tokens)]) == previous_tokens
        self.outcomes.append(result)
        assert_invariants(scheduler, self.accepted, self.terminal_snapshots)

    def _fault(self, operation):
        scheduler = self.scheduler
        stage = operation["stage"]
        target = scheduler.runner if stage in ("prefill", "decode") else scheduler.cache
        if stage == "sample":
            target = scheduler.default_sampler
        original = getattr(target, stage)
        triggered = []
        pending = list(scheduler.waiting) + list(scheduler.running)

        def fail_once(*args, **kwargs):
            if not triggered:
                triggered.append(True)
                if stage in ("allocate", "append"):
                    original(*args, **kwargs)
                raise InjectedFailure(stage)
            return original(*args, **kwargs)

        with patch.object(target, stage, side_effect=fail_once):
            try:
                getattr(scheduler, operation["mode"])()
            except InjectedFailure as error:
                assert str(error) == stage
                assert triggered == [True]
            else:
                assert not triggered
        if triggered:
            failed = [r for r in pending if r.status is RequestStatus.FAILED]
            assert len(failed) == 1
            if operation["mode"] == "step":
                assert {id(r) for r in scheduler.waiting} | {id(r) for r in scheduler.running} == {
                    id(r) for r in pending if r is not failed[0]
                }
            else:
                assert scheduler.is_done
                assert all(r.is_finished for r in pending)
        return {"fault_triggered": bool(triggered)}

    def snapshot(self):
        scheduler = self.scheduler
        return {
            "waiting": [scheduler.runner.generations[id(r)] for r in scheduler.waiting],
            "running": [scheduler.runner.generations[id(r)] for r in scheduler.running],
            "requests": [
                {"generation": i, "id": r.request_id, "status": r.status.name, "tokens": r.generated_ids}
                for i, r in enumerate(self.accepted)
            ],
            "blocks": {
                request_id: {"slots": table.num_tokens, "blocks": table.physical_blocks}
                for request_id, table in scheduler.cache.block_manager.block_tables.items()
            },
            "free": list(scheduler.cache.block_manager.allocator.free_blocks),
        }


def replay_operations(operations, *, num_blocks, block_size):
    """Replay the printed JSON operations without RNG or model downloads."""
    scenario = Scenario(num_blocks, block_size)
    for operation in operations:
        scenario.apply(operation)
    return scenario


def next_operation(rng, scenario):
    scheduler = scenario.scheduler
    active_ids = {r.request_id for r in list(scheduler.waiting) + list(scheduler.running)}
    roll = rng.randrange(100)
    if roll < 35:
        request_id = f"r{rng.randrange(5)}"
        kind = rng.choice(["valid", "valid", "zero", "duplicate", "reuse", "too_large", "negative", "empty"])
        if kind == "duplicate" and active_ids:
            request_id = rng.choice(sorted(active_ids))
        if kind == "reuse":
            reusable = sorted({r.request_id for r in scheduler.finished} - active_ids)
            if reusable:
                request_id = rng.choice(reusable)
        slots = scheduler.cache.capacity * scheduler.cache.pool.block_size
        prompt = [rng.randrange(2, 7) for _ in range(rng.randint(1, min(3, slots)))]
        limit = rng.randint(1, min(6, slots - len(prompt) + 1))
        invalid = False
        if kind == "zero":
            prompt, limit = [], 0
        elif kind == "too_large":
            prompt, limit, invalid = [2] * (slots + 1), 1, True
        elif kind == "negative":
            limit, invalid = -1, True
        elif kind == "empty":
            prompt, invalid = [], True
        return {
            "op": "add", "id": request_id, "prompt": prompt, "limit": limit,
            "eos_after": rng.choice([None, None, 1, 2, 3]),
            "reject": invalid or request_id in active_ids,
        }
    if roll < 60:
        return {"op": "step"}
    if roll < 75:
        return {"op": "cancel", "id": f"r{rng.randrange(7)}"}
    if roll < 80:
        return {"op": "close"}
    if roll < 90:
        return {"op": "run"}
    return {
        "op": "fault", "stage": rng.choice(["allocate", "prefill", "decode", "append", "sample"]),
        "mode": rng.choice(["step", "run"]),
    }


@pytest.mark.parametrize("seed", range(32))
def test_randomized_request_lifecycles_preserve_resources(seed):
    rng = random.Random(seed)
    block_size = (1, 2, 4)[seed % 3]
    num_blocks = seed % 4 + 1
    scenario = Scenario(num_blocks, block_size)
    trace = []
    try:
        for _ in range(200):
            operation = next_operation(rng, scenario)
            trace.append(operation)
            scenario.apply(operation)
        # Always finish with an explicit cleanup and an idempotency check.
        for operation in ({"op": "close"}, {"op": "close"}):
            trace.append(operation)
            scenario.apply(operation)
        assert scenario.scheduler.cache.block_manager.allocator.num_free == num_blocks
    except Exception as error:
        pytest.fail(
            f"seed={seed}, num_blocks={num_blocks}, block_size={block_size}\n"
            f"Replay: replay_operations(trace, num_blocks={num_blocks}, block_size={block_size})\n"
            f"trace={json.dumps(trace)}\n"
            f"outcomes={json.dumps(scenario.outcomes)}\n"
            f"state={json.dumps(scenario.snapshot())}\n"
            f"{type(error).__name__}: {error}",
            pytrace=True,
        )
