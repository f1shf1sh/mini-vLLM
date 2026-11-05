"""Module 1 data contracts; no scheduler, KV or transport implementation."""
from dataclasses import FrozenInstanceError, asdict, replace
import json
import pickle

import pytest
import torch

from llm.cache.blocks import BlockTableSnapshot
from llm.cache.pool import KVSpec
from llm.config import CacheConfig, ExecutorConfig, ModelConfig
from llm.executor.types import (
    EXECUTION_PROTOCOL_VERSION, ExecutionRequest, WorkerExecutionResult, WorkerInfo,
)
from llm.inference.types import ExecutionResult, ExecutionSpec, ScheduledItem, SchedulePlan
from llm.model.interfaces import ModelSpec
from llm.parallel.config import ParallelConfig, WorkerConfig


@pytest.fixture
def prefill_plan():
    return SchedulePlan(1, (
        ScheduledItem("A", (1, 2, 3), 0, BlockTableSnapshot((0,), 3), True),
        ScheduledItem("B", (4, 5, 6, 7, 8), 0, BlockTableSnapshot((1, 2), 5), True),
    ))


def test_prefill_decode_and_middle_chunk_are_plain_data(prefill_plan):
    decode = SchedulePlan(2, (
        ScheduledItem("A", (9,), 3, BlockTableSnapshot((0,), 4), True),
        ScheduledItem("B", (10,), 5, BlockTableSnapshot((1, 2), 6), True),
    ))
    chunk = SchedulePlan(3, (
        ScheduledItem("C", (5, 6, 7, 8), 4, BlockTableSnapshot((3, 4), 8), False),
    ))
    assert [len(item.input_ids) for item in prefill_plan.items] == [3, 5]
    assert [item.past_length for item in decode.items] == [3, 5]
    assert [item.input_ids for item in decode.items] == [(9,), (10,)]
    for plan in (prefill_plan, decode, chunk):
        assert pickle.loads(pickle.dumps(plan)) == plan
        json.dumps(asdict(plan))
    result = ExecutionResult(3, (), torch.empty(0, 16))
    assert result.logits.shape == (0, 16)
    assert chunk.items[0].needs_logits is False


def test_plan_snapshot_is_immutable_and_detached_from_source(prefill_plan):
    pages = [7, 2]
    snapshot = BlockTableSnapshot(tuple(pages), 5)
    pages[0] = 9
    assert snapshot.block_ids == (7, 2)
    with pytest.raises(FrozenInstanceError):
        snapshot.reserved_tokens = 6
    with pytest.raises(FrozenInstanceError):
        prefill_plan.items[0].past_length = 1
    with pytest.raises(FrozenInstanceError):
        prefill_plan.items = ()


def test_mutable_plan_containers_are_rejected(prefill_plan):
    with pytest.raises(TypeError):
        BlockTableSnapshot([0], 3)
    with pytest.raises(TypeError):
        replace(prefill_plan.items[0], input_ids=[1, 2, 3])
    with pytest.raises(TypeError):
        SchedulePlan(1, list(prefill_plan.items))


@pytest.mark.parametrize("change", [
    {"input_ids": ()},
    {"past_length": -1},
    {"past_length": 1},
])
def test_item_rejects_invalid_input_or_insufficient_reservation(prefill_plan, change):
    with pytest.raises(ValueError):
        replace(prefill_plan.items[0], **change)


def test_plan_rejects_empty_or_duplicate_members(prefill_plan):
    with pytest.raises(ValueError):
        SchedulePlan(1, ())
    with pytest.raises(ValueError):
        SchedulePlan(1, (prefill_plan.items[0], prefill_plan.items[0]))


@pytest.mark.parametrize("ids,logits", [
    (("A",), torch.zeros(1, 1, 16)),
    (("A", "B"), torch.zeros(1, 16)),
    (("A", "A"), torch.zeros(2, 16)),
    (("A",), torch.zeros(1, 16, dtype=torch.long)),
])
def test_result_rejects_invalid_shape_or_identity(ids, logits):
    with pytest.raises(ValueError):
        ExecutionResult(1, ids, logits)


def test_result_preserves_logits_and_freezes_only_fields():
    logits = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    result = ExecutionResult(1, ("B", "A"), logits)
    restored = pickle.loads(pickle.dumps(result))
    assert restored.request_ids == ("B", "A")
    torch.testing.assert_close(restored.logits, logits, rtol=0, atol=0)
    assert result.logits is logits
    with pytest.raises(FrozenInstanceError):
        result.logits = torch.zeros(2, 16)
    logits[0, 0] = 42
    assert result.logits[0, 0].item() == 42


def test_configs_reports_and_messages_round_trip(prefill_plan):
    parallel = ParallelConfig(tensor_parallel_size=2, driver_rank=1)
    worker = WorkerConfig("epoch-a", 1, 0, "cuda:3", ModelConfig("/local/model"),
                          parallel, block_size=4, max_num_blocks=96)
    model = ModelSpec(64, 16, 0, (15,), "cuda:3", "float16", "float32")
    kv = KVSpec(2, 1, 8, "cuda:3", "float16")
    report = WorkerInfo(1, model, kv, 4, 96)
    spec = ExecutionSpec(64, 16, (15,), 96, 4, "cpu", "float32")
    request = ExecutionRequest(EXECUTION_PROTOCOL_VERSION, "epoch-a", prefill_plan)
    values = (worker, report, spec, request, CacheConfig(4, 96),
              ExecutorConfig("multiprocess", "cpu", "env://"))
    for value in values:
        assert pickle.loads(pickle.dumps(value)) == value
        json.dumps(asdict(value))
    assert parallel.world_size == 2
    assert "world_size" not in asdict(parallel)
    assert worker.rank != worker.local_rank
    assert spec.logits_dtype != kv.dtype


def test_empty_driver_output_differs_from_non_driver_receipt():
    result = ExecutionResult(3, (), torch.empty(0, 16))
    driver = WorkerExecutionResult("epoch-a", 3, 0, result)
    other = WorkerExecutionResult("epoch-a", 3, 1, None)
    assert driver.output.logits.shape == (0, 16)
    assert other.output is None
