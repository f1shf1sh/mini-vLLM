"""Message data; receiving boundaries check versions, epochs and rank identity."""
from __future__ import annotations

from dataclasses import dataclass

from llm.cache.pool import KVSpec
from llm.inference.types import ExecutionResult, SchedulePlan
from llm.model.interfaces import ModelSpec


EXECUTION_PROTOCOL_VERSION = 1


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    protocol_version: int
    executor_epoch: str
    plan: SchedulePlan


@dataclass(frozen=True, slots=True)
class WorkerInfo:
    """Model-ready report; cache allocation has not been confirmed yet."""

    rank: int
    model_spec: ModelSpec
    kv_spec: KVSpec
    block_size: int
    max_num_blocks: int


@dataclass(frozen=True, slots=True)
class WorkerExecutionResult:
    """One rank's receipt; only the driver supplies output, including S=0.

    Executor checks plan identity, the current epoch, expected ranks and driver.
    Constructing this message does not prove device work has completed.
    """

    executor_epoch: str
    plan_id: int
    rank: int
    output: ExecutionResult | None
