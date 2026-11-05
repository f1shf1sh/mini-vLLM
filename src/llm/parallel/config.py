"""Serializable startup settings; topology and capacity checks belong to startup."""
from __future__ import annotations

from dataclasses import dataclass

from llm.config import ModelConfig


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    tensor_parallel_size: int = 1
    driver_rank: int = 0

    @property
    def world_size(self) -> int:
        return self.tensor_parallel_size


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """A local capacity budget; no model objects, callbacks or process groups."""

    executor_epoch: str
    rank: int
    local_rank: int
    device: str
    model: ModelConfig
    parallel: ParallelConfig
    block_size: int
    max_num_blocks: int
