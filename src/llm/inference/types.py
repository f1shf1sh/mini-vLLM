"""Public execution data; the legacy RunnerOutput remains until caller migration."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from llm.cache.blocks import BlockTableSnapshot


@dataclass(slots=True)
class RunnerOutput:
    """Legacy runner output [B, T, V], not an ExecutionResult."""

    logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class ScheduledItem:
    """Real input tokens for one request; no padding or device resources."""

    request_id: str
    input_ids: tuple[int, ...]
    past_length: int
    block_table: BlockTableSnapshot
    needs_logits: bool

    def __post_init__(self) -> None:
        if not isinstance(self.input_ids, tuple):
            raise TypeError("input_ids must be a tuple")
        if not self.input_ids or self.past_length < 0:
            raise ValueError("input must be nonempty and past_length nonnegative")
        if self.block_table.reserved_tokens < self.past_length + len(self.input_ids):
            raise ValueError("reserved slots must cover past_length plus input length")


@dataclass(frozen=True, slots=True)
class SchedulePlan:
    """One ordered, nonempty execution plan; IDs are assigned by Scheduler."""

    plan_id: int
    items: tuple[ScheduledItem, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            raise TypeError("items must be a tuple")
        if not self.items:
            raise ValueError("a plan must contain at least one item")
        request_ids = [item.request_id for item in self.items]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request IDs must be unique within a plan")


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Full-vocabulary logits [S, V]; frozen fields do not freeze the tensor.

    Scheduler checks plan identity and row order. ExecutionSpec fixes V, device
    and dtype. S=0 is valid and does not imply an empty execution plan.
    """

    plan_id: int
    request_ids: tuple[str, ...]
    logits: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.request_ids, tuple):
            raise TypeError("request_ids must be a tuple")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("result request IDs must be unique")
        if self.logits.ndim != 2 or self.logits.shape[0] != len(self.request_ids):
            raise ValueError("logits must have shape [len(request_ids), V]")
        if self.logits.shape[1] < 1 or not self.logits.is_floating_point():
            raise ValueError("logits must have a positive vocabulary width and floating dtype")


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """Confirmed control-side limits; contains no model or KV layout."""

    context_length: int
    vocab_size: int
    default_stop_token_ids: tuple[int, ...]
    num_blocks: int
    block_size: int
    result_device: str
    logits_dtype: str
