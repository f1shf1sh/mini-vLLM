"""Model-independent execution and request-resource boundaries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .types import RunnerOutput


@dataclass(frozen=True, slots=True)
class RunnerSpec:
    context_length: int
    device: torch.device
    dtype: torch.dtype


class CacheResources(Protocol):
    """Capacity units and request lifecycle, without a tensor-layout contract.

    Adapters construct the implementation. Scheduler owns lifecycle decisions.
    ``release`` must be safe for a request with no allocation, including after
    an unsuccessful allocation. Capacity is dedicated to one scheduler.
    """

    @property
    def capacity(self) -> int: ...

    def required_capacity(self, num_tokens: int) -> int: ...
    def allocate(self, request_id: str, num_tokens: int) -> None: ...
    def append(self, request_id: str) -> None: ...
    def release(self, request_id: str) -> None: ...


class Runner(Protocol):
    """Adapters own model forward, cache construction, and cache format.

    Logits have shape [1, sequence_length, vocabulary_size]. No model topology,
    attention layout, expert routing, or expert execution is exposed here.
    """

    spec: RunnerSpec
    cache: CacheResources

    def prefill(self, input_ids: torch.Tensor, request_id: str) -> RunnerOutput: ...
    def decode(self, input_ids: torch.Tensor, request_id: str) -> RunnerOutput: ...


class BatchDecodeRunner(Runner, Protocol):
    """Optional execution capability for one decode forward over several requests.

    Input IDs are [B, 1] and output logits are [B, 1, vocabulary_size]. Row i
    belongs to request_ids[i]. Each request has one reserved, unwritten KV slot;
    adapters handle independent context lengths, masks, positions and KV layout.
    """

    def decode_batch(
        self,
        input_ids: torch.Tensor,
        request_ids: list[str],
    ) -> RunnerOutput: ...
