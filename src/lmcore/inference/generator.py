from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .block_manager import BlockAllocator, BlockManager
from .kv_pool import KVPool
from .request import Request
from .sampler import Sampler, SamplingParams
from .scheduler import Scheduler


class BaseGenerator(ABC):
    """Common interface for single-request generation strategies."""

    @abstractmethod
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        raise NotImplementedError


class CachedGenerator(BaseGenerator):
    """Single-request adapter for the scheduler's paged KV inference path."""

    def __init__(
        self,
        model: torch.nn.Module,
        sampler: Sampler | None = None,
    ) -> None:
        self.model = model
        self.sampler = sampler or Sampler()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.size(0) != 1:
            raise ValueError("generate expects input_ids with shape [1, T].")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative.")
        if max_new_tokens == 0:
            return input_ids
        if input_ids.size(1) == 0:
            raise ValueError("generate expects at least one prompt token.")

        config = self.model.config
        cached_tokens = input_ids.size(1) + max_new_tokens - 1
        if cached_tokens > config.block_size:
            raise ValueError(
                f"Generation requires {cached_tokens} cached tokens, exceeding "
                f"block_size={config.block_size}."
            )
        block_size = min(16, config.block_size)
        num_blocks = (cached_tokens + block_size - 1) // block_size
        parameter = next(self.model.parameters())
        pool = KVPool(
            num_blocks=num_blocks,
            num_layers=config.n_layer,
            num_heads=config.n_head,
            block_size=block_size,
            head_dim=config.n_embd // config.n_head,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        block_manager = BlockManager(BlockAllocator(num_blocks), block_size)
        scheduler = Scheduler(self.model, pool, block_manager, self.sampler)
        request = Request(
            request_id="generate",
            input_ids=input_ids[0].tolist(),
            max_new_tokens=max_new_tokens,
            sampling_params=None,
        )
        scheduler.add_request(request)
        while not scheduler.is_done:
            scheduler.step()
        return input_ids.new_tensor([request.output_ids])


class NaiveGenerator(BaseGenerator):
    """Reserved baseline for comparing generation without KV cache."""

    def __init__(
        self,
        model: torch.nn.Module,
        sampler: Sampler | None = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.sampler = sampler or Sampler()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        raise NotImplementedError("NaiveGenerator is reserved for cache comparison.")


Generator = CachedGenerator


def generate_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    sampling_params: SamplingParams | None = None,
) -> torch.Tensor:
    sampler = Sampler(sampling_params)
    generator = CachedGenerator(model, sampler=sampler)
    return generator.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
    )
