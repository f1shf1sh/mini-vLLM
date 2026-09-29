from __future__ import annotations

import torch

from llm.cache.blocks import BlockAllocator, BlockManager
from llm.cache.paged import PagedCacheResources
from llm.cache.pool import KVPool
from llm.inference.runner import RunnerSpec
from llm.inference.types import RunnerOutput


class ModelRunner:
    """Adapt GPTMini forward and cache construction to the public engine."""

    def __init__(
        self,
        model: torch.nn.Module,
        kv_pool: KVPool,
        block_manager: BlockManager,
    ) -> None:
        self.model = model.eval()
        self.kv_pool = kv_pool
        self.block_manager = block_manager
        parameter = next(model.parameters())
        self.spec = RunnerSpec(model.config.block_size, parameter.device, parameter.dtype)
        self.cache = PagedCacheResources(kv_pool, block_manager)

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        *,
        num_blocks: int,
        block_size: int,
    ) -> ModelRunner:
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be positive.")
        config = model.config
        parameter = next(model.parameters())
        pool = KVPool(
            num_blocks, config.n_layer, config.n_head, block_size,
            config.n_embd // config.n_head,
            device=parameter.device, dtype=parameter.dtype,
        )
        manager = BlockManager(BlockAllocator(num_blocks), block_size)
        return cls(model, pool, manager)

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, request_id: str) -> RunnerOutput:
        if input_ids.ndim != 2 or input_ids.size(0) != 1:
            raise ValueError("prefill expects input_ids with shape [1, T].")
        if input_ids.size(1) < 1:
            raise ValueError("prefill expects at least one token.")
        logits, _ = self.model(input_ids, use_cache=True, kv_cache=self.cache.view(request_id))
        return RunnerOutput(logits=logits)

    @torch.no_grad()
    def decode(self, input_ids: torch.Tensor, request_id: str) -> RunnerOutput:
        if input_ids.ndim != 2 or input_ids.size(0) != 1:
            raise ValueError("decode expects input_ids with shape [1, 1].")
        if input_ids.size(1) != 1:
            raise ValueError("decode expects exactly one token per request.")
        logits, _ = self.model(input_ids, use_cache=True, kv_cache=self.cache.view(request_id))
        return RunnerOutput(logits=logits)
