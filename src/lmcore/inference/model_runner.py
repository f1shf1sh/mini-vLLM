from __future__ import annotations

import torch

from .types import RunnerOutput
from .kv_pool import KVPool, PagedKVCache
from .block_manager import BlockManager


class ModelRunner:
    """Thin inference wrapper around GPTMini.forward().

    Uses PagedKVCache so the model reads/writes KV directly from
    the global KVPool via block tables.  No contiguous KV cache is
    returned or managed by the caller.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        kv_pool: KVPool,
        block_manager: BlockManager,
    ) -> None:
        self.model = model
        self.kv_pool = kv_pool
        self.block_manager = block_manager
        self.model.eval()

    def _make_kv_cache(self, request_id: str) -> PagedKVCache:
        table = self.block_manager.block_tables[request_id]
        return PagedKVCache(
            pool=self.kv_pool,
            block_table=table,
            n_head=self.kv_pool.num_heads,
            head_dim=self.kv_pool.head_dim,
        )

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, request_id: str) -> RunnerOutput:
        if input_ids.ndim != 2:
            raise ValueError("prefill expects input_ids with shape [B, T].")
        if input_ids.size(1) < 1:
            raise ValueError("prefill expects at least one token.")

        self.block_manager.allocate_for_request(request_id, input_ids.size(1))
        kv_cache = self._make_kv_cache(request_id)

        logits, _ = self.model(
            input_ids,
            use_cache=True,
            kv_cache=kv_cache,
        )
        return RunnerOutput(logits=logits)

    @torch.no_grad()
    def decode(self, input_ids: torch.Tensor, request_id: str) -> RunnerOutput:
        if input_ids.ndim != 2:
            raise ValueError("decode expects input_ids with shape [B, 1].")
        if input_ids.size(1) != 1:
            raise ValueError("decode expects exactly one token per request.")

        self.block_manager.append_token(request_id)
        kv_cache = self._make_kv_cache(request_id)

        logits, _ = self.model(
            input_ids,
            use_cache=True,
            kv_cache=kv_cache,
        )
        return RunnerOutput(logits=logits)
