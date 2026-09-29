from __future__ import annotations

import torch

from .blocks import BlockManager, BlockTable
from .pool import KVPool


class PagedKVCache:
    """Per-request handle for reading/writing paged KV cache.

    Bundles a KVPool reference with a BlockTable so that model layers
    can read historical K/V and write new K/V without knowing about
    block internals.

    ``num_tokens`` is the total token count tracked by the BlockTable
    — this number may include one unwritten slot during decode (after
    ``BlockManager.append_token`` has incremented it).
    """

    def __init__(
        self,
        pool: KVPool,
        block_table: BlockTable,
        n_head: int,
        head_dim: int,
    ) -> None:
        self._pool = pool
        self._table = block_table
        self._n_head = n_head
        self._head_dim = head_dim

    @property
    def num_tokens(self) -> int:
        return self._table.num_tokens

    def read(
        self,
        layer: int,
        token_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one token's K/V for a layer. Each [num_heads, head_dim]."""
        block_id, offset = self._table.locate(token_pos)
        return self._pool.read(layer, block_id, offset)

    def write(
        self,
        layer: int,
        token_pos: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write one token's K/V for a layer at a specific position."""
        block_id, offset = self._table.locate(token_pos)
        self._pool.write(layer, block_id, offset, key, value)

    def gather(
        self,
        layer: int,
        num_past: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather ``num_past`` historical tokens into contiguous tensors.

        Returns (k, v), each [1, n_head, num_past, head_dim].
        """
        if num_past == 0:
            return (
                self._pool.pool.new_empty(1, self._n_head, 0, self._head_dim),
                self._pool.pool.new_empty(1, self._n_head, 0, self._head_dim),
            )
        k = self._pool.pool.new_empty(1, self._n_head, num_past, self._head_dim)
        v = torch.empty_like(k)
        for pos in range(num_past):
            kp, vp = self.read(layer, pos)
            k[0, :, pos, :] = kp
            v[0, :, pos, :] = vp
        return k, v

    def write_range(
        self,
        layer: int,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int = 0,
    ) -> None:
        """Write a contiguous range of new K/V starting at ``start_pos``.

        k / v shape: [1, n_head, S, head_dim]
        """
        S = k.size(2)
        for i in range(S):
            self.write(layer, start_pos + i,
                       key=k[0, :, i, :], value=v[0, :, i, :])


class PagedCacheResources:
    """Request resources backed by standard KV pages.

    Model adapters build the pool and block manager. This object exposes their
    capacity and lifecycle without making the scheduler inspect KV dimensions.
    """

    def __init__(self, pool: KVPool, block_manager: BlockManager) -> None:
        if pool.block_size <= 0 or pool.num_blocks <= 0:
            raise ValueError("Cache block_size and num_blocks must be positive.")
        allocator = block_manager.allocator
        if (pool.block_size != block_manager.block_size
                or pool.num_blocks != allocator.num_free + len(allocator.allocated)):
            raise ValueError("KV pool and block manager capacities must match.")
        self.pool = pool
        self.block_manager = block_manager

    @property
    def capacity(self) -> int:
        return self.pool.num_blocks

    def required_capacity(self, num_tokens: int) -> int:
        return (num_tokens + self.pool.block_size - 1) // self.pool.block_size

    def allocate(self, request_id: str, num_tokens: int) -> None:
        self.block_manager.allocate_for_request(request_id, num_tokens)

    def append(self, request_id: str) -> None:
        self.block_manager.append_token(request_id)

    def release(self, request_id: str) -> None:
        if request_id in self.block_manager.block_tables:
            self.block_manager.free_request(request_id)

    def view(self, request_id: str) -> PagedKVCache:
        return PagedKVCache(
            self.pool, self.block_manager.block_tables[request_id],
            self.pool.num_heads, self.pool.head_dim,
        )
