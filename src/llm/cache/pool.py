from __future__ import annotations

from enum import IntEnum

import torch


class PoolDim(IntEnum):
    """Index into the pool's K/V dimension."""
    KEY = 0
    VALUE = 1


class KVPool:
    """Global physical KV cache pool.

    Shape: [num_layers, 2, num_blocks, num_heads, block_size, head_dim]
           PoolDim.KEY = 0, PoolDim.VALUE = 1
    """

    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        num_heads: int,
        block_size: int,
        head_dim: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.block_size = block_size
        self.head_dim = head_dim

        self.pool = torch.zeros(
            num_layers, 2, num_blocks, num_heads, block_size, head_dim,
            device=device, dtype=dtype,
        )

    def write(
        self,
        layer: int,
        block_id: int,
        offset: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write one token's K/V for a single layer at (block_id, offset).

        key / value shape: [num_heads, head_dim]
        """
        self.pool[layer, PoolDim.KEY, block_id, :, offset, :] = key
        self.pool[layer, PoolDim.VALUE, block_id, :, offset, :] = value

    def read(
        self,
        layer: int,
        block_id: int,
        offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one token's K/V for a single layer. Returns (key, value),
        each shaped [num_heads, head_dim]."""
        k = self.pool[layer, PoolDim.KEY, block_id, :, offset, :]
        v = self.pool[layer, PoolDim.VALUE, block_id, :, offset, :]
        return k, v
