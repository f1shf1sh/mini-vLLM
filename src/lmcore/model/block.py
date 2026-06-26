"""Transformer block scaffolds."""

from __future__ import annotations

import torch
from torch import nn

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network scaffold."""

    def __init__(self, n_embd: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=bias),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-LN Transformer block.

    ``kv_cache`` is an optional ``PagedKVCache`` instance.  When
    provided together with ``use_cache=True``, the attention reads
    historical K/V from the pool and writes new K/V back; no
    intermediate tensor needs to be returned.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            dropout=dropout,
            bias=bias,
            layer_idx=layer_idx,
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd=n_embd, dropout=dropout, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        kv_cache: object | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), use_cache=use_cache, kv_cache=kv_cache)
        x = x + self.ffwd(self.ln2(x))
        return x
