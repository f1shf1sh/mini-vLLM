"""Transformer block scaffolds."""

from __future__ import annotations

import torch
from torch import nn
from .attention import MultiHeadAttention

class FeedForward(nn.Module):
    """Position-wise feed-forward network scaffold."""

    def __init__(self, n_embd: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        """
        While the linear transformations are the same across different positions, 
        they use different parameters from layer to layer. Another way of describing 
        this is as two convolutions with kernel size 1. 
        The dimensionality of input and output is dmodel = 512, 
        and the inner-layer has dimensionality dff = 2048.
        """
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=bias),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=bias),
            nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Block(nn.Module):
    """Pre-LN Transformer block scaffold."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.block_size = block_size
        self.dropout = dropout
        self.bias = bias

        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            dropout=dropout,
            bias=bias,
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(
            n_embd=n_embd,
            dropout=dropout,
            bias=bias,
        )
    def forward(self, 
                x: torch.Tensor,
                layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
                use_cache: bool = False,         
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, present = self.attn(
            self.ln1(x),
            layer_past,
            use_cache
        )
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x, present