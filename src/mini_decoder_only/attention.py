"""Attention module scaffolds for the decoder-only Transformer."""

from __future__ import annotations

import torch
from torch import nn


class Head(nn.Module):
    """Single causal self-attention head scaffold.

    Expected input shape:
        [B, S, H]

    Expected output shape:
        [B, S, Dh]
    """

    def __init__(
        self,
        n_embd: int,
        head_size: int,
        block_size: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.head_size = head_size
        self.block_size = block_size
        self.dropout = dropout
        self.bias = bias

        self.key = nn.Linear(n_embd, head_size, bias=bias)
        self.query = nn.Linear(n_embd, head_size, bias=bias)
        self.value = nn.Linear(n_embd, head_size, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size)),
        ) 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, S, _ = x.shape 
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        att = q @ k.transpose(-2, -1)
        att = att * (self.head_size ** -0.5)
        att = att.masked_fill(self.tril[:S, :S] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
           
        out = att @ v
        return out
 
class MultiHeadAttention(nn.Module):
    """Multi-head attention scaffold.

    Expected input shape:
        [B, S, H]

    Expected output shape:
        [B, S, H]
    """

    def __init__(
        self,
        n_head: int,
        n_embd: int,
        block_size: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("MultiHead Error")
        # Cal head size by n_embd // n_head
        head_size = n_embd // n_head
        
        self.head_size = head_size
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout
        self.bias = bias

        self.heads = nn.ModuleList(
            [
                Head(
                    n_embd=n_embd,
                    head_size=head_size,
                    block_size=block_size,
                    dropout=dropout,
                    bias=bias 
                )
                for _ in range(n_head)
            ]
        )
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.proj_dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_outputs = [head(x) for head in self.heads]
        out = torch.cat(head_outputs, dim=-1)
        out = self.proj(out)
        out = self.proj_dropout(out)
        
        return out