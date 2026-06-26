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
    """Multi-head attention with optional paged KV-cache support.

    When ``kv_cache`` is provided (inference), historical K/V are
    read from the pool and new K/V are written back — no intermediate
    contiguous cache is returned or managed by the caller.

    When ``kv_cache`` is None (training), the forward pass is a
    standard causal attention with no KV storage.
    """

    def __init__(
        self,
        n_head: int,
        n_embd: int,
        block_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.head_size = n_embd // n_head
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout
        self.bias = bias
        self.layer_idx = layer_idx

        self.key = nn.Linear(n_embd, n_embd, bias=bias)
        self.query = nn.Linear(n_embd, n_embd, bias=bias)
        self.value = nn.Linear(n_embd, n_embd, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        kv_cache: object | None = None,  # PagedKVCache, avoid circular import
    ) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.query(x)
        k_new = self.key(x)
        v_new = self.value(x)

        q = q.view(B, S, self.n_head, self.head_size).transpose(1, 2)
        k_new = k_new.view(B, S, self.n_head, self.head_size).transpose(1, 2)
        v_new = v_new.view(B, S, self.n_head, self.head_size).transpose(1, 2)

        if use_cache and kv_cache is not None:
            k_past, v_past = kv_cache.gather(self.layer_idx)
            k = torch.cat([k_past, k_new], dim=2)
            v = torch.cat([v_past, v_new], dim=2)
            kv_cache.write_range(self.layer_idx, k_new, v_new)
        else:
            k = k_new
            v = v_new

        q_len = q.size(2)
        k_len = k.size(2)

        att = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        causal_mask = torch.ones(
            q_len, k_len, device=x.device, dtype=torch.bool,
        ).tril(diagonal=k_len - q_len)
        att = att.masked_fill(~causal_mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, S, self.n_embd)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out
