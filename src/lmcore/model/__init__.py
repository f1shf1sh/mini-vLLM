"""Minimal decoder-only Transformer model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .block import Block
from ..config import ModelConfig


class GPTMini(nn.Module):
    """Minimal decoder-only Transformer.

    Training::

        logits, loss = model(idx, targets=targets)

    Inference with paged KV cache::

        logits, _ = model(idx, use_cache=True, kv_cache=paged_kv)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size <= 0:
            raise ValueError("config.vocab_size must be positive.")

        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embd=config.n_embd,
                    n_head=config.n_head,
                    block_size=config.block_size,
                    dropout=config.dropout,
                    bias=config.bias,
                    layer_idx=i,
                )
                for i in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        use_cache: bool = False,
        kv_cache: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, S = idx.shape

        past_length = kv_cache.num_tokens if kv_cache is not None else 0
        total_length = past_length + S
        if total_length > self.config.block_size:
            raise ValueError(
                f"Sequence length {total_length} exceeds "
                f"block_size={self.config.block_size}."
            )

        positions = torch.arange(past_length, total_length, device=idx.device)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(positions)
        x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x, use_cache=use_cache, kv_cache=kv_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.reshape(B * S, self.config.vocab_size)
            targets_flat = targets.reshape(B * S)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss
