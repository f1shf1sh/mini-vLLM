"""Minimal decoder-only Transformer model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .block import Block
from ..config import ModelConfig


class GPTMini(nn.Module):
    """Minimal decoder-only Transformer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size <= 0:
            raise ValueError("config.vocab_size must be positive.")

        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd=config.n_embd,
                    n_head=config.n_head,
                    block_size=config.block_size,
                    dropout=config.dropout,
                    bias=config.bias,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, S = idx.shape
        if S > self.config.block_size:
            raise ValueError(
                f"Sequence length {S} exceeds block_size={self.config.block_size}."
            )

        positions = torch.arange(S, device=idx.device)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(positions)

        x = tok_emb + pos_emb
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.reshape(B * S, self.config.vocab_size)
            targets_flat = targets.reshape(B * S)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]
            logits = logits / temperature

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                cutoff = values[:, [-1]]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
