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
        self.blocks = nn.ModuleList(
            [
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
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, 
               torch.Tensor | None,
               tuple[tuple[torch.Tensor, torch.Tensor], ...] | None,
        ]:
        B, S = idx.shape

        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(2)
        positions = torch.arange(past_length, past_length+S, device=idx.device)
        
        if past_length + S > self.config.block_size:
            raise ValueError( 
                f"Sequence length {S} exceeds block_size={self.config.block_size}."
            )
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(positions)

        x = tok_emb + pos_emb
        x = self.dropout(x)
        
        presents = [] if use_cache else None
        
        for layer_idx, block in enumerate(self.blocks):
            layer_past = None
            if past_key_values is not None:
                layer_past = past_key_values[layer_idx] 
            
            x, present = block(
                x,
                layer_past = layer_past,
                use_cache=use_cache,
            )
            
            if use_cache:
                presents.append(present)
         
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.reshape(B * S, self.config.vocab_size)
            targets_flat = targets.reshape(B * S)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss, tuple(presents) if use_cache else None

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

        # prefill stage
        logits, _, past_key_value = self(idx[:, -self.config.block_size :], use_cache = True)
        for _ in range(max_new_tokens):
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
            logits, _, past_key_value = self(
                idx_next,
                past_key_values = past_key_value,
                use_cache = True,
            )
        return idx
