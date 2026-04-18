"""Dataset utilities for the character-level language model scaffold."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


class CharTokenizer:
    """A minimal reversible character tokenizer."""

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        if not chars:
            raise ValueError("Tokenizer text must contain at least one character.")
        self.stoi = {ch: idx for idx, ch in enumerate(chars)}
        self.itos = {idx: ch for ch, idx in self.stoi.items()}
        
    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.itos[idx] for idx in ids)


class TextDataset:
    """Random-window dataset used for next-token training."""

    def __init__(
        self,
        token_ids: Sequence[int],
        block_size: int,
        batch_size: int,
        train_split: float = 0.9,
        device: str = "cpu",
    ) -> None:
        if not 0.0 < train_split < 1.0:
            raise ValueError("train_split must be between 0 and 1.")
        if block_size < 1:
            raise ValueError("block_size must be positive.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        ids = torch.tensor(token_ids, dtype=torch.long)
        if ids.numel() <= block_size + 1:
            raise ValueError("token_ids must be longer than block_size + 1.")

        # Safe split 
        split_idx = int(ids.numel() * train_split)
        split_idx = min(max(split_idx, block_size + 1), ids.numel() - (block_size + 1))

        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.train_ids = ids[:split_idx]
        self.val_ids = ids[split_idx:]

    def get_batch(self, split: str) -> tuple[Tensor, Tensor]:
        if split == "train":
            source = self.train_ids
        elif split == "val":
            source = self.val_ids
        else:
            raise ValueError("split must be 'train' or 'val'.")

        max_start = source.numel() - self.block_size - 1
        if max_start < 0:
            raise ValueError(f"split '{split}' is too short for block_size={self.block_size}.")

        starts = torch.randint(0, max_start + 1, (self.batch_size,))
        x = torch.stack(
            [source[start : start + self.block_size] for start in starts.tolist()]
        )
        y = torch.stack(
            [source[start + 1 : start + self.block_size + 1] for start in starts.tolist()]
        )
        return x.to(self.device), y.to(self.device)
