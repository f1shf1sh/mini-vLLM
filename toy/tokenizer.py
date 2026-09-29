"""Learning implementation of character tokenization, kept for existing toy checkpoints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class CharTokenizer:
    """A minimal reversible character tokenizer."""

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        if not chars:
            raise ValueError("Tokenizer text must contain at least one character.")
        self.stoi = {ch: idx for idx, ch in enumerate(chars)}
        self.itos = {idx: ch for ch, idx in self.stoi.items()}

    @classmethod
    def from_vocab(cls, raw_stoi: dict[Any, Any], raw_itos: dict[Any, Any]) -> "CharTokenizer":
        tokenizer = cls.__new__(cls)
        tokenizer.stoi = {str(ch): int(idx) for ch, idx in raw_stoi.items()}
        tokenizer.itos = {int(idx): str(ch) for idx, ch in raw_itos.items()}
        return tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        missing = sorted({ch for ch in text if ch not in self.stoi})
        if missing:
            raise ValueError(f"Input contains unseen characters: {missing!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.itos[int(idx)] for idx in ids)
