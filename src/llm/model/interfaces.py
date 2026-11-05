"""Model metadata shared with the execution layer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Logical model limits plus rank-local execution metadata."""

    context_length: int
    vocab_size: int
    pad_token_id: int | None
    default_stop_token_ids: tuple[int, ...]
    device: str
    dtype: str
    logits_dtype: str
