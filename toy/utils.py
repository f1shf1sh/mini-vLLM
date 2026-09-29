"""Small shared utilities used by the project scaffold."""

from __future__ import annotations

from contextlib import contextmanager
import random
import time
from collections.abc import Iterator

import torch


def set_seed(seed: int) -> None:
    """Seed common RNGs for repeatable experiments."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def timer() -> Iterator[callable]:
    """Yield a closure that reports elapsed wall time in seconds."""

    start = time.perf_counter()

    def elapsed() -> float:
        return time.perf_counter() - start

    yield elapsed


def tokens_per_second(num_tokens: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        return 0.0
    return num_tokens / elapsed_seconds
