"""Process-local environment contract; no communication implementation."""
from __future__ import annotations

from typing import Protocol

import torch


class ParallelContext(Protocol):
    """Metadata required by model execution, never transported to Scheduler."""

    rank: int
    local_rank: int
    world_size: int
    driver_rank: int
    device: torch.device
