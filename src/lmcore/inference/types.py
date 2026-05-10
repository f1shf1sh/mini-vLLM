from __future__ import annotations
from dataclasses import dataclass

import torch

KVCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]

@dataclass(slots=True)
class RunnerOutput:
    logits: torch.Tensor
    kv_cache: KVCache