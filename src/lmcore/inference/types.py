from __future__ import annotations
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class RunnerOutput:
    logits: torch.Tensor
    # kv_cache removed — KV is stored in the global KVPool, accessed via PagedKVCache.
