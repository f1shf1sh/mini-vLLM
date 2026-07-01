from __future__ import annotations
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class RunnerOutput:
    logits: torch.Tensor
