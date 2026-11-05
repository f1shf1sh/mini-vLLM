from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    do_sample: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive when provided.")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")


class Sampler:
    def __init__(
        self,
        params: SamplingParams | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.params = params or SamplingParams()
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

    def synchronize(self) -> None:
        """Wait for sampling work on the fixed device."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert full-vocabulary logits [R, V] into token IDs [R, 1]."""
        if not self.params.do_sample:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / self.params.temperature
        if self.params.top_k is not None:
            k = min(self.params.top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
        if self.params.top_p is not None and self.params.top_p < 1:
            sorted_logits, indices = torch.sort(logits, descending=True, dim=-1)
            cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove = cumulative > self.params.top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            mask = torch.zeros_like(remove).scatter(-1, indices, remove)
            logits = logits.masked_fill(mask, float("-inf"))
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
