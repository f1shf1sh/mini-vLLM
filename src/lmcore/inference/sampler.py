from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    do_sample: bool = True

class Sampler:
    def __init__(self, params: SamplingParams | None = None) -> None:
        self.params = params or SamplingParams() 
    
     
    def sample(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits[:, -1, :]
        logits = logits / self.params.temperature
        
        """This can be extended later, for example with top-p sampling."""
        
        if self.params.top_k is not None:
            k = min(self.params.top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            cutoff = values[:, [-1]]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    