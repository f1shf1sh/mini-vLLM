from __future__ import annotations

import torch

from .types import KVCache, RunnerOutput

class ModelRunner:
    """Thin inference wrapper around GPTMini.forward()."""
    
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.model.eval()
    
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> RunnerOutput:
        if input_ids.ndim != 2:
            raise ValueError("prefill expects input_ids with shape [B, T].")
        if input_ids.size(1) < 1:
            raise ValueError("prefill expects at least one token.")
        
        logits, _, kv_cache = self.model(
            idx = input_ids,
            use_cache = True,
        )
        
        if kv_cache is None:
            raise RuntimeError("model did not return KV cache during prefill.")
        
        return RunnerOutput(
            logits=logits,
            kv_cache=kv_cache,
        )
        
    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache,
    ) -> RunnerOutput:
        if input_ids.ndim != 2:
            raise ValueError("decode expects input_ids with shape [B, 1].")
        if input_ids.size(1) != 1:
            raise ValueError("decode expects exactly one token per request.") 
          
        logits, _, new_kv_cache = self.model(
            input_ids,
            past_key_values = kv_cache,
            use_cache = True,
        )
        
        if new_kv_cache is None:
            raise RuntimeError("model did not return KV cache during decode.")
        
        return RunnerOutput(
            logits=logits,
            kv_cache=new_kv_cache,
        )