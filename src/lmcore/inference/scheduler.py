from __future__ import annotations

import torch

from .request import Request
from .types import KVCache
from .sampler import Sampler
from .model_runner import ModelRunner

class Scheduler:
    def __init__(self,
                 model: torch.nn.Module,
                 sampler: Sampler | None = None
        ):
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self._kv_caches: dict[str, KVCache] = {} # key = request.request_id
        self._next_inputs: dict[str, torch.Tensor] = {}
        self.runner = ModelRunner(model)
        self.sampler = sampler or Sampler()

    def step(self) -> list[Request]:
        finished: list[Request] = []
        
        if self.waiting:
            request = self.waiting.pop()
            request.mark_prefilling()
            input_tensor = torch.tensor([request.input_ids], dtype=torch.long)
            output = self.runner.prefill(input_tensor)
            request.mark_decoding()
        elif self.running:
            request = self.running.pop()
            output = self.runner.decode(
                self._next_inputs[request.request_id],
                self._kv_caches[request.request_id],
            )
        else:
            return finished
        
        token = self.sampler.sample(output.logits)
        request.append_token(token.item())
        self._kv_caches[request.request_id] = output.kv_cache
        self._next_inputs[request.request_id] = token
        
        if request.is_finished:
            self.finished.append(request)
            finished.append(request)
            del self._kv_caches[request.request_id]
            del self._next_inputs[request.request_id]
        else:
            self.running.append(request)
            
        return finished
        
    @property 
    def is_done(self) -> bool:
        if not self.waiting and not self.running:
            return True
        return False 
    
    def add_request(self, request: Request) -> None:
        self.waiting.append(request)