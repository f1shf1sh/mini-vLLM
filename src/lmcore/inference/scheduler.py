from __future__ import annotations

from collections import deque

import torch

from .request import Request
from .sampler import Sampler
from .model_runner import ModelRunner
from .block_manager import BlockManager
from .kv_pool import KVPool


class Scheduler:
    def __init__(self,
                 model: torch.nn.Module,
                 kv_pool: KVPool,
                 block_manager: BlockManager,
                 sampler: Sampler | None = None,
        ):
        self.waiting: deque[Request] = deque()
        self.running: deque[Request] = deque()
        self.finished: list[Request] = []
        self._next_inputs: dict[str, torch.Tensor] = {}
        self._samplers: dict[str, Sampler] = {}
        self.runner = ModelRunner(model, kv_pool, block_manager)
        self.block_manager = block_manager
        self.default_sampler = sampler or Sampler()
        self._device = next(model.parameters()).device

    def add_request(self, request: Request) -> None:
        if request.request_id in self._samplers:
            raise ValueError(f"Duplicate request_id: {request.request_id!r}")
        self._samplers[request.request_id] = (
            Sampler(request.sampling_params)
            if request.sampling_params is not None
            else self.default_sampler
        )
        self.waiting.append(request)

    def step(self) -> list[Request]:
        finished: list[Request] = []

        if self.waiting:
            request = self.waiting.popleft()
            if request.is_finished:
                self.finished.append(request)
                finished.append(request)
                del self._samplers[request.request_id]
                return finished

            request.mark_prefilling()
            self.block_manager.allocate_for_request(
                request.request_id, len(request.input_ids),
            )
            input_tensor = torch.tensor(
                [request.input_ids], dtype=torch.long, device=self._device,
            )
            output = self.runner.prefill(input_tensor, request.request_id)
            request.mark_decoding()
        elif self.running:
            request = self.running.popleft()
            output = self.runner.decode(
                self._next_inputs[request.request_id], request.request_id,
            )
        else:
            return finished

        sampler = self._samplers[request.request_id]
        token = sampler.sample(output.logits)
        request.append_token(token.item())
        self.block_manager.append_token(request.request_id)
        self._next_inputs[request.request_id] = token

        if request.is_finished:
            self.finished.append(request)
            finished.append(request)
            del self._next_inputs[request.request_id]
            del self._samplers[request.request_id]
            self.block_manager.free_request(request.request_id)
        else:
            self.running.append(request)

        return finished

    @property
    def is_done(self) -> bool:
        return not self.waiting and not self.running
