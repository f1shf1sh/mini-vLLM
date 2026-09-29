from __future__ import annotations

from collections import deque

import torch

from .request import Request, RequestStatus
from .sampler import Sampler
from .model_runner import ModelRunner
from ..cache.blocks import BlockManager
from ..cache.pool import KVPool


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
        self._context_length = model.config.block_size
        self._num_blocks = kv_pool.num_blocks

    def _required_blocks(self, request: Request) -> int:
        if request.max_new_tokens == 0:
            return 0
        cached_tokens = request.prompt_length + request.max_new_tokens - 1
        return (cached_tokens + self.block_manager.block_size - 1) // self.block_manager.block_size

    def _can_admit(self, request: Request) -> bool:
        # Leave enough capacity for every running request to finish decoding.
        reserved = sum(self._required_blocks(r) for r in self.running)
        return reserved + self._required_blocks(request) <= self._num_blocks

    def add_request(self, request: Request) -> None:
        if request.request_id in self._samplers:
            raise ValueError(f"Duplicate request_id: {request.request_id!r}")
        if request.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative.")
        if request.max_new_tokens > 0 and not request.input_ids:
            raise ValueError("A request must contain at least one prompt token.")
        cached_tokens = request.prompt_length + max(request.max_new_tokens - 1, 0)
        if cached_tokens > self._context_length:
            raise ValueError(
                f"Request requires {cached_tokens} cached tokens, exceeding "
                f"block_size={self._context_length}."
            )
        if self._required_blocks(request) > self._num_blocks:
            raise ValueError("Request requires more KV blocks than the pool provides.")
        self._samplers[request.request_id] = (
            Sampler(request.sampling_params)
            if request.sampling_params is not None
            else self.default_sampler
        )
        self.waiting.append(request)

    def step(self) -> list[Request]:
        finished: list[Request] = []

        if self.waiting and self._can_admit(self.waiting[0]):
            request = self.waiting.popleft()
            if request.is_finished:
                request.status = RequestStatus.FINISHED
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

        if request.is_finished:
            self.finished.append(request)
            finished.append(request)
            self._next_inputs.pop(request.request_id, None)
            del self._samplers[request.request_id]
            self.block_manager.free_request(request.request_id)
        else:
            # The final sampled token is returned without another forward pass,
            # so only unfinished requests need a new KV slot.
            self.block_manager.append_token(request.request_id)
            self._next_inputs[request.request_id] = token
            self.running.append(request)

        return finished

    @property
    def is_done(self) -> bool:
        return not self.waiting and not self.running
