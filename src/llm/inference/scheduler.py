from __future__ import annotations

from collections import deque

import torch

from .request import Request, RequestStatus
from .runner import Runner
from .sampler import Sampler


class Scheduler:
    """FIFO admission and round-robin decode with injected model execution.

    Reserve enough capacity for admitted requests to finish. Each step executes
    one request; this is not tensor-level continuous batching or paged attention.
    """

    def __init__(self, runner: Runner, sampler: Sampler | None = None) -> None:
        self.waiting: deque[Request] = deque()
        self.running: deque[Request] = deque()
        self.finished: list[Request] = []
        self._next_inputs: dict[str, torch.Tensor] = {}
        self._samplers: dict[str, Sampler] = {}
        self.runner = runner
        self.cache = runner.cache
        self.default_sampler = sampler or Sampler()

    def _required_capacity(self, request: Request) -> int:
        if request.max_new_tokens == 0:
            return 0
        return self.cache.required_capacity(
            request.prompt_length + request.max_new_tokens - 1,
        )

    def _can_admit(self, request: Request) -> bool:
        reserved = sum(self._required_capacity(r) for r in self.running)
        return reserved + self._required_capacity(request) <= self.cache.capacity

    def _finish_request(self, request: Request, status: RequestStatus) -> None:
        self.cache.release(request.request_id)
        self._next_inputs.pop(request.request_id, None)
        self._samplers.pop(request.request_id, None)
        request.status = status
        self.finished.append(request)

    def add_request(self, request: Request) -> None:
        if request.request_id in self._samplers:
            raise ValueError(f"Duplicate request_id: {request.request_id!r}")
        if request.status is not RequestStatus.WAITING or request.generated_ids:
            raise ValueError("Only fresh waiting requests can be submitted.")
        if request.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative.")
        if request.max_new_tokens > 0 and not request.input_ids:
            raise ValueError("A request must contain at least one prompt token.")
        cached_tokens = request.prompt_length + max(request.max_new_tokens - 1, 0)
        if cached_tokens > self.runner.spec.context_length:
            raise ValueError(
                f"Request requires {cached_tokens} cached tokens, exceeding "
                f"context_length={self.runner.spec.context_length}."
            )
        if self._required_capacity(request) > self.cache.capacity:
            raise ValueError("Request requires more cache capacity than is available.")
        self._samplers[request.request_id] = (
            Sampler(request.sampling_params)
            if request.sampling_params is not None
            else self.default_sampler
        )
        self.waiting.append(request)

    def step(self) -> list[Request]:
        if self.waiting and self._can_admit(self.waiting[0]):
            request = self.waiting.popleft()
            prefill = True
        elif self.running:
            request = self.running.popleft()
            prefill = False
        else:
            return []

        try:
            if request.is_finished:
                self._finish_request(request, RequestStatus.FINISHED)
                return [request]
            if prefill:
                request.mark_prefilling()
                self.cache.allocate(request.request_id, request.prompt_length)
                input_ids = torch.tensor(
                    [request.input_ids], dtype=torch.long,
                    device=self.runner.spec.device,
                )
                output = self.runner.prefill(input_ids, request.request_id)
                request.mark_decoding()
            else:
                output = self.runner.decode(
                    self._next_inputs[request.request_id], request.request_id,
                )
            token = self._samplers[request.request_id].sample(output.logits)
            request.append_token(token.item())
            if request.is_finished:
                self._finish_request(request, RequestStatus.FINISHED)
                return [request]
            # The last sampled token is returned without another forward pass.
            self.cache.append(request.request_id)
            self._next_inputs[request.request_id] = token
            self.running.append(request)
        except BaseException:
            self._finish_request(request, RequestStatus.FAILED)
            raise
        return []

    def run(self) -> list[Request]:
        """Drain submitted requests, releasing pending resources on failure."""
        try:
            while not self.is_done:
                self.step()
        except BaseException:
            self.close()
            raise
        return self.finished

    def close(self) -> None:
        """Cancel pending work and release resources; safe to call repeatedly."""
        while self.waiting:
            self._finish_request(self.waiting.popleft(), RequestStatus.CANCELLED)
        while self.running:
            self._finish_request(self.running.popleft(), RequestStatus.CANCELLED)

    @property
    def is_done(self) -> bool:
        return not self.waiting and not self.running
