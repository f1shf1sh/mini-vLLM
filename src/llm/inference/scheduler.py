from __future__ import annotations

from collections import deque
from collections.abc import Callable
from time import perf_counter_ns
from typing import cast

import torch

from .request import Request, RequestStatus, RequestTiming
from .runner import BatchDecodeRunner, Runner
from .sampler import Sampler


class Scheduler:
    """FIFO admission and round-robin decode with injected model execution.

    Reserve enough capacity for admitted requests to finish. Prefill executes
    one request; decode can execute up to max_batch_size requests in one forward.
    Scheduling and cancellation are synchronous, between steps.
    """

    def __init__(
        self,
        runner: Runner,
        sampler: Sampler | None = None,
        *,
        max_batch_size: int = 1,
        record_timing: bool = False,
        clock: Callable[[], int] = perf_counter_ns,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if max_batch_size > 1 and not callable(getattr(runner, "decode_batch", None)):
            raise ValueError("max_batch_size > 1 requires a batch decode runner")
        self.max_batch_size = max_batch_size
        self.waiting: deque[Request] = deque()
        self.running: deque[Request] = deque()
        self.finished: list[Request] = []
        self._next_inputs: dict[str, torch.Tensor] = {}
        self._samplers: dict[str, Sampler] = {}
        self.runner = runner
        self.cache = runner.cache
        self.default_sampler = sampler or Sampler()
        self.record_timing = record_timing
        self._clock = clock

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
        if request.timing is not None:
            request.timing.finished_at_ns = self._clock()

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
        sampler = (
            Sampler(request.sampling_params)
            if request.sampling_params is not None
            else self.default_sampler
        )
        timing = RequestTiming(self._clock()) if self.record_timing else None
        self._samplers[request.request_id] = sampler
        request.timing = timing
        self.waiting.append(request)

    def cancel_request(self, request_id: str) -> bool:
        """Cancel one active request between steps, preserving other queue order.

        Unknown or terminal IDs return False. A reused ID refers to its current
        active request; cancellation does not interrupt an in-flight forward.
        """
        for queue in (self.waiting, self.running):
            for index, request in enumerate(queue):
                if request.request_id == request_id:
                    del queue[index]
                    self._finish_request(request, RequestStatus.CANCELLED)
                    return True
        return False

    def step(self) -> list[Request]:
        if self.waiting and self._can_admit(self.waiting[0]):
            request = self.waiting.popleft()
            prefill = True
        elif self.running:
            if self.max_batch_size > 1:
                return self._step_decode_batch()
            request = self.running.popleft()
            prefill = False
        else:
            return []

        try:
            if request.is_finished:
                self._finish_request(request, RequestStatus.FINISHED)
                return [request]
            if prefill:
                if request.timing is not None:
                    request.timing.admitted_at_ns = self._clock()
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
            if request.timing is not None:
                request.timing.token_ready_at_ns.append(self._clock())
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

    def _step_decode_batch(self) -> list[Request]:
        # Admission is already complete before dequeuing. No other request is
        # admitted until this synchronous batch finishes, so reservations remain
        # valid even while its members are temporarily outside self.running.
        batch = [
            self.running.popleft()
            for _ in range(min(self.max_batch_size, len(self.running)))
        ]
        completed = []
        completed_ids = set()
        continuing = []
        try:
            request_ids = [request.request_id for request in batch]
            input_ids = torch.cat([self._next_inputs[key] for key in request_ids], dim=0)
            runner = cast(BatchDecodeRunner, self.runner)
            output = runner.decode_batch(input_ids, request_ids)
            if (output.logits.ndim != 3 or output.logits.shape[:2] != (len(batch), 1)
                    or output.logits.shape[2] < 1):
                raise ValueError("Batch decode logits must have shape [B, 1, vocabulary_size]")
            # Keep each request's sampling policy while sharing the model forward.
            # Sample every row before committing any result: a sampling failure
            # invalidates the whole executing batch, whose KV may already change.
            tokens = [
                self._samplers[request.request_id].sample(output.logits[i:i + 1])
                for i, request in enumerate(batch)
            ]
            token_ids = torch.cat(tokens, dim=0).flatten().cpu().tolist()
            for request, token_id in zip(batch, token_ids):
                request.append_token(token_id)
            # All rows are observed ready after the same CPU transfer. Do not
            # introduce artificial per-row latency by synchronizing each token.
            if self.record_timing:
                ready_at_ns = self._clock()
                for request in batch:
                    request.timing.token_ready_at_ns.append(ready_at_ns)
            for request, token in zip(batch, tokens):
                if request.is_finished:
                    self._finish_request(request, RequestStatus.FINISHED)
                    completed.append(request)
                    completed_ids.add(request.request_id)
                else:
                    self.cache.append(request.request_id)
                    self._next_inputs[request.request_id] = token
                    continuing.append(request)
            # Delay requeueing until all rows commit, so failures cannot leave a
            # FAILED request in running or finish a previously completed row twice.
            self.running.extend(continuing)
        except BaseException:
            for request in batch:
                if request.request_id not in completed_ids:
                    self._finish_request(request, RequestStatus.FAILED)
            raise
        return completed

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
