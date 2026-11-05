from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from time import perf_counter_ns

import torch

from llm.cache.blocks import BlockManager

from .request import Request, RequestStatus, RequestTiming
from .sampler import Sampler
from .types import ExecutionResult, ExecutionSpec, ScheduledItem, SchedulePlan


class Scheduler:
    """Plan work and commit results without owning model execution.

    Waiting requests have FIFO priority; admitted requests rotate after each
    execution. Capacity commitments cover completion, while reserve() only
    allocates slots needed by the current plan. B=1 uses the same batch path.
    """

    def __init__(
        self,
        block_manager: BlockManager,
        execution_spec: ExecutionSpec,
        sampler: Sampler | None = None,
        *,
        max_decode_batch_size: int = 1,
        max_prefill_batch_size: int = 1,
        max_prefill_tokens: int | None = None,
        record_timing: bool = False,
        clock: Callable[[], int] = perf_counter_ns,
    ) -> None:
        self.block_manager = block_manager
        self.execution_spec = execution_spec
        self.max_decode_batch_size = max_decode_batch_size
        self.max_prefill_batch_size = max_prefill_batch_size
        self.max_prefill_tokens = (
            execution_spec.context_length
            if max_prefill_tokens is None
            else max_prefill_tokens
        )
        if min(max_decode_batch_size, max_prefill_batch_size, self.max_prefill_tokens) < 1:
            raise ValueError("Batch sizes and the prefill token budget must be positive.")
        # TODO(module 2): BlockManager must provide the new control-side API.
        if (block_manager.capacity != execution_spec.num_blocks
                or block_manager.block_size != execution_spec.block_size):
            raise ValueError("BlockManager capacity must match ExecutionSpec.")

        self._sampling_device = torch.device(execution_spec.result_device)
        if self._sampling_device.type == "cuda" and self._sampling_device.index is None:
            self._sampling_device = torch.device("cuda", torch.cuda.current_device())
        self._logits_dtype = getattr(torch, execution_spec.logits_dtype)
        self.default_sampler = sampler or Sampler(device=self._sampling_device)
        if self.default_sampler.device != self._sampling_device:
            raise ValueError("Sampler device must match the result device.")

        self._requests: dict[str, Request] = {}
        self.waiting: deque[str] = deque()
        self.running: deque[str] = deque()
        self._preparing_ids: list[str] = []
        self._active_plan: SchedulePlan | None = None
        self._result_committed = False
        # Keep the original objects even if an ID is reused after a partial commit.
        self._active_requests: tuple[Request, ...] = ()
        self._capacity_commitments: dict[str, int] = {}
        self._samplers: dict[str, Sampler] = {}
        self._finished_pending: list[Request] = []
        self._next_plan_id = 0
        self.record_timing = record_timing
        self._clock = clock

    def add_request(self, request: Request) -> None:
        """Register a fresh request after checking that it can finish."""
        request_id = request.request_id
        if request_id in self._requests:
            raise ValueError(f"Duplicate active request_id: {request_id!r}")
        if (request.status is not RequestStatus.WAITING
                or request.generated_ids or request.num_computed_tokens != 0):
            raise ValueError("Only fresh waiting requests can be submitted.")
        if request.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative.")
        if request.max_new_tokens > 0:
            if not request.input_ids:
                raise ValueError("A generating request needs at least one prompt token.")
            cached_tokens = request.prompt_length + request.max_new_tokens - 1
            if cached_tokens > self.execution_spec.context_length:
                raise ValueError("Request exceeds the model context length.")
            if self._required_capacity(request) > self.block_manager.capacity:
                raise ValueError("Request exceeds the total cache capacity.")
            if request.prompt_length > self.max_prefill_tokens:
                raise ValueError("Request prompt exceeds the prefill token budget.")

        if request.sampling_params is None:
            sampler = self.default_sampler
        else:
            sampler = Sampler(request.sampling_params, device=self._sampling_device)
        request.timing = RequestTiming(self._clock()) if self.record_timing else None
        self._requests[request_id] = request
        self._samplers[request_id] = sampler
        self.waiting.append(request_id)

    def schedule(self) -> SchedulePlan | None:
        """Select requests, reserve their input slots, and publish one plan."""
        if self._active_plan is not None or self._preparing_ids:
            raise RuntimeError("Settle the previous plan before scheduling another.")

        try:
            self._select_prefill()
            if not self._preparing_ids:
                self._select_decode()
            if not self._preparing_ids:
                return None

            items = []
            requests = []
            for request_id in self._preparing_ids:
                request = self._requests[request_id]
                past_length = request.num_computed_tokens
                if past_length < request.prompt_length:
                    input_ids = tuple(request.input_ids[past_length:])
                    request.mark_prefilling()
                else:
                    output_start = past_length - request.prompt_length
                    input_ids = tuple(request.generated_ids[output_start:])
                    request.mark_decoding()

                if request.timing is not None and request.timing.admitted_at_ns is None:
                    request.timing.admitted_at_ns = self._clock()
                target_length = past_length + len(input_ids)
                self.block_manager.reserve(request_id, target_length)
                block_table = self.block_manager.snapshot(request_id)
                items.append(ScheduledItem(
                    request_id=request_id,
                    input_ids=input_ids,
                    past_length=past_length,
                    block_table=block_table,
                    needs_logits=True,
                ))
                requests.append(request)

            plan = SchedulePlan(self._next_plan_id, tuple(items))
        except BaseException:
            # No plan reached EngineCore, so Scheduler owns preparation cleanup.
            selected = [self._requests[key] for key in self._preparing_ids]
            self._finish_requests(selected, RequestStatus.FAILED)
            self._preparing_ids.clear()
            raise

        self._active_plan = plan
        self._active_requests = tuple(requests)
        self._next_plan_id += 1
        self._preparing_ids.clear()
        return plan

    def update(self, plan_id: int, result: ExecutionResult) -> None:
        """Sample all output rows, then commit progress exactly once.

        If update fails, the caller must establish a safe boundary and abort.
        A cleanup failure does not permit resampling the same plan.
        """
        plan = self._active_plan
        if plan is None or plan.plan_id != plan_id or result.plan_id != plan_id:
            raise ValueError("Result does not belong to the active plan.")
        if self._result_committed:
            raise ValueError("Result was already committed; abort the unsettled plan.")

        expected_ids = tuple(item.request_id for item in plan.items if item.needs_logits)
        if result.request_ids != expected_ids:
            raise ValueError("Result rows must follow the plan's sampling request order.")
        if result.logits.shape != (len(expected_ids), self.execution_spec.vocab_size):
            raise ValueError("Result logits must have shape [S, vocab_size].")
        if (result.logits.device != self._sampling_device
                or result.logits.dtype != self._logits_dtype):
            raise ValueError("Result device and dtype must match ExecutionSpec.")

        token_ids = self._sample_tokens(result)
        ready_at_ns = self._clock() if token_ids and self.record_timing else None

        self._result_committed = True
        # Progress belongs to every plan item, including items without logits.
        for item, request in zip(plan.items, self._active_requests):
            request.num_computed_tokens = item.past_length + len(item.input_ids)
            if item.needs_logits:
                request.append_token(token_ids[item.request_id])
                if request.timing is not None:
                    request.timing.token_ready_at_ns.append(ready_at_ns)

        completed = []
        continuing = []
        for request in self._active_requests:
            if request.should_stop:
                completed.append(request)
            else:
                if request.num_computed_tokens >= request.prompt_length:
                    request.mark_decoding()
                continuing.append(request.request_id)

        # A release failure must not skip cleanup of the other completed rows.
        self._finish_requests(completed, RequestStatus.FINISHED)
        self.running.extend(continuing)
        self._result_committed = False
        self._active_plan = None
        self._active_requests = ()

    def abort(self, plan_id: int) -> bool:
        """Fail a matching plan only after execution has safely stopped."""
        if self._active_plan is None or self._active_plan.plan_id != plan_id:
            return False
        self._finish_requests(self._active_requests, RequestStatus.FAILED)
        self._result_committed = False
        self._active_plan = None
        self._active_requests = ()
        return True

    def cancel_request(self, request_id: str) -> bool:
        """Cancel between steps; this never interrupts device execution."""
        self._require_idle()
        request = self._requests.get(request_id)
        if request is None:
            return False
        self._finish_request(request, RequestStatus.CANCELLED)
        return True

    def cancel_all(self) -> None:
        """Cancel pending requests without unloading model or worker resources."""
        self._require_idle()
        self._finish_requests(list(self._requests.values()), RequestStatus.CANCELLED)
        self._preparing_ids.clear()

    def pop_finished(self) -> list[Request]:
        """Transfer terminal records without touching their released resources."""
        finished = self._finished_pending
        self._finished_pending = []
        return finished

    def synchronize_sampling(self) -> None:
        samplers = set(self._samplers.values())
        samplers.add(self.default_sampler)
        for sampler in samplers:
            sampler.synchronize()

    @property
    def is_done(self) -> bool:
        return not self._requests

    @property
    def num_active_requests(self) -> int:
        return len(self._requests)

    def _required_capacity(self, request: Request) -> int:
        if request.max_new_tokens == 0:
            return 0
        # The final sampled token is returned without another model execution.
        cached_tokens = request.prompt_length + request.max_new_tokens - 1
        return self.block_manager.required_capacity(cached_tokens)

    def _select_prefill(self) -> None:
        committed_capacity = sum(self._capacity_commitments.values())
        max_input_length = 0
        while self.waiting and len(self._preparing_ids) < self.max_prefill_batch_size:
            request_id = self.waiting[0]
            request = self._requests[request_id]
            if request.should_stop:
                self._finish_request(request, RequestStatus.FINISHED)
                continue

            batch_size = len(self._preparing_ids) + 1
            input_length = max(max_input_length, request.prompt_length)
            if batch_size * input_length > self.max_prefill_tokens:
                break
            request_capacity = self._required_capacity(request)
            if committed_capacity + request_capacity > self.block_manager.capacity:
                break

            self.waiting.popleft()
            self._preparing_ids.append(request_id)
            self._capacity_commitments[request_id] = request_capacity
            committed_capacity += request_capacity
            max_input_length = input_length

    def _select_decode(self) -> None:
        batch_size = min(self.max_decode_batch_size, len(self.running))
        for _ in range(batch_size):
            self._preparing_ids.append(self.running.popleft())

    def _sample_tokens(self, result: ExecutionResult) -> dict[str, int]:
        tokens = []
        for row, request_id in enumerate(result.request_ids):
            row_logits = result.logits[row:row + 1]
            token = self._samplers[request_id].sample(row_logits)
            tokens.append(token)
        if not tokens:
            return {}
        # One CPU transfer observes all sampled rows before any output is visible.
        token_ids = torch.cat(tokens, dim=0).flatten().cpu().tolist()
        return dict(zip(result.request_ids, token_ids))

    def _finish_request(self, request: Request, status: RequestStatus) -> None:
        request_id = request.request_id
        if self._requests.get(request_id) is not request:
            return

        if request_id in self._capacity_commitments:
            self.block_manager.release(request_id)
        request.num_computed_tokens = 0
        if request_id in self._preparing_ids:
            self._preparing_ids.remove(request_id)
        if request_id in self.waiting:
            self.waiting.remove(request_id)
        if request_id in self.running:
            self.running.remove(request_id)
        del self._requests[request_id]
        self._samplers.pop(request_id)
        self._capacity_commitments.pop(request_id, None)
        request.status = status
        if request.timing is not None:
            request.timing.finished_at_ns = self._clock()
        self._finished_pending.append(request)

    def _finish_requests(
        self,
        requests: Iterable[Request],
        status: RequestStatus,
    ) -> None:
        errors = []
        for request in requests:
            try:
                self._finish_request(request, status)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("Request cleanup failed", errors)

    def _require_idle(self) -> None:
        if self._active_plan is not None:
            raise RuntimeError("Settle the active plan before cancelling requests.")
