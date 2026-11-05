from __future__ import annotations

from llm.executor.base import Executor
from llm.inference.request import Request
from llm.inference.scheduler import Scheduler
from llm.inference.types import ExecutionSpec


class EngineCore:
    """Drive one round; Scheduler owns requests and Executor owns execution."""

    def __init__(
        self,
        scheduler: Scheduler,
        executor: Executor,
    ) -> None:
        self.scheduler = scheduler
        self.executor = executor

    def add_request(self, request: Request) -> None:
        self.scheduler.add_request(request)

    def step(self) -> None:
        """Run one plan without consuming finished requests.

        TODO(module 8B): coordinate execution failures and safe abort.
        Until then, a failed execute/update leaves the plan unsettled.
        """
        plan = self.scheduler.schedule()
        if plan is None:
            return
        result = self.executor.execute(plan)
        self.scheduler.update(plan.plan_id, result)

    def pop_finished(self) -> list[Request]:
        return self.scheduler.pop_finished()

    def cancel_request(self, request_id: str) -> bool:
        return self.scheduler.cancel_request(request_id)

    @property
    def execution_spec(self) -> ExecutionSpec:
        return self.scheduler.execution_spec

    @property
    def is_done(self) -> bool:
        return self.scheduler.is_done

    @property
    def num_active_requests(self) -> int:
        return self.scheduler.num_active_requests

    def run(self) -> list[Request]:
        # TODO(module 8B): drive steps, detect stalls, then collect terminal records.
        raise NotImplementedError("Engine run is completed in module 8B.")

    def cancel_pending(self) -> None:
        # TODO(module 8B): establish a safe boundary before cancelling requests.
        raise NotImplementedError("Engine cancellation is completed in module 8B.")

    def synchronize(self) -> None:
        # TODO(module 8B): synchronize both execution and control-side sampling.
        raise NotImplementedError("Engine synchronization is completed in module 8B.")

    def close(self) -> None:
        # TODO(module 8B): cancel requests and close the executor permanently.
        raise NotImplementedError("Engine shutdown is completed in module 8B.")
