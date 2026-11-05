"""The execution boundary used by EngineCore; no backend is implemented here."""
from __future__ import annotations

from typing import Protocol

from llm.inference.types import ExecutionResult, ExecutionSpec, SchedulePlan


class Executor(Protocol):
    def initialize(self) -> ExecutionSpec: ...

    def execute(self, plan: SchedulePlan) -> ExecutionResult: ...

    def synchronize(self) -> None: ...

    def close(self) -> None: ...
