from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from .sampler import SamplingParams

class RequestStatus(Enum):
    WAITING = auto()
    PREFILLING = auto()
    DECODING = auto()
    FINISHED = auto()
    FAILED = auto()
    CANCELLED = auto()
 
@dataclass(slots=True)
class RequestTiming:
    """Monotonic host timestamps; token readiness follows the CPU token read."""

    submitted_at_ns: int
    admitted_at_ns: int | None = None
    token_ready_at_ns: list[int] = field(default_factory=list)
    finished_at_ns: int | None = None


@dataclass(slots=True)
class Request:
    request_id: str
    input_ids: list[int]
    max_new_tokens: int
    sampling_params: SamplingParams | None = field(default_factory=SamplingParams)
    generated_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    
    stop_token_ids: tuple[int, ...] = ()
    timing: RequestTiming | None = field(default=None, init=False, repr=False)

    @property
    def prompt_length(self) -> int:
        return len(self.input_ids)
    
    @property
    def num_generated_tokens(self) -> int:
        return len(self.generated_ids)
    
    @property
    def output_ids(self) -> list[int]:
        return self.input_ids + self.generated_ids
    
    @property
    def is_finished(self) -> bool:
        return (
            self.status in (RequestStatus.FINISHED, RequestStatus.FAILED, RequestStatus.CANCELLED)
            or self.num_generated_tokens >= self.max_new_tokens
            or bool(self.generated_ids and self.generated_ids[-1] in self.stop_token_ids)
        )
    
    def mark_prefilling(self) -> None:
        self.status = RequestStatus.PREFILLING
        
    def mark_decoding(self) -> None:
        self.status = RequestStatus.DECODING

    def append_token(self, token_id: int) -> None:
        self.generated_ids.append(token_id)
        
        if self.is_finished:
            self.status = RequestStatus.FINISHED
