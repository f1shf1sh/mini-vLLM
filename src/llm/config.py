"""Inference settings; new data configs are checked at their startup boundaries."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from .inference.sampler import SamplingParams


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Serializable model source; loading belongs to Worker/Loader."""

    model: str
    dtype: str = "float32"


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """Common capacity confirmed by Executor, not a per-worker budget."""

    block_size: int
    num_blocks: int


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
    """Execution settings only; declaring a backend does not implement it."""

    backend: Literal["local", "multiprocess"] = "local"
    result_transport: Literal["local_device", "cpu"] = "local_device"
    init_method: str | None = None
    timeout_seconds: float = 30.0


@dataclass(slots=True)
class InferenceConfig:
    model: str = "/models/Qwen2.5-0.5B-Instruct"
    device: str = "cuda"
    dtype: str = "float32"
    num_blocks: int = 256
    block_size: int = 16
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    do_sample: bool = False

    def __post_init__(self) -> None:
        if self.num_blocks < 1 or self.block_size < 1:
            raise ValueError("num_blocks and block_size must be positive.")
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative.")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("dtype must be float32, float16, or bfloat16.")
        self.sampling_params()

    def sampling_params(self) -> SamplingParams:
        return SamplingParams(self.temperature, self.top_k, self.top_p, self.do_sample)


def load_config(path: str | Path) -> InferenceConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Inference config root must be a mapping.")
    return InferenceConfig(**raw)
