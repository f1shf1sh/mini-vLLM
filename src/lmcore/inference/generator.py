from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .model_runner import ModelRunner
from .sampler import Sampler, SamplingParams


class BaseGenerator(ABC):
    """Common interface for single-request generation strategies."""

    @abstractmethod
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        raise NotImplementedError


class CachedGenerator(BaseGenerator):
    """Single-request generator backed by prefill/decode KV cache."""

    def __init__(
        self,
        model: torch.nn.Module,
        sampler: Sampler | None = None,
    ) -> None:
        self.runner = ModelRunner(model)
        self.sampler = sampler or Sampler()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        if max_new_tokens == 0:
            return input_ids

        output = self.runner.prefill(input_ids)
        generated_ids = input_ids

        for _ in range(max_new_tokens):
            next_token = self.sampler.sample(output.logits)
            generated_ids = torch.cat((generated_ids, next_token), dim=1)
            output = self.runner.decode(next_token, output.kv_cache)

        return generated_ids


class NaiveGenerator(BaseGenerator):
    """Reserved baseline for comparing generation without KV cache."""

    def __init__(
        self,
        model: torch.nn.Module,
        sampler: Sampler | None = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.sampler = sampler or Sampler()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        raise NotImplementedError("NaiveGenerator is reserved for cache comparison.")


Generator = CachedGenerator


def generate_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    sampling_params: SamplingParams | None = None,
) -> torch.Tensor:
    sampler = Sampler(sampling_params)
    generator = CachedGenerator(model, sampler=sampler)
    return generator.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
    )
