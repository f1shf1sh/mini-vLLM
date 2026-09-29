"""Direct GPTMini generation without request scheduling or paged KV storage."""
from __future__ import annotations

import torch

from llm.inference.sampler import Sampler, SamplingParams


class Generator:
    """Run full-prefix model forward and reuse the shared stateless sampler."""

    def __init__(
        self,
        model: torch.nn.Module,
        sampler: Sampler | None = None,
    ) -> None:
        self.model = model.eval()
        self.sampler = sampler or Sampler()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.size(0) != 1:
            raise ValueError("generate expects input_ids with shape [1, T].")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative.")
        if max_new_tokens == 0:
            return input_ids
        if input_ids.size(1) == 0:
            raise ValueError("generate expects at least one prompt token.")
        forward_tokens = input_ids.size(1) + max_new_tokens - 1
        if forward_tokens > self.model.config.block_size:
            raise ValueError(
                f"Generation requires {forward_tokens} forward positions, exceeding "
                f"block_size={self.model.config.block_size}."
            )
        output_ids = input_ids
        for _ in range(max_new_tokens):
            logits, _ = self.model(output_ids)
            output_ids = torch.cat((output_ids, self.sampler.sample(logits)), dim=1)
        return output_ids


def generate_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    sampling_params: SamplingParams | None = None,
) -> torch.Tensor:
    return Generator(model, Sampler(sampling_params)).generate(input_ids, max_new_tokens)
