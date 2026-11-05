"""Direct GPTMini generation without request scheduling or paged KV storage."""
from __future__ import annotations

import torch

from llm.inference.sampler import Sampler, SamplingParams


class Generator:
    """Run full-prefix model forward and reuse the shared sampler."""

    def __init__(
        self,
        model: torch.nn.Module,
        sampler: Sampler | None = None,
    ) -> None:
        self.model = model.eval()
        if sampler is None:
            model_device = next(self.model.parameters()).device
            sampler = Sampler(device=model_device)
        self.sampler = sampler

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
            next_token_logits = logits[:, -1, :]
            next_token = self.sampler.sample(next_token_logits)
            output_ids = torch.cat((output_ids, next_token), dim=1)
        return output_ids


def generate_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    sampling_params: SamplingParams | None = None,
) -> torch.Tensor:
    sampler = Sampler(sampling_params, device=input_ids.device)
    return Generator(model, sampler).generate(input_ids, max_new_tokens)
