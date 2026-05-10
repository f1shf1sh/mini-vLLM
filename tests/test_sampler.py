import torch

from lmcore.inference.sampler import Sampler, SamplingParams


def test_sampler_returns_one_token_per_batch_item() -> None:
    torch.manual_seed(0)
    sampler = Sampler(SamplingParams(temperature=1.0, top_k=None))
    logits = torch.randn(2, 3, 5)

    next_token = sampler.sample(logits)

    assert next_token.shape == (2, 1)
    assert next_token.dtype == torch.long


def test_sampler_top_k_one_selects_highest_logit() -> None:
    torch.manual_seed(0)
    sampler = Sampler(SamplingParams(temperature=1.0, top_k=1))
    logits = torch.tensor(
        [
            [[0.0, 1.0, 5.0, 2.0]],
            [[3.0, 7.0, 1.0, 0.0]],
        ]
    )

    next_token = sampler.sample(logits)

    assert next_token.tolist() == [[2], [1]]
