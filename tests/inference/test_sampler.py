import torch

from llm.inference.sampler import Sampler, SamplingParams


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


def test_greedy_is_deterministic_without_consuming_rng():
    logits = torch.tensor([[[0.0, 3.0, 1.0]]])
    sampler = Sampler(SamplingParams(do_sample=False))
    before = torch.random.get_rng_state()
    assert sampler.sample(logits).item() == 1
    assert torch.equal(before, torch.random.get_rng_state())


def test_nucleus_sampling_keeps_threshold_crossing_token():
    # Probabilities 0.6, 0.3, 0.1: p=0.8 must retain the first TWO tokens.
    logits = torch.log(torch.tensor([[[0.6, 0.3, 0.1]]])).expand(512, -1, -1)
    torch.manual_seed(42)
    tokens = Sampler(SamplingParams(top_p=0.8)).sample(logits)
    assert set(tokens.flatten().tolist()) == {0, 1}


def test_invalid_sampling_parameters_are_rejected():
    import pytest
    for params in ({"temperature": 0}, {"temperature": float("nan")}, {"top_k": 0}, {"top_p": 0}, {"top_p": 1.1}):
        with pytest.raises(ValueError):
            SamplingParams(**params)
