import pytest
import torch

from llm.config import ModelConfig
from llm.inference.generator import CachedGenerator, generate_tokens
from llm.inference.sampler import SamplingParams
from llm.model import GPTMini


def make_model(device="cpu"):
    torch.manual_seed(17)
    return GPTMini(ModelConfig(
        vocab_size=16, block_size=32, n_layer=2, n_head=2, n_embd=8,
        dropout=0.0, bias=False,
    )).to(device).eval()


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
)])
def test_paged_generation_matches_full_forward_across_block_boundary(device):
    model = make_model(device)
    prompt = torch.tensor([[i % 16 for i in range(15)]], device=device)
    expected = prompt.clone()
    with torch.no_grad():
        for _ in range(5):
            logits, _ = model(expected)
            expected = torch.cat((expected, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
    actual = generate_tokens(model, prompt, 5, SamplingParams(top_k=1))
    assert torch.equal(actual, expected)
    assert actual.device == prompt.device


def test_final_sample_does_not_require_an_extra_context_position():
    model = make_model()
    prompt = torch.tensor([[1] * 32])
    actual = generate_tokens(model, prompt, 1, SamplingParams(top_k=1))
    assert actual.shape == (1, 33)


def test_generator_can_be_reused():
    generator = CachedGenerator(make_model())
    prompt = torch.tensor([[1, 2]])
    assert generator.generate(prompt, 2).shape == (1, 4)
    assert generator.generate(prompt, 3).shape == (1, 5)


def test_zero_tokens_preserves_input():
    prompt = torch.tensor([[1, 2]])
    assert CachedGenerator(make_model()).generate(prompt, 0) is prompt


@pytest.mark.parametrize("prompt,max_new_tokens,match", [
    (torch.tensor([[1, 2]]), -1, "nonnegative"),
    (torch.empty(1, 0, dtype=torch.long), 1, "prompt token"),
    (torch.tensor([[1], [2]]), 1, "shape"),
    (torch.tensor([[1] * 32]), 2, "exceeding"),
])
def test_generator_rejects_invalid_requests(prompt, max_new_tokens, match):
    with pytest.raises(ValueError, match=match):
        CachedGenerator(make_model()).generate(prompt, max_new_tokens)
