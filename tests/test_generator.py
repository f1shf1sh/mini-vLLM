import torch

from lmcore.config import ModelConfig
from lmcore.inference.generator import CachedGenerator, Generator, generate_tokens
from lmcore.inference.sampler import SamplingParams
from lmcore.model import GPTMini


def make_model() -> GPTMini:
    config = ModelConfig(
        vocab_size=16,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=False,
    )
    model = GPTMini(config)
    model.eval()
    return model


def test_generator_appends_new_tokens() -> None:
    torch.manual_seed(0)
    generator = Generator(make_model())
    input_ids = torch.tensor([[1, 2, 3]])

    output_ids = generator.generate(input_ids, max_new_tokens=2)

    assert output_ids.shape == (1, 5)
    assert output_ids[:, :3].tolist() == [[1, 2, 3]]


def test_generator_defaults_to_cached_generator() -> None:
    assert Generator is CachedGenerator


def test_generate_tokens_uses_sampling_params() -> None:
    torch.manual_seed(0)
    model = make_model()
    input_ids = torch.tensor([[1, 2, 3]])

    output_ids = generate_tokens(
        model=model,
        input_ids=input_ids,
        max_new_tokens=2,
        sampling_params=SamplingParams(temperature=1.0, top_k=1),
    )

    assert output_ids.shape == (1, 5)
