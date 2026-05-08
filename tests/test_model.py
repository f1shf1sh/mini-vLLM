import pytest
import torch

from lmcore.config import ModelConfig
from lmcore.model import GPTMini


def make_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=11,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=False,
    )


def test_model_forward_output_shape_and_loss() -> None:
    torch.manual_seed(0)
    model = GPTMini(make_model_config())
    idx = torch.randint(0, 11, (2, 5))
    targets = torch.randint(0, 11, (2, 5))

    logits, loss, _ = model(idx, targets)

    assert logits.shape == (2, 5, 11)
    assert loss is not None
    assert loss.ndim == 0


def test_model_forward_rejects_long_sequences() -> None:
    model = GPTMini(make_model_config())
    idx = torch.randint(0, 11, (2, 9))

    with pytest.raises(ValueError):
        model(idx)


def test_model_generate_appends_new_tokens() -> None:
    torch.manual_seed(0)
    model = GPTMini(make_model_config())
    idx = torch.randint(0, 11, (2, 3))

    out = model.generate(idx, max_new_tokens=4, temperature=1.0, top_k=3)

    assert out.shape == (2, 7)


def test_model_generate_rejects_non_positive_temperature() -> None:
    model = GPTMini(make_model_config())
    idx = torch.randint(0, 11, (1, 3))

    with pytest.raises(ValueError):
        model.generate(idx, max_new_tokens=2, temperature=0.0)
