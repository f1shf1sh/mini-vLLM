import pytest
import torch

from toy.config import ModelConfig
from toy.model import GPTMini


def make_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=8,
        dropout=0.0, bias=False,
    )


def test_model_forward_output_shape_and_loss() -> None:
    torch.manual_seed(0)
    model = GPTMini(make_model_config())
    idx = torch.randint(0, 11, (2, 5))
    targets = torch.randint(0, 11, (2, 5))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 5, 11)
    assert loss is not None
    assert loss.ndim == 0


def test_model_forward_rejects_long_sequences() -> None:
    model = GPTMini(make_model_config())
    with pytest.raises(ValueError):
        model(torch.randint(0, 11, (2, 9)))


def test_model_forward_without_targets() -> None:
    torch.manual_seed(0)
    model = GPTMini(make_model_config())
    logits, loss = model(torch.randint(0, 11, (2, 5)))
    assert logits.shape == (2, 5, 11)
    assert loss is None
