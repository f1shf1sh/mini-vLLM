import torch

from lmcore.config import ModelConfig
from lmcore.inference.model_runner import ModelRunner
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


def test_prefill_builds_cache_for_full_prompt() -> None:
    runner = ModelRunner(make_model())
    input_ids = torch.tensor([[1, 2, 3]])

    output = runner.prefill(input_ids)

    assert output.logits.shape == (1, 3, runner.model.config.vocab_size)
    assert len(output.kv_cache) == runner.model.config.n_layer
    key, value = output.kv_cache[0]
    assert key.shape == (1, runner.model.config.n_head, 3, 4)
    assert value.shape == (1, runner.model.config.n_head, 3, 4)


def test_decode_extends_cache_by_one_token() -> None:
    runner = ModelRunner(make_model())
    prefill_output = runner.prefill(torch.tensor([[1, 2, 3]]))

    decode_output = runner.decode(torch.tensor([[4]]), prefill_output.kv_cache)

    assert decode_output.logits.shape == (1, 1, runner.model.config.vocab_size)
    key, value = decode_output.kv_cache[0]
    assert key.shape == (1, runner.model.config.n_head, 4, 4)
    assert value.shape == (1, runner.model.config.n_head, 4, 4)
