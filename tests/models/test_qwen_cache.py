from types import SimpleNamespace

import pytest
import torch

from llm.models.qwen.cache import make_qwen_kv_pool
from llm.cache.pool import KVPool


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA unavailable",
        ),
    ),
])
def test_qwen_pool_uses_kv_heads_and_requested_device(device):
    config = SimpleNamespace(
        num_hidden_layers=24,
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )
    pool = make_qwen_kv_pool(
        config,
        num_blocks=8,
        block_size=4,
        device=device,
        dtype=torch.float32,
    )

    assert isinstance(pool, KVPool)
    assert pool.pool.shape == (24, 2, 8, 2, 4, 64)
    assert pool.pool.dtype == torch.float32
    assert pool.pool.device.type == device


def test_pool_dimensions_and_dtype_follow_arguments():
    config = SimpleNamespace(
        num_hidden_layers=3,
        hidden_size=48,
        num_attention_heads=6,
        num_key_value_heads=2,
    )
    pool = make_qwen_kv_pool(
        config,
        num_blocks=3,
        block_size=5,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert isinstance(pool, KVPool)
    assert pool.pool.shape == (3, 2, 3, 2, 5, 8)
    assert pool.pool.dtype == torch.float64
    assert pool.pool.device.type == "cpu"
