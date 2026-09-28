import pytest
import torch

from llm.model.attention import Head, MultiHeadAttention


def test_head_output_shape() -> None:
    torch.manual_seed(0)
    head = Head(n_embd=8, head_size=4, block_size=6, dropout=0.0, bias=False)
    out = head(torch.randn(2, 5, 8))
    assert out.shape == (2, 5, 4)


def test_head_causal_mask_blocks_future_tokens() -> None:
    torch.manual_seed(0)
    head = Head(n_embd=8, head_size=4, block_size=4, dropout=0.0, bias=False)
    x = torch.randn(1, 4, 8)
    x_future_changed = x.clone()
    x_future_changed[:, 2:, :] = torch.randn(1, 2, 8)
    out = head(x)
    out_future = head(x_future_changed)
    torch.testing.assert_close(out[:, :2, :], out_future[:, :2, :])


def test_multi_head_attention_output_shape() -> None:
    torch.manual_seed(0)
    attn = MultiHeadAttention(n_head=2, n_embd=8, block_size=6,
                              dropout=0.0, bias=False)
    out = attn(torch.randn(3, 5, 8))
    assert out.shape == (3, 5, 8)


def test_multi_head_attention_requires_divisible_dimensions() -> None:
    with pytest.raises(ValueError):
        MultiHeadAttention(n_head=3, n_embd=8, block_size=6,
                           dropout=0.0, bias=False)
