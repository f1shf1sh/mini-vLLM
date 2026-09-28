import torch

from llm.model.block import Block, FeedForward


def test_feedforward_output_shape() -> None:
    torch.manual_seed(0)
    ffwd = FeedForward(n_embd=8, dropout=0.0, bias=False)
    out = ffwd(torch.randn(2, 5, 8))
    assert out.shape == (2, 5, 8)


def test_block_output_shape() -> None:
    torch.manual_seed(0)
    block = Block(n_embd=8, n_head=2, block_size=6, dropout=0.0, bias=False)
    out = block(torch.randn(2, 5, 8))
    assert out.shape == (2, 5, 8)


def test_stacked_blocks_keep_shape() -> None:
    torch.manual_seed(0)
    blocks = torch.nn.ModuleList([
        Block(n_embd=8, n_head=2, block_size=6, dropout=0.0, bias=False),
        Block(n_embd=8, n_head=2, block_size=6, dropout=0.0, bias=False),
    ])
    x = torch.randn(2, 5, 8)
    for block in blocks:
        x = block(x)
    assert x.shape == (2, 5, 8)
