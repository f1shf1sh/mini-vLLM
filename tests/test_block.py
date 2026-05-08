import torch

from lmcore.model.block import Block, FeedForward


def test_feedforward_output_shape() -> None:
    torch.manual_seed(0)
    ffwd = FeedForward(n_embd=8, dropout=0.0, bias=False)
    x = torch.randn(2, 5, 8)

    out = ffwd(x)

    assert out.shape == (2, 5, 8)


def test_block_output_shape() -> None:
    torch.manual_seed(0)
    block = Block(
        n_embd=8,
        n_head=2,
        block_size=6,
        dropout=0.0,
        bias=False,
    )
    x = torch.randn(2, 5, 8)

    out, present = block(x)

    assert out.shape == (2, 5, 8)
    assert present is None


def test_stacked_blocks_keep_shape() -> None:
    torch.manual_seed(0)
    blocks = torch.nn.ModuleList([
        Block(n_embd=8, n_head=2, block_size=6, dropout=0.0, bias=False),
        Block(n_embd=8, n_head=2, block_size=6, dropout=0.0, bias=False),
    ])
    x = torch.randn(2, 5, 8)

    out = x
    for block in blocks:
        out, present = block(out)
        assert present is None

    assert out.shape == (2, 5, 8)
