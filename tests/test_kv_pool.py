import torch

from lmcore.inference.kv_pool import KVPool


def _make_pool(num_blocks: int = 10) -> KVPool:
    return KVPool(
        num_blocks=num_blocks,
        num_layers=2,
        num_heads=2,
        block_size=4,
        head_dim=4,
    )


# ── read / write correctness ──────────────────────────────────────

def test_write_and_read_same_location():
    pool = _make_pool()

    key = torch.ones(2, 4) * 0.5
    value = torch.ones(2, 4) * 0.8

    pool.write(layer=0, block_id=3, offset=1, key=key, value=value)
    k, v = pool.read(layer=0, block_id=3, offset=1)

    assert torch.equal(k, key)
    assert torch.equal(v, value)


def test_unwritten_location_is_zero():
    pool = _make_pool()
    k, v = pool.read(layer=1, block_id=5, offset=2)
    assert torch.equal(k, torch.zeros(2, 4))
    assert torch.equal(v, torch.zeros(2, 4))


def test_different_layers_independent():
    pool = _make_pool()
    key0 = torch.ones(2, 4)
    key1 = torch.ones(2, 4) * 2.0

    pool.write(layer=0, block_id=0, offset=0, key=key0, value=key0)
    pool.write(layer=1, block_id=0, offset=0, key=key1, value=key1)

    k0, _ = pool.read(layer=0, block_id=0, offset=0)
    k1, _ = pool.read(layer=1, block_id=0, offset=0)
    assert torch.equal(k0, key0)
    assert torch.equal(k1, key1)


def test_different_blocks_independent():
    pool = _make_pool()
    key_a = torch.ones(2, 4) * 3.0
    key_b = torch.ones(2, 4) * 7.0

    pool.write(layer=0, block_id=2, offset=0, key=key_a, value=key_a)
    pool.write(layer=0, block_id=9, offset=0, key=key_b, value=key_b)

    ka, _ = pool.read(layer=0, block_id=2, offset=0)
    kb, _ = pool.read(layer=0, block_id=9, offset=0)
    assert torch.equal(ka, key_a)
    assert torch.equal(kb, key_b)


def test_multiple_tokens_in_same_block():
    """Tokens at different offsets within the same block do not overwrite."""
    pool = _make_pool()
    k0 = torch.randn(2, 4)
    k1 = torch.randn(2, 4)
    k2 = torch.randn(2, 4)
    k3 = torch.randn(2, 4)

    pool.write(layer=0, block_id=0, offset=0, key=k0, value=k0)
    pool.write(layer=0, block_id=0, offset=1, key=k1, value=k1)
    pool.write(layer=0, block_id=0, offset=2, key=k2, value=k2)
    pool.write(layer=0, block_id=0, offset=3, key=k3, value=k3)

    r0, _ = pool.read(layer=0, block_id=0, offset=0)
    r1, _ = pool.read(layer=0, block_id=0, offset=1)
    r2, _ = pool.read(layer=0, block_id=0, offset=2)
    r3, _ = pool.read(layer=0, block_id=0, offset=3)

    assert torch.equal(r0, k0)
    assert torch.equal(r1, k1)
    assert torch.equal(r2, k2)
    assert torch.equal(r3, k3)


# ── shape ─────────────────────────────────────────────────────────

def test_pool_shape():
    pool = _make_pool(num_blocks=8)
    assert pool.pool.shape == (2, 2, 8, 2, 4, 4)
    # layers=2, K/V=2, blocks=8, heads=2, block_size=4, dim=4
