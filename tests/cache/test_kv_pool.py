import torch
import pytest

from llm.cache.blocks import BlockAllocator, BlockManager
from llm.cache.pool import KVPool
from llm.cache.paged import PagedKVCache


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


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
)])
def test_gather_preserves_pool_device_dtype_and_values(device):
    pool = KVPool(2, 1, 2, 4, 4, device=device, dtype=torch.float64)
    manager = BlockManager(BlockAllocator(2), 4)
    manager.allocate_for_request("r", 5)
    cache = PagedKVCache(pool, manager.block_tables["r"], 2, 4)
    expected_k = torch.randn(1, 2, 5, 4, device=device, dtype=torch.float64)
    expected_v = torch.randn_like(expected_k)
    cache.write_range(0, expected_k, expected_v)
    for count in (0, 5):
        k, v = cache.gather(0, count)
        assert k.device == expected_k.device
        assert v.device == expected_v.device
        assert k.dtype == v.dtype == torch.float64
        assert torch.equal(k, expected_k[:, :, :count])
        assert torch.equal(v, expected_v[:, :, :count])


def test_paged_resources_share_one_block_lifecycle():
    from llm.cache.paged import PagedCacheResources
    pool = KVPool(2, 1, 2, 2, 3)
    manager = BlockManager(BlockAllocator(2), 2)
    cache = PagedCacheResources(pool, manager)
    cache.allocate("request", 2)
    view = cache.view("request")
    assert cache.required_capacity(3) == 2
    cache.append("request")
    assert view.num_tokens == 3
    key = torch.randn(2, 3)
    value = torch.randn(2, 3)
    view.write(0, 2, key, value)
    actual_key, actual_value = view.read(0, 2)
    torch.testing.assert_close(actual_key, key)
    torch.testing.assert_close(actual_value, value)
    cache.release("request")
    cache.release("request")
    assert manager.allocator.num_free == cache.capacity == 2
    assert not manager.block_tables


def test_paged_resources_reject_mismatched_storage_and_allocator():
    import pytest
    from llm.cache.paged import PagedCacheResources
    with pytest.raises(ValueError, match="capacities"):
        PagedCacheResources(KVPool(2, 1, 1, 2, 3), BlockManager(BlockAllocator(1), 2))
