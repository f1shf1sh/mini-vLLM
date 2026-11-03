"""Isolated tests for QwenBatchedPrefillCache.update.

The constructor and get_seq_length remain learning TODOs. Fixtures install one
explicit, fixed metadata state to exercise update only; they do not supply a
replacement constructor or claim end-to-end batch-prefill coverage.
"""
from unittest.mock import patch

import pytest
import torch
from transformers.cache_utils import Cache

from llm.cache.blocks import BlockAllocator, BlockManager
from llm.cache.paged import PagedCacheResources
from llm.cache.pool import KVPool
from llm.model.qwen2_5_0_5b_instruct.cache import QwenBatchedPrefillCache


@pytest.fixture
def update_case(request):
    device = getattr(request, "param", "cpu")
    pool = KVPool(
        num_blocks=12,
        num_layers=2,
        num_heads=2,
        block_size=2,
        head_dim=8,
        device=device,
        dtype=torch.float32,
    )
    pool.pool.fill_(37.25)
    resources = PagedCacheResources(pool, BlockManager(BlockAllocator(12), 2))
    resources.allocate("short", 2)   # Physical page 0.
    resources.allocate("idle", 2)    # Physical page 1, never participating.
    resources.allocate("long", 2)    # Physical page 2.
    resources.allocate("single", 1)  # Physical page 3, with an unused tail.
    resources.append("long")
    resources.append("long")
    resources.append("long")        # Noncontiguous long pages: 2, 4, 5.

    # Bypass only the unfinished constructor. Metadata is a literal fixture,
    # not a general implementation that computes the constructor's state.
    adapter = object.__new__(QwenBatchedPrefillCache)
    Cache.__init__(adapter)
    adapter.views = (
        resources.view("long"),
        resources.view("single"),
        resources.view("short"),
    )
    adapter.valid_lengths = (5, 1, 2)
    adapter.max_length = 5
    adapter.written_lengths = [[0, 0, 0], [0, 0, 0]]

    keys = torch.arange(240, dtype=torch.float32, device=device).reshape(3, 2, 5, 8)
    values = keys + 1000
    try:
        yield adapter, resources, keys, values
    finally:
        # Direct cache tests own cleanup; this does not test Scheduler cleanup.
        for request_id in list(resources.block_manager.block_tables):
            resources.release(request_id)
        allocator = resources.block_manager.allocator
        assert not resources.block_manager.block_tables
        assert not allocator.allocated
        assert sorted(allocator.free_blocks) == list(range(12))


def resource_snapshot(resources):
    tables = {}
    for request_id, table in resources.block_manager.block_tables.items():
        tables[request_id] = (table.num_tokens, tuple(table.physical_blocks))
    allocator = resources.block_manager.allocator
    return tables, set(allocator.allocated), list(allocator.free_blocks)


# Fixed physical addresses independently specify the fixture's expected writes.
# Each tuple is (batch row, logical token, physical block, offset within block).
EXPECTED_WRITES = (
    (0, 0, 2, 0),
    (0, 1, 2, 1),
    (0, 2, 4, 0),
    (0, 3, 4, 1),
    (0, 4, 5, 0),
    (1, 0, 3, 0),
    (2, 0, 0, 0),
    (2, 1, 0, 1),
)


@pytest.mark.parametrize("update_case", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable",
    )),
], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 1])
def test_real_prefixes_cross_blocks_and_padding_never_touches_pages(update_case, layer_idx):
    adapter, resources, keys, values = update_case
    before_pool = resources.pool.pool.clone()
    before_resources = resource_snapshot(resources)

    # Invalid input padding must not contaminate either persistent or scratch KV.
    keys[1, :, 1:, :] = float("nan")
    values[1, :, 1:, :] = float("nan")
    keys[2, :, 2:, :] = float("nan")
    values[2, :, 2:, :] = float("nan")
    actual_keys, actual_values = adapter.update(
        keys, values, layer_idx,
        cache_kwargs={"cache_position": torch.arange(5, device=keys.device)},
    )

    assert actual_keys.shape == actual_values.shape == (3, 2, 5, 8)
    assert actual_keys.dtype == keys.dtype
    assert actual_keys.device == keys.device
    expected_keys = torch.zeros_like(keys)
    expected_values = torch.zeros_like(values)
    expected_keys[0] = keys[0]
    expected_values[0] = values[0]
    expected_keys[1, :, :1, :] = keys[1, :, :1, :]
    expected_values[1, :, :1, :] = values[1, :, :1, :]
    expected_keys[2, :, :2, :] = keys[2, :, :2, :]
    expected_values[2, :, :2, :] = values[2, :, :2, :]
    torch.testing.assert_close(actual_keys, expected_keys, rtol=0, atol=0)
    torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)

    expected_pool = before_pool.clone()
    for row, token, block, offset in EXPECTED_WRITES:
        expected_pool[layer_idx, 0, block, :, offset, :] = keys[row, :, token, :]
        expected_pool[layer_idx, 1, block, :, offset, :] = values[row, :, token, :]
    # This covers every page, untouched layers, idle owners, and unused tails.
    torch.testing.assert_close(resources.pool.pool, expected_pool, rtol=0, atol=0)
    assert resource_snapshot(resources) == before_resources
    assert adapter.written_lengths[layer_idx] == [5, 1, 2]
    assert adapter.written_lengths[1 - layer_idx] == [0, 0, 0]

    # Attention scratch storage must not alias the input or persistent pool.
    actual_keys.fill_(-999)
    actual_values.fill_(-999)
    torch.testing.assert_close(resources.pool.pool, expected_pool, rtol=0, atol=0)
    assert keys[0, 0, 0, 0].item() == 0
    assert values[0, 0, 0, 0].item() == 1000


def test_layers_have_independent_progress_and_repeated_layer_is_rejected(update_case):
    adapter, resources, keys, values = update_case
    adapter.update(keys, values, 1)
    assert adapter.written_lengths == [[0, 0, 0], [5, 1, 2]]
    adapter.update(keys + 10, values + 10, 0)
    assert adapter.written_lengths == [[5, 1, 2], [5, 1, 2]]
    before_pool = resources.pool.pool.clone()
    with pytest.raises(ValueError, match="exactly once"):
        adapter.update(keys, values, 1)
    torch.testing.assert_close(resources.pool.pool, before_pool, rtol=0, atol=0)


@pytest.mark.parametrize("invalid_case", [
    "key_rank", "value_shape", "empty_batch", "batch_size", "padded_length",
    "heads", "head_dim", "kv_dtype", "pool_dtype", "kv_device", "pool_device",
    "negative_layer", "large_layer", "bool_layer", "float_layer",
    "later_reservation", "later_view_shape", "later_pool_layer",
])
def test_invalid_update_is_rejected_before_any_row_is_written(update_case, invalid_case):
    adapter, resources, keys, values = update_case
    layer_idx = 0
    if invalid_case == "key_rank":
        keys = keys[0]
    elif invalid_case == "value_shape":
        values = values[:, :, :-1, :]
    elif invalid_case == "empty_batch":
        keys, values = keys[:0], values[:0]
    elif invalid_case == "batch_size":
        keys, values = keys[:2], values[:2]
    elif invalid_case == "padded_length":
        keys, values = keys[:, :, :4], values[:, :, :4]
    elif invalid_case == "heads":
        keys, values = keys[:, :1], values[:, :1]
    elif invalid_case == "head_dim":
        keys, values = keys[..., :7], values[..., :7]
    elif invalid_case == "kv_dtype":
        values = values.double()
    elif invalid_case == "pool_dtype":
        keys, values = keys.double(), values.double()
    elif invalid_case == "kv_device":
        values = values.to("meta")
    elif invalid_case == "pool_device":
        keys, values = keys.to("meta"), values.to("meta")
    elif invalid_case == "negative_layer":
        layer_idx = -1
    elif invalid_case == "large_layer":
        layer_idx = 2
    elif invalid_case == "bool_layer":
        layer_idx = True
    elif invalid_case == "float_layer":
        layer_idx = 0.0
    elif invalid_case == "later_reservation":
        resources.append("short")
    elif invalid_case == "later_view_shape":
        adapter.views[2]._n_head = 1
    elif invalid_case == "later_pool_layer":
        adapter.views[2]._pool = KVPool(
            num_blocks=12, num_layers=1, num_heads=2, block_size=2, head_dim=8,
            dtype=torch.float32,
        )
        layer_idx = 1

    before_pool = resources.pool.pool.clone()
    before_resources = resource_snapshot(resources)
    with patch.object(adapter.views[0], "write_range") as first_write:
        with pytest.raises(ValueError):
            adapter.update(keys, values, layer_idx)
    first_write.assert_not_called()
    torch.testing.assert_close(resources.pool.pool, before_pool, rtol=0, atol=0)
    assert adapter.written_lengths == [[0, 0, 0], [0, 0, 0]]
    assert resource_snapshot(resources) == before_resources


@pytest.mark.parametrize("failure_stage", ["write", "gather"])
def test_partial_failure_propagates_without_releasing_or_retrying(update_case, failure_stage):
    adapter, resources, keys, values = update_case
    before_pool = resources.pool.pool.clone()
    before_resources = resource_snapshot(resources)
    later_view = adapter.views[1]

    def fail_during_write(layer_idx, row_keys, row_values, start_pos=0):
        # Simulate a backend failure after a real persistent write.
        later_view.write(layer_idx, start_pos, row_keys[0, :, 0], row_values[0, :, 0])
        raise RuntimeError("injected write failure")

    method_name = "write_range" if failure_stage == "write" else "gather"
    failure = fail_during_write if failure_stage == "write" else RuntimeError("injected gather failure")
    with patch.object(later_view, method_name, side_effect=failure) as failing_call:
        with patch.object(resources, "release", side_effect=AssertionError("Caller owns release")):
            with pytest.raises(RuntimeError, match=f"injected {failure_stage} failure"):
                adapter.update(keys, values, 0)
    assert failing_call.call_count == 1
    assert resource_snapshot(resources) == before_resources

    expected_pool = before_pool.clone()
    for row, token, block, offset in EXPECTED_WRITES:
        if row < 2:
            expected_pool[0, 0, block, :, offset, :] = keys[row, :, token, :]
            expected_pool[0, 1, block, :, offset, :] = values[row, :, token, :]
    torch.testing.assert_close(resources.pool.pool, expected_pool, rtol=0, atol=0)
    if failure_stage == "write":
        # A failed write_range cannot certify that the whole prefix was written.
        assert adapter.written_lengths[0] == [5, 0, 0]
    else:
        assert adapter.written_lengths[0] == [5, 1, 0]
    assert adapter.written_lengths[1] == [0, 0, 0]
    # No recovery is attempted here. The fixture releases all requests, as the
    # caller must after a failed forward with possibly half-updated KV.
