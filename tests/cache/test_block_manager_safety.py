"""Resource ownership regressions for the scheduler's block manager."""

import pytest

from llm.cache.blocks import BlockAllocator, BlockManager


def snapshot(manager):
    return (
        tuple(sorted(manager.allocator.free_blocks)),
        frozenset(manager.allocator.allocated),
        {
            request_id: (table.num_tokens, tuple(table.physical_blocks))
            for request_id, table in manager.block_tables.items()
        },
    )


def assert_ownership(manager, total):
    free = list(manager.allocator.free_blocks)
    owned = [
        block_id
        for table in manager.block_tables.values()
        for block_id in table.physical_blocks
    ]
    assert len(free) == len(set(free)), "Duplicate block in free list"
    assert len(owned) == len(set(owned)), "A block has multiple owners"
    assert set(owned) == manager.allocator.allocated, "Untracked allocated block"
    assert set(free).isdisjoint(owned)
    assert set(free) | set(owned) == set(range(total))


@pytest.mark.parametrize("free_blocks", [0, 1, 2])
def test_refused_allocation_preserves_existing_request(free_blocks):
    manager = BlockManager(BlockAllocator(free_blocks + 1), block_size=4)
    manager.allocate_for_request("live", 4)
    live_table = manager.block_tables["live"]
    before = snapshot(manager)

    with pytest.raises(RuntimeError):
        manager.allocate_for_request("new", 9)  # Needs three blocks.

    assert snapshot(manager) == before, "Failed allocation leaked blocks"
    assert manager.block_tables["live"] is live_table
    assert_ownership(manager, free_blocks + 1)

    # Refusal must not poison the ID or prevent a later retry.
    manager.free_request("live")
    manager.allocate_for_request("new", 4)
    manager.free_request("new")
    assert_ownership(manager, free_blocks + 1)
    assert manager.allocator.num_free == free_blocks + 1


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_mid_allocation_exception_rolls_back_only_new_blocks(monkeypatch, failure_type):
    manager = BlockManager(BlockAllocator(4), block_size=4)
    manager.allocate_for_request("live", 4)
    before = snapshot(manager)
    original_allocate = manager.allocator.allocate
    calls = 0

    def fail_on_second_allocation():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure_type("injected allocation failure")
        return original_allocate()

    monkeypatch.setattr(manager.allocator, "allocate", fail_on_second_allocation)
    with pytest.raises(failure_type, match="injected allocation failure"):
        manager.allocate_for_request("new", 8)

    assert calls == 2
    assert snapshot(manager) == before, "Rollback must preserve other requests"
    assert_ownership(manager, 4)
    monkeypatch.setattr(manager.allocator, "allocate", original_allocate)
    manager.allocate_for_request("new", 8)
    manager.free_request("new")
    assert snapshot(manager) == before
    manager.free_request("live")
    assert manager.allocator.num_free == 4
    assert_ownership(manager, 4)


def test_failed_append_preserves_length_and_can_retry():
    manager = BlockManager(BlockAllocator(2), block_size=4)
    manager.allocate_for_request("live", 4)
    manager.allocate_for_request("other", 4)
    before = snapshot(manager)

    with pytest.raises(RuntimeError):
        manager.append_token("live")

    assert snapshot(manager) == before
    assert_ownership(manager, 2)
    manager.free_request("other")
    manager.append_token("live")
    assert manager.block_tables["live"].num_tokens == 5
    assert len(manager.block_tables["live"].physical_blocks) == 2
    manager.free_request("live")
    assert manager.allocator.num_free == 2
    assert_ownership(manager, 2)


def test_duplicate_free_does_not_duplicate_available_blocks():
    manager = BlockManager(BlockAllocator(3), block_size=4)
    manager.allocate_for_request("done", 4)
    manager.allocate_for_request("live", 4)
    released = manager.block_tables["done"].physical_blocks[0]
    manager.free_request("done")
    before = snapshot(manager)

    with pytest.raises(KeyError):
        manager.free_request("done")
    assert snapshot(manager) == before

    with pytest.raises(ValueError):
        manager.allocator.free(released)
    assert snapshot(manager) == before
    assert_ownership(manager, 3)

    manager.allocate_for_request("next", 8)
    assert_ownership(manager, 3)
    manager.free_request("next")
    manager.free_request("live")
    assert manager.allocator.num_free == 3
    assert_ownership(manager, 3)


def test_repeated_mixed_length_rounds_restore_all_blocks():
    manager = BlockManager(BlockAllocator(4), block_size=4)
    initial = snapshot(manager)

    for _ in range(8):
        for request_id, length in (("a", 1), ("b", 4), ("c", 5)):
            manager.allocate_for_request(request_id, length)
            assert_ownership(manager, 4)
        assert manager.allocator.num_free == 0

        for request_id in ("b", "a", "c"):
            manager.free_request(request_id)
            assert_ownership(manager, 4)
        assert snapshot(manager) == initial
        assert manager.allocator.num_free == 4


@pytest.mark.parametrize("num_tokens", [-1, -4])
def test_negative_allocation_preserves_live_request(num_tokens):
    manager = BlockManager(BlockAllocator(4), block_size=4)
    manager.allocate_for_request("live", 4)
    blocks = list(manager.block_tables["live"].physical_blocks)
    free = manager.allocator.num_free
    with pytest.raises(ValueError, match="nonnegative"):
        manager.allocate_for_request("invalid", num_tokens)
    assert manager.allocator.num_free == free
    assert set(manager.block_tables) == {"live"}
    assert manager.block_tables["live"].physical_blocks == blocks
