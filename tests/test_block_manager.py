import pytest

from llm.inference.block_manager import BlockTable, BlockAllocator

def test_block_table_empty():
    table = BlockTable(block_size=4)
    assert table.block_size == 4
    assert table.num_tokens == 0
    assert table.num_blocks() == 0
    assert table.physical_blocks == []

def test_num_logical_blocks():
    table = BlockTable(block_size=4)
    assert table.num_logical_blocks_for(0) == 0
    assert table.num_logical_blocks_for(1) == 1
    assert table.num_logical_blocks_for(4) == 1
    assert table.num_logical_blocks_for(5) == 2
    assert table.num_logical_blocks_for(8) == 2
    assert table.num_logical_blocks_for(9) == 3

def test_locate():
    table = BlockTable(block_size=4)
    table.physical_blocks = [7, 2, 9]

    # token 0 → 第0个逻辑块，第0个物理块(7)，偏移0
    assert table.locate(0) == (7, 0)
    # token 3 → 还在第0个逻辑块，偏移3
    assert table.locate(3) == (7, 3)
    # token 4 → 第1个逻辑块，物理块2，偏移0
    assert table.locate(4) == (2, 0)
    # token 6 → 第1个逻辑块，物理块2，偏移2
    assert table.locate(6) == (2, 2)
    # token 8 → 第2个逻辑块，物理块9，偏移0
    assert table.locate(8) == (9, 0)
 
def test_allocate_and_free():
    alloc = BlockAllocator(num_blocks=3)
    assert alloc.num_free == 3

    a = alloc.allocate()
    b = alloc.allocate()
    assert alloc.num_free == 1
    assert a != b

    alloc.free(a)
    assert alloc.num_free == 2

    # 释放后再分配，应该能复用
    c = alloc.allocate()
    assert c == a  # 复用刚释放的


def test_allocate_when_full():
    alloc = BlockAllocator(num_blocks=1)
    alloc.allocate()
    with pytest.raises(RuntimeError):
        alloc.allocate()


def test_free_unallocated():
    alloc = BlockAllocator(num_blocks=3)
    with pytest.raises(ValueError):
        alloc.free(999)


# ── BlockManager tests ──────────────────────────────────────────────

from llm.inference.block_manager import BlockManager


def _make_bm(num_blocks: int = 10, block_size: int = 4) -> BlockManager:
    return BlockManager(
        allocator=BlockAllocator(num_blocks=num_blocks),
        block_size=block_size,
    )


class TestBlockManagerAllocate:
    def test_creates_block_table_with_correct_blocks(self):
        bm = _make_bm(num_blocks=10)
        bm.allocate_for_request("req_A", num_tokens=10)

        table = bm.block_tables["req_A"]
        assert table.num_blocks() == 3          # ceil(10/4) = 3
        assert table.num_tokens == 10
        assert len(table.physical_blocks) == 3
        assert bm.allocator.num_free == 7       # 10 - 3

    def test_zero_tokens_allocates_no_blocks(self):
        bm = _make_bm(num_blocks=10)
        bm.allocate_for_request("req_A", num_tokens=0)

        table = bm.block_tables["req_A"]
        assert table.num_blocks() == 0
        assert table.num_tokens == 0
        assert bm.allocator.num_free == 10      # none consumed

    def test_duplicate_request_id_raises(self):
        bm = _make_bm(num_blocks=10)
        bm.allocate_for_request("req_A", num_tokens=5)

        with pytest.raises(ValueError, match="already exists"):
            bm.allocate_for_request("req_A", num_tokens=3)


class TestBlockManagerAppendToken:
    def test_within_block_does_not_allocate(self):
        bm = _make_bm(num_blocks=10)
        bm.allocate_for_request("req_A", num_tokens=3)  # 3 tokens, 1 block

        bm.append_token("req_A")  # token 3, still in block 0
        table = bm.block_tables["req_A"]
        assert table.num_tokens == 4
        assert table.num_blocks() == 1           # no new block
        assert bm.allocator.num_free == 9        # still only 1 used

    def test_at_block_boundary_allocates_new_block(self):
        bm = _make_bm(num_blocks=10)
        bm.allocate_for_request("req_A", num_tokens=4)  # 1 full block

        bm.append_token("req_A")  # token 4, needs new block
        table = bm.block_tables["req_A"]
        assert table.num_tokens == 5
        assert table.num_blocks() == 2           # new block allocated
        assert len(table.physical_blocks) == 2

    def test_multiple_boundary_crossings(self):
        bm = _make_bm(num_blocks=10)
        bm.allocate_for_request("req_A", num_tokens=2)  # 1 block, 2 used

        # generate 6 more tokens → cross boundaries at tokens 4 and 8
        for _ in range(6):
            bm.append_token("req_A")

        table = bm.block_tables["req_A"]
        assert table.num_tokens == 8
        assert table.num_blocks() == 2           # ceil(8/4) = 2
        assert bm.allocator.num_free == 8        # 2 used


class TestBlockManagerFreeRequest:
    def test_recycles_all_blocks(self):
        bm = _make_bm(num_blocks=10)
        bm.allocate_for_request("req_A", num_tokens=10)  # 3 blocks
        blocks_before = bm.allocator.num_free

        bm.free_request("req_A")
        assert bm.allocator.num_free == blocks_before + 3
        assert "req_A" not in bm.block_tables

    def test_nonexistent_request_raises_key_error(self):
        bm = _make_bm(num_blocks=10)
        with pytest.raises(KeyError):
            bm.free_request("ghost")


class TestBlockManagerLocate:
    def test_delegates_to_block_table(self):
        bm = _make_bm(num_blocks=10)
        bm.allocate_for_request("req_A", num_tokens=10)

        # Same semantics as BlockTable.locate
        assert bm.locate("req_A", token_pos=0) == (
            bm.block_tables["req_A"].physical_blocks[0], 0)
        assert bm.locate("req_A", token_pos=5) == (
            bm.block_tables["req_A"].physical_blocks[1], 1)


class TestBlockManagerEndToEnd:
    def test_multiple_requests_interleaved(self):
        bm = _make_bm(num_blocks=20)

        bm.allocate_for_request("req_A", num_tokens=6)   # 2 blocks
        bm.allocate_for_request("req_B", num_tokens=10)  # 3 blocks
        assert bm.allocator.num_free == 15               # 5 used

        # Different requests have disjoint physical blocks
        a_blocks = set(bm.block_tables["req_A"].physical_blocks)
        b_blocks = set(bm.block_tables["req_B"].physical_blocks)
        assert a_blocks.isdisjoint(b_blocks)

        # Free req_A, blocks return to pool
        bm.free_request("req_A")
        assert bm.allocator.num_free == 17

    def test_empty_prompt_grows_from_zero(self):
        """A request with 0 prompt tokens allocates its first block on decode."""
        bm = _make_bm(num_blocks=10)
        bm.allocate_for_request("req_A", num_tokens=0)

        assert bm.block_tables["req_A"].num_blocks() == 0

        bm.append_token("req_A")  # token 0 → allocates block 0
        assert bm.block_tables["req_A"].num_blocks() == 1
        assert bm.allocator.num_free == 9
