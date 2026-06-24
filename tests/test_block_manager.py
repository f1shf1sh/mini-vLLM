import pytest

from lmcore.inference.block_manager import BlockTable, BlockAllocator

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
