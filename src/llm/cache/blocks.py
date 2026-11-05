from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BlockTableSnapshot:
    """Immutable mapping data; page validity is checked at cache boundaries."""

    block_ids: tuple[int, ...]
    reserved_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.block_ids, tuple):
            raise TypeError("block_ids must be a tuple")


class BlockTable:
    '''
    if block_size == 4, the sequence of 10-tokens be cut at:
        logical block 0 -> tokens 0, 1, 2, 3
        logical block 1 -> tokens 4, 5, 6, 7
        logical block 2 -> tokens ...
    '''  
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self.physical_blocks: list[int] = []
        self.num_tokens: int = 0
        
    def num_blocks(self) -> int:
        '''
        the number of currently allocated physical blocks.
        ''' 
        return len(self.physical_blocks)
    
    def num_logical_blocks_for(self, n_tokens: int) -> int:
        '''
        return the number of logical blocks required for n_tokens tokens
        ''' 
        if n_tokens == 0:
            return 0
        return (n_tokens + self.block_size - 1) // self.block_size
    
    def append_block(self, physical_block_id: int) -> None:
        self.physical_blocks.append(physical_block_id)
        
    def locate(self, token_pos: int) -> tuple[int, int]:
        '''
        Return (physical_block_id, offset_in_block)
        TODO: Create example
        example token_pos = 6
        '''
        logical_block_id = token_pos // self.block_size
        offset = token_pos % self.block_size
        physical_block_id = self.physical_blocks[logical_block_id]
        
        return  physical_block_id, offset 

class BlockAllocator:
    '''
    global physical block allocator
    '''
    
    def __init__(self, num_blocks: int) -> None:
        self.free_blocks: deque[int] = deque(range(num_blocks))
        self.allocated: set[int] = set()
    
    def allocate(self) -> int:
        '''
        Return physical block id from free_blocks list
        And add it in set
        ''' 
        if not self.free_blocks:
            raise RuntimeError("No free blocks")
        
        physical_block_id = self.free_blocks.popleft()
        self.allocated.add(physical_block_id)
        
        return physical_block_id
    
    def free(self, block_id) -> None:
        '''
        Remove physical block from allocate set
        ''' 
        if block_id not in self.allocated:
            raise ValueError(
                f"Block {block_id} is not allocated"
            )
        self.allocated.remove(block_id)
        self.free_blocks.appendleft(block_id)
 
    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

class BlockManager:
    def __init__(self, allocator: BlockAllocator, block_size: int) -> None:
        self.allocator = allocator
        self.block_size = block_size
        self.block_tables: dict[str, BlockTable] = {}

    def allocate_for_request(self, request_id: str, num_tokens: int) -> None:
        if num_tokens < 0:
            raise ValueError("num_tokens must be nonnegative.")
        if request_id in self.block_tables:
            raise ValueError(
                f"request_id: {request_id} already exists in block_tables"
                )
        
        table = BlockTable(block_size=self.block_size)
        
        '''
        If num_tokens = 0, not execute next loop
        ''' 
        num_blocks = table.num_logical_blocks_for(num_tokens)

        try:
            for _ in range(num_blocks):
                table.append_block(self.allocator.allocate())

        except BaseException:
            for block_id in table.physical_blocks:
                self.allocator.free(block_id)

            raise

        table.num_tokens = num_tokens
        self.block_tables[request_id] = table 
        
    def append_token(self, request_id: str) -> None:
        table = self.block_tables[request_id]
        
        num_tokens = table.num_tokens
        if num_tokens % table.block_size == 0:
            table.append_block(self.allocator.allocate())
        
        table.num_tokens += 1

    def free_request(self, request_id: str) -> None:
        table = self.block_tables[request_id]
        
        for block_id in table.physical_blocks:
            self.allocator.free(block_id)
        
        del self.block_tables[request_id]
        
    def locate(self, request_id: str, token_pos: int) -> tuple[int, int]:
        table = self.block_tables[request_id]
        
        return table.locate(token_pos = token_pos)
