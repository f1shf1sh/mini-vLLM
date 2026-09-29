"""Map pretrained model configuration to the project's KV storage."""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache

from llm.cache.pool import KVPool
from llm.cache.paged import PagedKVCache

class QwenPagedCache(Cache):
    def __init__(
            self,
            paged_cache: PagedKVCache,
            num_layers: int,
            past_length: int = 0,
    ):
        super().__init__()

        self.paged_cache = paged_cache
        if past_length < 0 or past_length > paged_cache.num_tokens:
            raise ValueError("Invalid past cache length.")
        self.lengths = [past_length] * num_layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.lengths[layer_idx]

    def get_max_cache_shape(self):
        return None

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs=None,
    ) -> tuple:
        if key_states.ndim != 4 or key_states.shape[0] != 1:
            raise ValueError("Qwen cache expects shape [1, kv_heads, tokens, head_dim].")
        if value_states.shape != key_states.shape:
            raise ValueError("Key and value shapes must match.")
        token_pos = self.lengths[layer_idx]
        if token_pos + key_states.shape[2] > self.paged_cache.num_tokens:
            raise ValueError("Scheduler must reserve KV slots before forward.")
        self.paged_cache.write_range(layer_idx, key_states, value_states, start_pos=token_pos)

        new_token_pos = token_pos + key_states.shape[2]
        self.lengths[layer_idx] = new_token_pos

        return self.paged_cache.gather(layer_idx, new_token_pos)


def make_qwen_kv_pool(
    config,
    num_blocks: int,
    block_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> KVPool:
    """Create a pool from Qwen2-style config attributes and storage options."""

    if num_blocks < 1 or block_size < 1:
        raise ValueError("num_blocks and block_size must be positive.")
    num_hidden_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    head_dim = hidden_size // num_attention_heads

    return KVPool(
        num_blocks=num_blocks,
        num_layers=num_hidden_layers,
        num_heads=num_key_value_heads,
        block_size=block_size,
        head_dim=head_dim,
        device=device,
        dtype=dtype
    )
