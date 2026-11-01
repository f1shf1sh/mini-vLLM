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



class QwenBatchedDecodeCache(Cache):
    """One-forward view over independent request pages, with right-padded KV.

    Persistent pages contain only real tokens. Each layer writes one new token
    at that row's logical position, then gathers [B, Hkv, Lmax, D] scratch KV.
    Batch row order never changes block ownership.
    """

    def __init__(
        self,
        views: list[PagedKVCache],
        num_layers: int,
    ) -> None:
        super().__init__()
        if not views or any(view.num_tokens < 2 for view in views):
            raise ValueError("Decode requires nonempty histories and a reserved new slot.")
        self.views = tuple(views)
        self.valid_lengths = tuple(view.num_tokens for view in views)
        self.max_length = max(self.valid_lengths)
        # A scheduler-reserved last slot has not been written yet.
        self.lengths = [
            [length - 1 for length in self.valid_lengths]
            for _ in range(num_layers)
        ]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        # Transformers needs a scalar rectangular extent, not per-row positions.
        return max(self.lengths[layer_idx])

    def get_max_cache_shape(self):
        return None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (key_states.ndim != 4 or key_states.shape[0] != len(self.views)
                or key_states.shape[2] != 1):
            raise ValueError("Batch decode KV must have shape [B, kv_heads, 1, head_dim].")
        if value_states.shape != key_states.shape:
            raise ValueError("Key and value shapes must match.")
        positions = self.lengths[layer_idx]
        if any(pos + 1 != length for pos, length in zip(positions, self.valid_lengths)):
            raise ValueError("Each decode layer must write its reserved slot exactly once.")

        batch, heads, _, dim = key_states.shape
        # Zero padding also avoids uninitialized/NaN values entering QK or AV.
        keys = key_states.new_zeros((batch, heads, self.max_length, dim))
        values = value_states.new_zeros((batch, heads, self.max_length, dim))
        for row, (view, position) in enumerate(zip(self.views, positions)):
            # row:row+1 preserves the per-request [1, Hkv, 1, D] interface.
            view.write_range(
                layer_idx, key_states[row:row + 1], value_states[row:row + 1],
                start_pos=position,
            )
            length = position + 1
            k, v = view.gather(layer_idx, length)
            keys[row:row + 1, :, :length, :] = k
            values[row:row + 1, :, :length, :] = v
            positions[row] = length
        return keys, values


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
