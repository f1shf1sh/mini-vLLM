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



class QwenBatchedPrefillCache(Cache):
    """One-forward prefill view over independently owned request pages.

    Each view reserves only real prompt tokens. All layers initially have zero
    written tokens, even though view.num_tokens is already the prompt length.
    Per-layer written lengths are execution metadata, not a second allocator.

    Incoming and returned K/V have shape [B, Hkv, T, D], where T is the longest
    prompt. Persistent writes contain only each row's real prefix. Returned
    scratch tensors use zero padding and never expose unused physical slots.
    This adapter must not allocate, append, release, or mutate block ownership.
    """

    def __init__(
        self,
        views: list[PagedKVCache],
        num_layers: int,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("Prefill cache requires at least one layer.")
        if not views or any(view.num_tokens < 1 for view in views):
            raise ValueError("Prefill requires nonempty reserved prompts.")

        # Preserve batch row order and snapshot real lengths, excluding padding.
        # Scheduler has reserved these slots but has not written prompt KV yet.
        self.views = tuple(views)
        self.valid_lengths = tuple(view.num_tokens for view in views)
        self.max_length = max(self.valid_lengths)

        # Each iteration creates a separate mutable list for one layer.
        # Updating one layer must not change another layer's write progress.
        self.written_lengths = [
            [0] * len(self.views)
            for _ in range(num_layers)
        ]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the layer's scalar written extent for Transformers metadata.

        Before prefill it is zero. After the layer update it is the maximum
        written row length; it is not every request's true sequence length.
        """
        return max(self.written_lengths[layer_idx])

    def get_max_cache_shape(self):
        # The physical pool is paged; this adapter is not a static dense cache.
        return None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write one layer's real prompt KV and return zero-padded scratch KV.

        Inputs and outputs are [B, Hkv, T, D], with T equal to max_length.
        The constructor supplies views in batch order, valid_lengths excluding
        padding, and independent written_lengths for every layer. Each layer
        starts unwritten and may be updated only once by this prefill forward.

        Scheduler owns reservations and cleanup. If a write or gather fails,
        discard the executing requests: pages may already be partly updated.
        This method propagates failures and never allocates or releases blocks.
        """
        if key_states.ndim != 4 or value_states.shape != key_states.shape:
            raise ValueError("Prefill K/V must have matching [B, Hkv, T, D] shapes.")
        if type(layer_idx) is not int or not 0 <= layer_idx < len(self.written_lengths):
            raise ValueError("Invalid prefill cache layer index.")

        batch_size, num_kv_heads, padded_length, head_dim = key_states.shape
        if batch_size == 0 or batch_size != len(self.views):
            raise ValueError("Prefill batch size must match the request views.")
        if len(self.valid_lengths) != batch_size:
            raise ValueError("Each prefill row must have one valid length.")
        if padded_length != self.max_length or self.max_length != max(self.valid_lengths):
            raise ValueError("Prefill token dimension must equal the longest prompt.")
        if value_states.dtype != key_states.dtype or value_states.device != key_states.device:
            raise ValueError("Key and value dtype/device must match.")

        written_lengths = self.written_lengths[layer_idx]
        if len(written_lengths) != batch_size or any(length != 0 for length in written_lengths):
            raise ValueError("Each prefill layer must write fresh prompts exactly once.")

        # Validate every row before modifying any persistent page. In particular,
        # a later row with changed reservations or incompatible storage must not
        # cause earlier rows to be written before its error is discovered.
        for row, view in enumerate(self.views):
            real_length = self.valid_lengths[row]
            if real_length <= 0 or view.num_tokens != real_length:
                raise ValueError("Prefill requires unchanged, nonempty prompt reservations.")

            # PagedKVCache exposes storage through its internal pool. Keep this
            # model-specific shape check here rather than teaching Scheduler
            # about layers, heads, or the layout of physical KV tensors.
            pool = view._pool
            if layer_idx >= pool.num_layers:
                raise ValueError("Prefill layer index exceeds the KV pool.")
            if num_kv_heads != pool.num_heads or head_dim != pool.head_dim:
                raise ValueError("Prefill KV heads and head dimension must match the pool.")
            if num_kv_heads != view._n_head or head_dim != view._head_dim:
                raise ValueError("Prefill KV shape must match the request view.")
            if key_states.dtype != pool.pool.dtype or key_states.device != pool.pool.device:
                raise ValueError("Prefill KV dtype/device must match every request's pool.")

        # These are temporary attention inputs, not additional persistent slots.
        # Zero padding prevents unused physical data (possibly NaN) from entering
        # QK or AV: a mask alone cannot make 0 * NaN numerically harmless.
        output_shape = (batch_size, num_kv_heads, padded_length, head_dim)
        keys = key_states.new_zeros(output_shape)
        values = value_states.new_zeros(output_shape)

        for row, view in enumerate(self.views):
            real_length = self.valid_lengths[row]
            # The slice row:row+1 preserves batch dimension 1. Only :real_length
            # is written; padded input positions never touch persistent pages.
            row_keys = key_states[row:row + 1, :, :real_length, :]
            row_values = value_states[row:row + 1, :, :real_length, :]
            view.write_range(layer_idx, row_keys, row_values, start_pos=0)
            written_lengths[row] = real_length

            # gather follows this request's logical-to-physical block mapping,
            # including noncontiguous pages. Each result is [1, Hkv, Li, D].
            gathered_keys, gathered_values = view.gather(layer_idx, real_length)
            keys[row:row + 1, :, :real_length, :] = gathered_keys
            values[row:row + 1, :, :real_length, :] = gathered_values

        # cache_kwargs may contain rectangular cache_position and RoPE metadata.
        # Qwen has already applied RoPE. Row-local valid lengths, not this shared
        # cache_position, determine which tokens belong in persistent storage.
        return keys, values


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
