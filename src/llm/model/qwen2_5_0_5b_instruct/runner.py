"""Qwen forward using scheduler-owned standard KV pages."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

from llm.cache.blocks import BlockAllocator, BlockManager
from llm.cache.paged import PagedCacheResources
from llm.inference.runner import RunnerSpec
from llm.inference.types import RunnerOutput

from .cache import (
    QwenBatchedDecodeCache,
    QwenBatchedPrefillCache,
    QwenPagedCache,
    make_qwen_kv_pool,
)


class QwenModelRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        num_blocks: int,
        block_size: int,
    ) -> None:
        if model.config.model_type != "qwen2":
            raise ValueError("This adapter supports Qwen2/Qwen2.5 models.")
        self.model = model.eval()
        parameter = next(model.parameters())
        self.spec = RunnerSpec(
            model.config.max_position_embeddings, parameter.device, parameter.dtype,
        )
        pool = make_qwen_kv_pool(
            model.config, num_blocks, block_size,
            device=parameter.device, dtype=parameter.dtype,
        )
        self.cache = PagedCacheResources(
            pool, BlockManager(BlockAllocator(num_blocks), block_size),
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        num_blocks: int = 256,
        block_size: int = 16,
    ) -> QwenModelRunner:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=dtype, attn_implementation="eager",
        ).to(device)
        return cls(model, num_blocks, block_size)

    @torch.inference_mode()
    def _forward(
        self,
        input_ids: torch.Tensor,
        request_id: str,
        *,
        prefill: bool,
    ) -> RunnerOutput:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have shape [1, tokens] with at least one token.")
        if not prefill and input_ids.shape[1] != 1:
            raise ValueError("decode expects exactly one token.")
        view = self.cache.view(request_id)
        past_length = view.num_tokens - input_ids.shape[1]
        if past_length < 0 or (prefill and past_length != 0):
            raise ValueError("Input does not match scheduler-reserved KV slots.")
        if view.num_tokens > self.spec.context_length:
            raise ValueError("Request exceeds model context length.")
        cache = QwenPagedCache(view, self.model.config.num_hidden_layers, past_length)
        positions = torch.arange(past_length, view.num_tokens, device=self.spec.device)
        output = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones((1, view.num_tokens), device=self.spec.device, dtype=torch.long),
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        return RunnerOutput(logits=output.logits)

    def prefill(self, input_ids: torch.Tensor, request_id: str) -> RunnerOutput:
        return self._forward(input_ids, request_id, prefill=True)

    @torch.inference_mode()
    def prefill_batch(
        self,
        input_ids: list[list[int]],
        request_ids: list[str],
    ) -> RunnerOutput:
        """Execute one forward over fresh prompts and return logits [B, 1, V].

        Input rows contain only real tokens and follow request_ids order.
        Scheduler must reserve exactly each prompt's length before this call;
        no prompt KV has been written yet. Scheduler also owns growth, release,
        and cleanup if execution fails after partially writing the cache.

        This method owns input padding, RoPE positions, attention visibility,
        and selection of each row's last valid logits. The cache adapter owns
        the per-layer writes into the already-reserved request pages.
        """
        if not isinstance(input_ids, list) or not input_ids:
            raise ValueError("input_ids must be a nonempty list of prompt lists.")
        batch_size = len(input_ids)
        if not isinstance(request_ids, list) or len(request_ids) != batch_size:
            raise ValueError("request_ids must contain one ID per input row.")
        if any(not isinstance(request_id, str) for request_id in request_ids):
            raise ValueError("request_ids must be strings.")
        if len(set(request_ids)) != batch_size:
            raise ValueError("Duplicate request IDs would alias KV writes.")
        if self.model.config._attn_implementation != "eager":
            raise ValueError("Batched paged prefill currently requires eager attention.")
        if self.model.config.use_sliding_window:
            raise ValueError("Batched paged prefill currently requires full attention.")

        # Validate every prompt before constructing tensors or entering any
        # model layer. Use the actual embedding table's vocabulary bounds.
        vocab_size = self.model.get_input_embeddings().num_embeddings
        for prompt in input_ids:
            if not isinstance(prompt, list) or not prompt:
                raise ValueError("Each prompt must be a nonempty list of token IDs.")
            if len(prompt) > self.spec.context_length:
                raise ValueError("Request exceeds model context length.")
            for token_id in prompt:
                if type(token_id) is not int or not 0 <= token_id < vocab_size:
                    raise ValueError("Token IDs must be integers within the model vocabulary.")

        prompt_lengths = [len(prompt) for prompt in input_ids]
        max_prompt_length = max(prompt_lengths)
        views = []
        for request_id, prompt_length in zip(request_ids, prompt_lengths):
            view = self.cache.view(request_id)
            if view.num_tokens != prompt_length:
                raise ValueError("Prompt length must match scheduler-reserved KV slots.")
            views.append(view)
        # num_tokens counts reserved slots, not written tokens. Freshness is a
        # caller precondition; equal lengths alone cannot prove empty history.

        pad_token_id = self.model.config.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
        if type(pad_token_id) is not int or not 0 <= pad_token_id < vocab_size:
            raise ValueError("pad_token_id must be an integer within the model vocabulary.")

        # [B, T] stores rectangular model input only. PAD does not add logical
        # cache slots; a real prompt token equal to PAD is still a valid token.
        padded_input_ids = torch.full(
            (batch_size, max_prompt_length),
            pad_token_id,
            dtype=torch.long,
            device=self.spec.device,
        )
        for row, prompt in enumerate(input_ids):
            prompt_tensor = torch.tensor(
                prompt,
                dtype=torch.long,
                device=self.spec.device,
            )
            prompt_length = len(prompt)
            padded_input_ids[row, :prompt_length] = prompt_tensor

        lengths = torch.tensor(
            prompt_lengths,
            dtype=torch.long,
            device=self.spec.device,
        )  # [B]: real prompt lengths, excluding padding.
        positions = torch.arange(
            max_prompt_length,
            dtype=torch.long,
            device=self.spec.device,
        )  # [T]: positions in the rectangular input.

        # [B, 1] compared with [1, T] broadcasts to [B, T].
        row_lengths = lengths.unsqueeze(1)
        columns = positions.unsqueeze(0)
        valid_tokens = columns < row_lengths

        # Each real prompt starts at logical position zero. repeat creates
        # independent rows so padding positions can be set to zero in place.
        position_ids = columns.repeat(batch_size, 1)
        position_ids.masked_fill_(~valid_tokens, 0)

        # Additive mask [B, 1, T, T]: batch, shared head, query, key.
        # The complete 4-D mask must include causality as well as padding.
        blocked_value = torch.finfo(self.spec.dtype).min
        attention_mask = torch.full(
            (batch_size, 1, max_prompt_length, max_prompt_length),
            blocked_value,
            dtype=self.spec.dtype,
            device=self.spec.device,
        )
        for row, prompt_length in enumerate(prompt_lengths):
            # Keep the learner's explicit loop: a real query reads only its
            # own position and earlier real keys. Future/PAD columns stay blocked.
            for query_position in range(prompt_length):
                attention_mask[row, 0, query_position, :query_position + 1] = 0

            # Give unused queries a defined softmax row by allowing real key 0.
            # Their logits are discarded; their KV must be cropped by the cache
            # adapter. A fully finite-minimum row is not a zero-weight row.
            if prompt_length < max_prompt_length:
                attention_mask[row, 0, prompt_length:, 0] = 0

        cache = QwenBatchedPrefillCache(views, self.model.config.num_hidden_layers)
        # cache_position describes the rectangular extent, not valid row lengths.
        # Passing this cache is essential: a default Transformers cache would
        # leave Scheduler's persistent pages unwritten.
        output = self.model(
            input_ids=padded_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=positions,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

        # Pair row i with its last real position, Li - 1. The last rectangular
        # column may be PAD. Indexing yields [B, V]; restore [B, 1, V] for Sampler.
        batch_rows = torch.arange(batch_size, device=self.spec.device)
        last_token_positions = lengths - 1
        next_token_logits = output.logits[batch_rows, last_token_positions]
        next_token_logits = next_token_logits.unsqueeze(1)
        return RunnerOutput(logits=next_token_logits)

    def decode(self, input_ids: torch.Tensor, request_id: str) -> RunnerOutput:
        return self._forward(input_ids, request_id, prefill=False)


    @torch.inference_mode()
    def decode_batch(
        self,
        input_ids: torch.Tensor,
        request_ids: list[str],
    ) -> RunnerOutput:
        """Execute one real batched decode with row-local RoPE and KV ownership.

        Scheduler reserves one unwritten slot per request before this call.
        Input is [B, 1], output logits are [B, 1, V] in request_ids order.
        """
        batch_size = len(request_ids)
        if (batch_size == 0 or input_ids.ndim != 2
                or input_ids.shape != (batch_size, 1)):
            raise ValueError("decode_batch expects nonempty input_ids of shape [B, 1].")
        if any(not isinstance(key, str) for key in request_ids):
            raise ValueError("request_ids must be strings.")
        if len(set(request_ids)) != batch_size:
            raise ValueError("Duplicate request IDs would alias KV writes.")
        if input_ids.dtype != torch.long or input_ids.device != self.spec.device:
            raise ValueError("input_ids must be torch.long on the runner device.")
        if self.model.config._attn_implementation != "eager":
            raise ValueError("Batched paged decode currently requires eager attention.")
        if self.model.config.use_sliding_window:
            raise ValueError("Batched paged decode currently requires full attention.")

        # Resolve/validate every row before the first model layer can write KV.
        views = [self.cache.view(key) for key in request_ids]
        if any(view.num_tokens > self.spec.context_length for view in views):
            raise ValueError("Request exceeds model context length.")
        cache = QwenBatchedDecodeCache(views, self.model.config.num_hidden_layers)
        lengths = torch.tensor(cache.valid_lengths, device=self.spec.device)
        position_ids = (lengths - 1).unsqueeze(1)  # [B, 1], real RoPE positions
        columns = torch.arange(cache.max_length, device=self.spec.device)
        valid = columns.unsqueeze(0) < lengths.unsqueeze(1)  # [B, Lmax]
        attention_mask = torch.zeros(
            (batch_size, 1, 1, cache.max_length),
            dtype=self.spec.dtype, device=self.spec.device,
        )
        attention_mask.masked_fill_(
            ~valid[:, None, None, :], torch.finfo(self.spec.dtype).min,
        )
        # The explicit 4-D mask is already causal for one newest query per row.
        # cache_position describes only the padded rectangular extent; RoPE and
        # persistent writes use row-local positions above/in the cache adapter.
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=columns[-1:],
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        return RunnerOutput(logits=output.logits)
