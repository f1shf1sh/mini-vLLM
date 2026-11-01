"""Qwen forward using scheduler-owned standard KV pages."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

from llm.cache.blocks import BlockAllocator, BlockManager
from llm.cache.paged import PagedCacheResources
from llm.inference.runner import RunnerSpec
from llm.inference.types import RunnerOutput

from .cache import QwenBatchedDecodeCache, QwenPagedCache, make_qwen_kv_pool


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
