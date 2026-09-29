"""Qwen forward using scheduler-owned standard KV pages."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

from llm.cache.blocks import BlockAllocator, BlockManager
from llm.cache.paged import PagedCacheResources
from llm.inference.runner import RunnerSpec
from llm.inference.types import RunnerOutput

from .cache import QwenPagedCache, make_qwen_kv_pool


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
