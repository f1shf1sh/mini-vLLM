"""Explicit offline integration: set QWEN_MODEL_PATH to the mounted weights."""
import os

import pytest
import torch
from transformers import AutoTokenizer

from llm.inference.request import Request
from llm.inference.sampler import SamplingParams
from llm.inference.scheduler import Scheduler
from llm.models.qwen.runner import QwenModelRunner


@pytest.mark.skipif(not os.environ.get("QWEN_MODEL_PATH"), reason="QWEN_MODEL_PATH not provided; no weights are downloaded")
@torch.inference_mode()
def test_mounted_qwen_logits_greedy_scheduling_and_recovery():
    path = os.environ["QWEN_MODEL_PATH"]
    device = os.environ.get("QWEN_DEVICE", "cuda")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    runner = QwenModelRunner.from_pretrained(path, device=device, num_blocks=96, block_size=4)
    prompts = ["What is 2 + 2?", "Name one planet in the solar system."]
    requests = []
    references = []
    for index, prompt in enumerate(prompts):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
        )
        requests.append(Request(str(index), ids, 6, SamplingParams(do_sample=False)))
        tokens = torch.tensor([ids], device=device)
        for _ in range(6):
            logits = runner.model(tokens, use_cache=False).logits
            tokens = torch.cat([tokens, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        references.append(tokens[0].tolist())
    # Direct cache correctness with noncontiguous physical blocks.
    tokens = torch.tensor([references[0]], device=device)
    expected = runner.model(tokens, use_cache=False).logits
    runner.cache.allocate("verify", 3)
    actual = runner.prefill(tokens[:, :3], "verify").logits
    torch.testing.assert_close(actual, expected[:, :3], rtol=1e-4, atol=2e-4)
    errors = [float((actual-expected[:, :3]).abs().max())]
    runner.cache.allocate("blocker", 4)
    for position in range(3, 10):
        runner.cache.append("verify")
        actual = runner.decode(tokens[:, position:position+1], "verify").logits
        reference = expected[:, position:position+1]
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=2e-4)
        errors.append(float((actual-reference).abs().max()))
    runner.cache.release("verify")
    runner.cache.release("blocker")
    scheduler = Scheduler(runner)
    for request in requests:
        scheduler.add_request(request)
    scheduler.step()
    scheduler.step()
    assert len(scheduler.running) == 2
    scheduler.run()
    assert [r.output_ids for r in requests] == references
    assert runner.cache.pool.num_heads == runner.model.config.num_key_value_heads
    assert runner.cache.pool.num_heads != runner.model.config.num_attention_heads
    assert runner.cache.block_manager.allocator.num_free == runner.cache.capacity
    assert not runner.cache.block_manager.block_tables
    print(f"mounted Qwen: max_abs_logit_error={max(errors):.8f}; greedy requests=2; free_blocks={runner.cache.capacity}")
