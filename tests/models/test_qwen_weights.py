"""Explicit offline integration: set QWEN_MODEL_PATH to the mounted weights."""
import os

import pytest
import torch
from transformers import AutoTokenizer

from llm.inference.request import Request
from llm.inference.sampler import SamplingParams
from llm.inference.scheduler import Scheduler
from llm.model.qwen2_5_0_5b_instruct.runner import QwenModelRunner


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


@pytest.mark.skipif(not os.environ.get("QWEN_MODEL_PATH"), reason="QWEN_MODEL_PATH not provided; no weights are downloaded")
@torch.inference_mode()
def test_mounted_qwen_variable_batch_logits_and_kv_match_single():
    from unittest.mock import patch

    runner = QwenModelRunner.from_pretrained(
        os.environ["QWEN_MODEL_PATH"], device=os.environ.get("QWEN_DEVICE", "cuda"),
        num_blocks=64, block_size=4,
    )
    sequences = {
        "short": [1, 3, 5],
        "medium": [2, 4, 6, 8, 10, 12, 14],
        "long": [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
    }
    for name, ids in sequences.items():
        for prefix in ("batch-", "single-"):
            runner.cache.allocate(prefix + name, len(ids))
            runner.prefill(torch.tensor([ids], device=runner.spec.device), prefix + name)
    logit_error = 0.0
    kv_error = 0.0
    for step in range(3):
        order = list(sequences)
        if step % 2:
            order.reverse()
        tokens = torch.tensor([[41 + step + row] for row in range(3)], device=runner.spec.device)
        for name in order:
            runner.cache.append("batch-" + name)
            runner.cache.append("single-" + name)
        with patch.object(runner.model, "forward", wraps=runner.model.forward) as forward:
            actual = runner.decode_batch(tokens, ["batch-" + name for name in order]).logits
        assert forward.call_count == 1
        for row, name in enumerate(order):
            reference = runner.decode(tokens[row:row + 1], "single-" + name).logits
            torch.testing.assert_close(actual[row:row + 1], reference, rtol=1e-4, atol=2e-4)
            logit_error = max(logit_error, float((actual[row:row + 1] - reference).abs().max()))
            sequences[name].append(int(tokens[row, 0]))
            for layer in range(runner.model.config.num_hidden_layers):
                batched_kv = runner.cache.view("batch-" + name).gather(layer, len(sequences[name]))
                single_kv = runner.cache.view("single-" + name).gather(layer, len(sequences[name]))
                for actual_kv, reference_kv in zip(batched_kv, single_kv):
                    torch.testing.assert_close(actual_kv, reference_kv, rtol=1e-4, atol=2e-4)
                    kv_error = max(kv_error, float((actual_kv - reference_kv).abs().max()))
    for name in list(runner.cache.block_manager.block_tables):
        runner.cache.release(name)
    allocator = runner.cache.block_manager.allocator
    assert not allocator.allocated
    assert sorted(allocator.free_blocks) == list(range(runner.cache.capacity))
    print(f"mounted batch Qwen: max_abs_logit_error={logit_error:.8g}; max_abs_kv_error={kv_error:.8g}")
