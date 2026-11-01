import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from llm.inference.request import Request
from llm.inference.sampler import SamplingParams
from llm.inference.scheduler import Scheduler
from llm.model.qwen2_5_0_5b_instruct.runner import QwenModelRunner


def tiny_runner(device="cpu"):
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=32, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, attention_dropout=0.0,
        use_sliding_window=False,
    )
    config._attn_implementation = "eager"
    return QwenModelRunner(Qwen2ForCausalLM(config).to(device).eval(), 12, 2)


@pytest.mark.parametrize("prompt_length", [1, 3, 5])
@torch.inference_mode()
def test_qwen_paged_prefill_and_fragmented_decode_match_dense_logits(prompt_length):
    runner = tiny_runner()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    expected = runner.model(ids, use_cache=False).logits
    runner.cache.allocate("a", prompt_length)
    actual = runner.prefill(ids[:, :prompt_length], "a").logits
    torch.testing.assert_close(actual, expected[:, :prompt_length], rtol=1e-5, atol=1e-5)
    runner.cache.allocate("blocker", 2)
    for position in range(prompt_length, ids.shape[1]):
        runner.cache.append("a")
        actual = runner.decode(ids[:, position:position+1], "a").logits
        torch.testing.assert_close(actual, expected[:, position:position+1], rtol=1e-5, atol=1e-5)
    blocks = runner.cache.block_manager.block_tables["a"].physical_blocks
    assert any(b != a + 1 for a, b in zip(blocks, blocks[1:]))
    runner.cache.release("a")
    runner.cache.release("blocker")
    assert runner.cache.block_manager.allocator.num_free == runner.cache.capacity


@pytest.mark.parametrize("max_batch_size", [1, 2, 4])
@torch.inference_mode()
def test_qwen_scheduler_interleaves_gqa_requests_and_matches_greedy_reference(max_batch_size):
    runner = tiny_runner()
    requests = [Request("short", [1, 3], 4, SamplingParams(do_sample=False)),
                Request("long", [2, 4, 6, 8, 10], 4, SamplingParams(do_sample=False))]
    references = []
    for request in requests:
        ids = torch.tensor([request.input_ids])
        for _ in range(request.max_new_tokens):
            token = runner.model(ids, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat((ids, token), dim=1)
        references.append(ids[0].tolist())
    scheduler = Scheduler(runner, max_batch_size=max_batch_size)
    for request in requests:
        scheduler.add_request(request)
    scheduler.step()
    scheduler.step()
    assert len(scheduler.running) == 2
    scheduler.run()
    assert [r.output_ids for r in requests] == references
    assert runner.cache.pool.num_heads == 2
    assert runner.model.config.num_attention_heads == 4
    assert runner.cache.block_manager.allocator.num_free == 12
    assert not runner.cache.block_manager.block_tables


@pytest.mark.parametrize("max_batch_size", [1, 2])
@torch.inference_mode()
def test_qwen_cancellation_preserves_other_request_output_and_pages(max_batch_size):
    runner = tiny_runner()
    survivor = Request("survivor", [1, 3, 5], 4, SamplingParams(do_sample=False))
    cancelled = Request("cancelled", [2, 4], 5, SamplingParams(do_sample=False))
    expected = torch.tensor([survivor.input_ids])
    for _ in range(survivor.max_new_tokens):
        logits = runner.model(expected, use_cache=False).logits
        expected = torch.cat([expected, logits[:, -1].argmax(-1, keepdim=True)], dim=1)

    scheduler = Scheduler(runner, max_batch_size=max_batch_size)
    scheduler.add_request(survivor)
    scheduler.add_request(cancelled)
    scheduler.step()
    scheduler.step()
    table = runner.cache.block_manager.block_tables["survivor"]
    blocks = list(table.physical_blocks)
    pages = runner.cache.pool.pool[:, :, blocks].clone()
    generated = list(cancelled.generated_ids)

    assert scheduler.cancel_request("cancelled") is True
    assert runner.cache.block_manager.block_tables["survivor"] is table
    assert table.physical_blocks == blocks
    torch.testing.assert_close(runner.cache.pool.pool[:, :, blocks], pages, rtol=0, atol=0)
    scheduler.run()

    assert survivor.output_ids == expected[0].tolist()
    assert cancelled.status.name == "CANCELLED"
    assert cancelled.generated_ids == generated
    assert not runner.cache.block_manager.block_tables
    assert runner.cache.block_manager.allocator.num_free == runner.cache.capacity
