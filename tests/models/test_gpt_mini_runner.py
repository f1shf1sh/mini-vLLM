import torch

from toy.config import ModelConfig
from llm.cache.blocks import BlockManager, BlockAllocator
from llm.cache.pool import KVPool
from llm.model.gpt_mini.runner import GPTMiniRunner
from toy.model import GPTMini


def make_model() -> GPTMini:
    config = ModelConfig(
        vocab_size=16, block_size=8, n_layer=2, n_head=2, n_embd=8,
        dropout=0.0, bias=False,
    )
    model = GPTMini(config)
    model.eval()
    return model


def _make_runner():
    model = make_model()
    n_head = model.config.n_head
    head_dim = model.config.n_embd // n_head
    pool = KVPool(
        num_blocks=20, num_layers=model.config.n_layer,
        num_heads=n_head, block_size=4, head_dim=head_dim,
    )
    allocator = BlockAllocator(num_blocks=20)
    bm = BlockManager(allocator=allocator, block_size=4)
    return GPTMiniRunner(model, pool, bm), model.config


def test_prefill_produces_correct_logits_shape():
    runner, cfg = _make_runner()
    input_ids = torch.tensor([[1, 2, 3]])
    runner.block_manager.allocate_for_request("test", input_ids.size(1))
    output = runner.prefill(input_ids, request_id="test")
    assert output.logits.shape == (1, 3, cfg.vocab_size)


def test_decode_produces_correct_logits_shape():
    runner, cfg = _make_runner()
    runner.block_manager.allocate_for_request("test", 3)
    runner.prefill(torch.tensor([[1, 2, 3]]), request_id="test")
    runner.block_manager.append_token("test")
    output = runner.decode(torch.tensor([[4]]), request_id="test")
    assert output.logits.shape == (1, 1, cfg.vocab_size)


def test_kv_written_to_pool_during_prefill():
    runner, cfg = _make_runner()
    runner.block_manager.allocate_for_request("test", 3)
    runner.prefill(torch.tensor([[1, 2, 3]]), request_id="test")
    table = runner.block_manager.block_tables["test"]
    assert table.num_tokens == 3
    block_id, offset = table.locate(0)
    k, v = runner.kv_pool.read(layer=0, block_id=block_id, offset=offset)
    assert not torch.allclose(k, torch.zeros_like(k))


def test_kv_grows_during_decode():
    runner, cfg = _make_runner()
    runner.block_manager.allocate_for_request("test", 3)
    runner.prefill(torch.tensor([[1, 2, 3]]), request_id="test")
    runner.block_manager.append_token("test")
    runner.decode(torch.tensor([[4]]), request_id="test")
    table = runner.block_manager.block_tables["test"]
    assert table.num_tokens == 4


def test_paged_logits_match_full_forward_across_noncontiguous_blocks():
    torch.manual_seed(0)
    runner, _ = _make_runner()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    with torch.no_grad():
        expected_logits, _ = runner.model(input_ids)

    prompt_length = 3
    runner.block_manager.allocate_for_request("test", prompt_length)
    actual = runner.prefill(input_ids[:, :prompt_length], request_id="test")
    torch.testing.assert_close(actual.logits, expected_logits[:, :prompt_length])

    # Another request occupies the next physical block, so decode must follow
    # the block table when the first request grows beyond its initial block.
    runner.block_manager.allocate_for_request("other", 4)
    for token_pos in range(prompt_length, input_ids.size(1)):
        runner.block_manager.append_token("test")
        actual = runner.decode(input_ids[:, token_pos : token_pos + 1], request_id="test")
        torch.testing.assert_close(
            actual.logits, expected_logits[:, token_pos : token_pos + 1]
        )

    table = runner.block_manager.block_tables["test"]
    assert table.num_tokens == input_ids.size(1)
    assert table.physical_blocks == [0, 2]


@torch.no_grad()
def test_gpt_mini_runner_matches_direct_generation_and_releases_cache():
    from llm.inference.request import Request
    from llm.inference.sampler import SamplingParams
    from llm.inference.scheduler import Scheduler
    from toy.generator import generate_tokens

    for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
        model = make_model().to(device)
        runner = GPTMiniRunner.from_model(model, num_blocks=8, block_size=2)
        requests = [Request("a", [1, 2, 3], 4, SamplingParams(do_sample=False)),
                    Request("b", [4], 3, SamplingParams(do_sample=False))]
        expected = [generate_tokens(model, torch.tensor([r.input_ids], device=device),
                                    r.max_new_tokens, SamplingParams(do_sample=False))[0].tolist()
                    for r in requests]
        scheduler = Scheduler(runner)
        for request in requests:
            scheduler.add_request(request)
        scheduler.step()
        scheduler.step()
        assert len(scheduler.running) == 2
        scheduler.run()
        assert [r.output_ids for r in requests] == expected
        assert runner.cache.block_manager.allocator.num_free == 8
        assert not runner.cache.block_manager.block_tables
