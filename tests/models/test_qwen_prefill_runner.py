"""Runner-only prefill tests, independent of the paged-prefill cache TODO.

Numerical tests temporarily inject Transformers DynamicCache into the cache
constructor boundary. They execute a real batched Qwen forward, but do NOT
validate paged batch writes, padding exclusion from KVPool, or batch decode
after prefill. No replacement paged-writing algorithm is implemented here.
"""
import os
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache

from llm.model.qwen2_5_0_5b_instruct import runner as runner_module
from llm.model.qwen2_5_0_5b_instruct.runner import QwenModelRunner


def make_runner(device="cpu"):
    torch.manual_seed(19)
    config = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.0,
        use_sliding_window=False,
    )
    config._attn_implementation = "eager"
    return QwenModelRunner(Qwen2ForCausalLM(config).to(device).eval(), 64, 2)


def release_test_requests(runner):
    # These tests call Runner directly, so the test owns request cleanup.
    # This helper is not evidence of automatic Scheduler cleanup.
    for request_id in list(runner.cache.block_manager.block_tables):
        runner.cache.release(request_id)


def resource_snapshot(runner):
    manager = runner.cache.block_manager
    tables = {}
    for request_id, table in manager.block_tables.items():
        tables[request_id] = (table.num_tokens, tuple(table.physical_blocks))
    return tables, set(manager.allocator.allocated), list(manager.allocator.free_blocks)


def assert_runner_logits_match_references(runner, prompts, *, rtol, atol):
    request_ids = [f"batch-{row}" for row in range(len(prompts))]
    for request_id, prompt in zip(request_ids, prompts):
        runner.cache.allocate(request_id, len(prompt))
    # Execution order differs from allocation order.
    request_ids.reverse()
    inputs = [list(prompt) for prompt in reversed(prompts)]
    before_inputs = [list(prompt) for prompt in inputs]
    standard_cache = DynamicCache()
    full_logits = []
    original_forward = runner.model.forward

    def capture_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        full_logits.append(result.logits.clone())
        return result

    try:
        with patch.object(runner_module, "QwenBatchedPrefillCache", return_value=standard_cache) as factory:
            with patch.object(runner.model, "forward", side_effect=capture_forward) as forward:
                output = runner.prefill_batch(inputs, request_ids)
        assert factory.call_count == 1
        assert forward.call_count == 1
        assert inputs == before_inputs
        assert output.logits.shape == (len(prompts), 1, runner.model.config.vocab_size)
        assert torch.isfinite(full_logits[0]).all()
        kwargs = forward.call_args.kwargs
        assert kwargs["input_ids"].shape == (len(prompts), max(map(len, prompts)))
        assert kwargs["past_key_values"] is standard_cache
        assert kwargs["use_cache"] is True and kwargs["return_dict"] is True
        views, num_layers = factory.call_args.args
        assert num_layers == runner.model.config.num_hidden_layers
        for row, request_id in enumerate(request_ids):
            # Check that the constructor receives the row's actual owner view.
            table = runner.cache.block_manager.block_tables[request_id]
            assert views[row]._table is table
            assert views[row].num_tokens == len(inputs[row])

        max_error = 0.0
        for row, prompt in enumerate(inputs):
            tokens = torch.tensor([prompt], dtype=torch.long, device=runner.spec.device)
            # Dense reference uses neither our mask nor any custom cache.
            dense = runner.model(tokens, use_cache=False).logits
            reference_id = f"reference-{row}"
            runner.cache.allocate(reference_id, len(prompt))
            single = runner.prefill(tokens, reference_id).logits
            actual_prefix = full_logits[0][row:row + 1, :len(prompt), :]
            torch.testing.assert_close(actual_prefix, dense, rtol=rtol, atol=atol)
            torch.testing.assert_close(actual_prefix, single, rtol=rtol, atol=atol)
            torch.testing.assert_close(output.logits[row:row + 1], dense[:, -1:, :], rtol=rtol, atol=atol)
            torch.testing.assert_close(output.logits[row:row + 1], single[:, -1:, :], rtol=rtol, atol=atol)
            max_error = max(max_error, float((actual_prefix - dense).abs().max()))
        return max_error
    finally:
        release_test_requests(runner)


@pytest.mark.parametrize("lengths", [(1,), (2, 5), (1, 5, 9), (4, 4)])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable",
    )),
])
@torch.inference_mode()
def test_runner_tensor_pipeline_matches_single_and_dense_using_standard_cache(lengths, device):
    prompts = []
    for row, length in enumerate(lengths):
        prompts.append([(row * 7 + position) % 63 + 1 for position in range(length)])
    runner = make_runner(device)
    assert_runner_logits_match_references(runner, prompts, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("pad_token_id", [None, 7])
@torch.inference_mode()
def test_runner_supplies_explicit_positions_causal_mask_and_padding_query_rows(pad_token_id):
    runner = make_runner()
    runner.model.config.pad_token_id = pad_token_id
    prompts = [[7, 3], [2, 4, 6, 8, 10]]
    for request_id, prompt in zip(("short", "long"), prompts):
        runner.cache.allocate(request_id, len(prompt))
    standard_cache = DynamicCache()
    try:
        with patch.object(runner_module, "QwenBatchedPrefillCache", return_value=standard_cache):
            with patch.object(runner.model, "forward", wraps=runner.model.forward) as forward:
                output = runner.prefill_batch(prompts, ["short", "long"])
        kwargs = forward.call_args.kwargs
        padding = 0 if pad_token_id is None else pad_token_id
        assert kwargs["input_ids"].tolist() == [[7, 3, padding, padding, padding], prompts[1]]
        assert kwargs["input_ids"].dtype == torch.long
        # A real token equal to PAD must remain valid.
        assert kwargs["position_ids"].tolist() == [[0, 1, 0, 0, 0], [0, 1, 2, 3, 4]]
        assert kwargs["cache_position"].tolist() == [0, 1, 2, 3, 4]
        mask = kwargs["attention_mask"]
        assert mask.shape == (2, 1, 5, 5)
        assert mask.dtype == runner.spec.dtype
        blocked = torch.finfo(mask.dtype).min
        expected_short = torch.tensor([
            [0, blocked, blocked, blocked, blocked],
            [0, 0, blocked, blocked, blocked],
            [0, blocked, blocked, blocked, blocked],
            [0, blocked, blocked, blocked, blocked],
            [0, blocked, blocked, blocked, blocked],
        ], dtype=mask.dtype)
        expected_long = torch.tensor([
            [0, blocked, blocked, blocked, blocked],
            [0, 0, blocked, blocked, blocked],
            [0, 0, 0, blocked, blocked],
            [0, 0, 0, 0, blocked],
            [0, 0, 0, 0, 0],
        ], dtype=mask.dtype)
        torch.testing.assert_close(mask[0, 0], expected_short, rtol=0, atol=0)
        torch.testing.assert_close(mask[1, 0], expected_long, rtol=0, atol=0)
        assert output.logits.shape == (2, 1, 64)
    finally:
        release_test_requests(runner)


@pytest.mark.parametrize("inputs,request_ids,error", [
    ([], [], ValueError),
    (None, ["a"], ValueError),
    ([[1, 2]], "a", ValueError),
    ([[1, 2]], ["a", "b"], ValueError),
    ([[1, 2], [3, 4, 5]], ["a", "a"], ValueError),
    ([[1, 2], [3, 4, 5]], ["a", 3], ValueError),
    ([[], [3, 4, 5]], ["a", "b"], ValueError),
    ([[1, 2], "345"], ["a", "b"], ValueError),
    ([[1, 2], [3, 4, -1]], ["a", "b"], ValueError),
    ([[1, 2], [3, 4, 64]], ["a", "b"], ValueError),
    ([[1, 2], [3, 4, 1.5]], ["a", "b"], ValueError),
    ([[1, 2], [3, 4, True]], ["a", "b"], ValueError),
    ([[1, 2], [3, 4, "5"]], ["a", "b"], ValueError),
    ([[1] * 65], ["a"], ValueError),
    ([[1, 2], [3, 4]], ["a", "b"], ValueError),
    ([[1, 2], [3, 4, 5]], ["a", "missing"], KeyError),
])
def test_invalid_input_is_rejected_before_forward_or_cache_construction(inputs, request_ids, error):
    runner = make_runner()
    runner.cache.allocate("a", 2)
    runner.cache.allocate("b", 3)
    before_pool = runner.cache.pool.pool.clone()
    before_resources = resource_snapshot(runner)
    try:
        with patch.object(runner_module, "QwenBatchedPrefillCache") as factory:
            with patch.object(runner.model, "forward") as forward:
                with pytest.raises(error):
                    runner.prefill_batch(inputs, request_ids)
        factory.assert_not_called()
        forward.assert_not_called()
        assert resource_snapshot(runner) == before_resources
        torch.testing.assert_close(runner.cache.pool.pool, before_pool, rtol=0, atol=0)
    finally:
        release_test_requests(runner)


@pytest.mark.parametrize("pad_token_id", [-1, 64, True])
def test_invalid_padding_id_is_rejected_before_model_execution(pad_token_id):
    runner = make_runner()
    runner.cache.allocate("a", 2)
    runner.model.config.pad_token_id = pad_token_id
    try:
        with patch.object(runner_module, "QwenBatchedPrefillCache") as factory:
            with patch.object(runner.model, "forward") as forward:
                with pytest.raises(ValueError, match="pad_token_id"):
                    runner.prefill_batch([[1, 2]], ["a"])
        factory.assert_not_called()
        forward.assert_not_called()
    finally:
        release_test_requests(runner)


@pytest.mark.parametrize("setting,value", [
    ("_attn_implementation", "sdpa"),
    ("use_sliding_window", True),
])
def test_unsupported_attention_configuration_is_rejected(setting, value):
    runner = make_runner()
    runner.cache.allocate("a", 2)
    setattr(runner.model.config, setting, value)
    try:
        with patch.object(runner_module, "QwenBatchedPrefillCache") as factory:
            with patch.object(runner.model, "forward") as forward:
                with pytest.raises(ValueError):
                    runner.prefill_batch([[1, 2]], ["a"])
        factory.assert_not_called()
        forward.assert_not_called()
    finally:
        release_test_requests(runner)


def test_cache_constructor_failure_propagates_without_fallback_or_resource_release():
    runner = make_runner()
    runner.cache.allocate("a", 2)
    before = resource_snapshot(runner)
    try:
        with patch.object(
            runner_module, "QwenBatchedPrefillCache", side_effect=NotImplementedError("cache unavailable"),
        ):
            with patch.object(runner.model, "forward") as forward:
                with patch.object(runner.cache, "release", side_effect=AssertionError("Caller owns release")):
                    with pytest.raises(NotImplementedError, match="cache unavailable"):
                        runner.prefill_batch([[1, 2]], ["a"])
        forward.assert_not_called()
        assert resource_snapshot(runner) == before
    finally:
        release_test_requests(runner)


@torch.inference_mode()
def test_forward_failure_propagates_and_leaves_resource_cleanup_to_caller():
    runner = make_runner()
    runner.cache.allocate("a", 2)
    before = resource_snapshot(runner)
    original_forward = runner.model.forward

    def fail_after_forward(*args, **kwargs):
        original_forward(*args, **kwargs)
        raise RuntimeError("forward failed")

    try:
        with patch.object(runner_module, "QwenBatchedPrefillCache", return_value=DynamicCache()):
            with patch.object(runner.model, "forward", side_effect=fail_after_forward):
                with patch.object(runner.cache, "release", side_effect=AssertionError("Caller owns release")):
                    with pytest.raises(RuntimeError, match="forward failed"):
                        runner.prefill_batch([[1, 2]], ["a"])
        assert resource_snapshot(runner) == before
    finally:
        release_test_requests(runner)


@pytest.mark.skipif(
    not os.environ.get("QWEN_MODEL_PATH"), reason="QWEN_MODEL_PATH not provided; no weights are downloaded",
)
@torch.inference_mode()
def test_local_weights_runner_pipeline_matches_references_using_standard_cache():
    runner = QwenModelRunner.from_pretrained(
        os.environ["QWEN_MODEL_PATH"],
        device=os.environ.get("QWEN_DEVICE", "cuda"),
        num_blocks=64,
        block_size=4,
    )
    prompts = [[1], [2, 4, 6, 8, 10], [11, 13, 15, 17, 19, 21, 23, 25, 27]]
    max_error = assert_runner_logits_match_references(runner, prompts, rtol=1e-4, atol=2e-4)
    print(f"Runner-only prefill with DynamicCache: max_abs_logit_error={max_error:.8g}")
