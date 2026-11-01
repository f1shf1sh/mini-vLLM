"""Batched decode must match independent requests and preserve page ownership."""
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

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
    runner = QwenModelRunner(Qwen2ForCausalLM(config).to(device).eval(), 64, 2)
    # Unwritten physical slots are deliberately nonzero. Padding must never
    # expose these values to attention or overwrite them in the persistent pool.
    runner.cache.pool.pool.fill_(37.25)
    return runner


def prefill(runner, request_id, ids):
    runner.cache.allocate(request_id, len(ids))
    runner.prefill(torch.tensor([ids], device=runner.spec.device), request_id)


def pages(runner, request_id):
    table = runner.cache.block_manager.block_tables[request_id]
    return runner.cache.pool.pool[:, :, table.physical_blocks].clone()


def history(runner, request_id, length):
    return [
        tuple(value.clone() for value in runner.cache.view(request_id).gather(layer, length))
        for layer in range(runner.model.config.num_hidden_layers)
    ]


def assert_history_equal(actual, expected, *, exact=False):
    for actual_layer, expected_layer in zip(actual, expected):
        for actual_tensor, expected_tensor in zip(actual_layer, expected_layer):
            torch.testing.assert_close(
                actual_tensor, expected_tensor,
                rtol=0 if exact else 1e-5, atol=0 if exact else 1e-5,
            )


def assert_released(runner):
    for request_id in list(runner.cache.block_manager.block_tables):
        runner.cache.release(request_id)
    allocator = runner.cache.block_manager.allocator
    assert not allocator.allocated
    assert sorted(allocator.free_blocks) == list(range(runner.cache.capacity))


@pytest.mark.parametrize("prompt_lengths", [(1,), (1, 3), (5, 2, 7), (2, 2)])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable",
    )),
])
@torch.inference_mode()
def test_variable_length_batch_matches_single_and_dense_with_reordering(prompt_lengths, device):
    batched, serial = make_runner(device), make_runner(device)
    sequences = {
        f"r{i}": [(i * 7 + j) % 63 + 1 for j in range(length)]
        for i, length in enumerate(prompt_lengths)
    }
    for runner in (batched, serial):
        for request_id, ids in sequences.items():
            prefill(runner, request_id, ids)
            # Keep other owners between the initial pages and later growth.
            runner.cache.allocate(f"blocker-{request_id}", 2)

    for step in range(4):
        order = list(sequences)
        order = order[step % len(order):] + order[:step % len(order)]
        if step % 2:
            order.reverse()
        old_history = {
            request_id: history(batched, request_id, len(ids))
            for request_id, ids in sequences.items()
        }
        inputs = []
        for index, request_id in enumerate(order):
            inputs.append([(len(sequences[request_id]) * 3 + index) % 63 + 1])
            batched.cache.append(request_id)
            serial.cache.append(request_id)
        input_ids = torch.tensor(inputs, device=device)
        with patch.object(batched.model, "forward", wraps=batched.model.forward) as forward:
            actual = batched.decode_batch(input_ids, order).logits
        assert forward.call_count == 1
        assert actual.shape == (len(order), 1, 64)

        for row, request_id in enumerate(order):
            reference = serial.decode(input_ids[row:row + 1], request_id).logits
            old_length = len(sequences[request_id])
            sequences[request_id].append(inputs[row][0])
            dense = serial.model(
                torch.tensor([sequences[request_id]], device=device), use_cache=False,
            ).logits[:, -1:, :]
            torch.testing.assert_close(actual[row:row + 1], reference, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(actual[row:row + 1], dense, rtol=1e-5, atol=1e-5)
            assert_history_equal(
                history(batched, request_id, old_length),
                old_history[request_id], exact=True,
            )
            assert_history_equal(
                history(batched, request_id, old_length + 1),
                history(serial, request_id, old_length + 1),
            )
            assert batched.cache.view(request_id).num_tokens == len(sequences[request_id])

    tables = batched.cache.block_manager.block_tables
    assert any(
        right != left + 1
        for request_id in sequences
        for left, right in zip(tables[request_id].physical_blocks, tables[request_id].physical_blocks[1:])
    )
    assert_released(batched)
    assert_released(serial)


@torch.inference_mode()
def test_batch_uses_row_positions_and_masks_only_temporary_padding():
    runner = make_runner()
    prefill(runner, "short", [1, 3])
    prefill(runner, "long", [2, 4, 6, 8, 10, 12])
    runner.cache.append("short")
    runner.cache.append("long")
    before = runner.cache.pool.pool.clone()

    with patch.object(runner.model, "forward", wraps=runner.model.forward) as forward:
        output = runner.decode_batch(torch.tensor([[13], [14]]), ["short", "long"])
    assert forward.call_count == 1
    kwargs = forward.call_args.kwargs
    assert kwargs["input_ids"].shape == (2, 1)
    assert kwargs["position_ids"].tolist() == [[2], [6]]
    mask = kwargs["attention_mask"]
    assert mask.shape == (2, 1, 1, 7)
    assert mask.dtype == runner.spec.dtype
    assert mask[0, 0, 0, :3].eq(0).all()
    assert mask[0, 0, 0, 3:].eq(torch.finfo(mask.dtype).min).all()
    assert mask[1].eq(0).all()
    assert output.logits.shape == (2, 1, 64)
    assert torch.isfinite(output.logits).all()

    # Only each request's one newly reserved position may change. In particular,
    # the four padding columns for "short" must not become persistent KV writes.
    expected_unchanged = runner.cache.pool.pool.clone()
    for request_id in ("short", "long"):
        view = runner.cache.view(request_id)
        block, offset = runner.cache.block_manager.block_tables[request_id].locate(view.num_tokens - 1)
        expected_unchanged[:, :, block, :, offset, :] = before[:, :, block, :, offset, :]
    torch.testing.assert_close(expected_unchanged, before, rtol=0, atol=0)
    assert [runner.cache.view(r).num_tokens for r in ("short", "long")] == [3, 7]
    assert_released(runner)


@torch.inference_mode()
def test_nonparticipants_and_reused_pages_do_not_contaminate_other_requests():
    runner = make_runner()
    sequences = {"a": [1, 3], "b": [2, 4, 6, 8, 10], "idle": [7, 9, 11]}
    for request_id, ids in sequences.items():
        prefill(runner, request_id, ids)
    idle_before = pages(runner, "idle")
    for request_id in ("a", "b"):
        runner.cache.append(request_id)
    runner.decode_batch(torch.tensor([[13], [14]]), ["a", "b"])
    sequences["a"].append(13)
    sequences["b"].append(14)
    torch.testing.assert_close(pages(runner, "idle"), idle_before, rtol=0, atol=0)

    survivor_before = pages(runner, "b")
    released_blocks = set(runner.cache.block_manager.block_tables["a"].physical_blocks)
    runner.cache.release("a")
    sequences["a"] = [31, 29, 27]
    prefill(runner, "a", sequences["a"])
    assert released_blocks & set(runner.cache.block_manager.block_tables["a"].physical_blocks)
    torch.testing.assert_close(pages(runner, "b"), survivor_before, rtol=0, atol=0)
    torch.testing.assert_close(pages(runner, "idle"), idle_before, rtol=0, atol=0)

    # "idle" joins, "a" is a new generation with reused pages, and row order
    # differs from both allocation order and the previous execution batch.
    order = ["idle", "a", "b"]
    tokens = [17, 19, 21]
    for request_id in order:
        runner.cache.append(request_id)
    actual = runner.decode_batch(torch.tensor(tokens).unsqueeze(1), order).logits
    for row, request_id in enumerate(order):
        sequences[request_id].append(tokens[row])
        dense = runner.model(torch.tensor([sequences[request_id]]), use_cache=False).logits[:, -1:, :]
        torch.testing.assert_close(actual[row:row + 1], dense, rtol=1e-5, atol=1e-5)
    assert_released(runner)


@pytest.mark.parametrize("request_ids,shape,error", [
    ([], (0, 1), ValueError),
    (["a", "a"], (2, 1), ValueError),
    (["a", "b"], (1, 1), ValueError),
    (["a"], (1, 2), ValueError),
    (["a"], (1,), ValueError),
    (["a", "missing"], (2, 1), KeyError),
])
@torch.inference_mode()
def test_invalid_batch_is_rejected_before_model_or_cache_writes(request_ids, shape, error):
    runner = make_runner()
    for request_id, ids in (("a", [1, 2]), ("b", [3, 4, 5])):
        prefill(runner, request_id, ids)
        runner.cache.append(request_id)
    before = runner.cache.pool.pool.clone()
    tables = {
        key: (value.num_tokens, tuple(value.physical_blocks))
        for key, value in runner.cache.block_manager.block_tables.items()
    }
    with patch.object(runner.model, "forward", wraps=runner.model.forward) as forward:
        with pytest.raises(error):
            runner.decode_batch(torch.ones(shape, dtype=torch.long), request_ids)
    assert forward.call_count == 0
    torch.testing.assert_close(runner.cache.pool.pool, before, rtol=0, atol=0)
    assert tables == {
        key: (value.num_tokens, tuple(value.physical_blocks))
        for key, value in runner.cache.block_manager.block_tables.items()
    }
    assert_released(runner)
