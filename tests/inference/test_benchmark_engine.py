from itertools import count
import json

import pytest
import torch

from benchmarks.engine import make_input_ids, make_requests, run_closed_loop, write_records
from benchmarks.metrics import summarize_requests
from llm.cache.blocks import BlockAllocator, BlockManager
from llm.cache.paged import PagedCacheResources
from llm.cache.pool import KVPool
from llm.inference.runner import RunnerSpec
from llm.inference.types import RunnerOutput


class TinyRunner:
    def __init__(self, fail_on=None):
        self.spec = RunnerSpec(64, torch.device("cpu"), torch.float32)
        self.cache = PagedCacheResources(
            KVPool(16, 1, 1, 4, 1), BlockManager(BlockAllocator(16), 4),
        )
        self.fail_on = fail_on
        self.calls = []

    def prefill(self, input_ids, request_id):
        self.calls.append(("prefill", request_id))
        if request_id == self.fail_on:
            raise RuntimeError("intentional forward failure")
        return RunnerOutput(torch.tensor([[[0., 1., 2., 3.]]]))

    def decode(self, input_ids, request_id):
        self.calls.append(("decode", request_id))
        return RunnerOutput(torch.tensor([[[0., 1., 2., 3.]]]))


def test_closed_loop_caps_outstanding_refills_and_releases_resources(tmp_path):
    runner = TinyRunner()
    requests = make_requests([[1, 2]] * 5, 3, (), "r")
    synchronization = []
    result = run_closed_loop(
        runner, requests, 2,
        clock=count(1_000_000_000, 1_000_000).__next__,
        synchronize=lambda: synchronization.append(len(runner.calls)),
    )
    assert result["run_status"] == "success"
    assert result["peak_outstanding"] == 2
    assert result["cache_after"]["fully_released"]
    assert synchronization == [0, 15]
    assert [r.generated_ids for r in requests] == [[3, 3, 3]] * 5
    # The next request is submitted only when an outstanding request has completed.
    assert requests[2].timing.submitted_at_ns >= requests[0].timing.finished_at_ns
    write_records(tmp_path / "requests.jsonl", requests)
    records = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    summary = summarize_requests(records, result["start_ns"], result["end_ns"])
    assert summary["counts"]["successful_requests"] == 5
    assert summary["counts"]["successful_output_tokens"] == 15


def test_failure_preserves_partial_tokens_cancels_pending_and_leaves_unsubmitted(tmp_path):
    runner = TinyRunner(fail_on="r-000001")
    requests = make_requests([[1]] * 5, 3, (), "r")
    result = run_closed_loop(runner, requests, 3, clock=count(1_000_000_000, 1_000_000).__next__)
    assert result["run_status"] == "failed"
    assert result["errors"][0]["type"] == "RuntimeError"
    assert result["cache_after"]["fully_released"]
    write_records(tmp_path / "requests.jsonl", requests)
    records = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert [r["status"] for r in records] == ["CANCELLED", "FAILED", "CANCELLED", "NOT_SUBMITTED", "NOT_SUBMITTED"]
    assert records[0]["generated_ids"] == [3]
    summary = summarize_requests(records, result["start_ns"], result["end_ns"])
    assert summary["counts"]["unsuccessful_output_tokens"] == 1
    assert summary["throughput"]["successful_output_tokens_per_second"] == 0


@pytest.mark.parametrize("maximum,stop_ids,expected", [(0, (), []), (4, (3,), [3]), (4, (), [3] * 4)])
def test_zero_output_eos_and_fixed_length_modes(maximum, stop_ids, expected):
    runner = TinyRunner()
    requests = make_requests([[1, 2]], maximum, stop_ids, "r")
    result = run_closed_loop(runner, requests, 1)
    assert result["run_status"] == "success"
    assert requests[0].generated_ids == expected
    assert result["cache_after"]["fully_released"]
    if not maximum:
        assert requests[0].timing.admitted_at_ns is None
        assert not runner.calls


def test_synthetic_inputs_have_exact_length_and_seeded_variation():
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert not add_special_tokens
            return list(range(20))

    tokenizer = Tokenizer()
    first = make_input_ids(tokenizer, 31, 7, 0)
    assert first == make_input_ids(tokenizer, 31, 7, 0)
    assert first != make_input_ids(tokenizer, 31, 7, 1)
    assert all(len(ids) == 31 for ids in first)


def test_raw_files_are_never_overwritten(tmp_path):
    target = tmp_path / "requests.jsonl"
    write_records(target, [])
    with pytest.raises(FileExistsError):
        write_records(target, [])


def test_resource_check_rejects_duplicate_free_ids_with_matching_count():
    from benchmarks.engine import cache_snapshot
    runner = TinyRunner()
    free = runner.cache.block_manager.allocator.free_blocks
    free.pop()
    free.append(0)
    assert len(free) == runner.cache.capacity
    assert not cache_snapshot(runner)["fully_released"]
    with pytest.raises(ValueError, match="outstanding cache"):
        run_closed_loop(runner, make_requests([[1]], 1, (), "r"), 1)


def test_cpu_cli_runs_local_tiny_qwen_and_recomputes_each_repeat(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from benchmarks.engine import main
    from benchmarks.metrics import summarize_run

    model_dir = tmp_path / "tiny-qwen"
    tokenizer_backend = Tokenizer(models.WordLevel(
        {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "This": 3, "is": 4, "a": 5, "synthetic": 6, "benchmark": 7},
        unk_token="[UNK]",
    ))
    tokenizer_backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    config = Qwen2Config(
        vocab_size=8, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=32, bos_token_id=1, eos_token_id=2,
    )
    torch.manual_seed(0)
    Qwen2ForCausalLM(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    output = tmp_path / "results"
    assert main([
        "--model", str(model_dir), "--device", "cpu", "--dtype", "float32",
        "--num-blocks", "8", "--block-size", "2", "--prompt-length", "3",
        "--max-new-tokens", "2", "--num-requests", "3", "--concurrency", "2",
        "--warmup", "1", "--repeat", "2", "--output-dir", str(output),
    ]) == 0
    experiments = list(output.iterdir())
    assert len(experiments) == 1
    repeats = sorted(experiments[0].glob("repeat-*"))
    assert len(repeats) == 2
    for directory in repeats:
        saved = json.loads((directory / "summary.json").read_text())
        assert saved == summarize_run(directory)
        assert saved["run_status"] == "success"
        assert saved["counts"]["successful_output_tokens"] == 6
        assert saved["cache_after"]["fully_released"]
        actual = json.loads((directory / "config.json").read_text())
        assert "benchmarks/engine.py" in actual["code"]["source_files_sha256"]
        assert actual["workload"]["eos_policy"] == "fixed"
        assert len(actual["workload"]["input_ids"]) == 3
        assert "model.safetensors" in actual["model"]["files"]
        assert (directory / "command.txt").exists()
        assert (directory / "warmup_requests.jsonl").exists()
