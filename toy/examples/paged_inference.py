"""Run a trained checkpoint through paged prefill, decode, and scheduling.

Example:
    python -m toy.examples.paged_inference --device cuda --json-output outputs/inference-demo.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from toy.checkpoint import load_checkpoint
from toy.config import ProjectConfig
from llm.cache.blocks import BlockAllocator, BlockManager
from llm.cache.pool import KVPool
from toy.runner import ModelRunner
from llm.inference.request import Request, RequestStatus
from llm.inference.sampler import SamplingParams
from llm.inference.scheduler import Scheduler
from toy.model import GPTMini
from toy.learning.char_tokenizer import CharTokenizer


class RecordingAllocator(BlockAllocator):
    """Observe real physical allocations, including allocations freed in one step."""

    def __init__(self, num_blocks: int) -> None:
        super().__init__(num_blocks)
        self.peak_allocated = 0
        self.used_ids: set[int] = set()

    def allocate(self) -> int:
        block_id = super().allocate()
        self.peak_allocated = max(self.peak_allocated, len(self.allocated))
        self.used_ids.add(block_id)
        return block_id

    def reset_observations(self) -> None:
        self.peak_allocated = len(self.allocated)
        self.used_ids.clear()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def assert_recovered(manager: BlockManager, num_blocks: int) -> None:
    assert manager.allocator.num_free == num_blocks, "KV blocks leaked"
    assert not manager.allocator.allocated, "Allocated block IDs remain"
    assert not manager.block_tables, "Per-request block tables remain"
    assert len(set(manager.allocator.free_blocks)) == num_blocks, "Duplicate free blocks"


@torch.inference_mode()
def verify_paged_logits(
    model: GPTMini,
    tokenizer: CharTokenizer,
    pool: KVPool,
    manager: BlockManager,
) -> dict[str, Any]:
    """Compare trained-model logits across a deliberately fragmented KV boundary."""
    device = next(model.parameters()).device
    prompt_length = pool.block_size - 1
    decode_steps = 6
    total_length = prompt_length + decode_steps
    if total_length > model.config.block_size:
        raise ValueError("KV block size is too large for the checkpoint context length.")
    needed_blocks = (total_length + pool.block_size - 1) // pool.block_size + 1
    if needed_blocks > pool.num_blocks:
        raise ValueError(f"Logit verification needs at least {needed_blocks} KV blocks.")

    source_ids = tokenizer.encode("First Citizen:\nROMEO:\nThe king")
    ids = (source_ids * ((total_length + len(source_ids) - 1) // len(source_ids)))[:total_length]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    expected, _ = model(input_ids)
    runner = ModelRunner(model, pool, manager)
    max_error = 0.0
    synchronize(device)
    started = time.perf_counter()
    try:
        manager.allocate_for_request("verify-logits", prompt_length)
        actual = runner.prefill(input_ids[:, :prompt_length], "verify-logits").logits
        torch.testing.assert_close(actual, expected[:, :prompt_length], rtol=1e-4, atol=1e-4)
        max_error = float((actual - expected[:, :prompt_length]).abs().max().item())

        # Occupy the next physical block so the verified request cannot extend
        # contiguously; only its logical-to-physical block table finds the right KV.
        manager.allocate_for_request("verify-blocker", pool.block_size)
        blocker_id = manager.block_tables["verify-blocker"].physical_blocks[0]
        for token_pos in range(prompt_length, total_length):
            manager.append_token("verify-logits")
            actual = runner.decode(input_ids[:, token_pos : token_pos + 1], "verify-logits").logits
            reference = expected[:, token_pos : token_pos + 1]
            torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)
            max_error = max(max_error, float((actual - reference).abs().max().item()))

        physical_blocks = list(manager.block_tables["verify-logits"].physical_blocks)
        assert len(physical_blocks) >= 2, "Decode did not cross a KV block boundary"
        assert blocker_id not in physical_blocks
        assert any(b != a + 1 for a, b in zip(physical_blocks, physical_blocks[1:])), (
            "Verification did not exercise noncontiguous physical blocks"
        )
    finally:
        for request_id in ("verify-logits", "verify-blocker"):
            if request_id in manager.block_tables:
                manager.free_request(request_id)
    synchronize(device)
    assert_recovered(manager, pool.num_blocks)
    return {
        "passed": True,
        "prefill_tokens": prompt_length,
        "decode_steps": decode_steps,
        "physical_blocks": physical_blocks,
        "blocker_physical_block": blocker_id,
        "max_absolute_logit_error": max_error,
        "rtol": 1e-4,
        "atol": 1e-4,
        "free_blocks_after": manager.allocator.num_free,
        "elapsed_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def run_wave(
    scheduler: Scheduler,
    tokenizer: CharTokenizer,
    allocator: RecordingAllocator,
    wave: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    num_blocks: int,
) -> dict[str, Any]:
    prompts = ("First Citizen:", "ROMEO:", "The king")
    requests = [
        Request(
            request_id=f"wave-{wave}-request-{index + 1}",
            input_ids=tokenizer.encode(prompt),
            max_new_tokens=max(1, max_new_tokens - index * 8),
            sampling_params=SamplingParams(temperature=temperature, top_k=20),
        )
        for index, prompt in enumerate(prompts)
    ]
    arrivals = (0, 2, 4)
    finished_at: dict[str, int] = {}
    prefill_calls = 0
    decode_calls = 0
    peak_running = 0
    steps = 0
    finished_before = len(scheduler.finished)
    allocator.reset_observations()
    synchronize(device)
    started = time.perf_counter()
    expected_model_calls = sum(request.max_new_tokens for request in requests)

    for step in range(expected_model_calls + max(arrivals) + 1):
        for arrival, request in zip(arrivals, requests):
            if step == arrival:
                scheduler.add_request(request)
        generated_before = [request.num_generated_tokens for request in requests]
        had_work = not scheduler.is_done
        completed = scheduler.step()
        steps += int(had_work)
        for request, count_before in zip(requests, generated_before):
            if request.num_generated_tokens > count_before:
                assert request.num_generated_tokens == count_before + 1
                if count_before == 0:
                    prefill_calls += 1
                else:
                    decode_calls += 1
        for request in completed:
            finished_at[request.request_id] = step
        peak_running = max(peak_running, len(scheduler.running))
        if step >= max(arrivals) and scheduler.is_done:
            break
    else:
        raise AssertionError("Scheduler did not finish within the expected step bound")

    synchronize(device)
    elapsed = time.perf_counter() - started
    assert scheduler.is_done
    assert len(scheduler.finished) - finished_before == len(requests)
    assert prefill_calls == len(requests)
    assert decode_calls == sum(request.max_new_tokens - 1 for request in requests)
    assert steps == expected_model_calls
    assert_recovered(scheduler.cache.block_manager, num_blocks)
    for request in requests:
        assert request.status is RequestStatus.FINISHED
        assert request.num_generated_tokens == request.max_new_tokens
        assert request.request_id in finished_at

    return {
        "wave": wave,
        "scheduler_steps": steps,
        "prefill_calls": prefill_calls,
        "decode_calls": decode_calls,
        "peak_running_requests": peak_running,
        "peak_allocated_blocks": allocator.peak_allocated,
        "physical_blocks_used": sorted(allocator.used_ids),
        "free_blocks_after": allocator.num_free,
        "elapsed_seconds": elapsed,
        "generated_tokens": expected_model_calls,
        "generated_tokens_per_second": expected_model_calls / elapsed,
        "requests": [
            {
                "request_id": request.request_id,
                "arrival_step": arrival,
                "finish_step": finished_at[request.request_id],
                "status": request.status.name,
                "prompt_tokens": request.prompt_length,
                "requested_new_tokens": request.max_new_tokens,
                "generated_tokens": request.num_generated_tokens,
                "prompt": prompt,
                "generated_text": tokenizer.decode(request.generated_ids),
                "full_text": tokenizer.decode(request.output_ids),
            }
            for request, arrival, prompt in zip(requests, arrivals, prompts)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--ckpt", default="outputs/ckpt.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--kv-block-size", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.kv_block_size < 2 or args.num_blocks < 3:
        parser.error("max-new-tokens >= 1, kv-block-size >= 2, and num-blocks >= 3 are required")
    if args.temperature <= 0:
        parser.error("temperature must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable; use a GPU-enabled container or --device cpu")

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    started = time.perf_counter()
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    config = ProjectConfig.from_dict(checkpoint["config"])
    tokenizer = CharTokenizer.from_vocab(checkpoint["stoi"], checkpoint["itos"])
    config.model.vocab_size = tokenizer.vocab_size
    model = GPTMini(config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    parameter = next(model.parameters())
    pool = KVPool(
        num_blocks=args.num_blocks,
        num_layers=config.model.n_layer,
        num_heads=config.model.n_head,
        block_size=args.kv_block_size,
        head_dim=config.model.n_embd // config.model.n_head,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    allocator = RecordingAllocator(args.num_blocks)
    manager = BlockManager(allocator, args.kv_block_size)
    verification = verify_paged_logits(model, tokenizer, pool, manager)
    scheduler = Scheduler(ModelRunner(model, pool, manager))
    waves = [
        run_wave(scheduler, tokenizer, allocator, wave, args.max_new_tokens,
                 args.temperature, device, args.num_blocks)
        for wave in (1, 2)
    ]
    reused_blocks = sorted(set(waves[0]["physical_blocks_used"]) & set(waves[1]["physical_blocks_used"]))
    assert reused_blocks, "The second request wave did not reuse physical blocks"
    assert len(scheduler.finished) == 6
    assert_recovered(manager, args.num_blocks)
    synchronize(device)
    report = {
        "success": True,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(parameter.device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "dtype": str(parameter.dtype),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "model_context_length": config.model.block_size,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "seed": args.seed,
        "logit_verification": verification,
        "pool": {
            "num_blocks": args.num_blocks,
            "tokens_per_block": args.kv_block_size,
            "bytes": pool.pool.numel() * pool.pool.element_size(),
            "device": str(pool.pool.device),
            "peak_allocated_blocks": max(wave["peak_allocated_blocks"] for wave in waves),
            "free_blocks_after": allocator.num_free,
            "reused_blocks_across_waves": reused_blocks,
            "fully_recovered": True,
        },
        "finished_requests": len(scheduler.finished),
        "generated_tokens": sum(wave["generated_tokens"] for wave in waves),
        "waves": waves,
        "elapsed_seconds": time.perf_counter() - started,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
