"""Synthetic, in-process closed-loop benchmark of the shared Qwen inference engine."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import shlex
import subprocess
import sys
from time import perf_counter_ns
import traceback
import uuid

import torch

from benchmarks.metrics import summarize_run
from llm.inference.request import Request
from llm.inference.sampler import SamplingParams
from llm.inference.scheduler import Scheduler


SYNTHETIC_TEXT = (
    "This is a synthetic inference benchmark. A scientist measures the time needed "
    "to generate tokens while several requests share a language model. "
    "The input is repeated to produce a controlled number of tokens. "
)
REPOSITORY = Path(__file__).resolve().parents[1]


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_records(path: Path, requests: list[Request]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for request in requests:
            record = {
                "request_id": request.request_id,
                "status": request.status.name if request.timing is not None else "NOT_SUBMITTED",
                "input_ids": request.input_ids,
                "generated_ids": request.generated_ids,
                "prompt_tokens": request.prompt_length,
                "output_tokens": request.num_generated_tokens,
                "max_new_tokens": request.max_new_tokens,
                "stop_token_ids": list(request.stop_token_ids),
                "sampling_params": asdict(request.sampling_params) if request.sampling_params else None,
                "timing": asdict(request.timing) if request.timing is not None else None,
            }
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def cache_snapshot(runner) -> dict:
    """Inspect the concrete paged adapter in the benchmark, not in Scheduler."""
    manager = runner.cache.block_manager
    allocator = manager.allocator
    state = {
        "capacity_blocks": runner.cache.capacity,
        "free_blocks": allocator.num_free,
        "free_block_ids": sorted(allocator.free_blocks),
        "allocated_block_ids": sorted(allocator.allocated),
        "active_request_ids": sorted(manager.block_tables),
    }
    state["fully_released"] = (
        state["free_blocks"] == state["capacity_blocks"]
        and state["free_block_ids"] == list(range(state["capacity_blocks"]))
        and not state["allocated_block_ids"]
        and not state["active_request_ids"]
    )
    return state


def run_closed_loop(
    runner,
    requests: list[Request],
    concurrency: int,
    *,
    clock=perf_counter_ns,
    synchronize=lambda: None,
) -> dict:
    """Run a fixed workload; concurrency counts waiting plus running requests.

    Input preparation and the initial CUDA synchronization precede start_ns.
    End_ns follows close() and final CUDA synchronization. Request records are
    serialized later so disk I/O does not enter the measured interval.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    scheduler = Scheduler(runner, record_timing=True, clock=clock)
    before = cache_snapshot(runner)
    if not before["fully_released"]:
        raise ValueError("The runner has outstanding cache allocations before this run")
    synchronize()
    next_request = 0
    peak = 0
    errors = []
    start_ns = clock()
    try:
        while next_request < len(requests) or not scheduler.is_done:
            while next_request < len(requests) and len(scheduler.waiting) + len(scheduler.running) < concurrency:
                scheduler.add_request(requests[next_request])
                next_request += 1
                peak = max(peak, len(scheduler.waiting) + len(scheduler.running))
            if not scheduler.is_done:
                scheduler.step()
    except BaseException as exc:
        errors.append({"stage": "execution", "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
    finally:
        try:
            scheduler.close()
        except BaseException as exc:
            errors.append({"stage": "cleanup", "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        try:
            synchronize()
        except BaseException as exc:
            errors.append({"stage": "synchronize", "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        end_ns = clock()
    after = cache_snapshot(runner)
    if not after["fully_released"]:
        errors.append({"stage": "resource_check", "type": "ResourceLeak", "message": "KV blocks or block tables remain after close()."})
    if not errors and any(request.status.name != "FINISHED" for request in requests):
        errors.append({"stage": "completion_check", "type": "IncompleteWorkload", "message": "Not every planned request finished successfully."})
    return {
        "schema_version": 1,
        "clock": "time.perf_counter_ns",
        "start_ns": start_ns,
        "end_ns": end_ns,
        "run_status": "failed" if errors else "success",
        "errors": errors,
        "cache_before": before,
        "cache_after": after,
        "peak_outstanding": peak,
    }


def make_input_ids(tokenizer, prompt_length: int, count: int, seed: int) -> list[list[int]]:
    """Seeded cyclic slices of real tokenizer IDs; not a natural/chat workload."""
    source = tokenizer.encode(SYNTHETIC_TEXT, add_special_tokens=False)
    if not source:
        raise ValueError("Tokenizer produced no tokens for the synthetic source text")
    rng = random.Random(seed)
    prompts = []
    for _ in range(count):
        offset = rng.randrange(len(source))
        prompts.append([source[(offset + i) % len(source)] for i in range(prompt_length)])
    return prompts


def make_requests(inputs: list[list[int]], maximum: int, stop_ids: tuple[int, ...], prefix: str) -> list[Request]:
    return [
        Request(f"{prefix}-{index:06d}", list(ids), maximum, SamplingParams(do_sample=False), stop_token_ids=stop_ids)
        for index, ids in enumerate(inputs)
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_identity() -> dict:
    """Include untracked benchmark/source files, not only git's file list."""
    paths = sorted(
        list((REPOSITORY / "src" / "llm").rglob("*.py"))
        + list((REPOSITORY / "benchmarks").rglob("*.py"))
        + [REPOSITORY / "pyproject.toml"]
    )
    manifest = {str(path.relative_to(REPOSITORY)): sha256_file(path) for path in paths}
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()

    def git(*args):
        result = subprocess.run(["git", "-C", str(REPOSITORY), *args], text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain")
    return {
        "repository": str(REPOSITORY),
        "commit": git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "git_status": status,
        "source_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_files_sha256": manifest,
    }


def model_identity(directory: Path) -> dict:
    files = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix in {".safetensors", ".bin", ".json", ".model", ".txt", ".tiktoken"}
        and ".cache" not in path.relative_to(directory).parts
    )
    manifest = {
        str(path.relative_to(directory)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    }
    return {
        "local_directory": str(directory),
        "files": manifest,
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }


def runtime_identity(device: torch.device) -> dict:
    import transformers
    result = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["gpu"] = {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "uuid": str(getattr(properties, "uuid", "unavailable")),
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Existing local Qwen2/Qwen2.5 model directory; no download.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup requests before EACH repeat; excluded from measurements.")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eos-policy", choices=("fixed", "model"), default="fixed")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"), help="Parent directory; each invocation creates a unique child.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("num_blocks", "block_size", "prompt_length", "concurrency", "num_requests", "repeat"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_new_tokens", "warmup"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be in [0, 2**63)")
    args.model = args.model.expanduser().resolve()
    if not args.model.is_dir():
        parser.error("--model must be an existing local directory")
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    command = "cd " + shlex.quote(str(Path.cwd())) + "\n" + shlex.join([sys.executable, "-m", "benchmarks.engine", *actual_argv]) + "\n"
    invocation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    experiment = args.output_dir.expanduser().resolve() / invocation_id
    experiment.mkdir(parents=True, exist_ok=False)
    (experiment / "command.txt").write_text(command, encoding="utf-8")
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(experiment / "invocation.json", {"arguments": arguments, "argv": actual_argv, "cwd": str(Path.cwd())})
    print(f"Results: {experiment}", flush=True)
    try:
        from transformers import AutoTokenizer
        from llm.model.qwen2_5_0_5b_instruct.runner import QwenModelRunner
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu for a CPU run")
        torch.manual_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
        runner = QwenModelRunner.from_pretrained(
            str(args.model), device=str(device), dtype=getattr(torch, args.dtype),
            num_blocks=args.num_blocks, block_size=args.block_size,
        )
        device = runner.spec.device
        synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
        eos = runner.model.generation_config.eos_token_id
        if eos is None:
            eos = tokenizer.eos_token_id
        model_stop_ids = () if eos is None else (eos,) if isinstance(eos, int) else tuple(eos)
        stop_ids = model_stop_ids if args.eos_policy == "model" else ()
        inputs = make_input_ids(tokenizer, args.prompt_length, args.num_requests, args.seed)
        warmup_inputs = make_input_ids(tokenizer, args.prompt_length, args.warmup, args.seed + 1)
        maximum_cached_tokens = args.prompt_length + max(args.max_new_tokens - 1, 0)
        required_blocks = runner.cache.required_capacity(maximum_cached_tokens) if args.max_new_tokens else 0
        if maximum_cached_tokens > runner.spec.context_length:
            raise ValueError("The requested prompt/output length exceeds model context length")
        if required_blocks > runner.cache.capacity:
            raise ValueError("A single request exceeds cache capacity; increase --num-blocks")
        base_config = {
            "schema_version": 1,
            "arguments": arguments,
            "command": command,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "code": code_identity(),
            "model": model_identity(args.model),
            "runtime": runtime_identity(device),
            "runner": {"context_length": runner.spec.context_length, "device": str(device), "dtype": str(runner.spec.dtype), "attention_implementation": "eager"},
            "workload": {
                "kind": "synthetic_tokenizer_ids",
                "source_text": SYNTHETIC_TEXT,
                "construction": "Tokenize once without special tokens; take seeded random-offset cyclic slices of exactly prompt_length IDs. No chat template.",
                "input_ids": inputs,
                "warmup_input_ids": warmup_inputs,
                "sampling_params": asdict(SamplingParams(do_sample=False)),
                "stop_token_ids": list(stop_ids),
                "model_eos_token_ids": list(model_stop_ids),
                "eos_policy": args.eos_policy,
            },
            "capacity": {
                "required_blocks_per_request": required_blocks,
                "maximum_cached_tokens_per_request": maximum_cached_tokens if args.max_new_tokens else 0,
                "max_admitted_by_cache": runner.cache.capacity // required_blocks if required_blocks else None,
            },
            "measurement_policy": {
                "concurrency": "waiting + running; one request per Scheduler.step, no tensor batching",
                "interval": "after pre-run CUDA sync, before first submission, through drain/close/final CUDA sync",
                "excluded": ["model loading", "tokenization", "input/Request construction", "warmup", "metadata hashing", "result file writes"],
                "token_ready": "Scheduler timestamp after token.item() and append_token; no additional per-token synchronization",
                "warmup": "warmup requests before each repeat; RNG reseeded before measured requests",
            },
        }
        for repeat_index in range(args.repeat):
            run_dir = experiment / f"repeat-{repeat_index:03d}"
            run_dir.mkdir(exist_ok=False)
            config = {**base_config, "repeat_index": repeat_index}
            write_json(run_dir / "config.json", config)
            (run_dir / "command.txt").write_text(command, encoding="utf-8")
            if args.warmup:
                torch.manual_seed(args.seed)
                warmup_requests = make_requests(warmup_inputs, args.max_new_tokens, stop_ids, "warmup")
                warmup = run_closed_loop(runner, warmup_requests, args.concurrency, synchronize=synchronize)
                write_json(run_dir / "warmup.json", warmup)
                write_records(run_dir / "warmup_requests.jsonl", warmup_requests)
                if warmup["run_status"] != "success":
                    print(f"Warmup failed; raw results preserved in {run_dir}", file=sys.stderr)
                    return 1
            torch.manual_seed(args.seed)
            requests = make_requests(inputs, args.max_new_tokens, stop_ids, "request")
            measurement = run_closed_loop(runner, requests, args.concurrency, synchronize=synchronize)
            write_records(run_dir / "requests.jsonl", requests)
            write_json(run_dir / "measurement.json", measurement)
            summary = summarize_run(run_dir)
            write_json(run_dir / "summary.json", summary)
            result_line = (
                f"repeat={repeat_index} status={summary['run_status']} "
                f"requests/s={summary['throughput']['successful_requests_per_second']:.3f} "
                f"output_tokens/s={summary['throughput']['successful_output_tokens_per_second']:.3f} "
                f"free_blocks={measurement['cache_after']['free_blocks']}/{runner.cache.capacity}"
            )
            (run_dir / "stdout.log").write_text(result_line + "\n", encoding="utf-8")
            (run_dir / "stderr.log").write_text("\n".join(error.get("traceback", error["message"]) for error in measurement["errors"]), encoding="utf-8")
            print(result_line, flush=True)
            if measurement["run_status"] != "success":
                return 1
        return 0
    except BaseException as exc:
        write_json(experiment / "failure.json", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        print(f"Benchmark failed: {exc}. Details: {experiment / 'failure.json'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
