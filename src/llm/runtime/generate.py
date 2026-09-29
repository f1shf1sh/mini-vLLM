"""Generate with a local Qwen model through the shared request scheduler."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict

import torch
from transformers import AutoTokenizer

from llm.config import InferenceConfig, load_config
from llm.inference.request import Request
from llm.inference.scheduler import Scheduler
from llm.models.qwen.runner import QwenModelRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Inference YAML path.")
    parser.add_argument("--model", help="Local model directory or an already cached model ID.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--num-blocks", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--do-sample", action="store_true", default=None)
    parser.add_argument("--raw", action="store_true", help="Encode plain text without the model's chat template.")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else InferenceConfig()
    values = asdict(config)
    for key in values:
        value = getattr(args, key, None)
        if value is not None:
            values[key] = value
    config = InferenceConfig(**values)
    if not args.prompt:
        parser.error("prompt must not be empty")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; choose --device cpu")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model, local_files_only=True, trust_remote_code=False,
    )
    if not args.raw and tokenizer.chat_template:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True, add_generation_prompt=True,
        )
    else:
        ids = tokenizer.encode(args.prompt)
    runner = QwenModelRunner.from_pretrained(
        config.model, device=config.device, dtype=getattr(torch, config.dtype),
        num_blocks=config.num_blocks, block_size=config.block_size,
    )
    eos = runner.model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    stop_ids = () if eos is None else (eos,) if isinstance(eos, int) else tuple(eos)
    request = Request(
        "generate", ids, config.max_new_tokens, config.sampling_params(),
        stop_token_ids=stop_ids,
    )
    scheduler = Scheduler(runner)
    scheduler.add_request(request)
    scheduler.run()
    print(tokenizer.decode(request.generated_ids, skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
