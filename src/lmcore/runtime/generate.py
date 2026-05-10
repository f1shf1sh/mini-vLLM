"""Generation entry point for the minimal decoder-only Transformer."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch

from ..checkpoint import load_checkpoint
from ..config import ProjectConfig
from ..inference.generator import generate_tokens
from ..inference.sampler import SamplingParams
from ..model import GPTMini
from ..tokenizer import CharTokenizer
from ..utils import timer, tokens_per_second


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text with lmcore.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation length.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=None, help="Optional top-k sampling.")
    parser.add_argument(
        "--device",
        default=None,
        help="Override device from checkpoint config, e.g. cpu or cuda.",
    )
    return parser

def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.prompt:
        raise ValueError("prompt must not be empty.")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")
    if args.top_k is not None and args.top_k < 1:
        raise ValueError("top_k must be positive when provided.")

    ckpt = load_checkpoint(args.ckpt, device="cpu")
    config = ProjectConfig.from_dict(ckpt["config"])

    device = args.device or config.training.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")

    tokenizer = CharTokenizer.from_vocab(ckpt["stoi"], ckpt["itos"])
    config.model.vocab_size = tokenizer.vocab_size
    model = GPTMini(config.model)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    
    # close dropout
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with timer() as elapsed:
        out = generate_tokens(
            model=model,
            input_ids=idx,
            max_new_tokens=args.max_new_tokens,
            sampling_params=SamplingParams(
                temperature=args.temperature,
                top_k=args.top_k,
            ),
        )
    elapsed_seconds = elapsed()

    generated_ids = out[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    new_tokens = max(len(generated_ids) - len(prompt_ids), 0)
    tps = tokens_per_second(new_tokens, elapsed_seconds)

    print(f"Checkpoint: {args.ckpt}")
    print(f"Device: {device}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Generated text: {generated_text}")
    print(f"Elapsed: {elapsed_seconds:.4f}s")
    print(f"tokens/s: {tps:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
