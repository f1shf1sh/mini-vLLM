"""Training entry point scaffold."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from ..checkpoint import save_checkpoint
from ..config import ProjectConfig, load_config
from ..dataset import TextDataset
from ..model import GPTMini
from ..tokenizer import CharTokenizer
from ..utils import set_seed, timer, tokens_per_second


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training scaffold for lmcore.")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the YAML config file.",
    )
    return parser

def load_text(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Training text is empty.")
    return text

def estimate_loss(
    model: GPTMini,
    dataset: TextDataset,
    eval_iters: int,
) -> dict[str, float]:
    losses: dict[str, float] = {}

    # Switch to eval mode so dropout stays disabled during validation.
    model.eval()
    with torch.no_grad():
        for split in ("train", "val"):
            split_losses = []
            for _ in range(eval_iters):
                x, y = dataset.get_batch(split)
                _, loss = model(x, y)
                split_losses.append(loss.item())
            losses[split] = sum(split_losses) / len(split_losses)
    # Restore train mode before the main optimization loop continues.
    model.train()

    return losses

def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    set_seed(config.training.seed)
    text = load_text(config.paths.input_text)
    tokenizer = CharTokenizer(text)
    token_ids = tokenizer.encode(text)

    # The tokenizer determines the true vocabulary size used by embeddings.
    config.model.vocab_size = tokenizer.vocab_size

    dataset = TextDataset(
        token_ids=token_ids,
        block_size=config.model.block_size,
        batch_size=config.training.batch_size,
        train_split=config.training.train_split,
        device=config.training.device,
    )

    # Build the model only after vocab_size has been filled from the tokenizer.
    model = GPTMini(config=config.model)
    model = model.to(config.training.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    print(f"Config: {args.config}")
    print(f"Device: {config.training.device}")
    print(f"Text length: {len(text)}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(
        "Model summary:"
        f" layers={config.model.n_layer},"
        f" heads={config.model.n_head},"
        f" embd={config.model.n_embd},"
        f" block_size={config.model.block_size}"
    )

    for step in range(config.training.max_iters):
        with timer() as elapsed:
            x, y = dataset.get_batch("train")
            _, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        step_time = elapsed()
        if step % config.training.eval_interval == 0 or step == config.training.max_iters - 1:
            losses = estimate_loss(model, dataset, config.training.eval_iters)
            tokens_seen = config.training.batch_size * config.model.block_size
            tps = tokens_per_second(tokens_seen, step_time)

            print(
                f"step {step:4d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"step time {step_time:.4f}s | "
                f"tokens/s {tps:.2f}"
            )

            ckpt_path = save_checkpoint(
                output_dir=config.paths.output_dir,
                model=model,
                optimizer=optimizer,
                config=config,
                tokenizer=tokenizer,
                step=step,
            )
            print(f"Saved checkpoint to {ckpt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
