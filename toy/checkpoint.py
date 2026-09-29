"""Checkpoint helpers shared by training and generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import ProjectConfig
from .model import GPTMini
from .tokenizer import CharTokenizer


def save_checkpoint(
    output_dir: str | Path,
    model: GPTMini,
    optimizer: torch.optim.Optimizer,
    config: ProjectConfig,
    tokenizer: CharTokenizer,
    step: int,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_path / "ckpt.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.to_dict(),
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
            "step": step,
        },
        ckpt_path,
    )

    return ckpt_path


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location=device)
