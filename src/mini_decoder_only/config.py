"""Configuration helpers for the project scaffold."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 0
    block_size: int = 128
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_iters: int = 5000
    eval_interval: int = 200
    eval_iters: int = 50
    train_split: float = 0.9
    device: str = "cpu"
    seed: int = 42


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int | None = None


@dataclass(slots=True)
class PathsConfig:
    input_text: str = "data/input.txt"
    output_dir: str = "outputs"


@dataclass(slots=True)
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectConfig":
        return cls(
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            generation=GenerationConfig(**data.get("generation", {})),
            paths=PathsConfig(**data.get("paths", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ProjectConfig:
    """Load a YAML config file into a typed project config."""

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}.")
    return ProjectConfig.from_dict(raw)
