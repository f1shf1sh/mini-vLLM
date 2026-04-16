"""Configuration helpers for the project scaffold."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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
    raw = _load_simple_yaml(config_path.read_text(encoding="utf-8"))
    return ProjectConfig.from_dict(raw)


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by configs/base.yaml.

    Supported features:
    - nested dictionaries via indentation
    - integers and floats
    - booleans
    - null
    - plain strings
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"Invalid config line: {raw_line!r}")

        while indent <= stack[-1][0]:
            stack.pop()

        current = stack[-1][1]
        clean_value = value.strip()

        if not clean_value:
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
            continue

        current[key] = _parse_scalar(clean_value)

    return root


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value
