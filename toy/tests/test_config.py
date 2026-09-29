from pathlib import Path

from toy.config import load_config


def test_load_base_config() -> None:
    config = load_config(Path("toy/configs/base.yaml"))

    assert config.model.block_size == 64
    assert config.training.batch_size == 16
    assert config.paths.input_text == "data/input.txt"
