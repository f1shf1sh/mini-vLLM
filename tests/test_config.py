from pathlib import Path

from lmcore.config import load_config


def test_load_base_config() -> None:
    config = load_config(Path("configs/base.yaml"))

    assert config.model.block_size == 128
    assert config.training.batch_size == 32
    assert config.paths.input_text == "data/input.txt"
