from pathlib import Path

import pytest
import torch
import yaml

from llm.checkpoint import load_checkpoint
from llm.config import ProjectConfig
from llm.model import GPTMini
from llm.runtime.train import main


def test_training_cli_saves_usable_checkpoint_after_one_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_text("hello world\n" * 20, encoding="utf-8")

    config = ProjectConfig()
    config.model.block_size = 4
    config.model.n_layer = 1
    config.model.n_head = 2
    config.model.n_embd = 8
    config.model.dropout = 0.0
    config.training.batch_size = 2
    config.training.max_iters = 1
    config.training.eval_interval = 1
    config.training.eval_iters = 1
    config.training.device = "cpu"
    config.paths.input_text = str(input_path)
    config.paths.output_dir = str(tmp_path / "checkpoints")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")

    assert main(["--config", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "train loss" in output
    assert "val loss" in output

    checkpoint = load_checkpoint(tmp_path / "checkpoints" / "ckpt.pt")
    assert checkpoint["step"] == 0
    optimizer_states = checkpoint["optimizer_state_dict"]["state"]
    assert optimizer_states
    assert all(state["step"].item() == 1 for state in optimizer_states.values())

    restored_config = ProjectConfig.from_dict(checkpoint["config"])
    assert restored_config.model.vocab_size == len(checkpoint["stoi"])
    model = GPTMini(restored_config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    input_ids = torch.tensor([[checkpoint["stoi"][char] for char in "hell"]])
    with torch.no_grad():
        logits, loss = model(input_ids)
    assert logits.shape == (1, 4, restored_config.model.vocab_size)
    assert torch.isfinite(logits).all()
    assert loss is None
