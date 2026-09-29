from pathlib import Path

import torch

from toy.checkpoint import load_checkpoint, save_checkpoint
from toy.config import ProjectConfig
from toy.model import GPTMini
from toy.learning.char_tokenizer import CharTokenizer


def make_project_config() -> ProjectConfig:
    config = ProjectConfig()
    config.model.vocab_size = 11
    config.model.block_size = 8
    config.model.n_layer = 1
    config.model.n_head = 1
    config.model.n_embd = 8
    config.model.dropout = 0.0
    config.model.bias = False
    config.training.batch_size = 2
    config.training.learning_rate = 1e-3
    config.training.weight_decay = 0.0
    config.training.max_iters = 1
    config.training.eval_interval = 1
    config.training.eval_iters = 1
    config.training.train_split = 0.9
    config.training.device = "cpu"
    config.training.seed = 42
    return config


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = make_project_config()
    tokenizer = CharTokenizer("hello world")
    config.model.vocab_size = tokenizer.vocab_size

    model = GPTMini(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    ckpt_path = save_checkpoint(
        output_dir=tmp_path,
        model=model,
        optimizer=optimizer,
        config=config,
        tokenizer=tokenizer,
        step=7,
    )

    ckpt = load_checkpoint(ckpt_path, device="cpu")

    assert ckpt_path.exists()
    assert ckpt["step"] == 7
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert ckpt["stoi"] == tokenizer.stoi
