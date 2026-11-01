# Getting started

[Project home](../../README.md) | [中文](../zh-CN/getting-started.md)

## Run locally

Use Python 3.11+ and run commands from the repository root. GPU inference requires
a compatible PyTorch/CUDA environment. Dependencies are declared in `pyproject.toml`.

In a new Python environment, install the project and its dependencies:

```bash
python -m pip install -e .
```

If an existing environment already satisfies the dependencies, register the
project without replacing runtime libraries:

```bash
python -m pip install --no-deps -e .
```

Place existing model weights and tokenizer files in
`models/Qwen2.5-0.5B-Instruct/`, or point `--model` to your model directory:

```bash
python -m llm.runtime.generate \
  --config configs/qwen.yaml \
  --model models/Qwen2.5-0.5B-Instruct \
  --prompt 'What is 2 + 2?' \
  --max-new-tokens 16
```

`configs/qwen.yaml` defaults to the container path
`/models/Qwen2.5-0.5B-Instruct`; override it with `--model` for local execution.
After installation, `llm-generate` can replace `python -m llm.runtime.generate`.
Add `--device cpu` when no GPU is available.

Generation uses the model's chat template and greedy sampling by default;
`--raw` selects plain-text input. Options include `--do-sample`, `--temperature`,
`--top-k`, `--top-p`, device, dtype, and cache capacity. See
`python -m llm.runtime.generate --help` for the complete interface.
Models and tokenizers are loaded locally; missing files are not downloaded.

## Use the existing GPU container

Run from the Linux checkout root:

```bash
./docker/dev.sh shell
./docker/dev.sh python -m pytest tests toy/tests -q
./docker/dev.sh llm-generate --config configs/qwen.yaml \
  --prompt 'What is 2 + 2?' --max-new-tokens 16
./docker/dev.sh python examples/qwen_inference.py --max-new-tokens 16
./docker/dev.sh python benchmarks/kv_memory.py

./docker/dev.sh env QWEN_MODEL_PATH=/models/Qwen2.5-0.5B-Instruct \
  python -m pytest tests/models/test_qwen_weights.py -q -s
```

`docker/dev.sh` starts or uses the existing container. It does not install
dependencies or create a replacement environment. `MINIVLLM_CONTAINER` selects
the container name; the interpreter is `/opt/venv/bin/python`. Weights must
already be mounted at `/models/Qwen2.5-0.5B-Instruct`, or the path must be
overridden explicitly.

Existing real-model validation uses CUDA float32. Transformers 4.51.3 eager
attention may emit a sliding-window warning for this checkpoint; its actual
configuration has `use_sliding_window: false`.
