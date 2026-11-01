# 快速开始

[项目首页](../../README.zh-CN.md) | [English](../en/getting-started.md)

## 本地运行

使用 Python 3.11+，在仓库根目录运行命令。GPU 推理需要兼容的 PyTorch/CUDA 环境；
依赖声明见 `pyproject.toml`。

在新建的 Python 环境中安装项目及其依赖：

```bash
python -m pip install -e .
```

如果已有环境已满足项目依赖，只需注册项目，保留现有运行库：

```bash
python -m pip install --no-deps -e .
```

将已有的模型权重与 tokenizer 文件放入 `models/Qwen2.5-0.5B-Instruct/`，
或将 `--model` 改为自己的模型目录：

```bash
python -m llm.runtime.generate \
  --config configs/qwen.yaml \
  --model models/Qwen2.5-0.5B-Instruct \
  --prompt 'What is 2 + 2?' \
  --max-new-tokens 16
```

`configs/qwen.yaml` 默认使用容器路径 `/models/Qwen2.5-0.5B-Instruct`，
本地运行时通过 `--model` 覆盖。项目安装后，也可以用 `llm-generate`
替代 `python -m llm.runtime.generate`。没有可用 GPU 时可加 `--device cpu`。

生成入口默认使用模型的聊天模板和 greedy 采样；`--raw` 改为普通文本输入。
支持 `--do-sample`、`--temperature`、`--top-k`、`--top-p`，以及设备、
精度和缓存容量配置。完整参数见 `python -m llm.runtime.generate --help`。
模型和 tokenizer 均只从本地加载，不会自动下载缺失文件。

## 使用现有 GPU 容器

在 Linux 仓库根目录运行：

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

`docker/dev.sh` 启动或使用已有容器，不安装依赖、不创建替代环境。
`MINIVLLM_CONTAINER` 可指定容器名称；容器内解释器为 `/opt/venv/bin/python`。
模型需已挂载在 `/models/Qwen2.5-0.5B-Instruct`，或显式覆盖路径。

已有真实模型验证使用 CUDA float32。Transformers 4.51.3 的 eager attention
可能对此 checkpoint 发出 sliding-window 警告；其实际配置为 `use_sliding_window: false`。
