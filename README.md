# mini-vLLM

A pure-PyTorch teaching implementation of paged KV storage, prefill/decode
execution, and request scheduling. A small GPT model provides a controllable
testbed. Pretrained Qwen loading and continuous batching are future work.

## Layout

- `src/llm/inference/`: block allocation, KV storage, scheduling, sampling.
- `src/llm/model/`: toy GPT model.
- `src/llm/runtime/`: training and generation CLI.
- `tests/`: production regression tests.
- `configs/`: CPU and CUDA training configurations.
- `docker/`: GPU image and container development entry.
- `benchmarks/`: KV memory and scheduler block-usage experiments.
- `examples/`: checkpoint-based paged inference demo.
- `tools/`: dataset preparation.
- `data/` and `outputs/`: datasets and generated artifacts.
- Personal lessons and detailed documentation stay in Git-ignored `.local/`.

## Develop in the existing GPU container

On the Linux host, run from the repository root:

```bash
./docker/dev.sh shell
```

Inside the container:

```bash
python -m pytest tests -q
llm-train --config configs/base.yaml
llm-generate --ckpt outputs/ckpt.pt --prompt "hello"
python benchmarks/kv_memory.py
python benchmarks/block_usage.py
```

Or run one command from the host: `./docker/dev.sh python -m pytest tests -q`.
The helper starts a stopped container, but never creates one or installs packages.
Set `MINIVLLM_CONTAINER` to select a container other than `mini-vllm-dev`.

## Build an environment for a new container

Skip this section when the existing environment already works.
Requires Docker with NVIDIA GPU support. Use an unused container name.

```bash
docker build -f docker/Dockerfile \
  --build-arg DEV_UID="$(id -u)" \
  --build-arg DEV_GID="$(id -g)" \
  -t mini-vllm:dev .

mkdir -p "$HOME/models"
docker run -d --name mini-vllm-dev --gpus device=0 --shm-size=2g \
  --mount "type=bind,src=$PWD,dst=/workspace/mini-vLLM" \
  --mount "type=bind,src=$HOME/models,dst=/models,readonly" \
  --restart unless-stopped mini-vllm:dev
```

`docker/Dockerfile` owns container versions and installation. It inherits the base
image's CUDA PyTorch, installs the project in `/opt/venv`, and creates a user
matching the host UID/GID with writable `/home/dev`.
`pyproject.toml` declares Python package metadata and dependency requirements;
`uv.lock` is retained for standalone uv workflows, not used to replace container
PyTorch. The development Python version is 3.12 (package minimum: 3.11).
Rebuild the image after changing container dependencies.

Keep model weights outside the Git repository. The mount above exposes
`$HOME/models/Qwen2.5-0.5B-Instruct` as `/models/Qwen2.5-0.5B-Instruct`.
An existing container needs recreation to add a new bind mount; retain its
environment and data until the replacement has been verified.

## Attach an editor

Connect to the Linux host with VS Code Remote - SSH, then run
`Dev Containers: Attach to Running Container...` and open
`/workspace/mini-vLLM`. Select `/opt/venv/bin/python`.

For the existing container, save these values through
`Dev Containers: Open Named Configuration File`:

```json
{
  "workspaceFolder": "/workspace/mini-vLLM",
  "settings": {
    "python.defaultInterpreterPath": "/opt/venv/bin/python"
  },
  "remoteEnv": {
    "HOME": "/home/dev",
    "PATH": "/opt/venv/bin:${containerEnv:PATH}",
    "PYTHONDONTWRITEBYTECODE": "1"
  }
}
```

## CUDA training and inference demo

On the host, download the dataset and run the workload in the container:

```bash
./tools/download_data.sh
./docker/dev.sh llm-train --config configs/cuda.yaml
./docker/dev.sh python examples/paged_inference.py --device cuda
```

Training writes `outputs/ckpt.pt`; select another output directory in the
configuration to retain an earlier checkpoint. The character tokenizer and toy
GPT are not compatible with Qwen pretrained weights.

This project gathers paged K/V for attention computation; it does not implement
custom CUDA kernels or production serving.

Developed with AI assistance.
