# mini-vLLM

A PyTorch teaching project with a shared request scheduler, standard paged KV
storage, and model adapters. The `llm` package runs local Qwen2.5-0.5B-Instruct
weights through Transformers forward. The repository-local `toy` module keeps
the small GPT and its character-model training workflow.

The implementation gathers paged K/V into contiguous tensors for attention.
It does **not** implement a high-performance paged-attention kernel. Scheduling
interleaves one request per step; tensor-level continuous batching is future work.

## Layout and migration

```text
src/llm/
  cache/             blocks.py, pool.py, paged.py
  inference/         runner.py, scheduler.py, request.py, sampler.py, types.py
  models/qwen/       runner.py, cache.py
  runtime/generate.py
  config.py          real-model inference configuration

toy/
  model/             GPTMini, attention, block
  runner.py          adapter to the shared inference engine
  generator.py       character-model generation helper
  config.py, checkpoint.py, dataset.py, utils.py
  train.py, generate.py
  learning/char_tokenizer.py
  configs/, tests/, examples/, benchmarks/, tools/

tests/               cache/, inference/, models/
configs/             qwen.yaml
examples/            qwen_inference.py
benchmarks/          kv_memory.py (model-independent cache arithmetic)
docker/              Dockerfile, dev.sh
```

| Previous location | Current location |
| --- | --- |
| `llm.inference.block_manager` | `llm.cache.blocks` |
| `llm.inference.kv_pool` | `llm.cache.pool` and `llm.cache.paged` |
| `llm.inference.hf_cache`, `hf_model_runner` | `llm.models.qwen.cache`, `runner` |
| `llm.model`, training config/dataset/checkpoint | corresponding modules under `toy` |
| `llm.inference.model_runner`, `generator` | `toy.runner`, `toy.generator` |
| `llm.tokenizer.CharTokenizer` | `toy.learning.char_tokenizer.CharTokenizer` |
| `llm.runtime.train`, old toy generate | `toy.train`, `toy.generate` |
| training YAML, toy examples/tools/block benchmark | corresponding directories under `toy` |

The import package remains `llm`. Wheels and source distributions include the
inference package; `toy` runs from the checkout and is not an installed package.
Data and generated artifacts stay in ignored `data/` and `outputs/`; migration
does not move or overwrite existing checkpoints or datasets. Personal lessons
and detailed validation records remain in ignored `.local/`.

## Inference boundaries

`Scheduler(runner)` handles FIFO admission, round-robin decode, sampling,
termination, and resource cleanup. It reserves enough capacity for admitted
requests to finish. `RunnerSpec` exposes context length, device, and dtype;
`prefill` and `decode` return public `RunnerOutput` logits shaped `[1, T, V]`.
Adapters parse model-specific cache dimensions and construct resource objects.
The scheduler sees capacity, allocation, growth, and release operations, not a
fixed KV tensor layout.

- `BlockManager` executes block allocation, block-table updates, and reclamation.
- `KVPool` owns physical tensors; `PagedKVCache` maps logical positions and
  performs reads, writes, and gather.
- `PagedCacheResources` exposes this storage's lifecycle to the scheduler.
- `toy.runner.ModelRunner` adapts GPTMini; `QwenModelRunner` adapts Transformers
  forward. Qwen uses **KV heads**, separately from query heads.
- `QwenPagedCache` is a per-forward Transformers view. It writes already reserved
  slots and does not allocate/free requests or run its own generation loop.

The final sampled token is returned without another forward, so the maximum
cached length is `prompt_length + max_new_tokens - 1`. Requests release storage
on length/EOS completion, cancellation, and failure. `run()` cleans pending work
on failure; callers stepping manually can call `close()` to cancel remaining work.
Public cache code does not depend on scheduling or models. The shipped `llm`
package does not import toy training code.

For future MoE models with standard KV, expert routing and expert execution stay
in the model adapter/model layer; request scheduling remains unchanged. MLA and
other special cache representations are a separate extension point: their storage
and forward view need an appropriate adapter behind the resource boundary.
No MoE operators, expert parallelism, or plugin registry are implemented now.

## Run in the existing GPU container

From the Linux checkout `/home/yzl/mini-vLLM`:

```bash
./docker/dev.sh shell
./docker/dev.sh python -m pytest tests toy/tests -q
./docker/dev.sh llm-generate --config configs/qwen.yaml \
  --prompt 'What is 2 + 2?' --max-new-tokens 16
./docker/dev.sh python examples/qwen_inference.py --max-new-tokens 16
./docker/dev.sh python benchmarks/kv_memory.py
```

`llm-generate` is the real-model CLI. It applies the model's own chat template
unless `--raw` is supplied, uses greedy decoding by default, and supports
`--do-sample`, `--temperature`, `--top-k`, and `--top-p`. Both model and tokenizer
loading are offline (`local_files_only=True`). No missing model is downloaded.
The default path is `/models/Qwen2.5-0.5B-Instruct`; override `--model`, `--device`,
`--dtype`, `--num-blocks`, or `--block-size` as needed.

Run the explicit mounted-weight correctness check:

```bash
./docker/dev.sh env QWEN_MODEL_PATH=/models/Qwen2.5-0.5B-Instruct \
  python -m pytest tests/models/test_qwen_weights.py -q -s
```

Without that environment variable the mounted-weight test skips; tiny Qwen tests
still run without downloads. It checks GQA, fragmented block boundaries, logits,
two interleaved greedy requests, and complete resource reclamation. Float32 on
CUDA is the validated real-model configuration. The upstream Transformers 4.51.3
eager-attention warning about sliding windows can appear for this checkpoint;
its actual configuration has `use_sliding_window: false`.

## Keep using the toy model

Run from the repository root, inside the container or through `docker/dev.sh`:

```bash
python -m toy.train --config toy/configs/base.yaml
python -m toy.generate --ckpt outputs/ckpt.pt --prompt 'First Citizen:' \
  --device cuda --max-new-tokens 32
python -m toy.examples.paged_inference --device cuda
python -m toy.benchmarks.block_usage
```

For the existing TinyShakespeare workflow, use `toy/configs/cuda.yaml`. If its
input data is absent, explicitly run `bash toy/tools/download_data.sh` from the
host to fetch it. This is an optional existing workflow, not a prerequisite for
Qwen inference. Training writes `ckpt.pt` to `paths.output_dir`; choose a new
output directory to preserve an earlier checkpoint.

The old character checkpoints retain their vocabulary and dictionary format.
They are loaded by `toy.generate`, not the real-model `llm-generate` CLI. A
successful short training run only proves the workflow, not conversational quality.

## Future toy tokenizer, conversation training, and export

The preserved handwritten tokenizer lives in `toy/learning/char_tokenizer.py`.
Future implementation locations are `toy/tokenizer.py` for Hugging Face
`tokenizers`, `toy/chat.py` for conversation formatting, and `toy/export.py` for
inference exports. These files are planned, not empty placeholder implementations.
This round does not download new dialogue datasets or train a new chat model.

Dialogue datasets will be normalized to `messages`. Training and inference must
share the chat template, role/turn boundaries, generation-start marker, and
special token conventions (including BOS/EOS/PAD and their IDs).

A **training checkpoint** supports resuming training and can contain weights,
configuration, optimizer state, and training step. An **inference export** should
contain model configuration, weights, tokenizer, chat template, special token
IDs, and a format version. A new vocabulary must be paired with weights trained
for that vocabulary. Existing character checkpoints remain usable with their
original tokenizer; they are not automatically converted or overwritten.

## Container and package maintenance

`docker/dev.sh` starts or executes the existing container; it does not install
packages or create a new environment. `MINIVLLM_CONTAINER` selects another
container name. `/opt/venv/bin/python` is the intended interpreter. The existing
validation environment is Python 3.12, CUDA PyTorch 2.11.0+cu128, Transformers
4.51.3, and tokenizers 0.21.4. Package support remains Python 3.11+.

`docker/Dockerfile` owns container version pins and reuses the CUDA PyTorch base.
`pyproject.toml` and `uv.lock` describe package/standalone workflows. Do not run
`uv sync` over the active container environment. Refresh editable package metadata
without replacing runtime dependencies when CLI declarations change:

```bash
python -m pip install --no-deps -e .
```

For a **new** container only (keep the current container and weights):

```bash
docker build -f docker/Dockerfile \
  --build-arg DEV_UID="$(id -u)" --build-arg DEV_GID="$(id -g)" \
  -t mini-vllm:dev .

docker run -d --name mini-vllm-dev-new --gpus device=0 --shm-size=2g \
  --mount "type=bind,src=$PWD,dst=/workspace/mini-vLLM" \
  --mount "type=bind,src=$HOME/models,dst=/models,readonly" \
  --restart unless-stopped mini-vllm:dev
```

Use an unused container name. Datasets and weights are not baked into the image.
For VS Code Remote SSH, attach to the existing container, open
`/workspace/mini-vLLM`, and select `/opt/venv/bin/python`. Keep container setup in
`docker/`, not benchmark scripts. Personal learning tests run explicitly, e.g.
`python -m pytest .local/learning/lab03_paged_cache -q`.

Developed with AI assistance.
