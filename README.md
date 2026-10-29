# mini-vLLM

A PyTorch teaching project with a shared request scheduler, standard paged KV
storage, and model adapters. The `llm` package runs local Qwen2.5-0.5B-Instruct
weights through Transformers forward. The repository-local `toy` module keeps
the small GPT and its character-model training workflow.

The implementation gathers paged K/V into contiguous tensors for attention.
It does **not** implement a high-performance paged-attention kernel. Scheduling
interleaves one request per step; tensor-level continuous batching is future work.

## Layout and model adapters

```text
src/llm/
  cache/                        blocks.py, pool.py, paged.py
  inference/                    runner.py, scheduler.py, request.py, sampler.py, types.py
  model/
    gpt_mini/                   runner.py
    qwen2_5_0_5b_instruct/       runner.py, cache.py
  runtime/generate.py
  config.py                     real-model inference configuration

toy/
  model/                        GPTMini, attention, block
  tokenizer.py                  handwritten character tokenizer for learning
  generator.py                  direct model generation, without Scheduler
  config.py, checkpoint.py, dataset.py, utils.py
  train.py, generate.py
  configs/, tests/, tools/

tests/                          cache/, inference/, models/
configs/                        qwen.yaml
examples/                       qwen_inference.py
benchmarks/                     kv_memory.py (shared cache arithmetic)
docker/                         Dockerfile, dev.sh
```

Each model adapter directory uses the full model name, normalized to a Python
identifier: **GPTMini** becomes `gpt_mini`, and **Qwen2.5-0.5B-Instruct** becomes
`qwen2_5_0_5b_instruct`. Model-specific code belongs there; future models add their
own adapter directory. There is no registration system or model factory.

| Previous location | Current location |
| --- | --- |
| `llm.inference.block_manager` | `llm.cache.blocks` |
| `llm.inference.kv_pool` | `llm.cache.pool` and `llm.cache.paged` |
| `llm.models.qwen` | `llm.model.qwen2_5_0_5b_instruct` |
| `toy.runner.ModelRunner` | `llm.model.gpt_mini.runner.GPTMiniRunner` |
| `toy.learning.char_tokenizer.CharTokenizer` | `toy.tokenizer.CharTokenizer` |
| toy runner integration tests | `tests/models/test_gpt_mini_runner.py` |
| `toy/examples/`, `toy/benchmarks/` | removed; future engine benchmarks belong in root `benchmarks/` |

The import package remains `llm`. Wheels and source distributions include only
the inference package and metadata. `toy` runs from the checkout. The GPTMini
adapter accepts a model object supplied by the caller and does not import toy
training code. Data and generated artifacts remain in ignored `data/` and
`outputs/`; existing checkpoints and datasets are preserved. Personal lessons
and detailed validation records stay in ignored `.local/`.

## Runner and direct model execution

The public **Runner abstraction is retained** in `llm.inference.runner`.
`Scheduler(runner)` receives an implementation from the appropriate model adapter
and handles FIFO admission, round-robin decode, sampling, termination, and resource
cleanup. Adding a model requires implementing the Runner contract in its model
adapter and passing that instance to Scheduler; the public scheduler does not
need model-specific branches.

`RunnerSpec` exposes context length, device, and dtype. `prefill` and `decode`
return `RunnerOutput` logits shaped `[1, T, V]`. Adapters parse model-specific KV
specifications and construct cache resources. Scheduler uses capacity,
allocation, growth, and release operations without inspecting KV tensor layout.

- `BlockManager` executes block ownership and block-table updates.
- `KVPool` owns tensors; `PagedKVCache` maps positions and performs read/write/gather.
- `PagedCacheResources` exposes this storage's lifecycle to Scheduler.
- `GPTMiniRunner` adapts a supplied GPTMini model to mini-vLLM.
- `QwenModelRunner` adapts Transformers forward and uses KV heads separately
  from query heads. `QwenPagedCache` only writes already reserved slots; it does
  not own request allocation/release or an independent generation loop.

The final sampled token is returned without another forward, so maximum cached
length is `prompt_length + max_new_tokens - 1`. Storage is released on completion,
EOS, cancellation, or failure. `cancel_request(request_id)` cancels one waiting
or running request between steps and returns whether an active request was found.
It preserves other requests and returns `False` for unknown or terminal IDs.
`run()` cleans pending work on failure; callers stepping manually can call
`close()` to cancel remaining work.

`Scheduler(runner, record_timing=True)` records optional per-request monotonic
nanosecond timestamps in `request.timing`: submission, admission, each token's
CPU readiness, and completion after cleanup. Timing is disabled by default.
Zero-output requests have no admission/token timestamps, and cancelled or failed
requests retain any timestamps already recorded. Statistics belong to the
benchmark tooling, not the scheduler or cache.

For direct execution, `python -m toy.generate` calls GPTMini on the full prefix
for each new token. It does not construct a Runner, Scheduler, or paged cache.
It only reuses the stateless public sampler. The direct path is for running and
learning the toy model; engine integration correctness is tested in `tests/models`.
Future benchmarks target mini-vLLM in root `benchmarks/`, rather than maintaining
a separate toy benchmark/example hierarchy.

For future standard-KV MoE, expert routing and execution stay in the model layer.
MLA and other special cache representations are a separate extension point whose
storage and forward view require a suitable adapter behind the resource boundary.
No MoE operators, expert parallelism, or plugin registry are implemented here.
Paged storage with gather is not a high-performance paged-attention kernel.

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

`toy/tokenizer.py` is the existing handwritten character tokenizer for learning
tokenization. It also keeps old character checkpoints usable; a separate learning
directory is unnecessary. Future Hugging Face `tokenizers` integration should
preserve this educational implementation and be added when actual dialogue
training is implemented. `toy/chat.py` and `toy/export.py` remain planned locations
for conversation formatting and inference exports, without placeholder files.
This round does not download dialogue datasets or train a new chat model.

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
