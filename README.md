# mini-vLLM

A PyTorch teaching project with a shared request scheduler, standard paged KV
storage, and model adapters. The `llm` package runs local Qwen2.5-0.5B-Instruct
weights through Transformers forward. The repository-local `toy` module keeps
the small GPT and its character-model training workflow.

The implementation gathers paged K/V into contiguous tensors for attention.
It does **not** implement a high-performance paged-attention kernel. Scheduling
supports variable-length Qwen decode batches through one model forward; prefill
remains single-request. Mixed prefill/decode batches and paged-attention kernels
remain future work.

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
benchmarks/                     engine.py, metrics.py, kv_memory.py
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

The [engine benchmark guide](benchmarks/README.md) describes reproducible
in-process workloads, latency/throughput definitions, raw result files, and
summary recalculation. Concurrency counts waiting plus running requests;
`--max-batch-size` separately limits requests in one decode forward (default 1).
Model loading, prompt encoding and warm-up are
excluded from measurements. Results belong in ignored `outputs/benchmarks/`.

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


## 批量 decode：不同上下文长度

Qwen 支持 `Scheduler(runner, max_batch_size=4)`。默认 1 保留单请求路径，
其他仅实现单请求 Runner 的模型仍然可以使用；启用大于 1 的 batch 要求实现
可选的 `BatchDecodeRunner.decode_batch(input_ids, request_ids)` 接口。

```python
from llm.inference.request import Request
from llm.inference.sampler import SamplingParams
from llm.inference.scheduler import Scheduler

scheduler = Scheduler(runner, max_batch_size=4)
for index, prompt in enumerate(([1, 3], [2, 4, 6, 8, 10, 12])):
    scheduler.add_request(Request(
        str(index), list(prompt), 4, SamplingParams(do_sample=False),
    ))
completed = scheduler.run()
```

这里两个请求分别 prefill，随后在同一次模型 forward 中 decode。
prefill 仍优先接纳；decode 从 running 头部选择最多 B 条，未结束的请求放回尾部。
这是同步、逐迭代组批；取消在 step 之间执行。没有对等待请求增加异步线程。

以已写入 KV 的历史长度 `[2, 6]` 为例，Scheduler 已为各请求追加一个未写入槽：

| 内容 | 组织方式 |
| --- | --- |
| 输入 | `input_ids [2, 1]`，第 i 行对应 `request_ids[i]` |
| 位置编码 | `position_ids = [[2], [6]]`，RoPE 使用每行真实位置 |
| 层内新 KV | `[2, Hkv, 1, D]`，分别写入各自逻辑位置 2、6 |
| attention 临时 KV | `[2, Hkv, 7, D]`，短请求右侧补零 |
| mask | 加性 `[2, 1, 1, 7]`，有效列为 0，补齐列为 dtype 的最小有限值 |
| 输出 | `logits [2, 1, V]`，每行分别采样和判断结束 |

新输入就是各行最新的 query，真实历史和当前 token 都允许被读取；
短请求第 3..6 列是 padding，必须被 mask 屏蔽。
显式 4D mask 避免框架用单一全局长度推导变长请求的因果关系。
`cache_position=[6]` 只描述临时矩形布局的最大位置；真实 RoPE 位置来自
`position_ids`，持久 KV 写入位置由批量缓存适配器逐请求决定。

每个请求继续持有独立 BlockTable 和物理 block。QwenBatchedDecodeCache
只在一次 forward 内收集各请求视图，按行写入，再 gather 成补零的临时矩形 KV。
padding 不占持久逻辑槽，也不写入其他请求的页；batch 行号不是物理块号。
当前实现支持 eager、完整上下文 attention；其他 attention 后端与滑动窗口会明确拒绝。

Scheduler 在本批所有采样成功后一次传回 CPU，并记录共同的 token 就绪时刻。
每条请求独立判断 EOS/输出上限，只有继续执行的请求才预留下一个 KV 槽。
批次执行或采样失败后，step 清理执行批次，保留未参与的队列；
run 的异常处理则继续取消其他请求。资源扩容过程中已经完成的行不会重复释放。

验收测试：

```bash
python -m pytest tests/models/test_qwen_batch_decode.py \
  tests/inference/test_scheduler_batch.py -q
QWEN_MODEL_PATH=/absolute/path/to/Qwen2.5-0.5B-Instruct \
  python -m pytest tests/models/test_qwen_weights.py -q
```

小型模型在 CPU/CUDA 上比较单请求 decode 和 dense 无缓存参考，容差
`rtol=atol=1e-5`；本地真实权重比较单请求 decode，容差
`rtol=1e-4, atol=2e-4`。测试覆盖多步变长、重排行、单行尾批次、物理页分散、
释放后复用、非参与请求不变、padding 不写缓存和一次 forward 的调用计数。
历史 KV 与未参与页不变使用零容差。这里要求浮点容差内对齐，不要求 batch
改变计算形状后逐 bit 相同。

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
