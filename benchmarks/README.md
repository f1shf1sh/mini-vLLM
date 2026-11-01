# 推理性能基线

这里的基准直接调用 `QwenModelRunner` 和共享 `Scheduler`，测量进程内推理引擎。
模型加载、tokenizer 编码、请求对象创建、预热、元数据哈希和结果写盘均在测量区间之外。
没有 HTTP、网络传输或文本解码开销。

Scheduler 的 prefill 每次执行一个请求，decode 可以把不同上下文长度的请求放入一次 forward。
`concurrency` 是 **waiting + running** 的上限；`--max-batch-size` 是一次 decode
的请求数上限（默认 1，保留原串行基线）。实际 batch 大小还受 running 请求数限制。
可接纳的新请求优先 prefill，其余请求按轮转顺序 decode。

## 运行

在仓库根目录、已安装项目依赖的环境中运行。`--model` 必须替换为已有的本地
Qwen2/Qwen2.5 模型目录；脚本不会下载模型。

```bash
python -m benchmarks.engine \
  --model /absolute/path/to/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype float32 \
  --num-blocks 256 --block-size 16 \
  --prompt-length 128 --max-new-tokens 16 \
  --concurrency 4 --num-requests 12 \
  --warmup 4 --repeat 3 --seed 0 \
  --eos-policy fixed \
  --output-dir outputs/benchmarks
```

`--device cpu` 可用于功能冒烟验证。GPU 性能数据应在同一设备、相同精度和模型下比较，
每次只运行一个压测进程。完整参数见 `python -m benchmarks.engine --help`。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--model` | 必填 | 本地模型目录 |
| `--device` | `cuda` | 模型执行设备 |
| `--dtype` | `float32` | 支持 float32、float16、bfloat16 |
| `--num-blocks` / `--block-size` | `256` / `16` | KV 总块数、每块 token 数 |
| `--prompt-length` | `128` | 每个测量请求的精确输入 token 数 |
| `--max-new-tokens` | `16` | 输出上限，允许为零 |
| `--concurrency` | `1` | 同时未结束的请求数上限 |
| `--max-batch-size` | `1` | 单次 decode forward 的请求数上限，prefill 仍逐请求 |
| `--num-requests` | `12` | 每轮固定的请求总数 |
| `--warmup` | `2` | **每轮**测量前的预热请求数，可为零 |
| `--repeat` | `3` | 同配置重复测量轮数 |
| `--seed` | `0` | 合成输入和 PyTorch 随机种子 |
| `--eos-policy` | `fixed` | `fixed` 或 `model`，见下文 |
| `--output-dir` | `outputs/benchmarks` | 结果父目录，每次调用创建唯一子目录 |

输入是明确的 **synthetic benchmark**：先用真实 tokenizer 编码脚本内固定文本，
关闭 special tokens，再用种子选择偏移，循环截取恰好 `prompt_length` 个 token ID。
它不使用聊天模板，也不代表自然对话分布。所有输入 ID、原始文本和构造规则均保存在配置中。
各 repeat 使用相同的测量输入；预热使用独立的、同样可复现的一组输入。

采样固定为 greedy（`do_sample=False`）：

- `fixed`：停止 token 列表为空，生成恰好 `max_new_tokens` 个 token。即使模型产生 EOS，
  也继续生成；用于比较相同输入/输出工作量。
- `model`：使用模型 generation config 的 EOS，缺省时退回 tokenizer EOS；
  命中即结束。统计使用实际输出数，包括产生的停止 token。如果模型和 tokenizer 均无 EOS，
  则只能在输出上限结束。

## 闭环负载和测量边界

每轮先完成指定数量的预热请求，并检查全部缓存归还。重新设置 PyTorch 种子后，
准备好测量请求，在第一次提交之前开始计时：

1. 填入最多 `concurrency` 个请求。
2. 调用一次 `Scheduler.step()`。
3. 若有请求结束，立即补入后续请求，保持 outstanding 上限。
4. 固定工作量全部提交后继续执行，直到队列排空。
5. 调用 `close()`，同步 GPU，再记录测量结束时刻。

整轮吞吐使用这段实际墙钟时间，包含首次提交/填充、后续提交、排队、末尾排空、
异常处理、最终清理和最终同步。并发上限可能超过缓存允许同时接纳的请求数；
多出的请求停留在 waiting，配置会记录每请求最大所需块数和缓存接纳上限。

CUDA 操作是异步的。脚本只在每轮开始计时前和结束计时前调用 CUDA synchronize，
不为每个 forward 或每一层增加同步。逐 token 时间来自 Scheduler：
单请求在已有的 `token.item()` 后记录；批量 decode 一次传回整批采样 token，
写入各自 `generated_ids` 后给所有行记录同一就绪时刻。
因此这些是 CPU 观察到 token 就绪的时间，不是纯 GPU kernel 耗时。

所有事件使用 `time.perf_counter_ns()`；这是进程内单调时钟，数值不是 Unix 时间，
不能跨机器直接比较。目录名和配置中的 UTC 日期只用于定位运行记录。

## 指标定义

设请求提交时刻为 `s`，被选中执行 prefill 的时刻为 `a`，第 i 个输出 token 就绪时刻
为 `t_i`，清理完成时刻为 `f`，实际输出数为 `N`：

| 指标 | 计算 |
| --- | --- |
| queue | `a - s` |
| TTFT | `t_1 - s`，包含排队 |
| ITL | 每个 `t_i - t_(i-1)` |
| 每请求 TPOT | `(t_N - t_1) / (N - 1)` |
| E2E | `f - s`，包含请求资源清理 |
| 请求吞吐 | 成功请求数 / 整轮测量秒数 |
| 输出 token 吞吐 | 成功请求实际输出 token 总数 / 整轮测量秒数 |

延迟单位为毫秒。每项保存 count、min、max、mean、p50、p95、p99；分位数对排序后
`(n - 1) * q` 的位置做线性插值。`tpot_per_request` 将每个请求作为一个样本；
`itl` 将每个 token 间隔作为一个样本，两者权重不同。

零输出请求没有 TTFT、TPOT、ITL；单输出请求没有 TPOT、ITL。这些分布在没有样本时
为 `count=0`，数值字段为 `null`。零输出请求直接结束，不执行 prefill，`admitted_at_ns`
为 `null`，因此也没有 queue 样本。

延迟统计和吞吐分子只包含 `FINISHED` 请求。失败、取消请求的部分输出单独计数，
不混入成功吞吐；整轮分母仍包括它们耗费的时间。每轮都保留样本数；少量请求的 p99
只描述这批样本，不足以代表稳定的尾延迟。比较性能时同时观察多轮波动，不能只挑最快一轮。

## 结果与复现

每次调用创建一个带 UTC 时间和随机后缀的目录，已有结果不会被覆盖：

```text
outputs/benchmarks/<invocation-id>/
  command.txt                  # 原始命令和工作目录
  invocation.json              # CLI 参数和 argv
  repeat-000/
    command.txt
    config.json                # 生效参数、输入 ID、代码/模型标识、关键运行信息
    warmup.json                # 预热自己的区间和清理结果，warmup=0 时不存在
    warmup_requests.jsonl      # 预热原始记录，不混入测量指标
    requests.jsonl             # 每个计划请求的输入/输出 ID、状态、原始时间事件
    measurement.json           # 实际测量起止、错误、峰值 outstanding、缓存检查
    summary.json               # 从前两个原始文件计算出的结果
    stdout.log                 # 本轮基准摘要
    stderr.log                 # 本轮捕获的执行/清理异常
  repeat-001/
  ...
```

每条测量记录包含 `submitted_at_ns`、`admitted_at_ns`、`token_ready_at_ns` 列表、
`finished_at_ns`。失败时，尚未提交的计划请求记为 `NOT_SUBMITTED`，timing 为 `null`；
已提交但尚未完成的其他请求会被关闭，通常记为 `CANCELLED`。

`config.json` 保存代码 commit、dirty 状态，以及 `src/llm/`、`benchmarks/` 下全部
Python 文件和 `pyproject.toml` 的 SHA-256 清单；包含尚未被 git 跟踪的新源文件。
模型权重和 tokenizer/config 文件保存文件大小及 SHA-256。关键运行信息包括
Python、PyTorch、Transformers、CUDA runtime、GPU、device、dtype 和 CPU 线程数。
这里不导出 Conda/pip 环境快照。若代码 dirty，复现时需保留与清单一致的本地源文件；
只 checkout commit 不一定能恢复当时测量的代码。

从原始文件重算，不读取旧 summary：

```bash
python -m benchmarks.metrics outputs/benchmarks/<invocation-id>/repeat-000

# 保存到新文件；--output 拒绝覆盖已有文件。
python -m benchmarks.metrics outputs/benchmarks/<invocation-id>/repeat-000 \
  --output outputs/benchmarks/<invocation-id>/repeat-000/recomputed.json
```

重算会拒绝重复请求 ID、token 数与 ID/事件数不一致、越界或逆序时间、非末位 EOS、
未达到上限且没有 EOS 的伪成功请求，以及与成功状态矛盾的错误/资源清理信息。
运行状态和错误同时保留在结果中，不会把一次失败运行包装成成功基线。

每轮 `close()` 后检查 block table 为空、allocated 集合为空，并验证 free block ID
恰好为 `0..capacity-1`，防止只检查数量而漏掉重复/丢失的块。执行失败时保存原始记录，
停止后续 repeat，并以非零状态退出。加载模型等测量前失败保存在顶层 `failure.json`；
预热失败保存在本轮 warmup 文件中，该轮不会生成测量 summary。

## 回归验证

```bash
python -m pytest tests/inference/test_benchmark_metrics.py \
  tests/inference/test_benchmark_engine.py -q
```

测试覆盖数值口径、空/单 token、失败请求、分位数、损坏记录拒绝、并发上限、
失败回收、EOS/固定输出、种子复现和小型本地 Qwen 的 CPU CLI 全流程。
这些测试验证压测工具的正确性，不作为性能结果。


批量 decode 示例（模型路径替换为本地目录）：

```bash
python -m benchmarks.engine --model /absolute/path/to/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype float32 --prompt-length 32 --max-new-tokens 8 \
  --num-requests 8 --concurrency 4 --max-batch-size 4 --warmup 2 --repeat 3
```

当前使用 eager attention，分页 KV 在每层 gather 为右侧补零的矩形临时张量；
mask 屏蔽补齐区域，RoPE 使用各请求真实位置。这是批量执行的正确性实现，
临时 KV 搬运开销仍然存在。性能比较应保持 concurrency 和工作量一致，
单独改变 max-batch-size；不能直接把提高 concurrency 的收益归因于批处理。
