# 项目结构与架构

[项目首页](../../README.zh-CN.md) | [English](../en/architecture.md)

## 项目结构与模型适配器

```text
src/llm/
  cache/                        blocks.py, pool.py, paged.py
  inference/                    runner.py, scheduler.py, request.py, sampler.py, types.py
  model/
    gpt_mini/                   runner.py
    qwen2_5_0_5b_instruct/       runner.py, cache.py
  runtime/generate.py
  config.py                     Inference configuration

toy/
  model/                        GPTMini, attention, block
  tokenizer.py                  Educational character tokenizer
  generator.py                  Direct generation without Scheduler
  config.py, checkpoint.py, dataset.py, utils.py
  train.py, generate.py
  configs/, tests/, tools/

tests/                          cache/, inference/, models/
configs/                        qwen.yaml
examples/                       qwen_inference.py
benchmarks/                     engine.py, metrics.py, kv_memory.py
docker/                         Dockerfile, dev.sh
```

模型适配器使用规范化后的完整模型名称，例如 `gpt_mini` 和
`qwen2_5_0_5b_instruct`。新增模型应实现自己的 Runner，并将实例传给 Scheduler；
共享调度器中不添加模型专用分支。目前没有模型注册系统或模型工厂。

导入包名为 `llm`。发布包包含推理代码与元数据，`toy` 从仓库运行。
`GPTMiniRunner` 接收外部构造的模型，不依赖 toy 训练代码。
数据、权重和运行产物保留在相应目录；生成的数据与结果不应进入提交。
个人练习和详细验证记录放在 Git 忽略的 `.local/`。

历史路径对应关系：

| 旧位置 | 当前位置 |
| --- | --- |
| `llm.inference.block_manager` | `llm.cache.blocks` |
| `llm.inference.kv_pool` | `llm.cache.pool` 和 `llm.cache.paged` |
| `llm.models.qwen` | `llm.model.qwen2_5_0_5b_instruct` |
| `toy.runner.ModelRunner` | `llm.model.gpt_mini.runner.GPTMiniRunner` |
| `toy.learning.char_tokenizer.CharTokenizer` | `toy.tokenizer.CharTokenizer` |
| toy Runner 集成测试 | `tests/models/test_gpt_mini_runner.py` |
| `toy/examples/`、`toy/benchmarks/` | 已移除；推理引擎示例与基准位于根目录 `examples/`、`benchmarks/` |

## Runner、调度器与资源边界

公开 Runner 接口位于 `llm.inference.runner`：

- `RunnerSpec` 提供上下文长度、设备和数据类型。
- 单请求 `prefill`、`decode` 返回 `RunnerOutput`，logits 形状为 `[1, T, V]`。
- 可选的 `BatchDecodeRunner.decode_batch` 接收 `[B, 1]`，返回 `[B, 1, V]`。
- 适配器解析模型配置，创建缓存资源，并负责模型 forward 和缓存格式转换。

`Scheduler(runner)` 负责 FIFO 接纳、轮转 decode、采样、结束判断和资源回收。
它通过 `CacheResources` 查询容量、分配、扩容和释放，不检查模型结构或 KV 张量布局。

| 组件 | 职责 |
| --- | --- |
| `BlockManager` | 管理物理块归属和 BlockTable |
| `KVPool` | 持有实际 K/V 张量 |
| `PagedKVCache` | 映射逻辑 token 位置，执行读写与 gather |
| `PagedCacheResources` | 向调度器提供缓存生命周期接口 |
| `GPTMiniRunner` | 将外部传入的 GPTMini 模型接入推理引擎 |
| `QwenModelRunner` | 适配 Transformers forward，区分 query heads 和 KV heads |
| `QwenPagedCache` | 将模型产生的 KV 写入已预留槽位，不承担请求调度和分配/释放职责 |

最终采样出的 token 会直接返回，不再执行一次 forward。
因此，当输出上限大于零时，最大缓存长度为
`prompt_length + max_new_tokens - 1`。零输出请求直接结束，不分配缓存。

请求正常结束、命中 EOS、取消或执行失败时都会回收资源。
`cancel_request(request_id)` 在 step 之间取消一条 waiting/running 请求，
不影响其他请求；未知或已结束的 ID 返回 `False`，取消不会中断正在运行的 forward。
`run()` 在异常时清理剩余请求；手动调用 `step()` 的调用方可以用 `close()` 清理队列。

启用 `Scheduler(runner, record_timing=True)` 后，`request.timing` 记录提交、
接纳、每个 token 在 CPU 可见的就绪时刻，以及资源清理完成时刻。
时间使用单调时钟，单位为纳秒，默认关闭。零输出请求没有接纳/token 时间；
取消或失败请求保留已记录的事件。指标统计由 benchmark 负责。
