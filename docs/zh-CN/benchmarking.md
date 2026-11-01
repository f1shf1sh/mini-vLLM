# 性能基准

[项目首页](../../README.zh-CN.md) | [English](../en/benchmarking.md)

`benchmarks/engine.py` 驱动进程内闭环负载并保存原始记录；
`benchmarks/metrics.py` 验证这些记录，再计算吞吐量和延迟。
完整口径与文件格式见[性能基准说明](../../benchmarks/README.md)。

准备好项目依赖与本地权重后，在仓库根目录中运行：

```bash
python -m benchmarks.engine \
  --model models/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype float32 \
  --num-blocks 256 --block-size 16 \
  --prompt-length 128 --max-new-tokens 16 \
  --num-requests 12 --concurrency 4 --max-batch-size 4 \
  --warmup 4 --repeat 3 --seed 0 \
  --eos-policy fixed \
  --output-dir outputs/benchmarks
```

| 参数 | 含义 |
| --- | --- |
| `--num-requests` | 每轮测量的请求总数 |
| `--concurrency` | waiting + running 的上限；请求结束后补入后续请求 |
| `--max-batch-size` | 一次 decode forward 的请求数上限，默认 1 |
| `--warmup` | 每轮正式测量前的预热请求数 |
| `--repeat` | 同一配置重复测量的轮数 |
| `--eos-policy fixed` | 忽略 EOS，生成固定数量 token，便于比较相同工作量 |
| `--eos-policy model` | 使用模型或 tokenizer 的 EOS，按实际输出量统计 |

输入为可复现的合成 token 序列，采样固定为 greedy，不代表真实聊天流量。
模型加载、输入编码、请求构造、预热、元数据哈希与结果写盘不计入测量。
整轮墙钟时间包含提交、排队、执行、排空、清理和最终 GPU 同步。

吞吐量为成功请求数或实际输出 token 总数除以整轮秒数。
TTFT 包含排队；ITL 为相邻 token 就绪时刻之差；
TPOT 为每条请求首末 token 时间差除以 `输出数 - 1`；
E2E 包含资源清理。逐 token 时间表示 CPU 观察到的就绪时刻，不是单个 GPU kernel 耗时。

结果写入 `outputs/benchmarks/<invocation-id>/`，保存原始命令、配置、
输入/输出 ID、时间事件、缓存检查和汇总；每次调用创建新目录。
基准结束时检查缓存归还情况，包括空 block table、空 allocated 集合和完整的 free block ID。
保留关键运行信息及代码/模型哈希，不导出环境快照。

替换下面的占位目录，即可从原始记录重算统计：

```bash
python -m benchmarks.metrics 'outputs/benchmarks/<invocation-id>/repeat-000'
```

比较批量执行时，保持模型、设备、精度、输入/输出工作量和 concurrency 相同，
单独改变 `max_batch_size`，并观察多轮结果。分页 gather 和 padding 仍有开销，
不能预先假定批量版本一定更快。
