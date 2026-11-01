# mini-vLLM

**中文** | [English](README.md)

一个基于 PyTorch 的大模型推理学习项目，用于理解请求调度、KV 缓存管理和批量执行。
支持本地 Qwen2.5-0.5B-Instruct，并保留小型 GPT 的训练与生成示例。

## 功能

- 分页 KV 存储与请求级资源管理。
- 不同上下文长度的批量 decode。
- 请求取消、EOS 处理与异常资源回收。
- 可复现的吞吐量和延迟基准。

当前 prefill 逐请求执行；attention 使用分页 KV gather，尚未实现高性能 paged-attention kernel。
后续计划支持批量 prefill，并完善连续批处理策略。

## 文档

| 文档 | 内容 |
| --- | --- |
| [快速开始](docs/zh-CN/getting-started.md) | 环境准备、本地模型与运行方式 |
| [项目结构与架构](docs/zh-CN/architecture.md) | 目录划分、Runner、调度器与缓存职责 |
| [性能基准](docs/zh-CN/benchmarking.md) | 压测命令、并发参数与指标定义 |

本项目开发过程中使用了 AI 辅助。
