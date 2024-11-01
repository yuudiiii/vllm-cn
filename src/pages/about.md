---
title: 关于项目
---

# 欢迎来到 vLLM!

vLLM 是一个快速且易于使用的库，专为大型语言模型 (LLM) 的推理和部署而设计。

vLLM 的核心特性包括：

- 最先进的服务吞吐量

- 使用 **PagedAttention** 高效管理注意力键和值的内存

- 连续批处理传入请求

- 使用 CUDA/HIP 图实现快速执行模型

- 量化： [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), INT4, INT8, 和 FP8

- 优化的 CUDA 内核，包括与 FlashAttention 和 FlashInfer 的集成

- 推测性解码

- 分块预填充

vLLM 的灵活性和易用性体现在以下方面：

- 无缝集成流行的 HuggingFace 模型

- 具有高吞吐量服务以及各种解码算法，包括*并行采样*、*束搜索*等

- 支持张量并行和流水线并行的分布式推理

- 流式输出

- 提供与 OpenAI 兼容的 API 服务器

- 支持 NVIDIA GPU、AMD CPU 和 GPU、Intel CPU 和 GPU、PowerPC CPU、TPU 以及 AWS Neuron

- 前缀缓存支持

- 支持多 LoRA

欲了解更多信息，请参阅以下内容：

- [vLLM announcing blog post](https://vllm.ai) (PagedAttention 教程)

- [vLLM paper](https://arxiv.org/abs/2309.06180) (SOSP 2023)

- [How continuous batching enables 23x throughput in LLM inference
  ](https://www.anyscale.com/blog/continuous-batching-llm-inference) [while reducing p50
  ](https://www.anyscale.com/blog/continuous-batching-llm-inference)[ ](https://www.anyscale.com/blog/continuous-batching-llm-inference)[latency](https://www.anyscale.com/blog/continuous-batching-llm-inference)
  by Cade Daniel et al.

- vLLM 聚会

## 文档

### 入门

[安装](https://shimo.im/docs/N2A1g69E94Tr68qD)

使用 ROCm 进行安装

使用 OpenVINO 进行安装

使用 CPU 进行安装

使用 Neuron 进行安装

使用 TPU 进行安装

使用 XPU 进行安装

快速入门

调试提示

示例

### 服务

OpenAI 兼容服务器

使用 Docker 部署

分布式推理和服务

生产指标

环境变量

使用统计数据收集

集成

使用 CoreWeave 的 Tensorizer 加载模型

常见问题解答

### 模型

支持的模型

模型支持策略

添加新模型

启用多模态输入

引擎参数

使用 LoRA 适配器

使用 VLMs

在 vLLM 中使用推测性解码

性能和调优

### 量化

量化内核支持的硬件

AutoAWQ

BitsAndBytes

INT8 W8A8

FP8 W8A8

FP8 E5M2 KV 缓存

FP8 E4M3 KV 缓存

### 自动前缀缓存

简介

实现

广义缓存策略

### 性能基准

vLLM 的基准套件

### 开发者文档

采样参数

离线推理

LLM 类

LLM 输入

vLLM 引擎

LLM 引擎

AsyncLLMEngine

vLLM 分页注意力

输入

概念

查询

键

QK

Softmax

值

LV

输出

输入处理

指南

模块内容

多模态

指南

模块内容

Docker 文件

vLLM 性能分析

示例命令和用法：

离线推理：

OpenAI 服务器：

## 社区

vLLM 聚会

赞助商

# 索引和表格

- 索引

- 模块索引
