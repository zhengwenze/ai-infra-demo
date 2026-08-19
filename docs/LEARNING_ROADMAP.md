# AI Infra 推理优化零基础学习路线

> 面向零基础学习者，以 8 周完成一个可运行、可测试、可展示的 LLM 推理优化项目为目标。

如果希望先用一个月完成求职导向的集中训练，请从
[30 天 AI Infra 推理入门冲刺计划](30_DAY_AI_INFRA_PLAN.md)开始；本文保留更完整的 8 周学习范围。

## 1. 学习目标

AI Infra（人工智能基础设施）涵盖训练、推理、数据、调度、存储、网络和硬件等多个方向。本路线聚焦 **大模型推理优化**：在模型效果基本不变的前提下，让模型响应更快、吞吐更高、显存占用更低、服务更稳定、运行成本更低。

完成本路线后，你应该能够：

- 使用 Python、PyTorch 和 Hugging Face 运行开源模型；
- 解释 Transformer 自回归推理、Prefill、Decode 和 KV Cache；
- 使用 vLLM 部署兼容 OpenAI API 的推理服务；
- 测量 TTFT、TPOT、吞吐、并发能力和显存占用；
- 使用 Profiler 定位性能瓶颈，而不是凭感觉优化；
- 对 FP16/BF16、量化、批处理、Prefix Cache 等方案进行公平的 A/B 测试；
- 理解 GPU 内存层次、算子融合以及计算瓶颈与访存瓶颈；
- 使用 Triton 编写和测试简单 GPU 算子；
- 完成一份可以放入简历或作品集的推理优化实验报告。

## 2. 学习原则

### 2.1 先打通系统，再深入底层

推荐顺序：

```text
Python / Linux / Git
        ↓
PyTorch 与 Transformer 推理
        ↓
性能指标与基准测试
        ↓
vLLM 推理服务
        ↓
量化、缓存、批处理等优化
        ↓
Profiler、Triton 与 CUDA
```

不要一开始就阅读 CUDA、vLLM 或 TensorRT-LLM 的大量源码。先完整跑通一次“模型加载 → 请求处理 → 输出生成 → 性能测量”的链路，再逐层深入。

### 2.2 没有测量，就没有优化

每次实验只改变一个主要变量，并记录：

- GPU 型号、显存和驱动版本；
- CUDA、PyTorch、推理框架和模型版本；
- 输入长度、输出长度、并发数和请求数量；
- 精度类型和量化方案；
- 预热次数、正式测试次数和测试时长；
- TTFT、TPOT、P50/P95/P99、吞吐和峰值显存；
- 优化前后的结果与结论。

### 2.3 以项目驱动学习

每天至少一半时间用于敲代码、运行实验和记录结果。看懂教程不等于掌握；能够复现、解释和对比，才算真正学会。

## 3. 必须掌握的核心指标

| 指标 | 含义 | 关注点 |
| --- | --- | --- |
| TTFT | Time To First Token，从发送请求到收到首个 token 的时间 | 影响用户感受到的响应速度 |
| TPOT | Time Per Output Token，首个 token 之后每生成一个 token 的平均时间 | 影响持续生成速度 |
| ITL | Inter-Token Latency，相邻输出 token 的延迟 | 观察流式输出是否稳定 |
| Tokens/s | 每秒产生的 token 数 | 可用于单请求或系统整体评估 |
| Throughput | 单位时间内处理的请求或 token 数量 | 衡量服务整体处理能力 |
| P50/P95/P99 | 延迟分位数 | P99 能反映慢请求和尾延迟问题 |
| GPU Utilization | GPU 计算资源利用率 | 利用率低不一定代表 GPU 本身慢 |
| Peak VRAM | 峰值显存占用 | 决定模型能否运行及可支持的并发量 |

必须区分延迟和吞吐：增大批次通常可以提高总吞吐，但单个请求等待的时间可能增加。

## 4. 8 周学习计划

建议每天学习 2 小时、每周学习 6 天。每天可按“30 分钟学习 + 70 分钟实践 + 20 分钟记录”安排。

### 第 1 周：Python、Linux、Git 和开发环境

#### 学习内容

- Python：变量、条件、循环、函数、类、列表、字典、异常、文件读写；
- NumPy：数组、维度、数据类型、广播和矩阵乘法；
- Linux：目录、文件、进程、端口、日志、环境变量和权限；
- Git：`clone`、`status`、`diff`、`add`、`commit`；
- 虚拟环境：`venv` 或 Conda；
- Docker：镜像、容器、端口映射和目录挂载的基本概念。

#### 实践任务

1. 创建 Python 虚拟环境并安装依赖；
2. 编写矩阵乘法程序，输出输入和结果的 shape；
3. 编写一个命令行程序，读取参数并将结果保存为 JSON；
4. 使用 Git 提交代码；
5. 使用 Docker 运行一个简单 Python 程序。

#### 验收标准

- 能独立下载并运行一个 Python 项目；
- 能看懂常见报错中的文件、行号和异常类型；
- 能解释进程、端口、镜像和容器的区别；
- 能使用 Git 保存一次完整修改。

### 第 2 周：PyTorch 最小基础

#### 学习内容

- Tensor、shape、stride、dtype 和 device；
- CPU 与 GPU 之间的数据移动；
- 矩阵乘法、Softmax 和常用张量操作；
- `nn.Module`、参数和 `forward`；
- FP32、FP16 和 BF16；
- `torch.no_grad()` 或 `torch.inference_mode()`；
- GPU 异步执行及准确计时前同步的必要性。

#### 实践任务

1. 在 CPU 和 GPU 上分别执行矩阵乘法；
2. 比较 FP32、FP16、BF16 的耗时和显存；
3. 编写一个包含 Linear、激活函数和 Softmax 的小模型；
4. 为测试添加预热，并使用同步操作获得更可靠的 GPU 耗时。

#### 验收标准

- 能解释 `shape`、`dtype` 和 `device`；
- 能解释为什么 GPU 计时需要预热和同步；
- 能说明降低精度为什么可能减少显存并提高速度；
- 能完成一次具有可重复性的 PyTorch 性能测试。

#### 推荐资料

- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [PyTorch Profiler Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

### 第 3 周：Transformer 与 LLM 推理原理

#### 学习内容

- Tokenizer、Token、Vocabulary 和 Embedding；
- Transformer Decoder 的基本结构；
- Self-Attention 中 Q、K、V 的作用；
- Logits、Softmax、Sampling 和 Greedy Decoding；
- 自回归生成；
- Prefill 和 Decode 两个推理阶段；
- KV Cache 的作用和显存开销；
- Batch Size、Sequence Length 和 Context Length。

#### 必须理解的推理链路

```text
用户文本
  → Tokenizer
  → Input IDs
  → Transformer Forward
  → Logits
  → 选择下一个 Token
  → 将 Token 加回输入
  → 重复生成
  → Decode 为文本
```

#### 实践任务

1. 使用 Hugging Face Transformers 加载一个 0.5B～1.5B 的小模型；
2. 打印 token ID、输入长度和输出长度；
3. 分别测试短输入、长输入、短输出和长输出；
4. 比较启用与关闭 KV Cache 时的生成耗时；
5. 手写一个简化的逐 token 生成循环。

#### 验收标准

- 能解释为什么生成 100 个 token 需要进行多轮 Decode；
- 能解释 Prefill 与 Decode 的差别；
- 能解释 KV Cache 节省了什么计算、消耗了什么资源；
- 能画出一次请求从文本到输出文本的完整流程。

#### 推荐资料

- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)

### 第 4 周：Benchmark 与性能分析

#### 学习内容

- 延迟、吞吐、并发和尾延迟；
- Warm-up 与正式测量；
- 同步计时与异步计时；
- 冷启动与热启动；
- 固定输入与真实数据集测试；
- PyTorch Profiler 的基本使用；
- CPU、GPU、数据传输和模型计算之间的边界。

#### 实践任务

建立第一版 benchmark，至少覆盖：

- 输入长度：128、512、1024；
- 输出长度：32、128、256；
- Batch Size：1、2、4、8；
- 精度：FP32、FP16 或 BF16；
- 指标：耗时、tokens/s 和峰值显存。

将实验结果保存为 CSV 或 JSON，不要只打印在终端中。

#### 验收标准

- 能设计一组只改变一个变量的实验；
- 能解释冷启动数据为什么不能直接代表稳定性能；
- 能使用 Profiler 找出耗时较大的操作；
- 能根据数据形成结论，而不是只罗列数字。

### 第 5 周：vLLM 推理服务

#### 学习内容

- 离线推理与在线推理服务；
- OpenAI-compatible API；
- Continuous Batching；
- Paged KV Cache；
- 请求调度和并发；
- Streaming Response；
- 模型服务的启动参数和日志。

#### 实践任务

1. 使用 vLLM 启动一个小模型；
2. 通过 Python 或 `curl` 发送请求；
3. 编写并发请求脚本；
4. 使用 `vllm bench` 运行吞吐和在线服务测试；
5. 在相同模型、输入和硬件下比较 Transformers 与 vLLM；
6. 记录并发提高时 TTFT、TPOT、吞吐和显存的变化。

#### 验收标准

- 能启动、调用和停止模型服务；
- 能区分客户端耗时和服务端模型耗时；
- 能解释 Continuous Batching 的基本作用；
- 能给出一份 Transformers 与 vLLM 的公平对比报告。

#### 推荐资料

- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
- [vLLM CLI Guide](https://docs.vllm.ai/en/latest/cli/)
- [vLLM Benchmark API](https://docs.vllm.ai/en/latest/api/vllm/benchmarks/)

### 第 6 周：常见推理优化技术

#### 学习内容

| 技术 | 主要解决的问题 | 常见代价或限制 |
| --- | --- | --- |
| Continuous Batching | 提高多请求场景的 GPU 利用率和吞吐 | 调度策略会影响单请求延迟 |
| KV Cache | 避免 Decode 阶段重复计算历史 K/V | 占用显存，长上下文更明显 |
| Paged KV Cache | 降低 KV Cache 的碎片和管理成本 | 引入块管理机制 |
| Prefix Caching | 复用多个请求的公共前缀 | 只在前缀重复时收益明显 |
| FP16/BF16 | 减少显存和计算开销 | 依赖硬件支持，可能有数值差异 |
| INT8/INT4 量化 | 大幅降低权重显存和带宽压力 | 可能降低效果，且有量化/反量化开销 |
| Tensor Parallel | 将大模型分布到多张 GPU | 带来跨卡通信开销 |
| Chunked Prefill | 减少长 Prefill 对其他请求的阻塞 | 调度配置更复杂 |
| Speculative Decoding | 借助草稿模型一次验证多个 token | 依赖接受率和额外模型开销 |
| Operator Fusion | 减少中间数据读写和 Kernel Launch | 实现和调试复杂度增加 |

#### 实践任务

至少选择三项技术进行 A/B 测试，例如：

1. BF16 对比 FP16；
2. 原始权重对比 INT4 量化；
3. Prefix Caching 关闭对比开启；
4. 不同最大并发数对比；
5. 不同输入长度下 Chunked Prefill 对比。

#### 验收标准

每项实验都能回答：

- 它解决了什么瓶颈？
- 在什么负载下有效？
- 指标提升了多少？
- 是否增加显存、延迟或精度风险？
- 实验是否公平、是否可复现？

### 第 7 周：GPU、Triton 与 CUDA 入门

#### 学习内容

- GPU Thread、Block、Grid、Warp；
- SIMT 执行模型；
- Register、Shared Memory、L1/L2 Cache、Global Memory；
- 合并访存（Coalesced Memory Access）；
- 算术强度（Arithmetic Intensity）；
- Compute-bound 与 Memory-bound；
- Kernel Launch Overhead；
- Tiling 和 Operator Fusion；
- Roofline 模型的基本思想。

#### Triton 实践顺序

1. Vector Addition；
2. Fused Softmax；
3. Matrix Multiplication；
4. Layer Normalization；
5. Fused Attention。

每个算子都必须完成：

- 与 PyTorch 参考实现对比，验证正确性；
- 覆盖多个输入 shape；
- 预热后测量性能；
- 比较 Triton 与 PyTorch 的延迟；
- 分析快或慢的原因。

#### 验收标准

- 能解释 GPU 的内存层次；
- 能判断一个简单算子更可能受计算还是内存带宽限制；
- 能解释算子融合为什么可能减少 HBM 读写和启动开销；
- 能独立修改 Triton Kernel 的 block size 并比较性能。

#### 推荐资料

- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)

### 第 8 周：完成作品集项目

项目名称建议：**LLM Inference Optimization Lab**。

#### 项目结构建议

```text
ai-infra-demo/
├── README.md
├── requirements.txt
├── docker/
├── scripts/
│   ├── serve_vllm.sh
│   ├── run_benchmark.sh
│   └── collect_gpu_metrics.sh
├── benchmark/
│   ├── client.py
│   ├── metrics.py
│   └── datasets.py
├── experiments/
│   ├── baseline.yaml
│   ├── precision.yaml
│   ├── quantization.yaml
│   └── concurrency.yaml
├── results/
│   ├── raw/
│   ├── charts/
│   └── summary.md
└── tests/
```

#### 最小功能

- Transformers 原生推理基线；
- vLLM 在线推理服务；
- 自动并发压测；
- FP16/BF16 或量化对比；
- 不同输入长度、输出长度和并发数的对比；
- 自动保存 JSON/CSV 结果；
- 生成性能图表；
- 输出实验环境、方法、结果和结论。

#### 项目验收标准

- 新环境可以按照 README 复现实验；
- 脚本不会把模型输出长度和请求失败错误地计入成功吞吐；
- 所有对比使用相同模型、硬件和负载；
- 报告同时展示收益和代价；
- 至少有一项优化获得了数据支持；
- 即使优化无收益，也能基于 Profiler 或运行数据解释原因。

## 5. 推荐的实验记录模板

每次实验复制下面的模板：

```markdown
## 实验名称

### 目标

本次只验证什么问题？

### 环境

- GPU：
- CPU：
- OS：
- Driver/CUDA：
- PyTorch：
- 推理框架：
- 模型与版本：

### 固定条件

- 输入长度：
- 输出长度：
- 并发数：
- 请求数量：
- 精度：
- 预热方式：

### 唯一变量

- 对照组：
- 实验组：

### 结果

| 方案 | TTFT P50 | TTFT P99 | TPOT | 输出 tokens/s | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 对照组 | | | | | |
| 实验组 | | | | | |

### 结论

- 数据说明了什么？
- 优化在哪些条件下有效？
- 有什么代价或异常？
- 下一次需要验证什么？
```

## 6. 硬件与环境建议

### 没有 NVIDIA GPU

- 前三周可使用本机 CPU、Apple Silicon 的 MPS 或在线 Notebook；
- 从 vLLM、CUDA 和 Triton 阶段开始，建议租用 Linux + NVIDIA GPU；
- 先使用 0.5B～1.5B 小模型验证完整流程；
- 按小时租用 GPU 时设置消费上限，用完立即关闭实例；
- 实验完成后保存代码、依赖版本和原始结果，不必一直保留机器。

### 有 NVIDIA GPU

先记录：

```bash
nvidia-smi
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
```

不要默认“程序能跑”就代表环境正确。驱动、CUDA、PyTorch 和框架版本需要相互兼容。

## 7. 初学阶段暂时不要做的事

- 不要从零训练大模型；
- 不要同时学习 vLLM、SGLang、TensorRT-LLM 等全部框架；
- 不要一开始阅读大型推理框架的全部源码；
- 不要脱离具体实验死记 CUDA 名词；
- 不要用不同模型或不同 GPU 的数字直接得出框架优劣；
- 不要只报告平均延迟而忽略 P95/P99；
- 不要只看 GPU 利用率就判断系统是否达到最佳性能；
- 不要省略预热、输入分布、并发和输出长度；
- 不要为了得到“更快”的结论而隐藏精度、显存或稳定性代价。

## 8. 学习完成后的进阶方向

完成 8 周路线后，根据兴趣选择一个方向继续深入。

### 推理引擎与服务方向

- 深入 vLLM 调度器、Block Manager 和执行引擎；
- 学习 SGLang、TensorRT-LLM 或其他推理框架；
- 学习限流、排队、自动扩缩容和多租户隔离；
- 研究真实流量下的吞吐与尾延迟权衡。

### GPU Kernel 方向

- 深入 Triton 和 CUDA C++；
- 学习 Nsight Systems、Nsight Compute；
- 分析 GEMM、Attention、Normalization 和 Sampling；
- 研究 Tiling、Pipeline、Tensor Core 和低精度计算。

### 分布式推理方向

- Tensor Parallel、Pipeline Parallel 和 Expert Parallel；
- NCCL、NVLink、PCIe、RDMA；
- 通信与计算重叠；
- Prefill/Decode 分离和多机部署。

### 模型压缩方向

- Weight-only Quantization；
- SmoothQuant、GPTQ、AWQ 等方法；
- FP8、INT8、INT4；
- 蒸馏、剪枝和稀疏化；
- 精度评估与性能收益之间的权衡。

## 9. 最终自检清单

如果下面大部分问题都能回答并亲手验证，就已经完成入门：

- [ ] 我能运行一个 Hugging Face 模型并解释推理链路；
- [ ] 我能解释 Prefill、Decode 和 KV Cache；
- [ ] 我能准确测量 GPU 程序耗时；
- [ ] 我能解释 TTFT、TPOT、吞吐和 P99；
- [ ] 我能使用 vLLM 启动并调用模型服务；
- [ ] 我能编写并发压测程序；
- [ ] 我能公平比较两种推理配置；
- [ ] 我能使用 Profiler 找到主要耗时；
- [ ] 我能解释一种量化方案的收益和代价；
- [ ] 我能解释 Continuous Batching 和 Paged KV Cache；
- [ ] 我能描述 GPU 的主要内存层次；
- [ ] 我能编写并验证一个简单 Triton Kernel；
- [ ] 我拥有一份可复现的实验报告和一个可展示项目。

## 10. 一句话总结

AI Infra 推理优化的正确入门方式不是尽可能多地背概念，而是反复完成这条闭环：

```text
建立基线 → 找到瓶颈 → 提出假设 → 只改一个变量
→ 验证正确性 → 测量性能 → 分析收益与代价 → 记录并复现
```

只要能够用数据证明“为什么这里变快了、在什么条件下有效、付出了什么代价”，你就已经开始具备推理优化工程师的核心能力。
