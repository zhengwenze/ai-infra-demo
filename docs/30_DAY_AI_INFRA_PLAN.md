# 30 天 AI Infra 推理入门冲刺计划

> 面向零基础学习者，以 InferScope 为实验主线，在 30 天内完成“系统基础 → LLM 推理原理 →
> vLLM Serving → 真实 Benchmark → 面试表达”的第一轮闭环。

这份计划不是“30 天成为 CUDA 工程师”，而是帮助你达到以下入门状态：

> 能部署小型 LLM 推理服务，解释核心原理，设计可信 benchmark，并在实习面试中讲清项目设计、
> 实验结果和当前边界。

长期、完整的学习路线见 [AI Infra 推理优化零基础学习路线](LEARNING_ROADMAP.md)。本计划只聚焦
LLM Inference / Model Serving，训练 Infra、CUDA Kernel 和多机分布式放到后续阶段。

## 1. 30 天最终产出

完成计划后，应当留下以下可检查成果：

1. 一个 C++ 有界生产者—消费者队列；
2. 一份 LLM 推理流程图；
3. 一份 vLLM Scheduler、Continuous Batching、PagedAttention 原理笔记；
4. 一组真实 RTX 4060 benchmark 产物，或一份明确记录硬件阻塞的实验报告；
5. 至少两组只改变一个主要变量的对照实验；
6. 一份包含 TTFT、TPOT、Throughput 和 Goodput 的性能分析；
7. 一段能够在面试中讲 3～5 分钟的项目介绍；
8. 一份基于真实证据的 AI Infra 实习简历项目描述。

真实性边界：CPU fake server 只能验证请求、计时、聚合、校验和报告链路，不能代表真实模型或
GPU 性能。只有经过审核且状态为 `VALID` 的真实 GPU 实验才能形成性能结论。

## 2. 每天固定学习结构

默认每天投入 3～4 小时。如果每天只有 2 小时，优先完成编码任务和验收标准，减少扩展阅读。

| 时间 | 内容 |
| --- | --- |
| 45～60 分钟 | 理论学习 |
| 90～120 分钟 | 编码、实验或源码阅读 |
| 30 分钟 | 整理 Markdown 笔记 |
| 20 分钟 | 不看资料口述当天知识 |
| 20～30 分钟 | C++ 或算法基础练习，可选 |

每天都要留下四种证据：

```text
1. 今天理解了什么
2. 今天运行了什么命令
3. 今天写了什么代码
4. 今天能回答什么面试问题
```

不要把“看完视频”当作学习成果。

---

# 第一阶段：Linux、C++ 与系统基础（Day 1～7）

## Day 1：开发环境和能力基线

学习内容：

- Linux 文件、进程、端口和权限基础；
- 编译、链接和可执行文件；
- Git commit、branch、diff；
- Python、C++、CUDA 和 vLLM 分别处于哪一层。

实践任务：

```bash
uname -a
lscpu
free -h
df -h
ps aux
ss -lntp
nvidia-smi
g++ --version
python --version
git --version
```

画出自己的学习主线：

```text
Linux / C++
  ↓
Transformer Inference
  ↓
vLLM Serving
  ↓
Benchmark / Profiling
  ↓
CUDA / Triton
```

验收标准：

- 能解释源代码、编译器和可执行文件的关系；
- 能找到占用某个端口的进程；
- 建立第一份每日学习记录。

## Day 2：C++ 数据与内存基础

学习内容：

- 指针和引用；
- 栈与堆；
- `const`；
- `std::string`、`std::vector`；
- 值传递和引用传递。

编码任务：实现一个简单的请求模型，并通过 `vector<Request>` 保存、修改和遍历请求。

```cpp
struct Request {
    int id;
    std::string prompt;
    int max_tokens;
};
```

验收标准：

- 能解释指针和引用的区别；
- 能说明局部变量和动态分配对象的主要生命周期差异；
- 程序能在开启编译器警告后正常编译。

## Day 3：Class、RAII 与智能指针

学习内容：

- class、构造函数和析构函数；
- RAII；
- `unique_ptr`、`shared_ptr`；
- 系统代码中的资源生命周期。

编码任务：实现 `RequestTimer`，构造时记录开始时间，结束时输出持续时间；禁止复制，允许移动。

验收标准：

- 能解释 RAII 为什么可以减少资源泄漏；
- 能区分 `unique_ptr` 和 `shared_ptr` 的使用场景；
- 能解释持续时间为什么应使用单调时钟。

## Day 4：STL 与指标聚合

学习内容：

- `unordered_map`；
- lambda；
- `sort`；
- template 基础；
- 平均值和百分位。

编码任务：使用 C++ 实现简化指标聚合器，输入请求延迟，输出 mean、min、max、P50 和 P95。

验收标准：

- 能解释 `vector` 与 `unordered_map` 的使用差异；
- 不依赖第三方统计库完成百分位计算；
- 能说明平均值为什么不能代替 P95。

## Day 5：进程、线程与上下文切换

学习内容：

- 进程与线程；
- 用户态和内核态；
- 上下文切换；
- 阻塞与忙等待；
- CPU-bound 与 I/O-bound。

编码任务：创建两个线程执行不同任务，打印线程 ID，并对比单线程与双线程执行时间。

验收标准：

- 能解释线程共享什么、独立拥有什么；
- 能解释线程越多为什么不一定越快；
- 能说明推理压测客户端为什么通常偏 I/O-bound。

## Day 6：Mutex 与 Condition Variable

学习内容：

- race condition；
- mutex；
- lock guard；
- condition variable；
- producer-consumer。

编码任务：实现一个有界阻塞队列。

```text
Producer → Request Queue → Consumer
```

必须支持：

- 队列满时生产者等待；
- 队列空时消费者等待；
- 多线程安全退出；
- 不使用循环空转等待。

验收标准：

- 程序连续运行多次不死锁；
- 能解释为什么等待条件通常要放在循环中；
- 能画出请求队列的状态变化。

## Day 7：第一周复盘

整理本周产物：

```text
cpp-lab/
├── request.cpp
├── timer.cpp
├── metrics.cpp
└── blocking_queue.cpp
```

口述验收：

1. 进程和线程有什么区别？
2. mutex 解决了什么问题？
3. condition variable 为什么比循环检查好？
4. 推理服务器为什么需要请求队列？
5. 客户端并发过高可能造成什么问题？

答不出来的内容先补齐，再进入下一阶段。

---

# 第二阶段：LLM 推理原理（Day 8～14）

## Day 8：PyTorch Tensor 与推理模式

学习内容：

- Tensor、shape、dtype、device；
- CPU Tensor 与 CUDA Tensor；
- matrix multiplication；
- `torch.inference_mode()`；
- 推理为什么不保存反向传播中间状态。

实践任务：创建不同 shape 的 Tensor，执行矩阵乘法，对比 CPU/GPU，并使用 inference mode
执行一个小模型的推理。

验收标准：

- 能解释 shape、dtype 和 device；
- 能判断两个矩阵能否相乘；
- 能解释推理为什么通常比训练节省显存。

## Day 9：Tokenizer、Embedding 与采样

学习链路：

```text
Text → Tokenizer → Token IDs → Embedding → Transformer
     → Logits → Sampling → Next Token
```

实践任务：

- 对同一句中文和英文进行 tokenization；
- 查看 token IDs；
- 修改 temperature 和 top-p；
- 观察生成结果变化。

验收标准：

- 不把字符数等同于 token 数；
- 能解释 logits、temperature 和 top-p；
- 能解释 benchmark 为什么必须固定 tokenizer。

## Day 10：Transformer 与 Attention

学习内容：

- Q、K、V；
- Self-Attention；
- causal mask；
- 多头注意力；
- Transformer block。

编码任务：手写极简 scaled dot-product attention，打印 Q/K/V、attention score 和最终输出的
shape，并观察 mask 前后的结果。

验收标准：

- 能在纸上画出 Attention；
- 能解释生成模型为什么不能看到未来 token；
- 能解释 Attention 随序列长度增长的计算代价。

## Day 11：Prefill 与 Decode

重点理解：Prefill 一次处理完整 Prompt；Decode 每一步产生一个新 token。

实践任务：为“4 个输入 token、3 个输出 token”的请求画出 Prefill 与每轮 Decode，标记每步输入、
计算和输出。

验收标准：

- 能解释输入长度为什么主要影响 TTFT；
- 能解释输出长度为什么主要影响整体 Decode 时间；
- 能说明客户端 TTFT 不等于纯 GPU Prefill 时间。

## Day 12：KV Cache

学习内容：

- K/V 为什么可以复用；
- 为什么通常不缓存 Q；
- KV Cache 与 batch、序列长度的关系；
- KV Cache 的显存开销。

使用下面的简化公式估算一个模型的 KV Cache：

```text
KV Cache ≈ 2 × 层数 × KV Head 数 × Head Dim × Token 数 × dtype 字节数
```

验收标准：

- 能画出使用与不使用 KV Cache 的区别；
- 能解释 KV Cache 节省了什么计算、增加了什么资源；
- 能解释序列越长、并发越高时显存为什么增大。

## Day 13：推理性能指标

阅读 [Benchmark 方法论](BENCHMARK_METHODOLOGY.md)。

重点掌握：

- TTFT；
- TPOT；
- E2E；
- requests/s；
- tokens/s；
- P50/P95/P99；
- Goodput。

实践任务：给定五个请求的开始、首 token 和结束时间，手动计算每个请求的 TTFT、TPOT、E2E，
以及整体吞吐和满足 SLO 的请求数。

验收标准：

- 能解释 TTFT 和 TPOT 为什么必须拆开；
- 能解释吞吐与延迟为什么存在权衡；
- 能解释 Goodput 和普通吞吐的区别。

## Day 14：第二周复盘

画出完整推理链路：

```text
Prompt → Tokenizer → Embedding → Transformer → Prefill
       → KV Cache → Decode → Logits → Sampling → Streaming Token
```

必须能回答：

1. 为什么第一个 token 通常更慢？
2. 为什么 Decode 是逐 token 的？
3. KV Cache 缓存了什么？
4. 为什么并发会增加显存？
5. 为什么小样本 P99 不适合形成强结论？
6. 为什么高吞吐不一定代表用户体验好？

---

# 第三阶段：vLLM 与推理 Serving（Day 15～21）

## Day 15：启动真实推理服务

阅读 [RTX 4060 实验指南](RTX4060_GUIDE.md)。

在支持 NVIDIA 的 Linux 环境执行：

```bash
nvidia-smi
./scripts/serve_vllm_4060.sh
```

再检查服务：

```bash
uv run inferscope server check \
  --base-url http://127.0.0.1:8000
```

验收标准：

- 服务能够返回模型列表；
- 记录驱动、CUDA、vLLM、模型和启动参数；
- 如果启动失败，保存完整错误和环境信息。

如果当天没有 Linux GPU 环境，可以用 `scripts/serve_fake.sh` 完成接口链路，但必须标记为 CPU
smoke，不能形成 GPU 性能结论。

## Day 16：流式接口与 SSE

阅读：

- `src/inferscope/transport/sse.py`；
- `src/inferscope/transport/openai_client.py`。

重点理解：HTTP streaming、SSE event、`data:`、`[DONE]`、网络 chunk 与 SSE event 的区别，以及
首个角色消息为什么不能算首 token。

编码任务：用自己的语言写一个 100～150 行以内的简化 SSE decoder，处理半个 event、一个 chunk
中的多个 event、`[DONE]` 和非法 JSON。

验收标准：

- 能解释为什么不能把第一个 TCP chunk 当成首 token；
- 能说明项目如何记录 started、first content 和 finished；
- 能找到 malformed stream 的处理路径。

## Day 17：Continuous Batching

理解 static batching 的问题：batch 中较短请求可能等待最长请求。再理解 continuous batching 如何在
请求完成后释放位置，让新请求进入执行批次。

实践任务：用纸面或简单 Python 程序模拟三个不同输出长度的请求，对比 static batching 与
continuous batching。

验收标准：

- 能解释 continuous batching 为什么可能提高 GPU 利用率和吞吐；
- 能说明它可能如何影响单请求 TTFT；
- 不把 vLLM 的 batching 能力说成自己的实现。

## Day 18：PagedAttention

学习内容：

- 连续 KV Cache 分配的碎片问题；
- block/page；
- logical block 与 physical block；
- block table；
- PagedAttention 与虚拟内存的类比。

实践任务：画图演示三个不同长度请求如何分配 KV blocks，以及请求结束后 block 如何复用。

验收标准：

- 能解释 PagedAttention 解决的是 KV Cache 管理问题；
- 能说明 PagedAttention 不等于 FlashAttention；
- 能解释减少碎片为什么能支持更多并发序列。

## Day 19：Scheduler 与请求生命周期

阅读 [InferScope 架构](ARCHITECTURE.md)，并理解被测服务内部的请求生命周期：

```text
HTTP Request → Waiting Queue → Scheduler → Running Sequence
             → Prefill / Decode → Finished
```

实践任务：画出 scheduler 每轮需要考虑的请求状态、剩余 token、KV blocks、batch token budget、
等待/抢占和 Prefill/Decode 调度。

验收标准：

- 能解释 scheduler 为什么是推理服务核心；
- 能说明请求并发和实际执行 batch size 不完全相同；
- 能解释长 Prompt 可能如何影响其他请求的 TTFT。

## Day 20：vLLM 与 GPU 遥测

阅读：

- `src/inferscope/telemetry/vllm_metrics.py`；
- `src/inferscope/telemetry/gpu.py`；
- `src/inferscope/telemetry/sampler.py`。

理解 GPU utilization、GPU memory、power、服务端逻辑指标以及客户端 event-loop lag。

验收标准：

- 能解释 GPU utilization 100% 为什么不一定最优；
- 能说明遥测缺失为什么不能用 0 代替；
- 能区分压测客户端瓶颈和推理服务瓶颈。

## Day 21：第三周复盘

完成一张 Serving 架构图：

```text
Client → OpenAI API → Request Queue → Scheduler → Continuous Batch
       → KV Cache Manager → Model Executor → GPU → Streaming Response
```

必须能回答：

1. vLLM 为什么能提升多请求场景的吞吐？
2. Continuous Batching 解决什么问题？
3. PagedAttention 与 KV Cache 有什么关系？
4. Scheduler 需要管理什么资源？
5. 为什么 GPU 满载时 Goodput 仍可能下降？

---

# 第四阶段：真实 Benchmark 与项目表达（Day 22～30）

## Day 22：建立真实实验基线

使用：

- `configs/rtx4060_qwen05b.yaml`；
- `scripts/serve_vllm_4060.sh`；
- `scripts/run_4060.sh`。

运行前记录 GPU、驱动、CUDA、vLLM、模型 revision、dtype、启动参数和其他 GPU 进程。

验收标准：

- 每个 run 都有 config、manifest、requests、telemetry、validation 和 aggregate；
- 只把 `VALID` 实验用于正式性能比较；
- `INCONCLUSIVE` 只能用于诊断，不能包装成有效结果。

## Day 23：并发矩阵实验

运行并发 `1 / 2 / 4 / 8`，填写：

| Concurrency | TTFT P95 | TPOT P95 | Requests/s | Tokens/s | Goodput | GPU Memory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | | | | | | |
| 2 | | | | | | |
| 4 | | | | | | |
| 8 | | | | | | |

分析吞吐是否增加、TTFT 从哪个点开始恶化、TPOT 如何变化、Goodput 最高点在哪里，以及 GPU
memory 如何变化。

## Day 24：第一次结果分析

按照以下结构写分析：

```text
观察到的现象
  ↓
可能原因
  ↓
现有证据
  ↓
还缺什么证据
  ↓
能得出什么结论
  ↓
不能得出什么结论
```

验收标准：

- 区分数据事实和原因推测；
- 同时讨论吞吐、延迟、成功率和 Goodput；
- 能说明最高吞吐点是否也是最适合 SLO 的配置。

## Day 25：输入长度单变量实验

只改变输入长度，例如 `256 / 512 / 1024 tokens`，固定模型、输出长度、并发、temperature、seed
和服务启动参数。

重点观察 TTFT、E2E、GPU memory 和 KV Cache 使用。

验收标准：

- 一次只改变一个主要变量；
- 不把不同输入长度的吞吐当作同等工作量直接比较；
- 能解释输入变长为什么首先影响 Prefill。

## Day 26：输出长度单变量实验

只改变输出长度，例如 `32 / 64 / 128 tokens`，重点观察 TPOT、E2E、tokens/s、提前 EOS 和输出
长度门禁。

验收标准：

- 能解释输出长度为什么影响 Decode 总时间；
- 检查实际输出 token 是否接近目标；
- 不使用少生成 token 的实验获得虚假低延迟。

## Day 27：Serving 参数单变量实验

只选择一个参数做 A/B，例如将 `max-num-seqs` 从 4 调到 8。分析吞吐收益、TTFT/TPOT 代价、
GPU memory、OOM 和 Goodput。

验收标准：

- 明确 baseline 和 experiment；
- 保留失败实验；
- 能回答该参数为什么可能影响结果；
- 不同时改变模型、负载和多个服务参数。

## Day 28：重复性与稳定性

对关键配置至少重复三次，观察 mean、标准差、CV、异常值，以及温度、功耗和后台进程变化。

验收标准：

- 不只选择最好的一次；
- 报告全部重复实验；
- 不能稳定复现时，将结论标记为探索性。

## Day 29：整理项目材料

按照 [结果证据规则](results/README.md)整理经过审核的证据：

```text
docs/results/<experiment>/
├── README.md
├── manifest.json
├── validation.json
├── aggregate.json
├── summary.csv
└── performance.svg
```

同时更新 README 结果区、实验方法、当前限制、面试问答和简历项目描述。只有真实运行且审核过的
数据才能进入 `docs/results/`。

## Day 30：模拟面试与最终验收

进行一次 45～60 分钟模拟面试，至少回答：

1. 项目解决了什么问题？
2. 和普通压测脚本有什么区别？
3. TTFT、TPOT、Goodput 怎么计算？
4. 如何解析流式 SSE？
5. 为什么需要 warmup？
6. 为什么并发增加后 TTFT 可能恶化？
7. KV Cache 为什么占显存？
8. Continuous Batching 为什么有效？
9. PagedAttention 解决什么问题？
10. 怎么证明实验结果可信？
11. 你真正实现了什么？
12. 哪些是 vLLM 已有能力？
13. RTX 4060 实验发现了什么？
14. 下一步准备验证什么优化？
15. 为什么当前项目仍不能称为 CUDA Kernel 优化？

最终验收：

- 能进行 3 分钟项目介绍；
- 能在纸上画出完整推理链路；
- 能解释并发、队列、Scheduler 和 KV Cache；
- 能展示真实 benchmark 证据，或明确说明 GPU 实验阻塞；
- 能说明实验限制；
- 不夸大未完成能力。

## 3. 30 天内暂时不展开的内容

- CUDA Kernel 编程；
- PTX；
- Triton Kernel；
- FlashAttention 源码；
- TensorRT-LLM；
- 多机多卡；
- Tensor Parallel；
- NCCL/RDMA；
- Kubernetes GPU 集群；
- 训练 Infra、DDP、FSDP。

这些内容不是不重要，而是当前还缺少推理 Serving 与性能实验基础。此时过早展开，容易变成每个
名词都听过，但没有一个能够深入解释。

## 4. 30 天后的进阶方向

第二个月：

```text
GPU Architecture → CUDA execution model → Memory hierarchy
                 → CUDA profiling → Triton → 简单 Kernel optimization
```

第三个月：

```text
vLLM / SGLang 源码 → Scheduler / KV Cache 策略
                   → TensorRT-LLM → Tensor Parallel → NCCL
                   → Distributed Inference
```

长期坚持同一条实验闭环：

```text
建立 Baseline → 找到瓶颈 → 提出假设 → 一次改变一个变量
             → 检查正确性 → 重复测量 → 分析收益与代价
```

30 天后最有价值的成果不是“我学过 vLLM”，而是：

> 我能够部署一个 LLM 推理服务，设计可信实验，分析 TTFT、TPOT、吞吐、Goodput 和 GPU 资源，
> 并明确说明结论与证据边界。
