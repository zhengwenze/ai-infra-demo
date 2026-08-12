# InferScope 项目立项与路线图

## 1. 项目定位

InferScope 是一个面向 LLM 在线推理服务的 SLO-aware benchmark 与证据归档工具。它服务两个目标：

1. 用一个范围可控的项目掌握 AI Infra 推理服务的负载、时延、吞吐、遥测与可复现性；
2. 为实习面试提供可以从代码、测试和产物追溯的工程案例，而不是只包装一个开源命令。

项目不自研推理引擎。vLLM/兼容服务负责模型执行，InferScope 负责“怎么压、怎么量、实验是否有效、
如何形成可复核证据”。

## 2. 要解决的问题

常见 benchmark 容易出现以下问题：

- 只报平均值或峰值吞吐，隐藏 TTFT/TPOT 尾延迟与失败率；
- prompt、输出长度、到达模型和 warmup 不一致，无法公平比较；
- 客户端 event loop 已经过载或 GPU 被其他进程污染，却仍把数字归因给服务器；
- 环境、参数、原始请求和验证状态缺失，结果无法复现；
- 把 mock smoke 或某次最佳结果写成硬件性能结论。

InferScope 将指标、门禁、环境指纹与结构化产物放在一次 runner 中，优先拒绝不可信结论。

## 3. 当前范围

### 已验证（Verified）

- OpenAI Chat Completions 流式 SSE 请求链；
- synthetic 负载与 concurrency/fixed-rate/Poisson 到达计划；
- TTFT、TPOT、E2E、吞吐、成功率和 SLO Goodput；
- warmup、计时完整性、输出长度、成功率、event-loop lag 等门禁；
- raw/processed/figure evidence bundle；
- CPU fake-server 的 unit、contract、integration 和 smoke 链路。

### 已实现但目标环境未验证（Implemented）

- vLLM Prometheus 指标解析；
- NVML GPU 遥测采集；
- shared-prefix prompt 生成器；
- Pareto frontier 与重复实验稳定性纯函数；
- RTX 4060 + Qwen2.5-0.5B 的启动脚本和实验配置。

### 尚未实现（Planned）

- Hugging Face backend adapter 与跨引擎公平 A/B；
- completions 与真正的 mixed workload；
- 本地 tokenizer 对照和 token mismatch 门禁；
- Pareto/stability 的 CLI 与一等报告；
- 自动参数搜索、回归阈值和 GitHub Actions；
- 可复核的 RTX 4060 公开结果。

## 4. 核心设计

### 证据优先

每次运行先生成请求级事实、环境 manifest 和验证报告，再派生 aggregate、CSV、Markdown 和图表。
报告不是唯一事实源，任何结论都应该能回到 raw evidence。

### SLO-aware

项目同时观察用户等待首 token 的 TTFT、持续生成的 TPOT、E2E、失败率和吞吐。Goodput 只统计
满足时延约束的成功请求，并在运行级成功率不达标时归零。

### 无效实验显式化

结果分为 `VALID`、`INVALID` 和 `INCONCLUSIVE`。证据不足不会被当作通过，特别是 GPU 隔离和
token 统计等容易缺失的信号。

### 自研与复用

自研集中在协议状态机、负载计划、指标/门禁、遥测归一化、实验编排和 artifact contract；复用
httpx、Pydantic、Prometheus/NVML 接口与 vLLM 的服务能力。具体模块见 [架构](ARCHITECTURE.md)。

## 5. MVP 验收

MVP 不以某个吞吐数字为验收，而以以下闭环为准：

- YAML 配置错误能在请求前失败；
- fake server 可以完成端到端 smoke；
- SSE chunk 边界与首内容语义有 contract tests；
- 请求级指标可聚合、校验并重新读取；
- 无效或证据不足的运行不会返回成功状态；
- 产物不包含密钥和默认 prompt/response 正文；
- 文档明确区分 Verified、Implemented 与 Planned。

当前代码已经达到该 CPU MVP；GPU 验证仍是下一阶段。

## 6. 近期迭代

### P0：形成首个真实 GPU 证据包

1. 在 RTX 4060 8 GB 上固定软件栈和 GPU 状态；
2. 运行 Qwen2.5-0.5B、并发 1/2/4/8、3 repeats；
3. 审核 validation、token、客户端 event-loop lag 与遥测；
4. 将完整摘要与图表放入 `docs/results/`；
5. 只在证据支持后更新 README 结果表。

### P1：补强可信度

- 给 token mismatch 增加本地 tokenizer 对照；
- 明确 `/metrics` 和 NVML 缺失原因，完善 clean-GPU evidence；
- 将中断运行原子归档为 `ABORTED`；
- 把多 repeat stability 和 Pareto 接入 CLI/报告。

### P2：扩展比较面

- 抽象 backend adapter；
- 实现 HF baseline 与 completions；
- 在固定 workload 和门禁下做公平 A/B；
- 最后再做参数搜索与回归门禁。

## 7. 求职展示口径

面试时可以基于代码说明：

- 为什么 TTFT 与 TPOT 必须拆开；
- 为什么 closed-loop concurrency 与 fixed-rate 不是同一负载；
- 如何解析跨 chunk SSE 并避免角色 chunk 误判首 token；
- 为什么 event-loop lag 或缺失 GPU 证据会让结果失效；
- 如何通过 raw evidence、manifest 和 validation 防止“漂亮但不可复核”的数字。

不能声称：项目已优化 vLLM kernel、已完成 HF 对照、已在 RTX 4060 获得某个吞吐提升，或已具备
生产级 CI/CD。这些都需要后续代码与真实证据。

更完整的项目拷打见 [面试问答](INTERVIEW_QA.md)，指标口径见
[Benchmark 方法论](BENCHMARK_METHODOLOGY.md)。

## 8. 风险与控制

| 风险 | 控制 |
| --- | --- |
| 8 GB 显存 OOM | 小模型、保守模型长度/显存比例/序列数，单变量调整 |
| benchmark 客户端成为瓶颈 | 记录 event-loop lag，超阈值标 `INCONCLUSIVE` |
| 服务端少生成 token | 输出长度门禁与 usage 证据 |
| GPU 被其他进程污染 | 要求 clean GPU，缺证据标 `INCONCLUSIVE` |
| 文档超前于代码 | 状态标签、CLI help 对照、发布前文档审计 |
| 简历夸大 | 只使用已提交代码、测试和 evidence bundle 支持的结论 |

## 9. 成功标准

短期成功不是“代码量更多”，而是能够用一条真实实验回答：环境是什么、请求如何到达、指标怎么
计算、实验为什么有效、瓶颈在哪里、调整参数后为什么改善或恶化。首个 RTX 4060 evidence bundle
完成后，项目才从“方法链已验证”进入“真实硬件结论已验证”。
