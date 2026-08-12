# Benchmark 方法论

InferScope 的基本原则是：**先判断实验是否可用，再讨论性能数字。** 本文给出当前实现的指标、
负载、测量窗口、SLO 和有效性口径。

## 1. 请求级时间点

对每个流式请求记录三个单调时钟时间点：

- `started_ns`：客户端开始请求；
- `first_content_ns`：收到首个非空内容 token；
- `finished_ns`：响应流正常结束或失败被记录。

由此定义：

```text
TTFT = first_content_ns - started_ns
E2E  = finished_ns - started_ns
TPOT = (finished_ns - first_content_ns) / (output_tokens - 1)
```

TPOT 只有在首内容存在且输出 token 至少为 2 时才有定义。角色消息、空字符串和 `[DONE]` 不算首个
内容 token。失败请求不会被混入成功时延分布。

## 2. 运行级指标

| 指标 | 计算口径 |
| --- | --- |
| 成功率 | 成功请求数 / 总正式请求数 |
| 请求吞吐 | 成功请求数 / `measured_seconds` |
| 输出 token 吞吐 | 成功请求输出 token 总数 / `measured_seconds` |
| TTFT/TPOT/E2E 百分位 | 对有效请求样本使用线性插值百分位 |
| Goodput | 满足请求级 TTFT 与 TPOT 阈值的请求数 / 测量窗口 |

`measured_seconds` 是正式测量窗口，不包含 warmup。聚合函数要求调用方显式传入该窗口，避免用
请求 E2E 求和等错误分母。

小样本的 P95/P99 可以由数学插值得到，但统计意义很弱：少于 100 个样本时，P99 更适合检查链路，
不适合形成硬件性能结论。

## 3. Goodput 的语义

一个请求只有同时满足 TTFT 和 TPOT SLO 才算 qualifying request。若整个运行的成功率低于
`success_rate` SLO，本次 Goodput 直接为零，同时保留 qualifying request 数作为诊断证据。

```text
if run_success_rate < required_success_rate:
    goodput = 0
else:
    goodput = qualifying_requests / measured_seconds
```

配置字段名使用 `ttft_p95_ms` 和 `tpot_p95_ms`，但当前 Goodput 实现把这些值作为**逐请求上限**；
它不是“只检查该运行的 P95 是否达标”。报告和面试表述必须说明这一点，后续版本应将字段重命名
或拆分运行级与请求级 SLO。

## 4. 负载模型

### 闭环并发

保持最多 N 个在途请求，请求完成后补位。它适合观察并发提升时的饱和点，但不是固定外部到达率。

### 固定速率

按 `1 / requests_per_second` 的间隔安排请求。客户端 event-loop 采样用于判断负载生成器是否
可能过载；当前报告还没有独立聚合每个请求的 dispatch lag。

### Poisson 到达

相邻请求间隔服从指数分布，并通过 `seed` 保证计划可复现。这里随机的是 inter-arrival time，
不是请求服务时间。

## 5. Prompt 与输出控制

`synthetic` workload 按目标 prompt token 数近似构造输入；`shared_prefix` 在多请求间复用前缀，
用来研究 prefix cache 类场景。目标输出 token 通过请求参数传给服务端。

当前没有本地 tokenizer，无法独立验证服务端 usage 是否准确；`token_count_mismatch_ratio` 已进入
配置 schema，但尚未形成实际门禁。因此真实对比应锁定 tokenizer 与 chat template，并检查
服务端返回的 token 统计。

`mixed` 已被配置枚举接受，但 runner 目前没有独立混合负载生成逻辑，不能宣称已支持。

## 6. Warmup 与重复实验

每个负载点先运行 `warmup_requests`，这些请求不进入正式聚合。warmup 的目标是降低首次加载、
JIT、缓存冷启动对正式窗口的影响，但不能自动证明系统已稳定。

`repeat` 会生成多次独立 run。稳定性与 Pareto 纯函数已经实现并测试，但 CLI 尚未把多次 run 自动
合成稳定性报告；现阶段需要保留每次产物，避免只挑最好的一次。

## 7. 有效性门禁

validator 关注以下证据：

| 门禁 | 目的 | 可能结果 |
| --- | --- | --- |
| warmup 是否执行 | 避免把冷启动直接当稳态 | INVALID |
| 成功率 | 拒绝用大量失败换取高吞吐 | INVALID |
| timing completeness | 保证 TTFT/TPOT 样本可解释 | INVALID |
| 输出长度 | 防止服务端少生成 token 获得虚假优势 | INVALID |
| token count availability | 防止吞吐分子缺失 | INVALID/INCONCLUSIVE |
| client event-loop lag | 识别压测端可能过载 | INCONCLUSIVE |
| clean GPU evidence | 防止其他进程污染 GPU 对比 | INCONCLUSIVE |

`INCONCLUSIVE` 表示证据不足，而不是性能介于好坏之间。若配置要求 clean GPU，但 runner 无法证明，
即使请求本身全部成功，也不能把结果标成 `VALID`。

## 8. 公平对比清单

比较两个 backend 或两组参数前，至少固定：

1. GPU 型号、驱动、CUDA、功耗/频率策略和并发进程；
2. 模型 revision、dtype、量化方式、tokenizer 与 chat template；
3. prompt 集合、输入/输出 token 目标、seed、请求数和到达模型；
4. warmup、repeat、cooldown、超时、SLO 和错误处理；
5. vLLM/框架版本、启动参数和显存配置；
6. 两边相同的有效性门禁与测量窗口。

只报告“最佳吞吐”会隐藏尾延迟和失败率。建议同时给出 TTFT/TPOT P50/P95/P99、成功率、请求吞吐、
token 吞吐、Goodput、峰值 GPU 内存以及 `validation.status`。

## 9. 不能从当前结果推出什么

- CPU fake-server smoke 不能代表真实模型、GPU 或 vLLM 性能。
- 已实现 NVML/vLLM parser 不等于已在 NVIDIA 环境验证。
- 未提交完整 evidence bundle 的数字不能称为可复现 benchmark。
- 一张 8 GB 卡上的单模型结果不能推广到其他 GPU、模型或生产负载。
- 当前项目没有经过跨引擎公平 A/B，因此不能声称某引擎更快。

产物结构见 [结果证据说明](results/README.md)，首次单卡运行见
[RTX 4060 指南](RTX4060_GUIDE.md)。
