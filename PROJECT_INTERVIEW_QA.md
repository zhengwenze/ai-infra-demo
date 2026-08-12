# InferScope 项目面试拷打清单与参考答案

> 适用方向：AI Infra、LLM Serving、推理部署、推理性能测试、模型服务平台实习。  
> 项目状态：CPU fake server 链路已经验证；真实 RTX 4060 + vLLM 实验、HF baseline 和优化 A/B 尚未完成。  
> 回答原则：只讲当前代码和实验能够证明的事实，不把规划功能包装成已实现结果。

## 一、怎么使用这份清单

建议分三轮练习：

1. 第一轮只练“项目介绍、架构、指标”，每题控制在 30～60 秒；
2. 第二轮加入代码追问和反例，每题说完后主动指出一个边界；
3. 第三轮让同学随机连续追问，不看答案，用代码文件证明自己的说法。

面试回答可以采用下面的四句结构：

```text
先给结论 → 解释为什么 → 指出代码或实验怎么做 → 主动说明当前边界
```

例如：

> TTFT 从请求实际开始发送到第一个非空内容事件到达。项目使用单调时钟记录，避免系统时间回拨影响持续时间。对应实现位于 transport/openai_client.py 和 metrics/request.py。当前测到的是客户端观察到的 TTFT，包含网络和服务端排队，并不是纯 GPU Prefill 时间。

## 二、必须牢记的真实性边界

面试中可以说：

- 已实现异步 OpenAI-compatible Chat Completions 流式客户端；
- 已实现增量 SSE 解码、三种到达模型、TTFT/TPOT/吞吐/Goodput；
- 已实现 vLLM Prometheus 指标解析、NVML GPU 遥测和实验有效性门禁；
- 已实现 JSONL/JSON/CSV/Markdown/SVG 产物；
- CPU fake server 端到端链路已经验证，85 个非 GPU 测试通过；
- 已准备 RTX 4060 的 vLLM 启动脚本和实验配置。

面试中不能说：

- 已经证明 vLLM 比 HF 快多少；
- 已经在 RTX 4060 上获得某个吞吐或显存数字；
- 已经实现 PagedAttention、Continuous Batching 或 CUDA Kernel；
- 已经完成 Prefix Cache、Chunked Prefill 或量化优化实验；
- 已经支持 Kubernetes、多机多卡、PD 分离、RDMA 或弹性扩缩容；
- 已经接入 SGLang、TensorRT-LLM、GuideLLM 或生产 Prometheus/Grafana；
- 所有 `PROJECT_PROPOSAL.md`、`DEV_DOCUMENT.md` 中的规划功能都已经落地。

## 三、90 秒项目介绍

### Q1：请你完整介绍一下这个项目

**参考回答：**

> InferScope 是一个面向大模型在线推理服务的 SLO-aware benchmark 工具。它解决的问题不是单纯测一个 requests/s，而是在固定模型、负载和生成参数后，同时记录 TTFT、TPOT、端到端延迟、吞吐、成功率、Goodput、vLLM 指标和 GPU 遥测，并判断这次实验是否具备比较价值。
>
> 系统由严格 YAML 配置、确定性 workload、异步 OpenAI-compatible 流式客户端、多源遥测、实验有效性门禁、指标聚合和证据化报告组成。核心编排在 `src/inferscope/runner.py`。我自己实现的是 benchmark 和验证层，vLLM 作为被测推理后端复用。
>
> 当前 CPU fake server 已经跑通完整链路，自动化测试覆盖协议、指标、配置、遥测和 runner。RTX 4060 的配置已经准备好，但真实 GPU 数据尚未采集，所以我目前把它定位为“推理性能测量基础设施”，还不把它包装成已经完成的推理优化成果。

### Q2：这个项目最核心的技术价值是什么？

**参考回答：**

> 核心价值是让性能结论有证据链。很多压测只输出吞吐，但没有控制输入输出长度、预热、客户端饱和、token 计数和 GPU 环境，数字可能无法比较。InferScope 会保存解析后的配置、环境指纹、请求级时间线、遥测、校验结果和聚合报告；如果证据不足，就标记为 `INCONCLUSIVE`，而不是强行给出优化结论。

### Q3：为什么不直接使用 vLLM 自带 benchmark 或 GuideLLM？

**参考回答：**

> 成熟工具应该用于交叉验证，而不是重新造推理引擎。但只运行现成命令不容易展示我对请求时序、指标口径、实验有效性和结果追溯的理解。因此我实现了一个范围受控的 benchmark 核心，再计划用 GuideLLM 或 vLLM benchmark 验证趋势是否一致。当前交叉验证还没完成，这是后续必须补的证据。

### Q4：你到底优化了什么？

**参考回答：**

> 当前阶段我没有声称已经优化 vLLM Kernel 或调度器。我完成的是优化之前必须具备的测量、校验和诊断基础设施。真正的优化闭环应该是：建立 baseline、发现瓶颈、只修改一个主要变量、重复实验、比较 Goodput 和资源代价。目前最后三步还需要在 4060 上完成。

### Q5：这个项目和普通接口压测脚本有什么区别？

**参考回答：**

> 普通脚本通常只并发发请求并统计平均耗时。InferScope 还处理流式 SSE 边界、首个非空内容时间、服务端 token usage、开放和闭环负载、多源遥测、实验有效性、配置哈希、不可变产物目录以及 SLO Goodput。它更接近一个轻量性能实验系统，而不是一次性脚本。

### Q6：为什么这个项目适合 AI Infra，而不是普通后端？

**参考回答：**

> 它确实使用了后端工程能力，例如异步 I/O、协议解析、配置和测试；但指标和实验对象是 LLM serving，涉及 Prefill/Decode、KV Cache、首 token 延迟、逐 token 速度、GPU 显存和推理调度现象。它位于“后端服务工程进入推理 Infra”的交叉位置，但还没有深入到 CUDA 算子层。

## 四、架构与数据流追问

### Q7：一次 benchmark 从配置到报告经历了什么？

**参考回答：**

> CLI 读取 YAML，由 Pydantic 严格校验；runner 根据配置生成 workload 和 arrival plan；先执行 warmup，再启动客户端、vLLM 和 GPU 遥测；异步发送流式请求并保存请求级样本；测量结束后运行有效性门禁；随后计算 Goodput 和聚合指标，最后写入 raw、processed、Markdown 和 SVG 产物。

**代码锚点：**`src/inferscope/cli.py`、`config.py`、`runner.py`。

### Q8：为什么把 transport、metrics、validators、telemetry 分开？

**参考回答：**

> transport 只负责协议和原始时间事件；metrics 只做纯计算；validators 判断证据是否可信；telemetry 采集外部事实；runner 只负责编排。这样能避免网络层直接生成业务结论，也方便每个模块用确定性输入做单元测试。

### Q9：为什么使用严格配置模型？

**参考回答：**

> 性能实验最怕配置被静默转换。例如字符串 `"8"` 被当成整数、未知字段拼错但被忽略，都会让实验和预期不一致。项目使用 `extra="forbid"`、`strict=True` 和冻结模型，未知字段、隐式类型转换和运行中修改都会被拒绝。YAML 列表只会结构化转换成不可变 tuple，不做标量强制转换。

### Q10：配置哈希有什么用？

**参考回答：**

> 完整解析后的无密钥配置会用稳定 JSON 序列化，再计算 SHA-256。run id 包含哈希前缀，所以可以判断两次运行是不是同一配置，也能在报告和原始数据之间建立关联。哈希不能代替完整配置，因此 `config.resolved.yaml` 仍然会落盘。

### Q11：为什么每个 run 目录不可复用？

**参考回答：**

> 如果重复运行覆盖旧数据，就无法审计实验过程。`ArtifactStore.create()` 使用 `exist_ok=False`；JSON/YAML 通过临时文件原子替换，JSONL 写入后执行 flush 和 fsync。这样失败实验也能保留原始证据，而不是只保留最好结果。

### Q12：为什么不使用数据库？

**参考回答：**

> 首期规模是单机、单用户、离线实验，JSONL 和不可变目录更容易检查、版本化 schema 和导入 Pandas。数据库会增加部署和迁移成本。等需要多用户、跨机器任务调度、查询大量历史实验时，再引入数据库更合理。

### Q13：报告目录为什么不全放在 `results/`？

**参考回答：**

> 原始和聚合机器数据位于 `results/`，人类可读报告位于 `reports/generated/`。这是当前实现的实际布局。两类生成内容都被 gitignore；以后要展示作品，应筛选脱敏样例单独提交，而不是把所有运行数据直接入库。

### Q14：环境指纹记录了什么？

**参考回答：**

> 包括平台、Python、CPU 和内存、关键包版本、Git commit/branch/dirty 状态，以及可用时的 NVIDIA 设备、驱动等信息。只采集白名单字段，不扫描全部环境变量，避免把密钥写入 manifest。

## 五、TTFT、TPOT、吞吐与统计口径

### Q15：TTFT 怎么定义？

**参考回答：**

> TTFT 是第一个非空内容事件到达客户端的单调时间减去请求开始时间。角色信息、空 delta 和 usage 事件不会触发 TTFT。这个值包含客户端到服务端网络、排队、Prefill 和首 token 返回路径，因此不是纯 GPU Prefill 耗时。

### Q16：为什么是“第一个非空内容”，而不是第一个 SSE 事件？

**参考回答：**

> OpenAI-compatible 流可能先返回 role、空 delta 或其他元数据。如果把这些事件当作首 token，会系统性低估用户真正看到内容的时间。客户端只在 `delta.content` 是非空字符串时记录 `first_content_at_ns`。

### Q17：TPOT 怎么计算？为什么分母是 `output_tokens - 1`？

**参考回答：**

> 当前定义是 `(E2E - TTFT) / (output_tokens - 1)`。TTFT 已经覆盖第一个输出 token，因此剩余生成阶段对应后面的 `N-1` 个 token。输出 token 小于等于 1、缺少 TTFT 或缺少完成时间时，TPOT 返回 `None`，不伪造为 0。

### Q18：TPOT 和 ITL 是一回事吗？

**参考回答：**

> 不是。TPOT 是从总生成阶段和 token 数得到的平均值；ITL 是相邻 token 的时间间隔分布。客户端只能观察 SSE 内容 chunk，而一个 chunk 可能包含零个、一个或多个 token，所以项目把 chunk inter-arrival 标成代理指标，不冒充精确 token-level ITL。

### Q19：为什么 SSE chunk 不能直接当 token？

**参考回答：**

> TCP chunk 是网络分片，SSE event 是协议事件，token 是 tokenizer 的语义单位，这三个边界互不保证一致。同一个 UTF-8 字符甚至可能跨网络字节分片，一个 SSE 内容字段也可能包含多个 token。如果直接按 chunk 计 token，TPOT 和 ITL 都会失真。

### Q20：吞吐的分母为什么不能用所有请求耗时之和？

**参考回答：**

> 并发请求时间彼此重叠，累加请求延迟会重复计算时间。项目使用完整测量窗口：从开始调度正式 workload 到最后一个请求完成。请求吞吐是成功请求数除以这个 wall-clock window，token 吞吐同理。

### Q21：失败请求要不要算进吞吐？

**参考回答：**

> 请求吞吐当前统计成功请求率；scheduled、successful 和 valid 数量会分别保存。失败请求不能贡献有效服务吞吐，但必须进入成功率和校验门禁，避免系统通过大量快速失败获得漂亮数字。

### Q22：什么是 Goodput？

**参考回答：**

> Throughput 只问完成了多少请求，Goodput 还要求请求满足延迟 SLO。当前实现统计同时满足 TTFT 和 TPOT 阈值的请求速率；如果整次运行成功率低于 SLO，即使部分请求很快，Goodput 也记为 0，同时保留 qualifying count 解释原因。

### Q23：你配置字段叫 `ttft_p95_ms`，Goodput 却拿它和单请求 TTFT 比，是否有命名问题？

**参考回答：**

> 是，这是当前值得改进的 API 命名。聚合报告确实计算 TTFT P95；但 Goodput 计算把相同数值当作单请求阈值。更严谨的设计应该拆成 request-level SLO threshold 和 aggregate percentile gate，或者明确阈值语义。目前面试中我不会把二者混为一谈。

### Q24：P50、P95、P99 怎么计算？

**参考回答：**

> 项目使用 NumPy `quantile` 的 linear 方法，并在项目级常量中固定。空集合返回 count 0 和空统计值；NaN、Infinity 会被拒绝。样本数少于 100 时会标记 P99 small-sample warning，因为少量样本的 P99 很不稳定。

### Q25：为什么只跑一次不够？

**参考回答：**

> GPU 时钟、温度、缓存状态、系统噪声和请求调度都会造成波动。项目的稳定性分析使用吞吐均值、总体标准差和变异系数 CV；默认至少三次，CV 超阈值则标记 `INCONCLUSIVE`。不过该分析函数当前还没有完整接入 CLI 报告链路。

### Q26：为什么平均延迟不够？

**参考回答：**

> 在线服务的用户体验往往由尾部请求决定。平均值可能掩盖排队、长 Prefill 阻塞或偶发调度抖动，因此需要同时报告 P50、P95、P99，以及样本量和失败率。

## 六、异步并发与负载模型

### Q27：项目支持哪些负载模型？

**参考回答：**

> 支持闭环并发、固定速率和 Poisson 到达。闭环并发控制同时在途请求数；固定速率按确定间隔发请求；Poisson 用指数分布生成到达间隔，并通过 seed 复现。

### Q28：闭环并发和开放到达有什么区别？

**参考回答：**

> 闭环模型中，一个请求完成后才释放并发槽，慢服务会自动降低发出速率；开放模型按照外部时间计划到达，不因为服务变慢而减少新请求，更容易形成排队。闭环适合找安全并发，开放模型更接近独立用户流量和过载实验。

### Q29：并发限制怎么实现？

**参考回答：**

> runner 为全部 workload 创建协程，但使用 `asyncio.Semaphore(max_concurrency)` 限制同时进入请求执行区的数量。协程获得 semaphore 后记录 scheduled time，再调用异步客户端；完成后释放槽位。

### Q30：一次创建所有 coroutine 会不会占很多内存？

**参考回答：**

> 会随请求数增长。当前目标是几百级单机实验，全部 coroutine 加 semaphore 的实现简单可测。如果扩展到百万请求或长时间压测，应改为有界 producer-consumer 队列，避免一次性创建所有任务。

### Q31：固定速率如何避免累计 sleep 漂移？

**参考回答：**

> arrival plan 保存相对于测量起点的绝对 offset。执行时计算 `target_ns - current_ns`，而不是每次简单 sleep 固定间隔，因此前一次调度误差不会直接累加到下一次目标时间。但当前还没有单独聚合 dispatch lag，这是可以补充的客户端容量指标。

### Q32：Poisson 到达为什么更像线上流量？

**参考回答：**

> 当大量独立用户随机发起请求时，聚合到达常用 Poisson 过程近似，间隔服从指数分布。它比完全均匀的固定速率更容易产生短时突发。项目用局部随机数生成器和固定 seed，避免污染全局随机状态并保证复现。

### Q33：scheduled time 和 started time 有什么区别？

**参考回答：**

> scheduled 是负载计划期望或允许开始的时间，started 是客户端真正开始网络请求的时间。二者差值可以反映负载生成器调度延迟。闭环并发中 scheduled 在获得 semaphore 后记录，因此不会把等待并发槽的时间算进请求延迟；开放负载会保存原始目标 offset。

### Q34：为什么使用 `time.perf_counter_ns()`？

**参考回答：**

> 性能持续时间需要单调时钟，不能受 NTP 校时、时区或系统时间回拨影响。纳秒整数也避免多次浮点累计。UTC wall time只用于日志和跨进程关联，不用于本地延迟计算。

### Q35：客户端本身可能成为瓶颈吗？

**参考回答：**

> 会。Python event loop、CPU、连接数、JSON 解析和日志都可能限制发压。项目采集 event-loop lag、客户端 CPU 和 RSS；如果 GPU 利用率低、服务端没有排队，但 event-loop lag 很高，就应怀疑 load generator，而不是把结果归因于模型服务。

### Q36：为什么没有用多进程压测？

**参考回答：**

> 首期目标是 4060 单卡和较小并发，单进程异步 I/O 更容易统一时间线与复现。多进程会引入时钟对齐、结果合并和进程间协调问题。只有确认单进程客户端容量不足后，才应该引入分布式或多进程 load generator。

## 七、SSE 和网络协议拷打

### Q37：你的 SSEDecoder 为什么保持 bytes，而不是每个 chunk 立即 decode？

**参考回答：**

> UTF-8 字符可能在任意字节位置被 TCP 分片。如果每个网络 chunk 单独 decode，半个多字节字符会报错或被替换。decoder 先在 bytes 层累积到完整 SSE 行，再进行严格 UTF-8 解码。

### Q38：SSE event 的结束边界是什么？

**参考回答：**

> SSE 事件由空行分隔，不是由 TCP chunk 分隔。decoder 支持 LF、CRLF 和 bare CR，并能在一个网络 chunk 中解析多个事件，也能跨多个 chunk 拼出一个事件。

### Q39：流在半个 event 时断开怎么办？

**参考回答：**

> EOF 时调用 `decoder.finalize()`；如果还存在不完整行或事件，会抛出 `MalformedStreamError`，最终请求被标为 stream malformed。不能静默接受，否则 TTFT、内容和完成状态都可能基于不完整协议数据。

### Q40：为什么限制单个 SSE event 大小？

**参考回答：**

> 防止异常或恶意服务端持续发送无边界数据导致客户端内存无限增长。当前默认上限约 1 MiB，超出后按 malformed stream 失败。

### Q41：怎么判断一次流正常完成？

**参考回答：**

> 收到 `[DONE]`，或者存在有效的 `finish_reason`。如果连接关闭时两者都没有，就认为流异常结束。usage-only event 不会被误判成内容。

### Q42：网络超时、HTTP 错误和协议错误怎么区分？

**参考回答：**

> timeout 对应 `REQUEST_TIMEOUT`；连接和请求错误对应 `TARGET_UNAVAILABLE`；非法 Content-Type、JSON 或不完整 SSE 对应 `STREAM_MALFORMED`；HTTP 非 2xx 会保留状态码和截断后的错误正文。结果统一进入 `StreamResult`，而不是让普通请求错误打断整个矩阵。

### Q43：如何防止错误信息泄露 API Key？

**参考回答：**

> API key 只通过配置中的环境变量名解析，payload 不包含凭据。Authorization 只进入 header；错误消息和错误正文会做 Bearer/Authorization 模式脱敏并限制长度。当前 runner 不持久化原始 prompt 和 response，只记录哈希；配置中的正文保存开关尚未接入运行逻辑。

### Q44：支持请求取消吗？

**参考回答：**

> 数据模型预留了 `CANCELLED`，但当前客户端遇到 `asyncio.CancelledError` 会继续抛出，以保证协程取消语义正确；runner 尚未把取消转换成持久化 request sample。也就是说取消状态的完整落盘链路还没完成。

## 八、实验有效性与可信度

### Q45：有哪些实验有效性门禁？

**参考回答：**

> 当前包括 warmup 是否成功、成功率、成功请求时间戳完整性、输出 token 长度、token count 可用性、客户端 event-loop lag，以及 GPU 独占证据。门禁输出逐项 PASS/FAIL/WARN 和解释。

### Q46：`INVALID` 和 `INCONCLUSIVE` 有什么区别？

**参考回答：**

> `INVALID` 表示已观察到明确违反实验约束的问题，例如成功率或输出长度不达标；`INCONCLUSIVE` 表示缺少证据，例如 token count、客户端容量或 GPU 独占无法确认。前者不应参与比较，后者也不能形成正式结论，但原因不同。

### Q47：`ABORTED` 什么时候产生？

**参考回答：**

> enum 已定义，但当前 runner/CLI 在 KeyboardInterrupt 等中断场景下还没有写出完整的 `ABORTED` artifact。CLI 会以 130 退出。这是模型设计先于运行链路的一处未完成能力，不能说已经支持中断恢复。

### Q48：为什么输出长度也要校验？

**参考回答：**

> 如果一个配置因为提前 EOS 只生成少量 token，它的 E2E 和吞吐看起来可能更好，但工作量不同。项目按目标输出长度和容差检查成功请求，并建议固定温度、seed 和 EOS 策略。

### Q49：为什么 token count 缺失不能当成 0？

**参考回答：**

> 0 表示真实观测到没有 token，缺失表示不知道。混为一谈会抬高或降低 token throughput，并让 TPOT 失真。项目使用 `TokenCountSource` 标明 server usage、local tokenizer 或 unavailable；当前 runner 实际只接入 server usage 和 unavailable。

### Q50：`token_count_mismatch_ratio` 已经配置了，真的执行了吗？

**参考回答：**

> 还没有。配置模型已预留阈值，但 runner 尚未使用本地 tokenizer 重新计数，也没有 mismatch gate。因此我只能说“token usage 可用性检查已实现”，不能说“本地与服务端 token 一致性已验证”。

### Q51：GPU clean gate 真的有效吗？

**参考回答：**

> validator 支持 `gpu_is_clean`，但 runner 当前固定传入 `None`。当配置要求 clean GPU 且其他门禁通过时，实验会变成 `INCONCLUSIVE`，而不是假装已经独占。下一步需要在实验前后扫描 GPU compute process，并定义允许的服务进程。

### Q52：服务端 `/metrics` 拉取失败会怎样？

**参考回答：**

> 当前采集循环会捕获 HTTP 错误并继续 benchmark，避免遥测故障中断请求测量。但整个 scrape 失败时还没有写入明确的 unavailable snapshot，也没有专门门禁，所以这是当前可观测性链路的不足。更好的实现应记录失败时间、错误原因和缺失比例。

### Q53：为什么失败实验还要保存？

**参考回答：**

> 失败本身就是诊断证据。删除失败数据会产生幸存者偏差，也无法解释为什么某配置没有进入对比。项目保留原始请求和 validation，只限制它进入 Pareto 或正式结论。

## 九、vLLM、GPU 与 Transformer 推理基础

### Q54：Prefill 和 Decode 有什么区别？

**参考回答：**

> Prefill 一次处理全部输入 token，计算量大、并行度高，通常更接近 compute-bound，并影响 TTFT。Decode 每一步生成一个新 token，需要反复读取模型权重和 KV Cache，并行度相对低，常更接近 memory-bandwidth-bound，主要影响 TPOT。实际瓶颈仍要结合 profiler 和硬件指标判断，不能机械套结论。

### Q55：什么是 KV Cache？为什么占显存？

**参考回答：**

> 自回归生成时，每一层会缓存历史 token 的 Key 和 Value，后续 token 不必重复计算全部历史注意力。缓存规模随层数、KV head、head dimension、序列长度、并发序列数和数据类型增长，因此长上下文和高并发会快速消耗显存。

### Q56：PagedAttention 解决什么问题？你实现了吗？

**参考回答：**

> 它借鉴虚拟内存分页思想，把 KV Cache 管理成块，降低连续大块分配造成的碎片和预留浪费，也方便共享与调度。我没有实现 PagedAttention；它属于 vLLM 推理引擎能力，项目只把 vLLM 当被测后端并观察其行为。

### Q57：Continuous Batching 和静态 batching 有什么区别？

**参考回答：**

> 静态 batch 通常等待整批完成，短请求可能被长请求拖住。Continuous Batching 在迭代级别动态加入新请求、移除完成请求，提高 GPU 利用率和吞吐。代价是调度更复杂，过高负载可能导致排队和 TTFT 恶化。我没有实现调度器，只通过并发矩阵和服务端指标观察现象。

### Q58：Prefix Cache 为什么可能加速？

**参考回答：**

> 多个请求共享相同前缀时，可以复用这部分前缀的 KV Cache，减少重复 Prefill。收益取决于共享比例、缓存命中、缓存容量和驱逐策略。项目已有 shared-prefix workload 生成器，但还没完成真实 vLLM 的 cache on/off A/B，所以不能声称已经获得收益。

### Q59：Chunked Prefill 是什么？

**参考回答：**

> 它把长 Prefill 拆成较小块，与 Decode 或其他请求调度交错，避免一个超长输入长时间占据计算资源。可能改善短请求 TTFT，但也增加调度开销。当前项目尚未实现对应 A/B 配置与分组报告。

### Q60：如果并发提高后吞吐上涨、TTFT 急剧恶化，你怎么解释？

**参考回答：**

> 第一假设是排队和 batch 变大：GPU 利用率提升使吞吐增长，但请求等待调度或 Prefill 的时间增加。我要同时看 running/waiting requests、TTFT、TPOT、GPU 利用率和客户端 lag。如果 TPOT 相对稳定而 TTFT 上升，更像排队或 Prefill；如果 TPOT 也明显变差，可能存在 Decode 竞争、KV Cache 压力或内存带宽饱和。

### Q61：如果 GPU 利用率很低，但 TTFT 很高呢？

**参考回答：**

> 不能直接说 GPU 太弱。应依次检查客户端 event-loop lag、网络、服务端 waiting queue、模型加载/冷启动、CPU tokenizer、输入长度和采样间隔。如果客户端 lag 高，可能是发压器；如果服务端 waiting 高而 GPU 低，可能是 CPU/调度或指标采样问题。

### Q62：为什么 RTX 4060 配置选择 0.5B 模型？

**参考回答：**

> 8GB 显存首先要保证实验可运行和可重复。默认选择 Qwen2.5-0.5B-Instruct、4096 最大上下文、0.80 显存利用率和最多 8 条序列，优先降低 OOM 风险。后续可以逐步增加模型或上下文，而不是一开始用过大的模型导致无法形成完整数据。

### Q63：`--enforce-eager` 有什么取舍？

**参考回答：**

> 当前启动脚本使用 eager mode，目的是在小显存环境中降低图捕获相关的启动和额外显存风险，但可能损失 CUDA Graph 带来的性能。它应该作为实验变量或环境约束记录，而不是默认宣称最优。真实效果需要在同一模型和负载下 A/B。

### Q64：TP、PP、EP 分别是什么？项目支持吗？

**参考回答：**

> TP 把单层张量计算拆到多卡；PP 把不同层放到不同 stage；EP 主要把 MoE experts 分布到不同设备。它们涉及通信、负载均衡和并行效率。当前项目是单卡，不支持这些分布式模式；我只具备概念理解，不能把它当作项目经验。

### Q65：PD 分离和 RDMA 做了吗？

**参考回答：**

> 没有。PD 分离把 Prefill 和 Decode 放到不同资源池，RDMA 可用于低开销传输 KV 等数据，适合大规模 serving。当前项目首期明确是单机单卡，招聘材料中的这些内容是后续学习方向，不是已实现能力。

## 十、遥测与瓶颈定位

### Q66：项目采集哪些 GPU 指标？

**参考回答：**

> 通过 NVML 采集 GPU utilization、used memory、power 和 temperature；聚合报告目前保存平均利用率、显存峰值和平均功耗。某个设备字段不支持时跳过该 sample；NVML 整体不可用时保存 available=false 和原因，而不是写 0。

### Q67：如何对齐客户端、服务端和 GPU 指标？

**参考回答：**

> 每个遥测样本同时记录 UTC wall time 和本机 monotonic timestamp。本机持续时间和采样顺序用 monotonic，跨进程日志关联用 UTC。当前是单机设计；如果扩展到多机，需要 NTP/PTP 误差说明或集中采集，不能直接比较不同机器的 monotonic clock。

### Q68：vLLM 不同版本的指标名可能变化，怎么处理？

**参考回答：**

> 项目维护 logical metric 到多个 Prometheus alias 的映射，解析时选择当前版本实际存在的系列，并保留原始 Prometheus 名称。找不到的逻辑指标进入 `missing_metrics`，不会用 0 填充。这能缓解版本漂移，但 alias 表仍需要随官方版本维护。

### Q69：怎么判断是 Prefill 瓶颈还是 Decode 瓶颈？

**参考回答：**

> 看 workload 与指标联动：增加输入长度主要推高 TTFT，说明 Prefill 压力明显；增加输出长度或并发主要推高 TPOT，可能是 Decode/KV/带宽竞争。再结合 waiting requests、GPU 利用率、KV Cache 使用、功耗和 profiler 交叉确认。只凭一个 TTFT 或 GPU utilization 无法定因。

### Q70：GPU utilization 100% 就代表最优吗？

**参考回答：**

> 不代表。利用率高只说明采样窗口内设备忙，可能同时伴随排队、尾延迟恶化或无效计算。在线服务目标是满足 SLO 下的最大 Goodput，而不是单独追求 100% utilization。

### Q71：你会怎么设计真实 4060 实验？

**参考回答：**

> 固定驱动、vLLM、模型 revision、输入输出长度、生成参数和 seed；确认 GPU 无其他进程；先预热；并发 1/2/4/8，每点至少重复三次并冷却；保存全部 run；比较 TTFT/TPOT P95、吞吐、Goodput、显存和 CV。优化实验一次只改一个主要变量，并公开失败点和代价。

## 十一、测试与工程质量

### Q72：测试怎么分层？

**参考回答：**

> Unit 测试覆盖配置、workload、指标、遥测、验证和报告；Contract 测试固定 SSE 与 OpenAI-compatible 流式协议行为；Integration 测试用 fake server 跑 runner 到产物生成。当前有 85 个非 GPU 测试通过，但这不等价于真实 GPU 验证。

### Q73：SSE contract test 重点测什么？

**参考回答：**

> 包括 UTF-8 跨字节分片、CR/LF/CRLF、一个 chunk 多事件、一个事件跨多个 chunk、keep-alive、multi-line data、`[DONE]`、非法 JSON、错误 Content-Type、超大 event、半截 EOF、usage 和错误脱敏。

### Q74：fake server 的价值是什么？

**参考回答：**

> 它提供确定性的 SSE、usage 和 metrics，使 CI 或无 GPU 开发机也能验证协议、runner、门禁和 artifact 链路。它不能模拟真实模型的 Prefill、Decode、KV Cache 或 GPU 性能，所以 fake server 的吞吐不能写进简历。

### Q75：为什么 MyPy strict 和冻结数据模型有意义？

**参考回答：**

> 性能系统有很多 `None`、状态枚举和单位边界。严格类型能迫使代码显式处理 token count 缺失、未完成时间和不可用 GPU；冻结模型避免聚合阶段意外修改原始样本。它不能代替运行时测试，但能减少静默错误。

### Q76：当前测试最大的缺口是什么？

**参考回答：**

> 最大缺口是真实 NVIDIA GPU、真实 vLLM 版本和长时间负载。现有测试证明计算与协议实现按设计工作，但不能证明驱动兼容性、显存是否足够、vLLM metrics alias 是否覆盖目标版本，也不能证明性能数据准确。

## 十二、压力面与真实性拷打

### Q77：这个项目是不是复制了开源方案？

**参考回答：**

> 我参考了 vLLM、GuideLLM 和 AIPerf 的方法与指标，但没有复制 vLLM 的调度器或 Kernel。复用边界很明确：模型执行、Prometheus 协议和 NVML 来自成熟组件；我实现的是流式压测、到达模型、指标、有效性门禁、产物和报告层。方法参考需要承认，但核心代码可以逐模块解释。

### Q78：项目使用了 AI 编程工具吗？

**参考回答：**

> 使用了 AI 工具辅助设计、编码和测试，但我负责需求边界、指标口径、代码审查、运行验证和文档真实性。我不会声称每一行都纯手写。面试时我可以从 CLI 追到 runner、transport、metrics 和 artifacts，并解释关键取舍和已知缺陷；如果解释不了，就不应把它写成自己的能力。

### Q79：你没有真实 GPU 数据，为什么把项目写进简历？

**参考回答：**

> 目前可以证明的是性能测试基础设施和工程能力，不能证明优化收益。因此我会在简历中写“实现 benchmark、遥测和门禁”，不会写“提升 X%”。如果投递前能完成 4060 A/B 数据，项目说服力会显著提高；如果还没完成，我会主动说明状态。

### Q80：如果让我现场删掉一半功能，你保留什么？

**参考回答：**

> 保留 OpenAI streaming transport、确定性并发 workload、TTFT/TPOT/吞吐、实验有效性门禁和原始 artifact。Pareto、复杂报告、更多 arrival mode 都可以后置，因为测得准和可追溯比功能数量重要。

### Q81：项目最难的部分是什么？

**参考回答：**

> 最难的不是发 HTTP 请求，而是定义可信的时间和缺失语义。例如 TCP chunk、SSE event 和 token 边界不同；token count 缺失不能补 0；GPU 不可用不能写低利用率；失败实验不能删除。这些取舍决定结果是否可解释。

### Q82：当前代码里你最想重构什么？

**参考回答：**

> 第一是把 tokenizer 校准和 token mismatch gate 真正接入；第二是补齐 GPU process isolation；第三是把遥测 scrape failure 结构化保存；第四是拆分 Goodput 的 request-level threshold 与 aggregate P95 命名；第五是让 `output.formats`、请求类型和隐私开关真正影响运行行为。

### Q83：为什么没有直接做 Kubernetes？

**参考回答：**

> 当前核心风险是单机 benchmark 是否测得准，而不是部署规模。过早加入 K8s 会增加调度、网络和监控变量，反而难以判断误差来源。完成单卡 baseline 和优化闭环后，再把同一 artifact contract 扩展到 K8s 更合理。

### Q84：为什么用 Python，不用 C++？

**参考回答：**

> 项目主要是 I/O 密集的 load generator、配置、遥测和分析，Python 异步生态能快速实现并验证。真正的 GPU Kernel、runtime hot path 或极高 QPS 发压器可能需要 C++/Rust/CUDA，但当前瓶颈尚未证明在 Python。先测量再决定重写，而不是因为岗位写 C++ 就盲目换语言。

### Q85：如果面试官说“这不算推理优化”，你怎么回应？

**参考回答：**

> 我同意当前不能把它称为已经完成的优化结果，更准确是推理 benchmark、验证和诊断基础设施。我的下一步是用它完成一组受控 A/B，并根据 vLLM/GPU 指标解释收益和代价。这个边界我会主动说清楚，而不是和面试官争定义。

### Q86：为什么我们应该录用你？

**参考回答：**

> 我现在还不是 CUDA 或分布式推理专家，但已经能把后端工程能力迁移到 LLM serving，并且重视性能结论的可验证性。我能实现协议和工具，也能指出证据不足和下一步实验，而不是只展示一个可运行 Demo。对实习岗位，我的优势是工程落地、学习路径清楚、愿意用数据修正判断。

## 十三、场景题：让面试官继续追问

### Q87：并发 1、2、4、8 的吞吐持续上涨，但 Goodput 在 8 下降，你选哪个配置？

**参考回答：**

> 不能只选吞吐最高的 8。先确认实验有效；如果并发 8 因 TTFT/TPOT 超 SLO 导致 Goodput 下降，面向在线服务应优先选择 Goodput 更高的配置，例如 4。还要报告资源余量和波动，而不是把一个数字称为绝对最优。

### Q88：优化后吞吐提升 20%，但显存多占 30%，你怎么判断值不值？

**参考回答：**

> 取决于业务 SLO、模型副本密度和成本目标。若显存增加导致无法部署第二副本，系统总吞吐和容灾可能反而下降。应把 Goodput、显存、功耗、稳定性和部署密度放在多目标比较里，不只报告单点吞吐。

### Q89：两次实验配置相同，但结果差 15%，怎么办？

**参考回答：**

> 先不下结论。检查 config hash、Git dirty 状态、模型 revision、GPU 进程、温度/时钟、缓存冷热、客户端 lag、错误率和输出长度；增加 repeats，计算 CV；必要时随机化实验顺序，避免温度和顺序偏差。

### Q90：怎么证明你的工具测得对？

**参考回答：**

> 分三层：用单元测试证明公式和边界；用协议测试证明时间事件获取正确；用确定性 fake server 验证端到端 artifact；最后在真实 vLLM 上与官方 benchmark/GuideLLM 做趋势和口径交叉验证。当前前三层已具备，最后一层仍未完成。

### Q91：如果服务端不返回 usage，怎么计算 TPOT？

**参考回答：**

> 当前会把 token count 标为 unavailable，TPOT 和 token throughput 不形成可信结论。后续应使用与目标模型和 revision 一致的本地 tokenizer 重算，并记录 token count source，再检查本地与服务端差异，不能用字符串长度代替 token 数。

### Q92：如果让你两周内把项目变成可投递作品，你怎么排优先级？

**参考回答：**

> 第一，完成 tokenizer 和 clean GPU 两个可信度缺口；第二，在 4060 上完成 baseline、并发矩阵和一个优化 A/B；第三，加入三次重复和 GuideLLM 趋势对照；第四，提交脱敏样例 artifact、图表和结论；第五，再补 CI 和演示视频。K8s、CUDA Kernel 和多机不进入这两周范围。

## 十四、根据不同岗位调整重点

### AI 平台 / LLMOps / 模型服务岗位

重点讲：

- OpenAI-compatible streaming；
- 异步并发、错误分类和探活；
- 配置、环境指纹、artifact 和复现；
- 遥测、SLO、失败诊断和服务稳定性。

主动承认：当前没有 K8s、autoscaling 和生产多租户。

### 推理性能 / vLLM 岗位

重点讲：

- TTFT、TPOT、Goodput 和尾延迟；
- Prefill/Decode、KV Cache、continuous batching；
- 并发矩阵、shared prefix、GPU/vLLM 指标；
- 如何设计单变量 A/B 和识别客户端瓶颈。

主动承认：真实 vLLM A/B 尚未完成，PagedAttention 和调度器不是自研。

### CUDA / Kernel / TensorRT-LLM 岗位

重点讲：

- 当前项目可以作为性能测量入口；
- 自己对 compute-bound、memory-bound、带宽和 profiler 的基础理解；
- 下一步如何从瓶颈证据进入 Triton/CUDA 优化。

主动承认：当前没有 Kernel、CUDA profiler 报告和 C++ hot path 经验。这类岗位应作为冲刺，不应靠包装项目蒙混。

### 分布式 Serving / 调度岗位

重点讲：

- 开放与闭环负载；
- 排队、SLO、Goodput 和可观测性；
- 单卡实验如何扩展为多机 artifact contract。

主动承认：当前没有 K8s、PD/EP、RDMA、NCCL 和多机时钟对齐实践。

## 十五、面试时可以主动打开的代码

| 面试问题 | 建议打开的文件 |
| --- | --- |
| 项目主流程 | `src/inferscope/runner.py` |
| SSE 为什么不能按 chunk 解析 | `src/inferscope/transport/sse.py` |
| TTFT 如何触发 | `src/inferscope/transport/openai_client.py` |
| TTFT/TPOT 公式 | `src/inferscope/metrics/request.py` |
| 吞吐和分位数 | `src/inferscope/metrics/aggregate.py` |
| Goodput | `src/inferscope/analysis/goodput.py` |
| 实验是否可信 | `src/inferscope/validators/experiment.py` |
| 并发/固定速率/Poisson | `src/inferscope/workloads/arrival.py` |
| vLLM 指标兼容 | `src/inferscope/telemetry/vllm_metrics.py` |
| GPU 不可用语义 | `src/inferscope/telemetry/gpu.py` |
| 不可变产物 | `src/inferscope/artifacts.py` |
| 严格配置和密钥引用 | `src/inferscope/config.py` |
| 协议测试 | `tests/contract/test_sse.py`、`test_openai_client.py` |
| 端到端测试 | `tests/integration/test_runner.py` |

## 十六、面试前自测评分表

每项 0～2 分：0 分不会，1 分能背，2 分能脱稿并打开代码证明。

| 能力 | 分数 |
| --- | ---: |
| 90 秒项目介绍 | /2 |
| 从 CLI 讲完整调用链 | /2 |
| TTFT、TPOT、E2E、Throughput、Goodput | /2 |
| SSE event、TCP chunk、token 的区别 | /2 |
| 闭环并发、固定速率、Poisson | /2 |
| 单调时钟与完整测量窗口 | /2 |
| VALID、INVALID、INCONCLUSIVE | /2 |
| Prefill、Decode、KV Cache | /2 |
| PagedAttention、Continuous Batching | /2 |
| 客户端/vLLM/GPU 三类遥测 | /2 |
| 真实 4060 实验设计 | /2 |
| 主动说明五项当前限制 | /2 |
| 打开核心代码接受连续追问 | /2 |
| 不虚构优化数据 | /2 |

建议达到 24/28 再把本项目作为主项目接受深挖。能背答案但不能解释代码，只能算 1 分。

## 十七、反问面试官

可以根据岗位选择两到三个问题：

1. 团队当前更关注推理引擎、Serving 平台，还是 GPU 资源调度？
2. 线上最重要的性能目标是 TTFT、TPOT、Goodput、成本还是稳定性？
3. 团队主要使用 vLLM、SGLang、TensorRT-LLM，还是自研引擎？
4. 实习生会参与真实 benchmark、profiling 和线上故障定位吗？
5. 当前最大的瓶颈更常出现在 Prefill、Decode、KV Cache、通信还是调度层？
6. 团队如何保证性能实验的可复现性和配置公平性？
7. 对这个岗位而言，C++/CUDA 与服务工程能力的实际占比大约是多少？

不要一上来只问加班、转正和薪资；也不要为了显得专业堆砌 RDMA、EP、MTP 等自己无法继续讨论的词。

## 十八、最后的答题底线

当面试官问到未实现能力时，使用下面的模板：

> 这个能力我目前只理解原理，还没有在项目中完成，因此我不把它当作实践经验。当前代码已经完成的是……；如果继续实现，我会先……，再用……指标验证。

对实习生而言，“知道边界并给出验证路径”比编造一个看似完整的答案更可靠。
