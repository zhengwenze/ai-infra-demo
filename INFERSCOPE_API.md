# InferScope 接口与数据契约

> 版本：v1.0
> 更新时间：2026-08-12
> 状态：核心契约已实现，扩展命令待完成

## 一、文档范围

InferScope 首期没有用户账号、数据库和远程实验管理服务。本文件定义：

1. HF 基线服务必须实现的 OpenAI-compatible HTTP 接口；
2. vLLM 目标服务需要满足的最小协议；
3. 流式 SSE 解析规则；
4. 实验配置、请求样本和结果文件的 Schema；
5. CLI 命令契约；
6. 错误码和降级行为。

接口兼容的目标是让同一个压测客户端无需修改即可请求 HF 基线和 vLLM。兼容范围以本文列出的字段为准，不宣称完整实现 OpenAI API。

## 二、HTTP 基础规范

| 字段 | 开发默认值 |
| --- | --- |
| HF 基线 URL | `http://127.0.0.1:8001` |
| vLLM URL | `http://127.0.0.1:8000` |
| 请求格式 | `application/json` |
| 普通响应 | `application/json; charset=utf-8` |
| 流式响应 | `text/event-stream` |
| 认证 | 本机默认不认证；远程目标可配置 Bearer Token |
| 时间格式 | RFC 3339 UTC，例如 `2026-08-12T08:00:00Z` |
| 内部持续时间 | 纳秒整数，使用单调时钟 |

### 2.1 请求头

```http
Content-Type: application/json
Accept: text/event-stream
Authorization: Bearer <optional-token>
X-Request-ID: <uuid>
```

规则：

- `X-Request-ID` 由客户端生成；
- 服务端应尽可能在响应头回传同一个 ID；
- Bearer Token 只从环境变量或私密配置读取；
- Token 不得写入日志、manifest 或错误报告。

### 2.2 错误响应

HF 基线使用以下格式：

```json
{
  "error": {
    "message": "max_tokens must be greater than 0",
    "type": "invalid_request_error",
    "param": "max_tokens",
    "code": "IS_CONFIG_INVALID"
  },
  "request_id": "1fbf3f25-4078-4e09-914d-9d68f05a8a42"
}
```

压测客户端不得依赖所有第三方后端都使用此格式；非 2xx 响应必须同时保留 HTTP 状态、截断后的响应体和标准化错误码。

## 三、HTTP 接口清单

| 方法 | 路径 | 说明 | 首期要求 |
| --- | --- | --- | --- |
| GET | `/health` | HF 基线进程与模型状态 | HF 必须 |
| GET | `/v1/models` | 查询已加载模型 | HF/vLLM 必须 |
| POST | `/v1/completions` | 文本补全 | 可选 |
| POST | `/v1/chat/completions` | 对话补全及流式输出 | HF/vLLM 必须 |
| GET | `/metrics` | Prometheus 指标 | vLLM 必须；HF 可选 |

## 四、接口详情

### 4.1 GET `/health`

用途：HF 基线启动探活。不能仅以 TCP 端口开放判断模型已就绪。

成功响应：

```json
{
  "status": "ready",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "revision": "resolved-commit-or-tag",
  "device": "cuda:0",
  "dtype": "bfloat16"
}
```

状态码：

| HTTP | 含义 |
| --- | --- |
| 200 | 模型已加载，可接收请求 |
| 503 | 进程存活，但模型尚未就绪或加载失败 |

### 4.2 GET `/v1/models`

成功响应：

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen2.5-0.5B-Instruct",
      "object": "model",
      "created": 1786492800,
      "owned_by": "inferscope"
    }
  ]
}
```

探活客户端必须检查：

- HTTP 200；
- `data` 非空；
- 配置中的模型名能与返回模型匹配；
- 失败时进行有限次数、指数退避重试。

### 4.3 POST `/v1/chat/completions`

#### 请求

```json
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": "Explain KV cache in one paragraph."
    }
  ],
  "stream": true,
  "temperature": 0.0,
  "top_p": 1.0,
  "max_tokens": 128,
  "seed": 20260812,
  "stream_options": {
    "include_usage": true
  }
}
```

#### 字段约束

| 字段 | 类型 | 必填 | 规则 |
| --- | --- | --- | --- |
| `model` | string | 是 | 必须与目标服务模型匹配 |
| `messages` | array | 是 | 至少一条消息 |
| `messages[].role` | enum | 是 | system/user/assistant |
| `messages[].content` | string | 是 | 首期只支持文本 |
| `stream` | boolean | 是 | 正式 TTFT 测试必须为 `true` |
| `temperature` | number | 否 | 基准默认 0.0 |
| `top_p` | number | 否 | `(0, 1]`，基准默认 1.0 |
| `max_tokens` | integer | 是 | 大于 0，不得超过配置上限 |
| `seed` | integer | 否 | 后端不支持时必须记录 |
| `stream_options.include_usage` | boolean | 否 | 后端支持时设为 true |

#### 非流式成功响应

```json
{
  "id": "chatcmpl-01",
  "object": "chat.completion",
  "created": 1786492800,
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "KV cache stores previously computed keys and values..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 16,
    "completion_tokens": 42,
    "total_tokens": 58
  }
}
```

#### 流式 SSE 响应

```text
data: {"id":"chatcmpl-01","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-01","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"KV"},"finish_reason":null}]}

data: {"id":"chatcmpl-01","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" cache"},"finish_reason":null}]}

data: {"id":"chatcmpl-01","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":16,"completion_tokens":42,"total_tokens":58}}

data: [DONE]

```

### 4.4 POST `/v1/completions`

首期不是核心接口，仅在实现成本很低或 GuideLLM 交叉验证需要时提供。

请求示例：

```json
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "prompt": "KV cache is",
  "stream": true,
  "temperature": 0.0,
  "max_tokens": 64
}
```

### 4.5 GET `/metrics`

vLLM 暴露 Prometheus 文本格式。InferScope 只依赖实际发现且能够解析的指标，不硬编码版本间已经移除的字段。

优先采集的逻辑指标：

| 逻辑名称 | 常见 vLLM 指标 | 用途 |
| --- | --- | --- |
| Running Requests | `vllm:num_requests_running` | 当前执行请求 |
| Waiting Requests | `vllm:num_requests_waiting` | 排队压力 |
| KV Cache Usage | `vllm:kv_cache_usage_perc` | KV Cache 压力 |
| Prefix Cache Queries | `vllm:prefix_cache_queries` | 缓存查询量 |
| Prefix Cache Hits | `vllm:prefix_cache_hits` | 缓存命中 |
| Prompt Tokens | `vllm:prompt_tokens_total` | Prefill 工作量 |
| Generation Tokens | `vllm:generation_tokens_total` | Decode 工作量 |
| TTFT | `vllm:time_to_first_token_seconds` | 服务端延迟分布 |
| ITL | `vllm:inter_token_latency_seconds` | 服务端 token 间隔 |
| E2E | `vllm:e2e_request_latency_seconds` | 端到端延迟 |

兼容规则：

- 启动时抓取 `/metrics` 并建立名称映射；
- 找不到非关键指标时记录 `missing_metrics`，不伪造零值；
- 找不到关键队列/KV 指标时允许完成客户端测试，但报告降级；
- Prometheus histogram 不得只读取某个 bucket 当作 P95；
- 对 counter 使用窗口差分，并处理进程重启导致的回绕。

## 五、SSE 解析契约

### 5.1 解析规则

- 以 SSE 空行作为事件边界，而不是假设一个 TCP chunk 等于一个事件；
- 支持单个网络读取包含多个 SSE 事件；
- 支持一个 SSE 事件跨多个网络读取；
- 忽略注释行和空 keep-alive；
- 支持 `data:` 后可选单个空格；
- 收到 `[DONE]` 后正常结束；
- `[DONE]` 前断开且没有合法 finish reason，标记为中途断流；
- UTF-8 解码必须支持跨字节分片；
- 单个事件体设置最大字节限制。

### 5.2 TTFT 边界

以下事件不得触发 TTFT：

- 只有 `role=assistant`；
- `delta.content` 缺失；
- `delta.content` 为空字符串；
- keep-alive 或注释；
- 只有 usage；
- 只有 finish reason。

第一个包含非空 `delta.content` 的完整 SSE 事件到达时记录 TTFT。

### 5.3 SSE 块与 token

SSE 内容块不保证一块对应一个 token。因此：

- `chunk_times_ns` 表示内容块到达时间；
- 默认输出名称为 `chunk_interarrival_ms`；
- 只有通过后端契约确认一块一 token 时才输出客户端 ITL；
- TPOT 使用最终输出 token 数计算，不使用 SSE 块数量；
- 服务端返回 usage 时优先使用服务端 token 计数；
- 本地 tokenizer 计数必须使用与目标模型匹配的 revision。

## 六、CLI 契约

计划命令：

```text
inferscope env capture
inferscope server check
inferscope benchmark run --config <path>
inferscope benchmark validate --run-id <id>
inferscope analyze summarize --run-id <id>
inferscope analyze compare --run-id <id> --run-id <id>
inferscope analyze pareto --group <name>
inferscope report build --run-id <id>
```

### 6.1 `benchmark run`

```bash
inferscope benchmark run \
  --config configs/baseline.yaml \
  --results-dir results \
  --repeat 3
```

退出码：

| code | 含义 |
| ---: | --- |
| 0 | 实验完成且有效 |
| 2 | 配置错误 |
| 3 | 目标服务不可用 |
| 4 | 实验完成但验证失败 |
| 5 | 资源耗尽或运行异常 |
| 130 | 用户中断，已尽力落盘部分结果 |

### 6.2 `benchmark validate`

只读取已存在的原始产物，不发送网络请求。重复运行必须产生一致的验证结果。

### 6.3 `analyze compare`

比较前先验证：

- 模型与 revision 一致；
- GPU 型号一致；
- 输入/输出分布一致；
- 正式样本数量和 SLO 口径兼容；
- 除声明的实验变量外，其余关键配置一致。

不满足时默认拒绝生成“提升百分比”，用户显式指定 `--allow-noncomparable` 后只能生成带警告的探索性报告。

## 七、实验配置 Schema

### 7.1 完整示例

```yaml
schema_version: "1.0"
name: vllm-concurrency-sweep
seed: 20260812

target:
  backend: vllm
  base_url: http://127.0.0.1:8000
  model: Qwen/Qwen2.5-0.5B-Instruct
  api_key_env: INFERSCOPE_API_KEY
  request_type: chat_completions
  timeout_seconds: 120

generation:
  temperature: 0.0
  top_p: 1.0
  max_output_tokens: 128
  ignore_eos: false

workload:
  type: synthetic
  prompt_tokens: 1024
  output_tokens: 128
  num_requests: 100
  arrival:
    mode: concurrency
    values: [1, 2, 4, 8, 16]

warmup:
  requests: 5
  include_in_metrics: false

telemetry:
  vllm_metrics_url: http://127.0.0.1:8000/metrics
  gpu_index: 0
  interval_ms: 500

validation:
  min_success_rate: 0.99
  output_token_tolerance_ratio: 0.10
  token_count_mismatch_ratio: 0.02
  max_client_loop_lag_ms: 20
  require_clean_gpu: true

slo:
  ttft_p95_ms: 500
  tpot_p95_ms: 50
  success_rate_min: 0.99

execution:
  repeats: 3
  cooldown_seconds: 10
  max_matrix_combinations: 32

output:
  save_prompts: false
  save_responses: false
  formats: [json, csv, markdown, png]
```

### 7.2 校验规则

- `num_requests > 0`；
- `prompt_tokens > 0`；
- `output_tokens > 0`；
- concurrency/request rate 均大于 0；
- telemetry interval 至少 100ms；
- repeats 至少 1，正式实验建议至少 3；
- 所有比例在 `[0, 1]`；
- 参数矩阵组合数不得超过配置上限；
- 输出目录不得指向文件系统根目录、用户主目录或项目根目录本身；
- API Key 字段只能引用环境变量名称，不能保存明文。

## 八、原始结果 Schema

### 8.1 `requests.jsonl`

每行一个请求：

```json
{
  "schema_version": "1.0",
  "run_id": "20260812T080000Z-7c91d8a1",
  "request_id": "1fbf3f25-4078-4e09-914d-9d68f05a8a42",
  "sequence": 17,
  "scheduled_at_ns": 1120000000,
  "started_at_ns": 1121000000,
  "first_content_at_ns": 1300000000,
  "finished_at_ns": 2480000000,
  "input_tokens": 1024,
  "output_tokens": 128,
  "token_count_source": "server_usage",
  "chunk_times_ns": [1300000000, 1312000000, 1324000000],
  "status": "success",
  "http_status": 200,
  "finish_reason": "length",
  "error_code": null,
  "prompt_sha256": "...",
  "response_sha256": "..."
}
```

### 8.2 `validation.json`

```json
{
  "schema_version": "1.0",
  "run_id": "20260812T080000Z-7c91d8a1",
  "status": "VALID",
  "checks": [
    {
      "name": "warmup_excluded",
      "status": "PASS",
      "message": "5 warmup requests excluded from measured samples"
    },
    {
      "name": "success_rate",
      "status": "PASS",
      "value": 1.0,
      "threshold": 0.99
    }
  ],
  "warnings": []
}
```

状态枚举：

| 状态 | 含义 |
| --- | --- |
| `VALID` | 可以用于正式比较 |
| `INVALID` | 存在确定性错误，不得用于性能结论 |
| `INCONCLUSIVE` | 数据已生成，但证据不足或环境干扰明显 |
| `ABORTED` | 测试未完成 |

### 8.3 `summary.json`

```json
{
  "schema_version": "1.0",
  "run_id": "20260812T080000Z-7c91d8a1",
  "validation_status": "VALID",
  "measured_seconds": 30.0,
  "requests": {
    "scheduled": 100,
    "successful": 100,
    "valid": 100
  },
  "throughput": {
    "requests_per_second": 3.333,
    "input_tokens_per_second": 3413.0,
    "output_tokens_per_second": 426.7,
    "goodput_requests_per_second": 2.8
  },
  "latency_ms": {
    "ttft": {"p50": null, "p95": null, "p99": null},
    "tpot": {"p50": null, "p95": null, "p99": null},
    "e2e": {"p50": null, "p95": null, "p99": null}
  },
  "gpu": {
    "utilization_mean": null,
    "memory_peak_bytes": null,
    "power_mean_watts": null
  }
}
```

示例中的 `null` 表示文档不提供虚构测试值；实现后只能从真实原始样本计算。

## 九、错误码

| 标准错误码 | 触发条件 | HTTP/CLI 行为 |
| --- | --- | --- |
| `IS_CONFIG_INVALID` | 配置缺失、范围错误 | HTTP 400 / CLI 2 |
| `IS_TARGET_UNAVAILABLE` | 探活失败 | HTTP 503 / CLI 3 |
| `IS_REQUEST_TIMEOUT` | 请求超时 | 样本标记 timeout |
| `IS_STREAM_MALFORMED` | SSE 无法解析 | 样本 error，保留截断证据 |
| `IS_TOKEN_COUNT_MISMATCH` | token 计数差异过大 | INCONCLUSIVE |
| `IS_OUTPUT_LENGTH_INVALID` | 输出长度超出容差 | INVALID 或样本无效 |
| `IS_WARMUP_FAILED` | 预热失败 | 终止正式测试 |
| `IS_CACHE_STATE_UNKNOWN` | 缓存状态无法证明 | INCONCLUSIVE |
| `IS_CLIENT_SATURATED` | 客户端事件循环/CPU 饱和 | INCONCLUSIVE |
| `IS_GPU_TELEMETRY_UNAVAILABLE` | NVML/DCGM 失败 | 警告并降级 |
| `IS_RESOURCE_EXHAUSTED` | OOM、连接或文件句柄耗尽 | CLI 5 |
| `IS_EXPERIMENT_INCONCLUSIVE` | 稳定性或可比性不足 | CLI 4 |

## 十、接口测试清单

- [ ] `/health` 在模型加载前返回 503，加载后返回 200；
- [ ] `/v1/models` 返回配置模型；
- [ ] chat 非流式响应符合最小 Schema；
- [ ] chat 流式响应最终包含 `[DONE]`；
- [ ] role-only chunk 不触发 TTFT；
- [ ] content chunk 触发且只触发一次 TTFT；
- [ ] usage-only chunk 被正确解析；
- [ ] 单个读取包含多个 SSE 事件；
- [ ] 单个 SSE 事件跨多个网络读取；
- [ ] UTF-8 多字节字符跨边界；
- [ ] HTTP 429/500 标准化；
- [ ] 中途断流、超时和取消正确落盘；
- [ ] API Key 在日志中脱敏；
- [ ] `/metrics` 指标缺失时明确降级；
- [ ] 结果文件通过 Pydantic Schema 校验。
