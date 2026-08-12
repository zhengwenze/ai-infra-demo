# 配置与 CLI

本文是当前版本的可执行接口说明。没有出现在 `inferscope --help` 中的命令，不属于当前 CLI。

## 1. 安装

```bash
uv sync --all-groups
uv run inferscope --help
```

Python 要求为 3.11–3.13。`uv sync --all-groups` 安装开发依赖；真实 GPU 推理服务通常要在
Linux NVIDIA 环境另行安装 vLLM。

## 2. 当前 CLI

```text
inferscope env capture
inferscope server check --base-url URL
inferscope benchmark run --config FILE [--results-dir DIR] [--repeat N]
inferscope benchmark show --run-id ID [--results-dir DIR]
```

### 捕获环境指纹

```bash
uv run inferscope env capture
```

输出无密钥 JSON，包含 Python、平台、CPU、内存、Git 和可用 GPU 软件信息。可用
`--project-dir` 指定 Git 项目。该命令不会保存凭据和 prompt/response 正文。

### 检查服务

```bash
uv run inferscope server check \
  --base-url http://127.0.0.1:8000 \
  --timeout-seconds 5
```

只请求 `/v1/models`，不生成 token。HTTP 不可用时退出码为 `3`。

### 运行 benchmark

```bash
uv run inferscope benchmark run \
  --config configs/smoke.yaml \
  --results-dir results \
  --repeat 1
```

CLI 遍历 `workload.arrival.values`，按 repeat 生成独立 run。配置错误退出 `2`，服务不可用退出 `3`，
任一结果不是 `VALID` 退出 `4`，键盘中断退出 `130`。

### 查看已有运行

```bash
uv run inferscope benchmark show \
  --run-id <run-id> \
  --results-dir results
```

打印 `validation.json` 与 `aggregate.json`；找不到文件时退出 `2`。

## 3. 可运行的最小配置

以下内容与 `configs/smoke.yaml` 的 schema 一致：

```yaml
schema_version: "1.0"
name: cpu-smoke
seed: 20260812

target:
  backend: vllm
  base_url: http://127.0.0.1:18000
  model: inferscope/fake-model
  request_type: chat_completions
  timeout_seconds: 10.0

generation:
  temperature: 0.0
  top_p: 1.0
  max_output_tokens: 8
  ignore_eos: false

workload:
  type: synthetic
  prompt_tokens: 8
  output_tokens: 8
  num_requests: 12
  arrival:
    mode: concurrency
    values: [1, 2, 4]

warmup:
  requests: 2
  include_in_metrics: false

telemetry:
  vllm_metrics_url: http://127.0.0.1:18000/metrics
  gpu_index: 0
  interval_ms: 500

validation:
  min_success_rate: 1.0
  output_token_tolerance_ratio: 0.0
  token_count_mismatch_ratio: 0.02
  max_client_loop_lag_ms: 20.0
  require_clean_gpu: false

slo:
  ttft_p95_ms: 500.0
  tpot_p95_ms: 100.0
  success_rate_min: 1.0

execution:
  repeats: 1
  cooldown_seconds: 0.0
  max_matrix_combinations: 8

output:
  save_prompts: false
  save_responses: false
  formats: [json, csv, markdown]
```

## 4. Schema 字段

所有 model 都是 strict、frozen 且 `extra="forbid"`：未知字段、隐式标量强转和运行中修改都会被
拒绝。YAML list 仅结构化转换为 tuple，不把字符串数字变成数值。

### 顶层

| 字段 | 类型 | 当前作用 |
| --- | --- | --- |
| `schema_version` | literal `"1.0"` | schema 版本 |
| `name` | kebab-case string | 实验名称 |
| `seed` | non-negative integer | prompt 和 Poisson 计划 seed |
| `target` | object | 被测服务与模型 |
| `generation` | object | 确定性生成参数 |
| `workload` | object | prompt/output 规模和到达计划 |
| `warmup` | object | 非正式请求预热 |
| `telemetry` | object | vLLM/NVML 采样设置 |
| `validation` | object | 有效性阈值 |
| `slo` | object | Goodput 阈值 |
| `execution` | object | 重复、冷却和矩阵上限 |
| `output` | object | 输出偏好；当前未完整接线 |

### `target`

| 字段 | 当前状态 |
| --- | --- |
| `backend: vllm` | 当前 runner 的实际 OpenAI-compatible 执行路径 |
| `backend: hf` | **Planned**；schema 接受，但没有 HF adapter |
| `base_url` | 必须是无内嵌凭据的绝对 HTTP(S) URL |
| `model` / `model_revision` | 模型身份；revision 可选 |
| `api_key_env` | 只接受大写环境变量名，运行时读取 value |
| `request_type: chat_completions` | 当前执行路径 |
| `request_type: completions` | **Planned**；schema 接受但 runner 未接线 |
| `timeout_seconds` | 正数请求超时 |

### `generation`

`temperature >= 0`、`0 < top_p <= 1`、`max_output_tokens > 0`，并支持 `ignore_eos`。runner 用
`max_output_tokens` 发请求；`workload.output_tokens` 用于构造目标与校验，两者应保持一致。

### `workload` 与 `arrival`

| 字段/值 | 当前状态 |
| --- | --- |
| `type: synthetic` | 已实现并通过 smoke |
| `type: shared_prefix` | 生成器与 runner 路径已实现；需要 `common_prefix_tokens` |
| `type: mixed` | **Planned**；当前会落入 synthetic 生成路径 |
| `prompt_tokens` / `output_tokens` | 正整数目标 |
| `num_requests` | 每个 run 的正式请求数 |
| `common_prefix_tokens` | 必须小于 prompt tokens |
| `arrival.mode: concurrency` | values 必须是正整数最大并发 |
| `arrival.mode: fixed_rate` | values 是正数 requests/second |
| `arrival.mode: poisson` | values 是平均 requests/second |

### `warmup` 与 `telemetry`

- `warmup.requests >= 0`；`include_in_metrics` 当前只能为 `false`。
- `vllm_metrics_url` 可为空，否则必须是 HTTP(S) URL。
- `gpu_index >= 0`，采样 `interval_ms >= 100`。

### `validation`

| 字段 | 当前作用或限制 |
| --- | --- |
| `min_success_rate` | 运行级成功率下限 |
| `output_token_tolerance_ratio` | 实际输出偏离目标的容忍比例 |
| `max_client_loop_lag_ms` | event-loop lag 上限 |
| `require_clean_gpu` | runner 当前传入 unknown，要求为 true 时通常 `INCONCLUSIVE` |
| `token_count_mismatch_ratio` | schema 已有，**尚未接入本地 tokenizer 对照门禁** |

### `slo`

`ttft_p95_ms`、`tpot_p95_ms` 与 `success_rate_min` 用于 Goodput。当前字段名带 `p95`，实现却把
两个时延值作为逐请求 qualifying threshold；见
[Benchmark 方法论](BENCHMARK_METHODOLOGY.md#3-goodput-的语义)。

### `execution` 与 `output`

- `repeats >= 1`，可被 CLI `--repeat` 覆盖；run 间等待 `cooldown_seconds`。
- `max_matrix_combinations` 限制 arrival values 数量，防止意外展开过大矩阵。
- `formats` 只接受 `json`、`csv`、`markdown`、`png` 且不可重复。
- `formats`、`save_prompts` 和 `save_responses` 尚未影响 runner 的固定产物路径；正文当前不保存。

## 5. 路径与安全

- API key 只通过 `api_key_env` 指向环境变量，配置与 manifest 不保存 key value。
- 结果目录不能是文件系统根、用户 home 或项目根；其他路径会 resolve 后使用。
- run id 只允许字母、数字、点、下划线和短横线，并拒绝复用已有 raw run 目录。
- prompt/response 只保存 SHA-256，不保存正文。

## 6. 当前产物接口

```text
results/raw/<run-id>/config.resolved.yaml
results/raw/<run-id>/manifest.json
results/raw/<run-id>/requests.jsonl
results/raw/<run-id>/client_metrics.jsonl
results/raw/<run-id>/server_metrics.jsonl
results/raw/<run-id>/gpu_metrics.jsonl
results/raw/<run-id>/validation.json
results/processed/<run-id>/aggregate.json
results/processed/<run-id>/summary.csv
results/charts/<experiment-and-first-run-id>.svg
reports/generated/<run-id>.md
```

机器消费者应以 JSON/JSONL 为事实源。项目尚未承诺跨版本 schema 兼容性；集成时应固定版本并保留
raw evidence。

## 7. 常见误用

- 不把 fake server 毫秒数写成 GPU 性能。
- 不只复制 `aggregate.json` 而丢掉 validation、manifest 和 requests。
- 不把 `INCONCLUSIVE` 解释成“基本通过”。
- 不在不同模型、token 长度或到达模式之间直接比较吞吐。
- 不使用旧设计文档里未出现在 `--help` 的命令。
